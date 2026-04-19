from __future__ import annotations

import contextlib
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf

from configs.config import Config
from infer.modules.vc.modules import VC
from tools import process_mixed_long_audio as mixed_audio
from tools.rvc_pipeline.segmenter import build_clean_segments


def _progress_event(stage: str, progress: float, message: str) -> dict:
    return {"stage": stage, "progress": progress, "message": message}


def _report(
    progress_callback: Callable[[dict], None] | None,
    stage: str,
    progress: float,
    message: str,
) -> None:
    if progress_callback:
        progress_callback(_progress_event(stage, progress, message))


def _write_stage_manifest(job_dir: Path, pipeline: str, stages: list[dict]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = job_dir / f"{pipeline}_stages.json"
    manifest_path.write_text(
        json.dumps({"pipeline": pipeline, "stages": stages}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_stage_summary(job_dir: Path, stage_name: str, payload: dict) -> None:
    summary_path = job_dir / f"{stage_name}_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@contextlib.contextmanager
def _temporary_mixed_audio_settings(params: dict, male_model: dict, female_model: dict):
    old_values = {
        "MALE_MODEL": mixed_audio.MALE_MODEL,
        "MALE_INDEX": mixed_audio.MALE_INDEX,
        "FEMALE_MODEL": mixed_audio.FEMALE_MODEL,
        "FEMALE_INDEX": mixed_audio.FEMALE_INDEX,
        "MALE_PARAMS": mixed_audio.MALE_PARAMS,
        "FEMALE_PARAMS": mixed_audio.FEMALE_PARAMS,
        "UVR_MODEL": mixed_audio.UVR_MODEL,
        "READING_MODE": mixed_audio.READING_MODE,
        "SPEAKER_ENCODER_FAILED": mixed_audio.SPEAKER_ENCODER_FAILED,
        "SPEAKER_ENCODER": mixed_audio.SPEAKER_ENCODER,
    }
    try:
        mixed_audio.MALE_MODEL = male_model["model"]
        mixed_audio.MALE_INDEX = male_model.get("index", "")
        mixed_audio.FEMALE_MODEL = female_model["model"]
        mixed_audio.FEMALE_INDEX = female_model.get("index", "")
        mixed_audio.MALE_PARAMS = params["male_params"]
        mixed_audio.FEMALE_PARAMS = params["female_params"]
        mixed_audio.UVR_MODEL = params.get("uvr_model", mixed_audio.UVR_MODEL)
        mixed_audio.READING_MODE = bool(params.get("reading_mode", mixed_audio.READING_MODE))
        if not params.get("speaker_embedding", False):
            mixed_audio.SPEAKER_ENCODER_FAILED = True
            mixed_audio.SPEAKER_ENCODER = None
        yield
    finally:
        for key, value in old_values.items():
            setattr(mixed_audio, key, value)


def _run_long_mixed_process(
    input_path: Path,
    job_dir: Path,
    progress_callback: Callable[[dict], None] | None,
) -> tuple[list[dict], dict]:
    stages = [
        {"name": "prepare", "status": "completed"},
        {"name": "uvr_split", "status": "running"},
        {"name": "segment_merge", "status": "pending"},
        {"name": "convert_remix", "status": "pending"},
    ]
    _write_stage_manifest(job_dir, "long_mixed_pipeline", stages)

    _report(progress_callback, "converting", 0.02, "准备输出目录")
    job_dir.mkdir(parents=True, exist_ok=True)
    segment_inputs = job_dir / "segments_in"
    segment_outputs = job_dir / "segments_out"
    segment_inputs.mkdir(exist_ok=True)
    segment_outputs.mkdir(exist_ok=True)

    _report(progress_callback, "converting", 0.08, "分离人声和伴奏")
    vocal_path, instrumental_path = mixed_audio.run_uvr_split(input_path, job_dir)
    _write_stage_summary(
        job_dir,
        "uvr_split",
        {
            "vocal_path": str(vocal_path),
            "instrumental_path": str(instrumental_path),
            "input_duration_seconds": round(float(sf.info(input_path).duration), 4),
            "vocal_duration_seconds": round(float(sf.info(vocal_path).duration), 4),
            "instrumental_duration_seconds": round(float(sf.info(instrumental_path).duration), 4),
        },
    )
    stages[1]["status"] = "completed"
    stages[2]["status"] = "running"
    _write_stage_manifest(job_dir, "long_mixed_pipeline", stages)

    _report(progress_callback, "converting", 0.18, "检测人声片段")
    mono_vocal, vocal_sr, intervals = mixed_audio.detect_segments(vocal_path)
    instrumental_audio, instrumental_sr = sf.read(instrumental_path, always_2d=True)

    _report(progress_callback, "converting", 0.28, f"分析音高与音色，初始片段 {len(intervals)} 段")
    rmvpe = mixed_audio.create_rmvpe(Config())
    intervals = mixed_audio.merge_adjacent_same_route(mono_vocal, vocal_sr, intervals, rmvpe)
    analyzed_segments = mixed_audio.analyze_intervals(mono_vocal, vocal_sr, intervals, rmvpe)
    mixed_audio.absorb_short_passthrough_segments(analyzed_segments, vocal_sr)
    analyzed_segments = mixed_audio.merge_context_absorbed_segments(mono_vocal, vocal_sr, analyzed_segments)
    route_counts = Counter(str(item["route"]) for item in analyzed_segments)
    note_counts = Counter(str(item["note"]) for item in analyzed_segments)
    _write_stage_summary(
        job_dir,
        "segment_merge",
        {
            "input_interval_count": len(intervals),
            "merged_segment_count": len(analyzed_segments),
            "route_counts": dict(route_counts),
            "note_counts": dict(note_counts),
            "voiced_sample_rate": vocal_sr,
        },
    )
    _report(progress_callback, "converting", 0.36, f"片段合并完成，待处理 {len(analyzed_segments)} 段")
    stages[2]["status"] = "completed"
    stages[3]["status"] = "running"
    _write_stage_manifest(job_dir, "long_mixed_pipeline", stages)

    execution_context = {
        "segment_inputs": segment_inputs,
        "segment_outputs": segment_outputs,
        "vocal_path": vocal_path,
        "instrumental_path": instrumental_path,
        "mono_vocal": mono_vocal,
        "vocal_sr": vocal_sr,
        "instrumental_audio": instrumental_audio,
        "instrumental_sr": instrumental_sr,
        "analyzed_segments": analyzed_segments,
    }
    return stages, execution_context


def _load_long_mixed_segments(job_dir: Path) -> list[dict]:
    segments_path = job_dir / "segments.csv"
    segments: list[dict] = []
    if segments_path.exists():
        with open(segments_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                segments.append(
                    {
                        "start": float(row["start_sec"]),
                        "end": float(row["end_sec"]),
                        "route": row["route"],
                        "median_f0": float(row["median_f0_hz"]),
                        "voiced_ratio": float(row["voiced_ratio"]),
                        "note": row["note"],
                    }
                )
    return segments


def _finalize_long_mixed_output(job_dir: Path, output_path: Path) -> None:
    final_mix = job_dir / "final_mix.mp3"
    if not final_mix.exists():
        raise RuntimeError("long_mixed_pipeline did not produce final_mix.mp3")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_mix, output_path)


def _execute_long_mixed_convert_and_remix(
    execution_context: dict,
    job_dir: Path,
    progress_callback: Callable[[dict], None] | None,
) -> None:
    segment_inputs = execution_context["segment_inputs"]
    segment_outputs = execution_context["segment_outputs"]
    mono_vocal = execution_context["mono_vocal"]
    vocal_sr = execution_context["vocal_sr"]
    instrumental_audio = execution_context["instrumental_audio"]
    instrumental_sr = execution_context["instrumental_sr"]
    analyzed_segments = execution_context["analyzed_segments"]

    _report(progress_callback, "converting", 0.42, "加载男声与女声模型")
    male_vc = mixed_audio.instantiate_vc(mixed_audio.MALE_MODEL)
    female_vc = mixed_audio.instantiate_vc(mixed_audio.FEMALE_MODEL)
    male_index_path = mixed_audio.resolve_index_path(mixed_audio.MALE_INDEX)
    female_index_path = mixed_audio.resolve_index_path(mixed_audio.FEMALE_INDEX)

    converted_vocals = np.zeros(
        (instrumental_audio.shape[0], instrumental_audio.shape[1]), dtype=np.float32
    )
    decisions: list[mixed_audio.SegmentDecision] = []
    total_segments = max(len(analyzed_segments), 1)

    for segment_id, segment_meta in enumerate(analyzed_segments):
        route_label = {
            "male": "男声",
            "female": "女声",
            "passthrough": "直通",
        }.get(segment_meta["route"], segment_meta["route"])
        _report(
            progress_callback,
            "converting",
            0.45 + 0.40 * ((segment_id + 1) / total_segments),
            f"正在转换第 {segment_id + 1}/{total_segments} 段 | {route_label} | {segment_meta['duration_sec']:.2f}s",
        )
        start = segment_meta["start"]
        end = segment_meta["end"]
        start_sec = start / vocal_sr
        end_sec = end / vocal_sr
        duration_sec = segment_meta["duration_sec"]
        raw_segment = segment_meta["raw_segment"]
        segment_input_path = segment_inputs / f"segment_{segment_id:04d}.wav"
        sf.write(segment_input_path, raw_segment, vocal_sr)

        classification = segment_meta["classification"]
        route = segment_meta["route"]
        median_f0 = segment_meta["median_f0"]
        voiced_frames = segment_meta["voiced_frames"]
        voiced_ratio = segment_meta["voiced_ratio"]
        note = segment_meta["note"]

        model_name = ""
        index_name = ""
        params = None
        segment_audio = raw_segment.copy()

        existing_outputs = sorted(segment_outputs.glob(f"segment_{segment_id:04d}_*.wav"))
        if existing_outputs:
            output_segment_path = existing_outputs[-1]
            route = output_segment_path.stem.rsplit("_", 1)[-1]
            cached_audio, cached_sr = sf.read(output_segment_path, always_2d=True)
            segment_audio = mixed_audio.ensure_mono(cached_audio.astype(np.float32))
            if route == "male":
                model_name = mixed_audio.MALE_MODEL
                index_name = mixed_audio.MALE_INDEX
                params = mixed_audio.MALE_PARAMS
            elif route == "female":
                model_name = mixed_audio.FEMALE_MODEL
                index_name = mixed_audio.FEMALE_INDEX
                params = mixed_audio.FEMALE_PARAMS
            else:
                note = "reused_passthrough"
            if cached_sr != instrumental_sr:
                segment_audio = librosa.resample(segment_audio, orig_sr=cached_sr, target_sr=instrumental_sr)
            target_len = int(round(duration_sec * instrumental_sr))
            segment_audio = mixed_audio.fit_to_length(segment_audio, target_len).astype(np.float32)
            segment_audio = mixed_audio.apply_edge_fade(segment_audio, instrumental_sr)
            start_out = int(round(start_sec * instrumental_sr))
            end_out = min(start_out + target_len, converted_vocals.shape[0])
            usable = segment_audio[: max(0, end_out - start_out)]
            if usable.size > 0:
                converted_vocals[start_out:end_out, :] += usable[:, None]
            chosen_params = params or mixed_audio.MALE_PARAMS
            decisions.append(
                mixed_audio.SegmentDecision(
                    segment_id=segment_id,
                    start_sec=round(start_sec, 3),
                    end_sec=round(end_sec, 3),
                    duration_sec=round(duration_sec, 3),
                    median_f0_hz=round(median_f0, 3),
                    voiced_frames=voiced_frames,
                    voiced_ratio=round(voiced_ratio, 4),
                    classification=classification,
                    route=route,
                    low_confidence=note != "ok",
                    model_name=model_name,
                    index_name=index_name,
                    f0_up_key=chosen_params["f0_up_key"],
                    f0_method=chosen_params["f0_method"],
                    index_rate=chosen_params["index_rate"],
                    protect=chosen_params["protect"],
                    rms_mix_rate=chosen_params["rms_mix_rate"],
                    input_segment_path=str(segment_input_path),
                    output_segment_path=str(output_segment_path),
                    note=note,
                )
            )
            continue

        if route == "male":
            model_name = mixed_audio.MALE_MODEL
            index_name = mixed_audio.MALE_INDEX
            params = mixed_audio.MALE_PARAMS
            try:
                out_sr, converted, _ = mixed_audio.convert_with_vc(
                    male_vc, mixed_audio.MALE_MODEL, male_index_path, segment_input_path, params
                )
                segment_audio = librosa.resample(
                    mixed_audio.to_float_audio(mixed_audio.ensure_mono(converted)),
                    orig_sr=out_sr,
                    target_sr=instrumental_sr,
                )
            except Exception as exc:
                note = f"male_convert_failed:{exc}"
                route = "passthrough"
        elif route == "female":
            model_name = mixed_audio.FEMALE_MODEL
            index_name = mixed_audio.FEMALE_INDEX
            params = mixed_audio.FEMALE_PARAMS
            try:
                out_sr, converted, _ = mixed_audio.convert_with_vc(
                    female_vc, mixed_audio.FEMALE_MODEL, female_index_path, segment_input_path, params
                )
                segment_audio = librosa.resample(
                    mixed_audio.to_float_audio(mixed_audio.ensure_mono(converted)),
                    orig_sr=out_sr,
                    target_sr=instrumental_sr,
                )
            except Exception as exc:
                note = f"female_convert_failed:{exc}"
                route = "passthrough"

        if route == "passthrough":
            segment_audio = librosa.resample(raw_segment.astype(np.float32), orig_sr=vocal_sr, target_sr=instrumental_sr)

        target_len = int(round(duration_sec * instrumental_sr))
        segment_audio = mixed_audio.fit_to_length(segment_audio, target_len).astype(np.float32)
        segment_audio = mixed_audio.apply_edge_fade(segment_audio, instrumental_sr)

        output_segment_path = segment_outputs / f"segment_{segment_id:04d}_{route}.wav"
        sf.write(output_segment_path, segment_audio, instrumental_sr)

        start_out = int(round(start_sec * instrumental_sr))
        end_out = min(start_out + target_len, converted_vocals.shape[0])
        usable = segment_audio[: max(0, end_out - start_out)]
        if usable.size > 0:
            converted_vocals[start_out:end_out, :] += usable[:, None]

        chosen_params = params or mixed_audio.MALE_PARAMS
        decisions.append(
            mixed_audio.SegmentDecision(
                segment_id=segment_id,
                start_sec=round(start_sec, 3),
                end_sec=round(end_sec, 3),
                duration_sec=round(duration_sec, 3),
                median_f0_hz=round(median_f0, 3),
                voiced_frames=voiced_frames,
                voiced_ratio=round(voiced_ratio, 4),
                classification=classification,
                route=route,
                low_confidence=note != "ok",
                model_name=model_name,
                index_name=index_name,
                f0_up_key=chosen_params["f0_up_key"],
                f0_method=chosen_params["f0_method"],
                index_rate=chosen_params["index_rate"],
                protect=chosen_params["protect"],
                rms_mix_rate=chosen_params["rms_mix_rate"],
                input_segment_path=str(segment_input_path),
                output_segment_path=str(output_segment_path),
                note=note,
            )
        )

    converted_vocal_path = job_dir / "converted_vocals.wav"
    converted_mp3_path = job_dir / "converted_vocals.mp3"
    _report(progress_callback, "converting", 0.90, "写出转换后人声")
    vocals_peak = float(np.max(np.abs(converted_vocals))) if converted_vocals.size else 0.0
    if vocals_peak > 1.0:
        converted_vocals /= vocals_peak
    sf.write(converted_vocal_path, converted_vocals, instrumental_sr)
    mixed_audio.export_mp3(converted_vocal_path, converted_mp3_path)

    mixed_audio.write_json(job_dir / "segments.json", [mixed_audio.asdict(item) for item in decisions])
    if decisions:
        mixed_audio.write_csv(job_dir / "segments.csv", decisions)

    _report(progress_callback, "converting", 0.96, "混回伴奏并导出成品")
    final_mix = instrumental_audio.astype(np.float32) + converted_vocals.astype(np.float32)
    peak = float(np.max(np.abs(final_mix))) if final_mix.size else 0.0
    if peak > 1.0:
        final_mix /= peak
    final_mix_wav = job_dir / "final_mix.wav"
    final_mix_mp3 = job_dir / "final_mix.mp3"
    sf.write(final_mix_wav, final_mix, instrumental_sr)
    mixed_audio.export_mp3(final_mix_wav, final_mix_mp3)
    decision_routes = Counter(item.route for item in decisions)
    decision_notes = Counter(item.note for item in decisions)
    _write_stage_summary(
        job_dir,
        "convert_remix",
        {
            "segment_count": len(decisions),
            "route_counts": dict(decision_routes),
            "note_counts": dict(decision_notes),
            "converted_vocals_path": str(converted_mp3_path),
            "final_mix_path": str(final_mix_mp3),
            "instrumental_sample_rate": instrumental_sr,
        },
    )


def execute_single(input_path: Path, output_path: Path, plan: dict) -> tuple[list[dict], list[str]]:
    model = plan["models"][0]
    params = plan["parameters"]
    vc = VC(Config())
    vc.get_vc(model["model"])
    info, audio_tuple = vc.vc_single(
        0,
        str(input_path),
        int(params["f0_up_key"]),
        None,
        params["f0_method"],
        model.get("index", ""),
        model.get("index", ""),
        float(params["index_rate"]),
        int(params["filter_radius"]),
        int(params["resample_sr"]),
        float(params["rms_mix_rate"]),
        float(params["protect"]),
    )
    if audio_tuple is None:
        raise RuntimeError(info)

    tmp_wav = output_path.with_suffix(".auto.tmp.wav")
    sr, audio = audio_tuple
    sf.write(tmp_wav, audio, sr)
    mixed_audio.export_mp3(tmp_wav, output_path)
    tmp_wav.unlink(missing_ok=True)

    duration = float(sf.info(input_path).duration)
    return [
        {
            "start": 0.0,
            "end": round(duration, 4),
            "route": model["role"],
            "median_f0": None,
            "voiced_ratio": None,
            "note": "single_auto",
        }
    ], [info.strip()]


def execute_separate_bgm_voice(
    input_path: Path,
    output_path: Path,
    job_dir: Path,
    plan: dict,
    progress_callback: Callable[[dict], None] | None,
) -> tuple[list[dict], list[str]]:
    params = plan["parameters"]
    target_route = plan.get("target_route", "unknown")
    if target_route not in {"male", "female"}:
        raise RuntimeError(f"separate_bgm_voice requires concrete target_route, got {target_route}")

    chosen_model = next(item for item in plan["models"] if item["role"] == target_route)
    chosen_params = params["male_params"] if target_route == "male" else params["female_params"]

    job_dir.mkdir(parents=True, exist_ok=True)
    _report(progress_callback, "converting", 0.28, "分离人声和伴奏")
    vocal_path, instrumental_path = mixed_audio.run_uvr_split(input_path, job_dir)
    instrumental_audio, instrumental_sr = sf.read(instrumental_path, always_2d=True, dtype="float32")
    vocal_duration = float(sf.info(vocal_path).duration)
    _write_stage_summary(
        job_dir,
        "separate_bgm_voice",
        {
            "target_route": target_route,
            "vocal_path": str(vocal_path),
            "instrumental_path": str(instrumental_path),
            "vocal_duration_seconds": round(vocal_duration, 4),
        },
    )

    _report(progress_callback, "converting", 0.48, f"整条人声转换 | {target_route} | {vocal_duration:.2f}s")
    vc = mixed_audio.instantiate_vc(chosen_model["model"])
    out_sr, converted, info = mixed_audio.convert_with_vc(
        vc,
        chosen_model["model"],
        mixed_audio.resolve_index_path(chosen_model.get("index", "")),
        vocal_path,
        chosen_params,
    )
    converted_mono = librosa.resample(
        mixed_audio.to_float_audio(mixed_audio.ensure_mono(converted)),
        orig_sr=out_sr,
        target_sr=instrumental_sr,
    ).astype(np.float32)
    converted_mono = mixed_audio.fit_to_length(converted_mono, instrumental_audio.shape[0])
    converted_mono = mixed_audio.apply_edge_fade(converted_mono, instrumental_sr)
    converted_vocals = converted_mono[:, None]
    if instrumental_audio.shape[1] > 1:
        converted_vocals = np.repeat(converted_vocals, instrumental_audio.shape[1], axis=1)

    _report(progress_callback, "converting", 0.82, "写出转换后人声")
    converted_vocal_path = job_dir / "converted_vocals.wav"
    converted_mp3_path = job_dir / "converted_vocals.mp3"
    sf.write(converted_vocal_path, converted_vocals, instrumental_sr)
    mixed_audio.export_mp3(converted_vocal_path, converted_mp3_path)

    _report(progress_callback, "converting", 0.92, "混回伴奏并导出成品")
    final_mix = instrumental_audio.astype(np.float32) + converted_vocals.astype(np.float32)
    peak = float(np.max(np.abs(final_mix))) if final_mix.size else 0.0
    if peak > 1.0:
        final_mix /= peak
    final_mix_wav = job_dir / "final_mix.wav"
    final_mix_mp3 = job_dir / "final_mix.mp3"
    sf.write(final_mix_wav, final_mix, instrumental_sr)
    mixed_audio.export_mp3(final_mix_wav, final_mix_mp3)
    shutil.copy2(final_mix_mp3, output_path)
    _write_stage_summary(
        job_dir,
        "separate_bgm_voice_result",
        {
            "converted_vocals_path": str(converted_mp3_path),
            "final_mix_path": str(final_mix_mp3),
            "instrumental_sample_rate": instrumental_sr,
            "target_route": target_route,
        },
    )

    duration = float(sf.info(input_path).duration)
    segments = [
        {
            "segment_id": 0,
            "start": 0.0,
            "end": round(duration, 4),
            "route": target_route,
            "median_f0": None,
            "voiced_ratio": None,
            "note": "separate_bgm_voice",
            "duration_sec": round(duration, 4),
            "segment_type": "speech",
            "speaker_cluster_id": f"{target_route[:1]}1",
            "gender_confidence": 0.9,
        }
    ]
    mixed_audio.write_json(job_dir / "segments.json", segments)
    return segments, [info.strip()]


def execute_long_mixed_pipeline(
    input_path: Path,
    output_path: Path,
    job_dir: Path,
    plan: dict,
    progress_callback: Callable[[dict], None] | None,
) -> tuple[list[dict], list[str]]:
    params = plan["parameters"]
    male_model = next(item for item in plan["models"] if item["role"] == "male")
    female_model = next(item for item in plan["models"] if item["role"] == "female")

    with _temporary_mixed_audio_settings(params, male_model, female_model):
        stages, execution_context = _run_long_mixed_process(input_path, job_dir, progress_callback)
        _execute_long_mixed_convert_and_remix(execution_context, job_dir, progress_callback)
        stages[3]["status"] = "completed"
        stages.append({"name": "collect_outputs", "status": "running"})
        _write_stage_manifest(job_dir, "long_mixed_pipeline", stages)
        _finalize_long_mixed_output(job_dir, output_path)
        segments = _load_long_mixed_segments(job_dir)
        stages[-1]["status"] = "completed"
        stages.append({"name": "done", "status": "completed"})
        _write_stage_manifest(job_dir, "long_mixed_pipeline", stages)
    return segments, []


def execute_clean_voice_segments(
    input_path: Path,
    output_path: Path,
    job_dir: Path,
    plan: dict,
    progress_callback: Callable[[dict], None] | None,
) -> tuple[list[dict], list[str]]:
    params = plan["parameters"]
    male_model = next(item for item in plan["models"] if item["role"] == "male")
    female_model = next(item for item in plan["models"] if item["role"] == "female")

    with _temporary_mixed_audio_settings(params, male_model, female_model):
        job_dir.mkdir(parents=True, exist_ok=True)
        segment_inputs = job_dir / "segments_in"
        segment_outputs = job_dir / "segments_out"
        segment_inputs.mkdir(exist_ok=True)
        segment_outputs.mkdir(exist_ok=True)

        _report(progress_callback, "converting", 0.28, "检测原始音频人声片段")
        source_audio, source_sr = sf.read(input_path, always_2d=True, dtype="float32")
        mono_audio = mixed_audio.ensure_mono(source_audio)
        source_vocals = job_dir / "source_vocals.wav"
        sf.write(source_vocals, mono_audio, source_sr)
        rmvpe = mixed_audio.create_rmvpe(Config())
        analyzed_segments = build_clean_segments(
            source_audio_path=source_vocals,
            source_audio=mono_audio,
            source_sr=source_sr,
            rmvpe=rmvpe,
        )
        _report(progress_callback, "converting", 0.36, f"分析音高与音色，初始片段 {len(analyzed_segments)} 段")
        _report(progress_callback, "converting", 0.42, f"片段合并完成，待处理 {len(analyzed_segments)} 段")
        _report(progress_callback, "converting", 0.45, "加载男声与女声模型")

        male_vc = mixed_audio.instantiate_vc(male_model["model"])
        female_vc = mixed_audio.instantiate_vc(female_model["model"])
        male_index_path = mixed_audio.resolve_index_path(male_model.get("index", ""))
        female_index_path = mixed_audio.resolve_index_path(female_model.get("index", ""))

        converted_audio = np.zeros_like(source_audio, dtype=np.float32)
        decisions: list[mixed_audio.SegmentDecision] = []
        logs: list[str] = []
        total_segments = max(len(analyzed_segments), 1)

        for segment_id, segment_meta in enumerate(analyzed_segments):
            route_label = {
                "male": "男声",
                "female": "女声",
                "passthrough": "直通",
            }.get(segment_meta["route"], segment_meta["route"])
            _report(
                progress_callback,
                "converting",
                0.48 + 0.38 * ((segment_id + 1) / total_segments),
                f"正在转换第 {segment_id + 1}/{total_segments} 段 | {route_label} | {segment_meta['duration_sec']:.2f}s",
            )

            start = int(segment_meta["start"])
            end = int(segment_meta["end"])
            start_sec = start / source_sr
            end_sec = end / source_sr
            duration_sec = float(segment_meta["duration_sec"])
            raw_segment = segment_meta["raw_segment"].astype(np.float32)
            segment_input_path = segment_inputs / f"segment_{segment_id:04d}.wav"
            sf.write(segment_input_path, raw_segment, source_sr)

            classification = segment_meta["classification"]
            route = segment_meta["route"]
            median_f0 = float(segment_meta["median_f0"])
            voiced_frames = int(segment_meta["voiced_frames"])
            voiced_ratio = float(segment_meta["voiced_ratio"])
            note = str(segment_meta["note"])
            model_name = ""
            index_name = ""
            chosen_params = mixed_audio.MALE_PARAMS
            segment_audio = raw_segment.copy()

            if route == "male":
                model_name = male_model["model"]
                index_name = male_model.get("index", "")
                chosen_params = params["male_params"]
                try:
                    out_sr, converted, info = mixed_audio.convert_with_vc(
                        male_vc, model_name, male_index_path, segment_input_path, chosen_params
                    )
                    logs.append(info.strip())
                    segment_audio = librosa.resample(
                        mixed_audio.to_float_audio(mixed_audio.ensure_mono(converted)),
                        orig_sr=out_sr,
                        target_sr=source_sr,
                    )
                except Exception as exc:
                    note = f"male_convert_failed:{exc}"
                    route = "passthrough"
            elif route == "female":
                model_name = female_model["model"]
                index_name = female_model.get("index", "")
                chosen_params = params["female_params"]
                try:
                    out_sr, converted, info = mixed_audio.convert_with_vc(
                        female_vc, model_name, female_index_path, segment_input_path, chosen_params
                    )
                    logs.append(info.strip())
                    segment_audio = librosa.resample(
                        mixed_audio.to_float_audio(mixed_audio.ensure_mono(converted)),
                        orig_sr=out_sr,
                        target_sr=source_sr,
                    )
                except Exception as exc:
                    note = f"female_convert_failed:{exc}"
                    route = "passthrough"

            if route == "passthrough":
                segment_audio = raw_segment.copy()

            target_len = max(0, end - start)
            segment_audio = mixed_audio.fit_to_length(segment_audio, target_len)
            segment_audio = mixed_audio.apply_edge_fade(segment_audio.astype(np.float32), source_sr)

            output_segment_path = segment_outputs / f"segment_{segment_id:04d}_{route}.wav"
            sf.write(output_segment_path, segment_audio, source_sr)

            usable_end = min(start + target_len, converted_audio.shape[0])
            usable = segment_audio[: max(0, usable_end - start)]
            if usable.size > 0:
                converted_audio[start:usable_end, :] += usable[:, None]

            decisions.append(
                mixed_audio.SegmentDecision(
                    segment_id=segment_id,
                    start_sec=round(start_sec, 3),
                    end_sec=round(end_sec, 3),
                    duration_sec=round(duration_sec, 3),
                    median_f0_hz=round(median_f0, 3),
                    voiced_frames=voiced_frames,
                    voiced_ratio=round(voiced_ratio, 4),
                    classification=classification,
                    route=route,
                    low_confidence=note != "ok",
                    model_name=model_name,
                    index_name=index_name,
                    f0_up_key=chosen_params["f0_up_key"],
                    f0_method=chosen_params["f0_method"],
                    index_rate=chosen_params["index_rate"],
                    protect=chosen_params["protect"],
                    rms_mix_rate=chosen_params["rms_mix_rate"],
                    input_segment_path=str(segment_input_path),
                    output_segment_path=str(output_segment_path),
                    note=note,
                )
            )

        _report(progress_callback, "converting", 0.90, "写出 clean voice 分段转换结果")
        peak = float(np.max(np.abs(converted_audio))) if converted_audio.size else 0.0
        if peak > 1.0:
            converted_audio /= peak
        converted_wav = job_dir / "clean_voice_segments.wav"
        converted_mp3 = job_dir / "clean_voice_segments.mp3"
        sf.write(converted_wav, converted_audio, source_sr)
        mixed_audio.export_mp3(converted_wav, converted_mp3)
        shutil.copy2(converted_mp3, output_path)

        mixed_audio.write_json(job_dir / "segments.json", [item.__dict__ for item in decisions])
        if decisions:
            mixed_audio.write_csv(job_dir / "segments.csv", decisions)

        segments = [
            {
                "segment_id": decision.segment_id,
                "start": decision.start_sec,
                "end": decision.end_sec,
                "route": decision.route,
                "median_f0": decision.median_f0_hz,
                "voiced_ratio": decision.voiced_ratio,
                "note": decision.note,
                "duration_sec": decision.duration_sec,
                "segment_type": next(
                    (
                        item["segment_type"]
                        for item in analyzed_segments
                        if int(item["segment_id"]) == int(decision.segment_id)
                    ),
                    "speech",
                ),
                "speaker_cluster_id": next(
                    (
                        item["speaker_cluster_id"]
                        for item in analyzed_segments
                        if int(item["segment_id"]) == int(decision.segment_id)
                    ),
                    "",
                ),
                "gender_confidence": next(
                    (
                        item["gender_confidence"]
                        for item in analyzed_segments
                        if int(item["segment_id"]) == int(decision.segment_id)
                    ),
                    0.0,
                ),
            }
            for decision in decisions
        ]
        return segments, logs
