import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv

try:
    from speechbrain.inference.speaker import EncoderClassifier
except Exception:
    EncoderClassifier = None

ORIGINAL_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from configs.config import Config
from infer.lib.rmvpe import RMVPE
import infer.modules.uvr5.modules as uvr_modules
from infer.modules.uvr5.modules import uvr
from infer.modules.vc.modules import VC
from tools.device_policy import choose_uvr_device


UVR_MODEL = "auto"
UVR_DEVICE_MODE = "auto"
UVR_AUTO_CANDIDATES = ("HP5_only_main_vocal", "HP3_all_vocals")
MALE_MODEL = "man-B.pth"
MALE_INDEX = "man-B.index"
FEMALE_MODEL = "splicegirl_v3_e130_s26520.pth"
FEMALE_INDEX = "splicegirl_v3_e130_s26520.index"

MALE_PARAMS = {
    "f0_up_key": 0,
    "f0_method": "rmvpe",
    "index_rate": 0.75,
    "protect": 0.30,
    "rms_mix_rate": 0.20,
    "filter_radius": 3,
    "resample_sr": 0,
}
FEMALE_PARAMS = {
    "f0_up_key": 0,
    "f0_method": "rmvpe",
    "index_rate": 0.75,
    "protect": 0.25,
    "rms_mix_rate": 0.18,
    "filter_radius": 3,
    "resample_sr": 0,
}
READING_MODE = True

TOP_DB = 35
MIN_SEGMENT_SEC = 0.30
MAX_SEGMENT_SEC = 20.0
TARGET_CHUNK_SEC = 15.0
MERGE_GAP_SEC = 0.35
SEGMENT_PAD_SEC = 0.08
SHORT_SEGMENT_SEC = 0.60
SHORT_SEGMENT_JOIN_GAP_SEC = 0.25
SHORT_SEGMENT_PASSTHROUGH_SEC = 0.60
SHORT_PASSTHROUGH_CONTEXT_SEC = 1.50
SHORT_PASSTHROUGH_CONTEXT_GAP_SEC = 3.00
SHORT_FEMALE_HIGH_F0_PASSTHROUGH_SEC = 1.00
SHORT_FEMALE_HIGH_F0_HZ = 300.0
SAME_ROUTE_MERGE_GAP_SEC = 0.35
SAME_ROUTE_MAX_SEC = 8.0
TIMBRE_SIM_MERGE_GAP_SEC = 0.35
TIMBRE_SIM_MERGE_MAX_SEC = 8.0
TIMBRE_SIM_THRESHOLD = 0.90
TIMBRE_SIM_CROSS_ROUTE_THRESHOLD = 0.96
TIMBRE_CONTINUATION_GAP_SEC = 1.50
TIMBRE_CONTINUATION_MAX_SEC = 12.0
TIMBRE_CONTINUATION_THRESHOLD = 0.93
PASSTHROUGH_BRIDGE_MAX_SEC = 0.80
PASSTHROUGH_BRIDGE_GAP_SEC = 2.00
PASSTHROUGH_BRIDGE_TOTAL_SEC = 12.0
PASSTHROUGH_BRIDGE_SIM_THRESHOLD = 0.88
SHORT_SAME_ROUTE_CLUSTER_SEC = 1.20
SHORT_SAME_ROUTE_GAP_SEC = 2.00
SHORT_SAME_ROUTE_TOTAL_SEC = 6.0
READING_CLUSTER_UNIT_SEC = 3.50
READING_CLUSTER_GAP_SEC = 3.00
READING_CLUSTER_TOTAL_SEC = 14.0
READING_CLUSTER_SIM_THRESHOLD = 0.88
READING_CLUSTER_MAX_INTERNAL_SILENCE_SEC = 1.20
READING_PASSTHROUGH_BRIDGE_MAX_SEC = 0.80
READING_PASSTHROUGH_BRIDGE_TOTAL_SEC = 14.0
READING_PASSTHROUGH_F0_DIFF_HZ = 45.0
CONTEXT_SMOOTH_MAX_SEC = 1.20
CONTEXT_SMOOTH_GAP_SEC = 3.00
CONTEXT_SMOOTH_LOW_CONF_MAX_SEC = 18.0
CONTEXT_SMOOTH_LOW_CONF_GAP_SEC = 10.0
CONTEXT_SMOOTH_RUN_TOTAL_SEC = 30.0
CONTEXT_SMOOTH_LOW_VOICED_RATIO = 0.35
SPEAKER_SIM_THRESHOLD = 0.58
SPEAKER_CONTINUATION_THRESHOLD = 0.52
SPEAKER_CROSS_ROUTE_THRESHOLD = 0.72
UNVOICED_MIN_FRAMES = 10
UNVOICED_MIN_RATIO = 0.15
DECISION_F0_HZ = 190.0
HIGH_CONF_MALE_MAX_HZ = 165.0
HIGH_CONF_FEMALE_MIN_HZ = 190.0
MUSIC_RESIDUAL_MAX_VOICED_RATIO = 0.30
MUSIC_RESIDUAL_MIN_F0_HZ = 320.0
SEGMENT_EDGE_FADE_SEC = 0.025

SPEAKER_ENCODER = None
SPEAKER_ENCODER_FAILED = False
UVR_LOCK = threading.Lock()


@dataclass
class SegmentDecision:
    segment_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    median_f0_hz: float
    voiced_frames: int
    voiced_ratio: float
    classification: str
    route: str
    low_confidence: bool
    model_name: str
    index_name: str
    f0_up_key: int
    f0_method: str
    index_rate: float
    protect: float
    rms_mix_rate: float
    input_segment_path: str
    output_segment_path: str
    note: str


def analyze_intervals(
    audio: np.ndarray,
    sr: int,
    intervals: List[Tuple[int, int]],
    rmvpe: RMVPE,
) -> List[dict]:
    analyzed = []
    for start, end in intervals:
        raw_segment = audio[start:end]
        classification, route, median_f0, voiced_frames, voiced_ratio, note = classify_segment(
            raw_segment, sr, rmvpe
        )
        duration_sec = (end - start) / sr
        analyzed.append(
            {
                "start": start,
                "end": end,
                "duration_sec": duration_sec,
                "raw_segment": raw_segment,
                "classification": classification,
                "route": route,
                "median_f0": median_f0,
                "voiced_frames": voiced_frames,
                "voiced_ratio": voiced_ratio,
                "note": note,
            }
        )
    smooth_context_routes(analyzed, sr)
    return analyzed


def absorb_short_passthrough_segments(segments: List[dict], sr: int) -> None:
    if not READING_MODE:
        return
    for idx, item in enumerate(segments):
        is_forced_short = item["duration_sec"] < SHORT_SEGMENT_PASSTHROUGH_SEC
        is_context_passthrough = (
            item["route"] == "passthrough"
            and item["duration_sec"] <= SHORT_PASSTHROUGH_CONTEXT_SEC
            and item.get("note") == "low_voice"
        )
        is_short_female_exclaim = (
            item["route"] == "female"
            and item["duration_sec"] < SHORT_FEMALE_HIGH_F0_PASSTHROUGH_SEC
            and item["median_f0"] >= SHORT_FEMALE_HIGH_F0_HZ
        )
        if not (is_forced_short or is_context_passthrough or is_short_female_exclaim):
            continue

        prev_seg = segments[idx - 1] if idx > 0 else None
        next_seg = segments[idx + 1] if idx + 1 < len(segments) else None
        prev_gap_sec = (
            (item["start"] - prev_seg["end"]) / sr if prev_seg is not None else None
        )
        next_gap_sec = (
            (next_seg["start"] - item["end"]) / sr if next_seg is not None else None
        )
        candidate = None

        if (
            prev_seg
            and next_seg
            and prev_seg["route"] == next_seg["route"]
            and prev_seg["route"] != "passthrough"
            and (prev_gap_sec is None or prev_gap_sec <= SHORT_PASSTHROUGH_CONTEXT_GAP_SEC)
            and (next_gap_sec is None or next_gap_sec <= SHORT_PASSTHROUGH_CONTEXT_GAP_SEC)
        ):
            candidate = prev_seg
        elif (
            prev_seg
            and prev_seg["route"] != "passthrough"
            and (prev_gap_sec is None or prev_gap_sec <= SHORT_PASSTHROUGH_CONTEXT_GAP_SEC)
        ):
            candidate = prev_seg
        elif (
            next_seg
            and next_seg["route"] != "passthrough"
            and (next_gap_sec is None or next_gap_sec <= SHORT_PASSTHROUGH_CONTEXT_GAP_SEC)
        ):
            candidate = next_seg

        if candidate:
            item["route"] = candidate["route"]
            item["classification"] = candidate["classification"]
            item["note"] = (
                "short_absorbed_into_context"
                if is_forced_short
                else "short_passthrough_absorbed"
                if is_context_passthrough
                else "short_high_female_absorbed"
            )
            if candidate.get("median_f0", 0.0) > 0:
                item["median_f0"] = float(candidate["median_f0"])


def merge_context_absorbed_segments(
    audio: np.ndarray,
    sr: int,
    segments: List[dict],
) -> List[dict]:
    if not READING_MODE:
        return segments
    if not segments:
        return []

    merged = [segments[0].copy()]
    for item in segments[1:]:
        prev = merged[-1]
        gap_sec = (item["start"] - prev["end"]) / sr
        total_sec = (item["end"] - prev["start"]) / sr
        should_merge = (
            item["route"] == prev["route"]
            and item["route"] != "passthrough"
            and should_merge_reading_cluster(
                prev_duration_sec=prev["duration_sec"],
                item_duration_sec=item["duration_sec"],
                gap_sec=gap_sec,
                merged_duration_sec=total_sec,
                timbre_similarity=1.0,
            )
        )
        if should_merge:
            prev["end"] = item["end"]
            prev["duration_sec"] = (prev["end"] - prev["start"]) / sr
            prev["raw_segment"] = audio[prev["start"] : prev["end"]]
            prev["voiced_frames"] = int(prev["voiced_frames"] + item["voiced_frames"])
            prev["voiced_ratio"] = max(float(prev["voiced_ratio"]), float(item["voiced_ratio"]))
            prev_f0 = float(prev.get("median_f0", 0.0))
            item_f0 = float(item.get("median_f0", 0.0))
            if prev_f0 > 0 and item_f0 > 0:
                prev["median_f0"] = (prev_f0 + item_f0) / 2.0
            elif item_f0 > 0:
                prev["median_f0"] = item_f0
            if "absorbed" in item.get("note", ""):
                prev["note"] = item["note"]
        else:
            merged.append(item.copy())
    return merged


def should_merge_reading_cluster(
    prev_duration_sec: float,
    item_duration_sec: float,
    gap_sec: float,
    merged_duration_sec: float,
    timbre_similarity: float,
) -> bool:
    return (
        READING_MODE
        and prev_duration_sec <= READING_CLUSTER_UNIT_SEC
        and item_duration_sec <= READING_CLUSTER_UNIT_SEC
        and gap_sec <= READING_CLUSTER_GAP_SEC
        and gap_sec <= READING_CLUSTER_MAX_INTERNAL_SILENCE_SEC
        and merged_duration_sec <= READING_CLUSTER_TOTAL_SEC
        and timbre_similarity >= SPEAKER_CONTINUATION_THRESHOLD
    )


def ensure_environment() -> None:
    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("weight_root", "assets/weights")
    os.environ.setdefault("weight_uvr5_root", "assets/uvr5_weights")
    os.environ.setdefault("index_root", "logs")
    os.environ.setdefault("outside_index_root", "assets/indices")
    os.environ.setdefault("rmvpe_root", "assets/rmvpe")
    os.environ.setdefault("TEMP", "/tmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process long mixed audio with UVR split, gender routing, and RVC conversion."
    )
    parser.add_argument("input_audio", type=str, help="Path to the source audio file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to outputs/<input_stem>.",
    )
    parser.add_argument(
        "--skip-remix",
        action="store_true",
        help="Only produce separated and converted vocal tracks without final remix.",
    )
    return parser.parse_args(ORIGINAL_ARGV)


def canonical_output_dir(input_audio: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return (REPO_ROOT / "outputs" / input_audio.stem).resolve()


def find_single_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} under {directory}")
    return matches[-1]


def run_uvr_paths(
    paths: List[SimpleNamespace],
    uvr_vocal_dir: Path,
    uvr_inst_dir: Path,
    temp_dir: Path,
    model_name: str | None = None,
) -> List[str]:
    messages: List[str] = []
    selected_model = model_name or UVR_MODEL
    with UVR_LOCK:
        old_uvr_device = uvr_modules.config.device
        old_uvr_is_half = uvr_modules.config.is_half
        old_temp = os.environ.get("TEMP")
        try:
            runtime_config = Config()
            uvr_device, uvr_is_half = choose_uvr_device(
                UVR_DEVICE_MODE, runtime_config.device, runtime_config.is_half
            )
            temp_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TEMP"] = str(temp_dir)
            uvr_modules.config.device = uvr_device
            uvr_modules.config.is_half = uvr_is_half
            for message in uvr(
                selected_model,
                "",
                str(uvr_vocal_dir),
                paths,
                str(uvr_inst_dir),
                10,
                "wav",
            ):
                if message:
                    messages.append(message)
        finally:
            uvr_modules.config.device = old_uvr_device
            uvr_modules.config.is_half = old_uvr_is_half
            if old_temp is None:
                os.environ.pop("TEMP", None)
            else:
                os.environ["TEMP"] = old_temp
    return messages


def find_uvr_pair(input_name: str, raw_dirs: Tuple[Path, Path]) -> Tuple[Path | None, Path | None]:
    vocal_matches = sorted(
        path for directory in raw_dirs for path in directory.glob(f"vocal_{input_name}*.wav")
    )
    inst_matches = sorted(
        path
        for directory in raw_dirs
        for path in directory.glob(f"instrument_{input_name}*.wav")
    )
    return (
        vocal_matches[-1] if vocal_matches else None,
        inst_matches[-1] if inst_matches else None,
    )


def run_chunked_uvr_split(
    input_audio: Path,
    output_dir: Path,
    uvr_vocal_dir: Path,
    uvr_inst_dir: Path,
    model_name: str,
) -> Tuple[Path, Path, List[str]]:
    chunk_dir = output_dir / "uvr_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(input_audio, always_2d=True)
    chunk_size = int(30.0 * sr)
    chunk_paths = []
    for idx, start in enumerate(range(0, len(audio), chunk_size)):
        chunk = audio[start : start + chunk_size]
        chunk_path = chunk_dir / f"chunk_{idx:04d}.wav"
        if not chunk_path.exists():
            sf.write(chunk_path, chunk, sr)
        chunk_paths.append(chunk_path.resolve())

    messages = run_uvr_paths(
        [SimpleNamespace(name=str(path)) for path in chunk_paths],
        uvr_vocal_dir,
        uvr_inst_dir,
        output_dir / "uvr_tmp",
        model_name,
    )

    raw_dirs = (uvr_vocal_dir, uvr_inst_dir)
    vocal_chunks = []
    inst_chunks = []
    for chunk_path in chunk_paths:
        vocal_raw, instrumental_raw = find_uvr_pair(chunk_path.name, raw_dirs)
        if vocal_raw is None or instrumental_raw is None:
            raise RuntimeError(f"UVR chunk output missing for {chunk_path.name}.")
        vocal_audio, vocal_sr = sf.read(vocal_raw, always_2d=True)
        inst_audio, inst_sr = sf.read(instrumental_raw, always_2d=True)
        if vocal_sr != sr or inst_sr != sr:
            raise RuntimeError(
                f"Unexpected UVR chunk sample rate for {chunk_path.name}: "
                f"vocal={vocal_sr}, instrumental={inst_sr}, expected={sr}"
            )
        vocal_chunks.append(vocal_audio)
        inst_chunks.append(inst_audio)

    vocal_path = output_dir / "vocals.wav"
    instrumental_path = output_dir / "instrumental.wav"
    sf.write(vocal_path, np.concatenate(vocal_chunks, axis=0)[: len(audio)], sr)
    sf.write(instrumental_path, np.concatenate(inst_chunks, axis=0)[: len(audio)], sr)
    return vocal_path, instrumental_path, messages


def run_uvr_split_for_model(
    input_audio: Path, output_dir: Path, model_name: str
) -> Tuple[Path, Path]:
    uvr_vocal_dir = output_dir / "uvr_vocal_raw"
    uvr_inst_dir = output_dir / "uvr_inst_raw"
    uvr_vocal_dir.mkdir(parents=True, exist_ok=True)
    uvr_inst_dir.mkdir(parents=True, exist_ok=True)
    vocal_path = output_dir / "vocals.wav"
    instrumental_path = output_dir / "instrumental.wav"

    if vocal_path.exists() and instrumental_path.exists():
        return vocal_path, instrumental_path

    raw_dirs = (uvr_vocal_dir, uvr_inst_dir)
    existing_vocal, existing_inst = find_uvr_pair(input_audio.name, raw_dirs)
    if existing_vocal and existing_inst:
        shutil.copy2(existing_vocal, vocal_path)
        shutil.copy2(existing_inst, instrumental_path)
        return vocal_path, instrumental_path

    audio_info = sf.info(input_audio)
    if audio_info.duration > 90:
        vocal_path, instrumental_path, _ = run_chunked_uvr_split(
            input_audio,
            output_dir,
            uvr_vocal_dir,
            uvr_inst_dir,
            model_name,
        )
        return vocal_path, instrumental_path

    paths = [SimpleNamespace(name=str(input_audio))]
    messages = run_uvr_paths(
        paths, uvr_vocal_dir, uvr_inst_dir, output_dir / "uvr_tmp", model_name
    )

    vocal_raw = None
    instrumental_raw = None
    deadline = time.time() + 8.0
    while time.time() < deadline:
        vocal_raw, instrumental_raw = find_uvr_pair(input_audio.name, raw_dirs)
        if vocal_raw and instrumental_raw:
            break
        time.sleep(0.25)

    if vocal_raw is None or instrumental_raw is None:
        success_reported = any("Success" in message for message in messages)
        details = "\n".join(messages[-3:]).strip()
        if success_reported:
            raise RuntimeError(
                "UVR reported success but output files were not found."
                + (f"\nRecent messages:\n{details}" if details else "")
            )
        raise RuntimeError(
            "UVR split did not produce output files."
            + (f"\nRecent messages:\n{details}" if details else "")
        )

    shutil.copy2(vocal_raw, vocal_path)
    shutil.copy2(instrumental_raw, instrumental_path)
    return vocal_path, instrumental_path


def score_uvr_vocal_candidate(vocal_path: Path) -> dict:
    audio, sr = librosa.load(vocal_path, sr=None, mono=True)
    intervals = merge_intervals(librosa.effects.split(audio, top_db=TOP_DB), sr)
    voiced_samples = sum(end - start for start, end in intervals)
    duration_samples = max(len(audio), 1)
    return {
        "coverage": voiced_samples / duration_samples,
        "rms": float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0,
        "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
    }


def score_uvr_region_balance(vocal_path: Path, instrumental_path: Path) -> dict:
    vocal_audio, sr = librosa.load(vocal_path, sr=None, mono=True)
    instrumental_audio, inst_sr = librosa.load(instrumental_path, sr=sr, mono=True)
    if inst_sr != sr:
        raise RuntimeError(f"Unexpected UVR region sample rate mismatch: vocal={sr}, instrumental={inst_sr}")

    def region_metrics(start_sec: float, end_sec: float) -> dict | None:
        if sr <= 0:
            return None
        start = max(0, int(start_sec * sr))
        end = min(len(vocal_audio), int(end_sec * sr))
        if end - start < int(2.0 * sr):
            return None
        vocal_region = vocal_audio[start:end]
        instrumental_region = instrumental_audio[start:end]
        vocal_rms = float(np.sqrt(np.mean(vocal_region**2))) if vocal_region.size else 0.0
        instrumental_rms = (
            float(np.sqrt(np.mean(instrumental_region**2))) if instrumental_region.size else 0.0
        )
        return {
            "vocal_rms": vocal_rms,
            "instrumental_rms": instrumental_rms,
            "vocal_to_instrumental_ratio": vocal_rms / max(instrumental_rms, 1e-9),
        }

    regions = {}
    for name, start_sec, end_sec in (
        ("intro_0_12", 0.0, 12.0),
        ("early_7_18", 7.0, 18.0),
    ):
        metrics = region_metrics(start_sec, end_sec)
        if metrics is not None:
            regions[name] = metrics
    return regions


def select_auto_uvr_model(metrics: dict, source_seconds: float) -> tuple[str, str]:
    hp5 = metrics.get("HP5_only_main_vocal", {})
    hp3 = metrics.get("HP3_all_vocals", {})
    hp3_coverage = float(hp3.get("coverage", 0.0))
    hp5_coverage = float(hp5.get("coverage", 0.0))
    hp3_looks_continuous = hp3_coverage > 0.78
    hp5_keeps_enough_vocal = hp5_coverage > 0.10
    hp5_reduces_bleed = hp5_coverage <= hp3_coverage - 0.10
    hp3_strong_music_bleed = hp3_coverage >= 0.90 and hp5_coverage <= 0.82

    hp5_regions = hp5.get("regions") or {}
    hp3_regions = hp3.get("regions") or {}
    hp5_intro = hp5_regions.get("intro_0_12") or {}
    hp3_intro = hp3_regions.get("intro_0_12") or {}
    hp5_early = hp5_regions.get("early_7_18") or {}
    hp3_early = hp3_regions.get("early_7_18") or {}
    hp3_intro_ratio = float(hp3_intro.get("vocal_to_instrumental_ratio", 0.0))
    hp5_intro_ratio = float(hp5_intro.get("vocal_to_instrumental_ratio", 0.0))
    hp3_early_ratio = float(hp3_early.get("vocal_to_instrumental_ratio", 0.0))
    hp5_early_ratio = float(hp5_early.get("vocal_to_instrumental_ratio", 0.0))

    intro_music_bleed = (
        source_seconds >= 60.0
        and hp5_keeps_enough_vocal
        and hp3_intro_ratio >= 1.05
        and hp5_intro_ratio <= 0.95
        and hp3_early_ratio >= 1.15
        and hp5_early_ratio <= 0.90
    )
    if intro_music_bleed:
        return (
            "HP5_only_main_vocal",
            "HP3 vocal track looked music-heavy in the intro window; HP5 kept more of that region in instrumental.",
        )

    if hp3_looks_continuous and hp5_keeps_enough_vocal and (
        hp5_reduces_bleed or hp3_strong_music_bleed
    ):
        return (
            "HP5_only_main_vocal",
            "HP3 vocal track looked nearly continuous; HP5 reduced likely music bleed.",
        )
    return (
        "HP3_all_vocals",
        "HP3 did not show strong continuous-vocal over-detection on the sample.",
    )


def choose_auto_uvr_model(input_audio: Path, output_dir: Path) -> str:
    audio_info = sf.info(input_audio)
    audio, sr = sf.read(input_audio, always_2d=True, dtype="float32")
    sample_len = min(len(audio), int(60.0 * sr))
    sample_path = output_dir / "uvr_auto_sample.wav"
    sf.write(sample_path, audio[:sample_len], sr)

    metrics = {}
    for model_name in UVR_AUTO_CANDIDATES:
        candidate_dir = output_dir / "uvr_auto" / model_name
        vocal_path, instrumental_path = run_uvr_split_for_model(
            sample_path, candidate_dir, model_name
        )
        metrics[model_name] = score_uvr_vocal_candidate(vocal_path)
        metrics[model_name]["regions"] = score_uvr_region_balance(vocal_path, instrumental_path)

    selected_model, reason = select_auto_uvr_model(metrics, audio_info.duration)

    metadata = {
        "selected_model": selected_model,
        "reason": reason,
        "sample_seconds": sample_len / sr if sr else 0.0,
        "source_seconds": audio_info.duration,
        "metrics": metrics,
    }
    with open(output_dir / "uvr_auto_selection.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return selected_model


def run_auto_uvr_split(input_audio: Path, output_dir: Path) -> Tuple[Path, Path]:
    vocal_path = output_dir / "vocals.wav"
    instrumental_path = output_dir / "instrumental.wav"
    if vocal_path.exists() and instrumental_path.exists():
        return vocal_path, instrumental_path

    selected_model = choose_auto_uvr_model(input_audio, output_dir)
    return run_uvr_split_for_model(input_audio, output_dir, selected_model)


def run_uvr_split(input_audio: Path, output_dir: Path) -> Tuple[Path, Path]:
    if UVR_MODEL == "auto":
        return run_auto_uvr_split(input_audio, output_dir)
    return run_uvr_split_for_model(input_audio, output_dir, UVR_MODEL)


def merge_intervals(intervals: np.ndarray, sr: int) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    gap_limit = int(MERGE_GAP_SEC * sr)
    for start, end in intervals.tolist():
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap_limit:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def split_long_interval(
    audio: np.ndarray,
    start: int,
    end: int,
    sr: int,
) -> List[Tuple[int, int]]:
    duration_sec = (end - start) / sr
    if duration_sec <= MAX_SEGMENT_SEC:
        return [(start, end)]

    target_chunk = int(TARGET_CHUNK_SEC * sr)
    max_chunk = int(MAX_SEGMENT_SEC * sr)
    min_chunk = int(6.0 * sr)
    min_tail = int(5.0 * sr)
    chunks: List[Tuple[int, int]] = []
    cursor = start

    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    frame_times = librosa.frames_to_samples(np.arange(len(rms)), hop_length=hop_length)

    while end - cursor > max_chunk:
        preferred = cursor + target_chunk
        search_start = max(cursor + min_chunk, preferred - int(5.0 * sr))
        search_end = min(cursor + max_chunk, preferred + int(5.0 * sr), end - min_tail)
        if search_end <= search_start:
            chunk_end = min(preferred, end)
        else:
            mask = (frame_times >= search_start) & (frame_times <= search_end)
            if np.any(mask):
                candidate_frames = np.flatnonzero(mask)
                best_frame = candidate_frames[np.argmin(rms[candidate_frames])]
                chunk_end = int(frame_times[best_frame])
            else:
                chunk_end = min(preferred, end)

        if chunk_end <= cursor:
            chunk_end = min(cursor + target_chunk, end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    if chunks and (end - cursor) < min_tail:
        prev_start, _ = chunks[-1]
        chunks[-1] = (prev_start, end)
    else:
        chunks.append((cursor, end))
    return chunks


def merge_short_intervals(intervals: List[Tuple[int, int]], sr: int) -> List[Tuple[int, int]]:
    if not intervals:
        return []

    short_limit = int(SHORT_SEGMENT_SEC * sr)
    join_gap = int(SHORT_SEGMENT_JOIN_GAP_SEC * sr)
    merged = [list(item) for item in intervals]

    idx = 0
    while idx < len(merged):
        start, end = merged[idx]
        if (end - start) >= short_limit:
            idx += 1
            continue

        prev_gap = None
        next_gap = None
        if idx > 0:
            prev_gap = start - merged[idx - 1][1]
        if idx + 1 < len(merged):
            next_gap = merged[idx + 1][0] - end

        merged_into_neighbor = False
        if prev_gap is not None and prev_gap <= join_gap and (
            next_gap is None or prev_gap <= next_gap
        ):
            merged[idx - 1][1] = end
            del merged[idx]
            idx = max(0, idx - 1)
            merged_into_neighbor = True
        elif next_gap is not None and next_gap <= join_gap:
            merged[idx + 1][0] = start
            del merged[idx]
            merged_into_neighbor = True

        if not merged_into_neighbor:
            idx += 1

    return [tuple(item) for item in merged]


def detect_segments(vocal_path: Path) -> Tuple[np.ndarray, int, List[Tuple[int, int]]]:
    audio, sr = librosa.load(vocal_path, sr=None, mono=True)
    raw_intervals = librosa.effects.split(audio, top_db=TOP_DB)
    merged = merge_intervals(raw_intervals, sr)
    intervals: List[Tuple[int, int]] = []
    pad = int(SEGMENT_PAD_SEC * sr)
    for start, end in merged:
        if (end - start) / sr < MIN_SEGMENT_SEC:
            continue
        start = max(0, start - pad)
        end = min(len(audio), end + pad)
        intervals.extend(split_long_interval(audio, start, end, sr))
    intervals = merge_short_intervals(intervals, sr)
    return audio, sr, intervals


def create_rmvpe(config: Config) -> RMVPE:
    model_path = str(REPO_ROOT / os.environ["rmvpe_root"] / "rmvpe.pt")
    return RMVPE(model_path, is_half=config.is_half, device=config.device)


def classify_segment(
    audio_segment: np.ndarray, sr: int, rmvpe: RMVPE
) -> Tuple[str, str, float, int, float, str]:
    if sr != 16000:
        segment_16k = librosa.resample(
            audio_segment.astype(np.float32), orig_sr=sr, target_sr=16000
        )
    else:
        segment_16k = audio_segment.astype(np.float32)

    f0 = rmvpe.infer_from_audio(segment_16k, thred=0.03)
    voiced = f0[f0 > 0]
    voiced_frames = int(voiced.shape[0])
    voiced_ratio = voiced_frames / max(len(f0), 1)
    if voiced_frames < UNVOICED_MIN_FRAMES or voiced_ratio < UNVOICED_MIN_RATIO:
        return "unknown", "passthrough", 0.0, voiced_frames, voiced_ratio, "low_voice"

    median_f0 = float(np.median(voiced))
    if (
        voiced_ratio < MUSIC_RESIDUAL_MAX_VOICED_RATIO
        and median_f0 >= MUSIC_RESIDUAL_MIN_F0_HZ
    ):
        return "unknown", "passthrough", median_f0, voiced_frames, voiced_ratio, "music_residual"
    classification = "female" if median_f0 >= DECISION_F0_HZ else "male"
    high_confidence = (
        median_f0 <= HIGH_CONF_MALE_MAX_HZ or median_f0 >= HIGH_CONF_FEMALE_MIN_HZ
    )
    note = "ok" if high_confidence else "borderline_f0"
    return classification, classification, median_f0, voiced_frames, voiced_ratio, note


def compute_timbre_embedding(audio_segment: np.ndarray, sr: int) -> np.ndarray:
    y = audio_segment.astype(np.float32)
    if y.size == 0:
        return np.zeros(40, dtype=np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / max(np.max(np.abs(y)), 1e-6)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features = np.vstack([mfcc, delta, delta2, zcr, centroid, rolloff])
    emb = np.concatenate([features.mean(axis=1), features.std(axis=1)]).astype(np.float32)
    norm = float(np.linalg.norm(emb))
    if norm <= 1e-8:
        return emb
    return emb / norm


def get_speaker_encoder():
    global SPEAKER_ENCODER, SPEAKER_ENCODER_FAILED
    if SPEAKER_ENCODER is not None or SPEAKER_ENCODER_FAILED:
        return SPEAKER_ENCODER
    if EncoderClassifier is None:
        SPEAKER_ENCODER_FAILED = True
        return None
    try:
        savedir = str(REPO_ROOT / "TEMP" / "speechbrain_spkrec")
        SPEAKER_ENCODER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": "cpu"},
        )
        return SPEAKER_ENCODER
    except Exception:
        SPEAKER_ENCODER_FAILED = True
        return None


def compute_speaker_embedding(audio_segment: np.ndarray, sr: int) -> np.ndarray | None:
    encoder = get_speaker_encoder()
    if encoder is None:
        return None
    y = audio_segment.astype(np.float32)
    if y.size == 0:
        return None
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / max(peak, 1e-6)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    tensor = torch.from_numpy(y).unsqueeze(0)
    with torch.no_grad():
        emb = encoder.encode_batch(tensor).squeeze().detach().cpu().numpy().astype(np.float32)
    norm = float(np.linalg.norm(emb))
    if norm <= 1e-8:
        return None
    return emb / norm


def compute_segment_embedding(audio_segment: np.ndarray, sr: int) -> np.ndarray:
    speaker_embedding = compute_speaker_embedding(audio_segment, sr)
    if speaker_embedding is not None:
        return speaker_embedding
    return compute_timbre_embedding(audio_segment, sr)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


def f0_is_close(f0_a: float, f0_b: float, max_diff_hz: float) -> bool:
    if f0_a <= 0 or f0_b <= 0:
        return False
    return abs(float(f0_a) - float(f0_b)) <= max_diff_hz


def smooth_context_routes(provisional: List[dict], sr: int) -> None:
    if not READING_MODE:
        return
    if len(provisional) < 3:
        return

    max_gap = int(CONTEXT_SMOOTH_GAP_SEC * sr)
    max_single_duration = CONTEXT_SMOOTH_MAX_SEC
    max_low_conf_duration = CONTEXT_SMOOTH_LOW_CONF_MAX_SEC
    max_low_conf_gap = int(CONTEXT_SMOOTH_LOW_CONF_GAP_SEC * sr)
    max_run_duration = CONTEXT_SMOOTH_RUN_TOTAL_SEC

    def is_context_candidate(item: dict) -> bool:
        note = str(item.get("note") or "")
        duration_sec = float(item.get("duration_sec") or 0.0)
        voiced_ratio = float(item.get("voiced_ratio") or 0.0)
        low_confidence = (
            item.get("route") == "passthrough"
            or note in {"borderline_f0", "short_passthrough", "low_voice", "music_residual"}
            or voiced_ratio < CONTEXT_SMOOTH_LOW_VOICED_RATIO
        )
        if not low_confidence:
            return False
        duration_limit = max_low_conf_duration if voiced_ratio < CONTEXT_SMOOTH_LOW_VOICED_RATIO or note == "borderline_f0" else max_single_duration
        return duration_sec <= duration_limit

    def prefers_low_conf_gap(item: dict) -> bool:
        note = str(item.get("note") or "")
        voiced_ratio = float(item.get("voiced_ratio") or 0.0)
        return voiced_ratio < CONTEXT_SMOOTH_LOW_VOICED_RATIO or note == "borderline_f0"

    i = 1
    while i < len(provisional) - 1:
        item = provisional[i]
        if not is_context_candidate(item):
            i += 1
            continue

        start_idx = i
        end_idx = i
        run_duration = float(item.get("duration_sec") or 0.0)
        run_gap_limit = max_low_conf_gap if prefers_low_conf_gap(item) else max_gap
        while end_idx + 1 < len(provisional) - 1:
            nxt_item = provisional[end_idx + 1]
            if not is_context_candidate(nxt_item):
                break
            if nxt_item["start"] - provisional[end_idx]["end"] > run_gap_limit:
                break
            projected_duration = run_duration + float(nxt_item.get("duration_sec") or 0.0)
            if projected_duration > max_run_duration:
                break
            if prefers_low_conf_gap(nxt_item):
                run_gap_limit = max(run_gap_limit, max_low_conf_gap)
            end_idx += 1
            run_duration = projected_duration

        prev = provisional[start_idx - 1]
        nxt = provisional[end_idx + 1]
        if prev["route"] == nxt["route"] and prev["route"] != "passthrough":
            left_gap = provisional[start_idx]["start"] - prev["end"]
            right_gap = nxt["start"] - provisional[end_idx]["end"]
            if left_gap <= run_gap_limit and right_gap <= run_gap_limit:
                neighbor_f0 = [
                    float(prev.get("median_f0", 0.0)),
                    float(nxt.get("median_f0", 0.0)),
                ]
                voiced_neighbor_f0 = [f for f in neighbor_f0 if f > 0]
                smoothed_f0 = (
                    float(sum(voiced_neighbor_f0) / len(voiced_neighbor_f0))
                    if voiced_neighbor_f0
                    else 0.0
                )
                for idx in range(start_idx, end_idx + 1):
                    current = provisional[idx]
                    if current["route"] == prev["route"] and str(current.get("note") or "") == "ok":
                        continue
                    current["route"] = prev["route"]
                    current["classification"] = prev["classification"]
                    current["note"] = f"context_smoothed_{prev['route']}"
                    if smoothed_f0 > 0:
                        current["median_f0"] = smoothed_f0
        i = end_idx + 1


def merge_adjacent_same_route(
    audio: np.ndarray,
    sr: int,
    intervals: List[Tuple[int, int]],
    rmvpe: RMVPE,
) -> List[Tuple[int, int]]:
    if not intervals:
        return []

    provisional = []
    for start, end in intervals:
        seg = audio[start:end]
        classification, route, median_f0, voiced_frames, voiced_ratio, note = classify_segment(
            seg, sr, rmvpe
        )
        embedding = compute_segment_embedding(seg, sr)
        provisional.append(
            {
                "start": start,
                "end": end,
                "route": route,
                "classification": classification,
                "note": note,
                "median_f0": median_f0,
                "duration_sec": (end - start) / sr,
                "embedding": embedding,
            }
        )

    smooth_context_routes(provisional, sr)

    merged = [provisional[0].copy()]
    gap_limit = int(SAME_ROUTE_MERGE_GAP_SEC * sr)
    max_len = int(SAME_ROUTE_MAX_SEC * sr)
    timbre_gap_limit = int(TIMBRE_SIM_MERGE_GAP_SEC * sr)
    timbre_max_len = int(TIMBRE_SIM_MERGE_MAX_SEC * sr)
    continuation_gap_limit = int(TIMBRE_CONTINUATION_GAP_SEC * sr)
    continuation_max_len = int(TIMBRE_CONTINUATION_MAX_SEC * sr)
    passthrough_bridge_max = int(PASSTHROUGH_BRIDGE_MAX_SEC * sr)
    passthrough_bridge_gap = int(PASSTHROUGH_BRIDGE_GAP_SEC * sr)
    passthrough_bridge_total = int(PASSTHROUGH_BRIDGE_TOTAL_SEC * sr)
    reading_passthrough_bridge_max = int(READING_PASSTHROUGH_BRIDGE_MAX_SEC * sr)
    reading_passthrough_bridge_total = int(READING_PASSTHROUGH_BRIDGE_TOTAL_SEC * sr)
    short_cluster_len = int(SHORT_SAME_ROUTE_CLUSTER_SEC * sr)
    short_cluster_gap = int(SHORT_SAME_ROUTE_GAP_SEC * sr)
    short_cluster_total = int(SHORT_SAME_ROUTE_TOTAL_SEC * sr)
    reading_cluster_unit = int(READING_CLUSTER_UNIT_SEC * sr)
    reading_cluster_gap = int(READING_CLUSTER_GAP_SEC * sr)
    reading_cluster_total = int(READING_CLUSTER_TOTAL_SEC * sr)

    i = 1
    while i < len(provisional):
        item = provisional[i]
        prev = merged[-1]
        gap = item["start"] - prev["end"]
        merged_len = item["end"] - prev["start"]
        same_route_merge = (
            item["route"] == prev["route"]
            and item["route"] != "passthrough"
            and gap <= gap_limit
            and merged_len <= max_len
        )
        timbre_sim = cosine_similarity(prev["embedding"], item["embedding"])
        cross_route_merge = (
            item["route"] != "passthrough"
            and prev["route"] != "passthrough"
            and gap <= timbre_gap_limit
            and merged_len <= timbre_max_len
            and (
                (item["route"] == prev["route"] and timbre_sim >= SPEAKER_SIM_THRESHOLD)
                or timbre_sim >= SPEAKER_CROSS_ROUTE_THRESHOLD
            )
        )
        continuation_merge = (
            item["route"] == prev["route"]
            and item["route"] != "passthrough"
            and gap <= continuation_gap_limit
            and merged_len <= continuation_max_len
            and timbre_sim >= SPEAKER_CONTINUATION_THRESHOLD
        )
        short_cluster_merge = (
            item["route"] == prev["route"]
            and item["route"] != "passthrough"
            and prev["duration_sec"] <= SHORT_SAME_ROUTE_CLUSTER_SEC
            and item["duration_sec"] <= SHORT_SAME_ROUTE_CLUSTER_SEC
            and gap <= short_cluster_gap
            and merged_len <= short_cluster_total
        )
        reading_cluster_merge = (
            item["route"] == prev["route"]
            and item["route"] != "passthrough"
            and should_merge_reading_cluster(
                prev_duration_sec=prev["duration_sec"],
                item_duration_sec=item["duration_sec"],
                gap_sec=gap / sr,
                merged_duration_sec=merged_len / sr,
                timbre_similarity=timbre_sim,
            )
        )
        passthrough_bridge_merge = False
        if (
            i + 1 < len(provisional)
            and item["route"] == "passthrough"
            and (item["end"] - item["start"]) <= reading_passthrough_bridge_max
        ):
            nxt = provisional[i + 1]
            left_gap = item["start"] - prev["end"]
            right_gap = nxt["start"] - item["end"]
            bridge_total_len = nxt["end"] - prev["start"]
            bridge_sim = cosine_similarity(prev["embedding"], nxt["embedding"])
            bridge_f0_close = f0_is_close(
                prev.get("median_f0", 0.0),
                nxt.get("median_f0", 0.0),
                READING_PASSTHROUGH_F0_DIFF_HZ,
            )
            passthrough_bridge_merge = (
                prev["route"] == nxt["route"]
                and prev["route"] != "passthrough"
                and (
                    (
                        left_gap <= passthrough_bridge_gap
                        and right_gap <= passthrough_bridge_gap
                        and bridge_total_len <= passthrough_bridge_total
                        and bridge_sim >= SPEAKER_SIM_THRESHOLD
                    )
                    or (
                        READING_MODE
                        and
                        prev["duration_sec"] <= READING_CLUSTER_UNIT_SEC
                        and nxt["duration_sec"] <= READING_CLUSTER_UNIT_SEC
                        and left_gap <= reading_cluster_gap
                        and right_gap <= reading_cluster_gap
                        and bridge_total_len <= reading_passthrough_bridge_total
                        and (
                            bridge_sim >= SPEAKER_CONTINUATION_THRESHOLD
                            or bridge_f0_close
                        )
                    )
                )
            )
            if passthrough_bridge_merge:
                prev["end"] = nxt["end"]
                prev["duration_sec"] = (prev["end"] - prev["start"]) / sr
                prev["embedding"] = (prev["embedding"] + nxt["embedding"]) / 2.0
                prev["median_f0"] = float(
                    (
                        float(prev.get("median_f0", 0.0))
                        + float(nxt.get("median_f0", 0.0))
                    )
                    / 2.0
                )
                prev_norm = float(np.linalg.norm(prev["embedding"]))
                if prev_norm > 1e-8:
                    prev["embedding"] = prev["embedding"] / prev_norm
                i += 2
                continue

        if (
            same_route_merge
            or cross_route_merge
            or continuation_merge
            or short_cluster_merge
            or reading_cluster_merge
        ):
            prev["end"] = item["end"]
            prev["duration_sec"] = (prev["end"] - prev["start"]) / sr
            prev["embedding"] = (prev["embedding"] + item["embedding"]) / 2.0
            prev["median_f0"] = float(
                (
                    float(prev.get("median_f0", 0.0))
                    + float(item.get("median_f0", 0.0))
                )
                / 2.0
            )
            prev_norm = float(np.linalg.norm(prev["embedding"]))
            if prev_norm > 1e-8:
                prev["embedding"] = prev["embedding"] / prev_norm
        else:
            merged.append(item.copy())
        i += 1

    return [(item["start"], item["end"]) for item in merged]


def instantiate_vc(model_name: str) -> VC:
    config = Config()
    vc = VC(config)
    vc.get_vc(model_name)
    return vc


def convert_with_vc(
    vc: VC, model_name: str, index_path: str | Path, input_path: Path, params: dict
) -> Tuple[int, np.ndarray, str]:
    info, audio = vc.vc_single(
        0,
        str(input_path),
        params["f0_up_key"],
        None,
        params["f0_method"],
        str(index_path),
        str(index_path),
        params["index_rate"],
        params["filter_radius"],
        params["resample_sr"],
        params["rms_mix_rate"],
        params["protect"],
    )
    if audio is None:
        raise RuntimeError(info)
    return audio[0], audio[1], info


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def to_float_audio(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        max_val = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        return audio.astype(np.float32) / float(max_val)
    audio = audio.astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.5:
        audio = audio / peak
    return audio


def fit_to_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    if audio.shape[0] == target_len:
        return audio
    if audio.shape[0] > target_len:
        return audio[:target_len]
    return np.pad(audio, (0, target_len - audio.shape[0]))


def resolve_index_path(index_name: str) -> str:
    index_name = (index_name or "").strip()
    if not index_name:
        return ""
    index_path = Path(index_name)
    if not index_path.is_absolute():
        index_path = REPO_ROOT / "assets" / "indices" / index_name
    if index_path.is_dir():
        return ""
    return str(index_path)


def apply_edge_fade(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.size == 0:
        return audio
    fade_len = min(int(SEGMENT_EDGE_FADE_SEC * sr), audio.shape[0] // 3)
    if fade_len <= 1:
        return audio
    faded = audio.astype(np.float32, copy=True)
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    faded[:fade_len] *= fade_in
    faded[-fade_len:] *= fade_in[::-1]
    return faded


def write_json(path: Path, payload: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, decisions: List[SegmentDecision]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(decisions[0]).keys()))
        writer.writeheader()
        for decision in decisions:
            writer.writerow(asdict(decision))


def export_mp3(input_wav: Path, output_mp3: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_mp3),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def process_audio(
    input_audio: Path,
    output_dir: Path,
    remix: bool,
    progress_callback: Callable[[float, str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> None:
    started_at = time.time()

    def report(progress: float, message: str) -> None:
        if progress_callback:
            progress_callback(progress, message)
        if log_callback:
            log_callback(message)

    report(0.02, "准备输出目录")
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_inputs = output_dir / "segments_in"
    segment_outputs = output_dir / "segments_out"
    segment_inputs.mkdir(exist_ok=True)
    segment_outputs.mkdir(exist_ok=True)

    report(0.08, "分离人声和伴奏")
    vocal_path, instrumental_path = run_uvr_split(input_audio, output_dir)
    report(0.18, "检测人声片段")
    mono_vocal, vocal_sr, intervals = detect_segments(vocal_path)
    instrumental_audio, instrumental_sr = sf.read(instrumental_path, always_2d=True)

    report(0.28, f"分析音高与音色，初始片段 {len(intervals)} 段")
    rmvpe = create_rmvpe(Config())
    intervals = merge_adjacent_same_route(mono_vocal, vocal_sr, intervals, rmvpe)
    analyzed_segments = analyze_intervals(mono_vocal, vocal_sr, intervals, rmvpe)
    absorb_short_passthrough_segments(analyzed_segments, vocal_sr)
    analyzed_segments = merge_context_absorbed_segments(
        mono_vocal, vocal_sr, analyzed_segments
    )
    report(0.36, f"片段合并完成，待处理 {len(analyzed_segments)} 段")
    report(0.42, "加载男声与女声模型")
    male_vc = instantiate_vc(MALE_MODEL)
    female_vc = instantiate_vc(FEMALE_MODEL)

    converted_vocals = np.zeros(
        (instrumental_audio.shape[0], instrumental_audio.shape[1]), dtype=np.float32
    )
    decisions: List[SegmentDecision] = []

    male_index_path = resolve_index_path(MALE_INDEX)
    female_index_path = resolve_index_path(FEMALE_INDEX)

    total_segments = max(len(analyzed_segments), 1)
    for segment_id, segment_meta in enumerate(analyzed_segments):
        route_label = {
            "male": "男声",
            "female": "女声",
            "passthrough": "直通",
        }.get(segment_meta["route"], segment_meta["route"])
        report(
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
            segment_audio = ensure_mono(cached_audio.astype(np.float32))
            if route == "male":
                model_name = MALE_MODEL
                index_name = MALE_INDEX
                params = MALE_PARAMS
            elif route == "female":
                model_name = FEMALE_MODEL
                index_name = FEMALE_INDEX
                params = FEMALE_PARAMS
            else:
                note = "reused_passthrough"
            if cached_sr != instrumental_sr:
                segment_audio = librosa.resample(
                    segment_audio, orig_sr=cached_sr, target_sr=instrumental_sr
                )
            target_len = int(round(duration_sec * instrumental_sr))
            segment_audio = fit_to_length(segment_audio, target_len).astype(np.float32)
            segment_audio = apply_edge_fade(segment_audio, instrumental_sr)
            start_out = int(round(start_sec * instrumental_sr))
            end_out = min(start_out + target_len, converted_vocals.shape[0])
            usable = segment_audio[: max(0, end_out - start_out)]
            if usable.size > 0:
                converted_vocals[start_out:end_out, :] += usable[:, None]
            chosen_params = params or MALE_PARAMS
            decisions.append(
                SegmentDecision(
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
            model_name = MALE_MODEL
            index_name = MALE_INDEX
            params = MALE_PARAMS
            try:
                out_sr, converted, _ = convert_with_vc(
                    male_vc, MALE_MODEL, male_index_path, segment_input_path, params
                )
                segment_audio = librosa.resample(
                    to_float_audio(ensure_mono(converted)),
                    orig_sr=out_sr,
                    target_sr=instrumental_sr,
                )
            except Exception as exc:
                note = f"male_convert_failed:{exc}"
                route = "passthrough"
        elif route == "female":
            model_name = FEMALE_MODEL
            index_name = FEMALE_INDEX
            params = FEMALE_PARAMS
            try:
                out_sr, converted, _ = convert_with_vc(
                    female_vc,
                    FEMALE_MODEL,
                    female_index_path,
                    segment_input_path,
                    params,
                )
                segment_audio = librosa.resample(
                    to_float_audio(ensure_mono(converted)),
                    orig_sr=out_sr,
                    target_sr=instrumental_sr,
                )
            except Exception as exc:
                note = f"female_convert_failed:{exc}"
                route = "passthrough"

        if route == "passthrough":
            segment_audio = librosa.resample(
                raw_segment.astype(np.float32),
                orig_sr=vocal_sr,
                target_sr=instrumental_sr,
            )

        target_len = int(round(duration_sec * instrumental_sr))
        segment_audio = fit_to_length(segment_audio, target_len)
        segment_audio = segment_audio.astype(np.float32)
        segment_audio = apply_edge_fade(segment_audio, instrumental_sr)

        output_segment_path = segment_outputs / f"segment_{segment_id:04d}_{route}.wav"
        sf.write(output_segment_path, segment_audio, instrumental_sr)

        start_out = int(round(start_sec * instrumental_sr))
        end_out = min(start_out + target_len, converted_vocals.shape[0])
        usable = segment_audio[: max(0, end_out - start_out)]
        if usable.size > 0:
            converted_vocals[start_out:end_out, :] += usable[:, None]

        chosen_params = params or MALE_PARAMS
        decisions.append(
            SegmentDecision(
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

    converted_vocal_path = output_dir / "converted_vocals.wav"
    converted_mp3_path = output_dir / "converted_vocals.mp3"
    report(0.90, "写出转换后人声")
    vocals_peak = float(np.max(np.abs(converted_vocals))) if converted_vocals.size else 0.0
    if vocals_peak > 1.0:
        converted_vocals /= vocals_peak
    sf.write(converted_vocal_path, converted_vocals, instrumental_sr)
    export_mp3(converted_vocal_path, converted_mp3_path)

    write_json(output_dir / "segments.json", [asdict(item) for item in decisions])
    if decisions:
        write_csv(output_dir / "segments.csv", decisions)

    if remix:
        report(0.96, "混回伴奏并导出成品")
        final_mix = instrumental_audio.astype(np.float32) + converted_vocals.astype(
            np.float32
        )
        peak = float(np.max(np.abs(final_mix))) if final_mix.size else 0.0
        if peak > 1.0:
            final_mix /= peak
        final_mix_wav = output_dir / "final_mix.wav"
        final_mix_mp3 = output_dir / "final_mix.mp3"
        sf.write(final_mix_wav, final_mix, instrumental_sr)
        export_mp3(final_mix_wav, final_mix_mp3)
    elapsed = time.time() - started_at
    minutes = int(elapsed // 60)
    seconds = int(round(elapsed % 60))
    elapsed_text = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    report(1.0, f"处理完成 | 实际用时 {elapsed_text}")


def main() -> None:
    ensure_environment()
    args = parse_args()
    input_audio = Path(args.input_audio).expanduser().resolve()
    output_dir = canonical_output_dir(input_audio, args.output_dir)
    process_audio(input_audio, output_dir, remix=not args.skip_remix)
    print(output_dir)


if __name__ == "__main__":
    main()
