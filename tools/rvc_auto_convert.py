from __future__ import annotations

import json
import os
import re
import shutil
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

from configs.config import Config
from infer.modules.vc.utils import get_index_path_from_model
from tools import process_mixed_long_audio as mixed_audio
from tools.rvc_pipeline.classifier import build_analysis, select_processing_plan as build_processing_plan
from tools.rvc_pipeline.executor import (
    execute_clean_voice_segments,
    execute_long_mixed_pipeline,
    execute_separate_bgm_voice,
    execute_single,
)
from tools.rvc_pipeline.metadata import append_review_record, build_result_payload, failed_response_payload
from tools.rvc_pipeline.quality_gate import evaluate_quality_gate
from tools.rvc_pipeline.segmenter import summarize_segment_review
from tools.rvc_pipeline.sfx_detector import summarize_sfx_segments
from tools.rvc_pipeline.thresholds import ACTIVE_THRESHOLDS


AUTO_ROOT = REPO_ROOT / "outputs" / "rvc_auto_convert"
REVIEW_QUEUE_PATH = AUTO_ROOT / "review_queue.jsonl"
RVC_SAMPLE_ROOT = Path("/Users/yanwei/Downloads/RVC Sample")
SAMPLE_LIBRARY_ROOT = REPO_ROOT / "assets" / "sample_library"
SAMPLE_LIBRARY_LABELS_PATH = SAMPLE_LIBRARY_ROOT / "labels" / "manual_labels.jsonl"
SAMPLE_LIBRARY_CANDIDATE_PATHS = (
    SAMPLE_LIBRARY_ROOT / "review_candidates.json",
    SAMPLE_LIBRARY_ROOT / "review_candidates_round2.json",
    SAMPLE_LIBRARY_ROOT / "manifest.jsonl",
)
SAMPLE_ANALYSIS_DB = REPO_ROOT / "outputs" / "audio_analysis" / "book_res_full_20260418" / "audio_features.sqlite"
SAMPLE_CALIBRATION_CACHE_PATH = AUTO_ROOT / "sample_calibration_cache.json"
DEFAULT_PROFILE = "default"
DEFAULT_MALE_MODEL = "Myaz.pth"
DEFAULT_FEMALE_MODEL = "haoshengyin.pth"
MUSIC_RISK_SAFE_THRESHOLD = 0.25
LONG_AUDIO_SECONDS = 25.0
SHORT_CLEAN_VOICE_SECONDS = 8.0
SHORT_CLEAN_VOICE_MIN_RATIO = 0.55
SHORT_VOICE_ALLOWED_MUSIC_REASONS = {"rhythmic_onsets", "percussive_content"}
CLEAN_VOICE_ALLOWED_MUSIC_REASONS = {"rhythmic_onsets", "percussive_content"}
CALIBRATION_MIN_CONFIDENCE = 0.72
CALIBRATION_STRONG_CONFIDENCE = 0.85
FEATURE_ANALYSIS_MAX_SECONDS = 15.0

PROFILE_PRESETS = {
    "default": {
        "male_model": DEFAULT_MALE_MODEL,
        "female_model": DEFAULT_FEMALE_MODEL,
        "uvr_model": "auto",
        "reading_mode": True,
        "speaker_embedding": False,
        "male_params": {
            "f0_up_key": 0,
            "f0_method": "rmvpe",
            "index_rate": 0.0,
            "protect": 0.30,
            "rms_mix_rate": 0.20,
            "filter_radius": 3,
            "resample_sr": 0,
        },
        "female_params": {
            "f0_up_key": 0,
            "f0_method": "rmvpe",
            "index_rate": 0.75,
            "protect": 0.25,
            "rms_mix_rate": 0.18,
            "filter_radius": 3,
            "resample_sr": 0,
        },
    }
}


def ensure_environment() -> None:
    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("weight_root", "assets/weights")
    os.environ.setdefault("weight_uvr5_root", "assets/uvr5_weights")
    os.environ.setdefault("index_root", "logs")
    os.environ.setdefault("outside_index_root", "assets/indices")
    os.environ.setdefault("rmvpe_root", "assets/rmvpe")
    os.environ.setdefault("TEMP", "/tmp")
    AUTO_ROOT.mkdir(parents=True, exist_ok=True)


def resolve_index_for_model(model_name: str) -> str:
    auto_index = get_index_path_from_model(model_name)
    if auto_index:
        path = Path(auto_index)
        return str(path if path.is_absolute() else (REPO_ROOT / path).resolve())
    index_path = REPO_ROOT / "assets" / "indices" / f"{Path(model_name).stem}.index"
    return str(index_path.resolve()) if index_path.exists() else ""


def audio_duration_seconds(input_path: Path) -> float:
    import soundfile as sf

    return float(sf.info(input_path).duration)


def _normalized_audio(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / max(peak, 1e-6)
    return y


_RMVPE_MODEL = None


def get_rmvpe_model():
    ensure_environment()
    global _RMVPE_MODEL
    if _RMVPE_MODEL is None:
        config = Config()
        _RMVPE_MODEL = mixed_audio.create_rmvpe(config)
    return _RMVPE_MODEL


def _percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def _mean_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _std_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _tempo_from_onset_env(onset_env: np.ndarray, sr: int) -> float:
    if hasattr(librosa, "feature") and hasattr(librosa.feature, "tempo"):
        tempo_values = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        return float(tempo_values[0]) if np.size(tempo_values) else 0.0
    if hasattr(librosa, "beat") and hasattr(librosa.beat, "tempo"):
        tempo_values = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        return float(tempo_values[0]) if np.size(tempo_values) else 0.0
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    return float(np.asarray(tempo).reshape(-1)[0]) if np.size(tempo) else 0.0


def extract_audio_feature_bundle(
    y: np.ndarray,
    sr: int,
    duration_seconds: float,
    max_seconds: float = FEATURE_ANALYSIS_MAX_SECONDS,
) -> dict:
    if y.size == 0:
        return {
            "duration_seconds": float(duration_seconds),
            "voiced_frames": 0,
            "voiced_ratio": 0.0,
            "non_voiced_ratio": 0.0,
            "median_f0_hz": 0.0,
        }

    max_samples = int(max_seconds * sr) if max_seconds > 0 else len(y)
    if max_samples > 0 and len(y) > max_samples:
        y = y[:max_samples]
    y = _normalized_audio(y)
    intervals = librosa.effects.split(y, top_db=mixed_audio.TOP_DB)
    non_silent_samples = sum(int(end - start) for start, end in intervals)
    non_silent_ratio = non_silent_samples / max(len(y), 1)
    segment_durations = np.array([(end - start) / max(sr, 1) for start, end in intervals], dtype=np.float32)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = _tempo_from_onset_env(onset_env, sr)
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_rms = float(np.sqrt(np.mean(harmonic * harmonic))) if harmonic.size else 0.0
    percussive_rms = float(np.sqrt(np.mean(percussive * percussive))) if percussive.size else 0.0
    total_rms = float(np.sqrt(np.mean(y * y))) if y.size else 0.0
    harmonic_ratio = harmonic_rms / max(total_rms, 1e-8)
    percussive_ratio = percussive_rms / max(total_rms, 1e-8)

    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    rmvpe = get_rmvpe_model()
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    rmvpe_f0 = rmvpe.infer_from_audio(y16, thred=0.03)
    rmvpe_voiced = rmvpe_f0[rmvpe_f0 > 0]
    voiced_frames = int(rmvpe_voiced.shape[0])
    voiced_ratio = voiced_frames / max(len(rmvpe_f0), 1)
    non_voiced_ratio = max(0.0, non_silent_ratio - voiced_ratio)

    yin_f0 = librosa.yin(y, fmin=65, fmax=450, sr=sr)
    yin_voiced = yin_f0[np.isfinite(yin_f0)]

    return {
        "duration_seconds": float(duration_seconds),
        "voiced_frames": voiced_frames,
        "voiced_ratio": float(voiced_ratio),
        "non_voiced_ratio": float(non_voiced_ratio),
        "non_silent_ratio": float(non_silent_ratio),
        "segment_count": int(len(intervals)),
        "short_segment_ratio": float(np.mean(segment_durations < 0.75)) if segment_durations.size else 0.0,
        "rmvpe_p25": _percentile_or_zero(rmvpe_voiced, 25),
        "rmvpe_p50": _percentile_or_zero(rmvpe_voiced, 50),
        "rmvpe_p75": _percentile_or_zero(rmvpe_voiced, 75),
        "rmvpe_mean": _mean_or_zero(rmvpe_voiced),
        "rmvpe_std": _std_or_zero(rmvpe_voiced),
        "yin_p25": _percentile_or_zero(yin_voiced, 25),
        "yin_p50": _percentile_or_zero(yin_voiced, 50),
        "yin_p75": _percentile_or_zero(yin_voiced, 75),
        "yin_mean": _mean_or_zero(yin_voiced),
        "yin_std": _std_or_zero(yin_voiced),
        "median_f0_hz": _percentile_or_zero(rmvpe_voiced, 50),
        "flatness_mean": _mean_or_zero(spectral_flatness),
        "onset_mean": _mean_or_zero(onset_env),
        "tempo": float(tempo),
        "harmonic_ratio": float(harmonic_ratio),
        "percussive_ratio": float(percussive_ratio),
        "centroid_mean": _mean_or_zero(centroid),
        "rolloff_mean": _mean_or_zero(rolloff),
        "bandwidth_mean": _mean_or_zero(bandwidth),
        "zcr_mean": _mean_or_zero(zcr),
        "rms_mean": _mean_or_zero(rms),
        "mfcc1_mean": _mean_or_zero(mfcc[1]) if mfcc.shape[0] > 1 else 0.0,
        "mfcc2_mean": _mean_or_zero(mfcc[2]) if mfcc.shape[0] > 2 else 0.0,
        "mfcc3_mean": _mean_or_zero(mfcc[3]) if mfcc.shape[0] > 3 else 0.0,
    }


def estimate_music_risk_from_features(feature_bundle: dict) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if float(feature_bundle.get("duration_seconds", 0.0)) <= 0.0:
        return 0.0, ["empty_audio"]

    duration_seconds = float(feature_bundle.get("duration_seconds", 0.0))
    flatness_mean = float(feature_bundle.get("flatness_mean", 0.0))
    onset_mean = float(feature_bundle.get("onset_mean", 0.0))
    tempo = float(feature_bundle.get("tempo", 0.0))
    harmonic_ratio = float(feature_bundle.get("harmonic_ratio", 0.0))
    percussive_ratio = float(feature_bundle.get("percussive_ratio", 0.0))
    non_voiced_ratio = float(feature_bundle.get("non_voiced_ratio", 0.0))
    voiced_ratio = float(feature_bundle.get("voiced_ratio", 0.0))

    risk = 0.0
    if duration_seconds >= LONG_AUDIO_SECONDS:
        risk += 0.20
        reasons.append("long_audio")
    if flatness_mean < 0.035 and harmonic_ratio > 0.45:
        risk += 0.25
        reasons.append("strong_harmonic_content")
    if onset_mean > 0.45 and tempo > 40:
        risk += 0.25
        reasons.append("rhythmic_onsets")
    if percussive_ratio > 0.18:
        risk += 0.20
        reasons.append("percussive_content")
    if harmonic_ratio > 0.65 and percussive_ratio > 0.08:
        risk += 0.20
        reasons.append("music_like_harmonic_percussive_mix")

    # Speech-dominant clips often trigger onset/percussive heuristics without actual BGM.
    if voiced_ratio >= 0.65 and non_voiced_ratio <= 0.04 and reasons:
        speech_like_reasons = {"rhythmic_onsets", "percussive_content", "strong_harmonic_content"}
        if set(reasons).issubset(speech_like_reasons):
            risk = min(risk, 0.18)

    return min(risk, 1.0), reasons


def estimate_music_risk(y: np.ndarray, sr: int, duration_seconds: float) -> tuple[float, list[str]]:
    feature_bundle = extract_audio_feature_bundle(y, sr, duration_seconds)
    return estimate_music_risk_from_features(feature_bundle)


def classify_voice_from_features(feature_bundle: dict) -> tuple[str, float, float, int, float]:
    voiced_frames = int(feature_bundle.get("voiced_frames", 0))
    voiced_ratio = float(feature_bundle.get("voiced_ratio", 0.0))
    median_f0 = float(feature_bundle.get("median_f0_hz", 0.0))
    if voiced_frames <= 0:
        return "unknown", 0.0, 0.0, 0, 0.0
    if voiced_ratio < 0.08:
        return "unknown", 0.35, 0.0, 0, voiced_ratio
    if voiced_frames < mixed_audio.UNVOICED_MIN_FRAMES:
        return "unknown", 0.4, 0.0, voiced_frames, voiced_ratio

    male_score = 0.0
    female_score = 0.0

    rmvpe_p25 = float(feature_bundle.get("rmvpe_p25", 0.0))
    rmvpe_p50 = float(feature_bundle.get("rmvpe_p50", 0.0))
    rmvpe_p75 = float(feature_bundle.get("rmvpe_p75", 0.0))
    rmvpe_mean = float(feature_bundle.get("rmvpe_mean", 0.0))
    yin_p50 = float(feature_bundle.get("yin_p50", 0.0))
    yin_mean = float(feature_bundle.get("yin_mean", 0.0))

    male_score += max(0.0, 180.0 - rmvpe_p50) / 42.0
    male_score += max(0.0, 160.0 - rmvpe_mean) / 45.0
    male_score += max(0.0, 150.0 - rmvpe_p25) / 28.0
    male_score += max(0.0, 180.0 - yin_p50) / 55.0

    female_score += max(0.0, rmvpe_p50 - 190.0) / 36.0
    female_score += max(0.0, rmvpe_mean - 180.0) / 42.0
    female_score += max(0.0, rmvpe_p75 - 215.0) / 32.0
    female_score += max(0.0, yin_mean - 210.0) / 60.0

    route = "female" if female_score >= male_score else "male"
    total = max(male_score + female_score, 1e-6)
    margin = abs(female_score - male_score) / total
    confidence = min(0.98, 0.58 + margin * 0.35 + min(voiced_ratio, 0.25))
    return route, confidence, median_f0, voiced_frames, voiced_ratio


def classify_voice(y: np.ndarray, sr: int) -> tuple[str, float, float, int, float]:
    feature_bundle = extract_audio_feature_bundle(y, sr, len(y) / max(sr, 1))
    return classify_voice_from_features(feature_bundle)


def _gender_feature_vector(feature_bundle: dict) -> np.ndarray:
    return np.array(
        [
            feature_bundle.get("rmvpe_p25", 0.0),
            feature_bundle.get("rmvpe_p50", 0.0),
            feature_bundle.get("rmvpe_p75", 0.0),
            feature_bundle.get("rmvpe_mean", 0.0),
            feature_bundle.get("yin_p50", 0.0),
            feature_bundle.get("yin_mean", 0.0),
            feature_bundle.get("centroid_mean", 0.0),
            feature_bundle.get("rolloff_mean", 0.0),
            feature_bundle.get("bandwidth_mean", 0.0),
            feature_bundle.get("mfcc1_mean", 0.0),
            feature_bundle.get("mfcc2_mean", 0.0),
            feature_bundle.get("mfcc3_mean", 0.0),
        ],
        dtype=np.float32,
    )


def _music_feature_vector(feature_bundle: dict) -> np.ndarray:
    return np.array(
        [
            feature_bundle.get("voiced_ratio", 0.0),
            feature_bundle.get("non_voiced_ratio", 0.0),
            feature_bundle.get("flatness_mean", 0.0),
            feature_bundle.get("onset_mean", 0.0),
            feature_bundle.get("tempo", 0.0),
            feature_bundle.get("harmonic_ratio", 0.0),
            feature_bundle.get("percussive_ratio", 0.0),
            feature_bundle.get("centroid_mean", 0.0),
            feature_bundle.get("rolloff_mean", 0.0),
            feature_bundle.get("rms_mean", 0.0),
        ],
        dtype=np.float32,
    )


def _manual_gender_feature_vector(feature_bundle: dict) -> np.ndarray:
    return np.array(
        [
            feature_bundle.get("rmvpe_p25", 0.0),
            feature_bundle.get("rmvpe_p50", 0.0),
            feature_bundle.get("rmvpe_p75", 0.0),
            feature_bundle.get("rmvpe_mean", 0.0),
            feature_bundle.get("voiced_ratio", 0.0),
            feature_bundle.get("non_voiced_ratio", 0.0),
            feature_bundle.get("harmonic_ratio", 0.0),
            feature_bundle.get("percussive_ratio", 0.0),
            feature_bundle.get("flatness_mean", 0.0),
            feature_bundle.get("onset_mean", 0.0),
        ],
        dtype=np.float32,
    )


def _manual_music_feature_vector(feature_bundle: dict) -> np.ndarray:
    return np.array(
        [
            np.log1p(float(feature_bundle.get("duration_seconds", 0.0))),
            feature_bundle.get("voiced_ratio", 0.0),
            feature_bundle.get("non_voiced_ratio", 0.0),
            feature_bundle.get("non_silent_ratio", 0.0),
            feature_bundle.get("flatness_mean", 0.0),
            feature_bundle.get("onset_mean", 0.0),
            feature_bundle.get("harmonic_ratio", 0.0),
            feature_bundle.get("percussive_ratio", 0.0),
            feature_bundle.get("rms_mean", 0.0),
            np.log1p(float(feature_bundle.get("segment_count", 0.0))),
            feature_bundle.get("short_segment_ratio", 0.0),
        ],
        dtype=np.float32,
    )


def _infer_sample_labels(folder_name: str) -> tuple[str | None, str | None]:
    gender_label = None
    music_label = None
    if "纯男声" in folder_name:
        gender_label = "male"
        music_label = "no_music"
    elif "纯女声" in folder_name:
        gender_label = "female"
        music_label = "no_music"
    elif "男声+女声+音乐" in folder_name:
        music_label = "music"
    elif "男声+女声" in folder_name:
        music_label = "no_music"
    elif "音乐" in folder_name:
        music_label = "music"
    return gender_label, music_label


def _build_calibration_task(samples: list[dict], feature_set: str = "legacy_full_v1") -> dict | None:
    if len(samples) < 2:
        return None
    matrix = np.array([item["vector"] for item in samples], dtype=np.float32)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    label_counts: dict[str, int] = {}
    for item in samples:
        label = str(item.get("label") or "")
        label_counts[label] = label_counts.get(label, 0) + 1
    return {
        "feature_set": feature_set,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "samples": samples,
        "label_counts": label_counts,
    }


def _is_valid_calibration_model(model: dict | None) -> bool:
    if not isinstance(model, dict):
        return False
    gender = model.get("gender")
    music = model.get("music")
    if not isinstance(gender, dict) or not isinstance(music, dict):
        return False
    gender_samples = gender.get("samples") or []
    music_samples = music.get("samples") or []
    if len(gender_samples) < 2 or len(music_samples) < 2:
        return False
    gender_labels = {item.get("label") for item in gender_samples}
    music_labels = {item.get("label") for item in music_samples}
    return {"male", "female"}.issubset(gender_labels) and {"music", "no_music"}.issubset(music_labels)


def _load_json_array(path: Path) -> list[dict]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _sample_id_to_book_and_stem(sample_id: str) -> tuple[str, str] | None:
    normalized = sample_id
    if normalized.startswith("round2_"):
        normalized = normalized[len("round2_") :]
    match = re.match(r"^(tape[^_]+_\d{6})_(.+)$", normalized)
    if not match:
        return None
    return match.group(1), match.group(2)


def _recover_sample_item_from_analysis_db(sample_id: str) -> dict | None:
    parsed = _sample_id_to_book_and_stem(sample_id)
    if not parsed or not SAMPLE_ANALYSIS_DB.exists():
        return None
    book_id, stem = parsed
    import sqlite3

    conn = sqlite3.connect(SAMPLE_ANALYSIS_DB)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT book_id, abs_path, rel_path, duration_bucket, audio_type
            FROM files
            WHERE book_id = ? AND abs_path LIKE ?
            ORDER BY abs_path
            LIMIT 1
            """,
            (book_id, f"%/{stem}.mp3"),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return {
        "sample_id": sample_id,
        "path": str(row["abs_path"]),
        "book_id": str(row["book_id"]),
        "duration_bucket": str(row["duration_bucket"] or ""),
        "notes": [str(row["audio_type"] or "")],
    }


def _analysis_row_for_item(sample_id: str, item: dict | None) -> dict | None:
    if not SAMPLE_ANALYSIS_DB.exists():
        return None
    book_id = str((item or {}).get("book_id") or "").strip()
    item_path = Path(str((item or {}).get("path") or ""))
    basename = item_path.name
    parsed = _sample_id_to_book_and_stem(sample_id)
    stem = parsed[1] if parsed else ""
    if not book_id and parsed:
        book_id = parsed[0]

    predicates: list[tuple[str, tuple]] = []
    if item_path:
        predicates.append(("f.abs_path = ?", (str(item_path),)))
    if book_id and basename:
        predicates.append(("f.book_id = ? AND f.abs_path LIKE ?", (book_id, f"%/{basename}")))
    if basename:
        predicates.append(("f.abs_path LIKE ?", (f"%/{basename}",)))
    if book_id and stem:
        predicates.append(("f.book_id = ? AND f.abs_path LIKE ?", (book_id, f"%/{stem}.mp3")))

    if not predicates:
        return None

    import sqlite3

    conn = sqlite3.connect(SAMPLE_ANALYSIS_DB)
    conn.row_factory = sqlite3.Row
    try:
        for where_sql, params in predicates:
            row = conn.execute(
                f"""
                SELECT
                    f.rel_path,
                    f.abs_path,
                    f.book_id,
                    f.duration_seconds,
                    f.voiced_coverage,
                    f.segment_count,
                    f.short_segment_ratio,
                    f.silence_ratio,
                    f.spectral_flatness,
                    f.onset_strength,
                    f.harmonic_ratio,
                    f.percussive_ratio,
                    f.rms,
                    d.f0_median_hz,
                    d.f0_p10_hz,
                    d.f0_p90_hz,
                    d.f0_voiced_frames,
                    d.f0_voiced_ratio
                FROM files f
                LEFT JOIN deep_features d ON d.rel_path = f.rel_path
                WHERE {where_sql}
                ORDER BY f.abs_path
                LIMIT 1
                """,
                params,
            ).fetchone()
            if row:
                return dict(row)
    finally:
        conn.close()
    return None


def _feature_bundle_from_analysis_row(row: dict) -> dict:
    non_silent_ratio = max(0.0, min(1.0, 1.0 - float(row.get("silence_ratio") or 0.0)))
    voiced_ratio = float(row.get("f0_voiced_ratio") or 0.0)
    if voiced_ratio <= 0.0:
        voiced_ratio = float(row.get("voiced_coverage") or 0.0)
    non_voiced_ratio = max(0.0, non_silent_ratio - voiced_ratio)
    f0_p10 = float(row.get("f0_p10_hz") or 0.0)
    f0_p50 = float(row.get("f0_median_hz") or 0.0)
    f0_p90 = float(row.get("f0_p90_hz") or 0.0)
    return {
        "duration_seconds": float(row.get("duration_seconds") or 0.0),
        "voiced_frames": int(row.get("f0_voiced_frames") or 0),
        "voiced_ratio": float(voiced_ratio),
        "non_voiced_ratio": float(non_voiced_ratio),
        "non_silent_ratio": float(non_silent_ratio),
        "segment_count": int(row.get("segment_count") or 0),
        "short_segment_ratio": float(row.get("short_segment_ratio") or 0.0),
        "rmvpe_p25": f0_p10,
        "rmvpe_p50": f0_p50,
        "rmvpe_p75": f0_p90,
        "rmvpe_mean": f0_p50,
        "median_f0_hz": f0_p50,
        "flatness_mean": float(row.get("spectral_flatness") or 0.0),
        "onset_mean": float(row.get("onset_strength") or 0.0),
        "harmonic_ratio": float(row.get("harmonic_ratio") or 0.0),
        "percussive_ratio": float(row.get("percussive_ratio") or 0.0),
        "rms_mean": float(row.get("rms") or 0.0),
    }


def _load_manual_sample_library_items() -> dict[str, dict]:
    items: dict[str, dict] = {}
    for path in SAMPLE_LIBRARY_CANDIDATE_PATHS:
        for row in _load_json_array(path):
            sample_id = str(row.get("sample_id") or "").strip()
            if sample_id:
                items[sample_id] = row
    return items


def _load_manual_label_records() -> list[dict]:
    if not SAMPLE_LIBRARY_LABELS_PATH.exists():
        return []
    records = []
    for line in SAMPLE_LIBRARY_LABELS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _manual_label_signature(records: list[dict], items: dict[str, dict]) -> dict:
    mtimes = []
    for path in (*SAMPLE_LIBRARY_CANDIDATE_PATHS, SAMPLE_LIBRARY_LABELS_PATH):
        if path.exists():
            mtimes.append(int(path.stat().st_mtime))
    return {
        "source": "manual_sample_library_v1",
        "label_count": len({record.get("sample_id") for record in records}),
        "item_count": len(items),
        "max_mtime": max(mtimes) if mtimes else 0,
    }


def _manual_speaker_label(labels: dict) -> str | None:
    speaker_pattern = labels.get("speaker_pattern")
    if speaker_pattern == "single_male":
        return "male"
    if speaker_pattern == "single_female":
        return "female"
    return None


def _manual_music_label(labels: dict) -> str | None:
    music_pattern = labels.get("music_pattern")
    if music_pattern in {"no_music", "transient_sfx"}:
        return "no_music"
    if music_pattern in {"bgm", "song"}:
        return "music"
    return None


def build_manual_sample_calibration_model() -> tuple[dict | None, dict | None]:
    records = _load_manual_label_records()
    items = _load_manual_sample_library_items()
    if not records:
        return None, None

    latest: dict[str, dict] = {}
    for record in records:
        sample_id = str(record.get("sample_id") or "").strip()
        if sample_id:
            latest[sample_id] = record

    signature = _manual_label_signature(list(latest.values()), items)
    gender_samples: list[dict] = []
    music_samples: list[dict] = []
    recovered_count = 0
    missing_count = 0
    sqlite_feature_count = 0
    audio_feature_count = 0

    for sample_id, record in latest.items():
        labels = record.get("labels") or {}
        item = record.get("item") or items.get(sample_id) or _recover_sample_item_from_analysis_db(sample_id)
        if not item:
            missing_count += 1
            continue
        if sample_id not in items and not record.get("item"):
            recovered_count += 1
        path = Path(str(item.get("path") or ""))
        if not path.exists():
            missing_count += 1
            continue
        gender_label = _manual_speaker_label(labels)
        music_label = _manual_music_label(labels)
        if gender_label is None and music_label is None:
            continue
        feature_row = _analysis_row_for_item(sample_id, item)
        if feature_row:
            feature_bundle = _feature_bundle_from_analysis_row(feature_row)
            sqlite_feature_count += 1
        else:
            try:
                duration = audio_duration_seconds(path)
                y, sr = librosa.load(path, sr=None, mono=True)
                feature_bundle = extract_audio_feature_bundle(y, sr, duration)
                audio_feature_count += 1
            except Exception:
                missing_count += 1
                continue
        if gender_label is not None and float(feature_bundle.get("voiced_ratio", 0.0)) >= 0.08:
            gender_samples.append(
                {
                    "label": gender_label,
                    "vector": _manual_gender_feature_vector(feature_bundle).tolist(),
                    "path": str(path),
                    "sample_id": sample_id,
                }
            )
        if music_label is not None:
            music_samples.append(
                {
                    "label": music_label,
                    "vector": _manual_music_feature_vector(feature_bundle).tolist(),
                    "path": str(path),
                    "sample_id": sample_id,
                }
            )

    model = {
        "source": "manual_sample_library",
        "gender": _build_calibration_task(gender_samples, feature_set="manual_sample_library_v1"),
        "music": _build_calibration_task(music_samples, feature_set="manual_sample_library_v1"),
        "stats": {
            "gender_samples": len(gender_samples),
            "music_samples": len(music_samples),
            "recovered_items": recovered_count,
            "missing_items": missing_count,
            "sqlite_feature_samples": sqlite_feature_count,
            "audio_feature_samples": audio_feature_count,
        },
    }
    if not _is_valid_calibration_model(model):
        return None, signature
    return model, signature


def manual_sample_library_signature() -> dict | None:
    records = _load_manual_label_records()
    if not records:
        return None
    items = _load_manual_sample_library_items()
    latest_sample_ids = {record.get("sample_id") for record in records if record.get("sample_id")}
    return _manual_label_signature(
        [{"sample_id": sample_id} for sample_id in latest_sample_ids],
        items,
    )


@lru_cache(maxsize=1)
def get_sample_calibration_model() -> dict | None:
    ensure_environment()
    manual_signature = manual_sample_library_signature()
    if manual_signature:
        try:
            cached = json.loads(SAMPLE_CALIBRATION_CACHE_PATH.read_text(encoding="utf-8"))
            if cached.get("signature") == manual_signature and _is_valid_calibration_model(cached.get("model")):
                return cached.get("model")
        except Exception:
            pass
    manual_model, manual_signature = build_manual_sample_calibration_model()
    if manual_model is not None and manual_signature is not None:
        try:
            SAMPLE_CALIBRATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            SAMPLE_CALIBRATION_CACHE_PATH.write_text(
                json.dumps({"signature": manual_signature, "model": manual_model}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        return manual_model

    if not RVC_SAMPLE_ROOT.exists():
        return None

    sample_paths = sorted(RVC_SAMPLE_ROOT.rglob("*.mp3"))
    if not sample_paths:
        return None

    signature = {
        "count": len(sample_paths),
        "max_mtime": max(int(path.stat().st_mtime) for path in sample_paths),
    }
    try:
        cached = json.loads(SAMPLE_CALIBRATION_CACHE_PATH.read_text(encoding="utf-8"))
        if cached.get("signature") == signature and _is_valid_calibration_model(cached.get("model")):
            return cached.get("model")
    except Exception:
        pass

    gender_samples: list[dict] = []
    music_samples: list[dict] = []
    for path in sample_paths:
        gender_label, music_label = _infer_sample_labels(path.parent.name)
        if gender_label is None and music_label is None:
            continue
        try:
            duration = audio_duration_seconds(path)
            y, sr = librosa.load(path, sr=None, mono=True)
            feature_bundle = extract_audio_feature_bundle(y, sr, duration)
        except Exception:
            continue

        if gender_label is not None and float(feature_bundle.get("voiced_ratio", 0.0)) >= 0.2:
            gender_samples.append(
                {
                    "label": gender_label,
                    "vector": _gender_feature_vector(feature_bundle).tolist(),
                    "path": str(path),
                }
            )
        if music_label is not None:
            music_samples.append(
                {
                    "label": music_label,
                    "vector": _music_feature_vector(feature_bundle).tolist(),
                    "path": str(path),
                }
            )

    model = {
        "gender": _build_calibration_task(gender_samples),
        "music": _build_calibration_task(music_samples),
    }
    if not _is_valid_calibration_model(model):
        return None
    try:
        SAMPLE_CALIBRATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SAMPLE_CALIBRATION_CACHE_PATH.write_text(
            json.dumps({"signature": signature, "model": model}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
    return model


def _vector_for_calibration_task(feature_bundle: dict, calibration: dict | None, kind: str) -> np.ndarray:
    if calibration and calibration.get("feature_set") == "manual_sample_library_v1":
        if kind == "gender":
            return _manual_gender_feature_vector(feature_bundle)
        return _manual_music_feature_vector(feature_bundle)
    if kind == "gender":
        return _gender_feature_vector(feature_bundle)
    return _music_feature_vector(feature_bundle)


def predict_knn_label(vector: np.ndarray, calibration: dict | None, k: int = 5) -> tuple[str, float, dict]:
    if calibration is None or not calibration.get("samples"):
        return "unknown", 0.0, {}
    samples = calibration["samples"]
    mean = np.array(calibration["mean"], dtype=np.float32)
    std = np.array(calibration["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    normalized_vector = (np.asarray(vector, dtype=np.float32) - mean) / std

    ranked: list[tuple[float, str]] = []
    for sample in samples:
        sample_vector = (np.array(sample["vector"], dtype=np.float32) - mean) / std
        distance = float(np.linalg.norm(normalized_vector - sample_vector))
        ranked.append((distance, sample["label"]))
    ranked.sort(key=lambda item: item[0])

    scores: dict[str, float] = {}
    label_counts = calibration.get("label_counts") or {}
    for distance, label in ranked[: max(1, min(k, len(ranked)))]:
        weight = 1.0 / max(distance, 1e-6)
        class_count = max(1, int(label_counts.get(label) or 1))
        weight = weight / np.sqrt(class_count)
        scores[label] = scores.get(label, 0.0) + weight
    if not scores:
        return "unknown", 0.0, {}

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    label, top_score = ordered[0]
    total_score = sum(scores.values())
    confidence = top_score / max(total_score, 1e-6)
    return label, float(confidence), scores


def apply_calibrated_music_adjustment(
    base_risk: float,
    base_reasons: list[str],
    feature_bundle: dict,
    prediction: dict | None,
) -> tuple[float, list[str], dict]:
    adjusted_risk = float(base_risk)
    adjusted_reasons = list(base_reasons)
    calibration = {
        "used": False,
        "suppressed": False,
        "boosted": False,
    }
    if not prediction:
        return adjusted_risk, adjusted_reasons, calibration

    label = prediction.get("label")
    confidence = float(prediction.get("confidence", 0.0))
    voiced_ratio = float(feature_bundle.get("voiced_ratio", 0.0))
    non_voiced_ratio = float(feature_bundle.get("non_voiced_ratio", 1.0))
    duration_seconds = float(feature_bundle.get("duration_seconds", 0.0))
    calibration["used"] = label in {"music", "no_music"}
    calibration["label"] = label
    calibration["confidence"] = round(confidence, 4)

    if (
        label == "no_music"
        and confidence >= CALIBRATION_STRONG_CONFIDENCE
        and voiced_ratio >= SHORT_CLEAN_VOICE_MIN_RATIO
        and duration_seconds <= ACTIVE_THRESHOLDS.ultra_short_single_seconds
        and non_voiced_ratio <= 0.20
        and set(base_reasons).issubset(SHORT_VOICE_ALLOWED_MUSIC_REASONS)
    ):
        adjusted_risk = min(adjusted_risk, 0.12 if confidence >= CALIBRATION_STRONG_CONFIDENCE else 0.18)
        adjusted_reasons = ["speech_dominant_calibrated_no_music"]
        calibration["suppressed"] = True
    elif label == "music" and confidence >= CALIBRATION_MIN_CONFIDENCE:
        adjusted_risk = max(adjusted_risk, 0.35 if confidence >= CALIBRATION_STRONG_CONFIDENCE else 0.28)
        if "calibrated_music_bed" not in adjusted_reasons:
            adjusted_reasons.append("calibrated_music_bed")
        calibration["boosted"] = True

    return adjusted_risk, adjusted_reasons, calibration


def _is_borderline_pitch(feature_bundle: dict) -> bool:
    rmvpe_p50 = float(feature_bundle.get("rmvpe_p50", 0.0))
    rmvpe_mean = float(feature_bundle.get("rmvpe_mean", 0.0))
    return 160.0 <= rmvpe_p50 <= 220.0 or 170.0 <= rmvpe_mean <= 225.0


def apply_calibrated_voice_route(
    base_route: str,
    base_confidence: float,
    feature_bundle: dict,
    prediction: dict | None,
) -> tuple[str, float]:
    route = base_route
    confidence = float(base_confidence)
    if not prediction:
        return route, confidence

    label = prediction.get("label")
    pred_conf = float(prediction.get("confidence", 0.0))
    if label not in {"male", "female"}:
        return route, confidence

    if label == route:
        return route, min(0.99, max(confidence, 0.6 + pred_conf * 0.35))

    if pred_conf >= CALIBRATION_MIN_CONFIDENCE and _is_borderline_pitch(feature_bundle):
        return label, min(0.95, max(confidence + 0.15, pred_conf * 0.92))

    return route, max(0.52, confidence * 0.92)


def analyze_audio(input_path: Path) -> dict:
    ensure_environment()
    duration = audio_duration_seconds(input_path)
    y, sr = librosa.load(input_path, sr=None, mono=True)
    feature_bundle = extract_audio_feature_bundle(y, sr, duration)

    music_risk, music_reasons = estimate_music_risk_from_features(feature_bundle)
    dominant_route, voice_confidence, median_f0, voiced_frames, voiced_ratio = classify_voice_from_features(
        feature_bundle
    )

    calibration_model = get_sample_calibration_model()
    gender_prediction = None
    music_prediction = None
    if calibration_model:
        gender_task = calibration_model.get("gender")
        music_task = calibration_model.get("music")
        # The full-library SQLite cache is reliable for music/speech texture, but many
        # cached F0 medians are zero. Keep gender routing on live RMVPE unless the
        # calibration was built from direct audio features.
        if (
            calibration_model.get("source") == "manual_sample_library"
            and int((calibration_model.get("stats") or {}).get("audio_feature_samples") or 0) == 0
        ):
            gender_task = None
        gender_label, gender_confidence, gender_scores = predict_knn_label(
            _vector_for_calibration_task(feature_bundle, gender_task, "gender"),
            gender_task,
        )
        music_label, music_confidence, music_scores = predict_knn_label(
            _vector_for_calibration_task(feature_bundle, music_task, "music"),
            music_task,
        )
        gender_prediction = {
            "label": gender_label,
            "confidence": round(gender_confidence, 4),
            "scores": {key: round(value, 4) for key, value in gender_scores.items()},
        }
        music_prediction = {
            "label": music_label,
            "confidence": round(music_confidence, 4),
            "scores": {key: round(value, 4) for key, value in music_scores.items()},
        }
        dominant_route, voice_confidence = apply_calibrated_voice_route(
            dominant_route,
            voice_confidence,
            feature_bundle,
            gender_prediction,
        )
        music_risk, music_reasons, music_calibration = apply_calibrated_music_adjustment(
            music_risk,
            music_reasons,
            feature_bundle,
            music_prediction,
        )
    else:
        music_calibration = {"used": False, "suppressed": False, "boosted": False}

    analysis = build_analysis(
        duration=duration,
        music_risk=music_risk,
        music_reasons=music_reasons,
        dominant_route=dominant_route,
        voice_confidence=voice_confidence,
        median_f0=median_f0,
        voiced_frames=voiced_frames,
        voiced_ratio=voiced_ratio,
        thresholds=ACTIVE_THRESHOLDS,
    )
    analysis["feature_summary"] = {
        "non_voiced_ratio": round(float(feature_bundle.get("non_voiced_ratio", 0.0)), 4),
        "rmvpe_p50": round(float(feature_bundle.get("rmvpe_p50", 0.0)), 4),
        "rmvpe_mean": round(float(feature_bundle.get("rmvpe_mean", 0.0)), 4),
        "yin_p50": round(float(feature_bundle.get("yin_p50", 0.0)), 4),
        "harmonic_ratio": round(float(feature_bundle.get("harmonic_ratio", 0.0)), 4),
        "percussive_ratio": round(float(feature_bundle.get("percussive_ratio", 0.0)), 4),
    }
    analysis["calibration"] = {
        "sample_library_used": bool(calibration_model),
        "gender_prediction": gender_prediction,
        "music_prediction": music_prediction,
        "music_adjustment": music_calibration,
    }
    return analysis

def apply_short_clean_voice_override(analysis: dict) -> dict:
    adjusted = dict(analysis)
    reasons = set(adjusted.get("music_reasons") or [])
    route = adjusted.get("dominant_route")
    duration = float(adjusted.get("duration_seconds", 0.0))
    has_reliable_voice = (
        route in {"male", "female"}
        and float(adjusted.get("voiced_ratio", 0.0)) >= SHORT_CLEAN_VOICE_MIN_RATIO
        and int(adjusted.get("voiced_frames", 0)) >= mixed_audio.UNVOICED_MIN_FRAMES
    )
    has_only_speech_like_music_reasons = reasons.issubset(
        CLEAN_VOICE_ALLOWED_MUSIC_REASONS
    )
    is_short_clean_voice = (
        duration <= SHORT_CLEAN_VOICE_SECONDS
        and has_reliable_voice
        and reasons.issubset(SHORT_VOICE_ALLOWED_MUSIC_REASONS)
    )
    is_clean_voice_before_long_threshold = (
        duration < LONG_AUDIO_SECONDS
        and has_reliable_voice
        and has_only_speech_like_music_reasons
    )
    if is_short_clean_voice:
        adjusted["classification"] = route
        adjusted["music_risk_overridden"] = True
        adjusted["override_reason"] = "short_clean_voice"
    elif is_clean_voice_before_long_threshold:
        adjusted["classification"] = route
        adjusted["music_risk_overridden"] = True
        adjusted["override_reason"] = "clean_voice_no_music"
    else:
        adjusted["music_risk_overridden"] = False
    return adjusted


def profile_config(profile: str) -> dict:
    return PROFILE_PRESETS.get(profile or DEFAULT_PROFILE, PROFILE_PRESETS[DEFAULT_PROFILE])


def select_processing_plan(analysis: dict, profile: str = DEFAULT_PROFILE) -> dict:
    config = profile_config(profile)
    return build_processing_plan(
        analysis,
        config,
        resolve_index_for_model,
        thresholds=ACTIVE_THRESHOLDS,
        profile_name=profile or DEFAULT_PROFILE,
    )


def progress_event(stage: str, progress: float, message: str) -> dict:
    return {"stage": stage, "progress": progress, "message": message}


def convert_single(input_path: Path, output_path: Path, plan: dict) -> tuple[list[dict], list[str]]:
    return execute_single(input_path, output_path, plan)


def convert_safe_long(
    input_path: Path,
    output_path: Path,
    job_dir: Path,
    plan: dict,
    progress_callback: Callable[[dict], None] | None,
) -> tuple[list[dict], list[str]]:
    return execute_long_mixed_pipeline(input_path, output_path, job_dir, plan, progress_callback)


def convert_separate_bgm_voice(
    input_path: Path,
    output_path: Path,
    job_dir: Path,
    plan: dict,
    progress_callback: Callable[[dict], None] | None,
) -> tuple[list[dict], list[str]]:
    return execute_separate_bgm_voice(input_path, output_path, job_dir, plan, progress_callback)


def convert_long_mixed_pipeline(
    input_path: Path,
    output_path: Path,
    job_dir: Path,
    plan: dict,
    progress_callback: Callable[[dict], None] | None,
) -> tuple[list[dict], list[str]]:
    return execute_long_mixed_pipeline(input_path, output_path, job_dir, plan, progress_callback)


def convert_clean_voice_segments(
    input_path: Path,
    output_path: Path,
    job_dir: Path,
    plan: dict,
    progress_callback: Callable[[dict], None] | None,
) -> tuple[list[dict], list[str]]:
    return execute_clean_voice_segments(input_path, output_path, job_dir, plan, progress_callback)


def failed_response(input_path: Path, output_path: Path, error: str, log: list[str] | None = None) -> dict:
    return failed_response_payload(
        input_path=input_path,
        output_path=output_path,
        error=error,
        log=log,
    )


def load_stage_summaries(job_dir: Path) -> dict:
    summaries: dict[str, dict] = {}
    if not job_dir.exists():
        return summaries
    for path in sorted(job_dir.glob("*_summary.json")):
        try:
            summaries[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    stages_path = job_dir / "long_mixed_pipeline_stages.json"
    if stages_path.exists():
        try:
            summaries[stages_path.stem] = json.loads(stages_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return summaries


def _segment_duration_seconds(segment: dict) -> float:
    if segment.get("duration_sec") is not None:
        try:
            return max(0.0, float(segment.get("duration_sec") or 0.0))
        except (TypeError, ValueError):
            return 0.0
    try:
        return max(0.0, float(segment.get("end") or 0.0) - float(segment.get("start") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _duration_weighted_ratio(segments: list[dict], predicate: Callable[[dict], bool]) -> float:
    total = sum(_segment_duration_seconds(segment) for segment in segments)
    if total <= 0.0:
        return 0.0
    selected = sum(_segment_duration_seconds(segment) for segment in segments if predicate(segment))
    return selected / total


def apply_pipeline_quality_checks(
    quality_gate: dict,
    analysis: dict,
    selected_plan: dict,
    segments: list[dict],
    stage_summaries: dict,
) -> dict:
    adjusted = dict(quality_gate)
    warnings = list(adjusted.get("warnings") or [])
    plan_mode = selected_plan.get("processing_mode")
    reasons = set(analysis.get("music_reasons") or [])
    music_risk = float(analysis.get("music_risk", 0.0) or 0.0)
    total_segment_seconds = sum(_segment_duration_seconds(segment) for segment in segments)

    def add_warning(reason: str) -> None:
        if reason not in warnings:
            warnings.append(reason)

    def note_of(segment: dict) -> str:
        return str(segment.get("note") or "")

    def is_context_smoothed(segment: dict) -> bool:
        return note_of(segment).startswith("context_smoothed_")

    has_strong_music = (
        music_risk >= 0.75
        and "strong_harmonic_content" in reasons
        and "music_like_harmonic_percussive_mix" in reasons
    )
    if plan_mode == "clean_voice_segments" and has_strong_music:
        add_warning("clean_voice_segments_on_strong_music")

    if plan_mode == "long_mixed_pipeline" and total_segment_seconds >= 20.0:
        passthrough_ratio = _duration_weighted_ratio(
            segments,
            lambda segment: segment.get("route") == "passthrough"
            or note_of(segment) in {"low_voice", "music_residual"},
        )
        if passthrough_ratio >= 0.40:
            add_warning("dominant_passthrough_segments")

        speech_segments = [segment for segment in segments if segment.get("route") in {"male", "female"}]
        if speech_segments:
            low_voiced_ratio = _duration_weighted_ratio(
                speech_segments,
                lambda segment: (
                    float(segment.get("voiced_ratio") or 0.0) < 0.35
                    and not is_context_smoothed(segment)
                ),
            )
            if low_voiced_ratio >= 0.70:
                add_warning("low_voiced_long_speech_segments")

        unstable_ratio = _duration_weighted_ratio(
            segments,
            lambda segment: (
                not is_context_smoothed(segment)
                and note_of(segment) != "music_residual"
                and (
                    note_of(segment) != "ok"
                    or float(segment.get("voiced_ratio") or 0.0) < 0.35
                )
            ),
        )
        if unstable_ratio >= 0.30:
            add_warning("unstable_long_mixed_segments")

    if warnings:
        adjusted["passed"] = False
        adjusted["fallback_used"] = True
        adjusted["fallback_reason"] = adjusted.get("fallback_reason") or warnings[0]
    adjusted["warnings"] = warnings
    if stage_summaries:
        adjusted["pipeline_stage_summary_keys"] = sorted(stage_summaries.keys())
    return adjusted


def auto_convert_mp3(
    input_path: Path,
    output_path: Path,
    profile: str = DEFAULT_PROFILE,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    ensure_environment()
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    log: list[str] = []

    try:
        if not input_path.exists():
            raise FileNotFoundError(f"input_path does not exist: {input_path}")
        if progress_callback:
            progress_callback(progress_event("analyzing", 0.05, "analyzing audio"))
        analysis = analyze_audio(input_path)
        if progress_callback:
            progress_callback(progress_event("selecting_plan", 0.20, "selecting processing plan"))
        plan = select_processing_plan(analysis, profile=profile)

        job_dir = AUTO_ROOT / f"{input_path.stem}_{int(time.time())}"
        job_dir.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback(progress_event("converting", 0.25, plan["processing_mode"]))
        if plan["processing_mode"] == "single":
            segments, single_log = convert_single(input_path, output_path, plan)
            log.extend(single_log)
        elif plan["processing_mode"] == "clean_voice_segments":
            segments, clean_log = convert_clean_voice_segments(
                input_path, output_path, job_dir, plan, progress_callback
            )
            log.extend(clean_log)
        elif plan["processing_mode"] == "separate_bgm_voice":
            segments, long_log = convert_separate_bgm_voice(
                input_path, output_path, job_dir, plan, progress_callback
            )
            log.extend(long_log)
        elif plan["processing_mode"] == "long_mixed_pipeline":
            segments, long_log = convert_long_mixed_pipeline(
                input_path, output_path, job_dir, plan, progress_callback
            )
            log.extend(long_log)
        else:
            raise RuntimeError(f"unsupported processing mode: {plan['processing_mode']}")

        if progress_callback:
            progress_callback(progress_event("finalizing", 0.95, "finalizing output"))
        if not output_path.exists():
            raise RuntimeError("output_path was not written")

        stage_summaries = load_stage_summaries(job_dir)
        quality_gate = evaluate_quality_gate(input_path, output_path)
        quality_gate = apply_pipeline_quality_checks(quality_gate, analysis, plan, segments, stage_summaries)
        status = "succeeded"
        if quality_gate["fallback_used"]:
            fallback_reason = quality_gate.get("fallback_reason") or "quality_gate_failed"
            fallback_warnings = list(quality_gate.get("warnings") or [])
            shutil.copy2(input_path, output_path)
            quality_gate = evaluate_quality_gate(input_path, output_path)
            quality_gate["fallback_used"] = True
            quality_gate["fallback_reason"] = fallback_reason
            quality_gate["warnings"] = fallback_warnings or list(quality_gate.get("warnings") or [])
            status = "fallback"

        review = summarize_segment_review(segments)
        review["reasons"] = []
        if quality_gate["fallback_used"]:
            review["needs_review"] = True
            review["reasons"].append("quality_gate_fallback")
        if quality_gate.get("warnings"):
            review["quality_warnings"] = list(quality_gate["warnings"])
        if review.get("uncertain_segment_count", 0) > 0:
            review["needs_review"] = True
            review["reasons"].append("uncertain_segments")
        sfx_summary = summarize_sfx_segments(segments)
        review["sfx_summary"] = sfx_summary
        if stage_summaries:
            review["stage_summary_keys"] = sorted(stage_summaries.keys())

        result = build_result_payload(
            status=status,
            input_path=input_path,
            output_path=output_path,
            analysis=analysis,
            selected_plan=plan,
            segments=segments,
            log=log,
            error="",
            quality_gate=quality_gate,
            job_dir=job_dir,
            review=review,
            stage_summaries=stage_summaries,
        )
        with open(job_dir / "auto_result.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        append_review_record(result, REVIEW_QUEUE_PATH)
        if progress_callback:
            progress_callback(progress_event("finalizing", 1.0, "done"))
        return result
    except Exception as exc:
        error = f"{exc}\n{traceback.format_exc()}"
        result = failed_response(input_path, output_path, error, log)
        append_review_record(result, REVIEW_QUEUE_PATH)
        return result
