from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def _load_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    return y.astype(np.float32), sr


def _voiced_coverage(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    intervals = librosa.effects.split(y, top_db=35)
    voiced = sum(int(end - start) for start, end in intervals)
    return float(voiced / max(len(y), 1))


def _rms_db(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(y * y))) if y.size else 0.0
    return 20.0 * np.log10(max(rms, 1e-8))


def _silence_ratio(y: np.ndarray) -> float:
    if y.size == 0:
        return 1.0
    intervals = librosa.effects.split(y, top_db=35)
    voiced = sum(int(end - start) for start, end in intervals)
    return float(1.0 - voiced / max(len(y), 1))


def evaluate_quality_gate(input_path: Path, output_path: Path) -> dict:
    warnings: list[str] = []
    if not Path(output_path).exists():
        return {
            "passed": False,
            "fallback_used": True,
            "fallback_reason": "output_missing",
            "duration_delta_ratio": None,
            "voiced_coverage_delta": None,
            "rms_delta_db": None,
            "clipping_detected": False,
            "warnings": ["output_missing"],
        }

    try:
        input_info = sf.info(input_path)
        output_info = sf.info(output_path)
        input_y, _ = _load_mono(input_path)
        output_y, _ = _load_mono(output_path)
    except Exception:
        return {
            "passed": False,
            "fallback_used": True,
            "fallback_reason": "output_unreadable",
            "duration_delta_ratio": None,
            "voiced_coverage_delta": None,
            "rms_delta_db": None,
            "clipping_detected": False,
            "warnings": ["output_unreadable"],
        }

    duration_delta_ratio = abs(float(output_info.duration) - float(input_info.duration)) / max(
        float(input_info.duration), 1e-6
    )
    input_voiced = _voiced_coverage(input_y)
    output_voiced = _voiced_coverage(output_y)
    voiced_coverage_delta = output_voiced - input_voiced
    rms_delta_db = _rms_db(output_y) - _rms_db(input_y)
    input_silence = _silence_ratio(input_y)
    output_silence = _silence_ratio(output_y)
    silence_ratio_delta = output_silence - input_silence
    output_peak = float(np.max(np.abs(output_y))) if output_y.size else 0.0
    clipping_detected = bool(np.max(np.abs(output_y)) >= 0.999) if output_y.size else False

    if duration_delta_ratio > 0.05:
        warnings.append("duration_mismatch")
    if voiced_coverage_delta < -0.35:
        warnings.append("voiced_coverage_drop")
    elif input_voiced > 0.1 and output_voiced <= max(0.01, input_voiced * 0.25):
        warnings.append("voiced_coverage_drop")
    if rms_delta_db < -18.0:
        warnings.append("severe_rms_drop")
    if silence_ratio_delta > 0.40:
        warnings.append("silence_ratio_spike")
    if 0.0 < output_peak < 0.02:
        warnings.append("very_low_peak")
    if clipping_detected:
        warnings.append("clipping_detected")

    passed = not warnings
    return {
        "passed": passed,
        "fallback_used": not passed,
        "fallback_reason": warnings[0] if warnings else None,
        "duration_delta_ratio": round(float(duration_delta_ratio), 6),
        "voiced_coverage_delta": round(float(voiced_coverage_delta), 6),
        "rms_delta_db": round(float(rms_delta_db), 6),
        "silence_ratio_delta": round(float(silence_ratio_delta), 6),
        "output_peak": round(float(output_peak), 6),
        "clipping_detected": clipping_detected,
        "warnings": warnings,
    }
