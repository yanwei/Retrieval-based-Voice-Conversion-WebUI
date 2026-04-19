from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ThresholdProfile:
    light_sample_seconds: float = 15.0
    vocal_split_top_db: int = 35
    very_short_risk_seconds: float = 0.75
    ultra_short_single_seconds: float = 1.2
    short_single_voice_seconds: float = 3.0
    short_voice_upper_seconds: float = 8.0
    long_audio_min_seconds: float = 25.0
    short_clean_voice_min_ratio: float = 0.55
    music_risk_safe_threshold: float = 0.25
    music_candidate_harmonic_ratio: float = 0.68
    music_candidate_percussive_ratio: float = 0.22
    clean_speech_voiced_coverage: float = 0.70
    clean_mixed_voiced_coverage: float = 0.45
    min_voiced_frames: int = 16

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_THRESHOLDS = ThresholdProfile()
DEFAULT_SUMMARY_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "audio_analysis"
    / "book_res_full_20260418"
    / "summary.json"
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def build_dataset_thresholds(summary: dict) -> ThresholdProfile:
    duration = summary.get("duration_seconds", {})
    p05 = float(duration.get("p05", DEFAULT_THRESHOLDS.very_short_risk_seconds))
    p25 = float(duration.get("p25", DEFAULT_THRESHOLDS.ultra_short_single_seconds))
    p50 = float(duration.get("p50", DEFAULT_THRESHOLDS.short_single_voice_seconds))
    p75 = float(duration.get("p75", DEFAULT_THRESHOLDS.short_voice_upper_seconds))
    p90 = float(duration.get("p90", DEFAULT_THRESHOLDS.long_audio_min_seconds))

    return ThresholdProfile(
        light_sample_seconds=DEFAULT_THRESHOLDS.light_sample_seconds,
        vocal_split_top_db=DEFAULT_THRESHOLDS.vocal_split_top_db,
        very_short_risk_seconds=_clamp(p05 * 1.1, 0.35, 0.9),
        ultra_short_single_seconds=_clamp(p25 * 1.05, 0.8, 1.5),
        short_single_voice_seconds=_clamp(p50 * 1.2, 1.8, 4.0),
        short_voice_upper_seconds=_clamp(p75 * 1.4, 4.0, 10.0),
        long_audio_min_seconds=_clamp(p90 * 1.2, 20.0, 45.0),
        short_clean_voice_min_ratio=DEFAULT_THRESHOLDS.short_clean_voice_min_ratio,
        music_risk_safe_threshold=DEFAULT_THRESHOLDS.music_risk_safe_threshold,
        music_candidate_harmonic_ratio=DEFAULT_THRESHOLDS.music_candidate_harmonic_ratio,
        music_candidate_percussive_ratio=DEFAULT_THRESHOLDS.music_candidate_percussive_ratio,
        clean_speech_voiced_coverage=DEFAULT_THRESHOLDS.clean_speech_voiced_coverage,
        clean_mixed_voiced_coverage=DEFAULT_THRESHOLDS.clean_mixed_voiced_coverage,
        min_voiced_frames=DEFAULT_THRESHOLDS.min_voiced_frames,
    )


def load_dataset_thresholds(summary_path: Path | None = None) -> ThresholdProfile:
    candidate = summary_path or DEFAULT_SUMMARY_PATH
    try:
        payload = json.loads(Path(candidate).read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_THRESHOLDS
    return build_dataset_thresholds(payload)


ACTIVE_THRESHOLDS = load_dataset_thresholds()
