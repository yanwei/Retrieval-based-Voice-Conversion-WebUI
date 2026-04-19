from __future__ import annotations

from pathlib import Path
from typing import Callable

from .thresholds import DEFAULT_THRESHOLDS, ThresholdProfile


SHORT_VOICE_ALLOWED_MUSIC_REASONS = {"rhythmic_onsets", "percussive_content"}
CLEAN_VOICE_ALLOWED_MUSIC_REASONS = {"rhythmic_onsets", "percussive_content"}


def build_analysis(
    *,
    duration: float,
    music_risk: float,
    music_reasons: list[str],
    dominant_route: str,
    voice_confidence: float,
    median_f0: float,
    voiced_frames: int,
    voiced_ratio: float,
    thresholds: ThresholdProfile = DEFAULT_THRESHOLDS,
) -> dict:
    if music_risk >= thresholds.music_risk_safe_threshold:
        classification = "mixed_with_music"
        confidence = max(0.5, min(0.95, music_risk))
    else:
        classification = dominant_route
        confidence = voice_confidence

    analysis = {
        "classification": classification,
        "confidence": round(float(confidence), 4),
        "music_risk": round(float(music_risk), 4),
        "duration_seconds": round(float(duration), 4),
        "dominant_route": dominant_route,
        "median_f0_hz": round(float(median_f0), 4),
        "voiced_frames": int(voiced_frames),
        "voiced_ratio": round(float(voiced_ratio), 4),
        "music_reasons": music_reasons,
    }
    return apply_analysis_overrides(analysis, thresholds)


def apply_analysis_overrides(
    analysis: dict,
    thresholds: ThresholdProfile = DEFAULT_THRESHOLDS,
) -> dict:
    adjusted = dict(analysis)
    reasons = set(adjusted.get("music_reasons") or [])
    route = adjusted.get("dominant_route")
    duration = float(adjusted.get("duration_seconds", 0.0))
    has_reliable_voice = (
        route in {"male", "female"}
        and float(adjusted.get("voiced_ratio", 0.0)) >= thresholds.short_clean_voice_min_ratio
        and int(adjusted.get("voiced_frames", 0)) >= thresholds.min_voiced_frames
    )
    is_short_clean_voice = (
        duration <= thresholds.short_voice_upper_seconds
        and has_reliable_voice
        and reasons.issubset(SHORT_VOICE_ALLOWED_MUSIC_REASONS)
    )
    is_clean_voice_before_long_threshold = (
        duration < thresholds.long_audio_min_seconds
        and has_reliable_voice
        and reasons.issubset(CLEAN_VOICE_ALLOWED_MUSIC_REASONS)
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


def select_processing_plan(
    analysis: dict,
    profile_config: dict,
    resolve_index_for_model: Callable[[str], str],
    thresholds: ThresholdProfile = DEFAULT_THRESHOLDS,
    profile_name: str = "default",
) -> dict:
    route = analysis.get("dominant_route") or analysis.get("classification") or "unknown"
    classification = analysis.get("classification") or route
    music_risk = float(analysis.get("music_risk", 1.0))
    duration = float(analysis.get("duration_seconds", 0.0))
    voiced_ratio = float(analysis.get("voiced_ratio", 0.0))
    short_single_voice_candidate = (
        route in {"male", "female"}
        and duration <= thresholds.short_single_voice_seconds
        and voiced_ratio >= max(0.45, thresholds.short_clean_voice_min_ratio - 0.10)
        and (
            duration <= thresholds.ultra_short_single_seconds
            or music_risk < thresholds.music_risk_safe_threshold
            or analysis.get("music_risk_overridden")
        )
    )
    uses_clean_voice_segments = (
        analysis.get("override_reason") == "clean_voice_no_music"
        and route in {"male", "female"}
        and duration > thresholds.short_single_voice_seconds
        and duration < thresholds.long_audio_min_seconds
    )
    needs_complex_long = duration >= thresholds.long_audio_min_seconds or route not in {"male", "female"}
    needs_bgm_separation = (
        (music_risk >= thresholds.music_risk_safe_threshold and not analysis.get("music_risk_overridden"))
    )

    if short_single_voice_candidate:
        model_key = "male_model" if route == "male" else "female_model"
        params_key = "male_params" if route == "male" else "female_params"
        model_name = profile_config[model_key]
        return {
            "profile": profile_name,
            "processing_mode": "single",
            "target_route": route,
            "models": [
                {
                    "role": route,
                    "model": model_name,
                    "index": resolve_index_for_model(model_name),
                }
            ],
            "parameters": profile_config[params_key],
        }

    if uses_clean_voice_segments:
        return {
            "profile": profile_name,
            "processing_mode": "clean_voice_segments",
            "target_route": route,
            "models": [
                {
                    "role": "male",
                    "model": profile_config["male_model"],
                    "index": resolve_index_for_model(profile_config["male_model"]),
                },
                {
                    "role": "female",
                    "model": profile_config["female_model"],
                    "index": resolve_index_for_model(profile_config["female_model"]),
                },
            ],
            "parameters": {
                "reading_mode": profile_config["reading_mode"],
                "speaker_embedding": profile_config["speaker_embedding"],
                "male_params": profile_config["male_params"],
                "female_params": profile_config["female_params"],
            },
        }

    if needs_complex_long:
        return {
            "profile": profile_name,
            "processing_mode": "long_mixed_pipeline",
            "target_route": route,
            "models": [
                {
                    "role": "male",
                    "model": profile_config["male_model"],
                    "index": resolve_index_for_model(profile_config["male_model"]),
                },
                {
                    "role": "female",
                    "model": profile_config["female_model"],
                    "index": resolve_index_for_model(profile_config["female_model"]),
                },
            ],
            "parameters": {
                "uvr_model": profile_config["uvr_model"],
                "reading_mode": profile_config["reading_mode"],
                "speaker_embedding": profile_config["speaker_embedding"],
                "male_params": profile_config["male_params"],
                "female_params": profile_config["female_params"],
            },
        }

    if needs_bgm_separation:
        return {
            "profile": profile_name,
            "processing_mode": "separate_bgm_voice",
            "target_route": route,
            "models": [
                {
                    "role": "male",
                    "model": profile_config["male_model"],
                    "index": resolve_index_for_model(profile_config["male_model"]),
                },
                {
                    "role": "female",
                    "model": profile_config["female_model"],
                    "index": resolve_index_for_model(profile_config["female_model"]),
                },
            ],
            "parameters": {
                "uvr_model": profile_config["uvr_model"],
                "reading_mode": profile_config["reading_mode"],
                "speaker_embedding": profile_config["speaker_embedding"],
                "male_params": profile_config["male_params"],
                "female_params": profile_config["female_params"],
            },
        }

    model_key = "male_model" if route == "male" else "female_model"
    params_key = "male_params" if route == "male" else "female_params"
    model_name = profile_config[model_key]
    return {
        "profile": profile_name,
        "processing_mode": "single",
        "target_route": route,
        "models": [
            {
                "role": route,
                "model": model_name,
                "index": resolve_index_for_model(model_name),
            }
        ],
        "parameters": profile_config[params_key],
    }
