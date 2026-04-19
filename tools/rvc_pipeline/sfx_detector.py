from __future__ import annotations


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_segment_type(segment: dict) -> str:
    route = segment.get("route", "unknown")
    note = str(segment.get("note", ""))
    duration = _safe_float(segment.get("duration_sec", 0.0))
    voiced_ratio = _safe_float(segment.get("voiced_ratio", 0.0))

    if route == "passthrough" and (note.startswith("low_voice") or voiced_ratio < 0.1):
        return "sfx"
    if "absorbed" in note and duration <= 1.0:
        return "speech"
    if route in {"male", "female"}:
        return "speech"
    if duration <= 0.5 and voiced_ratio < 0.15:
        return "sfx"
    return "mixed"


def summarize_sfx_segments(segments: list[dict]) -> dict:
    sfx_segments = [segment for segment in segments if infer_segment_type(segment) == "sfx"]
    total_duration = sum(_safe_float(segment.get("duration_sec", 0.0)) for segment in sfx_segments)
    return {
        "count": len(sfx_segments),
        "total_duration_seconds": round(total_duration, 4),
        "segment_ids": [segment.get("segment_id") for segment in sfx_segments],
    }
