from __future__ import annotations


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_gender_confidence(segment: dict) -> float:
    note = str(segment.get("note", ""))
    voiced_ratio = _safe_float(segment.get("voiced_ratio", 0.0))
    duration = _safe_float(segment.get("duration_sec", 0.0))

    confidence = 0.55
    if note == "ok":
        confidence = 0.55 if voiced_ratio < 0.35 else 0.9
    elif "borderline" in note or "smoothed" in note:
        confidence = 0.65
    elif "failed" in note or segment.get("route") == "passthrough":
        confidence = 0.25

    if voiced_ratio >= 0.7:
        confidence += 0.05
    if duration >= 1.5:
        confidence += 0.05
    return round(min(confidence, 0.98), 4)


def infer_segment_duration(segment: dict) -> float | None:
    if segment.get("duration_sec") is not None:
        return _safe_float(segment.get("duration_sec"), 0.0)
    if segment.get("start") is None or segment.get("end") is None:
        return None
    duration = _safe_float(segment.get("end"), 0.0) - _safe_float(segment.get("start"), 0.0)
    return round(max(0.0, duration), 4)


def normalize_review_segment(segment: dict) -> dict:
    normalized = dict(segment)
    duration = infer_segment_duration(normalized)
    if duration is not None:
        normalized["duration_sec"] = duration
    if normalized.get("gender_confidence") is None:
        normalized["gender_confidence"] = infer_gender_confidence(normalized)
    return normalized


def assign_speaker_cluster_ids(segments: list[dict]) -> list[dict]:
    cluster_counter = {"male": 0, "female": 0, "passthrough": 0, "unknown": 0}
    previous_route = None
    previous_cluster = None
    assigned: list[dict] = []
    for segment in segments:
        route = segment.get("route") or "unknown"
        if route == previous_route and previous_cluster:
            cluster_id = previous_cluster
        else:
            cluster_counter.setdefault(route, 0)
            cluster_counter[route] += 1
            cluster_id = f"{route[:1]}{cluster_counter[route]}"
        enriched = dict(segment)
        enriched["speaker_cluster_id"] = cluster_id
        enriched["gender_confidence"] = infer_gender_confidence(segment)
        assigned.append(enriched)
        previous_route = route
        previous_cluster = cluster_id
    return assigned


def summarize_uncertain_segments(segments: list[dict]) -> list[dict]:
    uncertain = []
    for segment in segments:
        segment = normalize_review_segment(segment)
        note = str(segment.get("note", ""))
        if (
            segment.get("route") == "passthrough"
            or segment.get("gender_confidence", 0.0) < 0.7
            or note != "ok"
        ):
            uncertain.append(
                {
                    "segment_id": segment.get("segment_id"),
                    "route": segment.get("route"),
                    "speaker_cluster_id": segment.get("speaker_cluster_id"),
                    "gender_confidence": segment.get("gender_confidence"),
                    "note": note,
                    "duration_sec": segment.get("duration_sec"),
                }
            )
    return uncertain
