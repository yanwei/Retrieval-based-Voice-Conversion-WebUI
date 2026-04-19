from __future__ import annotations

from pathlib import Path

import soundfile as sf

from tools import process_mixed_long_audio as mixed_audio
from .sfx_detector import infer_segment_type
from .speaker_router import assign_speaker_cluster_ids, summarize_uncertain_segments


def build_clean_segments(
    *,
    source_audio_path: Path,
    source_audio,
    source_sr: int,
    rmvpe,
) -> list[dict]:
    _, _, intervals = mixed_audio.detect_segments(source_audio_path)
    intervals = mixed_audio.merge_adjacent_same_route(source_audio, source_sr, intervals, rmvpe)
    analyzed_segments = mixed_audio.analyze_intervals(source_audio, source_sr, intervals, rmvpe)
    mixed_audio.absorb_short_passthrough_segments(analyzed_segments, source_sr)
    analyzed_segments = mixed_audio.merge_context_absorbed_segments(
        source_audio, source_sr, analyzed_segments
    )

    normalized_segments = []
    for idx, segment in enumerate(analyzed_segments):
        enriched = dict(segment)
        enriched["segment_id"] = idx
        enriched["segment_type"] = infer_segment_type(segment)
        normalized_segments.append(enriched)

    return assign_speaker_cluster_ids(normalized_segments)


def summarize_segment_review(segments: list[dict]) -> dict:
    uncertain_segments = summarize_uncertain_segments(segments)
    needs_review = bool(uncertain_segments)
    return {
        "needs_review": needs_review,
        "uncertain_segment_count": len(uncertain_segments),
        "uncertain_segments": uncertain_segments,
    }
