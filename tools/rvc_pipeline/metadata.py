from __future__ import annotations

import json
from pathlib import Path


def build_result_payload(
    *,
    status: str,
    input_path: Path,
    output_path: Path,
    analysis: dict,
    selected_plan: dict,
    segments: list[dict],
    log: list[str],
    error: str,
    quality_gate: dict,
    job_dir: Path,
    review: dict | None = None,
    stage_summaries: dict | None = None,
) -> dict:
    return {
        "status": status,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "analysis": analysis,
        "selected_plan": selected_plan,
        "segments": segments,
        "log": log,
        "error": error,
        "quality_gate": quality_gate,
        "job_dir": str(job_dir),
        "stage_summaries": stage_summaries or {},
        "review": review
        or {
            "needs_review": False,
            "reasons": [],
            "uncertain_segment_count": 0,
            "uncertain_segments": [],
        },
    }


def failed_response_payload(
    *,
    input_path: Path,
    output_path: Path,
    error: str,
    log: list[str] | None = None,
    quality_gate: dict | None = None,
) -> dict:
    return {
        "status": "failed",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "analysis": {},
        "selected_plan": {},
        "segments": [],
        "log": log or [],
        "error": error,
        "quality_gate": quality_gate
        or {
            "passed": False,
            "fallback_used": False,
            "fallback_reason": None,
            "warnings": [],
        },
        "job_dir": "",
        "stage_summaries": {},
        "review": {
            "needs_review": True,
            "reasons": ["failed_response"],
            "uncertain_segment_count": 0,
            "uncertain_segments": [],
        },
    }


def append_review_record(result: dict, queue_path: Path) -> None:
    review = result.get("review") or {}
    quality_gate = result.get("quality_gate") or {}
    if not review.get("needs_review") and not quality_gate.get("fallback_used"):
        return

    queue_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "input_path": result.get("input_path"),
        "output_path": result.get("output_path"),
        "status": result.get("status"),
        "processing_mode": (result.get("selected_plan") or {}).get("processing_mode"),
        "analysis": result.get("analysis"),
        "quality_gate": quality_gate,
        "review": review,
        "job_dir": result.get("job_dir"),
        "stage_summaries": result.get("stage_summaries") or {},
    }
    with open(queue_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
