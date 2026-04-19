from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = REPO_ROOT / "outputs" / "rvc_auto_convert" / "review_queue.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "rvc_auto_convert" / "reviews"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review uncertain/fallback auto-convert jobs with local ollama.")
    parser.add_argument("--queue", type=str, default=str(DEFAULT_QUEUE))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", type=str, default="gemma4:26b")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--only-status", type=str, choices=["fallback", "succeeded", "failed"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_queue(queue_path: Path) -> list[dict]:
    if not queue_path.exists():
        return []
    items = []
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        items.append(json.loads(line))
    return items


def filter_queue(items: list[dict], *, only_status: str | None, limit: int) -> list[dict]:
    filtered = []
    for item in items:
        if only_status and item.get("status") != only_status:
            continue
        filtered.append(item)
        if len(filtered) >= limit:
            break
    return filtered


def build_prompt(items: list[dict]) -> str:
    return (
        "You are reviewing RVC auto-convert fallback and uncertain jobs.\n"
        "Return only valid JSON.\n"
        "Use keys: summary, recurring_patterns, suggested_actions, route_adjustments, top_examples.\n"
        "route_adjustments must be an array of objects with keys: input_path, likely_issue, suggested_mode, suggested_followup.\n"
        "Focus on why jobs were flagged, what kinds of audio they likely are, and what routing improvements are suggested.\n"
        "Do not invent fields beyond the requested JSON.\n\n"
        f"Jobs:\n{json.dumps(items, ensure_ascii=False, indent=2)}"
    )


def run_ollama_review(model: str, prompt: str) -> str:
    completed = subprocess.run(
        ["ollama", "run", model, prompt],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def parse_model_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("\n", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def main() -> int:
    args = parse_args()
    queue_path = Path(args.queue).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_queue(queue_path)
    selected = filter_queue(items, only_status=args.only_status, limit=args.limit)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    review_json = output_dir / f"review_{timestamp}.json"
    review_md = output_dir / f"review_{timestamp}.md"

    if not selected:
        payload = {
            "summary": "No matching review items.",
            "recurring_patterns": [],
            "suggested_actions": [],
            "top_examples": [],
        }
        review_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        review_md.write_text("# Review Summary\n\nNo matching review items.\n", encoding="utf-8")
        print(str(review_json))
        return 0

    prompt = build_prompt(selected)
    if args.dry_run:
        payload = {
            "summary": "Dry run only.",
            "recurring_patterns": [],
            "suggested_actions": [],
            "route_adjustments": [],
            "top_examples": selected[:3],
        }
    else:
        raw = run_ollama_review(args.model, prompt)
        payload = parse_model_json(raw)

    review_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Review Summary", "", f"- Queue: `{queue_path}`", f"- Model: `{args.model}`", ""]
    lines.append("## Summary")
    lines.append(payload.get("summary", ""))
    lines.append("")
    lines.append("## Recurring Patterns")
    for item in payload.get("recurring_patterns", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Suggested Actions")
    for item in payload.get("suggested_actions", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Route Adjustments")
    for item in payload.get("route_adjustments", []):
        lines.append(f"- `{json.dumps(item, ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Top Examples")
    for item in payload.get("top_examples", []):
        lines.append(f"- `{item}`" if isinstance(item, str) else f"- `{json.dumps(item, ensure_ascii=False)}`")
    lines.append("")
    review_md.write_text("\n".join(lines), encoding="utf-8")
    print(str(review_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
