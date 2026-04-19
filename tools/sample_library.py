from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_LIBRARY_ROOT = REPO_ROOT / "assets" / "sample_library"
MANIFEST_PATH = SAMPLE_LIBRARY_ROOT / "manifest.jsonl"
LABELS_PATH = SAMPLE_LIBRARY_ROOT / "labels" / "manual_labels.jsonl"

SOURCE_PRIORITY = {
    "golden_set": 50,
    "rvc_sample": 45,
    "book_res_error_case": 40,
    "review_queue": 35,
    "book_res_candidate": 20,
}


@dataclass
class SampleEntry:
    sample_id: str
    path: str
    source: str
    language: str
    duration_bucket: str
    speaker_pattern: str
    music_pattern: str
    content_type: str
    expected_processing_mode: str
    quality_label: str
    voice_age: str | None = None
    book_id: str | None = None
    notes: list[str] | None = None
    output_path: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["notes"] = list(self.notes or [])
        return payload


def infer_legacy_labels(folder_name: str) -> dict[str, str]:
    name = folder_name.strip()
    language = "mixed"
    if "中文" in name:
        language = "zh"
    elif "英文" in name:
        language = "en"

    duration_bucket = "medium"
    if "短音频" in name:
        duration_bucket = "short"
    elif "长音频" in name:
        duration_bucket = "long"

    speaker_pattern = "multi_speaker_other"
    if "纯男声" in name:
        speaker_pattern = "single_male"
    elif "纯女声" in name:
        speaker_pattern = "single_female"
    elif "男声+女声" in name:
        speaker_pattern = "male_female_mixed"

    music_pattern = "no_music"
    if "音乐" in name:
        music_pattern = "bgm"

    content_type = "clean_reading"
    if speaker_pattern == "male_female_mixed":
        content_type = "dialogue" if music_pattern == "no_music" else "mixed_lesson"
    if music_pattern == "bgm" and duration_bucket == "long":
        content_type = "mixed_lesson"

    expected_processing_mode = "single_voice"
    if speaker_pattern == "male_female_mixed" and music_pattern == "no_music":
        expected_processing_mode = "clean_voice_segments"
    elif music_pattern == "bgm" and duration_bucket == "short":
        expected_processing_mode = "separate_bgm_voice"
    elif music_pattern == "bgm" and duration_bucket == "long":
        expected_processing_mode = "long_mixed_pipeline"

    return {
        "language": language,
        "duration_bucket": duration_bucket,
        "speaker_pattern": speaker_pattern,
        "music_pattern": music_pattern,
        "content_type": content_type,
        "expected_processing_mode": expected_processing_mode,
    }


def dedupe_entries(entries: list[SampleEntry]) -> list[SampleEntry]:
    winners: dict[str, SampleEntry] = {}
    for entry in entries:
        key = str(Path(entry.path).as_posix())
        existing = winners.get(key)
        if existing is None:
            winners[key] = entry
            continue
        existing_priority = SOURCE_PRIORITY.get(existing.source, 0)
        current_priority = SOURCE_PRIORITY.get(entry.source, 0)
        if current_priority > existing_priority:
            winners[key] = entry
    return sorted(winners.values(), key=lambda item: (item.source, item.path))


def select_review_subset(entries: list[SampleEntry], max_items: int = 96) -> list[SampleEntry]:
    buckets: dict[tuple[str, str, str], list[SampleEntry]] = {}
    for entry in entries:
        key = (entry.language, entry.music_pattern, entry.speaker_pattern)
        buckets.setdefault(key, []).append(entry)

    for bucket_entries in buckets.values():
        bucket_entries.sort(
            key=lambda item: (
                SOURCE_PRIORITY.get(item.source, 0),
                item.book_id or "",
                item.sample_id,
            ),
            reverse=True,
        )

    selected: list[SampleEntry] = []
    seen: set[str] = set()
    while len(selected) < max_items:
        progressed = False
        for key in sorted(buckets.keys()):
            if not buckets[key]:
                continue
            candidate = buckets[key].pop(0)
            if candidate.sample_id in seen:
                continue
            selected.append(candidate)
            seen.add(candidate.sample_id)
            progressed = True
            if len(selected) >= max_items:
                break
        if not progressed:
            break
    return selected


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> list[dict]:
    if not manifest_path.exists():
        return []
    rows: list[dict] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def write_manifest(entries: list[SampleEntry], manifest_path: Path = MANIFEST_PATH) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(entry.to_dict(), ensure_ascii=False) for entry in entries]
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_label_record(labels_path: Path, payload: dict) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record.setdefault("updated_at", datetime.now().isoformat(timespec="seconds"))
    with open(labels_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
