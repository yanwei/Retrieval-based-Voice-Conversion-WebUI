from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sample_library import (
    LABELS_PATH,
    MANIFEST_PATH,
    SAMPLE_LIBRARY_ROOT,
    SampleEntry,
    dedupe_entries,
    infer_legacy_labels,
    select_review_subset,
    write_manifest,
)
DEFAULT_RVC_SAMPLE_ROOT = Path("/Users/yanwei/Downloads/RVC Sample")
DEFAULT_ANALYSIS_DB = (
    REPO_ROOT / "outputs" / "audio_analysis" / "book_res_full_20260418" / "audio_features.sqlite"
)
DEFAULT_BOOK_RES_ROOT = Path("/Users/yanwei/dev/namibox/book-res-studio/data/library/book-res")
REPRESENTATIVE_BOOKS = ("tape3a_002013", "tape3a_002011")
MAX_REVIEW_ITEMS = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build project-owned sample library assets.")
    parser.add_argument("--rvc-sample-root", type=Path, default=DEFAULT_RVC_SAMPLE_ROOT)
    parser.add_argument("--analysis-db", type=Path, default=DEFAULT_ANALYSIS_DB)
    parser.add_argument("--book-res-root", type=Path, default=DEFAULT_BOOK_RES_ROOT)
    parser.add_argument("--sample-root", type=Path, default=SAMPLE_LIBRARY_ROOT)
    parser.add_argument("--max-review-items", type=int, default=MAX_REVIEW_ITEMS)
    parser.add_argument("--copy", action="store_true", help="Copy RVC Sample instead of moving.")
    return parser.parse_args()


def migrate_legacy_rvc_sample(sample_root: Path, legacy_root: Path, *, copy_only: bool) -> list[SampleEntry]:
    entries: list[SampleEntry] = []
    if not legacy_root.exists():
        return entries

    golden_root = sample_root / "golden_set" / "rvc_sample"
    golden_root.mkdir(parents=True, exist_ok=True)
    for audio_path in sorted(legacy_root.rglob("*")):
        if not audio_path.is_file():
            continue
        if audio_path.suffix.lower() not in {".mp3", ".wav", ".m4a"}:
            continue
        relative_parent = audio_path.parent.relative_to(legacy_root)
        target_dir = golden_root / relative_parent
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / audio_path.name
        if copy_only:
            shutil.copy2(audio_path, target_path)
        else:
            shutil.move(str(audio_path), str(target_path))

        labels = infer_legacy_labels(relative_parent.name or legacy_root.name)
        sample_id = f"legacy_{target_path.stem}"
        entries.append(
            SampleEntry(
                sample_id=sample_id,
                path=str(target_path.resolve()),
                source="rvc_sample",
                language=labels["language"],
                duration_bucket=labels["duration_bucket"],
                speaker_pattern=labels["speaker_pattern"],
                music_pattern=labels["music_pattern"],
                content_type=labels["content_type"],
                expected_processing_mode=labels["expected_processing_mode"],
                quality_label="golden",
            )
        )
    return entries


def parse_voice_conversion_review_cases(
    metadata_path: Path,
    *,
    book_root: Path,
    book_id: str,
) -> list[SampleEntry]:
    if not metadata_path.exists():
        return []
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    files = (payload.get("conversion") or {}).get("files") or []
    rows: list[SampleEntry] = []
    for entry in files:
        review = entry.get("review") or {}
        quality_gate = entry.get("quality_gate") or {}
        if not (review.get("needs_review") or quality_gate.get("fallback_used") or entry.get("status") == "failed"):
            continue
        relative_path = entry.get("relative_path") or ""
        rel_parts = Path(relative_path).parts
        rel_without_book = Path(*rel_parts[1:]) if rel_parts and rel_parts[0] == book_id else Path(relative_path)
        source_path = (book_root / rel_without_book).resolve()
        if not source_path.exists():
            continue
        labels = infer_candidate_labels(
            classification=str(entry.get("classification") or ""),
            processing_mode=str(entry.get("processing_mode") or ""),
            file_name=str(entry.get("file_name") or source_path.name),
        )
        sample_id = f"{book_id}_{entry.get('resource_id') or source_path.stem}"
        rows.append(
            SampleEntry(
                sample_id=sample_id,
                path=str(source_path),
                output_path=str(entry.get("output_path") or ""),
                source="book_res_error_case",
                language=labels["language"],
                duration_bucket=labels["duration_bucket"],
                speaker_pattern=labels["speaker_pattern"],
                music_pattern=labels["music_pattern"],
                content_type=labels["content_type"],
                expected_processing_mode=labels["expected_processing_mode"],
                quality_label="known_failure",
                book_id=book_id,
                notes=list(review.get("reasons") or []),
            )
        )
    return rows


def infer_candidate_labels(*, classification: str, processing_mode: str, file_name: str) -> dict[str, str]:
    language = "zh" if "tape3a_002011" in file_name or "unit_" in file_name else "en"
    speaker_pattern = "multi_speaker_other"
    if classification == "male":
        speaker_pattern = "single_male"
    elif classification == "female":
        speaker_pattern = "single_female"
    elif "mixed" in classification:
        speaker_pattern = "male_female_mixed"
    music_pattern = "bgm" if "music" in classification or "bgm" in classification else "no_music"
    expected_processing_mode = processing_mode or "clean_voice_segments"
    content_type = "mixed_lesson" if "long" in expected_processing_mode or music_pattern == "bgm" else "clean_reading"
    return {
        "language": language,
        "duration_bucket": "short",
        "speaker_pattern": speaker_pattern,
        "music_pattern": music_pattern,
        "content_type": content_type,
        "expected_processing_mode": expected_processing_mode,
    }


def query_representative_book_candidates(analysis_db: Path, book_ids: tuple[str, ...]) -> list[SampleEntry]:
    conn = sqlite3.connect(analysis_db)
    conn.row_factory = sqlite3.Row
    rows: list[SampleEntry] = []
    try:
        for book_id in book_ids:
            cursor = conn.execute(
                """
                SELECT f.rel_path, f.abs_path, f.duration_bucket, f.audio_type, f.duration_seconds,
                       f.audio_kind, f.voiced_coverage, COALESCE(d.gender_guess, '') AS gender_guess,
                       COALESCE(d.strategy_suggestion, '') AS strategy_suggestion,
                       COALESCE(d.transient_event_count, 0) AS transient_event_count
                FROM files f
                LEFT JOIN deep_features d ON d.rel_path = f.rel_path
                WHERE f.rel_path LIKE ? AND f.suffix = '.mp3'
                ORDER BY f.audio_type, f.duration_bucket, f.rel_path
                """,
                (f"{book_id}/%",),
            )
            groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
            for row in cursor:
                key = (str(row["audio_type"] or ""), str(row["duration_bucket"] or ""))
                groups.setdefault(key, []).append(row)
            for key, items in groups.items():
                for row in items[:2]:
                    abs_path = Path(str(row["abs_path"]))
                    if not abs_path.exists():
                        continue
                    rows.append(
                        SampleEntry(
                            sample_id=f"{book_id}_{abs_path.stem}",
                            path=str(abs_path),
                            source="book_res_candidate",
                            language="en" if book_id == "tape3a_002013" else "zh",
                            duration_bucket=str(row["duration_bucket"] or "medium"),
                            speaker_pattern=map_audio_type_to_speaker_pattern(str(row["audio_type"] or "")),
                            music_pattern=map_audio_type_to_music_pattern(str(row["audio_type"] or "")),
                            content_type=map_audio_type_to_content_type(str(row["audio_type"] or "")),
                            expected_processing_mode=map_strategy_to_processing_mode(
                                str(row["strategy_suggestion"] or ""),
                                str(row["audio_type"] or ""),
                            ),
                            quality_label="candidate",
                            book_id=book_id,
                            notes=[str(row["audio_type"] or "")],
                        )
                    )
    finally:
        conn.close()
    return rows


def map_audio_type_to_speaker_pattern(audio_type: str) -> str:
    if audio_type == "clean_speech_candidate":
        return "single_female"
    if audio_type == "clean_or_mixed_speech_candidate":
        return "multi_speaker_other"
    if audio_type == "music_or_bgm_candidate":
        return "male_female_mixed"
    if audio_type == "fragmented_or_sfx_candidate":
        return "multi_speaker_other"
    return "multi_speaker_other"


def map_audio_type_to_music_pattern(audio_type: str) -> str:
    if audio_type == "music_or_bgm_candidate":
        return "bgm"
    if audio_type == "fragmented_or_sfx_candidate":
        return "transient_sfx"
    return "no_music"


def map_audio_type_to_content_type(audio_type: str) -> str:
    if audio_type == "music_or_bgm_candidate":
        return "mixed_lesson"
    if audio_type == "fragmented_or_sfx_candidate":
        return "word_list"
    if audio_type == "clean_or_mixed_speech_candidate":
        return "dialogue"
    return "clean_reading"


def map_strategy_to_processing_mode(strategy_suggestion: str, audio_type: str) -> str:
    if strategy_suggestion:
        return strategy_suggestion
    if audio_type == "music_or_bgm_candidate":
        return "separate_bgm_voice"
    if audio_type == "clean_or_mixed_speech_candidate":
        return "clean_voice_segments"
    return "single_voice"


def copy_review_assets(entries: list[SampleEntry], sample_root: Path) -> list[SampleEntry]:
    review_root = sample_root / "review_set"
    review_root.mkdir(parents=True, exist_ok=True)
    copied: list[SampleEntry] = []
    for entry in entries:
        source_path = Path(entry.path)
        if not source_path.exists():
            continue
        target_dir = review_root / (entry.book_id or "misc") / entry.quality_label
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied.append(
            SampleEntry(
                sample_id=entry.sample_id,
                path=str(target_path.resolve()),
                output_path=entry.output_path,
                source=entry.source,
                language=entry.language,
                duration_bucket=entry.duration_bucket,
                speaker_pattern=entry.speaker_pattern,
                music_pattern=entry.music_pattern,
                content_type=entry.content_type,
                expected_processing_mode=entry.expected_processing_mode,
                quality_label=entry.quality_label,
                book_id=entry.book_id,
                notes=list(entry.notes or []),
            )
        )
    return copied


def build_review_candidates(entries: list[dict] | list[SampleEntry], max_items: int = MAX_REVIEW_ITEMS) -> list[dict]:
    typed = [entry if isinstance(entry, SampleEntry) else SampleEntry(**entry) for entry in entries]
    selected = select_review_subset(dedupe_entries(typed), max_items=max_items)
    return [entry.to_dict() for entry in selected]


def main() -> None:
    args = parse_args()
    sample_root = args.sample_root
    sample_root.mkdir(parents=True, exist_ok=True)
    (sample_root / "labels").mkdir(parents=True, exist_ok=True)

    manifest_entries: list[SampleEntry] = []
    manifest_entries.extend(
        migrate_legacy_rvc_sample(sample_root, args.rvc_sample_root, copy_only=args.copy)
    )
    manifest_entries.extend(query_representative_book_candidates(args.analysis_db, REPRESENTATIVE_BOOKS))

    for book_id in REPRESENTATIVE_BOOKS:
        book_root = args.book_res_root / book_id
        manifest_entries.extend(
            parse_voice_conversion_review_cases(
                book_root / "voice_conversion.json",
                book_root=book_root,
                book_id=book_id,
            )
        )

    deduped = dedupe_entries(manifest_entries)
    review_candidates = [
        entry
        for entry in deduped
        if entry.source in {"book_res_candidate", "book_res_error_case", "review_queue"}
    ]
    copied_review_entries = copy_review_assets(
        select_review_subset(review_candidates, max_items=args.max_review_items),
        sample_root,
    )

    final_entries = dedupe_entries(
        [
            *(entry for entry in deduped if entry.source in {"rvc_sample", "golden_set"}),
            *copied_review_entries,
        ]
    )
    write_manifest(final_entries, MANIFEST_PATH)

    review_manifest = sample_root / "review_candidates.json"
    review_manifest.write_text(
        json.dumps([entry.to_dict() for entry in copied_review_entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not LABELS_PATH.exists():
        LABELS_PATH.write_text("", encoding="utf-8")
    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Review candidates written to {review_manifest}")


if __name__ == "__main__":
    main()
