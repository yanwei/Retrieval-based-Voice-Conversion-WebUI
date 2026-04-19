from __future__ import annotations

import json
import random
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sample_library import LABELS_PATH, SAMPLE_LIBRARY_ROOT

REVIEW_CANDIDATES_PATH = SAMPLE_LIBRARY_ROOT / "review_candidates.json"
ANALYSIS_DB = REPO_ROOT / "outputs" / "audio_analysis" / "book_res_full_20260418" / "audio_features.sqlite"
ROUND2_PATH = SAMPLE_LIBRARY_ROOT / "review_candidates_round2.json"
ROUND2_ROOT = SAMPLE_LIBRARY_ROOT / "review_set_round2"
REPORT_JSON_PATH = SAMPLE_LIBRARY_ROOT / "label_diff_report.json"
REPORT_MD_PATH = SAMPLE_LIBRARY_ROOT / "label_diff_report.md"
ROUND2_RANDOM_SEED = 20260418

ROUND2_DURATION_QUOTAS = {
    "<1s": 3,
    "1-3s": 10,
    "3-8s": 12,
    "8-25s": 8,
    "25-60s": 6,
    "60-180s": 5,
    ">180s": 4,
}
ROUND2_SHORT_CLEAN_REPETITIVE_LIMIT = 4

TARGET_BUCKETS = [
    ("zh", "single_male", "no_music"),
    ("zh", "single_female", "no_music"),
    ("zh", "male_female_mixed", "no_music"),
    ("zh", "multi_speaker_other", "transient_sfx"),
    ("zh", "single_male", "bgm"),
    ("zh", "single_female", "bgm"),
    ("zh", "male_female_mixed", "bgm"),
    ("zh", "single_male", "song"),
    ("zh", "single_female", "song"),
    ("en", "single_male", "no_music"),
    ("en", "single_female", "no_music"),
    ("en", "male_female_mixed", "no_music"),
    ("en", "single_male", "transient_sfx"),
    ("en", "single_female", "transient_sfx"),
    ("en", "male_female_mixed", "transient_sfx"),
    ("en", "single_male", "bgm"),
    ("en", "single_female", "bgm"),
    ("en", "male_female_mixed", "bgm"),
    ("en", "single_male", "song"),
    ("en", "single_female", "song"),
    ("en", "male_female_mixed", "song"),
]

TARGET_MIN_COUNTS = {
    ("zh", "single_male", "no_music"): 8,
    ("zh", "single_female", "no_music"): 8,
    ("zh", "male_female_mixed", "no_music"): 6,
    ("zh", "multi_speaker_other", "transient_sfx"): 4,
    ("zh", "single_male", "bgm"): 4,
    ("zh", "single_female", "bgm"): 4,
    ("zh", "male_female_mixed", "bgm"): 4,
    ("zh", "single_male", "song"): 3,
    ("zh", "single_female", "song"): 3,
    ("en", "single_male", "no_music"): 8,
    ("en", "single_female", "no_music"): 8,
    ("en", "male_female_mixed", "no_music"): 6,
    ("en", "single_male", "transient_sfx"): 4,
    ("en", "single_female", "transient_sfx"): 4,
    ("en", "male_female_mixed", "transient_sfx"): 4,
    ("en", "single_male", "bgm"): 4,
    ("en", "single_female", "bgm"): 4,
    ("en", "male_female_mixed", "bgm"): 4,
    ("en", "single_male", "song"): 4,
    ("en", "single_female", "song"): 4,
    ("en", "male_female_mixed", "song"): 4,
}


@dataclass
class Round2Candidate:
    sample_id: str
    path: str
    source: str
    language: str
    duration_bucket: str
    speaker_pattern: str
    music_pattern: str
    voice_age: str
    content_type: str
    expected_processing_mode: str
    quality_label: str
    book_id: str
    notes: list[str]
    duration_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "path": self.path,
            "source": self.source,
            "language": self.language,
            "duration_bucket": self.duration_bucket,
            "speaker_pattern": self.speaker_pattern,
            "music_pattern": self.music_pattern,
            "voice_age": self.voice_age,
            "content_type": self.content_type,
            "expected_processing_mode": self.expected_processing_mode,
            "quality_label": self.quality_label,
            "book_id": self.book_id,
            "notes": list(self.notes),
            "duration_seconds": self.duration_seconds,
        }


def load_review_candidates(path: Path = REVIEW_CANDIDATES_PATH) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if ROUND2_PATH.exists():
        rows.extend(json.loads(ROUND2_PATH.read_text(encoding="utf-8")))
    return {row["sample_id"]: row for row in rows}


def load_latest_labels(path: Path = LABELS_PATH) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        latest[payload["sample_id"]] = payload["labels"]
    return latest


def build_diff_summary(review_candidates: dict[str, dict], latest_labels: dict[str, dict]) -> dict[str, object]:
    field_names = ["language", "speaker_pattern", "voice_age", "music_pattern"]
    field_diff_counts = Counter()
    diff_patterns = Counter()
    actual_combo_counts = Counter()
    for sample_id, labels in latest_labels.items():
        candidate = review_candidates.get(sample_id, {})
        actual_combo_counts[
            (
                labels.get("language"),
                labels.get("speaker_pattern"),
                labels.get("voice_age", "unknown"),
                labels.get("music_pattern"),
            )
        ] += 1
        for field_name in field_names:
            predicted = candidate.get(field_name)
            actual = labels.get(field_name)
            if (predicted or "") != (actual or ""):
                field_diff_counts[field_name] += 1
                diff_patterns[(field_name, predicted or "", actual or "")] += 1

    single_male_mispredicted = []
    for sample_id, labels in latest_labels.items():
        if labels.get("speaker_pattern") != "single_male":
            continue
        candidate = review_candidates.get(sample_id, {})
        if candidate.get("speaker_pattern") != "single_male":
            single_male_mispredicted.append(
                {
                    "sample_id": sample_id,
                    "predicted_speaker_pattern": candidate.get("speaker_pattern"),
                    "predicted_music_pattern": candidate.get("music_pattern"),
                    "actual_music_pattern": labels.get("music_pattern"),
                    "path": candidate.get("path"),
                }
            )

    actual_song = [
        {
            "sample_id": sample_id,
            "predicted_music_pattern": review_candidates.get(sample_id, {}).get("music_pattern"),
            "path": review_candidates.get(sample_id, {}).get("path"),
        }
        for sample_id, labels in latest_labels.items()
        if labels.get("music_pattern") == "song"
    ]

    labeled_bucket_counts = Counter(
        (labels.get("language"), labels.get("speaker_pattern"), labels.get("music_pattern"))
        for labels in latest_labels.values()
    )
    deficits = []
    for bucket in TARGET_BUCKETS:
        deficits.append(
            {
                "bucket": bucket,
                "current": labeled_bucket_counts.get(bucket, 0),
                "target": TARGET_MIN_COUNTS[bucket],
                "missing": max(TARGET_MIN_COUNTS[bucket] - labeled_bucket_counts.get(bucket, 0), 0),
            }
        )
    deficits.sort(key=lambda item: (item["missing"], item["target"]), reverse=True)

    return {
        "labeled_sample_count": len(latest_labels),
        "field_diff_counts": dict(field_diff_counts),
        "top_diff_patterns": [
            {"field": field, "predicted": pred, "actual": actual, "count": count}
            for (field, pred, actual), count in diff_patterns.most_common(25)
        ],
        "single_male_mispredicted": single_male_mispredicted,
        "actual_song_count": len(actual_song),
        "actual_song_items": actual_song,
        "actual_combo_counts": [
            {"combo": list(combo), "count": count}
            for combo, count in actual_combo_counts.most_common()
        ],
        "target_bucket_deficits": deficits,
    }


def infer_language_from_book_id(book_id: str) -> str:
    if book_id.endswith("_002011"):
        return "zh"
    if book_id.endswith("_002013"):
        return "en"
    return "mixed"


def infer_round2_labels(row: sqlite3.Row) -> tuple[str, str, str, str, str]:
    rel_path = str(row["rel_path"])
    lower_rel = rel_path.lower()
    book_id = str(row["book_id"] or rel_path.split("/", 1)[0])
    language = infer_language_from_book_id(book_id)
    gender_guess = str(row["gender_guess"] or "")
    audio_type = str(row["audio_type"] or "")
    voiced_coverage = float(row["voiced_coverage"] or 0.0)
    transient_event_count = int(row["transient_event_count"] or 0)

    if "chant" in lower_rel or "song" in lower_rel:
        music_pattern = "song"
    elif audio_type == "music_or_bgm_candidate":
        music_pattern = "bgm"
    elif transient_event_count > 0 or audio_type == "fragmented_or_sfx_candidate":
        music_pattern = "transient_sfx"
    else:
        music_pattern = "no_music"

    if audio_type == "clean_or_mixed_speech_candidate":
        speaker_pattern = "male_female_mixed"
    elif gender_guess == "male":
        speaker_pattern = "single_male"
    elif gender_guess == "female":
        speaker_pattern = "single_female"
    else:
        speaker_pattern = "multi_speaker_other"

    voice_age = "child" if "child" in lower_rel or "boy" in lower_rel or "girl" in lower_rel else "adult"

    if music_pattern in {"bgm", "song"}:
        expected_processing_mode = "separate_bgm_voice" if speaker_pattern in {"single_male", "single_female"} else "long_mixed_pipeline"
    else:
        expected_processing_mode = "single_voice" if speaker_pattern in {"single_male", "single_female"} else "clean_voice_segments"

    return language, speaker_pattern, music_pattern, voice_age, expected_processing_mode


def sample_family_key(rel_path: str) -> str:
    stem = Path(rel_path).stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] in {"page", "unit"}:
        return "_".join(parts[:2])
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def is_likely_repetitive_word(row: sqlite3.Row, speaker_pattern: str, music_pattern: str) -> bool:
    duration = float(row["duration_seconds"] or 0.0)
    if duration <= 2.0 and music_pattern == "no_music":
        return True
    return False


def reorder_candidates_for_labeling(candidates: list[Round2Candidate]) -> list[Round2Candidate]:
    duration_order = [">180s", "25-60s", "3-8s", "60-180s", "8-25s", "1-3s", "<1s"]
    buckets: dict[str, list[Round2Candidate]] = defaultdict(list)
    for candidate in candidates:
        buckets[candidate.duration_bucket].append(candidate)

    for duration_bucket, items in buckets.items():
        items.sort(
            key=lambda item: (
                item.language,
                item.music_pattern,
                item.speaker_pattern,
                item.sample_id,
            )
        )

    ordered: list[Round2Candidate] = []
    last_combo: tuple[str, str, str] | None = None
    while sum(len(items) for items in buckets.values()):
        progressed = False
        for duration_bucket in duration_order:
            items = buckets.get(duration_bucket) or []
            if not items:
                continue
            pick_index = 0
            for idx, candidate in enumerate(items):
                combo = (candidate.language, candidate.speaker_pattern, candidate.music_pattern)
                if combo != last_combo:
                    pick_index = idx
                    break
            candidate = items.pop(pick_index)
            ordered.append(candidate)
            last_combo = (candidate.language, candidate.speaker_pattern, candidate.music_pattern)
            progressed = True
        if not progressed:
            break
    return ordered


def select_round2_candidates(
    analysis_db: Path,
    existing_paths: set[str],
    latest_labels: dict[str, dict],
    *,
    existing_family_keys: set[str] | None = None,
    max_items: int = 48,
) -> list[Round2Candidate]:
    labeled_bucket_counts = Counter(
        (labels.get("language"), labels.get("speaker_pattern"), labels.get("music_pattern"))
        for labels in latest_labels.values()
    )
    needed = {
        bucket: max(TARGET_MIN_COUNTS[bucket] - labeled_bucket_counts.get(bucket, 0), 0)
        for bucket in TARGET_BUCKETS
    }
    bucket_order = sorted(TARGET_BUCKETS, key=lambda bucket: (needed[bucket], TARGET_MIN_COUNTS[bucket]), reverse=True)

    conn = sqlite3.connect(analysis_db)
    conn.row_factory = sqlite3.Row
    selected: list[Round2Candidate] = []
    selected_paths: set[str] = set(existing_paths)
    selected_families: set[str] = set(existing_family_keys or set())
    selected_duration_counts: Counter[str] = Counter()
    short_word_count = 0
    short_word_bucket_counts: Counter[tuple[str, str, str]] = Counter()

    def try_add_row(row: sqlite3.Row, bucket: tuple[str, str, str], note_prefix: str) -> bool:
        nonlocal short_word_count
        abs_path = Path(str(row["abs_path"])).resolve()
        if str(abs_path) in selected_paths:
            return False
        language, speaker_pattern, music_pattern, voice_age, expected_processing_mode = infer_round2_labels(row)
        rel_path = str(row["rel_path"])
        book_id = rel_path.split("/", 1)[0]
        family_key = f"{book_id}:{sample_family_key(rel_path)}"
        if family_key in selected_families:
            return False
        duration_bucket = str(row["duration_bucket"] or "unknown")
        if (
            duration_bucket in ROUND2_DURATION_QUOTAS
            and selected_duration_counts[duration_bucket] >= ROUND2_DURATION_QUOTAS[duration_bucket]
        ):
            return False
        if is_likely_repetitive_word(row, speaker_pattern, music_pattern):
            if (
                short_word_count >= ROUND2_SHORT_CLEAN_REPETITIVE_LIMIT
                or short_word_bucket_counts[bucket] >= 1
            ):
                return False
        selected.append(
            Round2Candidate(
                sample_id=f"round2_{book_id}_{abs_path.stem}",
                path=str(abs_path),
                source="book_res_candidate_round2",
                language=language,
                duration_bucket=duration_bucket,
                speaker_pattern=speaker_pattern,
                music_pattern=music_pattern,
                voice_age=voice_age,
                content_type="mixed_lesson" if music_pattern in {"bgm", "song"} else "clean_reading",
                expected_processing_mode=expected_processing_mode,
                quality_label="candidate_round2",
                book_id=book_id,
                notes=[
                    str(row["audio_type"] or ""),
                    f"family={family_key}",
                    f"{note_prefix}={book_id}:{row['audio_type']}:{row['duration_bucket']}",
                ],
                duration_seconds=float(row["duration_seconds"] or 0.0),
            )
        )
        selected_paths.add(str(abs_path))
        selected_families.add(family_key)
        selected_duration_counts[duration_bucket] += 1
        if is_likely_repetitive_word(row, speaker_pattern, music_pattern):
            short_word_count += 1
            short_word_bucket_counts[bucket] += 1
        return True

    try:
        cursor = conn.execute(
            """
            SELECT f.book_id, f.rel_path, f.abs_path, f.duration_bucket, f.audio_type, f.voiced_coverage,
                   COALESCE(f.duration_seconds, 0.0) AS duration_seconds,
                   COALESCE(d.gender_guess, '') AS gender_guess,
                   COALESCE(d.transient_event_count, 0) AS transient_event_count
            FROM files f
            LEFT JOIN deep_features d ON d.rel_path = f.rel_path
            WHERE f.suffix = '.mp3'
            ORDER BY
              CASE WHEN f.book_id IN ('tape3a_002013', 'tape3a_002011') THEN 0 ELSE 1 END,
              f.duration_bucket, f.book_id, f.rel_path
            """
        )
        bucket_pool: dict[tuple[str, str, str], list[sqlite3.Row]] = defaultdict(list)
        proxy_pool: dict[tuple[str, str, str], list[sqlite3.Row]] = defaultdict(list)
        for row in cursor:
            abs_path = str(Path(str(row["abs_path"])).resolve())
            if abs_path in selected_paths:
                continue
            language, speaker_pattern, music_pattern, voice_age, expected_processing_mode = infer_round2_labels(row)
            bucket_key = (language, speaker_pattern, music_pattern)
            rel_path = str(row["rel_path"])
            book_id = rel_path.split("/", 1)[0]
            proxy_key = (book_id, str(row["audio_type"] or "unknown"), str(row["duration_bucket"] or "unknown"))
            proxy_pool[proxy_key].append(row)
            if bucket_key not in TARGET_MIN_COUNTS:
                continue
            bucket_pool[bucket_key].append(row)

        rng = random.Random(ROUND2_RANDOM_SEED)
        for rows in [*bucket_pool.values(), *proxy_pool.values()]:
            rng.shuffle(rows)
            rows.sort(
                key=lambda row: (
                    selected_duration_counts[str(row["duration_bucket"] or "unknown")],
                    rng.random(),
                )
            )

        bucket_selected_counts: Counter[tuple[str, str, str]] = Counter()
        while len(selected) < max_items:
            progressed = False
            for bucket in bucket_order:
                if needed[bucket] <= 0 or bucket_selected_counts[bucket] >= needed[bucket]:
                    continue
                rows = bucket_pool.get(bucket, [])
                while rows:
                    row = rows.pop(0)
                    if try_add_row(row, bucket, "target_bucket"):
                        bucket_selected_counts[bucket] += 1
                        progressed = True
                        break
                if len(selected) >= max_items:
                    return reorder_candidates_for_labeling(selected)
            if not progressed:
                break

        proxy_keys = list(proxy_pool.keys())
        rng.shuffle(proxy_keys)
        proxy_keys.sort(
            key=lambda key: (
                0 if "music" in key[1] or "sfx" in key[1] or "mixed" in key[1] else 1,
                rng.random(),
            )
        )
        proxy_offsets = Counter()
        while len(selected) < max_items:
            progressed = False
            for proxy_key in proxy_keys:
                rows = proxy_pool[proxy_key]
                offset = proxy_offsets[proxy_key]
                while offset < len(rows):
                    row = rows[offset]
                    proxy_offsets[proxy_key] += 1
                    offset += 1
                    language, speaker_pattern, music_pattern, *_ = infer_round2_labels(row)
                    bucket = (language, speaker_pattern, music_pattern)
                    if try_add_row(row, bucket, "proxy_bucket"):
                        progressed = True
                        break
                if len(selected) >= max_items:
                    return reorder_candidates_for_labeling(selected)
            if not progressed:
                break
    finally:
        conn.close()
    return reorder_candidates_for_labeling(selected)


def copy_round2_assets(candidates: list[Round2Candidate], root: Path = ROUND2_ROOT) -> list[dict[str, object]]:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    copied = []
    for candidate in candidates:
        source_path = Path(candidate.path)
        target_dir = root / candidate.book_id / f"{candidate.language}_{candidate.speaker_pattern}_{candidate.music_pattern}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        payload = candidate.to_dict()
        payload["path"] = str(target_path.resolve())
        copied.append(payload)
    return copied


def write_report_markdown(report: dict[str, object], target: Path = REPORT_MD_PATH) -> None:
    lines = [
        "# Sample Label Diff Report",
        "",
        f"- labeled samples: `{report['labeled_sample_count']}`",
        f"- field diff counts: `{json.dumps(report['field_diff_counts'], ensure_ascii=False)}`",
        "",
        "## Top diff patterns",
        "",
    ]
    for item in report["top_diff_patterns"]:
        lines.append(
            f"- `{item['count']}` x `{item['field']}`: `{item['predicted']}` -> `{item['actual']}`"
        )
    lines.extend(["", "## Single male mispredictions", ""])
    for item in report["single_male_mispredicted"][:30]:
        path = item.get("path") or item["sample_id"]
        lines.append(
            f"- `{Path(path).name}`: speaker `{item['predicted_speaker_pattern']}`, music `{item['predicted_music_pattern']}` -> actual `single_male/{item['actual_music_pattern']}`"
        )
    lines.extend(["", "## Song / chant mismatches", ""])
    lines.append(f"- actual `song` count: `{report['actual_song_count']}`")
    for item in report["actual_song_items"]:
        path = item.get("path") or item["sample_id"]
        lines.append(
            f"- `{Path(path).name}` predicted `{item['predicted_music_pattern']}` -> actual `song`"
        )
    lines.extend(["", "## Human-labeled bucket coverage", ""])
    for item in report["actual_combo_counts"]:
        combo = tuple(item["combo"])
        lines.append(f"- `{item['count']}` x `{combo}`")
    lines.extend(["", "## Missing MECE buckets", ""])
    for item in report["target_bucket_deficits"][:20]:
        lines.append(
            f"- `{tuple(item['bucket'])}` current `{item['current']}` / target `{item['target']}` / missing `{item['missing']}`"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    review_candidates = load_review_candidates()
    latest_labels = load_latest_labels()
    report = build_diff_summary(review_candidates, latest_labels)
    REPORT_JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report_markdown(report)

    existing_paths = {str(Path(item["path"]).resolve()) for item in review_candidates.values()}
    labeled_sample_ids = set(latest_labels.keys())
    existing_family_keys = {
        f"{item.get('book_id') or 'unknown'}:{sample_family_key(str(item.get('path') or ''))}"
        for item in review_candidates.values()
        if item.get("sample_id") in labeled_sample_ids
    }
    existing_paths.update(
        str(Path(item["path"]).resolve())
        for item in review_candidates.values()
        if item.get("sample_id") in labeled_sample_ids
    )
    round2 = select_round2_candidates(
        ANALYSIS_DB,
        existing_paths,
        latest_labels,
        existing_family_keys=existing_family_keys,
    )
    copied = copy_round2_assets(round2)
    ROUND2_PATH.write_text(json.dumps(copied, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Diff report written to {REPORT_MD_PATH}")
    print(f"Round2 candidates written to {ROUND2_PATH}")


if __name__ == "__main__":
    main()
