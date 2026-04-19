import json
import tempfile
import unittest
from pathlib import Path

from tools.sample_library import (
    SampleEntry,
    dedupe_entries,
    infer_legacy_labels,
    select_review_subset,
    write_label_record,
)
from tools.sample_library_labeler import create_app, load_labeled_sample_ids


class LegacyLabelInferenceTests(unittest.TestCase):
    def test_infer_labels_for_short_english_mixed_with_music(self):
        labels = infer_legacy_labels("【短音频】英文男声+女声+音乐")

        self.assertEqual(labels["language"], "en")
        self.assertEqual(labels["duration_bucket"], "short")
        self.assertEqual(labels["speaker_pattern"], "male_female_mixed")
        self.assertEqual(labels["music_pattern"], "bgm")
        self.assertEqual(labels["content_type"], "mixed_lesson")

    def test_infer_labels_for_short_chinese_single_female(self):
        labels = infer_legacy_labels("【短音频】中文纯女声")

        self.assertEqual(labels["language"], "zh")
        self.assertEqual(labels["duration_bucket"], "short")
        self.assertEqual(labels["speaker_pattern"], "single_female")
        self.assertEqual(labels["music_pattern"], "no_music")
        self.assertEqual(labels["content_type"], "clean_reading")


class EntrySelectionTests(unittest.TestCase):
    def test_dedupe_entries_prefers_higher_priority_source(self):
        low = SampleEntry(
            sample_id="a",
            path="x/a.mp3",
            source="book_res_candidate",
            language="zh",
            duration_bucket="short",
            speaker_pattern="single_female",
            music_pattern="no_music",
            content_type="clean_reading",
            expected_processing_mode="single_voice",
            quality_label="candidate",
            book_id="book_a",
        )
        high = SampleEntry(
            sample_id="b",
            path="x/a.mp3",
            source="book_res_error_case",
            language="zh",
            duration_bucket="short",
            speaker_pattern="single_female",
            music_pattern="no_music",
            content_type="clean_reading",
            expected_processing_mode="single_voice",
            quality_label="known_failure",
            book_id="book_a",
        )

        result = dedupe_entries([low, high])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source, "book_res_error_case")
        self.assertEqual(result[0].quality_label, "known_failure")

    def test_select_review_subset_respects_max_items(self):
        entries = [
            SampleEntry(
                sample_id=f"id_{idx}",
                path=f"x/{idx}.mp3",
                source="book_res_candidate",
                language="zh" if idx % 2 == 0 else "en",
                duration_bucket="short",
                speaker_pattern="single_male",
                music_pattern="no_music",
                content_type="clean_reading",
                expected_processing_mode="single_voice",
                quality_label="candidate",
                book_id="book_a",
            )
            for idx in range(20)
        ]

        selected = select_review_subset(entries, max_items=7)

        self.assertEqual(len(selected), 7)
        self.assertEqual(len({item.sample_id for item in selected}), 7)

    def test_regression_cases_include_known_good_guardrail(self):
        regression_path = Path("assets/sample_library/regression_cases.json")
        cases = json.loads(regression_path.read_text(encoding="utf-8"))

        by_resource_id = {item["resource_id"]: item for item in cases}
        self.assertEqual(by_resource_id["1370472"]["quality_label"], "known_good")
        self.assertEqual(by_resource_id["1370472"]["expected_quality"], "keep")
        self.assertEqual(by_resource_id["1370477"]["quality_label"], "known_failure")
        self.assertEqual(by_resource_id["1370347"]["expected_warning"], "dominant_passthrough_segments")
        for item in cases:
            self.assertTrue(Path(item["path"]).exists(), item["path"])


class LabelRecordTests(unittest.TestCase):
    def test_write_label_record_appends_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "labels.jsonl"
            write_label_record(
                target,
                {
                    "sample_id": "s1",
                    "language": "zh",
                    "speaker_pattern": "single_female",
                },
            )

            lines = target.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["sample_id"], "s1")


class SampleLibraryLabelerApiTests(unittest.TestCase):
    def test_labeler_lists_items_and_saves_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "manifest.jsonl"
            labels_path = tmp_path / "labels.jsonl"
            review_candidates_path = tmp_path / "review_candidates.json"
            sample_audio = tmp_path / "sample.mp3"
            sample_audio.write_bytes(b"fake")
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample-1",
                        "path": str(sample_audio),
                        "source": "golden_set",
                        "language": "zh",
                        "duration_bucket": "short",
                        "speaker_pattern": "single_female",
                        "voice_age": "unknown",
                        "music_pattern": "no_music",
                        "content_type": "clean_reading",
                        "expected_processing_mode": "single_voice",
                        "quality_label": "golden",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            review_candidates_path.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "candidate-1",
                            "path": str(sample_audio),
                            "source": "book_res_candidate",
                            "language": "zh",
                            "duration_bucket": "short",
                            "speaker_pattern": "single_female",
                            "voice_age": "unknown",
                            "music_pattern": "no_music",
                            "content_type": "clean_reading",
                            "expected_processing_mode": "single_voice",
                            "quality_label": "candidate",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            app = create_app(
                manifest_path=manifest_path,
                labels_path=labels_path,
                review_candidates_path=review_candidates_path,
                host="127.0.0.1",
                port=7869,
            )
            client = app.test_client()

            response = client.get("/api/items")
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertEqual(len(payload["items"]), 1)
            self.assertEqual(payload["items"][0]["sample_id"], "candidate-1")

            save_response = client.post(
                "/api/labels",
                json={
                    "sample_id": "candidate-1",
                    "labels": {
                        "language": "zh",
                        "speaker_pattern": "single_female",
                        "voice_age": "child",
                        "music_pattern": "no_music",
                    },
                    "item": {"path": str(sample_audio), "source": "book_res_candidate"},
                },
            )
            self.assertEqual(save_response.status_code, 200)
            saved = json.loads(labels_path.read_text(encoding="utf-8").strip())
            self.assertEqual(saved["sample_id"], "candidate-1")
            self.assertEqual(saved["labels"]["voice_age"], "child")
            self.assertNotIn("expected_processing_mode", saved["labels"])
            self.assertEqual(saved["item"]["path"], str(sample_audio))

    def test_load_labeled_sample_ids_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.jsonl"
            labels_path.write_text(
                json.dumps({"sample_id": "a", "labels": {}}, ensure_ascii=False) + "\n"
                + json.dumps({"sample_id": "b", "labels": {}}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(load_labeled_sample_ids(labels_path), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
