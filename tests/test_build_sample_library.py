import json
import tempfile
import unittest
from pathlib import Path

from tools.build_sample_library import (
    build_review_candidates,
    parse_voice_conversion_review_cases,
)


class VoiceConversionReviewCaseTests(unittest.TestCase):
    def test_parse_voice_conversion_review_cases_extracts_reviewable_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_audio = root / "track_audio" / "a.mp3"
            source_audio.parent.mkdir(parents=True, exist_ok=True)
            source_audio.write_bytes(b"audio")
            output_audio = root / "track_audio_rvc" / "a.mp3"
            output_audio.parent.mkdir(parents=True, exist_ok=True)
            output_audio.write_bytes(b"audio")
            payload = {
                "conversion": {
                    "files": [
                        {
                            "resource_id": "r1",
                            "relative_path": "book_x/track_audio/a.mp3",
                            "output_relative_path": "track_audio_rvc/a.mp3",
                            "file_name": "a.mp3",
                            "output_file_name": "a.mp3",
                            "classification": "mixed_with_music",
                            "processing_mode": "long_mixed_pipeline",
                            "review": {"needs_review": True, "reasons": ["uncertain_segments"]},
                        }
                    ]
                }
            }
            metadata_path = root / "voice_conversion.json"
            metadata_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            rows = parse_voice_conversion_review_cases(metadata_path, book_root=root, book_id="book_x")

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].source, "book_res_error_case")
            self.assertEqual(rows[0].quality_label, "known_failure")


class ReviewSubsetBuilderTests(unittest.TestCase):
    def test_build_review_candidates_limits_output(self):
        entries = []
        for idx in range(12):
            entries.append(
                {
                    "sample_id": f"id-{idx}",
                    "path": f"/tmp/{idx}.mp3",
                    "source": "book_res_candidate",
                    "language": "zh" if idx % 2 == 0 else "en",
                    "duration_bucket": "short",
                    "speaker_pattern": "single_male",
                    "music_pattern": "no_music",
                    "content_type": "clean_reading",
                    "expected_processing_mode": "single_voice",
                    "quality_label": "candidate",
                    "book_id": "book_x",
                }
            )

        result = build_review_candidates(entries, max_items=5)

        self.assertEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()
