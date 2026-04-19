import unittest

from tools.analyze_sample_labels import (
    ROUND2_DURATION_QUOTAS,
    ROUND2_RANDOM_SEED,
    Round2Candidate,
    build_diff_summary,
    infer_round2_labels,
    is_likely_repetitive_word,
    reorder_candidates_for_labeling,
    sample_family_key,
)


class LabelDiffSummaryTests(unittest.TestCase):
    def test_build_diff_summary_counts_key_mismatches(self):
        review_candidates = {
            "s1": {
                "sample_id": "s1",
                "path": "/tmp/a.mp3",
                "language": "en",
                "speaker_pattern": "male_female_mixed",
                "music_pattern": "bgm",
            },
            "s2": {
                "sample_id": "s2",
                "path": "/tmp/b.mp3",
                "language": "en",
                "speaker_pattern": "single_female",
                "music_pattern": "bgm",
            },
        }
        latest_labels = {
            "s1": {
                "language": "en",
                "speaker_pattern": "single_male",
                "voice_age": "adult",
                "music_pattern": "no_music",
            },
            "s2": {
                "language": "en",
                "speaker_pattern": "single_female",
                "voice_age": "adult",
                "music_pattern": "song",
            },
        }

        report = build_diff_summary(review_candidates, latest_labels)

        self.assertEqual(report["labeled_sample_count"], 2)
        self.assertEqual(report["field_diff_counts"]["speaker_pattern"], 1)
        self.assertEqual(report["field_diff_counts"]["music_pattern"], 2)
        self.assertEqual(report["actual_song_count"], 1)


class InferRound2LabelsTests(unittest.TestCase):
    def test_infer_round2_labels_detects_song_from_rel_path(self):
        row = {
            "book_id": "tape3a_002013",
            "rel_path": "tape3a_002013/track_audio/chant_song_001.mp3",
            "audio_type": "music_or_bgm_candidate",
            "gender_guess": "female",
            "voiced_coverage": 0.8,
            "transient_event_count": 0,
            "duration_seconds": 4.2,
        }

        language, speaker_pattern, music_pattern, voice_age, expected_processing_mode = infer_round2_labels(row)

        self.assertEqual(language, "en")
        self.assertEqual(speaker_pattern, "single_female")
        self.assertEqual(music_pattern, "song")
        self.assertEqual(voice_age, "adult")
        self.assertEqual(expected_processing_mode, "separate_bgm_voice")

    def test_family_key_groups_same_page_variants(self):
        self.assertEqual(
            sample_family_key("tape3a_002013/track_audio/page_301964_aaa.mp3"),
            "page_301964",
        )

    def test_likely_repetitive_word_flags_short_clean_single_voice(self):
        row = {"duration_seconds": 1.2}

        self.assertTrue(is_likely_repetitive_word(row, "single_male", "no_music"))
        self.assertTrue(is_likely_repetitive_word(row, "male_female_mixed", "no_music"))
        self.assertFalse(is_likely_repetitive_word(row, "single_male", "bgm"))

    def test_round2_duration_quotas_cover_target_size(self):
        self.assertEqual(sum(ROUND2_DURATION_QUOTAS.values()), 48)
        self.assertIsInstance(ROUND2_RANDOM_SEED, int)

    def test_reorder_candidates_interleaves_duration_buckets(self):
        candidates = []
        for idx, duration_bucket in enumerate(["1-3s", "1-3s", "1-3s", ">180s", "25-60s"]):
            candidates.append(
                Round2Candidate(
                    sample_id=f"id-{idx}",
                    path=f"/tmp/{idx}.mp3",
                    source="book_res_candidate_round2",
                    language="zh" if idx % 2 else "en",
                    duration_bucket=duration_bucket,
                    speaker_pattern="male_female_mixed",
                    music_pattern="no_music",
                    voice_age="adult",
                    content_type="clean_reading",
                    expected_processing_mode="clean_voice_segments",
                    quality_label="candidate_round2",
                    book_id="book",
                    notes=[],
                    duration_seconds=1.0,
                )
            )

        ordered = reorder_candidates_for_labeling(candidates)

        self.assertNotEqual([item.duration_bucket for item in ordered[:3]], ["1-3s", "1-3s", "1-3s"])


if __name__ == "__main__":
    unittest.main()
