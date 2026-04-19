import unittest
from unittest.mock import patch


from tools.device_policy import choose_rvc_force_cpu, choose_uvr_device
from tools.process_mixed_long_audio import (
    analyze_intervals,
    select_auto_uvr_model,
    should_merge_reading_cluster,
    smooth_context_routes,
)


class DevicePolicyTests(unittest.TestCase):
    def test_choose_rvc_force_cpu_defaults_to_gpu_capable(self):
        self.assertFalse(choose_rvc_force_cpu("auto"))
        self.assertFalse(choose_rvc_force_cpu("gpu"))
        self.assertTrue(choose_rvc_force_cpu("cpu"))

    def test_choose_uvr_device_respects_requested_mode(self):
        self.assertEqual(choose_uvr_device("cpu", "mps"), ("cpu", False))
        self.assertEqual(choose_uvr_device("gpu", "mps"), ("mps", False))
        self.assertEqual(choose_uvr_device("auto", "mps"), ("mps", False))
        self.assertEqual(choose_uvr_device("auto", "cpu"), ("cpu", False))

    def test_reading_cluster_merge_rejects_long_internal_silence(self):
        self.assertFalse(
            should_merge_reading_cluster(
                prev_duration_sec=2.0,
                item_duration_sec=2.5,
                gap_sec=2.0,
                merged_duration_sec=6.5,
                timbre_similarity=0.95,
            )
        )

    def test_reading_cluster_merge_keeps_short_pause_word_clusters(self):
        self.assertTrue(
            should_merge_reading_cluster(
                prev_duration_sec=1.2,
                item_duration_sec=1.4,
                gap_sec=0.8,
                merged_duration_sec=3.4,
                timbre_similarity=0.95,
            )
        )

    def test_reading_cluster_merge_rejects_context_merge_on_long_pause(self):
        self.assertFalse(
            should_merge_reading_cluster(
                prev_duration_sec=1.658,
                item_duration_sec=2.853,
                gap_sec=2.766,
                merged_duration_sec=7.254,
                timbre_similarity=1.0,
            )
        )

    def test_context_smoothing_bridges_borderline_run_between_same_route_neighbors(self):
        sr = 44100
        provisional = [
            {"start": 0, "end": int(10 * sr), "route": "female", "classification": "female", "note": "ok", "median_f0": 348.6, "duration_sec": 10.0, "voiced_ratio": 0.39},
            {"start": int(10 * sr), "end": int(24 * sr), "route": "female", "classification": "female", "note": "ok", "median_f0": 234.8, "duration_sec": 14.0, "voiced_ratio": 0.43},
            {"start": int(24 * sr), "end": int(35 * sr), "route": "female", "classification": "female", "note": "ok", "median_f0": 310.8, "duration_sec": 11.0, "voiced_ratio": 0.48},
            {"start": int(35 * sr), "end": int(50 * sr), "route": "male", "classification": "male", "note": "borderline_f0", "median_f0": 176.3, "duration_sec": 15.0, "voiced_ratio": 0.19},
            {"start": int(50 * sr), "end": int(60 * sr), "route": "male", "classification": "male", "note": "borderline_f0", "median_f0": 174.9, "duration_sec": 10.0, "voiced_ratio": 0.28},
            {"start": int(60 * sr), "end": int(72 * sr), "route": "female", "classification": "female", "note": "ok", "median_f0": 195.3, "duration_sec": 12.0, "voiced_ratio": 0.53},
            {"start": int(72 * sr), "end": int(82 * sr), "route": "female", "classification": "female", "note": "ok", "median_f0": 194.5, "duration_sec": 10.0, "voiced_ratio": 0.54},
        ]

        smooth_context_routes(provisional, sr)

        self.assertEqual(provisional[3]["route"], "female")
        self.assertEqual(provisional[4]["route"], "female")
        self.assertTrue(str(provisional[3]["note"]).startswith("context_smoothed_"))
        self.assertTrue(str(provisional[4]["note"]).startswith("context_smoothed_"))

    def test_context_smoothing_keeps_true_route_change(self):
        sr = 44100
        provisional = [
            {"start": 0, "end": int(8 * sr), "route": "male", "classification": "male", "note": "ok", "median_f0": 148.6, "duration_sec": 8.0, "voiced_ratio": 0.72},
            {"start": int(8.2 * sr), "end": int(16 * sr), "route": "female", "classification": "female", "note": "ok", "median_f0": 296.8, "duration_sec": 7.8, "voiced_ratio": 0.57},
        ]

        smooth_context_routes(provisional, sr)

        self.assertEqual(provisional[0]["route"], "male")
        self.assertEqual(provisional[1]["route"], "female")

    def test_analyze_intervals_applies_context_smoothing_to_reclassified_segments(self):
        sr = 44100
        intervals = [
            (int(7.072 * sr), int(17.392 * sr)),
            (int(17.392 * sr), int(32.415 * sr)),
            (int(32.415 * sr), int(42.781 * sr)),
            (int(51.724 * sr), int(66.641 * sr)),
            (int(66.641 * sr), int(76.649 * sr)),
            (int(76.649 * sr), int(88.503 * sr)),
            (int(88.503 * sr), int(98.045 * sr)),
        ]
        classifications = [
            ("female", "female", 348.6, 100, 0.39, "ok"),
            ("female", "female", 234.8, 100, 0.43, "ok"),
            ("female", "female", 310.8, 100, 0.48, "ok"),
            ("male", "male", 176.3, 100, 0.19, "borderline_f0"),
            ("male", "male", 174.9, 100, 0.28, "borderline_f0"),
            ("female", "female", 195.3, 100, 0.53, "ok"),
            ("female", "female", 194.5, 100, 0.54, "ok"),
        ]

        with patch("tools.process_mixed_long_audio.classify_segment", side_effect=classifications):
            analyzed = analyze_intervals([0.0] * int(82 * sr), sr, intervals, object())

        self.assertEqual(analyzed[3]["route"], "female")
        self.assertEqual(analyzed[4]["route"], "female")
        self.assertTrue(str(analyzed[3]["note"]).startswith("context_smoothed_"))
        self.assertTrue(str(analyzed[4]["note"]).startswith("context_smoothed_"))

    def test_select_auto_uvr_model_prefers_hp5_for_long_intro_music_bleed(self):
        metrics = {
            "HP5_only_main_vocal": {
                "coverage": 0.3893,
                "rms": 0.0428,
                "peak": 0.5749,
                "regions": {
                    "intro_0_12": {"vocal_to_instrumental_ratio": 0.8431},
                    "early_7_18": {"vocal_to_instrumental_ratio": 0.5780},
                },
            },
            "HP3_all_vocals": {
                "coverage": 0.7306,
                "rms": 0.0528,
                "peak": 0.5376,
                "regions": {
                    "intro_0_12": {"vocal_to_instrumental_ratio": 1.1861},
                    "early_7_18": {"vocal_to_instrumental_ratio": 1.6505},
                },
            },
        }

        selected_model, reason = select_auto_uvr_model(metrics, source_seconds=105.0)

        self.assertEqual(selected_model, "HP5_only_main_vocal")
        self.assertIn("intro", reason)

    def test_select_auto_uvr_model_keeps_hp3_without_intro_bleed_signature(self):
        metrics = {
            "HP5_only_main_vocal": {
                "coverage": 0.40,
                "rms": 0.05,
                "peak": 0.60,
                "regions": {
                    "intro_0_12": {"vocal_to_instrumental_ratio": 0.82},
                    "early_7_18": {"vocal_to_instrumental_ratio": 1.44},
                },
            },
            "HP3_all_vocals": {
                "coverage": 0.73,
                "rms": 0.05,
                "peak": 0.54,
                "regions": {
                    "intro_0_12": {"vocal_to_instrumental_ratio": 1.20},
                    "early_7_18": {"vocal_to_instrumental_ratio": 0.66},
                },
            },
        }

        selected_model, reason = select_auto_uvr_model(metrics, source_seconds=105.0)

        self.assertEqual(selected_model, "HP3_all_vocals")
        self.assertIn("did not show", reason)


if __name__ == "__main__":
    unittest.main()
