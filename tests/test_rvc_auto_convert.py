import unittest
from pathlib import Path
from unittest.mock import patch
import os
import tempfile

import numpy as np
import soundfile as sf

from tools.rvc_auto_convert import (
    apply_calibrated_music_adjustment,
    apply_pipeline_quality_checks,
    apply_calibrated_voice_route,
    apply_short_clean_voice_override,
    analyze_audio,
    auto_convert_mp3,
    build_manual_sample_calibration_model,
    estimate_music_risk,
    predict_knn_label,
    select_processing_plan,
)


class RvcAutoConvertPlanTests(unittest.TestCase):
    def test_selects_separate_bgm_voice_when_music_risk_is_uncertain(self):
        analysis = {
            "classification": "mixed_with_music",
            "confidence": 0.72,
            "music_risk": 0.35,
            "duration_seconds": 8.0,
            "dominant_route": "female",
            "voiced_frames": 100,
            "voiced_ratio": 0.7,
            "music_reasons": ["strong_harmonic_content"],
        }

        plan = select_processing_plan(analysis, profile="default")

        self.assertEqual(plan["processing_mode"], "separate_bgm_voice")
        self.assertEqual(plan["profile"], "default")
        self.assertEqual(plan["parameters"]["uvr_model"], "auto")

    def test_selects_long_mixed_pipeline_for_long_audio_even_without_music(self):
        analysis = {
            "classification": "female",
            "confidence": 0.86,
            "music_risk": 0.05,
            "duration_seconds": 45.0,
            "dominant_route": "female",
        }

        plan = select_processing_plan(analysis, profile="default")

        self.assertEqual(plan["processing_mode"], "long_mixed_pipeline")

    def test_selects_single_for_short_low_risk_voice(self):
        analysis = {
            "classification": "male",
            "confidence": 0.91,
            "music_risk": 0.03,
            "duration_seconds": 4.2,
            "dominant_route": "male",
        }

        plan = select_processing_plan(analysis, profile="default")

        self.assertEqual(plan["processing_mode"], "single")
        self.assertEqual(plan["models"][0]["role"], "male")

    def test_short_male_voice_overrides_onset_and_percussive_music_risk(self):
        analysis = {
            "classification": "mixed_with_music",
            "confidence": 0.5,
            "music_risk": 0.45,
            "duration_seconds": 1.8296,
            "dominant_route": "male",
            "median_f0_hz": 189.4715,
            "voiced_frames": 116,
            "voiced_ratio": 0.863,
            "music_reasons": ["rhythmic_onsets", "percussive_content"],
        }

        adjusted = apply_short_clean_voice_override(analysis)
        plan = select_processing_plan(adjusted, profile="default")

        self.assertEqual(adjusted["classification"], "male")
        self.assertTrue(adjusted["music_risk_overridden"])
        self.assertEqual(adjusted["override_reason"], "short_clean_voice")
        self.assertEqual(plan["processing_mode"], "single")
        self.assertEqual(plan["models"][0]["role"], "male")

    def test_short_female_voice_overrides_onset_music_risk(self):
        analysis = {
            "classification": "mixed_with_music",
            "confidence": 0.5,
            "music_risk": 0.25,
            "duration_seconds": 2.5,
            "dominant_route": "female",
            "median_f0_hz": 260.0,
            "voiced_frames": 120,
            "voiced_ratio": 0.8,
            "music_reasons": ["rhythmic_onsets"],
        }

        adjusted = apply_short_clean_voice_override(analysis)
        plan = select_processing_plan(adjusted, profile="default")

        self.assertEqual(adjusted["classification"], "female")
        self.assertTrue(adjusted["music_risk_overridden"])
        self.assertEqual(plan["processing_mode"], "single")

    def test_medium_clean_voice_uses_clean_voice_segments_mode(self):
        analysis = {
            "classification": "mixed_with_music",
            "confidence": 0.5,
            "music_risk": 0.45,
            "duration_seconds": 18.3368,
            "dominant_route": "male",
            "median_f0_hz": 160.9122,
            "voiced_frames": 1112,
            "voiced_ratio": 0.6112,
            "music_reasons": ["rhythmic_onsets", "percussive_content"],
        }

        adjusted = apply_short_clean_voice_override(analysis)
        plan = select_processing_plan(adjusted, profile="default")

        self.assertEqual(adjusted["classification"], "male")
        self.assertTrue(adjusted["music_risk_overridden"])
        self.assertEqual(adjusted["override_reason"], "clean_voice_no_music")
        self.assertEqual(plan["processing_mode"], "clean_voice_segments")
        self.assertEqual([item["role"] for item in plan["models"]], ["male", "female"])

    def test_medium_no_music_calibrated_voice_without_override_stays_bgm_separation(self):
        analysis = {
            "classification": "mixed_with_music",
            "confidence": 0.5,
            "music_risk": 0.45,
            "duration_seconds": 18.3368,
            "dominant_route": "male",
            "median_f0_hz": 150.9671,
            "voiced_frames": 833,
            "voiced_ratio": 0.555,
            "music_reasons": ["rhythmic_onsets", "percussive_content"],
            "calibration": {
                "music_prediction": {"label": "no_music", "confidence": 1.0},
            },
        }

        plan = select_processing_plan(analysis, profile="default")

        self.assertEqual(plan["processing_mode"], "separate_bgm_voice")

    def test_low_voiced_ratio_with_strong_music_stays_bgm_separation(self):
        analysis = {
            "classification": "mixed_with_music",
            "confidence": 0.9,
            "music_risk": 0.9,
            "duration_seconds": 9.3007,
            "dominant_route": "female",
            "median_f0_hz": 266.3363,
            "voiced_frames": 371,
            "voiced_ratio": 0.3985,
            "music_reasons": [
                "strong_harmonic_content",
                "rhythmic_onsets",
                "percussive_content",
                "music_like_harmonic_percussive_mix",
            ],
            "calibration": {
                "music_prediction": {"label": "no_music", "confidence": 1.0},
            },
        }

        plan = select_processing_plan(analysis, profile="default")

        self.assertEqual(plan["processing_mode"], "separate_bgm_voice")

    def test_long_pure_speech_with_strong_no_music_calibration_stays_long_mixed(self):
        analysis = {
            "classification": "mixed_with_music",
            "confidence": 0.9,
            "music_risk": 0.9,
            "duration_seconds": 68.0762,
            "dominant_route": "male",
            "median_f0_hz": 88.9519,
            "voiced_frames": 719,
            "voiced_ratio": 0.479,
            "music_reasons": [
                "long_audio",
                "strong_harmonic_content",
                "rhythmic_onsets",
                "percussive_content",
            ],
            "calibration": {
                "music_prediction": {"label": "no_music", "confidence": 1.0},
            },
        }

        plan = select_processing_plan(analysis, profile="default")

        self.assertEqual(plan["processing_mode"], "long_mixed_pipeline")

    def test_short_voice_does_not_override_harmonic_music_risk(self):
        analysis = {
            "classification": "mixed_with_music",
            "confidence": 0.5,
            "music_risk": 0.5,
            "duration_seconds": 3.0,
            "dominant_route": "male",
            "voiced_frames": 140,
            "voiced_ratio": 0.85,
            "music_reasons": ["strong_harmonic_content"],
        }

        adjusted = apply_short_clean_voice_override(analysis)
        plan = select_processing_plan(adjusted, profile="default")

        self.assertFalse(adjusted["music_risk_overridden"])
        self.assertEqual(plan["processing_mode"], "separate_bgm_voice")

    def test_long_voice_does_not_override_music_risk(self):
        analysis = {
            "classification": "mixed_with_music",
            "confidence": 0.5,
            "music_risk": 0.45,
            "duration_seconds": 45.0,
            "dominant_route": "male",
            "voiced_frames": 1000,
            "voiced_ratio": 0.9,
            "music_reasons": ["rhythmic_onsets", "percussive_content"],
        }

        adjusted = apply_short_clean_voice_override(analysis)
        plan = select_processing_plan(adjusted, profile="default")

        self.assertFalse(adjusted["music_risk_overridden"])
        self.assertEqual(plan["processing_mode"], "long_mixed_pipeline")

    def test_music_risk_estimation_is_compatible_with_installed_librosa(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        y = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

        risk, reasons = estimate_music_risk(y, sr, 1.0)

        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)
        self.assertIsInstance(reasons, list)

    def test_pipeline_quality_keeps_known_good_long_mixed_case(self):
        quality_gate = {"passed": True, "fallback_used": False, "warnings": []}
        plan = {"processing_mode": "long_mixed_pipeline"}
        analysis = {
            "music_risk": 1.0,
            "music_reasons": [
                "long_audio",
                "strong_harmonic_content",
                "rhythmic_onsets",
                "percussive_content",
            ],
        }
        segments = [
            {"start": 8.871, "end": 16.694, "route": "male", "voiced_ratio": 0.7241, "note": "ok"},
            {"start": 16.987, "end": 24.484, "route": "female", "voiced_ratio": 0.5733, "note": "ok"},
        ]

        adjusted = apply_pipeline_quality_checks(quality_gate, analysis, plan, segments, {})

        self.assertTrue(adjusted["passed"])
        self.assertFalse(adjusted["fallback_used"])
        self.assertEqual(adjusted["warnings"], [])

    def test_pipeline_quality_fails_dominant_passthrough_long_mixed_case(self):
        quality_gate = {"passed": True, "fallback_used": False, "warnings": []}
        plan = {"processing_mode": "long_mixed_pipeline"}
        analysis = {"music_risk": 0.85, "music_reasons": ["long_audio", "percussive_content"]}
        segments = [
            {"start": 0.0, "end": 15.232, "route": "passthrough", "voiced_ratio": 0.0958, "note": "low_voice"},
            {"start": 15.232, "end": 26.494, "route": "male", "voiced_ratio": 0.1517, "note": "ok"},
            {"start": 26.494, "end": 43.831, "route": "passthrough", "voiced_ratio": 0.0444, "note": "low_voice"},
        ]

        adjusted = apply_pipeline_quality_checks(quality_gate, analysis, plan, segments, {})

        self.assertFalse(adjusted["passed"])
        self.assertTrue(adjusted["fallback_used"])
        self.assertEqual(adjusted["fallback_reason"], "dominant_passthrough_segments")
        self.assertIn("dominant_passthrough_segments", adjusted["warnings"])

    def test_pipeline_quality_fails_low_voiced_long_speech_segments(self):
        quality_gate = {"passed": True, "fallback_used": False, "warnings": []}
        plan = {"processing_mode": "long_mixed_pipeline"}
        analysis = {"music_risk": 0.85, "music_reasons": ["long_audio", "percussive_content"]}
        segments = [
            {"start": 0.0, "end": 18.599, "route": "male", "voiced_ratio": 0.2505, "note": "ok"},
            {"start": 18.599, "end": 35.178, "route": "male", "voiced_ratio": 0.31, "note": "ok"},
            {"start": 35.178, "end": 54.879, "route": "male", "voiced_ratio": 0.1877, "note": "ok"},
        ]

        adjusted = apply_pipeline_quality_checks(quality_gate, analysis, plan, segments, {})

        self.assertFalse(adjusted["passed"])
        self.assertTrue(adjusted["fallback_used"])
        self.assertEqual(adjusted["fallback_reason"], "low_voiced_long_speech_segments")
        self.assertIn("low_voiced_long_speech_segments", adjusted["warnings"])

    def test_pipeline_quality_fails_clean_voice_on_strong_music(self):
        quality_gate = {"passed": True, "fallback_used": False, "warnings": []}
        plan = {"processing_mode": "clean_voice_segments"}
        analysis = {
            "music_risk": 0.9,
            "music_reasons": [
                "strong_harmonic_content",
                "rhythmic_onsets",
                "percussive_content",
                "music_like_harmonic_percussive_mix",
            ],
        }
        segments = [
            {"start": 0.0, "end": 11.088, "route": "female", "voiced_ratio": 0.5735, "note": "ok"},
            {"start": 11.088, "end": 22.597, "route": "female", "voiced_ratio": 0.7307, "note": "ok"},
        ]

        adjusted = apply_pipeline_quality_checks(quality_gate, analysis, plan, segments, {})

        self.assertFalse(adjusted["passed"])
        self.assertTrue(adjusted["fallback_used"])
        self.assertEqual(adjusted["fallback_reason"], "clean_voice_segments_on_strong_music")

    def test_pipeline_quality_accepts_context_smoothed_segments_and_short_residual(self):
        quality_gate = {"passed": True, "fallback_used": False, "warnings": []}
        plan = {"processing_mode": "long_mixed_pipeline"}
        analysis = {
            "music_risk": 0.82,
            "music_reasons": [
                "long_audio",
                "strong_harmonic_content",
                "rhythmic_onsets",
                "percussive_content",
            ],
        }
        segments = [
            {"start": 0.0, "end": 10.0, "route": "female", "voiced_ratio": 0.39, "note": "ok"},
            {"start": 10.0, "end": 24.0, "route": "female", "voiced_ratio": 0.43, "note": "ok"},
            {
                "start": 24.0,
                "end": 39.0,
                "route": "female",
                "voiced_ratio": 0.19,
                "note": "context_smoothed_female",
            },
            {
                "start": 39.0,
                "end": 49.0,
                "route": "female",
                "voiced_ratio": 0.28,
                "note": "context_smoothed_female",
            },
            {"start": 49.0, "end": 61.0, "route": "female", "voiced_ratio": 0.53, "note": "ok"},
            {"start": 61.0, "end": 66.4, "route": "passthrough", "voiced_ratio": 0.24, "note": "music_residual"},
            {"start": 66.4, "end": 76.4, "route": "female", "voiced_ratio": 0.54, "note": "ok"},
        ]

        adjusted = apply_pipeline_quality_checks(quality_gate, analysis, plan, segments, {})

        self.assertTrue(adjusted["passed"])
        self.assertFalse(adjusted["fallback_used"])
        self.assertEqual(adjusted["warnings"], [])

    def test_analyze_audio_initializes_required_environment(self):
        os.environ.pop("rmvpe_root", None)
        with (
            patch("tools.rvc_auto_convert.audio_duration_seconds", return_value=1.0),
            patch("tools.rvc_auto_convert.librosa.load", return_value=(np.zeros(16000), 16000)),
            patch("tools.rvc_auto_convert.estimate_music_risk", return_value=(0.0, [])),
            patch(
                "tools.rvc_auto_convert.classify_voice",
                return_value=("male", 0.9, 120.0, 100, 0.8),
            ),
        ):
            analyze_audio(Path("/tmp/fake.mp3"))

        self.assertEqual(os.environ["rmvpe_root"], "assets/rmvpe")

    def test_auto_convert_dispatches_clean_voice_segments_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.wav"
            output_path = tmp_path / "output.mp3"
            sf.write(input_path, np.zeros(1600, dtype=np.float32), 16000)
            analysis = {
                "classification": "male",
                "confidence": 0.5,
                "music_risk": 0.45,
                "duration_seconds": 10.0,
                "dominant_route": "male",
                "music_risk_overridden": True,
                "override_reason": "clean_voice_no_music",
            }
            plan = {
                "processing_mode": "clean_voice_segments",
                "models": [{"role": "male"}, {"role": "female"}],
                "parameters": {},
            }

            def fake_convert(input_arg, output_arg, job_dir, plan_arg, progress_callback):
                output_arg.write_bytes(b"mp3")
                return ([{"route": "male"}], ["clean"])

            with (
                patch("tools.rvc_auto_convert.analyze_audio", return_value=analysis),
                patch("tools.rvc_auto_convert.select_processing_plan", return_value=plan),
                patch(
                    "tools.rvc_auto_convert.convert_clean_voice_segments",
                    side_effect=fake_convert,
                ) as clean_convert,
                patch("tools.rvc_auto_convert.convert_single") as single_convert,
                patch("tools.rvc_auto_convert.convert_safe_long") as safe_long_convert,
                patch(
                    "tools.rvc_auto_convert.evaluate_quality_gate",
                    return_value={
                        "passed": True,
                        "fallback_used": False,
                        "fallback_reason": None,
                        "warnings": [],
                    },
                ),
            ):
                result = auto_convert_mp3(input_path, output_path)

            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["selected_plan"]["processing_mode"], "clean_voice_segments")
            self.assertIn("review", result)
            self.assertIn("quality_gate", result)
            clean_convert.assert_called_once()
            single_convert.assert_not_called()
            safe_long_convert.assert_not_called()

    def test_auto_convert_dispatches_separate_bgm_voice_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.wav"
            output_path = tmp_path / "output.mp3"
            sf.write(input_path, np.zeros(1600, dtype=np.float32), 16000)
            analysis = {
                "classification": "mixed_with_music",
                "confidence": 0.5,
                "music_risk": 0.45,
                "duration_seconds": 12.0,
                "dominant_route": "male",
            }
            plan = {
                "processing_mode": "separate_bgm_voice",
                "models": [{"role": "male"}, {"role": "female"}],
                "parameters": {},
            }

            def fake_convert(input_arg, output_arg, job_dir, plan_arg, progress_callback):
                output_arg.write_bytes(b"mp3")
                return ([{"route": "male"}], ["bgm"])

            with (
                patch("tools.rvc_auto_convert.analyze_audio", return_value=analysis),
                patch("tools.rvc_auto_convert.select_processing_plan", return_value=plan),
                patch(
                    "tools.rvc_auto_convert.convert_separate_bgm_voice",
                    side_effect=fake_convert,
                ) as bgm_convert,
                patch("tools.rvc_auto_convert.convert_single") as single_convert,
                patch("tools.rvc_auto_convert.convert_clean_voice_segments") as clean_convert,
                patch("tools.rvc_auto_convert.convert_long_mixed_pipeline") as long_convert,
                patch(
                    "tools.rvc_auto_convert.evaluate_quality_gate",
                    return_value={
                        "passed": True,
                        "fallback_used": False,
                        "fallback_reason": None,
                        "warnings": [],
                    },
                ),
            ):
                result = auto_convert_mp3(input_path, output_path)

            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["selected_plan"]["processing_mode"], "separate_bgm_voice")
            bgm_convert.assert_called_once()
            single_convert.assert_not_called()
            clean_convert.assert_not_called()
            long_convert.assert_not_called()

    def test_auto_convert_dispatches_long_mixed_pipeline_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.wav"
            output_path = tmp_path / "output.mp3"
            sf.write(input_path, np.zeros(1600, dtype=np.float32), 16000)
            analysis = {
                "classification": "mixed_with_music",
                "confidence": 0.5,
                "music_risk": 0.45,
                "duration_seconds": 45.0,
                "dominant_route": "male",
            }
            plan = {
                "processing_mode": "long_mixed_pipeline",
                "models": [{"role": "male"}, {"role": "female"}],
                "parameters": {},
            }

            def fake_convert(input_arg, output_arg, job_dir, plan_arg, progress_callback):
                output_arg.write_bytes(b"mp3")
                return ([{"route": "male"}], ["long"])

            with (
                patch("tools.rvc_auto_convert.analyze_audio", return_value=analysis),
                patch("tools.rvc_auto_convert.select_processing_plan", return_value=plan),
                patch(
                    "tools.rvc_auto_convert.convert_long_mixed_pipeline",
                    side_effect=fake_convert,
                ) as long_convert,
                patch("tools.rvc_auto_convert.convert_single") as single_convert,
                patch("tools.rvc_auto_convert.convert_clean_voice_segments") as clean_convert,
                patch("tools.rvc_auto_convert.convert_separate_bgm_voice") as bgm_convert,
                patch(
                    "tools.rvc_auto_convert.evaluate_quality_gate",
                    return_value={
                        "passed": True,
                        "fallback_used": False,
                        "fallback_reason": None,
                        "warnings": [],
                    },
                ),
            ):
                result = auto_convert_mp3(input_path, output_path)

            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["selected_plan"]["processing_mode"], "long_mixed_pipeline")
            long_convert.assert_called_once()
            single_convert.assert_not_called()
            clean_convert.assert_not_called()
            bgm_convert.assert_not_called()


class RvcAutoConvertCalibrationTests(unittest.TestCase):
    def test_predict_knn_label_prefers_nearest_class(self):
        calibration = {
            "mean": [0.0, 0.0],
            "std": [1.0, 1.0],
            "label_counts": {"male": 2, "female": 2},
            "samples": [
                {"label": "male", "vector": [-2.0, -1.8]},
                {"label": "male", "vector": [-1.7, -2.1]},
                {"label": "female", "vector": [2.0, 1.8]},
                {"label": "female", "vector": [1.9, 2.2]},
            ],
        }

        label, confidence, scores = predict_knn_label(
            np.array([2.1, 2.0], dtype=np.float32),
            calibration,
            k=3,
        )

        self.assertEqual(label, "female")
        self.assertGreater(confidence, 0.7)
        self.assertGreater(scores["female"], scores["male"])

    def test_predict_knn_label_rebalances_imbalanced_classes(self):
        calibration = {
            "mean": [0.0, 0.0],
            "std": [1.0, 1.0],
            "label_counts": {"no_music": 20, "music": 2},
            "samples": [
                {"label": "no_music", "vector": [0.1, 0.1]},
                {"label": "no_music", "vector": [0.12, 0.12]},
                {"label": "no_music", "vector": [0.15, 0.15]},
                {"label": "music", "vector": [0.11, 0.11]},
                {"label": "music", "vector": [0.13, 0.13]},
            ],
        }

        label, confidence, _scores = predict_knn_label(
            np.array([0.115, 0.115], dtype=np.float32),
            calibration,
            k=5,
        )

        self.assertEqual(label, "music")
        self.assertGreater(confidence, 0.5)

    def test_apply_calibrated_music_adjustment_suppresses_speech_like_music_risk(self):
        adjusted_risk, adjusted_reasons, calibration = apply_calibrated_music_adjustment(
            base_risk=0.45,
            base_reasons=["rhythmic_onsets", "percussive_content"],
            feature_bundle={"non_voiced_ratio": 0.01, "voiced_ratio": 0.82, "duration_seconds": 0.9},
            prediction={"label": "no_music", "confidence": 0.9},
        )

        self.assertLess(adjusted_risk, 0.25)
        self.assertEqual(adjusted_reasons, ["speech_dominant_calibrated_no_music"])
        self.assertTrue(calibration["suppressed"])

    def test_apply_calibrated_music_adjustment_keeps_strong_music_reasons(self):
        adjusted_risk, adjusted_reasons, calibration = apply_calibrated_music_adjustment(
            base_risk=0.9,
            base_reasons=["strong_harmonic_content", "rhythmic_onsets", "percussive_content"],
            feature_bundle={"non_voiced_ratio": 0.01, "voiced_ratio": 0.82, "duration_seconds": 0.8},
            prediction={"label": "no_music", "confidence": 1.0},
        )

        self.assertEqual(adjusted_risk, 0.9)
        self.assertEqual(
            adjusted_reasons,
            ["strong_harmonic_content", "rhythmic_onsets", "percussive_content"],
        )
        self.assertFalse(calibration["suppressed"])

    def test_apply_calibrated_music_adjustment_does_not_suppress_medium_audio(self):
        adjusted_risk, adjusted_reasons, calibration = apply_calibrated_music_adjustment(
            base_risk=0.45,
            base_reasons=["rhythmic_onsets", "percussive_content"],
            feature_bundle={"non_voiced_ratio": 0.01, "voiced_ratio": 0.82, "duration_seconds": 2.0},
            prediction={"label": "no_music", "confidence": 1.0},
        )

        self.assertEqual(adjusted_risk, 0.45)
        self.assertEqual(adjusted_reasons, ["rhythmic_onsets", "percussive_content"])
        self.assertFalse(calibration["suppressed"])

    def test_apply_calibrated_voice_route_can_override_borderline_pitch(self):
        route, confidence = apply_calibrated_voice_route(
            base_route="male",
            base_confidence=0.58,
            feature_bundle={"rmvpe_p50": 176.0, "rmvpe_mean": 196.0},
            prediction={"label": "female", "confidence": 0.86},
        )

        self.assertEqual(route, "female")
        self.assertGreater(confidence, 0.7)

    def test_manual_sample_library_builds_gender_and_music_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            male = root / "male.mp3"
            female = root / "female.mp3"
            song = root / "song.mp3"
            speech = root / "speech.mp3"
            for path in [male, female, song, speech]:
                path.write_bytes(b"audio")
            labels_path = root / "labels.jsonl"
            labels_path.write_text(
                "\n".join(
                    [
                        '{"sample_id":"male","labels":{"speaker_pattern":"single_male","music_pattern":"no_music"},"item":{"path":"%s"}}'
                        % male,
                        '{"sample_id":"female","labels":{"speaker_pattern":"single_female","music_pattern":"no_music"},"item":{"path":"%s"}}'
                        % female,
                        '{"sample_id":"song","labels":{"speaker_pattern":"male_female_mixed","music_pattern":"song"},"item":{"path":"%s"}}'
                        % song,
                        '{"sample_id":"speech","labels":{"speaker_pattern":"single_female","music_pattern":"transient_sfx"},"item":{"path":"%s"}}'
                        % speech,
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            def fake_features(_y, _sr, duration):
                return {
                    "duration_seconds": duration,
                    "voiced_ratio": 0.7,
                    "rmvpe_p25": 120.0,
                    "rmvpe_p50": 160.0,
                    "rmvpe_p75": 220.0,
                    "rmvpe_mean": 170.0,
                    "yin_p50": 160.0,
                    "yin_mean": 170.0,
                    "centroid_mean": 1200.0,
                    "rolloff_mean": 2500.0,
                    "bandwidth_mean": 900.0,
                    "mfcc1_mean": 1.0,
                    "mfcc2_mean": 2.0,
                    "mfcc3_mean": 3.0,
                    "non_voiced_ratio": 0.02,
                    "flatness_mean": 0.1,
                    "onset_mean": 0.2,
                    "tempo": 80.0,
                    "harmonic_ratio": 0.4,
                    "percussive_ratio": 0.1,
                    "rms_mean": 0.05,
                }

            with (
                patch("tools.rvc_auto_convert.SAMPLE_LIBRARY_LABELS_PATH", labels_path),
                patch("tools.rvc_auto_convert.SAMPLE_LIBRARY_CANDIDATE_PATHS", (root / "missing.json",)),
                patch("tools.rvc_auto_convert.audio_duration_seconds", return_value=1.0),
                patch("tools.rvc_auto_convert.librosa.load", return_value=(np.zeros(16000), 16000)),
                patch("tools.rvc_auto_convert.extract_audio_feature_bundle", side_effect=fake_features),
            ):
                model, signature = build_manual_sample_calibration_model()

        self.assertIsNotNone(model)
        self.assertIsNotNone(signature)
        self.assertEqual(model["source"], "manual_sample_library")
        self.assertEqual({item["label"] for item in model["gender"]["samples"]}, {"male", "female"})
        self.assertEqual({item["label"] for item in model["music"]["samples"]}, {"music", "no_music"})

    def test_manual_sample_library_prefers_cached_analysis_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            male = root / "male.mp3"
            female = root / "female.mp3"
            song = root / "song.mp3"
            for path in [male, female, song]:
                path.write_bytes(b"audio")
            labels_path = root / "labels.jsonl"
            labels_path.write_text(
                "\n".join(
                    [
                        '{"sample_id":"male","labels":{"speaker_pattern":"single_male","music_pattern":"no_music"},"item":{"path":"%s"}}'
                        % male,
                        '{"sample_id":"female","labels":{"speaker_pattern":"single_female","music_pattern":"no_music"},"item":{"path":"%s"}}'
                        % female,
                        '{"sample_id":"song","labels":{"speaker_pattern":"male_female_mixed","music_pattern":"song"},"item":{"path":"%s"}}'
                        % song,
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            def fake_row(sample_id, _item):
                f0 = {"male": 120.0, "female": 240.0}.get(sample_id, 180.0)
                return {
                    "duration_seconds": 2.0,
                    "voiced_coverage": 0.8,
                    "segment_count": 1,
                    "short_segment_ratio": 0.0,
                    "silence_ratio": 0.1,
                    "spectral_flatness": 0.02,
                    "onset_strength": 0.2,
                    "harmonic_ratio": 0.5,
                    "percussive_ratio": 0.05,
                    "rms": 0.05,
                    "f0_median_hz": f0,
                    "f0_p10_hz": f0 - 20.0,
                    "f0_p90_hz": f0 + 20.0,
                    "f0_voiced_frames": 80,
                    "f0_voiced_ratio": 0.8,
                }

            with (
                patch("tools.rvc_auto_convert.SAMPLE_LIBRARY_LABELS_PATH", labels_path),
                patch("tools.rvc_auto_convert.SAMPLE_LIBRARY_CANDIDATE_PATHS", (root / "missing.json",)),
                patch("tools.rvc_auto_convert._analysis_row_for_item", side_effect=fake_row),
                patch("tools.rvc_auto_convert.extract_audio_feature_bundle") as extract_features,
            ):
                model, signature = build_manual_sample_calibration_model()

        self.assertIsNotNone(model)
        self.assertIsNotNone(signature)
        self.assertEqual(model["gender"]["feature_set"], "manual_sample_library_v1")
        self.assertEqual(model["music"]["feature_set"], "manual_sample_library_v1")
        self.assertEqual(model["stats"]["sqlite_feature_samples"], 3)
        extract_features.assert_not_called()


if __name__ == "__main__":
    unittest.main()
