import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.rvc_pipeline.classifier import apply_analysis_overrides, build_analysis, select_processing_plan
from tools.rvc_pipeline.metadata import append_review_record, build_result_payload
from tools.rvc_pipeline.quality_gate import evaluate_quality_gate
from tools.rvc_pipeline.segmenter import summarize_segment_review
from tools.rvc_pipeline.sfx_detector import summarize_sfx_segments
from tools.rvc_pipeline.thresholds import DEFAULT_THRESHOLDS, build_dataset_thresholds


class ThresholdTests(unittest.TestCase):
    def test_build_dataset_thresholds_uses_summary_distribution_with_guardrails(self):
        summary = {
            "duration_seconds": {
                "p25": 1.15,
                "p50": 2.509,
                "p05": 0.706,
                "p75": 5.096,
                "p90": 17.426,
            }
        }

        thresholds = build_dataset_thresholds(summary)

        self.assertAlmostEqual(thresholds.very_short_risk_seconds, 0.7766, places=3)
        self.assertAlmostEqual(thresholds.ultra_short_single_seconds, 1.2075, places=3)
        self.assertAlmostEqual(thresholds.short_single_voice_seconds, 3.0108, places=3)
        self.assertAlmostEqual(thresholds.short_voice_upper_seconds, 7.1344, places=3)
        self.assertAlmostEqual(thresholds.long_audio_min_seconds, 20.9112, places=3)


class ClassifierTests(unittest.TestCase):
    def test_apply_analysis_overrides_marks_short_clean_voice(self):
        analysis = {
            "classification": "mixed_with_music",
            "confidence": 0.5,
            "music_risk": 0.45,
            "duration_seconds": 1.83,
            "dominant_route": "male",
            "median_f0_hz": 189.0,
            "voiced_frames": 120,
            "voiced_ratio": 0.86,
            "music_reasons": ["rhythmic_onsets", "percussive_content"],
        }

        adjusted = apply_analysis_overrides(analysis, DEFAULT_THRESHOLDS)

        self.assertEqual(adjusted["classification"], "male")
        self.assertTrue(adjusted["music_risk_overridden"])
        self.assertEqual(adjusted["override_reason"], "short_clean_voice")

    def test_select_processing_plan_returns_clean_voice_segments_for_medium_clean_voice(self):
        analysis = {
            "classification": "male",
            "confidence": 0.9,
            "music_risk": 0.45,
            "duration_seconds": 18.0,
            "dominant_route": "male",
            "music_risk_overridden": True,
            "override_reason": "clean_voice_no_music",
        }
        profile = {
            "male_model": "man.pth",
            "female_model": "woman.pth",
            "uvr_model": "auto",
            "reading_mode": True,
            "speaker_embedding": False,
            "male_params": {"f0_up_key": 0},
            "female_params": {"f0_up_key": 0},
        }

        plan = select_processing_plan(analysis, profile, lambda model: f"/tmp/{Path(model).stem}.index", DEFAULT_THRESHOLDS)

        self.assertEqual(plan["processing_mode"], "clean_voice_segments")
        self.assertEqual([item["role"] for item in plan["models"]], ["male", "female"])

    def test_select_processing_plan_prefers_single_for_ultra_short_single_voice(self):
        analysis = {
            "classification": "male",
            "confidence": 0.98,
            "music_risk": 0.18,
            "duration_seconds": 0.8892,
            "dominant_route": "male",
            "voiced_ratio": 0.8202,
            "music_risk_overridden": False,
            "calibration": {
                "music_prediction": {"label": "no_music", "confidence": 0.8156},
            },
        }
        profile = {
            "male_model": "man.pth",
            "female_model": "woman.pth",
            "uvr_model": "auto",
            "reading_mode": True,
            "speaker_embedding": False,
            "male_params": {"f0_up_key": 0},
            "female_params": {"f0_up_key": 0},
        }

        plan = select_processing_plan(analysis, profile, lambda model: f"/tmp/{Path(model).stem}.index", DEFAULT_THRESHOLDS)

        self.assertEqual(plan["processing_mode"], "single")
        self.assertEqual(plan["target_route"], "male")

    def test_select_processing_plan_prefers_single_for_short_strong_no_music_voice(self):
        analysis = {
            "classification": "female",
            "confidence": 0.98,
            "music_risk": 0.12,
            "duration_seconds": 1.908,
            "dominant_route": "female",
            "voiced_ratio": 0.6335,
            "music_risk_overridden": False,
            "calibration": {
                "music_prediction": {"label": "no_music", "confidence": 1.0},
            },
        }
        profile = {
            "male_model": "man.pth",
            "female_model": "woman.pth",
            "uvr_model": "auto",
            "reading_mode": True,
            "speaker_embedding": False,
            "male_params": {"f0_up_key": 0},
            "female_params": {"f0_up_key": 0},
        }

        plan = select_processing_plan(analysis, profile, lambda model: f"/tmp/{Path(model).stem}.index", DEFAULT_THRESHOLDS)

        self.assertEqual(plan["processing_mode"], "single")
        self.assertEqual(plan["target_route"], "female")

    def test_select_processing_plan_keeps_bgm_separation_for_short_mixed_voice_without_override(self):
        analysis = {
            "classification": "mixed_with_music",
            "confidence": 0.5,
            "music_risk": 0.45,
            "duration_seconds": 1.3856,
            "dominant_route": "male",
            "voiced_ratio": 0.5036,
            "music_risk_overridden": False,
            "music_reasons": ["rhythmic_onsets", "percussive_content"],
            "calibration": {
                "music_prediction": {"label": "no_music", "confidence": 1.0},
            },
        }
        profile = {
            "male_model": "man.pth",
            "female_model": "woman.pth",
            "uvr_model": "auto",
            "reading_mode": True,
            "speaker_embedding": False,
            "male_params": {"f0_up_key": 0},
            "female_params": {"f0_up_key": 0},
        }

        plan = select_processing_plan(analysis, profile, lambda model: f"/tmp/{Path(model).stem}.index", DEFAULT_THRESHOLDS)

        self.assertEqual(plan["processing_mode"], "separate_bgm_voice")
        self.assertEqual(plan["target_route"], "male")

    def test_select_processing_plan_does_not_use_clean_voice_for_long_mixed_with_no_music_calibration(self):
        analysis = {
            "classification": "mixed_with_music",
            "confidence": 0.65,
            "music_risk": 0.65,
            "duration_seconds": 93.4,
            "dominant_route": "female",
            "voiced_ratio": 0.42,
            "music_risk_overridden": False,
            "calibration": {
                "music_prediction": {"label": "no_music", "confidence": 1.0},
            },
        }
        profile = {
            "male_model": "man.pth",
            "female_model": "woman.pth",
            "uvr_model": "auto",
            "reading_mode": True,
            "speaker_embedding": False,
            "male_params": {"f0_up_key": 0},
            "female_params": {"f0_up_key": 0},
        }

        plan = select_processing_plan(analysis, profile, lambda model: f"/tmp/{Path(model).stem}.index", DEFAULT_THRESHOLDS)

        self.assertEqual(plan["processing_mode"], "long_mixed_pipeline")

    def test_build_analysis_applies_override_flow(self):
        analysis = build_analysis(
            duration=2.0,
            music_risk=0.3,
            music_reasons=["rhythmic_onsets"],
            dominant_route="female",
            voice_confidence=0.82,
            median_f0=240.0,
            voiced_frames=140,
            voiced_ratio=0.75,
            thresholds=DEFAULT_THRESHOLDS,
        )

        self.assertEqual(analysis["classification"], "female")
        self.assertTrue(analysis["music_risk_overridden"])


class QualityGateTests(unittest.TestCase):
    def test_quality_gate_passes_reasonable_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.wav"
            output_path = tmp_path / "output.wav"
            audio = 0.1 * np.sin(np.linspace(0, 40, 16000, dtype=np.float32))
            sf.write(input_path, audio, 16000)
            sf.write(output_path, audio * 0.9, 16000)

            result = evaluate_quality_gate(input_path, output_path)

        self.assertTrue(result["passed"])
        self.assertFalse(result["fallback_used"])

    def test_quality_gate_fails_when_output_is_mostly_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.wav"
            output_path = tmp_path / "output.wav"
            input_audio = 0.1 * np.sin(np.linspace(0, 40, 16000, dtype=np.float32))
            output_audio = np.zeros(16000, dtype=np.float32)
            sf.write(input_path, input_audio, 16000)
            sf.write(output_path, output_audio, 16000)

            result = evaluate_quality_gate(input_path, output_path)

        self.assertFalse(result["passed"])
        self.assertTrue(result["fallback_used"])
        self.assertTrue(
            {"voiced_coverage_drop", "severe_rms_drop"}.intersection(result["warnings"])
        )


class MetadataTests(unittest.TestCase):
    def test_build_result_payload_keeps_legacy_keys_and_adds_quality_gate(self):
        payload = build_result_payload(
            status="succeeded",
            input_path=Path("/tmp/input.mp3"),
            output_path=Path("/tmp/output.mp3"),
            analysis={"classification": "male"},
            selected_plan={"processing_mode": "single"},
            segments=[{"route": "male"}],
            log=["ok"],
            error="",
            quality_gate={"passed": True, "fallback_used": False, "warnings": []},
            job_dir=Path("/tmp/job"),
            stage_summaries={"uvr_split_summary": {"ok": True}},
        )

        self.assertEqual(payload["status"], "succeeded")
        self.assertEqual(payload["analysis"]["classification"], "male")
        self.assertIn("quality_gate", payload)
        self.assertEqual(payload["job_dir"], "/tmp/job")
        self.assertIn("stage_summaries", payload)
        json.dumps(payload)

    def test_append_review_record_only_writes_reviewable_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue_path = Path(tmp) / "review.jsonl"
            result = build_result_payload(
                status="fallback",
                input_path=Path("/tmp/input.mp3"),
                output_path=Path("/tmp/output.mp3"),
                analysis={"classification": "male"},
                selected_plan={"processing_mode": "single"},
                segments=[],
                log=[],
                error="",
                quality_gate={"passed": False, "fallback_used": True, "warnings": ["severe_rms_drop"]},
                job_dir=Path("/tmp/job"),
                review={"needs_review": True, "reasons": ["quality_gate_fallback"], "uncertain_segment_count": 0, "uncertain_segments": []},
            )

            append_review_record(result, queue_path)

            lines = queue_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["status"], "fallback")
            self.assertEqual(record["review"]["reasons"], ["quality_gate_fallback"])


class SegmentReviewTests(unittest.TestCase):
    def test_segment_review_collects_uncertain_segments(self):
        review = summarize_segment_review(
            [
                {
                    "segment_id": 0,
                    "route": "male",
                    "speaker_cluster_id": "m1",
                    "gender_confidence": 0.92,
                    "note": "ok",
                    "duration_sec": 1.2,
                },
                {
                    "segment_id": 1,
                    "route": "passthrough",
                    "speaker_cluster_id": "p1",
                    "gender_confidence": 0.25,
                    "note": "low_voice",
                    "duration_sec": 0.4,
                },
            ]
        )

        self.assertTrue(review["needs_review"])
        self.assertEqual(review["uncertain_segment_count"], 1)

    def test_segment_review_accepts_long_pipeline_ok_segments_without_precomputed_confidence(self):
        review = summarize_segment_review(
            [
                {"start": 8.871, "end": 16.694, "route": "male", "voiced_ratio": 0.7241, "note": "ok"},
                {"start": 16.987, "end": 24.484, "route": "female", "voiced_ratio": 0.5733, "note": "ok"},
            ]
        )

        self.assertFalse(review["needs_review"])
        self.assertEqual(review["uncertain_segment_count"], 0)

    def test_segment_review_flags_long_pipeline_low_voiced_ok_segments(self):
        review = summarize_segment_review(
            [
                {"start": 0.0, "end": 18.599, "route": "male", "voiced_ratio": 0.2505, "note": "ok"},
                {"start": 18.599, "end": 35.178, "route": "male", "voiced_ratio": 0.31, "note": "ok"},
            ]
        )

        self.assertTrue(review["needs_review"])
        self.assertEqual(review["uncertain_segment_count"], 2)
        self.assertEqual(review["uncertain_segments"][0]["duration_sec"], 18.599)

    def test_sfx_summary_accepts_none_voiced_ratio(self):
        summary = summarize_sfx_segments(
            [
                {
                    "segment_id": 0,
                    "route": "male",
                    "voiced_ratio": None,
                    "duration_sec": 1.0,
                    "note": "single_auto",
                }
            ]
        )

        self.assertEqual(summary["count"], 0)


if __name__ == "__main__":
    unittest.main()
