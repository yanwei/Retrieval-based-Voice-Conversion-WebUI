import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.analyze_book_res_audio import (
    analyze_audio_file,
    analyze_audio_file_job,
    analyze_deep_feature,
    choose_audio_type,
    parse_args,
    classify_duration_bucket,
    create_schema,
    insert_file_features,
    summarize_database,
)


class AnalyzeBookResAudioTests(unittest.TestCase):
    def test_classify_duration_bucket_covers_expected_ranges(self):
        self.assertEqual(classify_duration_bucket(0.5), "<1s")
        self.assertEqual(classify_duration_bucket(2.0), "1-3s")
        self.assertEqual(classify_duration_bucket(5.0), "3-8s")
        self.assertEqual(classify_duration_bucket(12.0), "8-25s")
        self.assertEqual(classify_duration_bucket(40.0), "25-60s")
        self.assertEqual(classify_duration_bucket(120.0), "60-180s")
        self.assertEqual(classify_duration_bucket(300.0), ">180s")

    def test_choose_audio_type_is_mece_for_common_light_features(self):
        self.assertEqual(
            choose_audio_type(
                duration_seconds=0.4,
                voiced_coverage=0.5,
                rms=0.03,
                spectral_flatness=0.08,
                onset_strength=0.1,
                harmonic_ratio=0.5,
                percussive_ratio=0.03,
                segment_count=1,
                short_segment_ratio=1.0,
            ),
            "short_speech_or_sfx",
        )
        self.assertEqual(
            choose_audio_type(
                duration_seconds=4.0,
                voiced_coverage=0.82,
                rms=0.05,
                spectral_flatness=0.04,
                onset_strength=0.1,
                harmonic_ratio=0.55,
                percussive_ratio=0.03,
                segment_count=1,
                short_segment_ratio=0.0,
            ),
            "clean_speech_candidate",
        )
        self.assertEqual(
            choose_audio_type(
                duration_seconds=30.0,
                voiced_coverage=0.65,
                rms=0.08,
                spectral_flatness=0.02,
                onset_strength=0.8,
                harmonic_ratio=0.72,
                percussive_ratio=0.22,
                segment_count=8,
                short_segment_ratio=0.1,
            ),
            "music_or_bgm_candidate",
        )

    def test_analyze_audio_file_extracts_light_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "book1" / "track_audio" / "tone.wav"
            audio_path.parent.mkdir(parents=True)
            sr = 16000
            t = np.linspace(0, 1.0, sr, endpoint=False)
            y = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            sf.write(audio_path, y, sr)

            feature = analyze_audio_file(root, audio_path, sample_seconds=5.0)

            self.assertEqual(feature.rel_path, "book1/track_audio/tone.wav")
            self.assertEqual(feature.book_id, "book1")
            self.assertEqual(feature.audio_kind, "track_audio")
            self.assertAlmostEqual(feature.duration_seconds, 1.0, places=2)
            self.assertGreater(feature.rms, 0.01)
            self.assertGreaterEqual(feature.voiced_coverage, 0.0)
            self.assertIn(feature.duration_bucket, {"1-3s", "<1s"})

    def test_analyze_audio_file_job_returns_error_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_path = root / "book1" / "track_audio" / "bad.wav"
            bad_path.parent.mkdir(parents=True)
            bad_path.write_text("not audio", encoding="utf-8")

            status, payload = analyze_audio_file_job(str(root), str(bad_path), 5.0)

            self.assertEqual(status, "error")
            self.assertEqual(payload["rel_path"], "book1/track_audio/bad.wav")
            self.assertIn("error_type", payload)

    def test_parse_args_supports_analysis_modes_and_deep_backend(self):
        args = parse_args(
            [
                "--analysis-mode",
                "deep",
                "--deep-f0-backend",
                "rmvpe",
                "--deep-device",
                "cpu",
            ]
        )

        self.assertEqual(args.analysis_mode, "deep")
        self.assertEqual(args.deep_f0_backend, "rmvpe")
        self.assertEqual(args.deep_device, "cpu")

    def test_analyze_deep_feature_records_backend_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "book1" / "track_audio" / "tone.wav"
            audio_path.parent.mkdir(parents=True)
            sr = 16000
            t = np.linspace(0, 1.0, sr, endpoint=False)
            y = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            sf.write(audio_path, y, sr)

            feature = analyze_deep_feature(
                root,
                "book1/track_audio/tone.wav",
                sample_seconds=2.0,
                f0_backend="librosa",
                device_mode="cpu",
            )

            self.assertIn("pitch_backend=librosa:cpu", feature.notes)

    def test_sqlite_summary_counts_files_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "features.sqlite"
            conn = sqlite3.connect(db_path)
            create_schema(conn)
            feature = analyze_audio_file.__annotations__
            del feature
            from tools.analyze_book_res_audio import AudioFeature

            insert_file_features(
                conn,
                AudioFeature(
                    rel_path="book1/track_audio/a.mp3",
                    abs_path="/tmp/a.mp3",
                    book_id="book1",
                    audio_kind="track_audio",
                    suffix=".mp3",
                    file_size=100,
                    sample_rate=16000,
                    channels=1,
                    frames=16000,
                    duration_seconds=1.0,
                    bitrate_kbps=128.0,
                    analyzed_seconds=1.0,
                    peak=0.1,
                    rms=0.02,
                    loudness_db=-34.0,
                    voiced_coverage=0.9,
                    segment_count=1,
                    short_segment_count=0,
                    short_segment_ratio=0.0,
                    silence_ratio=0.1,
                    spectral_flatness=0.04,
                    onset_strength=0.1,
                    harmonic_ratio=0.5,
                    percussive_ratio=0.05,
                    duration_bucket="1-3s",
                    audio_type="clean_speech_candidate",
                    flags="clean_speech_candidate",
                ),
            )
            conn.execute(
                "insert into errors(rel_path, abs_path, error_type, error_message) values(?, ?, ?, ?)",
                ("bad.mp3", "/tmp/bad.mp3", "DecodeError", "bad"),
            )
            conn.commit()

            summary = summarize_database(conn)

            self.assertEqual(summary["total_files"], 1)
            self.assertEqual(summary["total_errors"], 1)
            self.assertEqual(summary["by_audio_type"]["clean_speech_candidate"], 1)
            conn.close()


if __name__ == "__main__":
    unittest.main()
