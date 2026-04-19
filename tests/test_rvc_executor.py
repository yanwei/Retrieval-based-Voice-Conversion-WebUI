import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.rvc_pipeline.executor import (
    _finalize_long_mixed_output,
    _load_long_mixed_segments,
    execute_long_mixed_pipeline,
)


class LongMixedExecutorTests(unittest.TestCase):
    def test_load_long_mixed_segments_reads_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            (job_dir / "segments.csv").write_text(
                "start_sec,end_sec,route,median_f0_hz,voiced_ratio,note\n"
                "0.0,1.2,male,120.0,0.8,ok\n",
                encoding="utf-8",
            )

            segments = _load_long_mixed_segments(job_dir)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["route"], "male")
        self.assertEqual(segments[0]["note"], "ok")

    def test_finalize_long_mixed_output_copies_final_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            job_dir = tmp_path / "job"
            output_path = tmp_path / "result.mp3"
            job_dir.mkdir()
            (job_dir / "final_mix.mp3").write_bytes(b"mix")

            _finalize_long_mixed_output(job_dir, output_path)

            self.assertEqual(output_path.read_bytes(), b"mix")

    def test_execute_long_mixed_pipeline_writes_stage_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.wav"
            output_path = tmp_path / "result.mp3"
            job_dir = tmp_path / "job"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plan = {
                "models": [{"role": "male", "model": "man.pth"}, {"role": "female", "model": "woman.pth"}],
                "parameters": {
                    "male_params": {},
                    "female_params": {},
                    "uvr_model": "auto",
                    "reading_mode": True,
                    "speaker_embedding": False,
                },
            }

            def fake_run_long_mixed_process(_input_path, _job_dir, _progress_callback):
                _job_dir.mkdir(parents=True, exist_ok=True)
                stages = [
                    {"name": "prepare", "status": "completed"},
                    {"name": "uvr_split", "status": "completed"},
                    {"name": "segment_merge", "status": "completed"},
                    {"name": "convert_remix", "status": "running"},
                ]
                return stages, {"job_dir": _job_dir}

            def fake_convert_and_remix(execution_context, _job_dir, _progress_callback):
                self.assertEqual(execution_context["job_dir"], _job_dir)
                (_job_dir / "final_mix.mp3").write_bytes(b"mix")
                (_job_dir / "segments.csv").write_text(
                    "start_sec,end_sec,route,median_f0_hz,voiced_ratio,note\n"
                    "0.0,1.0,female,220.0,0.7,ok\n",
                    encoding="utf-8",
                )

            with (
                patch("tools.rvc_pipeline.executor._run_long_mixed_process", side_effect=fake_run_long_mixed_process),
                patch(
                    "tools.rvc_pipeline.executor._execute_long_mixed_convert_and_remix",
                    side_effect=fake_convert_and_remix,
                ),
            ):
                segments, logs = execute_long_mixed_pipeline(
                    input_path=input_path,
                    output_path=output_path,
                    job_dir=job_dir,
                    plan=plan,
                    progress_callback=None,
                )

            self.assertEqual(logs, [])
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0]["route"], "female")
            self.assertEqual(output_path.read_bytes(), b"mix")
            manifest = json.loads((job_dir / "long_mixed_pipeline_stages.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pipeline"], "long_mixed_pipeline")
            self.assertEqual(
                [item["name"] for item in manifest["stages"]],
                ["prepare", "uvr_split", "segment_merge", "convert_remix", "collect_outputs", "done"],
            )
            self.assertEqual(manifest["stages"][-1]["name"], "done")
            self.assertEqual(manifest["stages"][-1]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
