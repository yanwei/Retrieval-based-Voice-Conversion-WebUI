import json
import tempfile
import unittest
from pathlib import Path

from tools.review_auto_convert_queue import build_prompt, filter_queue, load_queue, parse_model_json


class ReviewAutoConvertQueueTests(unittest.TestCase):
    def test_filter_queue_respects_status_and_limit(self):
        items = [
            {"status": "fallback", "id": 1},
            {"status": "succeeded", "id": 2},
            {"status": "fallback", "id": 3},
        ]

        filtered = filter_queue(items, only_status="fallback", limit=1)

        self.assertEqual(filtered, [{"status": "fallback", "id": 1}])

    def test_build_prompt_contains_requested_json_contract(self):
        prompt = build_prompt([{"status": "fallback", "processing_mode": "single"}])

        self.assertIn("summary", prompt)
        self.assertIn("recurring_patterns", prompt)
        self.assertIn("suggested_actions", prompt)
        self.assertIn("route_adjustments", prompt)
        self.assertIn("top_examples", prompt)
        self.assertIn('"status": "fallback"', prompt)

    def test_load_queue_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue_path = Path(tmp) / "review_queue.jsonl"
            queue_path.write_text(
                json.dumps({"status": "fallback"}) + "\n" + json.dumps({"status": "succeeded"}) + "\n",
                encoding="utf-8",
            )

            items = load_queue(queue_path)

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["status"], "fallback")

    def test_parse_model_json_handles_fenced_blocks(self):
        payload = parse_model_json(
            "```json\n"
            '{"summary":"ok","recurring_patterns":[],"suggested_actions":[],"route_adjustments":[],"top_examples":[]}\n'
            "```"
        )

        self.assertEqual(payload["summary"], "ok")


if __name__ == "__main__":
    unittest.main()
