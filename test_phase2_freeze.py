import json
import tempfile
import unittest
from pathlib import Path

from training.phase2_freeze import build_phase2_freeze, write_freeze


class Phase2FreezeTests(unittest.TestCase):
    def test_freeze_matches_current_stage_order(self) -> None:
        freeze = build_phase2_freeze()
        self.assertEqual(
            freeze["execution_benchmark"]["stages"],
            ["single_step", "next_2_steps", "short_trace", "terminal_state"],
        )
        self.assertEqual(
            freeze["sft_export"]["stages"],
            ["single_step", "next_2_steps", "short_trace", "terminal_state"],
        )
        self.assertEqual(freeze["selected_model"]["training_mode"], "lora")

    def test_freeze_writes_json_file(self) -> None:
        freeze = build_phase2_freeze()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "phase2_freeze.json"
            write_freeze(output_path, freeze)
            written = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(written["status"], "scaffold_frozen")
        self.assertIn("required_training_stack", written["guardrails"])


if __name__ == "__main__":
    unittest.main()
