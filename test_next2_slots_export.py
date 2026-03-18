import json
import tempfile
import unittest
from pathlib import Path

from test_paths import repo_path
from training.next2_slots_export import build_slot_target, export_slots


class Next2SlotsExportTests(unittest.TestCase):
    def setUp(self) -> None:
        sample_path = repo_path("datasets", "mvp", "next_2_steps", "train.jsonl")
        self.sample = json.loads(
            sample_path.read_text(encoding="utf-8").splitlines()[0]
        )

    def test_build_slot_target_emits_ordered_labels(self) -> None:
        target = build_slot_target(self.sample["target"])
        self.assertIn("S1_IP=", target)
        self.assertIn("S1_REG=", target)
        self.assertIn("S2_ERROR=", target)
        self.assertNotIn("\nS1\n", target)

    def test_export_slots_writes_training_and_benchmark_views(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            training_dir = Path(temp_dir) / "training"
            benchmark_root = Path(temp_dir) / "benchmark"
            manifest = export_slots(
                source_root=repo_path("datasets", "mvp", "next_2_steps"),
                training_output_dir=training_dir,
                benchmark_output_root=benchmark_root,
            )
            self.assertEqual(manifest["dataset_type"], "next_2_steps_slots")
            self.assertTrue((training_dir / "train.jsonl").exists())
            self.assertTrue(
                (benchmark_root / "next_2_steps_slots" / "train.jsonl").exists()
            )


if __name__ == "__main__":
    unittest.main()
