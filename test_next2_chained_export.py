import json
import tempfile
import unittest
from pathlib import Path

from test_paths import repo_path
from training.next2_chained_export import (
    export_chained,
    step1_record,
    step2_record,
)


class Next2ChainedExportTests(unittest.TestCase):
    def setUp(self) -> None:
        sample_path = repo_path("datasets", "mvp", "next_2_steps", "train.jsonl")
        self.sample = json.loads(
            sample_path.read_text(encoding="utf-8").splitlines()[0]
        )

    def test_step_records_split_into_sequential_targets(self) -> None:
        step1_benchmark, step1_training = step1_record(self.sample, "train")
        step2_benchmark, step2_training = step2_record(self.sample, "train")
        self.assertEqual(step1_benchmark["dataset_type"], "next_2_chained_step1")
        self.assertEqual(step2_benchmark["dataset_type"], "next_2_chained_step2")
        self.assertIn("TASK: chained_step1", step1_benchmark["prompt"])
        self.assertIn("TASK: chained_step2", step2_benchmark["prompt"])
        self.assertIn("\nS1\n", step2_benchmark["prompt"])
        self.assertNotIn("\nS2\n", step1_training["completion"])
        self.assertNotIn("\nS1\n", step2_training["completion"])

    def test_export_chained_writes_training_and_benchmark_views(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            training_dir = Path(temp_dir) / "training"
            benchmark_root = Path(temp_dir) / "benchmark"
            manifest = export_chained(
                source_root=repo_path("datasets", "mvp", "next_2_steps"),
                training_output_dir=training_dir,
                benchmark_output_root=benchmark_root,
            )
            self.assertEqual(manifest["dataset_type"], "next_2_chained")
            self.assertTrue((training_dir / "train.jsonl").exists())
            self.assertTrue(
                (benchmark_root / "next_2_chained_step1" / "train.jsonl").exists()
            )
            self.assertTrue(
                (benchmark_root / "next_2_chained_step2" / "train.jsonl").exists()
            )


if __name__ == "__main__":
    unittest.main()
