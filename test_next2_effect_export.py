import json
import tempfile
import unittest
from pathlib import Path

from test_paths import repo_path
from training.next2_effect_export import build_effect_target, export_effects


class Next2EffectExportTests(unittest.TestCase):
    def setUp(self) -> None:
        sample_path = repo_path("datasets", "mvp", "next_2_steps", "train.jsonl")
        self.sample = json.loads(
            sample_path.read_text(encoding="utf-8").splitlines()[0]
        )

    def test_build_effect_target_emits_step_fields(self) -> None:
        target = build_effect_target(self.sample["prompt"], self.sample["target"])
        self.assertIn("STEP1_OP=READ", target)
        self.assertIn("STEP1_DST=R1", target)
        self.assertIn("STEP1_VALUE=9", target)
        self.assertIn("STEP2_DST=R2", target)

    def test_export_effects_writes_training_and_benchmark_views(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            training_dir = Path(temp_dir) / "training"
            benchmark_root = Path(temp_dir) / "benchmark"
            manifest = export_effects(
                source_root=repo_path("datasets", "mvp", "next_2_steps"),
                training_output_dir=training_dir,
                benchmark_output_root=benchmark_root,
            )
            self.assertEqual(manifest["dataset_type"], "next_2_effects")
            self.assertTrue((training_dir / "train.jsonl").exists())
            self.assertTrue(
                (benchmark_root / "next_2_effects" / "train.jsonl").exists()
            )


if __name__ == "__main__":
    unittest.main()
