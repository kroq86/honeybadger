import unittest
from pathlib import Path

from test_paths import repo_path
from sft_export import build_manifest, export_sft_splits


class SFTExportTests(unittest.TestCase):
    def test_export_preserves_prompt_and_target_shape(self) -> None:
        dataset_root = repo_path("datasets", "mvp")
        exported = export_sft_splits(dataset_root, ["single_step"])
        self.assertTrue(exported["train"])
        sample = exported["train"][0]
        self.assertIn("prompt", sample)
        self.assertIn("completion", sample)
        self.assertIn("metadata", sample)
        self.assertEqual(sample["metadata"]["dataset_type"], "single_step")

    def test_manifest_records_no_prompt_drift_policy(self) -> None:
        dataset_root = repo_path("datasets", "mvp")
        exported = export_sft_splits(dataset_root, ["single_step", "next_2_steps"])
        manifest = build_manifest(
            Path("/tmp/out"),
            dataset_root,
            ["single_step", "next_2_steps"],
            exported,
        )
        self.assertEqual(manifest["format"], "plain_prompt_completion_v1")
        self.assertEqual(
            manifest["prompt_drift_policy"], "reuse_eval_prompts_exactly"
        )
        self.assertEqual(
            manifest["completion_policy"], "reuse_eval_targets_exactly"
        )


if __name__ == "__main__":
    unittest.main()
