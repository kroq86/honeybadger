import json
import tempfile
import unittest
from pathlib import Path

from test_paths import repo_path
from sft_export import export_sft_splits
from training.next2_delta_export import (
    build_manifest,
    export_delta_splits,
    parse_next2_target,
    parse_prompt_s0,
    to_delta_record,
)


class Next2DeltaExportTests(unittest.TestCase):
    def setUp(self) -> None:
        exported = export_sft_splits(repo_path("datasets", "mvp"), ["next_2_steps"])
        if not exported["train"]:
            raise AssertionError(
                "no next_2_steps sample found in canonical export"
            )
        self.sample = exported["train"][0]

    def test_parser_extracts_s0_s1_s2(self) -> None:
        s0 = parse_prompt_s0(self.sample["prompt"])
        target = parse_next2_target(self.sample["completion"])
        self.assertEqual(s0["IP"], 0)
        self.assertIn("S1", target)
        self.assertIn("S2", target)

    def test_delta_record_changes_task_and_target_shape(self) -> None:
        record = to_delta_record(self.sample)
        self.assertIn("TASK: next_2_steps_delta", record["prompt"])
        self.assertIn("D1", record["completion"])
        self.assertIn("D2", record["completion"])
        self.assertEqual(
            record["metadata"]["dataset_type"], "next_2_steps_delta"
        )

    def test_export_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_root = Path(temp_dir) / "source"
            export_root.mkdir(parents=True, exist_ok=True)
            records = export_sft_splits(repo_path("datasets", "mvp"), ["next_2_steps"])
            for split_name, split_records in records.items():
                path = export_root / f"{split_name}.jsonl"
                path.write_text(
                    "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in split_records),
                    encoding="utf-8",
                )
            exported = export_delta_splits(export_root)
        self.assertTrue(exported["train"])
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = build_manifest(Path(temp_dir), Path("/tmp/source"), exported)
        self.assertEqual(manifest["dataset_type"], "next_2_steps_delta")
        self.assertEqual(manifest["supervision_shape"], "delta_only_changed_fields")


if __name__ == "__main__":
    unittest.main()
