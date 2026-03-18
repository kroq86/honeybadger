import json
import tempfile
import unittest
from pathlib import Path

from test_paths import repo_path
from sft_export import export_sft_splits
from training.next2_factorized_export import (
    build_manifest,
    export_factorized_splits,
    to_factorized_records,
)


class Next2FactorizedExportTests(unittest.TestCase):
    def setUp(self) -> None:
        exported = export_sft_splits(repo_path("datasets", "mvp"), ["next_2_steps"])
        if not exported["train"]:
            raise AssertionError(
                "no next_2_steps sample found in canonical export"
            )
        self.sample = exported["train"][0]

    def test_factorized_records_split_d1_and_d2(self) -> None:
        records = to_factorized_records(self.sample)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["metadata"]["factorized_step"], "D1")
        self.assertEqual(records[1]["metadata"]["factorized_step"], "D2")
        self.assertIn("KNOWN_D1", records[1]["prompt"])

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
            exported = export_factorized_splits(export_root)
        self.assertTrue(exported["train"])
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = build_manifest(Path(temp_dir), Path("/tmp/source"), exported)
        self.assertEqual(manifest["dataset_type"], "next_2_step_factorized")
        self.assertEqual(manifest["supervision_shape"], "teacher_forced_d1_d2")


if __name__ == "__main__":
    unittest.main()
