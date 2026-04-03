from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.build_search_benchmark import build_two_step_windows
from tools.split_search_benchmark import _read_jsonl, _write_jsonl
from dataset_pipeline import split_records


class SearchBenchmarkToolTests(unittest.TestCase):
    def test_build_two_step_windows_includes_family_and_category(self) -> None:
        rows = build_two_step_windows()

        self.assertTrue(rows)
        sample = rows[0]
        self.assertIn("program_name", sample)
        self.assertIn("split_family", sample)
        self.assertIn("category", sample)
        self.assertEqual(sample["dataset_type"], "next_2_steps_search_benchmark")

    def test_family_split_keeps_each_family_in_one_split(self) -> None:
        rows = build_two_step_windows()
        splits = split_records(rows, seed=7)

        family_to_split: dict[str, str] = {}
        for split_name, split_rows in splits.items():
            for row in split_rows:
                family = row["split_family"]
                if family in family_to_split:
                    self.assertEqual(family_to_split[family], split_name)
                family_to_split[family] = split_name

    def test_jsonl_helpers_round_trip_rows(self) -> None:
        rows = [
            {"program_name": "a", "split_family": "fam_a", "category": "branch"},
            {"program_name": "b", "split_family": "fam_b", "category": "loop"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            _write_jsonl(path, rows)
            loaded = _read_jsonl(path)

        self.assertEqual(rows, loaded)

    def test_split_manifest_shape_is_json_serializable(self) -> None:
        rows = build_two_step_windows()
        splits = split_records(rows, seed=7)
        manifest = {
            "split_counts": {split_name: len(split_rows) for split_name, split_rows in splits.items()}
        }

        payload = json.dumps(manifest)
        self.assertIn("train", payload)
        self.assertIn("test", payload)


if __name__ == "__main__":
    unittest.main()
