from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from search_trace_export import (
    build_search_trace_records,
    export_search_trace_splits,
)
from tools.build_search_benchmark import build_two_step_windows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


class SearchTraceExportTests(unittest.TestCase):
    def test_build_search_trace_records_returns_decision_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "benchmark.jsonl"
            rows = build_two_step_windows()[:6]
            _write_jsonl(dataset_path, rows)
            records = build_search_trace_records(
                dataset_path,
                candidate_limit=8,
                candidate_mode="program_global",
                ranker="heuristic",
                target_mode="final_state_only",
                node_budget=8,
                split_name="train",
            )

        self.assertTrue(records)
        sample = records[0]
        self.assertIn("prompt", sample)
        self.assertIn("completion", sample)
        self.assertIn("metadata", sample)
        self.assertEqual(sample["metadata"]["dataset_type"], "search_trace_branch_ranking")
        self.assertEqual(sample["metadata"]["split"], "train")
        self.assertIn("split_family", sample["metadata"])
        self.assertIn("category", sample["metadata"])

    def test_export_writes_manifest_and_respects_split_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "benchmark.jsonl"
            _write_jsonl(dataset_path, build_two_step_windows()[:6])
            manifest = export_search_trace_splits(
                dataset_path,
                tmp / "traces",
                candidate_limit=8,
                candidate_mode="program_global",
                ranker="heuristic",
                target_mode="final_state_only",
                node_budget=8,
                split_name="test",
            )
            self.assertEqual(manifest["format"], "search_trace_branch_ranking_v1")
            self.assertEqual(manifest["search_params"]["split_name"], "test")
            self.assertTrue((tmp / "traces" / "manifest.json").exists())
            self.assertTrue((tmp / "traces" / "test.jsonl").exists())
            self.assertFalse((tmp / "traces" / "train.jsonl").read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    unittest.main()
