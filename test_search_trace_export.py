from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from test_paths import repo_path
from search_trace_export import (
    build_search_trace_records,
    export_search_trace_splits,
)


class SearchTraceExportTests(unittest.TestCase):
    def test_build_search_trace_records_returns_decision_examples(
        self,
    ) -> None:
        rows = build_search_trace_records(
            repo_path("reports", "vmbench", "search", "two_step_windows_benchmark.jsonl"),
            candidate_limit=8,
            candidate_mode="program_global",
            ranker="heuristic",
            target_mode="final_state_only",
            node_budget=8,
        )
        self.assertTrue(rows)
        sample = rows[0]
        self.assertIn("prompt", sample)
        self.assertIn("completion", sample)
        self.assertIn("metadata", sample)
        self.assertEqual(sample["metadata"]["dataset_type"], "search_trace_branch_ranking")

    def test_export_writes_manifest_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = export_search_trace_splits(
                repo_path(
                    "reports",
                    "vmbench",
                    "search",
                    "two_step_windows_benchmark.jsonl",
                ),
                tmpdir,
                candidate_limit=8,
                candidate_mode="program_global",
                ranker="heuristic",
                target_mode="final_state_only",
                node_budget=8,
            )
            self.assertEqual(
                manifest["format"], "search_trace_branch_ranking_v1"
            )
            self.assertTrue((Path(tmpdir) / "manifest.json").exists())
            self.assertTrue((Path(tmpdir) / "train.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
