from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.search_ranker_pipeline import (
    train_search_ranker_from_benchmark,
    write_full_two_step_windows,
)


class SearchRankerPipelineTests(unittest.TestCase):
    def test_write_full_two_step_windows_writes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "two_step_windows.jsonl"
            written = write_full_two_step_windows(output_path)
            self.assertEqual(written, output_path)
            self.assertTrue(output_path.exists())
            with output_path.open("r", encoding="utf-8") as handle:
                self.assertGreater(sum(1 for _ in handle), 0)

    def test_train_search_ranker_from_benchmark_writes_model_and_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            benchmark_path = write_full_two_step_windows(tmp / "two_step_windows.jsonl")
            summary = train_search_ranker_from_benchmark(
                benchmark_path=benchmark_path,
                traces_output_dir=tmp / "traces",
                model_output_path=tmp / "model.json",
                candidate_limit=16,
                candidate_mode="program_global",
                seed_ranker="heuristic",
                target_mode="intermediate_oracle",
                node_budget=8,
            )
            self.assertEqual(summary["benchmark_path"], str(benchmark_path))
            self.assertTrue((tmp / "traces" / "manifest.json").exists())
            self.assertTrue((tmp / "traces" / "train.jsonl").exists())
            self.assertTrue((tmp / "model.json").exists())
            self.assertGreater(summary["model_summary"]["trained_examples"], 0)


if __name__ == "__main__":
    unittest.main()
