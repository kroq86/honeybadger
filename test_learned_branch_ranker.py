from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from candidate_generator import CandidateInstruction
from learned_branch_ranker import (
    load_learned_ranker,
    rank_candidates_with_model,
    save_learned_ranker,
    train_learned_ranker,
)
from search_trace_export import build_search_trace_records
from tools.build_search_benchmark import build_two_step_windows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _sample_records() -> list[dict]:
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "benchmark.jsonl"
        _write_jsonl(dataset_path, build_two_step_windows()[:8])
        return build_search_trace_records(
            dataset_path,
            candidate_limit=8,
            candidate_mode="program_global",
            ranker="heuristic",
            target_mode="intermediate_oracle",
            node_budget=8,
            split_name="train",
        )


class LearnedBranchRankerTests(unittest.TestCase):
    def test_train_and_save_model(self) -> None:
        model = train_learned_ranker(_sample_records())
        self.assertEqual(model["format"], "learned_branch_ranker_v1")
        self.assertGreater(model["trained_examples"], 0)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.json"
            save_learned_ranker(model, model_path)
            reloaded = load_learned_ranker(model_path)
            self.assertEqual(reloaded["format"], "learned_branch_ranker_v1")

    def test_rank_candidates_with_model_prefers_seen_pattern(self) -> None:
        model = train_learned_ranker(_sample_records())
        candidates = [
            CandidateInstruction("READ R1, input[0]", 1, "current_ip"),
            CandidateInstruction("ADD R3, R1, R2", 2, "next2_ip"),
        ]
        before_state_text = "\n".join(
            [
                "IP=0",
                "R0=0 R1=0 R2=0 R3=0 R4=0 R5=0 R6=0 R7=0",
                "Z=0 N=0 C=0",
                "OUT[0]=_",
                "HALTED=0",
                "ERROR=NONE",
            ]
        )
        ranked = rank_candidates_with_model(
            candidates,
            model=model,
            program_name="add_two_numbers",
            remaining_steps=2,
            before_state_text=before_state_text,
        )
        self.assertEqual(ranked[0].instruction_text, "READ R1, input[0]")

    def test_train_model_supports_feature_ablation_flags(self) -> None:
        model = train_learned_ranker(
            _sample_records(),
            include_program_name=False,
            include_candidate_source=False,
        )
        self.assertFalse(model["feature_flags"]["include_program_name"])
        self.assertFalse(model["feature_flags"]["include_candidate_source"])

    def test_train_model_supports_structural_source_hints(self) -> None:
        model = train_learned_ranker(
            _sample_records(),
            include_program_name=False,
            include_candidate_source=False,
            include_structural_source_hints=True,
        )
        self.assertTrue(model["feature_flags"]["include_structural_source_hints"])


if __name__ == "__main__":
    unittest.main()
