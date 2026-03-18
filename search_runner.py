from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidate_generator import generate_candidates
from branch_ranker import rank_candidates
from vm_transition_verifier import (
    VerificationResult,
    _section_map,
    load_jsonl,
    verification_mode_verdict,
    verify_single_step,
)


@dataclass(frozen=True)
class SearchAttempt:
    instruction_text: str
    source: str
    result: VerificationResult


@dataclass(frozen=True)
class SearchResult:
    solved: bool
    program_name: str
    attempts: tuple[SearchAttempt, ...]
    successful_path: tuple[str, ...]
    nodes_explored: int
    budget_exhausted: bool


def solve_next_2_steps_record(
    record: dict[str, Any],
    candidate_limit: int = 5,
    *,
    candidate_mode: str = "strict_local",
    ranker: str = "none",
    ranker_model_path: str | None = None,
    target_mode: str = "intermediate_oracle",
    node_budget: int | None = None,
) -> SearchResult:
    sections = _section_map(record["prompt"])
    target_sections = _section_map(record["target"])
    inputs = record.get("input_values")

    attempts: list[SearchAttempt] = []
    successful_path: list[str] = []
    nodes_explored = 0

    first_candidates = rank_candidates(
        generate_candidates(
            program_name=record["program_name"],
            before_state_text=sections["S0"],
            mode=candidate_mode,
            limit=candidate_limit,
        ),
        before_state_text=sections["S0"],
        program_name=record["program_name"],
        remaining_steps=2,
        strategy=ranker,
        model_path=ranker_model_path,
    )

    for candidate in first_candidates:
        if node_budget is not None and nodes_explored >= node_budget:
            return SearchResult(
                solved=False,
                program_name=record["program_name"],
                attempts=tuple(attempts),
                successful_path=(),
                nodes_explored=nodes_explored,
                budget_exhausted=True,
            )
        first = verify_single_step(
            program_name=record["program_name"],
            input_values=inputs,
            before_state_text=sections["S0"],
            instruction_text=candidate.instruction_text,
            target_state_text=target_sections["S1"] if target_mode in {"intermediate_oracle", "state_diff"} else None,
        )
        attempts.append(SearchAttempt(candidate.instruction_text, candidate.source, first))
        nodes_explored += 1
        first_valid, _ = verification_mode_verdict(first, mode=target_mode)
        if not first_valid:
            continue
        successful_path.append(candidate.instruction_text)

        second_candidates = rank_candidates(
            generate_candidates(
                program_name=record["program_name"],
                before_state_text=first.after_state_text,
                mode=candidate_mode,
                limit=candidate_limit,
            ),
            before_state_text=first.after_state_text,
            program_name=record["program_name"],
            remaining_steps=1,
            strategy=ranker,
            model_path=ranker_model_path,
        )

        for second_candidate in second_candidates:
            if node_budget is not None and nodes_explored >= node_budget:
                return SearchResult(
                    solved=False,
                    program_name=record["program_name"],
                    attempts=tuple(attempts),
                    successful_path=(),
                    nodes_explored=nodes_explored,
                    budget_exhausted=True,
                )
            second = verify_single_step(
                program_name=record["program_name"],
                input_values=inputs,
                before_state_text=first.after_state_text,
                instruction_text=second_candidate.instruction_text,
                target_state_text=target_sections["S2"] if target_mode in {"intermediate_oracle", "final_state_only", "state_diff"} else None,
            )
            attempts.append(SearchAttempt(second_candidate.instruction_text, second_candidate.source, second))
            nodes_explored += 1
            if target_mode == "intermediate_oracle":
                solved = second.valid
            else:
                solved = second.after_state_text == target_sections["S2"] and second.error_type in {None, "NONE"}
            if solved:
                successful_path.append(second_candidate.instruction_text)
                return SearchResult(
                    solved=True,
                    program_name=record["program_name"],
                    attempts=tuple(attempts),
                    successful_path=tuple(successful_path),
                    nodes_explored=nodes_explored,
                    budget_exhausted=False,
                )

        successful_path.clear()

    return SearchResult(
        solved=False,
        program_name=record["program_name"],
        attempts=tuple(attempts),
        successful_path=(),
        nodes_explored=nodes_explored,
        budget_exhausted=False,
    )


def run_next_2_steps_search(
    path: str | Path,
    candidate_limit: int = 5,
    *,
    candidate_mode: str = "strict_local",
    ranker: str = "none",
    ranker_model_path: str | None = None,
    target_mode: str = "intermediate_oracle",
    node_budget: int | None = None,
) -> dict[str, Any]:
    rows = load_jsonl(path)
    results = [
        solve_next_2_steps_record(
            row,
            candidate_limit=candidate_limit,
            candidate_mode=candidate_mode,
            ranker=ranker,
            ranker_model_path=ranker_model_path,
            target_mode=target_mode,
            node_budget=node_budget,
        )
        for row in rows
    ]
    solved = sum(1 for result in results if result.solved)
    attempts = [len(result.attempts) for result in results]
    nodes = [result.nodes_explored for result in results]
    budget_exhausted = sum(1 for result in results if result.budget_exhausted)
    return {
        "dataset_path": str(path),
        "candidate_limit": candidate_limit,
        "candidate_mode": candidate_mode,
        "ranker": ranker,
        "ranker_model_path": ranker_model_path,
        "target_mode": target_mode,
        "node_budget": node_budget,
        "total_records": len(results),
        "solved_records": solved,
        "solve_rate": (solved / len(results)) if results else 0.0,
        "avg_attempts": (sum(attempts) / len(attempts)) if attempts else 0.0,
        "avg_nodes_explored": (sum(nodes) / len(nodes)) if nodes else 0.0,
        "budget_exhausted_records": budget_exhausted,
        "records": [
            {
                "program_name": result.program_name,
                "solved": result.solved,
                "successful_path": list(result.successful_path),
                "attempt_count": len(result.attempts),
                "nodes_explored": result.nodes_explored,
                "budget_exhausted": result.budget_exhausted,
            }
            for result in results
        ],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run depth-2 search over next_2_steps records.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--candidate-mode", choices=["strict_local", "program_global"], default="strict_local")
    parser.add_argument("--ranker", choices=["none", "heuristic", "learned"], default="none")
    parser.add_argument("--ranker-model-path", default=None)
    parser.add_argument("--target-mode", choices=["intermediate_oracle", "final_state_only", "instruction_only", "state_diff"], default="intermediate_oracle")
    parser.add_argument("--node-budget", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = run_next_2_steps_search(
        args.dataset,
        candidate_limit=args.candidate_limit,
        candidate_mode=args.candidate_mode,
        ranker=args.ranker,
        ranker_model_path=args.ranker_model_path,
        target_mode=args.target_mode,
        node_budget=args.node_budget,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
