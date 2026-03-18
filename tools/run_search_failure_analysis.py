from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from branch_ranker import rank_candidates
from candidate_generator import generate_candidates
from search_runner import solve_next_2_steps_record
from vm_transition_verifier import _section_map, load_jsonl, verification_mode_verdict, verify_single_step


def classify_failure(
    record: dict,
    *,
    candidate_limit: int,
    candidate_mode: str,
    ranker: str,
    ranker_model_path: str | None,
    verifier_mode: str,
    node_budget: int,
) -> dict:
    result = solve_next_2_steps_record(
        record,
        candidate_limit=candidate_limit,
        candidate_mode=candidate_mode,
        ranker=ranker,
        ranker_model_path=ranker_model_path,
        target_mode=verifier_mode,
        node_budget=node_budget,
    )
    if result.solved:
        return {"category": "solved", "program_name": record["program_name"]}

    sections = _section_map(record["prompt"])
    target_sections = _section_map(record["target"])
    inputs = record.get("input_values")

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
    first_matches = []
    for idx, candidate in enumerate(first_candidates, start=1):
        first = verify_single_step(
            program_name=record["program_name"],
            input_values=inputs,
            before_state_text=sections["S0"],
            instruction_text=candidate.instruction_text,
            target_state_text=target_sections["S1"] if verifier_mode in {"intermediate_oracle", "state_diff"} else None,
        )
        first_ok, _ = verification_mode_verdict(first, mode=verifier_mode)
        if verifier_mode == "intermediate_oracle":
            reaches_first = first.valid
        else:
            reaches_first = first.after_state_text == target_sections["S1"] and first.error_type in {None, "NONE"}
        if first_ok and reaches_first:
            first_matches.append((idx, candidate, first))

    if not first_matches:
        return {"category": "first_step_unreachable", "program_name": record["program_name"]}

    best_first_rank, _, gold_first_result = first_matches[0]
    second_candidates = rank_candidates(
        generate_candidates(
            program_name=record["program_name"],
            before_state_text=gold_first_result.after_state_text,
            mode=candidate_mode,
            limit=candidate_limit,
        ),
        before_state_text=gold_first_result.after_state_text,
        program_name=record["program_name"],
        remaining_steps=1,
        strategy=ranker,
        model_path=ranker_model_path,
    )
    second_matches = []
    for idx, candidate in enumerate(second_candidates, start=1):
        second = verify_single_step(
            program_name=record["program_name"],
            input_values=inputs,
            before_state_text=gold_first_result.after_state_text,
            instruction_text=candidate.instruction_text,
            target_state_text=target_sections["S2"] if verifier_mode in {"intermediate_oracle", "final_state_only", "state_diff"} else None,
        )
        second_ok, _ = verification_mode_verdict(second, mode=verifier_mode)
        reaches_second = second.after_state_text == target_sections["S2"] and second.error_type in {None, "NONE"}
        if second_ok and reaches_second:
            second_matches.append((idx, candidate, second))

    if not second_matches:
        return {"category": "second_step_unreachable", "program_name": record["program_name"]}

    best_second_rank = second_matches[0][0]
    min_nodes_needed = best_first_rank + best_second_rank
    if result.budget_exhausted or min_nodes_needed > node_budget:
        return {
            "category": "budget_or_rank_order",
            "program_name": record["program_name"],
            "best_first_rank": best_first_rank,
            "best_second_rank": best_second_rank,
            "min_nodes_needed": min_nodes_needed,
        }
    return {
        "category": "search_path_miss",
        "program_name": record["program_name"],
        "best_first_rank": best_first_rank,
        "best_second_rank": best_second_rank,
        "min_nodes_needed": min_nodes_needed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze search failures on a held-out benchmark.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--candidate-mode", choices=["strict_local", "program_global"], default="program_global")
    parser.add_argument("--ranker", choices=["none", "heuristic", "learned"], required=True)
    parser.add_argument("--ranker-model-path", default=None)
    parser.add_argument("--verifier-mode", choices=["instruction_only", "state_diff", "final_state_only", "intermediate_oracle"], default="state_diff")
    parser.add_argument("--node-budget", type=int, default=12)
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    details = [
        classify_failure(
            row,
            candidate_limit=args.candidate_limit,
            candidate_mode=args.candidate_mode,
            ranker=args.ranker,
            ranker_model_path=args.ranker_model_path,
            verifier_mode=args.verifier_mode,
            node_budget=args.node_budget,
        )
        for row in rows
    ]
    counts = Counter(detail["category"] for detail in details)
    summary = {
        "dataset_path": args.dataset,
        "ranker": args.ranker,
        "verifier_mode": args.verifier_mode,
        "node_budget": args.node_budget,
        "candidate_limit": args.candidate_limit,
        "candidate_mode": args.candidate_mode,
        "counts": dict(counts),
        "details": details,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
