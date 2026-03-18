from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidate_generator import generate_candidates
from vm_transition_verifier import _section_map, load_jsonl, verify_single_step


def _single_step_before_state(prompt: str) -> str:
    sections = _section_map(prompt)
    return sections["STATE"]


def _next2_before_state(prompt: str) -> str:
    sections = _section_map(prompt)
    return sections["S0"]


def _next2_first_target(target: str) -> str:
    sections = _section_map(target)
    return sections["S1"]


def _compute_recall(rows: list[dict[str, Any]], *, dataset_type: str, limit: int) -> dict[str, Any]:
    hits = {k: 0 for k in range(1, limit + 1)}
    details: list[dict[str, Any]] = []

    for row in rows:
        if dataset_type == "single_step":
            before_state_text = _single_step_before_state(row["prompt"])
            gold_instruction = row["instruction"]
            validity = lambda instruction_text: instruction_text == gold_instruction
        elif dataset_type == "next_2_steps":
            before_state_text = _next2_before_state(row["prompt"])
            first_target_state = _next2_first_target(row["target"])

            def validity(instruction_text: str) -> bool:
                return verify_single_step(
                    program_name=row["program_name"],
                    input_values=row.get("input_values"),
                    before_state_text=before_state_text,
                    instruction_text=instruction_text,
                    target_state_text=first_target_state,
                ).valid
        else:
            raise ValueError(f"unsupported dataset_type: {dataset_type}")

        candidates = generate_candidates(
            program_name=row["program_name"],
            before_state_text=before_state_text,
            limit=limit,
        )
        matched_at = None
        for index, candidate in enumerate(candidates, start=1):
            if validity(candidate.instruction_text):
                matched_at = index
                break
        if matched_at is not None:
            for k in range(matched_at, limit + 1):
                hits[k] += 1
        details.append(
            {
                "program_name": row["program_name"],
                "matched_at": matched_at,
                "top_candidates": [candidate.instruction_text for candidate in candidates],
            }
        )

    total = len(rows)
    return {
        "dataset_type": dataset_type,
        "total_records": total,
        "recall": {f"recall@{k}": (hits[k] / total) if total else 0.0 for k in range(1, limit + 1)},
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure candidate generator recall.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-type", required=True, choices=["single_step", "next_2_steps"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    report = _compute_recall(rows, dataset_type=args.dataset_type, limit=args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
