from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search_runner import run_next_2_steps_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Run verifier granularity ablations over a search benchmark.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--candidate-mode", choices=["strict_local", "program_global"], default="program_global")
    parser.add_argument("--rankers", nargs="+", default=["none", "heuristic", "learned"])
    parser.add_argument("--ranker-model-path", default=None)
    parser.add_argument("--budgets", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--verifier-modes", nargs="+", default=["instruction_only", "state_diff", "final_state_only"])
    args = parser.parse_args()

    rows: list[dict] = []
    for ranker in args.rankers:
        for budget in args.budgets:
            for verifier_mode in args.verifier_modes:
                report = run_next_2_steps_search(
                    args.dataset,
                    candidate_limit=args.candidate_limit,
                    candidate_mode=args.candidate_mode,
                    ranker=ranker,
                    ranker_model_path=args.ranker_model_path,
                    target_mode=verifier_mode,
                    node_budget=budget,
                )
                rows.append(
                    {
                        "ranker": ranker,
                        "node_budget": budget,
                        "verifier_mode": verifier_mode,
                        "solve_rate": report["solve_rate"],
                        "avg_nodes_explored": report["avg_nodes_explored"],
                        "budget_exhausted_records": report["budget_exhausted_records"],
                        "solved_records": report["solved_records"],
                        "total_records": report["total_records"],
                    }
                )

    summary = {
        "dataset_path": args.dataset,
        "candidate_limit": args.candidate_limit,
        "candidate_mode": args.candidate_mode,
        "rankers": args.rankers,
        "budgets": args.budgets,
        "verifier_modes": args.verifier_modes,
        "rows": rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
