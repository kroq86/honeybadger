from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search_trace_export import export_search_trace_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Export search decision traces into branch-ranking training data.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--candidate-mode", choices=["strict_local", "program_global"], default="program_global")
    parser.add_argument("--ranker", choices=["none", "heuristic"], default="heuristic")
    parser.add_argument("--target-mode", choices=["intermediate_oracle", "final_state_only"], default="final_state_only")
    parser.add_argument("--node-budget", type=int, default=12)
    args = parser.parse_args()

    export_search_trace_splits(
        args.dataset,
        args.output_dir,
        candidate_limit=args.candidate_limit,
        candidate_mode=args.candidate_mode,
        ranker=args.ranker,
        target_mode=args.target_mode,
        node_budget=args.node_budget,
    )


if __name__ == "__main__":
    main()
