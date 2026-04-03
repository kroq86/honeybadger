from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search_ranker_pipeline import (
    train_search_ranker_from_benchmark,
    write_full_two_step_windows,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the full two-step benchmark, export search traces, and train a learned search ranker."
    )
    parser.add_argument("--benchmark-output", default=None)
    parser.add_argument("--benchmark-input", default=None)
    parser.add_argument("--traces-output-dir", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument(
        "--candidate-mode",
        choices=["strict_local", "program_global"],
        default="program_global",
    )
    parser.add_argument(
        "--seed-ranker",
        choices=["none", "heuristic"],
        default="heuristic",
    )
    parser.add_argument(
        "--target-mode",
        choices=["intermediate_oracle", "final_state_only"],
        default="intermediate_oracle",
    )
    parser.add_argument("--node-budget", type=int, default=12)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--drop-program-name", action="store_true")
    parser.add_argument("--drop-candidate-source", action="store_true")
    parser.add_argument("--structural-source-hints", action="store_true")
    parser.add_argument("--rank-difficulty-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--late-step-weight", type=float, default=0.0)
    parser.add_argument(
        "--trace-split-name",
        choices=["train", "val", "test"],
        default=None,
    )
    parser.add_argument("--summary-output", default=None)
    args = parser.parse_args()

    if args.benchmark_input and args.benchmark_output:
        parser.error("pass only one of --benchmark-input or --benchmark-output")
    if not args.benchmark_input and not args.benchmark_output:
        parser.error("pass --benchmark-input or --benchmark-output")

    if args.benchmark_input:
        benchmark_path = Path(args.benchmark_input)
    else:
        benchmark_path = write_full_two_step_windows(args.benchmark_output)

    summary = train_search_ranker_from_benchmark(
        benchmark_path=benchmark_path,
        traces_output_dir=args.traces_output_dir,
        model_output_path=args.model_output,
        candidate_limit=args.candidate_limit,
        candidate_mode=args.candidate_mode,
        seed_ranker=args.seed_ranker,
        target_mode=args.target_mode,
        node_budget=args.node_budget,
        smoothing=args.smoothing,
        include_program_name=not args.drop_program_name,
        include_candidate_source=not args.drop_candidate_source,
        include_structural_source_hints=args.structural_source_hints,
        trace_split_name=args.trace_split_name,
        rank_difficulty_weight=args.rank_difficulty_weight,
        hard_negative_weight=args.hard_negative_weight,
        late_step_weight=args.late_step_weight,
    )

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
