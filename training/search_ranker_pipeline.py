from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learned_branch_ranker import save_learned_ranker, train_learned_ranker
from search_trace_export import export_search_trace_splits
from tools.build_search_benchmark import build_two_step_windows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_full_two_step_windows(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_two_step_windows()
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return output_path


def train_search_ranker_from_benchmark(
    *,
    benchmark_path: str | Path,
    traces_output_dir: str | Path,
    model_output_path: str | Path,
    candidate_limit: int = 32,
    candidate_mode: str = "program_global",
    seed_ranker: str = "heuristic",
    target_mode: str = "intermediate_oracle",
    node_budget: int = 12,
    smoothing: float = 1.0,
    include_program_name: bool = True,
    include_candidate_source: bool = True,
    include_structural_source_hints: bool = False,
    trace_split_name: str | None = None,
    rank_difficulty_weight: float = 0.0,
    hard_negative_weight: float = 0.0,
    late_step_weight: float = 0.0,
) -> dict[str, Any]:
    manifest = export_search_trace_splits(
        benchmark_path,
        traces_output_dir,
        candidate_limit=candidate_limit,
        candidate_mode=candidate_mode,
        ranker=seed_ranker,
        target_mode=target_mode,
        node_budget=node_budget,
        split_name=trace_split_name,
    )
    train_jsonl = Path(traces_output_dir) / "train.jsonl"
    model = train_learned_ranker(
        _read_jsonl(train_jsonl),
        smoothing=smoothing,
        include_program_name=include_program_name,
        include_candidate_source=include_candidate_source,
        include_structural_source_hints=include_structural_source_hints,
        rank_difficulty_weight=rank_difficulty_weight,
        hard_negative_weight=hard_negative_weight,
        late_step_weight=late_step_weight,
    )
    save_learned_ranker(model, model_output_path)
    return {
        "benchmark_path": str(benchmark_path),
        "traces_output_dir": str(traces_output_dir),
        "model_output_path": str(model_output_path),
        "trace_manifest": manifest,
        "model_summary": {
            "trained_examples": model["trained_examples"],
            "positive_examples": model["positive_examples"],
            "feature_flags": model["feature_flags"],
            "training_weights": model.get("training_weights", {}),
        },
    }
