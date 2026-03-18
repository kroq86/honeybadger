from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from baseline_trainer import (
        chained_delta_metrics,
        load_split,
        next_2_effect_field_metrics,
        next_2_field_metrics,
        next_2_slots_field_metrics,
        repair_prediction_for_stage,
        state_field_metrics,
    )
    from training.eval_checkpoint import _generate
    from training.quick_stage_probe import _load_base_model
else:
    from baseline_trainer import (
        chained_delta_metrics,
        load_split,
        next_2_effect_field_metrics,
        next_2_field_metrics,
        next_2_slots_field_metrics,
        repair_prediction_for_stage,
        state_field_metrics,
    )
    from training.eval_checkpoint import _generate
    from training.quick_stage_probe import _load_base_model


def _prefilled_prompt(prompt: str, seed_prefix: str) -> str:
    return f"{prompt.rstrip()}\n{seed_prefix}"


def _repaired_with_seed(stage: str, raw_completion: str, seed_prefix: str) -> str:
    combined = f"{seed_prefix}{raw_completion}"
    return repair_prediction_for_stage(stage, combined)


def _score(stage: str, repaired_prediction: str, target: str, prompt: str) -> dict[str, float]:
    if stage in {"single_step", "terminal_state"}:
        metrics = state_field_metrics(repaired_prediction, target)
        return {
            "field_accuracy": float(metrics["field_accuracy"]),
            "delta_transition_score": float(metrics["field_accuracy"]),
            "delta_exact": 1.0 if repaired_prediction.strip() == target.strip() else 0.0,
        }
    if stage in {"next_2_chained_step1", "next_2_chained_step2"}:
        metrics = chained_delta_metrics(repaired_prediction, target, prompt, stage)
        return {
            "field_accuracy": float(metrics["field_accuracy"]),
            "delta_transition_score": float(metrics["delta_transition_score"]),
            "delta_exact": float(metrics["delta_exact"]),
        }
    if stage == "next_2_steps":
        metrics = next_2_field_metrics(repaired_prediction, target)
    elif stage == "next_2_steps_slots":
        metrics = next_2_slots_field_metrics(repaired_prediction, target)
    elif stage == "next_2_effects":
        metrics = next_2_effect_field_metrics(repaired_prediction, target)
    else:
        return {"field_accuracy": 0.0, "delta_transition_score": 0.0, "delta_exact": 0.0}
    return {
        "field_accuracy": float(metrics["field_accuracy"]),
        "delta_transition_score": float(metrics["field_accuracy"]),
        "delta_exact": 1.0 if repaired_prediction.strip() == target.strip() else 0.0,
    }


def run_probe(
    base_model_name: str,
    dataset_root: Path,
    stage: str,
    split: str,
    limit: int,
    max_new_tokens: int,
    trust_remote_code: bool,
    seed_prefix: str,
) -> dict[str, Any]:
    tokenizer, model, device = _load_base_model(base_model_name, trust_remote_code)
    examples = load_split(dataset_root, stage, split)[:limit]
    records: list[dict[str, Any]] = []
    field_accuracies: list[float] = []
    delta_scores: list[float] = []
    delta_exacts = 0

    for example in examples:
        forced_prompt = _prefilled_prompt(example.prompt, seed_prefix)
        raw_completion = _generate(tokenizer, model, forced_prompt, max_new_tokens, device)
        repaired_prediction = _repaired_with_seed(stage, raw_completion, seed_prefix)
        scored = _score(stage, repaired_prediction, example.target, example.prompt)
        field_accuracies.append(scored["field_accuracy"])
        delta_scores.append(scored["delta_transition_score"])
        if scored["delta_exact"]:
            delta_exacts += 1
        records.append(
            {
                "program_name": example.program_name,
                "split_family": example.split_family,
                "seed_prefix": seed_prefix,
                "prompt": example.prompt,
                "forced_prompt": forced_prompt,
                "target": example.target,
                "raw_completion": raw_completion,
                "repaired_prediction": repaired_prediction,
                "field_accuracy": scored["field_accuracy"],
                "delta_transition_score": scored["delta_transition_score"],
                "delta_exact": scored["delta_exact"],
            }
        )

    count = len(records)
    return {
        "base_model_name": base_model_name,
        "dataset_root": str(dataset_root),
        "stage": stage,
        "split": split,
        "limit": limit,
        "seed_prefix": seed_prefix,
        "avg_field_accuracy": (sum(field_accuracies) / count) if count else 0.0,
        "avg_delta_transition_score": (sum(delta_scores) / count) if count else 0.0,
        "delta_exact_rate": (delta_exacts / count) if count else 0.0,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny prefix-forced basin probe.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--seed-prefix", required=True)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = run_probe(
        base_model_name=args.base_model,
        dataset_root=Path(args.dataset_root),
        stage=args.stage,
        split=args.split,
        limit=args.limit,
        max_new_tokens=args.num_predict,
        trust_remote_code=args.trust_remote_code,
        seed_prefix=args.seed_prefix,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
