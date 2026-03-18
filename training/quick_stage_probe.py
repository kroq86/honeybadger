from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from baseline_trainer import load_split, repair_prediction_for_stage
    from training.eval_checkpoint import _generate
else:
    from baseline_trainer import load_split, repair_prediction_for_stage
    from training.eval_checkpoint import _generate


def _load_base_model(base_model_name: str, trust_remote_code: bool) -> tuple[Any, Any, str]:
    from training.eval_checkpoint import _load_eval_modules, _pick_device

    modules = _load_eval_modules()
    AutoModelForCausalLM = modules["AutoModelForCausalLM"]
    AutoTokenizer = modules["AutoTokenizer"]
    device = _pick_device(modules["torch"])
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=trust_remote_code)
    return tokenizer, model, device


def _score_field_accuracy(stage: str, repaired_prediction: str, target: str) -> float:
    from baseline_trainer import (
        chained_delta_metrics,
        next_2_effect_field_metrics,
        next_2_field_metrics,
        next_2_slots_field_metrics,
        state_field_metrics,
    )

    if stage in {"single_step", "terminal_state"}:
        return float(state_field_metrics(repaired_prediction, target)["field_accuracy"])
    if stage in {"next_2_chained_step1", "next_2_chained_step2"}:
        raise ValueError("chained stages require prompt-aware scoring")
    if stage == "next_2_steps":
        return float(next_2_field_metrics(repaired_prediction, target)["field_accuracy"])
    if stage == "next_2_steps_slots":
        return float(next_2_slots_field_metrics(repaired_prediction, target)["field_accuracy"])
    if stage in {"next_2_effects", "next_2_effects_target_anchor"}:
        return float(next_2_effect_field_metrics(repaired_prediction, target)["field_accuracy"])
    return 0.0


def run_probe(
    base_model_name: str,
    dataset_root: Path,
    stage: str,
    split: str,
    limit: int,
    max_new_tokens: int,
    trust_remote_code: bool,
) -> dict[str, Any]:
    tokenizer, model, device = _load_base_model(base_model_name, trust_remote_code)
    examples = load_split(dataset_root, stage, split)[:limit]
    records: list[dict[str, Any]] = []
    field_accuracies: list[float] = []
    exact_matches = 0
    repaired_exact_matches = 0

    for example in examples:
        raw_prediction = _generate(tokenizer, model, example.prompt, max_new_tokens, device)
        repaired_prediction = repair_prediction_for_stage(stage, raw_prediction)
        if stage in {"next_2_chained_step1", "next_2_chained_step2"}:
            from baseline_trainer import chained_delta_metrics

            chained_metrics = chained_delta_metrics(repaired_prediction, example.target, example.prompt, stage)
            field_accuracy = float(chained_metrics["delta_transition_score"])
        else:
            chained_metrics = None
            field_accuracy = _score_field_accuracy(stage, repaired_prediction, example.target)
        exact = raw_prediction.strip() == example.target.strip()
        repaired_exact = repaired_prediction.strip() == example.target.strip()
        if exact:
            exact_matches += 1
        if repaired_exact:
            repaired_exact_matches += 1
        field_accuracies.append(field_accuracy)
        records.append(
            {
                "program_name": example.program_name,
                "split_family": example.split_family,
                "prompt": example.prompt,
                "target": example.target,
                "raw_prediction": raw_prediction,
                "repaired_prediction": repaired_prediction,
                "field_accuracy": field_accuracy,
                "delta_transition_score": None if chained_metrics is None else chained_metrics["delta_transition_score"],
                "delta_exact": None if chained_metrics is None else chained_metrics["delta_exact"],
                "changed_field_matches": None if chained_metrics is None else chained_metrics["changed_field_matches"],
                "changed_register_matches": None if chained_metrics is None else chained_metrics["changed_register_matches"],
                "spurious_field_edits": None if chained_metrics is None else chained_metrics["spurious_field_edits"],
                "spurious_register_edits": None if chained_metrics is None else chained_metrics["spurious_register_edits"],
                "exact_match": exact,
                "repaired_exact_match": repaired_exact,
            }
        )

    count = len(records)
    return {
        "base_model_name": base_model_name,
        "dataset_root": str(dataset_root),
        "stage": stage,
        "split": split,
        "limit": limit,
        "exact_match_rate": (exact_matches / count) if count else 0.0,
        "repaired_exact_match_rate": (repaired_exact_matches / count) if count else 0.0,
        "avg_field_accuracy": (sum(field_accuracies) / count) if count else 0.0,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight base-model probe on one benchmark stage.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--num-predict", type=int, default=128)
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
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
