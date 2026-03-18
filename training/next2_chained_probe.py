from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "mvp"

if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))
    from baseline_trainer import (
        canonical_state_from_parsed,
        load_split,
        next_2_field_metrics,
        parse_labeled_states,
        parse_state_text,
        repair_prediction_for_stage,
        state_field_metrics,
    )
    from training.eval_checkpoint import _generate, _load_model_pair
else:
    from baseline_trainer import (
        canonical_state_from_parsed,
        load_split,
        next_2_field_metrics,
        parse_labeled_states,
        parse_state_text,
        repair_prediction_for_stage,
        state_field_metrics,
    )
    from training.eval_checkpoint import _generate, _load_model_pair


def split_prompt(prompt: str) -> tuple[str, str, str]:
    lines = prompt.strip().splitlines()
    e1_idx = lines.index("E1")
    e2_idx = lines.index("E2")
    prefix_lines = lines[:e1_idx]
    if prefix_lines[:2] == ["TASK: next_2_steps_execution", "Emit S1 and S2 only."]:
        prefix_lines = prefix_lines[2:]
    prefix = "\n".join(prefix_lines).strip()
    e1 = lines[e1_idx + 1].strip()
    e2 = lines[e2_idx + 1].strip()
    return prefix, e1, e2


def split_target(target: str) -> tuple[str, str]:
    sections = parse_labeled_states(target)
    return sections["S1"], sections["S2"]


def build_step1_prompt(prefix: str, e1: str) -> str:
    return "\n".join(
        [
            "TASK: chained_step1",
            "Emit next canonical machine state only.",
            prefix,
            "E1",
            e1,
        ]
    )


def build_step2_prompt(prefix: str, s1: str, e2: str) -> str:
    return "\n".join(
        [
            "TASK: chained_step2",
            "Emit next canonical machine state only.",
            prefix,
            "S1",
            s1,
            "E2",
            e2,
        ]
    )


STEP_SYSTEM_PROMPT = "AI-assembly VM executor.\nReturn canonical target text only.\nNo prose. No markdown. No extra lines.\nNext canonical machine state only."


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


def run_probe(
    base_model_name: str,
    adapter_path: Path | None,
    dataset_root: Path,
    split: str,
    limit: int,
    max_new_tokens: int,
    trust_remote_code: bool,
) -> dict[str, Any]:
    if adapter_path is None:
        tokenizer, active_model, device = _load_base_model(base_model_name, trust_remote_code)
        model_label = "base"
    else:
        tokenizer, _, finetuned_model, device = _load_model_pair(base_model_name, adapter_path, trust_remote_code)
        active_model = finetuned_model
        model_label = "finetuned"

    examples = load_split(dataset_root, "next_2_steps", split)[:limit]
    records: list[dict[str, Any]] = []
    step1_accuracies: list[float] = []
    step2_gold_accuracies: list[float] = []
    step2_pred_accuracies: list[float] = []

    for example in examples:
        prefix, e1, e2 = split_prompt(example.prompt)
        s1_target, s2_target = split_target(example.target)

        step1_raw = _generate(tokenizer, active_model, build_step1_prompt(prefix, e1), max_new_tokens, device)
        step1_repaired = repair_prediction_for_stage("single_step", step1_raw)
        step1_metrics = state_field_metrics(step1_repaired, canonical_state_from_parsed(parse_state_text(s1_target)))
        step1_accuracies.append(float(step1_metrics["field_accuracy"]))

        step2_gold_raw = _generate(tokenizer, active_model, build_step2_prompt(prefix, s1_target, e2), max_new_tokens, device)
        step2_gold_repaired = repair_prediction_for_stage("single_step", step2_gold_raw)
        step2_gold_metrics = state_field_metrics(step2_gold_repaired, canonical_state_from_parsed(parse_state_text(s2_target)))
        step2_gold_accuracies.append(float(step2_gold_metrics["field_accuracy"]))

        step2_pred_raw = _generate(tokenizer, active_model, build_step2_prompt(prefix, step1_repaired, e2), max_new_tokens, device)
        step2_pred_repaired = repair_prediction_for_stage("single_step", step2_pred_raw)
        step2_pred_metrics = state_field_metrics(step2_pred_repaired, canonical_state_from_parsed(parse_state_text(s2_target)))
        step2_pred_accuracies.append(float(step2_pred_metrics["field_accuracy"]))

        joint_metrics = next_2_field_metrics(repair_prediction_for_stage("next_2_steps", ""), example.target)
        records.append(
            {
                "program_name": example.program_name,
                "split_family": example.split_family,
                "step1_field_accuracy": step1_metrics["field_accuracy"],
                "step2_gold_s1_field_accuracy": step2_gold_metrics["field_accuracy"],
                "step2_predicted_s1_field_accuracy": step2_pred_metrics["field_accuracy"],
                "step1_prediction": step1_repaired,
                "step2_gold_s1_prediction": step2_gold_repaired,
                "step2_predicted_s1_prediction": step2_pred_repaired,
                "joint_reference_zero": joint_metrics["field_accuracy"],
            }
        )

    def avg(values: list[float]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    return {
        "model_label": model_label,
        "base_model_name": base_model_name,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "split": split,
        "limit": limit,
        "step1_avg_field_accuracy": avg(step1_accuracies),
        "step2_gold_s1_avg_field_accuracy": avg(step2_gold_accuracies),
        "step2_predicted_s1_avg_field_accuracy": avg(step2_pred_accuracies),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight chained next_2_steps probe.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    payload = run_probe(
        base_model_name=args.base_model,
        adapter_path=adapter_path,
        dataset_root=Path(args.dataset_root),
        split=args.split,
        limit=args.limit,
        max_new_tokens=args.num_predict,
        trust_remote_code=args.trust_remote_code,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
