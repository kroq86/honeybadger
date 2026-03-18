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
        load_split,
        next_2_field_metrics,
        parse_labeled_states,
        repair_prediction_for_stage,
    )
    from training.eval_checkpoint import _generate, _load_model_pair
else:
    from baseline_trainer import (
        load_split,
        next_2_field_metrics,
        parse_labeled_states,
        repair_prediction_for_stage,
    )
    from training.eval_checkpoint import _generate, _load_model_pair


def _load_base_model(base_model_name: str, trust_remote_code: bool) -> tuple[Any, Any, str]:
    from training.eval_checkpoint import _load_eval_modules, _pick_device

    modules = _load_eval_modules()
    torch = modules["torch"]
    AutoModelForCausalLM = modules["AutoModelForCausalLM"]
    AutoTokenizer = modules["AutoTokenizer"]

    device = _pick_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=trust_remote_code)
    return tokenizer, model, device


def classify_prediction(raw_prediction: str, repaired_prediction: str, target: str) -> dict[str, Any]:
    raw_sections = parse_labeled_states(raw_prediction)
    repaired_sections = parse_labeled_states(repaired_prediction)
    target_sections = parse_labeled_states(target)
    field_metrics = next_2_field_metrics(repaired_prediction, target)

    if not raw_prediction.strip():
        category = "empty_output"
    elif not raw_sections:
        category = "missing_labels"
    elif set(repaired_sections.keys()) != set(target_sections.keys()):
        category = "missing_state_block"
    elif field_metrics["field_accuracy"] == 0.0:
        category = "wrong_state_transition"
    elif field_metrics["field_accuracy"] < 1.0:
        category = "partial_state_match"
    else:
        category = "exact_state_match"

    return {
        "category": category,
        "field_accuracy": field_metrics["field_accuracy"],
        "raw_has_labels": sorted(raw_sections.keys()),
        "repaired_has_labels": sorted(repaired_sections.keys()),
    }


def collect_diagnostics(
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
    category_counts: dict[str, int] = {}

    for example in examples:
        raw_prediction = _generate(tokenizer, active_model, example.prompt, max_new_tokens, device)
        repaired_prediction = repair_prediction_for_stage("next_2_steps", raw_prediction)
        classified = classify_prediction(raw_prediction, repaired_prediction, example.target)
        category = classified["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
        records.append(
            {
                "program_name": example.program_name,
                "split_family": example.split_family,
                "selection_key": example.selection_key,
                "prompt": example.prompt,
                "target": example.target,
                "raw_prediction": raw_prediction,
                "repaired_prediction": repaired_prediction,
                **classified,
            }
        )

    return {
        "model_label": model_label,
        "base_model_name": base_model_name,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "split": split,
        "limit": limit,
        "category_counts": category_counts,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect per-example next_2_steps diagnostics for base or finetuned checkpoints.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--num-predict", type=int, default=192)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    result = collect_diagnostics(
        base_model_name=args.base_model,
        adapter_path=adapter_path,
        dataset_root=Path(args.dataset_root),
        split=args.split,
        limit=args.limit,
        max_new_tokens=args.num_predict,
        trust_remote_code=args.trust_remote_code,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
