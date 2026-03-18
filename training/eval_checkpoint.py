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
        exact_match,
        load_split,
        next_2_field_metrics,
        next_2_effect_field_metrics,
        next_2_slots_field_metrics,
        repair_prediction_for_stage,
        state_field_metrics,
    )
else:
    from baseline_trainer import (
        chained_delta_metrics,
        exact_match,
        load_split,
        next_2_field_metrics,
        next_2_effect_field_metrics,
        next_2_slots_field_metrics,
        repair_prediction_for_stage,
        state_field_metrics,
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_eval_config(config: dict) -> None:
    required = [
        "phase2_freeze_path",
        "benchmark_dataset_root",
        "benchmark_stages",
        "benchmark_splits",
        "metrics",
        "runner_defaults",
        "base_report_path",
        "finetuned_report_path",
    ]
    for key in required:
        if key not in config:
            raise ValueError(f"missing required eval config key: {key}")


def _load_eval_modules() -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _pick_device(torch: Any) -> str:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _read_freeze(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model_pair(base_model_name: str, adapter_path: Path, trust_remote_code: bool) -> tuple[Any, Any, Any, str]:
    modules = _load_eval_modules()
    torch = modules["torch"]
    AutoModelForCausalLM = modules["AutoModelForCausalLM"]
    AutoTokenizer = modules["AutoTokenizer"]
    PeftModel = modules["PeftModel"]

    device = _pick_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=trust_remote_code)
    base_model.eval()
    finetuned_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    finetuned_model.eval()

    return tokenizer, base_model, finetuned_model, device


def _generate(tokenizer: Any, model: Any, prompt: str, max_new_tokens: int, device: str) -> str:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt")
    if device != "cpu":
        encoded = {key: value.to(device) for key, value in encoded.items()}
        model = model.to(device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = encoded["input_ids"].shape[1]
    completion_ids = generated[0][prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def _score_prediction(stage: str, prediction: str, target: str, prompt: str) -> dict:
    try:
        repaired = repair_prediction_for_stage(stage, prediction)
        result = {
            "exact_match": exact_match(prediction, target),
            "repaired_exact_match": exact_match(repaired, target),
        }
        if stage in {"single_step", "terminal_state"}:
            result["field_accuracy"] = state_field_metrics(repaired, target)["field_accuracy"]
        elif stage in {"next_2_chained_step1", "next_2_chained_step2"}:
            chained_metrics = chained_delta_metrics(repaired, target, prompt, stage)
            result["field_accuracy"] = chained_metrics["delta_transition_score"]
            result["delta_exact"] = chained_metrics["delta_exact"]
        elif stage == "next_2_steps":
            result["field_accuracy"] = next_2_field_metrics(repaired, target)["field_accuracy"]
        elif stage == "next_2_steps_slots":
            result["field_accuracy"] = next_2_slots_field_metrics(repaired, target)["field_accuracy"]
        elif stage in {"next_2_effects", "next_2_effects_target_anchor"}:
            result["field_accuracy"] = next_2_effect_field_metrics(repaired, target)["field_accuracy"]
        return result
    except Exception:
        result = {
            "exact_match": False,
            "repaired_exact_match": False,
        }
        if stage in {"single_step", "terminal_state", "next_2_steps", "next_2_steps_slots", "next_2_effects", "next_2_effects_target_anchor", "next_2_chained_step1", "next_2_chained_step2"}:
            result["field_accuracy"] = 0.0
        if stage in {"next_2_chained_step1", "next_2_chained_step2"}:
            result["delta_exact"] = 0.0
        return result


def _evaluate_model(
    tokenizer: Any,
    model: Any,
    device: str,
    dataset_root: Path,
    stages: list[str],
    splits: list[str],
    max_new_tokens: int,
    eval_limit: int,
) -> dict:
    result = {"stages": {}}
    for stage in stages:
        stage_scores: dict[str, list[float]] = {
            "val_exact_match_rate": [],
            "test_exact_match_rate": [],
            "val_repaired_exact_match_rate": [],
            "test_repaired_exact_match_rate": [],
            "val_avg_field_accuracy": [],
            "test_avg_field_accuracy": [],
        }
        for split in splits:
            examples = load_split(dataset_root, stage, split)[:eval_limit]
            for example in examples:
                prediction = _generate(tokenizer, model, example.prompt, max_new_tokens, device)
                scored = _score_prediction(stage, prediction, example.target, example.prompt)
                stage_scores[f"{split}_exact_match_rate"].append(1.0 if scored["exact_match"] else 0.0)
                stage_scores[f"{split}_repaired_exact_match_rate"].append(1.0 if scored["repaired_exact_match"] else 0.0)
                if "field_accuracy" in scored:
                    stage_scores[f"{split}_avg_field_accuracy"].append(float(scored["field_accuracy"]))
        result["stages"][stage] = {
            key: (sum(values) / len(values) if values else None)
            for key, values in stage_scores.items()
        }
    return result


def _build_comparison_report(base_result: dict, finetuned_result: dict) -> str:
    lines = ["# Phase 2 Comparison", ""]
    for stage, base_metrics in base_result["stages"].items():
        lines.append(f"## {stage}")
        finetuned_metrics = finetuned_result["stages"].get(stage, {})
        for metric_name, base_value in base_metrics.items():
            finetuned_value = finetuned_metrics.get(metric_name)
            lines.append(f"- {metric_name}: base={base_value} finetuned={finetuned_value}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_eval(config: dict) -> dict:
    freeze = _read_freeze(Path(config["phase2_freeze_path"]))
    base_model_name = freeze["selected_model"]["primary"]
    trust_remote_code = bool(freeze["selected_model"].get("trust_remote_code", False))
    adapter_path = Path(freeze["paths"]["output_dir"])

    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter output directory does not exist: {adapter_path}")

    tokenizer, base_model, finetuned_model, device = _load_model_pair(base_model_name, adapter_path, trust_remote_code)
    dataset_root = Path(config["benchmark_dataset_root"])
    splits = list(config["benchmark_splits"])
    max_new_tokens = int(config["runner_defaults"]["num_predict"])
    eval_limit = int(config["runner_defaults"]["eval_limit"])
    evaluate_base = bool(config.get("evaluate_base", True))
    if evaluate_base:
        base_result = _evaluate_model(
            tokenizer=tokenizer,
            model=base_model,
            device=device,
            dataset_root=dataset_root,
            stages=config["benchmark_stages"],
            splits=splits,
            max_new_tokens=max_new_tokens,
            eval_limit=eval_limit,
        )
    else:
        base_result = {"stages": {}}
    finetuned_result = _evaluate_model(
        tokenizer=tokenizer,
        model=finetuned_model,
        device=device,
        dataset_root=dataset_root,
        stages=config["benchmark_stages"],
        splits=splits,
        max_new_tokens=max_new_tokens,
        eval_limit=eval_limit,
    )
    return {
        "base_model_name": base_model_name,
        "adapter_path": str(adapter_path),
        "evaluate_base": evaluate_base,
        "benchmark_splits": splits,
        "base": base_result,
        "finetuned": finetuned_result,
        "comparison_markdown": _build_comparison_report(base_result, finetuned_result),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 checkpoint eval entrypoint.")
    default_config_path = Path(__file__).resolve().parent / "configs" / "eval_phase2.json"
    parser.add_argument("--config", default=str(default_config_path))
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_json(config_path)
    validate_eval_config(config)
    payload = {
        "config_path": str(config_path),
        "dry_run": args.dry_run,
        "config": config,
    }

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    result = run_eval(config)
    Path(config["base_report_path"]).write_text(
        json.dumps(result["base"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(config["finetuned_report_path"]).write_text(
        json.dumps(result["finetuned"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(config["comparison_report_path"]).write_text(
        result["comparison_markdown"],
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "base_report_path": config["base_report_path"],
                "finetuned_report_path": config["finetuned_report_path"],
                "comparison_report_path": config["comparison_report_path"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
