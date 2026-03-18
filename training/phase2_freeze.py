from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.train_lora import REQUIRED_STACK
else:
    from training.train_lora import REQUIRED_STACK

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "training" / "phase2_freeze.json"
DEFAULT_LORA_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "lora_smoke.json"
DEFAULT_EVAL_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "eval_phase2.json"
DEFAULT_EXECUTION_MANIFEST_PATH = REPO_ROOT / "datasets" / "mvp" / "manifest.json"
DEFAULT_SFT_MANIFEST_PATH = REPO_ROOT / "training_data" / "sft_v1" / "manifest.json"
DEFAULT_REQUIREMENTS_PATH = REPO_ROOT / "training" / "requirements-phase2.txt"
DEFAULT_DECLARED_FALLBACK_MODEL = "microsoft/Phi-3-mini-4k-instruct"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_phase2_freeze(
    lora_config_path: Path = DEFAULT_LORA_CONFIG_PATH,
    eval_config_path: Path = DEFAULT_EVAL_CONFIG_PATH,
    execution_manifest_path: Path = DEFAULT_EXECUTION_MANIFEST_PATH,
    sft_manifest_path: Path = DEFAULT_SFT_MANIFEST_PATH,
    requirements_path: Path = DEFAULT_REQUIREMENTS_PATH,
) -> dict:
    lora_config = load_json(lora_config_path)
    eval_config = load_json(eval_config_path)
    execution_manifest = load_json(execution_manifest_path)
    sft_manifest = load_json(sft_manifest_path)

    return {
        "phase": "phase2",
        "status": "scaffold_frozen",
        "selected_model": {
            "primary": lora_config["model_name"],
            "fallback": None if lora_config["model_name"] == DEFAULT_DECLARED_FALLBACK_MODEL else DEFAULT_DECLARED_FALLBACK_MODEL,
            "training_mode": lora_config["training_mode"],
            "trust_remote_code": lora_config["trust_remote_code"],
        },
        "paths": {
            "lora_config": str(lora_config_path),
            "eval_config": str(eval_config_path),
            "execution_manifest": str(execution_manifest_path),
            "sft_manifest": str(sft_manifest_path),
            "requirements_path": str(requirements_path),
            "train_dataset_path": lora_config["dataset_path"],
            "output_dir": lora_config["output_dir"],
            "benchmark_dataset_root": eval_config["benchmark_dataset_root"],
        },
        "execution_benchmark": {
            "stages": eval_config["benchmark_stages"],
            "metrics": eval_config["metrics"],
            "runner_defaults": eval_config["runner_defaults"],
            "dataset_types": execution_manifest["dataset_types"],
            "split_counts": execution_manifest["split_counts"],
        },
        "sft_export": {
            "format": sft_manifest["format"],
            "stages": sft_manifest["stages"],
            "split_counts": sft_manifest["split_counts"],
            "prompt_drift_policy": sft_manifest["prompt_drift_policy"],
            "completion_policy": sft_manifest["completion_policy"],
        },
        "guardrails": {
            "required_training_stack": REQUIRED_STACK,
            "lora": {
                "max_seq_length": lora_config["max_seq_length"],
                "per_device_train_batch_size": lora_config["per_device_train_batch_size"],
                "gradient_accumulation_steps": lora_config["gradient_accumulation_steps"],
                "learning_rate": lora_config["learning_rate"],
                "num_train_epochs": lora_config["num_train_epochs"],
                "lora_r": lora_config["lora_r"],
                "lora_alpha": lora_config["lora_alpha"],
                "lora_dropout": lora_config["lora_dropout"],
                "lora_target_modules": lora_config.get("lora_target_modules", []),
            },
            "run_guardrails": lora_config.get("guardrails", {}),
        },
    }


def write_freeze(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the current Phase 2 training scaffold into one machine-readable manifest.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--lora-config", default=str(DEFAULT_LORA_CONFIG_PATH))
    parser.add_argument("--eval-config", default=str(DEFAULT_EVAL_CONFIG_PATH))
    parser.add_argument("--execution-manifest", default=str(DEFAULT_EXECUTION_MANIFEST_PATH))
    parser.add_argument("--sft-manifest", default=str(DEFAULT_SFT_MANIFEST_PATH))
    parser.add_argument("--requirements", default=str(DEFAULT_REQUIREMENTS_PATH))
    args = parser.parse_args()

    payload = build_phase2_freeze(
        lora_config_path=Path(args.lora_config),
        eval_config_path=Path(args.eval_config),
        execution_manifest_path=Path(args.execution_manifest),
        sft_manifest_path=Path(args.sft_manifest),
        requirements_path=Path(args.requirements),
    )
    write_freeze(Path(args.output), payload)
    print(json.dumps({"output_path": args.output, "status": payload["status"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
