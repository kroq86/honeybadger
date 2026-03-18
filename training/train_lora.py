from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_STACK = ["torch", "transformers", "peft", "accelerate", "datasets"]
DEFAULT_FALLBACK_TARGET_MODULE_SETS = [
    ["q_proj", "k_proj", "v_proj", "o_proj"],
    ["qkv_proj", "o_proj"],
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_missing_modules() -> list[str]:
    missing: list[str] = []
    for module_name in REQUIRED_STACK:
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    return missing


def validate_config(config: dict) -> None:
    required = [
        "model_name",
        "dataset_path",
        "output_dir",
        "training_mode",
        "trust_remote_code",
        "max_seq_length",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "num_train_epochs",
        "max_steps",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
    ]
    for key in required:
        if key not in config:
            raise ValueError(f"missing required training config key: {key}")
    if config["training_mode"] != "lora":
        raise ValueError("first Phase 2 training mode must stay on lora")


def build_plan(config: dict) -> dict:
    return {
        "model_name": config["model_name"],
        "dataset_path": config["dataset_path"],
        "output_dir": config["output_dir"],
        "training_mode": config["training_mode"],
        "trust_remote_code": config["trust_remote_code"],
        "max_seq_length": config["max_seq_length"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "effective_batch_hint": config["per_device_train_batch_size"] * config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "num_train_epochs": config["num_train_epochs"],
        "max_steps": config["max_steps"],
        "max_train_samples": config.get("max_train_samples"),
        "lora": {
            "r": config["lora_r"],
            "alpha": config["lora_alpha"],
            "dropout": config["lora_dropout"],
            "target_modules": config.get("lora_target_modules", []),
        },
        "guardrails": config.get("guardrails", {}),
    }


def _load_training_modules() -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_train_rows(config: dict) -> list[dict]:
    rows = _read_jsonl(Path(config["dataset_path"]))
    max_train_samples = config.get("max_train_samples")
    if isinstance(max_train_samples, int) and max_train_samples > 0:
        rows = rows[:max_train_samples]
    return rows


def _pick_device(torch: Any) -> str:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _prepare_tokenizer(tokenizer: Any) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def _resolve_lora_target_modules(model: Any, requested: list[str]) -> list[str]:
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    if requested and all(name in module_names for name in requested):
        return requested
    for candidate_set in DEFAULT_FALLBACK_TARGET_MODULE_SETS:
        if all(name in module_names for name in candidate_set):
            return candidate_set
    linear_suffixes = sorted(
        {
            name.split(".")[-1]
            for name, module in model.named_modules()
            if module.__class__.__name__ == "Linear"
        }
    )
    preferred = [name for name in linear_suffixes if any(token in name for token in ("proj", "gate", "down", "up"))]
    if preferred:
        return preferred[:8]
    raise ValueError("could not resolve LoRA target modules for the current model")


def _encode_example(tokenizer: Any, prompt: str, completion: str, max_seq_length: int) -> dict[str, Any]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_text = f"{prompt}\n{completion}"
    full_ids = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length,
    )["input_ids"]
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = list(full_ids)
    for index in range(prompt_len):
        labels[index] = -100
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


class CompletionOnlyCollator:
    def __init__(self, tokenizer: Any, torch: Any) -> None:
        self.tokenizer = tokenizer
        self.torch = torch

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            batch_labels.append(feature["labels"] + [-100] * pad_len)
        return {
            "input_ids": self.torch.tensor(batch_input_ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(batch_attention_mask, dtype=self.torch.long),
            "labels": self.torch.tensor(batch_labels, dtype=self.torch.long),
        }


def run_training(config: dict) -> dict[str, Any]:
    modules = _load_training_modules()
    torch = modules["torch"]
    Dataset = modules["Dataset"]
    LoraConfig = modules["LoraConfig"]
    get_peft_model = modules["get_peft_model"]
    AutoModelForCausalLM = modules["AutoModelForCausalLM"]
    AutoTokenizer = modules["AutoTokenizer"]
    Trainer = modules["Trainer"]
    TrainingArguments = modules["TrainingArguments"]

    rows = _build_train_rows(config)
    if not rows:
        raise ValueError("training dataset is empty")

    device = _pick_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        use_fast=True,
        trust_remote_code=config["trust_remote_code"],
    )
    _prepare_tokenizer(tokenizer)

    encoded_rows = [
        _encode_example(
            tokenizer=tokenizer,
            prompt=row["prompt"],
            completion=row["completion"],
            max_seq_length=config["max_seq_length"],
        )
        for row in rows
    ]
    dataset = Dataset.from_list(encoded_rows)

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        trust_remote_code=config["trust_remote_code"],
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    resolved_target_modules = _resolve_lora_target_modules(model, config.get("lora_target_modules", []))

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=resolved_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"],
        logging_steps=1,
        save_strategy=config.get("guardrails", {}).get("save_strategy", "epoch"),
        save_total_limit=config.get("guardrails", {}).get("max_checkpoints_to_keep", 2),
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=False,
        bf16=False,
        use_cpu=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CompletionOnlyCollator(tokenizer, torch),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "status": "completed",
        "device": device,
        "train_rows": len(rows),
        "output_dir": str(output_dir),
        "max_steps": config["max_steps"],
        "max_train_samples": config.get("max_train_samples"),
        "resolved_target_modules": resolved_target_modules,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 LoRA training entrypoint.")
    default_config_path = Path(__file__).resolve().parent / "configs" / "lora_smoke.json"
    parser.add_argument("--config", default=str(default_config_path))
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_json(config_path)
    validate_config(config)
    plan = build_plan(config)
    missing = detect_missing_modules()

    payload = {
        "config_path": str(config_path),
        "dry_run": args.dry_run,
        "missing_modules": missing,
        "plan": plan,
    }

    if missing:
        raise SystemExit(json.dumps(payload, ensure_ascii=True, indent=2))

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    summary = run_training(config)
    print(json.dumps({"config_path": str(config_path), "summary": summary}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
