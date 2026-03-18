from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from baseline_trainer import load_split
else:
    from baseline_trainer import load_split


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
    if device != "cpu":
        model = model.to(device)
    return tokenizer, model, device


def run_probe(base_model_name: str, dataset_root: Path, stage: str, split: str, limit: int, trust_remote_code: bool) -> dict[str, Any]:
    import torch

    tokenizer, model, device = _load_base_model(base_model_name, trust_remote_code)
    examples = load_split(dataset_root, stage, split)[:limit]
    summaries: list[dict[str, Any]] = []

    for example in examples:
        encoded = tokenizer(example.prompt, return_tensors="pt")
        if device != "cpu":
            encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, k=20)
        tokens = []
        for token_id, score in zip(topk.indices.tolist(), topk.values.tolist()):
            text = tokenizer.decode([token_id])
            tokens.append({"token_id": token_id, "token": text, "prob": float(score)})
        summaries.append(
            {
                "program_name": example.program_name,
                "prompt": example.prompt,
                "target": example.target,
                "top_tokens": tokens,
            }
        )

    return {
        "base_model_name": base_model_name,
        "dataset_root": str(dataset_root),
        "stage": stage,
        "split": split,
        "limit": limit,
        "records": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect first-token continuation mode for a benchmark stage.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = run_probe(
        base_model_name=args.base_model,
        dataset_root=Path(args.dataset_root),
        stage=args.stage,
        split=args.split,
        limit=args.limit,
        trust_remote_code=args.trust_remote_code,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
