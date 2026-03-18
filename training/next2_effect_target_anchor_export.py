from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "datasets" / "mvp" / "next_2_steps"
DEFAULT_TRAINING_OUTPUT_DIR = REPO_ROOT / "training_data" / "sft_next2_effects_target_anchor_v1"
DEFAULT_BENCHMARK_OUTPUT_ROOT = REPO_ROOT / "datasets" / "next2_effects_target_anchor_benchmark_v1"

if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))
    from training.next2_effect_export import read_benchmark_jsonl, convert_benchmark_record, write_jsonl
else:
    from training.next2_effect_export import read_benchmark_jsonl, convert_benchmark_record, write_jsonl


ANCHOR_PREFIX = "TARGET\nSTEP1_OP="


def _anchored_prompt(prompt: str) -> str:
    return f"{prompt.rstrip()}\n{ANCHOR_PREFIX}"


def _anchored_target(target: str) -> str:
    prefix = "STEP1_OP="
    if target.startswith(prefix):
        return target[len(prefix):]
    return target


def convert_benchmark_record_to_anchor(record: dict) -> dict:
    base = convert_benchmark_record(record)
    return {
        **base,
        "dataset_type": "next_2_effects_target_anchor",
        "prompt": _anchored_prompt(base["prompt"]),
        "target": _anchored_target(base["target"]),
    }


def convert_training_record_to_anchor(record: dict, split: str) -> dict:
    base = convert_benchmark_record(record)
    return {
        "prompt": _anchored_prompt(base["prompt"]),
        "completion": _anchored_target(base["target"]),
        "metadata": {
            "dataset_type": "next_2_effects_target_anchor",
            "source_dataset_type": record["dataset_type"],
            "program_name": record["program_name"],
            "split_family": record.get("split_family"),
            "category": record.get("category"),
            "split": split,
            "input_values": record.get("input_values"),
            "instruction": None,
            "num_steps": record.get("num_steps"),
            "selection_key": record.get("selection_key"),
            "answer_onset_contract": ANCHOR_PREFIX,
        },
    }


def export_target_anchor(source_root: Path, training_output_dir: Path, benchmark_output_root: Path) -> dict:
    split_counts: dict[str, int] = {}
    benchmark_stage_dir = benchmark_output_root / "next_2_effects_target_anchor"
    for split in ("train", "val", "test"):
        source_records = read_benchmark_jsonl(source_root / f"{split}.jsonl")
        benchmark_records = [convert_benchmark_record_to_anchor(record) for record in source_records]
        training_records = [convert_training_record_to_anchor(record, split) for record in source_records]
        write_jsonl(benchmark_stage_dir / f"{split}.jsonl", benchmark_records)
        write_jsonl(training_output_dir / f"{split}.jsonl", training_records)
        split_counts[split] = len(source_records)
    manifest = {
        "format": "plain_prompt_completion_v1",
        "dataset_type": "next_2_effects_target_anchor",
        "supervision_shape": "operator_effect_representation_target_anchor",
        "answer_onset_contract": ANCHOR_PREFIX,
        "source_root": str(source_root),
        "training_output_dir": str(training_output_dir),
        "benchmark_output_root": str(benchmark_output_root),
        "split_counts": split_counts,
        "stages": ["next_2_effects_target_anchor"],
        "prompt_drift_policy": "reuse_eval_prompts_exactly_with_target_anchor",
        "completion_policy": "completion_continues_after_STEP1_OP_prefix",
    }
    training_output_dir.mkdir(parents=True, exist_ok=True)
    (training_output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    benchmark_output_root.mkdir(parents=True, exist_ok=True)
    (benchmark_output_root / "manifest.json").write_text(
        json.dumps({"dataset_types": ["next_2_effects_target_anchor"], "split_counts": {"next_2_effects_target_anchor": split_counts}}, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export next_2_effects into TARGET-anchored effect format.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--training-output-dir", default=str(DEFAULT_TRAINING_OUTPUT_DIR))
    parser.add_argument("--benchmark-output-root", default=str(DEFAULT_BENCHMARK_OUTPUT_ROOT))
    args = parser.parse_args()

    manifest = export_target_anchor(
        source_root=Path(args.source_root),
        training_output_dir=Path(args.training_output_dir),
        benchmark_output_root=Path(args.benchmark_output_root),
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
