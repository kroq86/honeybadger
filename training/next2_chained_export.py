from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "datasets" / "mvp" / "next_2_steps"
DEFAULT_TRAINING_OUTPUT_DIR = REPO_ROOT / "training_data" / "sft_next2_chained_v1"
DEFAULT_BENCHMARK_OUTPUT_ROOT = REPO_ROOT / "datasets" / "next2_chained_benchmark_v1"

if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))
    from training.next2_delta_export import write_jsonl
    from training.next2_chained_probe import build_step1_prompt, build_step2_prompt, split_prompt, split_target
else:
    from training.next2_delta_export import write_jsonl
    from training.next2_chained_probe import build_step1_prompt, build_step2_prompt, split_prompt, split_target


def read_benchmark_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def step1_record(source_record: dict, split: str) -> tuple[dict, dict]:
    prefix, e1, _ = split_prompt(source_record["prompt"])
    s1, _ = split_target(source_record["target"])
    benchmark_record = {
        **source_record,
        "dataset_type": "next_2_chained_step1",
        "prompt": build_step1_prompt(prefix, e1),
        "target": s1,
    }
    training_record = {
        "prompt": benchmark_record["prompt"],
        "completion": benchmark_record["target"],
        "metadata": {
            "dataset_type": "next_2_chained_step1",
            "source_dataset_type": source_record["dataset_type"],
            "program_name": source_record["program_name"],
            "split_family": source_record.get("split_family"),
            "category": source_record.get("category"),
            "split": split,
            "input_values": source_record.get("input_values"),
            "instruction": None,
            "num_steps": 1,
            "selection_key": source_record.get("selection_key"),
        },
    }
    return benchmark_record, training_record


def step2_record(source_record: dict, split: str) -> tuple[dict, dict]:
    prefix, _, e2 = split_prompt(source_record["prompt"])
    s1, s2 = split_target(source_record["target"])
    benchmark_record = {
        **source_record,
        "dataset_type": "next_2_chained_step2",
        "prompt": build_step2_prompt(prefix, s1, e2),
        "target": s2,
    }
    training_record = {
        "prompt": benchmark_record["prompt"],
        "completion": benchmark_record["target"],
        "metadata": {
            "dataset_type": "next_2_chained_step2",
            "source_dataset_type": source_record["dataset_type"],
            "program_name": source_record["program_name"],
            "split_family": source_record.get("split_family"),
            "category": source_record.get("category"),
            "split": split,
            "input_values": source_record.get("input_values"),
            "instruction": None,
            "num_steps": 1,
            "selection_key": source_record.get("selection_key"),
        },
    }
    return benchmark_record, training_record


def export_chained(source_root: Path, training_output_dir: Path, benchmark_output_root: Path) -> dict:
    split_counts: dict[str, int] = {}
    benchmark_stage_step1_dir = benchmark_output_root / "next_2_chained_step1"
    benchmark_stage_step2_dir = benchmark_output_root / "next_2_chained_step2"
    for split in ("train", "val", "test"):
        source_records = read_benchmark_jsonl(source_root / f"{split}.jsonl")
        step1_benchmark_records: list[dict] = []
        step2_benchmark_records: list[dict] = []
        training_records: list[dict] = []
        for record in source_records:
            step1_benchmark, step1_training = step1_record(record, split)
            step2_benchmark, step2_training = step2_record(record, split)
            step1_benchmark_records.append(step1_benchmark)
            step2_benchmark_records.append(step2_benchmark)
            training_records.append(step1_training)
            training_records.append(step2_training)
        write_jsonl(benchmark_stage_step1_dir / f"{split}.jsonl", step1_benchmark_records)
        write_jsonl(benchmark_stage_step2_dir / f"{split}.jsonl", step2_benchmark_records)
        write_jsonl(training_output_dir / f"{split}.jsonl", training_records)
        split_counts[split] = len(training_records)
    manifest = {
        "format": "plain_prompt_completion_v1",
        "dataset_type": "next_2_chained",
        "supervision_shape": "sequential_stepwise_state_supervision",
        "source_root": str(source_root),
        "training_output_dir": str(training_output_dir),
        "benchmark_output_root": str(benchmark_output_root),
        "split_counts": split_counts,
        "benchmark_stage_split_counts": {
            "next_2_chained_step1": {split: split_counts[split] // 2 for split in split_counts},
            "next_2_chained_step2": {split: split_counts[split] // 2 for split in split_counts},
        },
        "stages": ["next_2_chained_step1", "next_2_chained_step2"],
        "prompt_drift_policy": "reuse_eval_prompts_exactly",
        "completion_policy": "reuse_eval_targets_exactly",
    }
    training_output_dir.mkdir(parents=True, exist_ok=True)
    (training_output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    benchmark_output_root.mkdir(parents=True, exist_ok=True)
    (benchmark_output_root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_types": ["next_2_chained_step1", "next_2_chained_step2"],
                "split_counts": manifest["benchmark_stage_split_counts"],
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export next_2_steps into a chained stepwise supervision format.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--training-output-dir", default=str(DEFAULT_TRAINING_OUTPUT_DIR))
    parser.add_argument("--benchmark-output-root", default=str(DEFAULT_BENCHMARK_OUTPUT_ROOT))
    args = parser.parse_args()

    manifest = export_chained(
        source_root=Path(args.source_root),
        training_output_dir=Path(args.training_output_dir),
        benchmark_output_root=Path(args.benchmark_output_root),
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
