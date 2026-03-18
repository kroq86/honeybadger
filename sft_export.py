from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_STAGE_ORDER = ["single_step", "next_2_steps", "short_trace", "terminal_state"]


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_stage_split(dataset_root: Path, stage: str, split: str) -> List[dict]:
    return read_jsonl(dataset_root / stage / f"{split}.jsonl")


def to_sft_record(source_record: dict, split: str) -> dict:
    return {
        "prompt": source_record["prompt"],
        "completion": source_record["target"],
        "metadata": {
            "dataset_type": source_record["dataset_type"],
            "program_name": source_record["program_name"],
            "split_family": source_record.get("split_family"),
            "category": source_record.get("category"),
            "split": split,
            "input_values": source_record.get("input_values"),
            "instruction": source_record.get("instruction"),
            "num_steps": source_record.get("num_steps"),
        },
    }


def export_sft_splits(dataset_root: Path, stages: List[str]) -> Dict[str, List[dict]]:
    exported = {"train": [], "val": [], "test": []}
    for stage in stages:
        for split in ["train", "val", "test"]:
            for record in load_stage_split(dataset_root, stage, split):
                exported[split].append(to_sft_record(record, split))
    return exported


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_manifest(output_dir: Path, dataset_root: Path, stages: List[str], splits: Dict[str, List[dict]]) -> dict:
    return {
        "output_dir": str(output_dir),
        "source_dataset_root": str(dataset_root),
        "format": "plain_prompt_completion_v1",
        "stages": stages,
        "split_counts": {split: len(records) for split, records in splits.items()},
        "prompt_drift_policy": "reuse_eval_prompts_exactly",
        "completion_policy": "reuse_eval_targets_exactly",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the execution benchmark into an SFT-ready prompt/completion format.")
    parser.add_argument("--dataset-root", default="datasets/mvp")
    parser.add_argument("--output-dir", default="training_data/sft_v1")
    parser.add_argument("--stages", nargs="*", default=DEFAULT_STAGE_ORDER)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    splits = export_sft_splits(dataset_root, args.stages)
    for split_name, records in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", records)
    manifest = build_manifest(output_dir, dataset_root, args.stages, splits)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
