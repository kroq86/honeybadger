from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "training_data" / "sft_next2_biased_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training_data" / "sft_next2_factorized_v1"

if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))
    from training.next2_delta_export import delta_lines, parse_next2_target, parse_prompt_s0, read_jsonl, write_jsonl
else:
    from training.next2_delta_export import delta_lines, parse_next2_target, parse_prompt_s0, read_jsonl, write_jsonl


def build_step1_prompt(prompt: str) -> str:
    return prompt.replace(
        "TASK: next_2_steps_execution\nEmit S1 and S2 only.",
        "TASK: next_2_step_factorized\nEmit D1 only. Report only changed fields after E1.",
    )


def build_step2_prompt(prompt: str, d1_completion: str) -> str:
    return (
        prompt.replace(
            "TASK: next_2_steps_execution\nEmit S1 and S2 only.",
            "TASK: next_2_step_factorized\nEmit D2 only. D1 is already known. Report only changed fields after E2.",
        )
        + "\nKNOWN_D1\n"
        + d1_completion.strip()
    )


def to_factorized_records(source_record: dict) -> list[dict]:
    metadata = source_record["metadata"]
    s0 = parse_prompt_s0(source_record["prompt"])
    target_sections = parse_next2_target(source_record["completion"])
    s1 = target_sections["S1"]
    s2 = target_sections["S2"]
    d1 = "\n".join(delta_lines(s0, s1, "D1"))
    d2 = "\n".join(delta_lines(s1, s2, "D2"))

    common_metadata = {
        "source_dataset_type": metadata.get("dataset_type"),
        "program_name": metadata["program_name"],
        "split_family": metadata.get("split_family"),
        "category": metadata.get("category"),
        "split": metadata["split"],
        "input_values": metadata.get("input_values"),
        "instruction": metadata.get("instruction"),
        "num_steps": metadata.get("num_steps"),
        "selection_key": metadata.get("selection_key"),
    }

    return [
        {
            "prompt": build_step1_prompt(source_record["prompt"]),
            "completion": d1,
            "metadata": {
                **common_metadata,
                "dataset_type": "next_2_step_factorized",
                "factorized_step": "D1",
            },
        },
        {
            "prompt": build_step2_prompt(source_record["prompt"], d1),
            "completion": d2,
            "metadata": {
                **common_metadata,
                "dataset_type": "next_2_step_factorized",
                "factorized_step": "D2",
            },
        },
    ]


def export_factorized_splits(source_root: Path) -> dict[str, list[dict]]:
    exported: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        records = read_jsonl(source_root / f"{split}.jsonl")
        next2_records = [r for r in records if r.get("metadata", {}).get("dataset_type") == "next_2_steps"]
        expanded: list[dict] = []
        for record in next2_records:
            expanded.extend(to_factorized_records(record))
        exported[split] = expanded
    return exported


def build_manifest(output_dir: Path, source_root: Path, exported: dict[str, list[dict]]) -> dict:
    return {
        "output_dir": str(output_dir),
        "source_root": str(source_root),
        "format": "plain_prompt_completion_v1",
        "dataset_type": "next_2_step_factorized",
        "split_counts": {split: len(records) for split, records in exported.items()},
        "supervision_shape": "teacher_forced_d1_d2",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export next_2_steps into a factorized teacher-forced supervision format.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    exported = export_factorized_splits(source_root)
    for split, records in exported.items():
        write_jsonl(output_dir / f"{split}.jsonl", records)
    manifest = build_manifest(output_dir, source_root, exported)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
