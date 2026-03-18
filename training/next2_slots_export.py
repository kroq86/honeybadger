from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "datasets" / "mvp" / "next_2_steps"
DEFAULT_TRAINING_OUTPUT_DIR = REPO_ROOT / "training_data" / "sft_next2_slots_v1"
DEFAULT_BENCHMARK_OUTPUT_ROOT = REPO_ROOT / "datasets" / "next2_slots_benchmark_v1"

if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))
    from training.next2_delta_export import parse_next2_target, write_jsonl
else:
    from training.next2_delta_export import parse_next2_target, write_jsonl


def read_benchmark_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def slot_lines_for_state(label: str, state: dict) -> list[str]:
    registers = state.get("registers", {})
    return [
        f"{label}_IP={state.get('IP', 0)}",
        f"{label}_REG=" + " ".join(f"R{i}={registers.get(f'R{i}', 0)}" for i in range(8)),
        f"{label}_FLAGS=Z={state.get('Z', 0)} N={state.get('N', 0)} C={state.get('C', 0)}",
        f"{label}_OUT={state.get('OUT[0]', '_')}",
        f"{label}_HALTED={state.get('HALTED', 0)}",
        f"{label}_ERROR={state.get('ERROR', 'NONE')}",
    ]


def build_slot_prompt(prompt: str) -> str:
    return prompt.replace(
        "TASK: next_2_steps_execution\nEmit S1 and S2 only.",
        (
            "TASK: next_2_steps_slots\n"
            "Emit only these labels in this exact order: "
            "S1_IP, S1_REG, S1_FLAGS, S1_OUT, S1_HALTED, S1_ERROR, "
            "S2_IP, S2_REG, S2_FLAGS, S2_OUT, S2_HALTED, S2_ERROR.\n"
            "Do not emit instruction text. Do not emit S1 or S2 blocks."
        ),
    )


def build_slot_target(target: str) -> str:
    sections = parse_next2_target(target)
    return "\n".join(slot_lines_for_state("S1", sections["S1"]) + slot_lines_for_state("S2", sections["S2"]))


def convert_benchmark_record(record: dict) -> dict:
    return {
        **record,
        "dataset_type": "next_2_steps_slots",
        "prompt": build_slot_prompt(record["prompt"]),
        "target": build_slot_target(record["target"]),
    }


def convert_training_record(record: dict, split: str) -> dict:
    return {
        "prompt": build_slot_prompt(record["prompt"]),
        "completion": build_slot_target(record["target"]),
        "metadata": {
            "dataset_type": "next_2_steps_slots",
            "source_dataset_type": record["dataset_type"],
            "program_name": record["program_name"],
            "split_family": record.get("split_family"),
            "category": record.get("category"),
            "split": split,
            "input_values": record.get("input_values"),
            "instruction": record.get("instruction"),
            "num_steps": record.get("num_steps"),
            "selection_key": record.get("selection_key"),
        },
    }


def export_slots(
    source_root: Path,
    training_output_dir: Path,
    benchmark_output_root: Path,
) -> dict:
    split_counts: dict[str, int] = {}
    benchmark_stage_dir = benchmark_output_root / "next_2_steps_slots"
    for split in ("train", "val", "test"):
        source_records = read_benchmark_jsonl(source_root / f"{split}.jsonl")
        benchmark_records = [convert_benchmark_record(record) for record in source_records]
        training_records = [convert_training_record(record, split) for record in source_records]
        write_jsonl(benchmark_stage_dir / f"{split}.jsonl", benchmark_records)
        write_jsonl(training_output_dir / f"{split}.jsonl", training_records)
        split_counts[split] = len(source_records)
    manifest = {
        "format": "plain_prompt_completion_v1",
        "dataset_type": "next_2_steps_slots",
        "supervision_shape": "grammar_locked_labeled_slots",
        "source_root": str(source_root),
        "training_output_dir": str(training_output_dir),
        "benchmark_output_root": str(benchmark_output_root),
        "split_counts": split_counts,
    }
    (training_output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    benchmark_manifest = {
        "dataset_types": ["next_2_steps_slots"],
        "split_counts": {
            "next_2_steps_slots": split_counts,
        },
    }
    benchmark_output_root.mkdir(parents=True, exist_ok=True)
    (benchmark_output_root / "manifest.json").write_text(json.dumps(benchmark_manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export next_2_steps into a grammar-locked slot format.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--training-output-dir", default=str(DEFAULT_TRAINING_OUTPUT_DIR))
    parser.add_argument("--benchmark-output-root", default=str(DEFAULT_BENCHMARK_OUTPUT_ROOT))
    args = parser.parse_args()

    manifest = export_slots(
        source_root=Path(args.source_root),
        training_output_dir=Path(args.training_output_dir),
        benchmark_output_root=Path(args.benchmark_output_root),
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
