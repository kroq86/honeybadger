from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "datasets" / "mvp" / "next_2_steps"
DEFAULT_TRAINING_OUTPUT_DIR = REPO_ROOT / "training_data" / "sft_next2_effects_v1"
DEFAULT_BENCHMARK_OUTPUT_ROOT = REPO_ROOT / "datasets" / "next2_effects_benchmark_v1"

if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))
    from training.next2_delta_export import parse_next2_target, parse_prompt_s0, write_jsonl
else:
    from training.next2_delta_export import parse_next2_target, parse_prompt_s0, write_jsonl


def read_benchmark_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_prompt_instructions(prompt: str) -> dict[str, str]:
    lines = prompt.strip().splitlines()
    parsed: dict[str, str] = {}
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped in {"E1", "E2"}:
            current = stripped
            continue
        if current and stripped:
            parsed[current] = stripped
            current = None
    return parsed


def parse_instruction(instr: str) -> tuple[str, list[str]]:
    opcode, *rest = instr.split(None, 1)
    operands = []
    if rest:
        operands = [part.strip() for part in rest[0].split(",")]
    return opcode.strip(), operands


def effect_lines(label: str, instruction: str, after_state: dict) -> list[str]:
    opcode, operands = parse_instruction(instruction)
    dst = operands[0] if operands else "NONE"
    src = operands[1] if len(operands) > 1 else "NONE"
    value = "NONE"
    if opcode == "READ" and dst.startswith("R"):
        value = str(after_state.get("registers", {}).get(dst, 0))
    elif dst.startswith("R"):
        value = str(after_state.get("registers", {}).get(dst, 0))
    elif dst.startswith("output["):
        value = str(after_state.get("OUT[0]", "_"))
    flags = f"Z={after_state.get('Z', 0)} N={after_state.get('N', 0)} C={after_state.get('C', 0)}"
    return [
        f"{label}_OP={opcode}",
        f"{label}_DST={dst}",
        f"{label}_SRC={src}",
        f"{label}_VALUE={value}",
        f"{label}_IP={after_state.get('IP', 0)}",
        f"{label}_FLAGS={flags}",
        f"{label}_OUT={after_state.get('OUT[0]', '_')}",
        f"{label}_HALTED={after_state.get('HALTED', 0)}",
        f"{label}_ERROR={after_state.get('ERROR', 'NONE')}",
    ]


def build_effect_prompt(prompt: str) -> str:
    return prompt.replace(
        "TASK: next_2_steps_execution\nEmit S1 and S2 only.",
        (
            "TASK: next_2_effects\n"
            "Emit only these labels in this exact order: "
            "STEP1_OP, STEP1_DST, STEP1_SRC, STEP1_VALUE, STEP1_IP, STEP1_FLAGS, STEP1_OUT, STEP1_HALTED, STEP1_ERROR, "
            "STEP2_OP, STEP2_DST, STEP2_SRC, STEP2_VALUE, STEP2_IP, STEP2_FLAGS, STEP2_OUT, STEP2_HALTED, STEP2_ERROR.\n"
            "Describe the effect of each step, not the whole machine state."
        ),
    )


def build_effect_target(prompt: str, target: str) -> str:
    instructions = parse_prompt_instructions(prompt)
    target_sections = parse_next2_target(target)
    lines = effect_lines("STEP1", instructions["E1"], target_sections["S1"])
    lines.extend(effect_lines("STEP2", instructions["E2"], target_sections["S2"]))
    return "\n".join(lines)


def convert_benchmark_record(record: dict) -> dict:
    return {
        **record,
        "dataset_type": "next_2_effects",
        "prompt": build_effect_prompt(record["prompt"]),
        "target": build_effect_target(record["prompt"], record["target"]),
    }


def convert_training_record(record: dict, split: str) -> dict:
    return {
        "prompt": build_effect_prompt(record["prompt"]),
        "completion": build_effect_target(record["prompt"], record["target"]),
        "metadata": {
            "dataset_type": "next_2_effects",
            "source_dataset_type": record["dataset_type"],
            "program_name": record["program_name"],
            "split_family": record.get("split_family"),
            "category": record.get("category"),
            "split": split,
            "input_values": record.get("input_values"),
            "instruction": None,
            "num_steps": record.get("num_steps"),
            "selection_key": record.get("selection_key"),
        },
    }


def export_effects(source_root: Path, training_output_dir: Path, benchmark_output_root: Path) -> dict:
    split_counts: dict[str, int] = {}
    benchmark_stage_dir = benchmark_output_root / "next_2_effects"
    for split in ("train", "val", "test"):
        source_records = read_benchmark_jsonl(source_root / f"{split}.jsonl")
        benchmark_records = [convert_benchmark_record(record) for record in source_records]
        training_records = [convert_training_record(record, split) for record in source_records]
        write_jsonl(benchmark_stage_dir / f"{split}.jsonl", benchmark_records)
        write_jsonl(training_output_dir / f"{split}.jsonl", training_records)
        split_counts[split] = len(source_records)
    manifest = {
        "format": "plain_prompt_completion_v1",
        "dataset_type": "next_2_effects",
        "supervision_shape": "operator_effect_representation",
        "source_root": str(source_root),
        "training_output_dir": str(training_output_dir),
        "benchmark_output_root": str(benchmark_output_root),
        "split_counts": split_counts,
        "stages": ["next_2_effects"],
        "prompt_drift_policy": "reuse_eval_prompts_exactly",
        "completion_policy": "reuse_eval_targets_exactly",
    }
    (training_output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    benchmark_output_root.mkdir(parents=True, exist_ok=True)
    (benchmark_output_root / "manifest.json").write_text(
        json.dumps({"dataset_types": ["next_2_effects"], "split_counts": {"next_2_effects": split_counts}}, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export next_2_steps into an effect representation format.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--training-output-dir", default=str(DEFAULT_TRAINING_OUTPUT_DIR))
    parser.add_argument("--benchmark-output-root", default=str(DEFAULT_BENCHMARK_OUTPUT_ROOT))
    args = parser.parse_args()

    manifest = export_effects(
        source_root=Path(args.source_root),
        training_output_dir=Path(args.training_output_dir),
        benchmark_output_root=Path(args.benchmark_output_root),
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
