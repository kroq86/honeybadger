from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from reference_vm import collect_transitions, run_program, wrap16


ACTIVE_DATASET_TYPES = ["single_step", "next_2_steps", "short_trace", "terminal_state"]


@dataclass(frozen=True)
class TaskProgram:
    program_name: str
    split_family: str
    category: str
    source: str
    input_cases: List[Dict[int, int]]


TASK_LIBRARY: List[TaskProgram] = [
    TaskProgram(
        program_name="add_two_numbers",
        split_family="add_two_numbers_family",
        category="arithmetic",
        source="""
READ R1, input[0]
READ R2, input[1]
ADD R3, R1, R2
WRITE output[0], R3
HALT
""".strip(),
        input_cases=[
            {0: 1, 1: 2},
            {0: 7, 1: 5},
            {0: -3, 1: 9},
            {0: 32767, 1: 1},
        ],
    ),
    TaskProgram(
        program_name="store_then_load",
        split_family="store_then_load_family",
        category="memory",
        source="""
CONST R1, 42
STORE [10], R1
LOAD R2, [10]
WRITE output[0], R2
HALT
""".strip(),
        input_cases=[{}],
    ),
    TaskProgram(
        program_name="branch_on_equality",
        split_family="branch_on_equality_family",
        category="branch",
        source="""
READ R1, input[0]
READ R2, input[1]
CMP R1, R2
JZ MATCH
CONST R3, 0
WRITE output[0], R3
HALT
MATCH:
CONST R3, 1
WRITE output[0], R3
HALT
""".strip(),
        input_cases=[
            {0: 5, 1: 5},
            {0: 2, 1: 9},
        ],
    ),
    TaskProgram(
        program_name="branch_non_equal_flag",
        split_family="branch_non_equal_flag_family",
        category="branch",
        source="""
READ R1, input[0]
READ R2, input[1]
CMP R1, R2
JNZ DIFFERENT
CONST R3, 1
WRITE output[0], R3
HALT
DIFFERENT:
CONST R3, -1
WRITE output[0], R3
HALT
""".strip(),
        input_cases=[
            {0: 8, 1: 3},
            {0: 5, 1: 5},
        ],
    ),
    TaskProgram(
        program_name="sum_three_constants",
        split_family="sum_three_constants_family",
        category="straight_line",
        source="""
CONST R1, 4
CONST R2, 6
ADD R3, R1, R2
CONST R4, 3
ADD R5, R3, R4
WRITE output[0], R5
HALT
""".strip(),
        input_cases=[{}],
    ),
    TaskProgram(
        program_name="countdown_to_zero",
        split_family="countdown_to_zero_family",
        category="loop",
        source="""
READ R1, input[0]
CONST R2, 0
CONST R3, 1
LOOP:
CMP R1, R2
JZ DONE
SUB R1, R1, R3
JMP LOOP
DONE:
WRITE output[0], R1
HALT
""".strip(),
        input_cases=[
            {0: 0},
            {0: 1},
            {0: 3},
            {0: 5},
        ],
    ),
    TaskProgram(
        program_name="countdown_steps",
        split_family="countdown_steps_family",
        category="loop",
        source="""
READ R1, input[0]
CONST R2, 0
CONST R3, 1
CONST R4, 0
LOOP:
CMP R1, R2
JZ DONE
SUB R1, R1, R3
ADD R4, R4, R3
JMP LOOP
DONE:
WRITE output[0], R4
HALT
""".strip(),
        input_cases=[
            {0: 0},
            {0: 2},
            {0: 4},
        ],
    ),
    TaskProgram(
        program_name="pair_sum_scan_3",
        split_family="pair_sum_scan_3_family",
        category="search",
        source="""
READ R1, input[0]
READ R2, input[1]
READ R3, input[2]
READ R4, input[3]
ADD R5, R1, R2
CMP R5, R4
JZ FOUND_01
ADD R5, R1, R3
CMP R5, R4
JZ FOUND_02
ADD R5, R2, R3
CMP R5, R4
JZ FOUND_12
CONST R6, -1
WRITE output[0], R6
HALT
FOUND_01:
CONST R6, 1
WRITE output[0], R6
HALT
FOUND_02:
CONST R6, 2
WRITE output[0], R6
HALT
FOUND_12:
CONST R6, 3
WRITE output[0], R6
HALT
""".strip(),
        input_cases=[
            {0: 2, 1: 7, 2: 11, 3: 9},
            {0: 3, 1: 2, 2: 4, 3: 6},
            {0: 1, 1: 5, 2: 9, 3: 20},
        ],
    ),
    TaskProgram(
        program_name="midpoint_search_4",
        split_family="midpoint_search_4_family",
        category="search",
        source="""
READ R1, input[2]
READ R2, input[4]
CMP R1, R2
JZ FOUND_MID
READ R3, input[0]
CMP R3, R2
JZ FOUND_LEFT
READ R4, input[1]
CMP R4, R2
JZ FOUND_LEFT_MID
READ R4, input[3]
CMP R4, R2
JZ FOUND_RIGHT
JMP NOT_FOUND
FOUND_LEFT:
CONST R5, 0
WRITE output[0], R5
HALT
FOUND_LEFT_MID:
CONST R5, 1
WRITE output[0], R5
HALT
FOUND_MID:
CONST R5, 2
WRITE output[0], R5
HALT
FOUND_RIGHT:
CONST R5, 3
WRITE output[0], R5
HALT
NOT_FOUND:
CONST R5, -1
WRITE output[0], R5
HALT
""".strip(),
        input_cases=[
            {0: 1, 1: 3, 2: 5, 3: 7, 4: 1},
            {0: 1, 1: 3, 2: 5, 3: 7, 4: 5},
            {0: 1, 1: 3, 2: 5, 3: 7, 4: 7},
            {0: 1, 1: 3, 2: 5, 3: 7, 4: 4},
        ],
    ),
    TaskProgram(
        program_name="first_match_scan_4",
        split_family="first_match_scan_4_family",
        category="search",
        source="""
READ R1, input[4]
READ R2, input[0]
CMP R2, R1
JZ FOUND_0
READ R2, input[1]
CMP R2, R1
JZ FOUND_1
READ R2, input[2]
CMP R2, R1
JZ FOUND_2
READ R2, input[3]
CMP R2, R1
JZ FOUND_3
CONST R3, -1
WRITE output[0], R3
HALT
FOUND_0:
CONST R3, 0
WRITE output[0], R3
HALT
FOUND_1:
CONST R3, 1
WRITE output[0], R3
HALT
FOUND_2:
CONST R3, 2
WRITE output[0], R3
HALT
FOUND_3:
CONST R3, 3
WRITE output[0], R3
HALT
""".strip(),
        input_cases=[
            {0: 2, 1: 7, 2: 11, 3: 15, 4: 2},
            {0: 2, 1: 7, 2: 11, 3: 15, 4: 11},
            {0: 2, 1: 7, 2: 11, 3: 15, 4: 15},
            {0: 2, 1: 7, 2: 11, 3: 15, 4: 9},
        ],
    ),
    TaskProgram(
        program_name="fib_iterative_5",
        split_family="fib_iterative_5_family",
        category="recurrence",
        source="""
READ R0, input[0]
CONST R1, 0
CONST R2, 1
CONST R3, 0
CONST R4, 1
CMP R0, R3
JZ OUT_A
LOOP:
ADD R5, R1, R2
MOV R1, R2
MOV R2, R5
ADD R3, R3, R4
CMP R3, R0
JNZ LOOP
WRITE output[0], R2
HALT
OUT_A:
WRITE output[0], R1
HALT
""".strip(),
        input_cases=[
            {0: 0},
            {0: 1},
            {0: 2},
            {0: 3},
            {0: 5},
        ],
    ),
    TaskProgram(
        program_name="weighted_sum_3",
        split_family="weighted_sum_3_family",
        category="accumulator",
        source="""
READ R1, input[0]
READ R2, input[1]
READ R3, input[2]
CONST R0, 0
ADD R7, R1, R1
SUB R7, R7, R1
ADD R0, R0, R7
ADD R7, R2, R2
ADD R0, R0, R7
ADD R7, R3, R3
SUB R7, R7, R3
ADD R0, R0, R7
WRITE output[0], R0
HALT
""".strip(),
        input_cases=[
            {0: 1, 1: 2, 2: 3},
            {0: 2, 1: 1, 2: 4},
        ],
    ),
]


def _format_inputs(inputs: Dict[int, int]) -> str:
    if not inputs:
        return "INPUT\n<none>"
    rows = ["INPUT"]
    for key in sorted(inputs):
        rows.append(f"input[{key}]={wrap16(inputs[key])}")
    return "\n".join(rows)


def _single_step_prompt(before_state: str, instruction: str) -> str:
    return "\n".join(
        [
            "TASK: single_step_execution",
            "Emit next canonical state only.",
            "STATE",
            before_state,
            f"EXEC\n{instruction}",
        ]
    )


def _short_trace_prompt(program_source: str, inputs: Dict[int, int]) -> str:
    return "\n".join(
        [
            "TASK: short_trace_execution",
            "Emit full state trace only.",
            "PROGRAM",
            program_source,
            _format_inputs(inputs),
        ]
    )


def _next_k_steps_prompt(task_name: str, program_source: str, inputs: Dict[int, int], next_k_steps: int) -> str:
    return "\n".join(
        [
            f"TASK: {task_name}",
            f"Emit first {next_k_steps} canonical trace steps only.",
            "PROGRAM",
            program_source,
            _format_inputs(inputs),
        ]
    )


def _next_2_micro_prompt(inputs: Dict[int, int], state0: str, instruction1: str, instruction2: str) -> str:
    parts = [
        "TASK: next_2_steps_execution",
        "Emit S1 and S2 only.",
        _format_inputs(inputs),
        "S0",
        state0,
        "E1",
        instruction1,
        "E2",
        instruction2,
    ]
    return "\n".join(parts)


def _next_2_micro_target(state1: str, state2: str) -> str:
    return "\n".join(
        [
            "S1",
            state1,
            "",
            "S2",
            state2,
        ]
    )


def _terminal_state_prompt(program_source: str, inputs: Dict[int, int]) -> str:
    return "\n".join(
        [
            "TASK: terminal_state_execution",
            "Emit final canonical state only.",
            "PROGRAM",
            program_source,
            _format_inputs(inputs),
        ]
    )


def _trace_target(trace_steps: Iterable) -> str:
    return "\n\n".join(step.serialize() for step in trace_steps)


def _state_trace_target(trace_steps: Iterable) -> str:
    lines: List[str] = []
    for step in trace_steps:
        lines.extend([f"S{step.step}", step.state_text, ""])
    return "\n".join(lines).strip()


def _input_values_record(input_case: Dict[int, int]) -> Dict[str, int]:
    return {str(k): wrap16(v) for k, v in sorted(input_case.items())}


def _opcode_pattern(*instructions: str) -> str:
    return "|".join(instruction.split(None, 1)[0] for instruction in instructions)


def generate_single_step_examples(limit: int | None, seed: int) -> List[dict]:
    rng = random.Random(seed)
    candidates: List[dict] = []

    for task in TASK_LIBRARY:
        for input_case in task.input_cases:
            final_state, transitions = collect_transitions(task.source, inputs=input_case, max_steps=64)
            if final_state.error != "NONE":
                continue
            for transition in transitions:
                candidates.append(
                    {
                        "dataset_type": "single_step",
                        "program_name": task.program_name,
                        "split_family": task.split_family,
                        "category": task.category,
                        "instruction": transition.instruction,
                        "input_values": _input_values_record(input_case),
                        "prompt": _single_step_prompt(transition.before_state_text, transition.instruction),
                        "target": transition.after_state_text,
                    }
                )

    rng.shuffle(candidates)
    return candidates if limit is None else candidates[:limit]


def generate_short_trace_examples(limit: int | None, seed: int) -> List[dict]:
    rng = random.Random(seed)
    candidates: List[dict] = []

    for task in TASK_LIBRARY:
        for input_case in task.input_cases:
            final_state, trace_steps = run_program(task.source, inputs=input_case, max_steps=64, trace=True)
            if final_state.error != "NONE" or not trace_steps:
                continue
            candidates.append(
                {
                    "dataset_type": "short_trace",
                    "program_name": task.program_name,
                    "split_family": task.split_family,
                    "category": task.category,
                    "input_values": _input_values_record(input_case),
                    "prompt": _short_trace_prompt(task.source, input_case),
                    "target": _state_trace_target(trace_steps),
                    "num_steps": len(trace_steps),
                }
            )

    rng.shuffle(candidates)
    return candidates if limit is None else candidates[:limit]


def generate_next_k_steps_examples(
    limit: int | None,
    seed: int,
    next_k_steps: int = 3,
    dataset_type: str | None = None,
) -> List[dict]:
    rng = random.Random(seed)
    candidates: List[dict] = []
    resolved_dataset_type = dataset_type or f"next_{next_k_steps}_steps"
    task_name = f"{resolved_dataset_type}_execution"

    for task in TASK_LIBRARY:
        for input_case in task.input_cases:
            if resolved_dataset_type == "next_2_steps" and next_k_steps == 2:
                final_state, transitions = collect_transitions(task.source, inputs=input_case, max_steps=64)
                if final_state.error != "NONE" or len(transitions) < 2:
                    continue
                first, second = transitions[0], transitions[1]
                candidates.append(
                    {
                        "dataset_type": resolved_dataset_type,
                        "program_name": task.program_name,
                        "split_family": task.split_family,
                        "category": task.category,
                        "selection_key": _opcode_pattern(first.instruction, second.instruction),
                        "input_values": _input_values_record(input_case),
                        "prompt": _next_2_micro_prompt(
                            input_case,
                            first.before_state_text,
                            first.instruction,
                            second.instruction,
                        ),
                        "target": _next_2_micro_target(first.after_state_text, second.after_state_text),
                        "num_steps": 2,
                        "full_trace_steps": len(transitions),
                    }
                )
                continue
            final_state, trace_steps = run_program(task.source, inputs=input_case, max_steps=64, trace=True)
            if final_state.error != "NONE" or not trace_steps:
                continue
            prefix_steps = trace_steps[: min(next_k_steps, len(trace_steps))]
            if not prefix_steps:
                continue
            candidates.append(
                {
                    "dataset_type": resolved_dataset_type,
                    "program_name": task.program_name,
                    "split_family": task.split_family,
                    "category": task.category,
                    "input_values": _input_values_record(input_case),
                    "prompt": _next_k_steps_prompt(task_name, task.source, input_case, len(prefix_steps)),
                    "target": _trace_target(prefix_steps),
                    "num_steps": len(prefix_steps),
                    "full_trace_steps": len(trace_steps),
                }
            )

    rng.shuffle(candidates)
    return candidates if limit is None else candidates[:limit]


def generate_terminal_state_examples(limit: int | None, seed: int) -> List[dict]:
    rng = random.Random(seed)
    candidates: List[dict] = []

    for task in TASK_LIBRARY:
        for input_case in task.input_cases:
            final_state, _ = run_program(task.source, inputs=input_case, max_steps=64, trace=False)
            if final_state.error != "NONE":
                continue
            candidates.append(
                {
                    "dataset_type": "terminal_state",
                    "program_name": task.program_name,
                    "split_family": task.split_family,
                    "category": task.category,
                    "input_values": _input_values_record(input_case),
                    "prompt": _terminal_state_prompt(task.source, input_case),
                    "target": final_state.serialize(),
                }
            )

    rng.shuffle(candidates)
    return candidates if limit is None else candidates[:limit]


def split_records(records: Sequence[dict], seed: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for record in records:
        family = str(record.get("split_family", record.get("program_name", "UNKNOWN")))
        grouped.setdefault(family, []).append(record)

    def family_cost(family_name: str) -> float:
        family_records = grouped[family_name]
        return sum(len(record.get("prompt", "")) + len(record.get("target", "")) for record in family_records) / max(len(family_records), 1)

    family_names = sorted(grouped.keys(), key=family_cost, reverse=True)

    total_families = len(family_names)
    train_end = int(total_families * train_ratio)
    val_end = train_end + int(total_families * val_ratio)

    train_families = set(family_names[:train_end])
    remaining_families = sorted(family_names[train_end:], key=family_cost)
    val_families = set(remaining_families[: max(0, val_end - train_end)])
    test_families = set(remaining_families[max(0, val_end - train_end):])

    splits = {"train": [], "val": [], "test": []}
    for family_name in family_names:
        target_split = "test"
        if family_name in train_families:
            target_split = "train"
        elif family_name in val_families:
            target_split = "val"
        elif family_name in test_families:
            target_split = "test"
        splits[target_split].extend(grouped[family_name])

    return splits


def _bucket_num_steps(num_steps: int) -> str:
    if num_steps <= 4:
        return "1-4"
    if num_steps <= 8:
        return "5-8"
    return "9+"


def _distribution(records: Sequence[dict], key: str) -> Dict[str, int]:
    counter = Counter(str(record.get(key, "UNKNOWN")) for record in records)
    return dict(sorted(counter.items()))


def _step_distribution(records: Sequence[dict]) -> Dict[str, int]:
    counter = Counter()
    for record in records:
        if "num_steps" in record:
            counter[_bucket_num_steps(int(record["num_steps"]))] += 1
    return dict(sorted(counter.items()))


def build_manifest(split_datasets: Dict[str, Dict[str, List[dict]]], seed: int, output_dir: Path) -> dict:
    overall_counts: Dict[str, int] = {}
    split_counts: Dict[str, Dict[str, int]] = {}
    category_distribution: Dict[str, Dict[str, Dict[str, int]]] = {}
    program_distribution: Dict[str, Dict[str, Dict[str, int]]] = {}
    split_family_distribution: Dict[str, Dict[str, Dict[str, int]]] = {}
    step_distribution: Dict[str, Dict[str, Dict[str, int]]] = {}

    for dataset_type, splits in split_datasets.items():
        overall_counts[dataset_type] = sum(len(records) for records in splits.values())
        split_counts[dataset_type] = {split_name: len(records) for split_name, records in splits.items()}
        category_distribution[dataset_type] = {
            split_name: _distribution(records, "category") for split_name, records in splits.items()
        }
        program_distribution[dataset_type] = {
            split_name: _distribution(records, "program_name") for split_name, records in splits.items()
        }
        split_family_distribution[dataset_type] = {
            split_name: _distribution(records, "split_family") for split_name, records in splits.items()
        }
        step_distribution[dataset_type] = {
            split_name: _step_distribution(records) for split_name, records in splits.items()
        }

    return {
        "seed": seed,
        "output_dir": str(output_dir),
        "task_library_programs": len(TASK_LIBRARY),
        "dataset_types": sorted(split_datasets.keys()),
        "overall_counts": overall_counts,
        "split_counts": split_counts,
        "category_distribution": category_distribution,
        "program_distribution": program_distribution,
        "split_family_distribution": split_family_distribution,
        "step_distribution": step_distribution,
    }


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_generation_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def build_generation_config(args: argparse.Namespace) -> dict:
    return {
        "seed": args.seed,
        "output_dir": str(Path(args.output_dir)),
        "single_step_limit": args.single_step_limit,
        "next_2_steps_limit": args.next_2_steps_limit,
        "short_trace_limit": args.short_trace_limit,
        "terminal_state_limit": args.terminal_state_limit,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "max_steps": 64,
        "split_strategy": "program_family_level",
        "split_key": "split_family",
        "dataset_types": ACTIVE_DATASET_TYPES,
        "task_library_programs": [task.program_name for task in TASK_LIBRARY],
        "task_library_split_families": [task.split_family for task in TASK_LIBRARY],
    }


def build_dataset_card(manifest: dict, config: dict) -> str:
    dataset_types = ", ".join(manifest["dataset_types"])
    lines = [
        "# Dataset Card: AI-Assembly ISA v0 MVP",
        "",
        "## Summary",
        "",
        "This dataset contains supervised execution examples generated from the reference AI-Assembly ISA v0 VM.",
        f"It includes the following dataset types: {dataset_types}.",
        "",
        "## Intended Use",
        "",
        "- single-step execution supervision",
        "- next-2-steps trace supervision",
        "- full short-trace supervision",
        "- terminal-state-only supervision",
        "- curriculum learning from explicit execution to compressed execution",
        "",
        "## Split Strategy",
        "",
        "Splits are performed at the program family level.",
        "All examples sharing the same `split_family` are assigned to exactly one of train, val, or test.",
        "This prevents leakage where the same program/input family appears across splits.",
        "",
        "## Generation",
        "",
        f"- Seed: {config['seed']}",
        f"- Max steps per program: {config['max_steps']}",
        f"- Split key: `{config['split_key']}`",
        f"- Split strategy: `{config['split_strategy']}`",
        "",
        "## Dataset Sizes",
        "",
        f"- Overall counts: {json.dumps(manifest['overall_counts'], ensure_ascii=True)}",
        f"- Split counts: {json.dumps(manifest['split_counts'], ensure_ascii=True)}",
        "",
        "## Categories",
        "",
        "Programs currently cover arithmetic, memory, branch, straight-line, and loop tasks.",
        "",
        "## Known Limitations",
        "",
        "- MVP ISA only; no stack, calls, dynamic addressing, or bitwise ops yet",
        "- Synthetic programs are still small and structured",
        "- Loop diversity is limited to bounded counter-style programs",
        "- Local `short_trace` evaluation should stay capped to short traces only",
        "",
        "## Files",
        "",
        "- `single_step/train.jsonl`, `single_step/val.jsonl`, `single_step/test.jsonl`",
        "- `next_2_steps/train.jsonl`, `next_2_steps/val.jsonl`, `next_2_steps/test.jsonl`",
        "- `short_trace/train.jsonl`, `short_trace/val.jsonl`, `short_trace/test.jsonl`",
        "- `terminal_state/train.jsonl`, `terminal_state/val.jsonl`, `terminal_state/test.jsonl`",
        "- `manifest.json`",
        "- `generation_config.json`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI-assembly MVP datasets.")
    parser.add_argument("--output-dir", default="datasets/mvp", help="Directory for generated dataset files.")
    parser.add_argument("--single-step-limit", type=int, default=256)
    parser.add_argument("--next-2-steps-limit", type=int, default=128)
    parser.add_argument("--short-trace-limit", type=int, default=128)
    parser.add_argument("--terminal-state-limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    generated = {
        "single_step": generate_single_step_examples(limit=args.single_step_limit, seed=args.seed),
        "next_2_steps": generate_next_k_steps_examples(
            limit=args.next_2_steps_limit,
            seed=args.seed,
            next_k_steps=2,
            dataset_type="next_2_steps",
        ),
        "short_trace": generate_short_trace_examples(limit=args.short_trace_limit, seed=args.seed),
        "terminal_state": generate_terminal_state_examples(limit=args.terminal_state_limit, seed=args.seed),
    }
    split_datasets = {
        dataset_type: split_records(records, seed=args.seed)
        for dataset_type, records in generated.items()
    }

    for dataset_type, splits in split_datasets.items():
        for split_name, records in splits.items():
            write_jsonl(output_dir / dataset_type / f"{split_name}.jsonl", records)

    manifest = build_manifest(split_datasets, seed=args.seed, output_dir=output_dir)
    config = build_generation_config(args)
    write_manifest(output_dir / "manifest.json", manifest)
    write_generation_config(output_dir / "generation_config.json", config)
    (output_dir / "DATASET_CARD.md").write_text(build_dataset_card(manifest, config), encoding="utf-8")


if __name__ == "__main__":
    main()
