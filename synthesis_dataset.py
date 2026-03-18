from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from dataset_pipeline import TASK_LIBRARY, TaskProgram, split_records, wrap16
from reference_vm import parse_program, run_program


SYNTHESIS_MVP_PROGRAMS = {
    "add_two_numbers",
    "sum_three_constants",
    "store_then_load",
    "branch_on_equality",
    "branch_non_equal_flag",
    "pair_sum_scan_3",
    "midpoint_search_4",
    "first_match_scan_4",
}

SYNTHESIS_SPEC_LINES = {
    "add_two_numbers": [
        "goal=read two inputs and write their sum to output[0]",
        "shape=straight_line",
    ],
    "sum_three_constants": [
        "goal=compute the constant value 13 and write it to output[0]",
        "shape=straight_line",
    ],
    "store_then_load": [
        "goal=store constant 42 into memory[10], load it back, and write it to output[0]",
        "shape=memory_straight_line",
    ],
    "branch_on_equality": [
        "goal=write 1 when input[0] equals input[1], else write 0",
        "shape=single_branch",
    ],
    "branch_non_equal_flag": [
        "goal=write -1 when input[0] differs from input[1], else write 1",
        "shape=single_branch",
    ],
    "pair_sum_scan_3": [
        "goal=check whether any pair among the first three inputs sums to the target in input[3]",
        "output=1 for pair (0,1), 2 for pair (0,2), 3 for pair (1,2), else -1",
        "shape=bounded_search",
    ],
    "midpoint_search_4": [
        "goal=search a sorted 4-value array for the target in input[4]",
        "output=index when found, else -1",
        "shape=bounded_search",
    ],
    "first_match_scan_4": [
        "goal=scan the first four inputs and return the first index equal to input[4]",
        "output=index when found, else -1",
        "shape=bounded_search",
    ],
}


def _selected_tasks() -> List[TaskProgram]:
    return [task for task in TASK_LIBRARY if task.program_name in SYNTHESIS_MVP_PROGRAMS]


def _format_inputs(input_case: Dict[int, int]) -> List[str]:
    if not input_case:
        return ["<none>"]
    return [f"input[{idx}]={wrap16(value)}" for idx, value in sorted(input_case.items())]


def _expected_output(task: TaskProgram, input_case: Dict[int, int]) -> int:
    final_state, _ = run_program(task.source, inputs=input_case, max_steps=64, trace=False)
    if final_state.error != "NONE" or 0 not in final_state.output:
        raise ValueError(f"task {task.program_name} failed to produce a valid output for {input_case}")
    return wrap16(final_state.output[0])


def build_synthesis_prompt(task: TaskProgram) -> str:
    io_sections: List[str] = []
    for index, input_case in enumerate(task.input_cases, start=1):
        io_sections.extend(
            [
                f"CASE {index}",
                "INPUT",
                *_format_inputs(input_case),
                "EXPECTED",
                f"output[0]={_expected_output(task, input_case)}",
            ]
        )

    spec_lines = SYNTHESIS_SPEC_LINES[task.program_name]
    return "\n".join(
        [
            "TASK: program_synthesis",
            "Emit program only.",
            "SPEC",
            f"program_name={task.program_name}",
            f"category={task.category}",
            *spec_lines,
            "ISA=v0_mvp",
            "IO_EXAMPLES",
            *io_sections,
        ]
    )


def generate_synthesis_examples(limit: int | None, seed: int) -> List[dict]:
    rng = random.Random(seed)
    records: List[dict] = []

    for task in _selected_tasks():
        records.append(
            {
                "dataset_type": "program_synthesis",
                "program_name": task.program_name,
                "split_family": task.split_family,
                "category": task.category,
                "prompt": build_synthesis_prompt(task),
                "target": task.source,
                "io_examples": [
                    {
                        "inputs": {str(idx): wrap16(value) for idx, value in sorted(input_case.items())},
                        "expected_output": _expected_output(task, input_case),
                    }
                    for input_case in task.input_cases
                ],
            }
        )

    rng.shuffle(records)
    return records if limit is None else records[:limit]


def validate_synthesis_target(program_source: str, io_examples: List[dict]) -> dict:
    try:
        parse_program(program_source)
    except Exception as exc:  # pragma: no cover - exact parse failures vary by source
        return {
            "syntactic_valid": False,
            "executable_valid": False,
            "functional_correct": False,
            "reason": f"parse_error: {exc}",
        }

    for example in io_examples:
        inputs = {int(idx): int(value) for idx, value in example["inputs"].items()}
        final_state, _ = run_program(program_source, inputs=inputs, max_steps=64, trace=False)
        if final_state.error != "NONE":
            return {
                "syntactic_valid": True,
                "executable_valid": False,
                "functional_correct": False,
                "reason": f"runtime_error: {final_state.error}",
            }
        if final_state.halted != 1:
            return {
                "syntactic_valid": True,
                "executable_valid": False,
                "functional_correct": False,
                "reason": "runtime_error: not_halted",
            }
        actual_output = wrap16(final_state.output.get(0, 0))
        if actual_output != int(example["expected_output"]):
            return {
                "syntactic_valid": True,
                "executable_valid": True,
                "functional_correct": False,
                "reason": f"wrong_output: expected={example['expected_output']} actual={actual_output}",
            }

    return {
        "syntactic_valid": True,
        "executable_valid": True,
        "functional_correct": True,
        "reason": "OK",
    }


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_manifest(split_datasets: Dict[str, List[dict]], output_dir: Path, seed: int) -> dict:
    split_counts = {split_name: len(records) for split_name, records in split_datasets.items()}
    family_distribution = {
        split_name: {
            family: family_count
            for family, family_count in sorted(
                (
                    (family, sum(1 for record in records if record["split_family"] == family))
                    for family in {record["split_family"] for record in records}
                ),
                key=lambda item: item[0],
            )
        }
        for split_name, records in split_datasets.items()
    }
    return {
        "seed": seed,
        "output_dir": str(output_dir),
        "dataset_type": "program_synthesis",
        "task_library_programs": len(_selected_tasks()),
        "split_counts": split_counts,
        "split_family_distribution": family_distribution,
        "program_names": sorted(task.program_name for task in _selected_tasks()),
    }


def build_dataset_card(manifest: dict) -> str:
    return "\n".join(
        [
            "# Dataset Card: Program Synthesis MVP",
            "",
            "## Summary",
            "",
            "This dataset is a separate synthesis track and does not modify the active execution benchmark ladder.",
            "Canonical format:",
            "- input: spec + io_examples",
            "- target: program",
            "",
            "## Scope",
            "",
            "- straight-line seeds",
            "- simple branch seeds",
            "- bounded search seeds",
            "- no loop synthesis in this first slice",
            "",
            "## Split Strategy",
            "",
            "Program family level.",
            "",
            f"## Split Counts\n\n`{json.dumps(manifest['split_counts'], ensure_ascii=True)}`",
            "",
            f"## Program Names\n\n`{', '.join(manifest['program_names'])}`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the program synthesis MVP dataset.")
    parser.add_argument("--output-dir", default="datasets/synthesis_mvp")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records = generate_synthesis_examples(limit=args.limit, seed=args.seed)
    split_datasets = split_records(records, seed=args.seed)
    for split_name, split_records_list in split_datasets.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", split_records_list)
    manifest = build_manifest(split_datasets, output_dir=output_dir, seed=args.seed)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    (output_dir / "DATASET_CARD.md").write_text(build_dataset_card(manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
