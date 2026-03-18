from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from dataset_pipeline import TASK_LIBRARY, split_records, wrap16
from reference_vm import run_program
from synthesis_dataset import SYNTHESIS_MVP_PROGRAMS


def _selected_tasks():
    return [task for task in TASK_LIBRARY if task.program_name in SYNTHESIS_MVP_PROGRAMS]


def _format_inputs(inputs: Dict[int, int]) -> str:
    if not inputs:
        return "INPUT\n<none>"
    rows = ["INPUT"]
    for key, value in sorted(inputs.items()):
        rows.append(f"input[{key}]={wrap16(value)}")
    return "\n".join(rows)


def build_probe_prompt(program_name: str, category: str, program_source: str, input_case: Dict[int, int]) -> str:
    return "\n".join(
        [
            "TASK: latent_terminal_probe",
            "Emit final canonical state only.",
            "SPEC",
            f"program_name={program_name}",
            f"category={category}",
            "compression_goal=final_state_without_explicit_trace",
            "PROGRAM",
            program_source,
            _format_inputs(input_case),
        ]
    )


def generate_latent_probe_examples(limit: int | None, seed: int) -> List[dict]:
    rng = random.Random(seed)
    records: List[dict] = []
    for task in _selected_tasks():
        for input_case in task.input_cases:
            final_state, _ = run_program(task.source, inputs=input_case, max_steps=64, trace=False)
            if final_state.error != "NONE":
                continue
            records.append(
                {
                    "dataset_type": "latent_terminal_probe",
                    "program_name": task.program_name,
                    "split_family": task.split_family,
                    "category": task.category,
                    "input_values": {str(k): wrap16(v) for k, v in sorted(input_case.items())},
                    "prompt": build_probe_prompt(task.program_name, task.category, task.source, input_case),
                    "target": final_state.serialize(),
                }
            )
    rng.shuffle(records)
    return records if limit is None else records[:limit]


def validate_probe_target(program_source: str, input_values: Dict[str, int], expected_target: str) -> bool:
    final_state, _ = run_program(
        program_source,
        inputs={int(key): int(value) for key, value in input_values.items()},
        max_steps=64,
        trace=False,
    )
    return final_state.error == "NONE" and final_state.serialize().strip() == expected_target.strip()


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a latent terminal-state probe dataset.")
    parser.add_argument("--output-dir", default="datasets/latent_probe_terminal")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records = generate_latent_probe_examples(limit=args.limit, seed=args.seed)
    split_datasets = split_records(records, seed=args.seed)
    for split_name, split_records_list in split_datasets.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", split_records_list)
    manifest = {
        "seed": args.seed,
        "output_dir": str(output_dir),
        "dataset_type": "latent_terminal_probe",
        "split_counts": {split_name: len(split_records_list) for split_name, split_records_list in split_datasets.items()},
        "program_names": sorted({record["program_name"] for record in records}),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
