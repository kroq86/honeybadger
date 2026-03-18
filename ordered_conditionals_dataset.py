from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from dataset_pipeline import split_records, wrap16
from reference_vm import run_program


ORDERED_TASKS = [
    {
        "program_name": "ordered_compare_two_numbers",
        "split_family": "ordered_compare_two_numbers_family",
        "category": "ordered_branch",
        "source": """
READ R1, input[0]
READ R2, input[1]
CMP R1, R2
JL LESS
JG GREATER
CONST R3, 0
WRITE output[0], R3
HALT
LESS:
CONST R3, -1
WRITE output[0], R3
HALT
GREATER:
CONST R3, 1
WRITE output[0], R3
HALT
""".strip(),
        "input_cases": [
            {0: 2, 1: 9},
            {0: 7, 1: 7},
            {0: 9, 1: 2},
        ],
    },
    {
        "program_name": "sign_bucket_test",
        "split_family": "sign_bucket_test_family",
        "category": "ordered_branch",
        "source": """
READ R1, input[0]
TEST R1
JL NEG
JG POS
CONST R2, 0
WRITE output[0], R2
HALT
NEG:
CONST R2, -1
WRITE output[0], R2
HALT
POS:
CONST R2, 1
WRITE output[0], R2
HALT
""".strip(),
        "input_cases": [
            {0: -5},
            {0: 0},
            {0: 8},
        ],
    },
]


def _format_inputs(inputs: Dict[int, int]) -> str:
    if not inputs:
        return "INPUT\n<none>"
    rows = ["INPUT"]
    for key, value in sorted(inputs.items()):
        rows.append(f"input[{key}]={wrap16(value)}")
    return "\n".join(rows)


def generate_ordered_conditionals_examples(limit: int | None, seed: int) -> List[dict]:
    rng = random.Random(seed)
    records: List[dict] = []
    for task in ORDERED_TASKS:
        for input_case in task["input_cases"]:
            final_state, _ = run_program(task["source"], inputs=input_case, max_steps=64, trace=False)
            if final_state.error != "NONE":
                continue
            records.append(
                {
                    "dataset_type": "ordered_conditionals_demo",
                    "program_name": task["program_name"],
                    "split_family": task["split_family"],
                    "category": task["category"],
                    "input_values": {str(k): wrap16(v) for k, v in sorted(input_case.items())},
                    "prompt": "\n".join(
                        [
                            "TASK: ordered_conditionals_demo",
                            "Emit final canonical state only.",
                            "PROGRAM",
                            task["source"],
                            _format_inputs(input_case),
                        ]
                    ),
                    "target": final_state.serialize(),
                }
            )
    rng.shuffle(records)
    return records if limit is None else records[:limit]


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a tiny ordered-conditionals side-track dataset.")
    parser.add_argument("--output-dir", default="datasets/ordered_conditionals_demo")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records = generate_ordered_conditionals_examples(limit=args.limit, seed=args.seed)
    split_datasets = split_records(records, seed=args.seed)
    for split_name, split_records_list in split_datasets.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", split_records_list)
    manifest = {
        "seed": args.seed,
        "output_dir": str(output_dir),
        "dataset_type": "ordered_conditionals_demo",
        "split_counts": {split_name: len(split_records_list) for split_name, split_records_list in split_datasets.items()},
        "program_names": sorted({record["program_name"] for record in records}),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
