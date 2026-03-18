from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Callable, Dict, List

from dataset_pipeline import split_records
from synthesis_dataset import (
    SYNTHESIS_MVP_PROGRAMS,
    SYNTHESIS_SPEC_LINES,
    generate_synthesis_examples,
    validate_synthesis_target,
)


REGISTER_TOKEN_RE = re.compile(r"\bR([0-7])\b")
CONST_LINE_RE = re.compile(r"^(CONST\s+R[0-7],\s*)(-?\d+)\s*$")


def _synthesis_records() -> List[dict]:
    records = generate_synthesis_examples(limit=None, seed=7)
    return [record for record in records if record["program_name"] in SYNTHESIS_MVP_PROGRAMS]


def _replace_first_register(line: str) -> str | None:
    def repl(match: re.Match[str]) -> str:
        value = int(match.group(1))
        return f"R{(value + 1) % 8}"

    replaced, count = REGISTER_TOKEN_RE.subn(repl, line, count=1)
    return replaced if count else None


def mutate_wrong_register(program_source: str) -> str | None:
    lines = program_source.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.endswith(":"):
            continue
        mutated = _replace_first_register(stripped)
        if mutated and mutated != stripped:
            lines[index] = mutated
            return "\n".join(lines)
    return None


def mutate_missing_halt(program_source: str) -> str | None:
    lines = program_source.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip() == "HALT":
            return "\n".join(lines[:index] + lines[index + 1 :])
    return None


def mutate_swapped_jump_condition(program_source: str) -> str | None:
    lines = program_source.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("JZ "):
            lines[index] = stripped.replace("JZ ", "JNZ ", 1)
            return "\n".join(lines)
        if stripped.startswith("JNZ "):
            lines[index] = stripped.replace("JNZ ", "JZ ", 1)
            return "\n".join(lines)
    return None


def mutate_wrong_constant(program_source: str) -> str | None:
    lines = program_source.splitlines()
    for index, line in enumerate(lines):
        match = CONST_LINE_RE.fullmatch(line.strip())
        if not match:
            continue
        prefix, raw_value = match.groups()
        mutated_value = int(raw_value) + 1
        lines[index] = f"{prefix}{mutated_value}"
        return "\n".join(lines)
    return None


MUTATORS: Dict[str, Callable[[str], str | None]] = {
    "wrong_register": mutate_wrong_register,
    "missing_halt": mutate_missing_halt,
    "swapped_jump_condition": mutate_swapped_jump_condition,
    "wrong_constant": mutate_wrong_constant,
}


def build_repair_prompt(record: dict, bug_class: str, buggy_program: str) -> str:
    spec_lines = SYNTHESIS_SPEC_LINES[record["program_name"]]
    io_sections: List[str] = []
    for index, example in enumerate(record["io_examples"], start=1):
        io_sections.extend(
            [
                f"CASE {index}",
                "INPUT",
                *(
                    [f"input[{key}]={value}" for key, value in sorted(example["inputs"].items(), key=lambda item: int(item[0]))]
                    or ["<none>"]
                ),
                "EXPECTED",
                f"output[0]={example['expected_output']}",
            ]
        )
    return "\n".join(
        [
            "TASK: program_repair",
            "Emit fixed program only.",
            "SPEC",
            f"program_name={record['program_name']}",
            f"category={record['category']}",
            *spec_lines,
            f"bug_class={bug_class}",
            "BUGGY_PROGRAM",
            buggy_program,
            "IO_EXAMPLES",
            *io_sections,
        ]
    )


def generate_repair_examples(limit: int | None, seed: int, allowed_bug_classes: List[str] | None = None) -> List[dict]:
    rng = random.Random(seed)
    records: List[dict] = []
    bug_classes = allowed_bug_classes or list(MUTATORS.keys())

    for record in _synthesis_records():
        for bug_class in bug_classes:
            mutator = MUTATORS[bug_class]
            buggy_program = mutator(record["target"])
            if not buggy_program:
                continue
            buggy_validation = validate_synthesis_target(buggy_program, record["io_examples"])
            fixed_validation = validate_synthesis_target(record["target"], record["io_examples"])
            if not buggy_validation["syntactic_valid"]:
                continue
            if not fixed_validation["functional_correct"]:
                continue
            if buggy_validation["functional_correct"]:
                continue
            records.append(
                {
                    "dataset_type": "program_repair",
                    "program_name": record["program_name"],
                    "split_family": record["split_family"],
                    "category": record["category"],
                    "bug_class": bug_class,
                    "prompt": build_repair_prompt(record, bug_class, buggy_program),
                    "target": record["target"],
                    "buggy_program": buggy_program,
                    "io_examples": record["io_examples"],
                    "buggy_validation": buggy_validation,
                }
            )

    rng.shuffle(records)
    return records if limit is None else records[:limit]


def validate_repair_target(fixed_program: str, io_examples: List[dict]) -> dict:
    return validate_synthesis_target(fixed_program, io_examples)


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_manifest(split_datasets: Dict[str, List[dict]], output_dir: Path, seed: int) -> dict:
    return {
        "seed": seed,
        "output_dir": str(output_dir),
        "dataset_type": "program_repair",
        "split_counts": {split_name: len(records) for split_name, records in split_datasets.items()},
        "bug_class_distribution": {
            split_name: {
                bug_class: sum(1 for record in records if record["bug_class"] == bug_class)
                for bug_class in sorted({record["bug_class"] for record in records})
            }
            for split_name, records in split_datasets.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the repair MVP dataset.")
    parser.add_argument("--output-dir", default="datasets/repair_mvp")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bug-classes", nargs="*", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records = generate_repair_examples(limit=args.limit, seed=args.seed, allowed_bug_classes=args.bug_classes)
    split_datasets = split_records(records, seed=args.seed)
    for split_name, split_records_list in split_datasets.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", split_records_list)
    manifest = build_manifest(split_datasets, output_dir=output_dir, seed=args.seed)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
