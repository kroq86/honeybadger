from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "training_data" / "sft_next2_biased_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training_data" / "sft_next2_delta_v1"


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def parse_canonical_state(text: str) -> dict:
    parsed: dict = {"registers": {}}
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("IP="):
            parsed["IP"] = int(line.split("=", 1)[1])
            continue
        if line.startswith("R0="):
            for token in line.split():
                name, value = token.split("=", 1)
                parsed["registers"][name] = int(value)
            continue
        if line.startswith("Z="):
            parts = dict(token.split("=", 1) for token in line.split())
            parsed["Z"] = int(parts["Z"])
            parsed["N"] = int(parts["N"])
            parsed["C"] = int(parts["C"])
            continue
        if line.startswith("OUT[0]="):
            parsed["OUT[0]"] = line.split("=", 1)[1]
            continue
        if line.startswith("HALTED="):
            parsed["HALTED"] = int(line.split("=", 1)[1])
            continue
        if line.startswith("ERROR="):
            parsed["ERROR"] = line.split("=", 1)[1]
            continue
    return parsed


def parse_next2_target(text: str) -> dict[str, dict]:
    sections: dict[str, list[str]] = {}
    current_label: str | None = None
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if line in {"S1", "S2"}:
            current_label = line
            sections[current_label] = []
            continue
        if current_label is not None and line:
            sections[current_label].append(line)
    return {label: parse_canonical_state("\n".join(lines)) for label, lines in sections.items()}


def parse_prompt_s0(text: str) -> dict:
    lines = text.strip().splitlines()
    try:
        start = lines.index("S0") + 1
    except ValueError as exc:
        raise ValueError("prompt missing S0 section") from exc
    state_lines: list[str] = []
    for line in lines[start:]:
        if line in {"E1", "E2"}:
            break
        if line.strip():
            state_lines.append(line)
    return parse_canonical_state("\n".join(state_lines))


def delta_lines(before: dict, after: dict, label: str) -> list[str]:
    lines = [label]
    if before.get("IP") != after.get("IP"):
        lines.append(f"IP={after['IP']}")

    register_changes = []
    before_registers = before.get("registers", {})
    after_registers = after.get("registers", {})
    for index in range(8):
        name = f"R{index}"
        if before_registers.get(name) != after_registers.get(name):
            register_changes.append(f"{name}={after_registers.get(name, 0)}")
    if register_changes:
        lines.append("REG " + " ".join(register_changes))

    flag_changes = []
    for flag in ("Z", "N", "C"):
        if before.get(flag) != after.get(flag):
            flag_changes.append(f"{flag}={after.get(flag, 0)}")
    if flag_changes:
        lines.append("FLAGS " + " ".join(flag_changes))

    if before.get("OUT[0]") != after.get("OUT[0]"):
        lines.append(f"OUT[0]={after.get('OUT[0]', '_')}")
    if before.get("HALTED") != after.get("HALTED"):
        lines.append(f"HALTED={after.get('HALTED', 0)}")
    if before.get("ERROR") != after.get("ERROR"):
        lines.append(f"ERROR={after.get('ERROR', 'NONE')}")

    if len(lines) == 1:
        lines.append("NOOP")
    return lines


def to_delta_record(source_record: dict) -> dict:
    metadata = source_record["metadata"]
    s0 = parse_prompt_s0(source_record["prompt"])
    target_sections = parse_next2_target(source_record["completion"])
    s1 = target_sections["S1"]
    s2 = target_sections["S2"]
    delta_target = "\n".join(delta_lines(s0, s1, "D1") + [""] + delta_lines(s1, s2, "D2"))
    delta_prompt = source_record["prompt"].replace(
        "TASK: next_2_steps_execution\nEmit S1 and S2 only.",
        "TASK: next_2_steps_delta\nEmit D1 and D2 only. Report only changed fields per step.",
    )
    return {
        "prompt": delta_prompt,
        "completion": delta_target,
        "metadata": {
            "dataset_type": "next_2_steps_delta",
            "program_name": metadata["program_name"],
            "split_family": metadata.get("split_family"),
            "category": metadata.get("category"),
            "split": metadata["split"],
            "input_values": metadata.get("input_values"),
            "instruction": metadata.get("instruction"),
            "num_steps": metadata.get("num_steps"),
            "selection_key": metadata.get("selection_key"),
            "source_dataset_type": metadata.get("dataset_type"),
        },
    }


def export_delta_splits(source_root: Path) -> dict[str, list[dict]]:
    exported: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        source_path = source_root / f"{split}.jsonl"
        records = read_jsonl(source_path)
        next2_records = [
            record
            for record in records
            if record.get("metadata", {}).get("dataset_type") == "next_2_steps"
        ]
        exported[split] = [to_delta_record(record) for record in next2_records]
    return exported


def build_manifest(output_dir: Path, source_root: Path, exported: dict[str, list[dict]]) -> dict:
    return {
        "output_dir": str(output_dir),
        "source_root": str(source_root),
        "format": "plain_prompt_completion_v1",
        "dataset_type": "next_2_steps_delta",
        "split_counts": {split: len(records) for split, records in exported.items()},
        "supervision_shape": "delta_only_changed_fields",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export next_2_steps into a delta-only supervision format.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    exported = export_delta_splits(source_root)
    for split, records in exported.items():
        write_jsonl(output_dir / f"{split}.jsonl", records)
    manifest = build_manifest(output_dir, source_root, exported)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
