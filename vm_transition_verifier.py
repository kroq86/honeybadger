from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_pipeline import TASK_LIBRARY
from reference_vm import MachineState, Instruction, parse_program, execute_instruction, INPUT_RE, wrap16


PROGRAM_SOURCE_BY_NAME = {task.program_name: task.source for task in TASK_LIBRARY}
PROGRAM_BY_NAME = {task.program_name: parse_program(task.source) for task in TASK_LIBRARY}


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    program_name: str
    expected_instruction: str
    actual_instruction: str
    before_state_text: str
    after_state_text: str
    target_state_text: str | None
    error_type: str | None
    notes: tuple[str, ...] = ()


def state_diff_summary(before_state_text: str, after_state_text: str) -> tuple[str, ...]:
    before = parse_state_text(before_state_text)
    after = parse_state_text(after_state_text)
    diffs: list[str] = []

    if before.ip != after.ip:
        diffs.append(f"ip:{before.ip}->{after.ip}")
    if (before.z, before.n, before.c) != (after.z, after.n, after.c):
        diffs.append(f"flags:{before.z}{before.n}{before.c}->{after.z}{after.n}{after.c}")
    if before.halted != after.halted:
        diffs.append(f"halted:{before.halted}->{after.halted}")
    if before.error != after.error:
        diffs.append(f"error:{before.error}->{after.error}")

    for index, (old, new) in enumerate(zip(before.registers, after.registers)):
        if old != new:
            diffs.append(f"R{index}:{old}->{new}")

    before_out0 = before.output.get(0)
    after_out0 = after.output.get(0)
    if before_out0 != after_out0:
        diffs.append(f"OUT0:{before_out0}->{after_out0}")

    for addr, (old, new) in enumerate(zip(before.memory, after.memory)):
        if old != new:
            diffs.append(f"M[{addr}]:{old}->{new}")

    return tuple(diffs)


def parse_state_text(state_text: str) -> MachineState:
    lines = [line.strip() for line in state_text.strip().splitlines() if line.strip()]
    state = MachineState()
    for line in lines:
        if line.startswith("IP="):
            state.ip = int(line.split("=", 1)[1])
        elif line.startswith("R0="):
            for token in line.split():
                name, value = token.split("=", 1)
                state.registers[int(name[1:])] = wrap16(int(value))
        elif line.startswith("Z="):
            parts = line.split()
            state.z = int(parts[0].split("=", 1)[1])
            state.n = int(parts[1].split("=", 1)[1])
            state.c = int(parts[2].split("=", 1)[1])
        elif line.startswith("OUT[0]="):
            value = line.split("=", 1)[1]
            if value != "_":
                state.output[0] = wrap16(int(value))
        elif line.startswith("HALTED="):
            state.halted = int(line.split("=", 1)[1])
        elif line.startswith("ERROR="):
            state.error = line.split("=", 1)[1]
        elif line.startswith("MEM[") and line.endswith("]"):
            payload = line[4:-1].strip()
            if payload:
                for token in payload.split():
                    left, value = token.split("=", 1)
                    addr = int(left[2:-1])
                    state.memory[addr] = wrap16(int(value))
    return state


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_inputs(input_values: dict[str, int] | dict[int, int] | None) -> dict[int, int]:
    if not input_values:
        return {}
    return {int(key): wrap16(value) for key, value in input_values.items()}


def _resolve_instruction(program_name: str, instruction_text: str) -> Instruction:
    program = PROGRAM_BY_NAME[program_name]
    for instruction in program.instructions:
        if instruction.text == instruction_text:
            return instruction
    parsed = parse_program(instruction_text.strip())
    if len(parsed.instructions) != 1:
        raise ValueError(f"expected exactly one instruction, got {len(parsed.instructions)} from {instruction_text!r}")
    return parsed.instructions[0]


def _section_map(text: str) -> dict[str, str]:
    labels = {"STATE", "EXEC", "INPUT", "S0", "E1", "E2", "S1", "S2"}
    current: str | None = None
    collected: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line in labels:
            current = line
            collected.setdefault(current, [])
            continue
        if current is not None:
            collected[current].append(raw_line)
    return {key: "\n".join(value).strip() for key, value in collected.items()}


def verify_single_step(
    *,
    program_name: str,
    input_values: dict[str, int] | dict[int, int] | None,
    before_state_text: str,
    instruction_text: str,
    target_state_text: str | None = None,
) -> VerificationResult:
    if program_name not in PROGRAM_SOURCE_BY_NAME:
        raise KeyError(f"unknown program_name: {program_name}")
    state = parse_state_text(before_state_text)
    program = PROGRAM_BY_NAME[program_name]
    if not (0 <= state.ip < len(program.instructions)):
        return VerificationResult(
            valid=False,
            program_name=program_name,
            expected_instruction="<ip_oob>",
            actual_instruction=instruction_text,
            before_state_text=before_state_text,
            after_state_text=state.serialize(),
            target_state_text=target_state_text,
            error_type="IP_OOB",
            notes=("state ip outside program instruction range",),
        )
    expected_instruction = program.instructions[state.ip].text
    execute_instruction(state, _resolve_instruction(program_name, instruction_text), program.labels, _normalize_inputs(input_values))
    after_state_text = state.serialize()
    notes: list[str] = []
    if instruction_text != expected_instruction:
        notes.append("instruction_mismatch")
    if target_state_text is not None and after_state_text != target_state_text:
        notes.append("target_mismatch")
    return VerificationResult(
        valid=(instruction_text == expected_instruction and (target_state_text is None or after_state_text == target_state_text)),
        program_name=program_name,
        expected_instruction=expected_instruction,
        actual_instruction=instruction_text,
        before_state_text=before_state_text,
        after_state_text=after_state_text,
        target_state_text=target_state_text,
        error_type=None if state.error == "NONE" else state.error,
        notes=tuple(notes),
    )


def verification_mode_verdict(result: VerificationResult, *, mode: str) -> tuple[bool, tuple[str, ...]]:
    if mode == "intermediate_oracle":
        return result.valid, result.notes
    if mode == "final_state_only":
        ok = result.error_type in {None, "NONE"}
        return ok, result.notes
    if mode == "instruction_only":
        ok = result.actual_instruction == result.expected_instruction and result.error_type in {None, "NONE"}
        extra = () if ok else ("instruction_only_reject",)
        return ok, result.notes + extra
    if mode == "state_diff":
        if result.target_state_text is None:
            ok = result.error_type in {None, "NONE"}
            return ok, result.notes
        actual_diff = state_diff_summary(result.before_state_text, result.after_state_text)
        target_diff = state_diff_summary(result.before_state_text, result.target_state_text)
        ok = actual_diff == target_diff and result.error_type in {None, "NONE"}
        extra = () if ok else ("state_diff_mismatch",)
        return ok, result.notes + extra
    raise ValueError(f"unsupported verification mode: {mode}")


def verify_single_step_record(record: dict[str, Any]) -> VerificationResult:
    sections = _section_map(record["prompt"])
    return verify_single_step(
        program_name=record["program_name"],
        input_values=record.get("input_values"),
        before_state_text=sections["STATE"],
        instruction_text=record["instruction"],
        target_state_text=record.get("target"),
    )


def _split_next2_target(target: str) -> tuple[str, str]:
    sections = _section_map(target)
    return sections["S1"], sections["S2"]


def verify_next_2_steps_record(record: dict[str, Any]) -> list[VerificationResult]:
    sections = _section_map(record["prompt"])
    state0 = sections["S0"]
    step1_target, step2_target = _split_next2_target(record["target"])

    first = verify_single_step(
        program_name=record["program_name"],
        input_values=record.get("input_values"),
        before_state_text=state0,
        instruction_text=sections["E1"],
        target_state_text=step1_target,
    )
    second = verify_single_step(
        program_name=record["program_name"],
        input_values=record.get("input_values"),
        before_state_text=first.after_state_text,
        instruction_text=sections["E2"],
        target_state_text=step2_target,
    )
    return [first, second]


def replay_dataset(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in load_jsonl(path):
        if record["dataset_type"] == "single_step":
            result = verify_single_step_record(record)
            rows.append(
                {
                    "dataset_type": "single_step",
                    "program_name": record["program_name"],
                    "valid": result.valid,
                    "notes": list(result.notes),
                    "error_type": result.error_type,
                }
            )
        elif record["dataset_type"] == "next_2_steps":
            results = verify_next_2_steps_record(record)
            rows.append(
                {
                    "dataset_type": "next_2_steps",
                    "program_name": record["program_name"],
                    "valid": all(result.valid for result in results),
                    "step_results": [
                        {
                            "valid": result.valid,
                            "notes": list(result.notes),
                            "error_type": result.error_type,
                            "expected_instruction": result.expected_instruction,
                            "actual_instruction": result.actual_instruction,
                        }
                        for result in results
                    ],
                }
            )
        else:
            raise ValueError(f"unsupported dataset_type for replay: {record['dataset_type']}")
    return rows
