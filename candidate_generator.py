from __future__ import annotations

from dataclasses import dataclass

from dataset_pipeline import TASK_LIBRARY
from reference_vm import Instruction, parse_program
from vm_transition_verifier import parse_state_text


PROGRAM_BY_NAME = {task.program_name: parse_program(task.source) for task in TASK_LIBRARY}


@dataclass(frozen=True)
class CandidateInstruction:
    instruction_text: str
    rank: int
    source: str


def _fallthrough_index(ip: int) -> int:
    return ip + 1


def _add_candidate(
    candidates: list[CandidateInstruction],
    seen: set[str],
    instruction: Instruction,
    *,
    rank: int,
    source: str,
) -> None:
    if instruction.text in seen:
        return
    seen.add(instruction.text)
    candidates.append(CandidateInstruction(instruction_text=instruction.text, rank=rank, source=source))


def generate_candidates(
    *,
    program_name: str,
    before_state_text: str,
    mode: str = "strict_local",
    limit: int | None = None,
) -> list[CandidateInstruction]:
    if program_name not in PROGRAM_BY_NAME:
        raise KeyError(f"unknown program_name: {program_name}")

    program = PROGRAM_BY_NAME[program_name]
    state = parse_state_text(before_state_text)
    instructions = program.instructions
    candidates: list[CandidateInstruction] = []
    seen: set[str] = set()

    if mode not in {"strict_local", "program_global"}:
        raise ValueError(f"unsupported candidate mode: {mode}")

    if mode == "strict_local":
        if 0 <= state.ip < len(instructions):
            current = instructions[state.ip]
            _add_candidate(candidates, seen, current, rank=1, source="current_ip")

            if current.opcode in {"JMP", "JZ", "JNZ", "JL", "JLE", "JG", "JGE"}:
                target_ip = program.labels[current.args[0]]
                _add_candidate(candidates, seen, instructions[target_ip], rank=2, source="jump_target")
                if current.opcode != "JMP":
                    fallthrough_ip = _fallthrough_index(state.ip)
                    if 0 <= fallthrough_ip < len(instructions):
                        _add_candidate(
                            candidates,
                            seen,
                            instructions[fallthrough_ip],
                            rank=3,
                            source="jump_fallthrough",
                        )

        for offset, source in ((-1, "prev_ip"), (1, "next_ip"), (2, "next2_ip")):
            idx = state.ip + offset
            if 0 <= idx < len(instructions):
                _add_candidate(candidates, seen, instructions[idx], rank=len(candidates) + 1, source=source)
    else:
        for idx, instruction in enumerate(instructions):
            if idx == state.ip:
                source = "current_ip"
            elif idx == state.ip - 1:
                source = "prev_ip"
            elif idx == state.ip + 1:
                source = "next_ip"
            elif idx == state.ip + 2:
                source = "next2_ip"
            else:
                source = "program_scan"
            _add_candidate(candidates, seen, instruction, rank=len(candidates) + 1, source=source)

    if limit is None:
        return candidates
    return candidates[:limit]
