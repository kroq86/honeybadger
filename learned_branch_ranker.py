from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from candidate_generator import CandidateInstruction
from vm_transition_verifier import parse_state_text


def _opcode(instruction_text: str) -> str:
    return instruction_text.split()[0]


def _opcode_family(opcode: str) -> str:
    if opcode in {"READ", "LOAD", "MOV"}:
        return "load"
    if opcode in {"WRITE", "STORE"}:
        return "store"
    if opcode in {"ADD", "SUB"}:
        return "arith"
    if opcode in {"CMP", "TEST"}:
        return "compare"
    if opcode.startswith("J"):
        return "branch"
    if opcode == "CONST":
        return "const"
    if opcode == "HALT":
        return "terminal"
    return "other"


def _control_role(opcode: str) -> str:
    if opcode in {"JZ", "JNZ", "JL", "JLE", "JG", "JGE"}:
        return "cond_branch"
    if opcode == "JMP":
        return "jump"
    if opcode in {"CMP", "TEST"}:
        return "compare"
    if opcode == "MOV":
        return "move"
    if opcode in {"ADD", "SUB"}:
        return "update"
    if opcode in {"READ", "LOAD"}:
        return "load"
    if opcode in {"WRITE", "STORE"}:
        return "store"
    if opcode == "CONST":
        return "const"
    if opcode == "HALT":
        return "terminal"
    return "other"


def _candidate_rank_bucket(rank: int) -> str:
    if rank <= 1:
        return "top1"
    if rank <= 3:
        return "top3"
    if rank <= 5:
        return "top5"
    return "tail"


def _register_tokens(instruction_text: str) -> tuple[str, ...]:
    return tuple(sorted(token for token in instruction_text.replace(",", " ").split() if token.startswith("R")))


def _argument_tokens(instruction_text: str) -> tuple[str, ...]:
    parts = instruction_text.replace(",", " ").split()
    return tuple(parts[1:])


def _rank_delta_bucket(candidate_rank: int, state_ip: int) -> str:
    delta = candidate_rank - (state_ip + 1)
    if delta <= -2:
        return "back_far"
    if delta == -1:
        return "back_one"
    if delta == 0:
        return "current"
    if delta == 1:
        return "next"
    if delta == 2:
        return "next2"
    if delta <= 4:
        return "near_scan"
    return "far_scan"


def _candidate_locality(candidate_rank: int, state_ip: int) -> str:
    delta = candidate_rank - (state_ip + 1)
    if delta < 0:
        return "backward_rank"
    if delta == 0:
        return "aligned_rank"
    if delta <= 2:
        return "near_forward_rank"
    return "far_rank"


def _state_shape(before_state_text: str) -> dict[str, str]:
    state = parse_state_text(before_state_text)
    nonzero_regs = sum(1 for value in state.registers if value != 0)
    touched_mem = sum(1 for value in state.memory if value != 0)
    return {
        "ip_bucket": str(min(state.ip, 6)),
        "nonzero_regs": str(min(nonzero_regs, 4)),
        "touched_mem": "0" if touched_mem == 0 else ("1" if touched_mem <= 2 else "many"),
        "has_output": "1" if 0 in state.output else "0",
        "halted": str(state.halted),
    }


def _state_register_overlap(before_state_text: str, instruction_text: str) -> dict[str, str]:
    state = parse_state_text(before_state_text)
    registers = _register_tokens(instruction_text)
    values = []
    for token in registers:
        try:
            values.append(state.registers[int(token[1:])])
        except (ValueError, IndexError):
            continue
    nonzero = sum(1 for value in values if value != 0)
    return {
        "mentioned_regs": str(min(len(values), 3)),
        "mentioned_nonzero_regs": str(min(nonzero, 3)),
        "has_nonzero_reg": "1" if nonzero else "0",
        "mentions_flagged_reg": "1" if any(token in {"R0", "R1", "R2", "R3"} for token in registers) else "0",
    }


def _instruction_shape(instruction_text: str) -> dict[str, str]:
    opcode = _opcode(instruction_text)
    args = _argument_tokens(instruction_text)
    joined = " ".join(args)
    return {
        "family": _opcode_family(opcode),
        "control_role": _control_role(opcode),
        "arity": str(min(len(args), 3)),
        "has_input_ref": "1" if "input[" in joined else "0",
        "has_output_ref": "1" if "output[" in joined else "0",
        "has_memory_ref": "1" if "mem[" in joined.lower() else "0",
        "has_label": "1" if any(arg.isidentifier() and not arg.startswith("R") and "[" not in arg for arg in args) else "0",
        "has_immediate": "1" if any(arg.lstrip("-").isdigit() for arg in args) else "0",
        "writes_register": "1" if args and args[0].startswith("R") and opcode not in {"CMP", "TEST"} else "0",
    }


def _branch_state_cues(before_state_text: str) -> dict[str, str]:
    state = parse_state_text(before_state_text)
    return {
        "z": str(state.z),
        "n": str(state.n),
        "c": str(state.c),
    }


def _branch_behavior(opcode: str, before_state_text: str) -> str:
    flags = _branch_state_cues(before_state_text)
    z = flags["z"] == "1"
    n = flags["n"] == "1"
    if opcode == "JZ":
        return "taken" if z else "not_taken"
    if opcode == "JNZ":
        return "taken" if not z else "not_taken"
    if opcode == "JL":
        return "taken" if n else "not_taken"
    if opcode == "JGE":
        return "taken" if not n else "not_taken"
    if opcode in {"JLE", "JG", "JMP"}:
        return "control"
    return "na"


def _register_value_relation(before_state_text: str, instruction_text: str) -> str:
    state = parse_state_text(before_state_text)
    args = _argument_tokens(instruction_text)
    reg_tokens = [arg for arg in args if arg.startswith("R")]
    if len(reg_tokens) < 2:
        return "na"
    try:
        left = state.registers[int(reg_tokens[0][1:])]
        right = state.registers[int(reg_tokens[1][1:])]
    except (ValueError, IndexError):
        return "na"
    if left == right:
        return "eq"
    if left == 0 or right == 0:
        return "one_zero"
    if left < right:
        return "lt"
    return "gt"


def candidate_feature_keys(
    *,
    program_name: str,
    remaining_steps: int,
    state_ip: int,
    candidate: CandidateInstruction,
    include_program_name: bool = True,
    include_candidate_source: bool = True,
    before_state_text: str | None = None,
    include_structural_source_hints: bool = False,
) -> list[str]:
    opcode = _opcode(candidate.instruction_text)
    source_token = candidate.source if include_candidate_source else "<src_hidden>"
    program_token = program_name if include_program_name else "<program_hidden>"
    register_signature = "|".join(_register_tokens(candidate.instruction_text)) or "<no_regs>"
    instruction_shape = _instruction_shape(candidate.instruction_text)
    keys = [
        f"steps={remaining_steps}|source={source_token}|opcode={opcode}",
        f"source={source_token}|opcode={opcode}",
        f"opcode={opcode}",
        f"control={instruction_shape['control_role']}",
        f"source={source_token}|control={instruction_shape['control_role']}",
        f"source={source_token}",
        f"ip={state_ip}|opcode={opcode}",
        f"steps={remaining_steps}|opcode={opcode}|rank_bucket={_candidate_rank_bucket(candidate.rank)}",
        f"opcode={opcode}|rank_bucket={_candidate_rank_bucket(candidate.rank)}",
        f"opcode={opcode}|regs={register_signature}",
        f"steps={remaining_steps}|opcode={opcode}|regs={register_signature}",
        f"opcode={opcode}|locality={_candidate_locality(candidate.rank, state_ip)}",
        f"control={instruction_shape['control_role']}|locality={_candidate_locality(candidate.rank, state_ip)}",
    ]
    if include_structural_source_hints:
        keys.extend(
            [
                f"opcode={opcode}|family={instruction_shape['family']}",
                f"opcode={opcode}|family={instruction_shape['family']}|arity={instruction_shape['arity']}",
                f"opcode={opcode}|rank_delta={_rank_delta_bucket(candidate.rank, state_ip)}",
                f"family={instruction_shape['family']}|rank_delta={_rank_delta_bucket(candidate.rank, state_ip)}",
                f"opcode={opcode}|has_input_ref={instruction_shape['has_input_ref']}",
                f"opcode={opcode}|has_output_ref={instruction_shape['has_output_ref']}",
                f"opcode={opcode}|has_label={instruction_shape['has_label']}",
                f"opcode={opcode}|has_immediate={instruction_shape['has_immediate']}",
                f"opcode={opcode}|writes_register={instruction_shape['writes_register']}",
                f"steps={remaining_steps}|family={instruction_shape['family']}|rank_bucket={_candidate_rank_bucket(candidate.rank)}",
            ]
        )
    if before_state_text is not None:
        shape = _state_shape(before_state_text)
        value_relation = _register_value_relation(before_state_text, candidate.instruction_text)
        keys.extend(
            [
                f"opcode={opcode}|ip_bucket={shape['ip_bucket']}",
                f"opcode={opcode}|nonzero_regs={shape['nonzero_regs']}",
                f"opcode={opcode}|touched_mem={shape['touched_mem']}",
                f"opcode={opcode}|has_output={shape['has_output']}",
                f"steps={remaining_steps}|opcode={opcode}|ip_bucket={shape['ip_bucket']}",
                f"control={instruction_shape['control_role']}|ip_bucket={shape['ip_bucket']}",
                f"control={instruction_shape['control_role']}|nonzero_regs={shape['nonzero_regs']}",
                f"opcode={opcode}|value_relation={value_relation}",
                f"control={instruction_shape['control_role']}|value_relation={value_relation}",
            ]
        )
        if instruction_shape["control_role"] == "cond_branch":
            branch_behavior = _branch_behavior(opcode, before_state_text)
            keys.extend(
                [
                    f"opcode={opcode}|branch_behavior={branch_behavior}",
                    f"source={source_token}|branch_behavior={branch_behavior}",
                    f"control=cond_branch|branch_behavior={branch_behavior}|ip_bucket={shape['ip_bucket']}",
                ]
            )
        if include_structural_source_hints:
            overlap = _state_register_overlap(before_state_text, candidate.instruction_text)
            keys.extend(
                [
                    f"opcode={opcode}|mentioned_regs={overlap['mentioned_regs']}",
                    f"opcode={opcode}|mentioned_nonzero_regs={overlap['mentioned_nonzero_regs']}",
                    f"family={instruction_shape['family']}|has_nonzero_reg={overlap['has_nonzero_reg']}",
                    f"family={instruction_shape['family']}|mentions_flagged_reg={overlap['mentions_flagged_reg']}",
                    f"opcode={opcode}|ip_bucket={shape['ip_bucket']}|rank_delta={_rank_delta_bucket(candidate.rank, state_ip)}",
                    f"opcode={opcode}|nonzero_regs={shape['nonzero_regs']}|has_input_ref={instruction_shape['has_input_ref']}",
                    f"opcode={opcode}|touched_mem={shape['touched_mem']}|has_memory_ref={instruction_shape['has_memory_ref']}",
                ]
            )
            if instruction_shape["family"] == "branch":
                flags = _branch_state_cues(before_state_text)
                keys.extend(
                    [
                        f"opcode={opcode}|z={flags['z']}",
                        f"opcode={opcode}|n={flags['n']}",
                        f"opcode={opcode}|c={flags['c']}",
                    ]
                )
    if include_program_name:
        keys.append(f"program={program_token}|steps={remaining_steps}|source={source_token}|opcode={opcode}")
        keys.append(f"program={program_token}|ip={state_ip}|source={source_token}|opcode={opcode}")
    return keys


def train_learned_ranker(
    records: list[dict[str, Any]],
    *,
    smoothing: float = 1.0,
    include_program_name: bool = True,
    include_candidate_source: bool = True,
    include_structural_source_hints: bool = False,
    rank_difficulty_weight: float = 0.0,
    hard_negative_weight: float = 0.0,
    late_step_weight: float = 0.0,
) -> dict[str, Any]:
    feature_counts: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    total_positive = 0.0
    total_examples = 0.0

    for row in records:
        metadata = row["metadata"]
        if not metadata.get("gold_available"):
            continue
        state_lines = row["prompt"].splitlines()
        state_idx = state_lines.index("STATE")
        final_idx = state_lines.index("TARGET_FINAL_STATE")
        before_state_text = "\n".join(state_lines[state_idx + 1 : final_idx])
        ip_line = next(line for line in state_lines if line.startswith("IP="))
        state_ip = int(ip_line.split("=", 1)[1])
        chosen_instruction = row["completion"]
        candidates_payload = metadata["candidates"]
        chosen_rank = next(
            (
                payload.get("rank", index + 1)
                for index, payload in enumerate(candidates_payload)
                if payload["instruction_text"] == chosen_instruction
            ),
            1,
        )
        row_weight = 1.0 + rank_difficulty_weight * max(chosen_rank - 1, 0)
        if metadata["remaining_steps"] == 1:
            row_weight *= 1.0 + late_step_weight

        for index, payload in enumerate(candidates_payload):
            candidate = CandidateInstruction(
                instruction_text=payload["instruction_text"],
                source=payload["source"],
                rank=payload.get("rank", index + 1),
            )
            label = 1.0 if candidate.instruction_text == chosen_instruction else 0.0
            example_weight = row_weight
            if label == 0.0 and candidate.rank < chosen_rank:
                example_weight *= 1.0 + hard_negative_weight
            total_positive += label * example_weight
            total_examples += example_weight
            for key in candidate_feature_keys(
                program_name=metadata["program_name"],
                remaining_steps=metadata["remaining_steps"],
                state_ip=state_ip,
                candidate=candidate,
                include_program_name=include_program_name,
                include_candidate_source=include_candidate_source,
                before_state_text=before_state_text,
                include_structural_source_hints=include_structural_source_hints,
            ):
                feature_counts[key][0] += label * example_weight
                feature_counts[key][1] += example_weight

    base_rate = (total_positive + smoothing) / (total_examples + 2 * smoothing)
    weights: dict[str, float] = {}
    for key, (positive, total) in feature_counts.items():
        rate = (positive + smoothing) / (total + 2 * smoothing)
        weights[key] = math.log(rate / (1.0 - rate)) - math.log(base_rate / (1.0 - base_rate))

    return {
        "format": "learned_branch_ranker_v1",
        "smoothing": smoothing,
        "base_rate": base_rate,
        "weights": weights,
        "trained_examples": int(total_examples),
        "positive_examples": int(total_positive),
        "feature_flags": {
            "include_program_name": include_program_name,
            "include_candidate_source": include_candidate_source,
            "include_structural_source_hints": include_structural_source_hints,
        },
        "training_weights": {
            "rank_difficulty_weight": rank_difficulty_weight,
            "hard_negative_weight": hard_negative_weight,
            "late_step_weight": late_step_weight,
        },
    }


def save_learned_ranker(model: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def load_learned_ranker(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_candidate_with_model(
    candidate: CandidateInstruction,
    *,
    model: dict[str, Any],
    program_name: str,
    remaining_steps: int,
    before_state_text: str,
) -> tuple[float, int]:
    state = parse_state_text(before_state_text)
    score = math.log(model["base_rate"] / (1.0 - model["base_rate"]))
    for key in candidate_feature_keys(
        program_name=program_name,
        remaining_steps=remaining_steps,
        state_ip=state.ip,
        candidate=candidate,
        include_program_name=model.get("feature_flags", {}).get("include_program_name", True),
        include_candidate_source=model.get("feature_flags", {}).get("include_candidate_source", True),
        before_state_text=before_state_text,
        include_structural_source_hints=model.get("feature_flags", {}).get("include_structural_source_hints", False),
    ):
        score += model["weights"].get(key, 0.0)
    return score, -candidate.rank


def rank_candidates_with_model(
    candidates: list[CandidateInstruction],
    *,
    model: dict[str, Any],
    program_name: str,
    remaining_steps: int,
    before_state_text: str,
) -> list[CandidateInstruction]:
    return sorted(
        candidates,
        key=lambda candidate: score_candidate_with_model(
            candidate,
            model=model,
            program_name=program_name,
            remaining_steps=remaining_steps,
            before_state_text=before_state_text,
        ),
        reverse=True,
    )
