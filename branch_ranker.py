from __future__ import annotations

from functools import lru_cache

from candidate_generator import CandidateInstruction
from learned_branch_ranker import load_learned_ranker, rank_candidates_with_model
from vm_transition_verifier import parse_state_text


SOURCE_PRIOR = {
    "current_ip": 100,
    "jump_target": 95,
    "jump_fallthrough": 90,
    "next_ip": 70,
    "next2_ip": 60,
    "prev_ip": 40,
    "program_scan": 10,
}

BRANCH_OPS = {"JMP", "JZ", "JNZ", "JL", "JLE", "JG", "JGE"}


def _score_candidate(candidate: CandidateInstruction, before_state_text: str) -> tuple[int, int]:
    state = parse_state_text(before_state_text)
    opcode = candidate.instruction_text.split()[0]
    score = SOURCE_PRIOR.get(candidate.source, 0)

    if state.halted and opcode == "HALT":
        score += 50
    if opcode == "HALT" and not state.halted:
        score -= 30
    if opcode in BRANCH_OPS and candidate.source == "program_scan":
        score -= 5
    if opcode in {"READ", "CONST"} and state.ip <= 2:
        score += 5
    return score, -candidate.rank


def rank_candidates(
    candidates: list[CandidateInstruction],
    *,
    before_state_text: str,
    program_name: str | None = None,
    remaining_steps: int | None = None,
    strategy: str = "heuristic",
    model_path: str | None = None,
) -> list[CandidateInstruction]:
    if strategy == "none":
        return list(candidates)
    if strategy == "heuristic":
        return sorted(candidates, key=lambda candidate: _score_candidate(candidate, before_state_text), reverse=True)
    if strategy == "learned":
        if model_path is None or program_name is None or remaining_steps is None:
            raise ValueError("learned ranker requires model_path, program_name, and remaining_steps")
        return rank_candidates_with_model(
            candidates,
            model=_cached_model(model_path),
            program_name=program_name,
            remaining_steps=remaining_steps,
            before_state_text=before_state_text,
        )
    else:
        raise ValueError(f"unsupported ranker strategy: {strategy}")


@lru_cache(maxsize=4)
def _cached_model(model_path: str) -> dict:
    return load_learned_ranker(model_path)
