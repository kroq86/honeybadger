from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from branch_ranker import rank_candidates
from candidate_generator import generate_candidates
from search_runner import solve_next_2_steps_record
from vm_transition_verifier import _section_map, load_jsonl, verification_mode_verdict, verify_single_step
from vmbench_product_surface import resolve_mcp_path


def load_benchmark_record(dataset_path: str, record_index: int) -> dict[str, Any]:
    rows = load_jsonl(resolve_mcp_path(dataset_path, must_exist=True))
    if not (0 <= record_index < len(rows)):
        raise IndexError(f"record_index {record_index} out of range for dataset with {len(rows)} rows")
    record = rows[record_index]
    if "S0" not in _section_map(record["prompt"]):
        raise ValueError("benchmark record must contain S0/S1/S2 sections")
    return record


def solve_record_payload(
    record: dict[str, Any],
    *,
    candidate_limit: int,
    candidate_mode: str,
    ranker: str,
    ranker_model_path: str | None,
    verifier_mode: str,
    node_budget: int | None,
) -> dict[str, Any]:
    result = solve_next_2_steps_record(
        record,
        candidate_limit=candidate_limit,
        candidate_mode=candidate_mode,
        ranker=ranker,
        ranker_model_path=ranker_model_path,
        target_mode=verifier_mode,
        node_budget=node_budget,
    )
    return {
        "program_name": record["program_name"],
        "solved": result.solved,
        "successful_path": list(result.successful_path),
        "nodes_explored": result.nodes_explored,
        "budget_exhausted": result.budget_exhausted,
        "attempts": [
            {
                "instruction_text": attempt.instruction_text,
                "source": attempt.source,
                "expected_instruction": attempt.result.expected_instruction,
                "error_type": attempt.result.error_type,
                "notes": list(attempt.result.notes),
            }
            for attempt in result.attempts
        ],
    }


def choose_next_step_payload(
    record: dict[str, Any],
    *,
    candidate_limit: int,
    candidate_mode: str,
    ranker: str,
    ranker_model_path: str | None,
    verifier_mode: str,
) -> dict[str, Any]:
    sections = _section_map(record["prompt"])
    target_sections = _section_map(record["target"])
    before_state_text = sections["S0"]
    candidates = rank_candidates(
        generate_candidates(
            program_name=record["program_name"],
            before_state_text=before_state_text,
            mode=candidate_mode,
            limit=candidate_limit,
        ),
        before_state_text=before_state_text,
        program_name=record["program_name"],
        remaining_steps=2,
        strategy=ranker,
        model_path=ranker_model_path,
    )
    ranked_candidates = []
    winner: dict[str, Any] | None = None
    for index, candidate in enumerate(candidates, start=1):
        result = verify_single_step(
            program_name=record["program_name"],
            input_values=record.get("input_values"),
            before_state_text=before_state_text,
            instruction_text=candidate.instruction_text,
            target_state_text=target_sections["S1"] if verifier_mode in {"intermediate_oracle", "state_diff"} else None,
        )
        accepted, notes = verification_mode_verdict(result, mode=verifier_mode)
        payload = {
            "rank": index,
            "instruction_text": candidate.instruction_text,
            "source": candidate.source,
            "verified": accepted,
            "matches_first_target_state": result.after_state_text == target_sections["S1"],
            "error_type": result.error_type,
            "notes": list(notes),
        }
        ranked_candidates.append(payload)
        if winner is None and accepted:
            winner = payload
    return {
        "program_name": record["program_name"],
        "verifier_mode": verifier_mode,
        "ranker": ranker,
        "winner": winner,
        "top_candidates": ranked_candidates,
    }


def failure_category_payload(
    record: dict[str, Any],
    *,
    candidate_limit: int,
    candidate_mode: str,
    ranker: str,
    ranker_model_path: str | None,
    verifier_mode: str,
    node_budget: int,
) -> dict[str, Any]:
    solve_payload = solve_record_payload(
        record,
        candidate_limit=candidate_limit,
        candidate_mode=candidate_mode,
        ranker=ranker,
        ranker_model_path=ranker_model_path,
        verifier_mode=verifier_mode,
        node_budget=node_budget,
    )
    if solve_payload["solved"]:
        return {"category": "solved", "details": solve_payload}
    sections = _section_map(record["prompt"])
    target_sections = _section_map(record["target"])
    before_state_text = sections["S0"]
    candidates = rank_candidates(
        generate_candidates(
            program_name=record["program_name"],
            before_state_text=before_state_text,
            mode=candidate_mode,
            limit=candidate_limit,
        ),
        before_state_text=before_state_text,
        program_name=record["program_name"],
        remaining_steps=2,
        strategy=ranker,
        model_path=ranker_model_path,
    )
    first_hit_rank = None
    for idx, candidate in enumerate(candidates, start=1):
        result = verify_single_step(
            program_name=record["program_name"],
            input_values=record.get("input_values"),
            before_state_text=before_state_text,
            instruction_text=candidate.instruction_text,
            target_state_text=target_sections["S1"] if verifier_mode in {"intermediate_oracle", "state_diff"} else None,
        )
        accepted, _ = verification_mode_verdict(result, mode=verifier_mode)
        if accepted and result.after_state_text == target_sections["S1"]:
            first_hit_rank = idx
            break
    category = "budget_or_rank_order" if solve_payload["budget_exhausted"] or first_hit_rank is not None else "first_step_unreachable"
    return {
        "category": category,
        "first_hit_rank": first_hit_rank,
        "details": solve_payload,
    }


def demo_reasoning_runtime_payload(
    record: dict[str, Any],
    *,
    dataset_path: str,
    record_index: int,
    candidate_limit: int,
    candidate_mode: str,
    verifier_mode: str,
    learned_model_path: str | None,
    budgets: list[int],
) -> dict[str, Any]:
    next_step = choose_next_step_payload(
        record,
        candidate_limit=min(candidate_limit, 8),
        candidate_mode=candidate_mode,
        ranker="learned" if learned_model_path else "heuristic",
        ranker_model_path=learned_model_path,
        verifier_mode=verifier_mode,
    )
    compare = {
        "policies": [
            {"policy": "none", **solve_record_payload(record, candidate_limit=candidate_limit, candidate_mode=candidate_mode, ranker="none", ranker_model_path=None, verifier_mode=verifier_mode, node_budget=max(budgets))},
            {"policy": "heuristic", **solve_record_payload(record, candidate_limit=candidate_limit, candidate_mode=candidate_mode, ranker="heuristic", ranker_model_path=None, verifier_mode=verifier_mode, node_budget=max(budgets))},
        ]
    }
    if learned_model_path is not None:
        compare["policies"].append(
            {
                "policy": "learned",
                **solve_record_payload(
                    record,
                    candidate_limit=candidate_limit,
                    candidate_mode=candidate_mode,
                    ranker="learned",
                    ranker_model_path=learned_model_path,
                    verifier_mode=verifier_mode,
                    node_budget=max(budgets),
                ),
            }
        )

    budget_curve = []
    for budget in budgets:
        row = {"node_budget": budget, "policies": []}
        for policy, model_path in [("none", None), ("heuristic", None), ("learned", learned_model_path)]:
            if policy == "learned" and model_path is None:
                continue
            solved = solve_record_payload(
                record,
                candidate_limit=candidate_limit,
                candidate_mode=candidate_mode,
                ranker=policy,
                ranker_model_path=model_path,
                verifier_mode=verifier_mode,
                node_budget=budget,
            )
            row["policies"].append(
                {
                    "policy": policy,
                    "solved": solved["solved"],
                    "nodes_explored": solved["nodes_explored"],
                    "budget_exhausted": solved["budget_exhausted"],
                    "successful_path": solved["successful_path"],
                }
            )
        budget_curve.append(row)

    failure_demo = failure_category_payload(
        record,
        candidate_limit=candidate_limit,
        candidate_mode=candidate_mode,
        ranker="none",
        ranker_model_path=None,
        verifier_mode=verifier_mode,
        node_budget=min(budgets),
    )
    return {
        "dataset_path": str(resolve_mcp_path(dataset_path, must_exist=True)),
        "record_index": record_index,
        "scenario_id": f"{record['program_name']}#{record_index}",
        "scenario_label": f"{record['program_name']} #{record_index}",
        "program_name": record["program_name"],
        "verifier_mode": verifier_mode,
        "candidate_mode": candidate_mode,
        "candidate_limit": candidate_limit,
        "budgets": budgets,
        "choose_next_step": next_step,
        "solve_with_budget_curve": budget_curve,
        "failure_demo": failure_demo,
        "compare_policies": compare,
    }


def demo_reasoning_runtime_bundle(
    *,
    dataset_path: str,
    record_indices: list[int],
    candidate_limit: int,
    candidate_mode: str,
    verifier_mode: str,
    learned_model_path: str | None,
    budgets: list[int],
) -> dict[str, Any]:
    scenarios = []
    for record_index in record_indices:
        record = load_benchmark_record(dataset_path, record_index)
        scenarios.append(
            demo_reasoning_runtime_payload(
                record,
                dataset_path=dataset_path,
                record_index=record_index,
                candidate_limit=candidate_limit,
                candidate_mode=candidate_mode,
                verifier_mode=verifier_mode,
                learned_model_path=learned_model_path,
                budgets=budgets,
            )
        )
    default_scenario_id = scenarios[0]["scenario_id"] if scenarios else None
    return {
        "dataset_path": str(resolve_mcp_path(dataset_path, must_exist=True)),
        "record_indices": list(record_indices),
        "default_scenario_id": default_scenario_id,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def write_demo_runtime_payload(
    *,
    dataset_path: str,
    record_index: int | None,
    output_path: str,
    candidate_limit: int = 32,
    candidate_mode: str = "program_global",
    verifier_mode: str = "state_diff",
    budgets: list[int] | None = None,
    learned_model_path: str | None = None,
    record_indices: list[int] | None = None,
) -> dict[str, Any]:
    active_record_indices = list(record_indices or ([] if record_index is None else [record_index]))
    if not active_record_indices:
        raise ValueError("record_index or record_indices must be provided")
    normalized_budgets = budgets or [2, 4, 8, 12]
    payload = demo_reasoning_runtime_bundle(
        dataset_path=dataset_path,
        record_indices=active_record_indices,
        candidate_limit=candidate_limit,
        candidate_mode=candidate_mode,
        verifier_mode=verifier_mode,
        learned_model_path=learned_model_path,
        budgets=normalized_budgets,
    )
    output = resolve_mcp_path(output_path, output=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{\n  \"status\": \"success\",\n  \"payload\": " + json.dumps(payload, ensure_ascii=True, indent=2).replace("\n", "\n  ") + "\n}\n", encoding="utf-8")
    return {"output_path": str(output), "payload": payload}
