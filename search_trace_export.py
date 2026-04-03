from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from branch_ranker import rank_candidates
from candidate_generator import CandidateInstruction, generate_candidates
from vm_transition_verifier import _section_map, load_jsonl, verify_single_step


def _split_for_search_id(search_id: str) -> str:
    bucket = int(hashlib.md5(search_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "val"
    return "train"


def _candidate_payload(candidates: list[CandidateInstruction]) -> list[dict[str, Any]]:
    return [
        {
            "instruction_text": candidate.instruction_text,
            "source": candidate.source,
            "rank": index,
        }
        for index, candidate in enumerate(candidates, start=1)
    ]


def _build_prompt(
    *,
    program_name: str,
    input_values: dict[str, int] | dict[int, int] | None,
    state_text: str,
    final_target_state_text: str,
    remaining_steps: int,
    candidates: list[CandidateInstruction],
) -> str:
    lines = [
        "TASK: branch_ranking_trace_v1",
        "Emit the best next instruction only.",
        f"PROGRAM: {program_name}",
        f"REMAINING_STEPS: {remaining_steps}",
        "INPUT",
    ]
    if input_values:
        for key, value in sorted(((int(k), v) for k, v in input_values.items()), key=lambda item: item[0]):
            lines.append(f"input[{key}]={value}")
    else:
        lines.append("<none>")
    lines.extend(["STATE", state_text, "TARGET_FINAL_STATE", final_target_state_text, "CANDIDATES"])
    for index, candidate in enumerate(candidates, start=1):
        lines.append(f"{index}. {candidate.instruction_text} || source={candidate.source}")
    return "\n".join(lines)


def _decision_record(
    *,
    record: dict[str, Any],
    state_text: str,
    final_target_state_text: str,
    target_next_state_text: str,
    remaining_steps: int,
    candidates: list[CandidateInstruction],
    attempted_candidates: list[dict[str, Any]],
    chosen_instruction: str | None,
    search_outcome: str,
    trace_kind: str,
    search_id: str,
    depth: int,
    search_params: dict[str, Any],
    split_name: str,
) -> dict[str, Any]:
    prompt = _build_prompt(
        program_name=record["program_name"],
        input_values=record.get("input_values"),
        state_text=state_text,
        final_target_state_text=final_target_state_text,
        remaining_steps=remaining_steps,
        candidates=candidates,
    )
    return {
        "prompt": prompt,
        "completion": chosen_instruction or "<no_solution>",
        "metadata": {
            "dataset_type": "search_trace_branch_ranking",
            "trace_kind": trace_kind,
            "split": split_name,
            "search_id": search_id,
            "program_name": record["program_name"],
            "split_family": record.get("split_family"),
            "category": record.get("category"),
            "case_index": record.get("case_index"),
            "window_start": record.get("window_start"),
            "depth": depth,
            "remaining_steps": remaining_steps,
            "search_outcome": search_outcome,
            "gold_available": chosen_instruction is not None,
            "chosen_instruction": chosen_instruction,
            "target_next_state_text": target_next_state_text,
            "target_final_state_text": final_target_state_text,
            "candidates": _candidate_payload(candidates),
            "attempted_candidates": attempted_candidates,
            **search_params,
        },
    }


def build_search_trace_records(
    dataset_path: str | Path,
    *,
    candidate_limit: int,
    candidate_mode: str,
    ranker: str,
    target_mode: str,
    node_budget: int | None,
    split_name: str | None = None,
) -> list[dict[str, Any]]:
    rows = load_jsonl(dataset_path)
    exported: list[dict[str, Any]] = []

    for record in rows:
        sections = _section_map(record["prompt"])
        target_sections = _section_map(record["target"])
        search_id = f"{record['program_name']}:{record.get('case_index', 'na')}:{record.get('window_start', 'na')}"
        record_split = split_name or record.get("split") or _split_for_search_id(search_id)
        search_params = {
            "candidate_limit": candidate_limit,
            "candidate_mode": candidate_mode,
            "ranker": ranker,
            "target_mode": target_mode,
            "node_budget": node_budget,
        }
        nodes_explored = 0
        search_outcome = "failed"

        first_candidates = rank_candidates(
            generate_candidates(
                program_name=record["program_name"],
                before_state_text=sections["S0"],
                mode=candidate_mode,
                limit=candidate_limit,
            ),
            before_state_text=sections["S0"],
            strategy=ranker,
        )

        first_attempts: list[dict[str, Any]] = []
        first_choice: tuple[str, str] | None = None
        second_export: dict[str, Any] | None = None

        for candidate in first_candidates:
            if node_budget is not None and nodes_explored >= node_budget:
                search_outcome = "budget_exhausted"
                break
            first = verify_single_step(
                program_name=record["program_name"],
                input_values=record.get("input_values"),
                before_state_text=sections["S0"],
                instruction_text=candidate.instruction_text,
                target_state_text=target_sections["S1"] if target_mode == "intermediate_oracle" else None,
            )
            nodes_explored += 1
            first_valid = first.valid if target_mode == "intermediate_oracle" else first.error_type in {None, "NONE"}
            first_attempts.append(
                {
                    "instruction_text": candidate.instruction_text,
                    "source": candidate.source,
                    "valid_transition": first_valid,
                    "matches_target_state": first.after_state_text == target_sections["S1"],
                    "error_type": first.error_type,
                }
            )
            if not first_valid:
                continue

            second_candidates = rank_candidates(
                generate_candidates(
                    program_name=record["program_name"],
                    before_state_text=first.after_state_text,
                    mode=candidate_mode,
                    limit=candidate_limit,
                ),
                before_state_text=first.after_state_text,
                strategy=ranker,
            )
            second_attempts: list[dict[str, Any]] = []
            second_choice: str | None = None
            local_outcome = "failed"

            for second_candidate in second_candidates:
                if node_budget is not None and nodes_explored >= node_budget:
                    local_outcome = "budget_exhausted"
                    break
                second = verify_single_step(
                    program_name=record["program_name"],
                    input_values=record.get("input_values"),
                    before_state_text=first.after_state_text,
                    instruction_text=second_candidate.instruction_text,
                    target_state_text=target_sections["S2"] if target_mode in {"intermediate_oracle", "final_state_only"} else None,
                )
                nodes_explored += 1
                solved = second.valid if target_mode == "intermediate_oracle" else (
                    second.after_state_text == target_sections["S2"] and second.error_type in {None, "NONE"}
                )
                second_attempts.append(
                    {
                        "instruction_text": second_candidate.instruction_text,
                        "source": second_candidate.source,
                        "valid_transition": second.error_type in {None, "NONE"},
                        "matches_target_state": second.after_state_text == target_sections["S2"],
                        "error_type": second.error_type,
                    }
                )
                if solved:
                    second_choice = second_candidate.instruction_text
                    local_outcome = "solved"
                    break

            second_export = _decision_record(
                record=record,
                state_text=first.after_state_text,
                final_target_state_text=target_sections["S2"],
                target_next_state_text=target_sections["S2"],
                remaining_steps=1,
                candidates=second_candidates,
                attempted_candidates=second_attempts,
                chosen_instruction=second_choice,
                search_outcome=local_outcome,
                trace_kind="depth2_decision",
                search_id=search_id,
                depth=2,
                search_params=search_params,
                split_name=record_split,
            )

            if local_outcome == "solved":
                first_choice = (candidate.instruction_text, first.after_state_text)
                search_outcome = "solved"
                break
            if local_outcome == "budget_exhausted":
                search_outcome = "budget_exhausted"
                break

        exported.append(
            _decision_record(
                record=record,
                state_text=sections["S0"],
                final_target_state_text=target_sections["S2"],
                target_next_state_text=target_sections["S1"],
                remaining_steps=2,
                candidates=first_candidates,
                attempted_candidates=first_attempts,
                chosen_instruction=first_choice[0] if first_choice else None,
                search_outcome=search_outcome,
                trace_kind="depth1_decision",
                search_id=search_id,
                depth=1,
                search_params=search_params,
                split_name=record_split,
            )
        )
        if second_export is not None:
            exported.append(second_export)

    return exported


def export_search_trace_splits(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    candidate_limit: int,
    candidate_mode: str,
    ranker: str,
    target_mode: str,
    node_budget: int | None,
    split_name: str | None = None,
) -> dict[str, Any]:
    rows = build_search_trace_records(
        dataset_path,
        candidate_limit=candidate_limit,
        candidate_mode=candidate_mode,
        ranker=ranker,
        target_mode=target_mode,
        node_budget=node_budget,
        split_name=split_name,
    )
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        splits[row["metadata"]["split"]].append(row)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        with (output_root / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    manifest = {
        "output_dir": str(output_root),
        "source_dataset_path": str(dataset_path),
        "format": "search_trace_branch_ranking_v1",
        "split_counts": {split: len(records) for split, records in splits.items()},
        "search_params": {
            "candidate_limit": candidate_limit,
            "candidate_mode": candidate_mode,
            "ranker": ranker,
            "target_mode": target_mode,
            "node_budget": node_budget,
            "split_name": split_name,
        },
        "label_summary": {
            split: {
                "with_gold": sum(1 for record in records if record["metadata"]["gold_available"]),
                "no_solution": sum(1 for record in records if not record["metadata"]["gold_available"]),
                "solved": sum(1 for record in records if record["metadata"]["search_outcome"] == "solved"),
                "budget_exhausted": sum(1 for record in records if record["metadata"]["search_outcome"] == "budget_exhausted"),
            }
            for split, records in splits.items()
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return manifest
