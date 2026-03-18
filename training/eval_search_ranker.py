from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learned_branch_ranker import load_learned_ranker, score_candidate_with_model
from candidate_generator import CandidateInstruction


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_ranker(records: list[dict], model_path: str | Path) -> dict:
    model = load_learned_ranker(model_path)
    total = 0
    top1 = 0
    top3 = 0
    gold_rows = 0

    for row in records:
        metadata = row["metadata"]
        if not metadata.get("gold_available"):
            continue
        gold_rows += 1
        state_lines = row["prompt"].splitlines()
        state_idx = state_lines.index("STATE")
        final_idx = state_lines.index("TARGET_FINAL_STATE")
        before_state_text = "\n".join(state_lines[state_idx + 1 : final_idx])
        candidates = [
            CandidateInstruction(
                instruction_text=payload["instruction_text"],
                source=payload["source"],
                rank=payload.get("rank", index + 1),
            )
            for index, payload in enumerate(metadata["candidates"])
        ]
        ranked = sorted(
            candidates,
            key=lambda candidate: score_candidate_with_model(
                candidate,
                model=model,
                program_name=metadata["program_name"],
                remaining_steps=metadata["remaining_steps"],
                before_state_text=before_state_text,
            ),
            reverse=True,
        )
        total += 1
        if ranked and ranked[0].instruction_text == row["completion"]:
            top1 += 1
        if any(candidate.instruction_text == row["completion"] for candidate in ranked[:3]):
            top3 += 1

    return {
        "total_gold_rows": gold_rows,
        "top1_accuracy": (top1 / total) if total else 0.0,
        "top3_accuracy": (top3 / total) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a cheap learned branch ranker on held-out trace rows.")
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = evaluate_ranker(_read_jsonl(Path(args.eval_jsonl)), args.model_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
