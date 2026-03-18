from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learned_branch_ranker import save_learned_ranker, train_learned_ranker


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a cheap learned branch ranker from exported search traces.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--drop-program-name", action="store_true")
    parser.add_argument("--drop-candidate-source", action="store_true")
    parser.add_argument("--structural-source-hints", action="store_true")
    args = parser.parse_args()

    model = train_learned_ranker(
        _read_jsonl(Path(args.train_jsonl)),
        smoothing=args.smoothing,
        include_program_name=not args.drop_program_name,
        include_candidate_source=not args.drop_candidate_source,
        include_structural_source_hints=args.structural_source_hints,
    )
    save_learned_ranker(model, args.output_model)


if __name__ == "__main__":
    main()
