from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_pipeline import TASK_LIBRARY


PROGRAM_META = {
    task.program_name: {
        "category": task.category,
        "split_family": task.split_family,
    }
    for task in TASK_LIBRARY
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a harder search benchmark slice from the two-step windows benchmark.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-window-start", type=int, default=2)
    parser.add_argument("--categories", nargs="+", default=["branch", "loop", "search"])
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with Path(args.input).open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            meta = PROGRAM_META[row["program_name"]]
            if row.get("window_start", 0) < args.min_window_start:
                continue
            if meta["category"] not in set(args.categories):
                continue
            row["category"] = meta["category"]
            row["split_family"] = meta["split_family"]
            dst.write(json.dumps(row, ensure_ascii=True) + "\n")
            kept += 1

    manifest = {
        "source_path": args.input,
        "output_path": str(output_path),
        "min_window_start": args.min_window_start,
        "categories": args.categories,
        "kept_records": kept,
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
