from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmbench_demo_runtime import write_demo_runtime_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate docs/demo-runtime-payload.json from the current vmbench reasoning runtime.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--record-index", type=int, default=None)
    parser.add_argument("--record-indices", nargs="+", type=int, default=None)
    parser.add_argument("--output-path", default="docs/demo-runtime-payload.json")
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--candidate-mode", default="program_global")
    parser.add_argument("--verifier-mode", default="state_diff")
    parser.add_argument("--budgets", nargs="+", type=int, default=[2, 4, 8, 12])
    parser.add_argument("--learned-model-path", default=None)
    args = parser.parse_args()

    result = write_demo_runtime_payload(
        dataset_path=args.dataset_path,
        record_index=args.record_index,
        record_indices=args.record_indices,
        output_path=args.output_path,
        candidate_limit=args.candidate_limit,
        candidate_mode=args.candidate_mode,
        verifier_mode=args.verifier_mode,
        budgets=args.budgets,
        learned_model_path=args.learned_model_path,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
