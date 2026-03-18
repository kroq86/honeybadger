from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vm_transition_verifier import replay_dataset

REPORT_ROOT = ROOT / "reports" / "vmbench" / "verifier"


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    valid = sum(1 for row in rows if row["valid"])
    return {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "valid_rate": (valid / total) if total else 0.0,
    }


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    single_path = ROOT / "datasets" / "matrix_seed11" / "single_step" / "test.jsonl"
    next2_path = ROOT / "datasets" / "matrix_seed11" / "next_2_steps" / "test.jsonl"

    single_rows = replay_dataset(single_path)
    next2_rows = replay_dataset(next2_path)

    payload = {
        "snapshot_name": "matrix_seed11_transition_replay",
        "single_step": summarize(single_rows),
        "next_2_steps": summarize(next2_rows),
        "single_step_path": str(single_path),
        "next_2_steps_path": str(next2_path),
    }
    out = REPORT_ROOT / "matrix_seed11_transition_replay.json"
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_path": str(out), "payload": payload}, ensure_ascii=True))


if __name__ == "__main__":
    main()
