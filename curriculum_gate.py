from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


GATES: Dict[str, Dict[str, Dict[str, float]]] = {
    "single_step": {
        "signal": {
            "val_avg_field_accuracy": 0.85,
            "test_avg_field_accuracy": 0.85,
        },
        "pass": {
            "val_repaired_exact_match_rate": 0.70,
            "test_repaired_exact_match_rate": 0.70,
        },
    },
    "next_2_steps": {
        "signal": {
            "val_avg_field_accuracy": 0.80,
            "test_avg_field_accuracy": 0.80,
        },
        "pass": {
            "val_repaired_exact_match_rate": 0.50,
            "test_repaired_exact_match_rate": 0.50,
        },
    },
    "short_trace": {
        "signal": {
            "val_repaired_exact_match_rate": 0.05,
            "test_repaired_exact_match_rate": 0.05,
        },
        "pass": {
            "val_repaired_exact_match_rate": 0.25,
            "test_repaired_exact_match_rate": 0.25,
        },
    },
    "terminal_state": {
        "signal": {
            "val_avg_field_accuracy": 0.85,
            "test_avg_field_accuracy": 0.85,
        },
        "pass": {
            "val_repaired_exact_match_rate": 0.50,
            "test_repaired_exact_match_rate": 0.50,
        },
    },
}

ACTIVE_CURRICULUM = ["single_step", "next_2_steps", "short_trace", "terminal_state"]


def _metric_ok(value: Any, threshold: float) -> bool:
    if value is None:
        return False
    return float(value) >= threshold


def evaluate_stage(stage_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    stage_gates = GATES.get(stage_name)
    if not stage_gates:
        return {
            "signal": False,
            "pass": False,
            "reason": "no_gate_defined",
            "checked_metrics": metrics,
        }

    signal_checks = {
        key: {
            "value": metrics.get(key),
            "threshold": threshold,
            "ok": _metric_ok(metrics.get(key), threshold),
        }
        for key, threshold in stage_gates["signal"].items()
    }
    pass_checks = {
        key: {
            "value": metrics.get(key),
            "threshold": threshold,
            "ok": _metric_ok(metrics.get(key), threshold),
        }
        for key, threshold in stage_gates["pass"].items()
    }

    signal_ok = all(check["ok"] for check in signal_checks.values())
    pass_ok = all(check["ok"] for check in pass_checks.values())

    return {
        "signal": signal_ok,
        "pass": pass_ok,
        "signal_checks": signal_checks,
        "pass_checks": pass_checks,
    }


def evaluate_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    stage_metrics = summary.get("stages", {})
    per_stage: Dict[str, Any] = {}
    next_focus = None

    for stage_name in ACTIVE_CURRICULUM:
        stage_result = evaluate_stage(stage_name, stage_metrics.get(stage_name, {}))
        present = stage_name in stage_metrics
        per_stage[stage_name] = {
            "present": present,
            **stage_result,
        }
        if next_focus is None and (not present or not stage_result["signal"]):
            next_focus = stage_name

    if next_focus is None:
        for stage_name in ACTIVE_CURRICULUM:
            if not per_stage[stage_name]["pass"]:
                next_focus = stage_name
                break

    return {
        "model": summary.get("model"),
        "active_curriculum": ACTIVE_CURRICULUM,
        "next_focus_stage": next_focus,
        "stages": per_stage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate execution-fidelity gates for a baseline summary.")
    parser.add_argument("summary_path", help="Path to baseline summary.json")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary_path).read_text(encoding="utf-8"))
    result = evaluate_summary(summary)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
