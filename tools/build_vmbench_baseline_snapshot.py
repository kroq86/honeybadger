from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import curriculum_gate

REPORTS_ROOT = ROOT / "reports" / "vmbench"
BASELINE_ROOT = REPORTS_ROOT / "baselines"


@dataclass
class BaselineRow:
    slug: str
    run_dir: str
    mode: str
    status: str
    model: str | None
    dataset_root: str | None
    stages: list[str]
    metrics: dict[str, Any]
    next_focus_stage: str | None
    notes: str | None = None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_subset(summary: dict[str, Any]) -> dict[str, Any]:
    stages = summary.get("stages", {})
    subset: dict[str, Any] = {}
    for stage_name, stage_metrics in stages.items():
        subset[stage_name] = {
            "test_avg_field_accuracy": stage_metrics.get("test_avg_field_accuracy"),
            "test_repaired_exact_match_rate": stage_metrics.get("test_repaired_exact_match_rate"),
            "test_exact_match_rate": stage_metrics.get("test_exact_match_rate"),
            "val_avg_field_accuracy": stage_metrics.get("val_avg_field_accuracy"),
            "val_repaired_exact_match_rate": stage_metrics.get("val_repaired_exact_match_rate"),
        }
    return subset


def _row_from_summary(slug: str, mode: str, run_dir: Path, notes: str | None = None) -> BaselineRow:
    summary_path = run_dir / "finetuned_summary.json"
    summary = _load_json(summary_path)
    gate = curriculum_gate.evaluate_summary(summary)
    return BaselineRow(
        slug=slug,
        run_dir=str(run_dir),
        mode=mode,
        status="success",
        model=summary.get("model"),
        dataset_root=summary.get("dataset_root"),
        stages=list(summary.get("stages", {}).keys()),
        metrics=_metrics_subset(summary),
        next_focus_stage=gate.get("next_focus_stage"),
        notes=notes,
    )


def _row_stub(
    slug: str,
    mode: str,
    run_dir: Path,
    status: str,
    notes: str,
) -> BaselineRow:
    return BaselineRow(
        slug=slug,
        run_dir=str(run_dir),
        mode=mode,
        status=status,
        model=None,
        dataset_root=None,
        stages=[],
        metrics={},
        next_focus_stage=None,
        notes=notes,
    )


def build_rows() -> list[BaselineRow]:
    rows: list[BaselineRow] = []

    rows.append(
        _row_from_summary(
            slug="signal_135m_native_runtime_ultra_smoke",
            mode="native_runtime_ultra_smoke",
            run_dir=REPORTS_ROOT / "native_runtime_smoke_20260317" / "signal_host_runtime_ultra_smoke",
            notes="Preferred canonical native host runtime smoke.",
        )
    )
    rows.append(
        _row_from_summary(
            slug="signal_135m_matrix_ultra_smoke",
            mode="matrix_ultra_smoke",
            run_dir=REPORTS_ROOT / "matrix_local_smallweights_20260317" / "signal_135m_ultra_smoke",
            notes="Ultra-smoke local small-weights matrix entry.",
        )
    )
    rows.append(
        _row_from_summary(
            slug="next2_bias_135m_matrix_ultra_smoke",
            mode="matrix_ultra_smoke",
            run_dir=REPORTS_ROOT / "matrix_local_smallweights_20260317" / "next2_bias_135m_ultra_smoke",
            notes="Ultra-smoke local small-weights matrix entry.",
        )
    )
    rows.append(
        _row_from_summary(
            slug="next2_factorized_135m_matrix_ultra_smoke",
            mode="matrix_ultra_smoke",
            run_dir=REPORTS_ROOT / "matrix_local_smallweights_20260317" / "next2_factorized_135m_ultra_smoke",
            notes="Ultra-smoke local small-weights matrix entry.",
        )
    )
    rows.append(
        _row_stub(
            slug="next2_factorized_360m_matrix_ultra_smoke",
            mode="matrix_ultra_smoke",
            run_dir=REPORTS_ROOT / "matrix_local_smallweights_20260317" / "next2_factorized_360m_ultra_smoke",
            status="incomplete",
            notes="Transport closed during native matrix attempt before finetuned summary was written.",
        )
    )
    rows.append(
        _row_stub(
            slug="effects_anchor_135m_matrix_ultra_smoke",
            mode="matrix_ultra_smoke",
            run_dir=REPORTS_ROOT / "matrix_local_smallweights_20260317" / "effects_anchor_135m_ultra_smoke",
            status="error",
            notes=f"Dataset missing: {ROOT / 'datasets' / 'next2_effects_target_anchor_benchmark_v1'}",
        )
    )

    return rows


def build_markdown(rows: list[BaselineRow]) -> str:
    lines = [
        "# VMBench Baseline Snapshot",
        "",
        "Canonical baseline snapshot for the current recovery plan.",
        "",
        "| slug | mode | status | model | stages | key metrics | next focus | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row.metrics:
            metrics_parts = []
            for stage_name, stage_metrics in row.metrics.items():
                metrics_parts.append(
                    f"{stage_name}: test_avg_field_accuracy={stage_metrics.get('test_avg_field_accuracy')}, "
                    f"test_repaired_exact_match_rate={stage_metrics.get('test_repaired_exact_match_rate')}"
                )
            metrics_text = "<br>".join(metrics_parts)
        else:
            metrics_text = "-"
        lines.append(
            f"| {row.slug} | {row.mode} | {row.status} | {row.model or '-'} | "
            f"{', '.join(row.stages) if row.stages else '-'} | {metrics_text} | "
            f"{row.next_focus_stage or '-'} | {row.notes or '-'} |"
        )
    lines.append("")
    lines.append("## Canonical Reference")
    lines.append("")
    lines.append("- `signal_135m_native_runtime_ultra_smoke` is the preferred native host runtime smoke reference.")
    return "\n".join(lines) + "\n"


def main() -> None:
    BASELINE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    payload = {
        "snapshot_name": "latest_matrix_snapshot",
        "generated_from": "tools/build_vmbench_baseline_snapshot.py",
        "rows": [asdict(row) for row in rows],
    }
    json_path = BASELINE_ROOT / "latest_matrix_snapshot.json"
    md_path = BASELINE_ROOT / "latest_matrix_snapshot.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(rows), encoding="utf-8")
    print(json.dumps({"json_path": str(json_path), "md_path": str(md_path)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
