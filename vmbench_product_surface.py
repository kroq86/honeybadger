from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

ROOT = Path(__file__).resolve().parent


def _workspace_root() -> Path:
    env_root = os.environ.get("VMBENCH_MCP_WORKSPACE_ROOT", "").strip()
    if env_root:
        return Path(env_root)
    workspace = Path("/workspace")
    if workspace.exists():
        return workspace
    return ROOT


def _product_root() -> Path:
    workspace_root = _workspace_root()
    if (workspace_root / "README.md").exists():
        return workspace_root
    return ROOT


def manifest_payload() -> dict[str, Any]:
    product_root = _product_root()
    return {
        "name": "vmbench",
        "product_direction": (
            "formal VM benchmark + synthetic dataset + eval toolkit"
        ),
        "primary_surface": "mcp",
        "secondary_surface": "cli",
        "core_modules": [
            str(product_root / "reference_vm.py"),
            str(product_root / "dataset_pipeline.py"),
            str(product_root / "baseline_trainer.py"),
            str(product_root / "curriculum_gate.py"),
            str(product_root / "sft_export.py"),
        ],
    }


def repo_map_payload() -> dict[str, str]:
    product_root = _product_root()
    return {
        "repo_product_map": str(product_root / "REPO_PRODUCT_MAP.md"),
        "productization_plan": str(product_root / "PRODUCTIZATION_PLAN.md"),
        "execution_plan": str(product_root / "VMBENCH_EXECUTION_PLAN.md"),
        "readme": str(product_root / "README.md"),
    }


def compare_summary_payload(
    base_summary_path: str | Path,
    candidate_summary_path: str | Path,
) -> dict[str, Any]:
    base_path = Path(base_summary_path)
    candidate_path = Path(candidate_summary_path)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    stage_names = sorted(
        set(base.get("stages", {}).keys())
        | set(candidate.get("stages", {}).keys())
    )
    metric_names = [
        "test_exact_match_rate",
        "test_repaired_exact_match_rate",
        "test_avg_field_accuracy",
        "val_exact_match_rate",
        "val_repaired_exact_match_rate",
        "val_avg_field_accuracy",
    ]
    stages: dict[str, Any] = {}
    for stage in stage_names:
        base_stage = base.get("stages", {}).get(stage, {})
        candidate_stage = candidate.get("stages", {}).get(stage, {})
        stage_payload: dict[str, Any] = {}
        for metric_name in metric_names:
            base_value = base_stage.get(metric_name)
            candidate_value = candidate_stage.get(metric_name)
            if base_value is None and candidate_value is None:
                continue
            stage_payload[metric_name] = {
                "base": base_value,
                "candidate": candidate_value,
                "delta": (
                    None
                    if base_value is None or candidate_value is None
                    else candidate_value - base_value
                ),
            }
        stages[stage] = stage_payload
    return {
        "base_summary_path": str(base_path),
        "candidate_summary_path": str(candidate_path),
        "base_model": base.get("model"),
        "candidate_model": candidate.get("model"),
        "stages": stages,
    }


def _host_workspace_hints() -> list[Path]:
    hints = []
    env_hint = os.environ.get("VMBENCH_HOST_WORKSPACE_ROOT", "").strip()
    if env_hint:
        hints.append(Path(env_hint))
    hints.append(_product_root())
    hints.append(_workspace_root())
    hints.append(Path("/workspace"))
    deduped: list[Path] = []
    seen = set()
    for hint in hints:
        key = str(hint)
        if key not in seen:
            seen.add(key)
            deduped.append(hint)
    return deduped


def _try_map_to_root(path: Path) -> Path | None:
    workspace_root = _workspace_root()
    path_str = str(path)
    for hint in _host_workspace_hints():
        hint_str = str(hint)
        if path_str == hint_str:
            return workspace_root
        if path_str.startswith(hint_str + "/"):
            rel = Path(path_str[len(hint_str) + 1:])
            return workspace_root / rel
    return None


def resolve_cli_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def resolve_mcp_path(
    path_str: str | Path,
    *,
    must_exist: bool = False,
    output: bool = False,
) -> Path:
    workspace_root = _workspace_root()
    path = Path(path_str)
    candidate: Path
    if path.is_absolute():
        try:
            path.relative_to(ROOT)
            candidate = path
        except ValueError:
            try:
                path.relative_to(workspace_root)
                candidate = path
            except ValueError:
                mapped = _try_map_to_root(path)
                if mapped is not None:
                    candidate = mapped
                else:
                    raise ValueError(
                        "path must be inside the mounted workspace rooted at "
                        f"{workspace_root}; got absolute path outside workspace: "
                        f"{path}"
                    )
    else:
        candidate = workspace_root / path

    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)

    if output:
        candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def running_inside_container() -> bool:
    return Path("/.dockerenv").exists()


def normalize_mcp_runtime_host(host: str) -> str:
    parsed = urlparse(host)
    hostname = parsed.hostname
    if not hostname:
        return host
    if not running_inside_container():
        return host
    if hostname not in {"127.0.0.1", "localhost"}:
        return host

    netloc = "host.docker.internal"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username:
        credentials = parsed.username
        if parsed.password:
            credentials = f"{credentials}:{parsed.password}"
        netloc = f"{credentials}@{netloc}"
    return urlunparse(parsed._replace(netloc=netloc))
