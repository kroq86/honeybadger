from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

# Log of every tool call (LLM steps) so you can see from MCP what the agent actually did
CALL_LOG: deque = deque(maxlen=200)
_CALL_LOG_FILE = (Path(__file__).resolve().parents[2] / ".vmbench_llm_call_log.jsonl").as_posix()


def _to_dict(obj: Any) -> dict | None:
    """Convert MCP/Pydantic argument object to a plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        d = getattr(obj, "__dict__")
        if isinstance(d, dict):
            return d
    # Mapping-like (e.g. MCP TypedDict or namespace)
    try:
        return dict(obj)
    except (TypeError, ValueError):
        pass
    if hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        try:
            return {k: obj[k] for k in obj.keys()}
        except (TypeError, KeyError):
            pass
    return None


def _unwrap_cursor_args(args: tuple, kwargs: dict) -> tuple[tuple, dict, dict]:
    """Unwrap Cursor/MCP arguments; return (args, kwargs, request_meta). MCP _meta is logged when present."""
    meta: dict = {}
    # Single positional = the tool's arguments (dict or params object). Unwrap even if kwargs present.
    if len(args) == 1:
        one = args[0]
        payload = _to_dict(one)
        if payload is None and hasattr(one, "arguments"):
            payload = _to_dict(getattr(one, "arguments"))
        # MCP SDK may pass params with "arguments" and optional "_meta" (reasoning, trace, etc.)
        if payload is not None and "arguments" in payload and set(payload.keys()) <= {"arguments", "name", "_meta"}:
            meta = dict(payload.get("_meta") or {})
            inner = payload.get("arguments")
            payload = inner if isinstance(inner, dict) else _to_dict(inner)
        if payload is not None and isinstance(payload, dict):
            # Sub-case: Cursor envelope with "args" and "kwargs" (sometimes JSON strings)
            if set(payload.keys()) <= {"args", "kwargs"} and "args" in payload and "kwargs" in payload:
                a, k = payload.get("args"), payload.get("kwargs")
                if isinstance(a, str):
                    a = json.loads(a) if (a and a.strip()) else []
                if isinstance(k, str):
                    k = json.loads(k) if (k and k.strip()) else {}
                if isinstance(a, list) and len(a) == 1 and a[0] in (None, {}):
                    a = []
                return (tuple(a) if isinstance(a, list) else (a,) if a is not None else (),), (k if isinstance(k, dict) else {}), meta
            # Otherwise treat the whole dict as kwargs for the tool
            return (), payload, meta
        # Single positional we couldn't convert: call tool with no args (fixes "takes 0 positional but 1 given")
        return (), {}, meta
    # Envelope as kwargs (Cursor sends 0 positional, 2 kwargs: "args" and "kwargs")
    if len(args) == 0 and "args" in kwargs and "kwargs" in kwargs:
        a, k = kwargs.get("args"), kwargs.get("kwargs")
        if isinstance(a, str):
            a = json.loads(a) if (a and a.strip()) else []
        if isinstance(k, str):
            k = json.loads(k) if (k and k.strip()) else {}
        # Cursor may send args="[{}]" or "[null]" → treat as no positionals for no-param tools
        if isinstance(a, list) and len(a) == 1 and a[0] in (None, {}):
            a = []
        return (tuple(a) if isinstance(a, list) else (a,) if a is not None else (),), (k if isinstance(k, dict) else {}), meta
    return args, kwargs, meta


def _llm_call_log(tool_name: str):
    """Decorator: record tool name, args, timestamp, status, and result/error for every call."""
    def decorator(f):
        def wrapper(*args, **kwargs):
            args, kwargs, request_meta = _unwrap_cursor_args(args, kwargs)
            entry = {
                "tool": tool_name,
                "args": dict(kwargs),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if request_meta:
                entry["request_meta"] = request_meta
            try:
                result = f(*args, **kwargs)
                entry["status"] = "success"
                raw = json.dumps(result, default=str) if isinstance(result, dict) else str(result)
                # No truncation: store full result so nested call-log output shows complete result_preview
                max_preview = 500_000
                entry["result_preview"] = raw[:max_preview] if raw else "(empty)"
                if len(raw) > max_preview:
                    entry["result_preview"] += "…"
                    entry["result_preview_truncated"] = True
                CALL_LOG.append(entry)
                _append_call_log_file(entry)
                return result
            except Exception as e:
                entry["status"] = "error"
                entry["error"] = str(e)
                CALL_LOG.append(entry)
                _append_call_log_file(entry)
                raise
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator


def _append_call_log_file(entry: dict[str, Any]) -> None:
    """Append one call entry to a JSONL file so you can tail -f from the host (e.g. Docker)."""
    import os
    path = os.environ.get("VMBENCH_MCP_CALL_LOG", _CALL_LOG_FILE)
    if not path or path == "0":
        return
    try:
        with open(path, "a") as fp:
            fp.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP

import curriculum_gate
from baseline_trainer import OllamaClient, check_ollama_available, evaluate_stage, write_json
from branch_ranker import rank_candidates
from candidate_generator import generate_candidates
from dataset_pipeline import (
    build_dataset_card,
    build_generation_config,
    build_manifest,
    generate_next_k_steps_examples,
    generate_short_trace_examples,
    generate_single_step_examples,
    generate_terminal_state_examples,
    split_records,
    write_generation_config,
    write_jsonl,
    write_manifest,
)
from learned_branch_ranker import load_learned_ranker
from search_runner import solve_next_2_steps_record
from sft_export import build_manifest as build_sft_manifest
from sft_export import export_sft_splits, write_jsonl as write_sft_jsonl
from vm_transition_verifier import _section_map, load_jsonl, verification_mode_verdict, verify_single_step
from vmbench_demo_runtime import (
    choose_next_step_payload as shared_choose_next_step_payload,
    demo_reasoning_runtime_payload as shared_demo_reasoning_runtime_payload,
    failure_category_payload as shared_failure_category_payload,
    load_benchmark_record as shared_load_benchmark_record,
    solve_record_payload as shared_solve_record_payload,
    write_demo_runtime_payload as shared_write_demo_runtime_payload,
)
from vmbench_product_surface import (
    compare_summary_payload as shared_compare_summary_payload,
    manifest_payload as vmbench_manifest_payload,
    normalize_mcp_runtime_host,
    repo_map_payload as vmbench_repo_map_payload,
    resolve_mcp_path,
)


server = FastMCP(
    name="vmbench",
    instructions=(
        "vmbench is a formal VM benchmark and synthetic training/eval toolkit. "
        "Use it to generate benchmark datasets, run bounded baseline evaluation, "
        "evaluate gate summaries, and export SFT prompt/completion data."
    ),
)


# File written by vmbench_set_next_reasoning; proxy reads it and attaches to the *next* tool call meta (then clears it).
_NEXT_REASONING_FILE = ROOT / ".vmbench_next_reasoning.txt"

TOOL_NAMES = [
    "vmbench_manifest",
    "vmbench_generate_dataset",
    "vmbench_run_baseline",
    "vmbench_run_checkpoint_eval",
    "vmbench_gate_summary",
    "vmbench_export_sft",
    "vmbench_repo_map",
    "vmbench_compare_reports",
    "vmbench_choose_next_step",
    "vmbench_solve_with_budget",
    "vmbench_explain_failure",
    "vmbench_compare_policies",
    "vmbench_demo_reasoning_runtime",
    "vmbench_generate_demo_payload",
    "vmbench_status",
    "vmbench_set_next_reasoning",
    "vmbench_llm_call_log",
]


def _wrap_success(payload: dict[str, Any]) -> dict[str, Any]:
    return {"status": "success", "payload": payload}


def _wrap_error(message: str, category: str = "validation", is_retryable: bool = False) -> dict[str, Any]:
    return {
        "status": "error",
        "error": {
            "message": message,
            "category": category,
            "isRetryable": is_retryable,
        },
    }


def _error_category(exc: Exception) -> tuple[str, bool]:
    if isinstance(exc, RuntimeError):
        return "transient", True
    return "validation", False


def _details_for_exception(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _generate_dataset(
    output_dir: str = "datasets/mvp",
    single_step_limit: int = 256,
    next_2_steps_limit: int = 128,
    short_trace_limit: int = 128,
    terminal_state_limit: int = 128,
    seed: int = 7,
) -> dict[str, Any]:
    output_dir_path = resolve_mcp_path(output_dir, output=True)
    generated = {
        "single_step": generate_single_step_examples(limit=single_step_limit, seed=seed),
        "next_2_steps": generate_next_k_steps_examples(
            limit=next_2_steps_limit,
            seed=seed,
            next_k_steps=2,
            dataset_type="next_2_steps",
        ),
        "short_trace": generate_short_trace_examples(limit=short_trace_limit, seed=seed),
        "terminal_state": generate_terminal_state_examples(limit=terminal_state_limit, seed=seed),
    }
    split_datasets = {
        dataset_type: split_records(records, seed=seed)
        for dataset_type, records in generated.items()
    }
    for dataset_type, splits in split_datasets.items():
        for split_name, records in splits.items():
            write_jsonl(output_dir_path / dataset_type / f"{split_name}.jsonl", records)
    args = argparse.Namespace(
        output_dir=output_dir,
        single_step_limit=single_step_limit,
        next_2_steps_limit=next_2_steps_limit,
        short_trace_limit=short_trace_limit,
        terminal_state_limit=terminal_state_limit,
        seed=seed,
    )
    manifest = build_manifest(split_datasets, seed=seed, output_dir=output_dir_path)
    config = build_generation_config(args)
    write_manifest(output_dir_path / "manifest.json", manifest)
    write_generation_config(output_dir_path / "generation_config.json", config)
    (output_dir_path / "DATASET_CARD.md").write_text(build_dataset_card(manifest, config), encoding="utf-8")
    return {
        "ok": True,
        "output_dir": str(output_dir_path),
        "dataset_types": list(split_datasets.keys()),
        "manifest_path": str(output_dir_path / "manifest.json"),
    }


def _summary_from_stage_reports(config: dict[str, Any], stage_reports: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config["model"],
        "requested_host": config["requested_host"],
        "effective_host": config["effective_host"],
        "train_shots": config["train_shots"],
        "eval_limit": config["eval_limit"],
        "temperature": config["temperature"],
        "num_predict": config["num_predict"],
        "max_trace_steps": config["max_trace_steps"],
        "timeout_seconds": config["timeout_seconds"],
        "stages": {
            stage: {
                "val_exact_match_rate": report["val"]["exact_match_rate"],
                "test_exact_match_rate": report["test"]["exact_match_rate"],
                "val_repaired_exact_match_rate": report["val"]["repaired_exact_match_rate"],
                "test_repaired_exact_match_rate": report["test"]["repaired_exact_match_rate"],
                "val_avg_field_accuracy": report["val"]["avg_field_accuracy"],
                "test_avg_field_accuracy": report["test"]["avg_field_accuracy"],
                "val_count": report["val"]["count"],
                "test_count": report["test"]["count"],
            }
            for stage, report in stage_reports.items()
        },
    }


def _run_baseline(
    dataset_root: str = "datasets/mvp",
    report_dir: str = "reports/baseline",
    model: str = "llama3.2:latest",
    host: str = "http://127.0.0.1:11434",
    train_shots: int = 1,
    eval_limit: int = 2,
    temperature: float = 0.0,
    num_predict: int = 192,
    max_trace_steps: int = 6,
    timeout_seconds: int = 12,
    stages: list[str] | None = None,
) -> dict[str, Any]:
    stages = stages or ["single_step", "next_2_steps"]
    runtime_host = normalize_mcp_runtime_host(host)
    available, details = check_ollama_available(runtime_host, min(timeout_seconds, 5))
    if not available:
        raise RuntimeError(f"Ollama runtime is unavailable at {runtime_host}: {details}")

    dataset_root_path = resolve_mcp_path(dataset_root, must_exist=True)
    report_dir_path = resolve_mcp_path(report_dir, output=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_slug = model.replace(":", "-").replace("/", "-")
    run_dir = report_dir_path / f"{timestamp}-{model_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    stage_reports: dict[str, Any] = {}
    for stage in stages:
        client = OllamaClient(
            host=runtime_host,
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            timeout_seconds=timeout_seconds,
        )
        stage_reports[stage] = evaluate_stage(
            client=client,
            dataset_root=dataset_root_path,
            stage=stage,
            train_shots=train_shots,
            eval_limit=eval_limit,
            max_trace_steps=max_trace_steps,
        )
        write_json(run_dir / f"{stage}.json", stage_reports[stage])

    summary = _summary_from_stage_reports(
        {
            "model": model,
            "requested_host": host,
            "effective_host": runtime_host,
            "train_shots": train_shots,
            "eval_limit": eval_limit,
            "temperature": temperature,
            "num_predict": num_predict,
            "max_trace_steps": max_trace_steps,
            "timeout_seconds": timeout_seconds,
        },
        stage_reports,
    )
    write_json(run_dir / "summary.json", summary)
    return {
        "ok": True,
        "run_dir": str(run_dir),
        "summary_path": str(run_dir / "summary.json"),
        "stages": stages,
        "requested_host": host,
        "effective_host": runtime_host,
    }


def _load_eval_checkpoint_module() -> Any:
    workspace_module_path = resolve_mcp_path("training/eval_checkpoint.py", must_exist=True)
    spec = importlib.util.spec_from_file_location("vmbench_eval_checkpoint", workspace_module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load eval module from {workspace_module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_summary_payload(
    *,
    model_label: str,
    stage_metrics: dict[str, Any],
    eval_limit: int,
    num_predict: int,
    max_trace_steps: int,
    train_shots: int,
    dataset_root: Path,
    adapter_path: str | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model_label,
        "eval_limit": eval_limit,
        "num_predict": num_predict,
        "max_trace_steps": max_trace_steps,
        "train_shots": train_shots,
        "dataset_root": str(dataset_root),
        "stages": stage_metrics,
    }
    if adapter_path is not None:
        payload["adapter_path"] = adapter_path
    return payload


def _resolve_freeze_paths_for_mcp(freeze: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    resolved = json.loads(json.dumps(freeze))
    paths = dict(resolved.get("paths", {}))
    for key, value in list(paths.items()):
        if not isinstance(value, str):
            continue
        try:
            paths[key] = str(resolve_mcp_path(value, must_exist=key != "output_dir"))
        except (ValueError, FileNotFoundError):
            continue
    resolved["paths"] = paths
    return resolved, Path(paths["output_dir"])


def _run_checkpoint_eval(
    phase2_freeze_path: str,
    dataset_root: str | None = None,
    report_dir: str = "reports/checkpoint_eval",
    stages: list[str] | None = None,
    splits: list[str] | None = None,
    eval_limit: int | None = None,
    num_predict: int | None = None,
    train_shots: int | None = None,
    max_trace_steps: int | None = None,
    evaluate_base: bool = True,
    fast_smoke: bool = False,
    run_name: str | None = None,
) -> dict[str, Any]:
    eval_module = _load_eval_checkpoint_module()
    freeze_path = resolve_mcp_path(phase2_freeze_path, must_exist=True)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    resolved_freeze, adapter_dir = _resolve_freeze_paths_for_mcp(freeze)

    benchmark_root = dataset_root or resolved_freeze["paths"]["benchmark_dataset_root"]
    benchmark_root_path = resolve_mcp_path(benchmark_root, must_exist=True)
    report_dir_path = resolve_mcp_path(report_dir, output=True)

    runner_defaults = dict(resolved_freeze["execution_benchmark"]["runner_defaults"])
    resolved_stages = stages or list(resolved_freeze["execution_benchmark"]["stages"])
    resolved_eval_limit = int(eval_limit if eval_limit is not None else runner_defaults["eval_limit"])
    resolved_num_predict = int(num_predict if num_predict is not None else runner_defaults["num_predict"])
    resolved_train_shots = int(train_shots if train_shots is not None else runner_defaults["train_shots"])
    resolved_max_trace_steps = int(
        max_trace_steps if max_trace_steps is not None else runner_defaults["max_trace_steps"]
    )
    resolved_splits = splits or ["val", "test"]
    resolved_evaluate_base = evaluate_base
    if fast_smoke:
        resolved_splits = splits or ["test"]
        resolved_evaluate_base = False if evaluate_base else evaluate_base
        resolved_eval_limit = int(eval_limit if eval_limit is not None else 2)
        resolved_num_predict = int(num_predict if num_predict is not None else 24)
        resolved_max_trace_steps = int(max_trace_steps if max_trace_steps is not None else min(4, resolved_max_trace_steps))

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_slug = run_name or f"{timestamp}-{adapter_dir.name}"
    run_dir = report_dir_path / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_freeze_path = run_dir / "resolved_phase2_freeze.json"
    write_json(resolved_freeze_path, resolved_freeze)

    config = {
        "phase2_freeze_path": str(resolved_freeze_path),
        "benchmark_dataset_root": str(benchmark_root_path),
        "benchmark_stages": resolved_stages,
        "benchmark_splits": resolved_splits,
        "evaluate_base": resolved_evaluate_base,
        "metrics": list(resolved_freeze["execution_benchmark"]["metrics"]),
        "runner_defaults": {
            "train_shots": resolved_train_shots,
            "eval_limit": resolved_eval_limit,
            "num_predict": resolved_num_predict,
            "max_trace_steps": resolved_max_trace_steps,
            "timeout_seconds": int(runner_defaults["timeout_seconds"]),
        },
        "base_report_path": str(run_dir / "base_report.json"),
        "finetuned_report_path": str(run_dir / "finetuned_report.json"),
        "comparison_report_path": str(run_dir / "comparison.md"),
    }
    eval_module.validate_eval_config(config)
    result = eval_module.run_eval(config)

    base_summary = _checkpoint_summary_payload(
        model_label=result["base_model_name"],
        stage_metrics=result["base"]["stages"],
        eval_limit=resolved_eval_limit,
        num_predict=resolved_num_predict,
        max_trace_steps=resolved_max_trace_steps,
        train_shots=resolved_train_shots,
        dataset_root=benchmark_root_path,
    )
    finetuned_summary = _checkpoint_summary_payload(
        model_label=f"{result['base_model_name']}+adapter:{adapter_dir.name}",
        stage_metrics=result["finetuned"]["stages"],
        eval_limit=resolved_eval_limit,
        num_predict=resolved_num_predict,
        max_trace_steps=resolved_max_trace_steps,
        train_shots=resolved_train_shots,
        dataset_root=benchmark_root_path,
        adapter_path=result["adapter_path"],
    )

    base_summary_path = run_dir / "base_summary.json"
    finetuned_summary_path = run_dir / "finetuned_summary.json"
    write_json(base_summary_path, base_summary)
    write_json(finetuned_summary_path, finetuned_summary)
    write_json(run_dir / "resolved_config.json", config)

    return {
        "ok": True,
        "run_dir": str(run_dir),
        "base_model_name": result["base_model_name"],
        "adapter_path": result["adapter_path"],
        "base_summary_path": str(base_summary_path),
        "finetuned_summary_path": str(finetuned_summary_path),
        "base_report_path": config["base_report_path"],
        "finetuned_report_path": config["finetuned_report_path"],
        "comparison_report_path": config["comparison_report_path"],
        "resolved_config_path": str(run_dir / "resolved_config.json"),
        "stages": resolved_stages,
        "splits": resolved_splits,
        "evaluate_base": resolved_evaluate_base,
        "fast_smoke": fast_smoke,
        "eval_limit": resolved_eval_limit,
        "num_predict": resolved_num_predict,
    }


def _load_benchmark_record(dataset_path: str, record_index: int) -> dict[str, Any]:
    return shared_load_benchmark_record(dataset_path, record_index)


def _solve_record_payload(
    record: dict[str, Any],
    *,
    candidate_limit: int,
    candidate_mode: str,
    ranker: str,
    ranker_model_path: str | None,
    verifier_mode: str,
    node_budget: int | None,
) -> dict[str, Any]:
    return shared_solve_record_payload(
        record,
        candidate_limit=candidate_limit,
        candidate_mode=candidate_mode,
        ranker=ranker,
        ranker_model_path=ranker_model_path,
        verifier_mode=verifier_mode,
        node_budget=node_budget,
    )


def _choose_next_step_payload(
    record: dict[str, Any],
    *,
    candidate_limit: int,
    candidate_mode: str,
    ranker: str,
    ranker_model_path: str | None,
    verifier_mode: str,
) -> dict[str, Any]:
    return shared_choose_next_step_payload(
        record,
        candidate_limit=candidate_limit,
        candidate_mode=candidate_mode,
        ranker=ranker,
        ranker_model_path=ranker_model_path,
        verifier_mode=verifier_mode,
    )


def _failure_category_payload(
    record: dict[str, Any],
    *,
    candidate_limit: int,
    candidate_mode: str,
    ranker: str,
    ranker_model_path: str | None,
    verifier_mode: str,
    node_budget: int,
) -> dict[str, Any]:
    return shared_failure_category_payload(
        record,
        candidate_limit=candidate_limit,
        candidate_mode=candidate_mode,
        ranker=ranker,
        ranker_model_path=ranker_model_path,
        verifier_mode=verifier_mode,
        node_budget=node_budget,
    )


def _demo_reasoning_runtime_payload(
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
    return shared_demo_reasoning_runtime_payload(
        record,
        dataset_path=dataset_path,
        record_index=record_index,
        candidate_limit=candidate_limit,
        candidate_mode=candidate_mode,
        verifier_mode=verifier_mode,
        learned_model_path=learned_model_path,
        budgets=budgets,
    )


@server.tool(description="Return product metadata and core module paths for vmbench.")
@_llm_call_log("vmbench_manifest")
def vmbench_manifest(_request: Any = None) -> dict[str, Any]:
    return _wrap_success(vmbench_manifest_payload())


@server.tool(description="Generate vmbench benchmark datasets.")
@_llm_call_log("vmbench_generate_dataset")
def vmbench_generate_dataset(
    output_dir: str = "datasets/mvp",
    single_step_limit: int = 256,
    next_2_steps_limit: int = 128,
    short_trace_limit: int = 128,
    terminal_state_limit: int = 128,
    seed: int = 7,
) -> dict[str, Any]:
    try:
        payload = _generate_dataset(
            output_dir=output_dir,
            single_step_limit=single_step_limit,
            next_2_steps_limit=next_2_steps_limit,
            short_trace_limit=short_trace_limit,
            terminal_state_limit=terminal_state_limit,
            seed=seed,
        )
        return _wrap_success(payload)
    except Exception as exc:
        category, is_retryable = _error_category(exc)
        payload = _wrap_error(str(exc), category=category, is_retryable=is_retryable)
        payload["error"]["details"] = _details_for_exception(exc)
        payload["error"]["next_action"] = "Use a path inside the mounted workspace or pass a repo-relative path."
        return payload


@server.tool(description="Run a bounded local baseline evaluation for vmbench.")
@_llm_call_log("vmbench_run_baseline")
def vmbench_run_baseline(
    dataset_root: str = "datasets/mvp",
    report_dir: str = "reports/baseline",
    model: str = "llama3.2:latest",
    host: str = "http://127.0.0.1:11434",
    train_shots: int = 1,
    eval_limit: int = 2,
    temperature: float = 0.0,
    num_predict: int = 192,
    max_trace_steps: int = 6,
    timeout_seconds: int = 12,
    stages: list[str] | None = None,
    ) -> dict[str, Any]:
    try:
        payload = _run_baseline(
            dataset_root=dataset_root,
            report_dir=report_dir,
            model=model,
            host=host,
            train_shots=train_shots,
            eval_limit=eval_limit,
            temperature=temperature,
            num_predict=num_predict,
            max_trace_steps=max_trace_steps,
            timeout_seconds=timeout_seconds,
            stages=stages,
        )
        return _wrap_success(payload)
    except Exception as exc:
        category, is_retryable = _error_category(exc)
        payload = _wrap_error(str(exc), category=category, is_retryable=is_retryable)
        payload["error"]["details"] = {
            **_details_for_exception(exc),
            "host": host,
            "runtime_host": normalize_mcp_runtime_host(host),
        }
        payload["error"]["next_action"] = "Verify Ollama is reachable from the MCP runtime. Dockerized MCP will auto-remap localhost to host.docker.internal; dataset/report paths must still stay inside the mounted workspace."
        return payload


@server.tool(description="Run a local checkpoint eval for vmbench using HF base weights plus a PEFT adapter.")
@_llm_call_log("vmbench_run_checkpoint_eval")
def vmbench_run_checkpoint_eval(
    phase2_freeze_path: str,
    dataset_root: str | None = None,
    report_dir: str = "reports/checkpoint_eval",
    stages: list[str] | None = None,
    splits: list[str] | None = None,
    eval_limit: int | None = None,
    num_predict: int | None = None,
    train_shots: int | None = None,
    max_trace_steps: int | None = None,
    evaluate_base: bool = True,
    fast_smoke: bool = False,
    run_name: str | None = None,
) -> dict[str, Any]:
    try:
        payload = _run_checkpoint_eval(
            phase2_freeze_path=phase2_freeze_path,
            dataset_root=dataset_root,
            report_dir=report_dir,
            stages=stages,
            splits=splits,
            eval_limit=eval_limit,
            num_predict=num_predict,
            train_shots=train_shots,
            max_trace_steps=max_trace_steps,
            evaluate_base=evaluate_base,
            fast_smoke=fast_smoke,
            run_name=run_name,
        )
        return _wrap_success(payload)
    except Exception as exc:
        category, is_retryable = _error_category(exc)
        payload = _wrap_error(str(exc), category=category, is_retryable=is_retryable)
        payload["error"]["details"] = _details_for_exception(exc)
        payload["error"]["next_action"] = (
            "Verify the freeze path, benchmark dataset, and adapter directory all exist inside the mounted workspace. "
            "This tool runs local HF/PEFT checkpoint eval, not Ollama."
        )
        return payload


@server.tool(description="Evaluate a vmbench summary.json against curriculum gates.")
@_llm_call_log("vmbench_gate_summary")
def vmbench_gate_summary(summary_path: str) -> dict[str, Any]:
    try:
        resolved = resolve_mcp_path(summary_path, must_exist=True)
        summary = json.loads(resolved.read_text(encoding="utf-8"))
        result = curriculum_gate.evaluate_summary(summary)
        return _wrap_success({"summary_path": str(resolved), "result": result})
    except Exception as exc:
        category, is_retryable = _error_category(exc)
        payload = _wrap_error(str(exc), category=category, is_retryable=is_retryable)
        payload["error"]["details"] = _details_for_exception(exc)
        payload["error"]["next_action"] = "Pass a summary path inside the mounted workspace or use a repo-relative path."
        return payload


@server.tool(description="Export vmbench benchmark data into SFT prompt/completion format.")
@_llm_call_log("vmbench_export_sft")
def vmbench_export_sft(
    dataset_root: str = "datasets/mvp",
    output_dir: str = "training_data/sft_v1",
    stages: list[str] | None = None,
) -> dict[str, Any]:
    stages = stages or ["single_step", "next_2_steps", "short_trace", "terminal_state"]
    try:
        dataset_root_path = resolve_mcp_path(dataset_root, must_exist=True)
        output_dir_path = resolve_mcp_path(output_dir, output=True)
        splits = export_sft_splits(dataset_root_path, stages)
        for split_name, records in splits.items():
            write_sft_jsonl(output_dir_path / f"{split_name}.jsonl", records)
        manifest = build_sft_manifest(output_dir_path, dataset_root_path, stages, splits)
        (output_dir_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        payload = {
            "output_dir": str(output_dir_path),
            "manifest_path": str(output_dir_path / "manifest.json"),
            "stages": stages,
        }
        return _wrap_success(payload)
    except Exception as exc:
        category, is_retryable = _error_category(exc)
        payload = _wrap_error(str(exc), category=category, is_retryable=is_retryable)
        payload["error"]["details"] = _details_for_exception(exc)
        payload["error"]["next_action"] = "Use dataset and output paths inside the mounted workspace or pass repo-relative paths."
        return payload


@server.tool(description="Return the key repo map and product plan paths.")
@_llm_call_log("vmbench_repo_map")
def vmbench_repo_map(_request: Any = None) -> dict[str, Any]:
    return _wrap_success(vmbench_repo_map_payload())


@server.tool(description="Compare two vmbench summary.json reports and return per-stage metric deltas.")
@_llm_call_log("vmbench_compare_reports")
def vmbench_compare_reports(base_summary_path: str, candidate_summary_path: str) -> dict[str, Any]:
    try:
        payload = shared_compare_summary_payload(
            resolve_mcp_path(base_summary_path, must_exist=True),
            resolve_mcp_path(candidate_summary_path, must_exist=True),
        )
        return _wrap_success(payload)
    except Exception as exc:
        category, is_retryable = _error_category(exc)
        payload = _wrap_error(str(exc), category=category, is_retryable=is_retryable)
        payload["error"]["details"] = _details_for_exception(exc)
        payload["error"]["next_action"] = "Pass summary paths inside the mounted workspace or use repo-relative paths."
        return payload


@server.tool(description="Choose the next verified step for one benchmark record and return ranked candidates plus the verified winner.")
@_llm_call_log("vmbench_choose_next_step")
def vmbench_choose_next_step(
    dataset_path: str,
    record_index: int,
    candidate_limit: int = 8,
    candidate_mode: str = "program_global",
    ranker: str = "heuristic",
    ranker_model_path: str | None = None,
    verifier_mode: str = "state_diff",
) -> dict[str, Any]:
    try:
        record = _load_benchmark_record(dataset_path, record_index)
        payload = _choose_next_step_payload(
            record,
            candidate_limit=candidate_limit,
            candidate_mode=candidate_mode,
            ranker=ranker,
            ranker_model_path=ranker_model_path,
            verifier_mode=verifier_mode,
        )
        payload["dataset_path"] = str(resolve_mcp_path(dataset_path, must_exist=True))
        payload["record_index"] = record_index
        return _wrap_success(payload)
    except Exception as exc:
        category, is_retryable = _error_category(exc)
        payload = _wrap_error(str(exc), category=category, is_retryable=is_retryable)
        payload["error"]["details"] = _details_for_exception(exc)
        return payload


@server.tool(description="Run budgeted search for one benchmark record and return solved status, path, nodes explored, and attempts.")
@_llm_call_log("vmbench_solve_with_budget")
def vmbench_solve_with_budget(
    dataset_path: str,
    record_index: int,
    candidate_limit: int = 32,
    candidate_mode: str = "program_global",
    ranker: str = "heuristic",
    ranker_model_path: str | None = None,
    verifier_mode: str = "state_diff",
    node_budget: int = 12,
) -> dict[str, Any]:
    try:
        record = _load_benchmark_record(dataset_path, record_index)
        payload = _solve_record_payload(
            record,
            candidate_limit=candidate_limit,
            candidate_mode=candidate_mode,
            ranker=ranker,
            ranker_model_path=ranker_model_path,
            verifier_mode=verifier_mode,
            node_budget=node_budget,
        )
        payload["dataset_path"] = str(resolve_mcp_path(dataset_path, must_exist=True))
        payload["record_index"] = record_index
        return _wrap_success(payload)
    except Exception as exc:
        category, is_retryable = _error_category(exc)
        payload = _wrap_error(str(exc), category=category, is_retryable=is_retryable)
        payload["error"]["details"] = _details_for_exception(exc)
        return payload


@server.tool(description="Explain why one benchmark record failed under the current budget, ranker, and verifier mode.")
@_llm_call_log("vmbench_explain_failure")
def vmbench_explain_failure(
    dataset_path: str,
    record_index: int,
    candidate_limit: int = 32,
    candidate_mode: str = "program_global",
    ranker: str = "heuristic",
    ranker_model_path: str | None = None,
    verifier_mode: str = "state_diff",
    node_budget: int = 12,
) -> dict[str, Any]:
    try:
        record = _load_benchmark_record(dataset_path, record_index)
        payload = _failure_category_payload(
            record,
            candidate_limit=candidate_limit,
            candidate_mode=candidate_mode,
            ranker=ranker,
            ranker_model_path=ranker_model_path,
            verifier_mode=verifier_mode,
            node_budget=node_budget,
        )
        payload["dataset_path"] = str(resolve_mcp_path(dataset_path, must_exist=True))
        payload["record_index"] = record_index
        return _wrap_success(payload)
    except Exception as exc:
        category, is_retryable = _error_category(exc)
        payload = _wrap_error(str(exc), category=category, is_retryable=is_retryable)
        payload["error"]["details"] = _details_for_exception(exc)
        return payload


@server.tool(description="Compare no-ranker, heuristic, and learned policy behavior for one benchmark record under the same budget.")
@_llm_call_log("vmbench_compare_policies")
def vmbench_compare_policies(
    dataset_path: str,
    record_index: int,
    candidate_limit: int = 32,
    candidate_mode: str = "program_global",
    verifier_mode: str = "state_diff",
    node_budget: int = 12,
    learned_model_path: str | None = None,
) -> dict[str, Any]:
    try:
        record = _load_benchmark_record(dataset_path, record_index)
        policies = []
        for ranker, model_path in [
            ("none", None),
            ("heuristic", None),
            ("learned", learned_model_path),
        ]:
            if ranker == "learned" and model_path is None:
                continue
            solve_payload = _solve_record_payload(
                record,
                candidate_limit=candidate_limit,
                candidate_mode=candidate_mode,
                ranker=ranker,
                ranker_model_path=model_path,
                verifier_mode=verifier_mode,
                node_budget=node_budget,
            )
            policies.append({"policy": ranker, **solve_payload})
        payload = {
            "dataset_path": str(resolve_mcp_path(dataset_path, must_exist=True)),
            "record_index": record_index,
            "program_name": record["program_name"],
            "verifier_mode": verifier_mode,
            "node_budget": node_budget,
            "policies": policies,
        }
        return _wrap_success(payload)
    except Exception as exc:
        category, is_retryable = _error_category(exc)
        payload = _wrap_error(str(exc), category=category, is_retryable=is_retryable)
        payload["error"]["details"] = _details_for_exception(exc)
        return payload


@server.tool(description="Return a one-call reasoning demo payload with next-step choice, budget curve, failure explanation, and policy comparison.")
@_llm_call_log("vmbench_demo_reasoning_runtime")
def vmbench_demo_reasoning_runtime(
    dataset_path: str,
    record_index: int,
    candidate_limit: int = 32,
    candidate_mode: str = "program_global",
    verifier_mode: str = "state_diff",
    budgets: list[int] | None = None,
    learned_model_path: str | None = None,
) -> dict[str, Any]:
    try:
        record = _load_benchmark_record(dataset_path, record_index)
        payload = _demo_reasoning_runtime_payload(
            record,
            dataset_path=dataset_path,
            record_index=record_index,
            candidate_limit=candidate_limit,
            candidate_mode=candidate_mode,
            verifier_mode=verifier_mode,
            learned_model_path=learned_model_path,
            budgets=budgets or [2, 4, 8, 12],
        )
        return _wrap_success(payload)
    except Exception as exc:
        category, is_retryable = _error_category(exc)
        payload = _wrap_error(str(exc), category=category, is_retryable=is_retryable)
        payload["error"]["details"] = _details_for_exception(exc)
        return payload


@server.tool(description="Generate a GitHub Pages demo payload file for the reasoning runtime.")
@_llm_call_log("vmbench_generate_demo_payload")
def vmbench_generate_demo_payload(
    dataset_path: str,
    record_index: int | None = None,
    record_indices: list[int] | None = None,
    output_path: str = "docs/demo-runtime-payload.json",
    candidate_limit: int = 32,
    candidate_mode: str = "program_global",
    verifier_mode: str = "state_diff",
    budgets: list[int] | None = None,
    learned_model_path: str | None = None,
) -> dict[str, Any]:
    try:
        payload = shared_write_demo_runtime_payload(
            dataset_path=dataset_path,
            record_index=record_index,
            record_indices=record_indices,
            output_path=output_path,
            candidate_limit=candidate_limit,
            candidate_mode=candidate_mode,
            verifier_mode=verifier_mode,
            budgets=budgets or [2, 4, 8, 12],
            learned_model_path=learned_model_path,
        )
        return _wrap_success(payload)
    except Exception as exc:
        category, is_retryable = _error_category(exc)
        payload = _wrap_error(str(exc), category=category, is_retryable=is_retryable)
        payload["error"]["details"] = _details_for_exception(exc)
        return payload


@server.tool(description="Summarize vmbench status, manifest, repo map, and available tools.")
@_llm_call_log("vmbench_status")
def vmbench_status(_request: Any = None) -> dict[str, Any]:
    payload = {
        "manifest": vmbench_manifest_payload(),
        "repo_map": vmbench_repo_map_payload(),
        "tools": TOOL_NAMES,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return _wrap_success(payload)


@server.tool(
    description="Set the reasoning for your *next* vmbench tool call. Call this immediately before another vmbench tool (e.g. vmbench_status). Your reasoning (e.g. 'Checking server is up before running baseline') is then attached to that next call in proxymetalog.jsonl so you see why you chose it. Keep it one short sentence."
)
@_llm_call_log("vmbench_set_next_reasoning")
def vmbench_set_next_reasoning(_request: Any = None, **kwargs: Any) -> dict[str, Any]:
    """Write reasoning to a file the proxy reads before the next tools/call; proxy attaches it to meta and clears the file."""
    reasoning = str(kwargs.get("reasoning") or "")
    if not reasoning and _request is not None:
        payload = _to_dict(_request)
        if isinstance(payload, dict):
            reasoning = str(payload.get("reasoning") or "")
    try:
        text = (reasoning or "").strip()[:10000]
        with open(_NEXT_REASONING_FILE, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        return _wrap_success({"ok": True, "message": "Reasoning will be attached to your next vmbench tool call in proxymetalog.jsonl."})
    except Exception as e:
        return {"status": "error", "payload": {"error": str(e)}}


@server.tool(
    description="Return the last N tool invocations (LLM call trace). Use this to see from the MCP side what the agent actually did: which tools were called, with what args, and success/error."
)
@_llm_call_log("vmbench_llm_call_log")
def vmbench_llm_call_log(_request: Any = None, limit: int = 50) -> dict[str, Any]:
    """Return recent tool calls so you can see the LLM's real steps for any call."""
    n = min(max(1, limit), len(CALL_LOG))
    calls = list(CALL_LOG)[-n:]
    return _wrap_success({"calls": calls, "total": len(CALL_LOG)})


def run_self_test() -> dict[str, Any]:
    manifest = vmbench_manifest_payload()
    temp_root = ROOT / ".vmbench_mcp_selftest"
    generate = _generate_dataset(output_dir=str(temp_root / "datasets"), single_step_limit=4, next_2_steps_limit=2, short_trace_limit=2, terminal_state_limit=2, seed=7)
    export = vmbench_export_sft(dataset_root=generate["output_dir"], output_dir=str(temp_root / "sft"), stages=["single_step"])
    return {
        "ok": True,
        "tools": TOOL_NAMES,
        "manifest": manifest["name"],
        "generate_dataset": generate["ok"],
        "export_sft": export.get("status") == "success",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="vmbench MCP server")
    parser.add_argument("--self-test", action="store_true", default=False)
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(run_self_test(), ensure_ascii=True, indent=2))
        return

    server.run(args.transport)


if __name__ == "__main__":
    main()
