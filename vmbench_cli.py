from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import curriculum_gate
from baseline_trainer import OllamaClient, check_ollama_available, evaluate_stage, write_json
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
from sft_export import build_manifest as build_sft_manifest
from sft_export import export_sft_splits, write_jsonl as write_sft_jsonl
from vmbench_product_surface import (
    compare_summary_payload,
    manifest_payload as vmbench_manifest_payload,
    normalize_mcp_runtime_host,
    repo_map_payload as vmbench_repo_map_payload,
    resolve_cli_path,
)


def cmd_generate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = resolve_cli_path(args.output_dir)
    generated = {
        "single_step": generate_single_step_examples(limit=args.single_step_limit, seed=args.seed),
        "next_2_steps": generate_next_k_steps_examples(
            limit=args.next_2_steps_limit,
            seed=args.seed,
            next_k_steps=2,
            dataset_type="next_2_steps",
        ),
        "short_trace": generate_short_trace_examples(limit=args.short_trace_limit, seed=args.seed),
        "terminal_state": generate_terminal_state_examples(limit=args.terminal_state_limit, seed=args.seed),
    }
    split_datasets = {
        dataset_type: split_records(records, seed=args.seed)
        for dataset_type, records in generated.items()
    }
    for dataset_type, splits in split_datasets.items():
        for split_name, records in splits.items():
            write_jsonl(output_dir / dataset_type / f"{split_name}.jsonl", records)
    manifest = build_manifest(split_datasets, seed=args.seed, output_dir=output_dir)
    config = build_generation_config(args)
    write_manifest(output_dir / "manifest.json", manifest)
    write_generation_config(output_dir / "generation_config.json", config)
    (output_dir / "DATASET_CARD.md").write_text(build_dataset_card(manifest, config), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "dataset_types": list(split_datasets.keys()),
        "manifest_path": str(output_dir / "manifest.json"),
    }


def _summary_from_stage_reports(args: argparse.Namespace, stage_reports: dict) -> dict:
    return {
        "model": args.model,
        "host": args.host,
        "train_shots": args.train_shots,
        "eval_limit": args.eval_limit,
        "temperature": args.temperature,
        "num_predict": args.num_predict,
        "max_trace_steps": args.max_trace_steps,
        "timeout_seconds": args.timeout_seconds,
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


def cmd_eval(args: argparse.Namespace) -> dict[str, Any]:
    runtime_host = normalize_mcp_runtime_host(args.host)
    available, details = check_ollama_available(runtime_host, min(args.timeout_seconds, 5))
    if not available:
        raise SystemExit(f"Ollama runtime is unavailable at {runtime_host}: {details}")

    dataset_root = resolve_cli_path(args.dataset_root)
    report_dir = resolve_cli_path(args.report_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_slug = args.model.replace(":", "-").replace("/", "-")
    run_dir = report_dir / f"{timestamp}-{model_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    stage_reports: dict = {}
    for stage in args.stages:
        client = OllamaClient(
            host=runtime_host,
            model=args.model,
            temperature=args.temperature,
            num_predict=args.num_predict,
            timeout_seconds=args.timeout_seconds,
        )
        stage_reports[stage] = evaluate_stage(
            client=client,
            dataset_root=dataset_root,
            stage=stage,
            train_shots=args.train_shots,
            eval_limit=args.eval_limit,
            max_trace_steps=args.max_trace_steps,
        )
        write_json(run_dir / f"{stage}.json", stage_reports[stage])

    summary = _summary_from_stage_reports(args, stage_reports)
    write_json(run_dir / "summary.json", summary)
    return {
        "run_dir": str(run_dir),
        "summary_path": str(run_dir / "summary.json"),
        "stages": list(stage_reports.keys()),
    }


def cmd_gate(args: argparse.Namespace) -> dict[str, Any]:
    summary = json.loads(resolve_cli_path(args.summary).read_text(encoding="utf-8"))
    result = curriculum_gate.evaluate_summary(summary)
    return {"result": result}


def cmd_export_sft(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = resolve_cli_path(args.dataset_root)
    output_dir = resolve_cli_path(args.output_dir)
    splits = export_sft_splits(dataset_root, args.stages)
    for split_name, records in splits.items():
        write_sft_jsonl(output_dir / f"{split_name}.jsonl", records)
    manifest = build_sft_manifest(output_dir, dataset_root, args.stages, splits)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "stages": args.stages,
        "manifest_path": str(output_dir / "manifest.json"),
    }


def cmd_repo_map(_: argparse.Namespace) -> dict[str, str]:
    return vmbench_repo_map_payload()


def cmd_compare(args: argparse.Namespace) -> dict[str, Any]:
    return compare_summary_payload(resolve_cli_path(args.base_summary), resolve_cli_path(args.candidate_summary))


def cmd_status(_: argparse.Namespace) -> dict[str, Any]:
    manifest_data = vmbench_manifest_payload()
    repo_data = vmbench_repo_map_payload()
    commands = ["generate", "eval", "compare", "gate", "export-sft", "repo-map", "status"]
    return {
        "manifest": manifest_data,
        "repo_map": repo_data,
        "available_commands": commands,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vmbench CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Generate benchmark datasets.")
    gen.add_argument("--output-dir", default="datasets/mvp")
    gen.add_argument("--single-step-limit", type=int, default=256)
    gen.add_argument("--next-2-steps-limit", type=int, default=128)
    gen.add_argument("--short-trace-limit", type=int, default=128)
    gen.add_argument("--terminal-state-limit", type=int, default=128)
    gen.add_argument("--seed", type=int, default=7)
    gen.set_defaults(func=cmd_generate)

    ev = subparsers.add_parser("eval", help="Run local baseline evaluation.")
    ev.add_argument("--dataset-root", default="datasets/mvp")
    ev.add_argument("--report-dir", default="reports/baseline")
    ev.add_argument("--model", default="llama3.2:latest")
    ev.add_argument("--host", default="http://127.0.0.1:11434")
    ev.add_argument("--train-shots", type=int, default=1)
    ev.add_argument("--eval-limit", type=int, default=2)
    ev.add_argument("--temperature", type=float, default=0.0)
    ev.add_argument("--num-predict", type=int, default=192)
    ev.add_argument("--max-trace-steps", type=int, default=6)
    ev.add_argument("--timeout-seconds", type=int, default=12)
    ev.add_argument("--stages", nargs="*", default=["single_step", "next_2_steps"])
    ev.set_defaults(func=cmd_eval)

    gate = subparsers.add_parser("gate", help="Score a summary against curriculum gates.")
    gate.add_argument("--summary", required=True)
    gate.set_defaults(func=cmd_gate)

    compare = subparsers.add_parser("compare", help="Compare two vmbench summary.json files.")
    compare.add_argument("--base-summary", required=True)
    compare.add_argument("--candidate-summary", required=True)
    compare.set_defaults(func=cmd_compare)

    export = subparsers.add_parser("export-sft", help="Export benchmark data to SFT prompt/completion format.")
    export.add_argument("--dataset-root", default="datasets/mvp")
    export.add_argument("--output-dir", default="training_data/sft_v1")
    export.add_argument("--stages", nargs="*", default=["single_step", "next_2_steps", "short_trace", "terminal_state"])
    export.set_defaults(func=cmd_export_sft)

    repo = subparsers.add_parser("repo-map", help="Print product map paths.")
    repo.set_defaults(func=cmd_repo_map)

    status = subparsers.add_parser("status", help="Show vmbench manifest, repo map, and command overview.")
    status.set_defaults(func=cmd_status)

    return parser


def _print_response(payload: dict[str, Any], status: str = "success", error: dict[str, Any] | None = None) -> None:
    response = {"status": status, "payload": payload}
    if error:
        response["error"] = error
    print(json.dumps(response, ensure_ascii=True, indent=2))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        payload = args.func(args)
        _print_response(payload)
        return 0
    except Exception as exc:
        _print_response({}, status="error", error={"message": str(exc), "type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
