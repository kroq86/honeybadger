# vmbench

`vmbench` is a formal VM benchmark and inspectable reasoning runtime for testing whether language models can follow machine-like execution semantics on synthetic tasks.

It gives you:

- a toy ISA and reference VM
- synthetic dataset generation for execution tasks
- bounded evaluation and stage gating
- an inspectable search runtime with verification and budget control
- SFT export and bounded training/eval helpers

## What This Is

This repo started as a research workbench around AI-native execution ideas.

The strongest reusable outcome is not a new programming language inside the model. It is a practical benchmark/runtime stack for asking a narrower question:

**Can a model follow formal machine semantics on synthetic execution tasks?**

The current public story is:

- **formal VM benchmark**
- **verifier-guided search runtime**
- **MCP-first tool surface**

Not:

- proof that a latent internal executor emerged in the weights
- a general-purpose reasoning system
- a productized wrapper around every research branch in the repo

## Product Shape

Primary surface:

- **MCP-first runtime and benchmark tool layer**

Secondary surface:

- **CLI**

Optional surface:

- **agent / IDE integrations**

### Public Contract

**Stable v1**

- MCP server
- CLI
- machine-readable outputs
- smoke-tested Docker/local flows
- release-facing docs

**Experimental**

- bounded training/eval helpers
- advanced compare/report workflows
- branch-specific research probes

**Internal-only**

- negative-result `next2_*` branches
- exploratory reports and ad hoc research scaffolding

## Core Pieces

- [reference_vm.py](reference_vm.py)
  - formal VM/parser/executor
- [dataset_pipeline.py](dataset_pipeline.py)
  - main benchmark dataset generator
- [baseline_trainer.py](baseline_trainer.py)
  - local inference/eval harness
- [curriculum_gate.py](curriculum_gate.py)
  - stage gate evaluation
- [sft_export.py](sft_export.py)
  - SFT export from benchmark datasets

## Runtime UX

The runtime surface is meant to make reasoning visible rather than hidden. A user can:

- run a benchmark and get a compact machine-readable summary
- inspect the next-step choice with ranked candidates and verifier outcomes
- solve one record under a fixed node budget and see the path and explored-node count
- ask for a structured failure explanation
- compare `none`, `heuristic`, and `learned` policies on the same record
- gate a run and compare two runs stage-by-stage

The key difference from a normal black-box LLM answer is that the user can see:

- which candidates existed
- which one was verified
- how much budget was spent
- why a run failed

## Quickstart

Generate the core benchmark dataset:

```bash
python3 vmbench_cli.py generate
```

Run a local baseline against the generated dataset:

```bash
python3 vmbench_cli.py eval \
  --model llama3.2:latest \
  --host http://127.0.0.1:11434
```

Host contract:
- `vmbench_cli.py` runs on the host, so `http://127.0.0.1:11434` is the normal local default.
- The Dockerized MCP server automatically remaps `localhost` and `127.0.0.1` to `http://host.docker.internal:11434` for Ollama-backed baseline runs.

Inspect the manifest, repo map, and available commands with structured JSON output:
```bash
python3 vmbench_cli.py status
```

Evaluate curriculum gates on a `summary.json` file:

```bash
python3 vmbench_cli.py gate \
  --summary reports/baseline/<run>/summary.json
```

Compare two `summary.json` files:

```bash
python3 vmbench_cli.py compare \
  --base-summary /path/to/base-summary.json \
  --candidate-summary /path/to/candidate-summary.json
```

Export benchmark data into prompt/completion SFT format:

```bash
python3 vmbench_cli.py export-sft
```

All CLI commands return structured JSON: `status` (`success`/`error`), `payload`, and `error` details when applicable.

## CLI Commands

- `generate`
  - generate benchmark datasets
- `eval`
  - run local baseline evaluation
- `gate`
  - score a summary against curriculum gates
- `compare`
  - compare two summary files stage-by-stage
- `export-sft`
  - export SFT-ready prompt/completion datasets
- `repo-map`
  - print the current product map path
- `status`
  - show manifest, repo map, and command overview with machine-readable JSON

## MCP Runtime Surface

The MCP server is the primary public interface. It exposes both benchmark/tooling flows and runtime-inspection flows.

Core benchmark flows:

- `vmbench_manifest`
- `vmbench_generate_dataset`
- `vmbench_run_baseline`
- `vmbench_run_checkpoint_eval`
- `vmbench_gate_summary`
- `vmbench_export_sft`
- `vmbench_compare_reports`
- `vmbench_status`

Reasoning-runtime flows:

- `vmbench_choose_next_step`
- `vmbench_solve_with_budget`
- `vmbench_explain_failure`
- `vmbench_compare_policies`
- `vmbench_demo_reasoning_runtime`
- `vmbench_generate_demo_payload`

All tools return structured envelopes with `status`, `payload`, and `error`.

## Repository Layout

- [REPO_PRODUCT_MAP.md](REPO_PRODUCT_MAP.md)
  - product-oriented map of the repo
- [PRODUCTIZATION_PLAN.md](PRODUCTIZATION_PLAN.md)
  - productization work plan
- [VMBENCH_EXECUTION_PLAN.md](VMBENCH_EXECUTION_PLAN.md)
  - active milestone execution plan with agent ownership
- [datasets/mvp](datasets/mvp)
  - main benchmark datasets
- [training](training)
  - bounded fine-tuning / eval tooling
- [reports](reports)
  - research and validation outputs

## Current Status

This repo contains a usable benchmark/runtime core.

It also contains a large amount of experimental research history, especially around the failed `next_2_steps` unlock branch.

The current product direction is:

**formal VM benchmark + dataset factory + eval toolkit + inspectable runtime**

not:

**proof of internal multi-step execution inside the model**

## Limitations

- The multi-step `next_2_steps` branch is currently a negative result; gains depend on benchmark structure and ranking/verification design.
- Results are on synthetic execution tasks and the current ISA/dataset family; performance may not generalize to other task distributions or ISAs.
- The learned ranker may partly compress heuristic behavior rather than discover a materially different general policy.
- Shortcut structure, especially candidate-source information, still matters on harder cases.
- MCP exposes runtime behavior and tool use; it does **not** prove the model’s internal chain-of-thought or latent execution.
- Many reports in the repo are research artifacts; not every helper in the tree belongs to the public product surface.

## Next Product Steps

- keep a clean product surface
- keep MCP and CLI contracts aligned
- strengthen failure analysis, harder benchmarks, and budget curves
- archive research-heavy branches out of the default user path
