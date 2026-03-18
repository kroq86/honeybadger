# vmbench MCP Spec

## Product Direction

`vmbench` exposes a small formal-VM benchmark and evaluation toolkit through an MCP-friendly surface. Benchmark results depend on the dataset, splits, and run setup; see the dataset card and README for scope and limitations.

The goal is to let agents and tools:

- inspect the benchmark workspace
- generate datasets
- run bounded evaluations
- compare results
- evaluate curriculum gates
- export training data
- inspect reasoning decisions record by record
- run budgeted search with explainable outcomes
- compare policies on the same reasoning instance

The intended product story is:

- **formal VM benchmark**
- **verifier-guided search runtime**
- **MCP-first interface**

Not:

- proof of latent internal execution
- a claim that all research branches are part of the stable product surface

## Contract Boundary

### Stable contract

- structured envelopes: `status`, `payload`, `error`
- bounded operations by default
- explicit file/path arguments
- stable reasoning-runtime methods for inspecting one record at a time

### Experimental contract

- checkpoint-eval helpers
- richer compare/report flows
- demo payload generation and front-end support

### Not part of the MCP contract

- chain-of-thought or hidden model reasoning
- negative-result research branches as first-class tools
- arbitrary access outside the mounted workspace

## Current Skeleton Methods

### `vmbench_manifest`

Returns:

- product metadata
- primary/secondary surface
- core module paths

### `vmbench_gate_summary`

Input:

- `summary_path`

Returns:

- curriculum gate evaluation

### `vmbench_export_sft`

Inputs:

- `dataset_root`
- `output_dir`
- `stages`

Returns:

- output directory
- exported stages

## Next MCP Methods

### `vmbench_generate_dataset`

Inputs:

- output dir
- dataset limits
- seed

Backed by:

- [`../../dataset_pipeline.py`](../../dataset_pipeline.py)

### `vmbench_run_baseline`

Inputs:

- dataset root
- model
- host
- bounded eval parameters

Backed by:

- [`../../baseline_trainer.py`](../../baseline_trainer.py)

### `vmbench_run_checkpoint_eval`

Inputs:

- `phase2_freeze_path`
- optional benchmark dataset override
- optional stage/eval overrides
- optional `splits`
- optional `evaluate_base`
- optional `fast_smoke`
- report dir / run name

Returns:

- run directory
- base summary path
- finetuned summary path
- comparison report path

Backed by:

- [`../../training/eval_checkpoint.py`](../../training/eval_checkpoint.py)

### `vmbench_compare_reports`

Inputs:

- two summary/report paths

Returns:

- compact comparison

### `vmbench_choose_next_step`

Inputs:

- `dataset_path`
- `record_index`
- candidate/ranker/verifier options

Returns:

- ranked candidates
- verified winner
- per-candidate verifier status

### `vmbench_solve_with_budget`

Inputs:

- `dataset_path`
- `record_index`
- candidate/ranker/verifier options
- `node_budget`

Returns:

- solved flag
- successful path
- attempts
- nodes explored
- budget exhausted flag

### `vmbench_explain_failure`

Inputs:

- same as `vmbench_solve_with_budget`

Returns:

- failure category
- local diagnostic details

### `vmbench_compare_policies`

Inputs:

- `dataset_path`
- `record_index`
- common candidate/verifier settings
- `node_budget`
- optional learned model path

Returns:

- side-by-side `no ranker` / `heuristic` / `learned` outcomes

### `vmbench_demo_reasoning_runtime`

Inputs:

- `dataset_path`
- `record_index`
- candidate / verifier settings
- optional budget list
- optional learned model path

Returns:

- one-call reasoning demo payload:
  - next-step suggestion
  - solve-with-budget curve
  - failure explanation
  - policy comparison

### `vmbench_generate_demo_payload`

Inputs:

- same core demo inputs as `vmbench_demo_reasoning_runtime`
- `output_path`
- optional `record_indices` for a multi-scenario bundle

Returns:

- written payload path
- the generated demo payload or multi-scenario bundle

### `vmbench_repo_map`

Returns:

- product map
- plan path
- clean surface references

## Seeing the LLM's real steps from MCP

For every tool call, the server records: **tool name**, **arguments**, **timestamp**, **status** (success/error), and a short **result_preview** or **error** message. So you see from the MCP side what the agent actually did—no need to rely on the chat.

- **`vmbench_llm_call_log(limit=50)`**  
  Returns the last `limit` invocations (default 50). Payload: `calls` (list of entries), `total` (number of entries in the ring buffer). Use this after a run to inspect the exact sequence of tool calls the LLM made.

- **File log (optional)**  
  If the server writes to a file (default repo root `.vmbench_llm_call_log.jsonl`, or path in env `VMBENCH_MCP_CALL_LOG`), you can `tail -f` that file to watch calls in real time (e.g. from the host when MCP runs in Docker). Set `VMBENCH_MCP_CALL_LOG=0` to disable file logging.

## What MCP Does Not Show

The MCP layer is an **external behavior trace**, not an internal reasoning trace. It does not show:

- the user prompt
- the model's chain-of-thought
- all rejected tool choices
- complete retry history outside the retained window
- a mechanistic trace of hidden execution inside the model

This distinction matters for claims: MCP proves runtime behavior, tool use, and structured outputs; it does not prove latent internal execution.

## Design Rules

- bounded by default
- always return manifest/report paths
- avoid exposing research-branch names in the main user surface
- treat branch-heavy `next2_*` artifacts as internal/research unless explicitly requested
- return structured success/error envelopes with `status`/`payload` and `error` fields so callers can gate automation
- include a `vmbench_status` tool that aggregates manifest, repo map, and current tool list for quick assessments
- when dockerized, treat local Ollama defaults (`localhost`, `127.0.0.1`) as host-runtime addresses and remap them to `host.docker.internal` rather than failing with a false local-loopback error
- keep checkpoint-eval outputs close to baseline summary shape so downstream gate/compare tooling can reuse them
- persist HF cache inside the mounted workspace so repeated checkpoint eval calls stay fast
- provide a fast-smoke checkpoint path that can validate MCP wiring in tens of seconds rather than minutes by skipping base eval and restricting to bounded test-only runs
- expose record-level reasoning tools that make search controllable, observable, and budgeted rather than returning only opaque final answers
