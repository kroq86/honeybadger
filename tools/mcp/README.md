# vmbench MCP Surface

This directory holds the MCP-first public runtime surface for `vmbench`.

Current scope:

- expose the stable product-facing methods
- keep the interface small and machine-readable
- wrap the reusable core rather than exposing research branches by default

## Public Contract

`vmbench` over MCP is intended to expose:

- benchmark workspace inspection
- dataset generation
- bounded model evaluation
- run gating and comparison
- inspectable reasoning decisions record by record
- budgeted search with failure explanation

It is **not** intended to prove hidden internal execution inside the model or to expose every research helper in the repo.

### Stable

- MCP server v1 methods
- structured result envelopes
- repo/workspace path contract
- local and Docker smoke paths

### Experimental

- bounded checkpoint-eval helpers
- richer compare/report flows
- demo payload generation and related front-end support

### Internal-only

- negative-result research branches
- ad hoc exploratory helpers not covered by smoke validation

Current methods:

- `vmbench_manifest`
- `vmbench_generate_dataset`
- `vmbench_run_baseline`
- `vmbench_run_checkpoint_eval`
- `vmbench_gate_summary`
- `vmbench_export_sft`
- `vmbench_repo_map`
- `vmbench_compare_reports`
- `vmbench_choose_next_step`
- `vmbench_solve_with_budget`
- `vmbench_explain_failure`
- `vmbench_compare_policies`
- `vmbench_demo_reasoning_runtime`
- `vmbench_generate_demo_payload`
- `vmbench_status`

All tools return structured envelopes with `status`, `payload`, and `error` details so automation can gate retries, retries, and escalation pathways.

The newer reasoning-oriented tools turn `vmbench` into a small inspectable reasoning runtime:

- `vmbench_choose_next_step`: show top candidates plus the first verified winner for one benchmark record
- `vmbench_solve_with_budget`: run one record under a fixed node budget and return the path, attempts, and explored-node count
- `vmbench_explain_failure`: classify why one record failed under the current budget/policy
- `vmbench_compare_policies`: compare `no ranker`, `heuristic`, and `learned` on the same record
- `vmbench_demo_reasoning_runtime`: return a one-call demo payload with next-step choice, budget sweep, failure explanation, and policy comparison
- `vmbench_generate_demo_payload`: write a fresh demo payload to `docs/demo-runtime-payload.json` or another target path so the GitHub Pages demo can be refreshed automatically; supports one record or a multi-scenario bundle via `record_indices`

## What This Surface Proves

The MCP layer can prove:

- which tool was called
- with which arguments
- what structured result came back
- how the runtime behaved on one record or one run

It does **not** prove:

- the model's chain-of-thought
- the user prompt that triggered the tool call
- all rejected alternatives considered by the model
- hidden internal execution inside the model weights

So the MCP surface should be read as a proof of **runtime observability and interface behavior**, not as a mechanistic interpretability surface.

## Path Contract

The Dockerized MCP server is workspace-scoped.

- Use repo-relative paths whenever possible.
- Absolute paths must still point inside the mounted workspace.
- Paths outside the mounted workspace now fail fast with a structured validation error instead of returning false-success output.
- For Ollama-backed baseline runs, `localhost` and `127.0.0.1` are automatically remapped to `host.docker.internal` when the MCP server is running inside Docker.
- For local checkpoint eval runs, use freeze/adapters/datasets that exist inside the mounted workspace; these runs use local HF/PEFT weights rather than Ollama.
- The Docker image now persists Hugging Face cache under `/workspace/.cache/huggingface` so repeated checkpoint eval calls do not redownload model artifacts.
- `vmbench_run_checkpoint_eval` supports a `fast_smoke` mode for quick MCP validation runs. It evaluates only the finetuned adapter on the `test` split with small bounded defaults unless you override them explicitly.

## Local Smoke

Local venv smoke:

```bash
source .venv-vmbench-mcp/bin/activate
cd <repo-root>
python tools/mcp/vmbench_mcp_server.py --self-test
```

Docker smoke:

```bash
cd <repo-root>
docker build -f tools/mcp/Dockerfile -t vmbench-mcp:local .
docker run --rm -v "$PWD:/workspace" -w /workspace vmbench-mcp:local --self-test
```

## Codex Config Example

See:

- [`codex_config.example.toml`](codex_config.example.toml)

## Limitations

- Behavior and paths depend on workspace mount and `cwd`; results are not guaranteed outside the documented setup.
- Cleaner repo-root anchoring and a final packaged install surface are still missing.
