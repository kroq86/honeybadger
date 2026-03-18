# vmbench Execution Plan

## Goal

Turn `vmbench` from a research-heavy repo into a usable product with:

- a clean CLI
- a real MCP server
- stable machine-readable outputs
- release-ready docs and public surface

Primary user experience:

- **MCP-first**

Secondary user experience:

- **CLI**

Supporting surfaces:

- **CI**
- **GitHub Pages**

## Product Thesis

`vmbench` is a formal VM benchmark and synthetic training/eval toolkit for checking whether language models can follow machine-like execution semantics.

It is **not** a product about proving hidden internal execution inside the model.

## Non-Goals

- shipping the failed `next2_*` research branch as product surface
- promising internal-executor claims
- making fine-tuning the primary user path
- overloading v1 with skill-first or research-first UX

## Working Rule

We only open longer implementation or validation runs after a bounded smoke on the same path passes.

## Public Surface Policy

### Stable

- MCP server v1
- CLI v1
- smoke CI
- Docker image
- README and Pages

### Experimental

- bounded training/eval helpers
- advanced compare/report workflows that are not yet covered by smoke validation

### Internal-Only

- negative-result research branches
- deep exploratory reports that are not part of the default product path

## Roles

### Main Thread

Owns:

- product scope
- milestone gates
- integration
- MCP bookkeeping
- final quality bar

### Docs Agent

Owns:

- README / docs structure proposals
- UX flows
- launch checklist
- release-facing wording
- acceptance criteria clarity

### Code Agent

Owns:

- bounded product-surface code changes
- CLI/MCP implementation
- smoke workflow hardening
- structured machine-readable outputs

## Milestone 1. Product Contract

### Outcome

Define one stable public contract for `vmbench`.

### Main Thread

- freeze product direction around benchmark/eval toolkit
- define canonical commands and MCP tools
- define success criteria for “usable product surface”
- cut research-heavy claims from the public contract

### Docs Agent

- write short public positioning
- define primary user journeys
- define command/tool naming guidance
- draft personas and release labels glossary

### Code Agent

- audit current CLI/MCP gaps against the contract
- list blockers:
  - hardcoded paths
  - cwd assumptions
  - missing compare/report APIs
  - inconsistent defaults

### Deliverables

- [VMBENCH_EXECUTION_PLAN.md](/Users/ll/honeybadger/VMBENCH_EXECUTION_PLAN.md)
- updated product contract in [README.md](/Users/ll/honeybadger/README.md)
- product gap report as generated output, not tracked source

### Acceptance Criteria

- README, Pages, MCP descriptions, and CLI help tell the same story
- primary user can answer:
  - what `vmbench` is
  - how to generate a dataset
  - how to run a benchmark
  - how to gate a run
  - how to export SFT data
- `next2_*` research paths do not leak into the main product contract

## Milestone 2. Stable CLI Surface

### Outcome

CLI returns stable JSON-shaped outputs and is safe for CI and wrappers.

### Main Thread

- define CLI response contract
- choose required commands for v1
- decide minimal supported workflow:
  - generate
  - eval
  - compare
  - gate
  - export-sft

### Docs Agent

- define command descriptions and examples
- verify onboarding clarity
- align quickstart with actual command flags and outputs

### Code Agent

- harden:
  - [vmbench_cli.py](/Users/ll/honeybadger/vmbench_cli.py)
- add:
  - `compare`
- ensure consistent:
  - success envelopes
  - error envelopes
  - return codes
  - predictable JSON output

### Deliverables

- hardened CLI commands:
  - `generate`
  - `eval`
  - `compare`
  - `gate`
  - `export-sft`
  - `repo-map`
- CLI smoke matrix as generated validation output, not tracked source

### Acceptance Criteria

- every command has:
  - deterministic stdout contract
  - non-zero exit on failure
  - machine-readable error payload where possible
- `compare` and `gate` are usable for CI gates

## Milestone 3. Real MCP Product Surface

### Outcome

`vmbench` MCP is useful as a real tool surface, not just a wrapper.

### Main Thread

- define which tools are v1
- define tool descriptions and error semantics
- keep MCP and CLI semantics aligned

### Docs Agent

- write tool descriptions and examples
- write MCP runbook / enablement note
- define error taxonomy wording
- define future MCP resource semantics

### Code Agent

- harden:
  - [vmbench_mcp_server.py](/Users/ll/honeybadger/tools/mcp/vmbench_mcp_server.py)
- add:
  - `vmbench_compare_reports`
- add stable envelopes and better failure handling
- reduce cwd fragility where possible
- keep self-test green

### Deliverables

- v1 MCP tools:
  - `vmbench_manifest`
  - `vmbench_generate_dataset`
  - `vmbench_run_baseline`
  - `vmbench_gate_summary`
  - `vmbench_export_sft`
  - `vmbench_repo_map`

### Acceptance Criteria

- self-test passes locally and in Docker
- tool descriptions are good enough for agent discovery
- failures are reported as structured error payloads
- bounded defaults are used by default

## Milestone 4. CI and Release Safety

### Outcome

Product surface is validated before shipping.

### Main Thread

- define minimum required checks before push/release
- reject public push of untested surface

### Docs Agent

- produce launch checklist
- produce release note template / public copy guidance
- produce public-safe wording pass

### Code Agent

- harden:
  - [product-smoke.yml](/Users/ll/honeybadger/.github/workflows/product-smoke.yml)
- ensure both CLI and MCP self-test are covered
- add packaging validation where cheap

### Deliverables

- release checklist
- product smoke workflow
- docker publish workflow sanity notes

### Acceptance Criteria

- CLI smoke runs in CI
- MCP self-test runs in CI
- Docker image path is validated before publish
- release checklist is executable by someone other than the author

## Milestone 5. Public Docs and Pages

### Outcome

A public visitor can understand and try `vmbench`.

### Main Thread

- align public docs with actual surface

### Docs Agent

- refine:
  - README
  - Pages copy
  - runbook
  - examples
  - limitations (scope, assumptions, and factors that influence results)

### Code Agent

- only bounded fixes required to keep docs truthful

### Deliverables

- public README
- GitHub Pages content
- examples consistent with current code

### Acceptance Criteria

- a new user can:
  - run one CLI command successfully
  - understand what MCP server does
  - find the repo map and plan
- docs do not promise unsupported behavior

## Milestone 6. Archive Boundary

### Outcome

Default user path is product-focused; research debris is demoted.

### Main Thread

- decide what stays first-class
- decide what becomes archival / secondary
- freeze unstable branches out of public narrative

### Docs Agent

- propose archive wording and nav changes

### Code Agent

- only minimal path/documentation changes needed to keep imports working

### Deliverables

- keep/archive boundary note
- updated repo map

### Acceptance Criteria

- default repo entrypoints no longer depend on deep research context

## Risks

1. Research leakage into public surface.
2. MCP/CLI divergence.
3. Hardcoded path and cwd assumptions.
4. Untested public commands or docs.
5. Over-scoped first release.
6. Claims or docs that overstate scope—results are limited to the current benchmark, dataset family, and toolkit; state limitations and assumptions clearly.

## Current Active Queue

1. Harden CLI response contract.
2. Harden MCP response/error contract.
3. Add `compare` as first-class CLI/MCP path.
4. Strengthen product smoke workflow.
5. Refresh top-level README against the real contract.
6. Fold launch checklist into release-safe flow.

## Agent Tasking

### Docs Agent Current Track

- detailed milestone proposal
- README/public-surface wording
- launch checklist
- release-facing runbook

### Code Agent Current Track

- stable machine-readable responses in CLI
- stable structured error handling in MCP server
- stronger smoke validation in CI

### Main Thread Current Track

- integrate both outputs
- preserve clean product scope
- reject untested surface changes

## Gates

### Gate A

Before any public release-oriented push:

- CLI smoke passes
- MCP self-test passes
- Docker self-test passes

### Gate B

Before claiming MCP-first usability:

- tool descriptions reviewed
- stable error envelopes implemented
- Codex enablement runbook exists

### Gate C

Before larger cleanup/archive work:

- README and Pages reflect real product behavior
