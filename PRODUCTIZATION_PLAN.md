# Productization Plan

## Goal

Turn the strongest reusable core of this repo into a product-shaped tool:

- formal VM execution core
- synthetic dataset factory
- model evaluation harness
- optional local fine-tuning/eval loop

Primary product direction:

- **MCP-first**

Secondary surfaces:

- **CLI**
- **Codex skill**

## Product Thesis

This repo should stop presenting itself as “latent internal executor research” and instead become:

**a VM execution benchmark and training/eval toolkit for language models**

Working product name candidates will be decided during Milestone 1.

## Milestone 1. Product Framing

### Outcome

Choose a product shape, naming direction, and clean public surface.

### Main Thread

- create top-level product plan
- define product scope
- define keep/archive boundary
- define core user flows

### Kepler

- naming options
- UX framing
- CLI vs MCP vs skill recommendation
- user-facing product language

### Ramanujan

- concrete inventory of what code can stay in product surface
- what should move to archive / research area
- detect packaging blockers for CLI/MCP

### Deliverables

- `PRODUCTIZATION_PLAN.md`
- concise product surface brief
- cleanup inventory for public repo boundary

## Milestone 2. Clean Product Surface

### Outcome

Separate reusable product core from research debris.

### Main Thread

- define canonical modules to keep public
- define archive candidates
- update repo structure proposal

### Kepler

- validate docs/navigation shape
- propose public folder naming

### Ramanujan

- check import graph and practical breakage risk
- identify directories/files safe to leave out of public surface

### Deliverables

- `REPO_PRODUCT_MAP.md`
- keep-set decision for the public product surface
- archive/remove-set decision for research debris

## Milestone 3. User Experience Definition

### Outcome

Define real user experience, not just code structure.

### Main Thread

- define product personas
- define primary use cases
- define canonical commands / actions

### Kepler

- propose user flows
- write command naming options
- propose docs IA

### Ramanujan

- verify command feasibility against existing code paths

### Deliverables

- user flows for the public product
- naming matrix for the public product
- command map for the public product

## Milestone 4. Top-Level README

### Outcome

Replace the implicit research narrative with a product README.

### Main Thread

- write README around:
  - what it is
  - who it is for
  - quickstart
  - core commands
  - MCP/CLI surfaces
  - repo structure

### Kepler

- tighten wording / onboarding clarity

### Ramanujan

- verify commands in README actually correspond to code

### Deliverables

- `README.md`

## Milestone 5. CLI Skeleton

### Outcome

Expose the clean core through one consistent CLI entrypoint.

### Target Shape

Suggested command family:

- `vmbench init`
- `vmbench generate`
- `vmbench eval`
- `vmbench compare`
- `vmbench gate`
- `vmbench export-sft`

### Main Thread

- create CLI module
- connect commands to existing code:
  - VM/dataset generation
  - baseline eval
  - curriculum gate
  - SFT export

### Kepler

- validate command UX and naming consistency

### Ramanujan

- smoke-check commands
- identify broken or missing paths

### Deliverables

- `vmbench_cli.py` or package entrypoint
- lightweight usage examples

## Milestone 6. MCP Skeleton

### Outcome

Expose the same core through an MCP-friendly interface.

### Main Thread

- design minimal MCP surface
- map product operations into MCP methods

### Suggested First MCP Methods

- `vmbench_manifest`
- `vmbench_generate_dataset`
- `vmbench_run_baseline`
- `vmbench_compare_reports`
- `vmbench_gate_summary`
- `vmbench_export_sft`

### Kepler

- propose tool descriptions and user-facing phrasing

### Ramanujan

- validate method-to-code mapping and runtime assumptions

### Deliverables

- `tools/mcp/` skeleton
- MCP method spec doc

## Milestone 7. Skill / Codex Surface

### Outcome

Turn the product into a usable Codex skill on top of MCP/CLI.

### Main Thread

- define skill behavior around the new product

### Kepler

- skill prompt wording

### Ramanujan

- verify steps are executable

### Deliverables

- skill draft

## Milestone 8. Validation

### Outcome

Check that the product story is coherent and the commands actually work.

### Main Thread

- run small validation
- document known limitations (scope of claims, assumptions, and factors that affect results)

### Kepler

- write concise product-ready limitations note stating what we do and do not claim

### Ramanujan

- run only bounded smoke validations

### Deliverables

- bounded product validation summary

## Agent Ownership

### Main Thread

- architecture
- product surface decisions
- repo restructuring proposals
- README
- CLI/MCP implementation
- final integration

### Kepler

- naming
- UX
- docs shape
- command language
- concise product-facing notes

### Ramanujan

- bounded inventory
- packaging feasibility
- smoke validations
- command/runtime verification

## Rules

- no long research loops
- no reopening failed `next_2_steps` branch as main narrative
- bounded validations only
- keep product surface clean and boring
- archive experimental clutter conceptually, even if physical moves happen later

## Immediate Execution Order

1. finish product framing
2. define keep/archive boundary
3. write top-level README
4. add CLI skeleton
5. add MCP skeleton
6. validate and package next step
