# Repo Product Map

## What This Repo Actually Is

This repo is not one product. It is a layered research workbench with four main parts:

1. a formal toy execution environment
2. dataset generation for execution-style supervision
3. local inference / fine-tuning / evaluation harnesses
4. a very large archive of experiment branches and reports

The strongest reusable code is in the execution environment and evaluation stack, not in the failed `next_2_steps` research branch.

## Repo Map

### 1. Core VM / Semantics

- [ISA_SPEC.md](/Users/ll/honeybadger/ISA_SPEC.md)
  - formal spec for the toy ISA
- [reference_vm.py](/Users/ll/honeybadger/reference_vm.py)
  - parser
  - machine state
  - execution engine
  - trace / transition collection

This is the cleanest core in the repo.

### 2. Dataset Generation

- [dataset_pipeline.py](/Users/ll/honeybadger/dataset_pipeline.py)
  - generates core execution datasets
  - current active MVP stages:
    - `single_step`
    - `next_2_steps`
    - `short_trace`
    - `terminal_state`
- [synthesis_dataset.py](/Users/ll/honeybadger/synthesis_dataset.py)
  - program synthesis dataset generator
- [repair_dataset.py](/Users/ll/honeybadger/repair_dataset.py)
  - program repair dataset generator
- [latent_probe_dataset.py](/Users/ll/honeybadger/latent_probe_dataset.py)
  - final-state probe dataset generator
- [ordered_conditionals_dataset.py](/Users/ll/honeybadger/ordered_conditionals_dataset.py)
  - extra targeted demo dataset

### 3. Local Inference / Benchmarking

- [baseline_trainer.py](/Users/ll/honeybadger/baseline_trainer.py)
  - local inference harness
  - parsing / repair logic
  - scoring
  - stage-specific prompts and metrics
- [curriculum_gate.py](/Users/ll/honeybadger/curriculum_gate.py)
  - gate logic for stage progression

This is the other main reusable core.

### 4. Fine-Tuning / Probe Tooling

- [training/train_lora.py](/Users/ll/honeybadger/training/train_lora.py)
  - LoRA training entrypoint
- [training/eval_checkpoint.py](/Users/ll/honeybadger/training/eval_checkpoint.py)
  - before/after checkpoint eval
- [training/check_env.py](/Users/ll/honeybadger/training/check_env.py)
  - environment check
- [training/export_sft.py](/Users/ll/honeybadger/training/export_sft.py)
  - SFT export wrapper

Probe / branch-specific tools:

- [training/prefix_forced_probe.py](/Users/ll/honeybadger/training/prefix_forced_probe.py)
- [training/quick_stage_probe.py](/Users/ll/honeybadger/training/quick_stage_probe.py)
- [training/first_token_mode_probe.py](/Users/ll/honeybadger/training/first_token_mode_probe.py)
- `training/next2_*_export.py`
- `training/configs/*.json`

These are useful, but many are branch-specific and should not all survive into a cleaned product.

### 5. Data / Artifacts

- [datasets/mvp](/Users/ll/honeybadger/datasets/mvp)
  - main benchmark dataset root
- [training_data](/Users/ll/honeybadger/training_data)
  - exported SFT datasets
- [training_runs](/Users/ll/honeybadger/training_runs)
  - model training outputs
- [reports](/Users/ll/honeybadger/reports)
  - experiment logs, verdicts, probes, templates

### 6. External Reference Material

- [fasm-handbook](/Users/ll/honeybadger/fasm-handbook)
  - reference repository
  - useful as inspiration and examples
  - not part of the product core

## What Is Productizable

### Productizable Core A: VM Execution Benchmark Kit

This is the strongest product candidate.

It would include:

- [reference_vm.py](/Users/ll/honeybadger/reference_vm.py)
- [dataset_pipeline.py](/Users/ll/honeybadger/dataset_pipeline.py)
- [baseline_trainer.py](/Users/ll/honeybadger/baseline_trainer.py)
- [curriculum_gate.py](/Users/ll/honeybadger/curriculum_gate.py)
- [datasets/mvp](/Users/ll/honeybadger/datasets/mvp)

What it becomes:

- a small benchmark/workbench for testing whether models can execute a formal VM
- a reproducible harness for:
  - `single_step`
  - `terminal_state`
  - optional multi-step probes

### Productizable Core B: Synthetic Training Data Factory

This would package:

- [dataset_pipeline.py](/Users/ll/honeybadger/dataset_pipeline.py)
- [synthesis_dataset.py](/Users/ll/honeybadger/synthesis_dataset.py)
- [repair_dataset.py](/Users/ll/honeybadger/repair_dataset.py)
- [latent_probe_dataset.py](/Users/ll/honeybadger/latent_probe_dataset.py)

What it becomes:

- a generator for synthetic execution / synthesis / repair datasets
- useful for training and eval research

### Productizable Core C: Local LLM Execution Eval Harness

This would package:

- [baseline_trainer.py](/Users/ll/honeybadger/baseline_trainer.py)
- [training/eval_checkpoint.py](/Users/ll/honeybadger/training/eval_checkpoint.py)
- [training/train_lora.py](/Users/ll/honeybadger/training/train_lora.py)
- [training/check_env.py](/Users/ll/honeybadger/training/check_env.py)

What it becomes:

- a local eval + LoRA smoke toolkit for “can this model follow execution-style targets?”

## What Is Mostly Research Debris

These are useful as history, but not good product surface:

- most of [reports](/Users/ll/honeybadger/reports)
- most `next2_*` branch-specific exports and configs
- old research plans, notes, and experiment scaffolds that are no longer part of the product contract
- negative-result branch artifacts around `next_2_steps`

Keep them as archive, but they should not define the public shape of the repo. Scope and limitations of current results are in the README and dataset card.

## Best Product Direction

The best realistic product from this code is:

## `vm-bench`

A compact developer tool for:

- defining a small formal ISA
- generating synthetic execution datasets
- evaluating local or remote LLMs on execution tasks
- comparing checkpoints or prompting strategies

In plain English:

not “AI-native language inside the model”

but:

**“a benchmark and tooling kit for measuring whether models can execute formal machine semantics.”**

That is credible, shippable, and already mostly present in the code.

## What To Cut If We Productize

If we turn this into a product, the first cleanup would be:

1. archive most of [reports](/Users/ll/honeybadger/reports)
2. keep only one training path in [training](/Users/ll/honeybadger/training)
3. keep only one benchmark root in [datasets/mvp](/Users/ll/honeybadger/datasets/mvp)
4. move branch-specific experimental probes into `research_archive/`
5. add one top-level README describing:
   - what the VM is
   - how to generate datasets
   - how to run eval
   - how to compare models

## Bottom Line

This repo does contain a product-shaped core.

It is not the “latent internal executor” story.

It is:

**a formal VM + synthetic dataset + eval harness stack for testing execution fidelity in language models.**
