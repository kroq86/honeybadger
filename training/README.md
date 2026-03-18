# Phase 2 Training Layout

This directory holds the first local fine-tuning scaffolding for Phase 2.

Current scope:

- freeze one trainable base model path
- freeze one SFT export format
- define one conservative local LoRA profile
- define one eval config for before/after comparison

Current selected trainable model path:

- primary: `microsoft/Phi-3-mini-4k-instruct`
- fallback: gated Meta-Llama path only if authenticated access exists

Current training data export:

- `training_data/sft_v1`

Current execution benchmark root:

- `datasets/mvp`

## Files

- `export_sft.py`
  - thin wrapper around the repo-level SFT export
- `train_lora.py`
  - first training entrypoint with config loading and guardrails
- `eval_checkpoint.py`
  - eval entrypoint placeholder for before/after benchmark runs
- `phase2_freeze.py`
  - writes one machine-readable freeze manifest tying model, datasets, stages, metrics, and guardrails together
- `check_env.py`
  - prints a JSON report for the required local training stack
- `requirements-phase2.txt`
  - minimal Python package target for the first local LoRA path
- `configs/lora_smoke.json`
  - conservative smoke profile
- `configs/eval_phase2.json`
  - before/after eval config

Current freeze manifest:

- `training/phase2_freeze.json`

## Current Guardrails

- LoRA only for the first pass
- no full fine-tune
- small sequence length
- small batch size
- gradient accumulation
- explicit output directory
- intended for one smoke run before any real run

Training and eval results depend on the dataset, splits, and compute; document or cite these when reporting.
