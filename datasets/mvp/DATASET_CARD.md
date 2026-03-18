# Dataset Card: AI-Assembly ISA v0 MVP

## Summary

This dataset contains supervised execution examples generated from the reference AI-Assembly ISA v0 VM.
It includes the following dataset types: next_2_steps, short_trace, single_step, terminal_state.

## Intended Use

- single-step execution supervision
- next-2-steps trace supervision
- full short-trace supervision
- terminal-state-only supervision
- curriculum learning from explicit execution to compressed execution

## Split Strategy

Splits are performed at the program family level.
All examples sharing the same `split_family` are assigned to exactly one of train, val, or test.
This prevents leakage where the same program/input family appears across splits.

## Generation

- Seed: 7
- Max steps per program: 64
- Split key: `split_family`
- Split strategy: `program_family_level`

## Dataset Sizes

- Overall counts: {"single_step": 128, "next_2_steps": 35, "short_trace": 35, "terminal_state": 35}
- Split counts: {"single_step": {"train": 87, "val": 1, "test": 40}, "next_2_steps": {"train": 22, "val": 1, "test": 12}, "short_trace": {"train": 27, "val": 4, "test": 4}, "terminal_state": {"train": 25, "val": 1, "test": 9}}

## Categories

Programs currently cover arithmetic, memory, branch, straight-line, and loop tasks.

## Known Limitations

- MVP ISA only; no stack, calls, dynamic addressing, or bitwise ops yet. Results do not generalize to full-feature ISAs.
- Synthetic programs are still small and structured; performance on larger or different program distributions is unknown.
- Loop diversity is limited to bounded counter-style programs; unbounded or irregular loops may hurt accuracy.
- Local `short_trace` evaluation should stay capped to short traces only.

## License and Citation

License: see repository root. If you use this dataset, cite the vmbench repository and the description of the AI-Assembly ISA and generation procedure.

## Files

- `single_step/train.jsonl`, `single_step/val.jsonl`, `single_step/test.jsonl`
- `next_2_steps/train.jsonl`, `next_2_steps/val.jsonl`, `next_2_steps/test.jsonl`
- `short_trace/train.jsonl`, `short_trace/val.jsonl`, `short_trace/test.jsonl`
- `terminal_state/train.jsonl`, `terminal_state/val.jsonl`, `terminal_state/test.jsonl`
- `manifest.json`
- `generation_config.json`
