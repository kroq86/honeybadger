# Papers To Extract For VMBench

## Purpose

Turn the relevant papers into concrete experiments for the current `verifier/search/ranker` branch.

## 1. Rethinking Optimal Verification Granularity

### What To Take

- verifier granularity is a controllable design variable
- finer or coarser verification can change search quality under fixed compute budget

### How To Map It Here

- compare:
  - full-step verifier
  - final-state-only verifier
  - state-diff verifier
  - any partial-step verifier we can define cleanly

### Immediate Experiment

- run the same held-out benchmark with budgets `4`, `8`, `12`
- produce one table:
  - solve rate
  - avg nodes
  - budget exhausted
  - false accept / false reject counts

## 2. Scaling Flaws of Verifier-Guided Search

### What To Take

- verifier-guided search can fail badly when local verification signal is wrong or misaligned

### How To Map It Here

- explicit failure buckets:
  - verifier false accept
  - verifier false reject
  - candidate omission
  - ranker miss
  - budget miss

### Immediate Experiment

- build a failure analyzer over held-out misses
- sample concrete failures and inspect them with per-step traces

## 3. Transformers Struggle to Learn to Search

### What To Take

- do not assume end-to-end training will spontaneously recover explicit search structure

### How To Map It Here

- keep the search loop explicit
- evaluate learned components as assist modules, not as replacements for the whole system

### Immediate Experiment

- compare:
  - heuristic ranker
  - learned ranker
  - no ranker
- under the same explicit search loop and node budgets

## 4. Accelerating LLM Reasoning via Speculative Search

### What To Take

- search quality is only half the story
- compute-budget efficiency matters

### How To Map It Here

- build budget curves, not just one-budget snapshots
- optimize nodes-to-solution and budget-exhaustion rate

### Immediate Experiment

- node budgets `2`, `4`, `8`, `12`, `16`
- compare `heuristic` vs `learned`
- report solve rate vs budget and avg nodes vs budget

## Hidden Questions These Papers Force

- which verifier granularity is actually robust?
- how much of learned success is heuristic imitation?
- how much of the current gain depends on leakage-like features?
- where does search stop being operationally cheap?
- can candidate-source shortcuts be replaced by lower-leakage structural features without losing harder-split quality?

## Decision Rule

If these experiments stay strong on a harder benchmark, the `neuro-symbolic VM` branch remains alive. If they collapse once verifier granularity, leakage, or benchmark easiness is controlled, we should stop pretending this is a general search-capable VM direction.

Either way, report scope clearly: which benchmarks, splits, and runs; variability and compute where relevant.
