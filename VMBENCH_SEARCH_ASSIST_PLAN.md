# VMBench Search-Assist Execution Plan

## Goal

Preserve the useful part of the VM direction by treating it as a `neuro-symbolic search system`, not as a pure latent-VM training problem.

The current target is not:

- another `next_2_steps` LoRA run
- another `single_step` metric bump
- a claim that internal symbolic execution emerged in weights

The current target is:

- deterministic transition verification
- bounded candidate generation
- budgeted search
- heuristic and learned branch ranking
- transfer beyond the current easy slices

## Status Snapshot

### Completed Milestones

- `M0 baseline snapshot`: done
- `M1 transition verifier`: done
- `M2 candidate generator`: done
- `M3 rule/budgeted search`: done
- `M4 heuristic ranker`: done
- `M5 search trace export`: done
- `M6 learned ranker smoke`: done

### Key Observed Results

(All from the stated benchmarks and splits; full reporting would give variability across runs and compute.)

- `single_step` candidate recall@1 on current slice: `1.0`
- `next_2_steps` first-step recall@1 on current slice: `1.0`
- expanded `two_step_windows` benchmark:
  - `no ranker`, budget `12`: solve_rate `0.4343`, avg_nodes `9.20`
  - `heuristic ranker`, budget `12`: solve_rate `0.9484`, avg_nodes `2.91`
- held-out test split of `two_step_windows`:
  - `no ranker`: solve_rate `0.3548`, avg_nodes `9.55`
  - `heuristic`: solve_rate `0.9677`, avg_nodes `2.84`
  - `learned`: solve_rate `0.9677`, avg_nodes `2.39`
- learned trace ranking on held-out trace rows:
  - top1 `0.9697`
  - top3 `0.9848`

### What Survived

- the `pure neural VM` story is weak
- the `search/verifier/ranker` story is strong
- `single_step` mattered as scaffolding, not as the final result

## Posterior View

This section is an operational posterior, not a publication claim.

| Next objective | Posterior |
| --- | ---: |
| Normal learned-ranker train/eval loop | `0.92` |
| Stable `learned > heuristic` on current benchmark | `0.58` |
| Transfer to a harder 2-step benchmark | `0.63` |
| Transfer beyond synthetic program-derived windows | `0.41` |
| Strong `next_2_steps` result from pure HF/LoRA without search assist | `0.14` |
| Practically useful neuro-symbolic branch | `0.76` |

## Hidden Variables

These are the main hidden risks that should shape the next phase.

### H1. Verifier Correctness Risk

- we already found one critical verifier bug
- any remaining verifier bug can fake search quality
- verifier confidence is now a gating dependency for all later claims

### H2. Benchmark Easiness Risk

- the first `matrix_seed11 next_2_steps` slice was too easy
- all records started from `IP=0`
- current `two_step_windows` benchmark is better, but still synthetic and derived from the program

### H3. Heuristic Imitation Risk

- the learned ranker may be compressing the heuristic, not discovering a better policy
- this is acceptable for engineering value
- it is not strong evidence of deeper algorithmic generalization

### H4. Program Leakage Risk

- the ranker currently benefits from `program_name`, `IP`, and candidate-source structure
- this is useful and intentional for system performance
- it weakens any claim about purely internal reasoning

### H5. Search-Cost Risk

- the system works because explicit search is doing real work
- if search cost scales badly, the whole branch becomes operationally expensive

### H6. Candidate-Source Shortcut Risk

- explicit `program_name` is not the dominant shortcut
- `candidate source` is currently the strongest shortcut-like feature
- removing `candidate source` drops harder-split quality much more than removing `program_name`
- richer state-only features improved trace ranking but did not recover harder-split search quality

## Research-Derived Direction

The next phase is informed by four external threads:

- `Rethinking Optimal Verification Granularity`
- `Scaling Flaws of Verifier-Guided Search`
- `Transformers Struggle to Learn to Search`
- `Accelerating Large Language Model Reasoning via Speculative Search`

These do not suggest “train the model a bit more”.
They suggest:

- study verifier granularity explicitly
- study verifier-guided failure modes explicitly
- treat search budget as a first-class metric
- treat learned ranking as search acceleration, not as end-to-end reasoning proof

## Revised Milestone Map

## Milestone 7: Verification Granularity Study

### Goal

Determine which verifier granularity gives the best quality/cost tradeoff:

- full-step transition verification
- partial-step verification
- state-diff verification
- final-state-only verification

### Why This Exists

This is now the biggest technical risk.
The papers strongly suggest that verifier granularity can dominate search behavior under fixed budget.

### What To Write

- a verifier-ablation harness
- explicit verifier modes
- a comparison report under fixed node budgets

### File Targets

- extend `vm_transition_verifier.py`
- add a verifier ablation runner under `tools/` or `training/`

### Required Runs

1. `two_step_windows`, budgets `4`, `8`, `12`
2. compare `intermediate_oracle`, `final_state_only`, and at least one new finer-grained mode
3. log solve rate, nodes, and false-accept behavior

### Done When

- we know which verification granularity is worth carrying forward

## Milestone 8: Failure Modes of Verifier-Guided Search

### Goal

Map where the current pipeline fails and whether the failure is caused by:

- verifier false accept
- verifier false reject
- candidate miss
- ranker miss
- budget exhaustion

### What To Write

- failure taxonomy
- per-record explain artifact
- aggregate failure report

### File Targets

- extend `search_runner.py` or add `search_failure_analysis.py`

### Required Runs

1. analyze failures on held-out `two_step_windows` test
2. analyze failures on a harder benchmark slice
3. compare `no ranker`, `heuristic`, `learned`

### Done When

- every unresolved failure falls into a known bucket

## Milestone 9: Harder Benchmark Construction

### Goal

Move beyond the current easy or program-derived slices.

### Benchmark Priorities

1. deeper `two_step_windows` with harder start states
2. held-out program families
3. longer horizons than 2 steps
4. specialized branches such as `next_2_effects_target_anchor`

### What To Write

- benchmark builder for harder held-out slices
- manifest with difficulty tags

### Required Runs

1. train on current search traces
2. evaluate on harder split with no retraining
3. compare `no ranker`, `heuristic`, `learned`

### Done When

- we know whether the pipeline transfers or only memorizes the current benchmark family

## Milestone 10: Compute-Budget Search

### Goal

Treat search cost as a first-class objective.

### What To Measure

- solve rate as a function of node budget
- nodes-to-solution
- budget-exhaustion rate
- quality under tiny budgets

### Why This Matters

The speculative-search papers suggest the next real gain may come from budget-aware routing and pruning, not from a more expressive ranker alone.

### Required Runs

1. budgets `2`, `4`, `8`, `12`, `16`
2. compare `heuristic` and `learned`
3. report area under budget curve or equivalent summary

### Done When

- we know where learned ranking actually buys budget efficiency

## Milestone 11: Learned Ranker Proper

### Goal

Turn the current smoke into a reproducible train/eval loop.

### What To Write

- stable train config
- stable eval config
- ablation by feature availability
- checkpointed reports

### Required Evals

1. current trace split
2. harder benchmark split
3. budget curve vs heuristic
4. ablation without `program_name`
5. ablation without candidate-source tags
6. replacement of candidate-source tags with lower-leakage structural candidate features

### Done When

- learned ranker either clearly beats heuristic or is proven not worth further work on the current features
- and we understand whether its gains survive without explicit candidate-source shortcuts

## Milestone 11A: Candidate-Source Replacement

### Goal

Reduce reliance on `candidate source` without collapsing search quality.

### Current Evidence

- `drop program_name` is nearly free
- `drop candidate source` causes a large drop
- richer state-only features help held-out trace accuracy
- richer state-only features do not yet restore harder-split search quality
- structural source-hint replacement also does not restore harder-split search quality:
  - held-out trace `top1 = 0.8182`, `top3 = 0.9697`
  - harder split `solve_rate = 0.8482`, `avg_nodes = 6.63`
  - this is worse than plain `no_source` on harder split (`0.8705`, `5.17`)

### What To Write

- a lower-leakage feature set for learned ranking based on:
  - candidate rank bucket
  - opcode family
  - register signature
  - state-shape features
  - local structural cues derived from candidate text and current state

### Required Evals

1. held-out trace top1/top3 without candidate source
2. harder split solve rate without candidate source
3. comparison against:
   - full learned ranker
   - no-program learned ranker
   - no-source baseline

### Done When

- we either recover most of the no-source loss with lower-leakage features
- or we conclude candidate-source structure is still doing irreplaceable work

## Milestone 12: Transfer or No-Go Gate

### Continue If

- learned or heuristic ranking still works on a harder slice
- verifier failure modes are understood
- budget cost remains acceptable

### No-Go If

- success collapses once benchmark easiness or leakage is reduced
- learned ranker only imitates heuristic and adds no transfer value
- verifier fragility dominates all gains

## Immediate Run Order

1. verification granularity ablation
2. failure-mode analysis on current held-out test
3. harder benchmark construction
4. budget curve for `heuristic` vs `learned`
5. learned-ranker ablation without leak-prone features
6. candidate-source replacement experiments
7. transfer test onto harder slice
8. specialized-branch recovery only after transfer signal is real

## Immediate Next Three Tasks

### Task 1

Write a verifier-granularity ablation runner and compare at least three verifier modes under fixed budgets.

### Task 2

Write a failure-analysis report that classifies every held-out miss into candidate, verifier, ranker, or budget failure.

### Task 3

Build a harder benchmark split with held-out program families or deeper starting states and rerun `no ranker` / `heuristic` / `learned`.

### Task 4

Design lower-leakage structural candidate features that can replace explicit `candidate source` and rerun the no-source learned-ranker benchmark.

## Go / No-Go Rules

### Go

- the system remains strong after harder benchmarking
- learned ranking buys budget efficiency or transfer
- verifier ablations identify a robust mode

### Caution

- learned equals heuristic on easy slices but does not transfer
- verifier mode changes metrics wildly
- budget efficiency gains disappear below small node budgets
- trace ranking improves but harder-split search quality does not

### No-Go

- results collapse once easy structure or leakage is reduced
- the only reliable winner is hand-coded heuristic plus explicit program-local priors

## What Not To Do Now

- do not spend time on more pure `next_2_steps` LoRA runs without search assist
- do not claim latent symbolic execution from the current results
- do not trust a benchmark until verifier correctness and benchmark hardness have been checked
- do not optimize only for solve rate without tracking budget
