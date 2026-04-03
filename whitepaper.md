# vmbench: A Formal VM Benchmark and Inspectable Reasoning Runtime

Kirill Ostapenko  
Independent Researcher, Georgia  
`djkroq@gmail.com`

April 2026

## Abstract

We present `vmbench`, a formal virtual-machine execution benchmark and inspectable reasoning runtime for testing whether language models can follow machine-like semantics on synthetic execution tasks. The central empirical claim is narrow: on this benchmark, pure-neural multi-step execution is weak, while explicit verifier-guided search with ranking is strong and inspectable under bounded compute. The benchmark contributes a deterministic toy ISA (`AI-Assembly v0`), a reference VM, canonical traces, and family-level data splits designed to reduce leakage. The runtime contributes bounded candidate generation, deterministic transition verification, heuristic and learned ranking, and an interface that exposes candidate choice, verification outcomes, budget use, and failure modes. On the reported cohorts, ranking plus verification substantially improves solve rate over unranked search under the same node budget. These results are limited to the current benchmark family and do not establish broad generalization to other ISAs, longer horizons, or latent symbolic execution in model weights.

## 1. Introduction

Language-model reasoning is often evaluated through free-form textual tasks, but many questions about algorithmic behavior are easier to study when correctness is defined by executable semantics. This paper studies a narrow version of that problem: whether a system can make semantically correct progress on controlled virtual-machine execution tasks, and whether explicit verification plus bounded search improves performance over unranked candidate search under the same compute budget.

Early work in this repository explored direct multi-step trace prediction. That direction did not produce a reliable pure-neural execution story. The stronger result was a different system design: keep execution semantics in a deterministic verifier, use ranking to allocate search budget, and expose the resulting decision process through an inspectable runtime. This shift defines the scope of the paper.

The main contributions are:

- a formal VM benchmark built around a deterministic toy ISA, reference VM, and program-family-level data splits;
- a verifier-guided search pipeline with candidate generation, transition verification, and heuristic or learned ranking under a fixed node budget;
- empirical evidence that search order and verification dominate raw next-step prediction quality on this benchmark;
- an inspectable runtime surface through CLI and MCP interfaces.

We do not claim that latent symbolic execution emerged in model weights. We also do not claim that the reported gains generalize beyond the present benchmark family.

## 2. Related Work and Positioning

This work is closest to research on formal evaluation, verifier-guided reasoning, and test-time search for language models. Chain-of-thought prompting showed that reasoning performance can improve when models are encouraged to generate intermediate steps (Wei et al., 2022), and self-consistency further improved such pipelines through multi-sample decoding and voting (Wang et al., 2023). Subsequent work studied more explicit search and deliberation, including Tree of Thoughts (Yao et al., 2023), verifier-based supervision (Cobbe et al., 2021), execution-guided decoding (Wang et al., 2018), and Program of Thoughts prompting for tasks where symbolic computation matters (Chen et al., 2023).

Our system is narrower than those general reasoning methods. The key question here is not whether a model can produce plausible explanations, but whether a system can follow machine-like state transitions when correctness is defined by a reference execution model. In that sense, vmbench should be read as a benchmark-and-runtime contribution rather than a general claim about neural reasoning.

The paper also relates to work on algorithmic or execution-oriented models, including Learning to Execute (Zaremba and Sutskever, 2014) and Neural Programmer-Interpreters (Reed and de Freitas, 2016), where correctness is grounded in explicit semantics rather than subjective judgment. Our contribution is modest within that space: a small deterministic ISA, a reproducible benchmark pipeline, and a runtime that keeps verification outside the model while exposing ranked search behavior to the user. At the systems level, the project also aligns with recent work on allocating inference-time compute more effectively rather than attributing gains to hidden internal execution alone (Snell et al., 2024).

## 3. Benchmark and Method

### 3.1 AI-Assembly ISA v0

The benchmark is built on a small deterministic ISA with:

- 8 signed 16-bit general-purpose registers;
- three flags (`Z`, `N`, `C`);
- flat memory with 256 cells;
- indexed input and output arrays;
- instructions including `CONST`, `MOV`, `LOAD`, `STORE`, `READ`, `WRITE`, `ADD`, `SUB`, `CMP`, `TEST`, conditional jumps, `JMP`, and `HALT`.

The MVP ISA excludes stacks, calls, dynamic addressing, and bitwise operations. A reference VM executes programs step by step and produces canonical traces. This provides the ground-truth semantics used by the verifier.

### 3.2 Dataset

The dataset is fully synthetic and is generated from the reference VM. No human subjects or crowdsourced data are used. The current pipeline produces `single_step`, `next_2_steps`, `short_trace`, and `terminal_state` tasks. Splits are defined at the program-family level so that all examples sharing a `split_family` belong to exactly one of train, validation, or test. The reported runs use seed `7` and a maximum of `64` steps per program.

### 3.3 Candidate Generation

Given a program identifier and serialized VM state, candidate generation proposes a compact set of plausible next instructions. The current implementation supports local candidates around the current instruction pointer and broader program scans. Candidates are deduplicated before ranking and retain provenance labels such as `current_ip`, `jump_target`, or `next_ip`. These labels are useful both for ranking and for runtime inspection.

Candidate generation is intentionally conservative. It narrows the branching factor, but it does not determine correctness.

### 3.4 Transition Verification

Transition verification is the semantic core of the system. Given a program, a current state, and a candidate instruction, the verifier resolves the instruction against the program, executes exactly one VM step, and returns a structured verdict. Failure modes include mismatched instruction text, incorrect control-flow target, and incorrect state transition.

This design keeps correctness outside the model. Learned or heuristic components may propose and prioritize candidates, but acceptance is determined by the formal execution contract.

### 3.5 Ranking and Budgeted Search

The runtime supports three ranking policies:

- no ranker;
- a heuristic ranker based on candidate provenance and lightweight state cues;
- a learned ranker trained on labeled search-trace examples.

All three policies operate over the same candidate set and the same verifier-backed search loop. For `next_2_steps` tasks, the runtime performs bounded depth-2 search: it ranks and verifies candidates for the first transition, then repeats the process for the second transition if the first step is accepted. Search stops when a valid path is found or the node budget is exhausted.

The learned component is a lightweight feature-weight model rather than a neural runtime policy. Its role is to improve search order under a fixed budget, not to replace formal verification.

## 4. Experimental Setup

### 4.1 Evaluation Cohorts

- current checked-in `next_2_steps` held-out test cohort: `N = 12`
- full harder-start `two_step_windows` slice built from `TASK_LIBRARY` with `window_start >= 2` and categories `branch`, `loop`, `search`: `N_harder = 224`
- exported branch-ranking traces from the checked-in train split: `N_train_trace = 39` train rows and `N_val_trace = 5` validation rows
- exported branch-ranking traces from the checked-in test split: `N_test_trace = 24` total rows after export, split internally into `16` train-bucket rows and `8` val-bucket rows by the trace exporter

### 4.2 Metrics

- `solve rate`: fraction of instances solved within the node budget
- `average nodes`: arithmetic mean of explored nodes per instance
- `budget-exhaustion rate`: fraction of unsolved instances that consume the full budget
- `top-k trace ranking`: fraction of held-out trace rows for which the gold candidate appears in the top `k`

### 4.3 Reproducibility Notes

The VM, transition verifier, and search loop are deterministic given fixed inputs. For the experiments added in this revision, budget sweeps were run with `candidate_mode=program_global`, `candidate_limit=32`, and `target_mode=intermediate_oracle`, which is the main setting in the current repo state where budget and ranking meaningfully separate policies. The learned ranker was trained from exported search traces generated from the checked-in train split. The repository contains the benchmark generator, verifier, search stack, CLI, MCP server, and tests. Main benchmark and runtime flows do not require specialized hardware; training-related paths may use a single GPU.

## 5. Results

All numbers below are from the stated benchmark cohorts and are not averaged over multiple random restarts unless explicitly stated.

### 5.1 Held-Out Two-Step Test

Node budget: `12`. Cohort size: `N = 12`.

| Policy     | Solve rate | Avg nodes | Successes |
|------------|------------|-----------|-----------|
| No ranker  | 1.0000     | 3.00      | 12        |
| Heuristic  | 1.0000     | 2.00      | 12        |
| Learned    | 1.0000     | 4.00      | 12        |

At budget `12`, all three policies solve the current checked-in test set. The meaningful difference at this budget is efficiency rather than solve rate: heuristic ranking uses fewer nodes on average than either no ranking or the learned ranker trained from the current trace export.

### 5.2 Budget Sweep

The sharper comparison in the current repo state comes from low budgets. On the held-out `N = 12` test set with `candidate_mode=program_global` and `target_mode=intermediate_oracle`, budget `2` separates policies strongly.

| Node budget | No ranker solve rate | Heuristic solve rate | Learned solve rate |
|------------|------------------------|----------------------|--------------------|
| 2          | 0.0000                 | 1.0000               | 0.0000             |
| 4          | 1.0000                 | 1.0000               | 1.0000             |
| 8          | 1.0000                 | 1.0000               | 1.0000             |
| 12         | 1.0000                 | 1.0000               | 1.0000             |
| 16         | 1.0000                 | 1.0000               | 1.0000             |

This sweep shows that the current held-out set is easy once the node budget reaches `4`, but it still captures a real low-budget effect: heuristic ordering solves every record at budget `2`, whereas no ranking and the current learned ranker exhaust budget on all `12/12` records.

### 5.3 Held-Out Trace Ranking

Cohort sizes for the exported trace-ranking eval are small in the current repo snapshot. The learned ranker trained from the checked-in train split reaches:

- on the exported test train-bucket rows: Top-1 `9/16 = 0.5625`, Top-3 `9/16 = 0.5625`
- on the exported test val-bucket rows: Top-1 `3/8 = 0.3750`, Top-3 `8/8 = 1.0000`

### 5.4 Verification-Granularity Ablation

We ran verifier-granularity ablations on the checked-in `N = 12` held-out test set over `instruction_only`, `state_diff`, `final_state_only`, and `intermediate_oracle`. At node budget `2`, all three non-oracle verification modes make the task trivially easy: every policy reaches solve rate `1.0000` with `2.0` average explored nodes. Under `intermediate_oracle`, however, the low-budget separation reappears: heuristic still solves `12/12`, while both no-ranker and learned solve `0/12`.

This ablation shows that verification granularity is not a cosmetic implementation detail. On the current snapshot, relaxed verifier modes collapse the distinction between policies, whereas the strict intermediate-oracle setting preserves the bounded-search question the benchmark is intended to study.

### 5.5 Harder-Start Transfer

To test whether the search-order story survives beyond the tiny checked-in test split, we rebuilt the full two-step benchmark from `TASK_LIBRARY` and filtered it to the harder-start slice with `window_start >= 2` and categories `branch`, `loop`, and `search`. This yields `N_harder = 224` instances.

At node budget `12` with `candidate_mode=program_global` and `target_mode=intermediate_oracle`, the policies differ substantially:

| Policy     | Solve rate | Avg nodes | Successes |
|------------|------------|-----------|-----------|
| No ranker  | 0.4464     | 10.86     | 100       |
| Heuristic  | 0.9420     | 3.63      | 211       |
| Learned    | 0.8482     | 7.91      | 190       |

This harder-start transfer restores a stronger version of the core claim. Search order matters substantially on this slice, and the heuristic ranker remains clearly stronger than no ranking. In the current reproducible workspace, however, the learned ranker does not match the heuristic on the harder split.

### 5.6 Reduced-Shortcut Candidate-Feature Ablations

We trained additional learned rankers with candidate-source shortcuts removed. Two variants were tested: `no_source` and `structural_no_source`, where structural source hints are reintroduced without the raw candidate-source feature.

On the checked-in `N = 12` test set, both ablated models still solve all records by budget `8`, but both are worse than the original learned ranker in node efficiency:

- `no_source`: solve rate `1.0000` at budget `8`, average explored nodes `6.08`
- `structural_no_source`: solve rate `1.0000` at budget `8`, average explored nodes `6.50`
- original learned model: solve rate `1.0000` at budget `8`, average explored nodes `4.00`

On the harder-start `N_harder = 224` split at budget `12`, shortcut removal causes a large drop:

- original learned model: `190/224 = 0.8482`
- `no_source`: `96/224 = 0.4286`
- `structural_no_source`: `98/224 = 0.4375`

So in the current workspace, shortcut dependence is empirically real: removing candidate-source information roughly halves harder-split solve rate, and the structural replacement does not recover the lost performance.

### 5.7 Interpretation

The strongest signal in the current reproducible workspace is narrower than the earlier draft version of this paper. Search order still matters, but the current checked-in held-out set only exposes that difference at the smallest tested budget. In that low-budget regime, heuristic ranking is clearly useful. The current learned ranker is mixed: it does not match the heuristic on this small setup, and its trace-ranking metrics indicate that it is not yet a robust replacement for the hand-designed ordering policy.

### 5.8 Failure Taxonomy

We ran explicit failure analysis at node budget `2`, the only regime in the current checked-in test set where failures occur. The resulting taxonomy is simple but informative:

- no ranker: `12/12` failures classified as `budget_or_rank_order`
- heuristic: `12/12` records solved
- learned: `12/12` failures classified as `budget_or_rank_order`

In other words, failures on this test set are not currently caused by missing first-step candidates or verifier rejection of the gold path. They are caused by candidate ordering under a tight budget. The learned ranker is slightly worse than the heuristic on this analysis: for representative failures, the best second-step candidate appears at rank `3` under the learned policy rather than rank `2` under the no-ranker baseline, increasing the minimum nodes needed from `3` to `4`.

## 6. Limitations

The scope of the claims is narrow.

- The benchmark is synthetic and based on a small toy ISA.
- Results are limited to the current benchmark family and short-horizon settings.
- The verifier is a critical dependency; bugs in the verifier would invalidate conclusions.
- The learned ranker benefits from structural cues such as candidate source, which weakens claims about low-leakage generalization.
- The current report does not include broader multi-seed variance or a longer-horizon search runtime beyond two-step search.

## 7. Future Work

The most important next experiments are:

- longer-horizon transfer with a search runtime that is not restricted to two-step search;
- broader multi-seed reporting for the learned-ranker path;
- stronger structural replacements for candidate-source shortcuts that preserve harder-split quality.

The budget sweep, failure-taxonomy pass, verification-granularity ablation, harder-start transfer, and reduced-shortcut ablations are now implemented and reported for the current reproducible workspace. The remaining empirical gates are therefore the ones above.

## 8. Reproducibility and Artifact Availability

The repository contains the reference VM, ISA specification, dataset generator, search stack, rankers, CLI, MCP server, and tests. The MVP dataset is synthetic and can be regenerated from the provided pipeline using the documented seed and split policy. The artifact is therefore intended to be reproducible at the level of benchmark generation, verifier behavior, ranking evaluation, and runtime inspection. In particular, the current repo includes dedicated scripts for budget sweeps, failure analysis, and verifier-granularity ablations over the `next_2_steps` benchmark.

## 9. Broader Impact

The main positive use of this work is as a benchmark and runtime for studying transparent, verifier-backed reasoning behavior. The main risk is misinterpretation: strong benchmark performance in this setting should not be read as evidence of broad reasoning competence or latent symbolic execution. The paper therefore keeps the claim boundary explicit.

## References

- Chen, W., Ma, X., Wang, X., and Cohen, W. W. (2023). Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks. arXiv:2211.12588.
- Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168.
- Nye, M., Andreassen, A., Gur-Ari, G., Michalewski, H., Austin, J., Bieber, D., Dohan, D., Lewkowycz, A., Bosma, M., Luan, D., Sutton, C., and Odena, A. (2022). Show Your Work: Scratchpads for Intermediate Computation with Language Models. arXiv:2112.00114.
- Reed, S., and de Freitas, N. (2016). Neural Programmer-Interpreters. arXiv:1511.06279.
- Snell, C., Lee, J., Xu, K., and Kumar, A. (2024). Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. arXiv:2408.03314.
- Wang, C., Tatwawadi, K., Brockschmidt, M., Huang, P.-S., Mao, Y., and Polozov, O. (2018). Robust Text-to-SQL Generation with Execution-Guided Decoding. arXiv:1807.03100.
- Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv:2203.11171.
- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q. V., and Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. arXiv:2201.11903.
- Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. arXiv:2305.10601.
- Zaremba, W., and Sutskever, I. (2014). Learning to Execute. arXiv:1410.4615.
