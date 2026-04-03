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

- `two_step_windows` held-out test cohort: `N = 31`
- harder split (`window_start >= 2`, filtered branch/loop/search slice): `N_harder = 224`
- held-out trace-ranking evaluation: `N_trace = 66`

### 4.2 Metrics

- `solve rate`: fraction of instances solved within the node budget
- `average nodes`: arithmetic mean of explored nodes per instance
- `budget-exhaustion rate`: fraction of unsolved instances that consume the full budget
- `top-k trace ranking`: fraction of held-out trace rows for which the gold candidate appears in the top `k`

### 4.3 Reproducibility Notes

The VM, transition verifier, and search loop are deterministic given fixed inputs. The learned ranker remains subject to training-stack and hardware details. The repository contains the benchmark generator, verifier, search stack, CLI, MCP server, and tests. Main benchmark and runtime flows do not require specialized hardware; training-related paths may use a single GPU.

## 5. Results

All numbers below are from the stated benchmark cohorts and are not averaged over multiple random restarts unless explicitly stated.

### 5.1 Held-Out Two-Step Test

Node budget: `12`. Cohort size: `N = 31`.

| Policy     | Solve rate | Avg nodes | Successes |
|------------|------------|-----------|-----------|
| No ranker  | 0.3548     | 9.55      | 11        |
| Heuristic  | 0.9677     | 2.84      | 30        |
| Learned    | 0.9677     | 2.39      | 30        |

The main effect is large: explicit ranking dramatically improves solve rate over unranked search under the same budget, and the learned ranker uses fewer nodes on average than the heuristic ranker on this cohort.

### 5.2 Harder Split

Cohort size: `N_harder = 224`.

| Policy     | Solve rate | Successes |
|------------|------------|-----------|
| Heuristic  | 0.9420     | 211       |
| Learned    | 0.9911     | 222       |

On this filtered harder slice, the learned ranker slightly outperforms the heuristic ranker.

### 5.3 Held-Out Trace Ranking

Cohort size: `N_trace = 66`.

- Top-1: `64/66 = 0.9697`
- Top-3: `65/66 = 0.9848`

### 5.4 Interpretation

The strongest signal in the current results is that search order matters more than one-shot next-step prediction quality. Moving from no ranking to heuristic or learned ranking yields a large gain, while keeping the verifier fixed. The learned ranker appears most useful when budget is tight and on the harder slice. These findings support an interpretation centered on verifier-guided search efficiency rather than latent internal execution.

## 6. Limitations

The scope of the claims is narrow.

- The benchmark is synthetic and based on a small toy ISA.
- Results are limited to the current benchmark family and short-horizon settings.
- The verifier is a critical dependency; bugs in the verifier would invalidate conclusions.
- The learned ranker benefits from structural cues such as candidate source, which weakens claims about low-leakage generalization.
- The current report does not include broader multi-seed variance, longer-horizon transfer, or full verification-granularity ablations.

## 7. Future Work

The most important next experiments are:

- budget sweeps across multiple node limits;
- explicit failure-taxonomy analysis;
- verification-granularity ablations;
- transfer tests on harder start states and longer horizons;
- reduced-shortcut settings for candidate features.

These are the right empirical gates before making any broader claim about general reasoning or generalization across execution domains.

## 8. Reproducibility and Artifact Availability

The repository contains the reference VM, ISA specification, dataset generator, search stack, rankers, CLI, MCP server, and tests. The MVP dataset is synthetic and can be regenerated from the provided pipeline using the documented seed and split policy. The artifact is therefore intended to be reproducible at the level of benchmark generation, verifier behavior, ranking evaluation, and runtime inspection.

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
