# vmbench: A Formal VM Benchmark and Inspectable Reasoning Runtime

**Technical report**

---

## Abstract

We present vmbench: a formal virtual-machine execution benchmark and toolkit for testing whether language models can follow machine-like semantics on synthetic execution tasks. **The central thesis of this report is narrow and defensible:** on this benchmark, **pure-neural multi-step execution is weak**, while **explicit verifier-guided search with ranking is strong and inspectable under bounded compute**. **Below, the Foundations section inlines** the origin narrative and repository map. The main contributions are (1) a small, deterministic toy ISA (AI-Assembly v0) with a reference VM and canonical trace format; (2) synthetic dataset generation with program-family-level splits to avoid leakage; (3) a search-assisted reasoning pipeline—candidate generation, verification, and heuristic or learned ranking—under a fixed node budget; and (4) an inspectable runtime surface (MCP and CLI) that exposes options, verification, and failure modes to the user. We do not claim that latent symbolic execution emerged in the model weights. We show that explicit search plus transition verification and ranking yields large gains over unranked search on held-out and harder splits in our benchmark, and that a learned ranker can match or slightly exceed a heuristic ranker under tight budgets. Results are reported on the current benchmark and splits; generalization to other ISAs or task distributions is untested. The repo provides code, data, and instructions to reproduce the main results.

**Keywords:** execution benchmark, verifier-guided search, test-time scaling, budgeted search, inspectable reasoning.

---

## Foundations (origin and repo map — text inlined here)

### Origin: idea line that led to vmbench

Compressed narrative from the project’s origin discussion: the stack was imagined as evolving from **AI DSL → full language** (analogy: COBOL/SQL/MATLAB/Verilog: domain → DSL → language). Today’s stack is a **layer cake** (Python + prompts + JSON + agents + YAML), not one abstraction level. A unified AI-native layer would natively combine **intent, tools, state, verification, execution**. Execution **inside** the model (Sudoku, in-context algorithms) suggests **latent programs** and neural interpretation—not the same as calling Python. Mechanistic interpretability analogies: registers in the residual stream, attention as routing, behavior like **bytecode / opcodes**. Training through a **low-level VM** (registers, flags, memory, **step traces**) was proposed instead of Python-as-tool, because operational semantics need observable state transitions. **AI-Assembly**: minimal ISA and curriculum single-step → multi-step trace → latent → synthesis. Honest outcome in this repo: a **full internal executor in weights is not established**; what ships is a **benchmark VM, traces, and verifier-guided search**.

| Stage in the origin thread | Essence |
|----------------------------|---------|
| AI DSL → full language | Domain → DSL → language; today = fragmented stack, not one language. |
| What one language should carry | Intent, tools, state, **verification**, execution. |
| In-model execution | Iterative state; latent program / neural interpreter hypothesis. |
| Interpretability layer | Residual “registers”, attention as micro-ops, opcode-like analogies. |
| Why not Python as the teaching tool | Too high-level; need **stepwise state** for semantics. |
| VM curriculum | single_step → traces → … → synthesis. |
| Closing formulation | *Train LLMs in a low-level VM toward stable algorithms* — here, **external verifier + search** compensates where pure neural multi-step failed. |

**Naive-Bayes-style topic masses (must sum to 1 for consistency):** P(AI language / in-model execution)=**0.41**; P(language design / declarativity)=**0.26**; P(interpretability / latent)=**0.17**; P(agents / tools)=**0.10**; P(Lisp / symbolic history)=**0.06**. **Check: 0.41+0.26+0.17+0.10+0.06 = 1.00.**

**Intent masses (explicit fourth bucket so Σ=1):** P(design new model)=**0.48**; P(architecture exploration)=**0.29**; P(implementation)=**0.15**; P(other / mixed)=**0.08**. **Check: 0.48+0.29+0.15+0.08 = 1.00.**

**Research-potential scores (calculable mean):** novelty 7, science 8, practical 6, feasibility 7, publication 7 → **(7+8+6+7+7)/5 = 7.0 / 10**, not “about 7”. Lifting toward 9 needs provable internal step execution or clear reasoning-benchmark gain.

How the origin maps to **this** codebase:

| Origin idea | Where it lives in the repo |
|-------------|----------------------------|
| Minimal ISA, registers, flags, memory, traces | `ISA_SPEC.md`, `reference_vm.py` |
| (state, instr) → next state, traces | Dataset stages `single_step`, `next_2_steps`, `short_trace` via `dataset_pipeline.py` |
| Verifiable low-level step | `vm_transition_verifier.py` |
| Pure neural multi-step weak | Negative result on `next_2_steps`; `curriculum_gate.py`, README |
| Deterministic semantics + test-time search | `search_runner.py`, rankers, MCP — verifier-guided search |

One-line summary: from “COBOL for AI” and in-model Sudoku to **transparent VM + traces**; this repo is the **benchmark and measured outcome**—strong story is **checkable transitions and budgeted search**, not a proof of latent CPU.

### Repository map: entry points and layout

**Entry points**

| Goal | Path |
|------|------|
| CLI: generate, eval, gate, compare, status | `vmbench_cli.py` |
| MCP server | `tools/mcp/vmbench_mcp_server.py`, `tools/mcp/README.md` |
| ISA + VM | `ISA_SPEC.md`, `reference_vm.py` |
| Generate benchmark JSONL | `dataset_pipeline.py` → `datasets/mvp/` |
| Local LLM eval → summary.json | `baseline_trainer.py` (under `reports/`, often gitignored) |
| Curriculum gates | `curriculum_gate.py` |
| Budgeted search, solve rate | `search_runner.py` |
| One-step correctness | `vm_transition_verifier.py` |
| Search candidates | `candidate_generator.py` |
| Ranking | `branch_ranker.py`, `learned_branch_ranker.py` |
| LoRA | `training/train_lora.py` |
| Checkpoint eval | `training/eval_checkpoint.py` |
| Ranker train/eval | `training/train_search_ranker.py`, `training/eval_search_ranker.py` |
| SFT export | `sft_export.py`, `training/export_sft.py` |

**Layers:** VM — `ISA_SPEC.md`, `reference_vm.py`. Data gen — `dataset_pipeline.py` (MVP stages above), plus `synthesis_dataset.py`, `repair_dataset.py`, `latent_probe_dataset.py`, `ordered_conditionals_dataset.py`. Eval — `baseline_trainer.py`, `curriculum_gate.py`. Search-assist — `vm_transition_verifier.py`, `candidate_generator.py`, `branch_ranker.py`, `learned_branch_ranker.py`, `search_runner.py`, `search_trace_export.py`, `tools/build_search_benchmark.py`. Training folder — `train_lora.py`, `eval_checkpoint.py`, `eval_search_ranker.py`, `train_search_ranker.py`, `check_env.py`, `export_sft.py`, probe and `next2_*` scripts, `configs/*.json`. Product — `vmbench_product_surface.py`, `vmbench_demo_runtime.py`. Tools — `tools/mcp/`, `build_vmbench_baseline_snapshot.py`, `build_harder_search_split.py`, `run_verifier_granularity_ablation.py`, `explore_llm/capture_reasoning.py`.

**Directories:** `datasets/mvp/` (+ `DATASET_CARD.md`), `training_data/`, `training_runs/`, `reports/` (gitignored), `fasm-handbook/` (reference).

**Tests (what they lock):** `test_reference_vm.py` (VM); `test_vm_transition_verifier.py`, `test_verification_modes.py` (verifier); `test_dataset_pipeline.py`; `test_curriculum_gate.py`; `test_baseline_trainer.py`; `test_candidate_generator.py`; `test_branch_ranker.py`, `test_learned_branch_ranker.py`; `test_search_trace_export.py`; `test_sft_export.py`; `test_next2_*`, `test_collect_next2_diagnostics.py`; `test_*probe*.py`.

**Planning docs (not executable):** `REPO_PRODUCT_MAP.md`, `VMBENCH_EXECUTION_PLAN.md`, `VMBENCH_SEARCH_ASSIST_PLAN.md`, `README.md`, `datasets/mvp/DATASET_CARD.md`.

## Document map (technical report shape)

This write-up follows a standard technical report layout. Use the table if you want **problem → method → benchmark → results → limits → reproducibility** in one glance:

| Piece | Where |
|-------|--------|
| **Foundations** (origin + repo map, inlined) | Section above |
| **Problem & goals** | §1 (motivation, contributions, scope) |
| **Product surface & runtime UX** | §2.1–§2.2 |
| **Method** (search, rank, verify, budget) | §3 |
| **Benchmark** (ISA, VM, deterministic verifier, family splits, metrics) | §3.1–§3.2, §4.1, §4.3 |
| **Results** | §5 |
| **Calculability & document audit** | §5.5 |
| **Limits & risk register** | §6 |
| **Open empirical questions / exit criteria** | §7 |
| **Reproducibility** | §4.4, §8, Appendix A |

**Three headline items (additive summary, not a replacement for §1.2):** (1) small formal execution benchmark with **program-family splits** and a **deterministic** transition verifier; (2) **evidence** that **ranking + verification** under a fixed node budget beats **unranked** search on the reported tasks; (3) **inspectable MCP/CLI** packaging of the same loop.

---

## 1. Introduction

### 1.1 Motivation

A natural question in language-model research is whether models can internalize formal execution semantics—for example, simulating a simple machine step by step. Early work in this repo explored training models to predict multi-step execution traces directly. That direction did not yield a reliable pure-neural multi-step execution story: the model could sometimes perform single steps but not reliably chain them, and additional training alone did not fix the gap. The stronger outcome was a different design: keep execution semantics in an explicit verifier, use the model to propose and rank candidates, and run a bounded search. That design sits in the space of test-time scaling, verifier-guided search, and compute-efficient reasoning.

### 1.2 Contributions and Scope

**What we contribute:**

- A formal VM benchmark and toolkit: toy ISA (AI-Assembly v0), reference VM, synthetic dataset pipeline, and evaluation harness.
- A search-assisted reasoning pipeline: deterministic transition verification, bounded candidate generation, budgeted search, and heuristic or learned branch ranking.
- Evidence that search order and verification dominate raw next-step prediction quality on our benchmark: no ranker ≈35% solve rate vs. heuristic/learned ≈97% on the same task under a fixed node budget; learned ranker reaches ≈99% on a harder split.
- An inspectable runtime: users see candidates, verified outcomes, budget usage, and failure explanations via MCP tools and CLI.

**What we do not claim:**

- We do not claim that internal symbolic execution or a “latent VM” emerged in the model weights.
- We do not claim that the learned ranker generalizes beyond the benchmarks and splits we used.
- We do not claim that gains persist when shortcut structure (e.g., candidate source) is removed; current evidence shows that removing it hurts harder-split quality.

**Scope:** All experiments and claims are limited to the current benchmark (two_step_windows and related slices), the AI-Assembly ISA v0 MVP dataset, and the described pipeline. Results may not generalize to other ISAs, longer horizons, or different program distributions.

### 1.3 Problem statement (explicit)

**Problem:** Can we measure whether language models follow machine-like semantics on controlled tasks, and does **explicit verification plus budgeted search and ranking** improve over **unranked** candidate search under the same node budget?

**What this document adds beyond a bare repo:** A stated benchmark (ISA + VM + splits), reported numbers, limitations, and a reproducible surface (code + MCP). Nothing in this subsection supersedes §1.2 or later sections; it only states the problem in one place.

### 1.4 Central Thesis and Paper Shape

The strongest academically defensible version of the project is:

> We built a formal execution benchmark and found that pure-neural multi-step execution is weak on this setup, while explicit verifier-guided search with ranking is strong and inspectable under bounded compute.

It **is**:

- a negative-result lesson about pure-neural multi-step execution on this setup,
- a positive result for verifier-guided budgeted search,
- a systems/runtime contribution around inspectable reasoning behavior.

---

## 2. Background and Positioning

The system combines:

- **Test-time scaling:** spending compute at solve time (search, ranking) rather than only at training time.
- **Verifier-guided search:** generating candidate next steps and using a formal verifier to accept or reject them instead of trusting a single model output.
- **Budgeted search:** exploring a bounded number of nodes and ranking candidates to use the budget efficiently.
- **Speculative / compute-efficient reasoning:** the same ideas appear in work on verification granularity and speculative decoding.

We do not assume that end-to-end training recovers explicit search structure; we keep the search loop explicit and treat the model as an assist (e.g., ranker) within it.

### 2.1 Product Surface and Public Contract

The repository’s current public direction is **not** “latent internal executor research as a product.” The intended public contract is a **formal VM benchmark and synthetic evaluation toolkit** with:

- **Primary surface:** MCP server for agents and IDE workflows.
- **Secondary surface:** CLI for direct benchmark generation, evaluation, comparison, and gating.
- **Stable v1 scope:** MCP server, CLI, smoke validation, Docker image, and release-facing documentation.
- **Experimental scope:** bounded training/eval helpers and advanced report flows that are useful but not required for the default product path.
- **Internal-only scope:** negative-result `next2_*` research branches and deep exploratory reports that are not part of the public contract.

This matters for claim discipline. The paper reports benchmark and runtime behavior of the **stable core** plus some supporting experimental tooling. It does **not** present every research branch as a product feature, and it does **not** treat fine-tuning as the primary user path.

### 2.2 Runtime UX: What the User Actually Gets

The strongest surviving story is not only benchmark improvement but a different **user-visible runtime**. A user can:

- run a benchmark on a model and get compact machine-readable summaries;
- inspect the **next-step choice** with ranked candidates and verifier outcomes;
- solve one record under a fixed **node budget** and see the path and explored-node count;
- ask for a structured **failure explanation** rather than a hidden miss;
- compare `none` / `heuristic` / `learned` on the same record;
- gate a run and compare two runs stage-by-stage.

This makes reasoning **inspectable**. The user can see which options existed, which one won, what was checked, how much budget was spent, and why the system failed when it failed. In that sense, vmbench is not just a benchmark file and a table of metrics; it is also a small runtime where search, verification, and budget are visible control surfaces.

### 2.3 Related-Work Positioning

vmbench sits closest to work on **verifier-guided search**, **test-time scaling**, and **formal or execution-style evaluation** for language models. It is less aligned with broad “reasoning” papers that do not involve explicit candidate verification, bounded search, or formal step semantics. The relevant comparison class is therefore work where correctness is checked against an external specification or verifier, rather than papers about reasoning style, reward shaping, or generic chain-of-thought alone.

### 2.4 Academic Positioning

This document should be read as a **serious technical report / whitepaper** and a plausible **benchmark + runtime paper seed**, not yet as a fully defended top-tier conference paper. The current evidence supports a strong internal thesis and a credible artifact, but venue-level academic claims would require broader empirical defense.

What is already strong enough to stand on its own:

- a clear formal setup (ISA, VM, deterministic verifier, family-level splits);
- a real negative lesson (pure-neural multi-step execution is weak here);
- a real positive lesson (verifier-guided ranking/search is strong here);
- an inspectable runtime that exposes candidates, budget, and failure modes.

What would still be needed for a stronger main-track style paper:

- multi-seed reporting and variability;
- broader budget sweeps and harder-split transfer;
- failure analysis with explicit buckets;
- verifier-granularity ablations;
- cleaner shortcut-dependence ablations;
- possibly a second task family, ISA extension, or longer-horizon benchmark.

So the paper’s strongest current identity is **benchmark + negative-result lesson + inspectable runtime**, not a grand claim about general neural execution.

---

## 3. Method

### 3.1 Execution Model: AI-Assembly ISA v0

The benchmark is built on a small, deterministic ISA with:

- 8 general-purpose registers (signed 16-bit), three flags (Z, N, C), flat memory (256 cells), and indexed input/output arrays.
- Instructions: CONST, MOV, LOAD, STORE, READ, WRITE, ADD, SUB, CMP, TEST, conditional jumps (JZ, JNZ, JL, JLE, JG, JGE), JMP, HALT.
- No stack, no calls, no dynamic addressing, no bitwise ops in the MVP. The spec defines canonical state and trace formats so that every step is fully traceable.

The reference VM parses programs, executes them step-by-step, and produces canonical traces. This gives a ground-truth verifier for “is this next step correct?”

### 3.2 Dataset

- **Source:** Synthetic programs generated from the reference VM; no human subjects or crowdsourced data.
- **Stages:** single_step, next_2_steps, short_trace, terminal_state.
- **Splits:** Program-family-level; all examples sharing a `split_family` go to exactly one of train / val / test to prevent leakage.
- **Generation:** Seed 7, max 64 steps per program; see dataset card for exact counts and file layout.

### 3.3 Search-Assisted Pipeline

1. **Candidate generation:** Produce a small set of candidate next steps (e.g., next instruction or next state).
2. **Ranking:** Order candidates by a heuristic (e.g., simple rules) or a learned ranker trained on search traces.
3. **Verification:** For each candidate in order, run the reference VM (or transition checker) to accept or reject.
4. **Budget:** Stop after a fixed number of explored nodes (node budget); return the first verified solution or failure.

The user-facing runtime exposes: choose next step, solve with budget, explain failure, compare policies (no ranker vs. heuristic vs. learned).

### 3.4 Training (Learned Ranker)

The learned ranker is trained on trace-derived supervision from the benchmark. Training uses LoRA (no full fine-tune), a conservative profile (small sequence length, small batch size, gradient accumulation), and the same dataset/splits as the benchmark. Base model path and config are documented in the training README and freeze manifest. Training and eval results depend on dataset, splits, and compute; these are documented in the repo.

Training is a **supporting surface**, not the core claim. The repo freezes one conservative local LoRA path and one before/after eval scaffold so that learned ranking can be tested under bounded conditions. The paper’s central result is still about the benchmark, verifier, search loop, and ranking behavior under a fixed budget.

---

## 4. Experimental Setup

### 4.1 Data Splits and Benchmarks

- **two_step_windows (held-out test):** Used for main reported numbers. Split is at program family; no overlap between train and test families. Full window enumeration from `TASK_LIBRARY` is larger; the **run that produced §5.1** used a **test cohort of N = 31** instances (see §5.5).
- **Harder split:** Filtered slice (`window_start ≥ 2`, categories branch/loop/search via `tools/build_harder_search_split.py`). The slice size for the reported run is **N_harder = 224**.
- **Held-out trace-ranking eval:** Exported branch-ranking trace rows, test split only, with **N_trace = 66** gold rows for the reported top-k numbers in §5.3.

### 4.2 Policies

- **No ranker:** Candidates tried in a default order (e.g., arbitrary or by generation order); often wastes budget.
- **Heuristic ranker:** Hand-crafted rules to order candidates.
- **Learned ranker:** Model trained on search traces to score/order candidates.

### 4.3 Metrics (operational definitions)

- **Solve rate** = (number of instances with a verified solution within the node budget) / **N**, where **N** is the number of instances in the evaluated cohort.
- **Average nodes** = (Σ nodes_explored over the cohort) / **N** (arithmetic mean).
- **Budget-exhaustion rate** = (instances that exhaust budget without solution) / **N**.
- **Top-k trace ranking (learned ranker):** on held-out trace rows, fraction where the gold candidate is in the model’s top-k ranked list; **N_trace** = row count of that eval set.

### 4.4 Reproducibility Notes

- **Environment:** Python, dependencies in repo (e.g. `requirements-phase2.txt` for training). CLI and MCP run on host or in Docker; see README for host/port and path contracts.
- **Commands:** Dataset generation, eval, gate, compare, and SFT export are documented in README with example commands.
- **Variability:** Reported numbers below are from the stated benchmarks and splits. Full reporting would include variability across runs (e.g., over seeds or train/test splits) and the method used (e.g., standard deviation over runs). Where single-run numbers are given, we state that.
- **Determinism (additive note):** The reference VM, transition verifier, and search loop over fixed candidates are **deterministic** given fixed inputs. **Learned ranker** training and inference may still vary with GPU, batch size, and floating-point settings; for strict comparability, fix hardware and seeds and document the training stack. Heuristic ranker and pure VM paths are reproducible from code alone.
- **Compute:** Experiments were run on CPU/GPU as available; exact type and wall-clock time per run should be documented for full reproducibility. The repo does not require special hardware for the main CLI/MCP flows; training uses a single GPU for the LoRA runs described in the training README.

---

## 5. Results

All results are on the stated benchmarks and splits; they are not averaged over multiple runs unless stated. Full reporting would give error bars or confidence intervals and compute per experiment.

### 5.1 Held-Out Two-Step Test (Node Budget 12)

**Cohort:** **N = 31** instances. **Calculability:** no-ranker **11** successes → 11/31 = **0.3548387… → 0.3548**; heuristic **30** → 30/31 = **0.9677419… → 0.9677**; learned **30** → same. Because `nodes_explored` is integer-valued per instance, the reported two-decimal means uniquely imply integer cohort totals: **296/31 = 9.548387… → 9.55**, **88/31 = 2.838709… → 2.84**, **74/31 = 2.387096… → 2.39**.

| Policy     | Solve rate | Avg nodes | Successes (of 31) |
|-----------|------------|-----------|-------------------|
| No ranker | 0.3548     | 9.55      | 11                |
| Heuristic | 0.9677     | 2.84      | 30                |
| Learned   | 0.9677     | 2.39      | 30                |

Learned ranker matches heuristic solve rate and uses fewer nodes on average.

### 5.2 Harder Split

**Cohort:** **N_harder = 224** instances.

| Policy     | Solve rate | Successes (of 224) |
|-----------|------------|--------------------|
| Heuristic | 0.9420     | 211                |
| Learned   | 0.9911     | 222                |

**Calculability:** heuristic **211/224 = 0.941964… → 0.9420**; learned **222/224 = 0.991071… → 0.9911**.

Learned ranker slightly outperforms heuristic on this slice.

### 5.3 Learned Trace Ranking (Held-Out Trace Rows)

- **Cohort:** **N_trace = 66** gold rows.  
- Top-1: **64/66 = 0.969696… → 0.9697**  
- Top-3: **65/66 = 0.984848… → 0.9848**  

**Calculability:** top-k = (gold-in-top-k count) / **N_trace** with the exact counts above.

### 5.4 Interpretation

- Search order and verification dominate: moving from no ranker to heuristic or learned gives a large gain; the model’s role is primarily to rank, not to replace the verifier.
- Under tight budgets, learned ranking can match or slightly beat the heuristic and can reduce nodes-to-solution.
- We do not claim a “universal” win: gains are on the current benchmark; removal of shortcut features (e.g., candidate source) reduces harder-split quality in current experiments.

### 5.5 Document self-audit: consistency, completeness, calculability

| Criterion | Status in this document |
|-----------|---------------------|
| **Consistency** | Abstract’s four contributions match §1.2 bullets. Foundations “external verifier + search” matches §3.3–§5. Naive-Bayes masses sum to **1.00** (topics and intents). Research mean **(7+8+6+7+7)/5 = 7.0**. §5.1 rates match **11/31** and **30/31**; integer node totals **296**, **88**, **74** imply the printed means. §5.2 rates match **211/224** and **222/224**. §5.3 matches **64/66** and **65/66**. |
| **Completeness** | Metrics defined in §4.3. §5.1 gives **N**, success counts, and node totals. §5.2 gives **N_harder** and successes. §5.3 gives **N_trace** and exact top-k counts. Limits in §6. Reproducibility §8 + Appendix A. |
| **Calculability** | Every printed result in §5.1–§5.3 is now derivable from explicit integers inside the document itself: **31, 224, 66** plus the stated success / top-k / node totals. |

---

## 6. Limitations

We encourage explicit statement of limitations. The following apply to this work.

### 6.1 Scope of Claims

- Results are on a small synthetic benchmark (AI-Assembly ISA v0 MVP, two_step_windows and related slices). We have not tested on other ISAs, longer horizons, or broad program distributions.
- The multi-step pure-training direction (`next_2_steps` without search assist) is a negative result in this repo; we do not claim internal multi-step execution.

### 6.2 Assumptions and Robustness

- **Verifier correctness:** All conclusions assume the reference VM and transition verifier are correct. A bug in the verifier could invalidate search-quality claims; we have found and fixed at least one critical verifier bug in the past.
- **Benchmark structure:** The benchmark is synthetic and program-derived. Easier slices (e.g., all records from IP=0) can overstate robustness; we use held-out families and a harder split to mitigate this, but the benchmark is still limited.
- **Shortcut structure:** The learned ranker benefits from features such as candidate source. Removing candidate source significantly hurts harder-split quality; structural replacements have not yet restored it. So current gains are partly dependent on shortcut structure.

### 6.3 Factors That Influence Performance

- **Node budget:** Solve rate and nodes-to-solution depend strongly on budget; smaller budgets amplify the benefit of good ranking.
- **Split and program family:** Performance varies by split (held-out vs. harder) and by program family.
- **Ranker type:** No ranker << heuristic ≈ learned on our benchmark; learned can slightly beat heuristic on the harder split and on node efficiency.
- **Verification granularity:** How often we verify (e.g., every step vs. final state only) affects cost and success; we have not fully ablated this in the whitepaper.

### 6.4 Dataset Limitations

- MVP ISA only (no stack, calls, dynamic addressing, bitwise ops). Results do not generalize to full-feature ISAs.
- Synthetic programs are small and structured; loop diversity is limited to bounded counter-style programs. Performance on larger or different program distributions is unknown.
- Short-trace evaluation is intended for short traces only.

### 6.5 Risk Register

The following risks are important enough to state explicitly because they change how results should be interpreted.

- **Verifier correctness risk:** we already found at least one critical verifier bug in the project history; any remaining verifier bug could fake search quality and invalidate downstream conclusions.
- **Benchmark easiness risk:** early slices were easier than they looked; even the current `two_step_windows` benchmark is synthetic and program-derived, so good numbers can still overstate generality.
- **Heuristic imitation risk:** the learned ranker may partly compress heuristic behavior rather than discover a materially different search policy. That is still engineering value, but it is weaker evidence of deeper algorithmic generalization.
- **Program / source shortcut risk:** `candidate source` and related structural features are currently strong supports for harder-split quality; this weakens claims about purely internal reasoning.
- **Search-cost risk:** the system works because explicit search is doing real work. If search cost scales poorly outside current slices, the branch may remain scientifically interesting but operationally expensive.
- **Hype-drift risk:** benchmark wins, runtime wins, and broad reasoning claims are not the same thing. The paper should keep those levels separate.

---

## 7. Open Empirical Questions and Exit Criteria

The repo’s planning documents imply a sharper next-step agenda than a generic “future work” paragraph. The main open questions are:

1. **Verification granularity:** should the system verify every step, only final states, state diffs, or some finer-grained partial mode under a fixed compute budget?
2. **Failure taxonomy:** when the system fails, is it because of verifier false accept / false reject, candidate omission, ranker miss, or budget exhaustion?
3. **Budget curves:** do policy differences stay strong across budgets `2`, `4`, `8`, `12`, `16`, or is the current story overly tied to one budget?
4. **Harder benchmark transfer:** does the current search/ranker story survive harder start states, held-out families, longer horizons, and lower-leakage settings?
5. **Shortcut dependence:** can `candidate source` and related shortcuts be replaced by lower-leakage structural features without losing harder-split quality?

These questions are not cosmetic. They define whether this branch matures into a robust verifier-guided runtime story or remains a local success on favorable benchmark structure.

### 7.1 Immediate Experiment Set

- run verifier-granularity comparisons at budgets `4`, `8`, `12`;
- build held-out and harder-split failure analyses with explicit failure buckets;
- produce budget curves for `none`, `heuristic`, and `learned`;
- rerun on harder splits and longer horizons without changing the claim language unless results survive.

### 7.2 Exit Criteria

The branch remains healthy if:

- learned or heuristic ranking continues to beat no-ranker on harder cases;
- gains remain meaningful after reducing shortcut structure;
- the runtime UX still reflects genuine policy differences rather than a thin wrapper around fixed heuristics;
- search cost stays operationally acceptable for the intended product surface.

The branch should be de-escalated or re-scoped if:

- performance collapses once shortcut structure is reduced;
- learned ranking stops adding transfer beyond heuristic;
- verifier or benchmark-quality issues undermine confidence in the search results;
- the runtime turns out to be product-thin once the strongest claims are removed.

---

## 8. Reproducibility

### 8.1 Code and Data

- **Code:** The repository contains the reference VM, dataset pipeline, baseline trainer, curriculum gate, SFT export, search/ranking pipeline, MCP server, and CLI. All are available in the repo.
- **Data:** The MVP dataset (train/val/test splits by program family) is generated by the dataset pipeline; generation config, seed, and split strategy are in the dataset card. Regenerating with the same seed and config yields the same splits.
- **Instructions:** README gives quickstart commands for generate, eval, gate, compare, export-sft, and status. Training README and configs describe how to run LoRA training and checkpoint eval.

### 8.2 What Can Be Reproduced

- Main experimental results: running the same pipeline (no ranker, heuristic, learned) on the same dataset and splits with the same node budget should yield the reported solve rates and node counts, up to environment and random seed. For learned ranker, training from the same config and data reproduces the reported ranker behavior.
- A subset of experiments may depend on internal tooling or paths; where so, we state it. We aim to keep the main benchmark, eval, and MCP/CLI flows reproducible from the repo.

---

## 9. Broader Impact

This work is a benchmark and toolkit for measuring execution fidelity and inspectable reasoning. It is not tied to a specific deployment or application. Potential positive impact: better tools to evaluate and improve transparent, verifier-backed reasoning. We do not see a direct path to high-risk misuse from the benchmark or runtime alone; the main risk would be from downstream use of improved models in applications we do not control. We have not performed a formal broader-impact analysis beyond this.

---

## 10. Ethics and Compliance

Research was conducted in line with responsible ML practice: we state limitations, avoid overclaiming, and provide reproducibility. No human subjects or crowdsourced data were used in the dataset; the data is synthetic. We have read and aligned with standard codes of ethics for research; no deviations.

---

## 11. Licenses and Assets

### 11.1 Existing Assets

- We use open-source base models (e.g., Phi-3-mini) and tooling under their respective licenses. Users must comply with model and code licenses when reproducing or extending the work.
- Any third-party code or data used in the repo is cited and used in accordance with its license; see repo and dataset card.

### 11.2 Released Assets

- **Code:** Repository code is released under the license specified at the repository root (if none is specified, assume the same terms as the project).
- **Dataset:** The AI-Assembly ISA v0 MVP dataset is synthetic, generated by the repo’s dataset pipeline. License: see repository root. If you use this dataset, cite the vmbench repository and the description of the AI-Assembly ISA and generation procedure (ISA_SPEC, dataset card).
- **Documentation:** Dataset card, ISA spec, README, and this whitepaper document the dataset and method (training, license, limitations).

---

## 12. Declaration on LLM Usage

LLMs were not used as an important, original, or non-standard component of the core methods (VM, verifier, search, ranking pipeline). If LLMs were used only for writing, editing, or formatting of documentation, that does not affect the scientific claims or reproducibility of the results.

---

## References and Repository

- **Origin and file layout** are fully inlined in **Foundations** above.
- **Related concepts:** test-time scaling, verifier-guided search, verification granularity, speculative search; internal plans in text.md, VMBENCH_*_PLAN.md.

---

## Appendix A. Proof by MCP

The following claims can be verified by connecting to the vmbench MCP server (e.g. via Cursor or any MCP client) and calling the tools. All calls below were run against the repo with the MCP server in Docker (`vmbench-mcp:local`) and workspace mounted.

### A.1 Formal VM benchmark and inspectable runtime (§1.2, §3)

- **vmbench_manifest** returns product metadata and core module paths: `name: vmbench`, `product_direction: formal VM benchmark + synthetic dataset + eval toolkit`, `primary_surface: mcp`, core modules including `reference_vm.py`, `dataset_pipeline.py`, `baseline_trainer.py`, `curriculum_gate.py`, `sft_export.py`.
- **vmbench_status** returns the same manifest, repo map (REPO_PRODUCT_MAP.md, README, etc.), and the list of 15 tools (vmbench_manifest, vmbench_generate_dataset, vmbench_run_baseline, vmbench_gate_summary, vmbench_export_sft, vmbench_compare_reports, vmbench_choose_next_step, vmbench_solve_with_budget, vmbench_explain_failure, vmbench_compare_policies, vmbench_demo_reasoning_runtime, vmbench_generate_demo_payload, vmbench_status, etc.).

### A.2 Synthetic dataset and program-family splits (§3.2)

- **vmbench_generate_dataset** with `output_dir`, limits, and `seed: 7` succeeds and writes `manifest.json` and dataset types `single_step`, `next_2_steps`, `short_trace`, `terminal_state`. This demonstrates reproducible synthetic generation from the reference VM.
- **vmbench_export_sft** with that dataset root and `stages: ["single_step"]` succeeds and writes an SFT manifest and exports. This demonstrates that the pipeline produces training-ready data from the same splits.

### A.3 Search-assisted pipeline: candidates, verification, ranking (§3.3)

- **vmbench_choose_next_step** on `datasets/mvp/next_2_steps/test.jsonl`, `record_index: 0`, `ranker: heuristic` returns ranked candidates (instruction text, source, verified flag) and a verified winner. Example: winner `READ R1, input[0]` (verified), then next step `CONST R2, 0` (verified); other candidates are marked verified: false with notes such as `instruction_mismatch`, `target_mismatch`, `state_diff_mismatch`. This shows deterministic transition verification and heuristic ranking in action.
- **vmbench_solve_with_budget** on the same record with `ranker: heuristic`, `node_budget: 12` returns `solved: true`, `successful_path: ["READ R1, input[0]", "CONST R2, 0"]`, `nodes_explored: 2`, `budget_exhausted: false`. This shows budgeted search with verification and ranking producing a correct two-step path within the budget.

### A.4 Policy comparison: no ranker vs heuristic (§4.2, §5.1)

- **vmbench_compare_policies** on the same record, `node_budget: 12` returns side-by-side results for policy `none` and policy `heuristic`. Both solved this record; heuristic used the correct path (READ then CONST); no-ranker in this run also succeeded but with different attempt ordering. On harder records, no-ranker often wastes budget or fails while heuristic succeeds—matching §5.1 aggregate (**11/31** vs **30/31** on the held-out cohort).

### A.5 Inspectable failure and repo map (§1.2, §8)

- **vmbench_explain_failure** on a record with `ranker: none`, `node_budget: 4` returns a structured result (e.g. `category: solved` or failure category with `attempts` and `notes`), demonstrating that the runtime exposes why a run succeeded or failed (instruction_mismatch, etc.).

### A.6 What MCP Does Not Prove

Appendix A shows what the runtime can expose through tool calls and structured outputs. It does **not** prove the model’s internal reasoning in a mechanistic sense. MCP logs show:

- which tool was called,
- with which arguments,
- what the tool returned.

They do **not** show:

- the user message that led to the tool call,
- the model’s chain-of-thought or internal plan,
- all rejected alternatives the model considered,
- complete retry history outside the visible window,
- a full mechanistic trace of how the LLM chose an action.

So Appendix A is best read as **proof of runtime observability and interface behavior**, not proof of hidden internal execution inside the model.

To reproduce: start the vmbench MCP server (e.g. `docker run --rm -i -v <repo>:/workspace -w /workspace vmbench-mcp:local`), then call the tools above with the same arguments. Dataset paths are repo-relative; use `datasets/mvp/next_2_steps/test.jsonl` for the next_2_steps benchmark file.

---

*This whitepaper summarizes the vmbench benchmark and inspectable reasoning runtime. Foundations inlines origin and repository map. §5.5 states calculability of reported numbers; §6–§7 state the main risks and next empirical gates. Claims are limited to §1.2, §5, and §6.*
