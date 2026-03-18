# vmbench Presentation Q&A

## Framing

This Q&A is for the presentation about vmbench as an inspectable reasoning runtime.

Important framing:

- do not claim latent symbolic execution unless directly supported
- do not oversell the learned ranker as general reasoning
- keep separating benchmark wins from product wins

## 1. Did the original latent-VM hypothesis work?

Not in the strong sense.

We saw some real signal in `single_step`, but that did not turn into a convincing pure-neural multi-step execution story.

The stronger result came from explicit search, verification, and ranking.

## 2. So is this basically just a heuristic system?

Not exactly.

The system has hand-built components, but there is still a real learned layer:

- learned branch ranking
- trace-derived supervision
- measurable improvement under fixed budget on harder splits

The honest claim is not “end-to-end reasoning emerged”.
The honest claim is “learning improved a search-assisted reasoning controller”.

## 3. Why is the UX actually different from a normal LLM chat?

Because the user can see:

- which candidates existed
- which one won
- what was verified
- how much budget was spent
- why the system failed when it failed

That turns reasoning from a hidden act into an inspectable process.

## 4. Why does budget matter so much here?

Because budget is where the policies separate.

If compute is effectively infinite, many systems can look decent.
Under tight node budgets, ordering quality becomes visible.

That is why `no ranker` looks much worse than `heuristic` or `learned`.

## 5. Is the learned ranker actually better than the heuristic?

On some held-out and harder slices we tested, yes, slightly.

The correct statement is:

- learned is clearly better than no-ranker
- learned is competitive with heuristic
- learned sometimes beats heuristic on harder budgeted search

It is not yet a giant universal win.

## 6. What is the biggest remaining weakness?

Shortcut structure.

More specifically:

- `candidate source` is still a strong dependency
- dropping it hurts harder-split search noticeably
- improvements in trace ranking do not automatically mean improvements in real search behavior

So claims about generalization beyond the current benchmarks and splits should be cautious. That is the main scientific caution flag right now.

## 7. So why keep going if shortcuts still matter?

Because the runtime is already useful even before the strongest scientific claim is settled.

It already supports:

- verifier-backed decisions
- budget-aware search
- inspectable failure modes
- side-by-side policy comparison

That is enough to be product-interesting.

## 8. Is this still an MCP project or did it turn into a research-only artifact?

It is definitely still an MCP project.

The product surface already includes tools like:

- `vmbench_choose_next_step`
- `vmbench_solve_with_budget`
- `vmbench_explain_failure`
- `vmbench_compare_policies`
- `vmbench_demo_reasoning_runtime`

So the system is not just evaluated offline.
It is already exposed as a runtime interface.

The careful version is:

- MCP proves runtime behavior and tool use
- it does not prove hidden internal execution or expose chain-of-thought

## 9. What would make this a no-go?

Three things would be strong no-go signals:

1. performance collapses once shortcut structure is reduced
2. learned ranking stops adding transfer over heuristic
3. the runtime UX turns out to be only a thin wrapper around fixed heuristics with no genuine adaptive value

## 10. What is the cleanest one-line takeaway?

The project did not prove that a latent VM emerged in the weights.

It did produce something productively new:

an inspectable reasoning runtime where search, verification, budget, and failure are visible to the user.

## 11. What is the public product surface right now?

Primary:

- MCP runtime

Secondary:

- CLI benchmark/eval interface

Supporting / experimental:

- bounded training and checkpoint-eval helpers

Not part of the stable public contract:

- negative-result `next2_*` research branches
