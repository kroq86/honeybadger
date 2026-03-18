# vmbench | A System That Shows Its Work

## Core framing

This presentation should be honest and simple from the first minute: state what we contribute and what we do not claim.

We did **not** get a clean story that the model magically learned everything by itself.

What we got is more grounded and more useful:

- deterministic transition verification
- bounded candidate generation
- budgeted search
- heuristic and learned ranking
- a product surface that exposes reasoning as a visible process

Public shape:

- MCP-first runtime
- CLI-second benchmark/eval surface
- training helpers as supporting / experimental, not the main user path

In current academic language, this sits near:

- `test-time scaling`
- `verifier-guided search`
- `compute-efficient reasoning`
- `speculative search`

So the talk is not “look, the model became smart in a mysterious way”.

The talk is:

1. where the original hypothesis stalled
2. what actually started helping
3. why that yields a new user experience
4. what still looks fragile

## Slide flow

### 1. Title

Position vmbench as a system that shows its work, not just a benchmark.

The audience should understand immediately:

- this is not another generic AI deck
- part of the first idea failed
- something useful still survived

### 2. What changed

Be explicit:

- the model could do some simple steps but not reliable multi-step work
- pure training did not become the breakthrough
- explicit search plus checking is where the real lift came from

This keeps the story credible.

### 3. Metrics

Say the numbers clearly, but explain them in plain language. When reporting, state how they were obtained (which splits, how many runs) and how much variability to expect; state compute used.

- held-out 2-step test
  - `no ranker`: `0.3548`
  - `heuristic`: `0.9677`
  - `learned`: `0.9677`
- harder split
  - `heuristic`: `0.9420`
  - `learned`: `0.9911`

The key message:

- the order of trying options matters a lot
- learned ranking matters most when time and budget are tight
- extra search-time compute can change outcomes a lot, which is exactly why `test-time scaling` is such an active topic

When presenting these numbers, keep the claim narrow:

- benchmark + runtime win
- not proof of latent execution in the weights

### 4. UX

This is the product turn, and it should sound human.

Do not describe the system as magic.
Describe it as:

- something you can control
- something you can watch
- something you can trust a bit more
- something that works under limits

Use the concrete UX primitives:

- choose next step
- solve with budget
- explain failure
- compare policies

### 5. Interactive runtime demo

This slide should feel operational and easy to follow.

Pick one scenario, move the budget slider, and narrate:

- no-ranker spends budget badly
- heuristic improves the order
- learned often solves with fewer explored nodes

Then point to the verified winner.

The sentence to land:

“this is what the user sees now, not just a hidden answer”

### 6. How it works

Explain it in plain language:

- make a few options
- sort them
- check them
- keep trying until budget runs out
- show the result to the user

Then give the academic names:

- candidate generation
- branch ranking
- verification granularity
- compute-bounded search

Important point:
the same parts that improved the benchmark are the parts the user now feels directly.

### 7. What is still not good enough

Do not hide the shortcut problem.

Say clearly:

- some of the quality still depends on shortcuts
- the strongest shortcut is where an option came from
- removing that shortcut hurts harder tasks

Use the latest negative result:

- structural replacement improves held-out trace accuracy
- but still fails to restore harder-split quality

Scope: these limitations apply to the benchmarks and setups we used; results may not generalize beyond them. This is the honesty slide.

It is also the place to mention a research warning:

- recent work on verifier-guided search shows that systems can look strong on easier slices and still depend on shortcuts
- MCP logs and runtime traces show visible behavior, not internal chain-of-thought

### 8. Why still build it

Why continue?

Because even with that weakness, the runtime already offers:

- visible option choice
- checked steps
- partial progress under limits
- failure explanation instead of mystery

That is a meaningful UX even before any grand claim about reasoning.

### 9. Next gate

Make the next step disciplined:

- keep going only if the system still works on harder cases
- stop if the gains vanish once shortcuts are reduced

In research terms, this is the difference between:

- a real gain in generalization under search-time compute
- and a local gain that only survives on easy benchmark structure

This protects the project from hype drift.

## Talk close

Strong closing sentence:

The strongest surviving story is not “the model became a machine”.

The strongest surviving story is:

**we built a system that shows its work.**
