# How loss attachment and carry shape update flow through a shared-weight slot loop

Date: 2026-09-10. Hardware: the arch server's RTX 3070 (`ssh wolfe@3070`), torch 2.14.0+cu130
in `/home/wolfe/venvs/toyloop`. Predictions were frozen in [PLAN.md](PLAN.md) before the grid
ran; the method amendments the gate forced are appended there with their reasons and dates.

## Why this study exists

Three measurements on the real MORPH slot loop, all from 2026-09-10, box the problem in.

1. The per-pass cotangent at the loop state is FLAT under the prelude entry — share 0.168 /
   0.168 / 0.166 / 0.160 / 0.154 / 0.183 over six passes
   ([slot-gradient-probe](../experiments/results/2026-09-10-slot-gradient-probe/README.md)).
   Gradient reaches every pass.
2. The coda's CE with the loop's exit equals its CE with the loop's ENTRY to within 0.0002
   to 0.0033 nats, while a gradient-fitted z is worth 0.9 to 2.6 nats
   ([slot-z-optimize](../experiments/results/2026-09-10-slot-z-optimize/README.md)). The
   reader works. The writer produces nothing the reader wants.
3. The six passes' updates to the shared core weights largely cancel:
   `|sum_t dW_t| / sum_t |dW_t|` = 0.520, against 0.408 for six orthogonal equal-norm
   updates and 1.0 for six aligned ones.

On web text those three cannot be separated from a fourth possibility: that the M-next
target does not need iteration, so no loop has anything to earn however it is wired. This
toy fixes the iteration requirement by construction and then varies the wiring.

## The model

`model.py`. One row is 8 spans; one span is `[3 token cells][1 slot cell][2 prefix cells]`,
so 48 cells and 24 token positions.

| part | value |
|---|---|
| d_model / heads / d_ff | 96 / 4 / 256 |
| prelude | 2 pre-norm blocks over all cells |
| core | **1** shared pre-norm block, applied T times to the 8 slot cells only |
| coda | 2 pre-norm blocks; token cells read RAW embeddings (`coda_token_input: embed`), token-to-token attention is restricted to the own span, slot cells are never keys |
| depth | T per slot ~ clamp(Poisson(6), 1, 8); a slot freezes at its depth |
| entry | `h_0 = W_in(e)`, `e` the prelude state at the slot cell; injection `h += 0.45^t · W_inj(e)` on the first 24 channels every pass |
| write | `z -> W_prefix -> the span's 2 prefix cells` |
| heads | token head and MUX head both tied to the input embedding |
| optimiser | AdamW, lr 2e-3, betas (0.9, 0.95), wd 0.01, 200-step warmup then cosine to 0.1x, clip 1.0 |
| budget | 4,000 steps, batch 128, 5 seeds per cell, 596,928 parameters, of which 110,784 are the shared core block |

**The geometry switch is the study's first result, so it is part of the model description.**

`geometry: permissive` is MORPH's own slot-loop geometry: the prelude is causal over every
cell, the core attends every earlier slot, the coda reads every earlier prefix cell.

`geometry: strict` closes the three routes that let something other than the loop compose
across spans: the prelude attends within a span only, the core attends the previous slot and
itself only, a token reads only the previous span's prefix cells, and prefix cells do not
chain to each other. Under `strict` the loop is the only cross-span mechanism and depth has
an exact price: the answer at the head of span j+1 needs j passes. That price is proven by
measurement, not by argument — `selfcheck.py::t_depth_requirement` differentiates the value
logit at the head of span 5 with respect to span 0's operator cell and reads 0.000e+00 at
depths 1, 2 and 3, then 9.685e-03 at depth 4.

## The tasks

`tasks.py`. Input alphabet: the 6 elements of the permutation group S_3. Answer alphabet: 6
output-only ids. **The answer never appears as an input token**; it is the LABEL at the head
of the NEXT span, which is the double-label shape of the real spec's section 5. Every other
token position carries the ordinary next-symbol label, uniform by construction with floor
ln 6 = 1.7918, exactly as most of the real token CE is local statistics the loop cannot help.

- **compose** (needs iteration). A span's operator is its FIRST symbol; the other two are
  distractors. The register is `R_i = R_{i-1} · P_i`. The label at the head of span i+1 is
  `R_i`. S_3 is not commutative, so no bag or average computes `R_i`.
- **summary** (one pass, the control). The label at the head of span i+1 is the most
  frequent symbol of span i. Slot i's own prelude already sees its whole span.

MUX target for slot i: `R_i` or the mode. The staged attachment's own-span target is `P_i`.

Under `strict`, the reachability ceiling at forced depth d — the lowest value CE any model
can reach — is `(7 - min(d+1, 7)) / 7 · ln 6`:

| d | 1 | 2 | 3 | 6 | 8 |
|---|---|---|---|---|---|
| best possible value CE | 1.2798 | 1.0239 | 0.7679 | 0.0000 | 0.0000 |

Those two numbers, **1.28 and 0.00**, are the whole story below. A model at 1.28 is using
one pass. A model at 0.00 has learned the chain.

## What the gate established before the grid ran

`ignored/experiment-artifacts/2026-09-10-toy-slot-loop/ladder3/`, one seed each, trained at
a fixed depth and read on the shared 2,048-row eval draw.

| cell | d=1 | d=2 | d=3 | d=6 | d=8 | d=12 |
|---|---|---|---|---|---|---|
| `strict compose`, trained d=1 | 1.281 | 1.295 | 1.294 | 1.387 | 1.541 | 1.729 |
| `strict compose`, trained d=2 | 1.282 | 1.281 | 1.281 | 1.352 | 1.367 | 1.384 |
| `strict compose`, trained d=3 | 1.420 | 1.281 | 1.281 | 1.284 | 1.288 | 1.323 |
| `strict compose`, trained d=6 | 2.172 | 1.293 | 1.281 | 1.281 | 1.281 | 1.281 |
| `strict compose`, trained d=8 | 12.987 | 10.260 | 7.569 | 0.134 | **0.000** | 0.000 |
| `strict compose`, coda blind to z | 1.797 | 1.797 | 1.797 | 1.797 | 1.797 | 1.797 |
| `strict summary`, trained d=1 | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
| `permissive compose`, trained d=1 | **0.283** | 0.340 | 0.552 | 1.322 | 1.901 | 2.595 |
| `permissive compose`, trained d=6 | 4.149 | 1.126 | 0.364 | 0.056 | 0.084 | 0.479 |

Four things are settled by that table.

**The channel gate holds.** With the coda blind to z the value CE is 1.797 at every depth,
which is chance to three decimals. z is the only route.

**The control is a control.** `summary` is solved at depth 1 to 0.000.

**The task is solvable, and only through depth.** Trained at fixed depth 8 the model reaches
0.000, and the same model reads 12.987 at depth 1. The chain is learnable.

**MORPH's own geometry does not need the loop.** Under `permissive`, one core pass reaches
0.283 nats and 66 % accuracy on an 8-span scan, because the prelude (2 layers, causal over
everything) and the coda (2 layers over a causal chain of prefix cells) can compose across
spans by themselves. This is a mechanism for the real model's flat K-curve that has nothing
to do with loss attachment: **in the shipped geometry the loop is one of at least three
cross-span paths, and it is the slowest to learn.**

## Results

110 cells, 5 seeds each of 22 configurations, 0 failures, 4,000 steps per cell. The per-pass
tap self-check (`sum_t dW_t == leaf.grad` in the same backward) reads a worst relative error
of **1.63e-07** over all 110 runs. Every table is mean [min, max] over the 5 seeds. Raw JSON:
`ignored/experiment-artifacts/2026-09-10-toy-slot-loop/grid/`, regenerate with
`python aggregate.py <dir>`.

Read every value CE against two constants: **1.2798** is the one-pass shortcut and **0.0000**
is the full chain.

### Question 1 — which attachment lets the loop earn depth?

Task `compose`, strict geometry, default entry and terms.

| attachment | escaped | escape step | value CE @6 | value acc @6 | K1-K6 (value CE) | s/run |
|---|---|---|---|---|---|---|
| `staged` | **5/5** | 900 | **0.0000** [0.0000, 0.0000] | 1.000 | +1.2829 [+1.2811, +1.2872] | 538 |
| `exit` | 2/5 | 625 | 0.7693 [0.0000, 1.2832] | 0.642 | +0.5135 [+0.0000, +1.2862] | 198 |
| `deep_coda` | 2/5 | 625 | 0.7693 [0.0000, 1.2833] | 0.643 | +0.5129 [-0.0000, +1.2842] | 523 |
| `mux_all_detach` | 0/5 | - | 0.8438 [0.5391, 1.0322] | 0.584 | +0.4564 [+0.2947, +0.7503] | 235 |
| `mux_all` | 1/5 | 750 | 1.0255 [0.0000, 1.2828] | 0.524 | +0.2570 [+0.0000, +1.2850] | 224 |
| `progressive` | 0/5 | - | 1.2812 [1.2804, 1.2821] | 0.404 | +0.0000 [-0.0000, +0.0000] | 217 |

Value CE against forced depth (the K-curve):

| attachment | d=1 | d=2 | d=3 | d=6 | d=8 | d=12 |
|---|---|---|---|---|---|---|
| `exit` | 1.283 | 1.180 | 1.078 | 0.769 | 0.769 | 0.769 |
| `mux_all` | 1.282 | 1.231 | 1.180 | 1.025 | 1.025 | 1.025 |
| `mux_all_detach` | 1.300 | 1.096 | 0.996 | 0.844 | 0.816 | 0.900 |
| `staged` | 1.283 | 1.026 | 0.770 | **0.000** | 0.000 | 0.882 |
| `deep_coda` | 1.282 | 1.180 | 1.077 | 0.769 | 0.769 | 0.769 |
| `progressive` | 1.281 | 1.281 | 1.281 | 1.281 | 1.281 | 1.282 |

Where each attachment stops along the chain. `Rj` is the answer at the head of span j+1 and
needs j passes, so the column index is the depth the answer demands.

| attachment | R0 | R1 | R2 | R3 | R4 | R5 | R6 |
|---|---|---|---|---|---|---|---|
| `exit` | 0.00 | 0.00 | 1.08 | 1.08 | 1.08 | 1.08 | 1.08 |
| `mux_all` | 0.00 | 0.00 | 1.44 | 1.43 | 1.44 | 1.44 | 1.44 |
| `mux_all_detach` | 0.00 | 0.00 | 0.25 | 1.10 | 1.23 | 1.55 | 1.77 |
| `staged` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `deep_coda` | 0.00 | 0.00 | 1.08 | 1.08 | 1.08 | 1.08 | 1.08 |
| `progressive` | 0.00 | 0.00 | 1.79 | 1.79 | 1.79 | 1.79 | 1.79 |

`staged` — own-span target at every non-final pass, next-span target at the exit — is the
only attachment that solves the chain, and it solves it on every seed. Everything else stops
between R1 and R2, which is where the prelude plus one core pass runs out.

Two negatives worth naming. `deep_coda`, the TRM/HRM shape, reproduces `exit` to four
decimals on the same two seeds at 2.6x the wall clock: running the coda on every pass adds
nothing the exit CE was not already saying. `progressive`, Bansal's random no-grad prefix,
is the only attachment that is strictly WORSE than plain exit-only, at 0/5 and a dead-flat
K-curve. The real tree already measured that arm flat on MORPH
([`2026-09-10-arc-slot-mnext-progressive`](../experiments/failures/2026-09-10-arc-slot-mnext-progressive.md));
this says the flatness is not a scale artefact.

### Question 2 — can a one-pass task ever make a loop earn depth?

Task `summary`, same six attachments, 5 seeds each.

Every one of the 30 cells reads value CE **0.0000**, accuracy **1.000**, escape step **250**,
and K1-K6 **+0.0000** to four decimals. Not one attachment moved a K-curve. The answer is no,
and it is not close.

### Question 3 — entry, carry and terms

All on `compose` at attachment `exit`, one factor at a time.

| cell | escaped | escape step | value CE @6 | K1-K6 (value CE) |
|---|---|---|---|---|
| `exit` (the reference) | 2/5 | 625 | 0.7693 [0.0000, 1.2832] | +0.5135 |
| **fixed-point term, lambda 1.0** | **5/5** | 1400 | **0.0000** [0.0000, 0.0000] | +1.2820 [+1.2809, +1.2835] |
| injection decay 0.9 | 3/5 | 1250 | 0.5133 [0.0000, 1.2832] | +0.7695 |
| injection off | 3/5 | 833 | 0.5133 [0.0000, 1.2833] | +0.7689 |
| per-pass LoRA rank 8 | 2/5 | 1375 | 0.7692 [0.0000, 1.2827] | +0.5130 |
| fixed depth 6 | 2/5 | 625 | 0.7688 [0.0000, 1.2824] | +2.0552 |
| truncated BPTT, last 2 passes | 2/5 | **3125** | 0.8360 [0.0544, 1.4037] | +0.4747 |
| noise entry (Parcae) | **0/5** | - | 1.5308 [1.4982, 1.5454] | +0.0189 |
| coda blind to z | 0/5 | - | 1.7936 [1.7925, 1.7950] | +0.0000 |

- **The terminal fixed-point penalty is the strongest single lever in the study**: 5 of 5
  seeds solve the chain against 2 of 5 without it. It is `lambda · ||z - h_{T-1}||^2 /
  ||z||^2` on the slot's own last pass, and it costs nothing (233 s against 198 s).
- **The noise entry is the strongest single poison**: 0 of 5 seeds, and at 1.531 it does not
  even reach the one-pass ceiling of 1.280. Its entry state has participation rank **95.4**
  in 96 dimensions (it is noise) and its exit state collapses to rank **1.0**.
- **Truncated BPTT is survivable, and it costs speed, not capability.** With gradient on the
  last 2 passes only the per-pass cotangent reads 0.000 / 0.000 / 0.000 / 0.000 / 0.521 /
  0.479 — the window is exactly what it claims — and 2 of 5 seeds still solve the chain, but
  they escape at step 3125 instead of 625. A shared map trained only at passes 5 and 6 is
  still the same map at passes 1 to 4.
- **Per-pass LoRA changes nothing**: 0.7692 against 0.7693, the same 2 of 5 seeds.
- **The channel gate holds**: blind to z the value CE is chance at every depth. Note its MUX
  CE is still 0.897 with K1-K6 +0.449 — the loop keeps learning the chain from the MUX even
  when nothing downstream can read it.
- Injection decay 0.45 (the default) is the worst of the three injection settings, at 2/5
  against 3/5. With 5 seeds that is not a result; it is written down so nobody reads the
  default as tuned.

Write contribution on task `compose`, attachment `exit`: value CE 0.769 with z = exit,
**2.282** with z = the entry state (zero passes) and 4.987 with z = 0. The entry-minus-exit
gap is **+1.513** nats, against the real model's +0.0015. That gap is the number the real
model does not have.

### Question 4 — the geometry, which is the study's first result

`compose` at attachment `exit`, MORPH's own geometry against the strict one.

| geometry | escaped | value CE @6 | K1-K6 | d=1 | d=2 | d=3 | d=6 |
|---|---|---|---|---|---|---|---|
| strict (loop is the only cross-span path) | 2/5 | 0.7693 | +0.5135 | 1.283 | 1.180 | 1.078 | 0.769 |
| **permissive (MORPH's)** | 2/5 | **0.4359** | **+0.1060** | 0.542 | 0.454 | 0.438 | 0.436 |
| permissive, `mux_all` | 1/5 | 0.6284 | +0.0653 | 0.694 | 0.647 | 0.636 | 0.628 |

Under MORPH's geometry the model is BETTER (0.436 against 0.769) and the loop matters LESS
(K1-K6 +0.106 against +0.514). One core pass already reaches 0.542. That is the real model's
signature — a respectable CE with a flat K-curve — reproduced here by geometry alone, with
the attachment held fixed.

### Question 5 — what the per-pass gradient actually says

Per-pass cotangent at the loop state, per-pass share of the shared core weight gradient, and
the agreement between passes. Source is the trained loss, forced depth 6, 4 x 64 rows.
`cancellation` is `|sum_t dW_t| / sum_t |dW_t|`: **0.408** for six orthogonal equal-norm
updates, **1.0** for six aligned ones.

| cell | cotangent share p1..p6 | dW share p1..p6 | cos(dW_t, total) p1..p6 | mean pairwise cos | cancellation |
|---|---|---|---|---|---|
| `exit` | 0.143 0.159 0.161 0.164 0.173 0.201 | 0.141 0.114 0.118 0.136 0.187 0.305 | +0.72 +0.80 +0.92 +0.99 +1.00 +0.99 | +0.851 | 0.959 |
| `mux_all` | 0.312 0.247 0.184 0.131 0.084 0.041 | 0.411 0.209 0.160 0.112 0.073 0.036 | +0.98 +0.99 +0.99 +0.99 +0.99 +0.94 | +0.966 | 0.987 |
| `mux_all_detach` | 0.306 0.147 0.136 0.111 0.103 0.198 | 0.505 0.127 0.120 0.072 0.068 0.109 | +0.62 +0.38 +0.42 +0.43 +0.44 +0.41 | +0.472 | 0.633 |
| `staged` | 0.293 0.226 0.172 0.121 0.082 0.107 | 0.227 0.204 0.188 0.145 0.075 0.161 | +0.89 +0.96 +0.92 +0.86 +0.75 **-0.51** | +0.309 | 0.668 |
| `deep_coda` | 0.142 0.159 0.161 0.165 0.174 0.198 | 0.171 0.117 0.121 0.136 0.178 0.277 | +0.77 +0.84 +0.93 +0.99 +0.99 +0.98 | +0.865 | 0.955 |
| `progressive` | 0.118 0.150 0.164 0.177 0.190 0.202 | 0.179 0.133 0.139 0.164 0.180 0.205 | +0.95 +0.97 +0.99 +0.98 +0.98 +0.98 | +0.947 | 0.978 |
| `fixed-point term` | 0.072 0.082 0.101 0.140 0.219 0.386 | 0.042 0.057 0.098 0.159 0.277 0.367 | +0.22 +0.16 +0.25 +0.09 -0.18 +0.69 | **-0.057** | **0.291** |
| `noise entry` | 0.176 0.169 0.166 0.164 0.163 0.162 | 0.157 0.158 0.159 0.164 0.178 0.184 | +0.93 +0.96 +0.99 +1.00 +0.98 +0.97 | +0.940 | 0.973 |
| `bptt last 2` | 0.000 0.000 0.000 0.000 0.521 0.479 | 0.000 0.000 0.000 0.000 0.521 0.479 | +0.00 +0.00 +0.00 +0.00 +0.96 +0.94 | +0.054 | 0.949 |
| `fixed depth 6` | 0.130 0.167 0.171 0.174 0.177 0.181 | 0.142 0.162 0.166 0.171 0.177 0.183 | +1.00 +1.00 +1.00 +1.00 +1.00 +1.00 | +0.997 | 0.999 |

Pooled over all 70 strict `compose` runs, split by outcome rather than by cell:

| group | n | value CE @6 | cotangent share p1..p6 | dW share p1..p6 | mean pairwise cos | cancellation | z rank |
|---|---|---|---|---|---|---|---|
| escaped | 27 | 0.008 [0.000, 0.156] | 0.146 0.140 0.134 0.135 0.193 0.252 | 0.107 0.091 0.100 0.121 0.218 0.363 | **+0.443** | **0.749** [0.197, 1.000] | 5.3 |
| stuck | 43 | 1.322 [0.539, 1.795] | 0.169 0.158 0.153 0.148 0.180 0.192 | 0.223 0.142 0.138 0.135 0.175 0.186 | **+0.845** | **0.946** [0.320, 1.000] | 1.4 |

That split is not an artefact of pooling different attachments. Within the 8 cells that
produced both outcomes, the escaped seeds read a LOWER cancellation ratio than the stuck
seeds in 7 of 8 (mean difference **-0.090**; the exception is fixed depth 6, where both
groups sit at 0.999):

| cell | escaped | cancellation, escaped seeds | cancellation, stuck seeds |
|---|---|---|---|
| `exit` | 2/5 | 0.898 | 1.000 |
| `deep_coda` | 2/5 | 0.887 | 1.000 |
| `mux_all` | 1/5 | 0.940 | 0.999 |
| `decay 0.9` | 3/5 | 0.906 | 0.997 |
| `injection off` | 3/5 | 0.911 | 1.000 |
| `per-pass LoRA` | 2/5 | 0.859 | 1.000 |
| `bptt last 2` | 2/5 | 0.873 | 1.000 |
| `fixed depth 6` | 2/5 | 0.999 | 0.999 |

Over the 60 grid-A runs, `corr(cancellation, K1-K6)` is **-0.604**. On `summary`, where no
loop can earn anything, the mean cancellation is **0.982**; on `compose` it is 0.863.

**This inverts the sign the real tree has been reading.** A shared-weight loop whose six
passes agree perfectly is a loop doing ONE thing six times. Disagreement is what a map with
six different jobs looks like.

## Mechanism: why each attachment does what it does

**What a loss at the exit alone asks of a shared map.** It says one thing: the composition
`f^T(h_0)` must decode to the answer. It says nothing about what `f` is. Every factorisation
that produces the right exit is equally acceptable, including "do all the work in pass 1 and
be the identity afterwards". On `compose` that shortcut is a genuine local optimum: it wins
the two answers that need at most one pass and abandons the other five, and it sits at
exactly 1.281 against the depth-1 ceiling of 1.2798. Improving from there is not incremental.
The map has to become a correct chain step for every position at once, and any half-move
breaks the answers it already has. That is why the outcome is bimodal — 27 of 70 strict runs
escaped, 43 sat at the shortcut — and why the honest metric is an escape rate, not a mean CE.

**Why the passes cancel, and why that is the healthy sign.** The measured association runs
the opposite way to the intuition the real tree has been working from. Stuck runs read a
cancellation ratio of 0.946 and a mean pairwise cosine of +0.845; escaped runs read 0.749 and
+0.443, and the within-cell comparison holds in 7 of the 8 cells that produced both outcomes.
The extreme cases make the reason plain. `mux_all` reads +0.966 pairwise, near-perfect
agreement, and escapes 1 of 5. The fixed-point arm reads **-0.057** pairwise and a
cancellation of **0.291**, which is BELOW the 0.408 that six orthogonal updates would give —
its passes actively oppose each other — and it escapes 5 of 5. `fixed depth 6` reads +0.997,
the most agreeing profile measured, on a setting that does nothing for the outcome. Six
passes of a chain have six different jobs: pass 1 combines spans i-1 and i, pass 5 combines a
5-span prefix with a 1-span suffix. Those are different functions of different state
distributions, so they ask for different weight changes. **Agreement between passes measures
how close the map is to doing one thing six times.** A high ratio is the signature of a
collapsed loop, not a healthy one.

**Why supervising the final answer at every pass makes it worse.** `mux_all` puts the exit
target on the state after every pass. For any answer that needs more than one pass that
demand is unsatisfiable, and the averaged term's best available response is the same at every
pass: emit the best one-pass answer. The gradient tables show that literally. The per-pass
cotangent under `mux_all` is a decaying ramp — 0.312 / 0.247 / 0.184 / 0.131 / 0.084 / 0.041,
so pass 1 receives 7.6x what pass 6 receives — and the pairwise cosine is +0.966. Six passes,
one instruction. The result is 1/5 escapes and a K-curve of 1.282 / 1.231 / 1.180 / 1.025 that
never reaches past R1. Per-pass supervision is not free credit assignment; it is an extra
constraint, and it is only useful when what it asks for at pass t is reachable at pass t.

**Why staged works.** `staged` asks the non-final passes for the OWN-span operator and the
exit for the composed register. The intermediate target is reachable in about one pass, so it
does not fight the chain; what it does is force the state to keep the span's own contribution
decodable at every pass, which is exactly the ingredient a later pass needs. The backward
shows the two jobs separating: the per-pass cosines to the total run +0.89 / +0.96 / +0.92 /
+0.86 / +0.75 and then **-0.51** at pass 6. The exit pass pulls opposite to the intermediate
passes, mean pairwise cosine drops to +0.309, and 5 of 5 seeds solve the chain. The exit state
also carries more: participation rank 6.5 against 1.2-2.4 everywhere else.

**Why detaching the carry stalls halfway.** `mux_all_detach` reaches R2 at 0.25 nats — one
span further than any other non-staged arm — and then stops dead at R3, on all 5 seeds, with
the tightest spread in the study (0.539 to 1.032, never at the shortcut). Detaching removes
the only term that tells pass t to prepare a state for pass t+1, so each pass is trained as a
one-step map from whatever it is handed to the answer. That is enough to learn one more link
and structurally cannot learn two.

**What the injection does to the backward.** Re-injecting the entry on a channel every pass
adds a path from `e` to `h_t` that skips passes 1..t-1, so a cotangent can reach an early pass
without traversing the whole product. In this toy that turns out to be a mild negative: decay
0.45 escapes 2/5 against 3/5 for decay 0.9 and 3/5 for no injection at all. Under the noise
entry the injection is the ONLY path and the arm dies completely (0/5, 1.531, worse than the
one-pass ceiling), with an entry state at participation rank 95.4 of 96 and an exit state at
rank 1.0 — the loop takes noise in and emits one direction. Worth naming: the toy did NOT
reproduce the real model's 33x cotangent ramp under the noise entry. Its shares are
0.176 / 0.169 / 0.166 / 0.164 / 0.163 / 0.162, essentially flat, because the injection gives
every pass its own path to `e`. So the toy explains why the noise entry is bad here (the
state has nothing in it to iterate on) and does not explain the real arm's ramp.

**Why a one-pass task can never make the loop earn depth.** If the target is computable at
pass 1, the exit loss is already minimised by a map that is the identity from pass 2 onward,
and no term in any of the six attachments asks for anything else. All 30 `summary` cells read
K1-K6 = +0.0000 and mean cancellation 0.982 — the loop's passes all agree, because they all
have nothing to do. **This is the control that says a flat K-curve is a statement about the
target, not about the wiring**, and it is why the geometry result matters: under MORPH's own
geometry, `compose` behaves like `summary`.

## Scorecard against the frozen predictions

| id | claim | outcome |
|---|---|---|
| P1 | exit-only earns depth on `compose`, K1-K6 > 0.30 | **TRUE** on the mean (+0.5135), but carried by 2 of 5 seeds; the modal seed earns nothing |
| P2 | exit-only earns nothing on `summary` | **TRUE**, K1-K6 +0.0000 at accuracy 1.000 |
| P3 | dense per-pass MUX is worse than exit-only | **TRUE**, 1.0255 against 0.7693, 1/5 against 2/5 |
| P4 | detached carry is the worst attachment | **FALSE**, it is third best and the only arm that beats the shortcut on every seed |
| P5 | staged beats dense | **TRUE**, 0.0000 and 5/5 against 1.0255 and 1/5 |
| P6 | deep coda within 0.05 of dense | **FALSE**, it equals exit-only to four decimals instead |
| P7 | progressive within 0.05 of exit-only, better at depth 12 | **FALSE** on both, 0.512 worse and flat at depth 12 |
| P8 | cancellation correlates POSITIVELY with earning | **FALSE, and the sign is the finding**: -0.604 across grid A, and `summary` reads a HIGHER ratio than `compose` |
| P9 | the noise entry hurts, and its cotangent is a 5x ramp | first clause **TRUE** (0.76 nats worse, 0/5); second clause **FALSE**, the ramp is flat |
| P10 | the fixed-point term hurts | **FALSE**, it is the strongest single lever measured (5/5 against 2/5) |
| P11 | per-pass LoRA changes nothing | **TRUE**, 0.7692 against 0.7693 |
| P12 | truncated BPTT destroys the task | **FALSE**, it costs 5x the escape time and 0.067 nats, not the capability |
| P13 | the coda blind to z sits at chance | **TRUE**, 1.7936 against ln 6 = 1.7918 |
| P14 | the write contribution is at least 0.30 nats | **TRUE**, +1.513 |

Seven of fourteen held. The three that matter most were wrong: the fixed-point term, the
direction of the cancellation ratio, and truncated BPTT.

## What I would carry to the real model, and what it must read

**A prediction about the arm that is running right now.** `morph/configs/tul_slot_mnext_mux_every_pass.yaml`
puts the M-next target on the state after every pass with a live carry and uniform weights.
That is bit-for-bit the intent of this study's `mux_all`, and `mux_all` is the worst live-carry
attachment measured here: 1 of 5 escapes against exit-only's 2 of 5, a K-curve that stops at
1.025, and a per-pass cotangent that decays 7.6x from pass 1 to pass 6 while the pairwise
cosine sits at +0.966. **I expect that arm to read flat or slightly worse than the ruler**, and
I expect its cancellation ratio to RISE above 0.520 rather than fall. If it instead earns
depth, this toy's central claim is wrong and should be said so.

**First arm I would run: the staged MUX target, under `norm_match`, with the fixed-point term
verified to be binding.** In MORPH the knob exists — `tul.mux_stage_own_iters` — but its
shipped semantics are two fixed stages: the own-span target for the first k iterations, the
next-span target at the exit. The toy's winning variant is that knob with **k set to the
loop's maximum depth**, so every non-final pass carries the own-span target. The only configs
that use it (`tul_to_mnext_y2_stage2.yaml`, `tul_to_mnext_y2_stage3.yaml`) are from the
2026-09-04 y2 lineage at k = 2 and k = 3, and neither runs under `norm_match`. In the toy this
is the only attachment that solves a task the loop must solve, on every seed, and it needs no
new code.

**Second, and cheaper to test: check that `core_fixed_point_lambda` is actually doing
something in `_tul_core`.** The ruler `slot-mux-norm-match` reports its fixed-point term at
**0.00302** and the A1 arm at 0.00023. A term that reads 0.003 against a loss of 11.28 is not
shaping anything. In the toy the same-shaped term at lambda 1.0 took escape from 2/5 to 5/5,
and the way it did it is visible in the backward: it turned a flat cotangent into a steep
rising ramp (0.072 to 0.386 over six passes) and drove the per-pass updates to mean pairwise
cosine -0.057. If the real term is inert, raising it until it is measurable is the cheapest
lever in this report.

**What the arm must read at 5,000 steps.** The ruler's numbers are token K1-K6 +0.0001 and
`mux_local` K1-K6 +0.0067 with K3-K6 +0.0005, and a combined cancellation ratio of 0.520.
A staged arm has earned the right to a longer run if:

1. `mux_local` K1-K6 > 0.02 with K3-K6 > 0.002 — the bar the credit-assignment note already
   set, and the only clause that speaks to depth;
2. the per-pass cosine profile from `slot_gradient_probe.py` develops at least one pass with
   a NEGATIVE cosine to the total, as the toy's staged arm does at pass 6 (-0.51);
3. **the combined cancellation ratio FALLS below the ruler's 0.520 rather than rising.**

Clause 3 contradicts the acceptance criterion in
[`2026-09-10-credit-assignment-in-the-slot-loop.md`](../../.agents/notes/proposed/architecture/2026-09-10-credit-assignment-in-the-slot-loop.md),
whose prediction P-d asks for the ratio to rise above 0.60. On the toy's evidence that
prediction is pointed the wrong way, and the progressive arm's filed result is consistent
with the toy: it moved the ratio from 0.520 to 0.497, a small move in the direction the toy
calls healthy, while every downstream K-curve stayed flat. Two readings survive that. Either
the ratio is not the lever (it moved and nothing followed), or 0.497 is nowhere near far
enough — the toy's escaped runs sit at 0.749 against stuck runs at 0.946, but its
best arm sits at 0.291. I would not run another arm on the ratio alone.

**What I would NOT carry.** The progressive loss (0/5 here, already flat on the real model),
deep supervision through the coda (identical to exit-only at 2.6x the cost), per-pass LoRA
(identical to exit-only), and dense per-pass MUX (actively worse). Three of those four are
either running or proposed in the real tree right now.

**The uncomfortable one.** The largest single effect in this study is not an attachment at
all. Holding the attachment fixed at exit-only and switching from the strict geometry to
MORPH's own moved the loop's contribution from +0.514 to +0.106 nats and made one core pass
worth 0.542 on a task that formally needs six. If that transfers, no loss attachment will fix
the real slot loop while the prelude and the coda can both compose across spans on their own.
The cheap test on the real model is not a new arm: it is the `tg_restrict` family that is
already in the tree, read as a K-curve rather than as a CE.

## What this toy cannot tell us

- **Scale.** 596,928 parameters (110,784 of them the shared core), 12 vocabulary ids, 48 cells, 4,000 steps. MORPH is 268 M
  parameters on 4,096-token rows for 100,000 steps. Basin structure at this size is not
  evidence about basin structure at that size.
- **Attention geometry.** The toy's core is one pre-norm block with ordinary softmax
  attention. MORPH's core is 6 blocks of CCA + CSA + HCA + XSA with a Cayley
  hyper-connection carrier of 4 streams, and the real slot states sit at effective rank
  1.7-4.8 in 1024 dimensions. Nothing here measures what those do to a loop.
- **Ternary weights.** Every toy weight is fp32 dense. The shipped recipe is ternary STE
  with `norm_match`, which is exactly the knob that changed the real loop's branch ratios
  from 0.2 to 0.9 on 2026-09-09. A toy in fp32 cannot speak to that.
- **The real data.** `compose` is a synthetic chain with a known, exact depth requirement
  and zero irreducible noise at the answer positions. Natural text has neither. The toy can
  say what happens WHEN a target needs iteration; it cannot say whether the M-next target
  on web text needs any.
- **Per-slot Poisson depth interacts with a chain in a way the real model may not share.**
  The toy's strict core is a chain, so slot i is worthless when slot i-1 drew a shorter
  depth. MORPH's core attends every earlier slot, so a short draw is less damaging there.
- **One optimiser, one learning rate, one budget.** No AdEMAMix, no beta1=0, no warmup
  study, no 20k-step horizon. Wolfe's rule that short-horizon CE cannot rank looped against
  unlooped applies to this toy too; the escape rate and the K-curve are used precisely
  because they are not a CE ranking.
- **No wandb run.** Every cell writes one JSON with its full config; there is no sweep
  dashboard and no hydra config for this lab spike.

---

## 2026-09-18 appendix: a task whose optimum is a SET

Nothing above this line is edited. This appendix records what was ADDED to the study on
2026-09-18 and why; the results of the `eliminate` grid are filed separately under
`lab/experiments/`, with the predictions frozen beforehand in
[`2026-09-18-toy-eliminate-deferred-commitment.md`](../experiments/planned/2026-09-18-toy-eliminate-deferred-commitment.md).

The study above answers "does the loop earn depth when the target needs iteration?" on two
tasks whose optimal intermediate state is a POINT: `compose` carries one group element and
`summary` carries nothing at all. The superposition literature (Zhu et al., NeurIPS 2025;
Rizvi-Martel et al., 2026) says a loop earns depth when the optimal intermediate state is a
SET of candidates that later evidence prunes, that the set emerges without being supervised,
that it collapses to one candidate under pretraining, and that it needs width. This repo has
already reproduced the collapse half on its own stack — the Thought Register arm seeded four
cells apart and read rank 1.24 of 4 — and had never built a task that asks for the set.

`eliminate` is that task, and its whole design is the ceiling table in
[PLAN.md](PLAN.md#2026-09-18-extension-the-eliminate-task-appended-nothing-above-is-edited):
four plateaus, 1.386 / 0.924 / 0.693 / 0.000, one for depth starvation and one for each
point at which a state could collapse to a single candidate. A stuck run's plateau says
where it committed. That is the same device as `compose`'s 1.28-against-0.00, with two extra
rungs that only exist because the intermediate state is a set.

Two of the three new instruments exist because of a trap that the older instruments walk
straight into. The mean mass on the survivor at a 3-candidate state is 1/3 for a genuine
superposition AND 1/3 for a state that committed to a uniformly random candidate, so
`candidate_mass` reports the entropy and the top mass next to the three masses. And a linear
probe finds the alive set in an UNTRAINED network, because a random map preserves linearly
decodable information, so `membership_probe` is read as a destruction test — a low accuracy
is evidence, a high accuracy is not — and the grid carries two random-init cells as its
baseline.

**What the grid found** (32 cells, 2026-09-18, filed at
[`lab/experiments/results/2026-09-18-toy-eliminate/README.md`](../experiments/results/2026-09-18-toy-eliminate/README.md)):
every one of the 30 trained cells solves the task to value CE 0.0000, so a 2-pass chain has
no basin and the escape rate says nothing. The set IS carried — a linear probe reads it at
1.000 the moment an attention hop can reach it — but it is not carried as a superposition of
candidate embeddings: the tied head's mass sits on the candidate the state can already prove
DEAD, in 20 of 20 solved cells. The same probe reads 0.95–1.00 on a RANDOM-INIT network, so
carrying a set through this loop costs nothing and needs no training; what training buys is
the readout. The one lever that makes a slot state read as a spread rather than a point is
the terminal fixed-point term, at entropy 0.989 of a possible ln 3 = 1.0986 against 0.352
for plain `exit` and 0.096 at random init — the same term that was the strongest lever in
the study above.

---

## 2026-09-19 appendix: six hops, and what the two-hop task got wrong

Nothing above this line is edited. `eliminate6` is `eliminate` with the chain deepened from
two hops to six: eight candidate symbols, six drawn per instance over spans 0-1, five
eliminations at spans 2-6, the answer at the head of span 7. Its ceilings, its grid and its
scorecard are filed at
[`lab/experiments/results/2026-09-19-toy-eliminate6/README.md`](../experiments/results/2026-09-19-toy-eliminate6/README.md),
with the predictions frozen beforehand in
[`2026-09-19-toy-eliminate6-hop-distance.md`](../experiments/successes/2026-09-19-toy-eliminate6-hop-distance.md).

**It overturns the previous appendix's headline.** "Carrying a set through this loop costs
nothing and needs no training" was measured on a two-hop chain. Read at the answer slot of
an UNTRAINED network, the probe for "is symbol c already eliminated" reads 1.000 at hop 0,
0.881 at hop 2 and **0.668 at hop 4**, and never recovers; every trained cell reads 0.999
or better at every hop. Carrying a fact one or two hops is free. Carrying it six is learned.

Three more things this task could say that the two-hop one could not. The chain fails at
its LAST two hops, and it fails in the state rather than in the head: every stuck cell's
`in_set` probe sits at 0.89, the ceiling reachable from the five eliminations alone, and
the cells that pass it are exactly the cells that solve the task. A ridge-fitted linear
reader on the exit state beats the coda by a median of +0.030 nats on stuck seeds, so the
tied head is not the bottleneck here. And width is worth more than any attachment measured
in this study: 96 dimensions solve 2 of 15 cells, 192 solve 8 of 15.

The fixed-point term beat plain exit-only for the third independent time (2/5 against 0/5
at d96, 4/5 against 3/5 at d192). `staged` was the worst attachment at 1 of 10, and its
middle slot puts per-symbol mass 0.333 on the dead candidates and 0.000 on the survivor —
its own-span target asks for exactly that.
