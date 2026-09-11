# Planned: gradient-conditioned passes on the slot loop — the loop reads its own error

Status: failure

Date: 2026-09-10 (frozen before launch). Arc: `2026-09-04-loop-contribution-arc.md`.
Follows the five credit-assignment arms —
[`slot-mnext-progressive`](../failures/2026-09-10-arc-slot-mnext-progressive.md),
[`slot-mnext-per-pass-lora`](../failures/2026-09-10-arc-slot-mnext-per-pass-lora.md),
[`slot-mnext-mux-every-pass`](../failures/2026-09-10-arc-slot-mnext-mux-every-pass.md),
[`slot-mnext-staged`](2026-09-10-arc-slot-mnext-staged.md) and
[`slot-mnext-staged-all`](2026-09-10-arc-slot-mnext-staged-all.md) — and the toy study
built to explain why the first three read flat,
[`lab/toy_slot_loop/WRITEUP.md`](../../toy_slot_loop/WRITEUP.md). Design note:
[`2026-09-10-gradient-conditioned-slot-passes.md`](../../../.agents/notes/proposed/architecture/2026-09-10-gradient-conditioned-slot-passes.md).
Sources: Marino, Yue & Mandt 2018, *Iterative Amortized Inference* (arXiv:1807.09356);
Greff et al. 2019, *Multi-Object Representation Learning with Iterative Variational
Inference* (IODINE, arXiv:1903.00450).

## Question

Every credit-assignment arm on the slot loop so far changed the LOSS side. `progressive_p`
changed which passes carry the one exit gradient. `pass_lora_rank` gave the passes their
own parameters. `mux_every_pass`, `mux_stage_own_iters` and `mux_stage_all` changed what
the passes are asked for. All of them left one thing fixed: **what a pass can SEE**. A shared map applied
six times to a state plus a fixed seed is the same instruction six times, and the
measurements say that is what it behaves like — the per-pass cotangent is flat
(0.168/0.168/0.166/0.160/0.154/0.183 on the ruler), the six passes' updates to the shared
weights largely cancel (`|sum_t dW_t| / sum_t |dW_t|` = 0.520 against 0.408 for six
orthogonal equal-norm updates), and the coda's CE with the loop's exit equals its CE with
the loop's ENTRY to 0.0015 nats while a gradient-fitted z is worth 0.9-2.6 nats
([`slot_z_optimize`](../results/2026-09-10-slot-z-optimize/README.md)).

That last pair is the whole shape of the problem: **the reader works and the writer
produces nothing the reader wants**, and the thing a gradient-fitted z has that the loop's
z does not is a gradient. Marino et al. and IODINE both close exactly that gap: hand the
iterative network, at every step, the gradient of its own objective with respect to the
state it is refining, and it can learn to be an optimiser of that objective instead of
re-running one fixed update rule. The input then differs at every pass by construction,
because the error signal changes as the state moves.

Does giving each pass `dL_own/dh_t` as an extra input make the passes do different work,
and does the exit state the coda reads change as a result?

## Hypothesis

H-gp-1 (for): the loop is a fixed-point iteration with no error signal, so pass `t+1` has
no way to know what pass `t` got wrong. Feeding the gradient of a local, causal target
makes each pass a step of an optimiser: the own-span loss should FALL along the pass index
inside a forward (`loop/own_pass_t{t}`), the per-pass weight updates should point in more
distinct directions (a LOWER cancellation ratio), and the exit state should carry
something the entry does not (exit-minus-entry CE up).

H-gp-0 (against): the own-span content is already in the slot's entry state — the prelude
sees the whole span before the slot position — so the own-span gradient is small and
nearly the same vector at every pass, and `W_g(g_t)` degenerates into one more constant
seed injection. On this reading the arm reads exactly like the ruler with an extra
injection channel, which the levers panel already measured as worth 0.002-0.003 nats.

The two readings disagree about `loop/own_pass_t{t}`, which is why it is logged and why
P-e below is a prediction and not a readout.

## Method

`tul_slot_mnext_gradpass` = `tul_slot_mux_norm_match` (the ruler `slot-mux-norm-match`)
plus ONE change: `tul.grad_pass: true` (`grad_pass_scale: 0.1`, `grad_pass_norm: rms`, both
defaults).

**The mechanism, as implemented.** Before each pass `t` of `_tul_core`, the model takes the
current slot carry `h_t`, detaches it, computes `L_own(h_t) = _tul_mux_loss(h_t,
target="own")` — the slot's OWN span read through the tied head, the same head and the same
weighted-CE code the exit MUX uses — and calls `torch.autograd.grad(L_own, h_t,
create_graph=False)`. The result is reduced over the Hyper-Connection streams (the MUX
head's readout starts with a stream mean, so the gradient at each stream is the same vector
over `n`), divided by its own per-slot RMS, multiplied by `grad_pass_scale`, mapped through
a ZERO-INITIALISED `W_g` (`[C, C]`, `morph/model/tul.py::TULGradPass`), and broadcast-added
to the state the core step receives through `_apply_injection` — the same single-stream
insertion every other injection into this carrier uses. `_tul_core` gained one keyword,
`input_ids`, because the own target needs the row's token ids.

**Causality.** `mux_span_targets(target="own")` gives slot `i` the tokens of span `i`, and
slot `i` sits after every one of them, so a generator can compute the same feature with no
lookahead. The forecast gradient (`target="next"`) would NOT be causal — that is the
`slot_z_optimize.py` oracle, and it is deliberately not what this arm feeds.

**The gradient is a FEATURE.** `create_graph=False`: the injected tensor has no `grad_fn`,
so the token CE never differentiates through the inner backward and no second-order term
exists. `W_g` is the only route the outer graph has into the feature, and it is a live one
— `dL/dW_g = (dL/dh_in) (x) g` does not vanish at `W_g = 0`, so the zero init is a starting
point, not a trap. This is IODINE's own choice (§3.1); the second-order alternative
(`create_graph=True`, the learned-optimiser objective of Andrychowicz et al. 2016) is NOT
built and is the first follow-up if the feature acts and the arm still reads flat.

**The exit target is unchanged.** The loop's LOSS is the ruler's, exactly: one M-next
forecast MUX at beta 1 on the exit state, plus the token CE. This knob adds an input, never
a term. The toy study's constraint — the local target must DIFFER from the exit target, or
per-pass supervision becomes depth-independence training (`mux_all`, 1/5 seeds; measured on
MORPH as `slot-mnext-mux-every-pass`, flat) — is respected by construction: own span
inside, next span at the exit.

**Interactions, each one decided rather than assumed.**

* *Checkpointing.* The feature is built OUTSIDE `_core_step`. `ckpt_grad_iters` wraps
  `_core_step` in `torch.utils.checkpoint`, whose recompute re-runs that function under a
  fresh grad context, and running an inner `autograd.grad` there is not a shape this tree
  has ever executed. Building the feature in the loop body and letting only the finished
  tensor cross the checkpoint boundary keeps selective core checkpointing working
  unchanged, so the knob does NOT force `ckpt_grad_iters=0` and pays no extra memory for it.
* *The gain hinge.* It receives the INJECTED `_h_in`, so it probes the map at the operating
  point the run actually reached. Its finite difference still treats the feature as a
  CONSTANT (the feature is a function of `h`, not of the probe's perturbation) — the same
  "exogenous input" reading IODINE takes — which means the hinge does not bound the
  feature's own contribution to the gain. Named here, not hidden.
* *The terminal fixed-point term* applies at each slot's last pass: untouched.
* *The Jacobian capture* records the PRE-feature `h`; the loop probes read `in_norm`
  off `h` and `out_norm` off the post-feature step, so on this arm `core_gain`
  includes whatever the feature adds and is NOT comparable pass-for-pass with the
  ruler's. Both are stated in the code at the place they are taken.
* *Truncated BPTT.* At a no-grad iteration the term is still built; the step that consumes
  it runs under `torch.no_grad()`, so no edge to `W_g` survives and the window is what it
  was. Moot on this arm (`bptt_depth 8` >= `max_depth 8` = full BPTT).
* *SCSE, `db_loop`, the paid loop, `mux_beta 0`, `n_core 0`, a caller that omits
  `input_ids`*: all refused, with tests.

**Cost.** Per pass the feature costs one `[B, S, V]` fp32 readout of the tied head and the
backward of that readout with respect to z alone (`mux_detach_head: true` keeps the
unembedding out of it): about `2 * B * S * C * V * 2` = ~77 GFLOP at B 6, S 64, C 1024,
V 49169, times the batch's deepest slot (<= 8) = ~0.6 TFLOP per step. `autograd.grad` frees
each inner graph, so the memory is one transient pair of `[6, 64, 49169]` fp32 buffers
(~151 MB), NOT one retained per pass. The anchor is `slot-mnext-mux-every-pass`, which
computes MORE than this (T live MUX terms, retained through the outer backward) and read
0.99x the ruler's wall clock at 12,063 tok/s and a 12.62 GB smoke peak against the ruler's
10.1 GB. Expectation: 1.0-1.15x wall clock, smoke peak near 11 GB, tok/s comfortably above
the 8,086 floor. That is arithmetic plus one anchor, not a measurement.

**Run.** 5,000 steps, same recipe as the ruler in every other respect: M-next MUX at beta 1
through the tied head, prelude entry, hinge lambda 100 at target 0.9,
`core_fixed_point_lambda` 1.0, `slot_cot_clip` 4.0, `ternary_scale_mode: norm_match`, seq
1024, batch 6, seed 1, ramp 1,000, `tg_scoped_kernels`. Runner `arc/run_slotloop3.sh`
(file-driven queue, commit pinned in `arc/slotloop3_arms.txt`), a 12-step smoke first, the
draw under the sustained tripwire (`lab/divergence/tripwire_sustained.py`). Rate rule: tok/s
below 8,086 at step 200 skips the arm and the queue continues (the ruler read 12,429).

**Readout.** `core_depth_sweep.py` at checkpoints 2,500 and 5,000, depths 1,2,3,6,9,12,16 —
the token K-curve and the `mux_local` forecast K-curve on 480 rows. `worth_profile.py` and
`slot_state_probe.py` at 5,000 (the runner's `slot` kind does both). `slot_gradient_probe.py`
and `slot_z_optimize.py` at 5,000 by hand: the gradient probe carries the cancellation ratio
and the per-pass cotangent/cosine profile, the z-optimisation probe carries the coda's
exit-minus-entry CE. New wandb keys this arm writes: `loop/own_pass_t{t}` (the local
target's value at each pass — the ladder H-gp-1 and H-gp-0 disagree about) and
`loop/gp_rel_t{t}` (the injected term's norm over the state's, which reads 0 at step 0 by
construction and says whether the optimiser ever moved `W_g` off zero).

The ruler's numbers, cited once: 480-row CE at depth 6 = **4.3290** at step 5,000; token
K1-K6 +0.0001 [-0.0000, +0.0002], K3-K6 -0.0000; `mux_local` (forecast) K1-K6 +0.0067
[+0.0053, +0.0081], K3-K6 +0.0005; wall clock 52 min 38 s for 5,000 steps; combined
cancellation ratio 0.520; per-pass cotangent share 0.168/0.168/0.166/0.160/0.154/0.183;
per-pass cosine to total 0.33/0.27/0.37/0.63/0.75/0.67; z-optimisation `ce_loop` 4.1173 with
the loop ENTRY worth +0.0015.

## Predictions (frozen)

- **P-a (survival).** HEALTHY to 5,000, no sustained tripwire (`preclip/total > 1e4` at step
  >= 200): **82 %**. Reasoning: the loss is byte-for-byte the ruler's and `W_g` starts at
  zero, so the arm enters training as the ruler and moves off it gradually. The 18 % is for
  the one thing the ruler's stability apparatus does not cover — the hinge treats the
  feature as exogenous, so a `W_g` that grows large adds gain the constraint cannot see.
- **P-b (the rate rule).** tok/s at step 200 above 8,086: **92 %**. Reasoning: the
  `mux_every_pass` anchor computed more per step and read 0.99x. The 8 % is for the inner
  backward's launch overhead at T up to 8 on the eager TG-scoped path, which the anchor
  does not cover (its terms went through ONE outer backward, not T small ones).
- **P-c (tokens).** Token K1-K6 at 5,000 above 0.03 (the plain prelude-entry ruler
  `plain-panel-norm-match`): **14 %**. Above 0.10 (the plain NOISE-entry value): **5 %**.
  Reasoning: nothing here touches the reader, and every arm measured so far has left the
  coda's exit-minus-entry CE in a narrow band. Set slightly above the staged arm's 12 %
  because this is the first arm that changes the loop's INPUT rather than its loss, and the
  input is the only thing that has never been varied.
- **P-d (forecast K-curve, the bar).** `mux_local` K1-K6 above 0.02 at 5,000 (the ruler:
  0.0067): **30 %**. K3-K6 above 0.002 (the ruler: +0.0005): **20 %**. Reasoning: this is
  the arm's direct claim — if the passes become optimiser steps, later passes should reach
  a forecast earlier passes cannot. Set above the staged arm's 28 %/18 % because the
  mechanism makes the passes' inputs differ, which is a stronger form of "the passes are
  not redundant" than giving one pass a different target; set well under 50 % because
  H-gp-0 is a live reading and the own-span content is already in the entry state.
- **P-e (the ladder, and the mechanism's own signature).** `loop/own_pass_t{t}` FALLS with
  the pass index at step 5,000 — `own_pass_t0 - own_pass_t_last` above 0.02 nats, averaged
  over the last 100 logged steps: **35 %**. Reasoning: this is the most direct test of "the
  loop became an optimiser of the target it is handed the gradient of", and it does not
  require the coda to read anything. It is the highest probability in this file because it
  is the closest to the mechanism and the furthest from the reader. At step 0 of the CPU
  smoke the ladder RISES (4.614 / 4.807 / 5.019): the untrained loop degrades its own span,
  so there is room to fall.
- **P-f (the feature acts at all).** `loop/gp_rel_t0` above 0.01 at step 5,000 (the
  injected term is at least 1 % of the state's norm): **60 %**. Reasoning: the falsifier for
  "the optimiser left `W_g` near zero and the arm is the ruler under another name". Set high
  because `W_g` has a live gradient from step 1 and the CPU smoke moves it (‖W_g‖ 0 ->
  0.064 -> 0.107 -> 0.147 over three AdamW steps at lr 1e-3); set below 90 % because 5,000
  steps of a ramped 1e-4 schedule is not three steps at 1e-3, and weight decay pulls the
  other way. If P-f is FALSE the arm is UNINFORMATIVE about P-c/P-d and the honest next
  move is a scale sweep, not a verdict.
- **P-g (a negative-cosine pass).** At least one pass reads a NEGATIVE cosine to the total
  core-weight gradient in `slot_gradient_probe.py` (the ruler and every arm read so far are
  positive at every pass; the toy's `staged` cell reached -0.51): **28 %**. Reasoning: the
  toy's separation signature. Different inputs per pass is a plausible route to it, but the
  two jobs here (refine the own span, forecast the next) are closer to each other than the
  toy's compose-vs-shortcut pair.
- **P-h (cancellation).** Combined cancellation ratio on the shared core weights BELOW the
  ruler's 0.520: **35 %**. Reasoning: the toy found -0.604 correlation between cancellation
  and K1-K6 across grid A, and `mux_every_pass` ROSE to 0.771 while reading flat. A fall is
  the toy's signature of a loop doing more than one thing. Set above the staged arm's 30 %
  for the same reason as P-d: this arm makes the inputs differ, not just the target.
- **P-i (val CE).** 480-row CE at depth 6 within 0.05 of the ruler's 4.3290: **55 %**.
  Reasoning: not a verdict either way (Wolfe 2026-09-09: short-horizon CE cannot rank looped
  against unlooped); recorded because every arm files it.
- **P-j (cost).** Wall clock within 1.5x of the ruler's 52 min 38 s: **90 %**. Within 1.2x:
  **72 %**. Reasoning: see the cost paragraph; the anchor read 0.99x for more FLOPs, and the
  1.5x bar is generous enough that only a launch-overhead surprise breaks it.

Honest framing before the run: the toy study's second finding (Q4) is the biggest reason to
expect a flat reading, and it is orthogonal to everything this arm changes. Under MORPH's
own PERMISSIVE geometry — prelude causal over every cell, coda reading a causal chain of
prefix cells — the toy's iteration-forced `compose` task behaved like its one-pass control:
one core pass reached 0.283 nats and 66 % accuracy, because the prelude and the coda compose
across spans by themselves. If the loop has no job in MORPH's geometry, making its passes
better optimisers of a local target changes what the loop DOES without changing what the
model NEEDS. That is why P-e (the ladder) and P-f (the feature acts) are the two predictions
this file is really built on: they are readable whether or not the loop matters, and they
are what tells the next arm whether to change the mechanism or change the geometry.

## Binding

- P-e TRUE and P-d TRUE ⇒ the mechanism transfers: queue the second-order form
  (`create_graph=True`, one Hessian-vector product per pass) and a matched run under
  `tg_restrict`, where the loop is the only cross-span route and the toy predicts the
  largest effect.
- P-e TRUE, P-d FALSE ⇒ the loop DID become an optimiser of its local target and the
  forecast did not follow. That separates "the loop cannot iterate" from "iteration buys
  nothing here", which no arm has separated yet. Next is the geometry, not the loop: read
  the `tg_restrict` family as a K-curve.
- P-e FALSE, P-f TRUE ⇒ the feature acts and the loop does not descend its own target. Read
  `loop/gp_rel_t{t}` against the pass index first: a term that grows with `t` while the loss
  does not fall says `W_g` learned a state-dependent injection, not an optimiser step, and
  the next test is the scale (`grad_pass_scale` 0.03 / 0.3), not a new mechanism.
- P-f FALSE ⇒ UNINFORMATIVE, not a failure. `W_g` stayed near zero and the arm ran the
  ruler. Re-run at `grad_pass_scale` 1.0 or with `W_g` at a small random init before reading
  P-c/P-d/P-g/P-h at all.
- All of P-c, P-d, P-e, P-g, P-h FALSE with P-f TRUE ⇒ this closes the sixth and last
  credit-assignment attack on `slot-mux-norm-match`. Combined with the four filed arms, the
  loss side, the parameter side, the target side and now the INPUT side have all been
  changed one at a time with no effect, and the honest next step is the toy's Q4 taken at
  face value: test whether MORPH's permissive geometry already does the job any loop
  attachment could do.

## Not verified before launch

The arm has NOT run on a GPU, and no GPU job was launched from this session. `torch.compile`
behaviour of an inner `torch.autograd.grad` inside the traced forward at the real shape
(B=6, S=64, L_total=1152, T up to 8) is **untested** and is the single most likely way this
arm fails to start — the runner's 12-step smoke is the first real evidence either way, and a
graph break there would show up as a rate stop rather than an error. Memory at the real
shape is arithmetic plus one anchor, not a measurement. The interaction of the feature with
`tg_scoped_kernels` (the fused/eager split the ruler runs) is unmeasured. The gain hinge
reads the map with the feature held constant, so the arm's realised gain may exceed what the
hinge bounds; `loop/core_gain*` and the sustained tripwire are the only watch on that. The
interaction with `mux_every_pass`, `mux_stage_own_iters`, `mux_stage_all` and
`progressive_p` is ALLOWED by the config (no refusal) and has never been run — this arm
sets none of them. `tul.mux_readout: full` (finding F2, master `f8dea98`) changes the
reduction the feature uses: under `mean` every Hyper-Connection stream carries the same
gradient vector (measured spread exactly 0.0) and under `full` they differ (spread 7.9e-5
against a max |g| of 4.1e-3, about 2 %), so the mean over streams becomes a summary rather
than a recovery of one vector. A CPU test holds the contract under both; NEITHER has run on
a GPU and this arm uses the default, `mean`. `ckpt_grad_iters` is not overridden and the
interaction was reasoned about, not measured: no run has yet checkpointed a core step whose
input carries a feature built outside the checkpoint.
CPU evidence only: `pytest tests/ -q` reads 1,063 passed, 9 skipped, 1 xfailed, and a
3-step AdamW loop on the tiny slot-loop model moves ‖W_g‖ 0 -> 0.064 -> 0.107 -> 0.147 with
the loss falling 9.339 -> 8.354 -> 7.724.

## Results

Filed 2026-09-11 05:01. `slot-mnext-gradpass` (commit `8a287ed`; the ruler plus `tul.grad_pass: true`,
`TULGradPass` with `W_g` at zero init, the own-span loss gradient injected before each pass):
HEALTHY to 4,999, tripwire max 61.7 at step 245 (ruler 78.7 at the same step), 12,261 tok/s at
step 200 (ruler 12,429), smoke peak 10.87 GB, training peak 11.98 GB, wall clock 56 min 09 s
(ruler 52 min 38 s, 1.07x). Runner final val_loss 4.3263 (ruler 4.3775). Over steps 4,900 to
4,999: `gain_est` 0.885 (hinge bound on 0 of 5,000 steps), `loop/delta_ratio_last` 0.061
(ruler 0.065), `loop/core_gain_t0` 2.19, fixed-point term 0.010.

The mechanism's own instruments, averaged over steps 4,900 to 4,999:

| | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `loop/gp_rel_t` (injected term / state norm) | **0.086** | 0.034 | 0.028 | 0.026 | 0.024 | 0.021 | 0.019 | 0.016 |
| `loop/own_pass_t` (own-span CE from the state at pass t) | 7.105 | **6.740** | 6.703 | 6.713 | 6.729 | 6.731 | 6.724 | 6.721 |

The feature acts (8.6 % of the state at pass 0, never under 1.5 %). The ladder falls 0.385
nats from t0 to t7, of which 0.366 is the FIRST pass; t1 to t7 is 0.019 and passes 3 to 7 sit
within 0.03 of each other, slightly rising. The loop takes one descent step on its own target
and then holds. The ruler logs no ladder, so the size of the t0 to t1 drop has no paired
control (the untrained CPU smoke's ladder ROSE 4.61 / 4.81 / 5.02, so the direction is learned).

Sweeps, 480 rows, forced depths 1/2/3/6/9/12/16:

| step | tokens K1−K6 | tokens K3−K6 | forecast K1−K6 | forecast K3−K6 | forecast @1 / @3 / @6 (ruler @1 / @6) | CE@6 (ruler) |
| --- | --- | --- | --- | --- | --- | --- |
| 2,500 | +0.0012 [+0.0009, +0.0014] | +0.0001 | +0.0159 [+0.0136, +0.0184] | +0.0010 [+0.0002, +0.0018] | 6.8720 / 6.8571 / 6.8561 (6.8766 @6) | 4.7106 (4.7194) |
| 5,000 | +0.0014 [+0.0012, +0.0017] | +0.0001 [+0.0000, +0.0002] | **+0.0236 [+0.0207, +0.0269]** | +0.0016 [+0.0009, +0.0023] | 6.7793 / 6.7573 / **6.7557** (6.7791 / 6.7724) | 4.2772 (4.3290) |

The forecast K1−K6 crosses the 0.02 bar, and it does so the right way round: depth 1 matches
the ruler's depth-1 forecast to 0.0002 and depth 6 is 0.017 nats BETTER than the ruler's
depth-6 forecast, the first slot arm on this tree whose exit beats the ruler's by more than
the fullread arm's 0.016. The gain is in passes 1 to 3 (K1−K3 +0.0220); K3−K6 +0.0016 is
under its bar. Tokens read +0.0014, 14x the ruler and still an order of magnitude under the
0.03 bar. CE is 0.052 better than the ruler at depth 6 (the ruler family at 5k spans 4.25 to
4.33 across arms; a horizon reading, not a verdict).

Worth profile at 5,000 (offsets 0..6): zero +0.080 [+0.070, +0.089], +0.073, +0.039, +0.037,
+0.027, +0.020, +0.017 (ruler +0.094 .. +0.006); shuffle +0.053, +0.052, +0.034, +0.033,
+0.020, +0.013, +0.005; wrong_seed +0.060, +0.048, +0.030, +0.026, +0.017, +0.014, +0.010.
State probe: |h| 165 at depth 1 → 185 → 195 → 209 at 6 → 226 at 16; relative distance from
depth 1 0.167 / 0.260 / 0.409 / 0.618, cos 0.994 / 0.985 / 0.962 / 0.910 (masked ruler-family
0.152 / 0.232 / 0.350 / 0.516): a contracting trajectory of the ruler's kind, moving a little
more per pass. (Correction 05:06: the first filing said no state probe ran; it did, and its
log and JSON are in the artifact directory.)

Gradient probe at 5,000 (12 rows, batch 2, depth 6, eager, train-mode dropout seeded per
batch; self-checks PASS at max rel err 1.9e-7): combined cancellation ratio **0.502** (ruler
0.520; the family spans 0.50 to 0.56, so this is inside the spread); MUX 0.504, token CE
0.551. Per-pass share of the shared core weight gradient 0.138 / 0.129 / 0.113 / 0.138 /
0.208 / 0.275; per-pass cosine to the total +0.09 / +0.05 / +0.57 / +0.76 / +0.73 / +0.59
(passes 1 and 2 near orthogonal to the total, none negative). Cotangent share 0.171 / 0.176
/ 0.175 / 0.163 / 0.149 / 0.165, clip never binds. Parameter-group norms: prelude 2.98, coda
2.23, core.residual 1.73, core.attention 1.15, core.mlp 1.15, `tul_grad_pass` 0.175,
W_prefix 0.10. z-optimisation probe (12 rows, batch 2, 200 Adam steps, bit-exact): ce_loop
4.0699, entry **+0.0075** (ruler +0.0015; 5x, under the 0.02 bar), zero +0.0250, shuffle
+0.0179, fitted z −0.839 (random start −0.702; cos(z*, z_loop) +0.919), loop z rank 10.5.

Probes ran 04:53 to 05:00 after the staged-20k arm's rate check, 13.2 GB free at start.
Artifacts: `lab/experiments/results/2026-09-10-slot-mnext-gradpass/`; npz under
`ignored/experiment-artifacts/2026-09-10-slot-mnext-gradpass/`; probe JSON under
`ignored/experiment-artifacts/2026-09-10-slot-gradient-probe/` and `-slot-z-optimize/`.

## Verdict

P-a TRUE. P-b TRUE. P-c FALSE at both bars as predicted (+0.0014). **P-d TRUE at the K1−K6
bar** (+0.0236 > 0.02, given 30 %), FALSE at the K3−K6 bar as predicted (+0.0016). **P-e
TRUE** (0.385 > 0.02, given 35 %; one pass deep). **P-f TRUE** (0.086 > 0.01). P-g FALSE as
predicted (no negative pass). **P-h TRUE** (0.502 < 0.520, given 35 %; inside the family's
spread, so a letter-true reading, not a signature). P-i FALSE by its letter and on the good
side (0.052 better). P-j TRUE at both bars (1.07x). Three mechanism predictions came in TRUE
against the file's majority lean (P-d, P-e, P-h), so by the arc's convention (the predictions
did not hold) the file goes under failures; the arm itself is the first credit-assignment arm
whose exit beats the ruler's forecast.

Binding clause 1 fires by its letter (P-e TRUE and P-d TRUE): the mechanism transfers. Its
size has to be read with it: the loop takes ONE descent step on its own target (the ladder's
fall is at pass 1; t1 to t7 is 0.019), the forecast gain is in passes 1 to 3 and the exit is
0.017 nats better than the ruler's, and passes 3 to 6 are as redundant as on every other arm
(K3−K6 0.0016). The reader's use of the exit rose 5x (entry-vs-exit +0.0075) and stays under
the 0.02 bar; the tokens read +0.0014. The clause names the next two arms: the second-order
form (`create_graph=True`, one Hessian-vector product per pass) and a matched run under
`tg_restrict`. Both are recorded here as the proposal and NOT queued tonight (Wolfe's call
on the morning of 2026-09-11; the strict-geometry twins of this batch both read flat under
`norm_match`, which lowers the second one's prior).

## Updated hypothesis

Handing a pass the gradient of a local target changes what the FIRST pass does, and only the
first: the loop becomes a one-step optimiser of its own span and then a fixed point of the
same kind as the ruler's. That is the same shape as every other arm on the tree (one pass does
the work), with a better first pass. The measured order of exit improvements on the ruler
family is now gradpass 0.017 > fullread 0.016 > staged 0.003 > ruler, and none of them moves
K3−K6 past 0.002. The lane stays the target: an own-span gradient is a target the loop can
descend, and it descends it in one step because the span is already in the entry state; a
target the entry state does NOT already contain is what would make a second step worth
taking. The Parcae core (a4610cf's filing) is the place to build that, at 0.63x the cost.
