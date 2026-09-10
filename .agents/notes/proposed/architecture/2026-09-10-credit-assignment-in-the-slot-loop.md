# Agent Note: Credit assignment in the slot loop

Status: proposed

Date: 2026-09-10. Arms:
[`slot-mnext-progressive`](../../../../lab/experiments/failures/2026-09-10-arc-slot-mnext-progressive.md),
[`slot-mnext-per-pass-lora`](../../../../lab/experiments/failures/2026-09-10-arc-slot-mnext-per-pass-lora.md)
and
[`slot-mnext-mux-every-pass`](../../../../lab/experiments/failures/2026-09-10-arc-slot-mnext-mux-every-pass.md),
all one-factor arms on the ruler `slot-mux-norm-match`
(`morph/configs/tul_slot_mux_norm_match.yaml`).

## Problem

The slot loop is stable and empty. Under the shipped ternary rule the M-next arm reads a
token K-curve of +0.0001 nats from depth 1 to depth 6 over 480 rows and a forecast
(`mux_local`) K-curve of +0.0067, with K3-K6 at 0.0005. Every FORWARD-side lever tried
since has read flat too: the noise entry (+0.002), the terms off (+0.003), the reread
(+0.0003), a fixed depth (a bowl, at a worse CE), a wider slot channel, a different coda.
The panel's structural finding is that the core blocks move the slot state by 2-20 % per
pass while the same blocks move token states by 86-110 %.

The per-pass gradient probe
([`lab/experiments/results/2026-09-10-slot-gradient-probe/README.md`](../../../../lab/experiments/results/2026-09-10-slot-gradient-probe/README.md))
closed the obvious backward explanation and opened a new one:

1. The cotangent does NOT vanish through the loop under the prelude entry. Per-pass shares
   on `slot-mux-norm-match` are 0.168 / 0.168 / 0.166 / 0.160 / 0.154 / 0.183. There is no
   vanishing-gradient problem to fix.
2. The MUX pays for the loop 7.3x over the token CE (3.5e-3 vs 4.8e-4 into the loop state).
   Without the MUX the shared core receives about 1 % of the prelude's gradient norm.
3. **The passes fight each other.** `|sum_t dW_t| / sum_t |dW_t|` on the shared core
   weights is 0.520 combined, 0.601 for the token CE alone and 0.550 for the MUX alone.
   Six orthogonal equal-norm vectors read 0.41; six aligned ones read 1.0. Per-pass cosines
   to the total run from 0.27 to 0.75, and pass 6 carries 36 % of the norm sum while passes
   2-4 carry 10-12 % each.

Reading (3) as the live hypothesis: gradient reaches every pass, and what arrives asks one
weight-shared map to be six different things at once. Wolfe's framing on 2026-09-10 —
"the slot loop fails on gradients in the loop", "the loop issue is credit assignment".

Nothing here claims (3) is the CAUSE. It is equally consistent with the map being
near-inert for a reason upstream of the gradient, in which case near-orthogonal per-pass
updates are what you would expect from six noisy readings of nearly nothing. That
ambiguity is what the two arms are built to separate.

## Proposal

Two one-factor arms on the same ruler, from the two published methods that attack
per-pass credit assignment from opposite directions. Neither changes the loss, the entry,
the target, the depth draw or the recipe.

**Arm 1, `slot-mnext-progressive` (`tul.progressive_p: 0.5`).** Bansal, Schwarzschild et
al. 2022, the Deep Thinking progressive loss. With probability 0.5 a slot of realised depth
`T_i >= 2` draws `k_i` uniform in `[1, T_i - 1]` and its first `k_i` passes have their
OUTPUT detached, so those passes contribute no weight gradient through that slot. The map
is then trained to improve states it did not itself produce, and cannot co-adapt with its
own trajectory or settle at the identity. Asks: is the disagreement a training artefact?

- Implementation: `_tul_core` draws a per-slot `[B, S]` prefix length and applies one
  `torch.where(prefix, h_new.detach(), h_new)` per pass. `k_i <= T_i - 1` keeps every
  slot's LAST pass in the gradient window, so the terminal fixed-point term and the exit
  state always sit on a grad pass. The gain hinge is masked to GRAD passes only — its
  penalty shapes the core weights, and measuring at a detached position would constrain the
  map exactly where the objective was cut away. Zero new parameters, zero extra core
  applications.
- Detaching the pass INPUT as well was implemented, measured and REMOVED: for `t >= 1` it
  is redundant (the previous pass's output detach already makes the carry a constant at
  that slot, and a prefix is a contiguous initial run), and at `t == 0` it would also cut a
  GRAD slot's own read of a neighbouring slot's entry state, which the progressive loss
  never asked to remove. The slots share one sequence; a prefix slot still serves K/V to
  grad slots, and that read belongs to a grad pass.

**Arm 2, `slot-mnext-per-pass-lora` (`tul.pass_lora_rank: 32`).** Bae et al. 2024, Relaxed
Recursive Transformers. The core blocks stay weight-shared and pass `t` adds its own
rank-32 additive branch `y_t = sublayer(x) + B_t (A_t x)`, `B` zero-init. Asks: if the
passes genuinely want to be different maps, does giving them the capacity make the loop
earn depth?

- Implementation: one `PassLoRA` module per core block (`morph/model/mhc.py`), stacked
  parameters `A [T, r, C]` / `B [T, C, r]` per target, indexed by `pass_idx` — an index
  into a parameter, not a Python branch, so `torch.compile` sees no data-dependent control
  flow. `pass_idx` is threaded from `MORPHTransformer._apply_core_step` as `iter_idx`.
- Cost, measured by building both models: 6,291,456 parameters (268,223,095 ->
  274,514,551, +2.35 %) and +1.52 % of a core block's MLP FLOPs per pass.
- Granularity differs from the paper and is named as such: one delta per SUBLAYER (the
  attention branch, the SwiGLU) rather than per linear. It cannot express a delta acting
  inside the SwiGLU nonlinearity or on q/k/v before the score.

Both arms carry a bit-identity contract at their off value, verified against the
pre-change tree rather than by reading the diff: `progressive_p: 0` and
`pass_lora_rank: 0` reproduce loss `9.1006078720092773` and sha256 `32b174ef…2853f8` over
all 208 gradient tensors of the tiny CPU model at master `c429e22`, and the zero-init
deltas reproduce the same numbers with the module BUILT.

### Arm 3, `slot-mnext-mux-every-pass` (`tul.mux_every_pass: true`)

Added 2026-09-10 after arms 1 and 2 read flat. They both left the LOSS alone and changed how
the one exit gradient is delivered. This one changes what the passes are asked for.

**Problem.** The slot loop is scored ONLY at its exit state, through two doors: the token CE
via the coda and the two prefix cells, and the MUX loss computed ONCE on the exit state
(`_forward_tul`). No intermediate pass carries a loss. Every measurement of that shape says
the same thing: the per-pass cotangent is FLAT (0.168 / 0.168 / 0.166 / 0.160 / 0.154 /
0.183 on the ruler — one gradient through a near-identity map), the coda's CE with the exit
equals its CE with the ENTRY to 0.0015 nats on the ruler and 0.0002–0.006 on every arm read
(`slot_z_optimize.py`), and every token K-curve is at or under 0.0006. Wolfe: "this is
likely where our issue actually lives. that one arm needs to be run for sure."

**Proposal.** Put the CONFIGURED MUX target (`mux_target: next`) on the state after EVERY
pass of a LIVE carry — no detach anywhere — for the slots whose realised depth reaches that
pass, plus the ruler's own final term on every valid slot. Average the terms with uniform
weights summing to 1, so `mux_beta` keeps its meaning, and report stats from the FINAL term
so `mux_local` / `mux_rel` stay comparable with every earlier arm.

- Implementation: `_tul_core` keeps the same live-carry trajectory the staged target uses,
  one entry per pass, and RETURNS the per-pass supervision mask beside it
  (`slot_valid & (depths >= j)`, ANDed with `~prefix` when the progressive draw is on). A
  return value, not an attribute: rebuilding the mask in the caller would duplicate the
  conditions (`progressive_p`, `self.training`) that decide it. Off, no trajectory is kept
  and no mask is built. Zero new parameters, zero extra core applications; the cost is `T`
  extra `[B, S, V]` fp32 readouts per step and their backward.
- TRAINING only. An eval forward computes the single final-state term, which is what leaves
  `core_depth_sweep.py`'s forced-depth `mux_local` column identical to the ruler's — the
  arm's own readout must not be changed by the arm.
- Refused where a per-pass term has no meaning: the paid loop, `mux_beta <= 0`, `db_loop`
  (detached carry, the opposite contract), `mux_stage_own_iters` (two stages, two targets),
  `cond_layers` (the think-once stack would read the final term through a stack the per-pass
  terms never see) and SCSE (the carried state is a deviation, not the slot state).
- The honest tension, written down before the run: per-pass supervision toward ONE target is
  a deep-supervision recipe, and deep supervision makes EARLY exits good. That is the same
  direction as depth-independence. The averaged objective is minimised just as well by a map
  whose first pass reaches the target and whose later passes sit near the identity — which is
  the failure already measured. The arm is worth running because it is the first one that
  touches the loss the passes see at all, not because the sign is obvious.

## Alternatives considered

- **DEQ / implicit differentiation** (Bai et al. 2019). Solve for the fixed point and
  differentiate through it with the implicit function theorem, so the backward is a single
  linear solve instead of a product over passes — which would make "which pass gets credit"
  a non-question. Rejected for now on two grounds. First, the probe says the backward
  product is NOT the problem here: the per-pass cotangent is flat, so the thing implicit
  differentiation fixes is not broken. Second, MORPH's loop is a per-sample Poisson depth
  of mean 6 with a MUX loss read at the realised depth, not an iteration to convergence;
  making it a DEQ changes the objective, the depth draw and the stability story at once,
  which is three factors, not one.
- **HRM / TRM deep supervision with detached segments.** Supervise the state after each
  segment and detach between segments. Rejected as an ARM because this tree already ran its
  shape twice: `tul.db_loop` (detached carry, per-iteration local MUX) and
  `tul.mux_stage_own_iters` (staged targets, live carry) are both in the tree, and the
  progressive loss is the strictly more informative version of the same idea — a random cut
  rather than a fixed one, which is what forces the map to handle arbitrary states instead
  of only the segment boundaries it was trained on. Its detached-segment half is a special
  case of `progressive_p` with a deterministic `k`.
- **Universal Transformer timestep embedding.** Give each pass an explicit index signal.
  Already run: `tul.core_stage_cond: "iter"` (arc E2) and the think-once `iter` / `iter_all`
  arms. E2's finding was that per-iteration conditioning earns nothing past iteration 3
  once the loop is stable (`iter-all-r` K3-K6 +0.0001), and the 2026-08-30 cond-zero probe
  found iteration conditioning poisons depth-earning during formation. Per-pass LoRA is the
  same intent with a different lever — a per-pass WEIGHT delta rather than a per-pass
  activation shift — which is why the two are refused together in config
  (`tul.pass_lora_rank` with `tul.core_stage_cond` raises).
- **Coconut-style curriculum** (progressively replacing reasoning steps with latents).
  Rejected: Wolfe has ruled out staged-recipe curricula on this tree as unfriendly recipes,
  the same call that rejected dense-then-ternary warmup. A method whose result depends on a
  schedule nobody can reproduce from the config is not a result.
- **Deep supervision through the CODA at every pass** (the token CE, not the MUX, read at
  each pass). This is the version that would test whether the READER can use an intermediate
  state, which is the closer question to the measured failure. Rejected on cost: the coda is
  three blocks over the full 1,152-position sequence plus a full-vocabulary CE, so a term per
  pass is ~6x the coda's FLOPs and its activations, against ~1 % of a step for the MUX
  readout over 64 slot cells. If the MUX ladder moves anything, this is the follow-up worth
  the price.
- **A token CE per pass on a sub-sample of positions** (say 1/6 of the rows per pass), which
  buys the coda-side signal at the MUX's price. Rejected as a FIRST arm: it changes the
  gradient's variance and its target at once, and the sub-sample size is a tuned number
  nobody has a prior for on this tree. It is the natural fallback if the coda-side question
  survives the MUX ladder.
- **The staged target `tul.mux_stage_own_iters`** (already in the tree): the state after
  iteration k toward the span the slot terminates, the final state toward the next span. Two
  fixed stages, two different targets, and only one intermediate state supervised. It answers
  "does an intermediate state have a job", not "does every pass have one", and its mixed
  targets make `mux_local` a mixture. Refused beside `mux_every_pass` for that reason.
- **The detached DB-loop `tul.db_loop`** (already in the tree): evenly spaced supervised
  iterations with the carry DETACHED, so each term trains exactly ONE core application. That
  is the HRM/TRM shape and it deliberately removes the trajectory the arm is asking about. A
  live carry is the whole point here: a term at pass `j` must reach every earlier pass.
- **Doing nothing on the loop and attacking the state's rank instead.** The levers panel's
  structural finding is that ~50 slot cells sit at effective rank 1.7-4.8 in 1024
  dimensions. This is the real competing explanation and it is deliberately NOT one of
  these two arms: it changes what the loop is fed rather than how the loop is trained, and
  mixing it in would make the credit-assignment question unanswerable. It is the binding
  outcome if both arms read flat.

## Acceptance criteria

Per-arm predictions are frozen in the two planned files and are the binding record. At the
level of this note:

- Each arm's off value is bit-identical to the pre-change tree — loss, base weights and
  every base gradient. Verified, cited above.
- Each arm's mechanism is proven to ACT by a test that fails when the mechanism is removed.
  Done: the progressive output detach and the hinge mask each fail a sabotage run, and the
  pass-LoRA zero-init and index-threading each fail a sabotage run.
- The pass-LoRA parameters are invisible to ternary QAT, the CMS prune, the MORTAR carve
  and the deploy packer, and this is a property of their TYPE (plain `nn.Parameter` on a
  non-Linear module), not of an exclusion list. Two tests hold it.
- Either arm clears `mux_local` K1-K6 > 0.02 at 5,000 steps (the ruler: 0.0067), or the
  arm reports flat and says so.
- The per-pass gradient probe at 5,000 reports the cancellation ratio on the SHARED core
  weights for every arm, against the ruler's 0.520.
- `mux_every_pass: false` is bit-identical to the pre-change tree: loss `9.30903148651123`
  and sha256 `ee9ffa17…7bea2` over all 208 gradient tensors of the tiny CPU model at master
  `f4a284f` and after the change. Three sabotage runs fail the suite (dropping the `~prefix`
  mask, dropping the per-pass terms, detaching the trajectory).
- The every-pass ladder does not change the EVAL readout: an eval forward's `mux_local` is
  equal, tensor for tensor, with the knob on and off. A test holds it.

## Risks

- **Both arms read flat.** This is the modal outcome by the base rate of this tree: every
  forward-side lever since 2026-09-04 has read flat, and these are the first two
  training-side ones. That result is still worth having — it would move the failure past
  the backward and onto the map's near-inertness, and it retires "credit assignment" as
  the working explanation rather than leaving it as an untested intuition.
- **The LoRA arm detonates.** It adds a per-pass degree of freedom to the map's gain while
  the hinge still samples ONE random grad iteration per step, which arc E2 showed is the
  wrong instrument for a per-iteration map. `loop/core_gain_t0` is the leading indicator
  (~500 steps of warning on E7 and E13). Survival is predicted at 70 %, the lowest of the
  two.
- **A CE gain on the LoRA arm gets misread.** +2.35 % parameters usually buys a small CE
  at a fixed step count. That is a capacity reading. Only the K-curves and the cancellation
  ratio speak to depth (Wolfe 2026-09-09: short-horizon CE cannot rank looped against
  unlooped, or one density against another).
- **The deltas absorb the disagreement without changing the shared map.** H-lora-1 could be
  "true" in the sense that the per-pass parameters move and the loop still learns nothing
  new, because the shared weights end up exactly as scattered as before. The cancellation
  ratio measured on the shared weights alone is the instrument that catches this, and it is
  a frozen prediction on that arm.
- **The MUX ladder flattens the curve it is meant to steepen.** Deep supervision toward one
  target rewards a good FIRST pass, which is depth-independence. If `mux_local` at depth 1
  improves and K1-K6 shrinks, the arm has bought early-exit quality and paid in depth. That
  is a real finding and the prereg's second binding clause names it, but it is the outcome
  most likely to be misread as "the arm did nothing".
- **The memory cost is arithmetic, not a measurement.** Each per-pass term retains a 75 MB
  fp32 logit tensor and its log-softmax output at B=6, S=64, V=49169; eight extra terms hold
  on the order of 1.2 GB more on a card the ruler already runs at ~11 GB peak. The smoke's
  peak is the first real number.
- **No arm has run on the GPU.** Everything above is CPU builds, CPU tests and config
  compose checks. `torch.compile` on the new index path and the new mask path is untested
  until the runner's 12-step smoke.
