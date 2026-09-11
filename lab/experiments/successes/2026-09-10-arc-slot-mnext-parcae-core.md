# Planned: the slot loop on a plain Parcae core — is the fault the core or the mechanism?

Status: success

Date: 2026-09-10 (frozen before launch). Arc: `2026-09-04-loop-contribution-arc.md`.
Sits beside the four credit-assignment arms —
[`slot-mnext-progressive`](../failures/2026-09-10-arc-slot-mnext-progressive.md),
[`slot-mnext-per-pass-lora`](../failures/2026-09-10-arc-slot-mnext-per-pass-lora.md),
[`slot-mnext-mux-every-pass`](../failures/2026-09-10-arc-slot-mnext-mux-every-pass.md)
and [`slot-mnext-staged`](2026-09-10-arc-slot-mnext-staged.md) — and asks the one question
none of them can: those four all varied the LOSS. This one varies the CORE.
Design note:
[`2026-09-10-tul-on-a-parcae-core.md`](../../../.agents/notes/proposed/architecture/2026-09-10-tul-on-a-parcae-core.md).
Evidence it is built on:
[`slot-geometry-audit`](../results/2026-09-10-slot-geometry-audit/README.md),
[`slot-z-optimize`](../results/2026-09-10-slot-z-optimize/README.md),
[`lab/toy_slot_loop/WRITEUP.md`](../../toy_slot_loop/WRITEUP.md),
[E20](../failures/2026-09-09-arc-e20-loop-depth-candidates.md),
[the per-pass-strength panel](../successes/2026-09-09-arc-per-pass-strength.md).

## Question

The slot loop reads flat on every arm. The 2026-09-10 geometry audit closed the obvious
forward explanation: the core is NOT muted on the compact sequence — one pass moves a slot
state 0.65 / 2.94 / 247x its own norm against 1.04 / 2.78 / 161x for the same weights on a
token sequence, and the prefix write hands the coda a cell that is 1.07x, 6.8x and 802x the
loop's own work. What it could not ask is whether a DIFFERENT core, on the SAME mechanism,
would earn depth where this one does not.

Two things are on the table and today's instruments cannot separate them.

* **H-core.** MORPH's core at the SLOT shape is the fault. That core is ternary-STE weights
  (a rule already measured to be the per-pass map's limiter on the plain loop: K1−K6 0.033
  under absmean, 0.185 under norm_match, 0.168 with a bf16 core — the per-pass-strength
  panel and E20's `depthcand-dense-core`), CCA-compressed attention whose HCA compressed
  branch is EXACTLY DEAD at 64 cells while the gate still spends 0.42-0.48 of its mixture on
  the zero tensor (audit finding F1), CSA block selection that never fires (`tk` 8 of 8),
  XSA self-exclusion that leaves query 0 with an all `-inf` softmax row, a 4-stream Cayley
  hyper-connection carrier, and a diagonal carry on 256 of 1024 dims. Not one of those was
  designed for a 64-cell weight-shared recurrence.
* **H-mech.** The TUL mechanism or its target is the fault, and ANY core would read flat.
  The toy study's permissive-geometry cell is the concrete version of this: with MORPH's own
  geometry the prelude and the coda compose across spans by themselves, so one core pass
  reaches 0.283 nats on a task that needs iteration and the loop has nothing left to earn.

Does the slot loop earn depth when the looped core is a plain Parcae-style block stack?

## Hypothesis

Two branches, and the arm is built so that exactly one of them fires.

**H-core**: with the ternary snap off the core, a dense causal attention that actually sees
all 64 cells, and a plain single-stream residual, the per-pass map becomes strong enough
that the slot loop's later passes do something the first pass could not. Reads as a token
K1−K6 that leaves the ruler's +0.0001 and a `mux_local` K-curve that keeps rising past pass
3. If this fires, MORPH's core at the slot shape is the fault and the next question is
WHICH feature of it.

**H-mech**: the swap changes the map completely and the K-curve does not move. The loop
still reaches its neighbourhood in one pass, the coda's CE is still flat along the
directions the loop moves, and the passes' updates still cancel. If this fires, the core is
exonerated — five very different maps (ternary MORPH, bf16-core MORPH, a noise entry, a
7.3x-expansive free map, and now a plain dense transformer) all read flat, and the lane is
the target and the geometry, not the block.

## Method

`tul_slot_mnext_parcae_core` = the ruler `tul_slot_mux_norm_match` (`slot-mux-norm-match`)
with ONE factor changed: `model.core_impl: parcae`.

What that one knob swaps, all inside `morph/model/parcae_core.py` and selected at BUILD
time (no runtime branch in the forward):

| ruler's core block | this arm's core block |
|---|---|
| CCA + CSA/HCA + XSA + residual attention + conv + indexer + QK-Norm | one dense causal softmax over all 64 slot cells, 8 heads x d_head 64, the tree's own CoPE rotary, query 0 attends itself |
| `_SwiGLUMortar` (`CMSBlockLinear`), ternary STE under `norm_match` | plain dense `_SwiGLU` of `nn.Linear`, bf16, no ternary parametrization |
| 4-stream Cayley hyper-connection residual | plain additive pre-norm residual, single stream |
| prunable / carvable / routable / packable | invisible to all four walks by construction |

The carrier boundary: `_apply_core_step` collapses the `[B, S, n, C]` carrier by the stream
MEAN at the core's entry and broadcasts the pass output back over the n streams at its exit.
That is the same reduction `_readout` and `TULSlots.unpack` already use to read this carrier,
and it keeps the PRELUDE AND CODA as untouched MORPH blocks with their Cayley carrier — which
is what makes this a core swap rather than a different model.

**The swap's second component, named because it is not free.** Parcae's recurrence is
`h_t = A h_{t-1} + B e + R(h_{t-1}, e)` (arXiv 2604.12946 §3, App. C): `A`/`B` is MORPH's
`DiagonalInjection` and `R` is the block stack. A Parcae core under MORPH's ctx-only carry
would leave 768 of 1024 carrier dims with no source term at all, so this arm carries the
widened carry with a learned identity-init `B` — `injection_channels: all`,
`injection_all_decay: 0.447`, `injection_all_dt: 0.8`, `injection_B: true`, the same four
values `slot-mnext-noise-entry` carries. The loop ENTRY stays the ruler's
(`core_state_init: prelude`), per the knob. The carry alone is already measured INERT on this
family — `slot-unpack-noise-entry` K1−K6 0.002, `slot-mnext-noise-entry` K1−K6 −0.0001 — so
it is not a plausible cause of a positive result; if H-core fires anyway, the first follow-up
separates the two components before anything else.

Everything else is the ruler's, unchanged: slot layout, boundary rule, packer, prefix_k 2,
max_slots 64, M-next MUX at beta 1 through the tied head, `emit_weight` 0, token-state
dropout 0.15, the Poisson depth draw (mean 6, max 8), full BPTT, the gain hinge lambda 100 at
target 0.9, `slot_cot_clip` 4.0, `core_fixed_point_lambda` 1.0, `ternary_scale_mode:
norm_match` on prelude / coda / embeddings, seq 1024, batch 6, seed 1, ramp 1,000, 5,000
steps, prune / carve / route off.

Runner `arc/run_slotloop3.sh` (file-driven queue, commit pinned in `arc/slotloop3_arms.txt`),
a 12-step smoke first, the draw under the sustained tripwire
(`lab/divergence/tripwire_sustained.py`). Rate rule: tok/s below 8,086 at step 200 skips the
arm and the queue continues (the ruler read 12,429). Same launch line as every other slot arm
in this queue, `tg_scoped_kernels` included.

Readout, identical to the other arms so the numbers compare: `core_depth_sweep.py` at
checkpoints 2,500 and 5,000, depths 1,2,3,6,9,12,16, both the token K-curve and the
`mux_local` forecast K-curve on 480 rows; `worth_profile.py` and `slot_state_probe.py` at
5,000 (the runner's `slot` kind does both); `slot_gradient_probe.py` and `slot_z_optimize.py`
at 5,000 by hand — the gradient probe carries the cancellation ratio and the per-pass
cotangent/cosine profile, the z-optimisation probe carries the coda's exit-minus-entry CE.
`slot_anatomy.py` at 5,000 for the per-block branch ratios, which is the instrument that says
whether the swapped block is a STRONGER per-pass map than the ternary CCA one (the
per-pass-strength panel's `core MLP branch out/in` reading: 0.2 under absmean, 0.86-1.10 under
norm_match on the plain loop; the slot loop's own audit read 2-20 % on slot states).

Parameter counts, measured at the real shape on this tree (CPU build, both configs):

| | ruler | this arm |
|---|---|---|
| whole model | 268,223,095 | 265,734,591 |
| core (6 shared blocks) | 68,038,200 (25.4 %) | 64,499,712 (24.3 %) |
| per core block | 11,446,836 | 10,749,952 |
| — attention | 2,400,724 | 2,097,152 |
| — MLP | 8,650,752 | 8,650,752 (identical) |
| — residual / other | 395,360 (HC Cayley) | 2,048 (two RMSNorms) |
| diagonal carry | 640 | 1,050,624 (the `[1024, 1024]` B) |

The core is 5.2 % SMALLER than the one it replaces and the MLP is identical, so a depth gain
cannot be read as extra core capacity. The whole model is 0.9 % smaller.

Cost, arithmetic not measurement: the MLP FLOPs are identical; the Parcae attention is
2.10 M params against 2.40 M and runs ONE softmax instead of a window branch, a compressed
branch and a gate; the HC-Cayley mixer's 6 blocks x 2 residuals x ~6 passes are gone; the new
B matmul adds 2 x 64 x 1024^2 per pass per row (~0.8 GFLOP per row per forward, under 0.5 % of
the step). Core FLOPs are roughly a fifth of a forward, so the arithmetic says at or slightly
below the ruler's cost, with the kernel mix (eager SDPA and cuBLAS in place of fused Triton
window + fused HC) the only reason it could go the other way.

**Amendment 2026-09-10 18:18 (build failure, not a result).** The runner's 12-step smoke at `b94bd02` died at build: `CoreSpectralPenalty found 0 core MLP linears` (`morph/training/spectral_penalty.py::collect_core_linears` knew only `MortarLinear`; the Parcae MLP is dense `nn.Linear`). Fixed in `5790f09` (the enumerator also takes the plain-linear direct children of `blk.mlp`, tested), plus `15838c7` (the morph gradient-hash pin now runs under one CPU thread; it depended on the thread count). Re-queued at `15838c7`. No prediction changed; no draw started.

## Predictions (frozen)

The ruler's numbers, cited once, all at step 5,000: 480-row CE at depth 6 **4.3290**; token
K1−K6 **+0.0001** [−0.0000, +0.0002], K3−K6 −0.0000; `mux_local` K1−K6 **+0.0067**
[+0.0053, +0.0081], K3−K6 +0.0005; wall clock **52 min 38 s**; combined cancellation ratio
**0.520**; per-pass cotangent share 0.168/0.168/0.166/0.160/0.154/0.183; z-optimisation
`ce_loop` 4.1173 with the loop ENTRY worth **+0.0015**.

- **P-a (survival).** HEALTHY to 5,000, no sustained tripwire (`preclip/total > 1e4` at step
  >= 200): **72 %**. Reasoning: the ramp plus the fixed-point term hold every slot arm drawn
  since 2026-09-04, and the hinge is well defined here (the entry is still `h_0 = e`). The
  28 % is for a map nobody has run: a bf16 dense core has no ternary snap damping its
  per-pass gain, the widened carry detonated 2 of 2 draws under `warmup: 0` in E9, and the
  scale mode (`loop/core_gain_t0`) is the failure the fixed-point term is blind to.
- **P-b (rate).** tok/s at step 200 above the 8,086 floor: **90 %**. Reasoning: fewer core
  params, one attention call instead of three branches and a gate, no HC mixer in the core.
  The 10 % is the kernel mix — the fused Triton window and the fused HC kernel both stop
  running in the core and eager SDPA takes over.
- **P-c (tokens, the shallow bar).** Token K1−K6 at 5,000 above **0.03** (the plain
  prelude-entry ruler `plain-panel-norm-match`): **30 %**. Above **0.10** (the plain
  noise-entry value, and the neighbourhood of E20's bf16-core 0.168): **12 %**. Reasoning:
  the ONE thing measured to move a K-curve on this tree is the core's per-pass strength
  under a PLAIN loop — norm_match 0.033 → 0.185, bf16 core 0.168 — and this arm gives the
  slot loop the strongest per-pass map it has ever had. Against that: every slot arm to date
  has read K1−K6 in [−0.0000, +0.0006] whatever the map, including a free map that grows the
  state 7.3x per loop, and the audit says the coda's CE is flat along the directions the loop
  moves rather than starved of movement. 30 % is the honest weight on "the map was the
  binding constraint after all".
- **P-d (forecast K-curve, the real bar).** `mux_local` K1−K6 above **0.02** at 5,000 (ruler
  0.0067): **32 %**. K3−K6 above **0.002** (ruler +0.0005): **20 %**. Reasoning: the MUX
  target is the one loss that reads the loop state directly, so it is where a stronger map
  should show first, and it is already the only slot instrument with a nonzero reading. K3−K6
  is set lower than K1−K6 because "the loop is done by pass 3" has survived every intervention
  including a mean-12 draw, a fixed depth and per-pass parameters.
- **P-e (the reader).** The coda's exit-minus-entry CE from `slot_z_optimize.py` above **0.02**
  nats (ruler +0.0015; every arm read so far 0.0002-0.107, and only the free-map unpack arm
  above 0.02): **25 %**. Reasoning: this is the quantity the mechanism has to move for any of
  the rest to matter, and a dense core with a widened carry moves z further than the ruler's
  does. It is not higher because `slot-unpack-free` already moved z by 7.3x its entry norm and
  bought 0.107 nats — 4 % of what a gradient-fitted z buys the same frozen coda.
- **P-f (cancellation).** Combined cancellation ratio on the shared core weights BELOW the
  ruler's **0.520**: **35 %**. Reasoning: the toy study found a −0.604 correlation between
  cancellation and K1−K6 across its grid, and a different map with different per-pass
  Jacobians has no particular reason to agree with itself pass to pass. Under 50 % because
  every MORPH slot arm measured so far has landed in 0.52-0.77 regardless of what changed.
- **P-g (val CE).** 480-row CE at depth 6 within **0.10** of the ruler's 4.3290: **55 %**.
  Reasoning: recorded, NOT a verdict (Wolfe 2026-09-09: at 5,000 steps a final loss cannot
  rank looped against unlooped or one precision against another). The band is 0.10 rather
  than the staged arm's 0.05 because a bf16 dense core is a genuinely different model, not a
  loss-attachment change; E20's bf16-core arm moved CE by 0.08 at 5k on the plain loop.
- **P-h (cost).** Wall clock within **1.2x** of the ruler's 52 min 38 s: **85 %**. Reasoning:
  the FLOP arithmetic above says at or below 1.0x. The 15 % is entirely the kernel mix and
  the extra `[B, S, n, C]` materialisation the stream broadcast makes once per pass.

## Binding

- **P-c TRUE or P-d TRUE (H-core fires).** MORPH's core at the slot shape is the fault. The
  next arm does NOT chase the loss: it ports ONE MORPH core feature at a time back onto the
  Parcae core and reads the K-curve after each, in this order — (1) ternary STE on the core
  under `norm_match` (the feature with the largest measured effect on a plain loop's
  K-curve), (2) the 4-stream Cayley carrier, (3) CCA + CSA/HCA attention (with
  `core_hca_compress_ratio: 16` so F1 is not what is being measured). The FIRST feature that
  collapses the K-curve is the answer. Before any of that, one control separates this arm's
  two components: the same Parcae core with the ruler's ctx-only carry.
- **P-c and P-d FALSE, P-e TRUE.** The exit state moved and the K-curve did not. That is the
  audit's "the coda's loss is flat along the directions the loop moves" measured on a fifth
  map; read `slot_z_optimize`'s fitted-z gap and the worth profile before choosing, and treat
  the reader (the coda's access to z) as the lane rather than the core.
- **ALL FLAT (P-c, P-d, P-e, P-f all FALSE, curves inside the ruler's CIs) — H-mech.** The
  core is exonerated. Five maps that differ in weight precision, attention, residual
  topology, carry width and per-pass gain all leave the slot loop's contribution inside a
  0.002-nat band, which is a statement about the mechanism, not the block. The lane is then
  the toy study's two survivors, in this order: STAGED targets (`slot-mnext-staged`, already
  queued) and the STRICT geometry — closing the prelude's and the coda's cross-span routes so
  the loop is the only path, which on the toy moved one core pass from 0.283 nats to 1.281
  and is the only condition under which any attachment earned. No further core arm is queued
  under this branch.
- **P-a FALSE (detonation).** Read `loop/core_gain_t0` and the hinge's binding fraction
  first. If the hinge was binding on most steps the arm was confounded and is re-drawn at
  `slot_gain_lambda: 0`; if the scale mode fired with a passing hinge it is E7's mode on a
  new map and goes in the divergence README, not in this arc's K-curve table.

## Not verified before launch

The arm has NOT run on a GPU, and no number in this file except the parameter counts and the
ruler's own history is measured. Specifically:

* **The hinge's operating point on a dense bf16 core is unmeasured.** The ruler's hinge binds
  on under 1 % of steps at a ternary map that reads 0.87-0.91; a dense core with no ternary
  snap has no measured typical gain at all. On a tiny CPU model at init this arm reads
  `gain_est` 0.277 with the penalty at exactly 0.0, which says the machinery runs and says
  NOTHING about step 3,000. If `loss/gain_est` sits above 0.9 for most of the run the arm is
  a hinge fight, not a core swap — that is P-a's failure branch above, and it must be checked
  at step 200 rather than after the fact.
* **The two components of the swap are confounded by construction.** The block stack and the
  widened learned-B carry change together. The carry alone is measured inert twice on this
  family, which is why the arm is drawn this way, but a positive result needs the ctx-only
  control named in the Binding before it is believed.
* **`torch.compile` on the Parcae core at the real shape is untested** beyond the runner's
  12-step smoke. The block is plain `F.linear` + `F.scaled_dot_product_attention`, which the
  rest of the tree does not use in the core, and the `.contiguous()` stream broadcast is a new
  op inside a checkpointed region.
* **Memory at the real shape is arithmetic, not measurement.** The single-stream core should
  cut core activations by roughly 4x and the broadcast adds one `[6, 64, 4, 1024]` bf16
  tensor per pass; the smoke's peak is the first real number. The 5090 has ~1 GB of slack on
  the ruler's recipe.
* **Nothing was run against a checkpoint.** `core_depth_sweep.py`, `slot_state_probe.py`,
  `worth_profile.py` and `slot_z_optimize.py` were confirmed to have NO coupling to the core's
  module structure (they drive the forward, and `slot_z_optimize` hooks only `core_init` and
  `core[0]`, both present); `slot_gradient_probe.py`'s taps and its per-pass self-check were
  REHEARSED on a tiny CPU `core_impl: parcae` slot-loop model — 16 of 16 core weights tapped,
  max relative error 7.1e-8, zero weights without a gradient, cotangents recorded at every
  pass — and `slot_anatomy.py`'s hook sites (`blk.attention`, `blk.mlp`, the block, and
  `core_init`) all resolve, with its `_valid_rows` / `_valid_full` helpers already handling a
  3-D carrier. None of that is the same as running any probe on a real GPU checkpoint.
* **`slot_anatomy`'s `*_full` columns degenerate on this arm.** They report "every stream, no
  averaging"; a single-stream core has one, so `slot/token` stream-mean-survival readings from
  the audit have no counterpart here. Not a defect, but do not compare that column across arms.
* **The dead-branch defect F1 is removed rather than fixed.** This arm cannot be read as
  evidence about `core_hca_compress_ratio: 16` on a MORPH core; that repair is still not run.
* **`use_kernels` and `tg_scoped_kernels` do nothing inside the Parcae core**, because it
  calls no fused Triton kernel. A kernels-on / kernels-off comparison on this arm is a
  prelude/coda comparison.

## Results

Filed 2026-09-11 03:51. `slot-mnext-parcae-core` (commit `15838c7`; `model.core_impl: parcae`, six plain
pre-norm blocks with dense causal softmax attention, SwiGLU d_ff 2816, single-stream residual,
64.50M core params in bf16 and never ternarised, under the ruler's slot layout, M-next MUX,
prelude entry, hinge and fixed-point term): HEALTHY to 4,999, tripwire max 115 at step 3255
(ruler 78.7), **23,122 tok/s** at step 200 (ruler 12,429, 1.86x), smoke peak 10.54 GB,
training peak 11.65 GB, wall clock **33 min 05 s** (ruler 52 min 38 s, 0.63x). Runner final
val_loss 4.3338 (ruler 4.3775). Over steps 4,900 to 4,999: `gain_est` **0.222** (ruler 0.87;
the hinge at target 0.9 bound on 0 of 5,000 steps), `loop/delta_ratio_last` 0.055 (ruler
0.065), `loop/core_gain_t0` 4.4, fixed-point term 0.016. The first build attempt died at
construction (`CoreSpectralPenalty found 0 core MLP linears`, fixed in 5790f09 and hash-pinned
in 15838c7; Method amendment above).

Sweeps, 480 rows, forced depths 1/2/3/6/9/12/16:

| step | tokens K1−K6 | tokens K3−K6 | forecast K1−K6 | forecast K3−K6 | forecast @1 / @6 (ruler @6) | CE@6 (ruler) |
| --- | --- | --- | --- | --- | --- | --- |
| 2,500 | −0.0001 [−0.0001, −0.0000] | −0.0001 | −0.0009 [−0.0015, −0.0003] | −0.0008 | 6.8668 / 6.8677 (6.8766) | 4.7166 (4.7194) |
| 5,000 | **+0.0001 [+0.0000, +0.0002]** | −0.0000 [−0.0001, +0.0000] | **+0.0019 [+0.0012, +0.0027]** | −0.0004 [−0.0005, −0.0002] | 6.7686 / 6.7668 (6.7724) | 4.2940 (4.3290) |

Token K1−K6 is the ruler's number to four decimals. The forecast K-curve is SMALLER than the
ruler's (+0.0019 vs +0.0067) with K3−K6 negative. CE is 0.035 better than the ruler at depth 6
and 0.044 better at the runner's val, at 0.63x the wall clock (a horizon reading; not a
verdict on the core).

Worth profile at 5,000 (offsets 0..6): zero +0.103 [+0.094, +0.115], +0.050, +0.032, +0.022,
+0.017, +0.012, +0.009 (ruler +0.094 .. +0.006); shuffle +0.051, +0.054, +0.039, +0.035,
+0.021, +0.013, +0.005; wrong_seed +0.037, +0.040, +0.029, +0.024, +0.017, +0.013, +0.010.
State probe: |h| 297 at depth 1 → 372 → 403 → 425 at 6 → 427 at 16; relative distance from
depth 1 0.267 / 0.378 / 0.460 / 0.469; cos 0.998 / 0.996 / 0.994 / 0.994; per-pass relative
step 0.267 / 0.089 / 0.062 / 0.008. The Parcae map converges to a fixed point by pass 6 and
sits there through pass 16, which is what a typical gain of 0.22 says it should do.

Gradient probe at 5,000 (12 rows, batch 2, depth 6, eager; 48 tapped core weights; self-checks
PASS at max rel err 1.4e-7): combined cancellation ratio **0.915** (ruler 0.520; every MORPH
slot arm 0.34 to 0.77); MUX 0.913, token CE 0.926. Per-pass share of the shared core weight
gradient 0.028 / 0.040 / 0.068 / 0.122 / 0.232 / **0.510**; per-pass cosine to the total
+0.51 / +0.67 / +0.79 / +0.90 / +0.97 / +0.95. On a contractive map the passes AGREE: the
gradient is a geometric series back from the exit and every pass points the same way. This
is the cleanest credit-assignment picture any slot arm has produced, and the K-curve is flat
under it. Parameter-group norms: prelude 2.74, coda 2.34, core.attention 0.71, core.mlp
0.64, injection 0.48 (14x the ruler-family's 0.03 to 0.05: the widened carry is used).
z-optimisation probe (12 rows, batch 2, 200 Adam steps, bit-exact): ce_loop 4.0769, entry
**+0.0060** (ruler +0.0015), zero +0.0183, shuffle +0.0145, fitted z −0.534 with cos(z*,
z_loop) +0.978 (norm 425; the oracle is norm-limited at 200 steps, so its size is not
comparable to the ruler's −0.944), loop z rank 10.0.

Probes ran 03:42 to 03:49 after the gradpass arm's rate check, 13.8 GB free at start.
Artifacts: `lab/experiments/results/2026-09-10-slot-mnext-parcae-core/`; npz under
`ignored/experiment-artifacts/2026-09-10-slot-mnext-parcae-core/`; probe JSON under
`ignored/experiment-artifacts/2026-09-10-slot-gradient-probe/` and `-slot-z-optimize/`.

## Verdict

P-a TRUE. P-b TRUE (23,122). P-c FALSE at both bars as predicted (+0.0001). P-d FALSE at both
bars as predicted (+0.0019; K3−K6 −0.0004). P-e FALSE as predicted (+0.0060). P-f FALSE as
predicted, and by a margin no one set a bar for (0.915 against 0.520). P-g TRUE (0.035
better). P-h TRUE (0.63x). Every prediction held on its majority side, so the file goes under
successes. **Binding clause 3 fires, ALL FLAT, H-mech: the core is exonerated.** Five maps
that differ in weight precision (ternary / bf16), attention (CCA+CSA+HCA / dense softmax),
residual topology (4-stream Cayley / single stream), carry width (ctx channel / all 768 dims),
per-pass gain (0.87 / 0.22) and now credit structure (cancellation 0.52 / 0.92) all leave the
slot loop's token contribution inside [−0.0001, +0.0006] and its forecast contribution inside
[0.002, 0.015]. That is a statement about the mechanism, not the block.

The clause's named lane is already run in this batch, and both survivors are closed: the
STAGED target (`successes/2026-09-10-arc-slot-mnext-staged.md`: signatures transfer, exit
equals the ruler) and the STRICT geometry (`failures/2026-09-10-arc-slot-mux-mask-norm-match.md`
tokens +0.0009, `failures/2026-09-10-arc-slot-loop-mask-norm-match.md` −0.0001). Wolfe's
question (2026-09-10, "TUL on Parcae directly") has its answer: the slot loop behaves the same
on the reference looped transformer as on MORPH's core, and it does so with a better CE and
half the wall clock.

## Updated hypothesis

The slot loop's flatness is a property of what the slot is asked for and what it is given,
not of the map that iterates it. On a contractive Parcae core the loop reaches its fixed point
by pass 6, the passes' gradients agree at 0.92, and the exit is worth 0.006 nats over the
entry to the coda; on MORPH's core the same numbers are 0.52 and 0.0015. Both loops deliver
what one pass makes because the M-next target through the tied head is a one-pass job and the
slot's input sits at rank 10 to 13. The remaining arms in this batch are on that side: the
slot's INPUT (`slot-mux-bagmean-norm-match`) and its WRITE WIDTH (`slot-mux-prefix4-norm-match`).
After them the lane is a per-slot target that needs more than one pass and that the coda can
read, designed from what the coda needs at the tokens, and it should be developed on the
Parcae core, which is now the cheaper and better-behaved testbed for the mechanism.
