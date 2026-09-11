# Planned: the mask on the ruler, to find out what killed the mask's token dependence

Status: failure

Date: 2026-09-10 (frozen before launch). Arc: `2026-09-04-loop-contribution-arc.md`.
One factor against the ruler `slot-mux-norm-match`
(`morph/configs/tul_slot_mux_norm_match.yaml`) and one factor against
[`slot-mnext-staged-mask`](../failures/2026-09-10-arc-slot-mnext-staged-mask.md), which
ran today. Design note:
[`2026-09-10-credit-assignment-in-the-slot-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-10-credit-assignment-in-the-slot-loop.md).
Source of the geometry claim: question 4 of
[`lab/toy_slot_loop/WRITEUP.md`](../../toy_slot_loop/WRITEUP.md).

## Question

For six arms the mask was the ONE lever on this tree that bought a token K-curve. Under
`tg_restrict` at scope "all" the prelude and the coda are masked to same-span-or-slot, the
slot is the only route from an earlier span to a later token, and every arm that ran it
read a token K1-K6 between 0.020 and 0.060: E4 `to-mnext-y2-mask` 0.0204, E13
`m12-mnext-mask` 0.033 and 0.049, E18 `mask-k2/k4/k8` 0.024 to 0.049, E16 0.406 on clean
math. Every one of those ran under the OLD ternary rule, `absmean`.

Today `slot-mnext-staged-mask` read **0.0033**. The mask's forced token dependence was
gone. That arm changed two things at once against the record: the ternary rule is now
`norm_match` (shipped 2026-09-09) and the loss carries the staged own-span term at pass 3.
Its own filing says so in as many words: no plain-mask arm exists under `norm_match`, so
whether the rule or the loss removed the dependence is one arm away.

This is that arm. The ruler plus the mask block, nothing else. If the token curve comes
back, the staged loss removed it. If it stays near zero, the ternary rule did, and the
mask's whole six-arm lineage has to be re-read as an artefact of `absmean`.

The question matters beyond bookkeeping. The mask's K-curve is the only evidence on this
tree that the tokens ever read the slot's depth. If that evidence belongs to a ternary rule
we no longer ship, the arc has no surviving demonstration of token-side depth dependence at
all.

## Hypothesis

H-rule: the ternary rule removed it. Under `absmean` the looped core was starved (branch
ratios K1-K6 0.033 against 0.185 under `norm_match`; the carved MLP branch out/in 0.2
against 0.9), so the map's per-pass contribution was small and NOISY, and the tokens, given
no other route, had to track whatever the slot's later passes happened to add. Under
`norm_match` the same passes are better conditioned and ONE pass already produces the exit
state the coda wants, so forcing the route buys nothing extra with depth. Prediction: token
K1-K6 near 0.003, the staged-mask value, and the staged term is exonerated.

H-loss: the staged own-span term removed it. Supervising pass 3 toward the span the slot
terminates makes the middle of the trajectory a memory state, and passes 4 to 6 then spend
themselves converting it back (own-span CE 3.24 at pass 3, next-span 8.64 at pass 3 falling
to 6.79 at pass 6). A token reading the exit sees a state that ROUND-TRIPPED rather than
one that accumulated, so depth stops helping it. Prediction: token K1-K6 back at 0.02 or
above, and the ternary rule is exonerated.

The two are not exhaustive. Both could be partly true, and a value between 0.008 and 0.015
is the outcome that says so.

## Method

`tul_slot_mux_mask_norm_match` = `tul_slot_mux_norm_match` (the ruler) plus
`tul.tg_restrict: true` at the default `tg_restrict_scope: all`, with
`model.use_kernels: false` (forced by the restriction at construction) and
`model.tg_scoped_kernels: true`. Nothing else changes: M-next MUX at beta 1 through the
tied head, exit-only attachment, prelude entry, hinge lambda 100 at target 0.9,
`core_fixed_point_lambda` 1.0, `slot_cot_clip` 4.0, `ternary_scale_mode: norm_match`,
seq 1024, batch 6, seed 1, 5,000 steps, ramp 1,000, checkpoints at 2,500 and 5,000.

Scope. "all" masks the PRELUDE and the CODA, which is the geometry every mask arm in the
lineage ran and the geometry the toy's question 4 is about. Scope "coda" (the
`slot-unpack-*` family) leaves the prelude global on purpose and is the wrong control.

Kernels, and why they are not a free choice. `tg_restrict` forces `model.use_kernels:
false` at construction, which turns every fused kernel off process-wide.
`model.tg_scoped_kernels: true` is legal at scope "all" because every TG-restricted branch
stays eager by construction (a prelude or coda window call always carries `tg_allow` and
routes to the reference path, and the TG compressed branches are pure eager functions), and
it lets HC-Cayley, the CCA prologue and the core-region window run fused. Two reasons it is
required. Rate: E4's fully eager twin read 9,168 tok/s at step 200 against the floor of
8,086, and today's scoped-kernel mask arm read 9,349. The hinge: the eager finite
difference reads gain 0.94 +- 0.014 on a map whose true gain is 0.87 (noise bias at
`slot_gain_eps` 0.02), so a fully eager arm's hinge at target 0.9 fires on noise while the
ruler's, running fused, does not
(`.agents/notes/proposed/bug-fix/2026-09-08-slot-gain-eps-noise-bias.md`).

Side effect, named because it is a confound and not a choice: under `tg_restrict` the
pooled `GatedPoolCompressor` is not built on any layer (`_CCAHCAAttention.__init__` sets
`self.compressor = None`), so the HCA compressed branch that reads exactly 0.000 at S = 64
on the ruler cannot exist here. This arm therefore changes the geometry AND removes audit
finding F1. `slot-mux-hca-fix` is the arm that removes F1 alone, and it is parked.

Which comparisons are one-factor, stated so the readout is not over-read. Clean pairs: the
ruler (the mask on or off) and `slot-mnext-staged-mask` (the staged term on or off). E4's
`to-mnext-y2-mask` is the same family under the old rule but ran fully eager, so against
E4 this arm differs by the ternary rule AND by the hinge's noise bias. E4's +0.0209 is a
lineage number here, not a paired control.

Runner `arc/run_slotloop3.sh` (file-driven queue, commit pinned in `arc/slotloop3_arms.txt`),
a 12-step smoke first, the draw under the sustained tripwire
(`lab/divergence/tripwire_sustained.py`). Rate rule: tok/s below 8,086 at step 200 skips the
arm and the queue continues.

Readout: `core_depth_sweep.py` at 2,500 and 5,000, depths 1,2,3,6,9,12,16, giving the token
K-curve and the forecast column `mux_local`; `worth_profile.py` and `slot_state_probe.py` at
5,000 (the runner's `slot` kind does both); `slot_gradient_probe.py` and
`slot_z_optimize.py` at 5,000 by hand. No new wandb keys.

The numbers this arm is read against, cited once. RULER `slot-mux-norm-match` at 5,000:
480-row CE at depth 6 **4.3290**; token K1-K6 +0.0001 [-0.0000, +0.0002], K3-K6 -0.0000;
`mux_local` K1-K6 +0.0067 [+0.0053, +0.0081], K3-K6 +0.0005; forecast at depth 1 / 6 =
6.7791 / 6.7724; worth zero at offset 0 +0.094; z-optimisation `ce_loop` 4.1173 with the
loop ENTRY worth +0.0015; combined cancellation ratio 0.520; per-pass cotangent share
0.168 / 0.168 / 0.166 / 0.160 / 0.154 / 0.183 and per-pass cosine to the total 0.33 / 0.27 /
0.37 / 0.63 / 0.75 / 0.67; fixed-point term 0.00302; 12,429 tok/s at step 200; wall clock
52 min 38 s; tripwire max 78.7; runner final val_loss 4.3775. TODAY'S MASK ARM
`slot-mnext-staged-mask` at 5,000: token K1-K6 +0.0033 [+0.0028, +0.0037], K3-K6 +0.0022;
forecast K1-K6 +0.0604; CE at depth 6 **4.4373**; worth zero +0.336; z-opt entry +0.0235;
cancellation 0.385 with two negative-cosine passes; 9,349 tok/s; wall clock 51 min 07 s.
LINEAGE, all under `absmean`: E4 `to-mnext-y2-mask` token K1-K6 +0.0209 [+0.0199, +0.0219],
forecast +0.187, CE at depth 6 4.336 against its unmasked ruler's 4.205, worth zero 0.213
and shuffle 0.310 at offset 0, 9,168 tok/s.

## Predictions (frozen)

- **P-a (survival AND rate).** HEALTHY to 5,000 with no sustained tripwire AND tok/s at step
  200 at or above 8,086: **85 %**. Today's twin cleared both comfortably at the same kernel
  settings and one extra loss term; this arm has one term FEWER. The 15 % is the detonation
  base rate plus the chance the unmasked ruler's slightly different step shape lands under
  the floor, which nothing suggests.
- **P-b (the arm's whole point: tokens).** Token K1-K6 at 5,000 above 0.010: **45 %**.
  Above 0.020, which is where the `absmean` lineage sat: **30 %**. Below 0.006, the
  staged-mask value: **45 %**. These are deliberately close to even because the arm is built
  on a genuine two-way ambiguity; anything sharper here would be theatre. The slight tilt
  toward the rule as the cause comes from the size of the measured ternary effect on the
  plain loop (K1-K6 0.033 against 0.185 between entries, and the carved-branch ratios moving
  4x), against a staged term whose measured effect on the UNMASKED token curve was +0.0019.
- **P-c (forecast).** `mux_local` K1-K6 above 0.10 at 5,000 (E4 under `absmean` read
  +0.187; the ruler reads +0.0067): **35 %**. Above 0.02: **70 %**. The mask has always
  moved the forecast curve far more than the token curve, and today's staged-mask arm read
  +0.060 with the SAME exit value as the ruler, so a large K1-K6 here is mostly a statement
  about depth 1 being worse, not depth 6 being better. Scored, and read with the end point.
- **P-d (the reader).** The coda's exit-minus-entry CE from `slot_z_optimize.py` above 0.02
  nats (the ruler +0.0015; today's masked arm +0.0235): **65 %**. The highest non-survival
  probability in this file and close to mechanical: with the coda masked, the exit state
  carries information no other path supplies. If a mask arm reads low here the probe itself
  is suspect.
- **P-e (worth).** `worth_profile.py` zero at offset 0 above 0.20 nats (the ruler +0.094;
  today's masked arm +0.336; E4 0.213): **75 %**. The mask makes the slot load-bearing
  whatever the loop does with it, and two mask arms under two different ternary rules have
  already read above 0.20.
- **P-f (cancellation).** Combined cancellation ratio BELOW the ruler's 0.520: **60 %**.
  Today's masked arm read 0.385 and the unmasked staged arm 0.442; the mask routes the token
  CE through the loop for the first time, and that is the change most likely to make the
  passes disagree. The 40 % is that the staged term, absent here, may have been the actual
  cause of today's 0.385.
- **P-g (a negative pass).** At least one pass reads a NEGATIVE cosine to the total
  core-weight gradient in `slot_gradient_probe.py`: **30 %**. Every arm before the staged
  ones was positive at every pass; today's masked arm had two negatives, but it also had the
  staged term, which is the toy's named mechanism for the separation.
- **P-h (val CE, the price).** 480-row CE at depth 6 within 0.05 of the ruler's 4.3290:
  **12 %**. The mask has cost 0.1 to 0.2 nats on every panel that ran it (today's arm 0.108
  behind; E4 0.131 behind its ruler). Recorded as the PRICE, not as a verdict (Wolfe
  2026-09-09: short-horizon CE cannot rank looped against unlooped).
- **P-i (cost).** Wall clock within 1.2x of the ruler's 52 min 38 s: **80 %**. Today's twin
  ran at 0.97x with one more loss term.

## Binding

- P-b TRUE at the 0.020 bar ⇒ the STAGED LOSS removed the mask's token dependence, the
  ternary rule is exonerated, and the six-arm mask lineage stands. The staged attachment is
  then not merely flat but actively harmful to the one signal this tree ever had, and the
  loss-attachment lane closes with a negative result rather than a null one.
- P-b FALSE at the 0.006 bar (the token curve stays at the staged-mask value) ⇒ the TERNARY
  RULE removed it. Every mask K-curve in the record was measured on a starved core, and the
  arc loses its only demonstration that tokens read the slot's depth. The honest next step
  is then NOT another loss or geometry arm: it is to state that under the shipped rule no
  configuration of this loop has ever produced token-side depth dependence, and to ask what
  a slot could be asked for that needs more than one pass.
- P-b between the bars (0.006 to 0.020) ⇒ both contribute and neither dominates. The panel
  needs the fourth cell of the 2x2 (`absmean` + staged + mask) before anything is claimed,
  and that cell is cheap.
- P-d TRUE with P-b FALSE ⇒ the reader needs the slot and the loop's depth is still not what
  it needs. That reproduces today's masked reading with one fewer moving part and makes the
  exit state, not the training, the object of study.
- A rate stop ⇒ the arm is skipped and the queue continues.

## Not verified before launch

The arm has not run on a GPU. What was run: a Hydra compose check printing
`tul.tg_restrict` True at scope "all", `model.use_kernels` False, `model.tg_scoped_kernels`
True, `model.core_fixed_point_lambda` 1.0, `tul.mux_beta` 1.0 with `mux_target` next,
`tul.mux_stage_own_iters` absent, `ternary_scale_mode` norm_match, 5,000 steps at batch 6
and seq 1024; and a tiny CPU build that ran a training forward and a backward reaching the
core with these knobs (loss 9.305812, core gradient sum 4.01e+02). No test in this change
exercises the mask: `tg_restrict` has its own suite (`tests/test_tg_restrict.py`) and this
arm adds no code. The RATE is an extrapolation from two measurements, not a measurement:
9,349 tok/s is today's staged twin and 12,429 is the unmasked ruler, and this arm is
neither. Memory at the real shape is unmeasured (today's twin peaked at 11.18 GB in its
smoke, with one loss term more). The claim that `tg_restrict` removes audit finding F1 is
read from `_CCAHCAAttention.__init__`, not measured on a checkpoint. The 2x2's fourth cell
(`absmean` + staged + mask) is not queued, so a "both contribute" outcome cannot be resolved
from this batch alone.

## Results

Filed 2026-09-11 01:18. `slot-mux-mask-norm-match` (commit `ab2e31f`; the ruler `tul_slot_mux_norm_match`
plus `tg_restrict: true` at scope "all", `use_kernels: false`, `tg_scoped_kernels: true`):
HEALTHY to 4,999, tripwire max 31.2 at step 607 (ruler 78.7; staged-mask 151), 13,858 tok/s at
step 200 (floor 8,086; ruler 12,429; staged-mask 9,349), smoke peak 10.96 GB, training peak
14.00 GB, wall clock 46 min 37 s (ruler 52 min 38 s, 0.89x). Runner final val_loss 4.4986
(ruler 4.3775; staged-mask 4.5062). Fixed-point term 0.0092 at step 4,999 (ruler 0.0030),
`loop/delta_ratio_last` 0.062, `loop/core_gain_t0` 1.99.

Sweeps, 480 rows, forced depths 1/2/3/6/9/12/16 (`mux_local` is the M-next forecast, the
training objective on this arm):

| step | tokens K1−K6 | tokens K3−K6 | tokens K1−K16 | forecast K1−K6 | forecast @1 / @6 (ruler @6) | CE@6 (ruler; staged-mask) |
| --- | --- | --- | --- | --- | --- | --- |
| 2,500 | +0.0003 [+0.0000, +0.0005] | −0.0001 | | +0.0054 [+0.0043, +0.0066] | 6.8926 / 6.8872 (6.8766) | 4.7438 (4.7194; 4.8183) |
| 5,000 | **+0.0009 [+0.0007, +0.0011]** | −0.0000 [−0.0001, +0.0001] | −0.0006 | +0.0091 [+0.0075, +0.0109] | 6.7988 / 6.7897 (6.7724) | 4.4194 (4.3290; 4.4373) |

The mask on the ruler, with no staged term, reads a token K1−K6 of 0.0009: below the
staged-mask arm's 0.0033 and an order of magnitude under the 0.020 to 0.060 that six
`absmean` mask arms read. The forecast curve is the ruler's (+0.0091 vs +0.0067), with the
exit 0.017 nats worse than the ruler's depth-6 forecast. The mask price on the tokens is 0.090
nats behind the ruler at 5k (staged-mask paid 0.108).

Worth profile at 5,000 (offsets 0..6 after the slot): zero **+0.554** [+0.497, +0.625], +0.182,
+0.128, +0.109, +0.083, +0.057, +0.042 (staged-mask +0.336 .. +0.059; ruler +0.094 .. +0.006);
shuffle +0.665, +0.210, +0.146, +0.114, +0.080, +0.049, +0.022; wrong_seed +1.139, +0.067,
+0.055, +0.051, +0.036, +0.026, +0.021. The slot state carries more of the first token after
the slot than on any arm so far (1.6x the staged-mask arm, 5.9x the ruler), and the seed
matters most of all (wrong_seed +1.14 at offset 0). State probe: |h| 123 at depth 1 → 136 →
143 → 152 at 6 → 165 at 16 (staged-mask 187 → 230 → 238; ruler-family unmasked 449 → 511);
relative distance from depth 1 0.152 / 0.232 / 0.350 / 0.516, cos 0.995 / 0.988 / 0.973 / 0.943.

Gradient probe at 5,000 (12 rows, batch 2, depth 6, eager; three self-checks PASS at max rel
err 3.3e-7): combined cancellation ratio **0.556** (ruler 0.520, staged 0.442, staged-mask
0.385, every-pass 0.771); MUX alone 0.563, token CE alone 0.554. Per-pass share of the shared
core weight gradient 0.208 / 0.095 / 0.104 / 0.129 / 0.163 / 0.300; per-pass cosine to the
total +0.49 / **−0.33** / +0.26 / +0.71 / +0.80 / +0.79 (one negative pass, pass 2). The
token-CE source alone puts 0.318 of its core gradient on pass 1 and 0.204 on pass 6; the MUX
source 0.185 and 0.311. Cotangent share at the loop state 0.167 / 0.168 / 0.166 / 0.159 /
0.155 / 0.184 (the ruler's numbers to three digits), clip never binds. Parameter-group norms:
prelude 11.1, embed 5.02, coda 2.99, core.attention 1.66, core.residual 1.34, core.mlp 1.15,
W_prefix 0.32 (staged-mask: prelude 31.1, core.residual 20.6, coda 3.26). z-optimisation probe
(12 rows, batch 2, 200 Adam steps, bit-exact reproduction): ce_loop 4.1961, entry **+0.0159**
(ruler +0.0015; staged-mask +0.0235; staged +0.0073), zero +0.0939, shuffle +0.0891, fitted z
−1.134 (random start −0.943; cos(z*, z_loop) +0.879), loop z rank 12.0; per-bucket the loop's
z beats the entry by 0.023 on the first 8 tokens after a slot and 0.012 on the tail.

Probes ran 01:10 to 01:17 after the next arm's rate check, behind the 11 GB guard (12.5 GB
free at start); nothing else touched the GPU. Artifacts:
`lab/experiments/results/2026-09-10-slot-mux-mask-norm-match/`; npz under
`ignored/experiment-artifacts/2026-09-10-slot-mux-mask-norm-match/`; probe JSON under
`ignored/experiment-artifacts/2026-09-10-slot-gradient-probe/` and `-slot-z-optimize/`.

## Verdict

P-a TRUE. **P-b: the 0.006 outcome** (0.0009; given 45 %, against 45 % for above 0.010 and
30 % for above 0.020). P-c FALSE at both bars (+0.0091 against 0.02 given 70 %). P-d FALSE
(+0.0159 against 0.02 given 65 %, though 10x the ruler and the second-highest reading on the
tree). P-e TRUE (+0.554 > 0.20). P-f FALSE (0.556 is ABOVE the ruler's 0.520, given 60 % for
below). P-g TRUE (pass 2 at −0.33, given 30 %). P-h FALSE as predicted (0.090 behind). P-i
TRUE (0.89x). Three of the majority-confidence predictions missed (P-c, P-d, P-f), so the
file goes under failures; the Question itself is answered. Binding clause 2 fires: the
TERNARY RULE removed the mask's token dependence. The staged loss is exonerated (this arm has
no staged term and reads LOWER than the staged-mask arm), and every mask K-curve in the record
(E4, E13, E16, E18) was measured on the starved `absmean` core. Under the shipped rule no
configuration of this loop has produced token-side depth dependence: the ruler 0.0001, staged
0.0020, staged-mask 0.0033, this arm 0.0009, all with the slot as the ONLY road on the two
masked arms. Read with the forecast: the mask leaves the forecast K-curve at the ruler's
+0.009, so under `norm_match` neither the tokens nor the slot's own objective moves with depth
when the geometry forces every cross-span bit through the loop.

What the mask DID change, measured: the reader's use of the slot (worth zero 0.094 → 0.554,
entry-vs-exit 0.0015 → 0.0159, both the largest on the unstaged family) and nothing about
the passes (cotangent shares identical to the ruler to three digits; cancellation 0.556 vs
0.520, inside the seed spread). The staged term, not the mask, is what made the passes
disagree on the staged-mask arm (0.385 there, 0.442 unmasked-staged, 0.556 here).

## Updated hypothesis

The `absmean` mask K-curves were a starved-core artefact: with the core's per-pass
contribution small and noisy, the tokens, given no other route, tracked whatever the later
passes added. Under `norm_match` one well-conditioned pass makes the exit the coda wants, and
forcing the route buys nothing with depth. The coda reads the slot hard when it must (0.55
nats on the first token after the slot; the fitted z would give it 1.13), and what it reads
is what pass 1 made. The `absmean` + staged + mask cell of the 2x2 is no longer needed to
attribute the loss; the question that remains is the one this file named as the honest next
step: what a slot can be asked for that needs more than one pass. On this batch that is the
target arms, `slot-loop-mask-norm-match` (the coda's need as the ONLY loss under this mask,
running now) and the input/width arms `slot-mux-bagmean-norm-match` and
`slot-mux-prefix4-norm-match`.
