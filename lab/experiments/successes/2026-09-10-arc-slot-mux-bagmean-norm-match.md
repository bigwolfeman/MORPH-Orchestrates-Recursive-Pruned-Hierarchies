# Planned: the spec's slot seed on the ruler, under the shipped ternary rule

Status: success

Date: 2026-09-10 (frozen before launch). Arc: `2026-09-04-loop-contribution-arc.md`.
One factor against the ruler `slot-mux-norm-match`
(`morph/configs/tul_slot_mux_norm_match.yaml`). The factor is the slot's INPUT.
Prior measurements of the same axis:
[`2026-09-01-bound-seed-rank.md`](../failures/2026-09-01-bound-seed-rank.md) (the static
seed probe) and [`2026-09-01-write-side-ladder.md`](../failures/2026-09-01-write-side-ladder.md)
(the trained arms W1 and W2). Wolfe's 2026-09-10 reading, which put this axis back on the
table: the slot's target and the slot's input/state capacity are the two levers left.

## Question

The slot's state is the thing the coda reads and the thing the loop iterates, and it sits at
effective rank 1.7 to 4.8 in 1024 dimensions on every checkpoint the campaign has probed.
The ruler itself reads `val/slot_eff_rank` 6.3389 with `val/slot_pairwise_cos` 0.7388 at step
5,000: 54.8 slots per row that are 0.74 alike.

The seed is where that rank is set. The ruler uses `slot_seed: boundary`, shipped with arm
TG4b: `E_slot + W_sent . embed(t_last)`, one learned projection of the span's LAST token. The
spec's original seed, `bag_mean`, is `E_slot + mean_j embed(t_j)` over the whole span. One
reads a token, the other reads the span.

The seed axis has never been measured under `norm_match`. Every reading on it predates the
ternary rule change of 2026-09-09, and the rule is the one intervention that has moved loop
contribution on this tree by a factor of five (plain K1-K6 0.033 under `absmean` against
0.185 under `norm_match`). Does a stronger per-pass map change what the seed is worth?

## Hypothesis

H-seed-0, the record's answer, and the favoured one. The write-side ladder already ran this
axis to a verdict: "No seed mode moves loop earning; the core ignores its input rank." W1
(`slot_seed: content`, which is `bag_mean` minus the shared `E_slot` term) read K1-K6 +0.0007
against R0 boundary's +0.0113, and W2 (`bound`, the HRR binding) +0.0019, both at the floor,
while the trained `val/slot_eff_rank` moved 40.2 to 73.3 to 48.2 across the three. Rank and
depth-earning did not move together. TG4a found the seed moves CE by 0.003 nats. Prediction:
this arm reads the ruler's K-curve inside its CI and the axis closes under the shipped rule
too.

H-seed-1, the reason to run it anyway. The static seed probe measured the two seeds this arm
compares and ranked them the opposite way from the motivation: on 201 rows, `ship` (boundary)
reads unit rank 1.58 and mean pairwise cosine 0.773, `bag` (bag_mean) reads 1.13 and 0.941.
bag_mean is the LOWER-rank, MORE collinear seed. So if the input's rank is what limits the
loop, this arm makes it WORSE, and the prediction is a measurable fall in the loop's
contribution and in `val/slot_eff_rank`. A flat reading on a seed the record says should hurt
is stronger evidence for H-seed-0 than a flat reading on a seed that should help.

H-seed-2, the split the ladder actually found. W1 read 0.13 nats BETTER absolute CE than R0
boundary (4.3395 against 4.4689) while its depth earning fell to the floor. If that
reproduces here, "the model is better" and "the loop earns" are separable by the seed alone,
and every CE comparison in this campaign has to be read with the seed held fixed. That is a
result, not a cure.

## Method

`tul_slot_mux_bagmean_norm_match` = `tul_slot_mux_norm_match` (the ruler) plus ONE change,
`tul.slot_seed: bag_mean`. Everything else is the ruler: M-next MUX at beta 1 through the tied
head, exit-only attachment, prelude entry, slot loop (`tokens_through_core: false`),
`prefix_k` 2, `max_slots` 64, hinge lambda 100 at target 0.9, `core_fixed_point_lambda` 1.0,
`slot_cot_clip` 4.0, `ternary_scale_mode: norm_match`, fused kernels, seq 1024, batch 6, seed
1, 5,000 steps, ramp 1,000, checkpoints at 2,500 and 5,000.

What changes in the model, exactly. `bag_mean` does not build `W_sent`, so the arm has
1024 x 1024 = 1.05 M FEWER parameters than the ruler and the slot input is a mean over the
span's token embeddings instead of a learned projection of one token. The TUL parameters are
constructed LAST (runtime-invariants 6b, `tests/test_tul_forward.py`), so every non-TUL
weight is the ruler's at the same seed; the TUL block's own draws differ because `W_sent` is
absent from the stream. The layout is untouched: same boundary rule, same `L_total` 1152,
same 1014.0 tokens and 54.8 spans per row.

`tul.center_bag_mean` stays false (its default). It is legal only with `bag_mean` and turning
it on would make this a two-factor arm.

Runner `arc/run_slotloop3.sh` (file-driven queue, commit pinned in `arc/slotloop3_arms.txt`),
a 12-step smoke first, the draw under the sustained tripwire
(`lab/divergence/tripwire_sustained.py`). Rate rule: tok/s below 8,086 at step 200 skips the
arm and the queue continues.

Readout: `core_depth_sweep.py` at 2,500 and 5,000, depths 1,2,3,6,9,12,16, 480 rows, giving
the token K-curve and the forecast column `mux_local`; `worth_profile.py` and
`slot_state_probe.py` at 5,000 (the runner's `slot` kind does both); `slot_gradient_probe.py`
and `slot_z_optimize.py` at 5,000 by hand. The trainer's own `val/slot_eff_rank`,
`val/slot_pairwise_cos` and `val/slot_norm_mean` at step 5,000 are read from the run log; they
are the only instrument in the readout that looks at the rank the hypothesis is about. No new
wandb keys.

The numbers this arm is read against, cited once. RULER `slot-mux-norm-match` at 5,000:
480-row CE at depth 6 **4.3290**; token K1-K6 +0.0001 [-0.0000, +0.0002], K3-K6 -0.0000;
`mux_local` K1-K6 +0.0067 [+0.0053, +0.0081], K3-K6 +0.0005; forecast at depth 1 / 6 =
6.7791 / 6.7724; worth zero at offset 0 +0.094; z-optimisation `ce_loop` 4.1173 with the loop
ENTRY worth +0.0015; combined cancellation ratio 0.520; per-pass cotangent share 0.168 /
0.168 / 0.166 / 0.160 / 0.154 / 0.183 and per-pass cosine to the total 0.33 / 0.27 / 0.37 /
0.63 / 0.75 / 0.67; `val/slot_eff_rank` 6.3389, `val/slot_pairwise_cos` 0.7388,
`val/slot_norm_mean` 29.9062; 12,429 tok/s at step 200; wall clock 52 min 38 s; tripwire max
78.7; runner final val_loss 4.3775; smoke peak 10.1 GB. STATIC SEED PROBE (201 rows,
2026-09-01, an UNTRAINED measurement of the seed alone): ship/boundary unit rank 1.58, raw
1.30, cos 0.773; bag/bag_mean 1.13, 1.14, 0.941. THE TRAINED LADDER (2026-09-02, a different
and weaker recipe, 48 paired rows, and its R1 control was a weak draw): R0 boundary K1-K6
+0.0113 with CE 4.4689 and `slot_eff_rank` 40.2; W1 content +0.0007, 4.3395, 73.3; W2 bound
+0.0019, 4.4170, 48.2.

## Predictions (frozen)

- **P-a (survival AND rate).** HEALTHY to 5,000 with no sustained tripwire AND tok/s at step
  200 at or above 8,086: **90 %**. The arm removes a 1024x1024 matmul from the slot seed and
  changes nothing in the loop, the optimiser or the geometry. The 10 % is the campaign's
  detonation base rate, which no slot arm under the constraint has hit since 2026-09-04.
- **P-b (tokens).** Token K1-K6 at 5,000 above 0.010: **8 %**. Above 0.003, the staged arm's
  value: **15 %**. Inside the ruler's CI, so at or below 0.0005: **55 %**. The ladder's
  verdict is the reason these are low: the seed axis has been run to a negative result once
  already, on three modes, and the mode this arm picks is the one the static probe ranks
  LOWEST on rank.
- **P-c (forecast, the loop's own curve).** `mux_local` K1-K6 above 0.02 at 5,000 (the ruler
  +0.0067): **20 %**. Above the ruler's 0.0067 at all: **45 %**. Set near even on the second
  clause because the forecast curve is the one instrument that has moved on almost every arm
  this month, usually by depth 1 getting worse rather than depth 6 getting better; scored with
  the end point.
- **P-d (the reader).** The coda's exit-minus-entry CE from `slot_z_optimize.py` above 0.02
  nats (the ruler +0.0015; the largest reading of any unmasked arm is the staged arm's
  +0.0073): **10 %**. Nothing in this change touches the reader or the loss.
- **P-e (worth).** `worth_profile.py` zero at offset 0 above 0.15 nats (the ruler +0.094; the
  staged arm +0.078; the masked arms 0.21 to 0.34): **20 %**. A span-mean seed carries more of
  the span than a one-token seed does, so the slot could become more load-bearing without the
  loop earning anything; that is the mechanism this clause tests and it is the most plausible
  non-null outcome in this file.
- **P-f (the rank, the hypothesis's own instrument).** `val/slot_eff_rank` at step 5,000 BELOW
  the ruler's 6.3389: **60 %**. The static probe says bag_mean is the lower-rank seed (1.13
  against 1.58), and the trained ladder says training moves the number a long way from its
  static value, which is why this is 60 % and not 85 %.
- **P-g (CE, the price or the discount).** 480-row CE at depth 6 BETTER than the ruler's
  4.3290 by more than 0.05 nats: **35 %**. W1, the nearest relative of this seed, read 0.13
  nats better than boundary on the old recipe. Recorded as a PRICE or a discount, not as a
  verdict (Wolfe 2026-09-09: a 5,000-step CE cannot rank looped against unlooped).
- **P-h (cost).** Wall clock within 1.1x of the ruler's 52 min 38 s: **90 %**. The arm removes
  a matmul and adds a segment mean, on 54.8 slots of a 1,152-position row.

## Binding

- P-b FALSE and P-c FALSE (both curves inside the ruler's band) with P-f TRUE (the rank fell)
  ⇒ the input axis is CLOSED under the shipped rule as well as under `absmean`: the loop's
  contribution does not follow the seed's rank in either ternary regime, and no further seed
  arm runs in this arc. Wolfe's "input capacity" lever then means the slot's STATE capacity
  (`prefix_k`, the write width, and the carrier dimensions the loop actually uses), not its
  seed, and `slot-mux-prefix4-norm-match` is the arm that reads that half.
- P-b FALSE and P-c FALSE with P-f FALSE (the rank did NOT fall) ⇒ the static seed probe does
  not predict the trained rank, which the ladder already suspected; the rank instrument is
  then not a design lever and should stop being cited as one.
- P-g TRUE with P-b FALSE ⇒ H-seed-2: the seed buys CE and not depth. Every CE comparison in
  this campaign must then name its seed, and the ruler's own 4.3290 is partly a seed choice.
  This changes how the panel is read; it does not open a new arm.
- P-e TRUE with P-b FALSE ⇒ the slot became more load-bearing without the loop earning. That
  is the `slot-unpack` pattern again and it says the coda's use of the slot is set by what the
  slot CONTAINS, not by how many passes made it.
- A rate stop ⇒ the arm is skipped and the queue continues.

## Not verified before launch

The arm has not run on a GPU. What was run: a Hydra compose check printing `tul.slot_seed`
bag_mean against the ruler's boundary with every other key in the panel list identical
(`prefix_k` 2, `max_slots` 64, `tokens_through_core` False, `mux_beta` 1.0 / `mux_target`
next, `core_fixed_point_lambda` 1.0, `slot_gain_lambda` 100 at 0.9, `slot_cot_clip` 4.0,
`ternary_scale_mode` norm_match, `ademamix_t_beta3` 3500, `ademamix_t_alpha` 1600, 5,000 steps
at batch 6 and seq 1024, `L_total` 1152); and a tiny CPU build on the
`tests/test_tul_forward.py` fixtures that ran a training forward and a backward with each
setting, loss 9.673785 (boundary) against 9.550279 (bag_mean), `W_sent` present against
absent, TUL parameter count 12,416 against 8,320 (the 4,096-element difference is the
fixture's 64x64 `W_sent`), and the backward reaching 59 core tensors with summed absolute
gradient 288.3 against 226.8.

Not verified: the rate and the memory at the real shape; both are extrapolations from the
ruler, and this arm is the first on the panel to remove a parameter rather than add one. The
`bag_mean` code path has not run on a GPU under `norm_match`, under the gain constraint, with
the MUX, or at 5,000 steps on this recipe; the last arms that ran a bag-family seed were W1
and W2 on 2026-09-02 under a different recipe and the OLD ternary rule, and their control
draw was weak by their own filing. `val/slot_eff_rank` is a trainer eval metric over the
WRITTEN slot states, not over the seeds, so P-f is measured downstream of the quantity the
static probe measured, and the two numbers are not the same quantity. No test in this change
exercises the seed mode on the shipped forward; `tests/test_slot_seed.py` and
`tests/test_slot_seed_modes.py` own that contract and this arm adds no code.

## Results

Filed 2026-09-11 11:03. `slot-mux-bagmean-norm-match` (commit `01f88ce`; the ruler with `tul.slot_seed`
bag_mean, so `E_slot` + the span's token-embedding mean and no `W_sent`): HEALTHY to 4,999,
tripwire max 72.8 at step 244 (ruler 78.7 at 245), hinge binding on 238 of 5,000 steps (ruler
6), 12,996 tok/s at step 200 (ruler 12,429), smoke peak 10.81 GB, training peak 11.92 GB, wall
clock 51 min 07 s (ruler 52 min 38 s, 0.97x). Runner final val_loss 4.4174 (ruler 4.3775).
Over steps 4,900 to 4,999: `gain_est` 0.894, `loop/delta_ratio_last` 0.048, `loop/core_gain_t0`
1.56, fixed-point term 0.0057. `val/slot_eff_rank` at 5,000: **6.5247** (ruler 6.3389; the
static probe had ranked bag_mean BELOW boundary, 1.13 against 1.58).

Sweeps, 480 rows, forced depths 1/2/3/6/9/12/16:

| step | tokens K1−K6 | tokens K3−K6 | forecast K1−K6 | forecast K3−K6 | forecast @1 / @6 (ruler @6) | CE@6 (ruler) |
| --- | --- | --- | --- | --- | --- | --- |
| 2,500 | +0.0002 [+0.0000, +0.0003] | −0.0001 | +0.0043 [+0.0032, +0.0053] | −0.0002 | 6.8741 / 6.8699 (6.8766) | 4.7372 (4.7194) |
| 5,000 | **+0.0005 [+0.0003, +0.0006]** | +0.0000 [−0.0000, +0.0001] | +0.0057 [+0.0046, +0.0071] | +0.0005 [+0.0001, +0.0011] | 6.7759 / 6.7702 (6.7724) | 4.3536 (4.3290) |

Both curves sit inside the ruler's band (tokens at the 0.0005 edge, forecast below the
ruler's +0.0067). CE at depth 6 is 0.025 behind the ruler; the runner's val 0.040 behind.

Worth profile at 5,000 (offsets 0..6): zero +0.087 [+0.077, +0.097], +0.054, +0.035, +0.032,
+0.019, +0.016, +0.013 (ruler +0.094 .. +0.006); shuffle +0.070, +0.055, +0.045, +0.036,
+0.025, +0.016, +0.007; **wrong_seed +0.007 [+0.004, +0.010]**, +0.007, +0.006, +0.008,
+0.003, +0.003, +0.002 (ruler +0.035 at offset 0; the prefix4 arm +0.424). Swapping in
another span's mean seed costs the first token 0.007 nats: a span mean carries a fifth of the
span identity that the boundary token does. State probe: |h| 111 at depth 1 → 125 → 131 →
137 at 6 → 142 at 16; relative distance from depth 1 0.150 / 0.225 / 0.316 / 0.410, cos 0.997
/ 0.992 / 0.982 / 0.963.

Gradient probe at 5,000 (12 rows, batch 2, depth 6, eager; self-checks PASS): combined
cancellation ratio 0.521 (ruler 0.520); MUX 0.549, token CE 0.429. Per-pass share of the
shared core weight gradient 0.137 / 0.109 / 0.108 / 0.117 / 0.167 / 0.362; per-pass cosine to
the total +0.23 / −0.19 / +0.19 / +0.62 / +0.73 / +0.81. Parameter-group norms: prelude 3.02,
coda 2.23, **E_slot 1.76** (ruler 0.26: with no `W_sent`, the one shared embedding takes the
seed's whole gradient), core.residual 1.29, core.mlp 1.02, core.attention 0.74, W_prefix
0.086. z-optimisation probe (12 rows, batch 2, 200 Adam steps, bit-exact): ce_loop 4.1399,
entry **+0.0038** (ruler +0.0015), zero +0.0212, shuffle +0.0220, fitted z −1.032 (random
start −0.829; cos(z*, z_loop) +0.876), loop z rank 11.9.

Probe note: the runner's own probe slot was SKIPPED at 09:38 (8.05 GB free beside the
prefix4 arm, under the 11 GB guard); the probes were re-armed to run after the prefix4 arm's
probes and ran 10:55 to 11:02 beside the hca-fix draw with 13.9 GB free. Artifacts:
`lab/experiments/results/2026-09-10-slot-mux-bagmean-norm-match/`; npz under
`ignored/experiment-artifacts/2026-09-10-slot-mux-bagmean-norm-match/`; probe JSON under
`ignored/experiment-artifacts/2026-09-10-slot-gradient-probe/` and `-slot-z-optimize/`.

## Verdict

P-a TRUE. **P-b: the inside-the-ruler outcome** (+0.0005, given 55 %; the two upper bars
FALSE as predicted). P-c FALSE at both clauses as predicted (+0.0057 is below the ruler's
+0.0067). P-d FALSE as predicted (+0.0038). P-e FALSE as predicted (+0.087). **P-f FALSE**
(6.52 is ABOVE 6.34, given 60 % for below; the one majority miss). P-g FALSE as predicted
(0.025 behind, not 0.05 better). P-h TRUE (0.97x). Seven of eight held, so the file goes
under successes. Binding clause 2 fires: **the static seed probe does not predict the trained
rank** (it ranked bag_mean lowest and training put it above boundary), so the rank instrument
is not a design lever and stops being cited as one. Clause 1's substance also holds: the
input axis is closed under the shipped rule as under `absmean`, the loop's contribution does
not follow the seed in either regime, and no further seed arm runs in this arc. Clause 3 does
NOT fire: this seed is a small PRICE (0.025 to 0.040 nats), not the 0.13 discount W1 read on
the old recipe.

## Updated hypothesis

The slot's input is not what limits the loop: a lower-identity seed (wrong_seed 0.035 →
0.007) trains to a slightly higher rank, the loop moves the state the same way, the coda
reads the exit the same way, and the tokens pay 0.025 nats for the missing boundary token.
With `slot-mux-prefix4-norm-match` (the write width, filed the same morning) the channel is
closed at both ends. The remaining lever is the target, and the target work belongs on the
Parcae core.
