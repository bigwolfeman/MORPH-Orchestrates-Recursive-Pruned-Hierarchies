# Failure: ARC E8 — parallel multi-token prediction on the coda (the target lever, strict form)

Status: failure
Date: 2026-09-07 (frozen before any smoke; Wolfe: "an arm for testing contribution that uses
parallel multitoken prediction on the coda. It may help the contribution and assignment.
It is worth trying.")
Arc: `2026-09-04-loop-contribution-arc.md`, branch (b) TARGET. Follows E6
(`successes/2026-09-07-arc-e6-deep-recurrence-draw.md`).

## Question

The arc's hypothesis (b) says a loop earns only on the compute-limited part of its loss,
and next-token prediction on web text is mostly entropy-limited. E0 measured that every
MORPH loop's earning is a flat refinement that avoids the hard rows. A parallel
multi-token target (Gloeckle et al. 2024) makes each position also predict t+2, t+3, t+4
through heads that read ONLY the position's readout state, so whatever lookahead the
model can compute must be carried in that state; the gradient of the future flows into
the position. Does that target make the loop earn past iteration 3 on the mean-6 draw
(contribution), and does it move where the earning sits (assignment)? On the deep draw,
does it turn E6's dependence into value?

## Method

`model.mtp_heads 4`, `mtp_weight 1.0` (equal weights, the paper's setting): three heads,
each RMSNorm → identity-init Linear(d, d) → the tied LM head, loss CE_1 + Σ CE_j, tail
labels −100. Two arms, 5000 steps each, seed 1, the ramp, seq 1024, batch 6:

- **E8-6** `notul_mtp4.yaml`: the mean-6 plain loop (control: notul at 5000,
  `results/2026-09-04-arc-e0/sweep_notul_5000.json`, same rows and step).
- **E8-16** `notul_deep16_mtp4.yaml`: the mean-16 draw of E6 (control: E6 at 5000,
  `results/2026-09-07-arc-e6/`).

Readout: `token_depth_sweep.py --profile` at forced depths 1, 3, 6, 8 (E8-6) and 1, 3, 6,
8, 12, 16, 24 (E8-16) on the arc's 480 rows at 2500 and 5000, now also writing each
head's CE(depth) with per-row sums; paired bootstraps; `score_arc_e0.py` at (1, 6) and,
for E8-16, (3, 12). train/loss and val are the next-token CE; the heads log as
`train/ce_mtp_j` and `val/ce_mtp_j`. Sustained tripwire on both. Output
`results/2026-09-07-arc-e8/`. Runs after E7 in `arc/run_e8.sh`.

Cost, not measured: three extra fused CE passes per step (each streams the 49k vocab in
chunks); the smoke gives the step time. Peak memory grows by the head activations only.

## Predictions (frozen)

- **P8a (next-token price, E8-6).** Next-token CE at depth 6 on the 480 rows is WORSE
  than notul's 3.9707 by more than 0.02: **60 %** (Gloeckle: MTP costs next-token loss
  below ~1B parameters).
- **P8b (contribution, E8-6).** Next-token K3−K6 > 0.01 with the paired CI above 0
  (notul: 0.0016): **40 %**.
- **P8c (the heads carry the depth, E8-6).** The head-4 (t+4) K1−K6 exceeds the
  next-token K1−K6 by a factor above 1.5 with non-overlapping CIs: **55 %** (a lookahead
  target is less entropy-limited per bit the model can get, so the loop's work should
  show more there).
- **P8d (assignment, E8-6).** Spearman(row CE_1, next-token earning K1−K6) rises above
  E0's notul reading of −0.18 by more than 0.10 (earning moves toward the hard rows):
  **45 %**.
- **P8e (value, E8-16).** E8-16's next-token CE at depth 16 minus E6's at depth 16 on
  the same rows is below −0.02 (the target improves the deep model): **35 %**; and
  E8-16 at 16 minus notul at 6 is below +0.104 (the deep gap narrows): **50 %**.
- **P8f (stability).** Both arms reach 5000 with the sustained tripwire silent: **70 %**
  (the summed loss is ~4x the next-token loss; the clip is global).

## Binding

- P8b TRUE ⇒ the target is a lever on the mean-6 loop's contribution (the first
  target-side lever to move it). The next arm puts the same heads on the block-loop's
  token coda (needs the layout-aware label shift), scored on the masked arm's bars.
- P8b FALSE and P8c TRUE ⇒ the loop carries lookahead for the heads but not for the
  next token: the contribution is target-specific, and the next-token bar is the wrong
  bar for a think-once design. Record it and score TUL arms on the span's CE, not t+1.
- P8b FALSE and P8c FALSE ⇒ the target lever (b) is closed at this scale on web text,
  alongside (a); the remaining lever is Design 2 (per-position depth) or a data mix.
- P8e's first half TRUE ⇒ E6's dependence can be made value by the target; E5 at the
  deep draw runs with the heads on.
- P8f FALSE ⇒ the loss weighting is the first suspect; a `mtp_weight 0.3` draw before
  any reading.

## Not verified before launch

Step time and memory with three fused CE passes (no smoke has run); the fused CE's
`n_valid` host sync per head (three more per step; it exists on the main head already);
the identity-init heads' early loss (equal to the next-token CE against a shifted label,
so the total starts near 4x log V); whether `train/loss` subtraction of `mtp_weighted`
is exact under autocast (it is computed from the same tensors, one float cast).

## Results (2026-09-07 04:34; `arc/run_e8.sh` on worktree 901237a; files in `results/2026-09-07-arc-e8/`)

**E8-6** (`notul_mtp4`, mean 6, full BPTT): 5000 steps, healthy (max `preclip/total` 332 at
2154), 1.45 steps/s, peak 10.1 GB, final val 4.328 (notul run: 4.138). Forced depth on the
480 rows at 5000:

| depth | next token | t+2 head | t+3 head | t+4 head |
|---|---|---|---|---|
| 1 | 4.368 | 5.949 | 6.541 | 6.789 |
| 3 | 4.317 | 5.914 | 6.518 | 6.773 |
| 6 | 4.315 | 5.912 | 6.517 | 6.772 |
| 8 | 4.317 | 5.913 | 6.518 | 6.773 |

Paired (`paired_ci.txt`): next-token K1−K3 +0.0505, K3−K6 +0.0021 [+0.0017, +0.0025],
K6−K8 −0.0019, K1−K6 +0.0526; E8-6 at depth 6 minus notul at depth 6 on the same rows
**+0.3445 [+0.3298, +0.3597]**. Heads' K1−K6: +0.0374 (t+2), +0.0238 (t+3), +0.0171 (t+4),
ratios of means to the next-token K1−K6 0.71, 0.45, 0.32; heads' K3−K6 ≤ +0.0019.
`score_arc_e0.py` at (1, 6): Spearman(row CE₁, earning) −0.139 [−0.230, −0.044] (notul in
E0: −0.18); offset-0 earning 0.79x the later offsets; top-decile earning share 0.090 against
loss share 0.123.

**E8-16** (`notul_deep16_mtp4`, mean 16 / max 24 / bptt 8): DETONATED at 3446 (rows > 1e4 at
3429 and 3446, max 1.7e4); its control E6 ran 5000 steps at max 35. Pre-onset 2500 sweep:
next-token 5.183 / 4.935 / 4.844 / 4.822 / 4.804 / 4.800 at depths 1 / 3 / 6 / 8 / 12 / 16
(E6 at 2500: 5.198 / 4.622 / 4.560 / 4.549 / 4.539 / 4.537); K3−K6 +0.0918, K6−K12 +0.0397,
K12−K16 +0.0039, K16−K24 −0.0064; **E8-16 at 16 minus E6 at 16, paired, +0.2634 [+0.2578,
+0.2697]**. Heads' K1−K6 ratios to the next-token's: 0.63, 0.44, 0.34.

Scored:

- **P8a TRUE** (+0.345 against a 0.02 bar).
- **P8b FALSE** (K3−K6 +0.0021 against 0.01).
- **P8c FALSE** (the t+4 head's K1−K6 is 0.32x the next-token's; the bar was 1.5x). The
  further ahead the target, the LESS depth it uses, on both draws.
- **P8d FALSE** (Spearman moved +0.04, bar +0.10).
- **P8e FALSE** at the only readable checkpoint (E8-16 is 0.26 nats behind E6 at 2500);
  unscored at 5000.
- **P8f FALSE** (E8-16 detonated at 3446; E8-6 healthy).

## Verdict

The multi-token target is not a lever on the loop's contribution or its assignment at this
scale. On the mean-6 loop it leaves the saturation at 3 untouched, the lookahead heads use
depth less than the next-token head, the earning shape stays MORPH's, and it costs 0.34 nats
of next-token CE at matched steps (Gloeckle's small-model penalty, larger here). On the deep
draw it adds 0.26 nats over E6 and detonates where E6 did not: the four-fold summed loss on
the truncated deep loop (which trains on ~73 % of its samples' loop gradients, the E7-era
defect) is a stability load, not a contribution one.

Not verified: a one-layer head with attention (the paper's head) rather than the strict
readout-only head; a lower `mtp_weight` (the prereg's P8f follow-up, not run because the
mean-6 arm was stable and the deep arm's problem is the draw); whether the heads help at
20k (deep models converge slower; the 0.34 gap is a 5k number).

## Updated hypothesis

Target-side levers on next-token web text do not move the loop: the forecast target (E1–E3),
the masked route (E7) and the multi-token target (E8) all leave K3−K6 near zero on a stable
map. Branch (b) is closed at this width alongside branch (a). What remains is the loop's
own dynamics (E9, E10: the carry, the fixed point, the directional hinge) and, for TUL, the
mask's reading on a stable map (E4).
