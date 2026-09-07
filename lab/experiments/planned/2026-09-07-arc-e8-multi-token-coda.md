# Planned: ARC E8 — parallel multi-token prediction on the coda (the target lever, strict form)

Status: planned
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
