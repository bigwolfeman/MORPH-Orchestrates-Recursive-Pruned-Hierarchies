# Failure: ARC E9 — widen the diagonal carry to the whole residual (Parcae's ρ(A) < 1) under the detonation assay

Status: failure
Date: 2026-09-07 (frozen before any smoke; Wolfe: "we wanna try widening the diag carry")
Arc: `2026-09-04-loop-contribution-arc.md`. Follows E7
(`failures/2026-09-07-arc-e7-block-loop.md`) and the Parcae re-read
(`docs/references/looping-depth/parcae/2026-09-07-parcae-vs-morph-mechanisms.md`).

## Question

Parcae's stability claim (arXiv 2604.12946 §4.1, Table 1, Fig. 3) is a diagonal state
carry with spectral radius below 1 on the WHOLE residual; runs that diverge are the ones
whose carry learns ρ(A) ≥ 1. MORPH's `DiagonalInjection` applies that carry to the
256-dim context channel only; the other 512 dims ride the HC-Cayley residual, which is
norm-preserving (Table 1's "marginally stable" row). E7's Jacobian shows the failure that
row predicts: one iteration sets the carrier's scale (485 → 1.5e7) and the rest rotate.
Does a carry with ρ < 1 on every dim remove the early detonation that the LR ramp only
avoids?

## Method

The measured assay (`lab/divergence/DIVERGENCE-README.md` §A): the plain winner recipe
(`notul`, seq 1024, batch 6, ternary, AdEMAMix β1=0, alpha_cap 3.5) with `warmup 0`
detonates in ~70 % of draws (17 detonations, all with the first crossing of 1e3 at steps
200–775; 0 of 44 healthy runs ever exceeded 830), and every run that reached step 1000
finished. Abort rule: `preclip/total > 1e4` at any step ≥ 200 (`tulfm/tripwire.py`).

- **Arm:** `notul_carry_all.yaml` (`model.injection_channels: all`, `injection_all_decay
  0.9`: the context dims keep decay 0.447 / dt 1.0, the other 512 start at decay 0.9 /
  dt 0.1 so the linear carry's fixed point is e). Six draws, seeds 201–206, 1200 steps
  each, single-step tripwire, verdict = detonated / healthy per draw.
- **Control:** `notul_carry_ctx_wu0.yaml` (identical, carry on ctx only). Three draws,
  seeds 201–203.
- Readouts per draw: the tripwire verdict and first-crossing step, `preclip/total`
  maximum, val CE at 400/800/1200, `loss/gain_est` is absent on the plain path so the
  per-step `preclip/core_block_gain` stands in. Output `arc/results/2026-09-07-arc-e9/`,
  copied to `results/2026-09-07-arc-e9/` when filed. Runner `arc/run_e9.sh`, after E8.
  ~13 min per draw at ~1.6 steps/s: ~2 h for nine draws.

Cost of the widened carry per step: one elementwise op over 768 instead of 256 dims per
iteration; negligible.

## Predictions (frozen)

- **P9a (the claim).** At most 1 of the 6 widened draws detonates: **45 %** (under the
  ~70 % base rate the chance of ≤ 1 in 6 is about 1 %, so 0 or 1 is decisive either way).
- **P9b (the base rate holds).** At least 2 of the 3 control draws detonate: **70 %**.
- **P9c (the mechanism's signature).** Among widened draws that survive, the maximum
  `preclip/total` after step 200 stays below 830 (the healthy band): **55 %**.
- **P9d (no price).** The surviving widened draws' val CE at 1200 is within 0.05 of the
  surviving control draws' (if any survive; otherwise against the ramped notul at 1200
  from the E0 tree, read from its run log): **50 %**.
- **P9e.** The widened carry's learned decay on the non-context dims moves off 0.9 by
  more than 0.05 (mean over dims) by step 1200: **60 %** (Parcae Fig. 3: the carry is
  learned, and divergent runs push it to 1).

## Binding

- P9a TRUE and P9b TRUE ⇒ the carry span is the structural cause of the early detonation
  (Parcae's mechanism, on MORPH). `injection_channels: all` becomes a candidate default;
  the next run is the base recipe (ramp on) with the wide carry at 5000 steps against
  notul, scored on val CE and the depth sweep, and E7's rerun gets the wide carry before
  any renorm.
- P9a FALSE ⇒ the carry span is not sufficient; the early detonation lives elsewhere
  (ternary trigger, the optimizer, or the write scale). The next structural test is the
  bounded write (Fully-Looped attention injection, 2605.18797) or the directional
  penalty (STARS, 2605.26733).
- P9b FALSE (the control does not detonate at ≥ 2 of 3) ⇒ the assay's base rate has
  moved on this tree; the arm is unreadable at n = 6 and the assay must be re-measured
  before any carry claim.

## Not verified before launch

The widened carry under torch.compile (the injection is a plain elementwise op, but the
768-wide parameters are new shapes); whether 0.9 / 0.1 is a sensible init for the
semantic and bigram channels (Parcae never ablates its init; a second decay value is a
follow-up, not part of this prereg); the tripwire's exit-code handling in the runner's
draw loop at 1200 steps (the E6/E7 runner used the sustained variant).

### Method amendment, 2026-09-07 04:45 (draw count cut; predictions untouched)

After the first two widened draws detonated (seeds 201 and 202, at steps 419 and 200)
P9a ("at most 1 of 6") was decided, and the remaining four widened draws and two control
draws were cancelled to hand the window to E10 and E4 (`arc/cut_e9.sh`). P9b therefore has
one control draw (seed 201, detonated at 201), not three; P9c–P9e have no surviving
widened draw to read.

## Results (2026-09-07 04:34–04:44; `arc/run_e9.sh` on worktree 1d6e292; files in `results/2026-09-07-arc-e9/`)

| draw | carry | verdict | first crossing | max `preclip/total` |
|---|---|---|---|---|
| carry-all-s201 | all 768 dims | DETONATED | 419 | 1.29e4 at 419 |
| carry-ctx-s201 | ctx (control) | DETONATED | 201 | 9.06e5 at 201 |
| carry-all-s202 | all 768 dims | DETONATED | 200 | 4.61e8 at 225 |

Both widened draws detonated inside the assay's window (200–775); the second reached 4.6e8
by step 225, larger than any recorded control. The one control detonated at 201.

- **P9a FALSE** (2 of 2 widened draws detonated; the bar was ≤ 1 of 6).
- **P9b** consistent (1 of 1 control detonated) but under-sampled by the cut.
- **P9c, P9d, P9e** unreadable (no surviving widened draw).

## Verdict

A ρ < 1 diagonal carry on every carrier dim, at init decay 0.9 / dt 0.1 on the non-context
dims, does not remove or delay the early detonation on the ternary + AdEMAMix recipe at
warmup 0. Parcae's constraint is on the state carry; MORPH's early detonation is not on the
state carry, or not only. Consistent with the Parcae reader's own table: MORPH detonates at
an LR 20–80x below the LRs at which Parcae's UNconstrained baselines diverge, so the trigger
here (ternary is the measured trigger surface; the optimizer's slow EMA fills during the
window) is one Parcae never faced. The E7 power-iteration picture stands unexplained by the
carry: the 3e4x first-iteration jump is a property of the composed block map, which the
diagonal carry does not touch.

Not verified: other inits for the widened dims (0.447 everywhere, or a learned Δ as in
Parcae's ZOH form); the widened carry under the ramp (it may still change the map's late
dynamics without touching the early detonation, which is what the constraint is for in
Parcae's 1.3B late-instability case); the four cancelled draws.

## Updated hypothesis

The early detonation is not a carry-spectrum event. Its remaining candidates are the block
write's scale (E7: block 5's σ_max 9e4 at iteration 0), which a bounded write (Fully-Looped
attention injection) or a directional hinge (E10b) addresses, and the ternary/optimizer
trigger, which no map-side term addresses. E10 reads the first; the second needs the
divergence README's open items (code-assignment hysteresis on the ternary cusp).
