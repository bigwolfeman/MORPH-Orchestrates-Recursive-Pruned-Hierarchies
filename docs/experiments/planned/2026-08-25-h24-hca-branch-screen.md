# Experiment: H24 screen — does reviving the HCA compressed branch change the core's state geometry?

Status: planned

Ledger: `lab/divergence/takeover-campaign.md` H24.
Agent Note: [`.agents/notes/proposed/bug-fix/2026-08-25-hca-compressed-branch-dead-on-slot-path.md`](../../../.agents/notes/proposed/bug-fix/2026-08-25-hca-compressed-branch-dead-on-slot-path.md)
Source of the finding: [`2026-08-25-h18-positional-attention-sink.md`](../failures/2026-08-25-h18-positional-attention-sink.md) Phase 0.

## Question

Measured 2026-08-25: on the TUL slot path the HCA compressed branch is identically zero
(`hca_compress_ratio` 256 against 64 slot positions gives `n_blocks = 0`), so three of the
six core blocks output half of what they were built to output. The token path, at the same
weights, has a live branch. A1 (slots) diverges; A0 (tokens) does not.

Does turning that branch back on change the ONE forward quantity this campaign has
confirmed moves at the onset — the slot-state effective rank across the loop (H4)?

## Why this is cheap and why it is legitimate

`GatedPoolCompressor`'s projections are per-position linears; `m` enters in the reshape
that follows. The screen is two forward passes per rung at fixed weights.

**Method Amendment 1, 2026-08-25, written after the first attempt failed and before any
revived-arm number was read.** The paragraph above originally claimed `m` is in NO weight
shape. That was WRONG: `GatedPoolCompressor` carries `B_a` of shape `[m, c]`, a learned
gate bias per within-block position, and `model.hca_compress_ratio=16` fails to load the
checkpoint with seven size mismatches (log: `h24_screen.log`, `REVIVE exit=1`). The screen
therefore needs an approximation and a scope, both fixed here:

1. `B_a` is SLICED to its first `m_new` rows. Those rows are trained — every HCA block in
   prelude and coda uses all 256 at S = 1152 — but the slice repurposes a 256-wide
   positional gate as a 16-wide one. The screen is a forward-map probe of "what if this
   branch contributed", NOT a faithful model of the trained alternative. That weakens what
   a positive result can claim and does not weaken a negative one.
2. The surgery is confined to the CORE blocks. Rewriting prelude and coda would move the
   carrier that ENTERS the loop and would break validity gate V2 by construction.
3. Both arms run through the SAME script, `lab/divergence/h24_screen.py`, the control with
   `--control`, so they differ by the surgery alone.

The predictions below are UNCHANGED and were committed before any of this.

**Scope limit, stated before the run.** These weights were trained WITH the dead branch.
The screen measures what the branch does to the forward map at fixed weights. It cannot
show what training with the branch alive would do. A positive screen justifies the arm; it
does not replace it.

## Method

    PY=lab/divergence/jac_ladder.py
    OV=training.batch_size=6,model.use_kernels=false
    python $PY --state-probe --ckpt-dir checkpoints/morph/onset-capture \
        --overrides $OV --out h24_ctrl.json
    python $PY --state-probe --ckpt-dir checkpoints/morph/onset-capture \
        --overrides $OV,model.hca_compress_ratio=16 --out h24_revive.json

`jac_ladder.py --state-probe` is the instrument that produced H4. It reports, per loop
iteration, the participation ratio of the squared singular values of the `[S_valid, C]`
slot-state matrix, both raw and on UNIT-NORMALISED states, plus the mean pairwise cosine.

Statistic: `ratio = eff_rank_unit(last iteration) / eff_rank_unit(iteration 0)`. Above 1
the loop DIVERSIFIES the slot states; below 1 it collapses them.

**Rung classes**, fixed here, the same split H18 used and taken from the ladder README's
core-share column:

- HEALTHY: 1625, 1650, 1675, 1700, 1725, 1750, 1775
- SICK: 1800, 1850, 1866
- AMBIGUOUS, excluded from both: 1825

Published control values on unit-normalised states
(`docs/experiments/failures/2026-08-24-tul-takeover-cure.md`): 1625 = 1.41, 1700 = 1.28,
1750 = 1.47, 1800 = 0.92, 1850 = 0.71, 1866 = 1.06.

## Predictions

**Validity gate. Runs first, refuses the panel.**

- V1 the CONTROL arm reproduces the six published unit-rank ratios to within +/- 0.15. If
  it does not, this is not the same measurement and nothing below is readable.
- V2 `eff_rank_unit` at ITERATION 0 is identical between the two arms to within 1e-3.
  Iteration 0 is the carrier before the first core step, so a difference there means the
  config change leaked outside the core and the comparison is void.

**S1 direction.** The unit-rank ratio RISES in the revived arm at all three SICK rungs.

**S2 size.** The rise at rung 1850 — the cleanest sick rung, control 0.71, the only rung
where the campaign's two rank measures agreed — is at least +0.15.

**S3 crossing.** At rung 1850 the revived ratio is above 1.0, i.e. the loop stops
collapsing the states.

**S4 specificity.** The revived arm does NOT lower the ratio at any HEALTHY rung by more
than 0.15. A change that helps the sick rungs by breaking the healthy ones is not a fix.

**REFUTER.** If `|revived - control| < 0.05` at EVERY rung, the branch does nothing to the
state geometry at fixed weights. H24 then does not act through the mechanism H4 measured,
and the expensive arm is not justified on this evidence.

## What would make this inconclusive, and why that is a failure

A validity-gate failure is filed under `failures/` with the gate named, and the next
planned experiment fixes the gate. "The numbers were unclear" is not an outcome.

## Declared not verified

- fixed weights trained with the dead branch; this is a forward-map screen, not the arm
- seed 0 only, one batch
- the HCA compressor weights also serve the TOKEN path at `m = 256`; this screen changes
  `m` on BOTH paths, so it is not the same single-variable change the arm would be
