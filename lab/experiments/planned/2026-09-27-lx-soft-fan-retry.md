# Planned: retry the two soft-credit LX-Fan arms that detonated (lxfan4, lxfan6)

Status: planned

Date: 2026-09-27 11:28 (frozen before either retry trained).
Parent: [`../failures/2026-09-26-lx-credit-arms.md`](../failures/2026-09-26-lx-credit-arms.md):
(a) `lxtul-lxfan4-fp01` detonated at 3311 and `lxtul-lxfan6-fp01` at 1545 after the pass-0 write
grew 14-42x; the three WTA fans (a2 at 5k and 10k, (b)) never detonated. Wolfe (2026-09-27):
retry the detonated runs. The first draws are kept as `<tag>-det1` (checkpoints and
`/home/wolfe/morph-scratch/abc/<tag>-det1/`, which holds the probe the filed anatomy read).

## Question

Is the soft fans' detonation structural, or one bad draw each? MORPH runs are not
bit-reproducible (they decorrelate within ~11 steps at a fixed seed), so a rerun of the same
config at the same seed is a new draw of the same distribution.

## Hypothesis

H: the detonation is structural for the soft fans: both retries detonate before step 5000.

## Predictions (mine, orchestrator, before any retry step)

- **P-1**: the lxfan4 retry detonates before 5000. 55 %.
- **P-2**: the lxfan6 retry detonates before 5000. 65 %.
- **P-3**: in any retry that detonates, `loop/core_gain_t0` (25-step median) exceeds 10 at
  least 300 steps before the tripwire. 70 %.
- **P-4**: if the lxfan4 retry survives, its gap to a2 at depth 6 (same tokens) is within
  +-0.0137. 50 %.
- **P-5**: if the lxfan4 retry survives, its K1-K6 exceeds a2's +0.0057. 55 %.

## Method

- Same configs, same code as the parent (`ce1b7dc` model code; this prereg's commit), seed 1,
  5000 steps, the parent's runner and readouts (`/home/wolfe/morph-scratch/abc/runner.sh`), a
  detonation guard with the parent's sustained-tripwire rule. Order: lxfan4, then lxfan6.
- Survivors: paired vs a2 5k and vs (b) `lxtul-lxfan4-wta-fp01` 5k on the 480 sweep rows.
- Every run's `core_gain_t0` / hinge / `preclip` trajectory is read as in the parent's
  `detonation_anatomy.txt`.

## Reading rules

- H holds only if P-1 and P-2 both hold. One of two surviving means the soft fans are
  unstable at some rate below 1; that is filed as such, not as "stable".
