# Planned: retry the two soft-credit LX-Fan arms that detonated (lxfan4, lxfan6)

Status: failure

Date: 2026-09-27 11:28 (frozen before either retry trained).
Parent: [`../failures/2026-09-26-lx-credit-arms.md`](2026-09-26-lx-credit-arms.md):
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

## Results (filed 2026-09-27 16:18)

Code `591d99a`, seed 1, 5000 steps, the parent's runner; 480 rows at depth 6. Artifacts:
[`../results/2026-09-27-lx-soft-fan-retry/`](../results/2026-09-27-lx-soft-fan-retry/).

| run | outcome | gap to plain | K1-K6 | K3-K6 | worth zero / shuffle |
|---|---|---|---|---|---|
| lxfan4 draw 1 (parent) | DETONATED 3311 | | | | |
| **lxfan4 draw 2** | 5000, HEALTHY (max 222) | +0.2495 [+0.2345, +0.2665] | +0.0161 [+0.0151, +0.0170] | +0.0024 [+0.0021, +0.0028] | 0.195 / 0.173 |
| lxfan6 draw 1 (parent) | DETONATED 1545 | | | | |
| lxfan6 draw 2 | DETONATED 2113 (guard kill, max 8.16e4) | | | | |

Paired on the same tokens at depth 6 (`paired_vs_ruler.py`), lxfan4 draw 2 against:

| ruler | arm - ruler @6 | arm @1 - ruler @6 |
|---|---|---|
| (b) lxfan4-wta (the one-factor WTA pair) | **+0.0303 [+0.0280, +0.0325]** | +0.0464 |
| a2 (fan4-all-fp01) | +0.0317 [+0.0292, +0.0342] | +0.0477 |
| fp01 | -0.0157 [-0.0185, -0.0128] | +0.0002 |

Val at 5000 (not graded), against (b): `fan/single_ce - fan/mixed_ce` 0.164 (b 0.049);
`fan/oracle_gap` 0.121 (b 0.088); `slot_cell_eff_rank` 2.22 (b 2.72); `fan/stream_rank_t1` /
`_t6` 2.80 / 1.97 (b 2.92 / 2.54).

Anatomy ([`retry_anatomy.txt`](../results/2026-09-27-lx-soft-fan-retry/retry_anatomy.txt)):
lxfan6 draw 2's pass-0 write passed 10x (25-step median) at 1295 and reached 35-52x; the hinge
reading crossed 1.0 at 1264 and it charged 32-42 per step from ~1600 until the tripwire at 2113,
the same course as draw 1. lxfan4 draw 2's pass-0 write passed 10x at 1930 and PLATEAUED at
12-14x to step 5000 with the hinge at 0.95-0.97 and the per-slot maximum at 1.25-1.31; draw 1
reached 14-21x and its per-slot maximum climbed to 4.5 before 3311. The WTA fans hold it near 3x.

## Clause by clause

- **P-1 fails.** lxfan4 draw 2 did not detonate.
- **P-2 holds.** lxfan6 draw 2 detonated at 2113.
- **P-3 holds.** lxfan6 draw 2's pass-0 write passed 10x at 1295, 818 steps before the tripwire.
- **P-4 fails.** lxfan4 - a2 = +0.0317, outside +-0.0137.
- **P-5 holds.** K1-K6 +0.0161 > a2's +0.0057.
- **H fails** (P-1).

## Verdict

Failure on H: the soft 4-cell fan is unstable at some rate (1 of 2 draws), the soft 6-cell fan
detonated 2 of 2. The surviving draw gives the one-factor WTA reading under the codes, and it
splits the two effects cleanly:

- **WTA buys CE, not depth.** Without WTA the model is 0.0303 nats worse at depth 6, with the
  same depth use (K1-K6 +0.0161 vs +0.0162; K3-K6 +0.0024 vs +0.0020).
- **The codes on the fan buy depth, not CE.** Both code fans read K1-K6 ~0.016 against a2's
  +0.0057; (b) matches a2's CE.
- **Without WTA the cells blur into a committee.** A single cell read alone is 0.164 nats
  behind the joint read (with WTA 0.049), the cells are less distinct (rank 2.22 vs 2.72), and
  by pass 6 the streams have collapsed further (1.97 vs 2.54 of 3).
- Every soft-fan draw grew its pass-0 write past 10x; every WTA fan held it near 3x.

One surviving draw, 5000 steps: read directions, not sizes.

## Updated hypothesis

A per-cell training signal (WTA) is what keeps the fan's cells individually useful, and that is
where the fan's CE comes from; the depth use comes from the LX codes and does not need WTA.
Next: (b)'s second seed, and a2 with WTA 0 and no codes (the WTA effect without the codes).

## Amendment 2026-09-28

"Every WTA fan held it near 3x" was one draw. (b) at seed 2 grew its pass-0 write almost
linearly to 23x by step 5000 and did not detonate (tripwire max 38.5):
[`../successes/2026-09-27-lxfan-wta-seed2.md`](../successes/2026-09-27-lxfan-wta-seed2.md). The
second seed also reads K1−K6 +0.0093 against seed 1's +0.0162, so the K1−K6 seed spread on this
recipe is about 0.007. The soft retry's +0.0161 against (b)'s +0.0162 is well inside that
spread; "the codes buy the depth" rests on both arms with codes beating a2's one draw.
