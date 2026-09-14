# Four zero-training loop diagnostics (2026-09-14)

Prereg: [`lab/experiments/failures/2026-09-14-arc-loop-diagnostics.md`](../../failures/2026-09-14-arc-loop-diagnostics.md)
(moved from `planned/` — three of four predictions read false; see Verdict there).
Scripts: `lab/divergence/{aa_score,jacobian_gap_vs_k,sink_mass_per_pass,basin_map}.py`,
shared plumbing in `lab/divergence/_diag_common.py`. Run on the DGX Spark
(`ssh dgx-spark`, GB10), branch `loop-diagnostics-2026-09-14-v3`. 96 rows for
instruments 1-3, 6 (row, slot) pairs for instrument 4. `strict` =
`tul_slot_spandec_strict/step_5000.pt`, `plain` = `notul_norm_match_20k/step_5000.pt`,
both at step 5000. `run_all.log` (copied here as `run_all.txt`) is the full provenance
trail: every stage's start/end timestamp and exit code (all 0).

## 1. AA score (Anil et al. 2211.09961) — path independence

| arm | AA(swap) mean / median | AA(noise) mean / median |
| --- | --- | --- |
| strict (slot loop) | 0.8644 / 0.8749 | 0.8187 / 0.8227 |
| plain (noise entry) | 0.9777 / 0.9859 | 0.9998 / 0.9999 |

**P-1 FAILS on both halves.** Predicted strict AA > 0.9 with plain lower: strict sits
well BELOW 0.9 on both re-init constructions (swap 0.86, noise 0.82), and plain is
dramatically HIGHER, not lower (0.98-0.9998). The strict slot loop's own eval-depth
run is NOT close to path-independent — the loop visibly still depends on where the
core state started, which argues against the "already an uninformative fixed point"
reading the prereg leaned on. The plain arm's near-1.0 noise-AA is close to the
construction-risk flagged in the prereg: the plain entry is itself an independent
`N(0, std^2)` draw (`core_state_init: noise`), so its "once" and "noise-twice" runs are
two draws from the SAME distribution — a ~1.0 AA there is close to a tautology of the
model family, not evidence the loop converges to an informative fixed point. The swap
number (0.9777, a genuinely different row's converged state, not a re-drawn noise
sample) is the more informative plain reading and still says the readout barely moves
when the core entry is swapped for another row's own converged state.

## 2. Jacobian spectral gap vs K1-K6 (Wu/Zhang/Cao 2606.00605) — power method

| arm | sigma1 median | sigma2 median | ratio median | rho(sigma1, K) | rho(ratio, K) |
| --- | --- | --- | --- | --- | --- |
| strict | 5.9741 | 5.5219 | 1.0521 | +0.1106 | +0.0315 |
| plain | 31.6828 | 28.1683 | 1.1051 | -0.0510 | -0.0188 |

**P-2 FAILS.** Predicted strict ratio median > 3 with a negative Spearman correlation
against K1-K6: the measured ratio is 1.05, essentially NO spectral gap (the top two
singular values of the per-pass core map are nearly equal), and the correlation with
K1-K6 is weak and POSITIVE (+0.11), not negative. The 2026-08-24 "amplifying directions
align x2.9" reading does not show up here as a wide top-two gap at iteration 0 on THIS
checkpoint (strict geometry, spandec target, step 5000) — a narrow/absent gap is
consistent with the loop needing MANY iterations for the power-method story to apply,
not with the story explaining why this loop earns almost nothing in 6 passes. Read this
as: whatever caps the strict loop's earning, it is not "one direction dominates and the
rest is free" in the Wu/Zhang/Cao sense, at least not visible at iteration 0. The plain
arm (much larger sigma1, ~32 vs ~6, consistent with a noise-scaled entry into a
full-sequence forward) shows the same absent-gap pattern (ratio 1.11) with a weak
negative correlation — directionally closer to P-2's shape but nowhere near the
predicted magnitude, and this was never P-2's primary target (P-2 named the slot loop).

## 3. Sink mass per pass (SMELT 2609.01343 §6.4)

| arm | pass 1 | pass 2 | pass 3 | pass 4 | pass 5 | pass 6 | max rel dev (2-6) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strict | 0.0836 | 0.0842 | 0.0842 | 0.0842 | 0.0842 | 0.0843 | 0.09% |
| plain | 0.0214 | 0.0212 | 0.0211 | 0.0210 | 0.0210 | 0.0210 | 0.67% |

**P-3 HOLDS, strongly.** Predicted flat within 10% relative across passes 2-6: both
arms come in under 1% relative deviation, an order of magnitude flatter than the
prediction's own threshold. SMELT's own "extractability moves while CE doesn't" escape
hatch does not appear here — sink mass on the compact index-0 position (slot loop) or
token position 0 (plain) is essentially constant from pass 1 onward. This is the one
prediction that reads exactly as expected, and it corroborates the tree's existing
cancellation/K3-K6-flat readings from a fully independent angle (attention re-weighting,
not loss or state geometry).

## 4. Basin map (Lai et al. 2609.04963, Appendix B)

| arm | grid | pairs | mean entropy (nats) | mean frac differing from centre |
| --- | --- | --- | --- | --- |
| strict | 41x41 (disclosed as planned) | 6 | 0.5363 | 0.2519 |
| plain | 15x15 (**disclosed deviation**, see below) | 6 | 0.0000 | 0.0000 |

Per-pair breakdown, strict (the mean hides a bimodal split):

| pair | entropy | frac differing from centre | settling-time counts |
| --- | --- | --- | --- |
| 0 | 1.6085 | 0.9173 | {1:313, 2:66, 3:477, 4:499, 5:187, 6:139} |
| 1 | 1.6096 | 0.5943 | {1:682, 2:286, 3:241, 4:174, 5:161, 6:137} |
| 2 | 0.0000 | 0.0000 | {1:1681} |
| 3 | 0.0000 | 0.0000 | {1:1681} |
| 4 | 0.0000 | 0.0000 | {1:1681} |
| 5 | 0.0000 | 0.0000 | {1:1681} |

Per-pair breakdown, plain (all six pairs land on a single settling time, entropy 0):
counts are {4:225}, {2:225}, {1:225}, {1:225}, {4:225}, {1:225} — a different constant
settling time per pair, but each pair's own 15x15 grid is internally uniform.

**P-4 FAILS on strict, HOLDS on plain.** Predicted both arms flat with < 5% of grid
points differing from the centre point: plain matches this cleanly (0% differing, entropy
0). Strict does NOT — the mean (25.2% differing) is five times the threshold, but the
mean hides the real shape of the result: 4 of 6 (row, slot) pairs are perfectly flat
(entropy 0, 0% differing, exactly the pre-bifurcation / purely-contractive signature
the prereg predicted), while 2 of 6 pairs show real basin structure — entropy 1.6 nats
(close to the max possible for up to 6 settling-time buckets) and 60-92% of the 1681
grid points settling to a DIFFERENT depth than the unperturbed centre point. This is
the single most surprising number in the whole panel: it says at least some slots sit
near a genuine multistable/weakly-unstable regime in Lai et al.'s sense, not uniformly
in the flat regime the tree's gain readings (0.2-0.9, monotone convergence) predicted
for the checkpoint as a whole. Two pairs out of six is not enough to characterize what
distinguishes a structured slot from a flat one — that is the natural next question,
not answered here.

**Disclosed deviation:** `basin_plain` ran at grid=15 (225 points/pair) instead of the
prereg's 41x41 (1681 points/pair), reduced for measured wall-clock reasons — the plain
model's full-sequence, eager `_core_region` forward is far more expensive per grid
chunk than the slot loop's compact ~64-position core (confirmed on the Spark: jac_plain
ran roughly 3-4x slower per row than jac_strict for the same 96 rows). This was decided
live during the run, exactly as the prereg's own "Not verified before launch" section
flagged as a possibility, and is recorded in `run_all.sh`'s own comment (preserved in
`run_all.txt` here). A flat plain result at grid=15 cannot rule out structure that only
appears at a larger radius or finer grid than this smaller run reached; it is read as
"no structure detected at this grid," not "no structure exists."

## Cross-checks against the coordinator's independently-read headlines

jac strict ratio 1.052 rho 0.03 — matches (ratio median 1.0521, rho(ratio,K)=0.0315).
jac plain ratio 1.105 rho -0.02 — matches (ratio median 1.1051, rho(ratio,K)=-0.0188).
AA strict noise 0.82 swap 0.86 — matches (noise mean 0.8187, swap mean 0.8644).
AA plain noise 0.9998 swap 0.978 — matches (noise mean 0.9998, swap mean 0.9777).

## What this panel adds to the tree's existing "one pass does everything" reading

Three of four independent instruments (AA score, spectral gap, basin map on the strict
arm) do NOT corroborate the flat/fixed-point story the prereg leaned toward at 55/40/70%
confidence — each finds real structure the K-curve, rank and cosine instruments already
on this tree do not report: the loop is not path-independent (AA well below 0.9), the
per-pass map has essentially no spectral gap (ratio ~1.05, not the power-method
signature), and a minority of slots sit in a genuinely multistable basin (2 of 6 pairs,
entropy 1.6 nats). Only the sink-mass instrument (P-3) reads flat, and it reads flatter
than predicted. The binding section's "if any one fails, especially P-2 or P-3" branch
fires for P-2: a failing P-2 says the map's directions are NOT what caps this
checkpoint's loop earning, which does not reconcile cleanly with the 2026-08-24
alignment reading (x2.9) — that reading was taken at the TUL-takeover ONSET on a
different checkpoint family, not at a converged step-5000 strict-geometry checkpoint,
so the two are not directly comparable, but the disagreement is real and unresolved
here, not explained away.

## Not verified / caveats (No Theater)

* The AA-noise construction's calibration (`N(0, e.std()^2)`) remains a judgment call,
  as flagged in the prereg — not re-validated here.
* The basin map's radius (`--radius 1.0`) was not swept; the strict arm's 2-structured/
  4-flat split could shift at a different radius, and the plain arm's flatness at
  grid=15 could hide structure a finer grid or larger radius would find.
* Instrument 2's Jacobian is measured at iteration 0 only (not averaged over the loop's
  full trajectory).
* `--n-power-iter 20` was not calibrated against a real (as opposed to synthetic) core
  map spectrum; a narrower true gap could need more iterations to resolve cleanly, which
  would bias the measured ratio toward 1 (i.e. could partly explain, not fully explain,
  the near-1 ratios above).
* Everything here is EAGER (`model.use_kernels=false`, `model.tg_scoped_kernels=false`)
  on the Spark GPU — no fused-kernel cross-check was run.
* The strict-arm basin structure (2 of 6 pairs) is a small sample; which slot property
  predicts "structured vs flat" was not investigated in this panel.
