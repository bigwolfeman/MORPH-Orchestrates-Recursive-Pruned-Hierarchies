# Planned: ARC E18 — the slot-channel width sweep: does a wider slot close the mask arm's gap to the plain model?

Status: planned
Date: 2026-09-08 (frozen before launch; Wolfe: "We could also try increasing the width." →
"okay do the sweep then"). Arc: `2026-09-04-loop-contribution-arc.md`, row E18. Launch is
chained behind E17's completion in the queue (`arc/run_e18_chain.sh`).

## Question

The masked arm (`tg_restrict`: a token reaches an earlier span ONLY through that span's slot
positions) is the one arm where the slot is load-bearing by construction, and it is behind the
plain model everywhere: 0.09–0.36 nats on web text (08-27), 0.11 behind mnext (E13), 0.185
behind plain on clean math (E16), depth-flat on Sudoku (E17). Every one of those arms carried
the same channel: `prefix_k = 2` positions per slot, one slot per span of ~12 tokens on web
text. The restriction itself is verified (`tests/test_tg_restrict.py`: severed channel ⇒
exactly zero gradient from a later span to an earlier token; unrestricted ⇒ nonzero; intact
restricted ⇒ nonzero — 3 of 3 on 2026-09-08). Nothing has ever varied the channel's width.
E18 asks whether the gap is the width: prefix_k 2 (the arm as run), 4 and 8, against the plain
control, at E13's recipe, scored on depth contribution AND on CE at matched inference cost.

## Hypothesis

H18: width helps but does not close the gap — CE improves monotonically with prefix_k by a few
hundredths of a nat and the widest arm still sits > 0.05 nats behind the plain model at its
trained depth. H18′ (the lever): k8 comes within 0.05 nats of plain, paired. H18″ (the null):
k8 − k2 < 0.01 nats, the width is not the constraint.

## Method

Arms (configs `tul_m12_mask_k{2,4,8}.yaml` = `tul_m12_mnext_mask` + `prefix_k` + scoped
kernels; `notul_e18.yaml` = `notul` at the same length and batch): E13's mean-12 recipe (slot
Poisson mean 12 / max 16, full BPTT, hinge 0.9, fixed-point 1.0, ramp 1000, flat 1e-4), seq
1024, batch 6, 5,000 steps, web text, checkpoints at 2,500 and 5,000. `L_total` = 1024 +
prefix_k × 64 = 1152 / 1280 / 1536. Order: k2, k4, k8, notul. Runner `arc/run_e18.sh`
(12-step smoke per arm — the k4/k8 smokes are the first time a wider prefix runs end to end;
a failed smoke skips that arm and is recorded), the sustained tripwire, then
`lab/divergence/core_depth_sweep.py` at both checkpoints, 480 rows, batch 3, depths
1,2,3,6,9,12,16 → `results/2026-09-08-arc-e18/sweep_<arm>_<ck>.json`. Commit pinned in
`arc/E18_COMMIT`.

**The second metric, inference cost.** Block-passes per row as the FLOPs proxy (attention
and MLP cost scale with positions × blocks; the compact slot core is cheap):

    plain,  core depth T:   1024 × (3 + 6·T + 3)                      = 6144 + 6144·T
    mask-k, slot depth T':  L_total × (3 + 3) + 64 × 6 × T'          = 6·L_total + 384·T'

so k2 at T' = 16 costs 13,056, k8 at T' = 16 costs 15,360, plain at T = 1 costs 12,288 and at
T = 2 costs 18,432. Every mask arm at every slot depth costs less than plain at depth 2; the
matched-cost comparisons are mask-k@T' against plain@1 and plain@2. Scored from the sweep
JSONs by `score_e18.py` (written after launch; reads only JSONs and these constants).

Analysis: between-arm CE differences paired over the same 480 rows with 2,000-draw
bootstrap CIs; K-differences per arm; `loss/gain_est` and the hinge fraction from the probes.

**Method amendment 1 (2026-09-08 23:20, k2 and k4 scored, k8 and notul not yet).** Two
facts about the sweep tool, found while writing `score_e18.py`:

1. *Rows do not pair across cuts.* Every arm scores the same validation stream, but a row
   of 1024 + 64·k positions holds a different number of tokens at each width (k2 mean
   1044, k4 mean 1064.8 per row), so the rows drift apart along the stream: the row means
   of k2 and k4 correlate at 0.098 across the 480 rows against 1.000 within an arm across
   depths. "Paired over the same 480 rows" in Method holds WITHIN an arm (the K-differences)
   and between arms of the same width only; every cross-width and mask-versus-plain
   comparison on the row sums is unpaired, with a floor of about ±0.06 nats. Fix:
   `core_depth_sweep.py` now writes a per-token CE array keyed by stream index beside its
   JSON (shared packer `lab/divergence/_rows.py`, the one `olympiad_sweep.py` uses), and
   `score_e18.py` pairs at the token level with a block bootstrap over 1,024-token stream
   blocks. The k2/k4 sweeps predate the change and are re-run from their checkpoints; the
   K-differences already on disk are unaffected.
2. *The plain arm's sweep would crash.* `notul_e18` builds no TUL runtime and
   `core_depth_sweep.py` dereferenced it unconditionally; the same change gives it the
   plain path (the trainer's cut, `cfg.mean_depth` as the depth lever, as
   `token_depth_sweep.py` does). Its first result is checked against the trainer's
   held-out loss at the same step before it is read (the E17 rule).

The predictions are untouched; P18b and P18d are scored on the token-paired numbers.

## Predictions (frozen)

- **P18a (survival).** k2 **90 %**, k4 **85 %**, k8 **80 %**, notul **95 %** reach 5,000 with
  the tripwire silent. The k4 and k8 smokes pass: **80 %** each.
- **P18b (width).** Token CE at the trained depth (12) at 5,000 improves monotonically k2 →
  k4 → k8 with each step > 0.01 nats, paired: **45 %**. k8 − k2 < 0.01 (the null, H18″):
  **30 %**. k8 within 0.05 nats of notul at depth 6, paired (H18′): **25 %**.
- **P18c (depth).** Any mask arm's tokens K3−K6 > 0.005 with the CI above 0 at 5,000:
  **20 %** (E13: 0.002). All three arms' K1−K6 > 0.03 (the depth-1 hole persists at every
  width): **80 %**.
- **P18d (matched cost).** k8 at slot depth 12 beats notul at core depth 1 in token CE,
  paired: **30 %**; beats notul at depth 2: **15 %**. Any width beats notul at depth 1:
  **35 %**.
- **P18e (the hinge at width).** k8 `loss/gain_est` mean over 1,000–5,000 under 0.9 and the
  penalty nonzero on under 50 % of steps: **40 %** (E16 mask: 80 %; E17 mask: 27 %).
- **P18f (cost).** Each mask arm ≤ 2.5 h (E13's eager mask: 112 min; scoped is faster, k8 has
  1.33× the positions): **60 %**. k8 peak allocated under 26 GB: **60 %**.
- **P18g (the order).** Final val CE orders notul < k8 < k4 < k2: **40 %**.

## Binding

- H18′ (P18b's third clause TRUE) ⇒ the channel width was the constraint all along: the next
  run widens further (k16, or two slots per span) and revisits the write side at width.
- H18″ (k8 − k2 < 0.01) ⇒ width is not the lever and the load-bearing slot has no remaining
  free parameter in this family: the slot-loop lane is closed for good; the record says so
  and the next work is the paid loop / write-back or something else entirely (Wolfe's call).
- P18d TRUE at any width ⇒ the arm survives as an EFFICIENCY design (less inference cost at
  equal CE) even if it never beats plain at plain's trained depth; the matched-cost curve
  becomes a standing metric on every future arm.
- A k4/k8 smoke FAILURE ⇒ the arm is skipped, the failure is recorded here, and the sweep is
  scored on what ran; the wider prefix's bug is fixed in a separate change.
- P18a FALSE on a mask arm ⇒ its trip step and gain instruments go into the divergence
  README's second-hold table.
- NO 20k run from this experiment.

## Not verified before launch

Any forward with prefix_k ≠ 2 (the smokes are the first); memory at L_total 1536 with the
eager restricted branches at batch 6; `core_depth_sweep.py` on a prefix_k 4/8 checkpoint;
whether the bigram hash's one-token crack at span starts (the previous span's boundary token)
matters at any width — left unchanged so E18 compares to every prior mask arm.

## Results

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
