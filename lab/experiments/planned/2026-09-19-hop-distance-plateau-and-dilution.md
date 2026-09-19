# Hop-distance probe, second pass: the plateau at every depth, and the near-bin loss

Status: planned

Date: 2026-09-19. Follows `../failures/2026-09-18-hop-distance-earning.md`, which found a
hop staircase on `slot-spandec-strict-prev-reach1` @ 5000 but could not read its plateau
clause (depths 4 and 5 were not swept), ran a planted instrument five times weaker than its
bars assumed, and could not separate "farther content overwrites nearer content" from "every
pass degrades what the cell holds". Written before any of the three runs below.

## Question

1. On `prev-reach1`, does a token h spans from its source finish earning at pass h − 1 and
   stop there (a plateau), or does it keep moving after arrival?
2. Is the loss with depth in the near bins (h = 1, 2) and the far bin (h = 6) caused by content
   from farther spans arriving (dilution), or by the map degrading held content at every pass
   regardless of what arrives?
3. Does the staircase reproduce on a second draw of rows (seed 1)?

## Hypothesis

Arrival is a step: the h-bin's CE falls between pass h − 2 and h − 1 and moves by less than a
fifth of that fall afterwards. The near-bin loss is dilution: it appears exactly when the
first farther span arrives (pass 2 for h = 1, since the coda reads cell j − 1 and the h = 2
span arrives there at pass 1, the h = 3 span at pass 2), and it disappears when arrivals are
blocked. The staircase is a property of the arm, not of the rows.

## Predictions (frozen)

Same instrument, same checkpoint, `--depths 1,2,3,4,5,6,7,8,12,16`, 480 rows, seed 0 unless
stated. "Fall" = CE(1) − CE(6) for the bin.

- **P1 (plateau).** For h = 5: `CE(4) − CE(6) ≤ 0.20 · (CE(1) − CE(6))` and
  `CE(3) − CE(4) ≥ 0.60 · (CE(1) − CE(6))` (most of the fall happens at pass 4). For h = 4:
  `CE(3) − CE(6) ≤ 0.20 · (CE(1) − CE(6))`. For h = 6: the bin does not fall at pass 5:
  `CE(5) ≥ CE(1) − 0.005`.
- **P2 (dilution timing).** In the h = 1 bin, CE is within 0.003 of CE(1) at pass 2 and rises
  by at least 0.005 between pass 2 and pass 4. In the h = 2 bin, CE rises by at least 0.008
  between pass 1 and pass 3 and by less than 0.005 between pass 3 and pass 6.
- **P3 (dilution mechanism).** A new probe option that zeroes the loop's read of cell j − 1
  at every pass after pass p (`--cut-after p`, so no content from more than p + 1 spans back
  can arrive) with p = 1: the h = 1 and h = 2 bins' CE at depth 6 is within 0.004 of their CE
  at depth 1 (the loss is gone), while the h = 3 bin's fall is also gone (K1−K6 within
  ±0.010). If the h = 1, 2 loss persists under the cut, the map degrades held content and P3
  fails.
- **P4 (planted, stronger signal).** With a two-token planted copy (`--planted-len 2`), the
  g = 0 benefit at depth 6 is ≥ 0.30 nats on every arm (was 0.11 to 0.16). On `prev-reach1`
  the arrival depth stays g − 1 with the exact zeros before it, and `benefit(g − 1)` for
  g = 3, 4 is ≥ 0.08.
- **P5 (rows).** Seed 1 on `prev-reach1`, depths 1,2,3,6: the six bins' K1−K6 lie inside the
  seed-0 bootstrap intervals widened by 0.005 on each side, with the same signs.

Rejection of the reading: if P5 fails, the staircase is a property of the 480 rows and the
whole 2026-09-18 result is rejected. If P3 fails, the "dilution" wording in the 2026-09-18
updated hypothesis is withdrawn and the loss is filed as per-pass degradation.

## Method

- Host: 3070, `/mnt/bigdata/morph-instruments/MORPH`, checkpoint
  `slot-spandec-strict-prev-reach1_step_5000.pt` (already there, md5 as in the first pass).
- Two probe changes BEFORE the runs, each with a test in `tests/test_hop_distance_probe.py`:
  `--planted-len N` (plant N consecutive rare ids; the copy target is the second one, so the
  first is the cue) and `--cut-after p` (a forward hook that zeroes the loop's cross-cell read
  for passes > p; it must reproduce the uncut numbers at p ≥ max depth, asserted by the
  test). Commit both, then run.
- Run 1: depths 1..8,12,16, seed 0, planted-len 2 (P1, P2, P4). Run 2: same with
  `--cut-after 1`, depths 1,2,3,6 (P3). Run 3: seed 1, depths 1,2,3,6, single-token planted
  (P5). Strict null re-run only for P4's g = 0 clause (depths 1,6).
- Score with the same scorer, extended for the new clauses; file under `successes/` only if
  P1, P2, P3 and P5 all hold.

### Method amendment 1 (2026-09-19 16:20 UTC, after runs 1-3, before run 3 is re-run)

Run 3 as launched (`--seed 1`) returned bins IDENTICAL to seed 0 to four decimals (same
token counts, same K1−K6): the probe's `--seed` feeds only the planted source positions and
rare ids, and the rows are always the validation stream from its start. So the P5 run
measured nothing about rows. The probe gains `--row-offset N` (skip N rows' worth of the
stream before packing, a disjoint later stretch of text; `c813b1a` → this change), and P5
is re-run as `--row-offset 480 --seed 1`, depths 1, 2, 3, 6, single-token planted. The P5
prediction text is unchanged: the seed-1 clause is read on the offset draw. The identical
seed-1 JSON is kept in the results directory as the record of the mistake.

## Risks

- `--cut-after` changes the forward; if the arm's `loop_reach` plumbing does not expose a
  single read site, the option costs more than a day and the P3 clause is dropped by an
  amendment before any run, not after.
- Every depth from 1 to 8 doubles the corruption sweep's forwards; the run is about 4 h on
  the 3070.
