# Hop-distance probe, second pass: the plateau at every depth, and the near-bin loss

Status: failure

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
## Results

Four runs of `lab/divergence/hop_distance_probe.py` at `c813b1a` (the file shipped by scp onto
the 3070's `99bd7c9` checkout, md5 `51022c69` on both sides, recorded in
`run_hop_pass2.log`), 480 rows, batch 4, hops 6, planted, 2026-09-19. Artifacts:
`../results/2026-09-19-hop-distance-pass2/`. Intervals are the row bootstrap.

**Run 1: `prev-reach1`, pair, depths 1-8, 12, 16 (P1, P2, P4).** The corruption sweep and
the bins at the shared depths are identical to the first pass (same rows, same seed). The new
depths fill the curve:

| h | tokens | K1−K6 | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d12 | d16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 183k | +0.0109 | 4.2330 | 4.2281 | 4.2253 | 4.2237 | 4.2227 | 4.2221 | 4.2217 | 4.2218 | 4.2243 | 4.2300 |
| 2 | 73k | −0.0169 | 4.4800 | 4.4862 | 4.4931 | 4.4966 | 4.4974 | 4.4969 | 4.4960 | 4.4950 | 4.4924 | 4.4923 |
| 3 | 53k | +0.0640 | 4.5501 | 4.4849 | 4.4799 | 4.4830 | 4.4852 | 4.4861 | 4.4860 | 4.4857 | 4.4844 | 4.4841 |
| 4 | 44k | +0.0535 | 4.6011 | 4.5989 | 4.5578 | 4.5480 | 4.5472 | 4.5476 | 4.5479 | 4.5479 | 4.5479 | 4.5483 |
| 5 | 35k | +0.0288 | 4.5270 | 4.5321 | 4.5330 | 4.5101 | 4.5008 | 4.4982 | 4.4975 | 4.4976 | 4.4991 | 4.5005 |
| 6 | 33k | −0.0203 | 4.5054 | 4.5220 | 4.5312 | 4.5372 | 4.5310 | 4.5257 | 4.5228 | 4.5216 | 4.5218 | 4.5232 |

Planted pair (2778 sites; control 15.36 at d1, 15.40 at d6; benefit = control − planted):

| g | d1 | d2 | d3 | d4 | d5 | d6 | d8 | d12 | d16 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | +0.566 | +0.562 | +0.560 | +0.559 | +0.559 | +0.558 | +0.562 | +0.566 | +0.570 |
| 1 | +0.181 | +0.158 | +0.149 | +0.143 | +0.140 | +0.137 | +0.134 | +0.132 | +0.128 |
| 2 | +0.148 | +0.108 | +0.088 | +0.080 | +0.075 | +0.071 | +0.068 | +0.065 | +0.062 |
| 3 | 0.000 | +0.083 | +0.076 | +0.068 | +0.065 | +0.062 | +0.061 | +0.059 | +0.057 |
| 4 | 0.000 | 0.000 | +0.046 | +0.050 | +0.049 | +0.047 | +0.048 | +0.050 | +0.052 |
| 5 | 0.000 | 0.000 | 0.000 | +0.025 | +0.033 | +0.034 | +0.036 | +0.042 | +0.047 |
| 6 | 0.000 | 0.000 | 0.000 | 0.000 | +0.014 | +0.021 | +0.026 | +0.035 | +0.046 |

**Run 2: `prev-reach1`, pair, `--cut-after 1`, depths 1, 2, 3, 6 (P3).** Under the cut the
localiser reads ΔCE exactly 0 at every hop ≥ 3 (own 0.53, ΔCE(1) 0.24, ΔCE(2) 0.008), so the
bins are a different partition of the same tokens (h = 3 becomes a 101k catch-all) and only
within-run depth contrasts are read:

| bin (cut) | tokens | K1−K6 | d1 | d2 | d3 | d6 |
|---|---|---|---|---|---|---|
| 1 | 237k | +0.0076 [0.0064, 0.0088] | 4.2992 | 4.2945 | 4.2942 | 4.2916 |
| 2 | 83k | −0.0831 [−0.0854, −0.0808] | 4.4377 | 4.4874 | 4.5084 | 4.5209 |
| 3 | 101k | +0.0008 [−0.0013, 0.0028] | 4.6057 | 4.5959 | 4.5969 | 4.6049 |

Planted under the cut (control 15.3625 / 15.3387 / 15.3287 / 15.3235 at d1 / d2 / d3 / d6):
g = 0 +0.566 / +0.577 / +0.582 / +0.585; g = 1 +0.181 / +0.245 / +0.272 / +0.291;
g = 2 +0.148 / +0.073 / +0.050 / +0.035; g = 3 and g = 4: bit-identical to the control at
every depth (benefit 0.000).

**Run 3 as launched: `--seed 1` (kept as `hop2_prev-reach1-seed1_5000.json`).** Bins identical
to seed 0 to four decimals (same token counts 182993 / 72845 / 53132 / 43722 / 35214 / 33354,
same K1−K6). The seed never chose the rows (Method amendment 1). Re-run as
`--row-offset 480 --seed 1` (`hop2_prev-reach1-rows480_5000.json`), read under P5 below.

**Run 4: strict null, pair, depths 1 and 6 (P4's g = 0 on the null arm).** 2778 sites,
control 15.43 / 15.44. Benefits at d1 / d6: g = 0 +0.670 / +0.669; g = 1 +0.119 / +0.120;
g = 2 +0.082 / +0.083; g = 3 +0.066 / +0.067; g = 4 +0.058 / +0.059; g = 5 +0.049 / +0.051;
g = 6 +0.049 / +0.050. Every |benefit(6) − benefit(1)| ≤ 0.0016: on the arm whose coda reads
every cell directly, distance costs sit in the read and depth moves nothing.

Scorecard:

- **P1 (plateau): three clauses of four.** h = 4 settles by pass 3 (CE(3) − CE(6) = 0.0102 ≤
  0.0107, at the edge). h = 5 puts 79 % of its fall at pass 4 (0.0229 ≥ 0.0173, holds) but
  keeps falling through pass 5 (CE(4) − CE(6) = 0.0119 > 0.0058, FAILS): arrival spreads over
  passes h − 1 and h. h = 6 never falls below CE(1) (4.531 at pass 5 vs 4.505, holds).
- **P2 (dilution timing): h = 2 holds, h = 1 fails both clauses.** h = 2 rises 0.0131 by pass
  3 (≥ 0.008) and 0.0037 more by pass 6 (< 0.005). h = 1 does not rise: it falls 0.0049 by
  pass 2 (bar: within 0.003) and 0.0044 more by pass 4 (bar: rise ≥ 0.005), and only turns
  up at depths 12 and 16.
- **P3 (mechanism): FAILS in the direction the prereg named.** Under the cut, the h = 2 bin's
  CE rises 0.083 from depth 1 to 6 with nothing arriving from farther (bar: within 0.004),
  and planted g = 2 decays 0.148 → 0.035. The h = 1 bin does not lose (it gains 0.0076) and
  planted g = 1 grows 0.181 → 0.291. The h = 3 fall is gone (+0.0008, holds). Loop-carried
  content decays under the cell's own passes; own-span content, re-supplied by the injection
  every pass, is refined. The "dilution" wording of the 2026-09-18 record is withdrawn for
  the carried content and kept only for the displacement of g = 1 by later arrivals.
- **P4 (planted pair): three clauses of four.** g = 0 reads 0.56 (prev-reach1) and 0.67
  (strict) at every depth (bar 0.30). Exact zeros before pass g − 1 for every g. g = 3 at
  pass 2 is 0.083 (≥ 0.08, holds); g = 4 at pass 3 is 0.046 (FAILS).
- **P5 (rows): HOLDS on the disjoint draw** (`--row-offset 480 --seed 1`; token counts
  179355 / 70834 / 51566 / 42469 / 34978 / 33418, so the rows did change; localiser own 0.51,
  ΔCE(1) 0.19). Every bin lies inside the first draw's widened interval with the same sign:
  h = 1 +0.0127, h = 2 −0.0120, h = 3 +0.0665 (fall at pass 2), h = 4 +0.0545 (pass 3),
  h = 5 +0.0302 (between 3 and 6), h = 6 −0.0182. The single-token planted table repeats
  the exact zeros and the g − 1 arrival (g = 3 +0.082 at pass 2, g = 4 +0.050 at pass 3,
  g = 5 +0.033 and g = 6 +0.022 at pass 6; g = 0 0.19).

## Verdict

Failure by the filing rule (P1, P2, P3 and P5 had to hold; P2 and P3 did not), and a settled
answer to each of the three questions:

1. **Arrival is a step and a half.** Content h spans back arrives at pass h − 1 and the bin
   keeps improving for about one more pass (h = 5: 79 % of the fall at pass 4, the rest at
   pass 5), then holds. The plateau exists; its edge is one pass wider than predicted.
2. **The loss is not dilution, except for the nearest cell.** With the reach cut so that
   nothing arrives after pass 0, content the loop carried in decays under the cell's own
   later passes (planted g = 2: 0.148 → 0.035; the h = 2 bin: +0.083 nats worse by depth 6),
   while own-span content, re-supplied by the injection each pass, improves (g = 1: 0.181 →
   0.291). Uncut, g = 1 decays instead (0.181 → 0.137): later arrivals displace it. So the
   "dilution" wording of the 2026-09-18 record is withdrawn for carried content and kept for
   the displacement of the cell's own content. The rejection rule for P3 named this outcome
   and it is now the filed mechanism: **per-pass decay of loop-carried content.**
3. **The staircase is the arm's, not the rows'.** A disjoint 480-row draw reproduces every
   bin inside the first draw's interval with the same sign.

What the method could not do, for the record: the run-3-as-launched mistake (the seed never
chose the rows) cost one 40-minute run and is kept as `hop2_prev-reach1-seed1_5000.json`;
the cut re-bins the tokens (ΔCE beyond h = 2 is exactly 0), so only within-run depth
contrasts are read from it; and the depth-16 planted row (every g ≥ 2 near 0.05) is an
observation with no clause, read as the cell converging toward a distance-agnostic mixture.

## Updated hypothesis

The strict slot loop carries content one cell per pass, and the recurrent state loses part
of what it carried at every later pass while it keeps refining what its own span injects. At
5000 steps the decay of carried content is roughly a third per pass over the first three
passes (g = 2 planted: 0.148, 0.108, 0.088, 0.080), which is why h = 3 earns 0.064, h = 4
0.054, h = 5 0.029 and h = 6 nothing net: the farther the source, the more passes its content
spends decaying before the coda reads it. This is the same shape as the toy's untrained
six-hop carry (`../successes/2026-09-19-toy-eliminate6-hop-distance.md`) and it names the
lever: something that lets a cell KEEP what it read from its neighbour, on the same footing as
the injected own-span content. Candidates, in the order to try: a persistent slot component
written once on arrival and not re-processed (a gated carry), a per-pass write gate on the
cross-cell read, and width. Every candidate is scored on the per-hop K-curve and the planted
pair's decay row (g = 2 across depths), not on a whole-arm mean.
