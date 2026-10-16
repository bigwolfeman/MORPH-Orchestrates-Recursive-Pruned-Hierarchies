# Parcae minimal fan at 5k: the minimal slot loop with its 4 cells put back

Status: failure (predictions 2, 3 and 4 fail; filed 2026-10-14)

## Question

The minimal slot loop (`failures/2026-10-14-parcae-minimal-slot-loop-5k.md`) is fast (39.6k tok/s, 1.78x
strict LXTUL + pointer) and ties plain + pointer at depth 1, and it does not use depth: K1-K6 is 0.0000 in
all three runs. Its cell converges in about two passes (relative change 8.5 %, 0.45 %, 0.07 % per pass).
The old panel already held the reason, and the minimal prereg missed it: the one-cell `bare` arm, which
still had the gain hinge, the fixed-point term and the cotangent clip, was flat too (4.0826 at every
depth), and every 4-cell arm earned (ungraded fan +0.035, LXTUL +0.059, LXTUL with no span decoder
+0.133). So the stabilisers are not what buys depth on Parcae. The fan is.

Does the minimal loop with 4 cells per slot earn depth, at a CE close to plain + pointer, and how much
of the old arms' large K1-K6 comes from latent selection?

## Arms (all with the normalised pointer head, 5k steps, `lxtul.train` defaults)

| arm | config | seeds | change from minimal |
| --- | --- | --- | --- |
| minimal fan | `arm=minimal_fan` | 1, 2 | cells 4, select none: every cell written, the coda reads all 4 |
| minimal fan, latent selection | `arm=minimal_fan_lsel` | 1 | the above + router reset, winner-only write, EMA target |
| two-cell fan | `arm=minimal_fan arm.lxtul.cells=2` | 1 | cells 2 |

References, not re-run: plain + pointer seeds 1 / 2 (3.7155 / 3.7256, K1-K6 +0.0424 / +0.0377, pooled
3.7206 / +0.040); strict LXTUL + pointer 3.7608 / +0.0468; minimal 3.7647 / 3.7615, K1-K6 0.0000.

## Predictions

1. Minimal fan CE@6, mean of 2 seeds, in [3.735, 3.800] (60 %).
2. Minimal fan K1-K6, mean of 2 seeds, at least plain + pointer's pooled +0.040 (55 %); each seed's CI
   excludes 0 (80 %).
3. Latent selection: K1-K6 at least the minimal fan mean + 0.030 (55 %), and CE@6 no more than 0.030 above
   the minimal fan mean (55 %).
4. Two-cell fan: K1-K6 CI excludes 0 (60 %), and its K1-K6 is below the 4-cell mean (60 %).
5. Minimal fan run-log tok/s at least 31.0k, 1.40x strict LXTUL + pointer's 22.2k (65 %).
6. No non-finite loss, and no loss after step 1000 above 2x the median of the previous 100 steps (80 %).
7. Leak tripwire, not a prediction: CE@6 more than 0.30 below plain + pointer, or a head-off CE below
   the head-on CE, stops the panel and is filed as a leak.

## What each outcome changes

- Minimal fan K1-K6 at least plain's and CE gap at most +0.06: it is the slot-loop candidate for the flat
  references and the 40k prereg.
- Latent selection buys at least +0.03 K1-K6 for at most +0.03 CE: it is the candidate instead.
- Minimal fan K1-K6 below +0.02: something removed with the stabilisers mattered at 4 cells; add the
  fixed-point term back first (it is the only one that acts on every cell's last pass).

## Method

Code: `lxtul/minimal` 5717bc1 (worktree parcae-wt-min). One queue script
(`morph-scratch/abq/pz-panel/fan_5k.sh`), one job at a time under the GPU lock: minimal fan s1, its
head-off check, latent selection s1, its head-off check, minimal fan s2, two-cell fan s1. The two-cell fan
scores different tokens than the 4-cell rows, so its gap to plain uses `lxtul.paired_eval` (plain +
pointer s2 scored on the two-cell rows). Amendments get a date and a reason here.

## Results

Sweep CE on each run's own eval rows (4-cell rows 510,003 tokens; 2-cell rows differ), K1-K6 with its CI.
Head-off = pointer head off (`lxtul.mixture_off_eval`). Tok/s = median `perf/tok_s` of the run log.

| arm | seed | CE@1 | CE@6 | K1-K6 | head-off CE@6 | tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| minimal fan | 1 | 3.7447 | 3.7446 | +0.0000 [-0.0000, +0.0001] | 4.1498 | 32.7k |
| minimal fan | 2 | 3.7438 | 3.7438 | -0.0000 [-0.0001, +0.0000] | | 32.4k |
| minimal fan, latent selection | 1 | 3.7563 | 3.7563 | -0.0000 [-0.0001, +0.0001] | 4.1582 | 30.7k |
| two-cell fan | 1 | 3.7599 | 3.7599 | +0.0000 [-0.0001, +0.0001] | | 37.5k |

Two-cell fan against plain + pointer s2, scored on the two-cell rows (`lxtul.paired_eval`): depth 1
-0.0105 [-0.0131, -0.0081], depth 6 +0.0271 [+0.0247, +0.0294] (plain 3.7704 / 3.7328 on those rows).

Minimal fan mean of 2 seeds: CE@6 3.7442, K1-K6 0.0000. It is 0.0166 better than strict LXTUL + pointer
(3.7608) and 0.0236 behind plain + pointer's pooled 3.7206 (rows differ by width; the two-cell paired
number above is the like-for-like gap).

Losses: 201 logged steps per run, no non-finite value, no logged loss after step 1000 above 2x the median
of the previous 100 logged steps (the log interval is 25 steps).

Prediction check:
1. Minimal fan CE@6 mean 3.7442, inside [3.735, 3.800]. HOLDS.
2. Minimal fan K1-K6 0.0000; no seed's CI excludes 0. FAILS.
3. Latent selection K1-K6 0.0000 (needed +0.030); CE@6 +0.0121 over the fan mean. FAILS (the CE half
   holds, the K1-K6 half does not).
4. Two-cell K1-K6 CI includes 0. FAILS.
5. Minimal fan 32.7k / 32.4k tok/s, at least 31.0k. HOLDS.
6. No non-finite loss, no spike. HOLDS.
7. Tripwire not hit: head-off CE above head-on in both checked runs; no CE 0.30 below plain.

The hypothesis committed before the latent-selection result (`results/2026-10-14-parcae-minimal-fan-5k/
hypothesis-before-lsel.md`, a one-slot predictive target makes the loop use depth) is refuted by the same
run (`hypothesis-result.md`).

## Verdict

Failure. Four cells, two cells, and four cells with latent selection are all flat in depth, like one cell.
The fan does not buy depth on Parcae, and neither does latent selection without the span decoder and the
stabilisers. What the fan does buy is CE: the minimal fan is the best slot-loop arm on Parcae (3.7442), at
1.47x strict LXTUL + pointer's run-log rate. Every earning Parcae arm also carried epivol, the term that
pushes a slot's cells apart at passes 1 and 2 only.

## Updated hypothesis

The Parcae slot family's K1-K6 came from a pass-indexed term (epivol) or the stabilisers, not from the
cells or the selection; the epivol vs stabilisers split (`planned/2026-10-14-parcae-epivol-vs-stabilisers-5k.md`)
tests it. Independently, the minimal arms suggest the coda's 4 layers do the cross-span work over all
earlier slots, so the loop's first pass is enough; `planned/2026-10-14-parcae-shallow-coda-5k.md` tests
that with the main CE alone (Wolfe 2026-10-14: only the main CE may buy depth).
