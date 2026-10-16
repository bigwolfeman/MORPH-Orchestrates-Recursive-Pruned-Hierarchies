# Parcae minimal slot loop at 5k: remove the MORPH machinery, keep the strict loop

Status: failure (written 2026-10-14 before any run of the arm, committed 9f85db9; results filled 2026-10-14)

## Question

Wolfe (2026-10-14): "Slot loop is the paper." The Parcae-LXTUL recipe carries machinery that solved MORPH's
problems (ternary spikes, a flat loop, cell selection), not Parcae's. It costs FLOPs: LXTUL + pointer is
21.5 TFLOP/step, of which the loop core is 10.8 (about 3.3 useful), the span decoder and its CE 3.4 and the
EMA twin 0.8 (`morph-scratch/abq/parcae-speed/REPORT.md`). It also costs loop contribution: without the span
decoder the strict loop earned K1-K6 +0.1325 against +0.0585 (`failures/2026-10-06-parcae-lxtul-round2.md`).
Open geometry is not an option: it drops K1-K6 to +0.0042.

Does the minimal strict slot loop keep CE close to plain Parcae, earn more depth than plain, and run much faster?

## Arm

`arm=minimal`: strict geometry, ONE cell per slot (rows of 1024 + 64 positions), Parcae's diagonal injection,
token-state dropout 0.15, normalised pointer head (4 heads, output-only). Removed, with no compute and no
parameters: span decoder, gain hinge (and its 2 passes), fixed-point term, cotangent clip, latent selection
(router, EMA twin), epivol. Speed keys: pad-slot compaction, static depth, CE on loss rows, no pass
checkpoints. `arm=minimal_noptr`: the same with no pointer head.

## Runs (5k, 480 paired rows, `lxtul.train` defaults)

| run | seeds | reference |
| --- | --- | --- |
| minimal | 1, 2 | |
| minimal, no pointer | 1 | strict LXTUL 4.1242 / K1-K6 +0.0585; no span decoder 4.1304 / +0.1325 |
| plain + pointer, normalised | 2 | seed 1 is `plain-pointer-norm-5k`: 3.7155, K1-K6 +0.0424 |

Then the head-off check (`lxtul.mixture_off_eval`) on both minimal checkpoints. Strict LXTUL + pointer
(3.7608, K1-K6 +0.0468) is the old-recipe reference; it is not re-run.

## Predictions

1. Minimal CE@6, mean of 2 seeds, in [3.730, 3.800] (60 %).
2. Minimal gap to plain + pointer (pooled) at most +0.060 (60 %).
3. Minimal K1-K6 at least +0.060, above plain + pointer's +0.042 (60 %).
4. Minimal, no pointer: K1-K6 at least +0.080 and CE@6 within 0.03 of strict LXTUL's 4.1242 (55 %).
5. Minimal trains at least 1.6x strict LXTUL + pointer's run-log tok/s (22.2k), i.e. 35.5k tok/s or more (60 %).
6. No non-finite loss, and no loss after step 1000 above 2x the median of the previous 100 steps (the hinge,
   clip and fixed-point term are gone; Parcae is bf16 Muon, not ternary AdEMAMix) (80 %).
7. Leak tripwire, not a prediction: a minimal CE@6 more than 0.30 below plain + pointer, or a head-off CE
   below the head-on CE, stops the panel and is filed as a leak.

## Method

Builder gate first (`morph-scratch/abq/pz-min/gpu_todo.sh`: geometry tests with the minimal keys, a
bit-identity check of the merged tree against the pre-merge LXTUL trace, 300-step benches, a 600-step
seed-2 stability read, FLOPs). Then one queue script, one job at a time under the GPU lock. Amendments get
a date and a reason here.

## Results

Code: `lxtul/minimal` 08d43b9. Artifacts: `docs/experiments/results/2026-10-14-parcae-minimal-slot-loop-5k/`.
The head-off check crashed in the panel (`mixture_off_eval` built MORPH's 4-cell rows, 1280 positions,
for a 1-cell model, 1088); fixed in e8ede0a (`load_run` returns the run's own runtime) and run after.

| run | CE@1 | CE@6 | K1-K6 [95 % CI] | head-off CE@6 | run-log tok/s |
| --- | --- | --- | --- | --- | --- |
| minimal s1 | 3.7646 | 3.7647 | -0.0001 [-0.0001, -0.0000] | 4.1709 | 39.4k |
| minimal s2 | 3.7615 | 3.7615 | -0.0000 [-0.0001, +0.0000] | 4.1672 | 39.8k |
| minimal, no pointer s1 | 4.1265 | 4.1266 | -0.0000 [-0.0001, +0.0000] | | 46.3k |
| plain + pointer s2 | 3.7633 | 3.7256 | +0.0377 [+0.0366, +0.0388] | 4.1180 | |

Plain + pointer s1 (3.7155, +0.0424) pools to 3.7206 / +0.040. Run-log tok/s is the median of the
24-step windows after step 500.

Pairing. A 1-cell run's eval rows hold other tokens than a 4-cell run's: the packer fills unused slot
positions with tokens (494,924 vs 510,003 scored tokens on 480 rows), so the train-time sweeps are not
paired. `lxtul.paired_eval` (5717bc1) scores plain + pointer s2 on the minimal rows: 3.7646 @1, 3.7273
@6 (3.7256 on its own rows). Minimal - plain, paired: @1 +0.0000 [-0.0027, +0.0025] and -0.0031
[-0.0060, -0.0004]; @6 +0.0373 [+0.0349, +0.0397] and +0.0342 [+0.0317, +0.0367] (seeds 1, 2).
Plain + pointer s1 kept no checkpoint, so the paired gap has one plain seed.

Why the curve is flat (`loopprobe.py`, eager, 24 rows): not an eval bug. The speed keys on and off give
the same curve (CE @1 / @2 / @6 within 0.0002); depth 1 runs 1 pass and depth 6 runs 6. The slot is
read: zeroing every slot write costs 0.28 nats (pointer) and 0.41 (no pointer). All of it is in pass
1: the cell changes 8.5 % from pass 1 to 2, then 0.45 %, 0.07 %, 0.05 %, 0.05 %, and the coda's CE
does not move even across the 8.5 %. The injection still reads the state (decay 0.43-0.51, mean 0.46).

| # | prediction | result |
| --- | --- | --- |
| 1 | CE@6 mean in [3.730, 3.800] | holds (3.7631) |
| 2 | gap to plain + pointer at most +0.060 | holds (+0.036 paired, one plain seed; +0.043 unpaired pooled) |
| 3 | K1-K6 at least +0.060 | fails (0.0000) |
| 4 | no pointer: K1-K6 at least +0.080, CE within 0.03 of 4.1242 | fails (K1-K6 0.0000; CE 4.1266 holds) |
| 5 | at least 35.5k run-log tok/s | holds (39.4k / 39.8k, 1.78x) |
| 6 | no non-finite loss, no spike after step 1000 | holds |
| 7 | leak tripwire | not tripped (CE 0.036 above plain; head-off 0.41 above head-on) |

## Verdict

Failure: predictions 3 and 4 missed. The minimal slot loop is plain + pointer's depth-1 model at 1.6x
plain's row-compacted speed, with no loop contribution.

Process error, mine: the old panel's `bare` arm (one cell, no fan, no selection, no span decoder, WITH
the hinge, the fixed-point term and the clip) was already flat, CE 4.0826 at every depth
(`failures/2026-10-06-parcae-lxtul-5k-panel.md`). I wrote prediction 3 without reading it.

## Updated hypothesis

On Parcae's strict geometry a one-cell slot loop learns a one-pass map and a fast fixed point; the
stabilisers are not why it is flat (`bare` had them and was flat). Every 4-cell arm earned, so the fan
is the element that buys depth. Next: `planned/2026-10-14-parcae-minimal-fan-5k.md`, the minimal loop at
4 cells, with and without latent selection, and at 2 cells.
