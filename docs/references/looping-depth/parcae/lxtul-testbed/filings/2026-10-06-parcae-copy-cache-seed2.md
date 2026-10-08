# Parcae strict + copy cache, seed twin

Status: success (written 2026-10-06 17:46 CDT, before the run)

## Question

Strict + cache (seed 1) closed the gap to plain to +0.0215 [+0.011, +0.031] with K1-K6 +0.0491
(`failures/2026-10-06-parcae-copy-cache.md`). Is that a seed draw? Round 2 measured the seed
spread of strict LXTUL: CE@6 0.001, K1-K6 0.021.

## Run

`arm=lxtul_cache run.seed=2`, 5000 steps, commit e25088c's code path (HEAD 3c50a43 adds docs only).

## Predictions

1. CE@6 within 0.01 of seed 1's 3.8518 (85 %).
2. Gap to plain 5k in [+0.005, +0.04] (80 %).
3. K1-K6 in [+0.03, +0.08] (75 %).
4. No raise.

## Method

`panel2.sh`, then `python -m lxtul.readout --ref plain-5k`.

## Results

Run 17:46-18:09 CDT at 916c1fb, 480 paired rows (`results/2026-10-06-parcae-copy-cache-seed2/`).

| run | CE@1 | CE@6 | K1-K6 | gap to plain 5k |
| --- | --- | --- | --- | --- |
| strict + cache, seed 1 | 3.901 | 3.8518 | +0.0491 [+0.0475, +0.0508] | +0.0215 [+0.0110, +0.0306] |
| strict + cache, seed 2 | 3.922 | 3.8533 | +0.0690 [+0.0669, +0.0710] | +0.0230 [+0.0127, +0.0320] |

| # | prediction | result |
| --- | --- | --- |
| 1 | CE@6 within 0.01 of 3.8518 | holds (0.0015) |
| 2 | gap in [+0.005, +0.04] | holds (+0.0230) |
| 3 | K1-K6 in [+0.03, +0.08] | holds (+0.0690) |
| 4 | no raise | holds |

## Verdict

Success. The closed gap is not a seed draw: two seeds agree on CE to 0.0015 and on the gap to
0.0015. K1-K6 spreads by 0.020 across seeds, the same spread strict LXTUL showed (0.021).

## Updated hypothesis

Strict + copy cache is the Parcae recipe for the CE goal at 5k: gap +0.02, loop use at strict
LXTUL's level. Untested: plain + trained cache (the fair gap), 10k, the MORPH port.
