# Parcae pointer: the normalised head, and keys that read the loop's cells

Status: failure (written 2026-10-06 21:44 CDT, before any run; CPU tests 10 passed, GPU
geometry tests 9 passed, a 60-step smoke of the cell-key arm)

## Question

Like for like, strict LXTUL + pointer trails plain + pointer by +0.098, and with the heads off the
two models differ by only 0.035 (`failures/2026-10-06-parcae-plain-pointer.md`): most of the
remaining gap is head against head. Plain's pointer keys see the whole context; the strict keys
see their own span and the earlier cells. Wolfe (2026-10-06): "I am interested in that next lever.
I think its worth running." Does a key that also reads the loop's cells for its own span (the
loop's summary of the span the copied token sits in) close part of that?

The Parcae head also had a bug (its null mass was dropped: a sub-normalised mixture, pessimistic
CE). The fix returns the null mass to the model; both baselines are re-run with it.

## Runs (5k, seed 1, 480 paired rows)

| run | config |
| --- | --- |
| re-score of lxtul-pointer-5k and plain-pointer-5k with the normalised rule | `mixture_off_eval` (no training) |
| strict + pointer, normalised | `arm=lxtul_pointer` |
| plain + pointer, normalised | `arm=plain_pointer` |
| strict + pointer with cell keys | `arm=lxtul_pointer_cellkey` |

Then the head-off leak check on the three runs.

## Predictions

1. The normalised re-score lowers both old checkpoints' CE by 0 to 0.03 (80 %).
2. Normalised strict + pointer within [-0.04, +0.01] of the old 3.8047 (75 %).
3. Normalised plain + pointer within [-0.04, +0.01] of the old 3.7072 (75 %).
4. Cell keys: the gap to normalised plain + pointer at least 0.02 smaller than normalised strict +
   pointer's gap (45 %).
5. Cell keys: K1-K6 at least +0.04 (70 %).
6. Cell keys, head off: within 0.05 of normalised strict + pointer's head-off CE (60 %).
7. No run raises.

## Method

`/home/wolfe/morph-scratch/parcae/queue_cellkey.sh`, one job at a time under the GPU lock.

## Results

Run 21:44-22:59 CDT at 505ba1d, 480 paired rows (`results/2026-10-06-parcae-pointer-cellkey/`).

Re-score of the old checkpoints with the normalised rule: strict + pointer 3.80469 -> 3.80469,
plain + pointer 3.70718 -> 3.70719 (changes below 0.0001: the trained heads put almost no mass on
the null key, so dropping it cost almost nothing at scoring time).

| run (normalised head) | CE@1 | CE@6 | K1-K6 | gap to plain + pointer | head OFF CE@6 |
| --- | --- | --- | --- | --- | --- |
| plain + pointer | 3.758 | 3.7155 | +0.0424 [+0.0413, +0.0434] | - | 4.1087 |
| strict + pointer | 3.808 | 3.7608 | +0.0468 [+0.0452, +0.0485] | +0.0453 [+0.0427, +0.0478] | 4.1601 |
| strict + pointer + cell keys | 3.799 | 3.7534 | +0.0452 [+0.0438, +0.0467] | +0.0379 | 4.1594 |

Cell keys against strict + pointer, paired: -0.0074 [-0.0090, -0.0058]. Throughput 21.9k against
22.2k tok/s; 14.1 against 13.9 GiB.

| # | prediction | result |
| --- | --- | --- |
| 1 | re-score lowers both old CEs by 0 to 0.03 | holds (both change by under 0.0001) |
| 2 | normalised strict + pointer within [-0.04, +0.01] of 3.8047 | fails, better side (-0.0439) |
| 3 | normalised plain + pointer within [-0.04, +0.01] of 3.7072 | holds (+0.0083) |
| 4 | cell keys close at least 0.02 of the like-for-like gap | fails (0.0074) |
| 5 | cell keys K1-K6 at least +0.04 | holds (+0.0452) |
| 6 | cell keys head off within 0.05 of strict + pointer's head off | holds (4.1594 vs 4.1601) |
| 7 | no raise | holds |

## Verdict

Failure on 2 and 4. The normalisation fix mattered through TRAINING, not scoring: a head that can
abstain cleanly is worth -0.044 to strict and nothing to plain (+0.008), so the like-for-like gap
falls from +0.098 to +0.045. Strict LXTUL + pointer now recovers 85 % of the +0.294 gap with no
head on either side. Cell keys add a real but small -0.0074 (gap +0.038). With the heads off the
two normalised models sit 0.051 apart (4.160 vs 4.109), about the heads-on gap: the remaining gap
is now the models' own distributions, not the heads.

## Updated hypothesis

The pointer (normalised) is the LXTUL copy channel; cell keys are a small optional add. The
remaining +0.04 is in the model, not the copy channel. Next: the gap probe on strict + pointer
against plain + pointer (which tokens carry the last 0.04), then a seed twin and a 10k pair.
