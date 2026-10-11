# The span decoder reads the winner cell

Status: failure

## Question

On LXTUL + pointer the span decoder is 0.69 of the loop core's gradient (credit matrix,
`lab/experiments/results/2026-10-10-lxtul-credit-matrix/`). It grades the MEAN of the 4 exit
cells, but the coda reads only the router's winner cell, so the 3 losers carry 2.6x the winner's
part of that gradient. If the decoder grades the winner cell (`tul.spandec_read: winner`), do CE
and the loop's own contribution change?

## Hypothesis

Two accounts, and this run separates them.

- Mismatch is waste: the loop's main teacher grades a readout the model never uses. Grading the
  read cell aligns the decoder with the coda, so CE improves and K1-K6 holds or rises.
- Mean is a selection pool: all 4 exit cells start from the same reset winner and share weights,
  so the decoder's gradient through the losers keeps every candidate decodable, which the router
  needs. Taking it away lets the losers decay: the gap between the best cell and the routed cell
  grows, and CE gets worse.

My best guess is a small CE gain with K1-K6 unchanged, but I hold it weakly.

## Predictions

References, `lxtul_pointer_fast` from init at 5k, seed 1, 480 rows (2026-10-08 filings): CE@6
3.9248 / 3.9229 (mean 3.9239), K1-K6 +0.0104 / +0.0091 (mean +0.0098), final val 3.9161 / 3.9129,
`fan/oracle_gap` 0.0309 / 0.0347, `fan/mixed_ce` 4.5499 / 4.5415. Spread between independent 5k
runs: about 0.006 in K1-K6, 0.002 in CE@6.

Read as the mean of two arm runs:

- CE@6 best guess 3.918. At or below 3.918 (0.006 better, 3x the reference spread) = the winner
  read helps. At or above 3.930 = it harms. Between = no CE effect.
- K1-K6 best guess +0.011. Below +0.006 = harm (the C1d rule). Above +0.016 = help.
- Final val best guess 3.910.
- `fan/oracle_gap` best guess 0.040. At or above 0.045 is the selection-pool signature (losers
  decayed relative to the best cell).
- Trainer rate late within 2 % of the references' (no new compute: one gather).

Decision rule: help or no CE effect, with K1-K6 at or above +0.006: I propose to Wolfe that
`lxtul_pointer_fast` take `spandec_read: winner`. Harm on CE or K1-K6: the mean read is doing
work for the selection; file it, keep the mean, and the next arm is a relaxed read (winner plus a
small loser weight).

## Method

`--config-name lxtul_pointer_fast_spdwin` (= `lxtul_pointer_fast` + `tul.spandec_read: winner`),
from a frozen `git archive` of the commit that adds this file. Two runs, both seed 1 (the
references' seed; MORPH runs decorrelate within a few steps at a fixed seed), run one after the
other under `gpu.lock`, WANDB offline. Readout per run: `core_depth_sweep.py --ckpt
spdwin=lxtul_pointer_fast_spdwin=<step_5000> --depths 1,6 --rows 480 --batch 3`, the references'
rows and command. The fan readings come from each run's final val line.

## Results

Run 2026-10-10, 5000 steps, seed 1, two runs of the same config (winner read r1, r2), 480 rows.
Artifacts: `lab/experiments/results/2026-10-10-spandec-reads-winner/` (`sweep-r1.*`,
`sweep-r2.*`, `queue-log.txt`, `train-r1-tail.txt`, `train-r2-tail.txt`; these tails are the source of the
fan readings and the trainer rate).

| Run | CE@1 | CE@6 | K1-K6 | `fan/oracle_gap` | trainer tok/s late |
|---|---|---|---|---|---|
| winner read r1 | 3.9139 | 3.9093 | +0.0046 [+0.0042, +0.0050] | 0.0240 | 23,930 |
| winner read r2 | 3.9414 | 3.9290 | +0.0124 [+0.0118, +0.0131] | 0.0617 | 23,957 |
| two-run mean | | 3.9192 | +0.0085 | 0.0429 | 23.9k |
| references (mean) | | 3.9239 | +0.0098 | 0.0328 | 29.8k |

Span-decoder CE K1-K6: r1 +0.0188, r2 +0.0273.

Predictions against the two-run mean:
- CE@6 at or below 3.918 means help: 3.9192 is 0.001 above the line. Fails the help line. It does
  not reach 3.930 either. Reading: no CE effect. Held as "between", failed as the best guess 3.918
  by 0.001.
- K1-K6 best guess +0.011: measured +0.0085, no harm (above +0.006), no help (below +0.016).
  Best guess missed; no-harm reading held.
- Final val best guess 3.910: not logged in the files I filed from. r1 `val/ce_tokens` 3.9016, r2
  3.9215 (mean 3.9116), 0.002 above the guess.
- `fan/oracle_gap` at or above 0.045 is the selection-pool signature: r1 0.0240, r2 0.0617, mean
  0.0429. One run is on each side of the line. Not readable.
- Trainer rate within 2 % of 29.8k: 23.9k is 20 % lower. Failed.

## Verdict

Failure. The run does not tell the two accounts apart.

The same config run twice differed by 0.020 in CE@6 and 0.008 in K1-K6. The two-run CE@6 mean sits
0.005 from the references, and the K1-K6 mean sits 0.0013 from them. Both are inside noise. The
`fan/oracle_gap` of the two runs (0.024 and 0.062) straddles the 0.045 line, so neither account is
supported. The rate prediction failed by 20 %. Most panel runs ran at 22k to
28k tok/s, so I did not find the cause in the winner read, which adds only one gather. I did not
measure it.

What the method could not distinguish: a CE@6 shift under 0.020 and a K1-K6 shift under 0.008 on
a single run of 5000 steps.

## Updated hypothesis

The winner read has no effect on CE or K1-K6 that a pair of independent runs can resolve. The
decision rule says "help or no CE effect with K1-K6 at or above +0.006": that holds for the mean
(+0.0085) but not for r1 (+0.0046). I do not propose `spandec_read: winner` for the recipe on this
evidence.

Next experiment: a paired continuation from one shared checkpoint (three draws each of the mean
read and the winner read, as in `successes/2026-10-10-hinge-fresh-pass-paired.md`), which has a
K1-K6 noise of 0.0012 instead of 0.008.

## Method amendment 2026-10-11

The two runs were r1 and r2 of the same config. The filed noise from this pair (0.020 CE@6, 0.008
K1-K6) is the single-run noise used in all five credit-panel filings.
