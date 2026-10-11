# Credit panel, round 3: ungraded fan without the span decoder, and second runs

Status: failure

## Question

Round 2 left the ungraded fan + pointer as the design to adopt (Wolfe, 2026-10-10: distinct cells,
cos -0.09, no z machinery), at CE@6 3.9196 against the one-cell loop's 3.8997. Two open points:
(1) on the one-cell loop the span decoder bought nothing at 5k (3.8996 without it, 3.8997 with it,
26 % faster); does the ungraded fan need it? (2) two runs of one config (the winner read) differed
by 0.020 in CE@6 and 0.008 in K1-K6, so the one-cell lead over the fan rests on one run of each.

## Hypothesis

The fan does not need the span decoder for CE either; its cells stay distinct without it (LXTUL
without the decoder read cos 0.024). The one-cell lead over the fan is real at 5k (about 0.02).

## Predictions

Measured same-config spread: 0.020 CE@6, 0.008 K1-K6 (two runs); a single-run difference under
0.015 CE@6 reads as no effect.

| Arm | CE@6 best guess | K1-K6 best guess | other |
|---|---|---|---|
| ungraded fan, no span decoder | 3.920 | +0.003 | cell cos below 0.2; trainer rate 20 % or more above the fan's |
| ungraded fan + pointer, run 2 | 3.920 | +0.005 | cell cos below 0.2 |
| one-cell loop, run 2 | 3.900 | +0.002 | |

Decision rules:
- fan without the decoder within 0.015 of the fan's mean over its two runs: propose dropping the
  decoder from the fan (the speed lever). Worse by more: the decoder stays in the fan.
- one-cell minus fan, mean of two runs each, at or below -0.015: the 5k one-cell lead stands (as
  data; Wolfe's long-training bet on the fan is unchanged). Above -0.015: no 5k difference.

## Method

`nowta_pointer_fast_nospandec` (= `nowta_pointer_fast` + `tul.spandec: false`,
`model.spandec_ce_row_cap: false`), `nowta_pointer_fast`, `tul_onecell_pointer_fast`, from a
frozen `git archive` of the commit that adds this file, 5000 steps, seed 1, WANDB offline, one at
a time under `gpu.lock` by `/home/wolfe/morph-scratch/abq/runner.sh`. Readout as round 2.

## Results

Run 2026-10-10, 5000 steps, seed 1, fast recipe, 480 rows, one run per arm. Artifacts:
`lab/experiments/results/2026-10-10-credit-panel-round-3/` (`sweep-nowta-nospan.*`,
`sweep-nowta-ptr-r2.*`, `sweep-onecell-r2.*`, `queue-log.txt`).

| Arm | CE@1 | CE@6 | K1-K6 [95% CI] | write worth | cell cos | trainer tok/s |
|---|---|---|---|---|---|---|
| ungraded fan, no span decoder | 3.9018 | 3.8994 | +0.0024 [+0.0020, +0.0027] | 0.188 | -0.28 | 28,157 |
| ungraded fan + pointer, run 2 | 3.9266 | 3.9227 | +0.0039 [+0.0035, +0.0043] | 0.179 | -0.11 | 24,668 |
| one-cell loop, run 2 | 3.9048 | 3.9029 | +0.0019 [+0.0016, +0.0022] | 0.192 | | 28,235 |

Two-run means: ungraded fan + pointer CE@6 3.9212, K1-K6 +0.0042 (rate 24,665). One-cell loop CE@6
3.9013, K1-K6 +0.0021. Cell cos in the table is `val/slot_cell_pairwise_cos` from each run's final
val line.

Predictions:
- Fan without decoder CE@6 best guess 3.920: measured 3.8994. Failed (0.021 better than guess).
- Fan without decoder K1-K6 best guess +0.003: measured +0.0024. Held.
- Fan without decoder cell cos below 0.2: -0.28. Held.
- Fan without decoder rate 20 % or more above the fan's: 28,157 against 24,665 is 14 % above.
  Failed.
- Fan run 2 CE@6 best guess 3.920: 3.9227. Held. K1-K6 best guess +0.005: +0.0039. Held. Cell cos
  below 0.2: -0.11. Held.
- One-cell run 2 CE@6 best guess 3.900: 3.9029. Held. K1-K6 best guess +0.002: +0.0019. Held.

Decision rules:
- Fan without decoder is 0.022 better than the fan two-run mean (3.8994 against 3.9212). The rule
  says "within 0.015: propose dropping the decoder from the fan; worse by more: the decoder
  stays". The result is better, so the rule says drop the decoder. Caveat: one run on the
  no-decoder side. The 0.022 is at the 0.020 same-config noise floor, so the direction holds and
  the size does not.
- One-cell mean minus fan mean = 3.9013 - 3.9212 = -0.020, at or below -0.015: the 5k one-cell
  lead stands against the fan with the decoder. The fan without the decoder ties the one-cell loop
  (3.8994 against 3.9013, a 0.002 gap, inside noise).

## Verdict

Failure. The CE@6 prediction for the headline arm (the fan without decoder) missed by 0.021 and its
rate prediction missed. Seven of the nine prediction lines held, but the two that failed are the headline ones.
The two-run spreads are small (fan 0.003, one-cell 0.003), so the 0.020 one-cell lead over the
fan with decoder is stable in this pair. The 0.022 gain from removing the fan's decoder rests on
one run.

What the method could not distinguish: whether the fan without decoder is better than the fan
with it, or equal, from one run at the noise floor.

## Updated hypothesis

The span decoder is not needed by either loop at 5k: the one-cell loop (3.8996 without, 3.8997
with) and the ungraded fan (3.8994 without, 3.9212 with) both reach CE@6 of about 3.90 without it,
at 14 % to 26 % more speed. With the decoder removed, the fan and the one-cell loop tie at 5k.
Wolfe's position, recorded and not argued: he expects 4 cells to win after about 2B tokens, and
20k cannot show that knee. K1-K6 of all no-decoder arms is +0.0015 to +0.0024, a fifth of LXTUL's.

Next experiment: a second run of the fan without decoder (to confirm the 0.022), then the long
comparison of the one-cell loop against the cell fan,
`planned/2026-10-10-onecell-vs-lxtul-20k.md`, with the fan without decoder as the multi-cell arm, only on Wolfe's word (2026-10-10: "Do not queue the 20k steps right now").
