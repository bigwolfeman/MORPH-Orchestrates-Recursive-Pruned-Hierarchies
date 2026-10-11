# Credit panel, round 3: ungraded fan without the span decoder, and second runs

Status: planned

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
