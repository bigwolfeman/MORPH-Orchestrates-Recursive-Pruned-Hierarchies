# Agent Note: The span decoder reads the winner cell

Status: proposed

## Problem

On the latent-selected loop the coda reads only the router's winner cell, but the span decoder
grades the mean of the M exit cells. The credit matrix (2026-10-10) measures the decoder at 0.69
of the loop core's gradient, with the 3 losers carrying 2.6x the winner's part. The loop's main
teacher grades a readout the deployed model never uses.

## Proposal

`tul.spandec_read: winner` gathers the routed winner cell (the same index `_fan_route_cells`
keeps) and hands it to `_tul_spandec_loss` in place of the mean. `mean` stays the default and is
bit-identical (`tests/test_spandec_read_winner.py`). Refused without the hard-winner
latent-selected loop, without `spandec`, and with `spandec_reads_cells`.

## Alternatives considered

- Leave the mean: it may keep every candidate decodable for the router (the selection-pool
  account). The experiment tests this.
- The decoder reads each cell alone and averages the 4 losses: 4x the decoder's 5.68 TFLOP per
  step. Not built.
- `spandec_reads_cells` (cross-attention onto all M cells): a different reader, and it still
  hands the losers decoder gradient.

## Acceptance criteria

`lab/experiments/planned/2026-10-10-spandec-reads-winner.md`: CE@6 and K1-K6 against the two
`lxtul_pointer_fast` references.

## Risks

The losers lose their only text signal at the exit and may decay, making the router's choice a
choice among worse candidates. The mean of 4 unit-RMS cells has a smaller norm than one cell, so
the decoder's input scale changes at step 0.
