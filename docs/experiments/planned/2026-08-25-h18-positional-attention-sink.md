# Experiment: H18 — a positional attention sink in the looped core

Status: planned

Working document: [`lab/divergence/h18-attention-sink.md`](../../../lab/divergence/h18-attention-sink.md)
Ledger: `lab/divergence/takeover-campaign.md` H18.

## Question

Does the FORWARD attention inside the looped core concentrate its mass on a few fixed
slot POSITIONS, does that concentration grow across loop iterations, and does it change
at the takeover onset?

## Background, and why the question is not yet answered

- H4 CONFIRMED: the loop's effect on slot-state effective rank flips sign between step
  1750 and 1800. Healthy rungs diversify (rank ratio 1.23–1.48), sick rungs do not
  (1.01, 0.67, 0.87). It is the earliest indicator in this campaign and it is a FORWARD
  quantity.
- The COTANGENT already sits on a stable sink: the same top-3 slots at every one of the
  six core blocks (agreement 1.0 at five of six rungs), top slot's share 0.18 -> 0.54.
- H17 REFUTED: the LEARNABLE sink parameter never engages.
- Every weight-spectrum cure failed (H2, H3, and four more).

So a sink exists in the backward, it is not the learnable one, and the forward has never
been looked at.

## Phase 0, already measured and committed (2f8ae42)

Geometry only, no outcome. On the slot path (S = 64 slots): the three HCA core blocks
have `n_blocks = 0`, so their compressed branch output is identically zero; the three CSA
core blocks run `tk == n_blocks == 8`, so CSA's sparse selection never fires; the window
branch is dense causal with XSA over all 64 slots, so query `i` sees keys `0..i-1` only.

**Consequence for the method.** Early slots receive mass from many queries BY
CONSTRUCTION. A raw "slot 0 is the top key" reading would be an artifact. Every
prediction below is therefore a RELATIVE statement — across loop iterations at a fixed
mask, or across rungs — never an absolute magnitude.

## Hypothesis

The core's forward attention concentrates onto a few fixed slot positions; the
concentration compounds across the loop; and the compounding flips on at the onset, in
step with the state-rank flip H4 measured.

## Method

Probe: `lab/divergence/attn_sink_probe.py`. Scorer: `lab/divergence/score_h18.py`.
Both committed before the ladder is run.

Ladder: `checkpoints/morph/onset-capture` — `tul_a1`, seed 0, batch 6, `alpha_cap` 3.5,
`use_kernels=false`, replay-verified. Eleven rungs, `ROLL_step_{1625..1850}` at 25-step
spacing plus `TAKEOVER_step_1866`.

Per rung: two forward passes on two DIFFERENT validation batches (`skip_samples` 200k
and 250k, disjoint from the 50k the other probes use), `no_grad`, dropout off, the same
manual seed every time so the Poisson depth draw cannot move between rungs.

The attention weights are never materialized by the model, so the probe recomputes
`A = softmax(q k^T * scale + bias)` from the same `q, k` and SELF-TESTS that `A @ v`
reproduces the shipped `out_win`.

Statistic, per (rung, core block, loop iteration `t`), for the window branch and the CSA
branch separately: the mass each KEY POSITION receives, computed per row over that row's
valid slots and then averaged over rows. From it, `top1`, `top3`, the participation ratio
`pr = (sum m)^2 / sum m^2` (1 = one sink, n = uniform), the `argmax` position, and
`row_agree` — the fraction of rows whose own argmax equals the batch argmax.

**Rung classification, fixed here so it cannot be chosen later.** From the ladder's own
README core-share column:

- HEALTHY: 1625, 1650, 1675, 1700, 1725, 1750, 1775 (core share 0.012–0.054)
- SICK: 1800, 1850, 1866 (0.372, 0.890, 0.961)
- AMBIGUOUS, EXCLUDED FROM BOTH: 1825 (0.118, the README's "falls back" rung)

`ratio(rung, block) = pr(t = last) / pr(t = 0)` on the window branch. Below 1 means the
loop CONCENTRATES the attention; at or above 1 means it does not.

## Predictions

**Validity gate. Runs first and refuses the whole panel if it fails.**

- V1 window self-test relative error <= 2e-2 at every rung, block and iteration (the
  bf16 noise floor is 7.8e-3)
- V2 the recorded core-block call order is `0..n_core-1` repeated, at every rung
- V3 the loop-iteration count is IDENTICAL at every rung — a moved depth draw would make
  the rungs incomparable

**P1 compounding.** At every SICK rung, `ratio < 1` at 4 or more of the 6 core blocks.

**P2 the flip.** At every HEALTHY rung, `ratio < 1` at 3 or fewer of the 6 core blocks.

**P3 severity.** `mean(ratio) over HEALTHY rungs - mean(ratio) over SICK rungs >= 0.10`.

**P4 positional.** At the last iteration, `row_agree >= 0.8` at 4 or more of the 6
blocks, at every rung; and the cross-batch argmax agreement is `>= 0.8` at every rung.
A content-driven sink would give `row_agree ~ 1/n_valid ~ 0.02`.

**P5 the sink grows.** Window `top1` at the last iteration, meaned over blocks, is at
least 20 % higher at rung 1850 than at rung 1625.

**REFUTER.** If the window `pr` is within 10 % between the HEALTHY mean and the SICK
mean at EVERY (block, iteration) cell, H18 is REFUTED: the forward attention does not
behave differently at the onset, and the last cheap forward hypothesis in this campaign
dies.

The CSA branch is measured and reported with the same statistics but does NOT gate any
prediction: it covers three of six blocks, and Phase 0 showed its selection is inert.

## What would make this inconclusive, and why that is a failure

If the validity gate fails, the run is filed under `failures/` with the gate named, and
the next planned experiment fixes the gate. "The numbers were unclear" is not an outcome.

## Declared not verified

- fused vs eager window paths are not compared; the probe runs eager because the ladder
  was produced that way
- the ladder is seed 0 only, so a sink measured here is not shown to generalize
- a sink, if found, is a CORRELATE. This experiment cannot show it causes the takeover;
  that needs the Phase 4 intervention arm.
