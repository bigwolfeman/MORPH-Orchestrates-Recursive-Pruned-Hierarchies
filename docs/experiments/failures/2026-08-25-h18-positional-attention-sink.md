# Experiment: H18 — a positional attention sink in the looped core

Status: failure

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

## Results — run 2026-08-25

    PYTHONPATH=$PWD python lab/divergence/attn_sink_probe.py \
        --ckpt-dir checkpoints/morph/onset-capture --out attn_sink.json
    python lab/divergence/score_h18.py --json attn_sink.json

Validity gate PASSED: window self-test <= 2e-2 at every cell, the core-block call order
assert held at every rung, and the loop-iteration count was 8 everywhere.

`ratio = pr(t=7) / pr(t=0)` on the window branch, per core block:

| step | class | b0 | b1 | b2 | b3 | b4 | b5 | mean | blocks < 1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1625 | HEALTHY | 0.949 | 0.871 | 1.039 | 0.927 | 0.954 | 1.130 | 0.978 | 4 |
| 1650 | HEALTHY | 0.934 | 0.893 | 1.013 | 0.928 | 0.984 | 1.126 | 0.979 | 4 |
| 1675 | HEALTHY | 0.933 | 0.927 | 1.022 | 0.898 | 0.934 | 1.140 | 0.976 | 4 |
| 1700 | HEALTHY | 0.917 | 0.923 | 1.025 | 0.895 | 0.927 | 1.158 | 0.974 | 4 |
| 1725 | HEALTHY | 0.904 | 0.880 | 1.054 | 0.906 | 0.932 | 1.245 | 0.987 | 4 |
| 1750 | HEALTHY | 0.867 | 0.825 | 0.894 | 0.926 | 0.944 | 1.144 | 0.933 | 5 |
| 1775 | HEALTHY | 0.941 | 0.814 | 0.848 | 0.875 | 0.917 | 1.096 | 0.915 | 5 |
| 1800 | SICK | 0.899 | 0.753 | 0.790 | 0.842 | 0.860 | 1.042 | 0.864 | 5 |
| 1825 | ambig | 0.867 | 0.822 | 0.676 | 0.906 | 0.850 | 1.085 | 0.868 | 5 |
| 1850 | SICK | 0.784 | 0.833 | 0.818 | 0.947 | 0.756 | 1.146 | 0.881 | 5 |
| 1866 | SICK | 0.832 | 0.824 | 1.055 | 0.939 | 0.931 | 1.130 | 0.952 | 4 |

| prediction | outcome |
|---|---|
| P1 sick rungs concentrate at >= 4/6 blocks | **HELD** (5, 5, 4) |
| P2 healthy rungs concentrate at <= 3/6 blocks | **FAILED** — every healthy rung is at 4 or 5 |
| P3 mean(healthy) - mean(sick) >= 0.10 | **FAILED** — 0.963 - 0.899 = +0.064 |
| P4 positional: row_agree >= 0.8 at >= 4/6 blocks, every rung | **FAILED** — 0/6 to 4/6 |
| P5 top1 rises >= 20 % from 1625 to 1850 | **FAILED** — +14.2 % |
| REFUTER: healthy and sick `pr` within 10 % at EVERY cell | **did NOT fire** — 44 of 48 cells |

**1 of 5 predictions held, and the one that held does not discriminate**: healthy rungs
concentrate at 4 of 6 blocks too, so P1 separates nothing.

### The absolute numbers, which are what actually kill the mechanism

Not pre-registered, because the predictions were deliberately made relative. Reported
because they decide the reading. Window branch, `n_valid = 57` slots per row:

| step | pr(t=0) | pr(t=7) | pr/n_valid at t=7 | top1(t=0) | top1(t=7) |
|---:|---:|---:|---:|---:|---:|
| 1625 | 36.40 | 35.79 | 0.628 | 0.0685 | 0.0725 |
| 1750 | 37.05 | 34.80 | 0.611 | 0.0650 | 0.0759 |
| 1800 | 38.53 | 33.61 | **0.590** | 0.0645 | 0.0836 |
| 1850 | 37.71 | 33.54 | **0.588** | 0.0644 | 0.0828 |
| 1866 | 36.04 | 34.41 | 0.604 | 0.0704 | 0.0740 |

**The core's forward attention is DIFFUSE.** The participation ratio sits at 0.59 to 0.68
of the 57 valid slots, and the single top key holds 6.4 % to 8.4 % of the mass. A sink is
`pr -> 1` and `top1 -> 1`. Nothing here is within an order of magnitude of that.

The `argmax` is slot 0 or slot 1 at every rung, which is the artifact Phase 0 predicted:
causal + XSA makes query `i` see keys `0..i-1`, so slot 0 is the only key every query can
see. `row_agree` is 0.47 to 0.81, so the rows of a batch do not even agree on that.

CSA branch, for completeness (it gates nothing): `argmax` is block 0 at 9 of 11 rungs with
`row_agree` 0.78 to 1.00, but `top1` is 0.29 to 0.39 of 7.5 causally valid blocks and `pr`
is 4.1 to 4.8. Positionally stable, and equally diffuse. Its stability is also structural:
under the compressed causal mask, queries 8 to 15 can see block 0 and nothing else.

## Verdict

**H18 is NOT SUPPORTED.** There is no positional attention sink in the looped core. The
forward attention is diffuse at every rung, healthy and sick, and stays diffuse across the
loop.

Stated carefully, because the pre-registered refuter did NOT fire (44 of 48 cells, and it
needed 48). The effect that exists points the RIGHT way and is about half the size the
plan asked for: the loop's compression of attention entropy goes from -2 % at 1625 to
-13 % at 1800 and 1850, and it recovers to -5 % at 1866. That is the same shape H4 found
in the state rank and the same non-monotone tail. It is a weak CORRELATE inside a diffuse
regime, not a mechanism, and it is not a sink.

The campaign's own next-step table said "if attention is diffuse, the last cheap forward
hypothesis dies". It is diffuse. It dies.

## Updated hypothesis

The takeover is not driven by attention mass concentrating on a slot. The cotangent sink
measured in `2026-08-24-tul-takeover-cure.md` is therefore NOT the forward attention
looking at a few slots; it is the backward pass concentrating on the few slots whose
states still differ, which is a consequence of the state-rank collapse H4 measured, not a
cause of it.

## What Phase 0 found instead, and why it matters more

The geometry audit that made this experiment measurable turned up two architectural facts
about the slot path that no hypothesis in the campaign had used:

1. **The HCA compressed branch is identically zero on the slot path.**
   `hca_compress_ratio: 256` against 64 slot positions gives `n_blocks = 0`, so three of
   the six core blocks run with half their attention output exactly 0.0000 while the gate
   still spends `g_comp ~ 0.50` on that zero tensor. The TOKEN path at the same weights has
   `n_blocks = 4` and `|out_comp| ~ 1030`.
2. **CSA's sparse selection never fires on the short schedule.** `tk == n_blocks` and
   `distinct_top_idx == n_blocks` at S = 64 (8 of 8) and S = 1152 (144 of 144), because
   `top_k: 256` exceeds `n_blocks` at `seq_len: 1024`. At the deploy `seq_len: 4096` there
   are 512 blocks and selection does fire, so this is a short-schedule artifact.

Fact 1 is an ASYMMETRY BETWEEN THE SICK ARM AND THE HEALTHY ONE. A1 loops over slots and
its HCA blocks are half dead; A0 loops over tokens and they are not. That is exactly the
kind of difference the campaign has been looking for, and it is not weight-spectrum. It
becomes hypothesis H24 and needs its own pre-registered arm.

## Declared not verified

- fused vs eager window paths are not compared; the probe ran eager, as the ladder was produced
- seed 0 only
- H24 is a hypothesis generated by this run, NOT tested by it. Nothing here shows the dead
  HCA branch causes the takeover.
