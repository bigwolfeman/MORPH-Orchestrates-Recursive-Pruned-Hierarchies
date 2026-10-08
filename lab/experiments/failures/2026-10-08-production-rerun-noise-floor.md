# Production rerun: the same-code K1-K6 noise floor

Status: failure

Filed as a protocol failure: no prediction was written before the run (it was queued on
2026-10-07 evening as a noise-floor measurement and started 2026-10-08 01:24). The data stands;
the verdict below is about the protocol, not the numbers.

## Question

How far apart are two 5k runs of the SAME production recipe (`lxtul_pointer`, eager, seed 1) on
CE@6 and K1-K6? Without this number no single-run comparison of the winner can be read, including
the 2026-10-07 graph-step 5k check ([failure](2026-10-07-graph-step-5k.md)).

## Method

`perf/graph-step` at 014c5aad, every new key at its default (off; the off paths are gated
byte-identical to production over 45 and 200 deterministic steps). `--config-name lxtul_pointer
wandb.name=lxtul-pointer-rerun`, non-deterministic mode like the original run (MORPH's default
kernels use atomics, so the same seed does not give the same run). Readout:
`lab/divergence/core_depth_sweep.py --depths 1,6 --rows 480 --batch 3`, the same 480 rows as the
winner's filing. Artifacts: [`results/2026-10-08-production-rerun-noise-floor/`](../results/2026-10-08-production-rerun-noise-floor/).

## Results

| run | CE@1 | CE@6 | K1-K6 (ce_tokens, 480 rows) | spandec K1-K6 |
|---|---|---|---|---|
| eager winner, 2026-10-06 | 3.9435 | 3.9280 | +0.0155 [+0.0147, +0.0164] | +0.0133 |
| eager production rerun, 2026-10-08 | 3.9281 | 3.9187 | +0.0095 [+0.0089, +0.0100] | +0.0113 |
| graph recipe, 2026-10-07 | 3.9404 | 3.9316 | +0.0088 [+0.0082, +0.0094] | +0.0118 |

Final trainer val: rerun 3.9085, graph 3.9212.

## Verdict

Protocol failure (no prediction). The measurement: two runs of the same production code differ
by 0.0060 in K1-K6 and 0.0093 in CE@6. The 480-row CIs (about +-0.0006) are within-model
readout noise, not run-to-run noise; the run-to-run spread is ten times larger.

## Updated hypothesis

The graph recipe's K1-K6 (+0.0088) sits inside the production rerun spread (+0.0095 vs
+0.0155); the 2026-10-07 "43 % drop" is not evidence that the graph recipe changes training.
The winner's +0.0155 was a high draw. Consequence for the owner's rule ("keep loop contribution
within 5 % of baseline"): 5 % of K1-K6 is about 0.0006, ten times below the single-run spread,
so single 5k runs cannot test it. Numerics-only changes are tested at fixed weights (anchor 0);
changes to the training function need a paired design (resume from one checkpoint with and
without the change, same data order) or several seeds per arm.
