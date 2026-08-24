# Agent Note: an opt-in bit-reproducible training configuration

Status: implemented

## Problem

MORPH training runs were not reproducible at a fixed seed, and nobody knew by how much.
Two runs with identical code, config, seed and data order parted on the **first backward
pass** and reached a 10 % difference in the pre-clip gradient norm by step 11. One of the
pair finished healthy and the other suffered the core takeover — byte-identical commands,
opposite outcomes
([the failed gate](../../../../docs/experiments/failures/2026-08-23-tul-run-replication.md)).

This blocked an entire investigation. Two checkpoints at the same step from two runs are
not comparable if the runs are different trajectories, so no bisection, no onset study and
no mediation analysis was readable. Roughly 40 single-run arms had already been compared
against each other on the assumption that they were comparable.

## Decision

Ship `training.deterministic` (default `false`). When true it calls
`torch.use_deterministic_algorithms(True, warn_only=True)`, disables cuDNN autotuning, and
**raises** unless `CUBLAS_WORKSPACE_CONFIG` was exported before the process started.

It is only half the answer, and the code says so: the flag also prints a warning when
`model.use_kernels` is left on, because the fused CSA/HCA attention backward is a Triton
kernel and therefore outside PyTorch's determinism machinery entirely. Measured, six
repeated backwards over 150 parameters:

| configuration | non-identical | params affected |
|---|---:|---:|
| kernels ON | 6/6 | 119 / 150 |
| kernels ON + deterministic | 6/6 | 119 / 150 |
| kernels OFF | 1/6 | 36 / 150 |
| **kernels OFF + deterministic** | **0/6** | **0 / 150** |

Verified end to end: two 300-step training runs are bit-identical on all 300 probed steps
across all 85 probe series, with every console loss matching exactly.

Cost, measured: **2.28× fewer tokens per second** (24780 → 10845) and roughly half the
batch, because eager attention materialises what the kernels fuse (batch 12 OOMs at 31 GB;
batch 6 peaks at 18.6 GB).

## Alternatives considered

- **Make the Triton attention backward deterministic.** The real fix: reproducibility at
  full speed. Rejected for now on cost — it needs a split-K or per-block partial buffer and
  a second reduction kernel in two files, plus revalidation, and the blocked investigation
  needed an answer the same day. This note is the reason to come back to it.
- **Accept nondeterminism and use replicates everywhere.** Rejected as the primary route: at
  a median run-to-run spread of 6.5 % on the pre-clip gradient norm, and with a takeover that
  fires in one run of an identical pair, the replicate count needed to see a modest effect
  is larger than the GPU budget. It stays the fallback for anything that must run fast.
- **`torch.use_deterministic_algorithms` alone.** Measured and rejected: it does not touch
  the Triton kernels, which are the dominant source. Shipping it alone would have produced a
  configuration that looks reproducible and is not — which is why the flag warns.
- **Fixing `bag_mean` and stopping there.** Already done and already known to be
  insufficient: it cut the single-step gradient error 12× and changed nothing at run level.

## Consequences

- Experiments that must bisect or compare two runs use `deterministic: true` +
  `use_kernels: false` and pay 2.3× wall clock. Production runs do not.
- **The halved batch is a confound.** It changes the gradient noise scale, so a reproducible
  run is not the same experiment as a fast one. All arms in a comparison must share one
  configuration.
- The takeover statistics measured on the fast configuration do not transfer: a different
  attention implementation and half the batch. They must be re-measured.
- The fused attention backward is now a named, located defect rather than a suspicion:
  `morph/kernels/triton/fused_csa_attention.py:279` and `fused_hca_attention.py:288`.
- Not verified: that a **long** reproducible run stays bit-identical. 300 steps is measured;
  4000 is not.

Full measurements, including two probe bugs that produced wrong answers first:
[`the reproducibility result`](../../../../docs/experiments/results/2026-08-23-morph-bit-reproducible.md).
