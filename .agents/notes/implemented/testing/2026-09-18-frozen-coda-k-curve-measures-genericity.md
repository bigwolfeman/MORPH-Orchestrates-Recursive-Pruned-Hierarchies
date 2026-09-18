# Agent Note: a frozen-coda K-curve measures genericity, so the depth sweep says so

Status: implemented

## Problem

`lab/divergence/core_depth_sweep.py` reports `ce_tokens` across forced loop depths, and
`K1-K6` from it has been the depth instrument for the whole LCTUL code-target campaign. On
an arm whose coda is FROZEN the number does not mean what it looks like.

Such a coda is resumed from the VAE stage, where it was trained to read noised TRUE codes.
It prefers NO cell to a confidently-wrong one: the worth profile reads `zero - own` at
-4.64 nats. So it rewards a predicted cell that drifts toward the corpus mean and punishes
one that becomes more span-specific, and the depth curve tracks the cell's GENERICITY
rather than its match to the code. Measured 2026-09-18 on 480 rows, both directions:

| arm | reader | `pred.zbar` d1 -> d16 | frozen-coda CE d1 -> d16 | K1-K6 |
| --- | --- | --- | --- | --- |
| `tul-code-target` (A) | frozen | 0.3081 -> 0.3225 | 9.1713 -> 9.0759 | +0.0323 [+0.0236, +0.0415] |
| `tul-code-target-prog` | frozen | 0.3554 -> 0.3051 | 9.0482 -> 9.1685 | -0.1173 [-0.1304, -0.1040] |
| `tul-code-target-uf` | trained | 0.3002 -> 0.3125 | 4.1490 -> 4.1460 | +0.0047 [+0.0038, +0.0058] |

`pred.zbar` is cosine of the predicted cell to the corpus-mean code. The relation is
anti-correlated in both directions at matching magnitude (about -6.6 and -2.4 nats per unit).
At depth 1 the progressive arm is the MORE generic one (0.3554 vs 0.3081) and has the BETTER
CE (9.0482 vs 9.1713), which is the same effect seen without any depth at all.

The real loop behaviour is identical on all three arms and unremarkable: centred cosine to
the own code peaks at depth 6, the training mean, everywhere. Only the ADAPTED reader's CE
tracks it, with its CE minimum and its cosine maximum both at depth 6.

Consequence: arm A's +0.0323, the largest "depth earning" in this family, is its cell
collapsing toward the corpus mean. On a frozen-coda arm a LARGER `K1-K6` is worse news.

## Decision

`core_depth_sweep.py` prints a `[WARNING]` per checkpoint when `tul.code_target` is set and
no `coda.` prefix appears in `training.train_only`. The warning states that the K-curve
tracks genericity, that a larger value is not better, and names the companion instrument
(`code_target_mean_probe.py --depths`, read `pred.zbar`).

`tests/test_depth_sweep_frozen_reader_warning.py` pins the predicate on four configs, two
positive and two negative, asserts the message names both `pred.zbar` and the companion
probe, and asserts the discriminator is the `coda.` entry specifically. Inverting or
widening the predicate fails the suite (verified by sabotage: the `tul_code_target_uf` case
flips and the run reports `1 failed, 5 passed`).

## Alternatives considered

- **Refuse to sweep a frozen-coda arm.** Rejected: the curve is still the right instrument
  for the genericity question, and arm A's sweep is legitimate evidence about cell collapse.
  Refusing would have destroyed a reading rather than labelled it.
- **Emit `pred.zbar` from the sweep itself.** Rejected for now: the sweep has no encoder
  pass and would need the corpus-mean code, duplicating `code_target_mean_probe.py`. Pointing
  at the probe that already computes it keeps one home per measurement.
- **Put the caveat only in the experiment writeup.** Rejected: the next person runs the
  script, not the writeup. The instrument has to carry its own warning.
- **Detect the frozen reader from checkpoint weights rather than config.** Rejected as
  fragile: `train_only` is the declaration that produced the checkpoint, and reading it needs
  no heuristic about which tensors moved.

## Consequences

- Every depth sweep on `tul_code_target` and `tul_code_target_prog` now prints the warning;
  `tul_code_target_uf` and non-code-target arms stay silent.
- Historical `K1-K6` numbers on frozen-coda code-target arms are re-read as genericity drift.
  This does not touch plain or slot-loop K-curves, whose readers are trained on their own
  inputs.
- The progressive arm's negative K-curve is re-read as the cell becoming MORE span-specific
  with depth, which is the wanted direction punished by the wrong reader. Whether it inverts
  on an adapted reader is untested and is the obvious next arm.
- One seed per arm. The mechanism is consistent across three arms and two directions, but it
  has not been replicated.

Writeups: [`lab/experiments/successes/2026-09-17-lctul-target-unfreeze.md`](../../../../lab/experiments/successes/2026-09-17-lctul-target-unfreeze.md),
[`lab/experiments/planned/2026-09-17-lctul-target-progressive.md`](../../../../lab/experiments/planned/2026-09-17-lctul-target-progressive.md).
