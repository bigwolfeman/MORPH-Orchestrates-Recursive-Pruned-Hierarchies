# Planned: A2 future-leak probe — is the paid loop's fast descent reading the future?

Status: success
Date: 2026-09-02 (frozen ~19:45, before the probe runs; trigger: Wolfe — "That
may be indicative of some other kind of issue like answer leakage or something
else. That is a lot.")

## Question

Clean A2 (tul-a2, paid loop) drops 0.49 nats between step 2000 and 2250 after a
750-step plateau, reaches 4.68 at 2500 and 4.23 at 5000 (better than notul-20k
at the matched step), and detonates in ~2/3 of draws. Its depth sweep earns
K1−K6 = 0.1685 at 5k. The two known leak channels (retention_carry, the l2cap
cap) are OFF in this recipe, and the causality gate in the smoke runs at step 0
only. A LEARNED leak would pass that gate. No trained A2 checkpoint has been
probed. Does any information from token positions after a cut reach the CE at
positions before it?

## Hypothesis

H-causal: A2 is causal at every scored position; the descent and the earning
are real. Every attention branch is masked causally, the slot input is a
bag-mean over the span BEFORE the slot, retention is off, and the 2026-08-31
probe found the l2nc control bit-identical clean vs corrupt. Counter (H-leak):
some path in the paid core (tokens attending slots that sit after them, the
HCA compressed branch, the bag-mean) reads positions ≥ k. The 2026-08-31
lesson is that a leak can be invisible in the loss and only show under a
perturbation, which is why this is measured rather than argued.

## Method

`lab/divergence/future_leak_probe.py` (now arm-general: the forced-depth knob
is picked by `_build.DepthLever`, so on A2 it forces `model.cfg.mean_depth`,
the same lever `a2_depth_sweep.py` used for the 0.1685). Checkpoints
tul-a2/step_2500 and step_5000. 48 packed rows, batch 3, eager kernels, bf16
autocast, same validation stream and packing as the sweep. Two modes:

- v1: corrupt token positions with packed index > k, score every token
  position < k (labels ≤ k, clean). k ∈ {700, 900} of L_total 1152.
- boundary (`--boundary`): corrupt positions ≥ k, score ONLY the last token
  position before k (its label's input copy is corrupted).

Forced depths {1, 3, 6}. Per cell: CE_clean, CE_corrupt, earning = K1 − K6.

Sanity gate (aborts the read if it fails): v1 clean earning at step_5000,
k=900, lies in [0.10, 0.25] — the sweep's 0.1685 was over ALL token positions
of the same 48 rows; positions < 900 are a subset, so ±0.05 of slack.

## Predictions (frozen)

- **P-L1 (binding, v1).** |CE_corrupt − CE_clean| < 0.01 nats at every (ckpt,
  k, depth) cell: **85%**.
- **P-L2 (boundary).** The same < 0.01 bound at every boundary-mode cell:
  **80%**. (Lower because this cell scores ONE position per row; n = 48 per
  cell and bf16 noise has less to average over.)
- Derived, not scored separately: earning_corrupt within 0.02 of
  earning_clean everywhere.

## Binding

P-L1 and P-L2 TRUE ⇒ H-causal. The 5k table stands, the plateau-then-cliff
is a phase transition of a causal model, and the "aggressive minimum is
unstable" reading goes to the ρ ladder
(`2026-09-02-a2-core-jacobian-ladder.md`). Any cell ≥ 0.01 ⇒ H-leak: STOP.
No 20k on this recipe. Locate the path by ablating branches (HCA compressed,
bag-mean, slot attention) on the same checkpoint before anything else; the
5k earning table is suspect until the path is found.

## Not verified before run

The probe's A2 path end to end (the lever was smoke-tested on a tiny CPU
model; `tul_forward_ablated(plan_mode="normal")` on a full A2 checkpoint is
first exercised by this run). If the sanity gate fails the read is void and
the script, not the model, is the first suspect.

## Results (2026-09-02 20:18-20:20, tul-a2 step_2500 + step_5000)

Sanity gate: v1 clean earning at step_5000, k=900 = +0.1666 (band 0.10-0.25;
the full-row sweep gave 0.1685). Read is valid. Depth lever confirmed in the
log: `model.cfg.mean_depth`.

Max |CE_corrupt − CE_clean| over every cell (2 ckpts × 2 k × 3 depths), from
the JSON: **v1 = 0, boundary = 0.** Not "< 0.01": bit-identical, in bf16
autocast, at every depth, in both modes. Artifacts:
lab/experiments/results/a2_leak_probe_v1.json, a2_leak_probe_boundary.json;
logs $Q/a2probes/a2-leak-{v1,boundary}.log.

v1 (all token positions < k scored):

| ckpt | k | K1 | K3 | K6 | earning | n |
|---|---|---|---|---|---|---|
| 2500 | 700 | 4.7693 | 4.6590 | 4.6501 | +0.1191 | 30547 |
| 2500 | 900 | 4.7636 | 4.6523 | 4.6427 | +0.1209 | 39248 |
| 5000 | 700 | 4.3617 | 4.2118 | 4.1990 | +0.1627 | 30547 |
| 5000 | 900 | 4.3428 | 4.1896 | 4.1762 | +0.1666 | 39248 |

Boundary (last token before k only, its label's input copy corrupted):

| ckpt | k | K1 | K3 | K6 | earning | n |
|---|---|---|---|---|---|---|
| 2500 | 700 | 4.7283 | 4.6307 | 4.6091 | +0.1192 | 48 |
| 2500 | 900 | 5.2566 | 5.0920 | 5.0813 | +0.1753 | 48 |
| 5000 | 700 | 4.4105 | 4.2762 | 4.2317 | +0.1787 | 48 |
| 5000 | 900 | 4.6305 | 4.4777 | 4.4488 | +0.1817 | 48 |

Corrupt columns are omitted: they equal the clean columns exactly.

- **P-L1 (85%): TRUE.** Every v1 cell moves 0.
- **P-L2 (80%): TRUE.** Every boundary cell moves 0.

## Verdict

**SUCCESS — H-causal.** The paid loop's fast descent, the 0.49-nat step
2000→2250 cliff, and the K1−K6 earning (0.12 at 2500, 0.17 at 5000, and it
GROWS with training) are properties of a model that reads nothing after the
cut. The 5k table stands. Per the binding, the "aggressive minimum is
unstable" reading proceeds to the core-Jacobian ladder.

## Updated hypothesis

A2's earning is honest and growing (0.119 → 0.163 on the same positions
between 2500 and 5000). The open question is stability alone, and the ladder
(`2026-09-02-a2-core-jacobian-ladder.md`) is the instrument.
