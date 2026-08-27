# Experiment: does the MUX matched-control result replicate at fresh seeds?

Status: **planned.** No code change. Predictions frozen 2026-08-27 before any arm
ran.

## Question

The [08-27 arm panel](../failures/2026-08-27-warmup-sigreg-ntpdrop.md) found that
`tul_v1a2b` — the MUX head, detached, β=0.1 — beats the control on every measure:

```
                v1a2b (4 seeds)                control (3 seeds)
plan worth      0.0165 0.0201 0.0214 0.0330    0.0124 0.0148 0.0164
loop worth      0.0058 0.0082 0.0106 0.0107    0.0042 -0.0002 0.0013
takeover        0 of 4                         3 of 4
```

Complete separation on both worths, exact permutation p = 0.0286 — which is the
FLOOR for 4 against 3, so the test is maxed out rather than strong.

**That comparison is post-hoc.** The pre-registration froze an absolute threshold
of 0.0191 nats; the matched-control framing was devised after seeing the arm
numbers, because no control checkpoint at step 3000 existed when the panel was
written. A post-hoc separation at the smallest achievable p is exactly the shape
of a result that does not replicate. This is its confirmation.

## Arms and protocol

Fresh seeds — **4, 5, 6**, used by no arm in this campaign. `tul_v1a2b` and
`tul_a1` (control), three seeds each, six runs. Identical recipe to every other
arm: 3500 steps, batch 6, `ademamix_alpha_cap=3.5`, `use_kernels=false`,
`eval_every=250`, `ckpt_every=500`, `grad_probe_every=1`. Plan-off and loop worth
from `slot_path_worth.py --batches 8` on `step_3000.pt`.

**The control is re-run at the same fresh seeds rather than reused.** The old
three seeds would make this a comparison against a fixed reference again, which
is the defect being corrected.

## Predictions (frozen 2026-08-27, before any run)

- **M1 (plan worth):** `v1a2b` plan-off worth exceeds the **maximum of the three
  NEW control seeds** on **≥ 2 of 3** seeds.
- **M2 (loop worth):** the same, on loop worth. This is the stronger claim — the
  control's loop worth is indistinguishable from zero, so any consistent positive
  value is the loop doing work for the first time in this campaign.
- **M3 (takeover):** by the shipped 30 %-of-50 rule on the per-step probe, the
  takeover fires on **≤ 1 of 3** `v1a2b` seeds and **≥ 2 of 3** control seeds.
- **M4 (a test, not an eyeball):** exact one-sided permutation test on loop
  worth, 3 against 3, gives **p ≤ 0.10**. The floor for 3v3 is 1/20 = 0.05, so
  this needs near-complete separation and is not free.
- **M5 (THE DECISION):** M1 **and** M2 hold ⇒ the result replicates, and
  `tul_v1a2b` becomes the campaign's baseline arm in place of `tul_a1`. Either
  fails ⇒ the 08-27 separation was post-hoc luck, MUX is not promoted, and the
  loop-worth claim comes out of the record.

## Risks and confounds recorded up front

- **The control's takeover rate is seed-dependent.** The old four seeds gave 3 of
  4. Three fresh seeds could give 1 of 3 by chance alone, which would make M3
  fail without telling us anything about MUX. M3 is therefore reported but is not
  part of the M5 decision.
- **Broken runs score HIGH on plan worth** (measured: 0.0247 at ppl 305, 0.0301
  at ppl 603). Any seed whose `ce_main` at step 3000 exceeds 5.0 is reported with
  that flag and its worth is excluded from M4.
- **`ademamix_t_beta3` is null and falls back to `training.steps`.** Matched by
  construction here at 3500 steps for every arm, as it was for the originals.
- **Six runs at ~33 minutes is ~3.3 hours of sequential GPU** (UPS: one trainer at
  a time). Queued behind the token-tax sweep, which gates the architecture
  decision and therefore goes first.
