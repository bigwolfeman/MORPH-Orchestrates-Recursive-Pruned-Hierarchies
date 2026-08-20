# Agent Note: Loop Stp Revalidation

Status: implemented

Origin: Ai-notes/06-21-2026/Loop-STP-Revalidation/RESULT.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# Loop-STP Re-validation A/B — verdict: HARM (keep OFF)

**Date:** 2026-06-21
**Trigger:** Wolfe — "we need to test the stp on loop. I know this has been tested
before and it was fine, but we need to do it again."

## Result

Paired same-seed A/B, **only** variable = `model.loop_stp_lambda` (0.0 vs 0.02).
Both arms: `ademamix_b1zero` (coordcap cure, deploy default) + `adam8bit=true`,
prune OFF (`prune_start=999999`), TST OFF (`tst_bag_size=0`), flat LR, 5000 steps,
`eval_every=200`. Isolation script: `ignore/loop_stp_revalidate.sh`.
Logs: `ignore/loopstp_{off,on}.runlog`, overlay in `ignore/loopstp_master.log`.

```
signed_mean(on-off) = +0.0693 nats   abs_mean = 0.0693   worst = +0.1970
pos/neg = 24/0   (ALL 24 eval points: on-arm loss HIGHER)
verdict = HARM
```

Shape: largest harm during warmup (+0.197 @ step 200), decays to ~+0.043 @ 4800,
**never crosses zero, never trends negative.**

## Interpretation

- **Paired** same-seed design → the ~0.12–0.15 nat run-to-run floor (which is an
  *unpaired* phenomenon) does NOT apply. Only difference between arms is the loss
  coefficient. 24/24 positive ⇒ sign-test p ≈ 2⁻²⁴: the harm is real, not noise.
- At λ=0.02 on the looped core under the deploy optimizer, the loop-axis geodesic
  regularizer is a **small, consistent regression** — worst during early training.
- **Refutes** the prior "loop-STP was fine" belief *for this regime* (5k, prune-off,
  ademamix_b1zero, λ=0.02). Does NOT test other λ or other regimes (with-prune,
  longer horizon) — but at the model-matched λ it does not help.

## Decision

**Keep `loop_stp_lambda = 0.0` (already the base.yaml default).** No config change.
The A/B confirms the default is correct; turning it on is a measured regression.

## IMPORTANT disambiguation — two different STP knobs

- `loop_stp_lambda` (base.yaml:56, default **0.0** = OFF) — loop-axis STP, **this test**.
  transformer.py:783 gates it; loss term at :929. Verdict above: keep off.
- `stp_lambda` (base.yaml:54, default **0.02** = ON) — the *main* "Semantic Tube
  Predictor" (`STPLoss`, prediction.py:30), applied at transformer.py:894/929.
  **NOT part of this test.** Untouched.

## Follow-on (Wolfe, same session): "test stp off completely"

Zero the MAIN `stp_lambda` too. Built `ignore/stp_off_test.sh` — a **1-arm** test:
`self.stp(...)` is called unconditionally (transformer.py:894), so `stp_lambda` gates
only the gradient, not RNG/forward ⇒ a fresh `stp_lambda=0.0` arm is directly
comparable to the existing `loopstp_off.runlog` (stp=0.02). Diff isolates `stp_lambda`.
~70 min, one GPU arm. NOT launched yet (Wolfe needed the GPU ~1hr).
Launch: `setsid bash ignore/stp_off_test.sh > ignore/stpoff_master.log 2>&1 < /dev/null &`
