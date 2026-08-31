# Planned: gate-value probe — did the GRT gate ever open?

Status: failure
Date: 2026-08-30, frozen before the run (queued behind the gate-pair chain; runs
on the finished checkpoints, no training). Both gate arms are depth-flat (G1
+0.0002, G2 −0.0002). Two stories: (a) the gate never left its +4-bias identity
basin (g stayed ≈0.98 — the budget was too short for a mechanism GRT show
accrues late, and a longer run might differ); (b) the gate OPENED and the loop
is dead anyway (the shortcut absorbed formation — no budget rescues it).

## Predictions (frozen)

- **V1 (binding).** Mean realized gate value on real eval batches (48 rows,
  active slots only), both arms, stays > 0.90 (never opened): 65%. The +4 bias
  is 4σ_g from neutral and nothing in a flat-depth loss pushes it down.
- **V2.** G2's gate is FURTHER open (lower mean g) than G1's: 40% — the cap
  constrains the proposal branch, which weakly incentivises using it; but with
  no depth signal either way this is close to a coin.

## Method

One eval pass per arm (tul-g1, tul-g2 step_4500), forward hook on
`tul_recur_gate` collecting g over 48 rows; report mean/p10/p50/p90 over
active-slot elements and per-iteration means (does g drift open across t?).
Interpretation rule, fixed now: V1 holds ⇒ story (a) — record "undertrained
gate" as the open question, but the gate line STAYS closed at panel budget (the
decision rule already fired); V1 fails (gate open) ⇒ story (b) — input-blended
gates are dead at any budget, strengthen the output-bounded requirement.

## Not verified before launch

Whether hook capture at eval reflects training-time gating (no gate noise at
eval — by design, eval g is the deterministic gate).

## Results

48 rows per arm, 62.0M gate elements each, active slots only, eval (no gate noise).

| arm | mean | p10 | p50 | p90 | per-iteration means |
|---|---|---|---|---|---|
| tul-g1 | 0.9669 | 0.9258 | 0.9844 | 0.9922 | 0.945→0.977 (rising) |
| tul-g2 | 0.8551 | 0.4043 | 0.9766 | 0.9922 | 0.760→0.907 (rising) |

- **V1 FAIL** (65% prior): G2's mean 0.855 < 0.90 — the gate opened under the cap.
- **V2 PASS** (40% prior): G2 is further open than G1, as the long-shot predicted.

## Verdict

Both predictions landed against my priors → filed to failures per convention —
and the content is the day's most useful mechanism picture. G1 never left the
identity basin (story a). G2 opened — p10 0.40, a real proposal-using gate — and
its depth curve is still flat to ±0.0002 (story b), which closes the
"undertrained gate" escape for the capped variant. The rising per-iteration
means (0.76→0.91) show WHAT the open gate learned: front-load the
transformation into early iterations, then copy — the gate spends its freedom
shutting the loop's tail down. Input-blended gates are structurally aligned
with killing depth, not enabling it.

## Updated hypothesis

Folded into the gate-pair filing's updated hypothesis (gated-delta with a floor
vs identity-escape blends); see
`../successes/2026-08-30-tul-gate-pair.md`.
