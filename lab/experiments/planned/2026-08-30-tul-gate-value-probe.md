# Planned: gate-value probe — did the GRT gate ever open?

Status: planned
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
