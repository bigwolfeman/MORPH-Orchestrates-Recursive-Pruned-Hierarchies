# Planned: G1 + G2 — the GRT recurrence gate, alone and on top of the cap

Status: planned
Date: 2026-08-30 (frozen before launch). Program note:
`.agents/notes/proposed/architecture/2026-08-30-gate-ladder-program.md`. Machinery:
`morph/model/recur_gate.py` + `_tul_core` wiring, tests
`tests/test_tul_recur_gate.py` (8 passed; full suite 711 passed, 2 xfailed).
Configs: `tul_g1.yaml` (= tul_l1 + recur_gate grt — full BPTT, NO cap),
`tul_g2.yaml` (= tul_l2 + recur_gate grt — full BPTT + σ≤1.5 projection). Gate =
verified GRT Eqs. 4–5 (App. A defaults: bias +4, σ_g 0.1, τ 1.0), state+prelude
keyed only; their Eq. 2 re-injection is already covered by MORPH's per-iteration
injection (verified: `inj_core_terms` enters every `_apply_core_step`).

## Question

Contractivity by architecture vs by constraint. The uncapped full-BPTT loop (l1)
is depth-FLAT (earned 0.0016); only the cap has ever caused earning (l2cap,
0.2328). Does the gated convex blend CAUSE earning where the bare loop does not
(G1)? And does it preserve or destroy the cap's earning when stacked (G2)? The
sharp risk, from this morning's cond-zero probe: the gate's copy branch is an
easy identity shortcut — state-keyed, so not banned, but the experiment decides
whether state-keyed shortcuts are safe or only index-keyed ones were.

## Reference numbers (fixed, sweep instrument, 48 rows)

l1: K1 4.5202 → K6 4.5186, earned 0.0016. l2cap: K1 4.6220 → K6 4.3892, earned
0.2328; CE@4250 4.3489; 66 min; greedy rep4 0.61. l1 wall-clock 66 min. Panel:
batch 6, 4500 steps, seed 1, eager, mux on. GRT's own short-horizon note
(Table 7, 2k steps): the gate is a slow-accruing mechanism (−0.115 at 2k vs
−0.198 for re-injection) — 4500 steps may undersell it.

## Predictions (frozen)

- **P1 (G1 stability).** S1-clean: 80% (l1 itself completed; the gate starts
  near-identity, which can only calm the early loop).
- **P2 (G1, binding).** Depth sweep earned ≥ 0.10 nats: 30%. My prior leans
  fail at THIS budget: g≈0.98 at init passes 2% of the proposal per iteration,
  and 4500 steps is short for a mechanism GRT themselves show accrues late.
- **P3 (G1 CE).** CE@4250 ≤ 4.45: 55%.
- **P4 (G2, binding).** Depth sweep earned ≥ 0.15 nats (retains ≥64% of
  l2cap's): 50%. Coin: the cap's mechanism is proven, but the near-identity
  gate throttles early loop training and may starve it at this budget.
- **P5 (G2 CE).** CE@4250 < 4.3489 (beats l2cap): 30%.
- **P6 (wall-clock).** Each arm ≤ 72 min (l1/l2cap ~66 + gate MLP overhead):
  75%.
- **Decision rules (binding).** P2 passes ⇒ contractivity-by-architecture works
  here; next is G4 (uniform {1..R} depth) on G1's recipe, and the cap becomes
  optional. P2 fails but P4 passes ⇒ the gate is at best neutral: keep the cap,
  test G3 (decode-gate) on l2cap unchanged, drop gate-only lines. P2 and P4
  both fail ⇒ the state-keyed copy branch ALSO poisons formation at this
  budget — the "shortcut" hypothesis generalizes beyond index-keying; G3
  proceeds on l2cap, and any future gate must be output-bounded rather than
  input-blended (e.g. gated delta, not convex copy). If G1 diverges while G2
  trains clean, the gate alone is insufficient contractivity control at depth
  6-8 regardless of P2.

## Method

1. Smoke G1 first (steps=12, eval_every=5): gate on exit 0 + a gate-mean print
   is NOT built — gate liveness is proven by construction tests; the smoke
   gates VRAM/NaN only. Smoke G2 additionally requires the projection-armed
   line (`Core spectral PROJECTION ON: cap=1.5`).
2. Run tul-g1 then tul-g2, panel flags, seed 1; each + 48-row depth sweep
   (auto mode — no sigma path, so auto = forced-depth) + tul_samples;
   checkpoints pruned to step_4500.
3. Wall-clock from queue-log START→DONE stamps. Artifacts →
   `lab/experiments/results/2026-08-30-tul-gate-pair/`.

## Not verified before launch

The gate has never run on GPU (CPU tests only). Its interaction with the
spectral projection (G2) is untested beyond construction. The bf16 behaviour of
sigmoid(+4)-dominated blends over 4500 steps is unmeasured. Gate-MLP wall-clock
overhead is estimated, not measured.
