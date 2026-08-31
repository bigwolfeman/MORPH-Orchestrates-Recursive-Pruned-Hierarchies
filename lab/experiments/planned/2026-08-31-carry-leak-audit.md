# Planned: retention-carry leak audit of the l2cap depth-earning claim

Status: planned
Date: 2026-08-31 (frozen BEFORE any probe runs; written on discovery of the
tul-30k val CE anomaly, while arm A's gen samples were still generating).

## Question

Arm A (tul-30k) reached val CE 1.19 / PPL 3.3 at 30k steps — impossible as
language modeling at 283M on fresh OWT (train-side token CE on never-seen docs
is ~1.7 after subtracting the mux aux from the printed loss). The known
non-causality (`model.retention_carry: true` carries the GLA whole-sequence
final state into core iteration 2+; measured +0.143 nats on a 20k-step
truncated-BPTT arm, note 2026-08-23) is a LEARNED channel: full BPTT — the
l2cap recipe's key ingredient — is exactly what lets gradients train the
exploit, and every falsified ladder arm was truncated. Two questions:

1. How much of tul-30k's CE collapse is the carry channel?
2. Does the l2cap 4500-step depth-earning claim (0.233 nats, CE 4.622@K1 →
   4.389@K6) survive with the carry cut at eval? K=1 is carry-free by
   construction (no second iteration), so CE@K1 is already honest; the claim
   lives or dies at carry-off CE@K6.

## Method

`core_depth_sweep.py` with config override `model.retention_carry=false`
(same weights, carry cut at eval — the 2026-08-23 carry_leak_cost.py pattern),
48 rows, depths 1..8, paired against the existing carry-on sweeps
(tul-l2-cap 4500 from the ladder; tul-30k @10k/20k/30k from tonight):

- `l2cap4500-nocarry = tul_l2 = checkpoints/morph/tul-l2-cap/step_4500.pt = model.retention_carry=false`
- `tul30k-nocarry    = tul_l2 = checkpoints/morph/tul-30k/step_30000.pt   = model.retention_carry=false`

Runs ONLY in a GPU window (one-trainer rule): after the 30k pair completes,
or immediately if Wolfe kills arm B. Secondary no-GPU evidence, already in
flight: arm A's gen samples (generation cannot read the future — sample
quality tracks honest CE, so 4500-grade samples at "CE 1.19" confirm the leak).

### Method amendment — 2026-08-31 05:20 (before any probe ran)

Arm B is PAUSED at ~step 500 (nothing checkpointed; relaunch script
`$Q/run_30k_armB.sh` is byte-equivalent to the pair runner's B section) so the
audit runs now instead of in ~6.5 h. Rationale: B's carry summarizes all 1152
positions vs A's 64 slots, so leak-dominated CE makes H1/H2 a comparison of
exploitation capacity across geometries, and A's gen samples (greedy: phrase-
looping word salad, rep4 0.125 — a CE~4 model's output, not PPL 3.5) already
confirm the leak qualitatively. Pause cost ~15 min; audit ~10 GPU-min.

## Predictions (frozen)

- **P1 (leak magnitude at 30k).** tul-30k carry-off CE at its native depth
  (mean 6) is ≥ 3.0 nats, i.e. ≥ 1.8 nats of the collapse is the carry
  channel: 70%.
- **P2 (the recipe claim).** l2cap-4500 carry-off depth-earning
  CE@K1 − CE@K6 < 0.10 nats (less than half the claimed 0.233 survives): 55%.
  Prior lean: the mux targets its own span and the σ-cap bounds expansion —
  neither needs the future — but the carry is the one channel full BPTT can
  train that no truncated arm could, and the falsification table is exactly
  "full BPTT wins, everything else flat".
- **P3 (monotone leak).** tul-30k's carry-off deficit grows with forced depth
  (more iterations = more carry reads): 75%.
- **Binding.** P2 TRUE (earning < 0.10 honest) ⇒ the l2cap recipe note and the
  30k prereg H3/H5 are re-scored as leak-confounded; the winning-recipe note
  gets a dated correction; carry-off (or a causal carry) becomes a mandatory
  ingredient before any further loop claims. P2 FALSE (≥ 0.10 survives) ⇒ the
  recipe stands with a reduced number; correction records the honest value.
  Either way the 2026-08-23 bug-fix note is promoted from proposed to active
  work — a 30k run turning 0.14 nats into ~2+ makes it unshippable.

## Not verified before run

Whether retention_carry lands as an eval-time rebuild cleanly through
core_depth_sweep's build_cfg override path (carry_leak_cost.py used its own
loader); whether the sweep's skip_samples=0 rows (seen once at step ~0-100)
bias the absolute CE — the paired carry-on/off delta cancels row identity.
