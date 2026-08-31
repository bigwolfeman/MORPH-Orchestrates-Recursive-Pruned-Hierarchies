# Planned: retention-carry leak audit of the l2cap depth-earning claim

Status: success — all three predictions held; the l2cap depth-earning claim is falsified as leak artifact
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

## Results (2026-08-31, artifacts: /home/wolfe/morph-scratch/tulfm/depth_sweep_carry_off.{json,log}, 48 rows, paired vs the carry-on sweeps)

| checkpoint | mode | CE@K1 | CE@K6 | "earned" K1−K6 |
|---|---|---:|---:|---:|
| tul-l2-cap step_4500 | carry ON (the recipe claim) | 4.6220 | 4.3892 | +0.2328 |
| tul-l2-cap step_4500 | carry OFF | 4.6220 | 5.7418 | **−1.1198** |
| tul-30k step_30000 | carry ON | 4.2917 | 1.1192 | +3.1725 |
| tul-30k step_30000 | carry OFF | 4.2917 | 4.9646 | **−0.6729** |

CE@K1 is bit-identical between modes on both checkpoints (the carry is unused at
K1), confirming the override cut exactly the carry and nothing else. Carry-off
CE rises monotonically with depth on l2cap-4500 (5.148 / 5.509 / 5.671 / 5.731 /
5.742 at K2..K6) and near-monotonically on tul-30k. Span-first CE tells the same
story (30k K6: 0.719 carry-on vs 4.717 carry-off).

Supporting: carry-on CE@K1 across tul-30k checkpoints moved only 4.484 → 4.311 →
4.292 over steps 10k→30k while CE@K6 collapsed 3.782 → 2.250 → 1.119 — 25k steps
of training improved the honest single-iteration model by 0.19 nats and the leak
channel by 2.66. Gen samples corroborate (generation cannot read the future):
tul-30k greedy output is phrase-looping word salad, a CE≈4 model's text, at
teacher-forced CE 1.12.

## Verdict

P1 TRUE (leak at 30k native depth = 4.965−1.119 = 3.85 nats ≥ 1.8). P2 TRUE
(honest earning −1.12 < 0.10 — not reduced, INVERTED). P3 TRUE. The l2cap
recipe's depth-earning is retention-carry exploitation enabled by full BPTT;
with the channel cut, its iterations actively damage the state. No recipe in
the campaign has been shown to earn depth honestly (every other arm was flat
WITH the leak available). Binding consequences executed: dated correction in
the winning-recipe note; 30k head-to-head aborted (arm B cancelled at ~step
500 — its axes would measure leak-exploitation capacity); the 2026-08-23
retention-carry note is now the gating work item for ALL loop claims.

## Updated hypothesis

The looped core has never been shown to compose. The optimizer, given any
cheap channel (identity escape OR future access), takes the channel; full BPTT
selects which channels are learnable. The next loop experiment runs with
retention_carry=false (or a causal carry) from step 0, and every depth claim
must include a carry-off sweep as a standing control.
