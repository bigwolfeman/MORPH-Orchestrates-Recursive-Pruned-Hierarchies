# Planned: the write-side ladder — does restoring slot-seed rank let the core loop earn depth?

Status: failure
Date: 2026-09-01 (frozen before launch; follows on E2,
lab/experiments/failures/2026-09-01-bound-seed-rank.md)

## Question

E2 measured that the shared `E_slot` additive term collapses every slot seed
to ~rank-1 regardless of how the content term is built, and that the TUL
core's own loop earns only 0.015 nats (tul-20k core-axis K1-K6) against
0.207 nats for the token-axis loop of the matched notul-20k arm
(lab/experiments/successes/2026-08-31-tul-vs-notul-20k.md). Both numbers
come from the SAME winner recipe. Does restoring the write side — the rank
of what a slot's seed CARRIES INTO the core, before any loop iteration runs
— let the core CONVERT that restored rank into loop-earned depth, or does
the core ignore it regardless?

## Hypothesis

The core loops on slot states built from `E_slot + f(span)`. E2 showed
`f(span)` alone recovers 11x the rank of the composite seed (bound-minus-
E_slot: unit rank 38.30, |cos| 0.061, vs bag-minus-E_slot: 3.47 / 0.507) —
but the SHIPPED seed (with `E_slot`) sits at 1.13-3.07 regardless, because
the shared constant dominates every seed's norm. Dropping `E_slot`
entirely (arms W1/W2 below) tests the first half of E2's own contingency
clause directly, without also touching the rebalancing E2 deferred to E3.
If a rank-restored write side is the loop's actual bottleneck, the core's
per-iteration contribution should move off the 0.015-nat floor toward the
0.207-nat token-axis ceiling. The competing hypothesis (favoured, per the
`.agents/notes/implemented/architecture/2026-08-30-l2cap-winning-recipe.md`
identity-escape law and 15/15 prior TUL-core interventions failing): the
core's contractive map ignores the extra rank because nothing in the LOSS
rewards using it, and the loop stays near-identity regardless of what the
seed carries.

## Method

Four arms, sequential on the 5090, one panel:
`training.batch_size=6 training.seed=1 training.ademamix_alpha_cap=3.5
training.ademamix_t_beta3=3500 training.eval_every=250 training.gen_every=0
training.grad_probe_every=1`, eager kernels (TG-restrict requires it),
`training.steps=5000`, `training.ckpt_every=2500` (both `step_2500.pt` and
`step_5000.pt` are KEPT — no post-run prune, unlike the 20k arms — the
ladder needs the mid-run checkpoint too).

- **R0** — `tul_g0c0` unmodified (the existing 20k h2h TUL winner-recipe
  arm), wandb.name `tul-r0-5k`. **Note:** `tul_g0c0` inherits
  `tul.slot_seed: boundary` from `tul_tg4b` via the `tul_l2 <- tul_l1 <-
  tul_gl1b <- tul_gl1 <- tul_tg4b` chain (confirmed by hydra compose,
  2026-09-01) — R0 is the SHIPPED arm as trained, not a bag_mean baseline.
  Its slot seed is E2's "ship" column (rank_unit 1.58, |cos| 0.773), a
  different composite than either W1 or W2 tests, and closer to bag_mean's
  1.13 than to either no-E_slot column. Reused verbatim because R0's job is
  "the recipe as actually shipped", not "the spec default".
- **R1** — `notul_bg0c0` unmodified (the notul-20k arm's config), wandb.name
  `notul-r1-5k`. The token-axis ceiling: no slots, no TUL machinery, the
  same winner recipe, fused kernels (legal here — no TG-restrict).
- **W1** — `tul_w1` (`morph/configs/tul_w1.yaml`: `tul_g0c0` +
  `tul.slot_seed: content`) — bag-mean, E_slot term dropped. E2's
  "bag_noeslot" seed population made a model arm.
- **W2** — `tul_w2` (`morph/configs/tul_w2.yaml`: `tul_g0c0` +
  `tul.slot_seed: bound`) — HRR-bound span sum, E_slot term dropped. E2's
  "bound_noeslot" seed population made a model arm
  (`morph/model/tul.py::bound_seed`, rotations from
  `build_bound_rotations(d, span_cap, seed=17)` — the exact E2 probe math,
  pinned by `tests/test_slot_seed_modes.py::test_bound_seed_matches_the_probe_reference_math`).

Depth sweep on BOTH checkpoints of every arm (8 sweep invocations total):
`core_depth_sweep.py` (R0, W1, W2 — all TUL arms) / `token_depth_sweep.py`
(R1 — no TUL), depths 1..8, 48 rows, matching every prior ladder rung's
sweep protocol. Gen samples SKIPPED for all four arms — 5000-step
checkpoints are too early in training for gen-PPL/rep/distinct-n to be a
meaningful readout at this scale (every gen-sample rung in this campaign
has run at 20k+ steps); the sweep is the load-bearing instrument here.

Live mechanical gates read from `val/slot_eff_rank` and
`val/slot_pairwise_cos` (the training-time versions of E2's static probe,
already logged by the eval loop for TUL arms) at both checkpoints of W1/W2,
compared against R0.

## Predictions (frozen)

- **P-L1 (W1 rank).** W1's `val/slot_eff_rank` at step 5000 >= 3.0 (E2's
  static "bag_noeslot" unit rank was 3.47; a live, trained, non-boundary
  seed should land in that neighborhood): 70%.
- **P-L2 (W2 rank).** W2's `val/slot_eff_rank` at step 5000 exceeds W1's by
  a wide margin, tracking E2's static gap (bag_noeslot 3.47 vs
  bound_noeslot 38.30, an 11x ratio): 75%.
- **P-L3 (W1 depth-earning).** W1's core-axis K1-K6 (step 5000 sweep)
  exceeds R0's by >= 0.02 nats (clears the 0.015-nat floor measured for the
  shipped boundary seed at 20k): 25%.
- **P-L4 (W2 depth-earning).** W2's core-axis K1-K6 exceeds W1's by a
  margin proportional to its larger rank gain (i.e., rank and depth-earning
  move together across the W1/W2 pair, not just each arm vs R0):
  20%.
- **P-L5 (ceiling).** Neither W1 nor W2's K1-K6 reaches half of R1's OWN
  measured 5000-step token-axis K1-K6 (R1 is run at matched steps precisely
  so the ceiling is a same-horizon number, not the 20k value 0.207): 85%.
- **P-L6 (stability).** All four arms complete 5000 steps without a
  div-guard abort: 70%.
- **Binding.** Decision tree, scored at step 5000 (step 2500 checkpoints
  are read for the WITHIN-ARM trend, not a separate verdict):
  1. **Rank up (P-L1/P-L2 TRUE) AND depth-earning up (P-L3 and/or P-L4
     TRUE)** -> write-side rank is confirmed as a real lever on loop
     earning. Scale the better of W1/W2 to a 20k run and re-run the h2h
     against R0/R1 at matched scale.
  2. **Rank up, depth-earning FLAT (P-L1/P-L2 TRUE, P-L3 AND P-L4 FALSE)**
     -> the core's contractive map ignores the extra input rank regardless
     of what the write side carries — the bottleneck is downstream of the
     seed (the loss, or the core map's own geometry per the
     identity-escape law). Pivot to paid-axis arms (A2 tokens-through-core,
     or asymmetric per-slot depth) rather than further seed engineering.
  3. **Rank FLAT (both P-L1 and P-L2 FALSE)** -> the vectorized
     implementation (or the live eval probe reading it) has a bug — the
     static E2 math is not in question (pinned by
     `tests/test_slot_seed_modes.py`), so a flat live rank means the
     TRAINING-time signal never reaches the seed the way the CPU tests
     verify it does. Debug the seeding path before drawing any conclusion
     about depth-earning.

## Not verified before run

Whether `val/slot_eff_rank` / `val/slot_pairwise_cos` are logged for every
arm at both checkpoint steps or only at eval boundaries that may not align
exactly with `ckpt_every` (if they don't, the mechanical gate reads the
nearest eval instead and the offset gets recorded, not silently ignored).
Training stability of "content"/"bound" past what the 30-step smoke gate
exercises — first real exposure of either mode to a live gradient at scale.
Whether 5000 steps is long enough for a slot-seed rank change to show up as
a depth-earning signal at all (the ladder is short by design — a full 20k
commitment is gated on P-L1 AND (P-L3 or P-L4) per the binding).

## Method amendment — 2026-09-01 20:55 (before launch; predictions untouched)

The three TUL arms (R0/W1/W2) run with `+model.tg_scoped_kernels=true`
(commit this change): the process-global force_eager flag stays OFF so the
structurally-safe fused kernels engage (HC-Cayley, CCA prologue/conv, the
core-region window — where every position is a slot and the TG restriction
is vacuous). Every TG-restricted attention branch stays eager BY
CONSTRUCTION: prelude/coda window calls always carry tg_allow (extra_mask
routes to the reference path unconditionally, attention.py:773) and the TG
compressed branches are pure eager functions. CE stays chunked. R1 (notul)
is unchanged (already fused).

Reason: the post-sacg kineto profile ($Q/prof_tul_g0c0.kernels.txt) showed
no hot kernel — an eager elementwise soup (~13k copies/step) leaving the
GPU ~50% idle. Verification before this amendment: (1) GPU parity A/B, same
model + same packed val batches, eager vs scoped: max logit diff 0.073,
mean 0.0057, 98% argmax agreement (bf16 kernel-order numerics; a masking
bug would be O(1)+); (2) bitwise causality under scoped mode at 2 perturbed
token positions (prefix torch.equal, suffix changed), forward determinism
gate passed; (3) 30-step A/B training smoke: sps 1.41 -> 2.14 (+52%),
peak mem 17.25 -> 12.08 GB, loss inside the documented nondeterminism band
(final val 7.774 vs 7.809); (4) tests/test_tg_scoped_kernels.py (3 CPU
contract tests) + full suite 759 passed / 8 skipped / 1 xfailed.
Smoke gates now REQUIRE kernels=EAGER+TGSCOPED + the "TG SCOPED KERNELS ON"
print on the TUL arms.

All four arms remain internally comparable (identical treatment across
R0/W1/W2; R1 untouched); the depth sweeps still run use_kernels=false via
core/token_depth_sweep.py, so every prereg metric stays on the reference
eval path.

## Results (2026-09-02, runs tul-r0-5k / notul-r1-5k(+retry) / tul-w1 / tul-w2)

All numbers at step 5000 unless noted; sweeps over 48 paired rows in
$Q/nightladder/. R1's first attempt detonated at step 2080 (beta1=0
gradient explosion, preclip total 1.02e9 by step 2000, loss pinned ~7.4 by
clipping; step-1 probe identical to notul-20k's to 4 decimals — a bad
nondeterminism draw of the guardless recipe, not a code change); the
documented same-command retry ran clean and provides the sweep numbers,
with the caveat that the retry is itself a WEAK draw (train 6.16@3000 vs
notul-20k's 4.89@2000), so its loop contribution is the honest matched
number but its absolute CE understates notul.

| arm | K1−K6 | K6 ce_tokens | val/slot_eff_rank | pairwise_cos |
|---|---|---|---|---|
| R0 boundary | +0.0113 | 4.4689 | 40.2 | 0.18 |
| W1 content | +0.0007 | 4.3395 | 73.3 | 0.21 |
| W2 bound | +0.0019 | 4.4170 | 48.2 | 0.19 |
| R1 notul (retry) | +0.1937 | 4.7548 | — | — |

- **P-L1 (70%): TRUE.** W1 rank 73.3 >= 3.0 — but see the instrument
  caveat recorded mid-run (vlt tul-span-jepa, 21:40, before any scoring):
  `val/slot_eff_rank` measures the WRITTEN states (post-core h_slots via
  `_readout`), which climb with training in EVERY arm (R0's boundary seed
  reads 40.2; the sacg smoke read 1.73 at step 30). The threshold was
  anchored on E2's static SEED ranks and is trivially cleared. TRUE as
  written, weak as evidence.
- **P-L2 (75%): FALSE.** W2 48.2 < W1 73.3 — the ordering INVERTS the
  static E2 gap (bound 38.30 vs content 3.47). Static seed rank does not
  survive training contact.
- **P-L3 (25%): FALSE.** W1 K1−K6 = 0.0007 < 0.0313 (R0 + 0.02).
- **P-L4 (20%): FALSE.** W2 0.0019 vs W1 0.0007 — both at the floor;
  rank and depth-earning do not move together.
- **P-L5 (85%): TRUE.** Bar = 0.0968 (half of R1's 0.1937); W1 and W2 miss
  it by two orders of magnitude.
- **P-L6 (70%): FALSE.** R1 attempt 1 div-guard aborted (retry clean;
  2 of 4 paid-axis draws detonated across the night).

## Verdict

**FAILURE of the manipulated hypothesis — the favoured competing hypothesis
held.** No seed mode moves loop earning; the core ignores its input rank.
Binding branch 2 fires (rank moved where measurable, depth flat): stop seed
engineering, pivot to the paid axis. Executed the same night:
successes/2026-09-01-a2-paid-loop.md ran A2 (tokens_through_core) and the
loop earned 0.1685 nats with slots present — confirming the competing
hypothesis's own mechanism ("nothing in the LOSS rewards the loop" on the
free axis; when token CE depends on core output, the loop earns).

## Updated hypothesis

Loop earning follows PAYMENT, not seeding. The write side is a second-order
lever on absolute CE (W1's content seed is worth 0.13 nats over boundary
and stays worth keeping) but not on depth. Next: the efficient hybrid
(A2s / asymmetric depth) per the A2 binding, and the reopened stability
question for paid arms (2/4 draws detonated with no spectral guard).
