# Agent Note: the paid loop plus a 1000-step LR warmup is the TUL recipe

Status: implemented

## Problem

TUL as specified looped the core on slot positions only, so token CE never paid for
the loop and the loop never earned depth (0.015 nats at 20k; 0.357 nats behind notul).
Making the tokens run the core (arm A2) restored the earning (0.1685 at 5k, best 5k
checkpoint of any arm) and reopened an early-training detonation that belongs to the
winner recipe itself (retention off, cap 0, ternary on, flat 1e-4, warmup 0): ~70% of
draws, onset in steps 200 to 775, ternary QAT as the trigger surface, the healthy core
map expansive and drifting outward. The full record, with every number's file, is
[docs/tul-paid-loop-recipe.md](../../../../docs/tul-paid-loop-recipe.md).

## Decision

- The schedule gains a 1000-step linear LR ramp and stays flat at 1e-4 after it:
  `training.warmup: 1000` in `morph/configs/base.yaml` since 2026-09-03 (Wolfe,
  2026-09-02 23:35: "we update the config to that winner while we are in this branch").
  The ramp closed the detonation window 9 of 9 and is 0.14 nats better at 2500 and 0.09
  at 20k for the plain model. Seq length stays 1024. No master merge in this change.
- TUL stays default-off in `base.yaml`. The paid TUL arm is `tul_a2`
  (`tul.tokens_through_core: true`, `tg_restrict: false`, `mux_beta: 0`, fused kernels)
  on the `tul_g0c0` base (retention off, spectral cap 0). It is the arm to keep
  measuring, not the production recipe: at 20k on the ramp it is 0.022 nats behind the
  plain model on 480 identical rows at 1.33x the wall clock
  (`lab/experiments/failures/2026-09-02-warmup-20k-pair.md`).
- Any 20k comparison on the ramped schedule reruns the notul baseline on the same
  schedule. The flat-schedule ledger numbers stay valid for flat-schedule runs only.
- Every runner applies the measured abort rule: `preclip/total > 1e4` at any step ≥ 200
  kills the draw (0 false positives in 44 healthy runs). Trainer-side abort-and-retry
  is a follow-up.

## Alternatives considered

- **Spectral cap or projection on the core weights.** Rejected: four variants failed
  against the takeover, the cap kills depth-earning, and a uniform rescale cannot slow
  the alignment the Jacobian ladder shows.
- **Slowing or freezing the ternary scale gamma.** Rejected by measurement: EMA at 0.99
  detonated 2 of 3 and cost 0.30 nats; a hard freeze detonated 3 of 3 and never learned.
- **Dense warmup before ternary QAT.** Rejected by Wolfe: ternary weights organize
  differently from dense ones, so the switch is a re-init.
- **A seq-length curriculum instead of the LR ramp.** Deferred: a single stability probe
  was not worth the GPU; it is part of later warmup-on-length work.
- **Restoring GLA as the stabilizer.** Rejected by measurement (2026-09-03): under the
  ramp it is inert, 0.004 nats and 0.002 of earning for 19M parameters, so the frozen
  rule kept retention off for the pair. It stays a capability question (raven).
- **Making `tul_a2` the production recipe.** Rejected by the 20k pair: the plain model
  on the ramp is ahead at matched steps (0.022) and clearly ahead at matched wall clock
  (0.12). The ramp is what won.
- **Keeping the slot-only loop and fixing the slot input (write-side ladder).** Rejected
  by measurement: no seed mode moved the loop's earning above 0.011.

## Consequences

- The ramp trades loop earning for stability and CE. The plain model's K1−K6 is 0.04 at
  every checkpoint to 20k (0.207 on the flat schedule) and it is still the best CE in
  the campaign; A2's earning grows 0.041 → 0.100 by 20k and repairs a depth-1 path that
  is 0.084 worse. A2's matched-step deficit closes monotonically (0.132 at 5k → 0.012 at
  20k); the 40k continuation that would show a crossing is preregistered and unlaunched.
  The loop's value to the production recipe is now ~0.04 nats and has to be re-argued
  before the mean depth of 6 is kept.
- A late detonation past step 776 has never been observed but is not excluded; the map's
  outward drift is the reason to keep the tripwire on every long run.
- `lab/divergence/_build.py::DepthLever` is the one forced-depth knob for the offline
  probes; A2's knob is `model.cfg.mean_depth`, and the slot knobs are inert on A2.
- Supersedes nothing in `implemented/`; `2026-08-30-l2cap-winning-recipe.md` already
  carries its own correction (the cap and the carry were the leak).
