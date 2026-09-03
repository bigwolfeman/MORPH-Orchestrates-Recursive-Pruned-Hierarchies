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

- Production TUL arm is `tul_a2` (`tul.tokens_through_core: true`, `tg_restrict:
  false`, `mux_beta: 0`, fused kernels) on the `tul_g0c0` base (retention off,
  spectral cap 0).
- The schedule gains a 1000-step linear LR ramp (`training.warmup=1000`) and stays flat
  at 1e-4 after it. The ramp closed the detonation window 3 of 3 and is 0.14 nats better
  at step 2500. Seq length stays 1024. This is applied on the branch by config override
  today; the winner's config file is updated once the GLA arm and the 20k pair land
  (Wolfe, 2026-09-02 23:35: "we update the config to that winner while we are in this
  branch"). No master merge in this change.
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
- **Restoring GLA as the stabilizer.** Open: it cost 0.18 nats and 0.08 of earning in
  the bisect; three draws under warmup with a frozen selection rule decide whether it
  rides in the 20k pair.
- **Keeping the slot-only loop and fixing the slot input (write-side ladder).** Rejected
  by measurement: no seed mode moved the loop's earning above 0.011.

## Consequences

- The loop's earning under the ramp is 0.046 to 0.049 at 2500 against 0.121 on the flat
  schedule. Whether that is killed or delayed is the open measurement (wu5k, then the
  20k pair). If it is killed, the recipe trades depth-earning for stability and CE, and
  the loop's value has to be re-argued at 20k.
- A late detonation past step 776 has never been observed but is not excluded; the map's
  outward drift is the reason to keep the tripwire on every long run.
- `lab/divergence/_build.py::DepthLever` is the one forced-depth knob for the offline
  probes; A2's knob is `model.cfg.mean_depth`, and the slot knobs are inert on A2.
- Supersedes nothing in `implemented/`; `2026-08-30-l2cap-winning-recipe.md` already
  carries its own correction (the cap and the carry were the leak).
