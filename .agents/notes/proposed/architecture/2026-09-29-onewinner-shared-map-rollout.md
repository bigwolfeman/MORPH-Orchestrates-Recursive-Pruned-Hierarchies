# Agent Note: share the WTA winner across code_enum_k rollouts ("onewinner")

Status: proposed

**2026-09-29 update:** the ranking mechanism this note describes (a cheap mean-of-cells CE
PROXY, chosen specifically to avoid reordering the forward to reach the true posterior — see
this note's own "Alternatives considered") has been replaced by
[`2026-09-29-onewinner-perf-fused-ce-shared-posterior-grad-map.md`](2026-09-29-onewinner-perf-fused-ce-shared-posterior-grad-map.md):
`_tul_fan_all`'s "map" branch now runs the model's own deployed mixture pass itself, with
gradient, and picks the MAP rollout from the exact posterior `_enum_mix_losses` computes — the
"invasive" reorder this note rejected turned out to pay for itself once the pass is SHARED with
`_forward_tul`'s downstream call instead of duplicated. This note stays as the historical record
of the first shipped mechanism (still correct about the row-count motivation and the
`RolloutSharedDropout` gotcha, both unchanged); read the 2026-09-29 note for the current
ranking/grad-pass design.

## Problem

Arm (b) (`tul_slot_spandec_strict_lxfan4_wta_fp01`, LX-Fan + a2's winner-take-all,
`morph/model/transformer.py::_tul_fan_all`) picks each of the `tul.code_enum_k` = 4
rollouts' winning fan stream INDEPENDENTLY: the no-grad picking table runs M = 4 passes
per stream on the FULL K-fold rollout batch (K*B0 rows each). Those M no-grad passes are
2/3 of the arm's coda cost and most of its ~3.4x slowdown against the plain loop
(`morph/configs/tul_slot_spandec_strict_lxfan4_wta_fp01.yaml`'s own comment). If the K
rollouts' winners mostly agree anyway, that independence is bought for little.

## Proposal

`tul.fan_all_wta_winner: "map"` (default stays `"per_rollout"`, bit-identical to the
tree before this key): share ONE winner per slot across every rollout, picked from the
slot's MAP rollout — the rollout whose best-stream span CE is lowest under one cheap
RANKING pass (K*B0 rows, the mean-of-cells write `state`, already built for every
model) — then run the M picking passes on a B0-row batch (each slot's cells drawn from
its own MAP rollout) instead of K*B0. Net: the arm's total no-grad+grad coda row count
drops from `(M+2)*K*B0` to `3*K*B0 + M*B0` — 48 -> 32 row-units at this arm's K = M = 4,
a third less. Legal under the strict coda's "a prefix cell sees ITSELF and nothing else"
geometry (`morph/configs/tul_slot_spandec_strict.yaml`): a cell's own coda read never
depends on any other cell in the row, so swapping one slot's rollout only ever changes
TOKENS strictly after it — the same Frankenstein-of-rollouts construction the existing
grad pass already builds per FAN STREAM, one axis over.

The MAP rollout is NOT `_enum_mix_losses`'s exact mixture posterior (`softmax_k S_k`):
that runs on a coda pass LATER in the forward (`_forward_tul`'s deployed write), after
`_tul_fan_all` has already returned, so it is unavailable where the winner is picked.
`lab/divergence/wta_winner_agreement.py` (Part 1) measures how well the WTA table's own
per-rollout winners agree with each other, and with the rollout the exact mixture
posterior would call MAP, on two trained checkpoints
(`lxtul-lxfan4-wta-fp01/step_5000.pt`, `lxtul-lxfan4-wta-fp01-ev01/step_3000.pt`) — this
decides whether sharing one winner throws away real signal or a redundant copy of it.

Implementation: `morph/model/tul.py` (`TULConfig.fan_all_wta_winner` +
`__post_init__` refusals), `morph/model/transformer.py` (`_tul_fan_all`'s "map" branch,
`_batch_head`, `_fan_all_winner_capture`), `morph/model/rollout_dropout.py`
(`bypass_rollout_sharing`, see the 30-step-trace finding below), `morph/training/tul_setup.py`
(`KNOWN_TUL_KEYS`, `build_tul_runtime`, the wandb manifest, the build-time banner).
Configs: `tul_slot_spandec_strict_lxfan4_wta_fp01_mapwin{,_s2,_s3}.yaml` (3000 steps,
`lr_decay_steps: 5000`, the `..._ev001` audit's schedule pins). Tests:
`tests/test_tul_wta_shared_winner.py` (13 tests: default-is-unchanged, map-equals-
per_rollout when rollouts coincide, the shared winner matches an independent argmin of
the ranking pass, the pick batch reads only its own MAP rollout's cells — a pure-gather
check plus an end-to-end perturbation with a non-degeneracy precondition, the row-count
cost claim against `_back_region` call spies, config-compose, refusals, a real-dropout
regression on a non-multiple batch). Every claim sabotage-checked (flip argmin, corrupt
the gather indices, revert the B0-row cut, drop the `code_enum_k >= 2` refusal, break the
rollout broadcast, disable the dropout bypass) — each caught by a named test, then
restored.

**Found by the 30-step GPU trace, not by any CPU test (2026-09-29).** Every test above
builds its model at `dropout: 0.0` (this file's own "test fixtures run at dropout 0"
convention), which makes `RolloutSharedDropout.forward` a no-op (`if ... or self.p ==
0.0: return x`) and never exercises its `rows % n_rep` check. On the real
`..._mapwin.yaml` config (`model.dropout` inherited from arm (b), not 0), the M no-grad
picking passes' B0-row batch hit that check directly: `self.coda`'s dropout modules are
`RolloutSharedDropout(n_rep=R)` for the WHOLE model once `code_enum_k > 1`, built for the
R*B0-row passes every OTHER coda call in this method makes, and B0 is generally not a
multiple of R (2 vs 4 in the CPU fixture; 6 vs 4 in the GPU trace — the crash was
`RuntimeError: batch 6 is not a multiple of n_rep 4`). Fix: `bypass_rollout_sharing`
(`morph/model/rollout_dropout.py`) makes those dropout modules fall back to ordinary
independent-per-row dropout for exactly the M picking passes, wrapped in a context
manager scoped to `self.coda`. This is legal specifically because the picking passes run
under `torch.no_grad()` — `RolloutSharedDropout`'s `n_rep` is fixed at construction
because normally the mask decision must survive a gradient-checkpoint recompute, and a
no-grad call has no recompute, so nothing depends on the decision after the fact. It is
also the semantically CORRECT behavior, not merely a permissive workaround: the B0-row
picking batch holds ONE row per sample, each already carrying its slot's own MAP-picked
rollout's cells, so there is no rollout comparison left within a row for a shared mask
to protect. A new regression test (`test_map_picking_passes_survive_real_dropout_on_a_
non_multiple_batch`, dropout 0.2) pins this; sabotage (disabling the bypass) reproduces
the exact trace error and is caught. This is a genuine gap the CPU test suite's
dropout-0 convention could not have caught by construction — worth remembering for any
future coda-touching change under a code/rollout model.

## Alternatives considered

- **Reuse the full per-rollout table's own `ce.min(-1)` for free.** The FIRST
  implementation attempt: since the per-rollout table (`ce`, `[R*B0, S, M]`) is already
  built under `"per_rollout"`, reuse `ce.min(-1)` to rank rollouts with NO extra pass.
  Rejected after `test_the_picking_passes_run_on_b0_rows_not_kb0` caught it: that table
  IS the K*B0-row cost "map" exists to cut, so reusing it means building it AND the
  cheaper B0-row table on top — strictly MORE coda passes than `"per_rollout"`, not
  fewer. The shipped design skips that table entirely under `"map"` and pays for one
  cheaper K*B0-row ranking pass (the mean-of-cells write) instead.
- **Reorder the forward to reuse `_enum_mix_losses`'s exact posterior.** Would give the
  TRUE MAP rollout (the mixture posterior the model actually trains on) instead of the
  best-stream-CE proxy, but `_enum_mix_losses` runs on the DEPLOYED coda pass built well
  after `_tul_fan_all` returns (`_forward_tul`'s ordering: fan first, for the aux-reader
  seams; the main coda write and its mixture loss last). Moving that pass earlier is
  invasive (RNG ordering for the `token_state_dropout`/`fan_select_eps` draws, every
  `per_rollout`-mode caller's bit-identity claim) for a design whose whole selling point
  is a SMALL, contained diff. Rejected for this change; Part 1's agreement measurement
  is the check for whether the proxy is good enough to ship without it.
- **A single write (mean-of-cells) with no per-rollout ranking at all**, i.e. always use
  rollout 0 or a fixed rollout as MAP. Rejected: that is not "MAP" by any definition and
  would silently starve the other K-1 rollouts of ever supplying a winner — the same
  collapse `fan_select_eps`'s random-write guard exists to prevent, one level up (over
  rollouts instead of over fan streams).

## Acceptance criteria

- `tests/test_tul_wta_shared_winner.py` passes (12/12), the pre-existing LX-Fan/credit
  suite (93 tests) passes UNCHANGED with the default `"per_rollout"`.
- `lab/divergence/wta_winner_agreement.py` reports, on both checkpoints: the fraction of
  slots where all R rollouts agree, the mean pairwise agreement, the MAP-rollout
  agreement against each other rollout, and the chance level (1/M = 0.25) — read BEFORE
  trusting the design's premise (if MAP agreement is near chance, "map" throws away real
  per-rollout signal and should not be queued as a training recipe, only kept as a
  correctness-tested, unused knob).
- A 30-step GPU trace of `..._mapwin` against `..._ev01` (same steps, same schedule)
  shows a finite loss, the build banner's "winner='map'" line, and reports tok/s + peak
  memory for both so the row-count claim is checked against real wall clock, not only
  the CPU test's `_back_region` call counts.

## Risks

- The best-stream-CE ranking pass is a PROXY for the mixture posterior, not the
  posterior itself (see Alternatives). If Part 1 measures LOW agreement between the two,
  "map" would be sharing a winner picked by the wrong criterion — a correctness-clean
  but statistically weak design. The queue lines exist so a real run can show whether
  this matters for CE, but the arm should not be treated as a drop-in replacement for
  (b) until that run reports back.
- `"map"`'s stats (`wta_oracle_ce`, `wta_single_ce`, `wta_pick0`, `wta_share_k{i}`) read
  from the B0-row PICK table, broadcast to every rollout — NOT each rollout's own
  independent table, which no longer exists under `"map"`. A dashboard comparing these
  names across `"per_rollout"` and `"map"` arms is comparing two different readings
  under the same key names; this is documented in the field's own comment and in
  `_tul_fan_all`'s docstring, not hidden, but it is easy to misread on a wandb chart.
