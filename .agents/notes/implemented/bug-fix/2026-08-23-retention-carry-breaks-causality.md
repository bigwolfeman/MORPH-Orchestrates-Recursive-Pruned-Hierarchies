# Agent Note: the cross-iteration GLA carry breaks per-position causality

Status: implemented

## Problem

`docs/runtime-invariants.md` §5 states the contract: no module may pool statistics
across the sequence axis; every position's output must depend only on positions ≤ t.

Until 2026-08-31 it was violated on the default config by `model.retention_carry`
(then a bool, `true` in base.yaml and every TUL arm): the GLA retention branch
returns a `final_state` — the recurrent state after the WHOLE sequence — and the
core loop feeds it back as the `initial_state` of the next iteration
(`transformer.py` track_ret, in both `_forward_single` and `_tul_core`). From
core iteration 2 onward every position depends on positions after it.

Measured record (all pre-fix):
- Future-corruption probe on `tul-a0-acap1/step_20000`: corrupting tokens after
  position k moves logits at ≤ k by up to 4.08 against a mean |logit| of 2.41;
  exactly 0.000 with the carry off. Eager and fused paths identical (architectural,
  not a kernel bug). Bisect: first divergence at `core.1.retention` on its SECOND
  call — the signature of a carry, not a mask leak.
- The leak premium is LEARNED, not constant: +0.1433 nats val CE on that
  truncated-BPTT checkpoint, but **3.85 nats** on the 30k full-BPTT l2cap run
  (carry-off CE@K6 4.965 vs carry-on 1.119, same weights), whose honest carry-free
  CE@K1 moved only 0.19 nats over its last 20k steps. The l2cap recipe's entire
  "0.233 nats depth-earned" was this channel — carry-off inverts the depth curve
  (4.622@K1 → 5.742@K6). Full audit:
  `lab/experiments/successes/2026-08-31-carry-leak-audit.md`.

## Decision

Option (a) of the original proposal, shipped 2026-08-31 on
`fix/retention-carry-causality`:

- `MORPHConfig.retention_carry` is a mode string, default **"none"**: the core's
  GLA state resets every loop iteration — strictly causal. The pre-fix behaviour
  survives only as the explicit opt-in **"acausal_final"**, for loading and
  diagnosing checkpoints trained before the fix; building a model with it prints a
  loud warning. Bools map for config back-compat (False → "none",
  True → "acausal_final"). All read sites go through the normalizing property
  `retention_carry_mode` — a raw truthiness read would treat "none" as True and
  silently resurrect the leak (a revert-checked test guards this).
- `base.yaml` and `scale30b.yaml` set `"none"`; the whole `tul_*` chain inherits
  it. Historical experiment configs keep their `true` (→ acausal_final + warning)
  so they still describe the runs they produced.
- The KV-cache decode path gates its cross-iteration state seed on
  `retention_carry_mode == "acausal_final"`; it used to seed unconditionally,
  which would have decoded a different model than a "none"-trained one.
- Gate: `tests/test_causality_contract.py` — the old strict xfail is now a hard
  pass on the default config, on BOTH shipped forwards (plain and `_tul_core`
  with a slot layout). The TUL probe compares every packed position (slots
  included) before the corruption frontier: the leak surfaces FIRST at slot
  positions, and a token-only probe with the frontier before the first slot is
  blind — both blindness modes are asserted against in the probe itself.

## Alternatives considered

- **Chunk-boundary causal carry** (feed iteration k+1's chunk c the iteration-k
  state entering chunk c — causal because that state summarizes positions strictly
  before the chunk). Rejected for this fix, recorded for a future prereg: any
  causal mid-sequence injection hands position t TWO summaries of the same prefix
  (iteration k's injected state and iteration k+1's own accumulation), and every
  combine rule (add: double-counts; replace: severs within-iteration recurrence)
  is a new architecture whose training value is unmeasured. That is a
  preregistered-experiment decision, not a bug-fix decision — especially given the
  audit showed the acausal carry's measured "value" was 100% leak.
- **Per-position state carry** ([S, H, dk, dv]): exact and causal but
  memory-disproportionate, and it inherits the same double-counting question.
- **Leave it / fix only at inference**: rejected in the original proposal; the
  30k audit turned "larger than the effects being measured" into "3.85 nats and
  the headline result of the campaign was fake".
- **Silently map old `true` configs to "none"**: rejected — it would change what
  historical configs reproduce without saying so; they map to "acausal_final"
  with a build-time warning instead.

## Consequences

- Every future run is causal by default; teacher-forced CE and generation quality
  become commensurable again.
- Every pre-fix checkpoint is a different (leak-reliant) model: nothing is
  comparable across the change without a re-run, and any absolute CE/PPL quoted
  from a pre-fix run is inflated (0.14-3.85 nats depending on recipe/horizon).
- Carried forward from the original acceptance criteria, still OPEN: (3) the
  re-baseline number — val CE of a model TRAINED with `"none"` (needs a run);
  (4) the one-line which-side-of-the-fix annotation on every `docs/` page that
  states an absolute CE/PPL. Standing rule from the audit: no depth/loop claim is
  admissible without a carry-off (now: default-config) sweep.
- The JAX mirror (`morph/jax/`) was NOT audited for the same path (it lags the
  PyTorch tree; verify before use).
