# Agent Note: the coda reads the thought

Status: proposed

## Problem

The TUL spec (`docs/tul-spec.md` §3.4, line 418: `x_coda = input_norm(x)  # tokens SKIP
the core`) hands the coda every token's own prelude state. The prelude is a global
transformer over every position, so a token of span i+1 reaches the coda already carrying
all of span i, and the slot's looped state `z_i` is two extra cells among ~1,100 that the
coda has no reason to decode from. Every slot-loop reading in the record measures a coda
that was allowed to skip the loop: token K-curves flat to 1e-5 over forced slot depth on
the A1 arm under absmean (E13) and under norm_match (2026-09-09), plan worth 0.05 nats.
The mask arm (`tg_restrict`) tried to force reliance after the fact by starving the
prelude too, and is worse everywhere. Wolfe, 2026-09-09: "the design should be that the
prelude evolves the loop state, and the final z + MUX tokens are what conditions the
coda. This skipping will cause the coda to want to ignore the loop. [...] I see a mistake."

## Proposal

Wire the coda so that `z` is its only route to earlier spans, Block Transformer's contract
(the local decoder gets the block embedding plus the block's own tokens):

- `tul.coda_token_input: embed` — the coda's token carrier is the token embedding the
  prelude took in (`x0`), not the prelude's output; the coda's injections at slot cells
  are zeroed so a cell carries `z` alone.
- `tul.tg_restrict: true` with `tul.tg_restrict_scope: coda` — the same-span-or-slot mask
  on the coda only; the prelude stays global so the slot's seed `e_z` sees the whole past.
  In the coda a slot cell attends slot cells only (`tg_allow_mask(slot_queries_slots_only)`),
  and the CCA Q/K conv taps and the value shift are reset at span/slot boundaries
  (`attention.segment_causal_conv`, `tg_seg = 2·bag_id + slot_mask`).
- `tul.bcast: true` — spec §3.5's unpack row, implemented: `z_i` through offset-indexed
  zero-init linears (`TULSlots.W_bcast`, `[span_cap, d, d]`), added to span i+1's token
  coda inputs (`TULSlots.unpack`, `unpack_index`).

The contract is a test (`tests/test_tul_unpack.py`): with `z` zeroed, the logits of span
≥ 2 tokens are invariant to span 0's content; the shipped path leaks 0.09 nats of logit
movement. Arm: `tul_slot_unpack_norm_match.yaml`, prereg
`lab/experiments/planned/2026-09-09-arc-coda-reads-the-thought.md`.

## Alternatives considered

- **Keep the prelude state as the coda's token input and rely on the mask** (the shipped
  mask arm): the mask starves the prelude, and the coda still re-summarises each span at
  its own slot cells from the token states (measured: 0.69 nats of logit movement with
  `z` zeroed). Rejected.
- **Raw embeddings plus the mask, without the coda rule for slot cells**: the coda
  rebuilt a span summary at the cell from the span's embeddings (0.07 nats with `z`
  zeroed). Rejected; the coda rule is required.
- **Coda rule without the conv/value-shift reset**: the CCA conv (kernel 4, two stages)
  and `W_v_prev` hand the boundary token's coda state to the cell's key and value, a
  full-width bypass of `z` (0.07 nats; exactly 0.0000 with the taps cut). Rejected.
- **Additive conditioning of the coda's token states on `z` without the mask**
  (`cond_layers`, tested in the think-once panel): keeps the direct path, so the coda can
  still ignore `z`. Not the contract.
- **Per-offset low-rank or shared unpack linears**: cheaper, but the spec's row names
  full offset-indexed linears and the cost is ~15 GFLOP per forward slot-major. Kept the
  spec's form.

## Acceptance criteria

The arm's token K1−K6 over forced slot depth leaves zero with the CI above 0 (P-unpack-c
of the prereg), or the reading names which of map / target is the next limiter
(P-unpack-b true, c false). Either way the §3.4 sentence "tokens SKIP the core" and the
`x_coda = input_norm(prelude)` line are amended in the spec to state which coda input the
shipped TUL uses, and the prefix-route rationale in the spec's table cites this note.

## Risks

Eager attention only (the restriction is eager-only): ~1.3–1.5x the fused path's wall
clock. The raw-embedding coda is a weaker decoder than the prelude-state one; the price
at 5k may be large and is a horizon reading, not a verdict. The segment reset makes the
first three tokens of a span blind to the previous span's tail through the conv, which the
spec's window never restricted before.
