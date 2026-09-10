# Agent Note: the slot re-reads its span each pass

Status: rejected — the arm read flat: the read is used (5 % of the state every pass) and the depth curve is the unpack arm's

## Problem

The unpack arm (`2026-09-09-the-coda-reads-the-thought.md`) made the coda depend on the
looped slot state z: zeroing z costs 0.811 nats at the first token of a span. The depth
curve stayed flat (tokens K1−K6 +0.0005, K3−K6 +0.0000 at 5,000 steps): a slot looped six
times hands the coda the same z as a slot looped once. The coda's incentive is no longer
the suspect; the map on slot states is. Inside `_tul_core` the compact sequence holds only
slot cells. A pass can mix slots (causal attention over at most 64 summaries) and apply
the MLP to a summary the prelude already wrote. It cannot look at a token. The paid loop,
the one arm on this tree whose loop earns depth, re-reads every token state every pass
with a refined query. The slot loop has no equivalent of that read.

## Proposal

`tul.reread` (`morph/model/tul.py::TULReread`, wired in `_tul_core`): before the core
blocks run on each pass, the slot state becomes a query (RMSNorm, W_q) over keys and
values built ONCE per forward from `input_norm(prelude)` at the token positions (W_k, W_v);
the read goes through a zero-init W_o and is added to the slot's carrier on every stream.
`reread_scope` picks what a slot may read: `span` (its own span's tokens) or `causal` (its
own span and every earlier one); slot cells are never read (`reread_allow`). The read is
inside `_core_step`, so the gain hinge measures the map WITH the read and the checkpointed
step recomputes it. Plain `nn.Parameter` weights, never ternarised, initialised from a
private generator so the model's RNG stream and every other weight are untouched. At W_o
zero the forward is the no-reread forward bit for bit. Cost: S×L attention per pass (64 ×
1,152 at seq 1,024), 4.2M parameters at d_model 1,024.

## Alternatives considered

- **Tokens through the core (the paid loop).** Earns depth (K1−K6 0.120) and pays the loop
  on every position, which inverts TUL's purpose. Rejected by Wolfe 2026-09-09.
- **Re-seeding the slot each pass from the span (bag mean or boundary embedding).** A
  fixed input per pass; the injection already re-injects e each pass and the state still
  does not move. The read differs in that the QUERY changes with the state.
- **A larger compact sequence (prefix cells or several slots per span).** More summaries
  of the same content; no token in view.
- **Cross-attention in the coda to the slot trajectory (the spec's `xattn`).** Acts after
  the loop; it cannot give a pass something new to compute.

## Acceptance criteria

- `tests/test_tul_reread.py`: mask exactness for both scopes, no slot cell ever read;
  reread=false builds nothing and reread=true at W_o zero is bit-identical on the slot
  states and the logits; the read is live once W_o moves; a token of a later span never
  moves an earlier slot, with a fixture-sensitivity check (a broken mask must move it);
  the paid loop, `n_core` 0 and a bad scope refuse. 21 tests, passing.
- The arm `slot-unpack-reread` reads against `lab/experiments/planned/2026-09-10-arc-slot-reread.md`
  (P-rr-c: tokens K3−K6 above 0.002 with the CI above 0). The note moves to
  `implemented/` or `rejected/` on that reading.

## Risks

- The read gives the map a second look at the span on pass 1 and nothing on later passes
  (H-rr-1′ in the prereg); K1−K6 moves and K3−K6 does not.
- The hinge now measures a map that includes the read; its typical-gain reading may
  shift and the constraint may bind where it did not.
- The read is eager attention under the mask (no kernel); the rate rule guards.

## Rejection (2026-09-10)

The arm `slot-unpack-reread` ran (`lab/experiments/failures/2026-09-10-arc-slot-reread.md`):
healthy, the read trained (W_o norm 9.3 at 5,000) and adds a constant 5 % of the state on
every pass, and the depth curve is the unpack arm's (tokens K1−K6 +0.0003, K3−K6 0). The
read cannot start an iteration that the core blocks do not continue; the blocks output
1.4–5.5 % on slot states with or without it. Kept as a knob (`tul.reread`, default off,
bit-identical) until the TUL finalize, when it goes out with the other dead arms.
