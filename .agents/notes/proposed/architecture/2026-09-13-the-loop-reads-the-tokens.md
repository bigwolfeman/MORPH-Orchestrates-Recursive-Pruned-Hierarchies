# Agent Note: the loop reads the tokens, span-restricted

Status: proposed

## Problem

Inside `_tul_core` the compact sequence holds slot cells and nothing else, so a pass has
nothing new to look at: a slot's input is one span's summary, fixed before the loop
starts, and the state is settled after one pass on every target anyone has posed (K0−K6
0.0015 / 0.107 / 0.019 with passes 2−6 at ≤ 0.002, the 2026-09-10 geometry audit).

Two arms tried to give a pass new input and both read flat: `tul.reread` (the slot
cross-attends the FROZEN prelude token states each pass) K1−K6 **+0.0003**; and
`tul.core_token_aux` (tokens through the core in TRAINING only) **+0.0005**, while the
same six blocks became a **1.10-nat** better token map. The coretok verdict: *the core is
not under-trained, it is under-USED.*

## Proposal

`tul.loop_reads_tokens` (default false, bit-identical): the SHIPPED core stage runs
`_core_region` over every position — tokens and slot cells in one sequence, the per-SAMPLE
Poisson depth — under the span-restricted relation
`causal AND (bag_id[i] == bag_id[j] OR slot_mask[j])` (`_core_token_aux_kwargs`, the ONE
home the aux path already uses). A token reads its own span's tokens and reaches every
earlier span ONLY through a slot cell; the coda's reach is unchanged. There is no prefix
write: a cell's looped state is already at its own position when the core returns, and `z`
is `gather_valid` at the slot's FIRST cell, the same seam every other arm reads.

**No training arm is queued** (Wolfe, 2026-09-13). What runs is the eval probe: the
`--depths 1,2,3,6` extension of `core_token_aux_probe.py` on the existing
`slot-spandec-strict-coretok` checkpoint, which asks whether the token-reading core USES
its depth. That single reading decides whether the fault is the map or the input, and it
costs one checkpoint instead of a 55-minute run.

## Alternatives considered

* **The paid loop (`tul.tokens_through_core`).** Already shipped once and already
  refuted: the core is UNRESTRICTED over the packed row, so every token reads every
  earlier token directly, the cells carry nothing anyone needs, and the arm is 0.125 nats
  behind the plain model under `norm_match` (2026-09-09). The build refuses the two knobs
  together rather than letting a name promise the restriction.
* **Keeping `core_token_aux` and sweeping its weight.** Cheaper, and it leaves the shipped
  forward alone. Rejected as the mechanism because the aux is training-only by
  construction, so it can never test the half of the claim that matters — that the loop
  reads the tokens at INFERENCE. Kept as the probe's subject.
* **`tul.reread` with a trainable read instead of a frozen one.** Rejected: reread's `W_o`
  was already used at ~5 % per pass, so the bottleneck was not the read's freezing.
* **Running the arm anyway, at the panel's budget.** Rejected on cost: about 44
  block-passes per generated token against the slot loop's 10.59
  (`docs/tul-as-memory.md`), which abandons "think once, decode cheap" — the thing TUL is
  for. The knob stays so the arm can be queued deliberately rather than by accident.

## Acceptance criteria

1. OFF is byte-identical (`W_prefix` is the only `state_dict` difference; the relation
   builder is never reached). **Met.**
2. The span restriction holds two-sided, and the SHIPPED forward uses that exact relation
   (spied, not re-derived). **Met.**
3. Eval equals the training forward's CE at a pinned depth. **Met.**
4. The forced-depth sweep reads THIS mode's depth from `model.cfg.mean_depth`
   (`uses_sample_depth` in `lab/divergence/_build.py`, used by `core_depth_sweep.py`).
   **Met**, with a test that `slot_mean_depth` is NOT the lever here.
5. The probe answers whether the aux core earns depth. **Not met — queued.**

Prereg: `lab/experiments/planned/2026-09-13-arc-loop-reads-tokens.md`.

## Risks

* **The name oversells the mechanism, and an independent review measured why.** The knob
  LOOPS the token states — the coda reads the CORE's output at token positions — so it is
  the paid loop WITH a span restriction, not the slot loop. "Think once" is gone; only the
  cross-span restriction survives. That, more than the cost, is why no training arm is
  queued.
* **The leak test's scope is ONE core block.** It proves "no token reads another span's
  token" on an `n_core = 1` fixture. At `n_core = 2` the token → cell → token route is
  open BY DESIGN and reads 0.589 on the same fixture. That is the intended channel, not a
  leak, but the test cannot say "no cross-span information reaches a token".
* **The slot-loop levers were silently inert on this mode.** The config inherits
  `slot_gain_lambda: 100` and `slot_cot_clip: 4.0` from the slot-loop root, and they act
  only inside `_tul_core`. The `[slot-levers] ... INERT` predicate named the paid loop and
  not this one. Fixed, with a three-way test twin.
* **The mode holds no `W_prefix`** — 2.10 M fewer parameters than the ruler — so any CE
  comparison carries that confound.
* **It has never run a training step.** Built, tested, refused where it does not compose,
  OFF. The ON state has never seen a GPU.
