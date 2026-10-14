# Agent Note: FLOP diet keys, span decoder slot keep and gain hinge cadence

Status: proposed

## Problem

The flame4 ledger of the speed base puts the span decoder at 2.775 TFLOP of 14.244 per step
(23.6 ms) and the gain hinge at 1.069 TFLOP (9.5 ms). Both are training-only terms. Cutting
them changes the objective, so the cut needs two keys the 5k check can judge, both off by default.

## Proposal

`tul.spandec_slot_keep` (float, default 1.0). The packed span decoder trains on a random subset
of each row's slots per training step. Each kept slot keeps its full next-span target. The
subset is a per-row token budget, `round(keep x graded tokens)`: slots go in a random order and
the longest prefix that fits the budget is kept. The packed capacity is the row cap times keep
(integer arithmetic at 1/1024 resolution), so the decoder blocks and the vocab CE run on that
many rows. The loss is the mean over kept tokens. Eval grades every slot. The order is an
integer hash of the run seed, a device step counter, the row and the slot: no RNG stream
moves, and a CUDA-graph replay draws a new subset.

`model.slot_gain_every` (int, default 1). The one-pass hinge runs on every k-th training step
with its lambdas (hinge, tail, floor) times k. Off steps run no hinge forward or backward. In
the eager forward the cadence is a host count. Under `training.graph_step` the off step is one
more variant (`GAIN_OFF`, the last entry of `gain_pass_choices`), a recording with no hinge in
it, so graph count is cells x (passes + 1) x kinds.

## Alternatives considered

- A fixed slot COUNT per row (`round(keep x real slots)`). Rejected after working the bound:
  the packed capacity must be proven from shapes, and with keep 0.5 the worst case (the kept
  slots are the longest spans) is `max_n min(L - K n, J x round(0.5 (n-1)))`, 1024 against
  1132 at the lxtul shapes. The decoder would shrink 10 %, not 50 %.
- Capping tokens per span (`tul.spandec_max_tokens: 8`). Rejected earlier, CE@6 +0.026.
- A conditional graph node for the cadence (one recording, a device-side step counter feeds the
  condition). It skips the hinge backward only; the forward stays in the recording. A variant
  skips both, at the price of more recordings.
- `model.slot_gain_rows` (existing) cuts hinge rows, not steps. It stays independent.

## Acceptance criteria

Keys off: byte gate FINAL_SHA256 `4a73ebf5...` (met). Tests `tests/test_diet_keys.py`. The 5k
judgement of CE@6 and K1-K6 is the orchestrator's.

## Risks

- A random subset of whole spans weights long spans less than a uniform slot sample does (a long
  span fits the budget less often). Mean over kept tokens keeps the token weighting inside the
  subset.
- The cadence count and the subset counter restart at a resume.
- Two keys on one run change two terms at once; read them apart at 5k.
- Each cadence variant adds hinge-free recordings to the graph pool (memory not measured here
  beyond the 45-step gate).
