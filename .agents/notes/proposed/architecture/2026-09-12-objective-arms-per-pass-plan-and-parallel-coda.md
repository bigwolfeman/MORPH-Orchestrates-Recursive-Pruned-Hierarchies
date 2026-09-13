# Agent Note: two objective arms — a per-pass planning target, and parallel span decoding from the coda

Status: proposed

Date: 2026-09-12. Record:
[`lab/experiments/failures/2026-09-12-arc-objective-arms.md`](../../../../lab/experiments/failures/2026-09-12-arc-objective-arms.md).
Runs on the geometry of
[`2026-09-12-strict-slot-geometry.md`](2026-09-12-strict-slot-geometry.md) with coda reach
`all`, and follows
[`2026-09-11-span-decoder-target.md`](2026-09-11-span-decoder-target.md). Supersession
check: neither is superseded. The strict note owns the allow relation and this one changes
nothing about it; the span-decoder note owns the target and this one ADDS targets beside
it rather than replacing the exit term. Nothing archived in this change.

## Problem

The strict geometry did what it was built to do and the result did not follow.

On `slot-spandec-strict` the loop's write is the WHOLE cross-span channel — worth 0.187
nats, with `worth_profile`'s `all_slots` equal to its `zero` by construction — and the
passes still read **K1−K6 0.0016**. Twelve arms since 2026-09-04 now sit in [−0.0001,
+0.0033]: the core's blocks, the stability terms, the entry, the depth draw, the width, the
HCA fix, the seed, the state's geometry, the target, and now the allow relation.

One thing HAS moved the curve. `slot-spandec-strict-prev-reach1` (coda reach `prev`,
`tul.loop_reach` 1) reads tokens **K1−K6 +0.0163**, **K3−K6 +0.0042**, at depth-6 CE parity
with its control. It gets there by making a token blind to everything older than the
previous slot's cells, so the loop's chain of states is the only route left. Wolfe,
2026-09-12: "I am suspect of this method. It proves that TUL can work for sure. But the
amount of blindness here is concerning. z has to hold the history when it should hold the
present next thought that needs decoding. Our objectives are still poor."

That is the problem in one line. **The only lever that made passes matter did it by
amputation, and it made z the wrong thing.** What has never been tried is asking for depth
from the OBJECTIVE with the coda's reach intact.

The second half of the problem is the READER. Every span-decoder arm grades `z` through a
separate two-block `SpanDecoder` the token CE never touches. That reader moved a lot (the
slot channel went 0.115 → 0.182 with the whole-span target) and the loop did not move at
all. Nothing has ever asked the CODA — the reader the model ships — to produce a span.

## Proposal

Two knobs, two arms, one factor each against `slot-spandec-strict`. Both default off and
build nothing when off.

### `tul.spandec_per_pass` — a planning target that grows by a span a pass

The state after pass `t` is decoded into spans `s+1 .. s+min(t, tul.spandec_pass_horizon_max)`
at `tul.spandec_pass_tokens` tokens per span, through the SAME `SpanDecoder` the exit state
is graded by. Pass 1 is asked for the next span, pass 2 for the next two, and so on, so a
pass can only improve on its predecessor's target by planning one span further — the
property no unblinded objective has ever asked for. The EXIT term is untouched:
`tul.spandec_horizon` stays 1, so `z` remains "the present next thought that needs
decoding" and the arm does not quietly turn it into a history buffer.

* A slot is graded at pass `t` only when its REALISED depth reaches `t` (`tul.oracle_z`'s
  mask). A frozen slot's state is its final one; grading it again at every later pass would
  supervise the `torch.where` carry and over-weight shallow slots.
* Reduction: one CE per pass (a mean over that pass's graded tokens), then a plain mean over
  passes. Passes weigh equally.
* Nothing is detached. The term reaches pass `t`'s core application through the trajectory
  and, through the live carry, every earlier pass. The decoder trains on it too — one
  reader, one notion of what a decodable plan is.
* Its own ZERO-INIT position table (`SpanDecoder.pos_pass`). The per-pass sequence is `H_t`
  blocks of `pass_tokens`; the exit sequence is one block of `spandec_max_tokens`. Sharing
  `pos` would make row 8 mean "span s+2, token 0" to one term and "span s+1, token 8" to
  the other.

### `tul.coda_span_heads` — the coda emits the next span in parallel

At each slot's LAST prefix cell (`slot_index + prefix_k − 1`, the emitting position; under
strict that cell carries the looped state and nothing else) the coda's final readout goes
through `J` parallel offset heads — the `_MTPHead` construction, RMSNorm plus a `[d, d]`
linear at IDENTITY init — and head `j` is scored through the tied table against token `j`
of the next span. Non-autoregressive: no teacher forcing, no token path, all `J` offsets
from one state. The token CE is unchanged.

At identity init every head reproduces that position's own next-token head, so step 0 is
the ruler's forward. The tied table is read through `tul.mux_detach_head` (default true),
the rule every auxiliary head here follows. `tul.coda_span_source: "token"` moves the read
to the boundary TOKEN position and is a CONTROL, documented as one: that token sits before
its own slot's cells and the coda is causal, so its state has never seen its own slot's z.

Arms: `tul_slot_spandec_strict_perpass.yaml`, `tul_slot_spandec_strict_codaspan.yaml`
(J 8) and `tul_slot_spandec_strict_codaspan32.yaml` (J 32 = `span_cap`, the whole span).

## Alternatives considered

* **`reach2`, and more reach arms generally.** Amendment 3 already queues `prev-reach2`.
  More of that family buys more of the same reading: depth that exists because information
  was removed. Wolfe's objection is not that reach1 failed — it worked — but that it makes
  z hold history. Kept as the control these arms are read against, not extended.
* **A per-pass target that is the SAME span every pass** (just supervise `h_t` on span
  `s+1` at every `t`). That is `tul.mux_every_pass` with a decoder, and `mux_every_pass`
  already read flat. A target identical at every pass gives pass 6 no reason to differ from
  pass 1, which is the whole defect.
* **Weight the per-pass blocks by distance.** An unmeasured second knob on a target whose
  first question is whether the growth helps at all. Every supervised token counts once, the
  convention the H = 1 and H = 3 terms already have.
* **Token-weight the per-pass mean instead of weighting passes equally.** A pass-6 term
  carries six times pass 1's tokens, so a token-weighted mean is mostly a reading of the
  deepest pass. Rejected for the primary reduction; named in the record as unmeasured.
* **One merged CE call for all passes** (concatenate the flattened readouts, per-row
  `weights` = 1/n_t to keep the pass-equal reduction). It holds ONE `[V, d]` fp32 `grad_w`
  instead of six — about 1.0 GB saved — and it throws away `tul/spandec_pass_t{t}`, the
  per-pass CE. That column is the arm's honesty instrument: if it does not fall with `t`, a
  deeper pass is not making a better plan and no K-curve reading means anything. Kept the
  separate calls, wrote the gigabyte into the config header, and named the OOM as the arm's
  most likely death.
* **`tul.oracle_z` with the per-pass target.** REFUSED at construction. Both write a
  per-pass target onto the same trajectory — the oracle in STATE space, this in TOKEN space
  — and the oracle's teacher is computed FROM the decoder that the per-pass term is
  simultaneously training, so the teacher moves under the student. There is no reading of
  the conjunction, so it raises rather than running.
* **Let the coda heads run autoregressively** (a small decoder on the coda state, the
  `SpanDecoder` construction again). That is the arm we already have, moved one module
  over, and it re-introduces the teacher-forced token path — which is precisely the crutch
  Wolfe's question removes. "All the span at once" is the question; a token path would
  answer a different one.
* **Put the heads on every TOKEN position** (a slot-agnostic parallel span target, the E8
  MTP heads at span scale). It does not touch the loop's write at all: no token position's
  coda state reads a prefix cell of its own slot. The whole point of reading at the cell is
  that under strict the cell's only content is what the loop wrote.
* **A private `[V, d]` output projection for the heads instead of the tied table.** 50 M
  parameters at V 49169, d 1024, and it would grade the cell through a reader the model
  never uses. The tied head under `mux_detach_head` is the tree's rule for auxiliaries.
* **Give the per-pass term the decoder's own 32 tokens per span.** Σ_{t=1..6} t·32 = 672
  positions per slot per step, 84 decoder block-passes per token. The arm would not reach
  step 200 inside the rate floor. 8 is where the measured cross-span budget stops being
  front-loaded and is the `oracle_z_max_tokens` / `egrad_max_tokens` precedent.

## Acceptance criteria

1. **Off builds nothing and changes nothing.** `spandec_per_pass: false` builds no position
   table and never collects the trajectory; `coda_span_heads: 0` builds no module. Held by
   `tests/test_tul_spandec_per_pass.py` and `tests/test_tul_coda_span.py`.
2. **On is PURELY ADDITIVE and RNG-NEUTRAL.** Every shared parameter is byte-identical to an
   off-model from the same seed — the coda heads restore the global RNG state around their
   construction, because `nn.Linear.reset_parameters` draws before the identity init
   overwrites it — and `loss − <term>_weighted` equals the off-model's loss under
   `torch.equal`. Held, both arms.
3. **The targets are the right tokens**, checked against a Python oracle that walks
   `bag_id` itself: pass `t` block `h` holds span `s+h`'s tokens; coda head `j` at slot `k`
   holds span `k+1`'s token `j`. Held.
4. **The horizon at pass `t` is `min(t, cap)`** and **a slot of realised depth `d` is graded
   at exactly passes 1..d**, against a hand-built depth tensor. Held.
5. **The gradient reaches the core through pass 1 AND pass 3 individually** (every other
   pass detached), and the coda heads' term reaches both a coda parameter and
   `tul.W_prefix`. Held.
6. **The heads read the LAST prefix cell**, measured as the gradient with respect to the
   coda readout being non-zero at exactly those positions. Held.
7. **Six source-level sabotages, six catches** (wrong span set; no depth mask; detached
   trajectory; shifted head offsets; read the own span; no validity mask), each with its
   anchor asserted unique before the edit. Held, 2026-09-12.
8. **Every new `tul.*` key is in `KNOWN_TUL_KEYS`, in the wandb manifest and on a build
   banner**, and the validation refuses the combinations that have no reading. Held.
9. **Every config composes through Hydra, builds and RUNS at the PANEL's budgets** —
   `spandec_max_tokens` 0 → 32 beside per-pass tokens 8 (the J 8 vs 32 mismatch that killed
   the strict-oracle smoke), cap 6, J 8 and J 32. Held by
   `tests/test_tul_objective_arms.py`.
10. **The arms survive 5,000 steps and clear the rate floor.** NOT MET — no GPU step of
    either arm exists. That is what the panel is for.

## Risks

* **The per-pass arm may not fit.** Six live chunked-CE calls hold six `[V, d]` fp32
  `grad_w` accumulators, 1.21 GB, of which 1.01 GB is new. The 12-step smoke is where this
  is decided, and an OOM there is the expected failure mode.
* **The per-pass arm is the most expensive of the family**: 35.7 block-passes per real token
  against strict's 14.7, h3's 22.7 and the plain panel's 44.0. It may be skipped by the rate
  rule, by design.
* **The per-pass target may be met by the DECODER, not the state.** The decoder trains on
  every block; it can get better at the near tokens of each span without the loop's state
  changing. `tul/spandec_pass_t{t}` falling is necessary and not sufficient, and the K-curve
  is the arbiter.
* **The coda heads change what the CODA must do, not what a PASS must do.** The route from
  there to a per-pass contribution is indirect, and indirect pressure is what the eleven
  flat arms already applied. The outcome this arm is really for is the WRITE
  (`slot_z_optimize`'s `ce_entry − ce_loop`), not the K-curve.
* **Two auxiliary terms now compete for the coda's gradient** (the token CE and the heads)
  at an untuned weight of 1.0. Neither `spandec_pass_weight` nor `coda_span_weight` has been
  swept.
* **J 32's far heads are thin.** The packed mean span is ~20 tokens, so heads 21..32 are
  supervised on a minority of slots; `tul/coda_span_n_tokens` must be read before comparing
  J 32 with J 8.
* **`coda_span_source: "token"` has never been run.** Its causality consequence is reasoned
  from the coda's causal mask and pinned only as a read POSITION.
