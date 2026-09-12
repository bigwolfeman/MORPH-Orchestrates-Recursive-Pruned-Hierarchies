# Agent Note: the strict slot geometry — the loop as the only cross-span channel

Status: proposed

Date: 2026-09-12. Record:
[`lab/experiments/planned/2026-09-12-arc-strict-geometry.md`](../../../../lab/experiments/planned/2026-09-12-arc-strict-geometry.md).
Follows [`2026-09-11-span-decoder-target.md`](2026-09-11-span-decoder-target.md) and
[`2026-09-11-cross-span-budget.md`](2026-09-11-cross-span-budget.md); the latent-z arms of
[`2026-09-12-latent-z-gradient-loop.md`](2026-09-12-latent-z-gradient-loop.md) are re-pointed
onto it. Supersession check: none of those three is superseded — this note narrows the
GEOMETRY they all run on and none of them makes a claim about the allow relation that this
one contradicts. Nothing archived in this change.

## Problem

Eleven slot-loop arms since 2026-09-04 read a per-pass contribution of nothing: token
K1−K6 in [−0.0001, +0.0033], and on `slot-spandec-mask` specifically **0.0007** at 5,000
steps over 480 rows. Every lever tried — the core's blocks, the stability terms, the entry,
the depth draw, the width, the HCA fix, the seed, the geometry of the state — moved it by
less than 0.003.

The 2026-09-11 route split says why, and it is not subtle. `tul.tg_restrict`'s allow relation
is `causal AND (same span OR j is ANY slot cell)`, applied in the PRELUDE as well as the
coda. So:

* in the prelude, every token and every slot cell may attend every EARLIER slot cell;
* in the coda, a cell may attend other cells AND re-read its own span's token states;
* at every coda layer, `x0` and the bigram term re-add the cell's seed — `E_slot` plus a
  bag-mean of its own span's token embeddings — to the cell's carrier.

Measured, `slot-spandec-mask` at 5,000 on 480 rows: the whole slot channel (`all_slots`) is
worth **0.182** nats and the loop's own prefix write (`zero`) **0.078**. More than half of
what the slot channel delivers has never passed through an iteration of the loop. The loop
was being asked to justify itself on a forward where it is optional.

The information at stake is measured too: with no slot cells at all, cutting every cross-span
route costs **0.3994** nats [0.3838, 0.4162], of which **0.3149** [0.3025, 0.3314] is FLAT at
offsets eight or more tokens into a span.

## Proposal

`tul.tg_geometry: "strict"` (default `"restrict"`, bit-identical to every arm to date) makes
the slot loop the only route from one span to the next.

| stage | allow relation |
| --- | --- |
| prelude | `causal AND bag_id[i] == bag_id[j]`. A token sees its own span's tokens; a cell sees its own span's tokens and its own earlier cells. The seed is a pure summary of ONE span. |
| loop | unchanged. Slots attend earlier slots' states, causally. |
| coda | a TOKEN sees its own span plus the PREFIX CELLS of earlier slots; a PREFIX CELL sees ITSELF alone, so it carries the looped state and nothing else. |

`tul.tg_coda_prefix_reach` picks which cells a coda token may read: `"all"` (every earlier
slot's cells — the primary arm) or `"prev"` (only the cells of the slot terminating the
previous span, so everything older must flow through the chain of loop states; a tail
dump-bin token then reads no cell at all, the same conservative gating `tg_soft_prev_span`
already takes).

Three more cuts are PART of the geometry, not separate knobs, because a relation the next
operator undoes is not a relation:

1. the CCA causal conv and its `W_v_prev` value shift reset at every segment
   (`tg_segment_ids` = a span's tokens | that span's cells | the next span's tokens);
2. the retention carry resets on the same partition rather than on `bag_id`, which does not
   separate a span from its own cells (retention is off on every arm here — the reset is
   asserted, not assumed);
3. the coda's per-layer injections at the slot CELLS are zeroed. Load-bearing: without it a
   cell hands its span's token identities to every later span with no pass in between. A side
   effect worth stating is that `worth_profile`'s `all_slots` then EQUALS `zero` on a strict
   arm, which becomes a check on the geometry rather than a measurement.

The hash bigram needs no cut on a TUL row. A span's first token is always preceded by the
previous slot's prefix cells, whose input id is the constant `slot_id`, so the bigram carries
no information about the previous span. This is proved two-sided in the gate rather than
asserted.

Two levers ride on the same forward, each its own knob and its own arm:

* **`tul.loop_reach: w`** — inside the loop a slot attends slots `k-w .. k`, so a slot `m`
  spans back first reaches slot `k` at pass `ceil(m/w)` and long-range context REQUIRES depth
  by construction. The budget is PER PASS: it is spent in core layer 0 and closed in layers
  1..n-1, and the conv resets per cell in every layer. Spending `w` at every layer carries
  `w · n_core` per pass, which was measured on the CPU fixture and is why the first
  implementation was wrong. The price, stated: a reach arm's core mixes cells once per pass
  and is position-local for the rest of it.
* **`tul.oracle_z`** — a detached per-pass descent trajectory of the span decoder's next-span
  loss, with `‖h_t − z*_t‖²/d` added to the loss. **This breaks the standing rule against
  regressing onto the slot state** (LCM T3/T4, CoCoMix §6b, BT §4.2) on purpose, as a test of
  whether a per-pass target that a one-step optimiser could match produces a per-pass
  K-curve. It is a diagnosis, not a design, and Wolfe decides whether it ever ships.

And one target change, because the missing information is long-range: **`tul.spandec_horizon: H`**
grades z on spans `s+1 .. s+H` as one causal run. A target that ends at the next boundary
cannot ask z for anything past it, and the budget's missing part is flat at offsets 8+.

## Alternatives considered

* **Keep the mask and widen the write (`prefix_k` 2 → 4).** Already run —
  `slot-mux-prefix4-norm-match`, 2026-09-10 — and flat. It adds capacity to a route that was
  not the constraint; the constraint is that a cheaper route existed at all.
* **`tg_restrict_scope: "coda"`.** Restricts the coda and leaves the PRELUDE global, which is
  the larger half of the bypass: the prelude is where every cell reads every earlier cell.
  Kept as an existing arm, refused under strict.
* **`tg_soft_prev_span`.** Opens the previous span's TOKENS to a query — a second cross-span
  channel beside the loop, which is the opposite of the change. Refused under strict;
  `tg_coda_prefix_reach: "prev"` is the version of "one span back" that goes through the
  CELLS and therefore through the loop.
* **Cut the rows instead of masking** (train on one span per row). Removes the slot's reason
  to exist along with the bypass, and the 2026-09-11 budget arms already measured that
  no-slot world — its number is the 0.3994-nat budget this geometry is trying to route.
* **Drive the geometry from `model.span_mask`.** That machinery builds a Parcae core, refuses
  a slot layout and refuses `tg_restrict`; it is the NO-SLOT instrument and reusing it would
  mean re-deriving every slot relation inside it.
* **Budget the loop's reach per BLOCK instead of per pass.** Simpler to write and it is what
  the first implementation did. It makes the arm's central claim false — the pass count to
  reach `m` slots back becomes `ceil(m / (w · n_core))` — and the CPU fixture caught it
  (reach 2 over two core layers moved a slot four cells away at pass 1).
* **Budget the core's CCA conv instead of cutting it.** A kernel-4 two-stage conv reaches six
  cells back per block. No window relation expresses that, so a budgeted version would be a
  different relation per operator. Cut it and say so.
* **Let the oracle train the span decoder** (a joint objective). The decoder would then be
  shaped by what the loop can reach rather than by what the next span needs, and the loop
  would be graded against a target it helped choose. The trajectory is fully detached instead.
* **Give the oracle the decoder's own J = 32.** Arithmetic refuses it: one `[B, S, J, V]`
  readout is 38.7 GFLOP per token of J at the panel shape, so T 6 × J 32 is ~45 % of a step
  and the arm misses the queue's rate floor. `oracle_z_max_tokens: 8` — the `egrad_max_tokens`
  precedent, and the offset where the budget stops being front-loaded.
* **Weight the H spans of the downstream target by distance.** An unmeasured second knob on a
  target whose first question is whether distance matters at all. Every supervised token
  counts once; if the far spans turn out to need their own weight, that is the next arm.

## Acceptance criteria

1. **`tg_geometry: "restrict"` is bit-identical.** No new parameter, no RNG draw, the same
   state-dict keys, and a `restrict` model's logits equal a model built without the key.
   Held by `tests/test_tul_strict_geometry.py`.
2. **The leak test is EXACTLY zero and two-sided.** With the loop's prefix write zeroed, a
   token id in span 0 moves nothing outside span 0 (bit-exact, CPU fp32), the `restrict`
   model through the same probe DOES leak, the write-ON strict model DOES move a later span,
   and the edit moves its own span. Held.
3. **Five sabotages, five catches.** Prelude window widened, prelude compressed branch
   unmasked, coda cell query widened, conv/value-shift reset dropped, coda cell injections
   restored. Each is self-verifying: with no patch the probe reads exactly 0.0. Held.
4. **The reach law is exact.** A slot `m` back moves first at pass `ceil(m/w)` and is
   bit-exact zero before, at (w 1, m 2), (w 1, m 3) and (w 2, m 3), with an unlimited-reach
   control and three more sabotages on the core's routes. Held.
5. **The oracle does not train the decoder.** Every `tul_spandec.*` gradient is bit-identical
   with and without the term while core gradients differ, and `autograd.grad(term, decoder)`
   is None on every decoder parameter. Held by `tests/test_tul_oracle_z.py`.
6. **`spandec_horizon: 1` is bit-identical and H = 3 supervises exactly three spans**, pinned
   against a hand-built layout index set for index set. Held by
   `tests/test_tul_spandec_horizon.py`.
7. **Every arm's config composes through Hydra, builds and RUNS.** Not "a unit test passed":
   each new config is composed, mapped through `build_tul_runtime`, built into a model and
   forwarded. Held.
8. **Every new `tul.*` key is in `KNOWN_TUL_KEYS` and in the wandb manifest.** Held.
9. **The arms survive 5,000 steps and clear the rate floor.** NOT MET — no GPU step of any
   arm exists. This is the criterion the panel is for.

## Risks

* **Strict will probably cost CE at 5,000**, because ~0.10 nats that flowed through the cells
  must now flow through the loop or not at all. That is the price of the question, and the
  reading is the loop contribution, not the final CE (the standing short-horizon rule).
* **A prelude that sees only its own span is a large change to the seed.** Nothing says a
  strict model trains as well as a masked one; the gate proves leak-freedom at init on a tiny
  CPU model and says nothing about 5,000 steps.
* **`reach1` changes what the core's later layers ARE.** Five of six become position-local.
  It is inseparable from the reach limit, and it is a capacity cut nothing here prices.
* **`oracle_z` is a rule break.** The most likely outcome is that the loop matches the
  trajectory, `oracle_z` falls to near zero, and the token CE does not move — which says the
  passes can be driven anywhere and that where is not what the coda needs. Read
  `tul/oracle_z_l{t}` first: if the oracle's own readout does not fall, the target is noise.
* **`h3` may miss the rate floor** (14.7 → 22.7 block-passes per token) and be skipped by
  design.
* **`prev` starves the row's tail.** Dump-bin tokens read no prefix cell at all, and how many
  tokens that is on the real loader is unmeasured.
* **The latent-z arms' frozen predictions now sit against a different control.** Recorded as
  amendment 1 in their prereg and NOT repaired; that is the cost of moving a control after
  freezing.
