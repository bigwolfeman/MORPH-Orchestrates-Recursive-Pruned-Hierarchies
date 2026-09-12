# Agent Note: the span decoder — what the slot is asked for

Status: proposed

Date: 2026-09-11. Prereg:
[`lab/experiments/planned/2026-09-11-arc-span-decoder.md`](../../../../lab/experiments/planned/2026-09-11-arc-span-decoder.md).
Sibling: [`2026-09-10-credit-assignment-in-the-slot-loop.md`](2026-09-10-credit-assignment-in-the-slot-loop.md)
(how the slot loop is trained). This note is about what it is trained TOWARD.

## Problem

Eleven arms since 2026-09-04 have changed the slot loop's core, its entry, its stability
terms, its width, its readout, its geometry, its per-pass credit and its ternary rule.
Every one reads flat: token K1−K6 inside [−0.0001, +0.0033], the exit state equal to its
entry to 0.0015 nats, the coda's CE with the looped state equal to its CE with the state
the loop started from.

One thing has never changed: the slot's TARGET. Since arm v1a the slot has been graded by
the MUX local head — `z` read once through the tied head, scored against the geometric
superposition of the next span's tokens (`morph/model/tul.py::mux_span_targets`, rho 0.9).
That target is **order-free**. It is one categorical distribution per slot, so its
minimiser is the span's weighted unigram marginal, and a 1024-dimensional state that has
already reached that marginal has nothing left to gain from a second pass. Wolfe,
2026-09-11: "M-next asks for the next span's first token through the tied head; this is
probably 99 % of our problem. We need the whole span to decode from z, and we are
essentially getting like 2 tokens out of it."

The [cross-span budget](../../../../lab/experiments/failures/2026-09-11-arc-span-budget.md)
measured, the same day, what there is to carry and what SHAPE it has. Two plain models
differing only in whether information may cross a span boundary sit **0.3994 nats**
[0.3838, 0.4162] apart at 5k, and the gap is **not front-loaded**: 0.315 at every offset
eight or more tokens into a span, 0.958 at the span's first position, decaying over about
seven tokens. The slot channel's prefix write is worth 0.093 of that, with the decaying
shape. **The worth profile's front-loaded decay is the shape of what the write CARRIES,
not the shape of what is there to carry.**

A marginal can buy the spike. Nothing about a marginal can buy a flat long-range
component.

## Proposal

Five one-factor arms on the mask ruler `slot-mux-mask-norm-match`, plus one instrument and
one matched-compute control. The mechanism is `tul.spandec`
(`morph/model/tul_spandec.py`).

**The target.** For each slot, run a small teacher-forced causal decoder over
`[z, t_0 .. t_{J-2}]` predicting `t_0 .. t_{J-1}`, the tokens of the NEXT span
(`J = tul.span_cap`, 32). The loss is the mean of `-log p(t_j | z, t_{<j})` over the span's
tokens, weighted by `tul.spandec_weight`, pad slots and span 0 at `ignore_index`. The
gradient reaches `z` from every token of the span instead of once per span, and the
teacher-forced prefix is the token path the spec demands — never decode a span from one
vector plus an offset with no token path (Huginn 2026-08-16, MegaByte T7, Bowman T2,
Hourglass T6).

**Where `z` is read.** `_readout(h_slots)` BEFORE `prefix_project`: the same state the MUX
head, `slot_z_optimize.py` and the per-pass gradient probe read, so every instrument in
the campaign still points at the same object and the prefix-write worth stays the right
companion reading.

**The tied head.** The output head is `embed.lm_weight()` through
`fused_linear_cross_entropy` — the full readout would be `[B, S, J, V]`, 2.4 GB fp32 at
the panel shape. It obeys `tul.mux_detach_head`, the knob whose whole subject is "may an
auxiliary head train the tied table". The spandec configs set it TRUE where the ruler
lineage (`tul_gl1b`) sets it false; that is not a second factor on those arms, because
`mux_beta: 0` leaves the knob governing the decoder alone. The decoder's INPUT embedding
read is detached UNCONDITIONALLY: the MUX has no input-side read of the table, so there is
no precedent to inherit, and an undetached one would let the decoder reshape the table
that the slot's own seed (`E_slot` plus a bag-mean OF that table) is built from — the
feedback loop that diverged arm v1a at step 2800.

**The slot chain** (`tul.slot_chain`, arm `slot-spandec-chain-mask`). At every pass, slot
`k` receives `W_chain(h_t[k-1])`, zero-init, causal, not detached. It is the only
mechanism in the batch that can hold a running document state, which is what the budget's
flat component is.

*Named deviation.* The literal form — slot `k`'s seed gets slot `k-1`'s EXIT state — needs
slot `k-1`'s loop to finish before slot `k`'s starts, i.e. 64 sequential loops over the
compact sequence instead of the one masked update `runtime-invariants` §6b requires (a
frozen slot must keep serving the same K/V, which is why the slots loop together). What is
built is the WAVEFRONT form of the same recurrence: the injected state IS the previous
slot's exit for every neighbour whose realised depth is already spent, and its live state
otherwise. Recurrence depth is the pass count (≤ 8), not the 64 slots — which is why
`slot_chain_detach` defaults false: there is no deep-recurrence argument for detaching, and
detaching would make the chain a feature rather than a path an earlier span's loss can
shape.

**The instrument, `plan_mode="all_slots"`.** The budget result's own next step. `zero`
ablates ONE route, the prefix write. `all_slots` also zeroes the coda's per-layer
injections at the slot cells (`x0` carries the slot seed, the bigram term carries the span
bag-mean) and rebuilds `tg_allow` with `slot_queries_slots_only=True`, so a cell cannot
re-summarise its own span from the coda's token states. The three cuts together make the
slot channel carry nothing. It refuses on anything but a `tg_restrict` scope-"all"
slot-loop arm with a full-`L` coda.

*One route it does not cut, so every reading is a LOWER bound:* the CCA causal conv and its
`W_v_prev` value shift still read the positions before a slot cell. Cutting them needs a
`tg_seg` segment reset that also fires between one span's tokens and the next, which would
change the TOKEN positions' conv and put an operator change inside a paired CE difference.

`worth_profile.py --paired-rows` packs with the depth sweep's own
`lab/divergence/_rows.py::pack_rows` and records the stream index of every scored token, so
the budget curve, the prefix-write worth and the all-slot worth sit on the same tokens;
`--verify-tok-index` asserts that rather than assuming it. Measured 2026-09-11: the paired
and unpaired paths give bin-for-bin identical profiles on the same 6 rows, and the paired
`tok_index` is a prefix of the arm's own sweep npz (6,288 of 501,106 positions).

**Cost, measured from `_tul_layer_passes` on a real packed batch and confirmed by the
trainer's `lp/tok` line:** the ruler is 10.745 block-passes per token, the span-decoder arm
14.676, the decoder alone 3.931, a plain model at depth 6 is 44.0 and at depth 1 is 14.0.
`plain_coda_matched.yaml` (the panel ruler at depth 1) therefore matches the ARM's total to
4.6 %. A plain model matched to the decoder's 3.931 alone is not expressible: the plain
per-token count is `8 + 6d`, whose floor is 14.0.

## Alternatives considered

- **Widen the write instead of changing the target** (`prefix_k` 2 → 4). Already run:
  `slot-mux-prefix4-norm-match`, 2026-09-10, flat. The channel was not the binding
  constraint at the measured worth.
- **Every-pass span decoding** (a decoder term after each loop iteration). Rejected as the
  FIRST arm: `mux_every_pass` already ran that shape with the bag target and read flat, and
  the toy study says one instruction repeated six times is the signature of a loop that
  does not separate jobs. It is the follow-up if Test 1 moves anything.
- **A decoder over the OWN span** (`mux_target: "own"`'s conditional twin). Rejected: the
  own span is reachable in about one pass (`slot-mnext-gradpass` measured the loop
  descending it in one step), so the target would be satisfied before the loop begins. The
  staged form — own span at the intermediate passes, next span at the exit — is the natural
  second arm and is one knob away (`mux_stage_own_iters` beside `spandec`), deliberately not
  drawn here because it changes two things.
- **Regressing `z` onto a sentence encoder's embedding of the next span** (the LCM /
  CoCoMix shape). Refused by the root `CLAUDE.md`: never regress onto the slot state. A
  regression target has the same order-free defect as the bag plus a second model's biases.
- **Making the DECODER the deployed reader** (decode each span from its slot at inference,
  the literal "think once, decode cheaply"). Not proposed yet, and deliberately: the
  decoder here is a training-time SCORER whose only job is to say what `z` carries. Turning
  it into the generation path is a different change with a different contract (a KV cache,
  a halting rule, and the emit-source question), and it should be decided by what the
  scorer measures rather than before it.
- **A private `[V, d]` input embedding for the decoder.** Rejected on size: 37.7 M
  parameters at V=49169, d=768, and it would drift from the table the rest of the model
  uses. The detached tied table plus a learnable `d × d` map (`SpanDecoder.tok_in`) buys
  the same freedom for 1.05 M.
- **Ternarising the decoder.** Rejected: it is a training-only auxiliary and grading `z`
  through a quantised reader would mix the reader's precision into the target the loop is
  judged on. Every `nn.Linear` in it carries `_ternary_exclude` (the `ParcaeCoreBlock`
  precedent), and none is a `MortarLinear`, so the prune, the carve, the router and the
  deploy packer walk past by construction.
- **A post-loop causal SCAN over the exits** (`z'_k = z_k + W(z'_{k-1})`) instead of the
  wavefront chain. It reaches all 64 slots rather than ~6 and costs almost nothing — but it
  bypasses the loop entirely, which is the opposite of what a credit-assignment arm should
  do. Kept on the table as the cheap fallback if the wavefront's reach turns out to be the
  binding limit.
- **Doing nothing here and changing the corpus instead** (the code pair the budget result
  left unbuilt: the boundary rule cuts code twice as often). Still the right move if all
  five arms read flat, and the binding section of the prereg says so. It is not first
  because the budget on web text is 0.40 nats and rising, which is enough to measure a
  mechanism against.

## Acceptance criteria

Per-arm predictions are frozen in the prereg and are the binding record. At the level of
this note:

- `tul.spandec: false` and `tul.slot_chain: false` are the tree from before this change:
  no module built, no RNG drawn, the plain path (`slot_layout=None`) bit-identical with the
  decoder BUILT. Held by `tests/test_tul_spandec.py`.
- The mechanism is proven to ACT by tests that fail when it is removed. Done: 4 of 4
  sabotages caught on the decoder (span offsets shifted by one, `z` detached inside the
  loss, the chain read anti-causally, the term multiplied out of the loss) and 3 of 3 on
  `all_slots` (drop the injection cut, drop the attention cut, fall through to plain
  `zero`).
- The gradient reaches `z` from every token of the span, by autograd AND by a finite
  difference along that gradient. Held.
- Pad slots, span 0 and the dump bin carry no loss, checked two-sided: perturbing every id
  the decoder must not read leaves the term unmoved, and perturbing one it must read moves
  it.
- The chain is causal: `d term[k] / d h[j]` is nonzero only at `j = k-1`, checked at every
  slot index.
- Every new `tul.*` key is in `KNOWN_TUL_KEYS` and in the wandb manifest, and a
  misspelling raises.
- Full suite at this change: `pytest tests/ -q` → **1133 passed, 9 skipped, 1 xfailed**.
- Each new config builds and trains on the GPU: six 21-step smokes, all exit 0, zero NaN,
  peaks 9.04-14.52 GB, rates 10,829-20,318 tok/s.

## Risks

- **The overhead pays for the finding.** `slot-mnext-staged-20k` measured that a slot-side
  overhead costs the CE and the gap stays flat from 10k. This arm adds 3.93 block-passes per
  token, 37 % more than the ruler, and 27.83 M parameters. A CE that moves the wrong way is
  the MODAL outcome and must not be read as the target failing; the K-curve, the worth
  profile and `ce_entry − ce_loop` are what speak to the mechanism.
- **A new gradient on the tied table.** The decoder charges `J` conditionals per slot where
  the MUX charged one distribution. At fresh init on the real shapes the term delivers
  `|dL/dz|` = 2.18e-3 against the token CE's 7.61e-4 and the MUX's 4.70e-3 — the right
  order, but no arm on this tree has trained with it. `mux_detach_head: true` on these arms
  keeps it off the embedding table; `slot-spandec-mnext-mask` carries the same setting, so
  its clean partner is `slot-spandec-mask`, NOT the ruler.
- **The decoder solves the task without `z`.** A 2-block causal decoder over a span's own
  prefix is a competent short-context LM on its own, and `p(t_j | t_{<j})` alone already
  covers most of a span past offset 4. If it does, `spandec_ce` falls while `z` carries
  nothing new. The instrument for that is the `all_slots` worth profile, not the loss:
  the loss can improve with `z` ignored.
- **The wavefront chain is not the chain that was asked for.** Its direct path reaches ~6
  slots back per forward, and the core's attention over the compact sequence already lets
  slot `k` read slot `k-1`. If it reads flat, that is evidence about the wavefront, not
  about a true exit chain, and the post-loop scan in the alternatives is the next test.
- **`all_slots` is a lower bound.** The CCA conv and value-shift route at a slot cell is
  uncut and unmeasured, so "what the mask arm fails to route" computed from it is an upper
  bound.
- **Memory.** The decoder's `fused_linear_cross_entropy` call always accumulates a `[V, d]`
  fp32 `grad_w` (201 MB at V=49169, d=1024) and saves it for the backward, and with a
  DETACHED head that accumulator is computed and thrown away. It is the price of one
  chunked-CE implementation in the tree rather than two. The smoke's peak is 14.35 GB
  against the ruler's 12.83 GB; 5,000 steps with the full AdEMAMix moment set warm is
  unmeasured.

## Outcome (2026-09-12)

Six arms run and filed in
[`../../../../lab/experiments/successes/2026-09-11-arc-span-decoder.md`](../../../../lab/experiments/successes/2026-09-11-arc-span-decoder.md)
(8 of 10 predictions held). The whole-span target moves the slot channel (all-slot worth
0.115 → 0.182, CE −0.072 vs the mask ruler, flat across offsets); M-next on top costs 0.041;
the chain moves only the long-range prefix write (+0.008 at offsets 8-15) at a 0.007 CE cost
with per-pass cancellation 0.74 → 0.31; token-state dropout and the full-weight MUX were CE
taxes worth 0.043 and 0.046. No arm moved the loop (K1−K6 ≤ 0.0013; write contribution
≤ +0.006). At matched block-passes per token the family is 0.25-0.33 nats behind a depth-1
plain model. The gain rides on the slot cells the coda reads directly, bypassing the loop;
the next batch closes that route
([`2026-09-12-strict-slot-geometry.md`](2026-09-12-strict-slot-geometry.md)). The
`tul.spandec` target stays as the mask family's default target; `tul.slot_chain` stays
available, not default.
