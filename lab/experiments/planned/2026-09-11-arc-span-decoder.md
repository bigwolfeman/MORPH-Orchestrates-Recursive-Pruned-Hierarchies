# Planned: the span decoder — grade the slot on the WHOLE next span, not on its bag

Status: planned

Date: 2026-09-11 (frozen before launch; no arm has run past a 21-step smoke). Arc:
`2026-09-04-loop-contribution-arc.md`. Design note:
[`2026-09-11-span-decoder-target.md`](../../../.agents/notes/proposed/architecture/2026-09-11-span-decoder-target.md).

Evidence it stands on:
[the cross-span budget](../failures/2026-09-11-arc-span-budget.md) (0.3994 nats, flat 0.315
at offset 8+, 0.958 at a span's first position),
[`slot-mux-mask-norm-match`](../failures/2026-09-10-arc-slot-mux-mask-norm-match.md) (the
ruler: tokens K1−K6 +0.0009, worth-zero 0.554 at offset 0 and 0.042 at 16+),
[`slot-loop-mask-norm-match`](../failures/2026-09-10-arc-slot-loop-mask-norm-match.md) (the
no-MUX twin: the core got 0.5 % of the prelude's gradient norm and the exit equalled the
entry), and
[the credit-assignment note](../../../.agents/notes/proposed/architecture/2026-09-10-credit-assignment-in-the-slot-loop.md).

## Question

Nine arms have changed how the slot loop is TRAINED and one measurement has said how much
there is to carry. Nothing has changed what the slot is ASKED FOR.

The M-next MUX reads the exit state `z` once through the tied head and scores it against
the geometric superposition of the next span's tokens (`mux_span_targets`, rho 0.9). That
target is **order-free**: one categorical distribution per slot, so the minimiser is the
span's weighted unigram marginal and nothing beyond it. Wolfe, 2026-09-11: "M-next asks
for the next span's first token through the tied head; this is probably 99 % of our
problem. We need the whole span to decode from z, and we are essentially getting like 2
tokens out of it."

The budget says the shape of what is missing. The 0.399 nats a span needs from across its
boundary are **not** front-loaded: 0.315 of them sit at EVERY offset eight or more tokens
into the span, and only the 0.958-nat spike at the first position has the decaying shape
the prefix write already carries. A marginal can buy a spike. It cannot buy a flat
long-range component.

**Does grading `z` by a teacher-forced decoder of the whole next span — one conditional
per token instead of one bag per span — change what the slot carries, measured against the
budget's own curve?**

## Hypothesis

The slot's flat K-curves are the correct answer to the question the MUX asked. Replace the
question and the answer changes: the worth profile's decay flattens toward the budget's
shape, and the mask arm's CE gap to the plain model closes partly.

The competing hypothesis, and it is not a straw man: the slot state is a 1024-dimensional
vector read through a stream mean, and 0.315 nats at every offset is a running DOCUMENT
state, not a span summary. A better target on the same one-shot write may move the
readout and nothing else — which is what every reader-side arm on this tree has done
(`slot-mnext-staged-fullread` moved the exit 0.016 and the reader nothing;
`slot-mnext-gradpass` moved the exit 0.017 and tokens +0.0014). Test 3 exists because of
this hypothesis: the chain is the only arm that gives the slot a place to keep a running
state.

## Method

Five arms, all one-factor, all 5,000 steps on the panel recipe (seq 1024, batch 6, seed 1,
ramp 1,000, `norm_match` ternary, `core_fixed_point_lambda` 1.0, `ademamix_t_beta3` 3500,
prune / carve / route off, retention off, cap 0), all composing
`morph/configs/tul_slot_mux_mask_norm_match.yaml` — the mask ruler, `tg_restrict` at scope
"all", `tg_scoped_kernels` on.

| arm | config | one factor against |
|---|---|---|
| `slot-spandec-mask` | `tul_slot_spandec_mask.yaml` | the ruler — the slot's TARGET |
| `slot-spandec-mnext-mask` | `tul_slot_spandec_mnext_mask.yaml` | `slot-spandec-mask` — the bag target back on |
| `slot-spandec-chain-mask` | `tul_slot_spandec_chain_mask.yaml` | `slot-spandec-mask` — the slot chain |
| `slot-mask-dropout-off` | `tul_slot_mask_dropout_off.yaml` | the ruler — `token_state_dropout` |
| `slot-mask-mux-quarter` | `tul_slot_mask_mux_quarter.yaml` | the ruler — `mux_beta` 1.0 → 0.25 |

Plus `plain-coda-matched` (`plain_coda_matched.yaml`), the panel ruler at core depth 1: a
matched-COMPUTE plain control, see the cost table below.

### The mechanism

`tul.spandec` builds `morph/model/tul_spandec.py::SpanDecoder`: `spandec_layers` (2)
pre-norm causal blocks over the per-slot sequence `[z, t_0 .. t_{J-2}]`, predicting
`t_0 .. t_{J-1}` — the tokens of span `s+1`, `J = tul.span_cap` (32). The loss is the mean
over those tokens of `-log p(t_j | z, t_{<j})`, weighted by `tul.spandec_weight` (1.0), pad
slots and span 0 masked to `ignore_index`. Gradient reaches `z` from every token of the
span. There IS a token path (the teacher-forced prefix), which the spec demands.

`z` is `_readout(h_slots)` BEFORE `prefix_project` — the same state the MUX head,
`slot_z_optimize.py` and the per-pass gradient probe read, so every earlier instrument
still points at the same object.

The output head is the tied LM head through `fused_linear_cross_entropy` (the full readout
would be `[B, S, J, V]` = 2.4 GB fp32). It obeys `tul.mux_detach_head`, which the spandec
configs set TRUE against the ruler lineage's FALSE — not a second factor there, because
`mux_beta: 0` leaves that knob governing the decoder alone. The decoder's INPUT embedding
read is detached unconditionally: the MUX has no input-side read, so there is no precedent
to inherit, and an undetached one would reshape the table the slot seed is a bag-mean of.

`tul.slot_chain` (Test 3) adds `W_chain(h_t[k-1])` to slot `k`'s state at every pass —
zero-init, causal, not detached. **Named deviation:** the literal "slot `k`'s seed gets
slot `k-1`'s EXIT state" needs 64 sequential loops instead of the one masked update
`runtime-invariants` §6b requires. What is built is the wavefront form: at pass `t` the
injected state IS the previous slot's exit whenever that slot's realised depth is already
spent, and its live state otherwise. Recurrence depth is the pass count, not 64.

### Cost, measured not asserted

`MORPHTransformer._tul_layer_passes` on a real packed batch (2 rows, 2,084 token
positions, 110 valid slots, mean depth 6), confirmed by the trainer's own `lp/tok` line at
step 20 of each smoke:

| model | block-passes / token | params |
|---|---|---|
| `slot-mux-mask-norm-match` (ruler) | **10.745** (`lp/tok=10.76`) | 264.66 M |
| `slot-spandec-mask` | **14.676** (`lp/tok=14.70`) | 292.48 M |
| the span decoder alone | 3.931 | 27.83 M |
| the slot chain alone | ~0 (one `[B,64,1024]·[1024,1024]` per pass) | 1.05 M |
| plain panel at mean_depth 6 | 44.000 | 264.7 M |
| plain panel at mean_depth 1 (`plain-coda-matched`) | **14.000** | 265.1 M |

So `plain-coda-matched` matches the span-decoder arm's TOTAL to 4.6 %. A plain model
matched to the decoder's 3.931 alone is **not expressible** — the plain per-token count is
`n_prelude + n_core·d + n_coda` = 8 + 6d, whose floor is 14.0, because a plain model must
still run its prelude and coda on every token. Written down rather than faked.

The decoder's 27.83 M parameters are NOT in the token-CE path; they only shape `z`. A CE
move on these arms is therefore not a capacity reading. The wall clock is real: the smokes
read 10,829–10,888 tok/s against the ruler's 12,662 (0.855x), all above the 8,086 floor.

### The instrument the arms are scored with — `all_slots`, built in this change

`worth_profile.py`'s `zero` mode ablates ONE route, the prefix write, and the budget priced
that at 0.093 of 0.399 nats. `plan_mode="all_slots"` cuts all three tensors that reach a
coda slot cell: the prefix write, the cell's per-layer injections (`x0` carries the slot
seed, the bigram term carries the span bag-mean), and the cell's own attention over its
span (`tg_allow` rebuilt with `slot_queries_slots_only=True`). It refuses on anything but
a `tg_restrict` scope-"all" slot-loop arm.

**One route it does NOT cut, so every `all_slots` number is a LOWER bound:** the CCA causal
conv and its `W_v_prev` value shift still read the positions before a slot cell. Cutting
them needs a `tg_seg` reset that also fires between one span's tokens and the next, which
would change the TOKEN positions' conv and put an operator change inside a paired CE
difference.

`worth_profile.py --paired-rows` packs with `lab/divergence/_rows.py::pack_rows` — the
depth sweep's own packer — and writes the stream index of every scored token, so the
budget curve, the prefix-write worth and the all-slot worth can be read on the same
tokens. `--verify-tok-index <sweep .tokens.npz>` asserts the identity rather than assuming
it.

### Readout, per arm

Runner `arc/run_slotloop3.sh`, KIND `slot`, SWEEP_CKS 2,500 and 5,000, OUTDIR
`/home/wolfe/morph-scratch/arc/results/2026-09-11-spandec`, a smoke first, the draw under
the sustained tripwire. Then, on each 5,000 checkpoint:

1. `core_depth_sweep.py --depths 1,2,3,6,9,12,16 --rows 480` — the token K-curve, and the
   arm's own local readout: `spandec_ce` is now a sweep column beside `mux_local`, so a
   span-decoder arm's loop contribution on its own job is measured the way every MUX arm's
   is.
2. `worth_profile.py --rows 192 --paired-rows --verify-tok-index <that arm's sweep npz>` —
   the `zero`, `shuffle`, `wrong_seed` and `all_slots` profiles by offset, on the sweep's
   own tokens, against the budget's curve (0.958 / 0.690 / 0.591 / 0.508 / ~0.42 / ~0.31 /
   ~0.31 at worth bins 0 / 1 / 2 / 3 / 4-7 / 8-15 / 16+; the worth profile bins by the
   PREDICTING position, so worth bin k lines up with budget offset k+1).
3. `slot_z_optimize.py` — `ce_entry − ce_loop`, the write contribution, and the z-opt entry
   gap.
4. `slot_state_probe.py`, and `probe.jsonl`'s `loop/core_gain_t0` for the scale mode.

## Predictions (frozen)

The first four are the orchestrator's, frozen before any arm ran; the probabilities and
the reasoning under each are the builder's.

- **P-a (Test 1 flattens the worth profile toward the budget's shape).** On
  `slot-spandec-mask` at 5,000, the `zero` worth at offset bin 8-15 rises from the ruler's
  0.057 to **at least 0.12**, moving from 0.05 toward the budget's 0.31. **25 %.**
  Reasoning: the target now charges the decoder at every offset, which is the right
  direction, but the ruler's own profile is what a ONE-SHOT write into two coda cells can
  carry, and the decoder does not widen the write. The budget's flat component is a running
  state and Test 1 does not build one. I would take 25 % on "rises at all beyond its CI"
  being more likely than the full move: call it 60 % that offset 8-15 rises above 0.075 and
  25 % that it reaches 0.12.
- **P-b (Test 1 closes part of the CE gap to plain).** `slot-spandec-mask` sits **0.05 to
  0.15 nats** closer to `plain-panel-norm-match` than `slot-mux-mask-norm-match` does,
  token-paired at depth 6 on the 480-row sweep. **20 %.** Reasoning: the arm ADDS 3.93
  block-passes per token of overhead that buys the token CE nothing directly, and
  `slot-mnext-staged-20k` measured exactly this shape — "the overhead, not the loop, is
  what costs the CE", a gap flat at 0.16 from 10k. I expect the CE to move the WRONG way or
  not at all, and I am recording that against the orchestrator's prediction rather than
  quietly agreeing with it. 45 % that the gap does not close at all (moves ≤ 0).
- **P-c (the token K-curve stays flat).** `slot-spandec-mask` token K1−K6 **below 0.01**.
  **80 %.** Reasoning: every arm in the campaign reads ≤ 0.0033 and the ternary rule already
  removed the mask's historic token dependence. A target change acts on the slot's job, and
  the token K-curve measures the TOKEN path's depth dependence — the two have never moved
  together on this tree.
- **P-d (Test 2 is spikier than Test 1 at the same total).** `slot-spandec-mnext-mask`'s
  ratio of worth at offset bin 0 to worth at bin 8-15 is strictly greater than
  `slot-spandec-mask`'s. **55 %.** Reasoning: the bag term re-weights a span's first
  positions by `rho^j`, which is the spike's shape, so adding it should tilt the profile
  forward. Only 55 % because the two terms are averaged into one exit state and a 1024-dim
  state may satisfy both without a visible tilt.
- **P-e (Test 3 raises the offset-8+ worth above Test 1).** `slot-spandec-chain-mask`'s
  `zero` worth at offset bin 8-15 exceeds `slot-spandec-mask`'s, CIs disjoint. **35 %.**
  Reasoning: the chain is the only mechanism in the batch that can hold a running state,
  and the budget's flat component is exactly that. Against it: the core's attention over
  the 64-cell sequence ALREADY lets slot k read slot k-1 at every pass, so the chain buys a
  direct edge rather than a new reachability; and the wavefront's direct path reaches only
  T ≈ 6 slots back per forward.
- **P-f (Test 5, dropout off closes part of the gap to plain).** `slot-mask-dropout-off`
  sits closer to `plain-panel-norm-match` than the ruler does, by **at least 0.03 nats**,
  token-paired at depth 6. **65 %.** Reasoning: the dropout is a straight CE tax and under
  `tg_restrict` a dropped token's neighbours are its own span only, so the tax is heavier
  here than on any unmasked arm. The 35 % is that the dropout also protects the slot channel
  by forcing the coda to use it, and removing it could cost the channel more than it saves
  the CE.
- **P-g (Test 5, the quarter-weight MUX).** `slot-mask-mux-quarter`'s `zero` worth total
  lands strictly between the ruler's 0.093 and the no-MUX twin's (whose exit equalled its
  entry). **60 %.** Reasoning: the MUX's weight is what pays for the loop (7.3x the token
  CE into the loop state), and halving-and-halving it should interpolate. The 40 % is that
  the two published arms differ by presence, not by weight, and a quarter may be plenty.
- **P-h (survival).** All six arms HEALTHY to 5,000, no sustained tripwire
  (`preclip/total > 1e4` at step ≥ 200). **70 %.** Reasoning: the recipe carries the
  1,000-step ramp and the fixed-point term, which have held every arm since 2026-09-04, and
  the smokes' step-20 `preclip/total` is 22-51 against a 1e4 bar. The 30 % is that the
  spandec arms put a NEW auxiliary gradient on the shared core and prelude that is 2.9x the
  token CE's into `z` at init, and no arm on this tree has run with it.
- **P-i (the decode-cheap readout).** `slot-spandec-mask`'s `spandec_ce` at 5,000 is
  **below** the arm's own token CE at the same step. **50 %.** Reasoning: a span decoder
  conditioned on `z` plus the span's own prefix is predicting the same tokens the coda
  predicts, with less context (no cross-span attention at all) but with a dedicated
  2-block reader. Genuinely a coin flip, and it is the number Test 4 is: it says what the
  thought alone buys, at the decoder's cost rather than the coda's.
- **P-j (cost).** Every spandec arm finishes 5,000 steps inside **75 min**. **75 %.**
  Reasoning: 10,829-10,888 tok/s measured at step 20 of the smoke, 5,000 × 6,144 / 10,850 =
  47 min, plus eval and the sweeps. The 25 % is that a step-20 rate on a 21-step run is not
  a 5,000-step rate.

## Binding

- **P-a TRUE and P-e FALSE.** The target is the lever and the write is not the bottleneck.
  The next arm widens the target, not the channel: every-pass span decoding, or a decoder
  over the next TWO spans.
- **P-a FALSE and P-e TRUE.** The target was never the bottleneck; the slot needs somewhere
  to keep a running state. The next arm is the chain WITHOUT the decoder (one factor), and
  the state's rank becomes the measurement.
- **Both FALSE, K-curves flat, worth unmoved.** The slot channel's ceiling on web text is
  the one-shot write itself, and the arc's answer is that a per-span latent cannot carry a
  flat 0.31-nat long-range component whatever it is graded on. The lane then moves to the
  corpus (the code pair the budget result left unbuilt) or off the slot loop entirely.
- **P-b TRUE but P-a FALSE.** The decoder bought CE without changing what the slot carries,
  which means it acted as a regulariser on the shared core. Report it as such and do not
  call it a slot result.
- **P-h FALSE on a spandec arm.** Read `loop/core_gain_t0` and the fixed-point term first;
  a detonation traced to the auxiliary head goes on the record with
  `tul.mux_detach_head` as the first knob, since these arms are the first to put a
  J-token-per-slot gradient on the tied table.

## Not verified before launch

- **The 5,000-step conjunction.** The longest any new config has run is 21 steps.
- **Anything about a TRAINED span decoder.** Every gradient number below is at fresh init.
- **The `all_slots` reading at 192 or 480 rows.** It has been run once, on 6 rows, on
  `slot-mux-mask-norm-match` step 5,000: `zero` total +0.1049, `all_slots` total +0.1344.
  Six rows is a smoke, not a measurement.
- **The CCA conv / value-shift route at a slot cell**, which `all_slots` does not cut. Its
  size is unmeasured, so the gap between `all_slots` and the budget is an upper bound on
  what the mask arm fails to route.
- **The optimizer-state cost of 27.83 M extra parameters.** The smoke's peak (14.35-14.52 GB
  against the ruler's 12.83 GB) covers 21 steps of AdEMAMix, not 5,000 with the full moment
  set warm.
- **`tul.slot_chain` at a nonzero `W_chain`.** It is zero-init, so a 21-step smoke exercises
  the plumbing and a near-zero map. Whether the wavefront is stable once the map is real is
  unmeasured, and the gain hinge probes the map WITH the chain injected.
- **Generation.** No spandec arm has produced a sample. The decoder is a training-time
  scorer and is not in the deployed forward, but the arms' `gen_every` is 0 and the slot
  channel's generation behaviour under a changed target is untested.
