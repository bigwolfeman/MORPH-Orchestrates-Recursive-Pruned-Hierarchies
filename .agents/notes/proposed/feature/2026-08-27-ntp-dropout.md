# Agent Note: NTP dropout — run a fraction of steps with no slots at all

Status: proposed

## Problem

TUL trains every step with the slot layout on. Two consequences we have measured
or suspect:

1. **The plan is empty.** It carries ~0.07 nats of its span, and the MUX head
   converged to roughly the corpus marginal
   ([v1a](../../../../lab/experiments/failures/2026-08-25-mux-head-arm-v1a.md)).
2. **The core takes over** in 3 of 4 control seeds — its gradient share crosses
   0.5 between steps 2800 and 3000 while its Shapley value stays near zero.

Wolfe's framing: the model may need plain next-token prediction to build
representations before, and alongside, the structured objective. The warmup form
of this is a separate arm (activate the head only after step N). This note is the
STOCHASTIC form: keep a slice of ordinary NTP alive for the whole run.

## Proposal

With probability p ≈ 0.1 per optimizer step, run the forward with
`slot_layout=None` on a slot-free batch — the ordinary MORPH path, core looping
over token positions — instead of the TUL path.

It fits the architecture as designed rather than fighting it. `slot_layout` is
already a per-forward DATA argument (spec §8: "a data argument like `bag_size`"),
and `None` is bit-identical to the baseline forward. So this adds no runtime
feature flag and no branch inside the model.

**Batching is the one real implementation cost.** A TUL batch has slot tokens
inserted, so it is not a valid NTP batch — training on it with `slot_layout=None`
would teach the model to predict spurious `slot_id` positions. Two ways out:
emit the pre-insertion token sequence alongside the packed row (the packer in
`morph/model/tul_layout.py` already has it in hand, so this is nearly free), or
gather the non-slot positions back out with the existing `gather_positions` /
`compact_index` helpers and re-pad. The first is cheaper and less error-prone.

## Kernel impact (asked explicitly, answered explicitly)

Essentially none, for three reasons:

- **Inference is unaffected.** This is a training-time regularizer; deployment
  runs the TUL path only, so no deployed kernel gains a second case.
- **Two compiled variants, not thrash.** `torch.compile` caches per guard set,
  so the two shapes compile once each and then alternate against cache. The
  trainer already warms up several variants (active-set sizes [6,5,4,3,2,1] × 4,
  plus dynamic batch), so this is a smaller change than what already ships.
- **No new shape family.** The `slot_layout=None` path is the pre-TUL baseline
  path; its kernels already exist and are already exercised by arm A0.

The one caveat worth stating: any future kernel that fuses across the TUL
gather/scatter must still handle the plain path, but it has to anyway, because
A0 is a live arm.

## Alternatives considered

- **`token_state_dropout` (already shipped, 0.15).** Drops token STATES at the
  coda input. That is Bowman word dropout — a state-level tax that makes the
  coda need the plan. NTP dropout is the opposite direction and at a different
  level: it removes the PLAN, not the tokens. They are complementary, not
  substitutes, and their interaction is unmeasured.
- **Arm A4 (`coda_sees_slots: false`).** Permanent removal, an ablation rather
  than a regularizer. NTP dropout is stochastic and keeps both modes trained.
- **A Coconut-style curriculum** (anneal from all-NTP to all-TUL) instead of a
  fixed rate. Rejected as the FIRST version only because it adds a schedule
  shape to tune on top of an untested mechanism; the warmup arm already tests
  the annealed idea in its simplest form.
- **Do nothing and rely on the warmup arm.** Reasonable if the warmup alone
  fixes the plan. This note is explicitly gated on that result.

## Acceptance criteria

Gated: do not build until the NTP-warmup arm shows the head learns better when
representations exist first.

1. Core gradient share never crosses 0.5 in ≥ 3 of 4 seeds.
2. `ppl_tok` seed median at 3250 no worse than the control's.
3. **Plan-off ablation worth strictly greater than the arm's own p=0 comparison
   run.** This is the criterion that matters — see the risk below.

## Risks

**The central tension, and the reason this could backfire.** The goal is a plan
the coda RELIES on. Training the coda to work without the plan 10 % of the time
reduces its incentive to build machinery that reads the plan — the same shape as
posterior collapse, where a decoder strong enough to ignore the latent never
trains it. NTP dropout could therefore stabilise training while making the plan
MORE decorative, which would look like success on CE and takeover and be a
failure on the actual goal.

That is why criterion 3 is plan-off worth and not perplexity. An arm that
improves CE and reduces plan worth is a rejection, not a win.

Secondary risks: capacity split between two modes at a small model size; and
`p` becomes another hyperparameter in a campaign that already has too many.

Related: [[2026-08-25-mux-head-arm-v1a]] (the empty-plan measurement),
`docs/tul-spec.md` §3.4 (token-state dropout, the complementary tax).
