# Experiment: does the plan hold content the coda declines to use?

Status: **planned.** No code change needed — `tul.token_state_dropout` is already
a config knob (`morph/model/tul.py`, default 0.15). Predictions frozen
2026-08-27 before any arm ran.

## Question

The [objective split](../failures/2026-08-27-objective-split.md) refuted the
credit-assignment reading. On the clean control seed the coda's objective and the
slot's direct objective are **orthogonal to within ±0.04**, and `‖g_main‖` is
**82 % of `‖g_emit‖`**. The core is not starved of the coda's gradient. It gets a
large one, and the plan is still worth 0.015 nats.

A gradient is a SLOPE. The 0.015 nats is the achievable GAIN. So the binding
constraint is headroom, and there are exactly two ways headroom can be missing:

1. **The plan holds content the coda declines to use.** The coda can read span
   i's raw tokens, so it never has to consult the slot. Then the fix is to make
   the token path expensive — one config line.
2. **The plan holds nothing.** Then no amount of taxing helps, and only a NEW
   objective with its own headroom can put content there.

`tul.token_state_dropout` is the tax (Bowman word dropout on the coda's token
input; p is the probability a token state is DROPPED — at eval p=1.0 collapses
`ce_main` to 7.53). Raising it during TRAINING lets the coda adapt, which the
eval-time sweep could not.

**This experiment can kill the MTP plan before it is built.** If taxing works,
building an MTP chain is solving a problem we do not have.

## Arms and protocol

Control is the existing `ctrlworth-s{1,2,3}` at p=0.15 — same recipe, same
steps, already measured. New arms `tul_tax50` (p=0.5) and `tul_tax85` (p=0.85),
seeds 1, 2, 3 each. All: 3500 steps, batch 6, `ademamix_alpha_cap=3.5`,
`use_kernels=false`, `eval_every=250`, `ckpt_every=500`, `grad_probe_every=1`.
Plan-off worth from `slot_path_worth.py --batches 8` on `step_3000.pt`, the same
fixed eval set every other arm used.

**Matched reference, measured 2026-08-27 at step 3000 on three control seeds:**

```
plan worth   +0.0124  +0.0148  +0.0164     median 0.0148, max 0.0164
loop worth   +0.0042  -0.0002  +0.0013     median 0.0013, max 0.0042
ce_main       4.5459   4.4586   4.4680
```

### Method amended 2026-08-27, 15 minutes into the first arm

**Change:** a fourth rung, `tul_tax30` (p=0.3), seeds 1/2/3, queued after the two
frozen arms. **Predictions are untouched** — T1's monotonicity now reads over
p ∈ {0.15, 0.3, 0.5, 0.85}, and T2/T3/T5 still name p=0.85.

**Reason, and it is my error:** `base.yaml` line 477 states the designed range in
its own comment — *"arm sweep {0, 0.15, 0.3}"*. I chose 0.5 and 0.85 without
reading it. Fifteen minutes in, `tax50` seed 1 had already taken over: core share
**0.71 at step 200 and 0.98 by step 300**, 515 of its first 676 probed steps over
0.5, with validation loss RISING 7.147 → 7.777 where the control at the same
steps descends 6.465 → 6.054. If p=0.5 destabilises, p=0.85 certainly will, the
T5 health gate fails on every rung above the control, and **T2 and T3 become
unreadable** — the panel would answer nothing. p=0.3 is the rung that can be
healthy.

**Not cut:** the two frozen arms run to 3500 steps as pre-registered. Stopping a
diverging arm early after seeing its first seed would be a data-driven protocol
change, and one recorded precedent (`repl-det-a`) peaked at core share 0.9369 and
recovered.

**An observation, not yet a result:** raising the tax may itself trigger the
takeover — masking token inputs forces the coda onto the slot path, which
concentrates gradient on the core, which is the condition the takeover needs.
If that replicates across seeds it is a constraint on the whole "tax the token
path" family, separate from whatever T2/T3 say about content. It is one seed at
step 676.

## Predictions (frozen 2026-08-27, before any run)

- **T1 (dose–response):** median plan-off worth at step 3000 is monotone
  increasing across p ∈ {0.15, 0.5, 0.85}.
- **T2 (the tax clears the control band):** at p=0.85, plan-off worth exceeds
  **0.0164** — the maximum control seed, not the median — on **≥ 2 of 3** seeds.
- **T3 (the LOOP, not just the bag-mean):** at p=0.85, loop worth exceeds
  **+0.0042** on **≥ 2 of 3** seeds. **This is the one that matters.** Plan worth
  includes the span bag-mean, which the slot carries without looping at all; loop
  worth does not. An arm that raises plan worth and leaves loop worth at zero has
  made the bag-mean more useful, not the core.
- **T4 (the tax costs CE):** `ppl_tok` @3250 medians are ordered
  control < p=0.5 < p=0.85. Recorded so a free-lunch surprise is visible rather
  than assumed away.
- **T5 (health gate, and it runs first):** at least **2 of 3** seeds at p=0.85
  reach `ce_main < 5.0` at step 3000 (control 4.459–4.546; the two broken arms of
  the 08-27 panel sat at 5.869 and 6.354). **If T5 fails, T2 and T3 are
  unreadable** — broken runs score HIGH on plan worth by construction, measured:
  `warmup-s0` reads 0.0247 at ppl 305 and `center-s1` reads 0.0301 at ppl 603.
- **T6 (THE DECISION):** T2 **and** T3 fail while T5 passes ⇒ the plan has no
  content to route, **taxing is refuted as a lever**, and only a new objective can
  help — the MTP-shaped chain is indicated. T2 **and** T3 hold ⇒ the plan HAS
  content the coda declines to use, the fix is one config line, and **the MTP
  chain should not be built.**

## Risks and confounds recorded up front

- **p=0.85 may simply collapse.** At eval, p=1.0 puts `ce_main` at 7.53. If both
  high-tax seeds break, T5 fails and the panel falls back to p=0.5 alone — two
  points, which cannot establish the monotonicity T1 asks for. Recorded now.
- **Token dropout consumes RNG**, so a taxed arm is not step-comparable to the
  control at a fixed seed. Every comparison here is at the level of curves and
  medians, never bit equality.
- **The 0.0191 nats quoted elsewhere is not the reference here.** It came from a
  different run at step 1750. The band 0.0124–0.0164 is the matched one.
- **This tests the coda's willingness, not the plan's capacity.** A null result
  bounds what the CURRENT coda can extract; a differently-shaped decoder might
  extract more from the same plan. That limit is not addressed by any arm here.
