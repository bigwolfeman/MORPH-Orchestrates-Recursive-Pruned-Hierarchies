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
