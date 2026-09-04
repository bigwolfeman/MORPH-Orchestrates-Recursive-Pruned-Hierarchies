# Planned: ARC E4 — the TG restriction under the constraint (R5 rerun)

Status: planned
Date: 2026-09-04 (frozen; not launched; GPU time is Wolfe's call)
Arc: `2026-09-04-loop-contribution-arc.md`, branch (c) READER.

## Question

Every slot arm's token loss is flat to 0.001 over slot depth, and on a detached-z arm a
planted TRUE target moved token CE by 0.0000: the tokens have no gradient reason to read
the slot when the prefix is available through attention. The TG restriction makes the
slot the only route from an earlier span to a later token. R5 (M-next-mask) was the
panel's arm for this and the 2026-09-03 stop killed it at step 1639 with no reading. This
reruns it on the constrained arm, so the forecast face cannot detonate before 5000.

## Arm

| # | run | config | one line |
|---|---|---|---|
| E4 | `to-mnext-y2-mask` | `tul_to_mnext_y2_mask` | Y2 + `tg_restrict: true` (eager attention) |

Control: Y2 on disk (no mask). If E1/E2/E3 produced an arm that THINKS, the mask goes on
THAT arm instead (Method amendment naming it, dated, before launch; predictions here
stay as written and are scored against Y2 in either case).

## Readouts

As E1, plus `slot_path_worth.py` (plan-off / loop-off / shuffle) and `val/attn_lift`.
Wall clock against `to-mnext-mask`'s eager pace (1639 steps by the stop).

## Predictions (frozen)

- **P4a.** E4 reaches 5000 with the tripwire silent: **55%**.
- **P4b (the price).** Last-four val CE is within 0.15 of Y2's 4.2809: **50%** (the gist
  family paid 0.09–0.36 for this mask).
- **P4c (reliance).** Plan worth at offset 0, shuffle, ≥ 0.10 (Y2: 0.034): **60%**.
- **P4d (the reading).** Token K1−K6 at the slot ≥ 0.01 (the tokens read the loop's
  depth; every unmasked arm ≤ 0.0006): **25%**.
- **P4e.** Forecast `mux_local` K1−K6 on E4 exceeds Y2's 0.0135: **45%**.

## Decision rule (binding)

- P4d TRUE ⇒ (c) was binding: with the route forced, the depth reaches the tokens. E5
  on E4 (or on the masked THINK arm), scored at matched wall clock against A3-20k.
- P4c TRUE and P4d FALSE ⇒ the tokens read the slot's CONTENT and not its DEPTH: the
  loop's extra passes add nothing the reader can use even when it must read; (c) is
  closed and the loop's emptiness is a (b) fact.
- P4a FALSE ⇒ the mask's gradient into the shared core is the takeover fuel the 08
  campaign measured and the constraint does not cover it; the trip step and
  `preclip/*` shape are filed and the mask is retired from this arc.

## Not verified before launch

`tg_restrict` with `slot_gain_lambda` and `slot_cot_clip` in one build (each is tested
alone; the eager attention path under the hinge's extra core steps is the untested
conjunction). 12-step eager smoke first.
