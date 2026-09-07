# Success: ARC E4 — the TG restriction under the constraint (R5 rerun)

Status: success
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

### Method note, 2026-09-07 03:40 (queue only; predictions untouched)

Queued behind E10 in `arc/run_e4.sh` at the E10 commit (Wolfe: "add e4"): 12-step smoke,
the 5000-step draw under the SUSTAINED tripwire (`lab/divergence/tripwire_sustained.py`,
arc Method amendment 2), `core_depth_sweep.py` at depths 1–8 and `worth_profile.py` at
2500 and 5000 on the arc's 480 rows, then the Jacobian sweep at iterations 0, 3, 7. The
E7 reading to compare against: at a mean-16 slot draw the masked arm's token K1−K6 read
+0.026 at 2500 on a map that detonated 200 steps later. E4 asks the same question on the
constrained mean-6 map.

## Results (2026-09-07 08:22; `arc/run_e4.sh` on worktree c8b20c0; draw 07:02–08:10, 5000 steps, HEALTHY, max `preclip/total` 566 at 4904; files in `results/2026-09-07-arc-e4/`)

Val: 4.6589 at 2500; last four 4.4462, 4.3097, 4.3502, 4.4309 (mean 4.384; Y2 4.2809);
final 4.4178. Forced slot depth on the 480 rows:

| | token K1−K6 | token K3−K6 | forecast K1−K6 | forecast K3−K6 | token CE at depth 6 |
|---|---|---|---|---|---|
| E4 @2500 | +0.0204 [+0.0193, +0.0216] | −0.0003 | +0.178 [+0.168, +0.191] | +0.0021 | 4.667 |
| E4 @5000 | +0.0209 [+0.0199, +0.0219] | +0.0002 [0.0000, +0.0004] | +0.187 [+0.177, +0.199] | +0.0016 | 4.336 |
| Y2 @5000 (ruler) | +0.0003 | 0.0000 | +0.0135 | +0.0002 | 4.205 |
| E7 pre-onset @2500 (mask, mean 16) | +0.0263 | +0.0018 | +0.823 | +0.0279 | 4.892 |

Plan worth at offsets 0..3 (192 rows): at 5000 shuffle 0.310, 0.148, 0.095, 0.080; zero
0.213, 0.180, 0.109, 0.103; wrong-seed 0.989, 0.084, 0.049, 0.043 (at 2500 shuffle 0.263 at
offset 0). Jacobian sweep (fp32 power iteration): at 2500 / 5000 the loop entry norm 491 /
501 → 436 / 459 after iteration 0 (realized gain 0.97 / 1.02), 1.05e3 / 1.31e3 after
iteration 7; σ_max of the step map 11.9 / 16.4 at iteration 3 and 7.7 / 14.8 at iteration 7
with typical (rms) gain 0.80–0.81; effective rank 269 → 45 over 8 iterations; per-iteration
change ratio 0.45 / 0.43 at iteration 3. The healthy band (arc-a arms: σ 11–22, rms
0.90–0.93, rank → 40–47), with a slightly MORE contractive typical gain.

Scored:

- **P4a TRUE** (5000 steps, tripwire silent).
- **P4b TRUE** (last-four 4.384, +0.10 against Y2; bar 0.15).
- **P4c TRUE** (shuffle worth 0.310 at offset 0; bar 0.10; Y2 0.034).
- **P4d TRUE** (token K1−K6 +0.0209 [+0.0199, +0.0219]; bar 0.01; every unmasked arm
  ≤ 0.0006).
- **P4e TRUE** (forecast K1−K6 +0.187 against Y2's 0.0135).

## Verdict

Five of five. With the route forced, the tokens read the slot: they lose 0.31 nats at the
span's first token when the slot's content is shuffled and 0.021 nats when the slot loop is
cut from six iterations to one. That is (c) answered: the reader branch was binding, and on
a STABLE map (the E7 reading was on a map at gain 1.04 that detonated 200 steps later).
The bar the arc set for THINK is not met: token and forecast K3−K6 are +0.0002 and +0.0016,
so what the tokens read is what the slot computes in its first three iterations, and the
price is 0.13 nats of token CE at 5000 against Y2 on the same rows (the mask removes the
direct token route). E4's binding rule as written ⇒ E5 on E4 (20k, matched wall clock
against A3-20k); that run is Wolfe's call and is not queued.

Not verified: whether the 0.13-nat gap closes by 20k (deep models converge slower; the
matched pair's A2 gap closed 0.132 → 0.012 from 5k to 20k); the mask arm's generation
(the gate-spec leak check, gen_every 0 here); whether the slot's depth reading survives a
lower `prefix_k` or a shorter span cap.

## Updated hypothesis

The reader is not the loop's problem: under the mask the tokens use the slot fully, and the
slot's loop still stops at 3. The block-loop design (mask + deep slot draw) failed on
stability, not on the reader, so its rerun should carry the fixed-point term (E10a, 6 of 6
under the assay) ported to `_tul_core`, the per-sample gradient window, and this mask, at
mean 6 first, then the draw.
