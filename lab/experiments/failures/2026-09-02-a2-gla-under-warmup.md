# Planned: GLA back under warmup — does the retention branch still cost CE and earning once the ramp is the stabilizer?

Status: failure
Date: 2026-09-02 (frozen 23:40, before launch; trigger: Wolfe — "didn't we just
show that a huge portion of the stability was our warm up? We might be able to
bring GLA back. That's probably another arm we should test tonight")

## Question

The loop-killer bisect (2026-08-31) removed GLA and the spectral cap because
the conjunction killed depth-earning. The detonations began when GLA came out
(notul-bc0, GLA on, cap off: healthy 1/1; notul-bg0c0, both off: 1/2 and the
whole paid-axis campaign since). Tonight a 1000-step LR warmup gave 0/3
detonations without GLA. If the ramp is the stabilizer, is there anything
left for GLA to give back? In the bisect, GLA on (bc0) vs off (bg0c0) at 4500
steps on notul: val 4.3606 vs 4.1812 (+0.18 for GLA on), K1-K6 0.142 vs
0.220 (-0.08 for GLA on). So the prior says GLA costs both; Wolfe's read is
that its benefit was masked by the cap and the instability.

## Method

3 draws, `tul_a2` + panel flags + `training.warmup=1000` +
`model.retention=true` (tul_a2 already has `retention_carry: none`, the only
causal carry; the acausal carry is a known leak and is NOT re-enabled), cap 0,
2500 steps, ckpt_every 0, tripwire watcher. Draw-1 gate: the
"MORPHTransformer: N params" line reads >= 280M (A2 no-GLA is 268.2M; GLA
adds ~19M). Then `a2_depth_sweep.py --depths 1,6 --rows 48` on every healthy
draw's step_2500.pt. Scored against tonight's wu draws on the identical
instrument: val 4.5391 / 4.5440 / 4.5413 (mean 4.5415), K1-K6 0.0463 /
0.0489 / 0.0456 (mean 0.047).

## Predictions (frozen)

- **P-G1.** GLA detonations <= 1/3: **70%** (bc0 was clean without warmup;
  warmup is 0/3; the two stabilizers stack).
- **P-G2.** Healthy GLA draws' final val at 2500 within 0.10 of the wu mean
  4.5415: **35%** (the bisect's +0.18 says no; the arm is here because Wolfe's
  read says the bisect number was contaminated).
- **P-G3.** Healthy GLA draws' K1-K6 at 2500 >= 0.026 (wu mean minus 0.02):
  **50%**.

## Binding (feeds the 20k pair's frozen selection rule)

The 20k pair queues right after this arm. The pair runs WITH
`model.retention=true` iff at least 2 healthy GLA draws exist AND their mean
final val at 2500 is below the wu mean by >= 0.05 (i.e. <= 4.4915) AND their
mean K1-K6 >= 0.026. Otherwise the pair runs with retention off (the wu
recipe). The rule is in `run_gla_then_pair.sh` and is not edited after launch.

## Not verified before run

`model.retention=true` composed with `model.use_kernels=true` and
`tokens_through_core` has never trained (bc0 was notul; the fused-kernel GLA
path on the packed TUL row is exercised first by draw 1, hence the param-count
gate and the tripwire).

## Results (2026-09-03 00:19-02:28, runs tul-a2-gla1/2/3)

Draw-1 gate passed: 287.1M params (268.2M without retention). Tripwire silent
on all three; max preclip/total after step 200: 223 / 29.8 / 41.7.

| draw | final val @2500 | K1 | K6 | K1−K6 |
|---|---|---|---|---|
| gla1 | 4.5354 | 4.5519 | 4.5112 | 0.041 |
| gla2 | 4.5387 | 4.5753 | 4.5153 | 0.060 |
| gla3 | 4.5396 | 4.5609 | 4.5134 | 0.047 |
| mean | 4.5379 | | | 0.049 |
| wu1-3 (no GLA) mean | 4.5415 | | | 0.047 |

- **P-G1 (70%): TRUE.** 0/3.
- **P-G2 (35%): TRUE, against the majority prior.** Mean 4.5379 is 0.004
  BELOW the wu mean, inside the draw spread (0.005).
- **P-G3 (50%): TRUE.** Mean earning 0.049 >= 0.026; equal to the wu mean.

Frozen rule applied by the runner at 02:28:14: n=3, mean_val 4.5379 (needs
<= 4.4915), mean_earning 0.0494 -> **retention off** for the pair.

## Verdict

**FAILURE by protocol (P-G2 resolved on its minority side), and the answer is
clear: under the ramp GLA is neutral on all three axes.** It neither
destabilizes nor stabilizes (the ramp already does that), costs nothing and
buys nothing in CE at 2500, and leaves the loop's earning unchanged. The
bisect's +0.18 nats cost for GLA on the flat schedule was the instability
tax, not GLA's own price; the ramp removed the tax and revealed GLA as inert
at this horizon. It does not earn its 19M parameters, so the pair runs
without it and the recipe keeps retention off.

What this does not say: whether GLA's retention state pays off past 2500
steps or at longer context (its purpose is long-range memory; seq 1024 and
2500 steps are not where that would show). Raven (Wolfe's GLA variant) is the
next thing to test on that axis, once TUL is locked.

## Updated hypothesis

Every stabilizer measured so far (cap, GLA, gamma freeze) was either
harmful or inert once the LR ramp is in place; the ramp is the one stabilizer
that is also a CE win. Retention stays off by default; it is a capability
question, not a stability one, from here on.
