# Plain Parcae loop: does its depth value sit at span boundaries?

Status: failure (prediction 2 fails, in the opposite direction; filed 2026-10-14)

## Question

The trained plain + pointer loop (seed 2, 5k) earns K1-K6 +0.0377 (sweep: 3.7633 at depth 1, 3.7256 at
depth 6). The minimal slot loop ties plain at depth 1 and earns 0 from depth. Wolfe's TUL premise: the
depth a model needs is a per-span thought that every token of the next span reads. If that holds, a plain
loop that gives depth only to the boundary tokens (the last token of each span) should recover far more of
the depth gain on the FOLLOWING tokens than the same number of random tokens given depth.

## Method

Eval only, `lxtul/depth_mask_eval.py` (branch lxtul/depth-probe), on `parcae-runs/plain-pointer-norm-5k-s2`,
the trainer's 480 eval rows. Passes 2..6 update only the masked positions; the others keep their pass-1
state and still serve as keys and values. Conditions: all_1, all_6, boundary_6, random_6 (same count per
row, 3 draws), all_but_boundary_6. Readouts on fixed position sets: ce_all, ce_shallow (non-boundary),
ce_next4 (first 4 tokens after a boundary), ce_boundary_own.

R(cond, set) = (CE all_1 - CE cond) / (CE all_1 - CE all_6) on that set: the share of the depth gain the
condition recovers.

## Predictions

1. Gate PASS: all_1 and all_6 match sweep.json within 2e-3 (90 %).
2. R(boundary_6, next4) >= 1.5 x R(random_6, next4) (55 %).
3. R(boundary_6, all) <= 0.25 (75 %): most of the depth value is per token.
4. R(all_but_boundary_6, all) >= 0.85 (70 %): taking depth away from boundary tokens alone loses little.

## What each outcome changes

- 2 holds with R(boundary_6, next4) >= 0.2: the plain loop already concentrates span-level depth at
  boundaries. Next: a boundary-routed loop (pass 1 on all tokens, passes 2..T on boundary tokens only),
  pending Wolfe's call on the "no token loop" rule.
- Boundary is about equal to random: the plain loop's depth value is spread per token. The plain model gives
  no support for span-level depth at this scale; the weight moves to equal-wall-clock comparisons.

## Results

`results/2026-10-14-plain-loop-boundary-depth-probe/plain_s2.json`. Gate PASS: all_1 3.76328 / all_6 3.72564
against the sweep's 3.76325 / 3.72558, 510,003 tokens both. Position sets: all 510,003; shallow (non-boundary)
484,613; next4 100,674; boundary_own 25,390 (5.0 % of tokens). Random draws spread under 0.0004 nats.

| condition | ce_all | ce_shallow | ce_next4 | ce_boundary_own | R all | R shallow | R next4 | R own |
|---|---|---|---|---|---|---|---|---|
| all_1 | 3.7633 | 3.7851 | 3.7393 | 3.3477 | 0.000 | 0.000 | 0.000 | 0.000 |
| all_6 | 3.7256 | 3.7470 | 3.7050 | 3.3172 | 1.000 | 1.000 | 1.000 | 1.000 |
| boundary_6 | 3.7619 | 3.7851 | 3.7391 | 3.3203 | 0.035 | -0.001 | 0.004 | 0.898 |
| random_6 (mean of 3) | 3.7616 | 3.7833 | 3.7378 | 3.3476 | 0.044 | 0.046 | 0.044 | 0.002 |
| all_but_boundary_6 | 3.7272 | 3.7472 | 3.7055 | 3.3468 | 0.957 | 0.996 | 0.986 | 0.027 |

R = share of the all_1 -> all_6 gain on that position set (1 = full depth gain, 0 = none).

Read:
- Deep boundary states give the other tokens nothing: boundary_6 R shallow -0.001, R next4 0.004.
- Each token's depth gain is its own: boundary tokens given depth with every neighbour frozen at pass 1 keep
  0.898 of their own gain; boundary tokens frozen while every neighbour is deep get 0.027 of theirs.
- all_but_boundary_6 keeps 0.957 of the whole gain; the 5 % of positions it freezes lose almost exactly
  their own share.
- Random 5 % of positions recover 0.044 of the whole gain, about their count share; boundary positions
  recover 0.035, less than random.
- Boundary tokens' own depth gain (0.030 nats) is below the average token's (0.038).

Prediction check: 1 HOLDS (gate). 2 FAILS: R(boundary, next4) 0.004 against random's 0.044, the opposite
direction. 3 HOLDS (R(boundary, all) 0.035). 4 HOLDS (0.957).

## Verdict

Failure of the boundary hypothesis. In the trained plain Parcae loop the depth value is per token and
private: a token's extra passes improve its OWN next-token prediction, and almost nothing of it reaches
later tokens through attention, at boundaries or anywhere. Caveat: the probe is eval-only on a model
trained with one depth for every position, so it shows what this model does, not what a model trained to
share depth could learn.

## Updated hypothesis

On web text at this scale, the plain loop's 0.038 nats is per-token decode refinement, not a span-level
thought. A design that computes depth once per span and shares it (TUL, the boundary-routed loop) cannot
collect that value; it can only collect span-level depth, which no Parcae arm has shown without a
pass-indexed term. The boundary-routed loop is dropped. The open questions move to: does the shallow coda
force cross-span depth into the loop (`planned/2026-10-14-parcae-shallow-coda-5k.md`), and what does the
hierarchy buy at equal compute (`planned/2026-10-14-parcae-equal-compute-5k-budget.md`, LCM-Parcae).
