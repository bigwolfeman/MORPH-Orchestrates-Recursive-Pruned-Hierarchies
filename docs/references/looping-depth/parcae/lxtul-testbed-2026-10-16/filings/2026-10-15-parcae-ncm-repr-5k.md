# Parcae: span NCM v2 with a RePR span code, at 5k

Status: failure (filed 2026-10-16)

## Question

The span NCM v1 (`failures/2026-10-14-parcae-span-ncm-5k.md`) failed because its reconstruction-trained
code costs more to predict than it saves (level 1: 0.226 nats per token cost, 0.12 saved). RePR (arXiv
2610.20533) fits a representation so its dot products match the PMI that a generative model measures. Does a
span code fitted to the PMI between a span and its context cost less to predict per nat it saves?

## Arms (branch lxtul/ncm-repr; 5000 steps; batch 6; seed 1, as v1)

| stage | arm | what |
| --- | --- | --- |
| 1 | RePR span tokenizer | 4-block trunk, full and span-local views; PMI = local NLL - full NLL (stopgrad); R_ctx / R_loc heads, weight 0.01; code = R_loc at the span's last token; 4-stage residual VQ |
| 2 | RePR span plan on gpt8 | v1's plan arm on that code; levels [1, 1, 1, 1], 3 generated codes per span, m fixed at 3 |

References on the same tokens: gpt8 + pointer 3.7019 (2 seeds); v1 plan bound at m = 1 / 3: 3.8517 / 4.7815;
v1 level-1 code cost 0.226 nats per token.

## Predictions

1. Stage 1: mean PMI per scored token at least 0.5 nats on val (75 %).
2. Stage 1: R^2 of the repr fit at least 0.3 on val (50 %).
3. Stage 2: level-1 code cost per token below v1's 0.226 (60 %).
4. Stage 2: bound at m = 3 below v1's 4.7815 (80 %).
5. Stage 2: bound at m = 3 below gpt8 + pointer 3.7019 (10 %).
6. Stage 2: generator K1-K6 on the bound at least +0.005 (40 %).
7. Every run finite (90 %).

## Method

Configs `ncm_repr_tok.yaml`, `ncm_repr_plan.yaml`. Smokes passed (`morph-scratch/abq/repr/gpu_todo.log`):
stage 1 loss 24.26 -> 7.85 in 20 steps (step-1 CE above uniform is the testbed's tied-head start: gpt8
starts at 16.3, v1's plan at 19.2), stage 2 loss 17.63 -> 6.92. The first smoke attempt aborted on R^2 = NaN
at step 1, where the PMI target has zero variance; `ncm/pmi_var` now logs beside it.

## Results

The predictions above use the old numeric format. They are graded here under the 2026-10-16 protocol. The
percent figures are not graded. One seed per stage, so no seed spread was measured for this arm. The CE@6 seed
spread of about 0.010 nats from other Parcae arms is used as the noise band.

Sources: `/home/wolfe/parcae-runs/ncm-repr-tok-5k/summary.json` and `.../ncm-repr-plan-5k/summary.json` (key
`scalars`), `morph-scratch/abq/full1015/chain.log` lines 6 and 7. v1 numbers come from
`lxtul/ncm:docs/experiments/failures/2026-10-14-parcae-span-ncm-5k.md` (read with git show).
Per-level cost = bound(m) - bound(m-1) + (token CE(m-1) - token CE(m)); this equals the code NLL per token.

Stage 1 (RePR span tokenizer, `ncm-repr-tok-5k`): PMI mean per scored token 0.3775 (471,708 scored tokens);
R^2 of the repr fit -0.0206; local CE 4.3312, full CE 3.9642. Codebook use 0.40 / 0.42 / 0.44 / 0.43 at levels
1 to 4, perplexity 666 / 411 / 435 / 434 of 4096. Speed 54,270 tok/s, peak 5.07 GiB.

Stage 2 (RePR plan on gpt8, `ncm-repr-plan-5k`), by m, depth 6:

| m | bound | token CE given the plan | code cost per token | K1-K6 on the bound |
| --- | --- | --- | --- | --- |
| 0 | 3.8147 | 3.8147 | 0 | 0 |
| 1 | 3.9365 | 3.6761 | 0.2604 | +0.0007 |
| 2 | 4.1016 | 3.5883 | 0.5133 | +0.0011 |
| 3 | 4.2804 | 3.5002 | 0.7802 | +0.0014 [+0.0013, +0.0016] |

Per-level cost (bound step + token CE saved), v2 against v1:

| level | v2 bound step | v2 CE saved | v2 cost | v1 cost |
| --- | --- | --- | --- | --- |
| 1 | +0.1218 | 0.1386 | 0.2604 | 0.226 (v1 note) |
| 2 | +0.1651 | 0.0878 | 0.2529 | 0.466 (0.692 - 0.226) |
| 3 | +0.1788 | 0.0881 | 0.2669 | 0.983 (1.675 - 0.692) |

Whole plan, m = 3 against m = 0: v2 costs 0.7802 per token and saves 0.3145 (net bound +0.4657). v1 costs
1.6746 and saves 0.6374 (net bound +1.0372; v1 bounds 3.7443 and 4.7815, token CE 3.1069). Cost per nat saved:
v2 2.48, v1 2.63. At level 1 it is 1.88 against 1.91. The v2 plan has levels [1, 1, 1, 1] and v1 had levels
1, 2, 4, so the per-level rows are not the same code size and should not be read as a like-for-like cost.

Code CE per code at depth 6: v2 5.327 / 5.174 / 5.459 (levels 1 to 3) against v1 4.614 / 4.796 / 5.058.
Speed of stage 2: 34,030 tok/s, peak 12.94 GiB.

Prediction check:

1. Mean PMI at least 0.5 nats: 0.3775. Did not hold.
2. R^2 at least 0.3: -0.0206. Did not hold. The fit did not learn anything above the mean.
3. Level-1 code cost below v1's 0.226: 0.2604, higher by 0.034. Did not hold. This is 3x the 0.010 band, but
   the v1 row set and this row set were not confirmed identical, so read it as a direction, not a size.
4. Bound at m = 3 below v1's 4.7815: 4.2804, lower by 0.5011. Held. Most of it comes from the shorter plan
   (4 codes of 1 level each, not 1, 2, 4) and from m = 0 being 0.07 higher (3.8147 against 3.7443).
5. Bound at m = 3 below gpt8 + pointer 3.7019: 4.2804, over by 0.5785. Did not hold.
6. K1-K6 on the m = 3 bound at least +0.005: +0.0014. Did not hold. (v1: +0.0076.)
7. Every run finite: held (chain.log). A gate.

Also: m = 0 token CE is 3.8147, which is 0.113 above gpt8 + pointer's 3.7019 (the v1 note's mean of two seeds).
The plan-free path again pays for training with m drawn per span. Row set parity with the 3.7019 figure was
not checked here.

## Verdict

Not supported. Main question: does a PMI-fitted span code cost less to predict than the reconstruction code?
Level-1 code cost is 0.2604 per token against v1's 0.226, higher by 0.034, and it saves 0.1386 against v1's
0.118. Cost per nat saved is the same (1.88 against 1.91). The whole plan is cheaper (0.78 against 1.67) only
because it carries fewer codes and saves half as much.

A caveat that matters: stage 1 did not fit the PMI. R^2 was -0.02 and mean PMI 0.38 nats. So the code is not
a good PMI-fitted code, and this run does not test the idea of a good fit. It tests whether the fit as built
helps, and it does not.

## Updated hypothesis

The stage-1 repr fit failed to learn (R^2 below 0 against a fixed anchor of 0), so the span code carried no
more PMI than the reconstruction code. The next experiment fixes stage 1 first: the gate is R^2 above 0 on
val with `ncm/pmi_var` logged, and only then repeat stage 2 with the same level-1 cost readout. If R^2 stays
at 0, the PMI target (local NLL minus full NLL, stop-gradient) is too noisy at this scale and the arm stops.

## What this changes

The v1 failure stands: the plan costs more than it saves at both levels 1 and 3, and K1-K6 shrinks to
+0.0014. Do not build more on RePR spans until stage 1 passes its fit gate.

## Artifacts

- `/home/wolfe/parcae-runs/ncm-repr-tok-5k/summary.json`, `/home/wolfe/parcae-runs/ncm-repr-plan-5k/summary.json`
- `/home/wolfe/morph-scratch/abq/full1015/chain.log` (lines 6 and 7)
- v1 reference: `lxtul/ncm:docs/experiments/failures/2026-10-14-parcae-span-ncm-5k.md`
