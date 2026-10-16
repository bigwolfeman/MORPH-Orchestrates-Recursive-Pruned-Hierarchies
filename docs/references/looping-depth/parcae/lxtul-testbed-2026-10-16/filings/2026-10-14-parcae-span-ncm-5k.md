# Parcae: span-level Next-Concept Modeling (a frozen multiscale code plan per span) at 5k

Status: failure (prereg written 2026-10-14 before any code or run; filed 2026-10-15)

## Question

NCM (arXiv 2610.14162, HazyResearch) generates text from a frozen multiscale residual VQ tokenizer: level 1
holds 1 code, each later level more codes, and one shared transformer predicts each level from the
reconstruction of the earlier ones. Wolfe (2026-10-14): "It is almost everything LXTUL is trying to be."

Our LCM reference failed in the way NCM is built to avoid: the continuous concept decoded from a
predicted concept reads CE 15.10 (MSE) and 13.49 (diffusion), above uniform (ln 49152 = 10.80), while
the same decoder on the true concept reads 0.157. A regressed concept leaves the decoder's manifold.

Two questions on the Parcae testbed, same data and eval rows as the 5k references:

1. Span plan on gpt8: a token model of the gpt8 + pointer shape that, for each span, also reads a plan
   (the span's first m code levels from a frozen tokenizer), with a looped generator that predicts those
   codes from the earlier text. Its likelihood bound (code NLL + token NLL given the codes) is a valid
   upper bound on the true NLL. Does it beat gpt8 + pointer, and does the looped generator use depth?
2. Pure span NCM: the same plan and generator with NO token memory across spans (the generator reads
   only earlier spans' codes, the decoder reads only the plan and its own span). Is this the LCM
   reference repaired, and how far is it from the token models?

Why the bound is the right metric: the codes are a deterministic function of the text, so
-log p(x) <= -log p(codes) - log p(x | codes) (NCM Appendix A). Every bit the coda gets from the plan is
paid for by the generator's code NLL. The per-pass loop has NO per-pass target: Parcae's Poisson depth
runs over the generator; each code level is a set of positions in one forward (teacher forced, block
diagonal by level), not a loop pass. The code NLL is a term of the likelihood, so the rule "only the
main CE buys depth" holds: the main objective here is the bound.

## Arms (branch lxtul/ncm; seed 1; 5000 steps; batch 6; the minimal arm's rows, one slot position per span)

| arm | what |
| --- | --- |
| span tokenizer | stage 1: span-local bidirectional encoder (2 blocks) to a 256-wide latent per token; residual VQ over levels of 1, 2, 4, 8, 16, 32 segment means (capped at the span length), 4096 codes per level, EMA codebooks, dead-code restart; nearest-upsample reconstruction; span-local bidirectional decoder (2 blocks); nested dropout 0.5 |
| span plan on gpt8 | stage 2: gpt8 + pointer token path; the coda input adds the plan of the label's span (m drawn per span from {0, 1, 2, 3} in training, zero-init projection); generator: 7 code positions per span (levels 1, 2, 4), read the plan-free prelude states of earlier spans, Parcae's core loop (Poisson depth as plain), per-level code heads; input-code corruption 0.5 / 0.25 |
| pure span NCM | stage 2: generator reads only earlier spans' codes (one summary position per span), 4 levels (15 code positions); the decoder is span-local and reads the plan at m = 4 |

## Readouts

- Tokenizer: reconstruction CE and exact token accuracy at each cutoff level; per-level codebook use and perplexity.
- Bound (ELBO-CE per token) per m and per generator depth 1, 2, 3, 4, 6, 8, from per-row sums in the
  standard `sweep.json` format, paired against gpt8 + pointer (2 seeds), plain + pointer and the minimal
  fan on the same tokens. The token NLL does not depend on the generator's depth (the coda reads the
  true codes), so K1-K6 moves only through the code NLL.
- Per-level code CE (anchor: ln 4096 = 8.32 per code is uniform).

## Predictions

1. Tokenizer: exact token accuracy with all 6 levels at least 99 % (70 %).
2. Span plan on gpt8, m = 0 (no plan): within +-0.03 of gpt8 + pointer's 3.6996 (65 %).
3. Span plan on gpt8: the best m beats its own m = 0 bound (50 %).
4. Span plan on gpt8: the best m beats gpt8 + pointer by at least 0.01 (25 %).
5. Span plan on gpt8: generator K1-K6 on the m = 3 bound at least +0.01 (45 %).
6. Every per-level code CE below 8.32 at the end of training (95 %).
7. Pure span NCM: bound at least 0.30 worse than gpt8 + pointer on the same tokens (70 %).
8. Pure span NCM: bound at least 5 nats below the continuous Base-LCM's ce_pred on the same scored set (85 %).
9. Every run finishes with finite losses (85 %).

## What each outcome changes

- 3 and 4 hold: the discrete plan carries more than it costs; span NCM becomes the TUL candidate and the
  next step is the speed account (think once per span, shallow coda) and a second seed.
- 3 fails with 2 holding: the plan's code bits cost more than they save at 5k; read the per-level
  code CE to see which level is too expensive, and test m = 1 only.
- 5 holds: a looped generator earns depth on the likelihood with no per-pass target; the first such
  reading on Parcae.
- 7 and 8 hold: discrete codes repair the LCM failure; the token memory remains what makes the token
  models win.

## Method

Code: `lxtul/ncm.py` on branch lxtul/ncm (from lxtul/lcm 7784701), built by a sonnet agent under the
builder rules; existing arm kinds untouched. GPU chain `morph-scratch/abq/ncm/gpu_todo.sh`, one job at a
time under the GPU lock, after Wolfe releases the GPU. Amendments get a date and a reason here.

Amendment 2026-10-14 (before any run): the NCM arms read the one-cell rows of minimal-5k-s2, which hold
494,924 scored tokens, not the 510,003 of the four-cell rows. Every absolute reference in the predictions
(gpt8 + pointer 3.6996, plain + pointer) is read from those runs RE-SCORED on the one-cell rows by
`paired_eval`; the minimal fan reads four-cell rows and is not paired. The token path matches gpt8 +
pointer exactly at m = 0 (value embeddings, tied head, same pointer; CPU test to 1e-9).

## Results

Sources: `morph-scratch/abq/ncm/gpu_todo.log`, the runs' `summary.json`, `eval/paired-plan.json`,
`eval/ncm-pure-5k/summary.json`. Seed 1, 5k. The chain's paired eval died twice (an argparse
error, then two reference runs with no `final.pt`: `plain-pointer-5k`, `plain-pointer-norm-5k`); it was
rerun by hand on 2026-10-15 with the references that still have checkpoints.

Span tokenizer (`ncm-tok-5k`), reconstruction per cutoff on its eval rows:

| levels kept | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| recon CE | 6.949 | 6.236 | 5.390 | 3.982 | 1.962 | 0.297 |
| exact token accuracy | 0.059 | 0.139 | 0.229 | 0.407 | 0.699 | 0.962 |

Level-1 codebook use 28 % (perplexity 247 of 4096), level 2 44 % (466), level 3 72 % (888).

Span plan on gpt8 (`ncm-plan-5k`), every token label of the packed rows (the paired set):

| m | token CE given the plan | code cost / token | bound @6 | K1-K6 |
| --- | --- | --- | --- | --- |
| 0 | 3.7443 | 0 | 3.7443 | 0 |
| 1 | 3.6262 | 0.226 | 3.8517 | +0.0003 |
| 2 | 3.4295 | 0.692 | 4.1219 | +0.0044 |
| 3 | 3.1069 | 1.675 | 4.7815 | +0.0076 [+0.0071, +0.0080] |

Code CE per code at depth 6: level 1 4.614, level 2 4.796, level 3 5.058 (uniform 8.32).
Same tokens (paired, 480 rows): gpt8 + pointer s1 3.7057, s2 3.6981 (mean 3.7019); plain + pointer s2
3.7273; minimal-5k-s2 3.7616. Plan at its default m = 3: +1.0757 [+1.0612, +1.0915] against gpt8 s1.

Pure span NCM (`ncm-pure-5k`), target tokens of complete spans with index >= 1 (480,118 tokens):

| m | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| bound @6 | 5.9723 | 5.6067 | 5.5248 | 5.6699 | 6.2612 |
| K1-K6 | 0 | +0.0005 | +0.0035 | +0.0072 | +0.0185 |

Same tokens: gpt8 + pointer s1 3.6920 (pure m = 4 is +2.569 [+2.527, +2.618]); Base-LCM ce_pred 14.912,
diffusion LCM 13.448.

Prediction check:
1. FAILS. Exact accuracy with all 6 levels 96.2 %, under 99 %.
2. FAILS. m = 0 3.7443 is +0.042 against gpt8 + pointer's 3.7019, outside +-0.03.
3. FAILS. The best m is m = 0; every plan raises the bound.
4. FAILS. No m beats gpt8 + pointer.
5. FAILS. K1-K6 on the m = 3 bound +0.0076, under +0.01.
6. HOLDS. Every per-level code CE (4.6 to 5.1) is below 8.32.
7. HOLDS. Pure is 2.57 nats worse than gpt8 + pointer on its tokens.
8. HOLDS. Pure m = 4 is 8.65 nats below Base-LCM and 7.19 below the diffusion LCM.
9. HOLDS. All three runs finished with finite losses.

## Verdict

Failure, 4 of 9 hold. The discrete plan carries real information (token CE falls 0.64 nats with three
levels) but costs more to predict than it saves: level 1 costs 0.23 nats per token and saves 0.12. With
exact models the bound equals the true NLL, so the gap is the generator predicting codes worse than the
token model already predicts the same information. Training with m drawn per span also costs the
plan-free path: m = 0 is 0.042 behind gpt8 + pointer. The looped generator earns depth only on the code
NLL (K1-K6 grows with the code count, 0.0076 at m = 3), with the whole bound getting worse.
The discrete codes do repair the LCM interface: the pure arm is 7 to 9 nats better than both continuous
LCMs on the same tokens. They do not replace the token memory.

## Updated hypothesis

A coarse plan read by a coda that sees the full token history has no headroom at 5k, as LCTUL-D's oracle
plan already showed (2026-09-16). The span-level depth seen in the generator is real but lives on an
added target. The next test puts the loop output into the token state with no added target
(SpiralFormer on Parcae, `2026-10-15-parcae-spiral-5k.md`).
