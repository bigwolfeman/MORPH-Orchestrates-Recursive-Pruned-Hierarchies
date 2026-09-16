# Experiment: TUL-Code conditioned thinker — CFG and predictability pressure on E

Status: failure
Date: 2026-09-15
Owner: Claude (session f9558148), for Wolfe ("Agreed. Let's do it.")
Design: `.agents/notes/proposed/architecture/2026-09-15-tul-code-conditioned-thinker.md`

## Question

The 20k panel's sampled code is an unconditional draw (residual 1.7–1.9× the code's
variance; worth the top 32 of 1024 code directions; flow loss at its floor by step 4000;
+0.63 nats one-draw against the strict ruler). Two independent causes: the target has
almost no predictable content (a lossless copy), and the sampler under-uses the
conditioning it has (band-0 flow ratio 0.48 against a 0.64 blind floor). Does either fix
make the sample carry the span?

## Hypothesis

H-B1 (sampler): classifier-free guidance (`code_cfg_drop` 0.1, `code_cfg_scale` 2.0) pulls
the draw toward the conditional mode: the residual falls, the k-curve keeps part of its
phase-2 dependence after the rollout, the gap narrows.
H-A1 (target): letting the flow loss reach E at `code_target_lambda` 0.1 moves z toward
what the past can predict (the plan); ce_tf rises, rank falls, the residual falls, the gap
narrows without collapse (`code_rank_abort` 4).

## Predictions (frozen)

Paired numbers on the shared tokens against `slot-spandec-strict-20k` at 20k; the sample
residual and k-curve from the Spark probes (96 rows, 8 draws); reference `tul-code-20k`:
residual 1.65–1.92 (head, rank 32–256), k16−k1 +0.014 at 20k / −1.29 at 10k, gap +0.626,
ce_tf 0.35 at the trained statistic, rate 30k / 24k tok/s (phase 2 / 3).

- P-1 (B1). `tul-code-cfg` sample residual over z variance at 20k ≤ 1.4 in the rank-128
  head (from 1.83–1.85). 55 %.
- P-2 (B1). `tul-code-cfg` marginal(k=16) − marginal(k=1) at 20k ≤ −0.10. 45 %.
- P-3 (B1). `tul-code-cfg` one-draw `ce_k1` − ruler at 20k ≤ +0.45, CI excluding +0.55.
  45 %.
- P-4 (B1). Guidance sweep on the 20k checkpoint (w ∈ {1, 1.5, 2, 3}): one-draw CE is
  not monotone in w — an interior optimum exists. 60 %.
- P-5 (A1). `tul-code-jepa` reaches 20k without the rank abort (val/code_eff_rank ≥ 4
  throughout). 60 %.
- P-6 (A1). `tul-code-jepa` ce_tf at 20k ≥ 1.0 (the code stops being a copy) AND its
  one-draw `ce_k1` − ruler ≤ +0.45. 40 %.
- P-7 (A1). `tul-code-jepa` sample residual at 20k ≤ 1.4 in the rank-128 head. 50 %.
- P-8. Both arms healthy (no tripwire); `tul-code-cfg` phase-3 rate ≥ 15k tok/s (two
  thinker passes per Euler step at rollout). 80 %.

## Binding

- P-1 and P-3 hold → the sampler was the lever; combine with A1 and k 32 next.
- P-6 and P-7 hold → the target was the lever; sweep λ (0.03 / 0.3).
- Neither residual moves (P-1 and P-7 fail) → the flow parameterisation itself is the
  question (discrete plan code, A2), not its conditioning.
- P-5 fails → λ 0.1 is too strong for the CE anchor; λ 0.03 before any other change.

## Method

Two 20k arms on the panel recipe (`tul_code` lineage: seq 1024, batch 6, warmup 1000,
phases at 2000 / 10000, `code_noise_renorm: true` on both), fresh runs from step 0, one
factor each against `tul-code-20k` plus the renorm (the renorm's own effect is the
`tul-code-renorm-r10k` arm, running first):

- `tul-code-cfg` (`morph/configs/tul_code_cfg.yaml`): `code_cfg_drop` 0.1,
  `code_cfg_scale` 2.0 (rollout and eval samples guided).
- `tul-code-jepa` (`morph/configs/tul_code_jepa.yaml`): `code_target_lambda` 0.1,
  `code_rank_abort` 4.

Runner sweeps at 5k/10k/15k/20k; Spark k-sweep at each checkpoint, subspace probe and
span samples at 20k; `paired_vs_ruler.py` against the ruler twin; for P-4 the k-sweep
re-run with `tul.code_cfg_scale` overrides on the 20k cfg checkpoint.

## Not verified before launch

- No GPU smoke beyond the runner's 12-step one; the guided rollout's memory (two thinker
  passes) is unmeasured at batch 6.
- One seed each.

## Results

Both arms ran to 20000, healthy (runner verdict HEALTHY, exit 0). Artifacts in
`lab/experiments/results/2026-09-15-tul-code-cond/`. Paired numbers on the shared 501k
tokens (490 blocks) against `slot-spandec-strict-20k` and `tul-code-20k`; Spark probes on
the same 96 rows as every other arm.

`tul-code-cfg` (B1: CFG drop 0.1, scale 2.0; wandb 6r187tjp; phase-3 rate 19.7k tok/s):

| reading at 20k | cfg | parent |
|---|---|---|
| sample residual, rank-128 head (cells 0 / 1) | 1.81 / 1.84 | 1.85 / 1.83 |
| sample residual, full rank | 1.89 / 1.91 | 1.92 / 1.91 |
| 8-draw marginal k = 1 → k = 16 | 4.380 → 4.375 (−0.005 [−0.007, −0.002]) | 4.357 → 4.371 (+0.014) |
| one draw vs ruler, k = 1 | +0.648 [+0.640, +0.655] | +0.626 |
| one draw vs parent, k = 1 | +0.022 [+0.019, +0.025] | — |
| guidance sweep, k = 1 marginal at w = 1 / 1.5 / 2 / 3 | 4.382 / 4.377 / 4.380 / 4.386 | — |

Through phase 2 (10k, trusting coda) guidance was worth 0.6 nats on a one-draw CE; after
rollout the discounting coda erased it. At 20k the sample is still the unconditional
draw, and the guidance weight moves the one-draw CE by under 0.01 nats across w ∈ [1, 3].

`tul-code-jepa` (A1: `code_target_lambda` 0.1, rank abort 4; wandb 6qz1gq7x, own run):

| reading at 20k | jepa | parent |
|---|---|---|
| val/code_eff_rank (min over the run) | 31 (16.3 at step 10500) | 38–48 |
| ce_tf (encoder code, trained statistic) | 0.324–0.33 | 0.35 |
| flow probe share (fresh pairs) | 0.274 (bands .421/.292/.199/.192) | 0.31 |
| sample residual, rank-128 head | 1.84 / 1.84 | 1.85 / 1.83 |
| sample residual, full rank | 1.91 / 1.91 | 1.92 / 1.91 |
| 8-draw marginal k = 1 → k = 16 | 4.390 → 4.373 (−0.017 [−0.021, −0.014]) | +0.014 |
| one draw vs ruler, k = 1 | +0.654 [+0.646, +0.662] | +0.626 |
| one draw vs parent, k = 1 | +0.010 [+0.008, +0.012] | — |
| plan-worth (zero / shuffle the sampled cells) | +0.034 / +0.001 | — |

The flow gradient did reach the encoder: the code rank slid from 70 to 19 through
phase 2 (rebounding to 31 under rollout), the flow probe share fell 0.04, and on the
phase-2 trusting coda a one-step sample read 2.5 nats better than the parent's (8.01 vs
10.52 at 10k, k = 1 marginal) with an INVERTED k-curve (+0.46: more Euler steps hurt on
this code). None of it survived rollout: the coda that learned to discount samples lands
on the parent's curve to the third decimal (sampled val 4.615 vs 4.617 at 15k), and the
sample residual is unchanged. The code never stopped being a copy (ce_tf 0.32).

Scores:

- P-1 (cfg residual ≤ 1.4): FAIL, 1.81–1.84.
- P-2 (cfg k-curve ≤ −0.10): FAIL, −0.005.
- P-3 (cfg gap ≤ +0.45): FAIL, +0.648.
- P-4 (w sweep not monotone): HOLDS at the noise floor — an interior minimum at w = 1.5,
  0.003–0.009 nats below the ends; nothing to act on.
- P-5 (jepa no rank abort): HOLDS, minimum 16.3.
- P-6 (jepa ce_tf ≥ 1.0 AND gap ≤ +0.45): FAIL on both halves, 0.32 and +0.654.
- P-7 (jepa residual ≤ 1.4): FAIL, 1.84.
- P-8 (healthy, cfg rate ≥ 15k): HOLDS, 19.7k.

## Verdict

Status: failure. The two deciding predictions on each arm (P-1/P-3, P-6/P-7) failed.
Binding case "neither residual moves": the flow parameterisation itself is the question,
not its conditioning. Guidance sharpens a sample the trusting coda likes and the
discounting coda ignores; predictability pressure makes the code lower-rank and easier
to fit without making it anything but a copy, and the copy's unpredictable part is still
the whole gap. Together with the thinker-only run (40k steps on a fixed target, floor by
10k), the sampler's conditioning, the target's motion, training time and the target's
rank are all measured levers that do not move the sample off an unconditional draw.

## Updated hypothesis

The bound is what the code IS: a verbatim copy of the next span, whose entropy given the
past is most of its content. Two ways remain and both change the object rather than the
sampler: (1) Explorative Modeling (arXiv 2607.27372) on the training rule, running now
as `tul-code-xm` / `tul-code-xmc` (`planned/2026-09-15-tul-code-xm.md`), which can make
the field commit to a mode and the coda read the right parts of a sample but cannot add
information; (2) a code with less capacity by construction (fewer floats, a noise floor
the span cannot fit under, or a discrete plan code, A2), so the encoder is forced to keep
the predictable part and the coda does the language work. Not verified: a λ sweep
(0.03, 0.3) on the jepa arm, and a second seed of either arm.
