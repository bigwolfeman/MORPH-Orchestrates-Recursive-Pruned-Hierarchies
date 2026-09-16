# Experiment: TUL-Code with Explorative Modeling — K samples per slot, train on the nearest

Status: failure
Date: 2026-09-15
Owner: Claude (session f9558148), for Wolfe ("Have we run the xm like arm? I think it
solves training for us.")

## Question

Explorative Modeling (Gladstone, Ji, Du, arXiv 2607.27372, Forward XM): draw K candidate
generations, keep the one nearest the data, apply the standard loss to that one. A
regression toward the data averages the modes into the mean; a min over K commits to a
mode. On TUL-Code the thinker's sample has read as an unconditional draw on every arm
(residual 1.7–1.9× the code's variance), and the coda in phase 3 trains on whichever draw
came out, so it learns to discount samples. Does XM on the thinker (the flow pair takes
the winning seed) and on the coda (it reads the winning sample) move the sample toward
the truth, and does the coda then read it?

Two arms from scratch, the panel settings (seq 1024, batch 6, 20k steps, phases at 2k /
10k): `tul-code-xm` (selection by squared error to E's code, the paper's rule) and
`tul-code-xmc` (selection by the coda's CE on the true next span, K extra no-grad coda
passes). K = 4. Truth cell renormed. Parent for pairing: `tul-code-20k`; ruler:
`slot-spandec-strict-20k`.

## Hypothesis

H-X1: the min-over-K coupling lets the field commit; the sample's residual falls below
the joint runs' floor, the k-curve regains a slope after rollout, and the paired one-draw
gap against the ruler closes part of the way.
H-X0 (null): K candidates are diverse but none is nearer the truth than chance; the
selected seed is a lucky draw, the field learns nothing it could not before, and the
residual and the gap stay where the panel left them. The target bounds the guess.

## Predictions (frozen)

Reference at 20k (one seed each): parent residual (rank-128 head) 1.83–1.85, full 1.92;
rollout1 (E frozen) 1.62 / 1.65; k-curve after rollout 0.00 / +0.01; paired one-draw gap
vs the strict ruler +0.63 (parent), +0.65 (cfg); flow probe share 0.31.

- P-X1. Flow probe share (fresh random pairs, `code_flow_probe.py`) at 20k ≤ 0.28 on the
  l2 arm. 50 %.
- P-X2. Sample residual over code variance in the rank-128 head at 20k ≤ 1.60 on at
  least one arm. 45 %.
- P-X3. Paired one-draw gap against the strict ruler at 20k ≤ +0.45 on at least one arm
  (from +0.63). 40 %.
- P-X4. k-curve (8-draw marginal, k = 16 − k = 1) at 20k ≤ −0.10 on at least one arm. 35 %.
- P-X5. The coda-selected arm beats the l2 arm on the paired one-draw CE at 20k by ≥ 0.05
  (selection in the space the reader cares about). 50 %.
- P-X6. The candidates spread: `train/code_xm_score_best` / `train/code_xm_score_mean`
  ≤ 0.90 by step 5000 on both arms, and the ratio does not rise afterwards. 70 %.
- P-X7. Rate: l2 arm ≥ 28k tok/s, coda arm ≥ 22k tok/s (parent 34k); peak ≤ 28 GB. 70 %.

## Binding

- P-X2 and P-X3 hold → the sampler commits and the coda reads it: K sweep (2, 8), then
  XM with the FPF-style rollout from partially-noised truth.
- P-X6 holds and P-X2 fails → the candidates are diverse and none is nearer: the target
  bounds the guess (H-X0, the thinker-only verdict again); the code must change.
- P-X5 holds and P-X3 fails → the coda-space selection helps the reader without closing
  the gap: selection is a reader trick, not a thinker one.

## Method

`morph/configs/tul_code_xm.yaml` (= `tul_code` + `code_noise_renorm: true`,
`code_xm_k: 4`, `code_xm_select: l2`); the coda arm overrides `tul.code_xm_select=coda`
and `wandb.name=tul-code-xmc`. Mechanism (`docs/tul-code-spec.md` §6 step 3): from phase 2,
each step draws K samples per valid slot (`code_rollout_steps` = 8 Euler steps each, no
grad, K = 4 sampler runs); score by squared error of the rms-normed sample to E's code
(l2) or by the coda's summed CE on the true next span with that sample in the cells
(coda; dropout off, the training call's injections and TG restriction); the flow loss
runs on the pair (winner's z_0, E's code); in phase 3 the winner is the sample the coda
reads at the rollout slots (`code_rollout_p` 0.5). Runner queue after `tul-code-jepa`:
`training.steps=20000 training.ademamix_t_beta3=20000 training.ckpt_every=5000`;
12-step runner smoke as the gate; Spark watcher: marginal sweep at 5k/10k/15k/20k,
subspace, samples and flow probe at 20k; paired scoring vs the ruler and the parent at
20k (`lab/divergence/paired_vs_ruler.py`).

## Not verified before launch

- The XM path under torch.compile on the real model: the CPU tests cover the toy model
  (48 pass in `tests/test_tul_code.py` + setup keys + checkpoint compat); a 12-step Spark
  smoke of both arms runs before the queue entry, the runner's smoke is the gate.
- Memory of the coda selection (K no-grad coda passes with a per-row [L, V] logits
  chunk): estimated under 1 GB extra, not measured.
- The l2 selection compares the rms-normed sample to E's unit-RMS code; the paper's
  squared error is on its raw generation. Same argmin up to the norm.

## Results

The l2 arm (`tul-code-xm`, wandb nbzufde8, dac9fe8, 20k steps, one seed). Artifacts:
`lab/experiments/results/2026-09-15-tul-code-xm/` (flow probe, subspace probe, 8-draw
marginal at 5k–20k, span samples, the 20k depth sweep, paired scoring, the wandb series).
The coda arm (`tul-code-xmc`) is queued behind the 50k continuations; its rows are
appended when it lands (P-X5 stays open until then).

| prediction | bar | l2 arm at 20k | parent | verdict |
|---|---|---|---|---|
| P-X1 flow probe share (fresh pairs, 96 rows × 2) | ≤ 0.28 | 0.304 ± 0.001 (bands 0.449 / 0.327 / 0.225 / 0.212) | 0.31 | fail |
| P-X2 sample residual / code variance, rank-128 head | ≤ 1.60 | 1.88 (cell 0), 1.85 (cell 1); full 1.94 / 1.93 | 1.83–1.85 / 1.92 | fail |
| P-X3 paired one-draw gap vs strict ruler (501k tokens) | ≤ +0.45 | +0.627 [+0.619, +0.634] at k = 1 | +0.63 | fail |
| P-X4 8-draw marginal k = 16 − k = 1 | ≤ −0.10 | +0.016 [+0.014, +0.019] (k1 4.355, k16 4.371) | jepa −0.017 | fail |
| P-X5 coda arm beats l2 arm by ≥ 0.05 | | pending (xmc) | | open |
| P-X6 selection ratio ≤ 0.90 by 5k, not rising | | first ≤ 0.90 at step 2820, 0.91 over 3k–10k, 0.86–0.89 in phase 3 | | fail (narrow) |
| P-X7 rate ≥ 28k tok/s, peak ≤ 28 GB | | 15.2k tok/s in phase 3 (22.5k in phase 1); peak 15.8 GB reserved | 24.0k in phase 3 | fail |

Other readings. XM vs the parent on the same 501k tokens: −0.017 at k = 1 against the
parent's k = 6 (the parent's own k = 1 is 4.438 vs XM's 4.439). One-draw depth curve
k1 → k16: 4.439 → 4.479 (parent 4.438 → 4.459); every extra Euler step costs. wandb flow
share over 15k–20k: 0.312 vs the parent's 0.309. The oracle line in the span samples
returns the true span word for word (the code is still a copy); GREEDY and SAMPLE lines
are newswire-shaped sentences with no thread to the context, as on every arm.

The rate prediction was wrong by construction: each candidate is a full
`code_rollout_steps` = 8 Euler generation, so K = 4 adds 32 core passes per step, not
"K samples". The paper's own Diffusion/Flow hybrid costs K passes; that arm is
`tul-code-xmn` (`planned/2026-09-16-tul-code-xmn.md`).

## Verdict

Failure; H-X0 on the l2 arm. The best-of-4 selection ratio sits at the order statistic
of interchangeable draws (0.91 in phase 2), the flow share, the residual, the paired gap
and the k-curve all read as the parent's, and the selected seed teaches the field nothing
it did not learn from a random one. The binding line "P-X6 holds and P-X2 fails" needs
both arms; the coda arm is still queued, so the note stays `proposed/` until it lands.

## Updated hypothesis

Best-of-K over complete generations cannot help when every generation is an
unconditional draw from the same distribution: the nearest of four such draws is nearer
by the order statistic and carries no information about the past. Selection can only
sharpen a field that already curves toward the target; ours is straight (a flat Euler
k-curve is a straight mean path). The two open XM readings are the coda-space selection
(`tul-code-xmc`) and the paper's coupling search over the corruption noise
(`tul-code-xmn`), both at K = 4. If both read flat, XM is closed at this K and the
target definition remains the lever.
