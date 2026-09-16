# Agent Note: TUL-Code trains on the best of K thinker samples (Explorative Modeling)

Status: proposed

Date: 2026-09-15. Prereg: `lab/experiments/planned/2026-09-15-tul-code-xm.md`. Follows
[`2026-09-15-tul-code-conditioned-thinker.md`](2026-09-15-tul-code-conditioned-thinker.md)
(which listed XM as a second-wave follow-on) and the thinker-only result
(`lab/experiments/failures/2026-09-15-tul-code-thinker-only.md`). Supersession check: no
active note makes a claim about the flow pair's coupling or the phase-3 sample choice;
nothing archived.

## Problem

Every TUL-Code arm reads the thinker's sample as an unconditional draw (residual 1.7–1.9×
the code's variance; word salad on a trusting coda), and the thinker-only arm showed that
forty thousand extra steps on a fixed target do not move it. Two averaging mechanisms sit
in the training rule itself. The flow loss pairs a random noise with the code, so the
velocity target at any point is an average over every noise that could have been drawn:
the field learns the mean path, and a mean path lands on the mean code. And in phase 3
the coda trains on whichever single draw came out, so for most slots it is taught to
decode a wrong sentence into the right one, which teaches it to ignore the cells.

## Proposal

Explorative Modeling (Gladstone, Ji, Du, arXiv 2607.27372, Forward XM), ported as
`tul.code_xm_k` and `tul.code_xm_select`. Each step draws K samples per slot from the
thinker (no grad), keeps the one nearest the data, and applies the standard losses to it:
the flow loss on the pair (the winner's seed, E's code), which is the paper's rule, and
in phase 3 the coda reads the winner rather than a fresh draw. The selection criterion is
either the paper's squared error to the code (`l2`, K sampler runs, no extra coda work)
or the coda's CE on the true next span (`coda`, K extra no-grad coda passes with the
training call's geometry and dropout off). Inference is unchanged: one sample.

## Alternatives considered

- **Minibatch-OT noise coupling** (B2 of the conditioned-thinker note): the same
  straightening at the noise end without a model call. Not chosen first because XM
  selects by the model's own generation, so the coupling follows what the field can
  reach, and because it also fixes the coda's phase-3 distribution in the same move.
- **Best-of-K on the coda only** (keep the flow loss as is): the earlier framing. It
  sharpens the coda's training distribution but leaves the field's marginal untouched;
  the paper's rule trains the seed, so the port carries both.
- **Selection by a learned critic** instead of the data: no data-free selector is
  defined here, and the coda's CE is the critic the design already has.
- **K at inference (best-of-K decoding)**: not a training fix and needs a selector with
  no truth to compare to; out of scope.

## Acceptance criteria

The prereg's P-X2 (residual in the rank-128 head ≤ 1.60 at 20k) and P-X3 (paired one-draw
gap vs the strict ruler ≤ +0.45) on one seed; the note moves to `implemented/` if both
hold and to `rejected/` if P-X6 holds while P-X2 fails on both arms.

## Risks

- The min over K is a biased estimate of the flow loss; the checkpoint flow probe (fresh
  pairs) is the honest number, and the training series will read lower than it.
- With `l2` the "data" is E's code, a verbatim copy of the span; the nearest candidate
  to a copy may still be far from it, and the port cannot create information the past
  does not carry (the thinker-only verdict). H-X0 is the expected null.
- One seed per arm; the panel's runs decorrelate in tens of steps.

## Addendum 2026-09-16: the paper's hybrid form as a second arm

What `tul-code-xm` runs is Algorithm 1 in its end-to-end form: K full generations, the
endpoint nearest the code wins, the flow loss trains the winner's straight-line pair. The
paper's Diffusion/Flow experiments (§4.1, App. C) use a different search: same data sample,
timestep and condition, K corruption noises, one velocity prediction each, the pair with
the lowest flow loss trains. The paper calls it a coupling search over the noise, not a
best-of-K against a fixed target. Wolfe's call: queue it too. Shipped as
`tul.code_xm_mode: noise` (`morph/configs/tul_code_xmn.yaml`), prereg
`lab/experiments/planned/2026-09-16-tul-code-xmn.md`. Cost K thinker passes per step
instead of K × 8; the selection is in the space of the trained loss. Reverse XM (fix the
generation, search K data targets) is recorded and not built: it needs a set of valid
target codes per past, which a continuous condition does not give.

Live reading of the sample form at step 18k (wandb): flow share equal to the parent
(0.312 vs 0.309), selection ratio 0.91 in phase 2 and 0.86–0.89 in phase 3, rate 15.2k
tok/s against the parent's 24.0k. P-X6 and P-X7 of its prereg fail; P-X1–P-X4 wait on
the 20k probes.

## Measured 2026-09-16 (l2 arm at 20k, one seed)

Filed in `lab/experiments/failures/2026-09-15-tul-code-xm.md`. Flow probe 0.304 (parent
0.31), residual 1.85–1.88 in the rank-128 head (bar 1.60), paired gap +0.627 vs the ruler
(parent +0.63), 8-draw marginal k16 − k1 +0.016, selection ratio 0.91 in phase 2 and
0.86–0.89 in phase 3, rate 15.2k tok/s. P-X1–P-X4, P-X6 and P-X7 fail. The coda arm
(P-X5) and the noise-search arm are queued; the note stays proposed until both land.

