# Experiment: LCTUL with the noise-search form of Explorative Modeling (the paper's Diffusion/Flow hybrid)

Status: planned
Date: 2026-09-16
Owner: Claude (session f9558148), for Wolfe ("what kind of xm is this? They have variants
in the paper" / "add that to the queue too")

## Question

`tul-code-xm` runs Forward XM in its Algorithm 1 form: K complete 8-step generations per
slot, the endpoint nearest E's code wins, the flow loss trains on that seed's straight-line
pair at a fresh t. The paper does not run that form on its Diffusion/Flow models. Its
hybrid (arXiv 2607.27372 §4.1, App. C) keeps the data sample, the timestep and the
condition fixed, draws K corruption noises, makes one velocity prediction per candidate,
and trains the pair with the lowest flow loss. The selection happens in the space of the
trained loss, and it costs K thinker passes instead of K × 8. Does the coupling search the
paper actually validated move the thinker where the end-to-end search did not?

One arm from scratch, the panel settings (seq 1024, batch 6, 20k steps, phases at 2k /
10k): `tul-code-xmn` (`morph/configs/tul_code_xmn.yaml`: `code_xm_mode: noise`, K = 4,
truth cell renormed). Parent for pairing: `tul-code-20k`; ruler:
`slot-spandec-strict-20k`; sibling: `tul-code-xm` (the sample form, same K).

## Hypothesis

H-N1: selecting the corruption by the flow loss straightens the coupling the field has to
learn (the paper's reading: the model fits the mixture its explored candidates form), so
the flow share falls below the joint-run floor and the sample residual moves.
H-N0 (null): at K = 4 the K corruptions of a 2048-float code differ by noise the field
cannot use; the selected pair's loss is lower by the order statistic of four draws and
nothing else, the field learns what the parent learned, the residual and the paired gap
stay where the panel left them.

## Predictions (frozen)

Reference at 20k (one seed each): parent flow share (probe) 0.31, wandb `code_fm_rel`
0.309 over 15k–20k; sample residual (rank-128 head) 1.83–1.85, full 1.92; k-curve after
rollout 0.00 / +0.01; paired one-draw gap vs the strict ruler +0.63; rate in phase 3
24.0k tok/s (the sample-form XM arm: 15.2k).

- P-N1. Flow probe share (`code_flow_probe.py`, fresh pairs) at 20k ≤ 0.28. 40 %.
- P-N2. Sample residual over code variance in the rank-128 head at 20k ≤ 1.60. 35 %.
- P-N3. Paired one-draw gap against the strict ruler at 20k ≤ +0.45. 30 %.
- P-N4. k-curve (8-draw marginal, k = 16 − k = 1) at 20k ≤ −0.10. 30 %.
- P-N5. Selection ratio `code_xm_score_best / code_xm_score_mean` ≤ 0.85 by step 5000
  and not rising afterwards (the order statistic of four χ²-like draws at this dimension
  sits near 0.9; a ratio well below it says the candidates differ in something the
  field sees). 50 %.
- P-N6. Rate ≥ 20k tok/s in phase 3 (K = 4 extra thinker passes on a step whose parent
  cost is dominated by the prelude and coda); peak ≤ 20 GB. 75 %.
- P-N7. On the wandb flow share over 15k–20k, the arm sits ≥ 0.02 below the parent
  (0.309) — the selected pairs are easier by construction, so this is the order
  statistic and NOT evidence on its own; the probe (P-N1, fresh pairs) is the reading.
  85 %.

## Binding

- P-N1 and P-N2 hold → the coupling search moves the field: K sweep (2, 8) and the
  paper's smooth form (log-mean-exp over the candidates) next.
- P-N7 holds and P-N1 fails → the arm only reports easier pairs; the training loss is
  lower and the field is the same (H-N0). The sample form and the noise form are then
  both closed at K = 4, and the remaining XM lever is Reverse XM, which needs a set of
  target codes per past that we do not have.
- P-N5 fails (ratio stays near 0.9) → the four corruptions are interchangeable to the
  field; a larger K is the only way to test the mechanism, at K thinker passes each.

## Method

`morph/configs/tul_code_xmn.yaml`. Mechanism (`docs/tul-code-spec.md` §6, "the noise-search
form"): from phase 2, the flow block draws t and the condition once per slot, then K = 4
z_0 draws for the same target; each candidate's flow loss is one no-grad thinker pass on
its (z_t, v_target) pair; the argmin pair is re-run with grad (the memory-saving mode of
App. C). The phase-3 rollout is a fresh 8-step sample as on the parent. `_code_last_passes`
= 1 + K + rollout. Runner queue after `tul-code-xmc`; 12-step runner smoke as the gate;
Spark watcher: marginal sweep at 5k/10k/15k/20k, subspace, samples and flow probe at 20k;
paired scoring vs the ruler and the parent at 20k (`lab/divergence/paired_vs_ruler.py`).

Tests: `tests/test_tul_code.py` — the K no-grad passes share one t and the grad pass runs
on the selected pair (`test_xm_noise_search_trains_the_lowest_flow_loss_pair_with_no_generation`),
the argmin rule pinned with a zero thinker (`test_xm_noise_search_selects_the_argmin_of_the_flow_loss`),
refusals (`code_xm_mode` outside {sample, noise}; noise with coda selection). 53 pass
with setup keys and checkpoint compat.

## Not verified before launch

- The noise search under torch.compile on the real model: the CPU tests cover the toy
  model; a 12-step Spark smoke runs before the queue entry, the runner's smoke is the gate.
- The order-statistic baseline for P-N5 (0.9) is an estimate from the sample-form arm's
  phase-2 reading (0.91), not a computation for this criterion.

## Results

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
