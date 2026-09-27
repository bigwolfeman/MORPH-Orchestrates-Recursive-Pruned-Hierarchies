# Agent Note: LX hard credit, a committing credit over the code rollouts

Status: proposed

## Problem

LX (`tul.code_enum_k: 4`,
[2026-09-23-provable-loop-contribution.md](2026-09-23-provable-loop-contribution.md)) trains
the exact per-span mixture `L = -log (1/K) sum_k exp(-CE_k(span))`. Its gradient gives
rollout k the posterior weight `w_k = softmax_k(-CE_k)`. The codes are small (0.1 rms) and
fixed. So the CEs start nearly equal, `w` is close to `1/K`, and every rollout gets the
same gradient. Nothing breaks that symmetry. We measured the rollouts to be an ensemble:
span-level selection over the fixed codes has about 0.0006 nats to buy
([2026-09-26-lx-selection-ceiling.md](../../../../lab/experiments/failures/2026-09-26-lx-selection-ceiling.md)).

The question: does a credit that COMMITS to one rollout per span make the rollouts
specialise?

## Proposal

A training-only knob, `tul.code_enum_credit: soft | hard` (default `soft`, the forward
from before the key), with `tul.code_enum_hard_eps` (default 0.05).

`hard` is Multiple Choice Learning with a relaxed winner. Per span g, the token loss is
`sum_k c_k(g) CE_k(g)`. Here `c` is a stop-graded relaxed one-hot: `1 - eps` on
`argmin_k CE_k(g)` (on a tie, the lowest k) and `eps / (K - 1)` on each other rollout.
`CE_k(g)` is the same per-span summed CE the mixture uses (`-S_k(g)` of `iw_span_bound`),
over the same scored tokens. No pass is added. The math is one function,
`rollout_mixture.hard_credit_span_sum`; `_enum_mix_losses` calls it at train only.

What each reader sees:

- **Eval.** Unchanged. The val forward, the deploy (Bayes) read and every val metric are
  the mixture. The branch is `self.training`-gated, so an eval forward cannot reach it.
- **train/loss.** Stays the mixture NLL, so the arm is comparable with fp01.
  `enum_ce_mix` is still the mixture. The objective is `tul/enum_hard_obj`. Its excess
  over the mixture is `enum_hard_weighted`, which train.py subtracts in both of its
  `*_weighted` lists. The train-side `ce_main` is "the loss before the core-loop terms"
  (`_apply_core_aux`). It already carries every `groups` fold (the gain hinge, the probe
  head), and it carries this one too.
- **Win statistics.** `tul/enum_code_win{k}` already was the fraction of spans rollout k
  explains best. It is the same argmax the hard credit uses, so no second key was added.
  `tul/enum_win_entropy` is new, and it is logged on every LX model: the entropy of those
  fractions over `log K`. At 1 every rollout wins equally often. At 0 one rollout wins
  every span.
- **The parallel probe head.** fp01's head is detached. Its mixture term trains the head
  alone and does not read the token credit, and a test pins its gradients as identical
  under soft and hard. A LIVE head (`spandec_parallel_detach: false`) is refused with
  `hard`: it would send the loop a posterior credit over the same rollouts beside the
  hard one.
- **The fixed-point term and the gain hinges.** They act on the loop trajectory, not on
  the token credit. Their forward values are identical between soft and hard (tested).
  Their gradients reach the same loop that the token loss reaches, so the loop's total
  gradient does change. That is the arm.

The arm is `morph/configs/tul_slot_spandec_strict_e4probe_fp01_hard.yaml`, wandb
`lxtul-e4probe-fp01-hard`: fp01 plus the two keys, 5000 steps.

## Alternatives considered

- **Pure winner-take-all (`eps = 0`).** Allowed by the knob and not the default. A rollout
  that never wins gets no gradient, and so it can never start to win (the fan-select
  arm's reason for `fan_select_eps`). eps 0.05 matches that arm's value.
- **Sharpened posterior (a temperature on the mixture).** This keeps the credit smooth.
  At the symmetric start it is still close to `1/K` for every temperature that leaves the
  loss finite. The argmin breaks the tie at step 0, and a temperature does not.
- **Hard credit on the parallel head as well.** Not needed for fp01, because the head is
  a detached probe. A live head is refused rather than given a second policy.
- **A separate `enum_win_frac_k{i}` key.** Rejected: it would be the same number as
  `enum_code_win{k}` under a second name.

## Acceptance criteria

- CPU, done in this change: `tests/test_tul_lx_credit.py` (the hard-credit half).
  - Soft credit gives the e951498 pins at dropout 0 and 0.1. `pin_compare.py` also finds
    every full tensor `torch.equal` to the pre-change tree.
  - The relaxed one-hot, the tie rule and the gradient into S match a hand-worked case.
  - On the model, the gradient into rollout k's coda output at span g is `c_k(g)` times
    that rollout's plain CE gradient. `c` is built from an independent per-span table.
  - Soft and hard models with the same weights give the same eval logits and eval keys.
  - The train forward is the same forward, and loss minus `enum_hard_weighted` equals
    the soft loss.
  - The win statistics match hand counts.
  - The refusals hold, and the config composes, builds and trains a step.
  - Each key test was sabotaged once and failed. The log is
    `/home/wolfe/morph-scratch/credit/sabotage_cb.log`.
- GPU (NOT run): the memory trace and 30-step compiled smoke in
  `/home/wolfe/morph-scratch/credit/mem_trace.sh`. Then a prereg (the orchestrator's) before
  the 5k run.

## Risks

- **Collapse onto one rollout.** The winner's gradient is 57 times a loser's at eps 0.05,
  K 4, so one rollout can take every span. Read `tul/enum_win_entropy` and
  `tul/enum_code_win{k}`. If one rollout wins every span, the arm is a one-code model
  that pays for K.
- **The objective is not the scored loss.** The run optimises `enum_hard_obj`, and it is
  scored on the mixture. The mixture NLL can rise while the objective falls. The
  difference is `tul/enum_hard_weighted`.
- **The logged `ce_main` moves.** The train-side `ce_main` now includes the hard excess.
  Compare arms on `train/loss` or `tul/enum_ce_mix`, never on the train-side `ce_main`.
