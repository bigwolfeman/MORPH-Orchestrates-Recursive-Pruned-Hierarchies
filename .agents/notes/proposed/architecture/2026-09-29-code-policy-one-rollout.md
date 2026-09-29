# Agent Note: The code policy, one rollout that explores across steps (arm B)

Status: proposed

## Problem

LX (`tul.code_enum_k: 4`, `morph/model/tul_code_enum.py`) runs K = 4 rollouts of the slot
loop and the coda per row. Rollout k re-adds a learned code `u_k` at the end of every
slot-loop pass (`h <- f(h) + 0.1 * rms(f(h)).detach() * u_k`, the `u_k` a learned regular
simplex), and the coda's loss is the exact per-span mixture over the K rollouts. Two
readings make that design expensive for what it buys:

- Cost. The LX K = 4 arms ran 5538-5723 tok/s at 5k steps against 9355-10089 tok/s for the
  plain arms (the orchestrator's reading of the 5k runs, 2026-09-29). The arm pays about K
  codas per row.
- The gain is mostly an ensemble. The codes are fixed, so a code cannot specialise per
  span: the same code wins the same span at depth 1 and at depth 6 about 80 % of the time,
  and a per-span selector's ceiling over the rollouts is small
  ([lab/experiments/failures/2026-09-26-lx-selection-ceiling.md](../../../../lab/experiments/failures/2026-09-26-lx-selection-ceiling.md)).

The open question is whether a code that is CHOSEN per slot from the slot's own content
can earn what a fixed code cannot, at about the cost of one rollout.

## Proposal

`tul.code_policy_k = C` (default 0, off; `morph/model/tul_code_policy.py`). Keep LX's
codes and run ONE rollout:

- The codes are `TULCodeEnum` with C vertices: the same basis, the same simplex and the
  same per-pass rule, added at the same line of `_tul_core`, indexed per slot
  (`TULCodeEnum.term_per_slot`). Pads get nothing. The code size moved into
  `TULCodeEnum._size`, the one home for both indexings.
- A linear policy head `d_model -> C` reads the slot's loop-entry state. That state is `e`,
  the prelude's output gathered at the slot after `input_norm`, which `core_init` turns
  into the first carrier. The head reads it DETACHED, averaged over the Hyper-Connection
  streams and scaled to unit L2 norm. A value head `d_model -> 1` reads the same input.
  Both heads start at zero, so the policy starts uniform.
- At train, one code per valid slot is sampled with the global RNG stream. At eval,
  `tul.code_policy_eval` decides: `argmax` (default) or `sample`. The eval draws use a
  private generator.
- The reward of slot s is minus the mean per-token CE of bag s+1, the span that slot's
  cell feeds (`span_ce_index`'s pairing). The CE is the model's own per-token CE with the
  slot id masked. It is computed under `no_grad` from the detached coda state with one
  forward-only head pass. Slots whose span has no scored token do not count.
- The baseline is `b_s = rbar_{-s} + v_s`: the leave-one-out batch mean reward plus the
  value head's prediction of the residual. A head that predicts the raw reward (-4 to -7
  nats) would start several nats off. Its bias gradient (about 10 at step 0) would dominate
  the global gradient clip of 1.0 and throttle every other update. At lr 1e-4 it would also
  need tens of thousands of steps to reach the reward's level.
- The objective is `lambda * mean(-A log pi(c)) - entropy * mean(H(pi)) + value_lambda *
  mean((v - (r - rbar_{-s}))^2)`, with `A = r - b` detached. It folds as ONE
  `code_policy_weighted` term at train only. train.py subtracts it from train/loss and
  from the val loss. REINFORCE reaches only the two heads. The loop and the codes learn
  from the ordinary token CE through the one coda pass.
- The decisive instrument is `val/code_policy_vs_random`, a val-only extra pass (train.py
  `evaluate`). It is the token CE with a uniformly random code per slot minus the token CE
  with the policy's argmax, in nats per token. Eval depth and eval dropout are
  deterministic, so the two passes differ in the codes alone.

Configs (5000 steps, paired with the 5k `lxtul-e4probe-fp01` run):
`tul_slot_spandec_strict_e1probe_fp01.yaml` is the K = 1 control (fp01 with
`code_enum_k: 1`, the key's off value). `tul_slot_spandec_strict_e1probe_fp01_policy4.yaml`
is that control plus `code_policy_k: 4`.

## Alternatives considered

- **K parallel rollouts (LX itself).** This is the design being replaced. Its gain was
  measured as an ensemble and it costs about K codas.
- **VIMCO.** A leave-one-out multi-sample baseline has lower variance than REINFORCE, but it
  needs several samples of the choice per slot, which means several rollouts and brings
  back the cost this arm exists to remove.
- **Gumbel-softmax, straight-through.** A relaxed code is a convex mix of codes in the
  forward (or a hard pick with a relaxed backward), so the loop trains on blurred means of
  hypotheses. The blurred mean has lost every time it was measured on this tree: the soft
  LX fans ([lab/experiments/failures/2026-09-27-lx-soft-fan-retry.md](../../../../lab/experiments/failures/2026-09-27-lx-soft-fan-retry.md))
  and four copies of one cell ([docs/slot-cells-distinct-vs-blurred.md](../../../../docs/slot-cells-distinct-vs-blurred.md)).
- **A learned-sigma stochastic latent.** This is refused on this tree: learned noise dies
  under the likelihood (project memory "no learned-sigma arms", and
  [lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md](../../../../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md)).
- **A value head that predicts the raw reward (`b_s = v_s`).** This is what the brief
  specified. It lost to the residual head because of the global-clip and lr arithmetic in
  the Proposal.

## Acceptance criteria

- `val/code_policy_vs_random` is above 0 at 5000 steps, by more than its spread across the
  val batches. At about 0 the policy picks noise and the arm is the control plus a code.
- The forced-depth K-curve and `val/loss` are read against BOTH controls on the same
  tokens: `lxtul-e1probe-fp01` (K = 1, no code) and `lxtul-e4probe-fp01` (LX K = 4).
- tok/s stays within about 10 % of the K = 1 control.
- `tul/code_policy_entropy` does not reach 0 with one `tul/code_policy_share_k{i}` at 1
  (collapse).

## Risks

- **Reward coupling.** Under the strict geometry a later span reads earlier slots' cells,
  so slot s's code also moves the CE of spans s+2 and later. Its per-span reward does not
  credit that, and the leave-one-out baseline depends on `c_s` through those spans at
  O(1/N).
- **Policy collapse to one code.** The entropy bonus (0.01) is the only force against it.
- **Train/deploy gap.** Train samples and eval takes the argmax.
  `tul.code_policy_eval: sample` measures the gap (`val/code_policy_sample_minus_argmax`).
- **Blind choice.** The policy reads the slot before any pass, so it chooses from the
  prelude's summary only.
- **REINFORCE variance.** A batch has hundreds of slots and one sample each.
- **RNG.** The train draw moves the global RNG stream, so the arm's later random draws are
  not its control's. The two runs are not bit-comparable after step 0 (they never were
  bit-comparable in their losses).
- **Extra cost.** One forward-only head pass per step for the reward, and one extra coda
  pass per val batch for the instrument.
