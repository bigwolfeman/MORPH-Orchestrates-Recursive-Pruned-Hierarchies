# Agent Note: Fixed-Point Forcing — train the refinement on its own rollouts, not on one pass

Status: proposed

## Problem

Every "iterate a one-pass-trained map at inference" arm in MORPH's record is depth-dead
or depth-inverted: the dbfix Euler ladder went K=1 4.4652 → K=8 4.5101, the db_cond
curve was flat to 0.005 nats, and the dmorph v1 tok arm's ladder head reads WORSE than
its own one-pass head at step 750 (`ladder_ce` 9.76 against `dm_ce` 5.57,
`lab/experiments/results/2026-09-03-dmorph-v1-panel/dmorph-tok-s1-5k.log`). Only the
core loop trained with gradient THROUGH the iteration (l2cap, full BPTT) ever earned
depth, at 3.5x the per-token cost of a flat stack.

Flow Reasoning Models (Helbling et al., arXiv 2606.29150, mirrored at
`docs/references/training-objectives/flow-reasoning-models/`) name the mechanism and a
cure that needs no BPTT. Their diagnosis (§3.1): a denoiser trained on one pass from the
ground-truth interpolant and then fed its own predictions back at inference sees a carry
distribution it was never trained on (Eq. 6); the mismatch grows with depth; overconfident
wrong predictions are fed back as evidence and the recurrence converges to a spurious fixed
point — "increasing gold-target cross-entropy with depth (Fig. 6)". That sentence is our
K-sweeps. Their cure (§3.2, Fig. 4): Fixed-Point Forcing (FPF). Sample the supervision
time `t ~ U(0,1)` and a start `t_start ~ U(0, t)`; build `x_start = (1 − t_start)ε +
t_start·y`; run the ordinary self-conditioned inference integrator from `t_start` to `t`
under stopgrad; take its final prediction as the carry `s`; supervise the SAME canonical
interpolant `x_t` with the same flow-matching CE, now conditioned on `s`. No gradient
through the rollout; rollout probability 0.5, depth 16; the path and the endpoint loss are
untouched. Results (Table 2, three seeds): Sudoku-Extreme 13.1 % (base flow) → 32.6 %
(self-conditioning, saturates with depth) → 99.2 % (FPF); accuracy keeps improving with
inference depth beyond the training horizon of 16; the convergence residual predicts
correctness with AUROC 1.00 under FPF against 0.50 without (Fig. 6c). Scope caveat: 7–25M
parameter non-causal DiTs on Sudoku, Zebra and mazes; no language modelling.

The prerequisite FRMs add and we lack is the **self-conditioning carry** (§2.2, Chen et
al. 2023): the denoiser takes its own previous clean-solution prediction `s` as an extra
input, `D(x_t | c, s)`, with `s = ∅` a zero carry; training runs a null-carry pass to make
a detached `s̃` and feeds it to the supervised pass half the time. That carry is what makes
each recurrent update directly supervised ("every update direct supervision, allowing
detached training without backpropagation through time") — the recurrence lives in the
OUTPUT space, where every iterate is a decodable candidate, not in a hidden state.

## Proposal

Two things, in this order; the second is a design question for the loop program, not a
build.

### dmorph v1.1: self-conditioning + FPF on the noisy stream

1. Add the carry input to `DmorphStream`: `s` is the previous clean-solution prediction in
   target space — for the tok arm the hard-bridged embedding row of the previous `D̂`
   (the paper's decodable candidate; unit L2, on the table manifold), for the hs arm the
   previous `D̂` itself. It enters the block the way `x_t` does (added to the noisy
   input through a zero-init projection, so `s = 0` at construction is bit-identical to
   v1) and is ALWAYS stopgrad.
2. Training (per the paper's Fig. 4 right, adapted to the band→block routing): with
   probability `p_fpf = 0.5` draw `t_start ~ U(0, t)`, form `x_start`, run the v1 ladder
   from `t_start` to `t` under `no_grad` — the bands crossed are `b(t_start) … b(t)`,
   so the rollout costs at most one forward of the flat stack, usually less — and take
   its last prediction as `s`; otherwise `s = ∅`. Supervise `x_t` exactly as v1 does
   (`dm_fm + dm_ce`), now with `s` in the input. `p_fpf`, the rollout's held-time
   recurrence count `k_train` (default 1) and the carry form are config keys under
   `dmorph.fpf`; `p_fpf: 0` rebuilds v1 bit-for-bit.
3. Inference: the paper's sampler — alternate a flow advance (one block, one Euler step)
   with `k` held-time recurrent updates `s ← D(x_t | c, s)` at the same `t` and block.
   `k` is a free eval dial; report the ladder head at `k ∈ {0, 1, 2, 4}` and, per Fig. 6c,
   the residual `‖s_{k+1} − s_k‖` against per-token correctness (AUROC).
4. Gates before any panel: `dmorph.fpf.p_fpf: 0` bit-identical to v1; the rollout runs
   under `no_grad` (no parameter receives gradient through it — assert on a spy);
   `s = 0` at construction; the mutation "feed the ground-truth `y` as the carry" must
   collapse `dm_ce` to ~0 (the leak the carry could open) and the leak test must still
   hold for the clean head.

Prereg: a NEW planned file (the v1 panel's predictions stay frozen); the primary read is
`ladder_ce` and ladder accuracy versus `dm_ce` and the clean head at the same step, and
whether the ladder improves with `k` (the one curve nobody has, in text).

### The core loop: is the paid loop paying for the wrong thing?

MORPH's core loop iterates a HIDDEN state and pays BPTT through it (`bptt_depth: 8`, the
l2cap lesson: truncation lost). FRMs get depth that generalises past the training horizon
with NO BPTT by (a) carrying a decodable prediction instead of a hidden state and (b)
training each step on states from the model's own rollout. MORPH's truncated BPTT already
had (b)'s shape — the first `T − bptt_depth` iterations run under `no_grad` and the last
ones get gradient — and it lost to full BPTT; what it never had is (a): its iterate is not
a prediction, so an intermediate iterate cannot be directly supervised. The honest
proposal for the loop program is a probe, not a build: log the loop's per-iteration
residual `‖h_{k+1} − h_k‖` against per-token correctness on the shipped paid loop (a K-sweep
already exists in `lab/divergence/a2_depth_sweep.py`). Under the paper's reading a healthy
recurrence shows residual→correctness AUROC well above 0.5; the free-ride loop should sit
at chance. That number decides whether a carry-in-output-space variant of the core loop is
worth designing.

## Alternatives considered

- **Skip self-conditioning and only do FPF on the v1 ladder.** Rejected: FPF's carry IS
  the self-conditioning input; without it there is nothing to feed the rollout's state
  into, and the paper's ablation shows the base flow at 13 % with neither.
- **Backpropagate through the rollout (BPTT over the ladder).** Rejected: that is the
  l2cap price the no-loop line exists to avoid, and the paper's result is specifically
  that stopgrad suffices once the carry distribution matches inference.
- **Apply FPF to the hidden-state loop directly (carry = `h`).** Not now: the paper's
  direct supervision of every update depends on the carry being decodable; a hidden-state
  carry brings back BPTT or an un-supervised iterate. Filed as the probe above.
- **Wait for the v1 panel before writing this.** The v1 ladder result at step 750 already
  shows the exposure-bias signature; writing the design now costs nothing and the panel's
  numbers slot into the Problem section when they land.

## Acceptance criteria

Listed under item 4 of the v1.1 proposal, plus: `pytest tests/` green; smokes for tok
and hs with `p_fpf: 0.5` exit 0; a throughput read next to v1's (the rollout's cost
measured, not assumed).

## Risks

- The rollout's carry for the tok arm is a hard-bridged token: on text the "candidate
  solution" is one token per position and the bridge is argmax, so the carry may be a
  near-copy of `x_t`'s own decode at high `t`; the `y`-as-carry mutation gate and the
  decodability read (`lab/dmorph/decodability.py`) must be re-run with the carry present.
- The paper's tasks are constraint-satisfaction with one correct answer; text has none.
  Convergence to a fixed point may not be the right notion for an entropy-bearing target;
  the residual→correctness AUROC is the measurement that says so either way.
- Rollout cost: ≤ one stack forward per training step at `p_fpf 0.5`, so ≤ +0.5x forward
  FLOPs on top of v1's 1.25x; report tok/s next to `flop_proxy`.

## Implementation record (v1.1, 2026-09-03)

Built on `feat/dmorph` the same day; the loop probe is NOT built.

- `morph/model/dmorph.py`: `DmorphConfig.fpf_p` / `recur`; `DmorphStream.W_s` (zero-init)
  and `carry_in(s) = W_s(stopgrad(s)·in_gain)`; `noisy_stream(..., s=None)`;
  `carry_of(D̂) = normalize(D̂)` (NOT bridged — a bridged carry cannot say "unsure");
  `integrate(...)` is THE integrator (any `t_start → t_end` per row, carry fed forward,
  `recur + 1` evaluations per step, inactive rows untouched via a band of −1), and the
  eval ladder (`ladder_run` / `ladder`), the generator (`dmorph_infer`) and the training
  rollout (`fpf_rollout`) all call it; `residual_auroc` is the Fig. 6c read.
- Training: `fpf_rollout` draws `t_start = t·U(0,1)` per row, builds `x_start` from the
  SAME `x0` as the supervised interpolant (the paper's `noise` is one variable), runs
  `integrate` under `no_grad` on the rows in `use ~ Bernoulli(fpf_p)`, and the loss-bearing
  pass at the untouched `x_t` gets that carry. The rollout runs in whatever mode the
  model is in (no `.eval()` toggle inside a forward — compile guards); the configs run
  dropout 0.
- Eval keeps the null carry for `dm_ce` (so v1 and v1.1 one-pass reads are the same
  instrument) and, under `fpf_p > 0`, adds `dm_ladder_ce_r{0,2}`, `dm_ladder_acc_r{0,2}`,
  `dm_resid_r2`, `dm_resid_auroc_r2`. `aggregate_eval` now drops NaN batches with their
  weight (an AUROC with one empty class).
- Deviation from the paper: their carry is the categorical prediction, ours is the
  direction of `D̂` on the target manifold (what the tied head reads); their integrator
  has 16 steps, ours has `n_blocks` (4) band steps — the band ladder IS the inference
  integrator here.
- Fixed in passing: `dmorph_infer` read the ladder through the RAW `D̂ · head_scale`
  while eval used `readout_state`; both now use `readout_state`.
- Compat: a v1 dmorph checkpoint has no `dmorph.W_s.weight`; the loaders leave it at
  zero (= v1 exactly) and warn (`tests/test_checkpoint_compat.py`).
- Gates: `tests/test_dmorph_fpf.py` (8 tests: config bounds, `fpf_p: 0` bit-identity with
  `W_s` untrained, null carry / zero `W_s` no-ops and a live carry not, rollout carry
  detached + `W_s` trains, the rollout runs exactly the blocks between `band(t_start)`
  and `band(t)`, `integrate(0 → 1)` is the v1 ladder and `recur` repeats each block,
  eval reads only under fpf, AUROC helper). `pytest tests/ -q`: 595 passed, 8 skipped,
  1 xfailed at this commit. Prereg: `lab/experiments/planned/2026-09-03-dmorph-fpf-tok.md`.
