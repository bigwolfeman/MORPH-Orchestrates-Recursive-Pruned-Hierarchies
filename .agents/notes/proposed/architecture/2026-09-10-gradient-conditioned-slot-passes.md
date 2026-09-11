# Agent Note: Gradient-conditioned slot passes

Status: proposed

Date: 2026-09-10. Arm: `slot-mnext-gradpass`
(`morph/configs/tul_slot_mnext_gradpass.yaml`), one factor on the ruler
`slot-mux-norm-match` (`morph/configs/tul_slot_mux_norm_match.yaml`). Prereg:
[`2026-09-10-arc-slot-mnext-gradpass.md`](../../../../lab/experiments/planned/2026-09-10-arc-slot-mnext-gradpass.md).
Sibling note, same ruler and same question from the loss side:
[`2026-09-10-credit-assignment-in-the-slot-loop.md`](2026-09-10-credit-assignment-in-the-slot-loop.md).

## Problem

Five one-factor arms have now changed the LOSS side of the slot loop and every one that
has been read so far read flat. `progressive_p` changed which passes carry the exit
gradient. `pass_lora_rank` gave the passes their own parameters. `mux_every_pass` put the
exit target on every pass. `mux_stage_own_iters` put a different target at one intermediate
pass, and `mux_stage_all` puts it at every non-final pass. Token K1-K6 stayed at or under
0.0006 through every one of them that has run.

One thing has never been varied: **what a pass can see**. Every pass of `_tul_core`
receives the same two things — the current carry and a fixed seed injection — so a shared
map applied six times is the same instruction six times. Three measurements from 2026-09-10
say that is exactly how it behaves. The per-pass cotangent is flat
(0.168/0.168/0.166/0.160/0.154/0.183), so gradient reaches every pass. The six passes'
updates to the shared weights largely cancel (`|sum_t dW_t| / sum_t |dW_t|` = 0.520 against
0.408 for six orthogonal equal-norm updates). And the decisive one
([`slot_z_optimize`](../../../../lab/experiments/results/2026-09-10-slot-z-optimize/README.md)):
the coda's CE with the loop's exit equals its CE with the loop's ENTRY to 0.0015 nats,
while a z fitted by gradient descent on the same frozen coda is worth 0.9-2.6 nats.

The reader works. The writer produces nothing the reader wants. And the one thing the
fitted z has that the loop's z does not is a gradient.

## Proposal

`tul.grad_pass`: before each pass `t`, hand the loop the gradient of a local objective with
respect to the state it is refining, as an INPUT.

* **The objective.** `L_own(h_t) = _tul_mux_loss(h_t, target="own")` — the slot's own span
  read through the tied head, the same head and the same weighted-CE code the exit MUX
  uses. It is CAUSAL for the slot: `mux_span_targets(target="own")` gives slot `i` the
  tokens of span `i`, and slot `i` sits after every one of them, so a generator can compute
  the same feature with no lookahead.
* **The feature.** `g_t = torch.autograd.grad(L_own, h_t.detach(), create_graph=False)`,
  reduced over the Hyper-Connection streams (the MUX readout starts with a stream mean),
  divided by its own per-slot RMS, scaled by `tul.grad_pass_scale`, mapped through a
  ZERO-INITIALISED `W_g` (`[C, C]`, `TULGradPass`), and broadcast-added into the carrier
  through `_apply_injection`.
* **The loss is untouched.** The exit still carries the ordinary M-next forecast MUX at
  beta 1. This knob adds an input, never a term. The toy study's constraint — the local
  target must DIFFER from the exit target, or per-pass supervision becomes
  depth-independence training — holds by construction: own span inside, next span at the
  exit.
* **Where it is built.** In the `_tul_core` loop body, OUTSIDE `_core_step`. That is the
  checkpointing decision: `ckpt_grad_iters` wraps `_core_step` in
  `torch.utils.checkpoint`, whose recompute re-runs that function under a fresh grad
  context, and an inner `autograd.grad` there is a shape this tree has never executed.
  Building the feature in the loop body and letting only the finished tensor cross the
  checkpoint boundary keeps selective core checkpointing working unchanged, so the knob
  does not force `ckpt_grad_iters=0` and pays no memory for it.
* **Readouts.** `loop/own_pass_t{t}` (the local target's value at each pass — a loop that
  is descending the objective it is handed reads a falling ladder) and `loop/gp_rel_t{t}`
  (the injected term's norm over the state's — 0 at step 0 by construction, and the
  falsifier for "the optimiser left `W_g` at zero and the arm is the ruler").

Sources: Marino, Yue & Mandt 2018, *Iterative Amortized Inference* (arXiv:1807.09356);
Greff et al. 2019, *Multi-Object Representation Learning with Iterative Variational
Inference* (IODINE, arXiv:1903.00450). Both make an iterative network an optimiser of its
own objective by feeding it that objective's gradient at every step, and both report the
detached gradient is what works.

## Alternatives considered

* **`create_graph=True` — the second-order, learned-optimiser objective** (Andrychowicz et
  al. 2016). It is the theoretically right thing: the outer loss would then shape `W_g`
  through how the feature changes the trajectory, not only through where it lands. Rejected
  for the first draw on cost and risk: it retains one inner graph per pass (against a freed
  one) and puts a Hessian-vector product in every training step at `[B, S, V]` scale, and
  IODINE reports the detached form works. It is the named first follow-up if the feature
  acts and the arm still reads flat.
* **The staged target without the gradient feature** — `mux_stage_own_iters` and its
  every-non-final-pass form `mux_stage_all`, the two arms queued immediately before this
  one. They put the own-span target on the loop as a LOSS. That changes what a pass is
  asked for; it does not change what any pass can see, and the other loss-side arms with
  the same property read flat. The arms are complementary and all three are queued: if the
  staged arms move the forecast curve and this one does not, the target mattered and the
  input did not, and the reverse reading is equally available.
* **Feeding the gradient of the NEXT-span (forecast) loss.** Strictly stronger as a
  training signal and it is what `slot_z_optimize.py` fits. Rejected as a MECHANISM: the
  next span's tokens sit after the slot, so the feature is not computable at inference. An
  arm built on it would measure an oracle, which the tree already has as an instrument and
  must not confuse for a model.
* **A learned-optimiser hypernetwork** — a small recurrent network that reads `(h_t, g_t)`
  and emits the update, replacing the core step for the slot positions. Rejected as the
  first draw for the reason `pass_lora_rank` was informative and not decisive: it changes
  the map AND the conditioning at once, so a flat reading would not say which. `W_g` at
  zero init is the smallest change that makes the input differ per pass, and it keeps the
  arm one factor from the ruler.
* **Injecting the gradient on the ctx channel only**, mirroring `injection_channels: ctx`.
  Rejected: the carry's ctx channel covers 256 of 1024 dims (arc E7's second verified
  defect), and there is no reason to route an error signal through the same narrow pipe the
  seed already saturates. The feature goes to all dims, on every stream, through the same
  `_apply_injection` every other single-stream term uses.

## Acceptance criteria

1. `tul.grad_pass: false` is the forward from before the knob: same loss, same gradient on
   every parameter, no module built, no RNG consumed — with a sensitivity fixture so the
   check cannot pass vacuously.
2. With the knob ON and `W_g` at its zero init, the loss and every other parameter's
   gradient equal the off model's, and `W_g` still receives a nonzero gradient (it escapes
   zero on the first backward).
3. The injected feature at pass `t` equals `torch.autograd.grad` of the own-span loss at
   that pass's state, recomputed by hand to 1e-5.
4. The feature carries no `grad_fn`: no outer gradient and no second-order term flows
   through the inner backward.
5. The injected term is exactly proportional to `tul.grad_pass_scale`, and it moves the
   state pass 1 is handed.
6. The feature runs at eval too — it is part of the map, so `core_depth_sweep.py` measures
   the function that trained.
7. Refusals with tests: the paid loop, `mux_beta 0`, `db_loop`, SCSE, `n_core 0`, a zero
   scale, an unknown norm, and a caller that omits `input_ids`.
8. `tests/test_tul_grad_pass.py` plus the full CPU suite green, and the arm composes and
   builds through `build_tul_runtime`.

All eight hold on CPU as of this note (19 tests in the new file; 1,063 in `tests/`).

## Risks

* **`torch.compile` and an inner `autograd.grad`.** The single most likely way this arm
  fails to start. An `autograd.grad` inside the traced forward at the real shape (B=6,
  S=64, T up to 8) has never run in this tree; a graph break would show as a rate stop
  rather than an error. The runner's 12-step smoke is the first real evidence.
* **The gain hinge is blind to the feature.** It probes at the injected operating point but
  treats the feature as a constant (it is a function of `h`, not of the probe's
  perturbation), so a `W_g` that grows large adds gain the constraint cannot see. This is
  the same "exogenous input" reading IODINE takes, and it is the reason P-a in the prereg
  is 82 % and not higher. The sustained tripwire and `loop/core_gain_t0` are the watch.
* **The feature may be near-constant across passes.** The own span is already in the slot's
  entry state, so `g_t` may barely move as the loop runs and `W_g(g_t)` may degenerate into
  one more seed injection — worth 0.002-0.003 nats on the levers panel. `loop/own_pass_t{t}`
  and `loop/gp_rel_t{t}` are what separate that reading from the mechanism working.
* **`W_g` may stay near zero.** Zero init buys the identity and costs the escape rate. If
  `loop/gp_rel_t0` is still under 0.01 at 5,000 steps the arm is UNINFORMATIVE, not a
  failure, and the next move is the scale, not a new mechanism. The prereg's binding says
  so explicitly.
* **Cost.** One `[B, S, V]` fp32 readout and its backward per pass, ~0.6 TFLOP per step at
  the arm's shape. Estimated at 1.0-1.15x the ruler's wall clock from the
  `slot-mnext-mux-every-pass` anchor (0.99x for strictly more work). Not measured.
* **Untested combinations.** `grad_pass` with `mux_every_pass`, `mux_stage_own_iters`,
  `mux_stage_all` or `progressive_p` is ALLOWED by the config and has never been run. This
  arm sets none of them. If a later arm wants one, the interaction needs its own test
  first. The one combination that IS covered by a test is `tul.mux_readout: full`
  (finding F2): under `mean` every stream carries the same gradient vector (measured spread
  exactly 0.0), under `full` they differ by about 2 %, so the mean over streams becomes a
  summary rather than a recovery of one vector. The docstring says which is which.

## Outcome (2026-09-11)

Filed under `lab/experiments/failures/2026-09-10-arc-slot-mnext-gradpass.md` (failures by the
prediction convention; the arm is the first credit-assignment arm whose exit beats the
ruler's). The feature acts (`loop/gp_rel_t0` 0.086), the loop takes ONE descent step on its
own span (`loop/own_pass_t` 7.105 → 6.740 at pass 1, then flat to 6.721), and the M-next
forecast at the exit is 0.017 nats better than the ruler's with the depth-1 forecast equal to
it (K1−K6 +0.0236, K3−K6 +0.0016). Tokens +0.0014, entry-vs-exit +0.0075, cancellation 0.502,
CE 0.052 better at 5k, 1.07x wall clock. Consequence: the binding's next arms (second-order
form, `tg_restrict` twin) are a proposal for Wolfe, not queued; the reading is that a target
the entry state already contains is descended in one step, so the lane is a target it does
not contain.
