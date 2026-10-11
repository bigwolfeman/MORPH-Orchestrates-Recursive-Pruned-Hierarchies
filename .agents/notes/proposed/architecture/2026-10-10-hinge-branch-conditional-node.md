# Agent Note: Skip the gain hinge's backward with a CUDA conditional node

Status: proposed

## Problem

The gain hinge (`model.slot_gain_lambda`) applies the core twice more at the drawn pass and
charges `lam * mean relu(gain - target)^2`. Its backward is about 11 ms of the 230 ms
lxtul_pointer_fast step (the 2026-10-10 profile: 16.6 ms for the hinge, 0.30 TFLOP forward
and 0.62 TFLOP backward). When no row's gain is over the target the hinge is exactly 0 and
its whole backward adds exact zeros. In the fresh-pass paired runs that was 57-65 % of the
logged steps (steps 2500-3480); the tracker estimated 71 %. Which case a step is in is known
only on the device after the forward, and the step is one replayed CUDA graph
(`training.graph_step`), so the host cannot choose without a sync.

## Proposal

`morph/training/graph_step.py` records a step whose forward reports the hinge
(`out["gain_pen"]`, the penalty as it enters the loss; `out["gain_active"]`, the device flag
"a charged term is not 0", `_slot_gain_penalty`) as four torch graphs in the shared pool:

    pre  (torch replay) -> IF gain_active: A ELSE: B (one CondBranch launch) -> post (torch replay)

* A is the eager backward, recorded with `retain_graph=True`.
* B zero-fills A's gradient buffers and runs the backward from the loss chain's other addends
  (`_cut_roots`: GradientEdge roots below the pass-through adds and the `/ 1`), so no hinge
  node is reached.
* `morph/training/graph_cond.py` builds `[set kernel] -> IF/ELSE(child A, child B)` with the
  CUDA driver API (`cuda.bindings` 12.9; torch 2.11 has no conditional nodes); the set kernel
  is compiled once with NVRTC.

Why B is the eager step: the autograd engine runs the reachable nodes in descending creation
order (each node's consumers are newer, so the newest unexecuted node is always ready when its
turn comes). Leaving the hinge's nodes out keeps the order of every other node and the
summation order of every input buffer, and only drops addends that are exact zeros. Two
differences remain possible: the sign of a zero gradient element, and a step on which a
hinge-side activation is inf (A computes 0 x inf = NaN and the found-inf flag skips the step;
B does not compute it). A flag that reads NaN is active (`~(hinge <= 0)`).

Also in this change: a replay binds again the tensor attributes its recording set
(`_Captured.rebind`, e.g. `_loop_cot_ref`), so a reader after a replay sees that replay's
values; the bodies must not draw random numbers (the generator cannot be read during a
capture, so the eager warm-up steps measure the backward and a backward that drew makes the
recording raise).

## Alternatives considered

* A host read of the flag between a forward graph and two backward graphs: no driver-API
  code, but a device sync in the middle of every step and the same two-backward recording.
  The brief asked for the conditional inside the step's graphs.
* Run the hinge's backward as its own `autograd.grad` inside the IF body and the main backward
  always: one backward recorded, but on active steps the gradients are summed in another order
  (not bit-exact to today's step).
* Clone body A and delete the hinge's kernel nodes: the hinge's contributions are added into
  shared buffers by kernels outside its nodes; removing them safely needs the autograd engine's
  bookkeeping.
* A global "undefined gradient" gate on the penalty: the engine still walks the hinge's nodes,
  and the custom autograd Functions in the core materialise zeros and run their kernels.

## Acceptance criteria

* `tests/test_graph_step.py`: the replay equals the eager step bit for bit (loss, grad norm,
  every `out` value, gradient, parameter, buffer, EMA twin tensor, optimizer state and the
  hook-set `_loop_cot_ref`) with the hinge always active (`slot_gain_target=0`) and never
  active (`slot_gain_target=100`), and on every older parametrization; body B is a smaller
  graph than A. Measured 2026-10-10: 19 passed.
* `tests/test_graph_hinge_branch.py`: `_cut_roots` gives exactly the loss-without-penalty
  gradient and never runs the penalty's nodes; `CondBranch` runs the body the device flag
  names over 24 steps; the model's flag reads False / True / NaN-as-active.
* Byte gate on lxtul_pointer_fast (45 deterministic steps; 7 of them have the hinge inactive):
  identical to master (trace and FINAL_SHA256), alone and with C3 + E + Q. Measured
  2026-10-10.
* Speed, measured 2026-10-10 (bench.sh, 420 steps, wall ms/step over steps 200-400):

  | Case | master | hinge branch | change |
  |---|---|---|---|
  | hinge never active (`slot_gain_target=100`) | 245.5, 245.9 | 237.3, 237.7 | -8.2 |
  | hinge always active (`slot_gain_target=0`) | 246.1, 245.7 | 246.0, 248.4, 246.5, 246.6, 246.6, 246.4 | +0.9 |
  | always active, body A replayed with no conditional (diagnostic) | | 245.7, 245.7 (torch), 245.6, 245.7 (driver) | -0.2 |

  The conditional node itself costs about 0.9 ms per step; instantiate flags 0 (247.3,
  246.5) and an explicit `cuGraphUpload` (246.4, 246.4) do not remove it. Break-even is an
  inactive share of about 10 %. At the 57-65 % measured at steps 2500-3480 the expected change
  is -4.3 to -5.0 ms per step; early in a run, where the hinge is active on every step (0 of
  10 logged steps inactive at steps 200-400), it costs 0.9 ms. The replay-trace profile shows
  a 4.4-4.9 ms GPU idle gap after the set kernel; the benches do not, so most of that gap is
  CUPTI tracing.

## Risks

* Driver-API code outside torch: a body with an event or host node fails at instantiation
  (loud). Torch's own replay prologue (philox offsets) does not run for the bodies; guarded
  by the eager RNG measurement.
* Body A is recorded with `retain_graph`, so its temporaries cannot reuse the forward's saved
  tensors; the pool grows: 21.98 vs 21.00 GiB reserved after recording (gate), 22.73 GiB
  with every lever. The recording takes 7.3 s, not 4.6 s.
* The zero fill of every gradient runs on each inactive step (about 1.2 GB written).
