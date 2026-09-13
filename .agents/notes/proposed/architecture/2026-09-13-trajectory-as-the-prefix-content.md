# Agent Note: the coda reads the loop's trajectory, not only its exit

Status: proposed

## Problem

`prefix_project` writes ONE state — the loop's last iteration — into all `prefix_k` cells
through `W_prefix[k]`. Whatever the intermediate passes computed is discarded at the
boundary. Twelve arms have read token K1−K6 inside [−0.0001, +0.0033] (the strict ruler
**+0.0016 [+0.0013, +0.0019]**, K3−K6 **+0.0002**), and in every one of them the
instrument measures depth through a one-state window. "The coda only ever sees the exit"
has stood as an untested explanation for that flatness for a month.

It is now testable, and there is a theory saying what the answer has to be.

## Proposal

`tul.prefix_source: "exit" | "trajectory" | "exit_repeat" | "entry_exit"`, default `exit`
= today's behaviour, bit-identical.

* **`trajectory`** — cell k of a slot receives the loop state AFTER pass k, through the
  SAME `W_prefix[k]`, plus a zero-init per-cell pass embedding `E_pass[k]`. The EXIT state
  is always written, always into the LAST cell, never duplicated: a slot that ran d
  iterations fills `min(d, K)` cells and the rest are PAD. A pad's carrier is exactly
  zero, it is cut out of the coda's key set, and it keeps only its own self-edge.
* **`exit_repeat`** — the same cell count with the exit copied into every cell. The
  content-vs-count control: `trajectory` must be read against it, never alone.
* **`entry_exit`** — cell 0 holds the ENTRY `z1 = core_init(e)`, the last cell holds the
  exit, the cells between are exit copies. **The binding control**, and the reason is the
  2026-09-13 sorry-free Lean result: for a deterministic map,
  `I((z1,…,zT); Y) = I(z1; Y)` and `I(zT; Y) ≤ I(z1; Y)`. A trajectory prefix cannot carry
  more than the entry does, so any win it shows is information the exit LOST from the
  entry — a recovery, not a loop gain. Two cells holding `z1` and `zT` should therefore
  match the whole trajectory.

## Alternatives considered

* **A residual ENTRY path into the coda** (write `z1` alongside `zT` at `prefix_k: 2`).
  Cheaper, and if the theory holds it is the RIGHT fix. Rejected as the first arm because
  it assumes the answer: `entry_exit` at the same `prefix_k` as `trajectory` is the arm
  that tests it, and this becomes the shipped change if the pair lands where the theory
  says.
* **Per-pass auxiliary losses on the intermediate states** (`tul.spandec_per_pass`,
  `mux_every_pass`). Already built, already run, already flat. Those grade the passes;
  this changes what the coda READS. Different mechanism, and the refusals keep them from
  composing.
* **A learned pooling of the trajectory into one cell.** It would fit `prefix_k: 2` and
  cost nothing at the boundary. Rejected because the pooling is a second mechanism that
  could itself be the reason the arm works or fails, and because the coda already IS a
  learned reader — giving it the cells directly is the weaker assumption.
* **Keeping the exit in cell 0 instead of the last cell.** Rejected: putting the exit
  always in the LAST cell makes its position independent of the realised depth, so the
  coda reads one consistent slot for "the answer" and the pad cells are a contiguous
  prefix of the tail.

## Acceptance criteria

1. `exit` is bit-identical to the pre-knob tree on pinned values. **Met.**
2. Trajectory cells are actually distinct; the exit is always written and always last;
   `min(d, K)` cells are non-pad. **Met**, guarded at forced depths.
3. A PAD cell is inert four ways: no logit moves when it is edited, it is out of the
   coda's key set structurally, it keeps its self-edge, and the SHIPPED `_back_region` is
   handed the narrowed relation. **Met.**
4. Gradient reaches EVERY written pass directly through its own cell. **Met.**
5. The strict leak test still holds. **Met**, with a `tg_restrict` control.
6. `CE(traj) − CE(entry_exit) ≥ −0.005`. **Not met — unrun.** This is the theory's
   prediction and the panel's point.

Prereg: `lab/experiments/planned/2026-09-13-arc-trajectory-prefix.md`.

## Risks

* **`exit_repeat` is built and NOT queued** (scope call, 2026-09-13). Without it
  `trajectory` has no same-cell-count content control, and its comparison against
  `slot-spandec-strict` mixes cell CONTENT with cell COUNT (k=6 vs k=2). Two of the
  Theory agent's three frozen predictions become unscorable as written. Recorded, not
  hidden.
* **The packer confound.** `prefix_k: 6` makes `L_total` 1408 against the ruler's 1152, so
  these arms see about 1093 real tokens a row against 1033. The three `prefix_source` arms
  pair with each other, not with the ruler.
* **A real defect, found and fixed here.** `E_pass` is added AFTER the projection, so
  zeroing the pad's SOURCE state is not enough: a pad would carry `E_pass[k]`, a learned
  constant announcing how deep the slot went, and the CCA causal conv plus the `W_v_prev`
  value shift carry it into the later cells of the same slot INCLUDING the exit. Measured
  before the fix: `E_pass.grad.abs().sum()` **7407** against **0.25** on the
  `exit_repeat` twin. The fix zeroes `values` at pad cells in `_forward_tul`.
* **A guard-theater defect, found by sabotage.** The pad-narrowing test called
  `_tul_pad_cell_narrow` itself, so deleting the call from `_forward_tul` left it green.
  "The guard passed" and "the guard is watching the shipped path" are different claims;
  this is the second instance of that class in two days. Fixed with a spy on
  `_back_region`'s `attn_kwargs`.
* **The realised fill rate is unmeasured.** At mean depth 6 with a Poisson draw, a large
  fraction of the six cells are PAD on real rows, and nothing has counted them.
* **`prefix_k: 6` and the zero-init unregularised `E_pass` are choices, not measurements.**
