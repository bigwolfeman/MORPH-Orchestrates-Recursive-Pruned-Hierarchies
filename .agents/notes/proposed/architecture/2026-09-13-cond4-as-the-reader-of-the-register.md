# Agent Note: cond4 as the reader — a non-shared stack between the loop and z

Status: proposed

## Problem

Two problems, one mechanism.

**The knob was never measured.** `tul.cond_layers` is Wolfe's 2026-09-03 sketch: prelude →
core loop → a few NON-SHARED blocks → z → coda. It has been in the tree since the
think-once branch and its only two runs, R7f and R7d
([`failures/2026-09-03-tul-think-once-panel.md`](../../../../lab/experiments/failures/2026-09-03-tul-think-once-panel.md)),
detonated at step ~1010 under `warmup: 0` and the absmean ternary rule. Both of those are
gone — the 1,000-step LR ramp and `norm_match` are the recipe now — and the knob's only
config, `tul_to_cond4.yaml`, composes on the retired `tul_to_mnext` lineage. Nobody has a
number for it.

**On a register model the stack read the wrong thing.** `_tul_core` returns the S·M
compact CELL axis. The Thought Register block reduced it to the MEAN and stashed the raw
cells for the coda's 1:1 prefix write; the stack ran AFTER that. So on
`slot_cells > 1` the stack read S means while the coda read the RAW loop cells. It was the
reader of neither of the register's two readers, and only the span decoder saw it. A knob
that claims "the coda reads the conditioning stack's output" would have been false on
every register arm.

Beside that sits the arc's standing result: twelve strict arms read token K1−K6 inside
[−0.0001, +0.0033] and K3−K6 inside [−0.0008, +0.0003], across every lever tried. The
information note
([`2026-09-13-information-view-of-the-slot-loop.md`](2026-09-13-information-view-of-the-slot-loop.md))
argues the loop's value can only come from the reader's observation, and measures that
reader's per-pass headroom at 0.001–0.003 nats. Four non-shared blocks are the direct
lever on that reader, and nothing in the arc has pulled it.

## Proposal

Two arms, each one factor.

* **`slot-spandec-strict-cond4`** (`morph/configs/tul_slot_spandec_strict_cond4.yaml`) —
  `slot-spandec-strict` plus `tul.cond_layers: 4`. Config only; no code change was needed
  for the M = 1 path.
* **`slot-register-m4-cond4`** (`morph/configs/tul_slot_register_m4_cond4.yaml`) —
  `slot-register-m4` plus `tul.cond_layers: 4`, and the code change that makes the stack
  the register's reader.

**The change.** `_tul_cond_apply` runs straight out of `_tul_core`, BEFORE the register's
mean, on the S·M cell axis, under the SAME in-loop relation the register loops with. The
relation now comes from ONE builder, `transformer.slot_cell_relation`, which the core
stage calls too. Both readers then read the stack: the coda's per-cell prefix write and
the span decoder's mean. `_tul_layer_passes` counts the stack per CELL of a real slot, the
`TUL THINK-ONCE` banner says what the stack reads and under which relation, and
`tul_slot_state_probe` runs the stack so `val/slot_eff_rank` measures what the coda reads.

`slot_cells: 1` is bit-identical: the register block is a no-op there and plain causal over
S slots IS `slot_cell_relation` at M = 1, so the stack keeps its no-mask call. Proved by
running the pre-change source (`b1075f2`) and this tree in two processes on the same
fixture — loss, logit sum, layer passes and grad sum all match to the last printed digit.

`tul.loop_reach > 0` with a stack now RAISES. The reach budget spends every cross-cell
route in core layer 0 and closes it in layers 1..n−1; a stack running the unbudgeted
relation afterwards would carry cells the budget cut.

**A correction the build forced, and it is about the shipped register, not the stack.**
`blk[p][q] = slot(p) >= slot(q)` is a SUPERSET of flattened causal, and every branch that
reads `tg_allow` / `tg_comp_allow` ANDs it into an already-causal relation and can only
NARROW (`attention._tg_slot_attention`, `attention._window_fallback`). So at `loop_reach 0`
the register's documented "own slot full, including the cells after it" relation EXECUTES
as plain flattened causal: a cell never reads a later cell of its own slot, in the loop or
in the stack. Measured two-sided — an all-TRUE mask gives the identical loss at reach 0
and a different one at reach 1. The register's comments and its banner claimed the wider
relation and now say what runs. It is not patched: expressing the two-sided read needs a
branch that can widen past causality, which nothing in this tree has, and
`slot-register-m4` is already queued at the shape it has.

## Alternatives considered

* **Keep the mean and run the stack on the means.** The shipped placement, and it is the
  cheapest: four blocks over 64 positions instead of 256. Rejected because it makes the
  stack the reader of neither of the register's readers — the coda would still read the
  raw loop cells while only the span decoder saw the stack. The arm's own claim ("the coda
  reads the stack's output") would be false, which is the defect, not a cost saving.
* **A span decoder that cross-attends to the M cells, instead of a stack.** This is the
  register note's own named follow-up and it attacks the same thing from the target side:
  the decoder stops grading one pooled state. Rejected for THIS pair because it changes
  the TARGET, so a result would not be attributable to reader capacity. It stays the next
  arm if the stack moves CE.
* **The stack INSIDE the loop, as extra shared passes.** Cheaper in parameters and it is
  what "more depth" would normally mean. Rejected because the arc has already measured
  that lever to exhaustion: more passes of the shared map read K3−K6 ≤ 0.002 whatever the
  target, core, stability term or depth draw. Non-shared blocks OUTSIDE the loop are a
  different mechanism — capacity on the reader, not depth on the map — which is the whole
  reason to run it.
* **Give the stack plain causal at M > 1 instead of the loop's relation.** Simpler, and it
  is what the code did before this change. Rejected because the stack would then be a
  different mechanism from the loop it conditions, and the difference would be invisible:
  at `loop_reach 0` the two execute identically, so only a structural test can tell them
  apart. One builder, one relation, one test.
* **Fix the register's relation so the within-slot forward read really fires.** Tempting,
  since the mask says it should. Rejected: it needs an attention branch that can widen past
  causality, `slot-register-m4` and `-m8` are already queued at the current shape, and
  changing the mechanism under a queued arm makes its own prereg unscoreable. Recorded in
  the builder's docstring instead, so no later arm claims the wider relation.

## Acceptance criteria

1. `cond_layers: 0` builds no stack, draws no RNG and leaves every shared weight
   byte-identical. **Met** —
   `tests/test_tul_think_once.py::test_cond_stack_is_built_last_and_shared_weights_are_byte_identical`.
2. `slot_cells: 1` with a stack is bit-identical to the pre-change placement. **Met** —
   run on two processes, pinned in `tests/test_tul_cond4_strict.py`.
3. Both readers read the stack's output: the coda's prefix write (cell by cell on a
   register) and the span decoder's grade. **Met**, guarded, with the bypass caught.
4. The stack's relation is the loop's own builder, load-bearing, and causal across slots;
   pad cells never reach a valid cell and a pad slot's write still goes to the dump row.
   **Met**, guarded.
5. The interactions hold: the strict leak cut, `zero == all_slots`, the slot-gain
   constraint with no INERT notice, and the forced-depth lever the K-curve instrument
   uses. **Met**, guarded.
6. Token K3−K6 below 0.002 on both arms (the theory's prediction). **Not met — unrun.**
7. A CE or worth move against the one-factor partner. **Not met — unrun.**

Prereg:
[`lab/experiments/planned/2026-09-13-arc-cond4-reader.md`](../../../../lab/experiments/planned/2026-09-13-arc-cond4-reader.md).

## Risks

* **The theorem says this cannot move the K-curve, and that is the point.** If the CE does
  not move either, the arm has spent two runs to close a lane. That is a real cost and it
  is the reason the pair is two arms and not six.
* **The stack may collapse the register's cells.** It is four blocks of mixing over cells
  that already read each other in the loop, and `val/slot_cell_eff_rank` now reads the
  stack's output. A register whose cells survive the loop and die in the stack would look
  like a failed register unless the pair is read together.
* **Eager attention on the register arm.** The S·M relation tensors force the eager path
  for the stack, as they already do for the register's core stage. A second difference
  from the ruler, sized only by the smoke.
* **`slot-register-m4` has not run.** Its cond4 partner is unreadable until it does, and
  two of the predictions are stated against a number that does not exist yet.
* **The executed-relation correction was measured on a 64-dimensional CPU fixture.** The
  argument (an AND into causality cannot widen) is shape-independent; the measurement is
  not.
* **Nothing has run on a GPU.** No smoke, no wall clock, no memory figure at `d_model`
  1024.
