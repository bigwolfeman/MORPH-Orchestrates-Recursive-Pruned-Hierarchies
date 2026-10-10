# Agent Note: The gain hinge reads a fresh pass every step

Status: proposed

## Problem

The slot loop's gain hinge (`model.slot_gain_lambda`) reads the map's gain at ONE loop pass per
training step. The pass was drawn with `torch.randint` from the global CPU stream, and the
stream was put back after the draw (34d94a04, 2026-09-04) so the draw would not move any other
random number. Nothing else in a training step reads the CPU stream, so every step drew the same
pass: 54 of 54 draws were pass 6 on seed 1 and pass 3 on seed 2 (tracker A1,
`/home/wolfe/morph-scratch/perf/hingepass/`). Every winner number trained with one pass guarded,
chosen by the seed. Under `training.graph_step` the pass is also a Python branch inside the
recorded step, so even a moving host draw would not reach a replay. Wolfe decided on 2026-10-10
(tracker A2): a fresh, uniformly drawn grad pass every step.

## Proposal

Built in the working tree, pending the paired check below.

- The model owns a PRIVATE CPU generator, `MORPHTransformer._gain_pass_gen`, seeded at build
  from `torch.initial_seed()` (the trainer's `training.seed`) plus a fixed salt. One float64
  uniform per training step: pass = n_nograd + floor(u * n_grad). No other stream moves.
- `model.slot_gain_pass_force` (default -1 = the fresh draw; k >= 0 pins pass k, for gates and
  tests; refused when inert or past the last pass; a pass outside a step's grad range raises).
- The generator rides the checkpoint (`gain_pass_rng`); a checkpoint from before the key starts
  the draw from the run seed and says so.
- `training.graph_step`: the pass is a GraphStep VARIANT. The trainer passes the model's
  `gain_pass_choices()`, `draw_gain_pass` and `pin_gain_pass`; GraphStep records one graph per
  (step kind, pass), 16 on the winner, in one pool, draws the pass on the host before every
  step and replays that pass's graph. The recorded forwards read a PIN, never a draw; a draw
  inside a CUDA graph capture raises. Eager and replayed runs consume the generator the same
  way, so they draw the same pass sequence (tests/test_graph_step.py, bit for bit).
- Kept gradients (`keep_grads`, the instrument step): each kept recording ends by copying its
  gradients into ONE buffer set outside the pool, shared by all graphs, instead of each graph
  keeping its own (8 instrument graphs x 1.2 GB otherwise).
- The trainer's warm-up compile pins every pass once, so the capture (`fail_on_recompile`)
  finds every pass's compiled hinge frames; the forwards beyond the usual four run with the RNG
  streams and the buffers put back.
- `loop/gain_pass` is logged every step from the host value (no device sync).

## Alternatives considered

- **A device-side index into one recording** (store each grad pass's detached operating point,
  gather the drawn one, run the hinge once). Rejected: under `model.slot_compact` each pass's
  hinge runs on that pass's own row count, so the operating points have different shapes; and
  the hinge's dropout masks come from the CUDA stream at its call site, so one shared site
  draws different masks. It would be a different function (a full-shape hinge at another RNG
  site), not the eager hinge picked late, and could not pass a byte gate at a forced pass.
  Its upside is one recording instead of 16.
- **Keep one graph per kind and let the host draw move** (the pre-change GraphStep with the
  CPU check removed). Rejected: the replay would still read the recorded pass.
- **Every pass every step** (`slot_gain_all_iters`). Rejected in A2: 8x the hinge cost.
- **Counter-based draw** (pass = hash(seed, step)). Needs the step index inside the model and
  is no simpler than a generator whose state is saved; the trainer's step index is not a model
  input today.
- **Kept gradients per graph, as before.** Rejected on memory (above). Computing the per-region
  gradient norms inside the instrument graph would also remove the kept set; it changes the
  trainer's log path and its bit-for-bit test, so the copy was chosen.

## Acceptance criteria

- With `slot_gain_pass_force: 6` on seed 1, a 45-step `lxtul_pointer_fast` run and a 45-step
  eager `lxtul_pointer` run are byte-identical to 6a33958e on loss, grad norm and the gain terms.
- 50-step fresh-draw runs on seeds 1 and 2 log a moving, near-uniform `loop/gain_pass`.
- tests/test_gain_pass_draw.py (fresh, reproducible, RNG-neutral, checkpoint round trip, force
  key, host draw = eager draw), tests/test_graph_step.py on CUDA (replay = eager with several
  passes replayed; the drawn pass's graph runs; a draw inside a capture raises) and the slot-gain
  tests green; every new test sabotaged once.
- ms/step and peak memory of `lxtul_pointer_fast` before and after reported, with the
  recording time per (re)record.
- The paired check `lab/experiments/planned/2026-10-10-hinge-fresh-pass-paired.md` holds.

## Risks

- Recording time: 16 recordings instead of 2 after every eval (eval releases the graphs).
- The fresh draw changes the training function of every hinge arm; numbers filed before
  2026-10-10 read one pass.
- C4 (skipping the inactive hinge's backward) now needs its conditional node inside each of the
  16 graphs; the hinge site is fixed within a recording, so nothing here blocks it.
- The plain-loop twin `model.core_gain_lambda` (`_core_region`, `_t_cg`) draws its pass the
  same frozen way and is NOT changed here (no decision covers it).

Links: tracker [`2026-10-10-phase-2-tracker.md`](../process/2026-10-10-phase-2-tracker.md)
items A1-A4; graph step [`2026-10-07-graph-captured-training-step.md`](2026-10-07-graph-captured-training-step.md)
task 1.2; the side finding in `lab/experiments/mixed/2026-10-08-fast2-package-paired.md`.
