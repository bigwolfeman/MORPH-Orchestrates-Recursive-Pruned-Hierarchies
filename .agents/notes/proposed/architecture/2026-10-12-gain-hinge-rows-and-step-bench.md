# Agent Note: Gain hinge on k rows, and one speed bench

Status: proposed

## Problem

Two problems, one change set.

1. The slot loop's gain hinge runs two extra core applications over all B batch rows every step. The flame ledger (`/home/wolfe/morph-scratch/perf/flame2/LEDGER.md`) puts that at 17.5 ms of a ~190 ms step. The hinge reads a gain estimate averaged over rows, so a subset of rows may give a usable estimate for less time.
2. Two agents read the same recipe at 31k and 40k tok/s from different lines of the trainer log. The logged `sps` is a rolling 100-step mean that still holds the compile warm-up, and it prints every 20 steps. The project has no single speed number.

## Proposal

**`model.slot_gain_rows` (default 0 = every row, byte-identical to master).** With k > 0 the hinge's two applications run on k of the B rows. A batch row never attends to another batch row, so the kept rows' values are exact. `_slot_gain_penalty` cuts `h`, the injection source, the injection stack (batch axis 1), the mask, the `renorm_to` target, and the compact plan with its detached attention caches to the same k rows, then runs the unchanged body. The penalty formula is unchanged. The gain is the mean over the kept rows.

The kept rows rotate: call n reads rows `(n*k + 0..k-1) mod B`. `n` is a device buffer (`_gain_rows_ctr`, not persistent) that the graph advances, so a `training.graph_step` replay moves the rows. A host int would freeze into the graph. The rotation touches no RNG stream, so the hinge's RNG restore still holds.

Refusals (raise, no fallback): k < 0; k > 0 with `slot_gain_lambda` 0; k > 0 with `slot_compact: off` (the compact guard is what bars every per-pass feature whose batch-shaped state the slice would also have to cut: retention, carry, reused f0, reread, SCSE); k >= B at the first hinge.

**`training.step_time_print` (default false) and `lab/perf/bench_step.py`.** The key prints `[steptime] step= t= tokens=` once per step, with the wall clock at the step end. A step is asynchronous and only the logging steps (every 20) sync, so a per-step duration is noise (p10 of 0.5 ms in the first draft of the bench). Two stamps 20 steps apart, both at a sync step, are exact. The bench cuts steps `--skip`..end into 20-step blocks, turns each into ms/step, and reports median, p10, p90 over blocks, tok/s from the median, per-repeat medians, and for two arms the ratio of medians. It runs arms interleaved, one `train` process each, under the GPU lock, WANDB offline, own `hydra.run.dir`, checkpoint dir removed after.

Use `bench_step.py` for every speed claim: `python lab/perf/bench_step.py --arm a=CONFIG --arm b=CONFIG:"model.x=1" --repeats 2 --out r.json`.

## Alternatives considered

- Sample rows at random from a host generator. Rejected: under graph_step the draw freezes into the graph, the defect `_gain_pass_for_step` already documents for the pass draw.
- Randomly permuted rows from a device RNG. Rejected: it would move the global CUDA stream the hinge restores, and rotation covers every row equally.
- Support `slot_gain_rows` outside `slot_compact`. Rejected for now: the plain `_core_step` closure captures batch-shaped masks, the reread, carry, SCSE state. The compact guard already refuses all of those, so the slice stays small and exact.
- Time per step with `torch.cuda.synchronize()` each step. Rejected: it adds a host launch bubble per step and measures a different pipeline than a real run.
- Add `training.log_every` as the task first suggested. Rejected: the existing line holds a rolling mean, so a faster print interval would not fix it. A stamp line is the missing data.

## Acceptance criteria

- Key 0: byte gate SHA `4a73ebf5...bd70d7` unchanged (met 2026-10-12).
- `tests/test_slot_gain_rows.py`: key-0 build, refusals, slice equals the key-0 hinge on the same rows (bit for bit), a full-batch core step's kept rows equal the sliced step (< 1e-4), rows rotate, a CUDA graph replays a new choice (met).
- `tests/test_graph_step.py` case `gain_rows`: graph replay equals the eager step bit for bit with k=1 (met).
- `tests/test_bench_step.py`: parse and aggregate (met).
- Orchestrator 5k quality check of k=2 against k=0 (NOT run). A gain estimate on 2 of 6 rows is noisier; the hinge has no measured tolerance for that yet. Read `tul/gain_est` for drift and the spike-train tripwire (`lab/divergence/DIVERGENCE-README.md`) before adopting.

## Risks

- The hinge is the break-glass divergence control. A noisier estimate may let a detonation through that the full-batch hinge catches. 300 steps cannot show it.
- `B` here is the micro-batch. At B=6 and k=2 each row is read every third step.
- The bench measures blocks of 20 steps, so a run needs `--steps` of at least `skip` + 40 for two blocks. Default 300/100 gives 10 per run.
- Measured on one machine on 2026-10-12: noise floor of the bench (same config twice) is 0.1 to 0.4 %.
