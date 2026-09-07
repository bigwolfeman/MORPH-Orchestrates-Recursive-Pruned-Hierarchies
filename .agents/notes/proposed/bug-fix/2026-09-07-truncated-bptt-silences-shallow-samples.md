# Agent Note: truncated BPTT silences every sample shallower than the batch's no-grad prefix

Status: proposed

## Problem

Both loop paths compute one no-grad prefix per batch, `n_nograd = total_iters −
bptt_depth` with `total_iters` the deepest draw in the batch (`morph/model/transformer.py`,
`_core_region` ~L1790 and `_tul_core` ~L2436), and run iterations `t < n_nograd` under
`torch.no_grad()` for every active sample. A sample (or slot) whose Poisson depth is at or
below `n_nograd` therefore runs ALL of its iterations without a graph: its loop output is
detached, and neither the core nor the prelude receives gradient from it. Only the deepest
draws of a batch train the loop. Simulated fractions of silent samples: mean 16 / max 24 /
bptt 8: 27 % (plain, batch 6), 57 % (slot loop, ~300 slots per batch); the historical
`bptt_depth 4` at mean 6 / max 8: 25 %. At `bptt_depth ≥ max_depth` (full BPTT, the
current base) the defect is invisible, which is why it was never seen. Parcae (arXiv
2604.12946, Alg. 2) avoids it by front-padding: each sequence starts its loop at
`T_max − T_i`, so its LAST `µ_bwd` iterations always carry gradient. Found 2026-09-07
reading the E6/E7 deep-draw runs (`lab/experiments/successes/2026-09-07-arc-e6-*`,
`failures/2026-09-07-arc-e7-*`).

## Proposal

Make the gradient window per sample: a sample of depth `d_i` runs its first
`max(0, d_i − bptt_depth)` iterations under no_grad and its last `min(d_i, bptt_depth)`
with grad. With the sorted-active-set loop this is a per-iteration split of the active
prefix into a no-grad head and a grad tail (samples sorted by depth descending: at
iteration `t` the samples with `d_i − t ≤ bptt_depth` are in their grad window). Keep the
checkpointing on the grad iterations. Bit-identical at `bptt_depth ≥ max_depth` (no sample
has a no-grad iteration), which is the invariant the tests must pin. Same change in
`_tul_core` for the per-slot depths. Log the fraction of samples with a nonzero loop
gradient each step so a regression is visible.

## Alternatives considered

- Front-padding as in Parcae (start shallow samples late so every sample ends at
  `T_max`). Equivalent gradient coverage, but it changes which iteration index a sample's
  first step runs at, which interacts with the per-iteration conditioning
  (`core_stage_cond`) and the gain hinge's iteration draw; the per-sample split keeps
  iteration indices as they are.
- Leave it and raise `bptt_depth` to the max depth (full BPTT) for deep draws. Costs the
  activation memory the truncation exists to save, and Parcae/2604.21106 measure that the
  loop trains worse under truncation anyway; but full BPTT at 24 iterations is not a
  5090 option at seq 1024 batch 6.

## Acceptance criteria

- A test with `bptt_depth 2`, depths `[1, 2, 5]` in one batch: every sample's loss has a
  nonzero gradient on the core; the depth-1 and depth-2 samples' gradients equal those of
  a full-BPTT run (their whole trajectory is inside the window).
- Bit-identical loss and gradients at `bptt_depth ≥ max_depth` on the existing
  `test_tul_forward.py` fixtures.
- The logged silent-sample fraction reads 0.0 on `notul_deep16.yaml`.

## Risks

The per-iteration split adds a second `_core_step` call per iteration when both a no-grad
head and a grad tail are active (two launches instead of one); the active-set machinery
already varies shapes per iteration, so compile shapes grow. The E6 result (0.104 nats
behind mean-6) was measured WITH the defect; the fix changes that baseline.
