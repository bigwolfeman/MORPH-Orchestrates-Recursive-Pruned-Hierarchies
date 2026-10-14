# Agent Note: Parcae-LXTUL speed work and the 40k head-to-head

Status: proposed

Written 2026-10-14 for the agent that runs the MORPH optimization branch. Do this work AFTER the
MORPH speed round ends. Wolfe's direction, in his words: "seeing what we can do about the tok/s
of lxtul vs parcae combined with 40k runs will give us better loop contribution metrics to look
at too and answer a lot of questions." He rejected a broad ablation panel. Research files behind
this note: `/home/wolfe/morph-scratch/abq/design/publish-2026-10-13/` (`vs-parcae.md`,
`ablation-ledger.md`, `related-work.md`).

## Problem

Wolfe wants to show training-compute savings on a clean backbone that reviewers know. His target
speed ratios, same width and same tokens per step:

| Model | Target | Measured today (5k trainer rate, eager, batch 6 x 1024) |
|---|---|---|
| Flat transformer (`gpt8`) | 1.0 | 80.3k tok/s (1.00) |
| LXTUL-Parcae + pointer | 0.7 (0.5 is acceptable) | 22.2k tok/s (0.28) |
| Plain Parcae loop + pointer | about 0.3 | 21.6k tok/s (0.27) |
| Bare strict loop (no fan, no decoder) | | 37.6k tok/s (0.47) |

Today LXTUL ties the plain loop. The slot idea itself is cheap (the bare loop runs 1.7x the
plain loop). The cost is in what the recipe adds on the loop batch and around it. From
`/home/wolfe/parcae/lxtul/model.py` (lines 386-470, read 2026-10-14):

- Every slot runs `T = dslot.max()` passes. With Poisson(6) clamped to [1, 8] and about 300
  slots per batch, T is always 8. Finished slots still compute (`torch.where` keeps their old
  state). About 1.4x the needed core work (8 against a mean near 5.6).
- `ckpt_passes: 4` recomputes the forward of passes 0-3. About +17 % on the core.
- The fan is M = 4 cells: 4x the core work of one cell.
- The gain hinge runs one extra pass (forward and backward) at a random pass `t_gain`.
- Host syncs each step (`int(dslot.max())`, `int(torch.randint(...))`) block CUDA graphs.
- The span decoder is training-only and costs about as much as the whole loop. On MORPH
  (flame4 ledger, `/home/wolfe/morph-scratch/perf/flame4/LEDGER.md`) the split per step at M = 4
  is: prelude 2.12, coda 2.34, LM head 2.09, core 3.65, span decoder 2.78, hinge 1.07 TFLOP.

A measurement agent ran on 2026-10-14. Its report is
`/home/wolfe/morph-scratch/abq/parcae-speed/REPORT.md`: FLOPs per step by region for gpt8, plain +
pointer, LXTUL + pointer and the bare loop; eager and compiled ms/step; a profile; the code
that blocks compile and graphs; and whether the 480 eval rows overlap the training stream. READ
IT FIRST and re-rank Steps 2 and 3 by its numbers.

## Proposal

Work in `/home/wolfe/parcae` (branch `lxtul-testbed`; package `lxtul/`; read `lxtul/CLAUDE.md`
and `lxtul/SPEC.md`). Runs go to `/home/wolfe/parcae-runs` (ZFS). Use sonnet builders for code,
one key per change, default off and byte-identical when off. Run `lxtul/tests/test_geometry.py`
after any change to masks, the write or the layout. Filings go in
`/home/wolfe/parcae/docs/experiments/` (planned, then successes or failures).

### Step 1. Validation hygiene (blocks Step 5)

If the report finds that the 480 eval rows (or the trainer's val batches) come from documents
the training stream also reads, move them to a held-out split that the training stream never
reaches, for every arm. A 245.8M-token run must not train on its own eval rows (MORPH memory
note `owt-val-overlaps-training.md`). Re-read the existing 5k plain + pointer and LXTUL + pointer
checkpoints on the new rows so the old filings stay comparable.

### Step 2. Exact engineering (same function; rounding-only at most)

Give EVERY arm the same treatment where it applies (gpt8, plain, LXTUL), so the speed ratios are
fair.

| Change | Arms | Expected effect |
|---|---|---|
| Active-slot compaction: sort slots by drawn depth, pass t computes only slots with depth > t | LXTUL | core work about / 1.4 |
| Depth draw and hinge pass drawn on the device, static T = max_depth, no host syncs | LXTUL, plain if it syncs | enables CUDA graphs |
| `ckpt_passes: 0` if memory allows (check peak GiB) | LXTUL, plain | core about -15 % |
| Static slot layout: fixed slots per row, padded, so compile never recompiles (today `recompile_limit: 64`) | LXTUL | removes recompiles |
| CUDA-graph the whole train step (reference: MORPH `morph/training/graph_step.py`) | all | removes launch overhead |
| Packed span decoder on real span positions (MORPH `model.spandec_packed`) and the LM head on loss rows only (MORPH speed package, filed bit-identical) | LXTUL | decoder and head work cut by the pad share |

Gate for each change: a 300-step loss trace, key off vs on, same seed, equal or rounding-only;
`test_geometry.py` green; a synced bench (median of 200 steps after 100 warm-up). Timing-only
changes also get a 600-step, 2-seed stability read (MORPH memory note
`byte-gate-misses-launch-order-bugs.md`).

### Step 3. The lean recipe (objective changes; a 5k check before 40k)

These change what the model trains on. Each needs a 5k run against the current LXTUL + pointer
(normalised) arm. Read CE@6 (seed spread on Parcae is about 0.001) and check that K1-K6 stays
well above zero. Parcae's K1-K6 seed spread is about 0.021, so single-seed K1-K6 differences
under 0.02 are unreadable; do not gate on them.

| Lever | Source of the idea | Build |
|---|---|---|
| Ungraded fan (no latent selection), at Poisson cell mean 2 or one cell | MORPH fan recipe; MORPH diet chain 2026-10-13 (at cell mean 2, M = 1 read the same as M = 4 on seed 1) | `select: none` exists (`arm/ungraded.yaml`); cell-count draw is new on Parcae |
| Set reader: the span decoder cross-attends to the cells as a set | MORPH `tul.spandec_reads_cells` (2026-10-11 winner) | port |
| One decoder block instead of two | MORPH diet chain, filing `lab/experiments/*/2026-10-13-flop-diet-5k.md` | config |
| Decoder on half the slots per step | MORPH `tul.spandec_slot_keep` (branch perf/dietkeys) | port |
| Decoder CE on a random half of span tokens (NOT the first 8: that cap lost 0.026 CE on MORPH) | new | build |
| Hinge every 4th step at 4x lambda | MORPH `model.slot_gain_every` (branch perf/dietkeys) | port |

Use the MORPH diet-chain filing to decide which levers to carry. Stack the ones that held into
ONE lean arm and check it at 5k on 2 seeds. Do not run each lever alone unless the stack fails.

### Step 4. One bench, all arms

Same harness, same batch, synced timing, compiled and graphed where Step 2 allows: gpt8, a flat
model with the same layer count as the loop backbone (prelude 4 + core 6 + coda 4 = 14 layers,
no loop, for the same-parameter comparison), plain + pointer, LXTUL + pointer, the lean arm.
Report tok/s, TFLOP/step (FlopCounterMode, optimizer separately) and the ratio to each flat
model. Wolfe picks the denominator.

### Step 5. The 40k runs (prereg first)

40,000 steps x 6,144 tokens = 245.8M tokens per run. Arms:

1. Plain Parcae loop + pointer (normalised).
2. LXTUL-Parcae + pointer (normalised), fast path from Step 2.
3. Lean LXTUL from Step 3, if its 5k check held.
4. gpt8, run to the same total FLOPs as arm 1 (about 3x the steps), so every arm sits on one
   CE-vs-FLOP curve.

- Seed 1 for every arm first. A second seed for arms 1 and 2 (or 1 and 3) if time allows.
- Checkpoints at 5k, 10k, 20k, 30k, 40k. At each one run the depth sweep (depths 1, 2, 3, 4, 6,
  8) on the 480 held-out rows. This gives K1-K6 as a curve over training, the main
  loop-contribution reading Wolfe wants.
- Log synced wall time per step (eval excluded) for CE-vs-wall curves.
- Schedule caveat: `cooldown_fraction: 0.5` puts the LR cooldown over steps 20k-40k. Keep it the
  same for all arms. The 5k and 10k checkpoints are therefore NOT comparable to the filed 5k runs
  (which cooled down from step 2,500); say so in the filing.
- Time at today's speed: about 3.4 h per looped arm (277-284 ms/step + eval) and about 2.5 h for
  gpt8 to matched FLOPs. Step 2 should cut the LXTUL arms.

## Alternatives considered

- Run the 40k pair now on the current slow code. CE against tokens does not depend on speed, so
  this was possible. Rejected: Wolfe wants the speed savings shown on the code that produced the
  curves, and Step 1 must land first anyway.
- Do this on MORPH instead of Parcae. Rejected: MORPH's loop signal is about 10x smaller (fan
  K1-K6 about +0.005 against Parcae's +0.05), and MORPH's HC, ternary weights and kernels need
  explaining in a paper.
- A broad component-ablation panel first. Rejected by Wolfe on 2026-10-14; the 40k curves come
  first.

## Acceptance criteria

- Step 1: eval rows proven disjoint from the training stream (by document id), or moved.
- Step 2: every change passes its gate; LXTUL + pointer bench at 0.5x gpt8 or better is the goal
  (0.7x is Wolfe's ideal). Plain and gpt8 get the same engineering.
- Step 4: one table, all arms, tok/s and TFLOP/step from the same harness.
- Step 5: a prereg with numeric predictions committed before launch; all seeds reported; K1-K6
  and CE@6 curves at every checkpoint for both loops; CE-vs-FLOP and CE-vs-wall curves for all
  arms.

## Risks

- Compaction or graph changes alter the function. The 300-step trace gate and the geometry test
  catch most of it; the 600-step stability read catches launch-order bugs.
- The lean recipe may cost CE. That is why Step 3 runs at 5k before 40k, and why the faithful
  LXTUL + pointer arm stays in the 40k set.
- The CE gap may widen with training: MORPH's one-cell pair went from 0.265 at 5k to 0.333 at
  10k. That is a result, not a failure of the plan; report it.
- GPU contention with MORPH work: one model on the GPU at a time, every job under
  `/home/wolfe/morph-scratch/gpu.lock`.
