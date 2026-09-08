# Planned: ARC E17 — the Sudoku-Extreme depth grid: does the loop's accuracy on hard boards rise with iterations?

Status: planned
Date: 2026-09-08 (frozen before launch; Wolfe: "this was clearly a failure to capture depth
dependent thinking ... maybe we should try different data?" → "okay we use the Sudoku dataset
then"). Arc: `2026-09-04-loop-contribution-arc.md`, row E17. Launch is Wolfe's call.

## Question

Every loop-contribution number in this arc is on data a single pass can retrieve: web text
(E1–E14) and Olympiad synthetic math whose high bands are 23–47 % unique docs (E16). On both,
every stable loop finishes by iteration 3, the plain model's own core loop is depth-flat
(K1−K6 0.024–0.037 nats), and the only depth-dependent arm (the masked slot loop) depends on
depth because depth 1 is broken for it. E17 asks the question on data that NEEDS serial
steps and cannot be memorized: HRM's Sudoku-Extreme, where 7M–27M recurrent nets beat large
LMs and recurrent depth is the whole story. The readout is a GRID, not a scalar: whole-board
solve rate and answer-cell accuracy by rating bucket (the board's difficulty: the `rating`
column, bucket edges 0 / 1 / 11 / 23 / 38) by forced loop depth T ∈ {1,2,3,4,6,9,12,16}. A
loop that thinks shows a diagonal: hard buckets keep gaining as T grows while easy ones
saturate early. A loop that does not shows flat rows past T ≈ 3 on every bucket.

## Hypothesis

H17: the loop does not iterate on Sudoku either. Whatever a model learns in 6,000 steps it
learns in its first 2–3 iterations, on every bucket; the masked arm again needs 2–3
iterations to climb out of its own depth-1 hole and is flat or worse after; the plain model
is at least as good as any TUL arm at its trained depth. Counter-hypothesis H17′: on the hard
buckets the gain from T = 3 to T = 12 is positive with the CI above 0 and larger than on the
easy buckets — the diagonal.

## Method

Data (`scripts/sudoku_shards.py`, decision record
`.agents/notes/implemented/feature/2026-09-08-sudoku-extreme-data-path.md`): 1,000 train
puzzles sampled from `sapientinc/sudoku-extreme` train.csv stratified over the five rating
buckets, × 1,000 HRM augmentations (digit relabel, band/stack and in-band row/column
shuffles, transpose) = 1,000,000 boards, 0 exact duplicates, 183M tokens; held-out shard =
3,000 test.csv boards, 600 per bucket, NO augmentation, 0 found in the train shard (exact
token-sequence check). One board = 9 puzzle rows (9 digit tokens + newline) + `<|A|>` + 9
solution rows + `<|/A|>` + EOS = exactly 183 tokens; every grid row ends at a newline so the
TUL boundary rule gives one slot per row (19 slots per board with EOS). `seq_len` 182 so a
packed row is exactly one board (`tests/test_sudoku_shards.py::test_packing_never_splits_a_board`
and its control). Loss = ordinary CE on every token, puzzle cells included (no loss mask
exists in the loader; the puzzle part is a constant floor ≈ 0.5 nats/token); scoring is on
the 81 solution cells.

Arms, in order: **mask** (`tul_sud_mask`: the masked slot target, `tg_scoped_kernels`, hinge
0.9, fixed-point 1.0, Poisson slot mean 12 / max 16, full BPTT) and **notul** (`notul_sud`:
the plain model, Poisson core depth mean 6 / max 8). mnext (`tul_sud_mnext`) and a1
(`tul_sud_a1`) are built and smoked and run only if Wolfe adds them (`ARMS=`). Every arm:
6,000 steps in ONE curriculum stage, micro 32 × accum 4 = 128 boards/step (5,824 tokens per
micro-batch), ramp 1000, `ckpt_grad_iters 4`, `tul.max_slots 0` (→ 22), eval every 250 on
the held-out shard (rewound; 20 batches × 32 boards), checkpoints at 1500/3000/4500/6000.
Commit pinned in `arc/E17_COMMIT`; worktree `/home/wolfe/morph-to`; runner
`arc/run_e17.sh`: 12-step smoke per arm, the sustained tripwire (`preclip/total > 1e4`
sustained), then `lab/divergence/olympiad_sweep.py` on ALL 3,000 held-out boards at every
checkpoint, depths 1,2,3,4,6,9,12,16, batch 16 → `results/2026-09-08-arc-e17/sweep_<arm>_<ck>.json`
with `solve_rate`, `band_solve_rate`, `band_acc_answer`, and per-board sums for pairing.
Analysis: the grid (bucket × T) of solve rate and answer-cell accuracy per arm and
checkpoint; K-differences paired over boards with 2,000-draw bootstrap CIs (`score_e17.py`,
written after launch, reads only the JSONs); between-arm differences paired over the same
3,000 boards at each arm's trained depth. The copy floor: about 28 % of solution cells are
givens, so answer-cell accuracy ≈ 0.28 means "copies the puzzle, solves nothing".

Smokes before this prereg (2026-09-08, 12 steps each, all exit 0, 0 NaN): mask peak 20.07 GB
(`EAGER+TGSCOPED`), notul 15.60 GB, mnext 19.82 GB, a1 19.54 GB. Steps/s not recorded by the
build; the launch smoke records it.

## Predictions (frozen)

- **P17a (survival).** Both arms reach 6,000 with the tripwire silent: mask **75 %**, notul
  **95 %**.
- **P17b (does anything learn).** Notul answer-cell accuracy at its trained depth 6 at
  6,000 > 0.50: **55 %**. Notul bucket-0 solve rate at depth 6 at 6,000 > 0.20: **40 %**.
  Notul bucket-4 solve rate < 0.05: **75 %**. Mask answer-cell accuracy at depth 12 at 6,000
  > 0.50: **45 %**.
- **P17c (depth beyond 3, the point).** Notul answer-cell accuracy K3−K12 on buckets 3–4
  (pooled) at 6,000 is negative (deeper helps) by more than 0.01 with the CI excluding 0:
  **20 %**. Same for mask: **25 %**. Notul K1−K3 on buckets 3–4 > +0.01 (the shallow part of
  the curve exists): **50 %**; mask K1−K3 > +0.05: **80 %** (the depth-1 hole).
- **P17d (the diagonal).** On either arm at 6,000, the accuracy gain from T = 3 to T = 12 on
  bucket 4 exceeds the gain on bucket 0 by more than 0.01 with the paired CI above 0:
  **20 %**. Solve-rate version (bucket-4 solve rate at T = 12 minus T = 3 > 0.02): **15 %**.
- **P17e (the control).** Notul's answer-cell accuracy at depth 6 ≥ mask's at depth 12 at
  6,000, paired over boards: **70 %**. Notul solve rate (all buckets) ≥ mask's: **70 %**.
- **P17f (cost).** Mask ≤ 4.0 h wall clock for 6,000 steps (23.4k tokens/step at E16's
  11.6k tok/s ≈ 2.0 s/step + 24 evals): **65 %**; notul ≤ 3.0 h: **65 %**; both peaks under
  24.0 GB: **80 %**.
- **P17g (the hinge).** Mask `loss/gain_est` mean over steps 1,000–6,000 under 0.9 AND the
  penalty nonzero on under 50 % of steps: **40 %** (E16: 0.8997 and 80.5 %).
- **P17h (the curve).** Notul held-out `val/loss` at 6,000 between 0.60 and 1.20 nats/token:
  **60 %** (the puzzle-cell floor ≈ 0.53 plus the solution cells). Its answer-cell accuracy at
  6,000 exceeds its 3,000 value by more than 0.05 (still learning): **60 %**.

## Binding

- P17d TRUE on either arm ⇒ the loop CAN think when the data demands it: text and math were
  the wrong rulers, and the arc reopens on Sudoku with the grid as the instrument (targets,
  write-back, the depth draw). The slot-loop lane's closure from E16 is lifted for Sudoku only.
- P17c and P17d FALSE on both arms while P17b's first clause is TRUE (the model learns, not
  through depth) ⇒ the architecture does not iterate on any data tried: the carrier / the
  write-back is the lever (the Spiral schedule, the paid loop), not the target or the data.
  The slot-loop lane stays closed.
- P17b FALSE on notul (accuracy at the ~0.28 copy floor) ⇒ the run could not distinguish
  H17 from H17′: filed under `failures/` as a protocol failure, with the next planned run
  (a longer budget, or a loss mask on the puzzle cells) written before anything else runs.
- P17e FALSE (mask ahead of the plain model, paired) ⇒ the first time a TUL arm beats its
  control on anything; the next run is the seed pair.
- P17a FALSE on mask ⇒ its trip step, `loop/core_gain_t0` and `loss/gain_est_max` go into the
  divergence README's second-hold table.
- NO 20k run from this experiment; Wolfe's call.

## Not verified before launch

The 6,000-step path (only 12-step smokes ran); steps/s at this geometry; the sweep at 3,000
docs (the build checked `--docs 8` on a smoke checkpoint; a bootstrap-CI NaN at n = 8 in
`lab/divergence/_stats.py` is a pre-existing small-n artifact); whether 6,000 steps × 128
boards gets an autoregressive LM off the copy floor on Sudoku-Extreme at all (HRM's 27M
recurrent model needed ~10 h on a laptop GPU); the augmentation's near-duplicate rate
(exact duplicates are 0).

## Results

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
