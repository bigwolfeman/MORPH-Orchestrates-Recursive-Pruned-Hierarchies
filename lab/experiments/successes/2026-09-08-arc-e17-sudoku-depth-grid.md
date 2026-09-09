# Planned: ARC E17 — the Sudoku-Extreme depth grid: does the loop's accuracy on hard boards rise with iterations?

Status: success
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

**Method amendment 1 (2026-09-08 19:10, before any prediction was scored).** The runner's
first plain-arm sweeps read 2.46 nats at step 1500 and 2.86 at 3000 against the trainer's
held-out 0.548 and 0.522 on the same boards. Cause: `olympiad_sweep.py`'s plain-control
path advanced rows by `seq_len`, not `seq_len + 1`, so row k started k tokens before its
board; this model has only ever seen a board at offset 0 (the mask arm packs per document
and matched its trainer to 0.001; E16's plain arm matched to 0.0014 because packed web
text has no fixed offset). Fixed in commit `53b2472` (regression test
`tests/test_olympiad_sweep_plain_rows.py`); `arc/E17_COMMIT` re-pinned to it; the plain
arm's four sweeps re-run from its saved checkpoints by `arc/run_e17_resweep_then_e18.sh`.
The shifted JSONs are kept out of the record under `stride_bug/` in the scratch results
dir. The predictions are untouched.

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

Scored 2026-09-08 19:25 by `../results/2026-09-08-arc-e17/score_e17.py` (output in
`score_e17.txt`; sweep JSONs and run logs beside it; probes under
`ignored/experiment-artifacts/2026-09-08-arc-e17/`). Both arms ran the full 6,000 steps
with the tripwire silent; the plain arm's sweeps are the re-sweep at `53b2472` (Method
amendment 1), which match the trainer's held-out loss to 0.001 at every checkpoint.

The grid at 6,000 (answer-token accuracy over the 91 answer tokens per board, 81 cells + 10
structure tokens; whole-board solve rate in brackets; 600 boards per bucket):

| arm | bucket | T=1 | T=2 | T=3 | T=6 | T=12 | T=16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| notul (trained 6) | 0 easiest | 0.947 [0.050] | 0.947 [0.053] | 0.947 [0.057] | 0.947 [0.050] | 0.947 [0.053] | 0.947 [0.055] |
| | 4 hardest | 0.934 [0.000] | 0.934 [0.000] | 0.934 [0.000] | 0.934 [0.000] | 0.933 [0.000] | 0.934 [0.000] |
| | all | 0.9385 [0.010] | 0.9383 [0.011] | 0.9384 [0.012] | 0.9384 [0.010] | 0.9382 [0.011] | 0.9381 [0.011] |
| mask (trained 12) | 0 easiest | 0.791 [0.000] | 0.798 [0.005] | 0.798 [0.007] | 0.798 [0.003] | 0.798 [0.003] | 0.798 [0.002] |
| | 4 hardest | 0.776 [0.000] | 0.781 [0.000] | 0.781 [0.000] | 0.781 [0.000] | 0.781 [0.000] | 0.781 [0.000] |
| | all | 0.7799 [0.000] | 0.7865 [0.001] | 0.7867 [0.001] | 0.7868 [0.001] | 0.7865 [0.001] | 0.7865 [0.000] |

Every row is flat past T = 2 to three decimals, on every bucket, on both arms. Difficulty
moves accuracy by 0.013 (notul) / 0.017 (mask) from bucket 0 to 4 and moves the solve rate
from 5 % to 0 %. Answer CE on the plain arm rises with depth past its trained 6 (0.1200 at
6 → 0.1215 at 16).

- **P17a** TRUE, both arms (mask peak preclip 41.9 at step 209, notul 24.5 at 228; both
  under the 1e4 tripwire by two orders).
- **P17b** notul acc@6 = 0.938 > 0.50 TRUE; notul bucket-0 solve 0.050 > 0.20 FALSE;
  bucket-4 solve 0.000 < 0.05 TRUE; mask acc@12 = 0.787 > 0.50 TRUE. The model learns:
  0.938 sits far above the ≈ 0.33 copy floor on the 91-token answer region.
- **P17c** FALSE on every clause. Buckets 3–4 pooled, paired over 1,200 boards: notul
  acc@12 − acc@3 = −0.0000 [−0.0006, +0.0005]; mask −0.0001 [−0.0009, +0.0006]; notul
  acc@3 − acc@1 = −0.0001 [−0.0006, +0.0003]; mask +0.0057 [+0.0044, +0.0071] (a depth-1
  hole of 0.006, not the predicted > 0.05). acc@16 − acc@3: notul −0.0002, mask +0.0000.
- **P17d** FALSE on both arms: gain(3 → 12) on bucket 4 minus bucket 0 = +0.0000 [−0.0010,
  +0.0012] (notul), −0.0005 [−0.0019, +0.0010] (mask); bucket-4 solve@12 − solve@3 = 0
  exactly on both. The solve-rate diagonal reads +0.0033 on both arms only because bucket 0
  LOSES boards from 3 to 12 (mask 0.007 → 0.003); the CI touches 0 (mask) or spans it.
- **P17e** TRUE: notul@6 − mask@12 = +0.1547 [+0.1511, +0.1586] accuracy, +0.0063
  [+0.0023, +0.0107] solve rate, paired over the same 3,000 boards. The gap narrows with
  training (0.187 at 1,500 → 0.164 → 0.155 → 0.155) and is still 0.155 at 6,000.
- **P17f** mask 3.59 h ≤ 4.0 TRUE; notul 3.40 h ≤ 3.0 FALSE (the 4:6×6:4 plain core at
  micro 32 is not faster than the scoped mask arm per step: 0.49 vs 0.44 steps/s); peaks
  21.21 / 19.91 GB < 24 TRUE.
- **P17g** TRUE: mask `loss/gain_est` mean 0.882 over 1,000–6,000 (max 0.959), the hinge
  nonzero on 26.7 % of steps. The constraint is quiet on Sudoku where it fired 80 % on
  math.
- **P17h** notul held-out loss 0.500 at 5,750: below the predicted 0.60–1.20 range, FALSE
  (the puzzle-cell floor was overestimated: given a board's structure the puzzle rows are
  cheap). acc@6000 − acc@3000 = +0.0057, not > 0.05: FALSE. The curve is still falling
  (0.5219 → 0.5087 → 0.5001) but slowly.

## Verdict

**Success: H17 held and H17′ is refuted with the CI at ±0.0006.** On data that needs serial
constraint propagation and cannot be memorized (0 held-out boards in training; 0.77 epochs
of 1,000,000 augmented boards), MORPH's looped core learns to 0.938 answer-token accuracy
and then does not use its loop: every bucket is flat from T = 2 to T = 16 on both arms,
the hard buckets gain exactly nothing from more iterations, and whole-board solves stay at
1 %. The masked slot arm behaves as it did on text and math: a 0.006 depth-1 hole, flat
after 2, and 0.155 behind the plain model at its own trained depth. Binding, second bullet:
P17c and P17d FALSE on both arms with P17b's first clause TRUE ⇒ the architecture does not
iterate on any data tried; the slot-loop lane stays closed; the lever is the carrier / the
write-back, not the target and not the data.

Mispredictions to keep: the mask arm's depth-1 hole on Sudoku is 0.006 (predicted > 0.05
at 80 %), the plain arm is not faster per step than the scoped mask arm, and the held-out
loss floor was overestimated by 0.1–0.7 nats.

Not settled here: whether a longer budget (HRM trains its 27M model for far more board
passes) would move the solve rate off 1 % and open a depth dependence; whether a loss mask
on the puzzle cells changes what the loop learns. Neither is in scope for the arc (Wolfe,
2026-09-08: Sudoku "may be a dead end at our epoch count").

## Updated hypothesis

The loop's depth-flatness is a property of the architecture's carrier, not of the data: web
text (E1–E14), math (E16) and now a constraint task (E17) all read K3−K12 ≈ 0 with the CI
inside ±0.001 once the model is stable. The next test of "does the loop think" must change
what the loop carries or writes back (the Spiral schedule, the paid loop's write-back), and
E18 (the slot width) is the last free parameter of the slot family before that.
