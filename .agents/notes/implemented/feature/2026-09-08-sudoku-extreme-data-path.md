# Agent Note: Sudoku-Extreme data path for ARC E17

Status: implemented

## Problem

ARC E17 asks whether MORPH's loop earns solve accuracy on hard-rated Sudoku boards as
loop iterations T increase (HRM's `sapientinc/sudoku-extreme` dataset). Before any
training run, the data path needs: pretok shards MORPH's curriculum loader can read, a
held-out shard + sweep jsonl, Hydra configs for the 4 TUL/plain arms, and a GPU smoke
per arm — everything up to the preregistration (`lab/experiments/planned/`, not covered
here).

## Decision

`scripts/sudoku_shards.py` serializes each board as a FIXED-length token list built by
direct token-id construction, never by re-tokenizing text: 9 puzzle rows (9 digit tokens
+ newline each) + `<|A|>` + 9 solution rows (same shape) + `<|/A|>` + EOS = exactly 183
tokens for every board (digits 0-9 are single starcoder2 tokens — verified by encoding
each digit alone — so there is no BPE merge to make the length vary). `<|A|>`/`<|/A|>`
reuse Olympiad-AI's ids (49154/49155, `docs/olympiad-interop.md`) rather than minting new
ones, so `lab/divergence/olympiad_sweep.py` reads a sudoku held-out jsonl with only one
line of code changed (the `solve_rate`/`band_solve_rate` addition below).

`morph/configs/sudoku_data.yaml` pins `data.seq_len` (and the one curriculum stage's
`seq_len`) to 182 = 183 − 1. `MultiSourceCurriculumLoader._fill` draws whole documents
and concatenates them into a buffer, cutting it into `batch_size*(seq_len+1)`-token rows
with the remainder carried to the next call — it has no notion of document boundaries at
cut time. When every document is the SAME fixed length D and `seq_len+1` is a multiple
of D, the buffer is provably a multiple of D between calls (0, then +D repeatedly, never
overshooting a target that is itself a multiple of D), so the cut always lands exactly on
a document boundary and a board never splits across two packed rows. `seq_len=182` is the
smallest value for which this holds. `tests/test_sudoku_shards.py::
test_packing_never_splits_a_board` proves it over 24 boards / 6 batches; its sibling
`test_a_seq_len_off_the_board_length_does_split` is the control showing the SAME loader
DOES split a board at `seq_len=200` (not `183k − 1`).

Augmentation (`augment_board`) follows HRM's recipe: a random 1-9 digit relabeling, an
independent random permutation of the 3 row-bands (and, within each band, its 3 rows),
the same for column-stacks, and a coin-flip transpose. Each step is individually
validity-preserving; `is_valid_solution`/`puzzle_consistent` check this on 200+40×5
augmented samples in tests, and the production build's 0-duplicate, 0-leak counts
(the shipped build: 1,000 puzzles × 1,000 augmentations = 1,000,000 train boards, 0 exact-sequence dups, 183M tokens; 3,000 holdout boards, 0 found in train; a 200-augmentation build gave the same zeros)
are consistent with a correct, sufficiently diverse augmentation.

Rating buckets use edges `[0, 1, 11, 23, 38]`, chosen from `test.csv`'s quintiles
(bucket fractions 14.5/24.9/20.3/19.4/20.9%), encoded as Olympiad-style
`"<bucket>.<rating>"` stage strings so `lab/divergence/olympiad_sweep.py`'s
`band = int(stage.split('.')[0])` needs no change.

`lab/divergence/olympiad_sweep.py` gained `solve_rate` (fraction of docs whose answer
region is entirely correct at a depth) and `band_solve_rate` (same, per rating bucket),
computed from the per-doc correct/total answer-token arrays it already tracked — no new
forward pass, every existing key unchanged (14 insertions, 1 deletion). Verified against
a live checkpoint (`--docs 8`, see Consequences).

Four arm configs (`tul_sud_mask`, `notul_sud`, `tul_sud_mnext`, `tul_sud_a1`) each
compose `[<oly-arm-twin>, sudoku_data, _self_]`, mirroring the shipped Olympiad panel
(`tul_oly_*` / `notul_oly` / `oly_data.yaml`) with the data mixin swapped. Composing them
surfaced a real bug, not just a smoke-scale artifact: the mean-12 arm lineage
(`tul_m12_* → … → tul_a2`) sets `tul.max_slots: 64`, tuned for `tul_a2`'s `seq_len 1024`
(≈16 tokens/span on web text). At `seq_len 182` that gives `L_total = 182 + 2·64 = 310`
when a sudoku board needs only 19 slots (18 newline-ended rows + the EOS boundary), and
`tul.prefix_project`'s per-slot matmul OOMed at `micro_batch 32` on a 32 GB card (tried to
allocate 8 GiB with 2.53 GiB free). `sudoku_data.yaml` sets `tul.max_slots: 0`, which
falls back to the spec's own default `seq_len // 8 = 22` — comfortably above the 19
slots actually used (confirmed by the smoke's `span_spans=19.0156` eval metric) — fixing
the OOM for the real 6000-step run, not only the 12-step smoke.

## Alternatives considered

- **Re-tokenize the board as decoded text** instead of constructing token ids directly.
  Rejected: BPE could in principle merge adjacent digit characters into a different
  token depending on context, which would make the board length non-constant and break
  the seq_len=182 no-split proof. Direct id construction sidesteps this entirely.
- **Mint new sudoku-specific `<|A|>`/`<|/A|>` ids** instead of reusing Olympiad's.
  Rejected: the task explicitly asks to reuse the same ids so the sweep script's
  hardcoded `ANSWER_OPEN`/`ANSWER_CLOSE` constants work unchanged; new ids would need a
  sweep-script edit for no benefit.
- **A seq_len larger than 182** (e.g. 512, matching the Olympiad panel). Rejected: any
  seq_len that is not `183k − 1` for some integer k lets the carry-split packer land
  mid-board (proven by the control test), which would put the model's `<|A|>`/answer
  region in a truncated context on some fraction of rows — exactly the ambiguity T2 was
  built to rule out. 182 is the smallest, cheapest such value.
- **Leave `tul.max_slots` at the inherited 64.** Rejected once measured: it OOMs the
  mask arm's real forward at the config's own `micro_batch: 32`, so it would have failed
  the actual 6000-step run, not just this smoke.
- **All 3.8M train puzzles, lightly augmented**, instead of 1,000 puzzles × up to 1,000
  augmentations. Rejected per the arc note's own design point (`lab/experiments/planned/
  2026-09-04-loop-contribution-arc.md`, E17 row): 1,000 puzzles × HRM augmentation is the
  frozen design; the script supports `--num-aug` up to 1000 (probed at 5 puzzles × 1000
  aug = 5,000 unique boards, 0 dups) without needing to touch the puzzle count.

## Consequences

`scripts/sudoku_shards.py` built the shipped shards in 1 min 10 s (`--num-aug 1000`, 2026-09-08 09:50): train (1,000,000 docs, 183M tokens, 0 dups, 366 MB; a first 200-augmentation build gave 200,000 docs,
36.6M tokens, 0 dups, 47 MB) and holdout (3,000 docs, 549k tokens, 0 leaked into train,
1.9 MB + a 2.5 MB `eval_holdout.jsonl`), under `/home/wolfe/morph-scratch/data/sudoku/`.
All 4 arm configs compose cleanly (`hydra.compose`) and passed a 12-step GPU smoke each
(mask 59s/20.07GB, notul 52s/15.60GB, mnext 66s/19.82GB, a1 54s/19.54GB; 0 NaNs, exit 0
on all four); every smoke checkpoint directory was deleted after use. The sweep's new
`solve_rate`/`band_solve_rate` fields were exercised end to end against the mask arm's
smoke checkpoint (`--docs 8`) and returned sane values (0.0 solve rate on a
near-random 12-step model; `None` for a band with zero sampled answer tokens, no
divide-by-zero). `tests/test_sudoku_shards.py` (22 tests) plus the full suite (911
passed, 1 skipped, 1 xfailed) both pass; `ruff` and `black` are clean on the new files.
Not covered here: the preregistration (predictions, method) and the actual training
launch — both are the orchestrator's next step, per the task split.
