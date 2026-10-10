# LXTUL + pointer: credit assignment by loss term, part and path

Status: measured 2026-10-10. Predictions were written before the run (unchanged below).

## Question

Wolfe: "What is the credit assignment for each of the prelude, the loop, and the coda?"
Checkpoint: `lxtul-pointer-rerun/step_5000.pt` (config `lxtul_pointer`, the same-seed rerun of
the winner; K1-K6 +0.0095 on the sweep rows). Hypothesis: the loop gets little, diffuse credit
from the coda's token CE, and most of its training signal comes from the span decoder (the
no-span-decoder arm lost 0.065 nats CE and K1-K6 fell +0.010 -> +0.002,
`lab/experiments/failures/2026-10-10-lxtul-fast-no-spandec-5k.md`).

## Predictions (written before the run)

- P1. Loop-core column: the span decoder's projection share is above 0.5 and the token CE's
  is below 0.3.
- P2. At the loop exit cells, the span decoder's gradient norm is at least 2x the token CE's.
- P3. The token CE's gradient at the prelude output (token positions) is at least 10x its
  gradient at the loop exit cells.
- P4. The fixed-point term and the gain hinge each hold under 0.2 of the loop column
  (projection share), unless the hinge binds on these batches.
- P5. Ablations on 480 trainer-val rows: zeroing the loop's write costs more than +0.10
  nats; K1-K6 is between 0 and +0.02; pointer off costs +0.1 to +0.3; the coda reading token
  embeddings instead of prelude states costs more than +1 nat.

Refutation of the hypothesis: the token CE's projection share in the loop-core column is equal
to or larger than the span decoder's (P1 fails in that direction), or the token CE's gradient at
the exit cells is equal to or larger than the span decoder's (P2 fails in that direction).

## Verdict

The hypothesis holds on this checkpoint. The span decoder, not the coda's token CE, is the
main trainer of the loop.

- P1 held. Loop-core column, projection share: span decoder +0.69 (sd 0.27), token CE +0.20
  (sd 0.15). The span decoder's share is larger than the CE's on 12 of 12 batches. On one
  batch the CE's share is negative (-0.145): the CE pulled the loop against the column total.
  The cell norm and the slot seed + register show the same split (0.69 / 0.31 and 0.64 / 0.36).
- P2 held. At the loop exit cells the span decoder's gradient is 4.4x the CE's (0.0289 vs
  0.0066). On the winner cell alone, the only cell the coda reads, it is still 2.2x
  (0.0144 vs 0.0066). The three loser cells get gradient from the span decoder only (0.025).
- P3 held, at the edge. The CE's gradient at the prelude output (token positions) is 10.7x its
  gradient at the exit cells (0.0709 vs 0.0066).
- P4 held for the pass this run trained with (pass 6): fixed point 0.00, hinge +0.07 (from one
  batch, penalty 0.004). It FAILS under HEAD's per-step hinge draw; see "The gain hinge" below.
- P5: three of four held. Write zeroed +0.158, K1-K6 +0.0102, coda tokens read embeddings
  +1.37. Pointer off cost +0.439, above the predicted +0.1 to +0.3.

## What each part gets, in words

- Prelude: 91 % of its gradient (projection share) comes from the token CE, 10 % from the span
  decoder. Only 6.5 % of the CE's prelude gradient arrives through the loop exit; the rest
  comes straight down the coda's token path. Its worth as the coda's token carrier is large:
  replacing the prelude states with the token embeddings at the coda's token positions costs
  +1.37 nats (an upper bound; the model never trained that way).
- Loop (core blocks + injection + core x0 injections): the span decoder 0.69, token CE 0.20,
  repulsion 0.04 (but 0.24 of the norm: a large component nearly orthogonal to the total),
  fixed point 0.00, hinge 0.07 (one batch). ALL of the CE's loop gradient enters through the
  winner cell (fraction via exit 1.0000; via loser cells 0). The span decoder reaches the loop
  through all four cells, and the part that comes through the three LOSER cells has 2.6x the
  norm of the part through the winner (0.19 vs 0.074). The coda never reads the loser cells at
  the exit.
- The loop's worth: zeroing the write costs +0.158 nats; shuffling it across slots +0.148, so
  nearly all of the write's worth is span-specific. Depth is worth +0.0102 (K1-K6), and the
  curve is flat from depth 3 on (K3-K6 +0.0006): passes 4 to 6 buy nothing at eval.
- Coda: all of its gradient is token CE (no other term reaches it). `W_prefix`, the write
  into the coda, is also 100 % token CE. The pointer head is 100 % token CE and is worth
  +0.44 nats.
- Readout (lm_mixer + final_norm): CE 0.94, span decoder 0.06 (the decoder reads `z` through
  `_readout`). The tied table's head use is 100 % CE (the decoder's head is detached); its input
  use is CE 0.92, span decoder 0.08.
- Router and latent head: router CE 0.93, latent WTA 0.07. Neither reaches the loop
  (`fan_lsel_head_input: detached`, router inputs detached): the latent-selected loop's terms
  add no loop credit. The online floor (`lsel_enc`) was 0 on every batch.

So the answer to Wolfe's question: the coda's token CE trains the prelude, the coda, the write
and the pointer almost alone; the loop gets about one fifth of its gradient from the token CE,
through one cell per slot, and about two thirds from the span decoder, mostly through cells the
coda does not read. This agrees with the no-span-decoder arm losing most of its K1-K6.

## The gain hinge (a finding outside the question)

The hinge reads the slot map's gain at ONE pass per forward. On this checkpoint the reading is
1.8 to 2.1 at passes 0 and 1 (penalty 80 to 146 at lambda 100) and 0.82 to 0.87 at passes 3 to
7 (penalty 0). This run trained under the pre-e19234c2 rule (one fixed pass, pass 6 on seed 1),
so the hinge never charged passes 0 to 2. HEAD (e19234c2, 2026-10-10) draws a fresh pass per
step. The first grad run of this probe used HEAD's draw (`grad_headdraw.json`): on the 4 of 12
batches that drew pass 0 or 1, the hinge's loop-core gradient norm was 230 to 770 against about
0.25 on the other batches, and its projection share was 1.00. Under `grad_clip 1.0` such a step
is the hinge alone. Any continuation of a pre-e19234c2 checkpoint on HEAD meets this. Not
verified: the fp32 map gain at passes 0 and 1 (`core_map_fd.py`); the numbers above are the
model's own bf16 finite difference in this eager build.

## Results

### Gradient credit matrix (12 train batches x 6 rows, train skip 5000 batches, hinge pass pinned to 6)

Term values (mean over batches): `token_ce` 4.287, `fan_repel` -0.3882, `fp` 0.002098, `gain_reg` 0.0003411, `spandec` 4.706, `fan_lsel.lsel_lat` 0.2696, `fan_lsel.lsel_enc` 0, `fan_lsel.lsel_router_ce` 1.237

Entry = projection share mean (spread = sd over batches) / norm share mean. Projection shares sum to 1 per row; norm shares sum to 1 per row. Rows with no gradient from a term show `.`.

| group | params | total norm | `token_ce` | `fan_repel` | `fp` | `gain_reg` | `spandec` | `fan_lsel.lsel_lat` | `fan_lsel.lsel_enc` | `fan_lsel.lsel_router_ce` |
|---|---|---|---|---|---|---|---|---|---|---|
| embed_input | 100.68M | 0.382 | +0.92 (0.06) / 0.78 | +0.00 (0.00) / 0.03 | -0.00 (0.00) / 0.00 | . | +0.08 (0.06) / 0.19 | . | . | . |
| lm_head_tied | 50.35M | 0.396 | +1.00 (0.00) / 1.00 | . | . | . | . | . | . | . |
| prelude | 45.65M | 2.47 | +0.91 (0.07) / 0.80 | -0.00 (0.01) / 0.02 | -0.01 (0.00) / 0.01 | . | +0.10 (0.07) / 0.18 | . | . | . |
| slot_seed_register | 4.20M | 0.121 | +0.36 (0.14) / 0.37 | -0.00 (0.01) / 0.07 | -0.00 (0.00) / 0.01 | . | +0.64 (0.14) / 0.55 | . | . | . |
| loop_core | 68.48M | 0.294 | +0.20 (0.15) / 0.21 | +0.04 (0.12) / 0.24 | +0.00 (0.00) / 0.01 | +0.07 (0.24) / 0.05 | +0.69 (0.27) / 0.49 | . | . | . |
| cell_norm | 0.00M | 0.00689 | +0.31 (0.11) / 0.35 | +0.00 (0.00) / 0.04 | -0.00 (0.00) / 0.02 | . | +0.69 (0.12) / 0.59 | . | . | . |
| router_latent_head | 2.23M | 0.36 | . | . | . | . | . | +0.07 (0.02) / 0.22 | . | +0.93 (0.02) / 0.78 |
| W_prefix | 4.19M | 0.0256 | +1.00 (0.00) / 1.00 | . | . | . | . | . | . | . |
| coda | 45.65M | 0.59 | +1.00 (0.00) / 1.00 | . | . | . | . | . | . | . |
| pointer | 0.53M | 0.651 | +1.00 (0.00) / 1.00 | . | . | . | . | . | . | . |
| span_decoder | 27.83M | 1.3 | . | . | . | . | +1.00 (0.00) / 1.00 | . | . | . |
| readout_mixer_norm | 1.05M | 0.225 | +0.94 (0.05) / 0.81 | . | . | . | +0.06 (0.05) / 0.19 | . | . | . |

Additivity (max over batches of |sum of term grads - total| / |total|, clip off): embed_input 3.6e-03, lm_head_tied 0.0e+00, prelude 2.9e-03, slot_seed_register 4.0e-03, loop_core 1.3e-02, cell_norm 9.2e-04, router_latent_head 0.0e+00, W_prefix 2.3e-03, coda 3.0e-04, pointer 0.0e+00, span_decoder 0.0e+00, readout_mixer_norm 0.0e+00

Cotangent clip (slot_cot_clip 4.0), clipped vs unclipped total, min over batches: embed_input ratio 1.0000 cos 1.00000, lm_head_tied ratio 1.0000 cos 1.00000, prelude ratio 1.0000 cos 1.00000, slot_seed_register ratio 1.0000 cos 1.00000, loop_core ratio 1.0000 cos 1.00000, cell_norm ratio 1.0000 cos 1.00000, router_latent_head ratio 1.0000 cos 1.00000, W_prefix ratio 1.0000 cos 1.00000, coda ratio 1.0000 cos 1.00000, pointer ratio 1.0000 cos 1.00000, span_decoder ratio 1.0000 cos 1.00000, readout_mixer_norm ratio 1.0000 cos 1.00000

### Gradient arriving at each tensor (L2 norm, mean (sd) over batches)

| term | prelude_tokens | prelude_slots | exit_all | exit_winner | exit_losers | written_winner_cell | spandec_input |
|---|---|---|---|---|---|---|---|
| `token_ce` | 0.0709 (0.006) | 0.000253 (3e-05) | 0.00662 (0.0006) | 0.00662 (0.0006) | 0 | 0.00662 (0.0006) | 0 |
| `fan_repel` | 0.000226 (1e-05) | 2.36e-05 (2e-06) | 0 | 0 | 0 | 0 | 0 |
| `fp` | 4.6e-05 (2e-05) | 3.04e-06 (7e-07) | 0 | 0 | 0 | 0 | 0 |
| `gain_reg` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `spandec` | 0.00261 (0.0002) | 0.000601 (8e-05) | 0.0289 (0.003) | 0.0144 (0.001) | 0.025 (0.002) | 0 | 0.0577 (0.005) |
| `fan_lsel.lsel_lat` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `fan_lsel.lsel_enc` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `fan_lsel.lsel_router_ce` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### VJP from the loop exit back to the parameters

frac = <VJP(exit -> group), g_term> / |g_term|^2: the part of the term's gradient in the group that enters through the exit cells (1 = all of it).

| term | group | via winner cell | via loser cells | term norm | frac via exit |
|---|---|---|---|---|---|
| `token_ce` | embed_input | 0.0665 | 0 | 0.366 | 0.0396 |
| `token_ce` | lm_head_tied | 0 | 0 | 0.396 | 0.0000 |
| `token_ce` | prelude | 0.357 | 0 | 2.27 | 0.0645 |
| `token_ce` | slot_seed_register | 0.0547 | 0 | 0.0547 | 1.0000 |
| `token_ce` | loop_core | 0.0994 | 0 | 0.0994 | 1.0000 |
| `token_ce` | cell_norm | 0.00308 | 0 | 0.00308 | 1.0000 |
| `token_ce` | W_prefix | 0 | 0 | 0.0256 | 0.0000 |
| `token_ce` | coda | 0 | 0 | 0.59 | 0.0000 |
| `token_ce` | pointer | 0 | 0 | 0.651 | 0.0000 |
| `token_ce` | readout_mixer_norm | 0 | 0 | 0.217 | 0.0000 |
| `spandec` | embed_input | 0.0479 | 0.0915 | 0.116 | 0.9999 |
| `spandec` | prelude | 0.206 | 0.384 | 0.523 | 1.0000 |
| `spandec` | slot_seed_register | 0.0337 | 0.0692 | 0.0908 | 1.0000 |
| `spandec` | loop_core | 0.0736 | 0.19 | 0.239 | 1.0000 |
| `spandec` | cell_norm | 0.0021 | 0.00426 | 0.00552 | 1.0000 |
| `spandec` | span_decoder | 0 | 0 | 1.3 | 0.0000 |
| `spandec` | readout_mixer_norm | 0 | 0 | 0.0569 | 0.0000 |

### Worth by ablation (480 trainer-val rows, eval, paired)

Anchor: per-token CE of batch 0 4.43856, the labelled forward's ce_tokens 4.43856.

| arm | CE | delta vs full [95% CI] |
|---|---|---|
| full | 4.0354 | +0.0000 [+0.0000, +0.0000] |
| write_zeroed | 4.1933 | +0.1579 [+0.1533, +0.1629] |
| write_shuffled | 4.1835 | +0.1481 [+0.1424, +0.1540] |
| depth_1 | 4.0456 | +0.0102 [+0.0097, +0.0108] |
| depth_2 | 4.0446 | +0.0092 [+0.0086, +0.0097] |
| depth_3 | 4.0360 | +0.0006 [+0.0005, +0.0008] |
| depth_4 | 4.0355 | +0.0001 [+0.0000, +0.0002] |
| depth_6 | 4.0354 | -0.0000 [-0.0000, +0.0000] |
| depth_8 | 4.0354 | +0.0000 [-0.0001, +0.0001] |
| coda_tokens_read_embedding | 5.4040 | +1.3686 [+1.3521, +1.3849] |
| pointer_off | 4.4748 | +0.4394 [+0.4094, +0.4712] |
| full_again | 4.0354 | -0.0000 [-0.0000, +0.0000] |

K1-K6 = +0.0102 [+0.0097, +0.0108] (eval depth 6).

### Gain hinge reading per loop pass (train mode, target 0.9, lambda 100.0)

| batch | pass 0 | pass 1 | pass 2 | pass 3 | pass 4 | pass 5 | pass 6 | pass 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.836 / 88.03 | 2.088 / 141.30 | 1.122 / 9.40 | 0.871 / 0.00 | 0.839 / 0.00 | 0.839 / 0.00 | 0.831 / 0.00 | 0.830 / 0.00 |
| 1 | 1.791 / 79.59 | 2.107 / 145.86 | 0.944 / 0.27 | 0.845 / 0.00 | 0.842 / 0.00 | 0.850 / 0.00 | 0.842 / 0.00 | 0.843 / 0.00 |
| 2 | 1.792 / 79.82 | 2.030 / 128.33 | 1.008 / 1.38 | 0.850 / 0.00 | 0.857 / 0.00 | 0.858 / 0.00 | 0.834 / 0.00 | 0.824 / 0.00 |
| 3 | 1.809 / 82.69 | 2.080 / 139.29 | 0.992 / 0.99 | 0.860 / 0.00 | 0.851 / 0.00 | 0.847 / 0.00 | 0.841 / 0.00 | 0.845 / 0.00 |


## Method notes

- Checkpoint `/home/wolfe/morph-wt-graph/checkpoints/morph/lxtul-pointer-rerun/step_5000.pt`,
  config `lxtul_pointer` at worktree HEAD 1f271b2c, eager build (`load_ckpt`, no compile),
  `tg_scoped_kernels` on as in the config.
- Train batches: the trainer's own train stream fast-forwarded 5000 batches (the batches step
  5001+ would read), 12 batches x 6 rows. Train mode, bf16 autocast, CUDA seed 1000 + batch.
  The hinge is pinned to pass 6 (`slot_gain_pass_force`) to match how this run trained.
- Terms: every `*_weighted` key the forward folded into `out["loss"]`, live tensors from a
  TorchFunctionMode `.detach()` tap, token CE from `_tul_group_losses`. Closure
  `CE + terms == out["loss"]` checked on every batch (the run raises otherwise).
  `fan_lsel` split into its three summands and the split checked against the folded value.
- One forward per batch; one `autograd.grad` per term on the retained graph. The cotangent
  clip is OFF for the per-term gradients (it is not linear in the loss); the trainer's clipped
  total is read separately: the clip did not bind on any batch (ratio 1.0000, cos 1.00000 in
  every group). Additivity of the per-term gradients to the unclipped total: worst 1.3e-2
  relative (loop core), 4e-3 or less elsewhere. Read the 1 % as the run-to-run noise of the
  recomputed (checkpointed, bf16) backward.
- The tied table: the head's use is read through shadow leaves (`lm_head_tied`), checked
  bit-equal to the real table before the run.
- Group membership: `x0_injects.i` with its own layer's section; `input_norm` with the
  prelude; `tul.E_mask` (the token-dropout replacement on the coda input) with the coda.
  `prelude_tokens` path numbers include the register's pooling of the span's prelude token
  states into the loop's cells, not only the coda's token path.
- Worth: eval mode, 80 batches x 6 rows of the trainer's val stream (skip 50,000 docs), the
  label-free forward's pointer-mixture log-probs, per-row sums, paired bootstrap (2000
  resamples). `full_again` after every restore equals `full` (delta 0.0000).

## Reproduce

```
D=/some/dir; C=.../lxtul-pointer-rerun/step_5000.pt
python lab/divergence/credit_matrix.py --part data  --batches-file $D/batches.pt --train-skip 5000 --train-batches 12 --val-batches 80
python lab/divergence/credit_matrix.py --part grad  --ckpt $C --batches-file $D/batches.pt --n-batches 12 --gain-pass 6 --out $D/grad.json
python lab/divergence/credit_matrix.py --part grad  --ckpt $C --batches-file $D/batches.pt --n-batches 12 --gain-pass -1 --out $D/grad_headdraw.json
python lab/divergence/credit_matrix.py --part hinge --ckpt $C --batches-file $D/batches.pt --n-batches 4 --out $D/hinge.json
python lab/divergence/credit_matrix.py --part worth --ckpt $C --batches-file $D/batches.pt --n-val 80 --depths 1,2,3,4,6,8 --out $D/worth.json
python lab/divergence/credit_matrix.py --part report --results-dir $D
```

`grad_headdraw.json` was made by an earlier version of the script that booked `tul.E_mask` with
the slot seed group; only its loop-core column and the hinge rows are quoted above, and those
do not depend on that grouping. HEAD's draw is seeded from `torch.initial_seed()`, so a rerun
of that line draws different passes.

## Not verified

- One checkpoint, one seed, step 5000. Shares at other steps or on the 10k checkpoints are
  not measured.
- Gradient norm is not the optimizer's update: AdEMAMix normalises per coordinate, so a term
  with a small norm can still move parameters. The matrix says who sends gradient, not who
  moves the weights.
- The compiled training forward was not used; the eager build's bf16 numerics differ slightly.
- The hinge's pass-0/1 gains are not checked in fp32 (`core_map_fd.py`).
- The coda-reads-embedding arm is an upper bound of damage on an untrained configuration.
