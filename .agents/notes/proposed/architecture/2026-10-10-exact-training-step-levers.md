# Agent Note: Exact training-step levers for lxtul_pointer_fast

Status: proposed

## Problem

The shipped fast recipe (`morph/configs/lxtul_pointer_fast.yaml`) runs at 230 ms per step
(26.7k tok/s) at batch 6 x 1280 on the 5090. Wolfe wants MORPH to train clearly faster than
Parcae (23.5k tok/s). The 2026-10-10 profile of one replayed step (8 replay steps from step
61, aligned to a labelled graph_step-off trace) put 242.7 ms of GPU span on 15.8 TFLOP, and
showed work that changes no number the model learns from:

* the token head (`fused_linear_label_logprob`) ran its vocab GEMM twice, once in the
  forward and again in the backward (3.9 ms);
* the int6 embedding parametrization quantised the euclidean table three times per step and
  the bigram table once, as nine eager kernels each (7.7 ms in all), and `lm_weight()` built
  the tied head twice;
* the token head ran its vocab GEMMs on all 7680 rows, 13.3 % of which are slot rows whose
  log-prob no loss reads (about 0.34 TFLOP).

The gain hinge's backward on steps where the hinge is 0 is the fourth lever; it has its own
note: [2026-10-10-hinge-branch-conditional-node.md](2026-10-10-hinge-branch-conditional-node.md).

## Proposal

Four changes, on branch `perf/train-speed-free` (working tree, not committed):

1. **Kept logits (tracker C3).** `fused_linear_label_logprob(keep_logits=True)` saves the
   forward's bf16 logits tiles and the backward reads them instead of recomputing the GEMM.
   Bit-identical (the tiles are the same GEMM output). Cost: 7680 x 49280 x 2 bytes = 0.76 GB
   held from forward to backward (gate: 21.65 vs 21.00 GiB reserved, with E). Only the token
   head of `_tul_group_losses` asks for it. A second backward through kept tiles raises,
   except under CUDA graph capture (the hinge branch records two backwards, a replay runs one).
2. **One quantisation and one head per forward (E).** `MORPHEmbedding.forward_cache()`, opened
   by `_forward_single` around the TUL forward, opens `IntNRowSTE.scope()` on each int-N
   quantiser and caches `HybridEmbedding.lm_weight()`. A global
   `torch.nn.utils.parametrize.cached()` was tried first and is wrong: it also caches the
   ternary weights, whose per-use gradients are bf16 and would then be summed in bf16 before
   the cast back (the 45-step gate moved at step 4). The scoped cache is byte-identical.
3. **One-pass quantiser (Q).** `morph/kernels/triton/int_row_quant.py` computes
   `IntNRowSTE`'s value in one Triton pass with the eager operations in the eager order:
   the scale as a multiply by the fp32 reciprocal of `hi` (ATen's tensor / CPU-scalar path),
   `div_rn`, `rint`, NaN pass-through clamps, floating-point contraction off. Pinned bit for
   bit against `IntNRowSTE.eager` at both table shapes.
4. **Token-head rows (tracker C2).** `model.lmhead_row_cap` runs the head on the rows a loss
   reads (labelled token rows, plus emitting slot rows when `tul.emit_weight` is not 0), packed
   on the device into a fixed row count proven from shapes (`_lmhead_row_cap`: 1140 of 1280
   rows per row at the lxtul shapes). Reduction order changes, so it is a config key, on in
   `lxtul_pointer_fast.yaml`. `static_row_pack` is now the one packer shared with the span
   decoder's `row_cap`.

Pads (brief lever 4): `slot_compact: gather` with the stratified depth already runs pad
cells at pass 0 only, and pass 0 needs capacity 64 per row because a row can hold 64 valid
slots. No exact fixed-shape saving is left there.

## Alternatives considered

* Global `parametrize.cached()` for E: rejected, not exact (above).
* Keep logits in fp32: doubles the memory for no change in the result (the kernel path
  computes from the bf16 tile either way).
* C2 by a host-counted row set (`ce_compact_rows`): one host sync per call, illegal under
  `training.graph_step`.
* C2 bound per batch from `slot_valid`: needs a host read; the shape-only bound costs 1140 vs
  the measured mean of about 1110 labelled rows per row.

## Acceptance criteria

* Byte gate (`/home/wolfe/morph-scratch/hingedraw/gate.sh`, 45 deterministic steps, loss,
  grad norm and seven gain terms in hex, FINAL_SHA256): C3 alone, E alone, C3 + E and Q
  identical to master 1f271b2c. Measured 2026-10-10: C3, E (scoped cache), C3 + E, Q, and
  C3 + E + Q + the hinge branch all identical (trace and FINAL_SHA256).
* C2: step 0 loss identical, grad norm 6.7e-8 relative; over 45 steps the trajectories
  decorrelate (max loss deviation 0.055 nats at loss 24, the early chaotic phase). Error to
  an fp64 reference no larger than the all-rows head's (`tests/test_lmhead_row_cap.py`).
* Bench (`/home/wolfe/morph-scratch/perf/r1007/bench.sh`, 420 steps, wall ms/step over steps
  200-400, two or more runs each, interleaved with master), measured 2026-10-10:

  | Tree | ms/step (runs) | vs master |
  |---|---|---|
  | master 1f271b2c | 245.5, 246.2, 245.6, 245.6 (a first run read 267.1) | |
  | C3 | 241.5, 241.4 | -4.1 |
  | E | 241.5, 241.6 | -4.0 |
  | Q | 239.1, 239.5 | -6.3 |
  | C3 + E | 237.5, 238.1 | -7.8 |
  | C3 + E + C2 | 237.3, 236.6 | -8.6 |
  | C3 + E + Q + C2 + hinge branch | 233.4, 233.6 | -12.1 (25.0k -> 26.3k tok/s) |

  In this window the hinge is active on every logged step, so the hinge branch only costs
  here; its own numbers are in the hinge-branch note. Replay-trace GPU busy per step (8
  steps from step 61): master 244.0, C3 239.6, E 237.6, Q 235.3, all levers 228.2 ms.

## Risks

* C3's 0.76 GB is held across the backward; the 5090 has about 6 GB of slack at this recipe.
* C2's bound assumes every span, the open tail included, is at most `span_cap` tokens (the
  boundary rule cuts at the cap). A batch that beat the bound gives a NaN loss, never a silent
  drop; real packer rows (every span at the cap, every 4 tokens, random, DITTO) stay under it.
* Q depends on Triton's `libdevice.rint` and `div_rn` matching CUDA's `nearbyintf` and IEEE
  division; the tests pin it at the production shapes, not for every possible table.
