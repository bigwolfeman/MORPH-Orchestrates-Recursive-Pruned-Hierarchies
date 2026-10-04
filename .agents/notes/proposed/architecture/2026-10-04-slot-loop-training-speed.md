# Agent Note: slot-loop training speed, pass 1

Status: proposed

Date: 2026-10-04. Branch `wt-perf` from master `3f107d0`. Arm:
`tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank_cnorm` (batch 6, seq 1024,
the lead slot-loop arm). Scratch tools, logs and profiles: `/home/wolfe/morph-scratch/perf/`
(private, not in the repo).

## Problem

The lead arm trained at 620.9 ms per step (9896 tok/s) on the 5090. A training run of this
arm is 10k to 20k steps, so the step time sets how many arms a night can hold.

How it was measured. One protocol for every number in this note: `scripts/bench.sh TAG 420`
(420 steps, eval and checkpoints off, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
the GPU lock held, the CPU otherwise quiet). The step time is the wall clock between the
logged step 199 and step 419, divided by 220. Two runs of the same code read 475.4 and
477.7 ms, so a difference under about 3 ms is noise.

The baseline profile (torch.profiler over 4 steps, `prof/base.json`; the profiler adds
CPU time, so its wall clock is 767 ms, not 621):

| What | ms per step |
|---|---|
| GPU busy | 496 (65 % of the profiled wall) |
| GPU idle, gaps over 1 ms | 53 |
| GPU idle, gaps 50 us to 1 ms | 50 |
| GPU idle, gaps under 50 us | 167 |
| kernels per step | 35 000 |
| backward, outside the recompute | 195 |
| forward blocks (prelude, core, coda, EMA twin) | 87 |
| span decoder and its CE | 62 |
| recompute of the checkpointed loop passes | 45 |
| token CE | 37 |
| optimizer step | 23 |
| epiplexity term (fp64) | 16 |

Two findings set the plan. First, the GPU sat idle for a third of the step. A sync map
(`torch.cuda.set_sync_debug_mode`, two steps) found about 190 host syncs per step: 20 in
the backward (boolean-mask indexing), about 60 in the forward (`float()` on stats,
`bool(mask.any())`, `x[mask]`, a pageable batch copy, the CE's `n_valid.item()`) and about
110 after the backward (the per-step gradient probe and the clip). Each sync in the forward
or backward empties the launch queue, so the GPU waits for the host. Second, the loop's
eight passes are checkpointed and recomputed in the backward, and the arm used only
16 GB of the card.

## Proposal

Six changes. Each is separable by file and by hunk. Five are bit-identical to master; one
(the CE kernel) is not and sits behind a key that is off by default.

| # | Change | Files | Parity | ms per step | tok/s |
|---|---|---|---|---|---|
| 0 | master | | | 620.9 | 9896 |
| 1 | `model.ckpt_grad_iters=0` (an existing key, run override) | none | bit-identical, gate below | 531.9 | 11552 |
| 2 | forward sync removal | `fused_ce.py`, `transformer.py`, `train.py`, `data_placement.py`, `tul_layout.py` | bit-identical | 516.1 | 11905 |
| 3 | host shadows and lazy fan readings | `host_shadow.py` (new), `transformer.py`, `tul_fan.py`, `attention.py`, `tul_layout.py`, `train.py` | bit-identical | 495.9 | 12390 |
| 4 | one-pass fp32 optimizer kernel, `training.ademamix_fused_fp32=true` | `ademamix_fp32_kernel.py` (new), `ademamix_b1zero.py`, `optimizer.py`, `base.yaml` | bit-identical | 476.5 (2 runs) | 12892 |
| 5 | CE operand casts hoisted out of the chunk loop | `fused_ce.py` | bit-identical | 470.2 | 13067 |
| 6 | CE softmax-gradient kernel, `model.ce_softmax_kernel=true` | `kernels/triton/ce_softmax_grad.py` (new), `fused_ce.py`, `transformer.py`, `train.py`, `base.yaml` | NOT bit-identical, bounds below | 435.5 | 14109 |

Each row includes the rows above it. Rows 1 to 5 together: 1.32x. All six: 1.43x
(620.9 to 435.5 ms). Peak memory: 15.89 GB allocated on master; 23.80 to 23.99 GB
allocated and 24.04 to 24.33 GB reserved from row 1 on (row 1 is the whole increase).

What each change does:

1. **No loop checkpointing.** The arm's activations fit, so the eight loop passes keep
   their activations instead of being recomputed. The key already existed and is exact.
   This is an override on the command line, not a config edit: the cnorm config's own test
   pins that it changes one key from its parent.
2. **Forward sync removal.** The fused CE keeps its normaliser on the device and
   reproduces ATen's division by a Python number (multiply by the fp32 reciprocal) bit for
   bit; the CPU keeps the old division. The span decoder's training stats stay 0-dim device
   tensors. `w[idx] = float` became `index_fill_`, `new_tensor(float)` became `new_full`,
   `torch.tensor(float, device=)` became `torch.full`. The spectral penalty at lambda 0
   returns an exact 0.0 on the host. The prefetcher pins each batch on its producer thread
   and the trainer copies with `non_blocking=True` (the data region went from 15.2 ms to
   0.4 ms).
3. **Host shadows.** `morph/model/host_shadow.py` records the CPU value of a small device
   mask on the tensor itself, keyed by its version counter, so an in-place write voids it.
   `SlotLayout.to` records `slot_mask` and `slot_valid`; the slot loop takes the depth
   table to the host in the ONE sync it needs anyway (`depths.max()` sets the pass count)
   and derives each pass's finish mask there. With the shadow, `x[mask]` becomes a gather
   at host-known indices (same forward values, same `index_put_` backward), and
   `bool(mask.any())` and the attention's slot count read the host copy. Without a shadow
   every reader takes the old code. The fan's no-grad per-pass readings
   (`fan_stream_cos_t*`, and `fan_epi_t*` / `fan_vol_t*` outside the charged passes) now
   run only on the steps train.py logs (`step % 20 == 0`,
   `MORPHTransformer._train_instruments`); the logged values are the same and no RNG is
   drawn. After this change the step has one host sync before the backward (the depth
   table) and none inside it.
4. **fp32 optimizer kernel.** AdEMAMixB1Zero keeps 32-bit state for the no-decay group
   (110.6 M params: the embedding tables) and for every tensor under 4096 elements. Those
   went through about 25 `_foreach` passes, 20.6 ms of GPU per step. One Triton pass now
   does the same IEEE operations in the same order: fused multiply-add only where ATen
   uses one (`add` with alpha, `addcmul`), the reciprocal multiply where ATen uses one
   (`div` by a scalar), `div_rn` and `sqrt_rn` elsewhere, contraction off. The optimizer's
   GPU time went from 24.3 ms to 6.6 ms.
5. **CE cast hoisting.** Under bf16 autocast each chunk's three GEMMs cast the fp32
   operands again: the [49280, 1024] head weight twice per chunk, 20 chunks per step. The
   casts now happen once per call. Same round-to-nearest cast, same GEMM inputs.
6. **CE softmax-gradient kernel.** Per [1024, 49280] logits tile the eager chunk body made
   about ten passes over an fp32 copy (upcast, two masked column writes, `logsumexp`,
   gather, `softmax`, `scatter_add_`, row-weight multiply, downcast). One Triton kernel now
   reads the bf16 tile twice (an online log-sum-exp, then the gradient) and writes the bf16
   gradient in place. Measured alone at the arm's shapes: the span decoder's CE goes from
   44.0 to 26.4 ms per forward and backward, the token CE from 28.6 to 17.4 ms.

### Parity evidence

The gate: `scripts/gate.sh TAG 20` runs 20 steps in deterministic mode and writes, per
step, the loss and the pre-clip global grad norm as float hex, then a SHA-256 over the
final state dict. Two runs are bit-identical when the files are byte-identical.

Making the gate deterministic needed one fix in train.py: `training.deterministic` called
`torch.use_deterministic_algorithms(True, warn_only=True)`, which left the SDPA
memory-efficient backward non-deterministic, and two runs of master split at step 2. With
`warn_only=False` two runs of master are byte-identical. The trace line also gained the
grad norm, so a change that only touches the backward cannot pass on the loss alone.

| Gate | Config | Result vs master |
|---|---|---|
| g0_ck0 | row 1 | identical, 20 steps, `FINAL_SHA256 cfeb29f4...` |
| g1_cum | rows 1 and 2 | identical |
| g2_shadow | rows 1 to 3 | identical |
| g4_fp32 | rows 1 to 4 | identical |
| g5_cast | rows 1 to 5 | identical |
| ov_a vs ov_b2 | rows 1 to 5, then row 6 on top, 300 steps each | first difference at step 2 |

Row 6 is not bit-identical, so it has two other checks:

- **Against fp64.** `tests/test_ce_softmax_grad.py` computes the loss and both gradients in
  fp64 from the same bf16 logits and asserts the kernel is at least as close to that
  reference as the eager body is (25 % slack plus a small floor), with and without
  autocast, padded vocab, the masked slot id, row weights and ignored rows. At V 49169 the
  measured max errors are equal for both paths to three digits: loss 3e-7 to 8e-7, grad_x
  4e-6 to 8e-6 (max |grad_x| 8e-4 to 1.3e-3), grad_w 4e-6 to 1e-5.
- **300-step overlay**, both runs deterministic, so the only difference is the kernel. The
  loss is identical at steps 0 and 1, and the step-0 grad norm differs by 4.3e-4 relative.
  The runs then decorrelate, as any two MORPH runs do: the per-step loss difference has a
  mean of 0.029 over steps 200 to 299, where the per-step loss itself has a standard
  deviation of 0.83. The mean loss over steps 200 to 299 is 15.4345 against 15.4398
  (+0.0053); over steps 280 to 299 it is 15.2658 against 15.2598 (-0.0060). The sign flips,
  so the overlay shows no bias at this length. It cannot show a bias under about 0.01 nats.

Tests, all run on this tree:

- CPU, the required command (`CUDA_VISIBLE_DEVICES=""`, one core): the six regression files
  plus the four new ones, 231 passed, 48 skipped (the CUDA-only cases). Adjacent files
  (`test_data_placement`, `test_optimizer_resume_hparams`, `test_tg_restrict`,
  `test_tul_fan_epi`, `test_tul_lxfan`, `test_tg_slot_attention_gather`,
  `test_tul_slot_register`, `test_tul_layout`, `test_checkpoint_compat`): 182 passed.
- CUDA, under the GPU lock: `test_ademamix_fp32_kernel`, `test_ce_softmax_grad`,
  `test_fused_ce_sync_free`, `test_host_shadow`: 59 passed.
- Sabotage checks: a two-rounding `addcmul` in the optimizer kernel fails 23 of 25 cases;
  an extra RNG draw in an eager gain-hinge path failed its test (that path was then
  removed, see below).

### The profile after rows 1 to 3

Profiled at row 3 (4 steps, `prof/cur.json`): GPU busy 442 ms per step, 76 % of the
profiled wall; idle in gaps over 1 ms fell from 53 to 1.5 ms per step; kernels per step
fell from 35 000 to 28 700 (the lazy fan readings). Rows 4 to 6 then removed GPU work.

## Alternatives considered

- **Compile the transformer blocks** (`compile_blocks`) on top of row 1: 524 ms against
  530 ms, inside the noise, and not bit-identical. Dropped.
- **Run the gain hinge's two extra core applications without checkpointing** (a key
  `model.slot_gain_ckpt` was built and gated bit-identical): it saves about 20 ms but adds
  about 2 GB, and on top of row 1 the run went out of memory at 25.6 GB allocated. It buys
  the same ms per GB as `ckpt_grad_iters` (about 10 ms per GB) and gives no new trade, so
  the key was removed before this note.
- **Make the per-step gradient probe lazy** (read its numbers one step late). It would let
  the host queue the optimizer before the probe's reads, but the takeover guard saves its
  emergency checkpoint BEFORE the optimizer step, and a late read would change what that
  checkpoint holds. Not done.
- **Drop the GradScaler under bf16.** Not done: it changes the skip-on-inf behaviour and
  the scale path, so it is not the same training.
- **Compact the CE rows to the labelled ones.** About half of the span decoder's 12288 rows
  are ignored but still pay three GEMMs. Not done in this pass; ranked below.

## Acceptance criteria

- Rows 1 to 5 are bit-identical to master: gate files byte-identical over 20 steps
  (met for each row).
- Row 6 is at least as close to fp64 as the eager body on every test case, and a 300-step
  deterministic overlay shows no loss shift beyond the run-to-run noise (met at 300 steps;
  a longer paired run is the open check, see Risks).
- The off path of each new key is master: `training.ademamix_fused_fp32` and
  `model.ce_softmax_kernel` default to false, and `tests/test_fused_ce_sync_free.py`
  pins the eager CE body bit for bit with and without autocast.
- The recommended run line for this arm is:
  `python -m morph.training.train --config-name tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank_cnorm model.ckpt_grad_iters=0 training.ademamix_fused_fp32=true model.ce_softmax_kernel=true`
  (drop the last override for a bit-identical run).

## Risks

- **Memory.** Row 1 takes reserved memory from about 16 to 24.3 GiB on a 31.4 GiB card that
  the desktop uses about 4.5 GiB of. That leaves about 2.5 GiB. A desktop that grows, or a
  second GPU job, will OOM the run. `ckpt_grad_iters=k` checkpoints the first `k` passes
  and is the dial if this bites; it was not measured at intermediate `k`.
- **Host shadows** rely on the autograd version counter to notice an in-place write. A
  write through `.data` or through a numpy view would not bump it. No code in the tree
  does that to a layout mask today; a future one would read a stale shadow.
- **CE kernel numerics** were checked over 300 steps only. A long paired run (a 5k-step
  pair at two seeds) is the real test; it was not run.
- **fp32 optimizer kernel and NaN.** On a NaN input `clamp` in ATen returns NaN and Triton's
  `maximum` may not. A NaN in the optimizer state means the run is already dead, but the
  two paths could then write different garbage.
- **Lazy fan readings** assume train.py reads the train-step fan values only on its
  20-step log line. A new reader that logs them on other steps would see them missing.

## What is left, ranked by estimated payoff

Estimates are from the row-3 profile; none of these was built.

1. **Fused HC backward kernels** (`_FusedHCPreMapBackward` 30 ms, `_FusedHCPostBackward`
   22 ms per step) and the compiled MLP backward (41 ms): kernel tuning, perhaps 15 to
   25 ms. Not bit-identical.
2. **CE row compaction**: run the span decoder's CE only on labelled rows (about 50 % of
   12288) and the token CE without its ignored rows. About 10 to 12 ms. The labelled count
   is host-known from the layout, so it can stay sync-free. Not bit-identical (chunking
   changes the sum order).
3. **Ternary STE once per step for the shared core weights.** The eight loop passes and
   the two hinge applications each re-quantise the same core weights. Unmeasured; measure
   the STE's share first. Bit-identity depends on whether the compiled MLP graph changes.
4. **Gain hinge without checkpointing** when memory allows: about 20 ms for about 2 GB
   (see Alternatives).
5. **Batch the gradient probe's 60 host reads into one** (`torch.stack(...).tolist()`):
   about 2 ms. Bit-identical in the logged values.
6. **Launch overhead.** About 28 700 kernels per step; the gaps under 50 us are most of the
   remaining idle time. Graph capture is blocked by the dynamic depth draw.
