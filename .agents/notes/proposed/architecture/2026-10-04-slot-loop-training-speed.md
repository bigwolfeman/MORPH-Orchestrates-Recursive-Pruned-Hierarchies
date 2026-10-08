# Agent Note: slot-loop training speed, passes 1 and 2

Status: proposed

Date: 2026-10-04. Pass 1: branch `wt-perf` from master `3f107d0`, committed as `83c7df8`.
Pass 2: the same worktree from `83c7df8` (section "Pass 2" below). Arm:
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

## Pass 2 (2026-10-04, from `83c7df8`)

Pass 2 had three asks, in order: a memory curve so the run can leave at least 4 GB free on
a desktop card; the launch overhead; and the ranked items left by pass 1. Same protocol
(420-step bench, 20-step deterministic gate, 300-step deterministic overlay for anything
that is not bit-identical).

### Memory: the `ckpt_grad_iters` curve

`model.ckpt_grad_iters=k` checkpoints the first `k` of the eight loop passes. The slot loop
runs every pass on the full cell axis (inactive slots are masked, not dropped), so every
pass holds the same memory: each checkpointed pass frees about 1.02 GiB and costs about
7 to 8 ms. Bit-identical at every `k` (checkpointing is exact; gate g0_ck0 and pass 1).

Bit-identical recipe (`training.ademamix_fused_fp32=true`, plus the pass-2 probe change at
`k = 0` only, worth 1.5 ms):

| k | ms/step | tok/s | peak alloc GiB | peak reserved GiB |
|---|---|---|---|---|
| 0 | 468.7 | 13110 | 23.99 | 24.33 |
| 1 | 477.4 | 12870 | 22.96 | 23.29 |
| 2 | 484.9 | 12671 | 21.94 | 22.27 |
| 3 | 491.4 | 12502 | 20.92 | 21.23 |
| 4 | 501.2 | 12258 | 19.90 | 20.22 |

Opt-in recipe (the above plus `training.compile_blocks=true`,
`training.compile_core_dynamic=false`, `model.ce_softmax_kernel=true`,
`model.ce_compact_rows=true`; not bit-identical, see below):

| k | ms/step | tok/s | peak alloc GiB | peak reserved GiB |
|---|---|---|---|---|
| 0 | 388.1 | 15831 | 21.66 | 21.85 |
| 1 | 395.3 | 15541 | 20.74 | 20.93 |
| 2 | 402.0 | 15283 | 19.81 | 20.00 |

Free memory = 31.84 GiB (the card) - 0.69 GiB (the process's non-PyTorch memory, read off
the pass-1 OOM message) - the desktop - peak reserved. For at least 4 GiB free:

- desktop at 4.5 GiB (its idle reading today): bit-identical `k = 2` (4.38 GiB free,
  484.9 ms), or the opt-in recipe at `k = 0` (4.80 GiB free, 388.1 ms);
- desktop at 6.0 GiB: bit-identical `k = 4` (4.93 GiB free, 501.2 ms; `k = 3` leaves 3.92),
  or the opt-in recipe at `k = 1` (4.22 GiB free, 395.3 ms).

### Changes

| # | Change | Files | Parity | ms/step | Note |
|---|---|---|---|---|---|
| 7 | The gradient probe reads its ~45 device scalars in one copy (`train._host_floats`) | `train.py`, `tests/test_host_floats.py` | gate identical; the probe log (146 keys x 20 steps) is byte-identical to the `83c7df8` tree's | 470.2 -> 468.7 | inside the noise |
| 8 | `training.compile_core_dynamic` (default true = the tree): false compiles the core's modules with static shapes | `train.py`, `base.yaml` | gate identical on the bit-identical recipe | 468.7 -> 465.9 | the win is with compile_blocks: 407.6 -> 388.1 with every opt-in on |
| 9 | `model.ce_compact_rows` (default false): the CE's vocab GEMMs run on labelled rows only, one host sync per CE call | `fused_ce.py`, `transformer.py` (`self._ce_kw` at all 11 CE calls), `train.py`, `base.yaml` | NOT bit-identical; fp64 bounds in `tests/test_ce_softmax_grad.py` | 435.5 -> 422.0 (on top of row 6) | 50.5 % of the span decoder's 12288 rows and 84 % of the token CE's 7680 carry loss |
| - | `training.compile_blocks=true` (existing key), re-measured on this tree | none | NOT bit-identical | 468.7 -> 456.5, and 2.15 GiB less memory | inductor keeps fewer saved activations |

All opt-ins together at `k = 0`: 388.1 ms, 15831 tok/s, 1.60x over master's 620.9 ms. The
bit-identical recipe at `k = 0`: 468.7 ms, 1.32x.

Overlays (300 steps, deterministic, each against the bit-identical recipe `ov_a`):

| Run | first differing step | step-0 loss rel. diff | step-0 grad-norm rel. diff | mean loss diff, steps 200-299 | steps 280-299 |
|---|---|---|---|---|---|
| CE kernel (`ov_b2`) | 2 | 0 | 4.3e-4 | +0.0053 | -0.0060 |
| CE kernel + compact rows (`ov_c`) | 2 | 0 | 4.3e-4 | +0.0251 | +0.0114 |
| compile_blocks (`ov_d`) | 0 | 1.2e-4 | 1.8e-3 | +0.0283 | -0.0204 |
| all opt-ins, static core (`ov_e`) | 0 | 1.2e-4 | 1.6e-3 | +0.0262 | -0.0070 |
| noise: the bit-identical recipe, non-deterministic (`ov_n1`) | 2 | 0 | 7.4e-8 | +0.0109 | +0.0031 |
| noise, second draw (`ov_n2`) | 2 | 0 | 7.4e-8 | +0.0191 | +0.0139 |

How to read it: the two noise rows run the SAME code as `ov_a` without deterministic
algorithms, so they show how far a run drifts from `ov_a` by chaos alone: +0.011 and +0.019
over steps 200-299. The opt-in rows sit at +0.005 to +0.028, with the 280-299 window
changing sign for three of the four. So no opt-in moves the loss by more than about
0.01-0.02 nats beyond what chaos alone does, and a 300-step overlay cannot resolve a
smaller bias. A paired long run is the open check (Risks). compile_blocks starts differing
at step 0 (loss 1.2e-4 relative): inductor fuses the block's elementwise chain, so its
roundings differ from eager's. The CE paths start at step 2.

### Launch overhead

Kernels per step and GPU idle, torch.profiler over 4 steps (the profiler inflates host
time, so the idle numbers are upper bounds):

| Profile | wall/step (profiled) | GPU busy | kernels/step | idle in gaps 50 us to 1 ms | idle in gaps < 50 us |
|---|---|---|---|---|---|
| master (`prof/base`) | 767 | 496 | 35 000 | 50 | 167 |
| pass 1, row 3 (`prof/cur`) | 584 | 442 | 28 660 | 26 | 113 |
| bit-identical recipe (`prof/exact2`) | 523 | 418 | 28 500 | 16 | 88 |
| + compile_blocks, dynamic core (`prof/cb`) | 507 | 388 | 21 840 | 69 | 48 |

Where the launches are on the bit-identical recipe (per step): the backward 15 800 (55 %),
the forward blocks 7 700 (27 %; 72 block calls: prelude 4, EMA twin 4, core 6 x 8, hinge
6 x 2, coda 4), and the slot loop's own per-pass tail (`_tul_core` body, `_lsel_pass`,
the cell norm, the fixed-point term) about 1 500 (5 %, about 190 per pass). A cast census
(profile with shapes, `prof/shapes.json`) finds 4 080 `_to_copy` per step, mostly
fp32 -> bf16 activations inside the eager attention ([6, 8, 256, 64] q/k/v, [6, 256, 1024]),
not weights.

What this means for the three options the coordinator listed:

- **Compile the per-pass small-op tail.** Not built: the tail is 5 % of the launches, so
  removing all of it is worth at most a few ms.
- **CUDA-graph the per-pass core step.** The slot loop needs no bucketing (its pass shapes
  are already static). Tried as `mode="reduce-overhead"` on the compiled core blocks
  (scratch patch `core_ro_patch.py`): it fails with
  `torch.utils.checkpoint.CheckpointError: Recomputed values ... different metadata` in
  the gain hinge's checkpoint, and cudagraph trees warn "pending, uninvoked backwards"
  (re-recording, no fast path) because the same block is replayed ten times in one
  forward. A graph of the per-pass step needs a hand-built static-buffer capture with the
  hinge's checkpoint restructured. Not done; it is the main item left.
- **Compile the blocks with static core shapes.** Done (rows 8 and compile_blocks): with
  the dynamic guards gone the compiled core stops being host-bound, -22 ms.

### Tried and dropped in pass 2

- **Ternary STE once per step** via `torch.nn.utils.parametrize.cached()` around the forward
  (scratch `pcache_patch.py`): -16 ms but +0.92 GiB peak (the cached weights live through
  the backward), and NOT bit-identical. The grad norm matched master for all 20 gate steps,
  but the loss differs from step 2 on and the final state hash differs. The cause was not
  found, so it is not shipped. **Cause found 2026-10-07** (graph-step task 2.3): parametrize
  keys its cache by the id of the module the property was injected on, held in a closure
  (`torch/nn/utils/parametrize.py`, `key = (id(module), tensor_name)`), and a deepcopy keeps
  the injected class. The fan target's EMA twin deep-copies the parametrised prelude, so
  inside `cached()` the twin read the LIVE prelude's quantised weights. The twin equals the
  live prelude until the first nonzero-lr step, hence step 2. Proof: the same probe with the
  twin's read uncached is byte-identical to the uncached baseline over 20 deterministic
  steps (`lxtul_pointer_graph`, graph_step off). The replacement is `model.ternary_step_cache`
  ([graph-step note](2026-10-07-graph-captured-training-step.md), task 2.3).
- **CUDA graphs** (reduce-overhead): see above.
- **HC fused backward / compiled MLP backward tuning**: not started (time).

## Alternatives considered

Pass 1:

- **Compile the transformer blocks** (`compile_blocks`) on top of row 1, on master: 524 ms
  against 530 ms, inside the noise, and not bit-identical. Dropped in pass 1. Pass 2
  re-measured it on the sync-free tree, where it IS worth 12 ms and 2.15 GiB (the host was
  no longer the bottleneck), and 22 ms more with static core shapes.
- **Run the gain hinge's two extra core applications without checkpointing** (a key
  `model.slot_gain_ckpt` was built and gated bit-identical): it saves about 20 ms but adds
  about 2 GB, and on top of row 1 the run went out of memory at 25.6 GB allocated. It buys
  the same ms per GB as `ckpt_grad_iters` (about 10 ms per GB) and gives no new trade, so
  the key was removed before this note.
- **Make the per-step gradient probe lazy** (read its numbers one step late). It would let
  the host queue the optimizer before the probe's reads, but the takeover guard saves its
  emergency checkpoint BEFORE the optimizer step, and a late read would change what that
  checkpoint holds. Not done; pass 2 batched the reads instead (row 7).
- **Drop the GradScaler under bf16.** Not done: it changes the skip-on-inf behaviour and
  the scale path, so it is not the same training.

Pass 2: the parametrize cache, CUDA graphs and the per-pass tail compile ("Tried and
dropped in pass 2" and "Launch overhead" above). For the CE row count, a host-known count
from the layout (no sync) was considered and not built: it must reproduce the span
decoder's valid mask exactly on the host, and the one sync it would save sits in front of
a GEMM-bound loop where it costs little.

## Acceptance criteria

- Rows 1 to 5, 7 and 8 are bit-identical to master: gate files byte-identical over 20
  steps (met for each row; row 7 also has a byte-identical probe log).
- Rows 6 and 9 are at least as close to fp64 as the eager CE body on every test case
  (met), and every opt-in's 300-step overlay sits within about 0.01-0.02 nats of the noise
  runs (met at 300 steps; a longer paired run is the open check).
- The off path of each new key is master: `training.ademamix_fused_fp32`,
  `model.ce_softmax_kernel`, `model.ce_compact_rows` default to false and
  `training.compile_core_dynamic` to true; `tests/test_fused_ce_sync_free.py` pins the
  eager CE body bit for bit with and without autocast.
- Run lines for this arm (pick `k` from the memory tables for the desktop's load):
  - bit-identical: `python -m morph.training.train --config-name tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank_cnorm model.ckpt_grad_iters=<k> training.ademamix_fused_fp32=true`
  - fastest (not bit-identical): add `training.compile_blocks=true training.compile_core_dynamic=false model.ce_softmax_kernel=true model.ce_compact_rows=true`.

## Risks

- **Memory.** At `k = 0` the bit-identical recipe reserves 24.33 GiB, which leaves about
  2.3 GiB with the desktop at 4.5 GiB. The tables above give the `k` for a 4 GiB margin.
  They are 420-step peaks; a longer run draws the same depth (Poisson mean 6, max 8, the
  table's max is 8 on almost every batch), so the peak should not grow, but a 10k run was
  not watched.
- **Host shadows** rely on the autograd version counter to notice an in-place write. A
  write through `.data` or through a numpy view would not bump it. No code in the tree
  does that to a layout mask today; a future one would read a stale shadow.
- **Opt-in numerics** (CE kernel, compact rows, compile_blocks) were checked over 300 steps
  only. A paired run of 5k steps at two seeds is the real test; it was not run.
- **compile_core_dynamic=false on a shrinking core.** A core that changes its batch per
  pass (the plain Parcae path) would see one static graph per shape; under the trainer's
  `eager_on_recompile` stance new shapes run eager. Keep the default there.
- **fp32 optimizer kernel and NaN.** On a NaN input `clamp` in ATen returns NaN and Triton's
  `maximum` may not. A NaN in the optimizer state means the run is already dead, but the
  two paths could then write different garbage.
- **Lazy fan readings** assume train.py reads the train-step fan values only on its
  20-step log line. A new reader that logs them on other steps would see them missing.

## What is left, ranked by estimated payoff

None of these was built.

1. **A captured per-pass core step** (static buffers, one CUDA graph per pass shape, the
   hinge's checkpoint restructured so it does not recompute inside a graph). The core's
   forward and backward are 6 x 10 block calls per step, and with compile_blocks the
   profiled idle in `_tul_core` and the hinge is about 57 ms per step (profiler-inflated).
   Estimate 15 to 30 ms. Bit-identity depends on the dropout RNG under capture.
2. **Fused HC backward kernels** (`_FusedHCPreMapBackward` and `_FusedHCPostBackward`,
   about 52 ms of GPU per step together on the bit-identical recipe) and the compiled MLP
   backward: block-size tuning, perhaps 10 to 20 ms. Not bit-identical if a reduction
   order changes.
3. **Why the parametrize cache diverges.** If the cause is benign, caching the ternary
   weights is -16 ms; it also costs +0.9 GiB, so it competes with the memory margin.
4. **Gain hinge without checkpointing** when memory allows: about 20 ms for about 2 GB.
5. **The eager attention's fp32 <-> bf16 activation casts** (4 080 `_to_copy` per step):
   keep q/k/v in bf16 through the TG-restricted branches. Not bit-identical.

## 2026-10-07 update: the MORPH-vs-Parcae round

[`lab/experiments/mixed/2026-10-07-morph-vs-parcae-speed.md`](../../../../lab/experiments/mixed/2026-10-07-morph-vs-parcae-speed.md)
measured the winner arm (`lxtul_pointer`) against the Parcae testbed at equal FLOPs. The
step is host-serial (forward + autograd CPU time = wall, about 20 us per launch, 31.3k
launches against Parcae's 8.8k), so this note's ranking changes: launch count is first. Item 1
above (a captured per-pass core step) stays first; a new second item is to register the HC,
CCA and conv Triton dispatchers as `torch.library` custom ops, because their
`torch.compiler.disable` wrappers cut every compiled MORPHBlock into fragments (60 graph
breaks in 12 steps). Swapping them for their torch references removes the breaks but is
slower (492.8 and 519.5 ms against 444.7).
