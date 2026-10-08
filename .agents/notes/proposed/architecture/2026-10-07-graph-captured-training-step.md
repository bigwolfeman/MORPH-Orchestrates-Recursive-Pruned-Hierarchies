# Agent Note: graph-captured training step, then targeted fused kernels

Status: proposed

## Problem

At equal FLOPs, the winner arm (`lxtul_pointer`) trains at 516 ms per step against Parcae's
277 ms. The step is host-serial: forward-thread CPU plus autograd-thread CPU equals the wall
clock, at about 20 us per launch and 31.3k launches per step, while the GPU is busy for
451 ms (measured: [speed round](../../../../lab/experiments/mixed/2026-10-07-morph-vs-parcae-speed.md)).
torch.compile cannot take the model whole. Wolfe, 2026-10-07: about two weeks of testing
are left, and every run takes twice as long as it should.

## Proposal

Two phases. Phase 1 removes launch cost by capturing the whole training step (forward,
backward, clip, optimizer) as CUDA graphs, with no torch.compile requirement. Phase 2 then
cuts GPU-side work with a few separate fused kernels, each with its own parity test. No
megakernel (Alternatives). Branch: `perf/graph-step` from master `5e913cad`. Every speed
number uses `bench.sh` (420 steps, ms between logged steps 200 and 400, GPU lock, CPU quiet);
every exact change passes the 20-step deterministic gate (`gate.sh`, loss + grad norm hex +
final state SHA-256).

### Phase 0: harness (no model change)

- [ ] 0.1 Sync census of the winner step under `torch.cuda.set_sync_debug_mode("error")`
      with probes off: list every host sync on the step path with file:line. Known now:
      the slot loop's depth table to the host, the gain hinge's pass draw
      (`transformer.py:6435`, CPU RNG `.item()`, no device sync but a Python branch), the
      12 host-shadow gathers (`host_shadow.py` readers), `scaler.step` (`train.py:3293`,
      found-inf read), the CE `ce_compact_rows` row count, the 20-step instrument cadence
      (`train.py:3471`).
- [ ] 0.2 A capture test, `tests/test_graph_step.py` (CUDA, tiny LXTUL + pointer model):
      capture one step, replay it on two different batches, compare loss, every grad and
      the post-step weights with an eager step on the same batches and RNG state. It must
      fail if any listed sync is put back (sabotage check).
- [ ] 0.3 Re-bench the starting point on the branch (L0 516.2, O1 444.7 ms) so the
      phase deltas share a protocol and a tree.

### Phase 1: graph-safe step and capture

- [ ] 1.1 Fixed pass count in training: run `slot_max_depth` (8) passes always, masked
      as today, instead of `depths.max()` from the host. Exact whenever the batch maximum
      is 8, which is almost every batch; measure the fraction over 1000 batches and report it.
- [ ] 1.2 Gain hinge pass: keep the CPU draw (it needs no device sync) and capture one
      graph per drawn pass (8 graphs sharing one memory pool); replay the drawn one.
      Exact. Fallback if memory does not allow it: a device-side one-hot over passes.
- [ ] 1.3 Replace the host-shadow gathers with fixed-shape masked ops (`torch.where`,
      index tensors of fixed length). Exact in forward values; gate the grads.
- [ ] 1.4 GradScaler: replace `scaler.step` / `scaler.update` with a device-side
      found-inf skip inside the optimizer kernel (bf16 needs no loss scaling; the skip-on-
      inf behaviour is kept). Gate it against the scaler path on finite steps.
- [ ] 1.5 Optimizer capturable: learning rate, step count and every scheduled scalar as
      device tensors updated outside the graph; the fused AdEMAMix fp32 kernel reads them.
      `clip_grad_norm_` stays (it is capturable without `error_if_nonfinite`).
- [ ] 1.6 Static batch buffers: ids, labels and every `SlotLayout` tensor copied into
      fixed buffers before replay (the packer already gives fixed shapes). DITTO rows use
      the same buffers.
- [ ] 1.7 Instrument steps (every 20th) and eval run eager, outside the graph; log values
      come from static outputs only on log steps. Record the eager-step share in the run log.
- [ ] 1.8 Capture: warmup on a side stream, `torch.cuda.graph` over forward + backward +
      clip + optimizer, `training.graph_step: true` key (default false = today's path).
      `ce_compact_rows` refuses the key (data-dependent shape).
- [ ] 1.9 Measure: bench L0 and O1 recipes with the key; profile 4 steps; report wall,
      GPU busy and memory. Expected (estimate, not measured): wall falls to about the
      GPU-busy time, O1 about 388 ms.
- [ ] 1.10 A paired 5k run (key on vs off, same seed) on `lxtul_pointer`: CE@6 on the 480
      rows and K1-K6 must sit within seed noise (0.009).

### Phase 2: targeted fused kernels (after Phase 1, once block internals are settled)

- [ ] 2.1 Fused strict-mask attention, forward and backward: bf16 q/k/v in, the
      TG relation computed in the kernel from the layout's index tensors (prelude/coda
      `tg_allow` / `tg_comp_allow` / `tg_seg`, core `tg_relation`), covering both the
      compressed slot branch (`_tg_slot_attention`) and the window branch
      (`_window_fallback`). Today: about 10k launches and about 57 ms per step in the core
      alone, with fp32 casts and a mask rebuilt per call. Parity: fp64 reference test, then
      the gate within kernel tolerance.
- [ ] 2.2 HC pre-map emits the normalised bf16 sublayer input (fold the following RMSNorm
      and the cast into `_hc_premap_fwd_kernel`; eager RMSNorm is 18.4 ms per step).
- [ ] 2.3 Per-step bf16 weight cache for the shared core weights (8 passes cast the same
      fp32 weights again; copy/cast is 59 ms per step on O1). The 2026-10-04 parametrize
      cache diverged for an unknown reason; find that cause first.
      Result (castcache): the 10-04 cause was parametrize's cache key (the id of the module
      the property was injected on, so the deepcopied EMA twin read the live prelude's
      weights). `model.ternary_step_cache` refreshes a bf16 copy once per forward, a compiled
      fill for weights inside compiled blocks (an eager fill differs from Inductor at step 0);
      45-step gate byte-identical; bench 416.1 -> 409.7 ms/step, reserved 20.00 -> 19.60 GB.
      The copy/cast category is activation casts: the weight casts were already fused into
      the in-block quantiser, which was 8.7 ms/step of GPU time.
- [x] 2.4 CE heads through the existing `model.ce_softmax_kernel` (built 2026-10-04);
      its open check is a paired long run, which 1.10's run can carry.
      Result (cehead, 2026-10-08): the key never reached the winner's token loss, which goes
      through `fused_linear_label_logprob` (pointer / DITTO mix the per-row log-prob), so it
      now also runs that function's forward log-sum-exp and backward softmax gradient as
      Triton passes, and both kernel paths accumulate grad_w inside the GEMM (fp32 output).
      Span decoder: the detached head's grad_w GEMM is skipped (always on, exact: 45-step
      gate byte-identical; bench -12.8 ms) and `model.spandec_ce_row_cap` runs its vocab CE on
      the labelled rows at a fixed cap proven from the shapes (1132 of 2048 per row; real max
      1120), NaN on overflow, no host sync. Keys on vs off (ABAB, 420 steps, FAST): 373.0 /
      372.5 -> 324.3 / 323.9 ms/step; graph profile: LM head + CE 49.2 -> 25.3 ms, span
      decoder 53.5 -> 28.3 ms. fp64 tests in tests/test_ce_logprob_kernel.py.
- [ ] 2.5 Re-profile and re-rank what is left before writing any further kernel.

## Alternatives considered

- **A training megakernel** (one persistent kernel for forward and backward). It removes
  launches too, but it must also replace GEMMs that already run at about 151 TFLOPS through
  cuBLAS/cutlass, and every change to the loop, a loss, the masks or HC means re-writing and
  re-validating its schedule and its backward. With two weeks of architecture changes left,
  rejected for training. A forward-only decode megakernel after the architecture freezes is
  a separate, reasonable deploy project.
- **torch.compile of whole blocks.** Measured: the HC / CCA / conv Triton dispatchers'
  `torch.compiler.disable` wrappers cut each block (60 breaks in 12 steps); their torch
  references remove the breaks but are slower (492.8 and 519.5 ms against 444.7).
- **`mode="reduce-overhead"` (cudagraph trees).** Tried 2026-10-04: fails on the gain
  hinge's checkpoint recompute and re-records because one block replays ten times per
  forward. A hand capture of the whole step avoids both.
- **Per-pass graphs instead of a whole-step graph.** Smaller graphs, but the forward glue,
  the losses and the whole backward (302 ms of host time) would stay launch-bound.

## Acceptance criteria

- Phase 1: `training.graph_step: true` passes the gate against the eager path on the
  exact changes (1.2-1.6, 1.8) and reports 1.1's exact-batch fraction; `tests/test_graph_step.py`
  passes and fails under its sabotage; the bench shows the wall within 10 % of the profiled
  GPU-busy time; the 5k pair (1.10) is inside seed noise.
- Phase 2: each kernel passes its fp64 parity test and its own sabotage check, and the
  bench shows a saving; the re-profile (2.5) is filed.

## Risks

- Memory: a captured step holds its pool for the run; 8 hinge graphs must share one pool,
  and the desktop needs about 4 GiB free (2026-10-04 memory tables).
- New code that adds a host sync or a data-dependent shape breaks capture. 0.2's test
  catches it loudly; the eager path stays the default until 1.10 passes.
- Fixed 8 passes changes the step on the rare batch whose maximum depth is below 8.
- Phase 2 kernels that touch block internals (2.1, 2.2) must wait if HC, attention or CCA
  still change.
