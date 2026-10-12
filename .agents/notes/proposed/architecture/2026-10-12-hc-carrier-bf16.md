# Agent Note: HC residual carrier stored in bf16 (model.hc_carrier_bf16)

Status: proposed

## Problem

The Hyper-Connection carrier `[B, S, 4, C]` (Cayley n=4) lives in fp32 between sublayers. The HC
Triton kernels read and write it and run at 84-98 % of DRAM peak, so they are bandwidth bound:
bytes are the cost. `.agents/notes/proposed/architecture/2026-10-08-lxtul-training-speed-plan.md`
section B measured the lever at about 21 ms of a 389 ms step and named the trap: the slot-loop gain
hinge (`_slot_gain_penalty`) is a finite difference at relative size 0.02 of the state, and on a
bf16-stored carrier it reads storage rounding as gain (step-0 total loss 24.32 -> 25.73, CE
unchanged). The only earlier attempt (`e316c05f`, 2026-08-22, never merged) also rounded Hres to
bf16, which the 2026-10-06 HC audit found slightly expansive (sigma_max 1.07).

## Proposal

Key `model.hc_carrier_bf16`, default false (key off is byte-identical to master).

- The region kernels (`hc_region_fused`: `_hcr_*`, `_hc_post_fwd_kernel`) already widen every load
  to fp32 and round only at the store, so they run on a bf16 carrier unchanged: `x_bar`, the exit
  output and the carrier grad take the carrier's dtype. Hres and Hpost_row stay fp32 (the entry
  returns them from the kernel; no `.to(dtype)`). `_HCRegionEntry.backward` computes the 48 x 4C
  projection-weight grad with a bf16 GEMM and an fp32 result (`out_dtype`).
- The carrier is created in bf16: the prelude's stream expand (`_front_tail`, both the
  `inject_fold` and the plain branch), and the slot loop's source and state (`_tul_core`: the
  normed gathered prelude state `xn`, so `e` and `h = core_init(e)`).
- Every op that touches the carrier keeps its dtype: `DiagonalInjection` (`keep_dtype`) and
  `_apply_injection` (now a method) round the fp32 sum back to the carrier's dtype. `ChannelInject`
  and the cell RMSNorm already did.
- The gain hinge (`_slot_gain_penalty`, and `_core_gain_penalty` for the plain loop) widens the
  detached state and the source to fp32 before its two applications; the carrier stays fp32 through
  both (dtype follows the input). `slot_gain_reuse_f0` raises with the key (its reused f(h) is the
  main loop's bf16 output).
- Refusals, no fallback: build raises without `hc_use_kernel` and `hc_region_fused`;
  `_hc_region_active(carrier_bf16=True)` raises at call time without CUDA + Triton, n=4, an fp32 or
  bf16 carrier and bf16 autocast.
- Entry points outside the trainer (2026-10-12): `MORPHTransformer.forward` and every other public
  method that runs the network (`tul_forward_ablated`, `tul_forward_halt`, `tul_forward_cw_arms`,
  `tul_forward_with_plan_nats`, `tul_fm_forward`, the slot-state, attention-lift and MUX probes,
  `fm_eval_probe`, `prelude_states`) carry `@_carrier_autocast`. Key on and autocast off, the
  method enters bf16 autocast itself and so computes the function it trained with. Under any
  autocast it adds nothing (another dtype still raises in the HC gate). No CUDA: a clear
  RuntimeError, no fp32 fallback. Key off: the call is untouched (byte gate unchanged). This
  replaces the autocast wrapper that `train.evaluate` carried; generation, `evaluate`, the
  probes, `code_eval` and `sft` all reach the model through these methods.
- The KV-cached and graphed decoders (`tul_generate_cached._check_supported`) refuse the key:
  they run submodules directly, store the carrier in fp32, and were not checked against the
  bf16 function. Eager `generate_tul` reproduces it.
- Carrier grads are bf16 because autograd matches a gradient to its tensor's dtype. A bf16 carrier
  with fp32 carrier grads is not possible without a custom cast at every sublayer boundary; not
  built, not measured.

## Alternatives considered

- fp32 carrier, fewer passes over it: that is `hc_region_fused` + `inject_fold`, already shipped;
  this key is the next halving on top of it.
- Probe the hinge by JVP instead of widening the state: exact, but the hinge reuses the checkpointed
  core step and its dropout/RNG discipline; widening to fp32 is one cast and keeps the finite
  difference the loop was tuned with.
- Round Hres and Hpost_row to bf16 too (the 2026-08-22 branch): rejected, audit found Hres
  expansive in bf16. Test `test_bf16_carrier_is_the_fp32_carrier_rounded_at_the_store` pins them to
  fp32 roundoff of the fp32 run.
- Cast at every HC exit (`out_dtype` argument) so a missed fp32 site self-heals: rejected, the
  hinge needs the exit to keep fp32 when it gets fp32.
- Keep `x_bar` fp32: not done; `x_bar` leaves the entry kernel in the carrier's dtype and the
  sublayer norm sees a bf16-rounded input (one rounding, the one autocast's first GEMM applies).

## Acceptance criteria

Measured 2026-10-12 on the 5090, recipe `nowta_pointer_fast_spdcells_speed`, scratch
`/home/wolfe/morph-scratch/abq/bf16-carrier/` (REPORT.md has the commands).

- Byte gate, key off: `FINAL_SHA256 4a73ebf5426b9dfa889c81a8608a23d23a6785bb225ff32e2d5dd03476bd70d7`
  (equals master).
- Step 0, key on vs off (real trainer, graph step): hinge `tul/gain_est` 0.8978 vs 0.8980, train CE
  10.6998 vs 10.6993, `loss_total` 21.8638 vs 21.8629. A hinge left on the bf16 carrier reads 0.92
  (+0.04) and moves the loss +0.02 to +0.09 (tiny model, 3 seeds, d 64 and 256).
- Speed, interleaved off/on/off/on, 301 steps, steady-state window (steps 200-300): 138.3 and 138.1
  ms/step off, 127.2 and 128.5 ms/step on (tok/s 44.4k, 44.5k vs 48.3k, 47.8k; about -7.5 %
  ms/step, +8 % tok/s). Reserved memory 21.63 -> 19.02 GB. Allocated peak 3.98 GB both (graph replays
  allocate nothing; the figure reads the eager steps).
- HC Triton kernels, 4-step graph-replay profile: 26.8 -> 17.4 ms/step (952 launches/step both
  ways); GPU busy 212.0 -> 196.8 ms/step (profiler-inflated).
- Quality proxy, 300 steps same seed: train CE 8.573/7.265/6.515 (off, run 1), 8.718/7.238/6.523 (off,
  run 2) against 8.296/7.237/6.532 and 8.562/7.232/6.512 (on) at steps 100/200/300. Off-run spread at step
  100 is 0.15; this is a breakage check, not a verdict.
- Tests: `tests/test_hc_carrier_bf16.py` 13 passed on GPU (kernel fwd/bwd against fp64 with the bf16
  store tolerance, fp32-equivalence at the store, dtypes, refusals, model carrier dtypes with the
  hinge fp32, step-0 gain, evaluate autocast); sabotage checks recorded in its docstring.
- Entry points, 2026-10-12: `tests/test_hc_carrier_bf16.py` 17 passed on GPU. A no-autocast
  `forward` and `tul_forward_ablated` match the autocast call to 1e-5 on the logits (two
  autocast calls of one model already differ by 2.4e-7 from slot-gather atomics, so bit equality is
  not available), the HC carrier reads bf16 in that call, eager `generate_tul` emits 6 tokens on
  the key-on tiny fan model, the cached decoder refuses, no CUDA raises. Sabotage (wrapper made a
  pass-through) fails the no-autocast and generate tests. Real trainer: a 30-step
  `nowta_pointer_fast_spdcells_speed_bf16c` run with `eval_every=15` evaluates (`[VAL 15]`
  prints) with the old evaluate wrapper removed. Byte gate key off:
  `4a73ebf5...bd70d7`; the 120-step deterministic trace with eval every 40 is identical to master.
- Not yet: the orchestrator's 5k quality check (K1-K6, CE, divergence) on the recipe.

## Risks

- Rounding changes the trained function; the 5k check is the verdict. Run-to-run spread at 5k is
  about 0.006 K1-K6 (independent runs).
- The AdEMAMix + ternary spike train and the detonation (`lab/divergence/DIVERGENCE-README.md`) are
  sensitive to the loop map's gain; only 300 steps ran. The hinge reads the map at the same point
  within 3e-3 of gain (tiny model) and 2e-4 (recipe, step 0), but the operating point itself is a
  bf16-rounded state.
- Decorated entry points enter autocast only when none is active. A caller that runs the model
  under fp16 autocast still raises. `code_eval` and `sft` were not run with the key on; they call
  `model(...)`, which is covered by the wrapper, but only `evaluate` (trainer smoke) and
  `generate_tul` (tiny model) were exercised.
- Carrier-sized fp32 copies remain: the hinge's widening, the fan diversity term's `.float()` of the
  cells. aten copy launches rose 1381 -> 1598 per step (+0.8 ms); inductor copy launches unchanged.
- `x_bar` is bf16, so the sublayer RMSNorm input is rounded once more than before.
