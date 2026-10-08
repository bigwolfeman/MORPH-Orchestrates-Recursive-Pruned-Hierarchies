# Agent Note: the eager HC reference rounded the residual skip to bf16; HC now runs its kernels on CUDA

Status: implemented

## Problem

The Hyper-Connection residual (`morph/model/hyper_connections.py`) has two
implementations. The fused Triton kernels (`morph/kernels/triton/fused_hyper_connection.py`)
are what `model.hc_use_kernel: true` runs. The pure-PyTorch references in the same file
were described as the "bit-faithful" reference arm. They were not.

Under bf16 autocast with an fp32 carrier, three reference ops ran in bf16:

1. `hc_post_reference` and `hc_pre_reference` computed `Hres·h` and `Hpre_cm·h` with
   `torch.einsum`, which dispatches to `bmm`, which autocast runs in bf16. That rounds
   `Hres` to bf16 (a diagonal entry of 0.99995 becomes 1.0, so the mixer is no longer
   orthogonal) and stores the skip term of the fp32 carrier in bf16 at every HC op. On
   real inputs captured from the winner, `x_mix` is 2.4e-3 rel-RMS off an fp64 evaluation
   of the definition on the eager path and 4.4e-8 on the kernel path.
2. `cayley_orthogonal` and `_cayley_ref` ran their two 4×4 matmuls in bf16 under
   autocast (2e-3 on the HC output once item 1 was fixed).
3. The mapping GEMV (`F.linear` under autocast) ran its backward in bf16. The kernel's
   backward GEMMs are fp32 (1e-3 on the carrier grad once items 1 and 2 were fixed).

The effect on a run is not roundoff. On `lxtul_pointer` at step 0 (same instance, batch
and RNG; compile off; deterministic), the kernel path's loss is 24.315407 and the eager
path's is 26.955385 (+10.9 %). The whole gap is one term, the slot gain hinge
(`model.slot_gain_lambda` 100, target 0.9, finite difference at `slot_gain_eps` 0.02):
`gain_est` reads 0.905 on the kernel path and 1.063 on the eager path, and
`gain_reg_weighted` reads 0.0031 and 2.6415. Bf16 rounding noise in the two applications
of the map adds `(noise/eps)²` to the measured gain. Splitting the cause on the same
batch: rounding only `Hres` moves the loss by +2.4e-4 relative; storing only the skip in
bf16 moves it by +5.2e-2.

The reference was not only a test reference. `MORPHTransformer` calls
`set_force_eager(not use_kernels and not tg_scoped_kernels)`, and every HC dispatcher
checked `force_eager()`. So every run with `model.use_kernels: false` and no
`tg_scoped_kernels` trained the bf16 skip. The einsum reference dates from a18e4515 and
ad8d2b2d (2026-06-05). Affected:

- The 47 configs that resolve that way at HEAD: `budget_root`, `budget_web_full`,
  `budget_web_reach1`, `budget_web_reach1_coda`, `budget_web_reach1_coda_s2`,
  `budget_web_reachall`, `budget_web_span`, `budget_web_span_s2`, `tul_a2s`,
  `tul_cap_c1`, `tul_db_cond`, `tul_dbfix`, `tul_g0c0`, `tul_g1`, `tul_g2`, `tul_gl1`,
  `tul_gl1_ctrl`, `tul_gl1b`, `tul_gl1bc`, `tul_gl1c`, `tul_ilv50`, `tul_l1`, `tul_l2`,
  `tul_l2cap_cond`, `tul_l2trunc`, `tul_l3`, `tul_l3wake`, `tul_l3wake_cap`,
  `tul_m12_mask_g102`, `tul_m12_mask_g102_rn`, `tul_m12_mnext_mask`,
  `tul_m12_mnext_mask_mtp4`, `tul_sac`, `tul_sacg`, `tul_tg1`, `tul_tg2`, `tul_tg2c64`,
  `tul_tg3`, `tul_tg3b`, `tul_tg4a`, `tul_tg4b`, `tul_to_mnext_mask`,
  `tul_to_mnext_y2_mask`, `tul_to_mnext_y2_mask_d16`, `tul_to_mown_mask`, `tul_w1`,
  `tul_w2`. A config's composition on the day it ran may have differed.
- Every TG arm before 2026-09-01. TG arms ran `use_kernels: false` until
  `tg_scoped_kernels` (5825dc97, 2026-09-01) put HC back on its kernels.
- Every `training.deterministic: true` gate on a config without `tg_scoped_kernels`.
  Deterministic mode requires `use_kernels: false` (`train.py`), so those gates ran the
  reference.

It also corrects a misattribution.
[`2026-09-08-slot-gain-eps-noise-bias.md`](../../proposed/bug-fix/2026-09-08-slot-gain-eps-noise-bias.md)
measured that the eager path reads the slot gain 0.07 high at eps 0.02 and attributed the
noise to "the eager attention's own bf16 rounding". On the winner, switching only HC
between kernel and reference reproduces a gap of this kind with every attention kernel
held fixed, so the eager HC reference is a sufficient cause. Whether the eager attention
added part of the 2026-09-08 reading is not measured.

The decode engine had a related defect. `StaticDecodeEngine._hc` (`morph/inference/engine.py`)
still launched `_hc_premap_fwd_kernel` with the signature from before the
2026-07-02 bias-under-rms fix (9fbe1dea): no `pb_ptr`, so every later pointer sat one slot
early and the launch raised `TypeError: missing 'nw_ptr'`. It also added the bias inside
the GEMV, before the `/rms` divide. The wide-carrier path (C > 2048,
`fused_decode_step._hcw_map_kernel`) ran and was wrong: bias before `/rms`, and the old
inverse-free fixed-point Cayley iteration that training removed on 2026-07-02 (6.6e-2 on
the HC output against the training module in the test).

## Decision

- **HC ignores the global `force_eager`.** The four dispatchers (`hc_pre`, `hc_post`,
  `hc_pre_map`, `hc_pre_map_fold`) no longer check it, so `model.use_kernels: false`
  runs the HC kernels on CUDA. A comment at each says why. On CUDA the references run only
  on the explicit opt-ins `model.hc_use_kernel: false` and `MORPH_HC_FORCE_EAGER=1`
  (debugging), and on CPU.
- **The references are exact to fp32 roundoff.** `hc_pre_reference` and
  `hc_post_reference` run with autocast off in the carrier dtype, as the kernels do.
  `cayley_orthogonal` and `_cayley_ref` run with autocast off. The mapping GEMV goes
  through `hc_map_gemv`, which under autocast keeps the kernel's contract (bf16 inputs
  and result forward, fp32 backward; `_HCMapGemv`). Without autocast every reference op is
  the op from before, bit for bit, so CPU pins are unchanged. The eager branch of
  `HyperConnectionResidual.forward` now calls `hc_pre_map_reference`, so one reference
  implements the eager mapping.
- **The decode engine computes the training function.** `_hc` runs a bias-free GEMV and
  passes the fp32 bias as `pb_ptr` (table `self._hcb`). `_hcw_map_kernel` adds the bias
  after `/rms` and uses the exact `_cayley_tile_YM`.
- `base.yaml` no longer calls the eager path "bit-faithful".

Tests: `tests/test_hc_kernel_vs_eager.py`. Kernel vs eager under autocast, forward and
every gradient, at the winner's shapes (6×1280 and 6×256 at C 1024) and edge shapes:
worst 4.7e-7. Both paths vs fp64 without autocast: at most 1.8e-7. A CPU test that the
references ignore autocast. `use_kernels: false` on CUDA still calls the HC kernels.
`StaticDecodeEngine._hc` equals the module (C 768, 1024 and 4096).

## Alternatives considered

- **Fix the reference and keep the `force_eager` routing.** That would make
  `use_kernels: false` runs correct but keep a second HC implementation in training
  service on every eager arm, with its own compile and speed behaviour. Wolfe's call
  (2026-10-08): take the eager HC path out of training service on CUDA.
- **Make `model.hc_use_kernel: false` raise on CUDA.** That was the fallback if the fixed
  reference was not exact to fp32 roundoff. It is (4.7e-7 worst), so the opt-in stays for
  debugging.
- **A full-fp32 reference GEMV.** Simpler, but it is not the kernel's function: the
  kernel's mapping GEMV is bf16 under autocast, so the two paths would differ at 1e-3 in
  the mappings.
- **Explicit multiply-adds instead of `einsum` for the 4×4 mix, to be safe if TF32 is
  enabled.** Not done. TF32 is off in the trainer and the test process
  (`allow_tf32 False`, matmul precision `highest`, printed in both), and the remaining 2e-3
  traced to the Cayley matmuls under autocast, not TF32. Rewriting the einsum would also
  change the CPU bits that pinned tests read. If anything enables TF32 later,
  `test_hc_kernel_vs_eager.py` fails first.

## Consequences

- Runs that used the eager reference cannot be reproduced on new code. A
  `use_kernels: false` config now trains the HC kernels, and the fixed reference no
  longer matches the old one. Their logged numbers (loss, `gain_est`, the hinge) belong
  to the bf16-skip function. Their slot gain hinge read high on noise.
- On a `use_kernels: false` arm the HC kernels now sit behind `kernel_fence`, so a
  compiled eager arm graph-breaks at each HC call instead of fusing the reference.
  Not measured.
- Step-0 loss on `lxtul_pointer` with `model.hc_use_kernel: false` after the fix:
  24.316305, against 24.315407 on the kernels (+3.7e-5 relative; a 1-ulp change in one
  RMSNorm moves this loss by 4.6e-5). Before the fix: 26.955385.
- `lab/divergence/hc_jpmhc_audit_check.py` still calls `HyperConnectionResidual._mappings`;
  `_mappings` uses the same fixed GEMV and Cayley.
