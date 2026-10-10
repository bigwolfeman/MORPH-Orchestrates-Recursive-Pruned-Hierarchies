# Agent Note: the fused CCA prologue rounded every intermediate to bf16; it now rounds once

Status: implemented

## Problem

`fused_cca_prologue` (`morph/kernels/triton/fused_cca_prologue.py`) runs the CCA attention
prologue (qk-mean coupling, q/k RMSNorm, temp, CoPE-RoPE, GQA expand, value-shift cat) on
the training forward and on the cached and graphed LXTUL decoders. Its forward kernels
(`_cca_q_fwd_kernel`, `_cca_k_fwd_kernel`, the row-tiled `_cca_rows_fwd_kernel` and its
`_rope_bf16`) cast to `tl.bfloat16` after every op: the qk-mean, the pre-norm sum, the
normalized value, norm * weight, `exp(temp)`, cos and sin, both RoPE products and their
sum. The casts did not depend on the input dtype. The comments say they copy a
bf16-parameter reference (the file's self-test builds bf16 weights).

The model does not run that reference. Measured at the production widths (8 heads, 4 kv
heads, d_head 64), B 6, S 1280, relative L2 error of q / k against `cca_prologue_reference`
in float64 on the same inputs:

| inputs, params | old kernel | eager reference | new kernel | one-rounding floor |
|---|---|---|---|---|
| fp32, fp32 (the fp32 decoders) | 3.79e-3 / 4.62e-3 | 6.1e-8 / 7.3e-8 | 8.1e-8 / 8.9e-8 | 2.5e-8 |
| bf16, fp32 (training autocast) | 3.87e-3 / 4.67e-3 | 2.63e-3 / 2.71e-3 (fp32 out); 3.11e-3 / 3.17e-3 rounded to bf16 | 1.66e-3 / 1.66e-3 | 1.66e-3 |
| bf16, bf16 | 3.88e-3 / 4.52e-3 | 4.09e-3 / 4.67e-3 | 1.66e-3 / 1.66e-3 | 1.66e-3 |

v is a copy and is exact in every case. "One-rounding floor" is the float64 result rounded
once to the output dtype. So with fp32 inputs the kernel quantized an fp32 model to bf16,
and under training autocast it was less exact than the eager path it replaces: the norm
weights and temp are fp32 parameters, so eager `bf16 * fp32` promotes and runs norm *
weight, temp and RoPE in fp32.

On the decoders this was the fixed logit gap. `lab/perf/bench_lxtul_decode.py --arms none
--check` on `lxtul-pointer-rerun/step_5000.pt`, worst row difference as log-probs against
the eager generator: cached 4.149e-3, graphed 4.084e-3 (the same with `--deterministic`,
where eager repeats are bit-equal).

The backward kernels already recomputed the forward in fp32 with no rounding, so the old
backward was the gradient of a different function from the one the forward computed.

## Decision

The forward kernels compute in fp32 from the load to the store and round once, to the
output dtype, at the store. Loads stay in the input dtype and stores in the output dtype
(`q_lat.dtype`, bf16 under autocast), so memory traffic and the output contract do not
change. The K forward sums the group's q_lat and halves `(qmean + k_lat)`, the order the K
backward recomputes. Backward kernels are unchanged.

`tests/test_cca_prologue_precision.py` holds the contract for both kernel families: fp32
inputs give q / k within 1e-6 relative of float64 and every input grad within 1e-5; bf16
inputs give q / k within 1.05x the one-rounding floor. All 12 cases fail on the old file
(q at 3.7e-3 to 3.9e-3 against bounds of 1e-6 and 1.75e-3).

## Alternatives considered

- **Round in the input dtype** (a constexpr compute dtype equal to the input's): exact for
  fp32 inputs and bit-identical training. Rejected: it keeps the training forward less
  exact than the eager autocast path (3.87e-3 against 2.63e-3), which is the defect, and
  keeps the forward and backward computing different functions.
- **Copy the eager autocast rounding exactly** (round the qk-mean and the pre-norm sum to
  bf16, the normalized value to bf16, the rest fp32, output bf16): reproduces an artifact
  of eager bf16 elementwise ops, and still lands above the floor (3.11e-3 rounded to bf16).
- **Return fp32 q / k under autocast** as the eager path does: would change the dtype the
  attention kernels downstream consume and double the q / k / v traffic. The one-rounding
  bf16 output is already closer to float64 than the eager fp32 output (1.66e-3 against
  2.63e-3).
- **Route the decoders through `cca_prologue_reference`**: fixes decode only, leaves the
  training forward as it was, and gives decode a different code path from training.

## Consequences

- Decoders: the cached / graphed vs eager worst log-prob row gap falls from 4.149e-3 /
  4.084e-3 to 2.45e-5 / 2.45e-5 (tokens equal throughout; same with `--deterministic`).
- Training forward is not bit-identical to the old kernel. One `lxtul_pointer` step-5000
  forward + backward at B 6, L 1280, train mode, bf16 autocast, gain hinge off: loss
  9.2085953 -> 9.2091169 (+5.2e-4), grad norm 3.623045 -> 3.616595, gradient relative L2
  difference 1.51e-2. The ruler for an accepted rounding-only change, the per-row against
  the row-tiled old kernels on the same batch: +5.9e-5 loss, 0.94e-2 gradient. Eval mode
  CE 3.926878 -> 3.926472. Paired per pinned hinge pass (0-7), `gain_est` moves by at most
  0.012, with mixed signs.
- Kernel time at training shapes (bf16, production layouts, median of `triton.testing.
  do_bench`, before -> after): per-row B6 S1280 fwd 538 -> 542 us, fwd+bwd 1228 -> 1240 us;
  tiled B6 S1280 fwd 49.2 -> 49.4 us, fwd+bwd 124 -> 124 us; tiled B6 S256 fwd 22.5 ->
  22.5 us. Within 1 %; the kernels are launch and memory bound.
- `tests/test_cca_prologue_tiled.py` compares the two families against float64 and still
  holds; the module docstring's "Precision" paragraph records the old numbers.
- The module's `__main__` self-test now runs at the production widths with fp32 norm
  weights and temp, and gates against float64 truth with no cosines (it gated against the
  bf16-parameter reference, the less exact path; its end-to-end part imported an
  `ignore/` harness that this tree does not have). Its end-to-end part captures the real
  `_CCABase._cca_project`'s prologue inputs at the call.
- Related: [2026-10-08-eager-hc-reference-bf16.md](2026-10-08-eager-hc-reference-bf16.md),
  the same class of defect in the HC residual's eager reference.
