# Per-pass gradient probe on the five slot-loop checkpoints (2026-09-10)

Instrument: [`lab/divergence/slot_gradient_probe.py`](../../../lab/divergence/slot_gradient_probe.py).
Test: `tests/test_slot_gradient_probe.py`.
JSON artifacts: `ignored/experiment-artifacts/2026-09-10-slot-gradient-probe/<label>.json`
(one per arm, plus the console log beside it).

The forward instruments (`slot_anatomy.py`, `slot_state_probe.py`) said the core blocks
barely move the slot state and every K-curve is flat. This reads the BACKWARD: how much
gradient reaches each loop pass, from which loss, and how much of the SHARED core weight
gradient each pass contributes.

Command, per arm (run one at a time beside a live trainer):

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python lab/divergence/slot_gradient_probe.py \
  --ckpt <label>=<config>=/home/wolfe/morph-to/checkpoints/morph/<label>/step_5000.pt \
  --rows 12 --batch 3 --depth 6 --out <out>/<label>.json
```

12 packed validation rows, batch 3 (4 batches), every slot forced to depth 6, `model.train()`
with dropout ON, seeded per batch, bf16 autocast — the trainer's arithmetic. ONE forward per
batch and three backwards on it (token CE alone, the weighted MUX alone, the combined loss),
so the three sources read the SAME forward realisation.

## What the probe changes about the training forward

| knob | trained value | probe value | why |
|---|---|---|---|
| `tul.slot_depth_fixed` | 0 (Poisson mean 6) except the fixed-depth arm | 6 | a per-pass table over ragged depths is an average of different things |
| `model.slot_gain_lambda` | 100 on the mux / loop / fixed-depth arms | 0 | `_slot_gain_penalty` applies the core step twice more at one iteration, which would shift the pass counter. The "total" source on those arms is therefore the training loss MINUS the hinge penalty |
| `model.ckpt_grad_iters` | -1 (all grad iterations checkpointed) | 0 | checkpointing re-runs the core forward inside the backward and would tap every weight twice under the wrong pass index. `use_reentrant=False` restores the RNG on recompute, so this is numerically identical |

## Mechanism and its checks

The measurement is a TAP: a `torch.nn.utils.parametrize` parametrization appended to every
core weight, whose forward is an identity and whose BACKWARD banks the incoming gradient
under the pass index the forward ran at. It sits AFTER the ternary STE, and `norm_match` is
a pure straight-through identity (`w + (s*q - w).detach()`), so the tap reads the EFFECTIVE
weight's gradient and the leaf gradient is the same number with no STE correction.

A tap and not module hooks, because the CCA / compressor / indexer projections and the HC
residual mixers do not call their `nn.Linear` forward at all — they read `.weight` and
matmul it themselves. Measured on the tiny CPU model: 8 of 31 core linears fire a forward
hook. The first version of this probe used hooks and would have reported the MLP as if it
were the core.

- **Self-check (mandatory).** `sum_t tap_t == leaf.grad` on every tapped weight. Max
  relative error over all five arms and all sources: **5.55e-07**. It proves coverage and
  that nothing outside the loop touched a core weight; it does NOT re-derive the pass
  index, because tap and counter share one counter.
- **Pass index.** Pinned three ways: `core[0]` must run exactly `--depth` times, every
  weight that carries a gradient must be seen in exactly those passes with the same access
  count each, and `_tul_core`'s own loop index (through `_loop_cot`) must produce exactly
  `--depth` keys. All held on all five arms.
- **Cross-check.** On the 24 layers whose module forward does fire, the first batch also
  computes `dW_t = g_t^T x_t` from forward/tensor hooks and compares it per pass against
  the tap. Max relative error 2.4e-3 on the bf16 path — that is bf16 rounding, not a
  bookkeeping error: the same probe run with `--fp32` on `slot-mux-norm-match` gives
  **exactly 0.0** over the same 144 (layer, pass) pairs. The cross-check tolerance is
  therefore 1e-2 on the bf16 path and the fp32 run is the evidence.
- Nine attention weights per mux/loop arm (`indexer.W_IQ`, `indexer.compressor.W_a*`)
  receive NO gradient at all — the CSA selection is not differentiated. They are listed in
  the JSON as `weights_without_gradient`, not silently dropped.

## Headline readings

1. **The cotangent does not vanish through the loop under the prelude entry.** On four of
   the five arms the per-pass share of the cotangent arriving at the loop state is FLAT:
   0.155-0.21 across six passes, i.e. pass 1 receives 73-96 % of what pass 6 receives.
   Gradient flow through the loop is not the bottleneck on those arms.
2. **The noise entry is the exception and it is a 33x ramp.** `slot-unpack-noise-entry`
   reads 0.016 / 0.030 / 0.058 / 0.119 / 0.253 / 0.523. Under Parcae's entry the early
   passes get almost nothing.
3. **The MUX, not the token CE, pays for the loop.** On `slot-mux-norm-match` the MUX
   backward puts 3.5e-3 into the loop state against the token CE's 4.8e-4 — 7.3x — and the
   combined per-pass weight-gradient profile is the MUX's, not the CE's. The two sources
   reach the SAME passes; they differ in magnitude, not in reach.
4. **Without the MUX the core barely gets a gradient at all.** `slot-loop-norm-match` (A1,
   no MUX): core.mlp 1.4e-2, core.attention 5.5e-3, core.residual 1.9e-2, against prelude
   1.41, coda 1.56, embed 2.03. The shared core weights receive about 1 % of the prelude's
   gradient norm. With the MUX on (`slot-mux-norm-match`) core.mlp rises to 7.0e-1 — a 50x
   difference — while prelude and coda barely move.
5. **The passes fight each other.** `|sum_t dW_t| / sum_t |dW_t|` is 0.52-0.60 on the mux
   arm, 0.34-0.57 on the A1 arm, and **0.20** on `slot-unpack-fixed-depth`. The per-pass
   cosines to the total run from -0.42 to 0.96, negative on several passes of the unpack
   arms and on pass 5 of `slot-unpack-free`'s token CE. Passes are not accumulating one consistent update to the shared map; a large part of what each
   pass asks for is cancelled by another pass.
6. **Clip-through-time is inert here.** `slot_cot_clip = 4.0` bound 0.000 of rows at every
   pass on every arm, so post-clip equals pre-clip everywhere in these tables.

## Per-arm tables


### slot-mux-norm-match (`tul_slot_mux_norm_match`, step 5000)

token CE 4.4787, mux_local 6.7985, combined 11.2803, fixed-point 0.00302. 147 core weights tapped (68.0M params), 138 carry a gradient.

| source | quantity | pass 1 | pass 2 | pass 3 | pass 4 | pass 5 | pass 6 |
|---|---|---|---|---|---|---|---|
| token_ce | cotangent at the loop state (pre-clip) | 4.918e-04 | 4.802e-04 | 4.700e-04 | 4.579e-04 | 4.565e-04 | 5.123e-04 |
| token_ce | its share | 0.171 | 0.167 | 0.164 | 0.160 | 0.159 | 0.179 |
| token_ce | core weight grad \|dW_t\| | 4.917e-02 | 2.414e-02 | 2.253e-02 | 2.535e-02 | 3.041e-02 | 4.787e-02 |
| token_ce | its share of the norm sum | 0.246 | 0.121 | 0.113 | 0.127 | 0.152 | 0.240 |
| token_ce | cos(dW_t, dW_total) | 0.550 | 0.485 | 0.462 | 0.643 | 0.734 | 0.671 |
| mux | cotangent at the loop state (pre-clip) | 3.504e-03 | 3.497e-03 | 3.473e-03 | 3.348e-03 | 3.227e-03 | 3.838e-03 |
| mux | its share | 0.168 | 0.167 | 0.166 | 0.160 | 0.154 | 0.184 |
| mux | core weight grad \|dW_t\| | 2.748e-01 | 1.786e-01 | 1.984e-01 | 2.139e-01 | 2.561e-01 | 6.417e-01 |
| mux | its share of the norm sum | 0.156 | 0.101 | 0.112 | 0.121 | 0.145 | 0.364 |
| mux | cos(dW_t, dW_total) | 0.333 | 0.267 | 0.373 | 0.625 | 0.750 | 0.672 |
| total | cotangent at the loop state (pre-clip) | 3.590e-03 | 3.580e-03 | 3.551e-03 | 3.422e-03 | 3.297e-03 | 3.911e-03 |
| total | its share | 0.168 | 0.168 | 0.166 | 0.160 | 0.154 | 0.183 |
| total | core weight grad \|dW_t\| | 2.876e-01 | 1.923e-01 | 2.116e-01 | 2.330e-01 | 3.298e-01 | 7.095e-01 |
| total | its share of the norm sum | 0.146 | 0.098 | 0.108 | 0.119 | 0.168 | 0.361 |
| total | cos(dW_t, dW_total) | 0.365 | 0.306 | 0.408 | 0.653 | 0.447 | 0.664 |

- `token_ce`: |sum_t dW_t| 1.198e-01 vs sum_t |dW_t| 1.995e-01 (cancellation ratio 0.601); self-check max rel err 1.85e-07 over 138 weights.
- `mux`: |sum_t dW_t| 9.700e-01 vs sum_t |dW_t| 1.763e+00 (cancellation ratio 0.550); self-check max rel err 2.01e-07 over 138 weights.
- `total`: |sum_t dW_t| 1.021e+00 vs sum_t |dW_t| 1.964e+00 (cancellation ratio 0.520); self-check max rel err 1.92e-07 over 138 weights.
- cross-check `g^T x` vs tap: max rel err 2.38e-03 over 144 (layer, pass) pairs on 24 layers (worst core.3.mlp.0.gate_up._cms@pass2).
- clip-through-time (`slot_cot_clip` 4.0) bound on 0.000 of rows at its worst pass.

Parameter-group gradient norms (combined backward):

| group | coda | core.attention | core.mlp | core.other | core.residual | embed | injection | other.final_norm | other.input_norm | prelude | tul.E_mask | tul.E_slot | tul.W_prefix | tul.W_sent | x0_injects |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| norm | 1.454e+00 | 3.010e-01 | 7.024e-01 | 1.749e-02 | 7.068e-01 | 3.204e+00 | 3.224e-02 | 6.510e-02 | 3.326e-02 | 2.117e+00 | 3.562e-01 | 9.400e-02 | 5.163e-02 | 9.833e-02 | 6.782e-02 |

### slot-loop-norm-match (`tul_slot_loop_norm_match`, step 5000)

token CE 4.3994, no MUX, combined 4.3996, fixed-point 0.00023. 147 core weights tapped (68.0M params), 138 carry a gradient.

| source | quantity | pass 1 | pass 2 | pass 3 | pass 4 | pass 5 | pass 6 |
|---|---|---|---|---|---|---|---|
| token_ce | cotangent at the loop state (pre-clip) | 4.724e-04 | 4.717e-04 | 4.716e-04 | 4.742e-04 | 4.916e-04 | 5.902e-04 |
| token_ce | its share | 0.159 | 0.159 | 0.159 | 0.160 | 0.165 | 0.199 |
| token_ce | core weight grad \|dW_t\| | 2.943e-03 | 2.912e-03 | 3.227e-03 | 4.763e-03 | 9.946e-03 | 1.544e-02 |
| token_ce | its share of the norm sum | 0.075 | 0.074 | 0.082 | 0.121 | 0.254 | 0.394 |
| token_ce | cos(dW_t, dW_total) | 0.630 | 0.694 | 0.760 | 0.724 | 0.584 | 0.450 |
| total | cotangent at the loop state (pre-clip) | 4.725e-04 | 4.717e-04 | 4.717e-04 | 4.743e-04 | 4.931e-04 | 5.913e-04 |
| total | its share | 0.159 | 0.159 | 0.159 | 0.159 | 0.166 | 0.199 |
| total | core weight grad \|dW_t\| | 3.067e-03 | 3.100e-03 | 3.511e-03 | 5.015e-03 | 2.534e-02 | 2.887e-02 |
| total | its share of the norm sum | 0.045 | 0.045 | 0.051 | 0.073 | 0.368 | 0.419 |
| total | cos(dW_t, dW_total) | 0.607 | 0.680 | 0.691 | 0.631 | 0.311 | 0.217 |

- `token_ce`: |sum_t dW_t| 2.253e-02 vs sum_t |dW_t| 3.923e-02 (cancellation ratio 0.574); self-check max rel err 1.86e-07 over 138 weights.
- `total`: |sum_t dW_t| 2.369e-02 vs sum_t |dW_t| 6.891e-02 (cancellation ratio 0.344); self-check max rel err 3.85e-07 over 138 weights.
- cross-check `g^T x` vs tap: max rel err 2.45e-03 over 144 (layer, pass) pairs on 24 layers (worst core.2.attention._impl.cca.W_up@pass1).
- clip-through-time (`slot_cot_clip` 4.0) bound on 0.000 of rows at its worst pass.

Parameter-group gradient norms (combined backward):

| group | coda | core.attention | core.mlp | core.other | core.residual | embed | injection | other.final_norm | other.input_norm | prelude | tul.E_mask | tul.E_slot | tul.W_prefix | tul.W_sent | x0_injects |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| norm | 1.562e+00 | 5.485e-03 | 1.417e-02 | 2.828e-04 | 1.883e-02 | 2.028e+00 | 1.169e-03 | 4.695e-02 | 2.262e-02 | 1.405e+00 | 5.235e-01 | 2.985e-02 | 2.648e-02 | 2.542e-02 | 1.908e-02 |

### slot-unpack-noise-entry (`tul_slot_unpack_noise_entry`, step 5000)

token CE 4.6618, mux_local 6.8189, combined 11.4807. 114 core weights tapped (66.5M params), 114 carry a gradient.

| source | quantity | pass 1 | pass 2 | pass 3 | pass 4 | pass 5 | pass 6 |
|---|---|---|---|---|---|---|---|
| token_ce | cotangent at the loop state (pre-clip) | 1.094e-04 | 2.074e-04 | 3.954e-04 | 8.118e-04 | 1.724e-03 | 3.568e-03 |
| token_ce | its share | 0.016 | 0.030 | 0.058 | 0.119 | 0.253 | 0.523 |
| token_ce | core weight grad \|dW_t\| | 2.327e-02 | 3.024e-02 | 6.149e-02 | 1.209e-01 | 2.611e-01 | 8.376e-01 |
| token_ce | its share of the norm sum | 0.017 | 0.023 | 0.046 | 0.091 | 0.196 | 0.628 |
| token_ce | cos(dW_t, dW_total) | -0.417 | 0.327 | -0.223 | 0.641 | 0.102 | 0.940 |
| mux | cotangent at the loop state (pre-clip) | 7.029e-04 | 1.387e-03 | 2.748e-03 | 5.867e-03 | 1.252e-02 | 2.810e-02 |
| mux | its share | 0.014 | 0.027 | 0.054 | 0.114 | 0.244 | 0.547 |
| mux | core weight grad \|dW_t\| | 7.730e-02 | 1.227e-01 | 2.403e-01 | 3.642e-01 | 9.176e-01 | 3.115e+00 |
| mux | its share of the norm sum | 0.016 | 0.025 | 0.050 | 0.075 | 0.190 | 0.644 |
| mux | cos(dW_t, dW_total) | -0.090 | 0.098 | -0.036 | 0.186 | 0.149 | 0.949 |
| total | cotangent at the loop state (pre-clip) | 7.316e-04 | 1.437e-03 | 2.832e-03 | 6.023e-03 | 1.283e-02 | 2.865e-02 |
| total | its share | 0.014 | 0.027 | 0.054 | 0.115 | 0.244 | 0.546 |
| total | core weight grad \|dW_t\| | 8.349e-02 | 1.366e-01 | 2.598e-01 | 4.278e-01 | 1.032e+00 | 3.450e+00 |
| total | its share of the norm sum | 0.015 | 0.025 | 0.048 | 0.079 | 0.191 | 0.640 |
| total | cos(dW_t, dW_total) | -0.156 | 0.134 | 0.001 | 0.294 | 0.218 | 0.949 |

- `token_ce`: |sum_t dW_t| 8.781e-01 vs sum_t |dW_t| 1.334e+00 (cancellation ratio 0.658); self-check max rel err 2.10e-07 over 114 weights.
- `mux`: |sum_t dW_t| 3.157e+00 vs sum_t |dW_t| 4.837e+00 (cancellation ratio 0.653); self-check max rel err 1.85e-07 over 114 weights.
- `total`: |sum_t dW_t| 3.630e+00 vs sum_t |dW_t| 5.390e+00 (cancellation ratio 0.674); self-check max rel err 1.52e-07 over 114 weights.
- cross-check `g^T x` vs tap: max rel err 2.33e-03 over 144 (layer, pass) pairs on 24 layers (worst core.3.attention._impl.cca.W_up@pass4).
- clip-through-time (`slot_cot_clip` 4.0) bound on 0.000 of rows at its worst pass.

Parameter-group gradient norms (combined backward):

| group | coda | core.attention | core.mlp | core.other | core.residual | embed | injection | other.final_norm | other.input_norm | prelude | tul.E_mask | tul.E_slot | tul.W_bcast | tul.W_prefix | tul.W_sent | x0_injects |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| norm | 1.991e+00 | 5.918e-01 | 4.409e-01 | 1.904e-02 | 3.557e+00 | 3.250e+00 | 4.726e-01 | 6.493e-02 | 3.470e-02 | 1.288e+00 | 6.829e-02 | 1.511e-01 | 1.095e-01 | 1.970e-01 | 1.024e-01 | 2.433e-02 |

### slot-unpack-free (`tul_slot_unpack_free`, step 5000)

token CE 4.7012, mux_local 6.8336, combined 11.5348. 114 core weights tapped (66.5M params), 114 carry a gradient.

| source | quantity | pass 1 | pass 2 | pass 3 | pass 4 | pass 5 | pass 6 |
|---|---|---|---|---|---|---|---|
| token_ce | cotangent at the loop state (pre-clip) | 2.648e-03 | 2.648e-03 | 2.643e-03 | 2.684e-03 | 2.844e-03 | 3.573e-03 |
| token_ce | its share | 0.155 | 0.155 | 0.155 | 0.158 | 0.167 | 0.210 |
| token_ce | core weight grad \|dW_t\| | 3.102e-01 | 1.866e-01 | 1.855e-01 | 2.292e-01 | 3.785e-01 | 1.260e+00 |
| token_ce | its share of the norm sum | 0.122 | 0.073 | 0.073 | 0.090 | 0.148 | 0.494 |
| token_ce | cos(dW_t, dW_total) | 0.203 | 0.211 | -0.020 | 0.476 | -0.423 | 0.829 |
| mux | cotangent at the loop state (pre-clip) | 1.757e-02 | 1.762e-02 | 1.750e-02 | 1.794e-02 | 1.900e-02 | 2.424e-02 |
| mux | its share | 0.154 | 0.155 | 0.154 | 0.158 | 0.167 | 0.213 |
| mux | core weight grad \|dW_t\| | 2.362e+00 | 1.003e+00 | 1.024e+00 | 1.072e+00 | 1.243e+00 | 1.063e+01 |
| mux | its share of the norm sum | 0.136 | 0.058 | 0.059 | 0.062 | 0.072 | 0.613 |
| mux | cos(dW_t, dW_total) | 0.218 | -0.053 | 0.104 | -0.073 | 0.047 | 0.964 |
| total | cotangent at the loop state (pre-clip) | 1.808e-02 | 1.814e-02 | 1.801e-02 | 1.847e-02 | 1.956e-02 | 2.497e-02 |
| total | its share | 0.154 | 0.155 | 0.154 | 0.158 | 0.167 | 0.213 |
| total | core weight grad \|dW_t\| | 2.547e+00 | 1.083e+00 | 1.080e+00 | 1.114e+00 | 1.264e+00 | 1.163e+01 |
| total | its share of the norm sum | 0.136 | 0.058 | 0.058 | 0.060 | 0.068 | 0.621 |
| total | cos(dW_t, dW_total) | 0.223 | -0.100 | 0.088 | -0.107 | -0.040 | 0.962 |

- `token_ce`: |sum_t dW_t| 1.092e+00 vs sum_t |dW_t| 2.550e+00 (cancellation ratio 0.428); self-check max rel err 1.54e-07 over 114 weights.
- `mux`: |sum_t dW_t| 1.079e+01 vs sum_t |dW_t| 1.733e+01 (cancellation ratio 0.623); self-check max rel err 3.27e-07 over 114 weights.
- `total`: |sum_t dW_t| 1.156e+01 vs sum_t |dW_t| 1.871e+01 (cancellation ratio 0.618); self-check max rel err 3.66e-07 over 114 weights.
- cross-check `g^T x` vs tap: max rel err 2.48e-03 over 144 (layer, pass) pairs on 24 layers (worst core.4.attention._impl.cca.W_up@pass4).
- clip-through-time (`slot_cot_clip` 4.0) bound on 0.000 of rows at its worst pass.

Parameter-group gradient norms (combined backward):

| group | coda | core.attention | core.mlp | core.other | core.residual | embed | injection | other.final_norm | other.input_norm | prelude | tul.E_mask | tul.E_slot | tul.W_bcast | tul.W_prefix | tul.W_sent | x0_injects |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| norm | 1.831e+00 | 5.795e-01 | 4.887e-01 | 1.947e-02 | 1.154e+01 | 3.343e+00 | 2.627e-02 | 7.000e-02 | 3.920e-02 | 1.429e+00 | 6.643e-02 | 1.514e-01 | 1.302e-01 | 1.624e-01 | 1.156e-01 | 2.686e-02 |

### slot-unpack-fixed-depth (`tul_slot_unpack_fixed_depth`, step 5000)

token CE 4.7271, mux_local 6.8246, combined 11.5519, fixed-point 0.00026. 114 core weights tapped (66.5M params), 114 carry a gradient.

| source | quantity | pass 1 | pass 2 | pass 3 | pass 4 | pass 5 | pass 6 |
|---|---|---|---|---|---|---|---|
| token_ce | cotangent at the loop state (pre-clip) | 2.910e-03 | 2.910e-03 | 2.919e-03 | 2.956e-03 | 3.156e-03 | 3.962e-03 |
| token_ce | its share | 0.155 | 0.155 | 0.155 | 0.157 | 0.168 | 0.211 |
| token_ce | core weight grad \|dW_t\| | 7.670e-01 | 1.029e+00 | 1.037e+00 | 3.119e+00 | 2.201e+00 | 2.426e+00 |
| token_ce | its share of the norm sum | 0.072 | 0.097 | 0.098 | 0.295 | 0.208 | 0.229 |
| token_ce | cos(dW_t, dW_total) | 0.334 | 0.430 | -0.075 | -0.220 | 0.557 | 0.389 |
| mux | cotangent at the loop state (pre-clip) | 1.747e-02 | 1.760e-02 | 1.766e-02 | 1.794e-02 | 1.891e-02 | 2.321e-02 |
| mux | its share | 0.155 | 0.156 | 0.157 | 0.159 | 0.168 | 0.206 |
| mux | core weight grad \|dW_t\| | 2.144e+00 | 1.172e+00 | 9.345e-01 | 2.989e+00 | 3.179e+00 | 3.198e+00 |
| mux | its share of the norm sum | 0.157 | 0.086 | 0.069 | 0.219 | 0.233 | 0.235 |
| mux | cos(dW_t, dW_total) | 0.685 | 0.082 | -0.178 | 0.190 | 0.023 | 0.309 |
| total | cotangent at the loop state (pre-clip) | 1.810e-02 | 1.823e-02 | 1.828e-02 | 1.857e-02 | 1.960e-02 | 2.413e-02 |
| total | its share | 0.155 | 0.156 | 0.156 | 0.159 | 0.168 | 0.206 |
| total | core weight grad \|dW_t\| | 2.436e+00 | 1.830e+00 | 1.318e+00 | 5.770e+00 | 4.735e+00 | 3.484e+00 |
| total | its share of the norm sum | 0.124 | 0.094 | 0.067 | 0.295 | 0.242 | 0.178 |
| total | cos(dW_t, dW_total) | 0.591 | 0.247 | -0.058 | 0.085 | 0.331 | 0.024 |

- `token_ce`: |sum_t dW_t| 2.103e+00 vs sum_t |dW_t| 1.058e+01 (cancellation ratio 0.199); self-check max rel err 2.00e-07 over 114 weights.
- `mux`: |sum_t dW_t| 3.028e+00 vs sum_t |dW_t| 1.362e+01 (cancellation ratio 0.222); self-check max rel err 5.55e-07 over 114 weights.
- `total`: |sum_t dW_t| 3.957e+00 vs sum_t |dW_t| 1.957e+01 (cancellation ratio 0.202); self-check max rel err 4.82e-07 over 114 weights.
- cross-check `g^T x` vs tap: max rel err 2.35e-03 over 144 (layer, pass) pairs on 24 layers (worst core.1.mlp.0.gate_up._cms@pass3).
- clip-through-time (`slot_cot_clip` 4.0) bound on 0.000 of rows at its worst pass.

Parameter-group gradient norms (combined backward):

| group | coda | core.attention | core.mlp | core.other | core.residual | embed | injection | other.final_norm | other.input_norm | prelude | tul.E_mask | tul.E_slot | tul.W_bcast | tul.W_prefix | tul.W_sent | x0_injects |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| norm | 2.077e+00 | 3.462e-01 | 4.030e-01 | 1.046e-02 | 3.923e+00 | 3.362e+00 | 2.074e-02 | 6.973e-02 | 3.333e-02 | 1.307e+00 | 6.747e-02 | 1.820e-01 | 1.262e-01 | 2.212e-01 | 1.335e-01 | 2.257e-02 |

## Not verified

- One seed, one 12-row validation draw, 5000-step checkpoints. The per-arm numbers are not
  a ranking; short-horizon CE cannot rank these arms and neither can one gradient draw.
- Depth 6 only. Nothing here says what the profile looks like at the trained Poisson draw
  (the fixed-depth arm is the only one that trained at a single depth).
- The hinge penalty's own contribution to the core weight gradient is NOT measured: the
  probe turns it off so the pass counter stays honest.
- The two panels differ in more than the loop: the unpack arms carry `tg_restrict`,
  `tg_restrict_scope: coda`, `coda_token_input: embed`, `bcast: true` and
  `tg_scoped_kernels: true` (which is why they tap 114 weights / 66.5M against the mux and
  A1 arms' 147 / 68.0M). Cross-panel comparisons of the absolute norms are not clean.
- No wandb run: this reads finished checkpoints and trains nothing.
