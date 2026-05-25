// Copyright 2026 TitanMAC Authors. All rights reserved.
//
// memory_kernels.cu — Fused CUDA kernels for neural memory update + retrieve.
//
// Replaces the torch.autograd.grad() graph break in NeuralMemory.update()
// with hand-rolled chain-rule backward, eliminating a ~2.5x compile slowdown.
//
// === MLP Architecture ===
//   h0 = k                               (input, [N, d])
//   z1 = h0 @ W0.T + b0                  (linear, [N, d])
//   h1 = silu(z1)                        (activation)
//   z2 = h1 @ W1.T + b1
//   h2 = silu(z2)
//   z3 = h2 @ W2.T + b2
//   h3 = silu(z3)
//   h4 = h3 @ W3.T + b3                  (no activation on output layer)
//   L  = mean((h4 - v)^2)               (MSE loss)
//
// === Backward chain rule ===
//   dh4  = 2*(h4 - v) / N               [N, d]  (mse_grad)
//   dW3  = dh4.T @ h3                   [d, d]  (GEMM)
//   db3  = dh4.sum(0)                   [d]     (reduce)
//   dh3  = dh4 @ W3                     [N, d]  (GEMM)
//   dz3  = dh3 * silu'(z3)              [N, d]  (silu_backward)
//   dW2  = dz3.T @ h2                   [d, d]
//   db2  = dz3.sum(0)                   [d]
//   dh2  = dz3 @ W2
//   dz2  = dh2 * silu'(z2)
//   dW1  = dz2.T @ h1
//   db1  = dz2.sum(0)
//   dh1  = dz2 @ W1
//   dz1  = dh1 * silu'(z1)
//   dW0  = dz1.T @ h0
//   db0  = dz1.sum(0)
//   (dh0 not needed — keys are inputs, not params)
//
// === Momentum + weight update (Titans Eq. 13-14) ===
//   grad_flat = [dW0, db0, dW1, db1, dW2, db2, dW3, db3]  (fp32, clipped)
//   S = eta * S - theta * grad_flat
//   W = (1 - alpha) * W + S
//
// Branch: 006-looped-block-ell

#include "memory_kernels.cuh"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <torch/extension.h>

// cuBLAS access through ATen
#include <ATen/cuda/CUDABlas.h>

namespace titan {

// ============================================================================
// Constants
// ============================================================================

static constexpr int kBlockSize = 256;  // threads per block for elementwise ops

// ============================================================================
// Device helpers
// ============================================================================

// Fast BF16 ↔ float conversion using hardware intrinsics where available.
// __bfloat162float / __float2bfloat16 are available on sm_80+.

__device__ __forceinline__ float bf16_to_float(at::BFloat16 x) {
    // at::BFloat16 stores its bits as uint16_t.
    // Cast via the bit representation to avoid software emulation.
    float f;
    uint32_t bits = static_cast<uint32_t>(x.x) << 16;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

__device__ __forceinline__ at::BFloat16 float_to_bf16(float x) {
    // Round-to-nearest-even via truncation of the lower 16 mantissa bits.
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    // Handle NaN
    if ((bits & 0x7FFFFFFF) > 0x7F800000) {
        bits = 0x7FC00000;  // canonical NaN
    }
    // Round: add 0x7FFF + (bit 16) for round-to-nearest-even
    bits += 0x7FFF + ((bits >> 16) & 1);
    at::BFloat16 result;
    result.x = static_cast<uint16_t>(bits >> 16);
    return result;
}

// SiLU forward: x * sigmoid(x)
// Uses fp32 for sigmoid to avoid precision loss.
__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

// SiLU backward: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
//              = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
__device__ __forceinline__ float silu_grad_f(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

// ============================================================================
// Kernel: silu_forward_kernel
//
// h[i] = silu(z[i])
// preact[i] = z[i]  (fp32, saved for backward)
// ============================================================================

template <typename scalar_t>
__global__ void silu_forward_kernel(
    const scalar_t* __restrict__ z,
    scalar_t*       __restrict__ h,
    float*          __restrict__ preact,
    int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float z_f = bf16_to_float(static_cast<at::BFloat16>(z[idx]));
    preact[idx] = z_f;
    float h_f = silu_f(z_f);
    h[idx] = static_cast<scalar_t>(float_to_bf16(h_f));
}

// Specialization for float (non-bf16 path, e.g. fallback testing)
template <>
__global__ void silu_forward_kernel<float>(
    const float* __restrict__ z,
    float*       __restrict__ h,
    float*       __restrict__ preact,
    int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float z_f = z[idx];
    preact[idx] = z_f;
    h[idx] = silu_f(z_f);
}

// ============================================================================
// Kernel: silu_backward_kernel
//
// dz[i] = dh[i] * silu'(preact[i])
// preact is fp32 (saved from forward)
// ============================================================================

template <typename scalar_t>
__global__ void silu_backward_kernel(
    const scalar_t* __restrict__ dh,
    const float*    __restrict__ preact,
    scalar_t*       __restrict__ dz,
    int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float dh_f = bf16_to_float(static_cast<at::BFloat16>(dh[idx]));
    float grad = silu_grad_f(preact[idx]);
    float dz_f = dh_f * grad;
    dz[idx] = static_cast<scalar_t>(float_to_bf16(dz_f));
}

template <>
__global__ void silu_backward_kernel<float>(
    const float* __restrict__ dh,
    const float* __restrict__ preact,
    float*       __restrict__ dz,
    int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    dz[idx] = dh[idx] * silu_grad_f(preact[idx]);
}

// ============================================================================
// Kernel: mse_grad_kernel
//
// d_pred[i] = 2 * (pred[i] - target[i]) / N
//
// N is the total number of elements (N_rows * d) for mean reduction.
// ============================================================================

template <typename scalar_t>
__global__ void mse_grad_kernel(
    const scalar_t* __restrict__ pred,
    const scalar_t* __restrict__ target,
    scalar_t*       __restrict__ d_pred,
    int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float p = bf16_to_float(static_cast<at::BFloat16>(pred[idx]));
    float t = bf16_to_float(static_cast<at::BFloat16>(target[idx]));
    float grad = 2.0f * (p - t) / static_cast<float>(N);
    d_pred[idx] = static_cast<scalar_t>(float_to_bf16(grad));
}

template <>
__global__ void mse_grad_kernel<float>(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float*       __restrict__ d_pred,
    int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    d_pred[idx] = 2.0f * (pred[idx] - target[idx]) / static_cast<float>(N);
}

// ============================================================================
// Kernel: bias_grad_reduce_kernel
//
// db[j] = sum_{i=0}^{N_rows-1} dz[i, j]
//
// Each block handles one output column j. Threads within the block
// cooperatively sum over rows using shared memory reduction.
//
// Grid: D blocks, kBlockSize threads/block
// Accumulates in fp32 to prevent bf16 overflow on large N_rows.
// ============================================================================

template <typename scalar_t>
__global__ void bias_grad_reduce_kernel(
    const scalar_t* __restrict__ dz,       // [N_rows, D]
    float*          __restrict__ db_fp32,  // [D] fp32 accumulation
    int64_t N_rows,
    int64_t D
) {
    const int64_t col = blockIdx.x;  // which output dim
    if (col >= D) return;

    __shared__ float sdata[kBlockSize];

    float acc = 0.0f;
    // Each thread strides over rows
    for (int64_t row = threadIdx.x; row < N_rows; row += blockDim.x) {
        acc += bf16_to_float(static_cast<at::BFloat16>(dz[row * D + col]));
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();

    // Tree reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        db_fp32[col] = sdata[0];
    }
}

template <>
__global__ void bias_grad_reduce_kernel<float>(
    const float* __restrict__ dz,
    float*       __restrict__ db_fp32,
    int64_t N_rows,
    int64_t D
) {
    const int64_t col = blockIdx.x;
    if (col >= D) return;

    __shared__ float sdata[kBlockSize];
    float acc = 0.0f;
    for (int64_t row = threadIdx.x; row < N_rows; row += blockDim.x) {
        acc += dz[row * D + col];
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) db_fp32[col] = sdata[0];
}

// ============================================================================
// Kernel: fp32_to_bf16_kernel
//
// Simple cast from fp32 accumulation buffer → bf16 output tensor.
// Used to convert bias gradients after reduction.
// ============================================================================

__global__ void fp32_to_bf16_kernel(
    const float*     __restrict__ src_fp32,
    at::BFloat16*    __restrict__ dst_bf16,
    int64_t D
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= D) return;
    dst_bf16[idx] = float_to_bf16(src_fp32[idx]);
}

// ============================================================================
// Kernel: momentum_weight_update_kernel
//
// Implements Titans Eq. 13-14 for a contiguous chunk of parameters:
//
//   S[i]  = eta * S[i]  - theta * grad[i]       (fp32, in-place)
//   W[i]  = (1 - alpha) * W[i] + S[i]           (bf16, in-place)
//
// The gradient has already been clipped to unit norm by the host.
// ============================================================================

template <typename scalar_t>
__global__ void momentum_weight_update_kernel(
    const float*  __restrict__ grad_flat,  // [n_params] fp32, clipped
    float*        __restrict__ S,          // [n_params] fp32 momentum (MUTATED)
    scalar_t*     __restrict__ W_flat,     // [n_params] bf16 weights (MUTATED)
    int64_t n_params,
    float   eta,
    float   theta,
    float   alpha
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_params) return;

    // Eq. 13: S_t = eta * S_{t-1} - theta * grad
    float s_new = eta * S[idx] - theta * grad_flat[idx];
    S[idx] = s_new;

    // Eq. 14: W_t = (1 - alpha) * W_{t-1} + S_t
    float w_curr = bf16_to_float(static_cast<at::BFloat16>(W_flat[idx]));
    float w_new = (1.0f - alpha) * w_curr + s_new;
    W_flat[idx] = static_cast<scalar_t>(float_to_bf16(w_new));
}

template <>
__global__ void momentum_weight_update_kernel<float>(
    const float* __restrict__ grad_flat,
    float*       __restrict__ S,
    float*       __restrict__ W_flat,
    int64_t n_params,
    float   eta,
    float   theta,
    float   alpha
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_params) return;
    float s_new = eta * S[idx] - theta * grad_flat[idx];
    S[idx] = s_new;
    W_flat[idx] = (1.0f - alpha) * W_flat[idx] + s_new;
}

// ============================================================================
// Device helper: compute MSE loss scalar from predictions + targets
//
// Returns sum of squared differences / N as fp32.
// Uses a simple parallel reduction on the GPU to avoid a device-to-host sync
// in the middle of the compute graph; the final scalar is retrieved once after
// all kernels complete.
// ============================================================================

// Reduction kernel: sum array[0..N) into out[0] (fp32)
__global__ void sum_reduce_kernel(
    const float* __restrict__ arr,
    float*       __restrict__ out,
    int64_t N
) {
    extern __shared__ float smem[];

    int64_t tid = threadIdx.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    float val = 0.0f;
    if (idx < N) val = arr[idx];
    smem[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out, smem[0]);
}

// Squared-error kernel: tmp[i] = (pred[i] - target[i])^2
template <typename scalar_t>
__global__ void sq_err_kernel(
    const scalar_t* __restrict__ pred,
    const scalar_t* __restrict__ target,
    float*          __restrict__ tmp,
    int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float diff = bf16_to_float(static_cast<at::BFloat16>(pred[idx]))
               - bf16_to_float(static_cast<at::BFloat16>(target[idx]));
    tmp[idx] = diff * diff;
}

template <>
__global__ void sq_err_kernel<float>(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float*       __restrict__ tmp,
    int64_t N
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float diff = pred[idx] - target[idx];
    tmp[idx] = diff * diff;
}

// ============================================================================
// Host helper: compute_loss_gpu
//
// Computes MSE(h4, values) entirely on GPU, returns float scalar.
// Allocates a temporary fp32 tensor for squared errors + a one-element
// accumulator; both are short-lived and freed when out of scope.
// ============================================================================

static float compute_loss_gpu(
    const at::Tensor& h4,      // [N, d] bf16
    const at::Tensor& values,  // [N, d] bf16
    cudaStream_t stream
) {
    int64_t total = h4.numel();

    at::Tensor tmp = at::zeros({total}, h4.options().dtype(at::kFloat));
    at::Tensor acc = at::zeros({1},     h4.options().dtype(at::kFloat));

    const int blocks_sq = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        h4.scalar_type(), "sq_err_kernel",
        [&]() {
            sq_err_kernel<scalar_t><<<blocks_sq, kBlockSize, 0, stream>>>(
                h4.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                tmp.data_ptr<float>(),
                total
            );
        }
    );
    C10_CUDA_CHECK(cudaGetLastError());

    const int blocks_sum = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    sum_reduce_kernel<<<blocks_sum, kBlockSize,
                        kBlockSize * sizeof(float), stream>>>(
        tmp.data_ptr<float>(),
        acc.data_ptr<float>(),
        total
    );
    C10_CUDA_CHECK(cudaGetLastError());

    // Sync to retrieve scalar — this is the only host-device sync in update()
    float loss_sum = acc.item<float>();
    return loss_sum / static_cast<float>(total);
}

// ============================================================================
// Host helper: mlp_forward_save_preacts
//
// Forward pass through the 4-layer MLP saving pre-activations for backward.
// Returns {h1, h2, h3, h4, z1_preact, z2_preact, z3_preact}.
//
// Uses at::addmm for fused bias-add + GEMM (delegates to cuBLAS).
//
//   z_i = h_{i-1} @ W_{i-1}.T + b_{i-1}
//   h_i = silu(z_i)                          (layers 1-3)
//   h4  = z4                                 (no activation on output)
//
// All intermediate tensors are bf16; preact buffers are fp32.
// ============================================================================

struct ForwardActivations {
    at::Tensor h1, h2, h3, h4;      // post-activation [N, d]
    at::Tensor z1_fp32, z2_fp32, z3_fp32;  // pre-activation fp32 [N, d]
};

static ForwardActivations mlp_forward_save_preacts(
    const at::Tensor& h0,       // [N, d] bf16 input (= keys)
    const at::Tensor& W0, const at::Tensor& b0,
    const at::Tensor& W1, const at::Tensor& b1,
    const at::Tensor& W2, const at::Tensor& b2,
    const at::Tensor& W3, const at::Tensor& b3,
    cudaStream_t stream
) {
    int64_t N = h0.size(0);
    int64_t d = h0.size(1);

    // ---- Layer 0 ----
    // z1 = h0 @ W0.T + b0   (addmm: y = beta*mat + alpha*(mat1@mat2))
    // b0 needs to be broadcast over rows: expand to [N, d] via addmm semantics
    // at::addmm(bias_row, mat1, mat2) computes mat1 @ mat2 + bias_row (broadcast)
    at::Tensor z1 = at::addmm(b0.unsqueeze(0).expand({N, d}), h0, W0.t());
    // Save pre-activation as fp32
    at::Tensor z1_fp32 = z1.to(at::kFloat);
    // h1 = silu(z1)
    at::Tensor h1 = at::zeros_like(z1);
    {
        int64_t elems = N * d;
        int blocks = static_cast<int>((elems + kBlockSize - 1) / kBlockSize);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            z1.scalar_type(), "silu_fwd_layer1",
            [&]() {
                silu_forward_kernel<scalar_t><<<blocks, kBlockSize, 0, stream>>>(
                    z1.data_ptr<scalar_t>(),
                    h1.data_ptr<scalar_t>(),
                    z1_fp32.data_ptr<float>(),
                    elems
                );
            }
        );
        C10_CUDA_CHECK(cudaGetLastError());
    }

    // ---- Layer 1 ----
    at::Tensor z2 = at::addmm(b1.unsqueeze(0).expand({N, d}), h1, W1.t());
    at::Tensor z2_fp32 = z2.to(at::kFloat);
    at::Tensor h2 = at::zeros_like(z2);
    {
        int64_t elems = N * d;
        int blocks = static_cast<int>((elems + kBlockSize - 1) / kBlockSize);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            z2.scalar_type(), "silu_fwd_layer2",
            [&]() {
                silu_forward_kernel<scalar_t><<<blocks, kBlockSize, 0, stream>>>(
                    z2.data_ptr<scalar_t>(),
                    h2.data_ptr<scalar_t>(),
                    z2_fp32.data_ptr<float>(),
                    elems
                );
            }
        );
        C10_CUDA_CHECK(cudaGetLastError());
    }

    // ---- Layer 2 ----
    at::Tensor z3 = at::addmm(b2.unsqueeze(0).expand({N, d}), h2, W2.t());
    at::Tensor z3_fp32 = z3.to(at::kFloat);
    at::Tensor h3 = at::zeros_like(z3);
    {
        int64_t elems = N * d;
        int blocks = static_cast<int>((elems + kBlockSize - 1) / kBlockSize);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            z3.scalar_type(), "silu_fwd_layer3",
            [&]() {
                silu_forward_kernel<scalar_t><<<blocks, kBlockSize, 0, stream>>>(
                    z3.data_ptr<scalar_t>(),
                    h3.data_ptr<scalar_t>(),
                    z3_fp32.data_ptr<float>(),
                    elems
                );
            }
        );
        C10_CUDA_CHECK(cudaGetLastError());
    }

    // ---- Layer 3 (output, no activation) ----
    at::Tensor h4 = at::addmm(b3.unsqueeze(0).expand({N, d}), h3, W3.t());

    return {h1, h2, h3, h4, z1_fp32, z2_fp32, z3_fp32};
}

// ============================================================================
// Host helper: mlp_backward
//
// Given saved activations + loss gradient dh4, compute per-parameter
// gradients via hand-rolled chain rule.
//
// Returns flat fp32 gradient tensor in parameter order:
//   [dW0, db0, dW1, db1, dW2, db2, dW3, db3]
//
// GEMMs use at::mm (delegates to cuBLAS).  Elementwise ops use custom kernels.
// ============================================================================

static at::Tensor mlp_backward(
    // Saved activations from forward
    const at::Tensor& h0,       // [N, d] bf16 input keys
    const ForwardActivations& fwd,
    // Network weights (needed for input gradient GEMMs)
    const at::Tensor& W0, const at::Tensor& W1,
    const at::Tensor& W2, const at::Tensor& W3,
    // Loss gradient w.r.t. h4
    const at::Tensor& dh4,      // [N, d] bf16
    cudaStream_t stream
) {
    int64_t N = h0.size(0);
    int64_t d = h0.size(1);
    int64_t elems = N * d;
    int blocks_elem = static_cast<int>((elems + kBlockSize - 1) / kBlockSize);

    // Helper: compute bias grad db = dz.sum(dim=0) via custom reduction
    // Returns bf16 tensor [d].
    auto compute_db = [&](const at::Tensor& dz) -> at::Tensor {
        at::Tensor db_fp32 = at::zeros({d}, dz.options().dtype(at::kFloat));
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            dz.scalar_type(), "bias_reduce",
            [&]() {
                bias_grad_reduce_kernel<scalar_t><<<
                    static_cast<int>(d), kBlockSize, 0, stream>>>(
                    dz.data_ptr<scalar_t>(),
                    db_fp32.data_ptr<float>(),
                    N, d
                );
            }
        );
        C10_CUDA_CHECK(cudaGetLastError());
        // Convert fp32 accumulation → bf16
        at::Tensor db = at::zeros({d}, dz.options());
        int blocks_cast = static_cast<int>((d + kBlockSize - 1) / kBlockSize);
        fp32_to_bf16_kernel<<<blocks_cast, kBlockSize, 0, stream>>>(
            db_fp32.data_ptr<float>(),
            reinterpret_cast<at::BFloat16*>(db.data_ptr()),
            d
        );
        C10_CUDA_CHECK(cudaGetLastError());
        return db;
    };

    // ========== Layer 3 backward ==========
    // dW3 = dh4.T @ h3    [d, d]   (no SiLU on output layer — dz4 = dh4)
    // db3 = dh4.sum(0)    [d]
    // dh3 = dh4 @ W3      [N, d]
    at::Tensor dW3 = at::mm(dh4.t(), fwd.h3);          // [d, d]
    at::Tensor db3 = compute_db(dh4);                   // [d]
    at::Tensor dh3 = at::mm(dh4, W3);                  // [N, d]

    // ========== Layer 2 backward ==========
    // dz3 = dh3 * silu'(z3_preact)
    at::Tensor dz3 = at::zeros_like(dh3);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        dh3.scalar_type(), "silu_bwd_layer3",
        [&]() {
            silu_backward_kernel<scalar_t><<<blocks_elem, kBlockSize, 0, stream>>>(
                dh3.data_ptr<scalar_t>(),
                fwd.z3_fp32.data_ptr<float>(),
                dz3.data_ptr<scalar_t>(),
                elems
            );
        }
    );
    C10_CUDA_CHECK(cudaGetLastError());

    at::Tensor dW2 = at::mm(dz3.t(), fwd.h2);          // [d, d]
    at::Tensor db2 = compute_db(dz3);                   // [d]
    at::Tensor dh2 = at::mm(dz3, W2);                  // [N, d]

    // ========== Layer 1 backward ==========
    at::Tensor dz2 = at::zeros_like(dh2);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        dh2.scalar_type(), "silu_bwd_layer2",
        [&]() {
            silu_backward_kernel<scalar_t><<<blocks_elem, kBlockSize, 0, stream>>>(
                dh2.data_ptr<scalar_t>(),
                fwd.z2_fp32.data_ptr<float>(),
                dz2.data_ptr<scalar_t>(),
                elems
            );
        }
    );
    C10_CUDA_CHECK(cudaGetLastError());

    at::Tensor dW1 = at::mm(dz2.t(), fwd.h1);          // [d, d]
    at::Tensor db1 = compute_db(dz2);                   // [d]
    at::Tensor dh1 = at::mm(dz2, W1);                  // [N, d]

    // ========== Layer 0 backward ==========
    at::Tensor dz1 = at::zeros_like(dh1);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        dh1.scalar_type(), "silu_bwd_layer1",
        [&]() {
            silu_backward_kernel<scalar_t><<<blocks_elem, kBlockSize, 0, stream>>>(
                dh1.data_ptr<scalar_t>(),
                fwd.z1_fp32.data_ptr<float>(),
                dz1.data_ptr<scalar_t>(),
                elems
            );
        }
    );
    C10_CUDA_CHECK(cudaGetLastError());

    at::Tensor dW0 = at::mm(dz1.t(), h0);              // [d, d]
    at::Tensor db0 = compute_db(dz1);                  // [d]
    // dh0 = dz1 @ W0 — not needed (keys are inputs, not updated params)

    // ========== Flatten all gradients into a single fp32 tensor ==========
    // Order: [dW0, db0, dW1, db1, dW2, db2, dW3, db3]
    // Convert to fp32 before flattening for gradient clipping and momentum.
    at::Tensor grad_flat = at::cat({
        dW0.to(at::kFloat).view(-1),
        db0.to(at::kFloat).view(-1),
        dW1.to(at::kFloat).view(-1),
        db1.to(at::kFloat).view(-1),
        dW2.to(at::kFloat).view(-1),
        db2.to(at::kFloat).view(-1),
        dW3.to(at::kFloat).view(-1),
        db3.to(at::kFloat).view(-1),
    }, /*dim=*/0);

    return grad_flat;  // [n_params] fp32
}

// ============================================================================
// Host helper: gradient_norm_l2
//
// Computes L2 norm of a flat fp32 tensor on the GPU using at::norm.
// ============================================================================

static float gradient_norm_l2(const at::Tensor& grad_flat) {
    return at::norm(grad_flat, 2).item<float>();
}

// ============================================================================
// Host function: memory_update_cuda
//
// Main entry point — fused forward + backward + momentum + weight update.
// ============================================================================

float memory_update_cuda(
    const at::Tensor& keys,    // [N, d] bf16
    const at::Tensor& values,  // [N, d] bf16
    MemoryMLPParams&  params,
    double eta,
    double theta,
    double alpha
) {
    TORCH_CHECK(keys.is_cuda(),   "keys must be on CUDA");
    TORCH_CHECK(values.is_cuda(), "values must be on CUDA");
    TORCH_CHECK(keys.dim() == 2 && values.dim() == 2,
                "keys/values must be 2D [N, d]");
    TORCH_CHECK(keys.sizes() == values.sizes(),
                "keys and values must have the same shape");

    // Guard: ensure all ops run on the correct device
    const c10::cuda::CUDAGuard device_guard(keys.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // All mutations happen outside autograd
    at::NoGradGuard no_grad;

    int64_t N = keys.size(0);
    int64_t d = keys.size(1);

    // ----------------------------------------------------------------
    // 1. Forward pass — save pre-activations for backward
    // ----------------------------------------------------------------
    ForwardActivations fwd = mlp_forward_save_preacts(
        keys,
        params.W0, params.b0,
        params.W1, params.b1,
        params.W2, params.b2,
        params.W3, params.b3,
        stream
    );

    // ----------------------------------------------------------------
    // 2. Compute MSE loss: L = mean((h4 - v)^2)
    // ----------------------------------------------------------------
    float loss = compute_loss_gpu(fwd.h4, values, stream);

    // ----------------------------------------------------------------
    // 3. Compute loss gradient: dh4 = 2*(h4 - v) / N_total
    // ----------------------------------------------------------------
    at::Tensor dh4 = at::zeros_like(fwd.h4);
    {
        int64_t total = N * d;
        int blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            fwd.h4.scalar_type(), "mse_grad",
            [&]() {
                mse_grad_kernel<scalar_t><<<blocks, kBlockSize, 0, stream>>>(
                    fwd.h4.data_ptr<scalar_t>(),
                    values.data_ptr<scalar_t>(),
                    dh4.data_ptr<scalar_t>(),
                    total
                );
            }
        );
        C10_CUDA_CHECK(cudaGetLastError());
    }

    // ----------------------------------------------------------------
    // 4. Backward pass — compute flat gradient
    // ----------------------------------------------------------------
    at::Tensor grad_flat = mlp_backward(
        keys, fwd,
        params.W0, params.W1, params.W2, params.W3,
        dh4, stream
    );
    // grad_flat: [n_params] fp32

    // ----------------------------------------------------------------
    // 5. Gradient clipping: if ||g|| > 1.0, scale to unit norm
    // ----------------------------------------------------------------
    float grad_norm = gradient_norm_l2(grad_flat);
    if (grad_norm > 1.0f) {
        grad_flat.mul_(1.0f / grad_norm);
    }

    // ----------------------------------------------------------------
    // 6. Fused momentum + weight update (Titans Eq. 13-14)
    //
    // For each parameter group the momentum buffer and weight tensor
    // are laid out contiguously in the same order as grad_flat:
    //   [W0, b0, W1, b1, W2, b2, W3, b3]
    //
    // We run one kernel call spanning all n_params at once.
    // ----------------------------------------------------------------
    int64_t n_params = grad_flat.numel();
    TORCH_CHECK(params.momentum_S.numel() == n_params,
                "momentum buffer size mismatch: expected ", n_params,
                " got ", params.momentum_S.numel());

    // Build a flat view of all weights in the same order as grad_flat.
    // We do NOT allocate a copy; instead we update each weight tensor
    // in-place using the corresponding slice of grad_flat and momentum_S.
    {
        // Process each parameter block: (weight_tensor, slice_of_grad, slice_of_S)
        struct ParamBlock {
            at::Tensor& weight;
            int64_t offset;
            int64_t numel;
        };

        // Layout: W0, b0, W1, b1, W2, b2, W3, b3
        std::vector<std::pair<at::Tensor*, int64_t>> param_list = {
            {&params.W0, params.W0.numel()},
            {&params.b0, params.b0.numel()},
            {&params.W1, params.W1.numel()},
            {&params.b1, params.b1.numel()},
            {&params.W2, params.W2.numel()},
            {&params.b2, params.b2.numel()},
            {&params.W3, params.W3.numel()},
            {&params.b3, params.b3.numel()},
        };

        int64_t offset = 0;
        for (auto& [param_ptr, numel] : param_list) {
            at::Tensor& param = *param_ptr;
            at::Tensor grad_slice = grad_flat.narrow(0, offset, numel);
            at::Tensor S_slice = params.momentum_S.narrow(0, offset, numel);

            int blocks = static_cast<int>((numel + kBlockSize - 1) / kBlockSize);
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half, at::ScalarType::BFloat16,
                param.scalar_type(), "momentum_weight_update",
                [&]() {
                    momentum_weight_update_kernel<scalar_t><<<
                        blocks, kBlockSize, 0, stream>>>(
                        grad_slice.data_ptr<float>(),
                        S_slice.data_ptr<float>(),
                        param.view(-1).data_ptr<scalar_t>(),
                        numel,
                        static_cast<float>(eta),
                        static_cast<float>(theta),
                        static_cast<float>(alpha)
                    );
                }
            );
            C10_CUDA_CHECK(cudaGetLastError());

            offset += numel;
        }
    }

    return loss;
}

// ============================================================================
// Host function: memory_retrieve_cuda
//
// Forward-only pass through the 4-layer MLP (no autograd, no backward).
// Used by NeuralMemory.retrieve() — replaces the with torch.no_grad() path.
// ============================================================================

at::Tensor memory_retrieve_cuda(
    const at::Tensor& query,
    const MemoryMLPParams& params
) {
    TORCH_CHECK(query.is_cuda(),  "query must be on CUDA");
    TORCH_CHECK(query.dim() == 2, "query must be 2D [N, d]");

    const c10::cuda::CUDAGuard device_guard(query.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    at::NoGradGuard no_grad;

    int64_t N = query.size(0);
    int64_t d = query.size(1);

    // Layer 0: z1 = query @ W0.T + b0, h1 = silu(z1)
    at::Tensor z1 = at::addmm(params.b0.unsqueeze(0).expand({N, d}), query, params.W0.t());
    at::Tensor z1_fp32 = z1.to(at::kFloat);
    at::Tensor h1 = at::zeros_like(z1);
    {
        int64_t elems = N * d;
        int blocks = static_cast<int>((elems + kBlockSize - 1) / kBlockSize);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            z1.scalar_type(), "retrieve_silu_l1",
            [&]() {
                silu_forward_kernel<scalar_t><<<blocks, kBlockSize, 0, stream>>>(
                    z1.data_ptr<scalar_t>(),
                    h1.data_ptr<scalar_t>(),
                    z1_fp32.data_ptr<float>(),
                    elems
                );
            }
        );
        C10_CUDA_CHECK(cudaGetLastError());
    }

    // Layer 1: z2 = h1 @ W1.T + b1, h2 = silu(z2)
    at::Tensor z2 = at::addmm(params.b1.unsqueeze(0).expand({N, d}), h1, params.W1.t());
    at::Tensor z2_fp32 = z2.to(at::kFloat);
    at::Tensor h2 = at::zeros_like(z2);
    {
        int64_t elems = N * d;
        int blocks = static_cast<int>((elems + kBlockSize - 1) / kBlockSize);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            z2.scalar_type(), "retrieve_silu_l2",
            [&]() {
                silu_forward_kernel<scalar_t><<<blocks, kBlockSize, 0, stream>>>(
                    z2.data_ptr<scalar_t>(),
                    h2.data_ptr<scalar_t>(),
                    z2_fp32.data_ptr<float>(),
                    elems
                );
            }
        );
        C10_CUDA_CHECK(cudaGetLastError());
    }

    // Layer 2: z3 = h2 @ W2.T + b2, h3 = silu(z3)
    at::Tensor z3 = at::addmm(params.b2.unsqueeze(0).expand({N, d}), h2, params.W2.t());
    at::Tensor z3_fp32 = z3.to(at::kFloat);
    at::Tensor h3 = at::zeros_like(z3);
    {
        int64_t elems = N * d;
        int blocks = static_cast<int>((elems + kBlockSize - 1) / kBlockSize);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            z3.scalar_type(), "retrieve_silu_l3",
            [&]() {
                silu_forward_kernel<scalar_t><<<blocks, kBlockSize, 0, stream>>>(
                    z3.data_ptr<scalar_t>(),
                    h3.data_ptr<scalar_t>(),
                    z3_fp32.data_ptr<float>(),
                    elems
                );
            }
        );
        C10_CUDA_CHECK(cudaGetLastError());
    }

    // Layer 3 (output, no activation): h4 = h3 @ W3.T + b3
    at::Tensor h4 = at::addmm(params.b3.unsqueeze(0).expand({N, d}), h3, params.W3.t());

    return h4;
}

// ============================================================================
// Host function: memory_update_retrieve_cuda
//
// Combined update + retrieve.  The retrieve uses the UPDATED weights, which
// is the correct behavior for the MAC (memory-augmented context) variant
// where the memory update and readout happen in the same block.
//
// Implementation: run update, then run retrieve with updated params.
// The forward activations from update are discarded (retrieve does another
// forward pass with updated weights).  This costs an extra forward pass but
// avoids stale activations from before the weight update.
// ============================================================================

std::pair<float, at::Tensor> memory_update_retrieve_cuda(
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& query,
    MemoryMLPParams&  params,
    double eta,
    double theta,
    double alpha
) {
    float loss = memory_update_cuda(keys, values, params, eta, theta, alpha);
    at::Tensor output = memory_retrieve_cuda(query, params);
    return {loss, output};
}

} // namespace titan

// ============================================================================
// Flat-parameter wrappers — global namespace, match memory_ops.cpp declarations
//
// These construct a titan::MemoryMLPParams from the individual tensors and
// forward to the titan:: implementations.  They are the actual symbols that
// the linker resolves when memory_ops.cpp's TORCH_LIBRARY_IMPL lambdas call
// memory_update_cuda / memory_retrieve_cuda.
// ============================================================================

torch::Tensor memory_update_cuda(
    const torch::Tensor& keys,
    const torch::Tensor& values,
    torch::Tensor& w0,
    torch::Tensor& w1,
    torch::Tensor& w2,
    torch::Tensor& w3,
    torch::Tensor& b0,
    torch::Tensor& b1,
    torch::Tensor& b2,
    torch::Tensor& b3,
    torch::Tensor& momentum,
    float eta,
    float theta,
    float alpha
) {
    titan::MemoryMLPParams params;
    params.W0 = w0;
    params.W1 = w1;
    params.W2 = w2;
    params.W3 = w3;
    params.b0 = b0;
    params.b1 = b1;
    params.b2 = b2;
    params.b3 = b3;
    params.momentum_S = momentum;

    float loss_val = titan::memory_update_cuda(
        keys, values, params,
        static_cast<double>(eta),
        static_cast<double>(theta),
        static_cast<double>(alpha)
    );

    // Return loss as a 1-element fp32 scalar tensor (matches torch.library schema
    // "-> Tensor" and the Meta impl which returns empty({1}, kFloat32)).
    return torch::tensor({loss_val},
                         keys.options().dtype(torch::kFloat32));
}

torch::Tensor memory_retrieve_cuda(
    const torch::Tensor& query,
    const torch::Tensor& w0,
    const torch::Tensor& w1,
    const torch::Tensor& w2,
    const torch::Tensor& w3,
    const torch::Tensor& b0,
    const torch::Tensor& b1,
    const torch::Tensor& b2,
    const torch::Tensor& b3
) {
    titan::MemoryMLPParams params;
    params.W0 = w0;
    params.W1 = w1;
    params.W2 = w2;
    params.W3 = w3;
    params.b0 = b0;
    params.b1 = b1;
    params.b2 = b2;
    params.b3 = b3;
    // momentum_S not needed for retrieve-only path; leave default-constructed.

    return titan::memory_retrieve_cuda(query, params);
}
