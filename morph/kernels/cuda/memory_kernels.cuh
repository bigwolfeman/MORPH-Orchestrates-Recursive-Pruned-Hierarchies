// Copyright 2026 TitanMAC Authors. All rights reserved.
//
// memory_kernels.cuh — CUDA kernel declarations for fused neural memory
// update and retrieve operations.
//
// This replaces the torch.autograd.grad() path in NeuralMemory.update(),
// which causes a torch.compile graph break and ~2.5x slowdown.
//
// Architecture:
//   - 4-layer MLP: h0=k, h1=silu(W0@h0+b0), h2=silu(W1@h1+b1),
//                  h3=silu(W2@h2+b2), h4=W3@h3+b3
//   - Loss: L = mean((h4 - v)^2)
//   - Momentum: S_t = eta * S_{t-1} - theta * grad  (Titans Eq. 13)
//   - Weight update: W_t = (1 - alpha) * W_{t-1} + S_t  (Titans Eq. 14)
//
// Design:
//   - GEMMs delegated to cuBLAS via at::mm / at::addmm (1024x1024 dense)
//   - Custom kernels for: SiLU fwd/bwd, MSE grad, bias reduction,
//     gradient clip, fused momentum+weight update
//
// Namespace: titan::
// Target: sm_90a (compatible with RTX 5090 sm_120)
//
// Branch: 006-looped-block-ell

#pragma once

#include <cstdint>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace titan {

// ---------------------------------------------------------------------------
// MemoryMLPParams
//
// Convenience struct bundling all MLP weight/bias/momentum pointers for a
// 4-layer neural memory MLP.  All weight/bias tensors are bf16; the
// momentum buffer is fp32 for numerical stability of the running average.
//
// Layout of the momentum buffer (contiguous, flat, fp32):
//   [W0_flat | b0_flat | W1_flat | b1_flat |
//    W2_flat | b2_flat | W3_flat | b3_flat]
//
// Total elements: 4 * (1024*1024 + 1024) = 4,198,400 (~4.2M)
// ---------------------------------------------------------------------------
struct MemoryMLPParams {
    // Weight matrices [d, d] bf16
    at::Tensor W0, W1, W2, W3;
    // Bias vectors [d] bf16
    at::Tensor b0, b1, b2, b3;
    // Flat momentum buffer [n_params] fp32
    at::Tensor momentum_S;
};

// ---------------------------------------------------------------------------
// Elementwise kernel declarations
// ---------------------------------------------------------------------------

// silu_forward_kernel
//
// Computes h = silu(z) = z * sigmoid(z) elementwise and saves z into
// preact for use during the backward pass.
//
// Grid: (N + 255) / 256 blocks, 256 threads/block
// Inputs:
//   z    [N] bf16 – pre-activation (linear output before SiLU)
//   h    [N] bf16 – output activation (written)
//   preact [N] fp32 – pre-activation saved for backward (written, fp32 for
//                     numerical stability in sigmoid)
//   N    – number of elements
template <typename scalar_t>
__global__ void silu_forward_kernel(
    const scalar_t* __restrict__ z,
    scalar_t*       __restrict__ h,
    float*          __restrict__ preact,
    int64_t N
);

// silu_backward_kernel
//
// Computes dz = dh * silu'(z) where:
//   silu'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
//            = sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z))
//
// Reads pre-activation from saved fp32 preact buffer.
//
// Grid: (N + 255) / 256 blocks, 256 threads/block
template <typename scalar_t>
__global__ void silu_backward_kernel(
    const scalar_t* __restrict__ dh,
    const float*    __restrict__ preact,
    scalar_t*       __restrict__ dz,
    int64_t N
);

// mse_grad_kernel
//
// Computes the gradient of MSE loss w.r.t. predictions:
//   d_pred = 2 * (pred - target) / N
//
// Result is bf16 (same dtype as pred/target).
//
// Grid: (N + 255) / 256 blocks, 256 threads/block
template <typename scalar_t>
__global__ void mse_grad_kernel(
    const scalar_t* __restrict__ pred,
    const scalar_t* __restrict__ target,
    scalar_t*       __restrict__ d_pred,
    int64_t N
);

// bias_grad_reduce_kernel
//
// Reduces dz [N_rows, D] → db [D] by summing over the row dimension.
// Uses fp32 accumulation internally to avoid bf16 overflow.
//
// Grid: D blocks, 256 threads/block (each block handles one output column)
template <typename scalar_t>
__global__ void bias_grad_reduce_kernel(
    const scalar_t* __restrict__ dz,
    float*          __restrict__ db_fp32,   // fp32 accumulation buffer
    int64_t N_rows,
    int64_t D
);

// fp32_to_bf16_kernel
//
// Converts fp32 bias gradient accumulation buffer → bf16 bias delta tensor.
//
// Grid: (D + 255) / 256 blocks, 256 threads/block
__global__ void fp32_to_bf16_kernel(
    const float*                      __restrict__ src_fp32,
    at::BFloat16*                     __restrict__ dst_bf16,
    int64_t D
);

// momentum_weight_update_kernel
//
// Fused Titans Eq. 13 + 14 for a contiguous parameter chunk:
//
//   S  = eta * S  - theta * grad   (in fp32)
//   W  = (1 - alpha) * W + S      (bf16 in, fp32 S applied, bf16 out)
//
// All mutation happens in-place.  grad is the clipped flat gradient fp32.
//
// Grid: (n_params + 255) / 256 blocks, 256 threads/block
template <typename scalar_t>
__global__ void momentum_weight_update_kernel(
    const float*  __restrict__ grad_flat,   // clipped gradient [n_params] fp32
    float*        __restrict__ S,           // momentum buffer [n_params] fp32 (MUTATED)
    scalar_t*     __restrict__ W_flat,      // weight flat view [n_params] bf16 (MUTATED)
    int64_t n_params,
    float   eta,
    float   theta,
    float   alpha
);

// ---------------------------------------------------------------------------
// Host function declarations — struct-based (internal / combined op)
// ---------------------------------------------------------------------------

// memory_update_cuda (struct API — used internally by memory_update_retrieve_cuda)
//
// Fused forward + backward + momentum + weight update for the 4-layer
// neural memory MLP.  Replaces the torch.autograd.grad() call in Python.
//
// All mutations are performed under no-grad context.
// Returns the scalar MSE loss as fp32.
float memory_update_cuda(
    const at::Tensor&    keys,       // [N, d] bf16
    const at::Tensor&    values,     // [N, d] bf16
    MemoryMLPParams&     params,
    double               eta,
    double               theta,
    double               alpha
);

// memory_retrieve_cuda (struct API — used internally)
//
// Forward-only pass through the 4-layer MLP (no gradient computation).
at::Tensor memory_retrieve_cuda(
    const at::Tensor&        query,
    const MemoryMLPParams&   params
);

// memory_update_retrieve_cuda
//
// Combined update + retrieve in one kernel launch sequence.
// Useful when both operations occur in the same forward pass (MAC variant).
//
// Returns: {loss, output}
std::pair<float, at::Tensor> memory_update_retrieve_cuda(
    const at::Tensor&    keys,
    const at::Tensor&    values,
    const at::Tensor&    query,
    MemoryMLPParams&     params,
    double               eta,
    double               theta,
    double               alpha
);

} // namespace titan

// ---------------------------------------------------------------------------
// Host function declarations — flat-parameter API (torch.library linkage)
//
// These are the symbols that memory_ops.cpp forward-declares and calls via
// the TORCH_LIBRARY_IMPL dispatch lambdas.  They live in the global namespace
// (no titan:: prefix) to match memory_ops.cpp's forward declarations exactly.
//
// Each flat function constructs a titan::MemoryMLPParams internally and
// delegates to the corresponding titan:: implementation above.
// ---------------------------------------------------------------------------

// memory_update_cuda (flat API)
//
// Mutates w0..w3, b0..b3, momentum in-place.
// Returns a 1-element fp32 scalar tensor containing the MSE loss.
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
);

// memory_retrieve_cuda (flat API)
//
// Read-only forward pass.  Returns output [N, d] bf16.
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
);
