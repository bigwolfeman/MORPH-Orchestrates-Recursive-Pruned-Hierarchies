"""
Triton kernels for Neural Memory MLP backward + update.

These kernels replace torch.autograd.grad() with manual backpropagation,
enabling fullgraph=True compilation and eliminating retain_graph overhead.

Key algorithms:
1. Manual gradient computation through 2-layer MLP
2. Fused gradient clipping + momentum + weight update

Shapes (typical):
- N = batch * seq = 2048
- D = d_memory = 512
- H = hidden_dim = 512
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple


# =============================================================================
# Helper: SiLU backward
# =============================================================================


def silu_backward(d_h: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of SiLU activation.

    SiLU: silu(x) = x * sigmoid(x)
    SiLU': silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    Args:
        d_h: Gradient w.r.t. silu output [N, H]
        z1: Pre-activation input to silu [N, H]

    Returns:
        Gradient w.r.t. silu input [N, H]
    """
    sigma = torch.sigmoid(z1)
    silu_grad = sigma * (1.0 + z1 * (1.0 - sigma))
    return d_h * silu_grad


# =============================================================================
# Triton kernel: Fused gradient clipping + momentum + weight update
# =============================================================================


@triton.jit
def fused_clip_momentum_update_kernel(
    # Gradients (read-only after computation)
    grad_ptr,
    # Momentum buffer (read-write)
    momentum_ptr,
    # Weights (read-write)
    weight_ptr,
    # Scale tensor pointer (computed on GPU)
    scale_ptr,
    # Scalars
    eta,  # momentum decay
    theta,  # learning rate
    alpha,  # forget factor (1 - alpha = retention)
    # Size
    n_elements,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for:
    1. Apply gradient scaling (clipping)
    2. Momentum update: S = η*S - θ*g*scale
    3. Weight update: W = (1-α)*W + S
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load scale (same for all blocks)
    grad_scale = tl.load(scale_ptr)

    # Load
    g = tl.load(grad_ptr + offs, mask=mask, other=0.0)
    s = tl.load(momentum_ptr + offs, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)

    # Scale gradient
    g_scaled = g * grad_scale

    # Momentum update: S = η*S - θ*g
    s_new = eta * s - theta * g_scaled

    # Weight update: W = (1-α)*W + S
    w_new = (1.0 - alpha) * w + s_new

    # Store
    tl.store(momentum_ptr + offs, s_new, mask=mask)
    tl.store(weight_ptr + offs, w_new, mask=mask)


# =============================================================================
# PyTorch wrapper for fused update
# =============================================================================


def fused_clip_momentum_update(
    grads: Tuple[torch.Tensor, ...],
    momentums: Tuple[torch.Tensor, ...],
    weights: Tuple[torch.Tensor, ...],
    grad_norm: torch.Tensor,
    max_grad_norm: float,
    eta: float,
    theta: float,
    alpha: float,
):
    """
    Apply fused gradient clipping + momentum + weight update.

    Args:
        grads: Tuple of gradient tensors
        momentums: Tuple of momentum buffers (updated in-place)
        weights: Tuple of weight tensors (updated in-place)
        grad_norm: Pre-computed gradient norm
        max_grad_norm: Maximum gradient norm for clipping
        eta: Momentum decay factor
        theta: Memory learning rate
        alpha: Forget factor
    """
    # Compute gradient scale on GPU (branchless, no .item()!)
    # scale = min(1.0, max_grad_norm / grad_norm)
    scale = torch.clamp(max_grad_norm / (grad_norm + 1e-8), max=1.0)

    BLOCK_SIZE = 1024

    for grad, momentum, weight in zip(grads, momentums, weights):
        n_elements = grad.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        # Flatten tensors for kernel
        grad_flat = grad.view(-1).contiguous()
        momentum_flat = momentum.view(-1)
        weight_flat = weight.view(-1)

        fused_clip_momentum_update_kernel[grid](
            grad_flat,
            momentum_flat,
            weight_flat,
            scale,  # Pass as tensor pointer
            eta,
            theta,
            alpha,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )


# =============================================================================
# Main backward + update function
# =============================================================================


def memory_mlp_backward_update(
    k: torch.Tensor,  # [N, D] keys
    v: torch.Tensor,  # [N, D] target values
    z1: torch.Tensor,  # [N, H] pre-activation (saved from forward)
    h: torch.Tensor,  # [N, H] hidden activation (saved from forward)
    y: torch.Tensor,  # [N, D] output (saved from forward)
    W1: torch.Tensor,  # [H, D] layer 1 weight (updated in-place)
    B1: torch.Tensor,  # [H] layer 1 bias (updated in-place)
    W2: torch.Tensor,  # [D, H] layer 2 weight (updated in-place)
    B2: torch.Tensor,  # [D] layer 2 bias (updated in-place)
    S_W1: torch.Tensor,  # [H, D] momentum for W1 (updated in-place)
    S_B1: torch.Tensor,  # [H] momentum for B1 (updated in-place)
    S_W2: torch.Tensor,  # [D, H] momentum for W2 (updated in-place)
    S_B2: torch.Tensor,  # [D] momentum for B2 (updated in-place)
    alpha: float,  # forget gate output
    eta: float,  # decay gate output
    theta: float,  # memory learning rate
    max_grad_norm: float = 1.0,
) -> torch.Tensor:
    """
    Compute gradients manually and apply momentum + weight update.

    This function replaces torch.autograd.grad() with manual backpropagation,
    enabling fullgraph=True compilation.

    The forward pass computed:
        z1 = k @ W1.T + B1
        h = silu(z1)
        y = h @ W2.T + B2
        loss = MSE(y, v)

    This function computes gradients of loss w.r.t. W1, B1, W2, B2 and
    applies the Titans memory update:
        S = η*S - θ*∇L
        W = (1-α)*W + S

    Args:
        k: Keys [N, D_in]
        v: Target values [N, D_out]
        z1, h, y: Saved activations from forward pass
        W1, B1, W2, B2: MLP parameters (updated in-place)
        S_W1, S_B1, S_W2, S_B2: Momentum buffers (updated in-place)
        alpha: Forget gate output (0=retain, 1=forget)
        eta: Decay gate output (momentum decay)
        theta: Memory learning rate
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        grad_norm: Gradient norm before clipping
    """
    N, D_out = y.shape
    _, H = h.shape
    _, D_in = k.shape

    # ==================== STEP 1: d_loss/d_y ====================
    # MSE loss: L = (1/N/D) * sum((y - v)^2)
    # dL/dy = (2/N/D) * (y - v)
    d_y = (2.0 / (N * D_out)) * (y - v)  # [N, D_out]

    # ==================== STEP 2: Layer 2 backward ====================
    # y = h @ W2.T + B2
    d_W2 = d_y.T @ h  # [D_out, H]
    d_B2 = d_y.sum(dim=0)  # [D_out]
    d_h = d_y @ W2  # [N, H]

    # ==================== STEP 3: SiLU backward ====================
    d_z1 = silu_backward(d_h, z1)  # [N, H]

    # ==================== STEP 4: Layer 1 backward ====================
    d_W1 = d_z1.T @ k  # [H, D_in]
    d_B1 = d_z1.sum(dim=0)  # [H]

    # ==================== STEP 5: Compute gradient norm ====================
    grad_norm_sq = d_W1.pow(2).sum() + d_B1.pow(2).sum() + d_W2.pow(2).sum() + d_B2.pow(2).sum()
    grad_norm = grad_norm_sq.sqrt()

    # ==================== STEP 6: Fused clip + momentum + update ====================
    fused_clip_momentum_update(
        grads=(d_W1, d_B1, d_W2, d_B2),
        momentums=(S_W1, S_B1, S_W2, S_B2),
        weights=(W1, B1, W2, B2),
        grad_norm=grad_norm,
        max_grad_norm=max_grad_norm,
        eta=eta,
        theta=theta,
        alpha=alpha,
    )

    return grad_norm


# =============================================================================
# Class wrapper for stateful usage
# =============================================================================


class MemoryMLPBackwardUpdate:
    """
    Stateful wrapper for memory MLP backward + update operations.

    Usage:
        updater = MemoryMLPBackwardUpdate()

        # In training loop:
        grad_norm = updater(
            k, v, z1, h, y,
            W1, B1, W2, B2,
            S_W1, S_B1, S_W2, S_B2,
            alpha, eta, theta,
        )
    """

    def __init__(self, max_grad_norm: float = 1.0):
        self.max_grad_norm = max_grad_norm

    def __call__(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        z1: torch.Tensor,
        h: torch.Tensor,
        y: torch.Tensor,
        W1: torch.Tensor,
        B1: torch.Tensor,
        W2: torch.Tensor,
        B2: torch.Tensor,
        S_W1: torch.Tensor,
        S_B1: torch.Tensor,
        S_W2: torch.Tensor,
        S_B2: torch.Tensor,
        alpha: float,
        eta: float,
        theta: float,
    ) -> torch.Tensor:
        return memory_mlp_backward_update(
            k,
            v,
            z1,
            h,
            y,
            W1,
            B1,
            W2,
            B2,
            S_W1,
            S_B1,
            S_W2,
            S_B2,
            alpha,
            eta,
            theta,
            self.max_grad_norm,
        )


# =============================================================================
# Verification function
# =============================================================================


def verify_against_autograd(
    k: torch.Tensor,
    v: torch.Tensor,
    W1: torch.Tensor,
    B1: torch.Tensor,
    W2: torch.Tensor,
    B2: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """
    Verify that manual backward matches torch.autograd.grad().

    Returns True if all gradients match within tolerance.
    """
    # Clone weights with requires_grad for autograd
    W1_auto = W1.clone().detach().requires_grad_(True)
    B1_auto = B1.clone().detach().requires_grad_(True)
    W2_auto = W2.clone().detach().requires_grad_(True)
    B2_auto = B2.clone().detach().requires_grad_(True)

    # Forward pass for autograd
    z1_auto = F.linear(k, W1_auto, B1_auto)
    h_auto = F.silu(z1_auto)
    y_auto = F.linear(h_auto, W2_auto, B2_auto)
    loss_auto = F.mse_loss(y_auto, v)

    # Autograd backward
    grads_auto = torch.autograd.grad(
        loss_auto,
        [W1_auto, B1_auto, W2_auto, B2_auto],
        retain_graph=False,
    )

    # Forward pass for manual (save intermediates)
    z1 = F.linear(k, W1, B1)
    h = F.silu(z1)
    y = F.linear(h, W2, B2)

    # Manual backward (just gradients, no update)
    N, D_out = y.shape
    d_y = (2.0 / (N * D_out)) * (y - v)
    d_W2 = d_y.T @ h
    d_B2 = d_y.sum(dim=0)
    d_h = d_y @ W2
    d_z1 = silu_backward(d_h, z1)
    d_W1 = d_z1.T @ k
    d_B1 = d_z1.sum(dim=0)

    grads_manual = [d_W1, d_B1, d_W2, d_B2]

    # Compare
    names = ["d_W1", "d_B1", "d_W2", "d_B2"]
    all_match = True
    for name, g_auto, g_manual in zip(names, grads_auto, grads_manual):
        if not torch.allclose(g_auto, g_manual, rtol=rtol, atol=atol):
            max_diff = (g_auto - g_manual).abs().max().item()
            print(f"MISMATCH in {name}: max_diff = {max_diff}")
            all_match = False
        else:
            print(f"OK: {name} matches")

    return all_match


if __name__ == "__main__":
    print("Testing memory MLP backward + update kernels...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    N, D, H = 256, 128, 128

    k = torch.randn(N, D, device=device, dtype=dtype)
    v = torch.randn(N, D, device=device, dtype=dtype)
    W1 = torch.randn(H, D, device=device, dtype=dtype) * 0.02
    B1 = torch.zeros(H, device=device, dtype=dtype)
    W2 = torch.randn(D, H, device=device, dtype=dtype) * 0.02
    B2 = torch.zeros(D, device=device, dtype=dtype)

    print("\n[Test 1] Gradient verification against autograd:")
    success = verify_against_autograd(k, v, W1, B1, W2, B2)
    print(f"Result: {'PASS' if success else 'FAIL'}")

    if success:
        print("\n[Test 2] Full backward + update:")
        # Create momentum buffers
        S_W1 = torch.zeros_like(W1)
        S_B1 = torch.zeros_like(B1)
        S_W2 = torch.zeros_like(W2)
        S_B2 = torch.zeros_like(B2)

        # Forward pass
        z1 = F.linear(k, W1, B1)
        h = F.silu(z1)
        y = F.linear(h, W2, B2)

        # Backward + update
        W1_before = W1.clone()
        grad_norm = memory_mlp_backward_update(
            k,
            v,
            z1,
            h,
            y,
            W1,
            B1,
            W2,
            B2,
            S_W1,
            S_B1,
            S_W2,
            S_B2,
            alpha=0.01,
            eta=0.9,
            theta=0.01,
        )

        W1_changed = not torch.allclose(W1, W1_before)
        print(f"Gradient norm: {grad_norm.item():.6f}")
        print(f"Weights updated: {W1_changed}")
        print(f"Result: {'PASS' if W1_changed else 'FAIL'}")
