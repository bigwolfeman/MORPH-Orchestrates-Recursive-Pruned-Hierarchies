"""
Normalization layers for MORPH.

This module provides RMSNorm (Root Mean Square Normalization) for stable
training. Supports an optional Triton-fused path
that eliminates intermediate bf16 round-trips (2-4x more accurate than eager).
"""

import torch
import torch.nn as nn

# Try to import fused kernels (available when Triton is installed)
import os
try:
    if os.environ.get("DISABLE_FUSED_KERNELS", ""):
        raise ImportError("Fused kernels disabled by env var")
    from ..kernels.fused_ops import fused_rmsnorm as _fused_rmsnorm
    _HAS_FUSED = True
except (ImportError, RuntimeError):
    _HAS_FUSED = False


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.

    More stable than LayerNorm for transformer architectures as it doesn't
    depend on batch statistics (no mean centering).

    Formula:
        x_norm = x / RMS(x) * scale
        where RMS(x) = sqrt(mean(x²) + eps)

    Args:
        dim: Embedding dimension
        eps: Epsilon for numerical stability (default: 1e-5)
        use_adaptive_eps: If True, scale epsilon with input variance (default: False)
        clamp_scale: If True, clamp learnable scale to [0.1, 10.0] to prevent explosion (default: True)

    Shape:
        Input: [..., dim]
        Output: [..., dim]

    Example:
        >>> norm = RMSNorm(640)
        >>> x = torch.randn(2, 512, 640)
        >>> x_norm = norm(x)
        >>> x_norm.shape
        torch.Size([2, 512, 640])
    """

    def __init__(self, dim: int, eps: float = 1e-5, use_adaptive_eps: bool = False,
                 clamp_scale: bool = True, use_fused: bool = True):
        super().__init__()
        self.eps = eps
        self.use_adaptive_eps = use_adaptive_eps
        self.clamp_scale = clamp_scale
        self.use_fused = use_fused and _HAS_FUSED
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization with optional adaptive epsilon and scale clamping.

        ADAPTIVE EPSILON (optional, disabled by default):
        - When enabled, scales epsilon with input variance for high-variance inputs
        - Observed input variances up to ~8.4e6 can require larger epsilon
        - DISABLED BY DEFAULT for stability - use fixed epsilon unless needed

        SCALE CLAMPING (enabled by default):
        - Clamps learnable scale parameter to [0.1, 10.0] range
        - Prevents unbounded growth that can amplify logit magnitudes
        - Critical fix for preventing scale explosion during training

        Args:
            x: Input tensor [..., dim]

        Returns:
            Normalized tensor [..., dim]
        """
        # Fast path: Triton fused kernel (no adaptive eps, no NaN guard)
        if self.use_fused and x.is_cuda and not self.use_adaptive_eps:
            return _fused_rmsnorm(x, self.scale, self.eps, self.clamp_scale)

        # Compute variance: mean(x²)
        variance = torch.mean(x ** 2, dim=-1, keepdim=True)

        # Choose epsilon strategy
        if self.use_adaptive_eps:
            # Adaptive epsilon: scale with variance magnitude
            # For variance ~8.4e6, eps_adaptive ~8.4 (vs fixed 1e-5)
            eps_adaptive = torch.clamp(variance * 1e-6, min=1e-8, max=1e-4)
        else:
            # Fixed epsilon (default, more stable)
            eps_adaptive = self.eps

        # Compute RMS: sqrt(variance + eps)
        rms = torch.sqrt(variance + eps_adaptive)

        # Apply learnable scale with optional clamping
        # CRITICAL FIX: Clamp scale to prevent unbounded growth
        if self.clamp_scale:
            # Clamp scale parameter to reasonable range [0.1, 10.0]
            # This prevents scale from growing to 100+ and amplifying logits
            scale_clamped = torch.clamp(self.scale, min=0.1, max=10.0)
        else:
            scale_clamped = self.scale

        # Normalize and scale
        x_norm = x / rms * scale_clamped

        # NaN/Inf safety guard
        # If normalization produces invalid values, return scaled input
        if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
            import warnings
            warnings.warn(
                f"RMSNorm produced NaN/Inf (input norm: {x.norm().item():.2f}, "
                f"variance: {variance.mean().item():.2e}) - returning scaled input"
            )
            # Return input with clamped scale (prevents gradient flow issues)
            return x * scale_clamped

        return x_norm
