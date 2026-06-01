"""Multi-Rate Residual (MRR) — dimension-splitting residual stream (JAX/Flax).

Port of TPU/subq-attention/mrr.py (PyTorch) to Flax linen.

Motivation
----------
A looped transformer with 7 competing signals all writing into a single d_model
residual stream creates destructive interference. Multi-Rate Residual splits the
stream into three *channels* with different mixing / retention rates:

  Channel 0 — Compute (384 dims): primary attention + MLP outputs. Fast rate.
  Channel 1 — Context (256 dims): x0 skip, value embeds, loop injection.
  Channel 2 — Slow    (128 dims): slow-rate persistent channel (γ≈0.1). Reserved
                                  — neural memory (which fed it) is deferred.

Each sublayer gets learned per-channel parameters:
  alpha (3, softplus + L1-normalize): input mixing weights (currently identity-pass).
  gamma (3, softplus):                per-channel additive gain.

This adds only 6 × 2 sublayers × n_layers scalars to the model (e.g. 144 new
parameters for 12 layers). The attention/MLP modules themselves are unchanged.

References: Hyper-Connections (arXiv 2409.19606).
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


# ── Channel layout ─────────────────────────────────────────────────────────────
# DEFAULT_CHANNEL_DIMS must sum to d_model (768).
DEFAULT_CHANNEL_DIMS: tuple[int, ...] = (384, 256, 128)


def _make_splits(channel_dims: Sequence[int]) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for each channel."""
    splits: list[tuple[int, int]] = []
    start = 0
    for d in channel_dims:
        splits.append((start, start + d))
        start += d
    return splits


class MultiRateResidual(nn.Module):
    """Wrap a sublayer with multi-rate residual dimension-split dynamics (Flax).

    Before the sublayer (input mixing):
        x_mixed = concat(α₀·h[ch0], α₁·h[ch1], α₂·h[ch2])
        Alpha weights are softplus + L1-normalized. Compute channel starts
        with ~99% of weight so signal strength is preserved at init.

    After the sublayer (additive residual with per-channel gain):
        h[chᵢ] = h[chᵢ] + γᵢ · o[chᵢ]
        Gamma is a learned per-channel scale (softplus). ADDITIVE —
        not interpolation — so features accumulate through layers.

    Attributes
    ----------
    channel_dims : tuple[int, ...]
        Width of each channel. Must sum to d_model.
    alpha_init : tuple[float, ...]
        Pre-softplus values for alpha. Default gives compute ~99.3% at init.
    gamma_init : tuple[float, ...]
        Pre-softplus values for gamma (additive gain per channel).
    """

    channel_dims: tuple[int, ...] = DEFAULT_CHANNEL_DIMS
    alpha_init: tuple[float, ...] = (3.0, 0.01, 0.01)
    gamma_init: tuple[float, ...] = (1.0, 0.5, 0.1)

    @nn.compact
    def __call__(
        self,
        h: jnp.ndarray,
        sublayer_fn: Callable[..., jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        n = len(self.channel_dims)
        splits = _make_splits(self.channel_dims)

        # Alpha: softplus + L1-normalize per channel.
        # Stored as pre-softplus values so gradient passes through softplus.
        # Init: log(exp(init) - 1) = log(expm1(init))
        alpha_raw = self.param(
            "alpha_raw",
            lambda rng, shape: jnp.array(
                [math.log(math.expm1(max(a, 1e-6))) for a in self.alpha_init],
                dtype=jnp.float32,
            ),
            (n,),
        )

        # Gamma: per-channel additive gain (softplus, always positive).
        gamma_raw = self.param(
            "gamma_raw",
            lambda rng, shape: jnp.array(
                [math.log(math.expm1(max(g, 1e-6))) for g in self.gamma_init],
                dtype=jnp.float32,
            ),
            (n,),
        )

        # gamma: [n] — cast to match h dtype for bf16 safety
        gamma = jax.nn.softplus(gamma_raw).astype(h.dtype)  # [n]

        # Sublayer sees full h unchanged (alpha mixing for future; gamma routing now).
        o = sublayer_fn(h, *args, **kwargs)  # [B, S, D]

        # Additive residual with per-channel gain: h_new = h + γᵢ·o[chᵢ]
        # Assemble output without in-place ops (XLA-friendly concat).
        out_chunks = [
            h[..., s:e] + gamma[i] * o[..., s:e]
            for i, (s, e) in enumerate(splits)
        ]
        return jnp.concatenate(out_chunks, axis=-1)


class ChannelInject(nn.Module):
    """Inject a signal into a specific channel slice of the residual stream.

    Useful for targeted injection of:
      - x0 skip connection → Context channel (dims 384:640)
      - value embeddings   → Context channel (dims 384:640)
      - diagonal injection → Context channel (dims 384:640)

    The injection is: h[..., start:end] += scale * project(signal)

    A learned scalar gate (init from init_scale) modulates magnitude.
    An optional linear projection handles d_signal ≠ channel_width.

    Attributes
    ----------
    channel_start : int
    channel_end   : int
    d_signal      : int
        Dimensionality of the injected signal.
    init_scale    : float
        Initial raw scale value (not passed through any activation — raw scalar).
    """

    channel_start: int
    channel_end: int
    d_signal: int
    init_scale: float = 0.0

    @nn.compact
    def __call__(self, h: jnp.ndarray, signal: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            h:      [..., D] full residual stream.
            signal: [..., d_signal] signal to inject.

        Returns:
            h with channel slice updated (new array, no in-place).
        """
        channel_width = self.channel_end - self.channel_start

        # Learned scalar gate (raw — no activation, matches PyTorch log_scale param).
        scale_raw = self.param(
            "log_scale",
            nn.initializers.constant(self.init_scale),
            (),
        )
        scale = scale_raw.astype(h.dtype)

        # Optional projection if signal dim differs from channel width.
        if self.d_signal != channel_width:
            s = nn.Dense(
                channel_width,
                use_bias=False,
                kernel_init=nn.initializers.normal(0.02),
                dtype=h.dtype,
                name="proj",
            )(signal)
        else:
            s = signal.astype(h.dtype)

        # Assemble updated h without mutation (XLA-friendly).
        prefix = h[..., :self.channel_start]
        target = h[..., self.channel_start:self.channel_end] + scale * s
        suffix = h[..., self.channel_end:]
        return jnp.concatenate([prefix, target, suffix], axis=-1)


class MORPHBlock(nn.Module):
    """TransformerBlock replacement using multi-rate residual dynamics (Flax).

    Wraps attention and MLP each in a MultiRateResidual. The underlying attention
    and MLP modules are unchanged (d_model-dim in, d_model-dim out). Only the
    residual connection has per-channel alpha/gamma parameters.

    RMSNorm operates on the full mixed-channel input (not per-channel).

    Attributes
    ----------
    attn_module : nn.Module
        Instantiated attention module that accepts (x, deterministic) → x.
    mlp_module  : nn.Module
        Instantiated MLP module that accepts (x, deterministic) → x.
    d_model     : int
        Full residual width (sum of channel_dims).
    norm_eps    : float
        RMSNorm epsilon.
    channel_dims: tuple[int, ...]
        Channel widths. Must sum to d_model.
    """

    attn_module: nn.Module
    mlp_module: nn.Module
    d_model: int
    norm_eps: float = 1e-5
    channel_dims: tuple[int, ...] = DEFAULT_CHANNEL_DIMS

    @nn.compact
    def __call__(self, h: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            h:             [B, T, D] residual stream.
            deterministic: dropout flag.

        Returns:
            h updated through attention + MLP with multi-rate residuals.
        """
        from .attention import RMSNorm

        norm_attn = RMSNorm(eps=self.norm_eps, name="norm_attn")
        norm_mlp  = RMSNorm(eps=self.norm_eps, name="norm_mlp")

        mrr_attn = MultiRateResidual(channel_dims=self.channel_dims, name="mrr_attn")
        mrr_mlp  = MultiRateResidual(channel_dims=self.channel_dims, name="mrr_mlp")

        def _attn_fn(x):
            return self.attn_module(norm_attn(x), deterministic=deterministic)

        h = mrr_attn(h, _attn_fn)

        def _mlp_fn(x):
            return self.mlp_module(norm_mlp(x), deterministic=deterministic)

        h = mrr_mlp(h, _mlp_fn)
        return h
