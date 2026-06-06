"""Multi-Rate Residual (MRR) — dimension-splitting residual stream.

Motivation
----------
A looped transformer with many competing signals (attention, MLP, x0 skip,
value embeds, diagonal injection, bigram) all writing into a single
d_model-dim residual stream creates destructive interference during
autoregressive generation even when teacher-forced PPL is low.

Multi-Rate Residual splits the stream into three *channels* with different
mixing / retention rates, so each signal type has a dedicated slice and
retention timescale:

  Channel 0 — Compute (384 dims): primary attention + MLP outputs. Fast rate.
  Channel 1 — Context (256 dims): x0 skip, value embeds, loop injection.
  Channel 2 — Slow    (128 dims): slow-rate persistent channel (γ≈0.1).
                                  Reserved — neural memory (which fed it) is
                                  deferred; the slow channel itself stays.

Each sublayer gets learned per-channel parameters:
  alpha (n, softplus + L1-normalize): input mixing weights across channels.
  gamma (n, softplus):                per-channel additive gain.
                                      compute≈1.0, context≈0.5, slow≈0.1.

This adds only n_channels × n_sublayers × n_layers scalars to the model.
For 3 channels × 2 sublayers × 12 layers = 72 new scalars (144 parameters
total for alpha + gamma).

Note: this is a simpler per-channel residual scaling — NOT the paper's full
mHC (multi-channel hyper-connections, arXiv 2409.19606) which also mixes
input representations across channels. We use the dimension-splitting residual
stream idea without the input mixing component.

Design notes
------------
- Standalone module: no imports from MORPH internals. Drop in to any model.
- bf16 compatible: all dtype casts handled at injection boundaries.
- torch.compile friendly: no Python control flow on tensor values, no in-place
  ops on views, no dynamic shapes.
- MORPHBlock accepts the attention and MLP as pre-built nn.Module instances,
  keeping this file decoupled from model architecture.
"""

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Channel layout ────────────────────────────────────────────────────────────

# DEFAULT_CHANNEL_DIMS must sum to d_model (768 default).
#   Ch0 — Compute (384): attention + MLP primary output. Fast.
#   Ch1 — Context (256): x0 skip, value embeds, loop injection.
#   Ch2 — Slow    (128): slow-rate persistent channel (γ≈0.1). Reserved.
DEFAULT_CHANNEL_DIMS: tuple[int, ...] = (384, 256, 128)


def _make_slices(channel_dims: tuple[int, ...]) -> list[slice]:
    """Convert channel widths to index slices into the d_model dimension."""
    slices: list[slice] = []
    start = 0
    for d in channel_dims:
        slices.append(slice(start, start + d))
        start += d
    return slices


# ── MultiRateResidual ─────────────────────────────────────────────────────────

class MultiRateResidual(nn.Module):
    """Wrap a sublayer with multi-rate residual (MRR) dimension-split dynamics.

    After the sublayer (additive residual with per-channel gain):
        h[chᵢ] = h[chᵢ] + γᵢ · o[chᵢ]

    Gamma is learned via softplus (always positive). Different channels get
    different update rates: compute≈1.0 (full update), context≈0.5, slow≈0.1.

    The sublayer always sees the full unchanged h — channel separation is
    achieved purely via the per-channel gamma on the output side. This keeps
    the attention and MLP modules unchanged and avoids input mixing at init.

    Parameters
    ----------
    channel_dims : tuple[int, ...]
        Width of each channel. Must sum to d_model.
    alpha_init : tuple[float, ...]
        Pre-softplus values for alpha (input mixing). Kept for API completeness;
        not applied in forward (sublayer always receives full h).
    gamma_init : tuple[float, ...]
        Pre-softplus values for gamma (additive gain per channel).
        Default (1.0, 0.5, 0.1) → compute gets full sublayer output, context
        gets half, slow channel gets 10%.
    """

    def __init__(
        self,
        channel_dims: tuple[int, ...] = DEFAULT_CHANNEL_DIMS,
        alpha_init: tuple[float, ...] = (3.0, 0.01, 0.01),
        gamma_init: tuple[float, ...] = (1.0, 0.5, 0.1),
    ):
        super().__init__()
        n = len(channel_dims)
        assert len(alpha_init) == n, "alpha_init length must match n_channels"
        assert len(gamma_init) == n, "gamma_init length must match n_channels"

        self.channel_dims = channel_dims
        self.slices = _make_slices(channel_dims)
        self.n_channels = n

        # Alpha: softplus + L1-normalize. Stored for potential future use /
        # regularization loss terms, but not applied in forward.
        self.alpha_raw = nn.Parameter(
            torch.tensor(
                [math.log(math.expm1(max(a, 1e-6))) for a in alpha_init]
            )
        )

        # Gamma: per-channel additive gain via softplus (always positive).
        self.gamma_raw = nn.Parameter(
            torch.tensor(
                [math.log(math.expm1(max(g, 1e-6))) for g in gamma_init]
            )
        )

    def _alpha(self) -> Tensor:
        """Normalized mixing weights, shape [n_channels]."""
        a = F.softplus(self.alpha_raw)
        return a / (a.sum() + 1e-8)

    def _gamma(self) -> Tensor:
        """Additive gain per channel, shape [n_channels]."""
        return F.softplus(self.gamma_raw)

    def forward(
        self,
        h: Tensor,
        sublayer_fn: Callable[..., Tensor],
        *args,
        **kwargs,
    ) -> Tensor:
        """Apply sublayer with MRR channel-split additive residual.

        Args:
            h:           [B, S, D] full residual stream.
            sublayer_fn: callable that takes h and optional args/kwargs,
                         returns [B, S, D] sublayer output.
            *args, **kwargs: forwarded to sublayer_fn.

        Returns:
            [B, S, D] updated residual stream.
        """
        gamma = self._gamma()           # [n_channels]

        o = sublayer_fn(h, *args, **kwargs)

        # Additive residual with per-channel gain: h_new = h + γᵢ·o per channel.
        slices = self.slices
        out_chunks = [
            h[..., slices[i]] + gamma[i] * o[..., slices[i]]
            for i in range(self.n_channels)
        ]
        return torch.cat(out_chunks, dim=-1)


class StandardResidual(nn.Module):
    """Plain additive residual: h_new = h + sublayer(h).

    The MRR-ablation alternative to MultiRateResidual. MRR's only effect over a
    standard residual is three learned scalar gains (γ ≈ compute 1.0 / context 0.5 /
    slow 0.1) applied to channel slices — and the slow channel existed to carry the
    (since-removed) neural-memory signal. This collapses MRR's 3-way slice+scalar+cat
    (~16 ms/step of copies) to a single fused add. Same forward(h, sublayer_fn, ...)
    interface as MultiRateResidual so MORPHBlock swaps it in at construction
    (branch-free; no runtime flag in the hot path). Channel slices still exist for
    ChannelInject — only the per-channel *residual gain* is removed.
    """

    def forward(
        self,
        h: Tensor,
        sublayer_fn: Callable[..., Tensor],
        *args,
        **kwargs,
    ) -> Tensor:
        return h + sublayer_fn(h, *args, **kwargs)


# ── ChannelInject ─────────────────────────────────────────────────────────────

class ChannelInject(nn.Module):
    """Inject a signal into a specific channel slice of the residual stream.

    Useful for targeted injection of:
      - x0 skip connection → Context channel (e.g. dims 384:640)
      - value embeddings   → Context channel
      - diagonal injection → Context channel

    Injection: h[..., start:end] += scale · project(signal)

    A learned raw scalar `log_scale` modulates magnitude (not sigmoid-gated —
    allows negative scales, simpler gradient flow early in training).
    An optional Linear projection handles d_signal ≠ channel_width.

    All tensors constructed without in-place ops for torch.compile safety.

    Args:
        channel_start: start index into d_model.
        channel_end:   end index into d_model.
        d_signal:      dimension of the injected signal.
        init_scale:    initial value of the raw scalar gate. Use 0.0 to
                       start with zero injection (safe for all signal types).
    """

    def __init__(
        self,
        channel_start: int,
        channel_end: int,
        d_signal: int,
        init_scale: float = 0.0,
    ):
        super().__init__()
        self.start = channel_start
        self.end   = channel_end
        channel_width = channel_end - channel_start

        self.log_scale = nn.Parameter(torch.tensor(float(init_scale)))

        if d_signal != channel_width:
            self.proj: nn.Module = nn.Linear(d_signal, channel_width, bias=False)
            nn.init.normal_(self.proj.weight, std=0.02)      # type: ignore[union-attr]
        else:
            self.proj = nn.Identity()

    def precompute(self, signal: Tensor) -> Tensor:
        """Project + scale a signal into the channel-width additive term.

        Returns ``scale · project(signal)`` of shape ``[..., channel_width]``.

        This is the loop-invariant part of :meth:`forward` for a signal that
        does not change across iterations (e.g. the cloned ``x0`` skip).
        Compute it once outside the loop, then feed each iteration through
        :meth:`apply_precomputed` to avoid recomputing the projection.
        """
        if isinstance(self.proj, nn.Linear):
            s = F.linear(signal, self.proj.weight.to(signal.dtype))
        else:
            s = signal
        scale = self.log_scale.to(s.dtype)
        return scale * s

    def apply_precomputed(self, h: Tensor, term: Tensor) -> Tensor:
        """Add a pre-projected additive ``term`` to the channel slice of h.

        ``term`` must be the output of :meth:`precompute` (shape
        ``[..., channel_width]``). Equivalent to :meth:`forward` but skips the
        projection + scale, which the caller has already done once.

        Stream-adaptive: when the carrier ``h`` is an ``[B, S, n, C]`` Hyper-Connection
        n-stream tensor but ``term`` is a single-stream ``[B, S, W]`` signal (x0 / value
        embeds, which live outside the streams), broadcast the signal into every stream by
        inserting the stream axis. A native n-stream signal (term.ndim == h.ndim) is added
        as-is. Single-stream carriers (h.ndim == 3) are unaffected.
        """
        if term.ndim == h.ndim - 1:
            term = term.unsqueeze(-2)            # [B,S,W] → [B,S,1,W] broadcasts over n streams
        prefix = h[..., :self.start]
        target = h[..., self.start:self.end] + term.to(h.dtype)
        suffix = h[..., self.end:]
        return torch.cat([prefix, target, suffix], dim=-1)

    def forward(self, h: Tensor, signal: Tensor) -> Tensor:
        """Inject signal into channel slice of h.

        Args:
            h:      [B, S, D] full residual stream.
            signal: [B, S, d_signal] signal to inject.

        Returns:
            [B, S, D] h with channel slice updated. No in-place ops.
        """
        # forward == precompute (projection) then apply. Kept as one path for
        # the prelude/coda (called once each, no loop-invariance to exploit).
        return self.apply_precomputed(h, self.precompute(signal))


# ── MORPHBlock ────────────────────────────────────────────────────────────────

class MORPHBlock(nn.Module):
    """TransformerBlock with multi-rate residual (MRR) dimension-split dynamics.

    Wraps attention and MLP each in a MultiRateResidual with independent
    per-channel alpha/gamma parameters. The underlying attention and MLP modules
    are unchanged (d_model-dim in, d_model-dim out). Only the residual connection
    has per-channel learned gain parameters.

    Accepts pre-built attention and MLP modules so this file stays decoupled
    from the MORPH model internals. The caller is responsible for constructing
    norm layers and sublayer modules.

    RMSNorm (or any norm) operates on the full mixed-channel input — not
    per-channel — because the channel split only affects the residual update,
    not the sublayer computation.

    Args:
        norm_attn:    normalization module for the attention sublayer.
        attn:         attention module. forward(x) → [B, T, D].
        norm_mlp:     normalization module for the MLP sublayer.
        mlp:          MLP module. forward(x) → [B, T, D].
        channel_dims: channel widths. Must sum to d_model.
        dropout:      dropout rate applied after each sublayer output.

    Usage::

        block = MORPHBlock(
            norm_attn=RMSNorm(d_model),
            attn=MyAttention(cfg),
            norm_mlp=RMSNorm(d_model),
            mlp=SwiGLU(d_model, d_ff),
            channel_dims=DEFAULT_CHANNEL_DIMS,
        )
        h = block(h)                         # basic forward
        h = block(h, attn_kwargs={"n_skip_rope": 2})  # pass kwargs to attn
    """

    def __init__(
        self,
        norm_attn: nn.Module,
        attn: nn.Module,
        norm_mlp: nn.Module,
        mlp: nn.Module,
        channel_dims: tuple[int, ...] = DEFAULT_CHANNEL_DIMS,
        dropout: float = 0.0,
        use_mrr: bool = True,
        residual_mode: str | None = None,
        d_model: int | None = None,
        hc_kwargs: dict | None = None,
    ):
        super().__init__()
        self.norm_attn = norm_attn
        self.attention = attn
        self.norm_mlp  = norm_mlp
        self.mlp       = mlp
        self.drop      = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Residual wrapper chosen at CONSTRUCTION (branch-free hot path). residual_mode
        # supersedes the legacy use_mrr bool (use_mrr → "mrr"/"standard" when mode unset):
        #   "mrr"      — MultiRateResidual: per-channel learned gain (3-way slice+cat).
        #   "standard" — StandardResidual:  plain fused add (the MRR-removal ablation).
        #   "hc_cayley"/"hc_sinkhorn" — HyperConnectionResidual: the *real* n-stream
        #                 hyper-connection (carrier becomes [B,S,n,C]); orthogonal (Cayley/
        #                 JPmHC) or doubly-stochastic (Sinkhorn/mHC) stream mixer.
        if residual_mode is None:
            residual_mode = "mrr" if use_mrr else "standard"
        self.residual_mode = residual_mode

        if residual_mode == "mrr":
            self.mrr_attn: nn.Module = MultiRateResidual(channel_dims=channel_dims)
            self.mrr_mlp:  nn.Module = MultiRateResidual(channel_dims=channel_dims)
        elif residual_mode == "standard":
            self.mrr_attn = StandardResidual()
            self.mrr_mlp  = StandardResidual()
        elif residual_mode in ("hc_cayley", "hc_sinkhorn"):
            from .hyper_connections import HyperConnectionResidual
            assert d_model is not None, "HC residual needs d_model"
            manifold = "cayley" if residual_mode == "hc_cayley" else "sinkhorn"
            hk = dict(hc_kwargs or {})
            self.mrr_attn = HyperConnectionResidual(d_model, manifold=manifold, **hk)
            self.mrr_mlp  = HyperConnectionResidual(d_model, manifold=manifold, **hk)
        else:
            raise ValueError(f"unknown residual_mode {residual_mode!r}")

    def forward(
        self,
        h: Tensor,
        attn_kwargs: dict | None = None,
        mlp_kwargs: dict | None = None,
        next_inject_term: Tensor | None = None,
    ) -> Tensor:
        """Forward pass: attention sublayer then MLP sublayer with MRR residuals.

        Args:
            h:           [B, T, D] residual stream.
            attn_kwargs: optional keyword arguments forwarded to attention.
            mlp_kwargs:  optional keyword arguments forwarded to mlp.
            next_inject_term: [B, S, C] | None — carrier-engine (HC only): the NEXT layer's
                         injection term, folded into THIS block's MLP-residual POST write
                         (the block's last carrier write), so the next layer skips a separate
                         _apply_injection. Only set in HC carrier-engine mode.

        Returns:
            [B, T, D] updated residual stream.
        """
        attn_kwargs = attn_kwargs or {}
        mlp_kwargs  = mlp_kwargs  or {}

        def _attn_fn(x: Tensor) -> Tensor:
            return self.drop(self.attention(self.norm_attn(x), **attn_kwargs))

        def _mlp_fn(x: Tensor) -> Tensor:
            return self.drop(self.mlp(self.norm_mlp(x), **mlp_kwargs))

        h = self.mrr_attn(h, _attn_fn)
        if next_inject_term is not None:
            # HC carrier-engine: fold the next layer's inject into the MLP POST write.
            h = self.mrr_mlp(h, _mlp_fn, post_inject=next_inject_term)
        else:
            h = self.mrr_mlp(h, _mlp_fn)
        return h
