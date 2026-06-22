"""Residual-stream building blocks for the MORPH transformer block.

Contents:
  ChannelInject — additive injection of a signal (x0 skip, value embeds, diagonal
                  injection) into a fixed channel slice of the residual stream.
  MORPHBlock    — pre-norm attention + MLP block whose residual is a
                  HyperConnectionResidual (Cayley/JPmHC n-stream mixer; see
                  hyper_connections.py), with an optional parallel GLA retention branch.

The residual stream is conceptually split into channels (DEFAULT_CHANNEL_DIMS) so each
injected signal type has a dedicated slice; ChannelInject targets those slices.

Design notes
------------
- bf16 compatible: dtype casts handled at injection boundaries.
- torch.compile friendly: no Python control flow on tensor values, no in-place ops on
  views, no dynamic shapes.
- MORPHBlock takes pre-built attention and MLP modules, keeping this file decoupled from
  the model internals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Channel layout ────────────────────────────────────────────────────────────

# Channel layout of the residual stream (must sum to d_model=768). ChannelInject targets
# these slices so each injected signal type has a dedicated region:
#   Ch0 — Compute (384): attention + MLP primary output.
#   Ch1 — Context (256): x0 skip, value embeds, loop injection.
#   Ch2 — Slow    (128): low-rate slice.
DEFAULT_CHANNEL_DIMS: tuple[int, ...] = (384, 256, 128)


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
    """Pre-norm attention + MLP transformer block with a HyperConnection residual.

    Each sublayer (attention, MLP) is wrapped in a HyperConnectionResidual (Cayley/JPmHC
    n-stream mixer; see hyper_connections.py) — the carrier is [B, S, n, C]. The attention
    and MLP modules are unchanged (single [B,S,C] in/out); only the residual connection
    mixes streams. An optional GLA retention branch can be attached in parallel to the
    attention sublayer (attach_retention), gated off at init.

    Accepts pre-built attention and MLP modules so this file stays decoupled from the model
    internals. RMSNorm operates on the full stream-averaged input, not per-channel.

    Args:
        norm_attn: normalization module for the attention sublayer.
        attn:      attention module. forward(x) → [B, T, D].
        norm_mlp:  normalization module for the MLP sublayer.
        mlp:       MLP module. forward(x) → [B, T, D].
        dropout:   dropout rate applied after each sublayer output.
        d_model:   per-stream feature width C (required — HyperConnectionResidual needs it).
        hc_kwargs: kwargs forwarded to HyperConnectionResidual (n_streams, tau, cayley_*, …).

    Note: the residual attributes are named ``mrr_attn`` / ``mrr_mlp`` for checkpoint
    compatibility with earlier runs; they hold HyperConnectionResidual modules.
    """

    def __init__(
        self,
        norm_attn: nn.Module,
        attn: nn.Module,
        norm_mlp: nn.Module,
        mlp: nn.Module,
        dropout: float = 0.0,
        d_model: int | None = None,
        hc_kwargs: dict | None = None,
    ):
        super().__init__()
        self.norm_attn = norm_attn
        self.attention = attn
        self.norm_mlp  = norm_mlp
        self.mlp       = mlp
        self.drop      = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Residual = HyperConnectionResidual (Cayley/JPmHC), the sole supported residual:
        # an n-stream [B,S,n,C] carrier with an orthogonal stream mixer (exact dynamical
        # isometry for the weight-tied loop). Built once here (branch-free hot path). The
        # attribute names mrr_attn / mrr_mlp are kept for checkpoint compatibility.
        from .hyper_connections import HyperConnectionResidual
        assert d_model is not None, "HyperConnectionResidual needs d_model"
        hk = dict(hc_kwargs or {})
        self.mrr_attn: nn.Module = HyperConnectionResidual(d_model, **hk)
        self.mrr_mlp:  nn.Module = HyperConnectionResidual(d_model, **hk)

        # Retention branch (#230) — attached post-construction so it does NOT perturb
        # the base init RNG (keeps the rest of the model byte-identical to the baseline, so the
        # ablation isolates the retention branch and nothing else). None unless attach_retention.
        self.retention: nn.Module | None = None
        self.norm_ret: nn.Module | None = None
        self.ret_gate: nn.Parameter | None = None

    def attach_retention(self, gla: nn.Module, norm: nn.Module, gate_init: float) -> None:
        """Add a gated GLA branch in PARALLEL to the attention sublayer.

        sublayer output becomes  attn(x) + sigmoid(ret_gate) · gla(norm_ret(x), state).
        gate_init very negative → sigmoid ≈ 0 → branch ≈ off at init (identity to baseline);
        the gate is learnable so the model can open it if retention helps (the key diagnostic).
        """
        self.retention = gla
        self.norm_ret = norm
        self.ret_gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        h: Tensor,
        attn_kwargs: dict | None = None,
        mlp_kwargs: dict | None = None,
        next_inject_term: Tensor | None = None,
        ret_state: Tensor | None = None,
        ret_capture: dict | None = None,
    ) -> Tensor:
        """Forward pass: attention sublayer then MLP sublayer with HC residuals.

        Args:
            h:           [B, T, D] residual stream.
            attn_kwargs: optional keyword arguments forwarded to attention.
            mlp_kwargs:  optional keyword arguments forwarded to mlp.
            next_inject_term: [B, S, C] | None — carrier-engine (HC only): the NEXT layer's
                         injection term, folded into THIS block's MLP-residual POST write
                         (the block's last carrier write), so the next layer skips a separate
                         _apply_injection. Only set in HC carrier-engine mode.
            ret_state:   [B, H, dk, dv] | None — retention (GLA) initial state for this block's
                         branch (the cross-iteration carry in the core loop). None → zero state.
            ret_capture: dict | None — if given and this block has retention, the new GLA state
                         is written to ret_capture["state"]. The CALLER must RETURN that state
                         from any checkpointed region (side-channel capture is not checkpoint-safe
                         on its own); _core_step does exactly that.

        Returns:
            [B, T, D] updated residual stream.
        """
        attn_kwargs = attn_kwargs or {}
        mlp_kwargs  = mlp_kwargs  or {}

        def _attn_fn(x: Tensor) -> Tensor:
            a = self.attention(self.norm_attn(x), **attn_kwargs)
            if self.retention is not None:
                g_out, s_out = self.retention(self.norm_ret(x), initial_state=ret_state)
                if ret_capture is not None:
                    ret_capture["state"] = s_out
                a = a + torch.sigmoid(self.ret_gate).to(a.dtype) * g_out
            return self.drop(a)

        def _mlp_fn(x: Tensor) -> Tensor:
            return self.drop(self.mlp(self.norm_mlp(x), **mlp_kwargs))

        h = self.mrr_attn(h, _attn_fn)
        if next_inject_term is not None:
            # HC carrier-engine: fold the next layer's inject into the MLP POST write.
            h = self.mrr_mlp(h, _mlp_fn, post_inject=next_inject_term)
        else:
            h = self.mrr_mlp(h, _mlp_fn)
        return h
