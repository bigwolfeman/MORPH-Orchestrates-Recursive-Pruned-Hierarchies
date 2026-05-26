"""MORPH neural memory integration layer (PyTorch).

Clean wrapper around titans_core.memory.NeuralMemory (paper-faithful v3).
Handles all MORPH-specific integration patterns:

  SSM top-inject   — memory hidden state conditions the entire input at t=0,
                     via a single learned query expanded to batch.
  MAG inject       — gated residual into memory channel slice of hidden states.
  Forward update   — gradient-based surprise update on the forward pass.
  Split retrieve   — stable path + gated mutable delta.
  MAC tokens       — dynamic KV-prepend tokens for attention context.

Design principles:
  - No runtime feature flags in forward methods.
  - All methods always active (caller decides which to call and when).
  - bf16 compatible: parameter dtype coercion handled internally.
  - torch.compile friendly: @torch._dynamo.disable on NeuralMemory.update()
    is handled inside titans_core — MemorySystem itself compiles cleanly.

Source lineage:
  titans_core/memory/neural_memory.py     (NeuralMemory v3 implementation)
  TPU/subq-attention/looped_model_mhc.py  (integration patterns)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .titans_core.neural_memory import NeuralMemory


class MemorySystem(nn.Module):
    """MORPH wrapper around NeuralMemory v3 with all integration methods.

    Bundles the NeuralMemory module with the projection layers needed for
    SSM-inject, MAG-inject, and MAC-token retrieval patterns.

    Args:
        d_model: Hidden dimension of the main model.
        n_layers: Depth of the memory MLP. Must be >= 2. Default 2.
        n_memory_tokens: Number of MAC tokens to prepend for KV attention. Default 32.
        d_memory_channel: Width of the MAG memory channel slice (a sub-range
                          of d_model dims that the MAG gate writes into). The
                          caller is responsible for selecting the right slice
                          indices via channel_start / channel_end in mag_inject().
                          Default = d_model // 6 (≈128 for d_model=768).
        chunk_size: Tokens per chunk for gate computation. Default 128.
        max_lr: Upper bound for learned theta gate. Default 1.0.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_memory_tokens: int = 32,
        d_memory_channel: Optional[int] = None,
        chunk_size: int = 128,
        max_lr: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_memory_tokens = n_memory_tokens
        self.d_memory_channel = d_memory_channel or max(64, d_model // 6)

        # ── Core memory module (paper-faithful v3) ─────────────────────────
        self.memory = NeuralMemory(
            d_model=d_model,
            n_memory_layers=n_layers,
            chunk_size=chunk_size,
            max_lr=max_lr,
        )

        # ── MAG inject ────────────────────────────────────────────────────
        # Gated residual write into the memory channel.
        # Gate: sigmoid(Linear(d_model → d_memory_channel)), init near 0.
        # Proj: d_model → d_memory_channel (what gets written).
        self.mag_gate = nn.Sequential(
            nn.Linear(d_model, self.d_memory_channel),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.mag_gate[0].weight, std=0.02)
        nn.init.zeros_(self.mag_gate[0].bias)

        self.mag_proj = nn.Linear(d_model, self.d_memory_channel, bias=False)
        nn.init.normal_(self.mag_proj.weight, std=0.02)

        # ── MAC tokens ────────────────────────────────────────────────────
        # Dynamic MAC: pool the prelude output, project to n_memory_tokens
        # query vectors, retrieve from memory, project back to d_model.
        self.mac_gate_raw = nn.Parameter(torch.tensor(-5.0))  # softplus → ≈0.007 initially
        self.mac_query_proj = nn.Linear(d_model, n_memory_tokens * d_model, bias=False)
        nn.init.normal_(self.mac_query_proj.weight, std=0.01)

        self.mac_out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.mac_out_proj.weight, std=0.02)

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"MemorySystem: d_model={d_model}, n_layers={n_layers}, "
            f"n_mac_tokens={n_memory_tokens}, d_mem_ch={self.d_memory_channel}, "
            f"total_params={n_params / 1e3:.1f}K"
        )

    # ── MAG inject ─────────────────────────────────────────────────────────────

    def mag_inject(
        self,
        x: Tensor,
        channel_start: int,
        channel_end: int,
        scale: float = 1.0,
    ) -> Tensor:
        """MAG inject: gated residual update into memory channel slice.

        Retrieves from memory, projects into the memory channel width, gates it,
        and adds into x[..., channel_start:channel_end]. The rest of x is
        unchanged. This is a pure residual — no in-place ops.

        Args:
            x: [B, T, d_model] hidden states.
            channel_start: Start index of the memory channel in d_model.
            channel_end: End index (exclusive). Must equal channel_start + d_memory_channel.
            scale: Multiplicative scale on the gated signal (used for warmup ramp).

        Returns:
            [B, T, d_model] with memory channel updated.
        """
        param_dtype = self.mag_proj.weight.dtype
        x_cast = x.to(param_dtype) if x.dtype != param_dtype else x

        retrieval = self.memory.retrieve(x_cast)         # [B, T, d_model]
        mem_signal = self.mag_proj(retrieval)             # [B, T, d_memory_channel]
        gate = self.mag_gate(x_cast)                     # [B, T, d_memory_channel]
        gated = gate * mem_signal * scale

        prefix = x_cast[..., :channel_start]
        mem_ch = x_cast[..., channel_start:channel_end] + gated
        suffix = x_cast[..., channel_end:]
        return torch.cat([prefix, mem_ch, suffix], dim=-1)

    # ── Forward update ─────────────────────────────────────────────────────────

    def update(self, x: Tensor, suppress_decay: bool = False) -> Tensor:
        """Run the forward-pass gradient update on memory MLP.

        Processes x in chunks, running inner gradient descent on the memory MLP
        to minimize the associative loss. The probe loss for the last chunk is
        differentiable w.r.t. gate networks (alpha, eta, theta), allowing the
        outer optimizer to tune how memory learns.

        Must be called only during training (caller's responsibility):
            if self.training:
                mem_loss = memory_system.update(x)

        @torch._dynamo.disable is applied inside NeuralMemory.update — this
        method itself is compile-clean.

        Args:
            x: [B, T, d_model] hidden states at the update point.
            suppress_decay: Force alpha=0 (sets forget gate to zero).
                            Use during warmup to prevent premature decay.

        Returns:
            Differentiable probe loss scalar for the last chunk.
        """
        param_dtype = next(self.memory.parameters()).dtype
        x_cast = x.to(param_dtype) if x.dtype != param_dtype else x
        return self.memory.update(x_cast, suppress_decay=suppress_decay)

    # ── Split-path retrieve ─────────────────────────────────────────────────────

    def retrieve(self, x: Tensor) -> Tensor:
        """Split-path retrieve from memory.

        out = stable_norm(q) + sigmoid(delta_gate) * memory_mlp(q)

        Stable path: always active, trained only by outer optimizer.
        Delta path: MLP updated by inner GD, zero-init gate, grows with training.

        Args:
            x: [B, T, d_model] query hidden states.

        Returns:
            [B, T, d_model] retrieved memory representation.
        """
        param_dtype = next(self.memory.parameters()).dtype
        x_cast = x.to(param_dtype) if x.dtype != param_dtype else x
        return self.memory.retrieve(x_cast)

    # ── MAC tokens ─────────────────────────────────────────────────────────────

    def get_mac_tokens(self, x: Tensor, n_tokens: Optional[int] = None) -> Tensor:
        """Generate dynamic MAC tokens for KV-prepend to attention.

        Pool x → project to n_tokens query vectors → retrieve from memory →
        project to d_model → gate by softplus (starts near zero).

        The softplus gate (mac_gate_raw initialized to -5) ensures MAC tokens
        contribute nothing at warmup. The outer optimizer gradually opens the
        gate as memory representations become useful.

        Args:
            x: [B, T, d_model] prelude output (conditioned on current segment).
            n_tokens: Override number of tokens. Default: self.n_memory_tokens.

        Returns:
            [B, N, d_model] memory tokens for KV prepend, N = n_tokens.
        """
        n = n_tokens or self.n_memory_tokens
        param_dtype = self.mac_query_proj.weight.dtype
        x_cast = x.to(param_dtype) if x.dtype != param_dtype else x

        B, T, D = x_cast.shape
        # Pool over T to get a single representation per sequence
        x_pool = x_cast.mean(dim=1)                          # [B, D]
        # Project to n_tokens * D then reshape to n_tokens query vectors
        q_flat = self.mac_query_proj(x_pool)                 # [B, n * D]
        queries = q_flat.view(B, n, D)                       # [B, n, D]

        raw = self.memory.retrieve(queries)                  # [B, n, d_model]
        tokens = self.mac_out_proj(raw)                      # [B, n, d_model]

        gate = F.softplus(self.mac_gate_raw)                 # scalar ≈0.007 initially
        return tokens * gate

    # ── Utilities ──────────────────────────────────────────────────────────────

    def reset_momentum(self):
        """Reset inner-loop momentum buffer between sequences."""
        self.memory.reset_momentum()

    def reset_memory(self):
        """Full reset: re-initialize MLP weights and momentum buffer."""
        self.memory.reset_memory()

    def get_memory_stats(self) -> dict[str, float]:
        """Statistics dict for wandb logging.

        Returns:
            Dict with keys: memory_param_norm, momentum_norm, delta_gate,
            alpha_t, eta_t, theta_t, grad_norm (last 4 only after first update).
        """
        return self.memory.get_memory_stats()

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_mac_tokens={self.n_memory_tokens}, "
            f"d_mem_channel={self.d_memory_channel}"
        )
