"""MORPH attention — CCA+CSA/HCA triple-axis compression.

Three-axis compression:
  1. CCA (channel): E → E/C via down-project, causal conv, QK-mean, value-shift,
     learnable temp, QK-RMSNorm, CoPE clipped-RoPE.
  2. CSA (even layers): two-stream gated pooling m=4, Lightning Indexer top-k,
     -inf causal masking before relu, gather + re-check validity mask.
  3. HCA (odd layers): single-stream pooling m=128, dense compressed attention,
     early-query guard (no_valid rows zeroed before and after softmax).

All layers: causal sliding-window local attention with XSA (self-token excluded).
Gate (sigmoid MLP) blends compressed and window branches.
Residual attention (learned α per head) added on top of gate output.
Up-projection restores to d_model.

Alternation is resolved at __init__ time. No runtime dispatch.

References:
  "Compressed Convolutional Attention" — Figliolia et al., arXiv:2510.04476
  DeepSeek-V4 Technical Report (2026)
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    pass  # no sys.path hack needed
    from morph.kernels.triton.fused_window_attention import fused_window_attention, TRITON_AVAILABLE
    _USE_FUSED_WINDOW = TRITON_AVAILABLE and not os.environ.get("DISABLE_FUSED_KERNELS")
except ImportError:
    _USE_FUSED_WINDOW = False

# Fused CCA prologue (qk-mean → RMSNorm → temp → CoPE-RoPE → GQA expand →
# value-shift) in one Triton kernel. Public fn falls back to a pure-PyTorch
# reference when Triton is unavailable, so a direct import is always safe.
from morph.kernels.triton.fused_cca_prologue import fused_cca_prologue
# Fused causal conv pair (depthwise + head-grouped), replacing the cuDNN convs
# whose grouped wgrad backward is slow on sm_120. Verified fwd/grad-exact.
from morph.kernels.triton.fused_cca_conv import fused_cca_conv
# Fused HCA compressed attention (flash online-softmax over blocks, sink + early-
# query guard folded in). Never materializes [B,H,S,n_blocks] scores → big memory
# win at scale. Verified fwd/grad-exact vs the eager einsum path.
from morph.kernels.triton.fused_hca_attention import fused_hca_attention
# Fused CSA compressed attention: gathers the top-k selected blocks ON THE FLY
# (never materializes C_sel [B,S,tk,D] ≈ 2GB/layer at scale — 11× attn memory),
# folds invalid-mask + sink into a flash online softmax. Verified fwd/grad-exact.
from morph.kernels.triton.fused_csa_attention import fused_csa_attention


# ─── RMSNorm ──────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ─── CoPE (Clipped RoPE) ─────────────────────────────────────────────────────


class CoPEEmbedding(nn.Module):
    """Rotary embeddings with cosine-tapered attenuation for long-wavelength dims.

    Frequencies whose wavelength exceeds context_len are attenuated toward zero
    rotation (cos→1, sin→0 = identity). Provides smooth length extrapolation
    without learned parameters.
    """

    def __init__(self, d_head: int, max_seq_len: int = 32768,
                 base: float = 10000.0, context_len: int = 4096):
        super().__init__()
        assert d_head % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        wavelengths = 2.0 * math.pi / inv_freq
        taper = torch.ones_like(inv_freq)
        long = wavelengths > context_len
        if long.any():
            log_w = torch.log(wavelengths[long])
            log_L = math.log(context_len)
            log_max = torch.log(wavelengths).max()
            ratio = (log_w - log_L) / (log_max - log_L + 1e-8)
            taper[long] = torch.cos(ratio * math.pi / 2).clamp(min=0.0)
        self.register_buffer("inv_freq", inv_freq * taper, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)

    def _rotate_half(self, x: Tensor) -> Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        S = q.shape[2]
        cos = self.cos_cached[:, :, :S].to(q.dtype)
        sin = self.sin_cached[:, :, :S].to(q.dtype)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ─── GatedPoolCompressor ─────────────────────────────────────────────────────


class GatedPoolCompressor(nn.Module):
    """Learned gated pooling: [B, S, d_model] → [B, n_blocks, c].

    two_stream=True (CSA, m=4): overlapping streams A and B with joint softmax
    over 2m elements per feature. Stream B is offset by one block, so block i
    fuses tokens from block i (stream A) and block i-1 (stream B). The joint
    softmax prevents hard boundaries at block edges.

    two_stream=False (HCA, m=128): within-block softmax pooling, per feature dim.
    """

    def __init__(self, d_model: int, c: int, m: int, two_stream: bool):
        super().__init__()
        self.c = c
        self.m = m
        self.two_stream = two_stream

        self.W_aKV = nn.Linear(d_model, c, bias=False)
        self.W_aZ = nn.Linear(d_model, c, bias=False)
        self.B_a = nn.Parameter(torch.zeros(m, c))

        if two_stream:
            self.W_bKV = nn.Linear(d_model, c, bias=False)
            self.W_bZ = nn.Linear(d_model, c, bias=False)
            self.B_b = nn.Parameter(torch.zeros(m, c))

    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        m, c = self.m, self.c
        n_blocks = S // m
        x_u = x[:, :n_blocks * m]

        C_a = self.W_aKV(x_u).reshape(B, n_blocks, m, c)
        Z_a = self.W_aZ(x_u).reshape(B, n_blocks, m, c) + self.B_a

        if not self.two_stream:
            w = torch.softmax(Z_a, dim=2)
            return (w * C_a).sum(dim=2)

        C_b = self.W_bKV(x_u).reshape(B, n_blocks, m, c)
        Z_b = self.W_bZ(x_u).reshape(B, n_blocks, m, c) + self.B_b

        # Shift B right by one block: block i uses B tokens from block i-1.
        # First block's B-stream gets -inf gates so its weight is exactly zero.
        C_b_prev = F.pad(C_b[:, :-1], (0, 0, 0, 0, 1, 0))
        Z_b_prev = F.pad(Z_b[:, :-1], (0, 0, 0, 0, 1, 0), value=float("-inf"))

        Z_joint = torch.cat([Z_a, Z_b_prev], dim=2)        # [B, nb, 2m, c]
        S_joint = torch.softmax(Z_joint.float(), dim=2).to(x.dtype)
        S_a, S_b = S_joint[:, :, :m], S_joint[:, :, m:]

        return (S_a * C_a).sum(dim=2) + (S_b * C_b_prev).sum(dim=2)


# ─── LightningIndexer ────────────────────────────────────────────────────────


class LightningIndexer(nn.Module):
    """Lightweight block scorer for CSA top-k selection.

    Scores compressed blocks via ReLU dot-product. Causal mask is applied as
    -inf BEFORE relu so future blocks produce score 0 and are never spuriously
    preferred by top-k (the causal-leak fix).

    Returns [B, S, n_blocks] non-negative scores; caller does .topk().
    """

    def __init__(self, d_model: int, d_indexer: int, m: int):
        super().__init__()
        self.W_IQ = nn.Linear(d_model, d_indexer, bias=False)
        self.compressor = GatedPoolCompressor(d_model, d_indexer, m, two_stream=False)

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        """causal_mask: [B, S, n_blocks] bool, True = block is causally valid."""
        q_I = self.W_IQ(x)           # [B, S, d_I]
        K_I = self.compressor(x)      # [B, n_blocks, d_I]
        raw = torch.bmm(q_I, K_I.transpose(1, 2)).float()   # [B, S, n_blocks]
        raw = raw.masked_fill(~causal_mask, float("-inf"))   # -inf BEFORE relu
        return F.relu(raw)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _compressed_causal_mask(S: int, n_blocks: int, m: int, device) -> Tensor:
    """[S, n_blocks] bool: block j is causal for query i iff (j+1)*m - 1 < i."""
    block_end = (torch.arange(n_blocks, device=device) + 1) * m - 1   # [nb]
    query_pos = torch.arange(S, device=device)                          # [S]
    return block_end.unsqueeze(0) < query_pos.unsqueeze(1)              # [S, nb]


def _window_fallback(q: Tensor, k: Tensor, v: Tensor,
                     window_size: int, device, scale: float,
                     n_skip_rope: int = 0) -> Tensor:
    """Causal sliding-window attention with XSA (self-token excluded).

    Position j is attended by query i iff:
      - j <= i  (causal)
      - i - j < window_size  (within window)
      - j != i  (XSA: exclude self-token)
    OR j >= S - n_skip_rope (MAC suffix tokens always visible to all queries)
    OR i >= S - n_skip_rope (MAC suffix queries can see all keys).
    """
    S = q.shape[2]
    row = torch.arange(S, device=device).unsqueeze(1)
    col = torch.arange(S, device=device).unsqueeze(0)
    dist = row - col

    mask = (dist >= 0) & (dist < window_size) & (dist != 0)
    if n_skip_rope > 0:
        is_mac_col = col >= S - n_skip_rope
        is_mac_row = row >= S - n_skip_rope
        mask = mask | is_mac_col | is_mac_row

    bias = torch.where(mask, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale)


# ─── CCA Base ─────────────────────────────────────────────────────────────────


class _CCABase(nn.Module):
    """Shared CCA infrastructure — channel compress, causal conv, QK machinery.

    Down-projects Q and K to compressed latent space (d_head = d_model/(C*n_heads)),
    applies two causal Conv1d (depthwise + head-grouped) to each stream, fuses
    pre-conv and post-conv via QK-mean, normalizes with RMSNorm per head_dim,
    scales keys by learnable exp(temp), applies CoPE-RoPE, shifts values.

    The gate, residual-alpha, and W_up live here so _gate_combine_up is shared
    by both CCACSAAttention and CCAHCAAttention without duplication.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 compression: int, max_seq_len: int, context_len: int,
                 window_size: int, init_alpha: float, conv_kernel: int):
        super().__init__()
        assert d_model % (compression * n_heads) == 0, (
            f"d_model={d_model} must be divisible by compression*n_heads="
            f"{compression * n_heads}")
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // (compression * n_heads)
        self.latent_q_dim = n_heads * self.d_head
        self.latent_k_dim = n_kv_heads * self.d_head
        self.window_size = window_size
        self._conv_pad = conv_kernel - 1

        # CCA down-projections
        self.W_down_q = nn.Linear(d_model, self.latent_q_dim, bias=False)
        self.W_down_k = nn.Linear(d_model, self.latent_k_dim, bias=False)

        # Causal convolutions: depthwise then head-grouped on Q and K streams
        self.conv_q_dw = nn.Conv1d(self.latent_q_dim, self.latent_q_dim,
                                    conv_kernel, groups=self.latent_q_dim, bias=False)
        self.conv_q_gp = nn.Conv1d(self.latent_q_dim, self.latent_q_dim,
                                    conv_kernel, groups=n_heads, bias=False)
        self.conv_k_dw = nn.Conv1d(self.latent_k_dim, self.latent_k_dim,
                                    conv_kernel, groups=self.latent_k_dim, bias=False)
        self.conv_k_gp = nn.Conv1d(self.latent_k_dim, self.latent_k_dim,
                                    conv_kernel, groups=n_kv_heads, bias=False)

        # Value-shift: half from current token, half from t-1 lookback
        v_half = n_kv_heads * (self.d_head // 2)
        self.W_v_curr = nn.Linear(d_model, v_half, bias=False)
        self.W_v_prev = nn.Linear(d_model, v_half, bias=False)

        # QK-Norm and learnable temperature (per KV head, applied as exp(temp))
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)
        self.temp = nn.Parameter(torch.zeros(n_kv_heads))

        # CoPE (Clipped RoPE) in compressed latent space
        self.rope = CoPEEmbedding(self.d_head, max_seq_len, context_len=context_len)

        # Residual attention: learned α per query head
        self.alpha = nn.Parameter(torch.full((n_heads, 1, 1), init_alpha))

        # Gate MLP: full d_model → 2 weights per head (compressed + window)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 4, n_heads * 2, bias=False),
        )

        # Attention sinks: learnable logit per head, appended before softmax
        self.sink_logits = nn.Parameter(torch.zeros(n_heads))

        # CCA up-projection from compressed latent back to d_model
        self.W_up = nn.Linear(self.latent_q_dim, d_model, bias=False)

    def _causal_conv(self, x_t: Tensor,
                     conv_dw: nn.Module, conv_gp: nn.Module) -> Tensor:
        # Fused depthwise+grouped causal conv (one Triton kernel, sm_120).
        # x_t is [B, C, S]. Replaces two cuDNN Conv1d whose grouped wgrad is slow.
        # Falls back to a PyTorch reference (matching the old two-pad+conv1d) when
        # Triton is unavailable. Weights cast to x dtype for the autocast bf16 path.
        return fused_cca_conv(
            x_t, conv_dw.weight.to(x_t.dtype), conv_gp.weight.to(x_t.dtype),
            conv_gp.groups, conv_dw.kernel_size[0],
        )

    def _cca_project(self, x: Tensor, n_skip_rope: int = 0):
        """CCA: down-project → conv → QK-mean → norm → temp → RoPE → value-shift.

        Returns q [B, H, S, D], k [B, H, S, D], v [B, H, S, D] all at d_head
        with K and V already GQA-expanded to n_heads.
        """
        B, S, _ = x.shape
        H, Hkv, D = self.n_heads, self.n_kv_heads, self.d_head

        q_lat = self.W_down_q(x)   # [B, S, latent_q_dim]
        k_lat = self.W_down_k(x)   # [B, S, latent_k_dim]

        q_conv = self._causal_conv(
            q_lat.transpose(1, 2), self.conv_q_dw, self.conv_q_gp).transpose(1, 2)
        k_conv = self._causal_conv(
            k_lat.transpose(1, 2), self.conv_k_dw, self.conv_k_gp).transpose(1, 2)

        # Value-shift latents: W_v_curr(x_t) || W_v_prev(x_{t-1})
        v_curr = self.W_v_curr(x)
        v_prev = self.W_v_prev(F.pad(x[:, :-1], (0, 0, 1, 0)))

        # Fused prologue (one Triton kernel, sm_120): QK-mean coupling → RMSNorm(q/k)
        # → learnable temp → CoPE-RoPE → GQA repeat → value-shift assembly. Replaces
        # ~10 eager launches (the launch-bound bottleneck). Returns q,k,v [B,H,S,D]
        # with K/V already GQA-expanded. Falls back to a PyTorch reference w/o Triton.
        cos = self.rope.cos_cached[:, :, :S]
        sin = self.rope.sin_cached[:, :, :S]
        q, k, v = fused_cca_prologue(
            q_lat, k_lat, q_conv, k_conv, v_curr, v_prev,
            self.q_norm.weight, self.k_norm.weight, self.temp,
            cos, sin,
            H, Hkv, D, n_skip_rope=n_skip_rope, eps=self.q_norm.eps,
        )

        return q, k, v

    def _window_attn(self, q: Tensor, k: Tensor, v: Tensor,
                     device, scale: float, n_skip_rope: int = 0) -> Tensor:
        if _USE_FUSED_WINDOW:
            return fused_window_attention(
                q, k, v, self.window_size, n_skip_rope, True, scale=scale)
        return _window_fallback(q, k, v, self.window_size, device, scale, n_skip_rope)

    def _gate_combine_up(self, x: Tensor,
                          out_comp: Tensor, out_win: Tensor) -> Tensor:
        """Sigmoid gate blend + residual-alpha + up-project back to d_model."""
        B, S, _ = x.shape
        H, D = self.n_heads, self.d_head

        g = torch.sigmoid(self.gate(x)).reshape(B, S, H, 2).permute(0, 2, 1, 3)
        combined = g[..., 0:1] * out_comp + g[..., 1:2] * out_win

        x_res = self.W_down_q(x).reshape(B, S, H, D).transpose(1, 2)
        out = combined + self.alpha * x_res

        return self.W_up(out.transpose(1, 2).reshape(B, S, self.latent_q_dim))


# ─── CCA + CSA ────────────────────────────────────────────────────────────────


class _CCACSAAttention(nn.Module):
    """CCA + CSA: channel compression + sparse global selection (even layers).

    Causal-leak fix (double guard):
      1. Lightning Indexer scores future blocks with -inf before relu → score 0.
      2. After gathering, re-derive validity per gathered index and apply -inf
         to attention logits — suppression is absolute, not just score-based.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 compression: int, csa_compress_ratio: int, top_k: int,
                 d_indexer: int, max_seq_len: int, context_len: int,
                 window_size: int, init_alpha: float, conv_kernel: int):
        super().__init__()
        self.top_k = top_k
        self.compress_ratio = csa_compress_ratio

        self.cca = _CCABase(d_model, n_heads, n_kv_heads, compression,
                            max_seq_len, context_len, window_size,
                            init_alpha, conv_kernel)

        self.compressor = GatedPoolCompressor(
            d_model, self.cca.d_head, csa_compress_ratio, two_stream=True)
        self.comp_norm = RMSNorm(self.cca.d_head)
        self.indexer = LightningIndexer(d_model, d_indexer, csa_compress_ratio)

    def forward(self, x: Tensor, n_skip_rope: int = 0) -> Tensor:
        B, S, _ = x.shape
        q, k, v = self.cca._cca_project(x, n_skip_rope)
        H, D = self.cca.n_heads, self.cca.d_head
        m = self.compress_ratio
        n_blocks = S // m
        scale = D ** -0.5

        C_comp = self.comp_norm(self.compressor(x))    # [B, n_blocks, D]
        causal = _compressed_causal_mask(S, n_blocks, m, x.device)
        causal_3d = causal.unsqueeze(0).expand(B, -1, -1)  # [B, S, n_blocks]

        scores = self.indexer(x, causal_3d)            # [B, S, n_blocks]
        tk = min(self.top_k, n_blocks)
        _, top_idx = scores.topk(tk, dim=-1)           # [B, S, tk]

        # Per-gathered-entry causal validity (future blocks → masked in the kernel)
        invalid_mask = ~causal_3d.gather(-1, top_idx)  # [B, S, tk]

        # Fused CSA gather-attention: gathers the top-k blocks ON THE FLY (never
        # materializes C_sel [B,S,tk,D] ≈ 2GB/layer at scale), folding the invalid
        # mask + per-head sink into a flash online softmax.
        out_comp = fused_csa_attention(
            q, C_comp, top_idx, invalid_mask, self.cca.sink_logits, scale)

        out_win = self.cca._window_attn(q, k, v, x.device, scale, n_skip_rope)
        return self.cca._gate_combine_up(x, out_comp, out_win)


# ─── CCA + HCA ────────────────────────────────────────────────────────────────


class _CCAHCAAttention(nn.Module):
    """CCA + HCA: channel compression + dense compressed attention (odd layers).

    Early-query guard: queries whose all compressed-block scores are -inf (i.e.,
    no valid causal block exists yet) have their softmax input zeroed so uniform
    weights don't leak future information, and their output is then zeroed again.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 compression: int, hca_compress_ratio: int,
                 max_seq_len: int, context_len: int,
                 window_size: int, init_alpha: float, conv_kernel: int):
        super().__init__()
        self.compress_ratio = hca_compress_ratio

        self.cca = _CCABase(d_model, n_heads, n_kv_heads, compression,
                            max_seq_len, context_len, window_size,
                            init_alpha, conv_kernel)

        self.compressor = GatedPoolCompressor(
            d_model, self.cca.d_head, hca_compress_ratio, two_stream=False)
        self.comp_norm = RMSNorm(self.cca.d_head)

    def forward(self, x: Tensor, n_skip_rope: int = 0) -> Tensor:
        B, S, _ = x.shape
        q, k, v = self.cca._cca_project(x, n_skip_rope)
        H, D = self.cca.n_heads, self.cca.d_head
        m = self.compress_ratio
        n_blocks = S // m
        scale = D ** -0.5

        C_comp = self.comp_norm(self.compressor(x))    # [B, n_blocks, D]

        # Fused HCA compressed attention: flash online-softmax over blocks with the
        # causal-block mask, per-head sink logit, and early-query guard folded in.
        # Never materializes the [B,H,S,n_blocks] scores tensor (memory win at scale).
        out_comp = fused_hca_attention(q, C_comp, self.cca.sink_logits, m, scale)

        out_win = self.cca._window_attn(q, k, v, x.device, scale, n_skip_rope)
        return self.cca._gate_combine_up(x, out_comp, out_win)


# ─── MORPHAttention ───────────────────────────────────────────────────────────


class MORPHAttention(nn.Module):
    """MORPH production attention module: CCA+CSA (even) / CCA+HCA (odd).

    Alternation is resolved at __init__ by instantiating exactly one of
    _CCACSAAttention or _CCAHCAAttention. The forward method calls through
    without any runtime dispatch.

    Args:
        d_model:            Model hidden dimension. Must be divisible by
                            compression * n_heads.
        n_heads:            Number of query heads.
        layer_idx:          Layer position. Even → CSA, odd → HCA.
        max_seq_len:        Maximum sequence length (for CoPE cache).
        n_kv_heads:         Number of KV heads (GQA). Must divide n_heads.
        compression:        Channel compression factor C. d_head = d_model/(C*n_heads).
        csa_compress_ratio: Tokens per CSA block (two-stream pooling ratio).
        hca_compress_ratio: Tokens per HCA block (single-stream pooling ratio).
        top_k:              Max compressed blocks selected per query (CSA layers).
        d_indexer:          Indexer projection dim for LightningIndexer (CSA).
        window_size:        Local sliding-window size.
        context_len:        CoPE taper threshold (usually = training seq_len).
        init_alpha:         Initial value for residual-attention α.
        conv_kernel:        Causal conv kernel width.

    Forward:
        x: [B, S, d_model]
        n_skip_rope: leading token count that skips CoPE-RoPE (persistent/sink tokens).
        → [B, S, d_model]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        layer_idx: int,
        max_seq_len: int = 32768,
        n_kv_heads: int = 4,
        compression: int = 2,
        csa_compress_ratio: int = 4,
        hca_compress_ratio: int = 128,
        top_k: int = 128,
        d_indexer: int = 32,
        window_size: int = 128,
        context_len: int = 4096,
        init_alpha: float = 0.1,
        conv_kernel: int = 4,
    ):
        super().__init__()

        shared = dict(
            d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
            compression=compression, max_seq_len=max_seq_len,
            context_len=context_len, window_size=window_size,
            init_alpha=init_alpha, conv_kernel=conv_kernel,
        )

        if layer_idx % 2 == 0:
            self._impl: nn.Module = _CCACSAAttention(
                csa_compress_ratio=csa_compress_ratio,
                top_k=top_k, d_indexer=d_indexer, **shared)
        else:
            self._impl = _CCAHCAAttention(
                hca_compress_ratio=hca_compress_ratio, **shared)

    def forward(self, x: Tensor, n_skip_rope: int = 0) -> Tensor:
        return self._impl(x, n_skip_rope)
