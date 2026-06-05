"""CCA + CSA/HCA triple-axis attention for MORPH — JAX/Flax port.

Three-axis compression:
  1. CCA (Compressed Convolutional Attention):
       channel dim D -> D/C via learned down-projection, causal depthwise+group
       convolutions, QK-mean sharpening, value-shift (t-1 lookback), learnable
       temperature on keys, QK-RMSNorm, CoPE (clipped RoPE), residual-alpha.
  2. CSA (even layers, Chunked Sparse Attention):
       Two-stream gated pooling at ratio m=4, Lightning Indexer (ReLU dot-product
       scorer) for top-k compressed block selection, attention with -inf causal
       masking before softmax (causal-leak free).
  3. HCA (odd layers, Heavily Compressed Attention):
       Single-stream pooling at ratio m=128, dense attention over all ~16
       compressed entries, attention sinks, early-query no-valid guard.

All layers: sliding-window local attention (pure JAX fallback, no Triton).
Gate (sigmoid MLP) blends compressed-global and window branches.
XSA (self-exclusion, diagonal mask) applied to window output.
Residual attention (learned α per head) added on top.
CoPE clipped-RoPE applied in compressed latent space.
QK-Norm (RMSNorm per head_dim) applied before CoPE.

Default wiring (CCACSAHCAAttention):
  layer_idx % 2 == 0  →  CCA + CSA
  layer_idx % 2 == 1  →  CCA + HCA

Reference:
  "Compressed Convolutional Attention" — Figliolia et al., arXiv:2510.04476
  DeepSeek-V4 Technical Report (2026)
"""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn


# ─── Utilities ────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """Per-feature RMSNorm. Applied per head_dim in QK-Norm usage."""
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        # Compute in float32, return in input dtype
        x_f32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.eps)
        return (x_f32 / rms * scale).astype(x.dtype)


def _cope_freqs(head_dim: int, max_seq_len: int, context_len: int,
                base: float = 10000.0) -> jnp.ndarray:
    """Precompute CoPE (Clipped RoPE) frequency table [max_seq_len, head_dim//2, 2].

    Frequencies whose wavelength exceeds context_len are cosine-tapered to zero,
    reducing their rotation angle toward 0 (cos→1, sin→0, identity transform).
    This prevents aliasing when extrapolating beyond training length.

    Returns real pairs [cos, sin] stacked on last axis.
    """
    assert head_dim % 2 == 0
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (jnp.arange(half, dtype=jnp.float32) / half))

    # Taper: wavelength = 2π / inv_freq. Frequencies with wavelength > context_len get tapered.
    wavelengths = 2.0 * math.pi / inv_freq  # [half]
    log_w = jnp.log(jnp.maximum(wavelengths, 1e-8))
    log_L = math.log(context_len)
    log_max = jnp.max(log_w)
    ratio = (log_w - log_L) / (log_max - log_L + 1e-8)
    # taper = cos(ratio * π/2) clamped ∈ [0,1]; frequencies within context_len keep taper=1
    taper = jnp.where(wavelengths > context_len,
                      jnp.clip(jnp.cos(ratio * math.pi / 2.0), 0.0, 1.0),
                      jnp.ones_like(inv_freq))
    inv_freq_tapered = inv_freq * taper

    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq_tapered)  # [T, half]
    return jnp.stack([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)  # [T, half, 2]


def _apply_rope(x: jnp.ndarray, freqs: jnp.ndarray, offset: int = 0) -> jnp.ndarray:
    """Apply rotary embeddings to x [B, H, S, D] using freqs [T, D//2, 2].

    offset: skip this many leading positions (for persistent/sink tokens).
    """
    B, H, S, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = freqs[offset:offset + S, :, 0][None, None]  # [1, 1, S, half]
    sin = freqs[offset:offset + S, :, 1][None, None]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def _causal_conv1d(x: jnp.ndarray, kernel: jnp.ndarray,
                   groups: int, pad: int) -> jnp.ndarray:
    """Causal depthwise/group Conv1d via jax.lax.conv_general_dilated.

    x:      [B, S, C]    (channels-last, matching JAX convention)
    kernel: [C_out, C_in/groups, k]  (kernel in channel-first for lax OIW)
    Returns [B, S, C] with causal left-padding of `pad` zeros.

    JAX lax dimension_numbers format: ('NWC', 'OIW', 'NWC')
      N = batch, W = spatial (sequence), C = channels
      O = output features, I = input features/groups, W = kernel width
    """
    # x is [B, S, C]; pad along sequence dim (axis=1) on the left
    x_padded = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)))  # [B, S+pad, C]
    out = jax.lax.conv_general_dilated(
        x_padded,
        kernel,
        window_strides=(1,),
        padding='VALID',
        feature_group_count=groups,
        dimension_numbers=('NWC', 'OIW', 'NWC'),
    )
    return out  # [B, S, C]


def _compressed_causal_mask(S: int, n_blocks: int, m: int) -> jnp.ndarray:
    """Boolean mask [S, n_blocks]: block j is causal for query i iff (j+1)*m-1 < i.

    Equivalent to: the entire block j lies strictly before query position i.
    block_end[j] = (j+1)*m - 1 → visible iff block_end < query_pos.
    """
    block_end = (jnp.arange(n_blocks) + 1) * m - 1   # [n_blocks]
    query_pos = jnp.arange(S)                          # [S]
    return block_end[None, :] < query_pos[:, None]     # [S, n_blocks]


def _window_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
                      window_size: int, n_skip: int = 0,
                      dtype: jnp.dtype = jnp.bfloat16) -> jnp.ndarray:
    """Causal sliding-window attention with XSA (exclude self-token diagonal).

    q, k, v: [B, H, S, d_head]
    n_skip:  leading positions that attend to everything (persistent/sink tokens).

    Mask logic: position j is attended by query i iff:
      - (i >= j)          — causal
      - (i - j < window)  — within window
      - (i != j)          — XSA self-exclusion
      OR j < n_skip        — sink tokens always visible

    Early-query guard: position 0 has no valid keys (XSA + causal with window > 0).
    Without the guard, JAX softmax over all-NEG_INF produces uniform weights
    (1/S) rather than zero, causing a causal leak. We detect no-valid rows and
    zero-fill the output — matching the documented XSA behavior: "first token
    gets zero attention output".
    """
    B, H, S, D = q.shape
    NEG_INF = jnp.finfo(jnp.float32).min / 2

    row = jnp.arange(S)
    col = jnp.arange(S)
    dist = row[:, None] - col[None, :]   # [S, S], dist[i,j] = i - j

    is_causal = dist >= 0
    is_window = dist < window_size
    is_self   = dist == 0                # XSA: exclude self
    is_skip   = col[None, :] < n_skip   # always allow sink/persistent tokens

    mask = (is_causal & is_window & ~is_self) | is_skip  # [S, S]
    # no_valid[i] = True iff query i has zero valid keys (e.g. position 0 with XSA)
    no_valid = ~mask.any(axis=-1)  # [S]

    additive = jnp.where(mask, 0.0, NEG_INF).astype(jnp.float32)  # [S, S]
    # For no-valid rows, fill with 0 so softmax gives uniform weights (we'll zero-mask output)
    additive = jnp.where(no_valid[:, None], 0.0, additive)

    scale = D ** -0.5
    scores = jnp.einsum('bhsd,bhkd->bhsk', q.astype(jnp.float32),
                         k.astype(jnp.float32)) * scale  # [B, H, S, S]
    scores = scores + additive[None, None]
    weights = jax.nn.softmax(scores, axis=-1).astype(dtype)
    out = jnp.einsum('bhsk,bhkd->bhsd', weights, v.astype(dtype))
    # Zero out positions with no valid keys to prevent spurious leakage
    return jnp.where(no_valid[None, None, :, None], 0.0, out)


# ─── Gated Pooling Compressor ─────────────────────────────────────────────────


class GatedPoolCompressor(nn.Module):
    """Learned gated pooling that compresses [B, S, d] → [B, n_blocks, c].

    Two-stream mode (CSA, m=4): streams A and B with overlapping windows.
    Stream B is shifted by one block (B's block i uses tokens from block i-1).
    Joint softmax over 2m features per channel dim ensures smooth boundaries.

    Single-stream mode (HCA, m=128): within-block softmax pooling.

    c is the compressed channel dim (usually d_head).
    m is the compression ratio (tokens-per-block).
    """
    c: int             # compressed channel dim
    m: int             # tokens per block
    two_stream: bool   # True for CSA, False for HCA
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: [B, S, d_model] → [B, n_blocks, c]"""
        B, S, _ = x.shape
        m, c = self.m, self.c
        n_blocks = S // m
        x_u = x[:, :n_blocks * m]  # trim to exact multiple

        # Stream A
        C_a = nn.Dense(c, use_bias=False, dtype=self.dtype, name="W_aKV")(x_u)
        Z_a = nn.Dense(c, use_bias=False, dtype=self.dtype, name="W_aZ")(x_u)
        B_a = self.param("B_a", nn.initializers.zeros, (m, c))

        C_a = C_a.reshape(B, n_blocks, m, c)
        Z_a = Z_a.reshape(B, n_blocks, m, c) + B_a  # [B, nb, m, c]

        if not self.two_stream:
            w = jax.nn.softmax(Z_a.astype(jnp.float32), axis=2).astype(self.dtype)
            return (w * C_a).sum(axis=2)  # [B, nb, c]

        # Stream B
        C_b = nn.Dense(c, use_bias=False, dtype=self.dtype, name="W_bKV")(x_u)
        Z_b = nn.Dense(c, use_bias=False, dtype=self.dtype, name="W_bZ")(x_u)
        B_b = self.param("B_b", nn.initializers.zeros, (m, c))

        C_b = C_b.reshape(B, n_blocks, m, c)
        Z_b = Z_b.reshape(B, n_blocks, m, c) + B_b

        # Shift B by one block: block i uses block i-1's tokens.
        # Pad first block with -inf gates so it contributes zero weight.
        NEG_INF_f32 = jnp.finfo(jnp.float32).min / 2
        C_b_prev = jnp.concatenate([jnp.zeros_like(C_b[:, :1]), C_b[:, :-1]], axis=1)
        Z_b_prev = jnp.concatenate(
            [jnp.full_like(Z_b[:, :1], NEG_INF_f32), Z_b[:, :-1]], axis=1)

        # Joint softmax over 2m positions (A + prev-B), per feature
        Z_joint = jnp.concatenate([Z_a, Z_b_prev], axis=2)  # [B, nb, 2m, c]
        S_joint = jax.nn.softmax(Z_joint.astype(jnp.float32), axis=2).astype(self.dtype)
        S_a, S_b = S_joint[:, :, :m], S_joint[:, :, m:]

        return (S_a * C_a).sum(axis=2) + (S_b * C_b_prev).sum(axis=2)


# ─── Lightning Indexer ────────────────────────────────────────────────────────


class LightningIndexer(nn.Module):
    """Lightweight scorer that produces causal block selection scores.

    Projects queries and compressed keys to a small d_indexer space.
    ReLU discards negative correlations (only positive similarity contributes —
    DeepSeek V4 design). Causal mask applied before ReLU via -inf fill so
    future blocks produce exactly 0 score (selected only if forced by top-k).

    Returns raw scores [B, S, n_blocks] — caller applies top-k.
    """
    d_model: int
    d_indexer: int
    m: int            # compress ratio (tokens per block) — used by internal compressor
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, causal_mask: jnp.ndarray) -> jnp.ndarray:
        """x: [B, S, d_model], causal_mask: [B, S, n_blocks] bool → [B, S, n_blocks]"""
        q_I = nn.Dense(self.d_indexer, use_bias=False, dtype=self.dtype, name="W_IQ")(x)

        # Internal single-stream compressor at d_indexer dim
        K_I = GatedPoolCompressor(
            c=self.d_indexer, m=self.m, two_stream=False, dtype=self.dtype,
            name="indexer_compressor",
        )(x)  # [B, n_blocks, d_indexer]

        raw = jnp.einsum('bsd,bnd->bsn', q_I.astype(jnp.float32),
                         K_I.astype(jnp.float32))  # [B, S, n_blocks]

        # Mask future blocks with -inf BEFORE relu — this is the causal-leak fix.
        # Without this, future blocks get relu(negative) = 0 which looks valid;
        # with -inf fill those scores become -inf → relu(-inf) = 0, and more
        # importantly top-k won't prefer them over valid blocks.
        NEG_INF = jnp.finfo(jnp.float32).min / 2
        raw = jnp.where(causal_mask, raw, NEG_INF)
        return jax.nn.relu(raw).astype(self.dtype)


# ─── CCA Base ─────────────────────────────────────────────────────────────────


class _CCABase(nn.Module):
    """Shared CCA infrastructure: down-project, causal conv, QK-mean, value-shift,
    learnable temperature, QK-RMSNorm, CoPE, gate, residual-alpha, up-project.

    Subclasses implement the compressed global attention branch (CSA or HCA)
    in _forward_compressed() and call _cca_project() + _gate_combine_up().

    CCA compresses Q/K into latent_q_dim = n_heads * d_head and latent_k_dim =
    n_kv_heads * d_head where d_head = d_model / (compression * n_heads).
    The window branch and gate operate fully in this compressed space.
    Up-projection restores to d_model.
    """
    d_model: int
    n_heads: int
    max_seq_len: int
    n_kv_heads: int
    compression: int        # channel compression factor C (d_head = d_model/(C*n_heads))
    window_size: int
    context_len: int        # for CoPE taper threshold
    init_alpha: float
    conv_kernel: int
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        assert self.d_model % (self.compression * self.n_heads) == 0, (
            f"d_model={self.d_model} must be divisible by "
            f"compression*n_heads={self.compression * self.n_heads}")
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads={self.n_heads} must be divisible by n_kv_heads={self.n_kv_heads}")

        self._d_head = self.d_model // (self.compression * self.n_heads)
        self._latent_q_dim = self.n_heads * self._d_head
        self._latent_k_dim = self.n_kv_heads * self._d_head
        self._n_rep = self.n_heads // self.n_kv_heads
        self._conv_pad = self.conv_kernel - 1

        # ── CCA down-projections ─────────────────────────────────────────────
        self.W_down_q = nn.Dense(
            self._latent_q_dim, use_bias=False, dtype=self.dtype, name="W_down_q")
        self.W_down_k = nn.Dense(
            self._latent_k_dim, use_bias=False, dtype=self.dtype, name="W_down_k")

        # ── Causal Conv1d params (depthwise + head-grouped) ──────────────────
        # We store kernels as parameters and apply via _causal_conv1d().
        # Depthwise: groups = latent_dim (each channel its own filter).
        # Head-grouped: groups = n_heads (each head filters n_heads//n_heads channels).
        k = self.conv_kernel
        self.conv_q_dw_w = self.param(
            "conv_q_dw_w", nn.initializers.glorot_uniform(),
            (self._latent_q_dim, 1, k))  # depthwise: [C, 1, k]
        self.conv_q_gp_w = self.param(
            "conv_q_gp_w", nn.initializers.glorot_uniform(),
            (self._latent_q_dim, self._d_head, k))  # group: [C, C/groups, k]
        self.conv_k_dw_w = self.param(
            "conv_k_dw_w", nn.initializers.glorot_uniform(),
            (self._latent_k_dim, 1, k))
        self.conv_k_gp_w = self.param(
            "conv_k_gp_w", nn.initializers.glorot_uniform(),
            (self._latent_k_dim, self._d_head, k))

        # ── Value projections: value-shift (half current, half t-1) ─────────
        v_half = self.n_kv_heads * (self._d_head // 2)
        self.W_v_curr = nn.Dense(v_half, use_bias=False, dtype=self.dtype, name="W_v_curr")
        self.W_v_prev = nn.Dense(v_half, use_bias=False, dtype=self.dtype, name="W_v_prev")

        # ── QK-Norm ──────────────────────────────────────────────────────────
        self.q_norm = RMSNorm(name="q_norm")
        self.k_norm = RMSNorm(name="k_norm")

        # ── Learnable temperature (per KV head, multiplied onto keys) ────────
        # temp param initialized to 0; applied as exp(temp) so starts at 1.
        # Shape [n_kv_heads] → broadcast over [B, n_kv_heads, S, d_head].
        # Stored as nn.Module param via param() at call time.

        # ── CoPE (Clipped RoPE) in compressed latent space ───────────────────
        # Precomputed at setup (static graph — not a parameter).
        self._rope_freqs = _cope_freqs(
            self._d_head, self.max_seq_len, self.context_len)

        # ── Residual attention: learned α per query head ──────────────────────
        self._alpha_init = self.init_alpha

        # ── Gate: sigmoid MLP from full x → 2 weights per head ───────────────
        # d_model -> d_model//4 -> n_heads*2
        self.gate_w1 = nn.Dense(
            self.d_model // 4, use_bias=False, dtype=self.dtype, name="gate_w1")
        self.gate_w2 = nn.Dense(
            self.n_heads * 2, use_bias=False, dtype=self.dtype, name="gate_w2")

        # ── Attention sinks: learnable logit per head ─────────────────────────
        # Shape [n_heads], broadcast to [B, H, S, 1].

        # ── CCA up-projection ─────────────────────────────────────────────────
        self.W_up = nn.Dense(
            self.d_model, use_bias=False, dtype=self.dtype, name="W_up")

    def _do_causal_conv(self, x_lat: jnp.ndarray, dw_w: jnp.ndarray,
                         gp_w: jnp.ndarray, n_groups_gp: int) -> jnp.ndarray:
        """Apply two causal Conv1d layers (depthwise then head-grouped).

        x_lat: [B, S, C] where C is latent dim (channels-last).
        dw_w:  [C, 1, k]   depthwise kernel
        gp_w:  [C, C//groups, k]  head-grouped kernel
        Returns [B, S, C].
        """
        B, S, C = x_lat.shape
        # _causal_conv1d operates channels-last: [B, S, C] → [B, S, C]
        x_lat = _causal_conv1d(x_lat, dw_w.astype(self.dtype), groups=C, pad=self._conv_pad)
        x_lat = _causal_conv1d(x_lat, gp_w.astype(self.dtype), groups=n_groups_gp, pad=self._conv_pad)
        return x_lat  # [B, S, C]

    @nn.compact
    def _cca_project(self, x: jnp.ndarray, n_skip: int = 0):
        """CCA projection: down-project → conv → QK-mean → norm → temp → CoPE → value-shift.

        n_skip: number of leading tokens (persistent/sink) that skip RoPE.

        Returns q, k, v each in head layout [B, n_heads, S, d_head].
        q and k have been expanded so all shapes match for n_heads.
        v is already expanded (n_kv_heads → n_heads via repeat).
        """
        B, S, _ = x.shape
        H = self.n_heads
        Hkv = self.n_kv_heads
        D = self._d_head

        q_lat = self.W_down_q(x)   # [B, S, latent_q_dim]
        k_lat = self.W_down_k(x)   # [B, S, latent_k_dim]

        q_conv = self._do_causal_conv(q_lat, self.conv_q_dw_w, self.conv_q_gp_w, H)
        k_conv = self._do_causal_conv(k_lat, self.conv_k_dw_w, self.conv_k_gp_w, Hkv)

        # QK-mean: average pre-conv Q and K expanded to Q-head count, add to post-conv
        q_pre = q_lat.reshape(B, S, H, D)                    # [B, S, H, D]
        k_pre = k_lat.reshape(B, S, Hkv, D)                  # [B, S, Hkv, D]
        k_pre_exp = jnp.repeat(k_pre, self._n_rep, axis=2)   # [B, S, H, D]
        qk_mean_q = (q_pre + k_pre_exp) * 0.5                # [B, S, H, D]
        # Collapse back to Hkv for k's mean component
        qk_mean_k = qk_mean_q.reshape(B, S, Hkv, self._n_rep, D).mean(axis=3)  # [B, S, Hkv, D]

        q = q_conv.reshape(B, S, H, D) + qk_mean_q           # [B, S, H, D]
        k = k_conv.reshape(B, S, Hkv, D) + qk_mean_k         # [B, S, Hkv, D]

        # Transpose to head-first for norm and RoPE: [B, H, S, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)

        # QK-Norm (RMSNorm on last dim = d_head)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Learnable temperature: exp(temp) multiplied onto keys per KV-head
        temp = self.param("temp", nn.initializers.zeros, (Hkv,))
        k = k * jnp.exp(temp).astype(self.dtype)[None, :, None, None]  # [B, Hkv, S, D]

        # CoPE (Clipped RoPE) — apply to non-skip positions
        freqs = self._rope_freqs.astype(jnp.float32)
        if n_skip > 0:
            q_pos = _apply_rope(q[:, :, n_skip:].astype(jnp.float32), freqs, offset=0)
            k_pos = _apply_rope(k[:, :, n_skip:].astype(jnp.float32), freqs, offset=0)
            q = jnp.concatenate([q[:, :, :n_skip], q_pos.astype(self.dtype)], axis=2)
            k = jnp.concatenate([k[:, :, :n_skip], k_pos.astype(self.dtype)], axis=2)
        else:
            q = _apply_rope(q.astype(jnp.float32), freqs).astype(self.dtype)
            k = _apply_rope(k.astype(jnp.float32), freqs).astype(self.dtype)

        # Expand KV heads to match Q heads
        k = jnp.repeat(k, self._n_rep, axis=1)  # [B, H, S, D]

        # Value-shift: concat half(t) and half(t-1) along channel dim
        # W_v_prev applied to x shifted right by one position (pad front with zero)
        x_prev = jnp.pad(x[:, :-1], ((0, 0), (1, 0), (0, 0)))  # [B, S, d_model]
        v_curr = self.W_v_curr(x)       # [B, S, Hkv * D//2]
        v_prev = self.W_v_prev(x_prev)  # [B, S, Hkv * D//2]
        v = jnp.concatenate([v_curr, v_prev], axis=-1)  # [B, S, Hkv * D]
        v = v.reshape(B, S, Hkv, D).transpose(0, 2, 1, 3)  # [B, Hkv, S, D]
        v = jnp.repeat(v, self._n_rep, axis=1)              # [B, H, S, D]

        return q, k, v

    @nn.compact
    def _gate_combine_up(self, x: jnp.ndarray,
                          out_comp: jnp.ndarray,
                          out_win: jnp.ndarray) -> jnp.ndarray:
        """Gate-blend compressed and window outputs, add residual-alpha, up-project.

        out_comp, out_win: [B, H, S, d_head]
        Returns: [B, S, d_model]
        """
        B, S, _ = x.shape
        H, D = self.n_heads, self._d_head

        # Sigmoid gate: d_model -> d_model//4 -> n_heads*2
        g = jax.nn.sigmoid(
            self.gate_w2(jax.nn.silu(self.gate_w1(x)))
        )  # [B, S, n_heads*2]
        g = g.reshape(B, S, H, 2).transpose(0, 2, 1, 3)  # [B, H, S, 2]

        combined = g[..., 0:1] * out_comp + g[..., 1:2] * out_win  # [B, H, S, D]

        # Residual attention: α (learned per head) * down-projected x
        alpha = self.param("alpha", lambda rng, shape: jnp.full(shape, self._alpha_init),
                           (H, 1, 1))
        x_res = self.W_down_q(x).reshape(B, S, H, D).transpose(0, 2, 1, 3)  # [B, H, S, D]
        out = combined + alpha * x_res  # [B, H, S, D]

        # Up-project back to d_model
        out_flat = out.transpose(0, 2, 1, 3).reshape(B, S, H * D)  # [B, S, latent_q_dim]
        return self.W_up(out_flat)  # [B, S, d_model]


# ─── CCA + CSA ────────────────────────────────────────────────────────────────


class CCACSAAttention(nn.Module):
    """CCA channel compression + CSA sparse global selection (even layers).

    Pipeline:
      1. CCA: down-project, causal conv, QK-mean, value-shift, QK-norm,
              learnable temp, CoPE-RoPE.
      2. CSA compressed branch:
         a. GatedPoolCompressor (two-stream, m=csa_compress_ratio) → C_comp [B, nb, D]
         b. LightningIndexer (ReLU dot-product) → top-k block indices
         c. Gather C_comp entries for selected indices → C_sel [B, S, tk, D]
         d. Apply gathered causal mask with -inf fill → masked attention scores
         e. Append attention sink logit → softmax → weighted sum over C_sel
      3. Window branch: causal sliding window with XSA (self-exclusion).
      4. Gate-blend compressed + window + residual-alpha + up-project.

    Causal correctness: _compressed_causal_mask ensures block j is only
    visible to query i when the entire block lies before i. The Lightning
    Indexer scores future blocks with -inf before relu, so they get 0 score
    and are never preferentially selected. After gathering, the per-block
    causal validity is re-checked and future entries get -inf in the attention
    scores — not just softmax-suppressed. This double guard is the causal-leak
    fix from the PyTorch CSA audit.
    """
    d_model: int
    n_heads: int
    max_seq_len: int
    n_kv_heads: int = 4
    compression: int = 2
    csa_compress_ratio: int = 4
    top_k: int = 128
    d_indexer: int = 32
    window_size: int = 128
    context_len: int = 4096
    init_alpha: float = 0.1
    conv_kernel: int = 4
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, n_skip: int = 0) -> jnp.ndarray:
        """x: [B, S, d_model], n_skip: leading tokens that skip RoPE → [B, S, d_model]"""
        B, S, _ = x.shape
        d_head = self.d_model // (self.compression * self.n_heads)

        # Shared CCA base embedded inline via nn.compact sub-modules
        base = _CCABase(
            d_model=self.d_model, n_heads=self.n_heads, max_seq_len=self.max_seq_len,
            n_kv_heads=self.n_kv_heads, compression=self.compression,
            window_size=self.window_size, context_len=self.context_len,
            init_alpha=self.init_alpha, conv_kernel=self.conv_kernel, dtype=self.dtype,
            name="cca_base",
        )
        q, k, v = base._cca_project(x, n_skip)  # [B, H, S, d_head]

        H = self.n_heads
        m = self.csa_compress_ratio
        n_blocks = S // m
        scale = d_head ** -0.5
        NEG_INF = jnp.finfo(jnp.float32).min / 2

        # ── Two-stream gated pooling → compressed KV ─────────────────────────
        C_comp = GatedPoolCompressor(
            c=d_head, m=m, two_stream=True, dtype=self.dtype, name="compressor")(x)
        C_comp = RMSNorm(name="comp_norm")(C_comp)  # [B, n_blocks, d_head]

        # ── Lightning Indexer: causal scoring with -inf then relu ─────────────
        causal_mask = _compressed_causal_mask(S, n_blocks, m)  # [S, n_blocks]
        causal_3d = causal_mask[None].repeat(B, axis=0)        # [B, S, n_blocks]

        scores = LightningIndexer(
            d_model=self.d_model, d_indexer=self.d_indexer, m=m,
            dtype=self.dtype, name="indexer",
        )(x, causal_3d)  # [B, S, n_blocks]

        # Clean causal top-k: re-mask future blocks to -inf AFTER relu so they are
        # strictly NEVER preferred over a causally-visible block.
        # The indexer already masks -inf before relu, but relu(-inf)=0 ties with a
        # valid block that also scores 0, and jax.lax.top_k can break ties in favour
        # of the future block. Pushing future blocks back to -inf before top_k ensures
        # a visible block is always preferred. Mirrors PyTorch attention.py:491.
        scores = jnp.where(causal_3d, scores.astype(jnp.float32), NEG_INF)

        # Top-k selection (static k for XLA compatibility)
        tk = min(self.top_k, n_blocks)
        _, top_idx = jax.lax.top_k(scores, tk)  # [B, S, tk]

        # Gather compressed entries: C_sel [B, S, tk, d_head]
        # C_comp: [B, n_blocks, d_head] → need [B, S, tk, d_head]
        # Use take_along_axis with carefully broadcasted indices.
        top_idx_exp = jnp.broadcast_to(
            top_idx[:, :, :, None],
            (B, S, tk, d_head),
        )  # [B, S, tk, d_head]
        C_comp_exp = jnp.broadcast_to(
            C_comp[:, None, :, :],
            (B, S, n_blocks, d_head),
        )  # [B, S, n_blocks, d_head]
        C_sel = jnp.take_along_axis(C_comp_exp, top_idx_exp, axis=2)  # [B, S, tk, d_head]

        # Re-derive validity of gathered blocks (causal check after gather)
        # gathered_valid[b, s, i] = causal_3d[b, s, top_idx[b, s, i]]
        gathered_valid = jnp.take_along_axis(causal_3d, top_idx, axis=2)  # [B, S, tk]
        invalid_mask = ~gathered_valid  # True where block is not yet causally visible

        # Compressed attention scores: q · C_sel^T
        # q: [B, H, S, d_head], C_sel: [B, S, tk, d_head]
        attn_scores = jnp.einsum(
            'bhsd,bstd->bhst',
            q.astype(jnp.float32),
            C_sel.astype(jnp.float32),
        ) * scale  # [B, H, S, tk]

        # Apply causal mask with -inf (not just zero) so future blocks never contribute
        attn_scores = jnp.where(invalid_mask[:, None], NEG_INF, attn_scores)

        # Attention sink: learnable logit appended before softmax
        sink = self.param("sink_logits", nn.initializers.zeros, (H,))
        sink_bc = jnp.broadcast_to(sink[None, :, None, None], (B, H, S, 1))
        scores_aug = jnp.concatenate(
            [attn_scores, sink_bc.astype(jnp.float32)], axis=-1)  # [B, H, S, tk+1]

        attn_w = jax.nn.softmax(scores_aug, axis=-1).astype(self.dtype)
        attn_w = attn_w[..., :-1]  # drop sink weight, keep its normalization effect

        # Compressed output: sum over tk selected entries
        out_comp = jnp.einsum(
            'bhst,bstd->bhsd',
            attn_w,
            C_sel.astype(self.dtype),
        )  # [B, H, S, d_head]

        # ── Sliding window (XSA included) ─────────────────────────────────────
        out_win = _window_attention(q, k, v, self.window_size, n_skip, self.dtype)

        return base._gate_combine_up(x, out_comp, out_win)


# ─── CCA + HCA ────────────────────────────────────────────────────────────────


class CCAHCAAttention(nn.Module):
    """CCA channel compression + HCA dense compressed attention (odd layers).

    Pipeline:
      1. CCA: same as CCACSAAttention.
      2. HCA compressed branch:
         a. GatedPoolCompressor (single-stream, m=hca_compress_ratio) → C_comp [B, nb, D]
         b. Dense attention over ALL compressed entries with causal bias (-inf for future)
         c. Early-query guard: tokens with no valid blocks get all-zero output
            (masked fill before final multiply avoids NaN from softmax over all-inf).
         d. Attention sink logit appended.
      3. Window branch: causal sliding window with XSA.
      4. Gate-blend + residual-alpha + up-project.

    At S=2048, m=128: n_blocks=16, ~negligible KV cache.
    """
    d_model: int
    n_heads: int
    max_seq_len: int
    n_kv_heads: int = 4
    compression: int = 2
    hca_compress_ratio: int = 128
    window_size: int = 128
    context_len: int = 4096
    init_alpha: float = 0.1
    conv_kernel: int = 4
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, n_skip: int = 0) -> jnp.ndarray:
        B, S, _ = x.shape
        d_head = self.d_model // (self.compression * self.n_heads)

        base = _CCABase(
            d_model=self.d_model, n_heads=self.n_heads, max_seq_len=self.max_seq_len,
            n_kv_heads=self.n_kv_heads, compression=self.compression,
            window_size=self.window_size, context_len=self.context_len,
            init_alpha=self.init_alpha, conv_kernel=self.conv_kernel, dtype=self.dtype,
            name="cca_base",
        )
        q, k, v = base._cca_project(x, n_skip)

        H = self.n_heads
        m = self.hca_compress_ratio
        n_blocks = S // m
        scale = d_head ** -0.5
        NEG_INF = jnp.finfo(jnp.float32).min / 2

        # ── Single-stream gated pooling → compressed KV ───────────────────────
        C_comp = GatedPoolCompressor(
            c=d_head, m=m, two_stream=False, dtype=self.dtype, name="compressor")(x)
        C_comp = RMSNorm(name="comp_norm")(C_comp)  # [B, n_blocks, d_head]

        # Causal bias: [1, 1, S, n_blocks] — future blocks get -inf
        causal_mask = _compressed_causal_mask(S, n_blocks, m)  # [S, n_blocks]
        causal_bias = jnp.where(causal_mask, 0.0, NEG_INF).astype(jnp.float32)  # [S, n_blocks]

        # Dense compressed scores
        scores = jnp.einsum(
            'bhsd,bnd->bhsn',
            q.astype(jnp.float32),
            C_comp.astype(jnp.float32),
        ) * scale + causal_bias[None, None]  # [B, H, S, n_blocks]

        # Attention sink
        sink = self.param("sink_logits", nn.initializers.zeros, (H,))
        sink_bc = jnp.broadcast_to(sink[None, :, None, None], (B, H, S, 1))
        scores_aug = jnp.concatenate(
            [scores, sink_bc.astype(jnp.float32)], axis=-1)  # [B, H, S, n_blocks+1]

        # Early-query guard: queries with NO valid blocks get all-zero output.
        # Detect: all non-sink scores are -inf.
        no_valid = (scores == NEG_INF).all(axis=-1, keepdims=True)  # [B, H, S, 1]
        # Zero-fill those rows so softmax sees 0s (uniform → uniform weights).
        scores_aug = jnp.where(no_valid, 0.0, scores_aug)

        attn_w = jax.nn.softmax(scores_aug, axis=-1).astype(self.dtype)
        attn_w = attn_w[..., :-1]  # drop sink
        # Zero out early queries so they produce zero output (not uniform blend)
        attn_w = jnp.where(no_valid, 0.0, attn_w)

        out_comp = jnp.einsum(
            'bhsn,bnd->bhsd',
            attn_w,
            C_comp.astype(self.dtype),
        )  # [B, H, S, d_head]

        # ── Sliding window (XSA included) ─────────────────────────────────────
        out_win = _window_attention(q, k, v, self.window_size, n_skip, self.dtype)

        return base._gate_combine_up(x, out_comp, out_win)


# ─── Alternating CCA+CSA / CCA+HCA ───────────────────────────────────────────


class CCACSAHCAAttention(nn.Module):
    """Alternating CCA+CSA (even) / CCA+HCA (odd) — default MORPH attention.

    layer_idx % 2 == 0  →  CCACSAAttention  (sparse global + window)
    layer_idx % 2 == 1  →  CCAHCAAttention  (dense compressed + window)

    All features are always active:
      CCA: channel compression, causal conv, QK-mean, value-shift, learnable temp,
           QK-RMSNorm, CoPE clipped-RoPE, gate, residual-alpha, up-project.
      CSA: two-stream pooling, Lightning Indexer, top-k gather, causal -inf masking,
           attention sinks.
      HCA: single-stream pooling, dense compressed attention, early-query guard,
           attention sinks.
      Window: causal sliding window with XSA (self-exclusion diagonal).

    No if-statements in forward. Alternation is resolved at init time by
    constructing exactly one of CCACSAAttention or CCAHCAAttention.
    """
    d_model: int
    n_heads: int
    max_seq_len: int
    layer_idx: int
    n_kv_heads: int = 4
    compression: int = 2
    csa_compress_ratio: int = 4
    hca_compress_ratio: int = 128
    top_k: int = 128
    d_indexer: int = 32
    window_size: int = 128
    context_len: int = 4096
    init_alpha: float = 0.1
    conv_kernel: int = 4
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, n_skip: int = 0) -> jnp.ndarray:
        """x: [B, S, d_model], n_skip: leading tokens that skip CoPE-RoPE.

        Returns [B, S, d_model].
        """
        shared = dict(
            d_model=self.d_model, n_heads=self.n_heads, max_seq_len=self.max_seq_len,
            n_kv_heads=self.n_kv_heads, compression=self.compression,
            window_size=self.window_size, context_len=self.context_len,
            init_alpha=self.init_alpha, conv_kernel=self.conv_kernel, dtype=self.dtype,
        )
        if self.layer_idx % 2 == 0:
            return CCACSAAttention(
                csa_compress_ratio=self.csa_compress_ratio, top_k=self.top_k,
                d_indexer=self.d_indexer, name="cca_csa", **shared,
            )(x, n_skip)
        else:
            return CCAHCAAttention(
                hca_compress_ratio=self.hca_compress_ratio, name="cca_hca", **shared,
            )(x, n_skip)


# ─── Standalone variants (for ablation) ───────────────────────────────────────


class CSAAttention(nn.Module):
    """CSA only (no CCA channel compression). For ablation against CCACSAAttention.

    Standard GQA + QK-Norm + CoPE-RoPE + GatedPoolCompressor (two-stream) +
    LightningIndexer top-k + sliding window XSA + gate + residual-alpha.

    d_head = d_model // n_heads (no channel compression factor).
    """
    d_model: int
    n_heads: int
    max_seq_len: int
    n_kv_heads: int = 4
    compress_ratio: int = 4
    top_k: int = 128
    d_indexer: int = 32
    window_size: int = 128
    context_len: int = 4096
    init_alpha: float = 0.1
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, n_skip: int = 0) -> jnp.ndarray:
        B, S, _ = x.shape
        d_head = self.d_model // self.n_heads
        H = self.n_heads
        Hkv = self.n_kv_heads
        n_rep = H // Hkv
        scale = d_head ** -0.5
        NEG_INF = jnp.finfo(jnp.float32).min / 2

        # GQA Q/K/V projections
        q = nn.Dense(H * d_head, use_bias=False, dtype=self.dtype, name="W_q")(x)
        k = nn.Dense(Hkv * d_head, use_bias=False, dtype=self.dtype, name="W_k")(x)
        v = nn.Dense(Hkv * d_head, use_bias=False, dtype=self.dtype, name="W_v")(x)
        q = q.reshape(B, S, H, d_head).transpose(0, 2, 1, 3)     # [B, H, S, D]
        k = k.reshape(B, S, Hkv, d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, Hkv, d_head).transpose(0, 2, 1, 3)

        # QK-Norm
        q = RMSNorm(name="q_norm")(q)
        k = RMSNorm(name="k_norm")(k)

        # CoPE-RoPE
        freqs = _cope_freqs(d_head, self.max_seq_len, self.context_len).astype(jnp.float32)
        if n_skip > 0:
            q_p = _apply_rope(q[:, :, n_skip:].astype(jnp.float32), freqs)
            k_p = _apply_rope(k[:, :, n_skip:].astype(jnp.float32), freqs)
            q = jnp.concatenate([q[:, :, :n_skip], q_p.astype(self.dtype)], axis=2)
            k = jnp.concatenate([k[:, :, :n_skip], k_p.astype(self.dtype)], axis=2)
        else:
            q = _apply_rope(q.astype(jnp.float32), freqs).astype(self.dtype)
            k = _apply_rope(k.astype(jnp.float32), freqs).astype(self.dtype)

        k = jnp.repeat(k, n_rep, axis=1)
        v = jnp.repeat(v, n_rep, axis=1)

        # CSA compressed branch
        m = self.compress_ratio
        n_blocks = S // m
        C_comp = GatedPoolCompressor(
            c=d_head, m=m, two_stream=True, dtype=self.dtype, name="compressor")(x)
        C_comp = RMSNorm(name="comp_norm")(C_comp)

        causal_mask = _compressed_causal_mask(S, n_blocks, m)
        causal_3d = causal_mask[None].repeat(B, axis=0)
        scores_idx = LightningIndexer(
            d_model=self.d_model, d_indexer=self.d_indexer, m=m,
            dtype=self.dtype, name="indexer",
        )(x, causal_3d)

        # Clean causal top-k: re-mask future blocks to -inf after relu (mirrors
        # CCACSAAttention and PyTorch attention.py:491 causal-leak fix).
        scores_idx = jnp.where(causal_3d, scores_idx.astype(jnp.float32), NEG_INF)

        tk = min(self.top_k, n_blocks)
        _, top_idx = jax.lax.top_k(scores_idx, tk)

        top_idx_exp = jnp.broadcast_to(top_idx[:, :, :, None], (B, S, tk, d_head))
        C_comp_exp = jnp.broadcast_to(C_comp[:, None, :, :], (B, S, n_blocks, d_head))
        C_sel = jnp.take_along_axis(C_comp_exp, top_idx_exp, axis=2)

        gathered_valid = jnp.take_along_axis(causal_3d, top_idx, axis=2)
        attn_scores = jnp.einsum(
            'bhsd,bstd->bhst',
            q.astype(jnp.float32), C_sel.astype(jnp.float32)) * scale
        attn_scores = jnp.where(~gathered_valid[:, None], NEG_INF, attn_scores)

        sink = self.param("sink_logits", nn.initializers.zeros, (H,))
        scores_aug = jnp.concatenate(
            [attn_scores, jnp.broadcast_to(sink[None, :, None, None], (B, H, S, 1)).astype(jnp.float32)],
            axis=-1)
        attn_w = jax.nn.softmax(scores_aug, axis=-1).astype(self.dtype)[..., :-1]
        out_comp = jnp.einsum('bhst,bstd->bhsd', attn_w, C_sel.astype(self.dtype))

        out_win = _window_attention(q, k, v, self.window_size, n_skip, self.dtype)

        # Sigmoid gate
        g = jax.nn.sigmoid(
            nn.Dense(H * 2, use_bias=False, dtype=self.dtype, name="gate_w2")(
                jax.nn.silu(
                    nn.Dense(self.d_model // 4, use_bias=False, dtype=self.dtype,
                             name="gate_w1")(x)
                )
            )
        ).reshape(B, S, H, 2).transpose(0, 2, 1, 3)

        # Residual-alpha
        alpha = self.param("alpha", lambda rng, sh: jnp.full(sh, self.init_alpha), (H, 1, 1))
        x_res = nn.Dense(H * d_head, use_bias=False, dtype=self.dtype, name="x_res_proj")(x)
        x_res = x_res.reshape(B, S, H, d_head).transpose(0, 2, 1, 3)

        out = g[..., 0:1] * out_comp + g[..., 1:2] * out_win + alpha * x_res
        out = out.transpose(0, 2, 1, 3).reshape(B, S, H * d_head)
        return nn.Dense(self.d_model, use_bias=False, dtype=self.dtype, name="W_o")(out)


class HCAAttention(nn.Module):
    """HCA only (no CCA channel compression). For ablation against CCAHCAAttention."""
    d_model: int
    n_heads: int
    max_seq_len: int
    n_kv_heads: int = 4
    compress_ratio: int = 128
    window_size: int = 128
    context_len: int = 4096
    init_alpha: float = 0.1
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, n_skip: int = 0) -> jnp.ndarray:
        B, S, _ = x.shape
        d_head = self.d_model // self.n_heads
        H = self.n_heads
        Hkv = self.n_kv_heads
        n_rep = H // Hkv
        scale = d_head ** -0.5
        NEG_INF = jnp.finfo(jnp.float32).min / 2

        q = nn.Dense(H * d_head, use_bias=False, dtype=self.dtype, name="W_q")(x)
        k = nn.Dense(Hkv * d_head, use_bias=False, dtype=self.dtype, name="W_k")(x)
        v = nn.Dense(Hkv * d_head, use_bias=False, dtype=self.dtype, name="W_v")(x)
        q = q.reshape(B, S, H, d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, Hkv, d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, Hkv, d_head).transpose(0, 2, 1, 3)

        q = RMSNorm(name="q_norm")(q)
        k = RMSNorm(name="k_norm")(k)

        freqs = _cope_freqs(d_head, self.max_seq_len, self.context_len).astype(jnp.float32)
        if n_skip > 0:
            q_p = _apply_rope(q[:, :, n_skip:].astype(jnp.float32), freqs)
            k_p = _apply_rope(k[:, :, n_skip:].astype(jnp.float32), freqs)
            q = jnp.concatenate([q[:, :, :n_skip], q_p.astype(self.dtype)], axis=2)
            k = jnp.concatenate([k[:, :, :n_skip], k_p.astype(self.dtype)], axis=2)
        else:
            q = _apply_rope(q.astype(jnp.float32), freqs).astype(self.dtype)
            k = _apply_rope(k.astype(jnp.float32), freqs).astype(self.dtype)

        k = jnp.repeat(k, n_rep, axis=1)
        v = jnp.repeat(v, n_rep, axis=1)

        m = self.compress_ratio
        n_blocks = S // m
        C_comp = GatedPoolCompressor(
            c=d_head, m=m, two_stream=False, dtype=self.dtype, name="compressor")(x)
        C_comp = RMSNorm(name="comp_norm")(C_comp)

        causal_mask = _compressed_causal_mask(S, n_blocks, m)
        causal_bias = jnp.where(causal_mask, 0.0, NEG_INF).astype(jnp.float32)

        scores = jnp.einsum(
            'bhsd,bnd->bhsn',
            q.astype(jnp.float32), C_comp.astype(jnp.float32)) * scale + causal_bias[None, None]

        sink = self.param("sink_logits", nn.initializers.zeros, (H,))
        scores_aug = jnp.concatenate(
            [scores, jnp.broadcast_to(sink[None, :, None, None], (B, H, S, 1)).astype(jnp.float32)],
            axis=-1)

        no_valid = (scores == NEG_INF).all(axis=-1, keepdims=True)
        scores_aug = jnp.where(no_valid, 0.0, scores_aug)
        attn_w = jax.nn.softmax(scores_aug, axis=-1).astype(self.dtype)[..., :-1]
        attn_w = jnp.where(no_valid, 0.0, attn_w)

        out_comp = jnp.einsum('bhsn,bnd->bhsd', attn_w, C_comp.astype(self.dtype))
        out_win = _window_attention(q, k, v, self.window_size, n_skip, self.dtype)

        g = jax.nn.sigmoid(
            nn.Dense(H * 2, use_bias=False, dtype=self.dtype, name="gate_w2")(
                jax.nn.silu(
                    nn.Dense(self.d_model // 4, use_bias=False, dtype=self.dtype,
                             name="gate_w1")(x)
                )
            )
        ).reshape(B, S, H, 2).transpose(0, 2, 1, 3)

        alpha = self.param("alpha", lambda rng, sh: jnp.full(sh, self.init_alpha), (H, 1, 1))
        x_res = nn.Dense(H * d_head, use_bias=False, dtype=self.dtype, name="x_res_proj")(x)
        x_res = x_res.reshape(B, S, H, d_head).transpose(0, 2, 1, 3)

        out = g[..., 0:1] * out_comp + g[..., 1:2] * out_win + alpha * x_res
        out = out.transpose(0, 2, 1, 3).reshape(B, S, H * d_head)
        return nn.Dense(self.d_model, use_bias=False, dtype=self.dtype, name="W_o")(out)
