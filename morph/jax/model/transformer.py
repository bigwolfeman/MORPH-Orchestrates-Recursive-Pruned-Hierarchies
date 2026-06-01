"""MORPH Transformer — JAX/Flax port of morph/model/transformer.py.

Parcae-style looped architecture with all features baked in.
Architecture: Embedding → Prelude → Core×T → Coda → STP → LM head

Loop hierarchy:
  Inner: Parcae core loop (T iterations, Poisson depth, BPTT via jax.checkpoint)
  Outer: (Zyphra RSA — deferred to inference-time, requires RL)

Design principles:
  - Python for-loop + jax.checkpoint for the core loop (NOT lax.scan — known
    gotcha from TPU/CLAUDE.md: Poisson depth requires per-sample masking, not
    static scan with fixed iteration count).
  - No runtime feature flags. All features always on.
  - bf16 compatible.

Forward returns a dict matching the PyTorch model exactly:
  {"logits": [B, T, vocab_size],
   "loss": scalar (if labels given),
   "stp_loss": scalar}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from .attention import CCACSAHCAAttention, RMSNorm
from .embeddings import MORPHEmbedding
from .mhc import MultiRateResidual, ChannelInject, DEFAULT_CHANNEL_DIMS
from .prediction import STPLoss


# ── MORPHTransformerBlock: MORPHBlock with n_skip support ─────────────────────


class MORPHTransformerBlock(nn.Module):
    """Transformer block with multi-rate residual dynamics and n_skip support.

    Passes n_skip to the attention module for the generic sink/persistent-token
    RoPE offset (leading positions that skip RoPE and attend to everything).
    Currently always called with n_skip=0 — the mechanism is retained but unused.

    Attributes match MORPHBlock:
        attn_module : CCACSAHCAAttention
        mlp_module  : SwiGLU
        d_model     : int
        norm_eps    : float
        channel_dims: tuple
    """

    attn_module: nn.Module
    mlp_module: nn.Module
    d_model: int
    norm_eps: float = 1e-5
    channel_dims: tuple = DEFAULT_CHANNEL_DIMS

    @nn.compact
    def __call__(
        self,
        h: jnp.ndarray,
        deterministic: bool = True,
        n_skip: int = 0,
    ) -> jnp.ndarray:
        norm_attn = RMSNorm(eps=self.norm_eps, name="norm_attn")
        norm_mlp  = RMSNorm(eps=self.norm_eps, name="norm_mlp")
        mrr_attn = MultiRateResidual(channel_dims=self.channel_dims, name="mrr_attn")
        mrr_mlp  = MultiRateResidual(channel_dims=self.channel_dims, name="mrr_mlp")

        def _attn_fn(x):
            return self.attn_module(norm_attn(x), n_skip=n_skip)

        h = mrr_attn(h, _attn_fn)

        def _mlp_fn(x):
            return self.mlp_module(norm_mlp(x), deterministic=deterministic)

        h = mrr_mlp(h, _mlp_fn)
        return h


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MORPHConfig:
    """Frozen configuration dataclass matching PyTorch MORPHConfig exactly."""
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 0        # 0 = auto (8/3 * d_model, rounded to 64)
    vocab_size: int = 49152
    max_seq_len: int = 4096

    n_prelude: int = 3
    n_core: int = 6
    n_coda: int = 3
    mean_depth: int = 6
    max_depth: int = 8
    bptt_depth: int = 4

    channel_dims: tuple = (384, 256, 128)

    # Attention
    compression: int = 2
    n_kv_heads: int = 4
    csa_compress_ratio: int = 4
    hca_compress_ratio: int = 128
    top_k: int = 128
    d_indexer: int = 32
    window_size: int = 128
    context_len: int = 4096
    conv_kernel: int = 4
    init_alpha: float = 0.1

    # Embeddings
    lorentz_fraction: float = 0.25
    bigram_hash_vocab: int = 49152

    # Prediction (STP — Semantic Tube Predictor, geometric regularizer)
    stp_lambda: float = 0.02
    stp_tau: int = 64

    # Training
    dropout: float = 0.1

    def d_ff_actual(self) -> int:
        return self.d_ff if self.d_ff > 0 else ((self.d_model * 8 // 3 + 63) // 64 * 64)


# ── DiagonalInjection ──────────────────────────────────────────────────────────


class DiagonalInjection(nn.Module):
    """SSM-style diagonal injection on the context channel only.

    h_ctx = decay * h_ctx + dt * e_ctx
    Spectral radius < 1 guaranteed: A = exp(log_A).clamp(max=0.9999).

    Attributes
    ----------
    channel_start : int
    channel_end   : int
    init_decay    : float   Initial log_A value.
    """

    channel_start: int
    channel_end: int
    init_decay: float = 0.447

    def setup(self):
        d = self.channel_end - self.channel_start
        # log_A: log(init_decay) per element
        self.log_A = self.param(
            "log_A",
            nn.initializers.constant(math.log(self.init_decay)),
            (d,),
        )
        # log_dt: zeros → dt = exp(0) = 1 at init
        self.log_dt = self.param(
            "log_dt",
            nn.initializers.zeros,
            (d,),
        )

    def __call__(self, h: jnp.ndarray, e: jnp.ndarray) -> jnp.ndarray:
        """Apply diagonal injection to the context channel.

        Args:
            h: [B, S, d_model] current hidden state.
            e: [B, S, d_model] input (post norm_input).

        Returns:
            [B, S, d_model] with context channel updated.
        """
        A = jnp.clip(jnp.exp(self.log_A.astype(jnp.float32)), a_max=0.9999).astype(h.dtype)
        dt = jnp.exp(self.log_dt.astype(jnp.float32)).astype(h.dtype)

        h_ctx = h[..., self.channel_start:self.channel_end]
        e_ctx = e[..., self.channel_start:self.channel_end]
        new_ctx = A * h_ctx + dt * e_ctx

        prefix = h[..., :self.channel_start]
        suffix = h[..., self.channel_end:]
        return jnp.concatenate([prefix, new_ctx, suffix], axis=-1)


# ── SwiGLU MLP ────────────────────────────────────────────────────────────────


class SwiGLU(nn.Module):
    """SwiGLU MLP: gate + up → silu(gate)*up → down.

    Attributes
    ----------
    d_model   : int   Input/output dimension.
    d_ff      : int   Expanded dimension (pre-split).
    dropout   : float Dropout rate.
    dtype     : dtype Compute dtype (bf16 recommended).
    """

    d_model: int
    d_ff: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # gate_up: [B, S, d_ff * 2]
        gu = nn.Dense(
            self.d_ff * 2,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="gate_up",
        )(x)
        gate, up = jnp.split(gu, 2, axis=-1)
        hidden = jax.nn.silu(gate) * up
        if self.dropout > 0.0:
            hidden = nn.Dropout(rate=self.dropout)(hidden, deterministic=deterministic)
        return nn.Dense(
            self.d_model,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="down",
        )(hidden)


# ── LMHeadMixer ───────────────────────────────────────────────────────────────


class LMHeadMixer(nn.Module):
    """3-channel mixer before LM head.

    Per-channel softplus scale + cross-channel linear (identity-init).

    Attributes
    ----------
    d_model      : int
    channel_dims : tuple[int, ...]
    """

    d_model: int
    channel_dims: tuple = DEFAULT_CHANNEL_DIMS

    def setup(self):
        n = len(self.channel_dims)
        # Init: softplus⁻¹(1) ≈ 0.541 → softplus(0.541) ≈ 1.0
        # Use zeros for simplicity; softplus(0) ≈ 0.693, close enough to start
        self.channel_scales_raw = self.param(
            "channel_scales",
            nn.initializers.zeros,
            (n,),
        )
        # Identity-initialized mix matrix
        def _eye_init(rng, shape, dtype=jnp.float32):
            assert len(shape) == 2 and shape[0] == shape[1], f"eye_init requires square matrix, got {shape}"
            return jnp.eye(shape[0], dtype=dtype)

        self.mix = nn.Dense(
            self.d_model,
            use_bias=False,
            kernel_init=_eye_init,
            name="mix",
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scales = jax.nn.softplus(self.channel_scales_raw.astype(jnp.float32)).astype(x.dtype)

        # Split and scale per channel
        chunks = []
        s = 0
        for i, d in enumerate(self.channel_dims):
            chunks.append(x[..., s:s + d] * scales[i])
            s += d
        scaled = jnp.concatenate(chunks, axis=-1)
        return self.mix(scaled)


# ── MORPHTransformer ──────────────────────────────────────────────────────────


class MORPHTransformer(nn.Module):
    """MORPH Transformer — full Parcae loop architecture.

    Exact JAX/Flax port of morph/model/transformer.py MORPHTransformer.

    Usage:
        cfg = MORPHConfig()
        model = MORPHTransformer(cfg=cfg)

        # Initialize
        variables = model.init(
            {"params": key, "dropout": d_key},
            input_ids,
            training=False,
        )
        params = variables["params"]

        # Training step
        out, grads = jax.value_and_grad(
            lambda p: model.apply(
                {"params": p},
                input_ids, labels=labels, training=True,
                rngs={"dropout": d_key},
            )["loss"],
        )(params)

    Attributes
    ----------
    cfg : MORPHConfig
    """

    cfg: MORPHConfig

    def setup(self):
        cfg = self.cfg
        d = cfg.d_model
        n_total = cfg.n_prelude + cfg.n_core + cfg.n_coda
        d_ff = cfg.d_ff_actual()

        # Channel boundaries
        ch = cfg.channel_dims
        assert sum(ch) == d, f"channel_dims {ch} must sum to d_model={d}"
        starts = []
        ends = []
        s = 0
        for c in ch:
            starts.append(s)
            ends.append(s + c)
            s += c
        self._ctx_start = starts[1]
        self._ctx_end = ends[1]
        self._ch_starts = starts
        self._ch_ends = ends

        dtype = jnp.bfloat16

        # ── Attention kwargs (shared across all layers) ───────────────────
        attn_kw = dict(
            d_model=d, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
            compression=cfg.compression,
            csa_compress_ratio=cfg.csa_compress_ratio,
            hca_compress_ratio=cfg.hca_compress_ratio,
            top_k=cfg.top_k,
            d_indexer=cfg.d_indexer,
            window_size=cfg.window_size,
            context_len=cfg.context_len,
            max_seq_len=cfg.max_seq_len,
            conv_kernel=cfg.conv_kernel,
            init_alpha=cfg.init_alpha,
            dtype=dtype,
        )

        def _make_block(layer_idx: int, name: str) -> MORPHTransformerBlock:
            # Note: do NOT set name= on inner modules (attn_module, mlp_module).
            # Flax derives their names from MORPHTransformerBlock's attribute names
            # ("attn_module" → "attn_module_0" etc) when they are stored in lists.
            # Explicit name= would collide across list elements.
            return MORPHTransformerBlock(
                attn_module=CCACSAHCAAttention(layer_idx=layer_idx, **attn_kw),
                mlp_module=SwiGLU(d_model=d, d_ff=d_ff, dropout=cfg.dropout, dtype=dtype),
                d_model=d,
                norm_eps=1e-5,
                channel_dims=ch,
                name=name,
            )

        # ── Embedding ─────────────────────────────────────────────────────
        self.embed = MORPHEmbedding(
            vocab_size=cfg.vocab_size,
            d_model=d,
            lorentz_fraction=cfg.lorentz_fraction,
            bigram_hash_vocab=cfg.bigram_hash_vocab,
            n_layers=n_total,
            name="embed",
        )

        # ── Prelude ───────────────────────────────────────────────────────
        self.prelude = [
            _make_block(i, f"prelude_{i}") for i in range(cfg.n_prelude)
        ]

        # ── Loop state transition ─────────────────────────────────────────
        self.input_norm = RMSNorm(eps=1e-5, name="input_norm")
        self.injection = DiagonalInjection(
            channel_start=self._ctx_start,
            channel_end=self._ctx_end,
            name="injection",
        )

        # ── Core (shared across loop iterations) ──────────────────────────
        self.core = [
            _make_block(cfg.n_prelude + i, f"core_{i}") for i in range(cfg.n_core)
        ]

        # ── Coda ──────────────────────────────────────────────────────────
        self.coda = [
            _make_block(cfg.n_prelude + cfg.n_core + i, f"coda_{i}")
            for i in range(cfg.n_coda)
        ]

        # ── x0 skip (inject into context channel) ─────────────────────────
        self.x0_injects = [
            ChannelInject(
                channel_start=self._ctx_start,
                channel_end=self._ctx_end,
                d_signal=d,
                init_scale=0.0,
                name=f"x0_inject_{i}",
            )
            for i in range(n_total)
        ]

        # ── Value embeddings (inject into context channel) ─────────────────
        n_ve = min(3, cfg.n_prelude)
        self._n_ve = n_ve
        self.value_embeds = [
            ChannelInject(
                channel_start=self._ctx_start,
                channel_end=self._ctx_end,
                d_signal=d,
                init_scale=0.0,
                name=f"value_embed_{i}",
            )
            for i in range(n_ve)
        ]
        self.value_embed_tables = [
            nn.Embed(
                num_embeddings=cfg.vocab_size,
                features=d,
                embedding_init=nn.initializers.normal(stddev=0.02),
                name=f"value_embed_table_{i}",
            )
            for i in range(n_ve)
        ]

        # ── LM head ───────────────────────────────────────────────────────
        self.lm_mixer = LMHeadMixer(d_model=d, channel_dims=ch, name="lm_mixer")
        self.final_norm = RMSNorm(eps=1e-5, name="final_norm")

        # ── Prediction (STP — Semantic Tube Predictor) ─────────────────────
        self.stp = STPLoss(mode="geodesic", tau=cfg.stp_tau, name="stp")

    # ── x0 and value-embed injection helpers ──────────────────────────────────

    def _apply_x0(self, x: jnp.ndarray, layer_idx: int, x0: jnp.ndarray) -> jnp.ndarray:
        return self.x0_injects[layer_idx](x, x0)

    def _apply_ve(self, x: jnp.ndarray, layer_idx: int, input_ids: jnp.ndarray) -> jnp.ndarray:
        # value embeds only apply to the first n_ve prelude layers (layer 0..n_ve-1)
        if layer_idx < self._n_ve:
            signal = self.value_embed_tables[layer_idx](input_ids)
            return self.value_embeds[layer_idx](x, signal)
        return x

    # ── Forward (dispatch to segmented or single) ─────────────────────────────

    def __call__(
        self,
        input_ids: jnp.ndarray,
        labels: Optional[jnp.ndarray] = None,
        training: bool = False,
        rng: Optional[jax.Array] = None,
    ) -> dict:
        """Full forward pass over the whole sequence.

        Args:
            input_ids: [B, T] int32 token ids.
            labels:    [B, T] int32 targets, or None for logits-only.
            training:  True during training (enables dropout + Poisson depth).
            rng:       PRNGKey for dropout + Poisson sampling.

        Returns:
            dict with "logits" and (if labels given) "loss", "stp_loss".
        """
        return self._forward_single(input_ids, labels, training, rng)

    def _forward_single(
        self,
        input_ids: jnp.ndarray,
        labels: Optional[jnp.ndarray],
        training: bool,
        rng: Optional[jax.Array],
    ) -> dict:
        """Full-sequence forward pass (core of the model)."""
        cfg = self.cfg
        B, T = input_ids.shape

        # ── Embedding ─────────────────────────────────────────────────────
        x = self.embed(input_ids)                         # [B, T, d_model]
        if training and cfg.dropout > 0.0 and rng is not None:
            rng, d_rng = jax.random.split(rng)
            x = nn.Dropout(rate=cfg.dropout)(x, deterministic=False)
        bigram_emb = self.embed.get_bigram(input_ids)     # [B, T, d_model]

        x0 = x  # x0 skip connection (immutable in JAX)

        # ── Prelude ────────────────────────────────────────────────────────
        for i, layer in enumerate(self.prelude):
            x = self._apply_x0(x, i, x0)
            x = self._apply_ve(x, i, input_ids)
            x = self.embed.bigram.inject(x, bigram_emb, i)
            x = layer(x, deterministic=not training)

        # ── Core loop ──────────────────────────────────────────────────────
        e = self.input_norm(x)   # [B, T, d_model]
        h = e                    # loop state

        if training and rng is not None:
            rng, depth_rng = jax.random.split(rng)
            # Poisson depth: per-sample, clamped to [1, max_depth]
            depths = jax.random.poisson(
                depth_rng, lam=cfg.mean_depth, shape=(B,)
            ).astype(jnp.int32)
            depths = jnp.clip(depths, 1, cfg.max_depth)
        else:
            depths = jnp.full((B,), cfg.mean_depth, dtype=jnp.int32)

        # At inference: use mean_depth fixed iterations (matching PyTorch).
        # At training: use max_depth (so we iterate all possible Poisson steps,
        # masking out completed samples via the 'active' boolean).
        # PyTorch: total_iters = int(depths.max()); at inference depths are all
        # mean_depth so total_iters == mean_depth.
        total_iters = int(cfg.max_depth) if training else int(cfg.mean_depth)
        n_nograd = max(0, total_iters - cfg.bptt_depth)

        for t in range(total_iters):
            # active[b] = True if sample b is still iterating at step t
            active = (t < depths)[:, None, None]  # [B, 1, 1]

            # Capture loop variables explicitly to avoid closure mutation issues
            _e, _ids, _x0, _bg = e, input_ids, x0, bigram_emb
            _det = not training

            def _core_step(h_in):
                h_inj = self.injection(h_in, _e)
                for ci, clayer in enumerate(self.core):
                    gi = cfg.n_prelude + ci
                    h_inj = self._apply_x0(h_inj, gi, _x0)
                    h_inj = self._apply_ve(h_inj, gi, _ids)
                    h_inj = self.embed.bigram.inject(h_inj, _bg, gi)
                    h_inj = clayer(h_inj, deterministic=_det)
                return h_inj

            if t < n_nograd:
                h_new = jax.lax.stop_gradient(_core_step(jax.lax.stop_gradient(h)))
            elif training:
                h_new = jax.checkpoint(_core_step)(h)
            else:
                h_new = _core_step(h)

            # Poisson depth masking: frozen samples get unchanged h
            h = jnp.where(active, h_new, h)

        x = h

        # ── Coda ───────────────────────────────────────────────────────────
        for i, layer in enumerate(self.coda):
            gi = cfg.n_prelude + cfg.n_core + i
            x = self._apply_x0(x, gi, x0)
            x = self._apply_ve(x, gi, input_ids)
            x = self.embed.bigram.inject(x, bigram_emb, gi)
            x = layer(x, deterministic=not training)

        # ── STP loss ───────────────────────────────────────────────────────
        stp_loss = self.stp(self.final_norm(x))

        # ── LM head ────────────────────────────────────────────────────────
        x_for_head = self.lm_mixer(x)
        x_for_head = self.final_norm(x_for_head)
        logits = self.embed.attend(x_for_head)

        out = {"logits": logits}

        if labels is not None:
            # Cross-entropy loss
            B_flat = B * T
            vocab = cfg.vocab_size
            log_probs = jax.nn.log_softmax(logits.reshape(B_flat, vocab).astype(jnp.float32), axis=-1)
            labels_flat = labels.reshape(B_flat)
            valid = labels_flat != -100
            ce_loss = -jnp.sum(
                jnp.where(valid, log_probs[jnp.arange(B_flat), jnp.where(valid, labels_flat, 0)], 0.0)
            ) / jnp.maximum(valid.sum(), 1)

            loss = ce_loss + cfg.stp_lambda * stp_loss
            out["stp_loss"] = stp_loss
            out["loss"] = loss

        return out
