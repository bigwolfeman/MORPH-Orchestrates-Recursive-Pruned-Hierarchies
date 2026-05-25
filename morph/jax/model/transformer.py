"""MORPH Transformer — JAX/Flax port of morph/model/transformer.py.

Parcae-style looped architecture with all features baked in.
Architecture: Embedding → SSM inject → Prelude → MAC tokens → Core×T
              → Strip MAC → Coda → STP → Z-latent → LM head

Three-loop hierarchy:
  Inner:  Parcae core loop (T iterations, Poisson depth, BPTT via jax.checkpoint)
  Middle: Neural memory SSM (gradient-based surprise update on forward pass)
  Outer:  (RSA — deferred to inference-time)

Design principles:
  - Python for-loop + jax.checkpoint for the core loop (NOT lax.scan — known
    gotcha from TPU/CLAUDE.md: Poisson depth requires per-sample masking, not
    static scan with fixed iteration count).
  - No runtime feature flags. All features always on.
  - bf16 compatible.
  - mutable "memory_state" collection carried alongside params.

Forward returns a dict matching the PyTorch model exactly:
  {"logits": [B, T, vocab_size],
   "loss": scalar (if labels given),
   "memory_loss": scalar (if training),
   "z_loss": scalar (if multi-segment),
   "stp_loss": scalar,
   "z_target": [B, T, d_z],
   "z_memory_raw": [B, T, d_z],
   "z_prelude": [B, T, d_z],
   "x_prelude_raw": [B, T, d_model]}

Segmented forward: when segment_size < T, forward splits into segments,
accumulates per-segment z-latents, and applies split_nsm_outer_loss.
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
from .mhc import MHCResidual, MHCChannelInject, DEFAULT_CHANNEL_DIMS
from .memory import MemorySystem
from .prediction import STPLoss, ZLatentHeads, split_nsm_outer_loss, _smooth_l1, SIGReg


# ── MORPHTransformerBlock: MHCTransformerBlock with n_skip support ────────────


class MORPHTransformerBlock(nn.Module):
    """Transformer block with mHC residual dynamics and n_skip support.

    Extends MHCTransformerBlock to pass n_skip to the attention module,
    which is needed for RoPE offset when MAC tokens are prepended.

    Attributes match MHCTransformerBlock:
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
        mhc_attn = MHCResidual(channel_dims=self.channel_dims, name="mhc_attn")
        mhc_mlp  = MHCResidual(channel_dims=self.channel_dims, name="mhc_mlp")

        def _attn_fn(x):
            return self.attn_module(norm_attn(x), n_skip=n_skip)

        h = mhc_attn(h, _attn_fn)

        def _mlp_fn(x):
            return self.mlp_module(norm_mlp(x), deterministic=deterministic)

        h = mhc_mlp(h, _mlp_fn)
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

    # Memory
    n_memory_layers: int = 4
    n_memory_tokens: int = 32

    # Prediction
    stp_lambda: float = 0.02
    stp_tau: int = 64
    d_z: int = 256
    segment_size: int = 1024

    # Training
    dropout: float = 0.1
    mac_warmup_steps: int = 2000

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

    The model carries a "memory_state" mutable collection for the neural memory
    MLP weights and momentum buffer, in addition to the standard "params"
    collection.

    Usage:
        cfg = MORPHConfig()
        model = MORPHTransformer(cfg=cfg)

        # Initialize
        variables = model.init(
            {"params": key, "dropout": d_key, "memory_init": m_key},
            input_ids,
            training=False,
        )
        params = variables["params"]
        mem_state = variables.get("memory_state", {})

        # Training step
        (out, mut_vars), grads = jax.value_and_grad(
            lambda p: model.apply(
                {"params": p, "memory_state": mem_state},
                input_ids, labels=labels, training=True,
                mutable=["memory_state"],
                rngs={"dropout": d_key},
            ),
            has_aux=True,
        )(params)
        mem_state = mut_vars["memory_state"]

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
        self._mem_start = starts[2]
        self._mem_end = ends[2]
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
            MHCChannelInject(
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
            MHCChannelInject(
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

        # ── Neural memory ─────────────────────────────────────────────────
        self.memory = MemorySystem(
            d_model=d,
            n_layers=cfg.n_memory_layers,
            n_memory_tokens=cfg.n_memory_tokens,
            d_memory_channel=ch[2],
            name="memory",
        )

        # ── LM head ───────────────────────────────────────────────────────
        self.lm_mixer = LMHeadMixer(d_model=d, channel_dims=ch, name="lm_mixer")
        self.final_norm = RMSNorm(eps=1e-5, name="final_norm")

        # ── Prediction (STP + z-latent) ────────────────────────────────────
        self.stp = STPLoss(mode="geodesic", tau=cfg.stp_tau, name="stp")
        self.z_heads = ZLatentHeads(d_model=d, d_z=cfg.d_z, name="z_heads")

    # ── Memory warmup helpers ──────────────────────────────────────────────────

    def _mem_scale(self, step: int) -> float:
        """Memory contribution scale: 0.0 during warmup, ramps to 1.0."""
        if step <= self.cfg.mac_warmup_steps:
            return 0.0
        ramp = 500
        return min(1.0, (step - self.cfg.mac_warmup_steps) / ramp)

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
        step: int = 0,
        rng: Optional[jax.Array] = None,
    ) -> dict:
        """Full forward pass.

        Args:
            input_ids: [B, T] int32 token ids.
            labels:    [B, T] int32 targets, or None for logits-only.
            training:  True during training (enables memory update, Poisson depth).
            step:      Global training step (for memory warmup ramp).
            rng:       PRNGKey for dropout + Poisson sampling + SIGReg.

        Returns:
            dict with at least "logits" and optionally "loss", "memory_loss",
            "z_loss", "stp_loss", "z_target", "z_memory_raw", "z_prelude",
            "x_prelude_raw".
        """
        cfg = self.cfg
        B, T = input_ids.shape
        seg = cfg.segment_size

        if seg is not None and seg < T:
            return self._forward_segmented(input_ids, labels, seg, training, step, rng)
        return self._forward_single(input_ids, labels, training, step, rng)

    def _forward_segmented(
        self,
        input_ids: jnp.ndarray,
        labels: Optional[jnp.ndarray],
        seg: int,
        training: bool,
        step: int,
        rng: Optional[jax.Array],
    ) -> dict:
        """Segmented forward with delayed cross-segment z-loss (split_nsm)."""
        B, T = input_ids.shape
        n_segs = T // seg

        all_logits = []
        total_loss = jnp.zeros(())
        total_mem_loss = jnp.zeros(())
        total_z_loss = jnp.zeros(())
        total_stp_loss = jnp.zeros(())
        n_loss_segs = 0

        seg_z_codas = []
        seg_z_memories = []
        seg_z_preludes = []
        seg_bb_preds = []

        rng_segs = jax.random.split(rng, n_segs) if rng is not None else [None] * n_segs

        for s in range(n_segs):
            seg_ids = input_ids[:, s * seg:(s + 1) * seg]
            seg_labels = labels[:, s * seg:(s + 1) * seg] if labels is not None else None

            out = self._forward_single(seg_ids, seg_labels, training, step, rng_segs[s])

            if "z_target" in out:
                seg_z_codas.append(out["z_target"])
            if "z_memory_raw" in out:
                seg_z_memories.append(out["z_memory_raw"])
            if "z_prelude" in out:
                seg_z_preludes.append(out["z_prelude"])
            if "x_prelude_raw" in out:
                seg_bb_preds.append(
                    self.z_heads.backbone_predict(out["x_prelude_raw"])
                )

            all_logits.append(out["logits"])
            if "loss" in out:
                total_loss = total_loss + out["loss"]
                n_loss_segs += 1
            if "memory_loss" in out:
                total_mem_loss = total_mem_loss + out["memory_loss"]
            if "z_loss" in out:
                total_z_loss = total_z_loss + out["z_loss"]
            if "stp_loss" in out:
                total_stp_loss = total_stp_loss + out["stp_loss"]

        # Delayed cross-segment z-loss (split_nsm) — training only
        # We implement inline (instead of calling split_nsm_outer_loss) because
        # split_nsm_outer_loss instantiates SIGReg without a Flax scope.
        # Instead, we reuse self.z_heads._sigreg which is already in scope.
        if training and len(seg_z_codas) > 1 and len(seg_z_memories) > 0:
            rng_z = rng if rng is not None else jax.random.PRNGKey(0)
            n_pairs = min(len(seg_z_memories), len(seg_z_codas) - 1)
            zero = jnp.zeros(())
            bb_losses = []
            mem_losses = []
            for i in range(n_pairs):
                if i < len(seg_bb_preds):
                    bb_pred = seg_bb_preds[i]
                    bb_target = seg_z_codas[i + 1].mean(axis=1, keepdims=True)
                    bb_pred_mean = bb_pred.mean(axis=1, keepdims=True)
                    bb_losses.append(_smooth_l1(bb_pred_mean, bb_target))
                if i + 1 < len(seg_z_preludes):
                    mem_mean = seg_z_memories[i].mean(axis=1, keepdims=True)
                    prelude_target = seg_z_preludes[i + 1].mean(axis=1, keepdims=True)
                    mem_losses.append(_smooth_l1(mem_mean, prelude_target))

            # SIGReg via bound module (avoids out-of-scope instantiation)
            rng_z, rng_sr = jax.random.split(rng_z)
            sigreg_val = self.z_heads.sigreg_loss(
                seg_z_codas[1].astype(jnp.float32), rng_sr
            )
            bb_loss  = jnp.stack(bb_losses).mean()  if bb_losses  else zero
            mem_loss = jnp.stack(mem_losses).mean() if mem_losses else zero
            z_fwd_loss = bb_loss + mem_loss + 0.02 * sigreg_val

            total_z_loss = total_z_loss + z_fwd_loss
            if n_loss_segs > 0:
                total_loss = total_loss + 0.1 * z_fwd_loss / n_loss_segs

        result = {"logits": jnp.concatenate(all_logits, axis=1)}
        if n_loss_segs > 0:
            result["loss"] = total_loss / n_loss_segs
            if total_mem_loss > 0:
                result["memory_loss"] = total_mem_loss / n_loss_segs
            if total_z_loss > 0:
                result["z_loss"] = total_z_loss / n_loss_segs
            if total_stp_loss > 0:
                result["stp_loss"] = total_stp_loss / n_loss_segs
        return result

    def _forward_single(
        self,
        input_ids: jnp.ndarray,
        labels: Optional[jnp.ndarray],
        training: bool,
        step: int,
        rng: Optional[jax.Array],
    ) -> dict:
        """Single-segment forward pass (core of the model)."""
        cfg = self.cfg
        B, T = input_ids.shape
        mem_scale = self._mem_scale(step)

        # ── Embedding ─────────────────────────────────────────────────────
        x = self.embed(input_ids)                         # [B, T, d_model]
        if training and cfg.dropout > 0.0 and rng is not None:
            rng, d_rng = jax.random.split(rng)
            x = nn.Dropout(rate=cfg.dropout)(x, deterministic=False)
        bigram_emb = self.embed.get_bigram(input_ids)     # [B, T, d_model]

        # ── SSM top-inject ─────────────────────────────────────────────────
        ssm_signal = self.memory.ssm_inject(x)
        x = x + mem_scale * ssm_signal

        x0 = x  # x0 skip connection (immutable in JAX)

        # ── Prelude ────────────────────────────────────────────────────────
        _mem_inject_idx = max(0, cfg.n_prelude - 1)
        memory_loss = None

        for i, layer in enumerate(self.prelude):
            if i == _mem_inject_idx:
                if training:
                    memory_loss = self.memory.update(
                        x, suppress_decay=(mem_scale == 0.0)
                    )
                x = self.memory.mag_inject(
                    x, self._mem_start, self._mem_end, scale=mem_scale
                )

            x = self._apply_x0(x, i, x0)
            x = self._apply_ve(x, i, input_ids)
            x = self.embed.bigram.inject(x, bigram_emb, i)

            x = layer(x, deterministic=not training, n_skip=0)

        x_prelude = x  # save for split_nsm

        # ── MAC tokens ─────────────────────────────────────────────────────
        # Always prepend MAC tokens. During warmup, mem_scale=0 → zero contribution.
        mac_tokens = self.memory.get_mac_tokens(x) * mem_scale  # [B, N, d_model]
        n_skip = cfg.n_memory_tokens

        x = jnp.concatenate([mac_tokens, x], axis=1)                  # [B, N+T, d_model]
        x0_ext = jnp.concatenate([jnp.zeros_like(mac_tokens), x0], axis=1)
        bigram_ext = jnp.pad(bigram_emb, ((0, 0), (n_skip, 0), (0, 0)))
        ids_ext = jnp.pad(input_ids, ((0, 0), (n_skip, 0)), constant_values=0)

        # ── Core loop ──────────────────────────────────────────────────────
        e = self.input_norm(x)   # [B, N+T, d_model]
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

        total_iters = int(cfg.max_depth)
        n_nograd = max(0, total_iters - cfg.bptt_depth)

        for t in range(total_iters):
            # active[b] = True if sample b is still iterating at step t
            active = (t < depths)[:, None, None]  # [B, 1, 1]

            # Capture loop variables explicitly to avoid closure mutation issues
            _e, _ids, _x0, _bg = e, ids_ext, x0_ext, bigram_ext
            _n_skip = n_skip
            _det = not training

            def _core_step(h_in):
                h_inj = self.injection(h_in, _e)
                for ci, clayer in enumerate(self.core):
                    gi = cfg.n_prelude + ci
                    h_inj = self._apply_x0(h_inj, gi, _x0)
                    h_inj = self._apply_ve(h_inj, gi, _ids)
                    h_inj = self.embed.bigram.inject(h_inj, _bg, gi)
                    h_inj = clayer(h_inj, deterministic=_det, n_skip=_n_skip)
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

        # ── Strip MAC tokens ───────────────────────────────────────────────
        x = x[:, n_skip:]
        x0 = x0_ext[:, n_skip:]
        bigram_emb = bigram_ext[:, n_skip:]
        input_ids_orig = ids_ext[:, n_skip:]

        # ── Coda ───────────────────────────────────────────────────────────
        for i, layer in enumerate(self.coda):
            gi = cfg.n_prelude + cfg.n_core + i
            x = self._apply_x0(x, gi, x0)
            x = self._apply_ve(x, gi, input_ids_orig)
            x = self.embed.bigram.inject(x, bigram_emb, gi)
            x = layer(x, deterministic=not training, n_skip=0)

        # ── STP loss ───────────────────────────────────────────────────────
        stp_loss = self.stp(self.final_norm(x))

        # ── Z-latent heads (ALWAYS called to ensure all params are initialized) ──
        x_coda_normed = self.final_norm(x)
        mem_ret = self.memory.retrieve(x)  # [B, T, d_model]
        # Call __call__ to guarantee all three Dense layers are init'd
        _z_rng = rng if rng is not None else jax.random.PRNGKey(0)
        z_all = self.z_heads(x_coda_normed, x_prelude, mem_ret, rng_sigreg=_z_rng)
        z_coda = z_all["z_coda"]          # [B, T, d_z]
        z_memory = z_all["z_memory"]      # [B, T, d_z]

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

            loss = ce_loss

            if memory_loss is not None:
                loss = loss + 0.05 * memory_loss
                out["memory_loss"] = memory_loss

            out["z_memory_raw"] = (z_memory * mem_scale)
            out["z_prelude"] = self.z_heads.project_prelude(self.final_norm(x_prelude))
            out["x_prelude_raw"] = x_prelude

            loss = loss + cfg.stp_lambda * stp_loss
            out["stp_loss"] = stp_loss
            out["loss"] = loss

        out["z_target"] = z_coda
        return out
