"""MORPH Transformer — Parcae-style looped architecture with all features baked in.

Architecture: prelude → core×T (diagonal injection) → coda
Loop hierarchy:
  Inner: Parcae core loop (T iterations with Poisson depth sampling)
  Outer: (Zyphra RSA — deferred, inference-time, requires RL)

All features always on. No runtime if-statements in the forward pass.
Config determines dimensions and sizes, not whether features exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .attention import MORPHAttention, RMSNorm
from .embeddings import MORPHEmbedding
from .fused_ce import fused_linear_cross_entropy
from .mhc import MultiRateResidual, ChannelInject, MORPHBlock, DEFAULT_CHANNEL_DIMS
from .prediction import STPLoss
from .sparsity import BlockELLLinear


@dataclass
class MORPHConfig:
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 0  # 0 = auto (8/3 * d_model, rounded to 64)
    vocab_size: int = 49152
    max_seq_len: int = 4096

    n_prelude: int = 3
    n_core: int = 6
    n_coda: int = 3
    mean_depth: int = 6
    max_depth: int = 8
    bptt_depth: int = 4

    channel_dims: tuple[int, ...] = (384, 256, 128)

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

    # LM head — fused chunked cross-entropy (training). Rows of [B·T] tokens
    # processed per chunk; smaller = less peak memory, more launch overhead.
    # Tune per target: large on high-VRAM (Pro 6000) for speed, small on tight
    # memory / very long context.
    ce_chunk_size: int = 1024

    # Training
    dropout: float = 0.1


class DiagonalInjection(nn.Module):
    """SSM-style diagonal injection on the context channel only.

    h_ctx = decay * h_ctx + dt * e_ctx
    Spectral radius < 1 guaranteed by construction.
    """

    def __init__(self, channel_start: int, channel_end: int, init_decay: float = 0.447):
        super().__init__()
        self.start = channel_start
        self.end = channel_end
        d = channel_end - channel_start
        self.log_A = nn.Parameter(torch.full((d,), float(init_decay)).log())
        self.log_dt = nn.Parameter(torch.zeros(d))

    def forward(self, h: Tensor, e: Tensor) -> Tensor:
        A = self.log_A.exp().clamp(max=0.9999)
        dt = self.log_dt.exp()
        h_ctx = h[..., self.start:self.end]
        e_ctx = e[..., self.start:self.end]
        new_ctx = A * h_ctx + dt * e_ctx
        return torch.cat([h[..., :self.start], new_ctx, h[..., self.end:]], dim=-1)


def _make_swiglu(d_model: int, d_ff: int, dropout: float,
                 use_block_ell: bool = False) -> nn.Module:
    """SwiGLU MLP: gate + up → silu(gate)*up → down."""
    mlp: nn.Module
    if use_block_ell:
        mlp = _SwiGLUBlockELL(d_model, d_ff)
    else:
        mlp = _SwiGLU(d_model, d_ff)
    if dropout > 0:
        return nn.Sequential(mlp, nn.Dropout(dropout))
    return mlp


class _SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_up = nn.Linear(d_model, d_ff * 2, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gu = self.gate_up(x)
        gate, up = gu.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class _SwiGLUBlockELL(nn.Module):
    """SwiGLU with BlockELLLinear for CMS pruning support.

    Identical computation to _SwiGLU during dense phase (density=1.0).
    After compact(), uses Triton Block-ELL sparse kernels for the forward pass.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_up = BlockELLLinear(d_model, d_ff * 2, bias=False, initial_density=1.0)
        self.down = BlockELLLinear(d_ff, d_model, bias=False, initial_density=1.0)

    def forward(self, x: Tensor) -> Tensor:
        gu = self.gate_up(x)
        gate, up = gu.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class LMHeadMixer(nn.Module):
    """3-channel mixer before LM head: learned per-channel scale + cross-channel linear."""

    def __init__(self, d_model: int, channel_dims: tuple[int, ...] = (384, 256, 128)):
        super().__init__()
        self.channel_dims = channel_dims
        self.channel_scales = nn.Parameter(torch.ones(len(channel_dims)))
        self.mix = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.mix.weight)

    def forward(self, x: Tensor) -> Tensor:
        scales = F.softplus(self.channel_scales)
        chunks = x.split(list(self.channel_dims), dim=-1)
        scaled = torch.cat([c * s for c, s in zip(chunks, scales)], dim=-1)
        return self.mix(scaled)


class MORPHTransformer(nn.Module):

    def __init__(self, cfg: MORPHConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        n_total = cfg.n_prelude + cfg.n_core + cfg.n_coda

        d_ff = cfg.d_ff if cfg.d_ff > 0 else ((d * 8 // 3 + 63) // 64 * 64)

        # Channel boundaries
        ch = cfg.channel_dims
        assert sum(ch) == d
        self._ch_starts = []
        self._ch_ends = []
        s = 0
        for c in ch:
            self._ch_starts.append(s)
            self._ch_ends.append(s + c)
            s += c
        self._ctx_start = self._ch_starts[1]
        self._ctx_end = self._ch_ends[1]

        # ── Embedding ─────────────────────────────────────────────────
        self.embed = MORPHEmbedding(
            vocab_size=cfg.vocab_size,
            d_model=d,
            lorentz_fraction=cfg.lorentz_fraction,
            bigram_hash_vocab=cfg.bigram_hash_vocab,
            n_layers=n_total,
        )
        self.embed_drop = nn.Dropout(cfg.dropout)

        # ── Attention kwargs (shared across all layers) ───────────────
        attn_kw = dict(
            d_model=d, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
            compression=cfg.compression, csa_compress_ratio=cfg.csa_compress_ratio,
            hca_compress_ratio=cfg.hca_compress_ratio, top_k=cfg.top_k,
            d_indexer=cfg.d_indexer,
            window_size=cfg.window_size, context_len=cfg.context_len,
            max_seq_len=cfg.max_seq_len,
            conv_kernel=cfg.conv_kernel,
            init_alpha=cfg.init_alpha,
        )

        def _make_block(layer_idx: int, use_block_ell: bool = False) -> MORPHBlock:
            return MORPHBlock(
                norm_attn=RMSNorm(d),
                attn=MORPHAttention(layer_idx=layer_idx, **attn_kw),
                norm_mlp=RMSNorm(d),
                mlp=_make_swiglu(d, d_ff, cfg.dropout, use_block_ell=use_block_ell),
                channel_dims=ch,
            )

        # ── Prelude ───────────────────────────────────────────────────
        self.prelude = nn.ModuleList([_make_block(i) for i in range(cfg.n_prelude)])

        # ── Loop state transition ─────────────────────────────────────
        self.input_norm = RMSNorm(d)
        self.injection = DiagonalInjection(self._ctx_start, self._ctx_end)

        # ── Core (shared across loop iterations — Block-ELL for CMS pruning)
        self.core = nn.ModuleList([
            _make_block(cfg.n_prelude + i, use_block_ell=True)
            for i in range(cfg.n_core)
        ])

        # ── Coda ──────────────────────────────────────────────────────
        self.coda = nn.ModuleList([
            _make_block(cfg.n_prelude + cfg.n_core + i) for i in range(cfg.n_coda)
        ])

        # ── x0 skip (inject into context channel) ────────────────────
        self.x0_injects = nn.ModuleList([
            ChannelInject(self._ctx_start, self._ctx_end, d, init_scale=0.0)
            for _ in range(n_total)
        ])

        # ── Value embeddings (inject into context channel) ────────────
        n_ve = min(3, cfg.n_prelude)
        self.value_embeds = nn.ModuleList([
            ChannelInject(self._ctx_start, self._ctx_end, d, init_scale=0.0)
            for _ in range(n_ve)
        ])
        self.value_embed_tables = nn.ModuleList([
            nn.Embedding(cfg.vocab_size, d) for _ in range(n_ve)
        ])
        for ve in self.value_embed_tables:
            nn.init.normal_(ve.weight, std=0.02)
        self._ve_layer_map = list(range(n_ve))

        # ── LM head ──────────────────────────────────────────────────
        self.lm_mixer = LMHeadMixer(d, channel_dims=ch)
        self.final_norm = RMSNorm(d)

        # ── Prediction (STP — Semantic Tube Predictor) ────────────────
        self.stp = STPLoss()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"MORPHTransformer: {n_params/1e6:.1f}M params, "
              f"loop {cfg.n_prelude}:{cfg.n_core}×{cfg.mean_depth}:{cfg.n_coda}")

    # ── Helpers ───────────────────────────────────────────────────────

    def _sample_depths(self, B: int, device: torch.device) -> Tensor:
        lam = float(self.cfg.mean_depth)
        depths = torch.poisson(torch.full((B,), lam, device=device)).long()
        return depths.clamp(min=1, max=self.cfg.max_depth)

    def _apply_x0(self, x: Tensor, layer_idx: int, x0: Tensor) -> Tensor:
        return self.x0_injects[layer_idx](x, x0)

    def _apply_ve(self, x: Tensor, layer_idx: int, input_ids: Tensor) -> Tensor:
        if layer_idx in self._ve_layer_map:
            ve_idx = self._ve_layer_map.index(layer_idx)
            signal = self.value_embed_tables[ve_idx](input_ids)
            return self.value_embeds[ve_idx](x, signal)
        return x

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict:
        return self._forward_single(input_ids, labels)

    def _forward_single(self, input_ids: Tensor,
                        labels: Tensor | None = None) -> dict:
        B, T = input_ids.shape
        x = self.embed_drop(self.embed(input_ids))
        bigram_emb = self.embed.get_bigram(input_ids)

        x0 = x.clone()

        # ── Prelude ───────────────────────────────────────────────────
        for i, layer in enumerate(self.prelude):
            x = self._apply_x0(x, i, x0)
            x = self._apply_ve(x, i, input_ids)
            x = self.embed.bigram.inject(x, bigram_emb, i)
            x = layer(x)

        # ── Core loop ─────────────────────────────────────────────────
        e = self.input_norm(x)
        h = e.clone()

        if self.training:
            depths = self._sample_depths(B, x.device)
        else:
            depths = torch.full((B,), self.cfg.mean_depth,
                                device=x.device, dtype=torch.long)

        total_iters = int(depths.max().item())
        n_nograd = max(0, total_iters - self.cfg.bptt_depth)

        # ── Hoist the loop-invariant x0 projection out of the loop ──────────
        # x0 is cloned once (constant across iterations) and each core layer's
        # ChannelInject applies scale·proj(x0). Both proj.weight and log_scale
        # are loop-invariant, so the additive term is identical every iteration.
        # Precompute it once → ~n_core × total_iters redundant [.,.,d]→[.,.,ctx]
        # matmuls collapse to n_core. Stacked as a checkpoint input so the
        # backward recompute also skips re-projecting; gradient to proj.weight
        # is the same sum-over-iterations as the per-iteration form.
        n_core = self.cfg.n_core
        np_ = self.cfg.n_prelude
        x0_core_terms = torch.stack(
            [self.x0_injects[np_ + i].precompute(x0) for i in range(n_core)],
            dim=0,
        )  # [n_core, B, S, ctx_width]

        def _core_step(h_in, e_in, ids, x0_terms, bg):
            h_injected = self.injection(h_in, e_in)
            for i, layer in enumerate(self.core):
                gi = np_ + i
                h_injected = self.x0_injects[gi].apply_precomputed(
                    h_injected, x0_terms[i]
                )
                h_injected = self._apply_ve(h_injected, gi, ids)
                h_injected = self.embed.bigram.inject(h_injected, bg, gi)
                h_injected = layer(h_injected)
            return h_injected

        # ── Active-set shrinking ────────────────────────────────────────────
        # A sample is updated only while iteration t < its Poisson depth, then
        # frozen. The old code computed the FULL batch every iteration and
        # discarded frozen samples via torch.where → ~(max_depth-mean_depth)
        # fraction of forward FLOPs wasted on already-frozen samples.
        # Instead: sort by depth descending so the still-active samples are a
        # contiguous prefix [:n_active], process only that prefix, and carry the
        # frozen suffix unchanged. Per-sample math is identical (no cross-batch
        # mixing in attn/MLP); the global per-iteration no_grad/grad/checkpoint
        # schedule is preserved, so gradients match the truncated-BPTT window.
        sort_depths, perm = torch.sort(depths, descending=True)
        inv_perm = torch.argsort(perm)
        h_s = h[perm]
        e_s = e[perm]
        ids_s = input_ids[perm]
        bg_s = bigram_emb[perm]
        x0_s = x0_core_terms[:, perm]            # [n_core, B, S, W]

        for t in range(total_iters):
            n_active = int((sort_depths > t).sum().item())
            if n_active == 0:
                break
            h_a = h_s[:n_active]
            args = (h_a, e_s[:n_active], ids_s[:n_active],
                    x0_s[:, :n_active], bg_s[:n_active])

            if t < n_nograd:
                with torch.no_grad():
                    h_new = _core_step(*args)
            elif self.training:
                h_new = checkpoint(_core_step, *args, use_reentrant=False)
            else:
                h_new = _core_step(*args)

            # updated active prefix + frozen suffix (no in-place op).
            h_s = h_new if n_active == h_s.shape[0] else \
                torch.cat([h_new, h_s[n_active:]], dim=0)

        x = h_s[inv_perm]                        # restore original batch order

        # ── Coda ──────────────────────────────────────────────────────
        for i, layer in enumerate(self.coda):
            gi = self.cfg.n_prelude + self.cfg.n_core + i
            x = self._apply_x0(x, gi, x0)
            x = self._apply_ve(x, gi, input_ids)
            x = self.embed.bigram.inject(x, bigram_emb, gi)
            x = layer(x)

        # ── STP ───────────────────────────────────────────────────────
        stp_loss = self.stp(self.final_norm(x).float(), tau=self.cfg.stp_tau)

        # ── LM head ──────────────────────────────────────────────────
        x = self.lm_mixer(x)
        x = self.final_norm(x)

        if labels is not None and self.training:
            # Training: fused chunked cross-entropy. Never materialises the
            # [B, T, V] logits (the dominant activation-memory cost at scale) —
            # computes loss + grads in vocab-row chunks against the tied weight.
            # grad flows to BOTH x and the embedding (via lm_weight's cat/log-map).
            w_full = self.embed.lm_weight()                       # [V, d_model]
            ce_loss = fused_linear_cross_entropy(
                x.reshape(-1, x.shape[-1]), w_full, labels.reshape(-1),
                ignore_index=-100, chunk_size=self.cfg.ce_chunk_size,
            )
            loss = ce_loss + self.cfg.stp_lambda * stp_loss
            # logits intentionally not materialised in training (unused by the
            # loss / loggers). Generation + eval take the full-logits branch.
            out = {"logits": None, "stp_loss": stp_loss, "loss": loss}
        else:
            logits = self.embed.attend(x)
            out = {"logits": logits}
            if labels is not None:
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, self.cfg.vocab_size),
                    labels.reshape(-1), ignore_index=-100,
                )
                loss = ce_loss + self.cfg.stp_lambda * stp_loss
                out["stp_loss"] = stp_loss
                out["loss"] = loss

        return out
