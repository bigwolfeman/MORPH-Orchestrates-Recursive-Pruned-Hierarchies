"""MORPH Transformer — Parcae-style looped architecture with all features baked in.

Architecture: prelude → core×T (diagonal injection) → coda
Three-loop hierarchy:
  Inner:  Parcae core loop (T iterations with Poisson depth sampling)
  Middle: Neural memory SSM (gradient-based surprise update on forward pass)
  Outer:  (RSA — deferred, inference-time)

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
from .mhc import MultiRateResidual, ChannelInject, MORPHBlock, DEFAULT_CHANNEL_DIMS
from .memory import MemorySystem
from .prediction import STPLoss, ZLatentHeads, SIGReg
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
        self._mem_start = self._ch_starts[2]
        self._mem_end = self._ch_ends[2]

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
            window_size=cfg.window_size, context_len=cfg.context_len,
            max_seq_len=cfg.max_seq_len,
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

        # ── Neural memory ─────────────────────────────────────────────
        self.memory = MemorySystem(
            d_model=d,
            n_layers=cfg.n_memory_layers,
            n_memory_tokens=cfg.n_memory_tokens,
            d_memory_channel=ch[2],
        )
        self._mac_warmup = cfg.mac_warmup_steps
        self._step = 0

        # ── LM head ──────────────────────────────────────────────────
        self.lm_mixer = LMHeadMixer(d, channel_dims=ch)
        self.final_norm = RMSNorm(d)

        # ── Prediction (STP + z-latent) ───────────────────────────────
        self.stp = STPLoss()
        self.z_heads = ZLatentHeads(d, cfg.d_z)
        self.sigreg = SIGReg()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"MORPHTransformer: {n_params/1e6:.1f}M params, "
              f"loop {cfg.n_prelude}:{cfg.n_core}×{cfg.mean_depth}:{cfg.n_coda}")

    # ── Helpers ───────────────────────────────────────────────────────

    def _mem_active(self) -> bool:
        return self._step > self._mac_warmup

    def _mem_scale(self) -> float:
        if not self._mem_active():
            return 0.0
        ramp = 500
        return min(1.0, (self._step - self._mac_warmup) / ramp)

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
        if self.training:
            self._step += 1
        return self._forward_single(input_ids, labels)

    def _forward_single(self, input_ids: Tensor,
                        labels: Tensor | None = None) -> dict:
        B, T = input_ids.shape
        x = self.embed_drop(self.embed(input_ids))
        bigram_emb = self.embed.get_bigram(input_ids)

        mem_scale = self._mem_scale()  # 0.0 during warmup, ramps to 1.0
        memory_loss = None

        # ── MAC tokens (append — always-visible suffix through ALL layers)
        # Generated from embedded input, persist through prelude+core+coda.
        # Real tokens see MAC tokens via always-visible mask in attention.
        # MAG handles memory→tokens injection; MAC handles bidirectional context.
        mac_tokens = self.memory.get_mac_tokens(x) * mem_scale
        x = torch.cat([x, mac_tokens], dim=1)
        bigram_emb = F.pad(bigram_emb, (0, 0, 0, self.cfg.n_memory_tokens))
        input_ids = F.pad(input_ids, (0, self.cfg.n_memory_tokens), value=0)
        n_mac = self.cfg.n_memory_tokens

        x0 = x.clone()

        # ── Prelude ───────────────────────────────────────────────────
        _mem_inject_idx = max(0, self.cfg.n_prelude - 1)
        for i, layer in enumerate(self.prelude):
            if i == _mem_inject_idx:
                if self.training:
                    memory_loss = self.memory.update(
                        x[:, :T], suppress_decay=(mem_scale == 0.0))
                x = self.memory.mag_inject(
                    x, self._mem_start, self._mem_end, mem_scale)

            x = self._apply_x0(x, i, x0)
            x = self._apply_ve(x, i, input_ids)
            x = self.embed.bigram.inject(x, bigram_emb, i)
            x = layer(x, attn_kwargs={"n_skip_rope": n_mac})

        x_prelude = x.clone()

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

        for t in range(total_iters):
            active = (t < depths).unsqueeze(-1).unsqueeze(-1)

            def _core_step(h_in, e_in, ids, x0_in, bg):
                h_injected = self.injection(h_in, e_in)
                for i, layer in enumerate(self.core):
                    gi = self.cfg.n_prelude + i
                    h_injected = self._apply_x0(h_injected, gi, x0_in)
                    h_injected = self._apply_ve(h_injected, gi, ids)
                    h_injected = self.embed.bigram.inject(h_injected, bg, gi)
                    h_injected = layer(h_injected, attn_kwargs={"n_skip_rope": n_mac})
                return h_injected

            if t < n_nograd:
                with torch.no_grad():
                    h_new = _core_step(h, e, input_ids, x0, bigram_emb)
            elif self.training:
                h_new = checkpoint(
                    _core_step, h, e, input_ids, x0, bigram_emb,
                    use_reentrant=False,
                )
            else:
                h_new = _core_step(h, e, input_ids, x0, bigram_emb)

            h = torch.where(active, h_new, h)

        x = h

        # ── Coda (MAC tokens persist — memory context through all layers)
        for i, layer in enumerate(self.coda):
            gi = self.cfg.n_prelude + self.cfg.n_core + i
            x = self._apply_x0(x, gi, x0)
            x = self._apply_ve(x, gi, input_ids)
            x = self.embed.bigram.inject(x, bigram_emb, gi)
            x = layer(x, attn_kwargs={"n_skip_rope": n_mac})

        # ── Strip MAC tokens (appended suffix → slice to original T)
        x = x[:, :T]
        x0 = x0[:, :T]
        bigram_emb = bigram_emb[:, :T]
        input_ids = input_ids[:, :T]

        # ── STP ───────────────────────────────────────────────────────
        stp_loss = self.stp(self.final_norm(x).float(), tau=self.cfg.stp_tau)

        # ── LM head ──────────────────────────────────────────────────
        x_coda = x
        x = self.lm_mixer(x)
        x = self.final_norm(x)
        logits = self.embed.attend(x)

        out = {"logits": logits}

        if labels is not None:
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                labels.reshape(-1), ignore_index=-100,
            )
            loss = ce_loss

            if memory_loss is not None:
                loss = loss + 0.05 * memory_loss
                out["memory_loss"] = memory_loss.detach()

            # Z-latent: single-pass window-based prediction
            seg = self.cfg.segment_size
            if seg is not None and T >= seg * 2:
                z_loss = self._window_z_loss(x_coda, x_prelude, mem_scale, seg, T)
                loss = loss + 0.1 * z_loss
                out["z_loss"] = z_loss.detach()

            loss = loss + self.cfg.stp_lambda * stp_loss
            out["stp_loss"] = stp_loss

            out["loss"] = loss

        return out

    def _window_z_loss(self, x_coda: Tensor, x_prelude: Tensor,
                       mem_scale: float, seg: int, T: int) -> Tensor:
        """Single-pass z-latent loss from representation windows.

        No repeated forward passes. Slices the single-pass output at segment
        boundaries and computes cross-window prediction losses:
          - Backbone: prelude window[i] predicts mean(coda z[i+1])
          - Memory: retrieve from window[i] predicts prelude z[i+1]
        """
        n_segs = T // seg
        losses: list[Tensor] = []
        all_z_codas: list[Tensor] = []

        for s in range(n_segs):
            start, end = s * seg, (s + 1) * seg
            z_coda_w = self.z_heads.project_coda(
                self.final_norm(x_coda[:, start:end]))
            all_z_codas.append(z_coda_w.detach())

            if s > 0:
                prev_start, prev_end = (s - 1) * seg, s * seg

                # Backbone: prelude of window[s-1] predicts mean z_coda of window[s]
                bb_pred = self.z_heads.backbone_predict(x_prelude[:, prev_start:prev_end])
                bb_target = z_coda_w.mean(dim=1, keepdim=True).detach()
                losses.append(F.smooth_l1_loss(bb_pred.mean(dim=1, keepdim=True), bb_target))

                # Memory: retrieve from window[s-1] predicts prelude z of window[s]
                mem_ret = self.memory.retrieve(x_coda[:, prev_start:prev_end])
                z_mem = self.z_heads.memory_predict(mem_ret) * mem_scale
                z_prel_target = self.z_heads.project_prelude(
                    self.final_norm(x_prelude[:, start:end])
                ).mean(dim=1, keepdim=True).detach()
                losses.append(F.smooth_l1_loss(z_mem.mean(dim=1, keepdim=True), z_prel_target))

        if not losses:
            return x_coda.new_zeros(())

        z_loss = torch.stack(losses).mean()

        # SIGReg collapse prevention
        all_z = torch.cat(all_z_codas, dim=1)
        z_loss = z_loss + 0.02 * self.sigreg(all_z.reshape(-1, all_z.shape[-1]).unsqueeze(0))

        return z_loss
