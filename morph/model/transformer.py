"""MORPH Transformer — Parcae-style looped architecture with all features baked in.

Architecture: prelude → core×T (diagonal injection) → coda
Loop hierarchy:
  Inner: Parcae core loop (T iterations with Poisson depth sampling)
  Outer: (Zyphra RSA — deferred, inference-time, requires RL)

All features always on. No runtime if-statements in the forward pass.
Config determines dimensions and sizes, not whether features exist.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
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

# Env-guarded profiler regions for carrier-copy attribution (default OFF → nullcontext,
# zero production cost). Set MORPH_PROFILE_REGIONS=1 to name forward carrier sites so the
# profiler attributes copy_/add/gather kernels to them (with_stack is blind to compiled +
# backward kernels; record_function is not). Used by ignore/profile_copy_stack.py.
_PROFILE_REGIONS = os.environ.get("MORPH_PROFILE_REGIONS", "0") == "1"
if _PROFILE_REGIONS:
    from torch.profiler import record_function as _record_function

    def _prof(name):
        return _record_function(name)
else:
    _NULLCTX = nullcontext()  # reentrant-safe singleton → zero alloc on the hot path

    def _prof(name):
        return _NULLCTX


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
    # Selective activation checkpointing of the core-loop grad-iterations (throughput knob).
    # The bptt_depth grad-iterations are checkpointed (recomputed in backward) to save activation
    # memory. ckpt_grad_iters = how many of them (counting from the FIRST grad iter) to checkpoint;
    # the remaining (LAST) grad-iterations run eager (activations retained → no recompute → faster).
    # Un-checkpointing the LAST iters first is the efficient frontier: active-set shrinking makes
    # them the smallest (least memory to retain) while still eliminating a recompute.
    # -1 → checkpoint ALL grad-iterations (default; BIT-IDENTICAL to pre-knob behaviour).
    # Checkpointing is mathematically exact, so this NEVER changes the gradient (ppl-neutral) —
    # it only trades activation memory for recompute. Tune against VRAM headroom.
    ckpt_grad_iters: int = -1

    # Cross-iteration KV sharing (CLA) — see docs/cross_layer_kv_sharing.md.
    # When on, the per-core-layer KV bundle (k_lat, k, v, C_comp, +CSA top_idx/
    # invalid_mask) is computed once at cla_share_start and reused (q-only recompute)
    # in later loop iterations. cla_share_start=-1 → auto = n_nograd (first grad
    # iteration), so K/V projections receive gradient (truncated-BPTT trap).
    use_cla: bool = False
    cla_share_start: int = -1

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

    # Master kernel switch. True = fused Triton attention + fused chunked CE
    # (the optimised stack). False = eager PyTorch references + full-logits CE
    # (the un-optimised baseline) — same architecture/weights, for A/B on memory
    # and throughput. The bit-exact loop opts (x0-hoist, active-set) stay on in
    # BOTH arms (they are not "kernels" and have no downside).
    use_kernels: bool = True
    # MRR ablation: True = MultiRateResidual (per-channel gain); False = plain residual
    # (StandardResidual). Resolved at construction → branch-free. False removes ~16 ms/step
    # of channel slice+cat copies + the dead alpha params; ablate quality vs the MRR baseline.
    use_mrr: bool = True

    # Residual mechanism (supersedes use_mrr when set to a non-null value):
    #   "mrr" | "standard"   — single-stream [B,S,C] carrier (use_mrr True/False).
    #   "hc_cayley"          — REAL Hyper-Connection, orthogonal stream mixer (JPmHC, Cayley).
    #   "hc_sinkhorn"        — REAL Hyper-Connection, doubly-stochastic mixer (mHC, Sinkhorn).
    # The hc_* modes widen the residual stream to n=hc_streams parallel C-dim streams
    # ([B,S,n,C]) across the WHOLE network (expand after embeddings, mean-reduce before the
    # LM head). The depth-composite ∏H^res is norm-preserving (orthogonal: exact dynamical
    # isometry; doubly-stochastic: spectral norm ≤1) — stabilises the deep weight-tied loop.
    residual_mode: str | None = None
    hc_streams: int = 4          # expansion rate n (paper default 4); n=1 ≡ plain residual
    hc_tau: float = 1.0          # softmax temperature for Hpre/Hpost
    hc_cayley_iters: int = 3     # Cayley fixed-point steps (s); s=2 paper, 3 = safety margin
    hc_cayley_alpha: float = 0.1 # Cayley step size α
    hc_sinkhorn_iters: int = 20  # Sinkhorn iterations (mHC value)
    hc_init_gain: float = 0.1    # W_fused init std = gain/sqrt(n*d) → ≈ plain residual at init
    hc_use_kernel: bool = True   # fused Triton HC kernels (cayley+cuda). False ⇒ eager refs
                                 # (bit-faithful, slower) — for the fused-vs-eager A/B reference arm.

    # Carrier-engine: fold the per-core-layer injection (_apply_injection broadcast-add)
    # into the PREVIOUS layer's fused HC post write (out[i] += term_next), killing 5/6 of
    # the per-iteration carrier read+writes. HC-only; default off (bit-identical baseline).
    carrier_engine: bool = False

    # L2 residency: mark the active carrier's address range PERSISTING (cudaAccessPolicyWindow)
    # so it survives the sublayer GEMMs' streaming between HC ops. Numerically a no-op (caching
    # hint); cc8.0+. Default off. (Mechanism proven -19.6% isolated; model benefit measured.)
    l2_persist: bool = False

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

        # ── Residual mechanism (single-stream MRR/standard vs n-stream Hyper-Connection) ──
        residual_mode = cfg.residual_mode or ("mrr" if cfg.use_mrr else "standard")
        self._residual_mode = residual_mode
        self._is_hc = residual_mode in ("hc_cayley", "hc_sinkhorn")
        self._n_streams = cfg.hc_streams if self._is_hc else 1
        hc_kwargs = dict(
            n_streams=cfg.hc_streams, tau=cfg.hc_tau,
            cayley_iters=cfg.hc_cayley_iters, cayley_alpha=cfg.hc_cayley_alpha,
            sinkhorn_iters=cfg.hc_sinkhorn_iters, init_gain=cfg.hc_init_gain,
            use_kernel=cfg.hc_use_kernel,
        ) if self._is_hc else None

        def _make_block(layer_idx: int, use_block_ell: bool = False) -> MORPHBlock:
            return MORPHBlock(
                norm_attn=RMSNorm(d),
                attn=MORPHAttention(layer_idx=layer_idx, **attn_kw),
                norm_mlp=RMSNorm(d),
                mlp=_make_swiglu(d, d_ff, cfg.dropout, use_block_ell=use_block_ell),
                channel_dims=ch,
                use_mrr=cfg.use_mrr,
                residual_mode=residual_mode,
                d_model=d,
                hc_kwargs=hc_kwargs,
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

        # Master kernel switch → drives the fused-Triton-vs-eager-reference
        # dispatch in the attention kernels (process-global flag). Set at build
        # so the choice is captured in the run; the fused-CE branch in forward()
        # reads self.cfg.use_kernels directly.
        from morph.kernels.triton._eager_flag import set_force_eager
        set_force_eager(not cfg.use_kernels)

        n_params = sum(p.numel() for p in self.parameters())
        _res = self._residual_mode + (f"(n={self._n_streams})" if self._is_hc else "")
        print(f"MORPHTransformer: {n_params/1e6:.1f}M params, "
              f"loop {cfg.n_prelude}:{cfg.n_core}×{cfg.mean_depth}:{cfg.n_coda} "
              f"(kernels={'fused' if cfg.use_kernels else 'EAGER'}, residual={_res})")

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

    # ── Merged injection (HC perf) ────────────────────────────────────────
    # x0, value-embed and bigram are all *additive* signals (x0/ve into the ctx
    # channel slice, bigram full-width), so by commutativity their sum applied
    # in one pass equals the old sequential x0→ve→bigram chain (bit-exact to the
    # bf16 floor). The old chain did 2-3 slice+cat passes over the FULL [B,S,n,C]
    # Hyper-Connection carrier per layer; this assembles ONE full-width term in
    # cheap single-stream [B,S,C] space (the only cat lands on that small tensor,
    # not the 4x carrier) and broadcast-adds it into the carrier exactly once.
    def _build_injection_term(self, layer_idx: int, x0_term: Tensor,
                              input_ids: Tensor, bigram_emb: Tensor,
                              dtype: torch.dtype) -> Tensor:
        """Combined single-stream additive injection [B,S,C] for `layer_idx`.

        ``lam*bigram`` (full width) + (x0_term + ve_term) placed in the ctx slice.
        ``x0_term`` is the pre-projected/scaled x0 signal (``ChannelInject.precompute``).
        """
        cs, ce = self._ctx_start, self._ctx_end
        lam = self.embed.bigram.lambdas[layer_idx].to(dtype)
        full = lam * bigram_emb.to(dtype)                      # [B,S,C] full-width bigram
        ctx = x0_term.to(dtype)                                # [B,S,ctx_w] x0 contribution
        if layer_idx in self._ve_layer_map:
            ve_idx = self._ve_layer_map.index(layer_idx)
            signal = self.value_embed_tables[ve_idx](input_ids)
            ctx = ctx + self.value_embeds[ve_idx].precompute(signal).to(dtype)
        # Drop x0(+ve) into the ctx slice — cat on the small single-stream term only.
        return torch.cat([full[..., :cs], full[..., cs:ce] + ctx, full[..., ce:]], dim=-1)

    @staticmethod
    def _apply_injection(h: Tensor, term: Tensor) -> Tensor:
        """Broadcast-add the [B,S,C] injection term into the carrier in ONE pass.

        For an HC ``[B,S,n,C]`` carrier the single-stream term is inserted on the
        stream axis so it broadcasts to every stream; for a plain ``[B,S,C]``
        carrier it adds directly.
        """
        with _prof("carrier::inject_add"):
            if term.ndim == h.ndim - 1:
                term = term.unsqueeze(-2)
            return h + term

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict:
        return self._forward_single(input_ids, labels)

    def _forward_single(self, input_ids: Tensor,
                        labels: Tensor | None = None) -> dict:
        B, T = input_ids.shape
        x = self.embed_drop(self.embed(input_ids))
        bigram_emb = self.embed.get_bigram(input_ids)

        x0 = x.clone()      # single-stream skip signal (broadcast into HC streams)

        # ── Hyper-Connection stream expansion ─────────────────────────
        # Widen the residual carrier to n parallel C-dim streams for the whole network.
        # All streams start equal, so with the ≈identity HC init the network reduces to a
        # plain residual at step 0 (verified). Injections (x0/ve/bigram/diagonal) are
        # single-stream signals that broadcast into every stream (ndim-adaptive modules).
        if self._is_hc:
            with _prof("carrier::expand_contig"):
                x = x.unsqueeze(2).expand(B, T, self._n_streams, x.shape[-1]).contiguous()

        # ── Prelude ───────────────────────────────────────────────────
        for i, layer in enumerate(self.prelude):
            term = self._build_injection_term(
                i, self.x0_injects[i].precompute(x0), input_ids, bigram_emb, x.dtype
            )
            x = self._apply_injection(x, term)
            x = layer(x)

        # ── Core loop ─────────────────────────────────────────────────
        e = self.input_norm(x)
        with _prof("carrier::h_clone"):
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

        def _core_step(h_in, e_in, ids, x0_terms, bg, cla_mode=None, kv_bundles=None):
            # cla_mode: None=normal, "compute"=stash per-layer KV bundle (returns
            # (h, captures)), "reuse"=q-only recompute consuming kv_bundles[i].
            h_injected = self.injection(h_in, e_in)
            captures = [] if cla_mode == "compute" else None
            engine = self.cfg.carrier_engine
            n_c = len(self.core)
            if engine:
                # Carrier-engine: fold each per-layer inject (carrier-independent term) into
                # the PREVIOUS block's MLP POST write. term_0 has no previous post (it follows
                # the diagonal injection) → apply it plainly; terms[1..n-1] ride out on
                # blocks[0..n-2]'s fused post (5/6 injects fused). Terms are built JUST-IN-TIME
                # (one-ahead) NOT all-upfront — holding all n terms alive cost +221 MB peak at
                # deploy shape (failed the mem iron gate); one-at-a-time is memory-neutral.
                h_injected = self._apply_injection(
                    h_injected,
                    self._build_injection_term(np_, x0_terms[0], ids, bg, h_injected.dtype),
                )
            for i, layer in enumerate(self.core):
                gi = np_ + i
                if engine:
                    nxt = (
                        self._build_injection_term(
                            np_ + i + 1, x0_terms[i + 1], ids, bg, h_injected.dtype
                        )
                        if (i + 1) < n_c else None
                    )
                else:
                    term = self._build_injection_term(
                        gi, x0_terms[i], ids, bg, h_injected.dtype
                    )
                    h_injected = self._apply_injection(h_injected, term)
                    nxt = None
                if cla_mode == "compute":
                    cap: dict = {}
                    h_injected = layer(h_injected, attn_kwargs={"cla_capture": cap},
                                       next_inject_term=nxt)
                    captures.append(cap)
                elif cla_mode == "reuse":
                    h_injected = layer(h_injected, attn_kwargs={"cla_kv": kv_bundles[i]},
                                       next_inject_term=nxt)
                else:
                    h_injected = layer(h_injected, next_inject_term=nxt)
            if cla_mode == "compute":
                return h_injected, captures
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
        with _prof("carrier::perm_gather"):
            h_s = h[perm]
            e_s = e[perm]
            ids_s = input_ids[perm]
            bg_s = bigram_emb[perm]
            x0_s = x0_core_terms[:, perm]            # [n_core, B, S, W]

        # ── CLA (cross-iteration KV sharing) scheduling ──────────────────────
        # Compute the per-core-layer KV bundle once at cla_start (the first grad
        # iteration, so K/V projections receive gradient), then reuse it (q-only
        # recompute) for later iterations. cla_start defaults to n_nograd.
        cla_on = bool(self.cfg.use_cla)
        cla_start = self.cfg.cla_share_start
        if cla_start < 0:
            cla_start = n_nograd
        # v1 shares only WITHIN the grad window (single-epoch): the compute-iteration
        # must be a grad iteration so K/V projections receive gradient (trap §7.2).
        # A smaller value would be shadowed by the `t < n_nograd` no-grad branch and
        # silently disable CLA — clamp it up so that can't happen.
        cla_start = max(cla_start, n_nograd)
        kv_bundles = None

        # Selective checkpointing: checkpoint the first `n_ckpt` grad-iterations, run the rest
        # (the last grad-iters) eager (activations retained → no backward recompute). -1 → all.
        # Exact: changes memory/recompute only, never the gradient.
        n_grad_iters = max(0, total_iters - n_nograd)
        _ck = self.cfg.ckpt_grad_iters
        n_ckpt = n_grad_iters if _ck < 0 else max(0, min(_ck, n_grad_iters))

        # Precompute every iteration's active-set count in ONE host transfer. The old
        # per-iteration `(sort_depths > t).sum().item()` forced a GPU->CPU sync EACH of
        # the up-to-max_depth iterations, draining the launch queue mid-loop (the model
        # is ~87% compute-bound but under launch pressure — perf pass OPT1). sort_depths
        # is sorted descending, so this is one [B, total_iters] compare reduced to a
        # per-t count, materialised once. Exact: identical counts, identical control flow.
        _t_range = torch.arange(total_iters, device=sort_depths.device)
        active_counts = (sort_depths.unsqueeze(1) > _t_range.unsqueeze(0)).sum(0).tolist()

        use_l2 = self.cfg.l2_persist
        if use_l2:
            from morph.kernels import l2_persist as _l2

        for t in range(total_iters):
            n_active = active_counts[t]
            if n_active == 0:
                break
            h_a = h_s[:n_active]
            if use_l2:
                _l2.set_carrier(h_a)   # mark the active carrier L2-persisting (no-op math)
            args = (h_a, e_s[:n_active], ids_s[:n_active],
                    x0_s[:, :n_active], bg_s[:n_active])

            do_compute = cla_on and t == cla_start
            do_reuse = cla_on and t > cla_start and kv_bundles is not None
            # Checkpoint this grad-iteration? Only in training, and only the first n_ckpt grad
            # iters (later ones run eager → no recompute). n_nograd iters are frozen (no_grad).
            do_ckpt = self.training and (t - n_nograd) < n_ckpt

            if t < n_nograd:
                # no-grad region; cla_start >= n_nograd so no CLA here.
                with torch.no_grad():
                    h_new = _core_step(*args)
            elif do_compute:
                # NOT checkpointed: retain activations so the captured KV bundle
                # carries gradient to W_down_k / compressor / indexer (BPTT trap §7.2).
                h_new, kv_bundles = _core_step(*args, cla_mode="compute")
            elif do_reuse:
                # slice the cached bundle to the current active-set prefix (trap §7.1).
                kvb = [{k: (v[:n_active] if torch.is_tensor(v) else v)
                        for k, v in b.items()} for b in kv_bundles]
                if do_ckpt:
                    h_new = checkpoint(_core_step, *args, cla_mode="reuse",
                                       kv_bundles=kvb, use_reentrant=False)
                else:
                    h_new = _core_step(*args, cla_mode="reuse", kv_bundles=kvb)
            elif do_ckpt:
                h_new = checkpoint(_core_step, *args, use_reentrant=False)
            else:
                # eval, OR a grad-iter we chose not to checkpoint (activations retained).
                h_new = _core_step(*args)

            # updated active prefix + frozen suffix (no in-place op).
            with _prof("carrier::loop_cat"):
                h_s = h_new if n_active == h_s.shape[0] else \
                    torch.cat([h_new, h_s[n_active:]], dim=0)

        if use_l2:
            _l2.reset()                          # clear the persisting window for the rest of the step

        with _prof("carrier::inv_perm_gather"):
            x = h_s[inv_perm]                    # restore original batch order

        # ── Coda ──────────────────────────────────────────────────────
        for i, layer in enumerate(self.coda):
            gi = self.cfg.n_prelude + self.cfg.n_core + i
            term = self._build_injection_term(
                gi, self.x0_injects[gi].precompute(x0), input_ids, bigram_emb, x.dtype
            )
            x = self._apply_injection(x, term)
            x = layer(x)

        # ── Hyper-Connection stream reduction ─────────────────────────
        # Collapse the n streams back to a single C-dim representation before the LM head.
        # Mean readout is scale-preserving and (with all streams equal at init) exactly
        # recovers the plain-residual output; learned asymmetry is read out as the mean.
        if self._is_hc:
            x = x.mean(dim=2)

        # ── STP ───────────────────────────────────────────────────────
        stp_loss = self.stp(self.final_norm(x).float(), tau=self.cfg.stp_tau)

        # ── LM head ──────────────────────────────────────────────────
        x = self.lm_mixer(x)
        x = self.final_norm(x)

        if labels is not None and self.cfg.use_kernels:
            # Fused chunked cross-entropy whenever we have labels (TRAINING **and**
            # EVAL). Never materialises the [B, T, V] logits — the dominant
            # activation-memory cost — nor the [B·T, V] fp32 log_softmax intermediate
            # that F.cross_entropy builds (~6 GiB at B=8/T=4096/V=49152). Eval only
            # needs the loss scalar, so the old `self.training` gate made eval ~6 GiB
            # heavier than training for no benefit and OOM'd the B8 arm on the
            # fragmented pool (see Ai-notes 06-01-2026). Computes loss in vocab-row
            # chunks against the tied weight; under @torch.no_grad() (eval) it runs
            # the forward only. grad (training) flows to BOTH x and the embedding
            # (via lm_weight's cat/log-map). Generation (labels=None) still takes the
            # full-logits else branch — it needs logits to sample, and is batch-1/cheap.
            w_full = self.embed.lm_weight()                       # [V, d_model]
            ce_loss = fused_linear_cross_entropy(
                x.reshape(-1, x.shape[-1]), w_full, labels.reshape(-1),
                ignore_index=-100, chunk_size=self.cfg.ce_chunk_size,
            )
            loss = ce_loss + self.cfg.stp_lambda * stp_loss
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
