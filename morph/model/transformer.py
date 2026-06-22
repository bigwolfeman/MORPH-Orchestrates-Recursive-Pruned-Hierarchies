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
from .fused_ce import (
    fused_linear_cross_entropy,
    fused_linear_cross_entropy_mce,
    multi_hot_cross_entropy_reference,
)
from .mhc import ChannelInject, MORPHBlock, DEFAULT_CHANNEL_DIMS
from .prediction import STPLoss
from .sparsity import MortarLinear

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
    # Loop-axis STP regularizer: same geodesic-smoothness geometry as STP, but
    # applied along the LOOP-ITERATION trajectory h_0→…→h_T instead of the token axis. Pushes the
    # inner map f_θ toward locally-affine behaviour → reduces the curvature that makes the AdEMAMix
    # slow-EMA go stale (the σ_max(J_core) inter-block-alignment runaway). 0.0 → OFF, bit-exact.
    loop_stp_lambda: float = 0.0

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

    # Residual = Hyper-Connection (JPmHC, Cayley): widens the residual stream to n=hc_streams
    # parallel C-dim streams ([B,S,n,C]) across the whole network (expand after embeddings,
    # mean-reduce before the LM head). The orthogonal Cayley mixer makes the depth-composite
    # ∏H^res norm-preserving (exact dynamical isometry) — stabilises the deep weight-tied loop.
    hc_streams: int = 4          # expansion rate n (paper default 4); n=1 ≡ plain residual
    hc_tau: float = 1.0          # softmax temperature for Hpre/Hpost
    hc_cayley_iters: int = 3     # Cayley fixed-point steps (s); s=2 paper, 3 = safety margin
    hc_cayley_alpha: float = 0.1 # Cayley step size α
    hc_init_gain: float = 0.1    # W_fused init std = gain/sqrt(n*d) → ≈ plain residual at init
    hc_use_kernel: bool = True   # fused Triton HC kernels (cayley+cuda). False ⇒ eager refs
                                 # (bit-faithful, slower) — for the fused-vs-eager A/B reference arm.

    # L2 residency: mark the active carrier's address range PERSISTING (cudaAccessPolicyWindow)
    # so it survives the sublayer GEMMs' streaming between HC ops. Numerically a no-op (caching
    # hint); cc8.0+. Default off. (Mechanism isolated -19.6%; model benefit measured net-
    # negative in-model; kept as a dormant knob.)
    l2_persist: bool = False

    # ── Retention branch (#230) ────────────────────────────────────────────
    # Gated Linear Attention (GLA) added in PARALLEL to the windowed attention in the
    # 2nd layer (index in retention_layers) of prelude / core / coda — a global-context /
    # cross-iteration memory branch. Off by default → GLA modules are NOT constructed, so
    # the model is bit-identical to the baseline (flag gates construction, not just the
    # forward branch — keeps init RNG draw identical).
    retention: bool = True
    retention_layers: tuple[int, ...] = (1,)   # which layer index per section gets the branch
    retention_heads: int = 0                   # 0 → use n_heads
    retention_chunk: int = 128
    retention_gate_init: float = -6.0          # branch-gate logit; sigmoid(-6)≈0.0025 ≈ identity@init
    retention_carry: bool = True               # core: carry GLA state across loop iterations
                                               # (False → reset each iter = global retention, no memory)
    retention_gate_bias: float = 2.0           # GLA internal forget-gate logit bias (α near 1 = long memory)

    # Training
    dropout: float = 0.1

    # L1 core-gain governor: cap the per-iteration looped-core
    # amplification ‖h_new‖/‖h_a‖ (per sample) to this ratio τ. The HC residual is
    # norm-preserving (gain≈1 healthy) so this is IDENTITY in the healthy regime and only
    # shrinks the runaway-gain step that the weight-shared core amplifies T× (the β1=0
    # gain runaway mode). 0.0 = OFF (bit-identical to baseline). Typical τ≈1.5–2.0.
    core_gain_clip: float = 0.0


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


class _KwargSequential(nn.Sequential):
    """nn.Sequential that forwards ``**kwargs`` to the FIRST submodule (the MLP) and runs
    the remaining modules (e.g. Dropout) positionally.

    The core loop passes ``mlp_kwargs={"iter_idx": t}`` to each block's MLP so the Phase-C
    ReMoE router knows which loop iteration it is. A plain ``nn.Sequential`` rejects kwargs
    (``Sequential.forward()`` takes only ``input``), which silently broke any forward once
    iteration-threading was added. Subclassing keeps the child registration identical to
    ``nn.Sequential`` (indices ``"0"``/``"1"``) so state_dicts stay byte-compatible with
    checkpoints saved before this class existed. ``enable_routing`` / ``d_ff`` delegate to
    the inner MLP so the router attaches and stats read through the Dropout wrapper.
    """

    def forward(self, x, **kwargs):
        it = iter(self)
        x = next(it)(x, **kwargs)   # inner MLP receives iter_idx (and any future kwargs)
        for m in it:
            x = m(x)                # Dropout etc. — positional only
        return x

    def enable_routing(self, *args, **kwargs):
        return self[0].enable_routing(*args, **kwargs)

    @property
    def router(self):
        return getattr(self[0], "router", None)

    @property
    def d_ff(self):
        return self[0].d_ff


def _make_swiglu(d_model: int, d_ff: int, dropout: float) -> nn.Module:
    """SwiGLU MLP: gate + up → silu(gate)*up → down.

    Always uses _SwiGLUMortar (CMS-prunable, MORTAR-carvable) — there is no plain
    dense fallback. Every MLP in prelude, core, and coda is MortarLinear so the
    whole backbone is prunable and carves to MORTAR BCSR at compact_step.
    """
    mlp: nn.Module = _SwiGLUMortar(d_model, d_ff)
    if dropout > 0:
        # _KwargSequential (not nn.Sequential) so iter_idx threads through to the MLP.
        return _KwargSequential(mlp, nn.Dropout(dropout))
    return mlp


class _SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_up = nn.Linear(d_model, d_ff * 2, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor, iter_idx: int = 0) -> Tensor:
        # iter_idx accepted-and-ignored: the dense SwiGLU has no router, but the core loop
        # threads iter_idx to every MLP uniformly. Keeps a plain-dense core callable.
        gu = self.gate_up(x)
        gate, up = gu.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class _SwiGLUMortar(nn.Module):
    """SwiGLU with MortarLinear for CMS pruning + MORTAR carving support.

    Identical computation to _SwiGLU during dense phase (density=1.0).
    After carve(), uses the MORTAR BCSR Triton kernel for the forward pass.

    Optionally hosts an iteration-aware ReMoE router (Phase C). The router gates the
    post-SiLU hidden h = silu(gate)·up over contiguous d_ff neuron-clusters: a clean
    PEER/MoE expert selection over the FF neuron bank (one gate per neuron, applied
    coherently — NOT gate_up's raw 2·d_ff output, which would gate the gate/up halves
    of a neuron independently). The router is None until enable_routing() is called, so
    the dense / prune / compact phases are byte-identical to the no-routing path.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_up = MortarLinear(d_model, d_ff * 2, bias=False, initial_density=1.0)
        self.down = MortarLinear(d_ff, d_model, bias=False, initial_density=1.0)
        self.d_model = d_model
        self.d_ff = d_ff
        # ReMoE routing (Phase C) — built lazily by enable_routing(). router=None → plain SwiGLU.
        self.router: nn.Module | None = None
        self._last_aux_loss: Tensor | None = None
        self._aux_detach_input = True   # detach router input → no routing-grad into the carrier

    def enable_routing(
        self,
        n_clusters: int = 16,
        activation_ratio: float = 0.5,
        aux_loss_coeff: float = 1e-2,
        n_iters: int = 1,
        n_sub_keys: int = 0,
        detach_input: bool = True,
    ) -> None:
        """Attach an iteration-aware TileRouter over the d_ff hidden neuron bank.

        Adds NEW parameters (router) → the optimizer MUST be rebuilt after calling this.
        n_iters should equal the max core-loop depth so each loop iteration gets its own
        (zero-initialized → no specialization at start) iteration embedding row.

        detach_input (default True): feed the router a detached copy of x. The router params
        still train (gradient flows to query_proj/sub_keys/group_bias/iter_embed from the
        detached input, and the gates still get gradient from the main loss), but the routing
        gradient does NOT flow back into the carrier x. In the LOOPED core this is REQUIRED for
        memory: the load-balance aux is summed over grad-iterations and each term depends on that
        iteration's carrier state x_t — letting its gradient into x_t extends the effective
        truncated-BPTT depth and retains cross-iteration activations (measured +7 GB / step at
        deploy shape; the post-compact "OOM"). Detaching restores the no-routing memory envelope
        while keeping the router trained. (Standard MoE practice: the load-balance aux shapes the
        gate, not the backbone representation.)
        """
        self._aux_detach_input = bool(detach_input)
        from .routing import TileRouter

        # Device of the host layer (post-compact the leaf is `values`, not `weight`, so go
        # through parameters() rather than a named attribute).
        try:
            dev = next(self.down.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")

        self.router = TileRouter(
            n_tile_groups=n_clusters,
            d_model=self.d_model,
            activation_ratio=activation_ratio,
            n_sub_keys=n_sub_keys,
            aux_loss_coeff=aux_loss_coeff,
            n_iters=n_iters,
        ).to(dev)   # a freshly-built nn.Module lands on CPU; move it onto the model's device
                    # or the first routed matmul fails with mat2-on-cpu vs activations-on-cuda.
        self.n_clusters = n_clusters
        # Contiguous neuron→cluster map over d_ff (matches compact_with_groups' contiguous
        # output-cluster convention). Remainder neurons fold into the leading clusters.
        base = self.d_ff // n_clusters
        rem = self.d_ff % n_clusters
        h2c = torch.empty(self.d_ff, dtype=torch.long)
        s = 0
        for c in range(n_clusters):
            sz = base + (1 if c < rem else 0)
            h2c[s:s + sz] = c
            s += sz
        self.register_buffer("hidden_to_cluster", h2c.to(dev))

    def forward(self, x: Tensor, iter_idx: int = 0) -> Tensor:
        gu = self.gate_up(x)
        gate, up = gu.chunk(2, dim=-1)
        h = F.silu(gate) * up                         # [B, T, d_ff] hidden neuron bank
        if self.router is not None:
            # Detach the router input (default) so routing gradient does not flow into the
            # carrier x — required for looped-core memory (see enable_routing docstring). The
            # router params still train (grad via the detached input + gates from the main loss).
            _rx = x.detach() if self._aux_detach_input else x
            gates, aux = self.router(_rx, iter_idx=iter_idx)   # gates: [B, T, n_clusters]
            # Stash the load-balance aux for the training loop to collect (collect_routing_aux_losses
            # after forward, before backward). With detach_input the aux's graph reaches only the
            # router params (the detached x is a leaf), so it is cheap and does NOT pin the looped
            # core's forward graph — that is what keeps gradient checkpointing intact for routed steps.
            self._last_aux_loss = aux
            # Gate the d_ff hidden bank per neuron-cluster. Active groups stay ~unit scale
            # (gates sum to activation_k); inactive groups → 0.
            gates = gates.to(h.dtype)
            if self.d_ff % self.n_clusters == 0:
                # Memory-efficient + BIT-IDENTICAL when clusters are equal-size: reshape h to
                # [B, T, n_clusters, cluster_size] and broadcast-multiply gates[..., None].
                # Avoids materializing the full [B, T, d_ff] index-expanded gates tensor
                # (gates[..., hidden_to_cluster]) — that index-expand cost ~one extra [B,T,d_ff]
                # buffer per core MLP, held across BPTT grad-iters (the routing memory blow-up).
                cs = self.d_ff // self.n_clusters
                h = (h.unflatten(-1, (self.n_clusters, cs)) * gates.unsqueeze(-1)).flatten(-2)
            else:
                # Uneven clusters (remainder neurons): fall back to the index-expand path.
                h = h * gates[..., self.hidden_to_cluster]
        return self.down(h)


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
        # MORPH_DIAG_CORECOS: log per-iteration carrier ROTATION (min per-token cos(h_new,h_a))
        # + paired magnitude gain, to test whether the β1=0 spike is a directional rotation
        # (the magnitude governor was magnitude-invariant). Cheap: tensor-reduced, 1 sync/forward.
        self._diag_corecos = bool(os.environ.get("MORPH_DIAG_CORECOS"))
        self._fwd_count = 0
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

        # ── Residual = n-stream Hyper-Connection (Cayley/JPmHC), the sole residual ──
        self._residual_mode = "hc_cayley"
        self._is_hc = True
        self._n_streams = cfg.hc_streams
        hc_kwargs = dict(
            n_streams=cfg.hc_streams, tau=cfg.hc_tau,
            cayley_iters=cfg.hc_cayley_iters, cayley_alpha=cfg.hc_cayley_alpha,
            init_gain=cfg.hc_init_gain, use_kernel=cfg.hc_use_kernel,
        )

        def _make_block(layer_idx: int) -> MORPHBlock:
            return MORPHBlock(
                norm_attn=RMSNorm(d),
                attn=MORPHAttention(layer_idx=layer_idx, **attn_kw),
                norm_mlp=RMSNorm(d),
                mlp=_make_swiglu(d, d_ff, cfg.dropout),
                d_model=d,
                hc_kwargs=hc_kwargs,
            )

        # ── Prelude ───────────────────────────────────────────────────
        # All sections use MortarLinear MLPs — whole-body CMS pruning.
        self.prelude = nn.ModuleList([
            _make_block(i) for i in range(cfg.n_prelude)
        ])

        # ── Loop state transition ─────────────────────────────────────
        self.input_norm = RMSNorm(d)
        self.injection = DiagonalInjection(self._ctx_start, self._ctx_end)

        # ── Core (shared across loop iterations — MortarLinear for CMS pruning)
        self.core = nn.ModuleList([
            _make_block(cfg.n_prelude + i)
            for i in range(cfg.n_core)
        ])

        # ── Coda ──────────────────────────────────────────────────────
        self.coda = nn.ModuleList([
            _make_block(cfg.n_prelude + cfg.n_core + i)
            for i in range(cfg.n_coda)
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

        # ── Retention branch (#230) ────────────────────────────────────
        # Attach AFTER all base modules → GLA's RNG draws are a tail, so the base model is
        # byte-identical to the baseline whether retention is on or off. With the branch-gate
        # near 0 at init, retention-on ≈ baseline at step 0, and the ablation isolates exactly
        # the retention branch (no confound from a different random init of the rest of the net).
        self._retention_layers = tuple(cfg.retention_layers)
        self._core_has_retention = cfg.retention and any(
            i in self._retention_layers for i in range(cfg.n_core))
        if cfg.retention:
            from .gla import GatedLinearAttention
            rheads = cfg.retention_heads or cfg.n_heads
            for section in (self.prelude, self.core, self.coda):
                for si, blk in enumerate(section):
                    if si in self._retention_layers:
                        blk.attach_retention(
                            GatedLinearAttention(
                                d, rheads,
                                mode="kernel" if cfg.use_kernels else "chunked",
                                chunk=cfg.retention_chunk,
                                gate_logit_bias=cfg.retention_gate_bias),
                            RMSNorm(d), gate_init=cfg.retention_gate_init)

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
                              dtype: torch.dtype,
                              ve_bagged: list[Tensor] | None = None) -> Tensor:
        """Combined single-stream additive injection [B,S,C] for `layer_idx`.

        ``lam*bigram`` (full width) + (x0_term + ve_term) placed in the ctx slice.
        ``x0_term`` is the pre-projected/scaled x0 signal (``ChannelInject.precompute``).

        ``ve_bagged`` (TST only): pre-bagged per-ve-layer ctx signals [B,L,ctx_w].
        When provided, the value-embed contribution uses the bag-mean instead of the
        raw per-token ``input_ids`` lookup (which would be [B,s·L], mismatching the
        bagged [B,L] carrier). None → the normal per-token lookup (bit-identical).
        """
        cs, ce = self._ctx_start, self._ctx_end
        lam = self.embed.bigram.lambdas[layer_idx].to(dtype)
        full = lam * bigram_emb.to(dtype)                      # [B,S,C] full-width bigram
        ctx = x0_term.to(dtype)                                # [B,S,ctx_w] x0 contribution
        if layer_idx in self._ve_layer_map:
            ve_idx = self._ve_layer_map.index(layer_idx)
            if ve_bagged is not None:
                ctx = ctx + ve_bagged[ve_idx].to(dtype)
            else:
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

    def _apply_core_step(self, h_in, e_in, ids, x0_terms, bg,
                         ret_state=None, iter_idx=0):
        """ONE core-loop step: SSM diagonal injection → the n_core shared blocks
        (each with per-layer x0/bigram injection + optional GLA retention carry).
        Returns ``(h, new_ret_state)`` (new_ret None unless a core layer carries retention).

        Lifted verbatim out of ``_forward_single``'s loop so the EXACT training-path core
        map ``f_θ`` is callable in isolation — for σ_max(J_core) probing and per-step
        contractivity diagnostics (the nested-dynamical-system inner map). The
        only former loop-local was ``np_`` (= n_prelude, a constant), recomputed here, so
        this is byte-identical to the in-loop closure (gated bit-exact).
        """
        np_ = self.cfg.n_prelude
        mlp_kw = {"iter_idx": iter_idx}
        h_injected = self.injection(h_in, e_in)
        ret_cap = {} if self._core_has_retention else None
        for i, layer in enumerate(self.core):
            gi = np_ + i
            term = self._build_injection_term(
                gi, x0_terms[i], ids, bg, h_injected.dtype
            )
            h_injected = self._apply_injection(h_injected, term)
            # Retention carry only for the designated core layer(s); others get None.
            is_ret = ret_cap is not None and (i in self._retention_layers)
            rs_arg = ret_state if is_ret else None
            rc_arg = ret_cap if is_ret else None
            h_injected = layer(h_injected, mlp_kwargs=mlp_kw,
                               ret_state=rs_arg, ret_capture=rc_arg)
        new_ret = ret_cap.get("state") if ret_cap is not None else None
        return h_injected, new_ret

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, input_ids: Tensor, labels: Tensor | None = None,
                bag_size: int = 0) -> dict:
        return self._forward_single(input_ids, labels, bag_size)

    def _forward_single(self, input_ids: Tensor,
                        labels: Tensor | None = None,
                        bag_size: int = 0) -> dict:
        # ── Token-Superposition Training input bagging (TST, arXiv 2605.06546) ──
        # bag_size==0 → baseline path, BIT-IDENTICAL to pre-TST (and what eval/gen
        # always use). bag_size==s>0 → the superposition phase: input_ids arrives as
        # [B, s·L] raw tokens; we average each contiguous bag of s token-embeddings
        # into one "s-token", so the model processes L = (s·L)/s positions — SAME
        # cost/VRAM as baseline. value-embeds fire only in the prelude → bag their
        # per-token ctx signal up front (ve_bagged); the core/coda never read input_ids.
        B, T_in = input_ids.shape
        s = bag_size
        if s > 0:
            T = T_in // s
            x = self.embed_drop(self.embed(input_ids).view(B, T, s, -1).mean(dim=2))   # [B,L,d]
            bigram_emb = self.embed.get_bigram(input_ids).view(B, T, s, -1).mean(dim=2)  # [B,L,d]
            n_ve = len(self._ve_layer_map)
            ve_bagged = ([
                self.value_embeds[k]
                    .precompute(self.value_embed_tables[k](input_ids))
                    .view(B, T, s, -1).mean(dim=2)                                       # [B,L,ctx_w]
                for k in range(n_ve)
            ] if n_ve > 0 else None)
        else:
            T = T_in
            x = self.embed_drop(self.embed(input_ids))
            bigram_emb = self.embed.get_bigram(input_ids)
            ve_bagged = None

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
                i, self.x0_injects[i].precompute(x0), input_ids, bigram_emb, x.dtype,
                ve_bagged=ve_bagged,
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

        def _core_step(h_in, e_in, ids, x0_terms, bg, ret_state=None, iter_idx=0):
            # Thin closure → the bound `_apply_core_step` method (single source of truth so
            # the σ_max probe / diagnostics exercise the EXACT training core map). Kept as a
            # closure so `checkpoint(_core_step, ...)` and the eager/no_grad call sites below
            # are unchanged; np_ (= cfg.n_prelude) is now recomputed inside the method.
            return self._apply_core_step(h_in, e_in, ids, x0_terms, bg,
                                         ret_state=ret_state, iter_idx=iter_idx)

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

        # ── Retention cross-iteration carry (#230) ────────────────────────────
        # GLA state for the core retention layer, carried iter→iter (fp32 accumulator, tiny).
        # Held in the SAME sorted/active-set order as h_s: slice [:n_active], carry the frozen
        # suffix unchanged — exactly like the carrier. The no_grad iterations produce a detached
        # state, so when it enters the first grad iteration the gradient does NOT flow back into
        # the frozen window (truncated-BPTT boundary, automatic). retention_carry=False → never
        # tracked (each iter reseeds zero = global retention with no memory).
        track_ret = self._core_has_retention and self.cfg.retention_carry
        if track_ret:
            _rh = self.cfg.retention_heads or self.cfg.n_heads
            _rdh = self.cfg.d_model // _rh
            ret_state_s = h_s.new_zeros(h_s.shape[0], _rh, _rdh, _rdh, dtype=torch.float32)
        else:
            ret_state_s = None

        _cc_meanmin = None  # MORPH_DIAG_CORECOS: min-over-iters of MEAN per-token cos(h_new,h_a)
        _cc_fracmax = None  # max-over-iters of FRACTION of tokens rotated >60° (cos<0.5)
        _cc_min = None      # min per-token cos (saturated order-stat; kept for reference)
        _cc_gain = None     # max per-sample magnitude gain (natural when governor off)
        # MORPH_DIAG_PERITER: keep the PER-ITERATION max_gain (realized one-step amplification,
        # a data-direction lower bound on σ_max(J_core)). If it COMPOUNDS across iteration index t
        # when σ_max grows large → σ_max-driven transient blowup through the loop (the
        # nested-dynamical-system frame). Reuses the validated _g; just doesn't max-reduce over t.
        _peri = self._diag_corecos and bool(os.environ.get("MORPH_DIAG_PERITER"))
        _peri_g = []        # per-iter max_gain (computed PRE-governor below)
        # Loop-axis STP: collect the grad-carrying iterates' carriers (stream-reduced) so the
        # post-loop geodesic smoothness over the ITERATION trajectory is differentiable. Only the
        # last bptt_depth iters carry grad; active sets are nested prefixes (depth-sorted), so a
        # consecutive triplet is sliced to the LATEST iter's prefix (samples active in all three).
        _loop_stp_on = self.training and self.cfg.loop_stp_lambda > 0.0
        _loop_carriers: list[Tensor] = []
        for t in range(total_iters):
            n_active = active_counts[t]
            if n_active == 0:
                break
            h_a = h_s[:n_active]
            args = (h_a, e_s[:n_active], ids_s[:n_active],
                    x0_s[:, :n_active], bg_s[:n_active])
            rs_a = ret_state_s[:n_active] if track_ret else None

            # Checkpoint this grad-iteration? Only in training, and only the first n_ckpt grad
            # iters (later ones run eager → no recompute). n_nograd iters are frozen (no_grad).
            do_ckpt = self.training and (t - n_nograd) < n_ckpt

            if t < n_nograd:
                with torch.no_grad():
                    h_new, rs_new = _core_step(*args, ret_state=rs_a, iter_idx=t)
            elif do_ckpt:
                h_new, rs_new = checkpoint(_core_step, *args, ret_state=rs_a, iter_idx=t,
                                           use_reentrant=False)
            else:
                # eval, OR a grad-iter we chose not to checkpoint (activations retained).
                h_new, rs_new = _core_step(*args, ret_state=rs_a, iter_idx=t)

            # ── L1 core-gain governor (#276) ──────────────────────────────────────────
            # Cap this iteration's per-sample looped-core amplification ‖h_new‖/‖h_a‖ ≤ τ.
            # IDENTITY when gain ≤ τ (healthy: the HC residual is norm-preserving so gain≈1 →
            # scale=1.0 → bit-exact x*1.0); only SHRINKS the runaway-gain step that the
            # weight-shared core would otherwise amplify T× (gain runaway mode). Applied
            # uniformly across the no_grad / checkpoint / eager branches (outside the checkpoint
            # so the scaling lives in the outer graph). τ=0 → skipped entirely → bit-identical.
            _tau = self.cfg.core_gain_clip
            if _tau > 0.0:
                _in_n = h_a.flatten(1).norm(dim=1)
                _out_n = h_new.flatten(1).norm(dim=1)
                _scale = torch.clamp(_tau * _in_n / (_out_n + 1e-6), max=1.0)
                h_new = h_new * _scale.view(-1, *([1] * (h_new.dim() - 1)))

            # Loop-STP: stash this grad-iter's carrier (stream-reduced → [n_active, S, C]).
            if _loop_stp_on and t >= n_nograd:
                _loop_carriers.append(h_new.mean(dim=2) if self._is_hc else h_new)

            if self._diag_corecos:
                _a = h_a.flatten(0, 1); _b = h_new.flatten(0, 1)          # [n*S, C] per-token
                _ct = (_a * _b).sum(-1) / (_a.norm(dim=-1) * _b.norm(dim=-1) + 1e-6)
                _mean = _ct.mean()                                        # avg token rotation
                _frac = (_ct < 0.5).float().mean()                       # frac rotated >60°
                _cm = _ct.min()
                _g = (h_new.flatten(1).norm(dim=1) / (h_a.flatten(1).norm(dim=1) + 1e-6)).max()
                _cc_meanmin = _mean if _cc_meanmin is None else torch.minimum(_cc_meanmin, _mean)
                _cc_fracmax = _frac if _cc_fracmax is None else torch.maximum(_cc_fracmax, _frac)
                _cc_min = _cm if _cc_min is None else torch.minimum(_cc_min, _cm)
                _cc_gain = _g if _cc_gain is None else torch.maximum(_cc_gain, _g)
                if _peri:
                    _peri_g.append(_g)   # per-iteration realized max_gain (raw when governor off)

            # updated active prefix + frozen suffix (no in-place op).
            with _prof("carrier::loop_cat"):
                h_s = h_new if n_active == h_s.shape[0] else \
                    torch.cat([h_new, h_s[n_active:]], dim=0)
            if track_ret and rs_new is not None:
                ret_state_s = rs_new if n_active == ret_state_s.shape[0] else \
                    torch.cat([rs_new, ret_state_s[n_active:]], dim=0)

        if self._diag_corecos and self.training and _cc_meanmin is not None:
            self._fwd_count += 1
            print(f"CORECOS fwd={self._fwd_count} mean_cos={_cc_meanmin.item():.4f} "
                  f"frac_rot={_cc_fracmax.item():.4f} min_cos={_cc_min.item():.4f} "
                  f"max_gain={_cc_gain.item():.3f}", flush=True)
            if _peri and _peri_g:
                _gv = torch.stack(_peri_g)                       # [n_iters], 1 sync
                _gs = ",".join(f"{x:.2f}" for x in _gv.tolist())
                print(f"PERITER fwd={self._fwd_count} n_iter={_gv.numel()} gains=[{_gs}]", flush=True)

        # ── Loop-axis STP loss ────────────────────────────────────────────────
        # Geodesic smoothness over the iteration trajectory: for each consecutive grad-iter
        # triplet (a,b,c), penalise 1 − cos(c−b, b−a) (the increments should be colinear =
        # locally-affine inner map). Active sets are nested prefixes, so slice a,b to c's prefix.
        loop_stp_loss = h_s.new_zeros(())
        if _loop_stp_on and len(_loop_carriers) >= 3:
            _terms: list[Tensor] = []
            for j in range(len(_loop_carriers) - 2):
                a, b, c = _loop_carriers[j], _loop_carriers[j + 1], _loop_carriers[j + 2]
                n = c.shape[0]                            # nested → latest iter = smallest prefix
                v_bwd = b[:n] - a[:n]                      # increment k     [n, S, C]
                v_fwd = c - b[:n]                          # increment k+1
                _terms.append((1.0 - F.cosine_similarity(v_fwd, v_bwd, dim=-1)).mean())
            if _terms:
                loop_stp_loss = torch.stack(_terms).mean()

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
            if labels.ndim == 3:
                # TST superposition phase (#274): labels arrive as [B, T, s] token
                # bags → multi-hot CE = mean of the s per-target CE terms against the
                # SAME logits. Init loss ≈ log(V) (~11), NOT log(V)/s (~1.8) — the
                # single-hot labels.reshape(-1) would truncate to the
                # first B·T entries. Reduces to single-hot at s=1.
                ce_loss = fused_linear_cross_entropy_mce(
                    x.reshape(-1, x.shape[-1]), w_full,
                    labels.reshape(-1, labels.shape[-1]),
                    ignore_index=-100, chunk_size=self.cfg.ce_chunk_size,
                )
            else:
                ce_loss = fused_linear_cross_entropy(
                    x.reshape(-1, x.shape[-1]), w_full, labels.reshape(-1),
                    ignore_index=-100, chunk_size=self.cfg.ce_chunk_size,
                )
            loss = ce_loss + self.cfg.stp_lambda * stp_loss + self.cfg.loop_stp_lambda * loop_stp_loss
            out = {"logits": None, "stp_loss": stp_loss, "loop_stp_loss": loop_stp_loss, "loss": loss}
        else:
            logits = self.embed.attend(x)
            out = {"logits": logits}
            if labels is not None:
                if labels.ndim == 3:
                    # 3-D bag labels in the eager path (eval/gen normally force
                    # bag_size=0, so this is defensive): full-logits MCE reference.
                    ce_loss = multi_hot_cross_entropy_reference(
                        logits.reshape(-1, self.cfg.vocab_size),
                        labels.reshape(-1, labels.shape[-1]), ignore_index=-100,
                    )
                else:
                    ce_loss = F.cross_entropy(
                        logits.reshape(-1, self.cfg.vocab_size),
                        labels.reshape(-1), ignore_index=-100,
                    )
                loss = ce_loss + self.cfg.stp_lambda * stp_loss + self.cfg.loop_stp_lambda * loop_stp_loss
                out["stp_loss"] = stp_loss
                out["loop_stp_loss"] = loop_stp_loss
                out["loss"] = loss

        return out
