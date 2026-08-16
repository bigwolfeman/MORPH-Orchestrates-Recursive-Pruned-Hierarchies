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
from .sparsity import MortarLinear
from .tul import TULConfig, TULSlots, compact_index, gather_positions, scatter_positions
from .tul_layout import SlotLayout

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


# ── MORPH_STATIC_GRAPHS: capture the static front/back of the step as CUDA graphs ──
# The step = [embed+prelude] → [Poisson-depth core loop] → [coda+head+CE]. The core loop
# is variable-shape (active-set shrinking) and stays eager; the FRONT (embed+dropout+
# bigram+HC-expand+prelude) and BACK (coda+HC-mean+lm_mixer+final_norm) are fixed-shape,
# once per step → captured once via torch.cuda.make_graphed_callables and replayed as 2
# graph launches (+2 bwd graph launches) instead of thousands of individual kernels.
# The fused-CE stays EAGER: fused_linear_cross_entropy computes n_valid via .item() — a
# host sync that is ILLEGAL during capture (and its python-float division is last-bit
# load-bearing; see the reverted 0-dim-tensor n_valid change).
# BIT-EXACTNESS (class A): same kernels, same order, same tensors. Dropout RNG is handled
# by torch's graph-safe philox mechanism — each replay advances the default CUDA
# generator EXACTLY as the eager region would (probed bitwise: 8-step training loop with
# graphed dropout regions interleaved with eager RNG consumers, losses/grads/params all
# torch.equal — ignore/perf/gpu_probe_rng_graph.py).
# Requirements handled in build_static_graphs (each probed, not assumed):
#   * build MUST run with no prior-step autograd graph alive (train.py dels loss/out +
#     gc.collect() first) — stale default-stream AccumulateGrad nodes invalidate capture.
#   * a FAILED capture leaves the CUDA generator in graph mode → the process cannot fall
#     back to eager RNG → build failures must abort loudly, never be swallowed.
#   * build warmup runs real fwd/bwd on dummy data → wrapped in fork_rng + a snapshot/
#     restore of every region buffer (router load-EMAs mutate in forward).
#   * params inside the regions get their grads as VIEWS of the bwd-graph static buffers
#     (AccumulateGrad steal) → tagged p._grad_via_graph_static so the optimizer CUDA
#     graph (MORPH_OPT_CUDA_GRAPH) keeps steal-path zeroing for them (stable data_ptrs
#     for free; in-place zeroing would alias-double via buffer.add_(buffer)).
# MEMORY COST (measured, mb4/seq4k d768 on the 32GB 5090): the graphs' private mempool
# permanently reserves ~9.3GB (front+back activations + static buffers become EXCLUSIVE
# to the graphs — the eager allocator can no longer time-share that memory with the core
# loop's transient peak). With the default allocator this OOMs locally at deploy shape;
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True makes it fit (peak reserved ~24.4GB).
# On the 96GB cloud target the pool is trivial.
# Default OFF until gated; in-process override for A/B: set_static_graphs(True/False).
_STATIC_GRAPHS = os.environ.get("MORPH_STATIC_GRAPHS", "0").lower() not in ("0", "", "false")


def set_static_graphs(enabled: bool) -> None:
    """In-process override of MORPH_STATIC_GRAPHS (A/B testing without env replumbing)."""
    global _STATIC_GRAPHS
    _STATIC_GRAPHS = bool(enabled)


class _StaticRegion(nn.Module):
    """Thin nn.Module wrapper for a region closure, registering the region's REAL
    submodules so make_graphed_callables includes their parameters in the graph's
    static input surface (param grads flow). NOT attached to the model tree —
    registration here must not change the model's state_dict or named_parameters.

    forward() enters autocast ITSELF (cache off — required under capture; the cache is
    a pure cast memoization, values identical). This is load-bearing for bit-exactness:
    make_graphed_callables must be called with NO ambient autocast, so that the FORWARD
    capture sees autocast dispatch (matching the eager forward under train.py's autocast)
    while the BACKWARD capture — autograd.grad, which runs after this forward returns —
    executes with autocast OFF, matching eager training where .backward() is called
    outside the autocast block. Capturing the backward under ambient autocast re-dispatches
    autocast-eligible ops inside backward to bf16 and was MEASURED as a real grad
    divergence (~1e-3 across 200+ params, loss bitwise but step+1 diverged)."""

    def __init__(self, fn, submodules):
        super().__init__()
        self._fn = fn
        self._mods = nn.ModuleList(submodules)

    def forward(self, *args):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            return self._fn(*args)


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
    # Value embeddings (token-value injection, modded-nanogpt trick): fresh per-layer
    # vocab lookups additively injected into the ctx channel at the first n_ve prelude
    # layers. None → min(3, n_prelude) (historical default, bit-identical). Set 0 to
    # ablate them entirely (memorization-capacity study), or a smaller int to reduce.
    n_ve: int | None = None

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

    # ── TUL — Thought Unpack Loop (docs/tul-spec.md) ──────────────────────
    # None → NO TUL parameters are constructed and the model is byte-identical to the
    # baseline (the `retention` convention: the flag gates CONSTRUCTION, not a forward
    # branch). Set it and the model gains E_slot / E_mask / W_prefix (spec §3.1-§3.4),
    # which stay inert — grad None, so the optimizer skips them — until a forward is
    # called with a `slot_layout`. `slot_layout=None` is bit-identical either way
    # (runtime-invariants §6b).
    tul: TULConfig | None = None

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
        n_ve = min(3, cfg.n_prelude) if cfg.n_ve is None else min(cfg.n_ve, cfg.n_prelude)
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

        # ── TUL slot parameters (docs/tul-spec.md §3.1-§3.4) ───────────────
        # Constructed LAST, after retention, for the same reason: all three inits are
        # DETERMINISTIC (zeros / identity — zero RNG draws), so a TUL model's base weights
        # are byte-identical to a baseline built with the same seed, and the arms differ by
        # the mechanism alone. E_slot is re-initialised from the live embedding table at the
        # activation step (Block Transformer §3.7) — see TULSlots.init_at_activation.
        self.tul: TULSlots | None = TULSlots(d, cfg.tul) if cfg.tul is not None else None

        # Master kernel switch → drives the fused-Triton-vs-eager-reference
        # dispatch in the attention kernels (process-global flag). Set at build
        # so the choice is captured in the run; the fused-CE branch in forward()
        # reads self.cfg.use_kernels directly.
        from morph.kernels.triton._eager_flag import set_force_eager
        set_force_eager(not cfg.use_kernels)

        # Static-region CUDA graphs (MORPH_STATIC_GRAPHS): plain dict attr — holds the
        # graphed front/back callables + capture shapes. Deliberately NOT a submodule.
        self._static_graphs: dict = {}

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
        if bigram_emb is not None:
            lam = self.embed.bigram.lambdas[layer_idx].to(dtype)
            full = lam * bigram_emb.to(dtype)                  # [B,S,C] full-width bigram
        else:
            # bigram disabled (bigram_hash_vocab == 0): zero full-width base so the
            # ctx-slice placement below is unchanged.
            full = x0_term.new_zeros(*x0_term.shape[:-1], self.cfg.d_model, dtype=dtype)
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
                         ret_state=None, iter_idx=0, inj_terms=None):
        """ONE core-loop step: SSM diagonal injection → the n_core shared blocks
        (each with per-layer x0/bigram injection + optional GLA retention carry).
        Returns ``(h, new_ret_state)`` (new_ret None unless a core layer carries retention).

        Lifted verbatim out of ``_forward_single``'s loop so the EXACT training-path core
        map ``f_θ`` is callable in isolation — for σ_max(J_core) probing and per-step
        contractivity diagnostics (the nested-dynamical-system inner map). The
        only former loop-local was ``np_`` (= n_prelude, a constant), recomputed here, so
        this is byte-identical to the in-loop closure (gated bit-exact).

        ``inj_terms`` (perf, launch-count): the per-core-layer additive injection term
        [n_core, n_active, S, C] is LOOP-INVARIANT (a function of x0/value-embed/bigram +
        input_ids only — none iteration-dependent), so ``_forward_single`` precomputes it
        ONCE and passes the active-set slice in. When provided we skip the per-layer
        ``_build_injection_term`` rebuild (was ~6-8 cast/mul/cat kernels × n_core ×
        total_iters redundant launches → n_core). BIT-IDENTICAL: the term equals the
        old per-iteration rebuild (same inputs), and the shared term added into each
        iteration's carrier accumulates the SAME sum-over-iterations gradient to
        proj/bigram/value-embed as the per-iteration form (identical to the x0 hoist).
        None → rebuild in-place (the σ_max probe / any caller without a precomputed stack).
        """
        np_ = self.cfg.n_prelude
        mlp_kw = {"iter_idx": iter_idx}
        h_injected = self.injection(h_in, e_in)
        ret_cap = {} if self._core_has_retention else None
        for i, layer in enumerate(self.core):
            gi = np_ + i
            if inj_terms is not None:
                term = inj_terms[i]
            else:
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

    # ── Static-region CUDA graphs (MORPH_STATIC_GRAPHS) ──────────────────────
    def static_graphs_invalidate(self, reason: str = "") -> None:
        """Drop the captured static-region graphs → permanent eager for this topology.

        Called at topology/context events that change what the graphs read by pointer or
        shape: the compact/route phase boundary (modules replaced) and RoPE set_context
        (cos/sin cache buffers rebuilt as NEW tensors). Un-tags _grad_via_graph_static so
        the optimizer graph's hybrid zeroing resumes owning those params' grad buffers
        (otherwise its address-signature would churn-recapture every step)."""
        if self._static_graphs:
            for w in (self._static_graphs.get("front_wrap"),
                      self._static_graphs.get("back_wrap")):
                if w is not None:
                    for p in w.parameters():
                        if getattr(p, "_grad_via_graph_static", False):
                            p._grad_via_graph_static = False
            self._static_graphs = {}
            print(f"  [static-graph] invalidated ({reason}) — regions run eager from here",
                  flush=True)

    def _drain_region_aux(self, roots) -> tuple[list, list[Tensor]]:
        """Drain the routing aux stashes under `roots` in model.modules() order,
        returning (stash_modules, aux_tensors). Inside a CAPTURED region fn this is
        load-bearing twice over:
        (1) the stash protocol (module attr set in forward, read+cleared by the train
            loop) is python-side and does NOT re-run on graph replay — a captured
            region would silently DROP its routers' aux loss from the training loss,
            and the stale stashed tensors keep the capture-time graph alive (which
            then kills the next capture via default-stream AccumulateGrad reuse).
        (2) the aux tensors are returned INDIVIDUALLY (not summed): the dispatch
            re-stashes each onto its own module so collect_routing_aux_losses adds
            them in the IDENTICAL order as eager — summing per-region here was
            measured as a real fp-reassociation (loss diverged 1.9e-3 by step 9 on a
            0-floor deterministic probe)."""
        mods, auxs = [], []
        for rm in roots:
            for mod in rm.modules():
                aux = getattr(mod, "_last_aux_loss", None)
                if aux is not None:
                    mods.append(mod)
                    auxs.append(aux)
                    mod._last_aux_loss = None
        return mods, auxs

    def build_static_graphs(self, sample_input_ids: Tensor) -> bool:
        """Capture front (embed→prelude) and back (coda→lm_mixer→final_norm) as CUDA
        graphs via torch.cuda.make_graphed_callables. Call ONCE from the training loop.

        HARD PRECONDITION (probed, ignore/perf/gpu_probe_rng_graph.py): no prior-step
        autograd graph may be alive — the caller must drop loss/out refs + gc.collect()
        first. Alive graphs keep params' AccumulateGrad nodes cached with default-stream
        metadata; the capture-stream backward then syncs with the uncapturable default
        stream → cudaErrorStreamCaptureInvalidated. And a FAILED capture leaves the CUDA
        generator in graph mode ("Offset increment outside graph capture" on the next
        eager RNG op) → the process is unrecoverable, so this method must NOT be wrapped
        in a fallback try/except — a build failure is a run-ending finding (no-theater).

        Build isolation: the API's warmup iterations run REAL fwd/bwd on dummy values →
        wrapped in fork_rng (CUDA stream position untouched) with every region buffer
        snapshotted/restored (router load-EMAs mutate in forward). Params are untouched
        (autograd.grad returns grads; nothing accumulates into .grad).
        """
        import gc

        if not _STATIC_GRAPHS:
            return False
        if not (self.training and torch.is_grad_enabled()):
            raise RuntimeError("build_static_graphs requires train mode with grad enabled")
        dev = sample_input_ids.device
        np_, nc = self.cfg.n_prelude, self.cfg.n_coda

        # Region wrappers referencing the REAL submodules (params → graph input surface).
        front_wrap = _StaticRegion(None, [
            self.embed, self.prelude,
            nn.ModuleList(list(self.x0_injects)[:np_]),
            self.value_embeds, self.value_embed_tables,
        ])
        back_mods = [
            self.coda,
            nn.ModuleList(list(self.x0_injects)[np_ + self.cfg.n_core:]),
            self.lm_mixer, self.final_norm,
        ]
        if getattr(self.embed, "bigram", None) is not None:
            back_mods.append(self.embed.bigram)   # lambdas[gi] read per coda layer
        back_wrap = _StaticRegion(None, back_mods)

        def _snap_buffers():
            return [(b, b.clone()) for w in (front_wrap, back_wrap) for b in w.buffers()]

        def _restore_buffers(snap):
            for b, sv in snap:
                b.copy_(sv)

        # ── Spec discovery: one throwaway eager front+back pass (shapes/dtypes of the
        # region boundary tensors + whether each region stashes routing aux). autocast
        # (bf16, cache_enabled=False) = the training dispatch (cache off is required by
        # make_graphed_callables; the cache is a pure cast memoization — values identical
        # either way). The region-aux collection also CLEARS the spec pass's stashes —
        # leaving them stashed would keep the spec graph alive into the capture (fatal,
        # see _collect_region_aux). ──
        snap = _snap_buffers()
        with torch.random.fork_rng(devices=[dev]):
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                x_spec, x0_spec, bg_spec = self._front_region(sample_input_ids)
                front_aux_mods, _fa = self._drain_region_aux((self.prelude,))
                xh_spec = self._back_region(x_spec, x0_spec, bg_spec)
                back_aux_mods, _ba = self._drain_region_aux((self.coda,))
        _restore_buffers(snap)
        has_bigram = bg_spec is not None
        n_front_aux, n_back_aux = len(front_aux_mods), len(back_aux_mods)
        specs = {
            "x": (x_spec.shape, x_spec.dtype), "x0": (x0_spec.shape, x0_spec.dtype),
            "bg": (bg_spec.shape, bg_spec.dtype) if has_bigram else None,
        }
        # Stale-graph rule applies to OUR OWN spec pass too: free it before capture.
        del x_spec, x0_spec, bg_spec, xh_spec, _fa, _ba, snap
        gc.collect()

        def _front_fn(ids):
            x, x0, bg = self._front_region(ids)
            outs = (x, x0, bg) if has_bigram else (x, x0)
            _, auxs = self._drain_region_aux((self.prelude,))
            return outs + tuple(auxs)

        def _back_fn(*args):
            x, x0 = args[0], args[1]
            bg = args[2] if has_bigram else None
            xh = self._back_region(x, x0, bg)
            _, auxs = self._drain_region_aux((self.coda,))
            return (xh,) + tuple(auxs)

        front_wrap._fn = _front_fn
        back_wrap._fn = _back_fn

        def _dummy(key, rg=True):
            # NO RNG (a device randn here would advance the training generator and
            # shift every later dropout/poisson draw off the baseline stream — the
            # 6.9e-2 probe divergence). Deterministic non-constant ramp: avoids the
            # all-zero RMSNorm edge while staying generator-free. Values are dummies —
            # warmup math is discarded.
            shape, dtype = specs[key]
            n = 1
            for d_ in shape:
                n *= int(d_)
            t = ((torch.arange(n, device=dev, dtype=torch.float32) % 977) / 977.0 - 0.5)
            return t.reshape(shape).to(dtype).requires_grad_(rg)

        front_samples = (sample_input_ids,)
        back_samples = ((_dummy("x"), _dummy("x0"), _dummy("bg")) if has_bigram
                        else (_dummy("x"), _dummy("x0")))

        # NO ambient autocast here — _StaticRegion.forward enters autocast itself, so
        # the fwd captures see eager-matching autocast dispatch while the bwd captures
        # (autograd.grad, after forward returns) run autocast-OFF exactly like eager
        # training's .backward() outside the autocast block. See _StaticRegion.
        snap = _snap_buffers()
        with torch.random.fork_rng(devices=[dev]):
            g_front, g_back = torch.cuda.make_graphed_callables(
                (front_wrap, back_wrap), (front_samples, back_samples),
                allow_unused_input=True,
            )
        _restore_buffers(snap)

        # Region params now receive grads as VIEWS of the bwd-graph static buffers
        # (AccumulateGrad steal) — stable data_ptrs by construction. Tag them so the
        # optimizer CUDA graph keeps steal-path (set_to_none) zeroing for them: in-place
        # zeroing + accumulate would alias-double (buffer.add_(buffer)).
        n_tagged = 0
        for w in (front_wrap, back_wrap):
            for p in w.parameters():
                p._grad_via_graph_static = True
                n_tagged += 1

        self._static_graphs = {
            "front": g_front, "back": g_back,
            "front_wrap": front_wrap, "back_wrap": back_wrap,
            "front_shape": sample_input_ids.shape, "back_shape": specs["x"][0],
            "has_bigram": has_bigram,
            "front_aux_mods": front_aux_mods, "back_aux_mods": back_aux_mods,
        }
        print(f"  [static-graph] captured front(embed+{np_} prelude) and "
              f"back({nc} coda+head) regions → 2 fwd + 2 bwd graph replays/step "
              f"({n_tagged} params tagged steal-path, bigram={has_bigram}, "
              f"aux front/back={n_front_aux}/{n_back_aux})", flush=True)
        return True

    # ── Static regions (single source of truth for eager AND graph capture) ──
    # Pure code motion out of _forward_single — the flag-OFF path calls these with the
    # identical ops in the identical order as the old inline code.

    def _front_tail(self, x: Tensor, input_ids: Tensor, bigram_emb,
                    ve_bagged) -> tuple[Tensor, Tensor]:
        """x0 skip-clone → HC stream expansion → prelude blocks. Returns (x, x0)."""
        B, T = x.shape[0], x.shape[1]
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
        return x, x0

    def _front_region(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """bag0 FRONT region: embed+dropout+bigram → _front_tail. Fixed shapes, no
        recurrence, one RNG site (embed_drop) + prelude MLP dropouts → graphable."""
        x = self.embed_drop(self.embed(input_ids))
        bigram_emb = self.embed.get_bigram(input_ids)
        x, x0 = self._front_tail(x, input_ids, bigram_emb, None)
        return x, x0, bigram_emb

    def _back_region(self, x: Tensor, x0: Tensor, bigram_emb,
                     input_ids: Tensor | None = None,
                     inject_keep: Tensor | None = None) -> Tensor:
        """BACK region: coda blocks → HC stream mean → lm_mixer → final_norm.
        input_ids is only threaded into _build_injection_term for signature parity —
        value-embeds fire exclusively in the prelude (gi ≥ n_prelude+n_core is never in
        _ve_layer_map), so None and the real ids are equivalent here. The fused CE stays
        OUTSIDE (its n_valid .item() host-syncs — cannot live in a captured graph).

        ``inject_keep`` (TUL only, [B, S, 1], 1.0 keep / 0.0 drop): zeroes the whole
        additive injection at token positions whose coda state was replaced by E_mask
        (spec §3.4 token-state dropout). x0 carries proj(embed(t)) and the bigram term
        carries hash(t, t−1), so without this the dropped token leaks straight back in
        and Bowman's word dropout becomes a no-op. None on every non-TUL path → the
        ops below are unchanged and bit-identical."""
        for i, layer in enumerate(self.coda):
            gi = self.cfg.n_prelude + self.cfg.n_core + i
            term = self._build_injection_term(
                gi, self.x0_injects[gi].precompute(x0), input_ids, bigram_emb, x.dtype
            )
            if inject_keep is not None:
                term = term * inject_keep.to(term.dtype)
            x = self._apply_injection(x, term)
            x = layer(x)

        # ── Hyper-Connection stream reduction ─────────────────────────
        # Collapse the n streams back to a single C-dim representation before the LM head.
        # Mean readout is scale-preserving and (with all streams equal at init) exactly
        # recovers the plain-residual output; learned asymmetry is read out as the mean.
        if self._is_hc:
            x = x.mean(dim=2)

        # ── LM head ──────────────────────────────────────────────────
        x = self.lm_mixer(x)
        x = self.final_norm(x)
        return x

    def _core_region(self, x: Tensor, x0: Tensor, bigram_emb,
                     input_ids: Tensor | None = None) -> Tensor:
        """CORE region: input_norm → the Poisson-depth core loop → the looped carrier.

        Pure code motion out of ``_forward_single`` (the ``_front_region`` /
        ``_back_region`` precedent): the ops and their order are IDENTICAL to the old
        inline block, so every non-TUL path is bit-identical. It is a method so the TUL
        arm A2 (``tokens_through_core``, spec §7.1) can run the SAME per-sample core over
        a sequence that happens to contain slot positions, instead of forking a second
        implementation of the loop."""
        B = x.shape[0]
        # ── Core loop ─────────────────────────────────────────────────
        # n_core == 0 → prelude output flows straight to the coda. The whole loop
        # machinery below (input_norm/h clone, depth sampling, x0 hoist, DiagonalInjection
        # via _apply_core_step) is core-only and must NOT run: with zero core blocks the
        # injection would still perturb the ctx channel every iteration. Used by seed models.
        if self.cfg.n_core > 0:
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

            # ── Hoist the loop-invariant PER-CORE-LAYER injection term out of the loop ──
            # `_build_injection_term(np_+i, x0_core_terms[i], input_ids, bigram_emb, dtype)`
            # depends on nothing iteration-varying — it is the SAME additive [B,S,C] term for
            # core layer i on every iteration. The old code rebuilt it inside `_apply_core_step`
            # every iteration (n_core × total_iters rebuilds, each ~6-8 cast/mul/cat kernels →
            # a big share of the launch-bound step's kernel soup; the `.to(dtype)` casts alone
            # are ~5k/step in the routed trace). Precompute the n_core distinct terms ONCE.
            # Bit-identical (same inputs ⇒ same value; the shared term added into each
            # iteration's carrier accumulates the identical sum-over-iterations gradient to
            # proj/value-embed/bigram-λ — exactly the validated x0-hoist argument). Built at the
            # carrier dtype `h.dtype` (== the old `h_injected.dtype`, bf16). Stacked so the
            # active-set slice is a cheap view and the checkpoint recompute reuses it (no rebuild
            # in backward either — doubles the saving on the checkpointed grad-iters).
            inj_core_terms = torch.stack(
                [self._build_injection_term(np_ + i, x0_core_terms[i], input_ids,
                                            bigram_emb, h.dtype)
                 for i in range(n_core)],
                dim=0,
            )  # [n_core, B, S, C]

            def _core_step(h_in, e_in, inj_terms, ret_state=None, iter_idx=0):
                # Thin closure → the bound `_apply_core_step` method (single source of truth so
                # the σ_max probe / diagnostics exercise the EXACT training core map). Kept as a
                # closure so `checkpoint(_core_step, ...)` and the eager/no_grad call sites below
                # are unchanged; np_ (= cfg.n_prelude) is now recomputed inside the method.
                # ids/x0_terms/bg are None here: the injection is precomputed (inj_terms) and
                # threaded as a checkpoint input so the recompute reuses it.
                return self._apply_core_step(h_in, e_in, None, None, None,
                                             ret_state=ret_state, iter_idx=iter_idx,
                                             inj_terms=inj_terms)

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
                # ids_s / bg_s / x0_s gathers are gone: the injection is precomputed
                # (inj_core_terms) and only IT needs sorting into active-set order. This also
                # drops 3 gather kernels/step (input_ids, bigram, x0-stack) from the hot loop.
                inj_s = inj_core_terms[:, perm]          # [n_core, B, S, C], sorted order

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
            # ── Eval-only loop-trajectory capture (interp: latent-MSE / decode fidelity) ──
            # Gated by an instance flag; OFF (default, getattr→False) → zero overhead and
            # bit-exact. Stashes the stream-reduced carrier z_0..z_T for the forecastability
            # probe (ignore/loop_latent_mse.py). Eval runs a UNIFORM depth so the active set is
            # the full batch in original order (perm == identity) → the stash is already in batch
            # order, no inv_perm needed. Do NOT set this during training.
            _capture_traj = getattr(self, "_capture_traj", False)
            _traj: list[Tensor] = []
            if _capture_traj:
                _traj.append((e_s.mean(dim=2) if self._is_hc else e_s).detach())  # z_0 (pre-loop)
            for t in range(total_iters):
                n_active = active_counts[t]
                if n_active == 0:
                    break
                h_a = h_s[:n_active]
                # inj_s[:, :n_active]: the precomputed injection sliced to the active prefix
                # (per-sample terms, no cross-sample mixing → slicing is exact). Passed as a
                # checkpoint input so backward recompute reuses it instead of rebuilding.
                args = (h_a, e_s[:n_active], inj_s[:, :n_active])
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

                if _capture_traj:  # eval-only interp: capture EVERY iteration's carrier (z_1..z_T)
                    _traj.append((h_new.mean(dim=2) if self._is_hc else h_new).detach())

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

            if _capture_traj:
                self._traj_carriers = _traj  # [z_0 .. z_T], each [B, S, C]; read by the interp probe

            if self._diag_corecos and self.training and _cc_meanmin is not None:
                self._fwd_count += 1
                print(f"CORECOS fwd={self._fwd_count} mean_cos={_cc_meanmin.item():.4f} "
                      f"frac_rot={_cc_fracmax.item():.4f} min_cos={_cc_min.item():.4f} "
                      f"max_gain={_cc_gain.item():.3f}", flush=True)
                if _peri and _peri_g:
                    _gv = torch.stack(_peri_g)                       # [n_iters], 1 sync
                    _gs = ",".join(f"{x:.2f}" for x in _gv.tolist())
                    print(f"PERITER fwd={self._fwd_count} n_iter={_gv.numel()} gains=[{_gs}]", flush=True)

            with _prof("carrier::inv_perm_gather"):
                x = h_s[inv_perm]                    # restore original batch order
        else:
            # n_core == 0 (seed models): the loop path hands the coda
            # h = input_norm(prelude_out) (+ core deltas), so the coreless path
            # must apply the same boundary norm. This makes a seed model
            # EXACTLY a target-with-silent-core — the growth invariant that
            # function-preserving core insertion depends on.
            x = self.input_norm(x)
        return x

    # ── TUL regions (docs/tul-spec.md §3) ─────────────────────────────────
    # Reached only when a `slot_layout` is passed. Every helper below is a no-op for
    # the plain path because the plain path never calls it.

    def _tul_front(self, input_ids: Tensor, layout: SlotLayout):
        """Embed + slot inputs + prelude over ALL positions (spec §3.2).

        The slot's input embedding is ``E_slot + mean_j embed(t_j)`` over its span's
        tokens, and its bigram / value-embed signals are the same bag-mean — this is
        exactly the TST ``ve_bagged`` path with a data-dependent bag map (spec §3.2;
        Dynamic Token Pooling mean-pool; BLT Eq. 5). The prelude itself is unchanged:
        the slot's output is the in-context pooled span summary (BLT §3.2.2).
        """
        tok_emb = self.tul.slot_input(self.embed(input_ids), layout, add_e_slot=True)
        x = self.embed_drop(tok_emb)
        _bg = self.embed.get_bigram(input_ids)
        bigram_emb = (self.tul.slot_input(_bg, layout, add_e_slot=False)
                      if _bg is not None else None)
        n_ve = len(self._ve_layer_map)
        ve_bagged = ([
            self.tul.slot_input(
                self.value_embeds[k].precompute(self.value_embed_tables[k](input_ids)),
                layout, add_e_slot=False)
            for k in range(n_ve)
        ] if n_ve > 0 else None)
        x, x0 = self._front_tail(x, input_ids, bigram_emb, ve_bagged)
        return x, x0, bigram_emb

    def _sample_slot_depths(self, layout: SlotLayout, device) -> Tensor:
        """``[B, max_slots]`` per-slot Poisson depth (spec §3.3 [W]).

        Parcae samples one depth per SEQUENCE; TUL samples one per SLOT — claim C1 is
        "depth per idea", so the depth must vary per idea. Eval is the deterministic
        mean depth. Pad slots get depth 1 so they never inflate ``total_iters``; their
        update is masked out regardless.
        """
        tc = self.cfg.tul
        mean_d = tc.slot_mean_depth or self.cfg.mean_depth
        max_d = tc.slot_max_depth or self.cfg.max_depth
        shape = layout.slot_index.shape
        if self.training:
            d = torch.poisson(torch.full(shape, float(mean_d), device=device)).long()
            d = d.clamp(min=1, max=max_d)
        else:
            d = torch.full(shape, int(mean_d), device=device, dtype=torch.long)
        return torch.where(layout.slot_valid, d, torch.ones_like(d))

    def _tul_core(self, x: Tensor, x0: Tensor, bigram_emb, layout: SlotLayout):
        """Gather slots → masked per-slot depth loop → looped states (spec §3.3).

        Returns ``(xn, h_slots, depths)``: ``xn = input_norm(prelude)`` for the whole
        carrier (token positions keep it — the ``n_core == 0`` seed path, BLT Eq. 9),
        and ``h_slots`` ``[B, max_slots, …]`` the looped state of each slot.

        Invariant 2 (runtime-invariants §6b): the depth is a MASKED UPDATE over the
        full compact slot sequence, never a per-position gather. The active-set
        shrinking of the token path is deliberately NOT used here — MORPH recomputes
        K/V from the current carrier every iteration, so shrinking the sequence would
        change what a frozen slot's keys are, and frozen slots must keep serving the
        same K/V. The compact sequence is 9-19× shorter than the token stream, which
        is what makes the lost shrink affordable (spec §3.3).
        """
        B, L = x.shape[0], x.shape[1]
        np_, n_core = self.cfg.n_prelude, self.cfg.n_core
        gidx = torch.where(layout.slot_valid, layout.slot_index, L)   # pads → dump row

        xn = self.input_norm(x)
        xn_pad = torch.cat([xn, xn.new_zeros(B, 1, *xn.shape[2:])], dim=1)
        e = gather_positions(xn_pad, gidx)                            # [B, S, n, C]
        with _prof("carrier::h_clone"):
            h = e.clone()

        # Loop-invariant injection, built ON THE COMPACT SEQUENCE (the x0/bigram hoist
        # of the token path, applied to 9-19× fewer positions). Value-embeds never fire
        # in the core (gi ≥ n_prelude is not in _ve_layer_map), so input_ids is not needed.
        x0_pad = torch.cat([x0, x0.new_zeros(B, 1, x0.shape[-1])], dim=1)
        x0_s = gather_positions(x0_pad, gidx)
        if bigram_emb is not None:
            bg_pad = torch.cat([bigram_emb, bigram_emb.new_zeros(B, 1, bigram_emb.shape[-1])],
                               dim=1)
            bg_s = gather_positions(bg_pad, gidx)
        else:
            bg_s = None
        inj = torch.stack(
            [self._build_injection_term(np_ + i, self.x0_injects[np_ + i].precompute(x0_s),
                                        None, bg_s, h.dtype)
             for i in range(n_core)], dim=0)

        depths = self._sample_slot_depths(layout, x.device)
        total_iters = int(depths.max().item())
        n_nograd = max(0, total_iters - self.cfg.bptt_depth)
        n_grad_iters = total_iters - n_nograd
        _ck = self.cfg.ckpt_grad_iters
        n_ckpt = n_grad_iters if _ck < 0 else max(0, min(_ck, n_grad_iters))

        track_ret = self._core_has_retention and self.cfg.retention_carry
        if track_ret:
            _rh = self.cfg.retention_heads or self.cfg.n_heads
            _rdh = self.cfg.d_model // _rh
            ret_state = h.new_zeros(h.shape[0], _rh, _rdh, _rdh, dtype=torch.float32)
        else:
            ret_state = None

        def _core_step(h_in, e_in, inj_terms, ret_state=None, iter_idx=0):
            return self._apply_core_step(h_in, e_in, None, None, None,
                                         ret_state=ret_state, iter_idx=iter_idx,
                                         inj_terms=inj_terms)

        for t in range(total_iters):
            active = depths > t                                   # [B, S]
            do_ckpt = self.training and (t - n_nograd) < n_ckpt
            if t < n_nograd:
                with torch.no_grad():
                    h_new, rs_new = _core_step(h, e, inj, ret_state=ret_state, iter_idx=t)
            elif do_ckpt:
                h_new, rs_new = checkpoint(_core_step, h, e, inj, ret_state=ret_state,
                                           iter_idx=t, use_reentrant=False)
            else:
                h_new, rs_new = _core_step(h, e, inj, ret_state=ret_state, iter_idx=t)

            _tau = self.cfg.core_gain_clip
            if _tau > 0.0:
                _in_n = h.flatten(1).norm(dim=1)
                _out_n = h_new.flatten(1).norm(dim=1)
                _scale = torch.clamp(_tau * _in_n / (_out_n + 1e-6), max=1.0)
                h_new = h_new * _scale.view(-1, *([1] * (h_new.dim() - 1)))

            h = torch.where(active.view(*active.shape, *([1] * (h.dim() - 2))), h_new, h)
            if track_ret and rs_new is not None:
                ret_state = rs_new
        return xn, h, depths

    def _tul_group_losses(self, x: Tensor, labels: Tensor, layout: SlotLayout | None) -> dict:
        """Grouped cross-entropy over the three label populations (spec §3.4, §5, §7.2).

        MORPH's layout puts the slot BETWEEN a span's last token and the next span's
        first token, so ``t_1(i+1)`` is predicted TWICE: once from ``t_last`` (plain LM,
        no plan) and once from the slot's emitting position (with the plan). Spec §5
        weights both terms 0.5 "so first tokens are not counted twice", which makes the
        denominator the number of DISTINCT target tokens:

            L = (Σ_ordinary + ½Σ_tlast + ½Σ_emit) / (n_ordinary + ½n_tlast + ½n_emit)

        Implemented as three calls to the existing fused CE against fixed-shape index
        tensors (a zero pad row absorbs invalid slots) — no kernel change, no
        data-dependent shape, and no extra host sync. Splitting ``t_last`` from ``emit``
        instead of pooling them into one "half" group costs nothing and hands §7.2 its
        metrics for free: ``val/first_tok_ce`` is the emit term and
        ``val/first_tok_counterfactual`` is ``CE(t_last) − CE(emit)``.

        ``layout=None`` (arm A4 / the plan-nats gather, where slots are not in the
        sequence at all) → a single ordinary-token CE.
        """
        B, L, C = x.shape
        w = self.embed.lm_weight()
        mask_id = self.cfg.tul.slot_id
        chunk = self.cfg.ce_chunk_size
        flat = x.reshape(-1, C)
        lab = labels.reshape(-1)
        BL = flat.shape[0]

        if layout is None:
            ce = fused_linear_cross_entropy(flat, w, lab, ignore_index=-100,
                                            chunk_size=chunk, mask_token_id=mask_id)
            return {"ce_main": ce, "n_main": (lab != -100).sum().to(ce.dtype)}

        lab_pad = torch.cat([lab, lab.new_full((1,), -100)], dim=0)
        flat_pad = torch.cat([flat, flat.new_zeros(1, C)], dim=0)
        row_off = (torch.arange(B, device=x.device) * L).unsqueeze(1)
        base = layout.slot_index + row_off
        # t_last is the position immediately before the slot — the layout guarantees a
        # slot never starts at position 0, so base-1 is always a real token position.
        p_idx = torch.where(layout.slot_valid, base - 1, BL).reshape(-1)
        z_idx = torch.where(layout.slot_valid, base + layout.prefix_k - 1, BL).reshape(-1)
        main_lab = lab_pad.scatter(0, torch.cat([p_idx, z_idx], dim=0), -100)[:BL]

        ce_main = fused_linear_cross_entropy(flat, w, main_lab, ignore_index=-100,
                                             chunk_size=chunk, mask_token_id=mask_id)
        out = {"ce_main": ce_main, "n_main": (main_lab != -100).sum().to(ce_main.dtype)}
        for tag, idx in (("plast", p_idx), ("emit", z_idx)):
            labs = lab_pad[idx]
            ce = fused_linear_cross_entropy(flat_pad[idx], w, labs, ignore_index=-100,
                                            chunk_size=chunk, mask_token_id=mask_id)
            out[f"ce_{tag}"] = ce
            out[f"n_{tag}"] = (labs != -100).sum().to(ce.dtype)
        return out

    def _forward_tul(self, input_ids: Tensor, labels: Tensor | None,
                     layout: SlotLayout, plan_nats: bool) -> dict:
        """The TUL forward (docs/tul-spec.md §3). One shared position axis."""
        if self.tul is None:
            raise RuntimeError(
                "forward(slot_layout=...) requires a model built with MORPHConfig(tul=...); "
                "this model has no TUL parameters (E_slot / E_mask / W_prefix)."
            )
        tc = self.cfg.tul
        if layout.prefix_k != tc.prefix_k:
            raise ValueError(f"layout prefix_k {layout.prefix_k} != model {tc.prefix_k}")
        B, L = input_ids.shape

        x, x0, bigram_emb = self._tul_front(input_ids, layout)

        if tc.tokens_through_core:
            # Arm A2 (slots-as-memory): tokens AND slots run the ordinary per-SAMPLE core.
            # RESOLVED SPEC AMBIGUITY — §7.1's A2 row says "Poisson/slot" in the depth
            # column but "uniform depth" in the isolates column. A2 must differ from A0 by
            # the presence of slots ALONE (it isolates C2), so it reuses today's core
            # region unchanged; a per-position Poisson depth would change two things at once.
            x_coda = self._core_region(x, x0, bigram_emb, input_ids)
            depths = None
        else:
            xn, h_slots, depths = self._tul_core(x, x0, bigram_emb, layout)
            values, pos = self.tul.prefix_project(h_slots, layout, L)
            x_coda = scatter_positions(xn, pos, values)

        x_coda, keep = self.tul.apply_token_dropout(x_coda, layout, self.training)

        out: dict = {"logits": None}
        if tc.coda_sees_slots:
            xh = self._back_region(x_coda, x0, bigram_emb, input_ids, inject_keep=keep)
            groups = self._tul_group_losses(xh, labels, layout) if labels is not None else None
            coda_positions = L
        else:
            xh, groups, coda_positions = self._tul_coda_without_slots(
                x_coda, x0, bigram_emb, keep, labels, layout)
        if plan_nats and labels is not None:
            # §7.2: CE over the same tokens with the slots removed from the coda sequence.
            # Reported MINUS the normal token CE; a positive value is the plan actually
            # being used (the h_z-ablation, the C2 number). Under coda_sees_slots=false the
            # normal pass IS the slots-removed pass, so plan_nats is 0 by construction.
            if tc.coda_sees_slots:
                _xh, g_pn, _ = self._tul_coda_without_slots(
                    x_coda, x0, bigram_emb, keep, labels, layout)
                out["ce_tokens_no_slots"] = g_pn["ce_main"]
            else:
                out["ce_tokens_no_slots"] = groups["ce_main"]

        if groups is not None:
            out.update(self._tul_reduce(groups))
        else:
            # Generation (labels=None): full logits, with the structural slot id masked
            # out of the head (spec §3.1 / invariant 4 — "masked … at generation").
            # index_fill is out-of-place, so this is safe under grad as well as no_grad.
            out["logits"] = self.embed.attend(xh).index_fill(
                -1, torch.tensor([tc.slot_id], device=xh.device), float("-inf"))
        out["layer_passes"] = self._tul_layer_passes(layout, depths, coda_positions)
        out["n_tokens"] = (~layout.slot_mask).sum()
        return out

    def _tul_coda_without_slots(self, x_coda, x0, bigram_emb, keep, labels, layout):
        """Run the coda on the TOKEN positions only (arm A4 and the plan-nats metric)."""
        cidx = compact_index(layout.slot_mask)
        B, L = layout.slot_mask.shape

        def _g(t, fill=None):
            if t is None:
                return None
            if fill is None:
                pad = torch.cat([t, t.new_zeros(B, 1, *t.shape[2:])], dim=1)
            else:
                pad = torch.cat([t, t.new_full((B, 1, *t.shape[2:]), fill)], dim=1)
            return gather_positions(pad, cidx)

        xc = _g(x_coda)
        x0c = _g(x0)
        bgc = _g(bigram_emb)
        keepc = _g(keep)
        xh = self._back_region(xc, x0c, bgc, None, inject_keep=keepc)
        groups = None
        if labels is not None:
            labc = _g(labels.unsqueeze(-1), fill=-100).squeeze(-1)
            groups = self._tul_group_losses(xh, labc, None)
        return xh, groups, L

    @staticmethod
    def _tul_reduce(g: dict) -> dict:
        """Weighted-mean reduction of the grouped CEs → the training loss + §7.2 metrics."""
        n_main, ce_main = g["n_main"], g["ce_main"]
        if "ce_emit" not in g:                      # no slots in the sequence (A4 / plan-nats)
            return {"loss": ce_main, "ce_tokens": ce_main, "n_targets": n_main}
        ce_p, n_p = g["ce_plast"], g["n_plast"]
        ce_z, n_z = g["ce_emit"], g["n_emit"]
        denom = n_main + 0.5 * (n_p + n_z)
        loss = (ce_main * n_main + 0.5 * (ce_p * n_p + ce_z * n_z)) / denom.clamp(min=1.0)
        ce_tokens = (ce_main * n_main + ce_p * n_p) / (n_main + n_p).clamp(min=1.0)
        return {
            "loss": loss,
            "ce_tokens": ce_tokens,               # → val/ppl_tokens
            "ce_first_tok": ce_z,                 # → val/first_tok_ce
            "ce_first_tok_plain": ce_p,
            "first_tok_counterfactual": ce_p - ce_z,   # → val/first_tok_counterfactual
            "n_targets": denom,
        }

    def _tul_layer_passes(self, layout: SlotLayout, depths: Tensor | None,
                          coda_positions: int) -> Tensor:
        """Total layer-passes in this batch (spec §2: 10.3 vs 44 at OWT span 19.2).

        prelude and coda run on every position; the core runs ``depth`` times on each
        REAL slot (or, for arm A2, on every position at the sampled per-sample depth).
        The caller divides by ``n_tokens`` to get the headline number.
        """
        cfg = self.cfg
        L = layout.l_total
        B = layout.slot_mask.shape[0]
        passes = torch.tensor(float(cfg.n_prelude * L * B + cfg.n_coda * coda_positions * B),
                              device=layout.slot_mask.device)
        if depths is None:                                   # arm A2: core over all positions
            passes = passes + float(cfg.n_core * L * B * cfg.mean_depth)
        else:
            passes = passes + cfg.n_core * (depths * layout.slot_valid).sum()
        return passes

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, input_ids: Tensor, labels: Tensor | None = None,
                bag_size: int = 0, seq_lens: Tensor | None = None,
                slot_layout: SlotLayout | None = None) -> dict:
        return self._forward_single(input_ids, labels, bag_size, seq_lens, slot_layout)

    def tul_forward_with_plan_nats(self, input_ids: Tensor, labels: Tensor,
                                   slot_layout: SlotLayout) -> dict:
        """Eval-only: also run the coda with the slots gathered out (spec §7.2).

        A separate entry point rather than a forward flag — the training path must not
        carry a branch that decides how much work to do (CONTRIBUTING: no runtime
        feature flags in hot paths), and this doubles the coda cost.
        """
        return self._forward_single(input_ids, labels, 0, None, slot_layout, _plan_nats=True)

    def _forward_single(self, input_ids: Tensor,
                        labels: Tensor | None = None,
                        bag_size: int = 0,
                        seq_lens: Tensor | None = None,
                        slot_layout: SlotLayout | None = None,
                        _plan_nats: bool = False) -> dict:
        if slot_layout is not None:
            if bag_size > 0:
                raise ValueError(
                    "slot_layout and bag_size are mutually exclusive: TUL activates AT the "
                    "TST switch (spec §5), and val/gen always run TUL on with bag_size 0 "
                    "(invariant 6)."
                )
            return self._forward_tul(input_ids, labels, slot_layout, _plan_nats)
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
            _bg_raw = self.embed.get_bigram(input_ids)
            bigram_emb = (_bg_raw.view(B, T, s, -1).mean(dim=2)                          # [B,L,d]
                          if _bg_raw is not None else None)
            n_ve = len(self._ve_layer_map)
            ve_bagged = ([
                self.value_embeds[k]
                    .precompute(self.value_embed_tables[k](input_ids))
                    .view(B, T, s, -1).mean(dim=2)                                       # [B,L,ctx_w]
                for k in range(n_ve)
            ] if n_ve > 0 else None)
            x, x0 = self._front_tail(x, input_ids, bigram_emb, ve_bagged)
        else:
            T = T_in
            _sg = self._static_graphs
            if (_sg.get("front") is not None and self.training
                    and torch.is_grad_enabled()
                    and input_ids.shape == _sg["front_shape"]):
                # Graphed FRONT replay (2 launches: input copy + cudaGraphLaunch).
                outs = _sg["front"](input_ids)
                _am = _sg["front_aux_mods"]
                if _am:
                    # Region routers' aux arrives as explicit graph OUTPUTS (the python
                    # stash protocol does not re-run on replay) → re-stash each onto
                    # its own module so collect_routing_aux_losses sums in the exact
                    # eager order (per-region pre-summing was a measured fp-reassoc).
                    n = len(_am)
                    for _mod, _aux in zip(_am, outs[-n:]):
                        _mod._last_aux_loss = _aux
                    outs = outs[:-n]
                if _sg["has_bigram"]:
                    x, x0, bigram_emb = outs
                else:
                    x, x0 = outs
                    bigram_emb = None
            else:
                x, x0, bigram_emb = self._front_region(input_ids)

        x = self._core_region(x, x0, bigram_emb, input_ids)

        # ── Coda + LM head (BACK region — graphed replay when captured) ──
        _sg = self._static_graphs
        if (_sg.get("back") is not None and self.training and torch.is_grad_enabled()
                and s == 0 and x.shape == _sg["back_shape"]):
            outs = (_sg["back"](x, x0, bigram_emb) if _sg["has_bigram"]
                    else _sg["back"](x, x0))
            _am = _sg["back_aux_mods"]
            for _mod, _aux in zip(_am, outs[1:]):
                _mod._last_aux_loss = _aux   # per-module re-stash (exact eager order)
            x = outs[0]
        else:
            x = self._back_region(x, x0, bigram_emb, input_ids)

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
            loss = ce_loss
            out = {"logits": None, "loss": loss}
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
                loss = ce_loss
                out["loss"] = loss

        return out
