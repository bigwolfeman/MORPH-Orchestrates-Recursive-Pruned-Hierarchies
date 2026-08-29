"""MORPH Transformer — Parcae-style looped architecture with all features baked in.

Architecture: prelude → core×T (diagonal injection) → coda
Loop hierarchy:
  Inner: Parcae core loop (T iterations with Poisson depth sampling)
  Outer: (Zyphra RSA — deferred, inference-time, requires RL)

All features always on. No runtime if-statements in the forward pass.
Config determines dimensions and sizes, not whether features exist.
"""

from __future__ import annotations

import math
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
from .sigreg import sigreg_epps_pulley
from .sparsity import MortarLinear
from .tul import (TULConfig, TULGate, TULGateConfig, TULSlots, compact_index,
                  cw2_retain_mask, gather_positions, gather_valid, mux_span_targets,
                  scatter_positions,
                  window_drop_mask)
from .tul_layout import SlotLayout, tg_allow_mask, tg_reset_mask

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

    # ── SCSE Stage 1 (arXiv:2607.27656) — the loop's initial deviation ──────────
    # MORPH starts the core loop at ``h_0 = e``, so with the natural input-conditioned
    # anchor ``h* = e`` the initial deviation is EXACTLY zero. The paper's Theorem 2 then
    # makes the whole loop trajectory the propagated forcing response
    # ``Delta_T = sum_k Phi_E(T, k+1) b_k(e)``, with nothing bounding a quantity Corollary 5
    # shows can grow like ``rho^T``. This scale gives the loop a state of its own:
    #     h_0 = e + core_init_scale * H_0(e)          (Listing 1's `init_delta_proj`)
    # 0.0 → the ``_CloneInit`` path, which has NO parameters, draws NO RNG at build, and is
    # bit-identical to the old ``h = e.clone()``. The paper uses 0.1.
    # Follows the `tul` / `retention` convention: the value gates CONSTRUCTION, never a
    # forward branch, so torch.compile still sees a straight-line graph.
    core_init_scale: float = 0.0

    # ── SCSE — Source-Centered State Evolution (docs/scse-spec.md) ───────────────
    # The FULL method of arXiv:2607.27656, not the Stage 1 initial-deviation probe above.
    # The abstract credits the gain to "the learned anchor and the anchor-coordinate
    # deviation recurrence", which are precisely the two things `core_init_scale` does NOT
    # implement, so these are separate switches and enabling both RAISES.
    #     h*        = e + scse_anchor_scale * a_omega(e)          built ONCE, held fixed
    #     Delta_0   = scse_init_scale * init_proj(e) - scse_anchor_scale * a_omega(e)
    #     Delta_t+1 = Delta_t + 1{||Delta_t||_F^2 > eps} * s * G_theta(Delta_t)
    #     h_T       = h* + Delta_T
    #     G_theta(D) = stack(D) - D    <-- the SUBTRACTION is load-bearing; see _SCSE.update
    # `stack` is the core block stack with NO source injection: the source enters through
    # the anchor once instead of on every iteration. `stack` carries its OWN residual, so
    # feeding stack(D) straight in would apply a residual TWICE (~1.41x gain per iteration).
    # Construction-time
    # only, like `tul` and `retention` — never a forward branch.
    scse_enabled: bool = False
    scse_step_scale: float = 0.5       # s — the paper's value for a ONE-block core (spec D7)
    scse_anchor_scale: float = 0.1     # Listing 1 default
    scse_init_scale: float = 0.1       # Listing 1 default
    scse_eps: float = 1.0e-8           # zero-deviation mask threshold
    scse_kappa: float = 0.0            # 0 → SCSE proper; > 0 builds cond_proj (SC-Cond control)
    # What the core map RECEIVES each loop iteration.
    #   "deviation" — Delta alone. SCSE proper, and what the 2026-08-25 arm ran.
    #   "state"     — h* + Delta, the full-size state; the update then accumulates the
    #                 CHANGE the core made to it. Arm C.
    # Why the choice exists: MORPH's blocks are PRE-NORM, so RMSNorm divides out the size
    # of the core's input and the output size comes from the weights. Measured on trained
    # weights (lab/divergence/scale_probe.py, onset-capture/ROLL_step_1750): shrinking the
    # input 1000x moves the output 31 %, and ||stack(D)||/||D|| goes 1.79 -> 1235. So a
    # deliberately small Delta comes back at the map's own scale after ONE iteration. The
    # "can only damp" argument in _SCSE.update holds at a normal-size D and fails at a
    # small one, which is why the deviation grew 230x in the first iteration.
    scse_input_mode: str = "deviation"
    # Cap on the deviation's per-example RMS. 0.0 = no cap. Bounds how far Delta travels;
    # it does NOT stop Delta changing, which is what makes the loop iterate.
    scse_delta_clip: float = 0.0

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
    # The CORE's HCA ratio, when it must differ from the rest of the stack. None inherits
    # `hca_compress_ratio` and is bit-identical to not having this field.
    #
    # It exists because the looped core does NOT run at the stack's sequence length. Under
    # TUL the core loops over SLOT positions — 64 with `tul.max_slots: 64` — while prelude
    # and coda run on all 1152. `GatedPoolCompressor` computes `n_blocks = S // m`, so a
    # ratio sized for the token stream floors to ZERO on the slot path and the compressed
    # branch produces nothing at all, silently, for a whole run. Measured 2026-08-25:
    # `|out_comp|` is exactly 0.0000 on core blocks 1/3/5 while the gate still spends
    # ~0.50 of its mixture on that zero tensor. See
    # `.agents/notes/proposed/bug-fix/2026-08-25-hca-compressed-branch-dead-on-slot-path.md`.
    #
    # Scoped to the core on purpose: setting `hca_compress_ratio` globally would also
    # re-block prelude and coda, which do not have the problem.
    core_hca_compress_ratio: int | None = None
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

    # ── FM1: the flow-matching planner arm (morph/model/tul_fm.py) ────────
    # None → NO planner is constructed and every path is byte-identical (the `tul`
    # convention). Set it and the model gains an FMPlanner whose Euler ladder produces
    # the slot states in place of the core loop. Requires `tul` (FM1 is a TUL arm) and
    # `n_core == 0` (the core loop is what FM1 removes) — both checked at construction.
    fm: "FMArmConfig | None" = None

    # L1 core-gain governor: cap the per-iteration looped-core
    # amplification ‖h_new‖/‖h_a‖ (per sample) to this ratio τ. The HC residual is
    # norm-preserving (gain≈1 healthy) so this is IDENTITY in the healthy regime and only
    # shrinks the runaway-gain step that the weight-shared core amplifies T× (the β1=0
    # gain runaway mode). 0.0 = OFF (bit-identical to baseline). Typical τ≈1.5–2.0.
    core_gain_clip: float = 0.0
    # WHICH loop iterations the governor applies to, inclusive, 0-indexed by iteration.
    # (0, -1) — the default — means every iteration and is exactly the behaviour above.
    # -1 as the upper bound means "no upper bound".
    #
    # This exists because the governor's cap is not applied where anyone assumed. Measured
    # on the divergent control (lab/experiments/results/2026-08-23-tul-onset-ordering.md):
    # the realized per-iteration gain is 1.422 at t=0 and 1.08–1.13 at t=1..7, so a typical
    # τ≈1.5 can only ever bind on the FIRST iteration. Selecting the range makes that
    # testable instead of assumed — see
    # lab/experiments/planned/2026-08-23-tul-iteration0-mediation.md.
    core_gain_clip_iter_lo: int = 0
    core_gain_clip_iter_hi: int = -1


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


class _CloneInit(nn.Module):
    """``h_0 = e`` — MORPH's historical loop entry, as a module.

    Exists so the forward path is ``h = self.core_init(e)`` with no flag read and no
    branch, which is what keeps the compiled graph straight-line (Design Principles: "No
    runtime feature flags"). It holds no parameters, so building it advances the RNG
    stream by nothing and a baseline model's weights stay byte-identical to a model built
    before this module existed.
    """

    def forward(self, e: Tensor) -> Tensor:
        return e.clone()


class _SCSEInit(nn.Module):
    """``h_0 = e + s * H_0(e)`` — SCSE Listing 1's ``init_delta_proj``, s = 0.1.

    The point is narrow and structural: it makes ``Delta_0 = h_0 - e`` NON-ZERO. At
    ``Delta_0 = 0`` the paper's bias-subtracted counterfactual is identically zero by
    induction, so the entire deviation trajectory IS the propagated forcing response and
    there is no off-anchor computation to preserve.

    ``bias=True`` matches the paper's reference implementation rather than MORPH's
    core-wide ``bias=False`` convention. That is deliberate: this projection runs ONCE at
    loop entry, not inside the recurrence, so it is not part of the ``G_theta(0) = 0``
    surface that Stage 3's zero-deviation mask depends on.
    """

    def __init__(self, d_model: int, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, e: Tensor) -> Tensor:
        return e + self.scale * self.proj(e)


class _SCSE(nn.Module):
    """Source-Centered State Evolution — the FULL method. Spec: ``docs/scse-spec.md``.

    Paper: "Looped Transformers with Source-Centered State Evolution", arXiv:2607.27656,
    Kim, Hayashi, Kamiya, Koyama, Iwasawa, Matsuo, 30 July 2026. Reference implementation
    is its Listing 1; the equation numbers below are the paper's.

    This holds the two learned modules and the four constants. It deliberately does NOT
    own the loop: the recurrence lives in ``_core_region`` / ``_tul_core`` because MORPH's
    per-sample Poisson depth, active-set shrinking and truncated-BPTT window all have to
    apply to the deviation exactly as they apply to the carrier today (spec D5).

    ``bias=False`` on both projections, where Listing 1's ``nn.Linear`` defaults to True
    (spec D2). Under TUL, ``gather_valid`` zeroes pad slots, so ``e = 0`` there; a bias
    would put ``h*`` and ``Delta_0`` off zero at pads and give padding a forward effect.
    """

    def __init__(self, d_model: int, *, step_scale: float, anchor_scale: float,
                 init_scale: float, eps: float, kappa: float,
                 input_mode: str = "deviation", delta_clip: float = 0.0):
        super().__init__()
        if input_mode not in ("deviation", "state"):
            raise ValueError(f"scse_input_mode must be 'deviation' or 'state', got {input_mode!r}")
        self.input_mode = str(input_mode)
        self.delta_clip = float(delta_clip)
        self.step_scale = float(step_scale)
        self.anchor_scale = float(anchor_scale)
        self.init_scale = float(init_scale)
        self.eps = float(eps)
        self.kappa = float(kappa)
        self.anchor_proj = nn.Linear(d_model, d_model, bias=False)   # a_omega
        self.init_proj = nn.Linear(d_model, d_model, bias=False)     # init_delta_proj / H_0
        # SCSE proper sets cond_proj=None and kappa=0 (Listing 1 caption). A non-zero kappa
        # builds the paper's source-conditioned anchor-coordinate (SC-Cond) reference, whose
        # core is NOT zero-preserving — the mask then supplies the boundary condition.
        self.cond_proj = nn.Linear(d_model, d_model, bias=False) if kappa != 0.0 else None

    def entry(self, e: Tensor) -> tuple[Tensor, Tensor]:
        """``(h*, Delta_0)`` for one forward. The ONLY way the loop is entered.

        ``h* = e + anchor_scale * a_omega(e)`` (Eq. 2), and
        ``Delta_0 = H_0(e) - h*`` with ``H_0(e) = e + init_scale * init_proj(e)``.

        Both come from one method, and ``a_omega(e)`` is evaluated ONCE and reused, for two
        reasons. It halves the projection cost, and it makes "the anchor is built exactly
        once per forward" checkable by counting calls to ``anchor_proj`` (invariant S2) —
        with a separate ``anchor()`` / ``initial_deviation()`` pair the honest count was two
        and the invariant could not be stated crisply.

        ``Delta_0`` is formed as ``init_scale*init_proj(e) - anchor_scale*anchor_proj(e)``,
        never as the literal ``H_0(e) - h*``: the ``e`` terms cancel exactly in real
        arithmetic, and subtracting two bf16 tensors of the carrier's magnitude to recover a
        quantity ~20x smaller would throw away most of its significant bits (spec D8). It
        also makes ``Delta_0`` EXACTLY zero wherever ``e`` is zero, which is what keeps TUL
        pad slots off the forward path (invariant S8).
        """
        a = self.anchor_scale * self.anchor_proj(e)
        return e + a, self.init_scale * self.init_proj(e) - a

    def recurrent_input(self, delta: Tensor, h_star: Tensor) -> Tensor:
        """What ``G_theta`` actually receives.

        ``"deviation"`` — the deviation ALONE. SCSE proper.
        ``"state"`` — ``h* + Delta``, the full-size state (arm C). The core then always sees
        an input at the scale it was trained on, so the pre-norm cannot erase the deviation;
        :meth:`update` subtracts the SAME tensor back off, so what accumulates into Delta is
        the CHANGE the core made. Delta still changes every iteration — it is the only thing
        that does — so the loop still iterates.
        """
        base = (h_star + delta) if self.input_mode == "state" else delta
        if self.cond_proj is None:
            return base
        return base + self.kappa * self.cond_proj(h_star)

    def update(self, delta: Tensor, stack_out: Tensor, rec_in: Tensor | None = None) -> Tensor:
        """``Delta_{t+1} = Delta_t + m * s * G(Delta_t)`` with ``G(D) = stack(D) - D``.

        THE SUBTRACTION IS NOT COSMETIC, and getting it wrong was a real bug in the first
        version of this port (found by audit, 2026-08-25).

        The paper's ``G_theta`` carries NO top-level identity: its residual has been hoisted
        to loop level, which is what "residual step scale" names. Proof from the paper's own
        text rather than from taste — the tuned adapter is
        ``h_{t+1} = h_t + s*B_theta(h_t + alpha*W_in*h*)``, and if ``B_theta`` contained the
        identity that map would gain ``(1+s)`` every step and reach ``1.5^48 ~ 1e8`` at the
        T = 48 the paper evaluates at. Its T = 48 numbers are ordinary.

        MORPH's core blocks are full residual blocks — the HyperConnection carrier passthrough
        is INSIDE ``stack`` — so ``Delta + s*stack(Delta)`` applies the residual twice.
        Measured on a real checkpoint at the converged operating point:
        ``cos(stack(D), D) = 0.88`` and ``||stack(D)||/||D|| = 0.90``, i.e. ``stack`` is
        essentially "identity plus an update of about half the size". The doubled form gains
        **1.414x per iteration** (16x over eight); this form gains **0.923x**.

        Subtracting ``delta`` restores the LOOP-LEVEL residual structure. It APPROXIMATES
        the paper's ``B_theta``; it is not an equivalence, and spec section 3.2 says so.
        MORPH's HyperConnection carry is an orthogonal Cayley MIX ``M``, not the identity,
        so ``stack(D) = M(D) + U(D)`` and ``stack(D) - D = U(D) + (M - I)(D)`` where the
        paper's update is ``U`` alone. Measured at init the mixers sit about 2 % off
        identity, so the extra term is small; and it is SAFE in the useful direction --
        the resulting carry ``(1 - s)I + s*M`` has every eigenvalue of magnitude <= 1 for
        orthogonal ``M``, so it can only DAMP the deviation, never expand it. Equivalently
        ``Delta_{t+1} = (1 - s)*Delta_t + s*stack(Delta_t)``, so ``s`` is a damping factor
        between "no update" (s = 0) and MORPH's own core map in deviation coordinates
        (s = 1). Zero-preservation survives: ``stack(0) = 0`` gives ``G(0) = 0``.

        Both loop bodies AND the drift probe call THIS method. A previous version inlined the
        arithmetic in three places, and an audit removed the mask from one of them without a
        single test failing.
        """
        base = delta if rec_in is None else rec_in
        d = delta + self.gate(delta) * (self.step_scale * (stack_out - base))
        return self.clip(d)

    def clip(self, delta: Tensor) -> Tensor:
        """Cap the deviation's per-EXAMPLE RMS at ``delta_clip``. 0.0 = off, and then this
        returns the tensor unchanged so the baseline graph is untouched.

        Per example, matching :meth:`gate`'s reduction: a per-position or per-stream cap
        would be a different method. Only ever SHRINKS (``min(1, cap/rms)``), so it can
        never inflate a deviation that is already small.
        """
        if self.delta_clip <= 0.0:
            return delta
        dims = tuple(range(1, delta.dim()))
        rms = delta.float().pow(2).mean(dim=dims, keepdim=True).sqrt()
        scale = (self.delta_clip / rms.clamp_min(1e-12)).clamp(max=1.0)
        return delta * scale.to(delta.dtype)

    def gate(self, delta: Tensor) -> Tensor:
        """``m_{b,t} = 1{ ||Delta_t^{(b)}||_F^2 > eps }`` (Eq. 4) — per EXAMPLE.

        Listing 1 reduces over ``dim=(1, 2)`` of a ``[B, S, C]`` tensor, i.e. everything
        except the batch axis, and the paper's text says "The per-example mask". MORPH's
        carrier carries an extra HyperConnection stream axis, so the reduction is over every
        axis except 0 (spec D1); reducing over fewer would silently make this per position
        or per stream, which is a different method.

        Accumulated in fp32 (spec D4): the training path runs bf16 autocast and a sum of
        ~1.2e7 squares in bf16 cannot support an ``eps = 1e-8`` comparison.
        """
        dims = tuple(range(1, delta.dim()))
        nsq = delta.float().pow(2).sum(dim=dims, keepdim=True)
        return (nsq > self.eps).to(delta.dtype)


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

    # Operating-point capture for the core-map Jacobian probe
    # (morph/training/core_jacobian.py). `None` — the default — makes every capture site
    # a Python-level no-op, so the forward is bit-identical when the probe is off. Set to
    # a list to collect one dict per core-loop iteration.
    _jac_capture: list | None = None

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

        # ── TG restriction (docs/tul-tg-spec.md) ────────────────────────────
        # Construction-time only: gates which attention modules get built (§3) and
        # what `_forward_tul` threads into every prelude/coda call. Validated HERE,
        # before anything else is built, so a bad config never gets partway through
        # constructing a model it is going to refuse.
        self._tg_restrict = bool(cfg.tul.tg_restrict) if cfg.tul is not None else False
        if self._tg_restrict and cfg.use_kernels:
            raise ValueError(
                "model.tul.tg_restrict=true requires model.use_kernels=false "
                "(docs/tul-tg-spec.md §2/§6): the TG arms run eager only — the fused "
                "window/CSA/HCA kernels do not know about the restriction, and a "
                "silent unmasked kernel path is forbidden.")

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
            tg_restrict=self._tg_restrict,
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

        # The core alone may re-block its HCA branch; see `core_hca_compress_ratio`.
        core_attn_kw = dict(attn_kw)
        if cfg.core_hca_compress_ratio is not None:
            core_attn_kw["hca_compress_ratio"] = int(cfg.core_hca_compress_ratio)

        def _make_block(layer_idx: int, kw: dict | None = None) -> MORPHBlock:
            return MORPHBlock(
                norm_attn=RMSNorm(d),
                attn=MORPHAttention(layer_idx=layer_idx, **(kw or attn_kw)),
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
            _make_block(cfg.n_prelude + i, core_attn_kw)
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
        # Auxiliary-objective gates (arm: warmup schedules). Non-persistent so no
        # checkpoint gains a key; the trainer writes them each step, and because
        # they are BUFFERS not Python floats, flipping one costs no recompile.
        self.register_buffer("mux_gate", torch.ones(()), persistent=False)
        self.register_buffer("sigreg_gate", torch.ones(()), persistent=False)


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
        # The gate is built AFTER TULSlots for the same reason and with the same
        # discipline: every one of its inits is a deterministic zero/one, so building it
        # advances the RNG stream by nothing and arm TUL-gate's base weights are
        # byte-identical to arm A1's (docs/tul-gate-spec.md §9 invariant 1).
        _gc = cfg.tul.gate if cfg.tul is not None else None
        self.tul_gate: TULGate | None = TULGate(d, _gc) if _gc is not None else None

        # ── FM1 planner (morph/model/tul_fm.py) ────────────────────────────
        # Built LAST so a non-FM model's weights are byte-identical to today's: every
        # parameter drawn below advances the global RNG, so any earlier placement would
        # change the baseline's own initialisation.
        self.fm_planner = None
        self._fm_schedule = None
        self._fm_loss_scale = 1.0
        if cfg.fm is not None:
            self._build_fm(cfg, d)

        # ── SCSE Stage 1 loop entry ────────────────────────────────────────
        # Built LAST, for the same reason TULSlots is: `_CloneInit` draws no RNG at all,
        # and `_SCSEInit` draws its Linear init AFTER every other parameter, so a baseline
        # model and an SCSE model built from the same seed share byte-identical weights
        # everywhere except this projection. The arms then differ by the mechanism alone.
        self.core_init: nn.Module = (
            _SCSEInit(d, cfg.core_init_scale) if cfg.core_init_scale > 0.0
            else _CloneInit())

        # ── SCSE, the full method (docs/scse-spec.md) ──────────────────────────────
        # Also built LAST, and after `core_init`, for the same RNG-neutrality reason: with
        # `scse_enabled: false` NO parameter is created and NO RNG is drawn, so a control
        # model's weights stay byte-identical to master (invariant S1).
        if cfg.scse_enabled and cfg.core_init_scale > 0.0:
            raise ValueError(
                "model.scse_enabled and model.core_init_scale are mutually exclusive. SCSE "
                "defines its own initial state (Delta_0 = H_0(e) - h*, spec section 2); "
                "core_init_scale is the Stage 1 probe that sets h_0 only and was measured "
                "0.815 nats WORSE (lab/experiments/failures/"
                "2026-08-25-scse-stage1-initial-deviation.md). Set core_init_scale=0.0.")
        if cfg.scse_enabled and cfg.core_gain_clip > 0.0:
            raise ValueError(
                f"model.scse_enabled with core_gain_clip={cfg.core_gain_clip} (spec D6). The "
                "governor caps ||h_new||/||h_old|| on the LOOP CARRIER, which under SCSE is "
                "the deviation, not the state — the same tau means a different constraint. "
                "Set core_gain_clip=0.0, or extend the governor deliberately and update D6.")
        if cfg.scse_enabled and cfg.n_core == 0:
            raise ValueError(
                "model.scse_enabled with n_core=0: there is no core loop to reparameterise.")
        self.scse: _SCSE | None = (
            _SCSE(d, step_scale=cfg.scse_step_scale, anchor_scale=cfg.scse_anchor_scale,
                  init_scale=cfg.scse_init_scale, eps=cfg.scse_eps, kappa=cfg.scse_kappa,
                  input_mode=cfg.scse_input_mode, delta_clip=cfg.scse_delta_clip)
            if cfg.scse_enabled else None)

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
                         ret_state=None, iter_idx=0, inj_terms=None, source_free=False):
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
        # `source_free` is SCSE's G_theta (docs/scse-spec.md section 3.2): the shared block
        # stack with NO source entering the recurrence. Both injections are skipped, not fed
        # zeros — feeding e = 0 would leave DiagonalInjection's `h_ctx <- A*h_ctx` decaying
        # the deviation's context channels by ~0.447 per iteration with nothing to refill
        # them (spec D3). It is a Python bool that is constant per call site, so it traces
        # out and the baseline graph is unchanged.
        h_injected = h_in if source_free else self.injection(h_in, e_in)
        ret_cap = {} if self._core_has_retention else None
        for i, layer in enumerate(self.core):
            gi = np_ + i
            if not source_free:
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
                    ve_bagged, attn_kwargs: dict | None = None,
                    ret_reset_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """x0 skip-clone → HC stream expansion → prelude blocks. Returns (x, x0).

        ``attn_kwargs`` / ``ret_reset_mask`` (docs/tul-tg-spec.md §§1-4): the SAME
        tg_allow / slot-mask / GLA-reset-mask dict, built ONCE per forward, threaded
        into every prelude block. None on every non-TG path → the calls below are
        exactly ``layer(x)`` as before (bit-identical, spec T4).
        """
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
            x = layer(x, attn_kwargs=attn_kwargs, ret_reset_mask=ret_reset_mask)
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
                     inject_keep: Tensor | None = None,
                     attn_kwargs: dict | None = None,
                     ret_reset_mask: Tensor | None = None) -> Tensor:
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
        ops below are unchanged and bit-identical.

        ``attn_kwargs`` / ``ret_reset_mask``: see :meth:`_front_tail` — the same
        per-forward tg_allow / slot-mask / GLA-reset-mask dict, threaded into every
        coda block. Only meaningful for the FULL-``L`` coda call (``coda_sees_slots
        and coda_token_cut == 0``); the gathered-subset coda callers never pass these
        (docs/tul-tg-spec.md does not define the restriction on a gathered index
        space — see ``_forward_tul``'s raise for that combination)."""
        for i, layer in enumerate(self.coda):
            gi = self.cfg.n_prelude + self.cfg.n_core + i
            term = self._build_injection_term(
                gi, self.x0_injects[gi].precompute(x0), input_ids, bigram_emb, x.dtype
            )
            if inject_keep is not None:
                term = term * inject_keep.to(term.dtype)
            x = self._apply_injection(x, term)
            x = layer(x, attn_kwargs=attn_kwargs, ret_reset_mask=ret_reset_mask)

        return self._readout(x)

    def _readout(self, x: Tensor) -> Tensor:
        """HC stream mean → lm_mixer → final_norm.

        Pure code motion out of the tail of :meth:`_back_region` (the ``_core_region``
        precedent): the ops and their order are IDENTICAL, so every existing path stays
        bit-identical.
        """
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

    def prelude_states(self, input_ids: Tensor, apply_input_norm: bool = True) -> Tensor:
        """``[B, L, d_model]`` FEATURE READ-OUT after the prelude. Adds no behaviour.

        Written for TUL-FM P1 (``lab/tulfm/``), which trains a separate planner on a
        FROZEN backbone and needs the backbone's states over the context. Nothing in the
        training or inference path calls it, so every existing path stays bit-identical:
        this is a new public entry point that reuses :meth:`_front_region`, the same
        boundary norm ``_core_region`` applies, and the same stream reduction
        :meth:`_readout` applies.

        With ``apply_input_norm=True`` on an ``n_core == 0`` model (arm A3), the returned
        tensor is EXACTLY what the coda consumes: ``_core_region`` reduces to
        ``self.input_norm(x)`` there. On a model with a core it is the state the core
        loop starts from, before any iteration.

        The HC carrier is reduced by the mean over streams — the same scale-preserving
        readout :meth:`_readout` uses — so the result is single-stream ``[B, L, d_model]``
        whatever ``hc_streams`` is.

        Raises in ``train()`` mode: embed/MLP dropout would make the "frozen features"
        stochastic, and a silently-noisy feature is worse than a missing one.
        """
        if self.training:
            raise RuntimeError(
                "prelude_states() is a frozen-feature read-out and must run in eval mode "
                "(dropout would make the features stochastic). Call model.eval() first.")
        x, _x0, _bigram = self._front_region(input_ids)
        if apply_input_norm:
            x = self.input_norm(x)
        if self._is_hc:
            x = x.mean(dim=2)
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
            # SCSE (docs/scse-spec.md): a Python-level constant, so every branch on it below
            # is resolved at trace time and the non-SCSE graph is unchanged.
            _scse = self.scse
            with _prof("carrier::h_clone"):
                if _scse is None:
                    h = self.core_init(e)
                    h_star = None
                else:
                    # THE LOOP CARRIER IS NOW THE DEVIATION. h* is built ONCE here and is
                    # never recomputed inside the loop (invariant S2); the absolute state is
                    # reconstructed as h* + Delta_T at the single loop exit below (S6).
                    h_star, h = _scse.entry(e)

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
            # SCSE runs a source-free core (spec D3), so neither stack is ever read. Skip
            # BUILDING them too: they are n_core full [B,S,*] projections per forward, and
            # constructing tensors only to leave them unused would be a real cost, not a
            # cosmetic one. `_inj_none` keeps the `args` tuple below a fixed 3-tuple so the
            # checkpoint call sites are identical in both modes.
            _inj_none = h.new_zeros(0)
            if _scse is not None:
                x0_core_terms = inj_core_terms = _inj_none
            else:
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
                if _scse is None:
                    return self._apply_core_step(h_in, e_in, None, None, None,
                                                 ret_state=ret_state, iter_idx=iter_idx,
                                                 inj_terms=inj_terms)
                # ── SCSE, Eqs. 3-5 ──────────────────────────────────────────────────
                # `h_in` IS Delta_t and `e_in` carries h* (used only when kappa > 0 builds
                # the SC-Cond reference; SCSE proper ignores it). The signature is kept
                # byte-for-byte so the checkpoint / no_grad / eager call sites below, and
                # the truncated-BPTT window they implement, are untouched.
                # `_rec` is bound once and handed to BOTH calls: whatever went INTO the
                # core is what `update` must subtract back off, or the two disagree the
                # moment `scse_input_mode` is not "deviation".
                _rec = _scse.recurrent_input(h_in, e_in)
                g_out, new_ret = self._apply_core_step(
                    _rec, None, None, None, None,
                    ret_state=ret_state, iter_idx=iter_idx, inj_terms=None, source_free=True)
                return _scse.update(h_in, g_out, _rec), new_ret    # Eqs. 3-5

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
                # Under SCSE `e_s` carries the ANCHOR, not the source: the loop's second
                # positional argument is what `_core_step` forwards to `recurrent_input`,
                # and SCSE's recurrence never sees `e` again after Delta_0 and h* are built.
                e_s = (h_star if _scse is not None else e)[perm]
                # ids_s / bg_s / x0_s gathers are gone: the injection is precomputed
                # (inj_core_terms) and only IT needs sorting into active-set order. This also
                # drops 3 gather kernels/step (input_ids, bigram, x0-stack) from the hot loop.
                inj_s = _inj_none if _scse is not None else inj_core_terms[:, perm]

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
                # Under SCSE the carrier is the deviation, so the absolute pre-loop state is
                # h* + Delta_0 (= e_s + h_s here). Recording the raw carrier would hand the
                # forecastability probe a different quantity under a different name.
                _z0 = (e_s + h_s) if _scse is not None else e_s
                _traj.append((_z0.mean(dim=2) if self._is_hc else _z0).detach())  # z_0 (pre-loop)
            for t in range(total_iters):
                n_active = active_counts[t]
                if n_active == 0:
                    break
                h_a = h_s[:n_active]
                # inj_s[:, :n_active]: the precomputed injection sliced to the active prefix
                # (per-sample terms, no cross-sample mixing → slicing is exact). Passed as a
                # checkpoint input so backward recompute reuses it instead of rebuilding.
                args = (h_a, e_s[:n_active],
                        _inj_none if _scse is not None else inj_s[:, :n_active])
                rs_a = ret_state_s[:n_active] if track_ret else None
                # Jacobian probe capture — see the twin in `_tul_core`. None by default,
                # so this branch traces out and the forward stays bit-identical.
                if self._jac_capture is not None:
                    self._jac_capture.append({
                        "h": h_a.detach(), "e": args[1].detach(), "inj": args[2].detach(),
                        "ret_state": None if rs_a is None else rs_a.detach(),
                        "iter_idx": t,
                        "active": h_a.new_ones(h_a.shape[:2], dtype=torch.bool),
                        # Under SCSE "h" is the DEVIATION and "e" is the ANCHOR. Any probe
                        # that reads these as (state, source) describes the wrong operator,
                        # so the mode travels WITH the data instead of being assumed.
                        "scse": _scse is not None,
                    })

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
                if _tau > 0.0 and self._clip_applies(t):
                    _in_n = h_a.flatten(1).norm(dim=1)
                    _out_n = h_new.flatten(1).norm(dim=1)
                    _scale = torch.clamp(_tau * _in_n / (_out_n + 1e-6), max=1.0)
                    h_new = h_new * _scale.view(-1, *([1] * (h_new.dim() - 1)))

                if _capture_traj:  # eval-only interp: capture EVERY iteration's carrier (z_1..z_T)
                    # Eval runs a UNIFORM depth, so perm is the identity and n_active is the
                    # full batch (see the comment where _capture_traj is read); e_s therefore
                    # aligns with h_new row for row and h* + Delta is the absolute state.
                    _zt = (e_s + h_new) if _scse is not None else h_new
                    _traj.append((_zt.mean(dim=2) if self._is_hc else _zt).detach())

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
                if _scse is not None:
                    # Eq. 5 tail: h_T = h* + Delta_T (invariant S6). The deviation exists
                    # ONLY between the two lines marked S2/S6 — everything downstream (coda,
                    # readout, scatter, gate head, every checkpoint key) sees the absolute
                    # carrier exactly as it does today. h_star is already in batch order.
                    x = h_star + x
        else:
            # n_core == 0 (seed models): the loop path hands the coda
            # h = input_norm(prelude_out) (+ core deltas), so the coreless path
            # must apply the same boundary norm. This makes a seed model
            # EXACTLY a target-with-silent-core — the growth invariant that
            # function-preserving core insertion depends on.
            x = self.input_norm(x)
        return x

    def _build_fm(self, cfg: "MORPHConfig", d: int) -> None:
        """Construct the FM1 planner and its σ/t machinery. Called only when ``cfg.fm``.

        Two hard preconditions, both checked rather than assumed:

        * ``cfg.tul`` must be set. FM1 is a TUL arm — it writes into the slot prefix
          positions through ``W_prefix``, which only exists on a TUL model.
        * ``cfg.n_core`` must be 0. The planner REPLACES the core loop; leaving a core
          in place would build two slot-state producers and silently use one.
        """
        from morph.model.fm_planner import analytic_null_floor, build_schedule
        from morph.model.tul_fm import FMArmConfig

        if not isinstance(cfg.fm, FMArmConfig):
            raise TypeError(f"cfg.fm must be an FMArmConfig, got {type(cfg.fm).__name__}")
        if cfg.tul is None:
            raise ValueError(
                "MORPHConfig(fm=...) requires tul=... — FM1 writes its plans into the "
                "slot prefix positions through W_prefix, which only a TUL model has.")
        if cfg.n_core != 0:
            raise ValueError(
                f"MORPHConfig(fm=...) requires n_core == 0, got {cfg.n_core}. The FM "
                "planner REPLACES the core loop; building both would leave two slot-state "
                "producers in the model and use only one of them.")
        if cfg.tul.tokens_through_core:
            raise NotImplementedError(
                "fm has no defined interaction with arm A2 (tokens_through_core): A2 runs "
                "the core over every position and FM1 has no core. Raises rather than "
                "silently picking a behaviour (the tul.gate precedent).")
        if cfg.tul.sigreg_lambda > 0.0:
            raise ValueError(
                "tul.sigreg_lambda regularises the CORE's slot states, which FM1 does not "
                "have — its slot states are DETACHED plans, so the term would have no "
                "gradient path at all. Use fm.sigreg_lambda, which regularises the pooled "
                "TARGETS (morph/model/tul_fm.py::fm_sigreg_loss).")

        from morph.model.fm_planner import FMPlanner
        fmc = cfg.fm
        # The slot budget and the padded row length the loader will produce. max_slots
        # follows TulDataConfig's rule (seq_len // 8 when unset) and l_total adds the
        # prefix positions; both are upper bounds, and the planner only needs them to
        # size its slot-index embedding and its position table.
        fallback_slots = cfg.max_seq_len // 8
        fallback_l = cfg.max_seq_len + cfg.tul.prefix_k * fallback_slots
        pcfg = fmc.planner_cfg(d, fallback_slots, fallback_l)
        self.fm_planner = FMPlanner(pcfg)
        self._fm_schedule = build_schedule(sigma_data=1.0)
        self._fm_loss_scale = (
            analytic_null_floor(pcfg, self._fm_schedule, e_y_sq=1.0)
            if fmc.loss_scale == "auto" else 1.0)

        # LOUD, because it is the one number most likely to be wrong. The targets are
        # UNIT L2 in d dims, so their per-component std is 1/sqrt(d); the matched CFM
        # source scale is therefore 1/sqrt(d), not 1. At source_std = 1 the source carries
        # d times the variance of the target, so ||v||^2 = ||y||^2 + d*s^2 is ~1 + d
        # instead of ~2 and the velocity the net must fit is dominated by reconstructing
        # x0. This is the DeepWeightFlow App. H failure mode and the direct analogue of
        # P1's sigma_data scar. Printed, not silently corrected: the value is a config key.
        matched = 1.0 / math.sqrt(d)
        note = "MATCHED" if abs(fmc.source_std - matched) < 0.25 * matched else "MISMATCHED"
        print(f"  FM1 planner: {sum(p.numel() for p in self.fm_planner.parameters())/1e6:.1f}M "
              f"params, objective={fmc.objective} T={fmc.infer_steps} "
              f"target R^{d} max_slots={pcfg.max_slots} "
              f"L_total={pcfg.max_ctx_len}", flush=True)
        print(f"  FM1 loss: fm_weight={fmc.fm_weight} loss_scale={fmc.loss_scale}"
              f"(-> {self._fm_loss_scale:.4f}) sigreg_lambda={fmc.sigreg_lambda} "
              f"M={fmc.sigreg_slices}", flush=True)
        print(f"  FM1 source_std={fmc.source_std} vs matched 1/sqrt(d)={matched:.4f} "
              f"[{note}] -> E||v||^2 = ||y||^2 + d*s^2 = "
              f"{1.0 + d * fmc.source_std ** 2:.1f}", flush=True)

    def _tul_fm_core(self, x: Tensor, layout: SlotLayout):
        """FM1's replacement for :meth:`_tul_core`. Returns ``(xn, h_slots, y, geom)``.

        1. ``xn = input_norm(prelude)`` — the SAME boundary norm the ``n_core == 0`` seed
           path applies, so the coda sees the carrier it always sees.
        2. pooled unit-L2 targets ``y`` for every slot's NEXT span, LIVE (not detached):
           this is SIGReg's gradient path into the backbone.
        3. plans ``z`` from the Euler ladder under ``no_grad``, then ``detach()``.

        The context handed to the planner is the stream-MEAN of the carrier — the same
        scale-preserving reduction :meth:`_readout` uses — detached, so the ladder can
        never backpropagate into the prelude.
        """
        from morph.model.fm_planner import generate_plans
        from morph.model.tul_fm import fm_geometry, fm_span_targets

        pc = self.fm_planner.cfg
        if layout.max_slots > pc.max_slots or layout.l_total > pc.max_ctx_len:
            raise ValueError(
                f"FM planner is sized for max_slots={pc.max_slots}, "
                f"max_ctx_len={pc.max_ctx_len} but the layout is "
                f"max_slots={layout.max_slots}, l_total={layout.l_total}. Set "
                f"fm.max_slots / fm.l_total (morph/training/fm_setup.py passes the "
                f"loader's exact values), or raise model.max_seq_len.")
        xn = self.input_norm(x)
        h_ctx = xn.mean(dim=2) if self._is_hc else xn          # [B, L, C]
        geom = fm_geometry(layout)
        y = fm_span_targets(h_ctx, layout, geom)               # LIVE — SIGReg reads this

        with torch.no_grad():
            z = generate_plans(self.fm_planner, h_ctx.detach().float(), geom,
                               self._fm_schedule,
                               n_steps=int(self.cfg.fm.infer_steps))
        # THE DETACH. Everything downstream of here is in the CE graph; the ladder is not.
        h_slots = z.detach().to(xn.dtype)                      # [B, S, C]
        if self._is_hc:
            # A single-stream signal broadcast into every stream — the same convention
            # `_front_tail` uses for the initial carrier expansion.
            h_slots = h_slots.unsqueeze(2).expand(-1, -1, self._n_streams, -1)
        return xn, h_slots, y, geom, h_ctx

    def _clip_applies(self, t: int) -> bool:
        """Does the core-gain governor apply at loop iteration ``t``?

        ``t`` is a Python int from the loop, so this is a trace-time constant per
        iteration: the branch never reaches the graph and the default range keeps the
        traced code identical to the un-ranged version.
        """
        lo = self.cfg.core_gain_clip_iter_lo
        hi = self.cfg.core_gain_clip_iter_hi
        return t >= lo and (hi < 0 or t <= hi)

    # ── TUL regions (docs/tul-spec.md §3) ─────────────────────────────────
    # Reached only when a `slot_layout` is passed. Every helper below is a no-op for
    # the plain path because the plain path never calls it.

    def _tul_front(self, input_ids: Tensor, layout: SlotLayout,
                   attn_kwargs: dict | None = None,
                   ret_reset_mask: Tensor | None = None):
        """Embed + slot inputs + prelude over ALL positions (spec §3.2).

        The slot's input embedding is ``E_slot + mean_j embed(t_j)`` over its span's
        tokens, and its bigram / value-embed signals are the same bag-mean — this is
        exactly the TST ``ve_bagged`` path with a data-dependent bag map (spec §3.2;
        Dynamic Token Pooling mean-pool; BLT Eq. 5). The prelude itself is unchanged:
        the slot's output is the in-context pooled span summary (BLT §3.2.2).

        ``attn_kwargs`` / ``ret_reset_mask``: passed straight through to
        :meth:`_front_tail` (docs/tul-tg-spec.md §§1-4).
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
        x, x0 = self._front_tail(x, input_ids, bigram_emb, ve_bagged,
                                 attn_kwargs=attn_kwargs, ret_reset_mask=ret_reset_mask)
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

    def _tul_core(self, x: Tensor, x0: Tensor, bigram_emb, layout: SlotLayout,
                  halt: bool = False):
        """Gather slots → masked per-slot depth loop → looped states (spec §3.3).

        Returns ``(xn, h_slots, depths, g_traj)``: ``xn = input_norm(prelude)`` for the
        whole carrier (token positions keep it — the ``n_core == 0`` seed path, BLT
        Eq. 9), ``h_slots`` ``[B, max_slots, …]`` the looped state of each slot, the
        realised per-slot ``depths``, and — when the gate is built — ``g_traj``
        ``[B, max_slots, T]``, the gate output after EVERY iteration
        (docs/tul-gate-spec.md §4). ``g_traj`` is a RETURN VALUE, never a side channel:
        the ``ret_capture`` lesson is that a side channel is not checkpoint-safe.

        ``halt`` (arm ``TUL-halt``, gate §7) replaces the Poisson depth with the gate's
        own stop decision — a slot loops until it asks for ``k ≥ 1`` token, capped at
        ``slot_max_depth``. EVAL ONLY: §4 teacher-forces the depth during training, so
        ``TUL-gate`` and ``TUL-halt`` are one training run scored twice, which makes the
        comparison exactly paired.

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
        gidx, gvalid = layout.slot_index, layout.slot_valid

        xn = self.input_norm(x)
        e = gather_valid(xn, gidx, gvalid)                            # [B, S, n, C]

        # ── n_core == 0: NO LOOP AT ALL (arm GL1, the gist baseline) ─────────
        # .agents/notes/proposed/architecture/2026-08-29-gist-loop.md. The slot state IS
        # the prelude's own output at the slot position, after the same boundary norm the
        # coreless TOKEN path applies (`_core_region`'s n_core == 0 branch) — so a
        # coreless TUL model is exactly a coreless baseline that happens to have slot
        # positions, which is the growth invariant the seed path already depends on.
        #
        # Nothing is detached. Under `tg_restrict` the slot is the only route from an
        # earlier span to a later one, so a later span's CE MUST backpropagate through
        # this state into the boundary tap and the prelude. That gradient-carrying write
        # is the arm's entire mechanism (gisting; TG paper Table 1: detaching the write
        # costs 10x PPL), and there is no iterated map left for it to unroll — which is
        # what makes it safe here and unsafe in every arm that kept the loop.
        #
        # Without this branch the code below raises "stack expects a non-empty
        # TensorList" on the x0/bigram injection stack, because that stack has one entry
        # per core layer. Verified before the branch existed.
        if n_core == 0:
            if halt:
                raise RuntimeError(
                    "halt=True needs a core loop to halt (docs/tul-gate-spec.md §7); "
                    "n_core == 0 has no iterations to stop.")
            if self.tul_gate is not None:
                raise NotImplementedError(
                    "tul.gate has no defined meaning at n_core == 0: §4 reads the span "
                    "length off the core's per-iteration trajectory and there is no "
                    "trajectory. Raises rather than silently emitting a one-step gate.")
            depths = torch.zeros_like(gidx)
            return xn, e, depths, None

        _scse = self.scse           # Python-level constant → every branch below traces out
        with _prof("carrier::h_clone"):
            if _scse is None:
                h = self.core_init(e)
                h_star = None
            else:
                # THE LOOP CARRIER IS THE DEVIATION (docs/scse-spec.md section 3.1). h* is
                # built ONCE (S2); the absolute slot state is rebuilt at the return (S6).
                # `gather_valid` zeroes pad slots, and both projections are bias-free, so a
                # pad has h* = 0 AND Delta_0 = 0 exactly — invariant S8.
                h_star, h = _scse.entry(e)

        # Loop-invariant injection, built ON THE COMPACT SEQUENCE (the x0/bigram hoist
        # of the token path, applied to 9-19× fewer positions). Value-embeds never fire
        # in the core (gi ≥ n_prelude is not in _ve_layer_map), so input_ids is not needed.
        # SCSE's core is source-free (spec D3), so the x0/bigram stack is never read — and
        # building it anyway would cost n_core projections over the compact sequence per
        # forward. `_inj_none` keeps the call signature below identical in both modes.
        _inj_none = h.new_zeros(0)
        if _scse is not None:
            x0_s = bg_s = None
            inj = _inj_none
        else:
            x0_s = gather_valid(x0, gidx, gvalid)
            bg_s = gather_valid(bigram_emb, gidx, gvalid) if bigram_emb is not None else None
            inj = torch.stack(
                [self._build_injection_term(np_ + i, self.x0_injects[np_ + i].precompute(x0_s),
                                            None, bg_s, h.dtype)
                 for i in range(n_core)], dim=0)
        # What the loop actually hands `_core_step`. Under SCSE the second argument carries
        # the ANCHOR (read only when kappa > 0) and the third is unused.
        _e_arg = h_star if _scse is not None else e
        _inj_arg = _inj_none if _scse is not None else inj

        if halt:
            if self.tul_gate is None:
                raise RuntimeError("halt=True needs a model built with tul.gate (§7)")
            if self.training:
                raise RuntimeError(
                    "halt=True is EVAL ONLY (docs/tul-gate-spec.md §4: training "
                    "teacher-forces the depth). Scoring a training step with the gate "
                    "driving the depth would make the LM loss chase the gate's error.")
            total_iters = self.cfg.tul.slot_max_depth or self.cfg.max_depth
            alive = layout.slot_valid.clone()
            depths = torch.zeros_like(layout.slot_index)
        else:
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
            if _scse is None:
                return self._apply_core_step(h_in, e_in, None, None, None,
                                             ret_state=ret_state, iter_idx=iter_idx,
                                             inj_terms=inj_terms)
            # ── SCSE, Eqs. 3-5 ──────────────────────────────────────────────────────
            # `h_in` IS Delta_t; `e_in` carries h*. Signature unchanged so the three call
            # sites (no_grad / checkpoint / eager) and the truncated-BPTT window they
            # implement are untouched.
            # `_rec` is bound once and handed to BOTH calls — see the twin in `_tul_core`.
            _rec = _scse.recurrent_input(h_in, e_in)
            g_out, new_ret = self._apply_core_step(
                _rec, None, None, None, None,
                ret_state=ret_state, iter_idx=iter_idx, inj_terms=None, source_free=True)
            return _scse.update(h_in, g_out, _rec), new_ret    # Eqs. 3-5

        g_list: list[Tensor] = []
        # ── Phase-1 onset probe (plan task 1.1) ───────────────────────────────
        # The GLA cross-iteration carry is a SECOND recurrent loop inside the core loop
        # and nothing watches it. Its forget gate is biased to alpha near 1
        # (retention_gate_bias 2.0), so it can integrate without bound while the carrier
        # norm stays flat and the loss stays flat. Collected on GPU per iteration and
        # read once per step by the trainer, so there is no sync inside the loop.
        # `_probe_loop` is a plain Python bool read at trace time: False (the default)
        # traces the identical graph as before and costs nothing.
        _probe = getattr(self, "_probe_loop", False)
        _pr_ret: list[Tensor] = []
        _pr_gain: list[Tensor] = []
        _pr_bind: list[Tensor] = []
        _pr_in: list[Tensor] = []
        _pr_out: list[Tensor] = []
        _pr_delta: list[Tensor] = []
        for t in range(total_iters):
            active = alive if halt else (depths > t)               # [B, S]
            # ── Jacobian probe capture (morph/training/core_jacobian.py) ──────────
            # `_jac_capture` is None by default, so this is a Python-level branch that
            # traces out and costs nothing; the forward stays bit-identical. When a list
            # is attached, the probe needs the EXACT operating point of one core step —
            # the map f_theta is `_apply_core_step` bound to (e, inj, ret_state, t), and
            # sigma_max of its Jacobian is only meaningful at the h the run actually
            # reached. Detached: the probe rebuilds its own graph.
            if self._jac_capture is not None:
                self._jac_capture.append({
                    "h": h.detach(), "e": _e_arg.detach(), "inj": _inj_arg.detach(),
                    # Under SCSE "h" is the DEVIATION and "e" is the ANCHOR — see the twin
                    # in `_core_region`. The mode travels with the data so no probe can read
                    # these as (state, source) by accident.
                    "scse": _scse is not None,
                    "ret_state": None if ret_state is None else ret_state.detach(),
                    # `active & slot_valid`, never `active` alone. A pad slot enters the
                    # loop at h = 0 (gather_valid zeroes it) and depths is 1 there, so it is
                    # "active" at t = 0 — and an RMSNorm at h = 0 has a Jacobian of order
                    # 1/eps, which puts the top singular direction entirely in the pad
                    # subspace and returns a sigma of ~1e6 that means nothing. Measured.
                    "iter_idx": t, "active": (active & layout.slot_valid).detach(),
                })
            do_ckpt = self.training and (t - n_nograd) < n_ckpt
            if t < n_nograd:
                with torch.no_grad():
                    h_new, rs_new = _core_step(h, _e_arg, _inj_arg, ret_state=ret_state,
                                               iter_idx=t)
            elif do_ckpt:
                h_new, rs_new = checkpoint(_core_step, h, _e_arg, _inj_arg,
                                           ret_state=ret_state, iter_idx=t,
                                           use_reentrant=False)
            else:
                h_new, rs_new = _core_step(h, _e_arg, _inj_arg, ret_state=ret_state,
                                           iter_idx=t)

            _tau = self.cfg.core_gain_clip
            if _tau > 0.0 and self._clip_applies(t):
                # PAD SLOTS ARE EXCLUDED from the norm. The gain clip is per SAMPLE, so a
                # row's pad slots would otherwise put the number of REAL slots in that row
                # into the scale applied to every real slot — padding would change the
                # forward. Pads start at 0 (gather_valid) but the first core step moves
                # them off zero, so zeroing here is not redundant. Dormant at the arms'
                # core_gain_clip = 0.0; correct if it is ever turned on (reviewer).
                _vm = layout.slot_valid.view(*layout.slot_valid.shape,
                                             *([1] * (h.dim() - 2))).to(h.dtype)
                _in_n = (h * _vm).flatten(1).norm(dim=1)
                _out_n = (h_new * _vm).flatten(1).norm(dim=1)
                _scale = torch.clamp(_tau * _in_n / (_out_n + 1e-6), max=1.0)
                h_new = h_new * _scale.view(-1, *([1] * (h_new.dim() - 1)))
                if _probe:
                    # Fraction of samples the cap actually SHRANK at this iteration.
                    # Where a clip is applied and where it acts are different questions.
                    _pr_bind.append((_scale < 1.0).float().mean().detach())
            elif _probe:
                _pr_bind.append(h.new_zeros(()))

            # ── gate readout (docs/tul-gate-spec.md §4) ────────────────────────
            # OUTSIDE the checkpoint / no_grad block on purpose: it then shapes the core
            # state on exactly the iterations inside the truncated-BPTT window and is a
            # pure readout on the frozen ones — the same window the token loss uses —
            # while the head itself (w, b, norm.scale) is supervised on EVERY iteration.
            # Read off h_new, not the masked h: for an active slot they are equal, and
            # the halting policy below needs this iteration's fresh state.
            if self.tul_gate is not None:
                # Under SCSE `h_new` is the deviation; the halting head is a readout of the
                # slot STATE, so it must see h* + Delta, not the deviation alone.
                g_t = self.tul_gate.readout(
                    (h_star + h_new) if _scse is not None else h_new)   # [B, S]
                g_list.append(g_t)

            if _probe:
                with torch.no_grad():
                    # Gain is measured on the ACTIVE slots only — a finished slot's state is
                    # frozen, so including it would dilute the runaway we are looking for.
                    _am = active.view(*active.shape, *([1] * (h.dim() - 2))).to(h.dtype)
                    _hi = (h * _am).flatten(1).float().norm(dim=1)
                    _ho = (h_new * _am).flatten(1).float().norm(dim=1)
                    _pr_gain.append((_ho / (_hi + 1e-6)).max().detach())
                    # SEPARATE the ratio's numerator from its denominator. A gain of 17 at
                    # iteration 0 and 1.1 everywhere else has two readings that this ratio
                    # alone cannot tell apart: the map amplifies more on its first
                    # application, or ‖h_in‖ is smaller there because iteration 0's input is
                    # input_norm(prelude) rather than a previous core output. Logging both
                    # norms and the RELATIVE UPDATE ‖h_new − h‖/‖h‖ separates them —
                    # delta_ratio is the size of what the core ADDS, independent of the
                    # carrier's scale, so it is the term a residual stream actually controls.
                    _pr_in.append(_hi.max().detach())
                    _pr_out.append(_ho.max().detach())
                    _pr_delta.append((((h_new - h) * _am).flatten(1).float().norm(dim=1)
                                      / (_hi + 1e-6)).max().detach())
                    _pr_ret.append((rs_new if (track_ret and rs_new is not None)
                                    else h.new_zeros(())).float().norm().detach())

            h = torch.where(active.view(*active.shape, *([1] * (h.dim() - 2))), h_new, h)
            if track_ret and rs_new is not None:
                ret_state = rs_new

            if halt:
                # A slot that asks for k ≥ 1 token has finished thinking (§7/§8). Slots
                # that never ask keep the cap, so the generator cannot hang.
                stop = alive & (self.tul_gate.choose_k(g_t) >= 1)
                depths = torch.where(stop, torch.full_like(depths, t + 1), depths)
                alive = alive & ~stop
                if not bool(alive.any()):
                    break
        if halt:
            depths = torch.where(depths > 0, depths, torch.full_like(depths, total_iters))
            depths = torch.where(layout.slot_valid, depths, torch.ones_like(depths))
        if _probe:
            # [T] each, still on GPU and detached. The trainer reads them once per step.
            self._loop_probe = {
                "core_gain": torch.stack(_pr_gain) if _pr_gain else None,
                "ret_state_norm": torch.stack(_pr_ret) if _pr_ret else None,
                "in_norm": torch.stack(_pr_in) if _pr_in else None,
                "out_norm": torch.stack(_pr_out) if _pr_out else None,
                "delta_ratio": torch.stack(_pr_delta) if _pr_delta else None,
                "clip_bind": torch.stack(_pr_bind) if _pr_bind else None,
            }
        g_traj = torch.stack(g_list, dim=-1) if g_list else None   # [B, S, T]
        if _scse is not None:
            # Eq. 5 tail: h_T = h* + Delta_T (invariant S6). The deviation lives ONLY inside
            # this function; `_forward_tul` scatters an absolute carrier exactly as today.
            h = h_star + h
        return xn, h, depths, g_traj

    def _tul_half_weights(self, labels: Tensor, layout: SlotLayout):
        """``([N] row weights, t_last index, emit index)`` for the §5 double label.

        MORPH's layout puts the slot BETWEEN a span's last token and the next span's
        first token, so ``t_1(i+1)`` is predicted TWICE: once from ``t_last`` (plain LM,
        no plan) and once from the slot's emitting position (with the plan). Spec §5
        weights both terms 0.5 "so first tokens are not counted twice", which makes the
        weighted-mean denominator the number of DISTINCT target tokens.

        The index tensors are fixed-shape: invalid (pad) slots address a trailing pad
        row, so nothing depends on the realised slot count and no host sync is needed.
        """
        B, L = labels.shape
        BL = B * L
        row_off = (torch.arange(B, device=labels.device) * L).unsqueeze(1)
        base = layout.slot_index + row_off
        # t_last sits immediately before the slot — the layout guarantees a slot never
        # starts at position 0, so base-1 is always a real token position.
        p_idx = torch.where(layout.slot_valid, base - 1, BL).reshape(-1)
        z_idx = torch.where(layout.slot_valid, base + layout.prefix_k - 1, BL).reshape(-1)
        w = labels.new_ones(BL + 1, dtype=torch.float32)
        w[p_idx] = self.cfg.tul.plast_weight
        w[z_idx] = self.cfg.tul.emit_weight
        return w[:BL], p_idx, z_idx

    def _tul_mux_loss(self, h_slots: Tensor, input_ids: Tensor,
                      layout: SlotLayout) -> Tensor:
        """MUX local head (arXiv 2607.18264): weighted CE of each slot's NEXT span.

        The slot's post-core state is read out through the model's OWN LM-head path
        (``_readout`` → unembedding) — zero new parameters — and trained toward the
        geometric superposition of its next span's tokens. The KL to that target
        equals this weighted CE up to the target's (constant) entropy, and the
        dense ``|V|`` target is never materialised: the loss gathers log-probs at
        the span's own token ids only (``mux_span_targets``).

        Why this exists: the plan's only direct supervision was ``ce_emit``, a
        one-token race the free token path wins (the 2026-08-25 pivot). This head
        gives z span-level content gradient that does not route through the coda's
        suppressed readout, and MUX Prop 16 shows a low local loss also protects
        the answer-side routing TO the latents. Reads h_slots BEFORE the gate's
        budget conditioning — the plan is supervised, not the budget arithmetic.

        Cost: logits over slots only, ``[B, S, V]`` fp32 ≈ 75 MB at B=6, S=64 —
        ~24x smaller than one row of full-sequence logits (why fused CE exists).
        """
        tc = self.cfg.tul
        z = self._readout(h_slots)                            # [B, S, C]
        # `lm_weight()` is WEIGHT-TIED to the input embeddings, so an undetached
        # head trains the embedding table on the auxiliary target — see
        # TULConfig.mux_detach_head for the measured consequence. Python-level
        # constant: the branch traces out, no runtime flag in the graph.
        w_head = self.embed.lm_weight()                       # [V, C]
        if tc.mux_detach_head:
            w_head = w_head.detach()
        logits = (z @ w_head.t()).float() / tc.mux_tau        # fp32: stable log_softmax
        logits = logits.index_fill(
            -1, torch.tensor([tc.slot_id], device=logits.device), float("-inf"))
        logp = torch.log_softmax(logits, dim=-1)              # [B, S, V]
        pos_valid, alpha, tgt_slot, sup = mux_span_targets(input_ids, layout, tc.mux_rho)
        B = input_ids.shape[0]
        V = logp.shape[-1]
        # logp[b, tgt_slot[b,p], input_ids[b,p]] without a [B, L, V] gather.
        # Invalid positions gather id 0, NOT their own id: a slot position's own id
        # is slot_id, whose logp is the masked -inf, and 0 * -inf = NaN (caught by
        # test_mux_loss_decomposes_as_ce_plus_weighted_term before it ever ran).
        safe_ids = torch.where(pos_valid, input_ids, torch.zeros_like(input_ids))
        lp = logp.reshape(B, -1).gather(1, tgt_slot * V + safe_ids)    # [B, L]
        ce = -torch.where(pos_valid, alpha * lp, torch.zeros_like(lp)).sum()
        return ce / sup.sum().to(ce.dtype).clamp(min=1.0)

    def _tul_sigreg_loss(self, h_slots: Tensor, layout: SlotLayout) -> Tensor:
        """SIGReg over the VALID slot states (LeJEPA; see morph/model/sigreg.py).

        Reads the same post-core plan state the MUX head reads, through the
        model's own readout, and pushes the distribution of those states toward
        an isotropic standard Gaussian. Pad slots are excluded — they are a
        fixed-shape artefact, and including them would let the regulariser
        "fix" the distribution by moving vectors that mean nothing.

        Applied to the plan states rather than to token states on purpose: the
        collapse this targets was measured THERE (effective rank 1.7-4.8, mean
        pairwise cosine +0.39..+0.71).
        """
        z = self._readout(h_slots)                          # [B, S, C]
        valid = layout.slot_valid.reshape(-1)               # [B*S]
        z = z.reshape(-1, z.shape[-1])[valid]               # [N_valid, C]
        return sigreg_epps_pulley(z, num_slices=self.cfg.tul.sigreg_slices)

    def _tul_group_losses(self, x: Tensor, labels: Tensor, layout: SlotLayout | None,
                          want_groups: bool = True) -> dict:
        """Training loss (ONE weighted CE) plus, at eval, the §7.2 metric breakdown.

        The training term is a single ``fused_linear_cross_entropy`` over every position
        with the §5 weights folded into the kernel's reduction. One call rather than one
        per label group matters: each call allocates and SAVES a ``[V, d]`` fp32
        ``grad_w`` accumulator (201 MB at V=49169, d=1024), so three calls cost ~0.5 GB
        of activation memory for arithmetic that a weight vector expresses exactly.

        The per-group CEs (``ce_main`` / ``ce_plast`` / ``ce_emit``) are METRICS — spec
        §7.2's ``val/first_tok_ce`` and ``val/first_tok_counterfactual``. They are
        computed only when ``want_groups`` (eval), where there is no backward graph to
        retain, and they carry no training signal that the weighted call does not
        already carry.

        ``layout=None`` (arm A4 / the plan-nats gather, where slots are not in the
        sequence at all) → a plain unweighted CE over the token positions.
        """
        B, L, C = x.shape
        w_head = self.embed.lm_weight()
        mask_id = self.cfg.tul.slot_id
        chunk = self.cfg.ce_chunk_size
        flat = x.reshape(-1, C)
        lab = labels.reshape(-1)
        BL = flat.shape[0]

        if layout is None:
            ce = fused_linear_cross_entropy(flat, w_head, lab, ignore_index=-100,
                                            chunk_size=chunk, mask_token_id=mask_id)
            return {"loss": ce, "ce_main": ce, "ce_tokens": ce,
                    "n_targets": (lab != -100).sum().to(ce.dtype)}

        row_w, p_idx, z_idx = self._tul_half_weights(labels, layout)
        loss = fused_linear_cross_entropy(flat, w_head, lab, ignore_index=-100,
                                          chunk_size=chunk, mask_token_id=mask_id,
                                          weights=row_w)
        valid = (lab != -100).to(row_w.dtype)
        out = {"loss": loss, "n_targets": (row_w * valid).sum()}
        if not want_groups:
            return out

        lab_pad = torch.cat([lab, lab.new_full((1,), -100)], dim=0)
        flat_pad = torch.cat([flat, flat.new_zeros(1, C)], dim=0)
        main_lab = lab_pad.scatter(0, torch.cat([p_idx, z_idx], dim=0), -100)[:BL]
        ce_main = fused_linear_cross_entropy(flat, w_head, main_lab, ignore_index=-100,
                                             chunk_size=chunk, mask_token_id=mask_id)
        out["ce_main"] = ce_main
        out["n_main"] = (main_lab != -100).sum().to(ce_main.dtype)
        for tag, idx in (("plast", p_idx), ("emit", z_idx)):
            labs = lab_pad[idx]
            ce = fused_linear_cross_entropy(flat_pad[idx], w_head, labs, ignore_index=-100,
                                            chunk_size=chunk, mask_token_id=mask_id)
            out[f"ce_{tag}"] = ce
            out[f"n_{tag}"] = (labs != -100).sum().to(ce.dtype)
        # val/ppl_tokens is over TOKEN positions only (ordinary + t_last), which keeps it
        # comparable to the baseline's token PPL (spec §4).
        out["ce_tokens"] = ((ce_main * out["n_main"] + out["ce_plast"] * out["n_plast"])
                            / (out["n_main"] + out["n_plast"]).clamp(min=1.0))
        out["ce_first_tok"] = out["ce_emit"]
        out["ce_first_tok_plain"] = out["ce_plast"]
        out["first_tok_counterfactual"] = out["ce_plast"] - out["ce_emit"]
        return out

    def _forward_tul(self, input_ids: Tensor, labels: Tensor | None,
                     layout: SlotLayout, plan_nats: bool, halt: bool = False,
                     plan_mode: str = "normal") -> dict:
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
        if tc.coda_token_cut >= L:
            raise ValueError(
                f"tul.coda_token_cut={tc.coda_token_cut} >= seq_len {L} "
                f"(docs/tul-compaction-window-spec.md): every token position would be "
                f"dropped from the coda, leaving nothing to predict. Lower the cut."
            )

        # ── TG restriction (docs/tul-tg-spec.md §§1-4) — built ONCE per forward ────
        # tg_attn_kwargs feeds the window branch's extra_mask (tg_allow) and the
        # compressed branch's slot mask; tg_reset feeds the GLA segment reset. Both
        # None on a tg_restrict=false model (bit-identical, spec T4).
        tg_attn_kwargs = tg_reset = None
        if self._tg_restrict:
            tg_allow = tg_allow_mask(layout, soft_prev_span=tc.tg_soft_prev_span)
            tg_attn_kwargs = {"tg_allow": tg_allow, "tg_slot_mask": layout.slot_mask}
            tg_reset = tg_reset_mask(layout)

        x, x0, bigram_emb = self._tul_front(input_ids, layout,
                                            attn_kwargs=tg_attn_kwargs,
                                            ret_reset_mask=tg_reset)

        if tc.gate is not None and tc.tokens_through_core:
            raise NotImplementedError(
                "tul.gate has no defined interaction with arm A2 (tokens_through_core): "
                "A2 has no per-slot looped state to read a length off. Not specified, so "
                "this raises rather than silently picking a behaviour.")
        if tc.sigreg_lambda > 0.0 and tc.tokens_through_core:
            raise NotImplementedError(
                "tul.sigreg_lambda has no defined interaction with arm A2 "
                "(tokens_through_core): A2 has no per-slot looped state to "
                "regularise. Raises rather than silently picking a behaviour.")
        if tc.mux_beta > 0.0 and tc.tokens_through_core:
            raise NotImplementedError(
                "tul.mux_beta has no defined interaction with arm A2 (tokens_through_core): "
                "A2 has no per-slot looped state to read a plan off. Raises rather than "
                "silently picking a behaviour (the tul.gate precedent).")
        if self._tg_restrict and tc.tokens_through_core:
            raise NotImplementedError(
                "tul.tg_restrict has no defined interaction with arm A2 "
                "(tokens_through_core): docs/tul-tg-spec.md's mental model is 'core loop "
                "on slots only, the one channel to the past' — A2 runs the core over "
                "every position instead. Not specified, so this raises rather than "
                "silently picking a behaviour (the tul.gate precedent).")
        if tc.tokens_through_core:
            # Arm A2 (slots-as-memory): tokens AND slots run the ordinary per-SAMPLE core.
            # RESOLVED SPEC AMBIGUITY — §7.1's A2 row says "Poisson/slot" in the depth
            # column but "uniform depth" in the isolates column. A2 must differ from A0 by
            # the presence of slots ALONE (it isolates C2), so it reuses today's core
            # region unchanged; a per-position Poisson depth would change two things at once.
            x_coda = self._core_region(x, x0, bigram_emb, input_ids)
            depths, g_traj, mux_loss, sigreg_loss = None, None, None, None
            fm_y = fm_geom = fm_ctx = None
        elif self.fm_planner is not None:
            # FM1 (morph/model/tul_fm.py). The planner replaces the core loop; the plan
            # is DETACHED before it reaches W_prefix, so the coda's CE never touches the
            # ladder and there is no BPTT through an iterated map.
            xn, h_slots, fm_y, fm_geom, fm_ctx = self._tul_fm_core(x, layout)
            depths, g_traj, mux_loss, sigreg_loss = None, None, None, None
            h_slots = self._tul_plan_ablate(h_slots, layout, plan_mode)
            values, pos = self.tul.prefix_project(h_slots, layout, L)
            x_coda = scatter_positions(xn, pos, values)
        else:
            fm_y = fm_geom = fm_ctx = None
            xn, h_slots, depths, g_traj = self._tul_core(x, x0, bigram_emb, layout,
                                                         halt=halt)
            mux_loss = (self._tul_mux_loss(h_slots, input_ids, layout)
                        if tc.mux_beta > 0.0 else None)
            sigreg_loss = (self._tul_sigreg_loss(h_slots, layout)
                           if tc.sigreg_lambda > 0.0 else None)
            if self.tul_gate is not None:
                budget_ids = self._tul_budget_ids(layout, depths, g_traj)
                h_slots = self.tul_gate.apply_budget(h_slots, budget_ids)
            # Same eval-only ablation the FM branch takes, applied at the same seam.
            # `normal` returns its input unchanged, so the training forward is
            # bit-identical to a model with no ablation code at all.
            h_slots = self._tul_plan_ablate(h_slots, layout, plan_mode)
            values, pos = self.tul.prefix_project(h_slots, layout, L)
            x_coda = scatter_positions(xn, pos, values)

        x_coda, keep = self.tul.apply_token_dropout(x_coda, layout, self.training)

        out: dict = {"logits": None}
        if tc.coda_sees_slots and tc.coda_token_cut == 0:
            xh = self._back_region(x_coda, x0, bigram_emb, input_ids, inject_keep=keep,
                                   attn_kwargs=tg_attn_kwargs, ret_reset_mask=tg_reset)
            groups = (self._tul_group_losses(xh, labels, layout, want_groups=not self.training)
                      if labels is not None else None)
            coda_positions = L
        else:
            # Arm A4 (coda_sees_slots=False) and/or arm CW (coda_token_cut>0, spec
            # docs/tul-compaction-window-spec.md) — ONE gather whose drop_mask is the
            # union of "every slot" and "every token below the cut". coda_token_cut=0
            # never reaches this branch when coda_sees_slots is True (checked above), so
            # the pre-CW A4 path is untouched when CW is off.
            if self._tg_restrict:
                raise NotImplementedError(
                    "tul.tg_restrict has no defined interaction with coda_sees_slots=False "
                    "or coda_token_cut>0: the coda then runs on a GATHERED subset of "
                    "positions, and neither tg_allow nor the GLA reset mask is re-derived "
                    "for that index space (docs/tul-tg-spec.md does not specify it). Not "
                    "run by the TG arms (tul_a1's coda_sees_slots=true, coda_token_cut=0); "
                    "raises rather than silently building an unrestricted or mis-masked "
                    "coda pass.")
            drop_mask = self._tul_coda_drop_mask(layout, tc)
            # CW keeps slots in the gathered sequence, and _tul_coda_gather scores a
            # PLAIN CE (layout=None) over everything it keeps — so the slot emit labels
            # (label = next span's first token) must be masked here or CW training
            # silently reinstates the emit loss at weight 1.0 and double-counts every
            # span's first token. The eval screen (tul_forward_cw_arms) already scores
            # token positions only; this makes training match it. A no-op for arm A4,
            # whose drop_mask removes the slot positions themselves.
            labels_g = labels
            if labels is not None and tc.coda_sees_slots:
                labels_g = torch.where(layout.slot_mask,
                                       torch.full_like(labels, -100), labels)
            xh, groups, coda_positions = self._tul_coda_gather(
                x_coda, x0, bigram_emb, keep, labels_g, layout, drop_mask)
        if plan_nats and labels is not None:
            # §7.2: CE over the same tokens with the slots removed from the coda sequence.
            # Reported MINUS the normal token CE; a positive value is the plan actually
            # being used (the h_z-ablation, the C2 number). Only when the normal pass IS
            # ALREADY exactly the slots-removed pass (A4 with coda_token_cut=0) can it be
            # reused — arm CW's normal pass also removes early tokens, a different
            # ablation, so it must be recomputed fresh from x_coda in that case too.
            if self._tg_restrict:
                raise NotImplementedError(
                    "plan_nats (§7.2) has no defined interaction with tg_restrict: it "
                    "removes every slot position from the coda's gathered sequence — "
                    "exactly the channel tg_restrict forces context through — and "
                    "docs/tul-tg-spec.md does not specify the resulting mask. The TG "
                    "arms' pre-registration "
                    "(lab/experiments/planned/2026-08-27-tg-restriction.md) reports plan "
                    "worth as 'enormous by construction, decides nothing' and does not "
                    "require this path.")
            if tc.coda_sees_slots or tc.coda_token_cut > 0:
                _xh, g_pn, _ = self._tul_coda_without_slots(
                    x_coda, x0, bigram_emb, keep, labels, layout)
                out["ce_tokens_no_slots"] = g_pn["ce_main"]
            else:
                out["ce_tokens_no_slots"] = groups["ce_main"]

        if self.tul_gate is not None and g_traj is not None:
            # `depths` here is the REALISED depth per slot — the Poisson draw in training
            # and at fixed-depth eval, the gate's own stop index under `halt`. §6's target
            # is defined against that, so the halting arm is scored on what it actually did.
            _g = self.tul_gate.loss(g_traj, depths, layout) if layout.span_len is not None \
                else {}
            for _k, _v in _g.items():
                out[f"gate/{_k}"] = _v
            # The realised depth: the Poisson draw at fixed depth, the gate's own stop
            # index under `halt`. This is what separates the two arms, so it is logged.
            _vm = layout.slot_valid.float()
            out["gate/depth_mean"] = (depths.float() * _vm).sum() / _vm.sum().clamp(min=1)
            if groups is not None and self.cfg.tul.gate.lam > 0.0:
                groups = dict(groups)
                groups["loss_tokens"] = groups["loss"]
                groups["loss"] = groups["loss"] + self.cfg.tul.gate.lam * _g["loss_gate"]

        if self.fm_planner is not None and groups is not None:
            # THREE TERMS, THREE GRADIENT PATHS (morph/model/tul_fm.py header table):
            #   ce     -> backbone + E_slot/E_mask/W_prefix (already in groups["loss"])
            #   fm     -> the planner ONLY (context detached, target detached)
            #   sigreg -> the backbone, THROUGH the live pooled targets
            from morph.model.fm_planner import fm_loss as _fm_loss
            from morph.model.tul_fm import fm_sigreg_loss as _fm_sigreg

            fmc = self.cfg.fm
            groups = dict(groups)
            fm_val, fm_stats = _fm_loss(
                self.fm_planner, fm_ctx.detach().float(), fm_geom, self._fm_schedule,
                y=fm_y.detach().float(), loss_scale=self._fm_loss_scale)
            groups["fm"] = fm_val.detach()
            groups["fm_rel"] = fm_val.new_tensor(fm_stats["rel_loss"])
            groups["fm_weighted"] = (fmc.fm_weight * fm_val).detach()
            total = groups["loss"] + fmc.fm_weight * fm_val
            if fmc.sigreg_lambda > 0.0:
                sig = _fm_sigreg(fm_y, fm_geom.valid, fmc.sigreg_slices)
                groups["fm_sigreg"] = sig.detach()
                groups["fm_sigreg_weighted"] = (fmc.sigreg_lambda * sig).detach()
                total = total + fmc.sigreg_lambda * sig
            # UNDETACHED on purpose (the gate's `loss_tokens` precedent). It is the
            # token-CE tensor with its graph intact, which is what lets
            # tests/test_tul_fm1.py assert — with autograd.grad(..., allow_unused=True)
            # — that NO planner parameter is in the CE graph at all. A detached copy
            # would make that assertion vacuous. train.py detaches it when logging.
            groups["loss_tokens_only"] = groups["loss"]
            groups["loss"] = total

        if sigreg_loss is not None and groups is not None:
            groups = dict(groups)
            groups["sigreg"] = sigreg_loss.detach()
            _sw = tc.sigreg_lambda * self.sigreg_gate * sigreg_loss
            groups["sigreg_weighted"] = _sw.detach()
            groups["loss"] = groups["loss"] + _sw

        if mux_loss is not None and groups is not None:
            groups = dict(groups)
            groups["mux_local"] = mux_loss.detach()
            # The WEIGHTED term, exposed so train.py can report train/loss and
            # val loss as the MODEL's CE (the spectral-penalty precedent: an
            # auxiliary term inside train/loss makes the arm incomparable to its
            # control and lets the ppl divergence guard fire on the objective).
            _mw = tc.mux_beta * self.mux_gate * mux_loss
            groups["mux_weighted"] = _mw.detach()
            groups["loss"] = groups["loss"] + _mw

        if groups is not None:
            out.update(groups)
        else:
            # Generation (labels=None): full logits, with the structural slot id masked
            # out of the head (spec §3.1 / invariant 4 — "masked … at generation").
            # index_fill is out-of-place, so this is safe under grad as well as no_grad.
            out["logits"] = self.embed.attend(xh).index_fill(
                -1, torch.tensor([tc.slot_id], device=xh.device), float("-inf"))
            if self.tul_gate is not None:
                # The generator needs the model's OWN budget for each slot: how many
                # tokens the plan it just built covers (§8). It is the same tensor the
                # coda was conditioned on, never a second, separately-decoded one.
                out["gate_k"] = budget_ids
        out["layer_passes"] = self._tul_layer_passes(layout, depths, coda_positions)
        out["n_tokens"] = (~layout.slot_mask).sum()
        return out

    def _tul_plan_ablate(self, h_slots: Tensor, layout: SlotLayout, mode: str) -> Tensor:
        """Eval-only plan ablations for ``val/plan_worth_*`` (docs/tul-fm-probing.md §1).

        Applies to EVERY slot path — the FM planner's plans, the core loop's looped
        states, and GL1's one-step tap states — because it operates on ``h_slots`` just
        before :meth:`TULSlots.prefix_project`, which is the single point where any of
        them becomes something the coda can read.

        ``normal`` is the shipped path and returns its input unchanged, so the training
        forward is bit-identical to a model that has no ablation code at all.

        ``zero`` removes the plan's CONTENT AND the fact that a plan is there; ``shuffle``
        permutes whole slots WITHIN a row, removing only the correspondence. The doctrine
        is emphatic that the shuffle COST is the number to report whenever the zero cost
        is not comfortably positive, because the specificity FRACTION's denominator
        collapses through zero (the tg3b −55.4 % reading).
        """
        if mode == "normal":
            return h_slots
        if mode == "zero":
            return torch.zeros_like(h_slots)
        if mode != "shuffle":
            raise ValueError(f"plan_mode must be normal|zero|shuffle, got {mode!r}")
        B, S = layout.slot_valid.shape
        # Pads sort last (score 2.0 > any uniform draw), so real slots are permuted
        # among the real slot POSITIONS only — SlotLayout guarantees pads are last.
        r = torch.rand(B, S, device=h_slots.device)
        r = torch.where(layout.slot_valid, r, torch.full_like(r, 2.0))
        perm = r.argsort(dim=1)
        idx = perm.reshape(B, S, *([1] * (h_slots.dim() - 2))).expand_as(h_slots)
        return h_slots.gather(1, idx)

    def tul_forward_ablated(self, input_ids: Tensor, labels: Tensor | None,
                            layout: SlotLayout, plan_mode: str = "normal") -> dict:
        """Eval-only forward with the slot state ablated. Works on ANY TUL arm.

        ``normal`` — the shipped path.
        ``zero``   — the slot values written into the coda are zeroed. Removes the
                     plan's content AND the fact that a plan is there.
        ``shuffle``— whole slots permuted WITHIN a row. Removes only the correspondence
                     between a slot and its span, which is what makes it the
                     span-SPECIFICITY number. Report the shuffle COST, never a
                     specificity fraction (docs/tul-fm-probing.md §4 rule 1).
        ``wrong_seed`` — THE WRONG-PLAN PROBE. Swaps ``tul.slot_seed`` for a mode the arm
                     was NOT trained on, so the slot carries a valid-but-wrong value
                     instead of no value. This is the instrument that caught TG4b's
                     value-sensitivity (0.48-0.56 nats where zeroing cost 0.10 —
                     "removing LESS hurts MORE"), and it works because
                     :meth:`TULSlots.slot_input` dispatches on ``slot_seed`` at CALL
                     time, not at construction (``lab/divergence/slot_path_worth.py``
                     ``seed_bagmean``). READ IT AS OOD SHOCK, NOT AS WORTH: that file
                     measured the forced fallback costing 6-7x more than zeroing the
                     plan outright, because swapping a seed the weights never saw
                     measures the distribution shift. It answers "does the coda read the
                     slot's VALUE at all", and nothing else.

        A separate entry point rather than a forward flag, for the reason
        :meth:`tul_forward_with_plan_nats` gives: the training path must not carry a
        branch that decides how much work to do.
        """
        if self.tul is None:
            raise RuntimeError(
                "tul_forward_ablated needs a model built with MORPHConfig(tul=...)")
        if plan_mode != "wrong_seed":
            return self._forward_single(input_ids, labels, 0, None, layout,
                                        _plan_mode=plan_mode)
        tc = self.cfg.tul
        orig = tc.slot_seed
        alt = "bag_mean" if orig != "bag_mean" else "e_slot"
        tc.slot_seed = alt
        try:
            return self._forward_single(input_ids, labels, 0, None, layout)
        finally:
            tc.slot_seed = orig

    def tul_fm_forward(self, input_ids: Tensor, labels: Tensor | None,
                       layout: SlotLayout, plan_mode: str = "normal") -> dict:
        """FM1's name for :meth:`tul_forward_ablated`, kept so the FM1 gates and the
        trainer's FM eval block keep pointing at the same call."""
        if self.fm_planner is None:
            raise RuntimeError("tul_fm_forward needs a model built with MORPHConfig(fm=...)")
        return self.tul_forward_ablated(input_ids, labels, layout, plan_mode)

    @torch.no_grad()
    def tul_slot_state_probe(self, input_ids: Tensor, layout: SlotLayout) -> dict:
        """Eval-only: the homogeneity dial. Effective rank and mean pairwise cosine of
        the WRITTEN slot states, read at the point the coda reads them.

        This is the number arm GL1 exists to move. TG4b measured the gisting pipe
        WORKING as a pipe — a wrong-but-present plan cost 0.48-0.56 nats where zeroing
        cost 0.10 — while shuffling whole slots cost ~0. Both readings are true at once
        only if the written states are near-IDENTICAL across slots: the coda reads the
        value, and every value is the same value. Slot states across the campaign sat at
        effective rank 1.7-4.8 in 1024 dims with mean pairwise cosine +0.39..+0.71.
        SIGReg is aimed exactly here, and without this probe its effect is invisible.
        """
        if self.tul is None:
            raise RuntimeError("tul_slot_state_probe needs MORPHConfig(tul=...)")
        from morph.model.fm_planner import effective_rank, mean_pairwise_cos

        x, x0, bigram = self._tul_front(input_ids, layout)
        _xn, h_slots, _d, _g = self._tul_core(x, x0, bigram, layout)
        z = self._readout(h_slots).float()                     # [B, S, C]
        valid = layout.slot_valid
        rows = z[valid]
        return {
            "slot_eff_rank": effective_rank(z, valid),
            "slot_pairwise_cos": mean_pairwise_cos(z, valid),
            "slot_norm_mean": float(rows.norm(dim=-1).mean()),
            # The scale SIGReg's statistic actually sees. `_readout` ends in RMSNorm, so
            # this sits at ~1 by construction and no standardisation is applied before
            # the statistic — standardising would make the loss vacuous (see
            # morph/model/sigreg.py). Logged so a change to the readout cannot silently
            # move the target the regulariser is chasing.
            "slot_component_std": float(rows.std()),
            "slot_component_mean": float(rows.mean()),
        }

    @torch.no_grad()
    def fm_eval_probe(self, input_ids: Tensor, layout: SlotLayout) -> dict:
        """Eval-only FM1 instruments: target geometry and the copy gap.

        Re-runs the prelude rather than plumbing tensors out of the training forward —
        the training path must not carry eval bookkeeping, and eval is 20 batches.

        ``copy_gap`` is the number P1 was taught by: a zero-parameter baseline that
        guesses "the next span looks like the current one" scored 0.0678 within-row
        top-1 while the trained standalone planner scored 0.0516. A retrieval figure
        that is not reported against that baseline says nothing.
        """
        if self.fm_planner is None:
            raise RuntimeError("fm_eval_probe needs a model built with MORPHConfig(fm=...)")
        from morph.model.fm_planner import effective_rank, mean_pairwise_cos
        from morph.model.tul_fm import copy_gap_scores

        x, _x0, _bg = self._tul_front(input_ids, layout)
        _xn, h_slots, y, geom, _ctx = self._tul_fm_core(x, layout)
        # Undo the stream broadcast: every stream carries the same plan by construction.
        z = h_slots[:, :, 0, :] if self._is_hc else h_slots
        out = copy_gap_scores(z.float(), y.float(), geom.valid)
        out["target_eff_rank"] = effective_rank(y.float(), geom.valid)
        out["target_pairwise_cos"] = mean_pairwise_cos(y.float(), geom.valid)
        out["target_norm_mean"] = float(y[geom.valid].norm(dim=-1).mean())
        out["fm_slots_valid"] = float(geom.valid.sum())
        return out

    def _tul_coda_drop_mask(self, layout: SlotLayout, tc: TULConfig) -> Tensor:
        """``[B, L]`` bool union drop-mask for :meth:`_tul_coda_gather`: every slot (arm
        A4, ``coda_sees_slots=False``) and/or every token below the cut (arm CW,
        ``coda_token_cut>0`` — docs/tul-compaction-window-spec.md)."""
        if not tc.coda_sees_slots:
            drop = layout.slot_mask
            if tc.coda_token_cut > 0:
                drop = drop | window_drop_mask(layout.slot_mask, tc.coda_token_cut)
            return drop
        return window_drop_mask(layout.slot_mask, tc.coda_token_cut)

    def _tul_budget_ids(self, layout: SlotLayout, depths: Tensor, g_traj: Tensor) -> Tensor:
        """``[B, max_slots]`` token budget to condition the coda on (gate §4/§5/§9).

        The layout carrying a length label IS the training/eval signal: teacher force the
        REALISED length there, and use the model's OWN choice when it does not (generation,
        where ``TulRowBuilder`` builds the layout and no label exists). Never a mixture —
        mixing makes the LM loss chase the gate's error, and scheduled sampling is §12's
        unbuilt key, not a silent default.
        """
        if layout.span_len is not None:
            return layout.span_len
        g_fin = g_traj.gather(2, (depths - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1)
        k = self.tul_gate.choose_k(g_fin).clamp(min=1)
        return torch.where(layout.slot_valid, k, torch.zeros_like(k))

    def tul_forward_halt(self, input_ids: Tensor, labels: Tensor | None,
                         slot_layout: SlotLayout) -> dict:
        """Eval-only: arm ``TUL-halt`` — the gate chooses each slot's loop depth (§7).

        A separate entry point rather than a forward flag, following
        :meth:`tul_forward_with_plan_nats`: the training path must not carry a branch that
        decides how much work to do. Scoring the SAME checkpoint through this and through
        the ordinary forward is the whole bake-off — §4 teacher-forces the depth, so the
        two arms share every weight and the comparison is exactly paired.
        """
        return self._forward_single(input_ids, labels, 0, None, slot_layout, _halt=True)

    def _tul_coda_without_slots(self, x_coda, x0, bigram_emb, keep, labels, layout):
        """Run the coda on the TOKEN positions only (arm A4 and the plan-nats metric)."""
        return self._tul_coda_gather(x_coda, x0, bigram_emb, keep, labels, layout,
                                     layout.slot_mask)

    def _tul_coda_gather(self, x_coda, x0, bigram_emb, keep, labels, layout, drop_mask):
        """Run the coda after gathering ``drop_mask`` positions out of the sequence.

        Generalises the old slots-only gather (``drop_mask = layout.slot_mask``, arm A4)
        to also serve arm CW (``drop_mask`` = every token below the cut, or that unioned
        with every slot — docs/tul-compaction-window-spec.md). The gather itself does not
        care what kind of position was dropped; it only needs the boolean mask, which is
        why one function now serves both arms with no new indexing logic (spec §"the
        change": "Reuse compact_index, gather_positions, and its _g padding helper").
        """
        cidx = compact_index(drop_mask)
        B, L = drop_mask.shape

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
            groups = self._tul_group_losses(xh, labc, None, want_groups=not self.training)
        return xh, groups, L

    def _tul_coda_prep(self, input_ids: Tensor, layout: SlotLayout):
        """Front + core + token-state-dropout — the part of :meth:`_forward_tul` that is
        IDENTICAL across every arm CW variant (docs/tul-compaction-window-spec.md): which
        positions the CODA sees is decided after this point, never before it. Shared by
        :meth:`_forward_tul` and :meth:`tul_forward_cw_arms` so the (expensive) core loop
        runs once per input, not once per arm.
        """
        tc = self.cfg.tul
        if tc.tokens_through_core:
            raise NotImplementedError(
                "tul.tokens_through_core (arm A2) has no defined interaction with arm CW "
                "(docs/tul-compaction-window-spec.md) — it is not specified, so this "
                "raises rather than silently picking a behaviour."
            )
        x, x0, bigram_emb = self._tul_front(input_ids, layout)
        xn, h_slots, depths, g_traj = self._tul_core(x, x0, bigram_emb, layout)
        if self.tul_gate is not None:
            h_slots = self.tul_gate.apply_budget(
                h_slots, self._tul_budget_ids(layout, depths, g_traj))
        values, pos = self.tul.prefix_project(h_slots, layout, layout.l_total)
        x_coda = scatter_positions(xn, pos, values)
        x_coda, keep = self.tul.apply_token_dropout(x_coda, layout, self.training)
        return x_coda, x0, bigram_emb, keep, depths

    def tul_forward_cw_arms(self, input_ids: Tensor, labels: Tensor, layout: SlotLayout,
                            cut: int, seed: int = 0) -> dict[str, dict]:
        """Eval-only: score CW0/CW1/CW2/CW3 in one pass (docs/tul-compaction-window-spec.md).

        Every arm scores CE over the SAME set of labels — original TOKEN positions with
        row index ``>= cut`` — so the four numbers are directly comparable; that
        restriction is applied ONCE, to ``labels``, before any arm's gather, rather than
        four separate times. The front/core/token-dropout prefix is shared (one core loop
        for all four arms, not four); only the coda gather differs per arm.

        Args:
            cut:  ``C`` in the spec. Must be ``0 <= cut < seq_len``.
            seed: seeds arm CW2's random retention (:func:`morph.model.tul.cw2_retain_mask`).
                  Log this — a different seed picks a different random subset.

        Returns:
            ``{"CW0": groups, "CW1": groups, "CW2": groups, "CW3": groups}``, each a
            :meth:`_tul_group_losses` dict (``layout=None`` convention: ``loss`` ==
            ``ce_main`` == ``ce_tokens``, plain unweighted CE, no slot half-weighting —
            every arm goes through the same code path so there is no weighting asymmetry
            between them). ``n_targets`` is identical across all four by construction.
        """
        if self._tg_restrict:
            raise NotImplementedError(
                "tul.tg_restrict has no defined interaction with arm CW "
                "(tul_forward_cw_arms always runs the gathered-subset coda — see the "
                "same raise in _forward_tul). docs/tul-tg-spec.md does not specify it.")
        B, L = layout.slot_mask.shape
        if not 0 <= cut < L:
            raise ValueError(
                f"arm CW cut={cut} must satisfy 0 <= cut < seq_len={L} "
                f"(docs/tul-compaction-window-spec.md)."
            )
        x_coda, x0, bigram_emb, keep, _depths = self._tul_coda_prep(input_ids, layout)

        pos = torch.arange(L, device=layout.slot_mask.device).unsqueeze(0).expand(B, L)
        score_mask = (~layout.slot_mask) & (pos >= cut)     # SAME labels for all four arms
        labels_scored = torch.where(score_mask, labels, torch.full_like(labels, -100))

        early_tok = window_drop_mask(layout.slot_mask, cut)         # candidates for CW2
        budget = layout.prefix_k * layout.slot_valid.sum(dim=1)      # spec: prefix_k * n_valid
        retain = cw2_retain_mask(early_tok, budget, seed)

        drop_masks = {
            "CW0": layout.slot_mask.new_zeros(layout.slot_mask.shape),        # ceiling
            "CW1": early_tok,                                                 # the claim
            "CW2": layout.slot_mask | (early_tok & ~retain),                  # the decider
            "CW3": layout.slot_mask | early_tok,                              # floor
        }
        out: dict[str, dict] = {}
        for name, drop_mask in drop_masks.items():
            _xh, groups, _pos = self._tul_coda_gather(
                x_coda, x0, bigram_emb, keep, labels_scored, layout, drop_mask)
            out[name] = groups
        return out

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

        Under ``tg_restrict`` the plan-nats gather is undefined (its no-slots coda runs
        on a gathered index space the tg masks are not re-derived for — see the raise
        in ``_forward_tul``) and the pre-registration
        (lab/experiments/planned/2026-08-27-tg-restriction.md) declares plan worth
        non-discriminating there ("enormous by construction"). Eval therefore SKIPS the
        ablation pass on a TG model: ``val/plan_nats`` is simply absent from the logs
        (evaluate() already guards on the key), instead of every TG training run dying
        at its first eval step. Plan/loop worth for the TG arms comes from
        ``lab/divergence/slot_path_worth.py``, which zeroes ``prefix_project`` VALUES
        on the full-L sequence — fully defined under the restriction.
        """
        return self._forward_single(input_ids, labels, 0, None, slot_layout,
                                    _plan_nats=not self._tg_restrict)

    def _forward_single(self, input_ids: Tensor,
                        labels: Tensor | None = None,
                        bag_size: int = 0,
                        seq_lens: Tensor | None = None,
                        slot_layout: SlotLayout | None = None,
                        _plan_nats: bool = False,
                        _halt: bool = False,
                        _plan_mode: str = "normal") -> dict:
        if slot_layout is not None:
            if bag_size > 0:
                raise ValueError(
                    "slot_layout and bag_size are mutually exclusive: TUL activates AT the "
                    "TST switch (spec §5), and val/gen always run TUL on with bag_size 0 "
                    "(invariant 6)."
                )
            return self._forward_tul(input_ids, labels, slot_layout, _plan_nats,
                                     halt=_halt, plan_mode=_plan_mode)
        if self._tg_restrict:
            # docs/tul-tg-spec.md builds the restriction as a per-forward DATA argument
            # derived from the layout — there is no defined "unrestricted" fallback for
            # a tg_restrict model, and every real call site (train.py, tul_generate.py)
            # always supplies a layout once TUL is active. A missing layout here would
            # otherwise silently run the plain/TST path with none of the restriction
            # applied — the exact silent-fallback theater the spec forbids.
            raise RuntimeError(
                "model built with tul.tg_restrict=true but forward() got no slot_layout "
                "(docs/tul-tg-spec.md): there is no unrestricted fallback path for a TG "
                "model. Pass slot_layout explicitly.")
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
