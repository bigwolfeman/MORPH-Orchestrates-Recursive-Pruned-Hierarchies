"""MORPH Transformer — Parcae-style looped architecture with all features baked in.

Architecture: prelude → core×T (diagonal injection) → coda
Loop hierarchy:
  Inner: Parcae core loop (T iterations with Poisson depth sampling)
  Outer: (Zyphra RSA — deferred, inference-time, requires RL)

All features always on. No runtime if-statements in the forward pass.
Config determines dimensions and sizes, not whether features exist.
"""

from __future__ import annotations

import copy
import math
import functools
import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .attention import MORPHAttention, RMSNorm
from .diffusion_blocks import euler_step
from .embeddings import MORPHEmbedding
from .fused_ce import (
    fused_linear_cross_entropy,
    fused_linear_cross_entropy_mce,
    multi_hot_cross_entropy_reference,
)
from .iter_cond import CoreStageConditioning, DB1Sampler, iter_stage_value
from .recur_gate import RecurrenceGate
from .mhc import ChannelInject, MORPHBlock, PassLoRA, DEFAULT_CHANNEL_DIMS
from .sigreg import sigreg_epps_pulley
from .sparsity import MortarLinear
from .tul import (TULCenterExit, TULConfig, TULGate, TULGateConfig, TULGradPass,
                  TULPassGate,
                  TULReread, TULRowContrast, TULSlotChain,
                  TULSlotRegister, TULSlots,
                  boundary_token_index,
                  compact_index, next_span_pool,
                  cw2_retain_mask, gather_positions, gather_valid, mux_span_targets,
                  scatter_positions,
                  window_drop_mask)
from .tul_carry import TULLoopCarry
from .tul_fan import (FanReservoir, TULFanMix, fan_epi_term, fan_repel_term, fan_stream_rank,
                      fan_stream_stats)
from .tul_egrad import (CriticEnergy, DiscEnergy, ReconEnergy,
                        slot_outcome_labels)
from .tul_spandec import SpanDecoder, horizon_span_slots, next_span_slots, span_slots
from .tul_vq import TULThoughtVQ
from .tul_code import (TULCodeEncoder, TULCodeHead, TULCodeProj, TULCodeTime,
                       cfm_null_floor, cfm_pair, code_rmsnorm, code_target_infonce,
                       code_target_regression, code_target_shuffled_cos,
                       code_thinker_relation, euler_sample,
                       code_grade_distinct2, code_grade_pref_loss,
                       TULCodeSymHead, mdm_loss, mdm_mask, maskgit_sample)
from .tul_layout import (SlotLayout, span_allow_mask, span_ids_from_ids,
                         slot_cell_inject_keep, span_start_mask, tg_allow_mask,
                         tg_reset_from_ids,
                         tg_reset_mask, tg_segment_ids, tg_strict_allow)

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


# Attention kwargs `_core_region` can thread through its active-set sort. Each is a
# PER-SAMPLE tensor with batch as dim 0, so `[perm]` and `[:n_active]` are exact. Anything
# else (e.g. `tg_span`, a dict of per-span index tensors) raises there rather than being
# silently dropped — see the raise for the reason. `tg_relation` (the Thought Register's
# cell relation) is deliberately absent: it is a [1,1,S*M,S*M] relation over the SLOT
# loop's compact cell axis, `tul.slot_cells>1` refuses `tokens_through_core`, and a
# batch-less tensor is not permutable by `[perm]`. Reaching `_core_region` with it is a
# bug and raises there.
_CORE_TG_KEYS = frozenset({"tg_allow", "tg_slot_mask", "tg_comp_allow", "tg_seg"})


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
    # ``depth_fixed``: every sample runs EXACTLY ``max_depth`` core iterations at train
    # (eval already runs ``mean_depth`` for every row). Off, ``_sample_depths`` draws
    # Poisson(mean_depth) clamped to [1, max_depth], so ``mean_depth == max_depth`` is NOT
    # a fixed depth: a mean-6 / max-6 model still runs about a third of its rows at depth
    # 1-5 (the 2026-09-14 LoopMTP smoke died on exactly that draw). Requires
    # mean_depth == max_depth. The depth-ladder rungs (notul_norm_match_20k_d{1,2,3})
    # keep the clamped draw on purpose; the LoopMTP arms and their *fixed controls set it.
    depth_fixed: bool = False

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

    # WHICH BLOCK the looped core is built from. "morph" (the default, and every run
    # before 2026-09-10) = the ordinary MORPHBlock: ternary-STE MortarLinear MLP,
    # CCA+CSA/HCA+XSA attention, 4-stream Cayley hyper-connection residual. "parcae" =
    # `morph/model/parcae_core.py`'s ParcaeCoreBlock: dense causal softmax attention,
    # a plain single-stream additive residual, a dense `_SwiGLU` MLP, never ternarised
    # and never pruned/carved. The prelude and the coda are MORPH blocks either way —
    # they run once and are not the question — so the stream boundary is one collapse
    # (stream mean) at the core's entry and one broadcast at its exit, both inside
    # `_apply_core_step`. Arm `slot-mnext-parcae-core`, the one-factor test of whether
    # the slot loop's flat K-curve is MORPH's core at the slot shape or the TUL
    # mechanism itself (lab/experiments/planned/2026-09-10-arc-slot-mnext-parcae-core.md).
    core_impl: str = "morph"
    # Head width of the Parcae core's attention. None -> `d_model // (compression *
    # n_heads)`, the SAME d_head MORPHAttention computes, so the swap is roughly
    # parameter-matched and a depth gain cannot be read as extra capacity.
    parcae_core_d_head: int | None = None

    # ── The cross-span information budget (model.span_mask) ─────────────────────────
    # "off"  (default) — the tree as it was. Nothing below is built or computed.
    # "row"  — the maskable path is built and every row is ONE span, so the relation
    #          degenerates to plain causal. Arm `budget-web-full`.
    # "span" — the same path with the ONE TUL BoundaryRule cutting the row. Arm
    #          `budget-web-span`: NOTHING crosses a span boundary.
    # "row" and "span" run the IDENTICAL code and differ only in the span ids computed
    # per batch, so their CE gap is the number of nats that live across span boundaries
    # and not an operator change. Requires `core_impl: parcae` (a pooled compressed
    # block spans several spans and has no same-span restriction) and `use_kernels:
    # false`. `span_rule` is the resolved BoundaryRule; `build_morph_config` fills it
    # from the tokenizer for "span" and leaves it None for "row".
    # Record: .agents/notes/proposed/architecture/2026-09-11-cross-span-budget.md
    span_mask: str = "off"
    span_rule: object | None = None

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

    # Parallel multi-token prediction on the coda readout (Gloeckle et al. 2024, arXiv
    # 2404.19737; arc E8, 2026-09-07 [W]). mtp_heads = the number of future tokens each
    # position is trained to predict: 1 = the plain next-token head only (bit-identical, no
    # module built); k > 1 adds k-1 parallel heads, each RMSNorm -> Linear(d, d) at IDENTITY
    # init (zero RNG draws) feeding the SAME tied LM head, with labels shifted by 1..k-1 and
    # the tail padded to -100. Loss = CE_1 + mtp_weight * sum_{j=2..k} CE_j. The heads read
    # the readout state ONLY (no attention of their own), so every bit of lookahead must
    # already be in the position's state: the strict form of the target-side lever (b).
    # Undefined (raises) with 3-D TST bag labels and on the TUL slot path.
    mtp_heads: int = 1
    mtp_weight: float = 1.0

    # ── LoopMTP (arXiv 2608.03624, Shomali et al. 2026) ─────────────────────────
    # Two independent knobs, BOTH off by default and bit-identical when off. They make
    # MORPH's core iteration t the paper's loop iteration t: the shared 6-block core
    # applied T times between the prelude and the coda.
    #
    # (a) `core_readout`. "last" = the coda reads the FINAL iterate (the tree as it is).
    #     "gated" = the coda reads the paper's content-conditional aggregate of ALL T
    #     iterates (Eq 9-11):
    #         g^(t)      = softplus(W_g x^(t) + beta_t * 1_d)          [Eq 9]
    #         gtilde^(t) = g^(t) / (sum_s g^(s) + eps)                 [Eq 10, ELEMENTWISE]
    #         z          = sum_t gtilde^(t) (*) x^(t)                  [Eq 11, t = 1..T]
    #     W_g is ONE [d, d] linear shared across iterations, zero-initialised, and beta is
    #     T scalars initialised to 0 — so at init every gate is softplus(0) and z is the
    #     UNIFORM mean of the T iterates (the paper's "All (uniform)" variant, its
    #     second-best in Fig 4 left). Both inits draw NO RNG, so a gated model shares every
    #     base weight with the same-seed ungated one.
    # (b) `loopmtp_weight` (lambda_align). The soft multi-token-prediction target of Eq 12-13:
    #         L_align^(t) = mean_i (1 - cos(x_i^(t), sg[E_{u_{i+t}}]))
    #         L_align     = 1/(T-1) sum_{t=2..T} L_align^(t)           [free first iterate]
    #     i.e. iteration t is asked to anticipate the token t steps ahead of position i,
    #     against the DETACHED tied output embedding. TRAINING ONLY (like
    #     core_fixed_point_lambda), so val CE stays the number every arm is compared on.
    #     `loopmtp_free_first` = the paper's "iteration 1 is left unconstrained"; False
    #     supervises iteration 1 too (target u_{i+1}) and averages over all T.
    # (c) `loopmtp_proj`. MORPH DEVIATION. In the paper x^(t) is the state that feeds the
    #     LM head directly, so the cosine against an unembedding row is a comparison in the
    #     head's own space and needs no projection. MORPH puts THREE coda blocks, the
    #     lm_mixer and final_norm between the core and the head, so the raw core state does
    #     not live in that space. "linear" = one SHARED RMSNorm -> Linear(d, d) at identity
    #     init (zero RNG) on the state side, never ternarised; "none" = the paper's literal
    #     form, kept as the ablation. Shared, not per-iteration, so the iterates themselves
    #     have to differ — a per-iteration projection could fake the differentiation.
    # (d) `loopmtp_ponder_weight` (lambda_ponder, paper default 0.05). The paper's ponder
    #     regulariser: KL(mean-over-dims gate distribution || uniform), averaged over
    #     positions, which pulls the aggregator away from collapsing onto one iterate.
    #     Defined only with core_readout "gated" (it acts on the gate), training only.
    # (e) `loopmtp_gate_eps`. The epsilon of Eq 10's denominator, the paper's `eps`. It
    #     only matters where every raw gate is near zero; it is a config field rather than
    #     a constant so a run is reproducible from its wandb config alone.
    # Both knobs REQUIRE a fixed loop depth (mean_depth == max_depth) and full BPTT
    # (bptt_depth >= mean_depth): the paper's T is a constant and every iterate is
    # supervised, and a Poisson draw would give different samples different target sets.
    # A Poisson config RAISES at build rather than silently averaging over depths.
    core_readout: str = "last"
    loopmtp_weight: float = 0.0
    loopmtp_free_first: bool = True
    loopmtp_proj: str = "linear"
    loopmtp_ponder_weight: float = 0.0
    loopmtp_gate_eps: float = 1.0e-6

    # The diagonal state carry (Parcae arXiv 2604.12946 §4.1: rho(A) < 1 on the WHOLE residual
    # is the paper's stability claim). "ctx" = the shipped DiagonalInjection on the context
    # channel only (256 of 768 dims; the rest ride the norm-preserving HC residual, Parcae's
    # Table 1 "marginally stable" row) — bit-identical. "all" = the same carry on every carrier
    # dim: the context channel keeps its legacy init (decay 0.447, dt 1.0); the other dims start
    # at decay `injection_all_decay` with dt = 1 - decay, so the linear part's fixed point is
    # e itself (scale-preserving at init). Arc E9, 2026-09-07 [W: "we wanna try widening the
    # diag carry"]. Old checkpoints do not load into an "all" model (the carry is 768-wide).
    injection_channels: str = "ctx"
    injection_all_decay: float = 0.9
    # Parcae's loop entry, faithful (arXiv 2604.12946 §4.1; measured on its 140m OWT
    # checkpoint 2026-09-09: the state starts as noise at 4 % of its converged scale, the
    # adapter re-injects e on EVERY channel through a learned B, and each recurrent block
    # moves the state by ~40 % of its norm per pass; MORPH's h_0 = e entry with the ctx-only
    # carry leaves the recurrence 0.02 nats of work — .agents/notes/proposed/architecture/
    # 2026-09-09-depth-lobotomy-candidates.md). `injection_all_dt` (with "all"): one dt for
    # EVERY dim, ctx included, replacing the legacy split init; None keeps the E9 init.
    # `injection_B`: a full [d, d] identity-initialised matrix on the injected e (Parcae's
    # `ssm_B_identity`); no weight decay (the `injection` keyword), never ternarised (a raw
    # parameter, not a Linear). `core_state_init`: "prelude" = h_0 = e (the shipped entry);
    # "noise" = h_0 ~ N(0, core_state_init_std^2) per element (Parcae's like-init; 0.02 is
    # its embedding-init std), REQUIRES injection_channels "all" — with the ctx-only carry
    # 704 of 1024 dims would stay noise forever.
    injection_all_dt: float | None = None
    injection_B: bool = False
    core_state_init: str = "prelude"
    core_state_init_std: float = 0.02

    # Arc E10 (2026-09-07), two loss terms on the PLAIN loop (`_core_region`), both 0 = off and
    # bit-identical; both training-only; both raise on the TUL slot path.
    # (a) Terminal fixed-point objective (arXiv 2608.18222): lambda * mean_b ||h_T - h_{T-1}||^2 /
    #     ||h_T||^2 at each sample's LAST iteration — asks the loop to settle.
    core_fixed_point_lambda: float = 0.0
    # (b) Directional gain hinge (STARS arXiv 2605.26733 by finite difference): at one random
    #     grad iteration, g = ||f(h + d) - f(h)|| / ||d|| with d = core_gain_eps * ||h|| * v;
    #     penalty lambda * relu(g - core_gain_target)^2. direction "power": v is a persistent
    #     buffer updated by one power step per training step (v <- normalize(mean_b (f(h+d) -
    #     f(h)))), so g tracks the map's TOP singular value (healthy arms read 11-22 there,
    #     E7's sick map 95; the target is on sigma_max, not the typical gain). "random": a
    #     fresh Gaussian v each step (the slot hinge's reading).
    core_gain_lambda: float = 0.0
    core_gain_target: float = 20.0
    core_gain_eps: float = 0.02
    core_gain_direction: str = "power"
    # Within-step power iterations for the directional hinge (arc E10c). 0 = the one-shot
    # reading (the direction is only refined ACROSS steps, and every step is a new batch, so
    # the batch-averaged response reads ~1.1-1.5 on the plain loop — E10b was inert). k > 0
    # applies k extra finite-difference power steps at the SAME input before the reading
    # (each is one more core-step application), warm-started from the buffer, so the reading
    # approaches sigma_max for this batch. With core_gain_target 0 the hinge is STARS' plain
    # lambda * g^2.
    core_gain_power_iters: int = 0

    # Master kernel switch. True = fused Triton attention + fused chunked CE
    # (the optimised stack). False = eager PyTorch references + full-logits CE
    # (the un-optimised baseline) — same architecture/weights, for A/B on memory
    # and throughput. The bit-exact loop opts (x0-hoist, active-set) stay on in
    # BOTH arms (they are not "kernels" and have no downside).
    use_kernels: bool = True
    tg_scoped_kernels: bool = False  # tg_restrict only: leave the process-global force_eager
                                     # flag OFF so the structurally-safe fused kernels engage
                                     # (HC-Cayley, CCA prologue/conv, the core-region window —
                                     # where the TG restriction is vacuous: every core position
                                     # is a slot). Every TG-restricted branch stays eager by
                                     # construction: prelude/coda window calls always carry
                                     # tg_allow (extra_mask routes to the reference path
                                     # unconditionally) and the TG compressed branches are
                                     # pure eager functions. CE stays chunked (that branch
                                     # reads use_kernels directly).

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
    retention_sections: tuple[str, ...] = ("prelude", "core", "coda")  # which sections get it
    retention_write_shift: bool = False  # FWA next-latent write alignment (k_{t-1}, v_t)
    retention_heads: int = 0                   # 0 → use n_heads
    retention_chunk: int = 128
    retention_gate_init: float = -6.0          # branch-gate logit; sigmoid(-6)≈0.0025 ≈ identity@init
    # Cross-iteration GLA carry mode. "none" (DEFAULT since 2026-08-31): the core's GLA
    # state resets each loop iteration — strictly causal (runtime-invariants §5).
    # "acausal_final": the pre-fix behaviour — iteration t's END-OF-SEQUENCE state (a
    # summary of ALL positions, future included) seeds iteration t+1, so from iteration 2
    # every position sees the future. It is a LEARNED leak: 0.14 nats on a truncated-BPTT
    # arm, 3.85 nats after 30k full-BPTT steps, and it faked the l2cap depth-earning
    # (lab/experiments/successes/2026-08-31-carry-leak-audit.md). Kept ONLY as an explicit
    # opt-in for loading/diagnosing checkpoints trained before the fix. Bools are accepted
    # for config back-compat: False → "none", True → "acausal_final".
    retention_carry: bool | str = "none"
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
    # Clip-through-time on the SLOT loop's backward (DIVERGENCE-README §D). In the backward
    # the cotangent arriving at each slot-loop iteration's output is rescaled, per row, so
    # its norm never exceeds this ratio times the norm of the cotangent that arrived at the
    # loop's EXIT state. The exit cotangent itself is never touched, the forward is never
    # touched, and 0.0 registers no hook at all (bit-identical). The forecast spike train is
    # exactly this product growing 39-2436x through eight iterations against 1.6-3x on calm
    # steps, with a flat forward — so the lever bounds the product and nothing else.
    slot_cot_clip: float = 0.0
    # Phase 2 of the contractivity programme (.agents/notes/proposed/architecture/
    # 2026-09-04-loop-contractivity-as-design.md), two levers on the SLOT loop's forward map:
    #
    # slot_state_renorm — after every iteration the carried slot state is rescaled, per slot,
    # to the norm it ENTERED the loop with (direction preserved, pads stay 0). Bounds the
    # forward by construction: on the clipped M-next draw the exit norm inflated 5x and
    # iteration 0's realised gain climbed 1.5 -> 9.6 while the backward was bounded. False
    # traces the identical graph.
    slot_state_renorm: bool = False
    # slot_gain_lambda — a hinge penalty on the map's TYPICAL gain at one grad iteration per
    # step: lambda * relu(g - slot_gain_target)^2 with g = ||f(h + d) - f(h)|| / ||d|| for a
    # random direction d of relative size slot_gain_eps per slot (finite difference, two
    # extra core steps at the same dropout masks, the global RNG left exactly where it was).
    # This is the quantity the capture measured drifting 0.87 -> 1.00 (`jac/rms_t3`), on the
    # whole map — the thing the four weight-spectrum caps could not see. 0.0 = OFF.
    slot_gain_lambda: float = 0.0
    slot_gain_target: float = 0.9
    slot_gain_eps: float = 0.02
    # slot_gain_all_iters: regularise EVERY grad iteration of the slot loop instead of one
    # random one per step (2·n_grad_iters extra core steps on the compact slot sequence).
    # Needed when the map differs per iteration (tul.core_stage_cond="iter"): one random
    # sample read 0.53–0.61 while another iteration's map went expansive and the arm
    # detonated (arc E2, 2026-09-04). The penalty is the SUM of the per-iteration hinges;
    # gain / gain_max are the mean / max over the iterations.
    slot_gain_all_iters: bool = False

    @property
    def retention_carry_mode(self) -> str:
        """Normalized ``retention_carry``: "none" | "acausal_final" (bools mapped).

        Every read site MUST use this property, never the raw field — a raw
        truthiness check reads the string "none" as True and silently resurrects
        the causality leak this normalization exists to kill."""
        v = self.retention_carry
        if isinstance(v, bool):
            v = "acausal_final" if v else "none"
        if v not in ("none", "acausal_final"):
            raise ValueError(
                f"retention_carry must be 'none', 'acausal_final', or a bool; got {v!r}")
        return v


class DiagonalInjection(nn.Module):
    """SSM-style diagonal injection on the context channel only.

    h_ctx = decay * h_ctx + dt * e_ctx
    Spectral radius < 1 guaranteed by construction.
    """

    def __init__(self, channel_start: int, channel_end: int, init_decay: float = 0.447,
                 init_decay_vec: Tensor | None = None, init_dt_vec: Tensor | None = None,
                 use_B: bool = False):
        super().__init__()
        self.start = channel_start
        self.end = channel_end
        d = channel_end - channel_start
        # Parcae's B (identity init, no RNG draw): new_ctx = A * h_ctx + dt * (e_ctx @ B^T).
        # Registered LAST so a model without it keeps byte-identical parameters and RNG.
        self.B: nn.Parameter | None = nn.Parameter(torch.eye(d)) if use_B else None
        if init_decay_vec is None:
            self.log_A = nn.Parameter(torch.full((d,), float(init_decay)).log())
            self.log_dt = nn.Parameter(torch.zeros(d))
        else:
            # Per-dim init (injection_channels="all"): deterministic, no RNG draw.
            assert init_decay_vec.shape == (d,) and init_dt_vec.shape == (d,)
            self.log_A = nn.Parameter(init_decay_vec.float().log())
            self.log_dt = nn.Parameter(init_dt_vec.float().log())

    def forward(self, h: Tensor, e: Tensor) -> Tensor:
        A = self.log_A.exp().clamp(max=0.9999)
        dt = self.log_dt.exp()
        h_ctx = h[..., self.start:self.end]
        e_ctx = e[..., self.start:self.end]
        if self.B is not None:
            e_ctx = e_ctx @ self.B.to(e_ctx.dtype).T
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


class _NoiseInit(nn.Module):
    """``h_0 ~ N(0, std^2)`` per element — Parcae's ``like-init`` loop entry.

    The loop state starts as small noise (Parcae 140m: std 0.023 against a converged state
    RMS of 0.56 per element) and the coda can only read what the recurrence builds from the
    injected ``e``: the identity-loop solution ``h_T = h_0`` is not available. No
    parameters, so building it draws no RNG; the forward draws ``randn`` on every call,
    training and eval alike (as Parcae does).
    """

    def __init__(self, std: float):
        super().__init__()
        self.std = float(std)

    def forward(self, e: Tensor) -> Tensor:
        return torch.randn_like(e) * self.std


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


class _MTPHead(nn.Module):
    """One parallel lookahead head: RMSNorm -> Linear(d, d) at identity init.

    Reads the coda readout (post lm_mixer/final_norm) and feeds the tied LM head, so at
    init every head predicts exactly what the next-token head predicts (and its loss
    starts at the next-token CE against a shifted label). Deterministic init: no RNG.
    """

    def __init__(self, d: int):
        super().__init__()
        self.norm = RMSNorm(d)
        self.proj = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(d))

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.norm(x))


class _LoopMTPGate(nn.Module):
    """LoopMTP's content-conditional aggregator, Eq 9-11 (arXiv 2608.03624 Sec 3.1).

        g^(t)      = softplus(W_g x^(t) + beta_t * 1_d)
        gtilde^(t) = g^(t) / (sum_s g^(s) + eps)      <- elementwise, over ITERATIONS
        z          = sum_{t=1..T} gtilde^(t) (*) x^(t)

    ``W_g`` is ONE ``[d, d]`` linear shared by every iteration (the paper's "single linear
    gate shared across all iterations"); ``beta`` is one scalar per iteration. Both start
    at zero, which draws no RNG and makes ``z`` the uniform mean of the T iterates at init.

    Carrier shape: MORPH's loop carrier is the Hyper-Connection ``[B, S, n, C]`` tensor, so
    the gate acts on the last axis and broadcasts across the ``n`` streams — the aggregate
    is a valid carrier the coda consumes unchanged. On a plain ``[B, S, C]`` carrier the
    same code is the paper's form verbatim.

    Precision and memory: the gate runs at the CARRIER dtype (bf16 under autocast), the
    dtype the rest of the loop already uses. The normalising sum is a ``torch.sum``
    reduction, which accumulates in fp32 on both CPU and CUDA, so the division is accurate
    even when the stored gates are bf16. Nothing is stacked at ``[T, B, S, n, C]``: the
    running sum and ``z`` are accumulated in a Python loop, and the only per-iteration
    tensor kept for the caller is the reduced ``[T, B*S]`` gate mass the ponder term and
    the Fig-4 read-out want.

    Depths beyond ``T``: a forced-depth sweep may run MORE iterations than the model was
    trained with (``core_depth_sweep.py --depths ...,9,12,16``). ``beta`` holds only T
    entries, so iteration ``t > T`` reuses ``beta_T``. The gate stays content-conditional
    through ``W_g``; only the per-iteration bias is held. This is a READ-OUT convention for
    the sweep — training always runs exactly T iterations and never reaches it.
    """

    def __init__(self, d: int, n_iters: int, eps: float = 1.0e-6):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.proj.weight)
        # Never ternarised: a {-1,0,+1} gate matrix cannot express a small content
        # perturbation inside a softplus — the same rationale as the HC coefficient
        # projection and the DiagonalInjection control matrices.
        self.proj._ternary_exclude = True
        self.beta = nn.Parameter(torch.zeros(int(n_iters)))
        self.eps = float(eps)

    def raw_gates(self, states: list[Tensor]) -> list[Tensor]:
        """``g^(t)`` of Eq 9, one tensor per iteration, un-normalised."""
        n_beta = int(self.beta.shape[0])
        out = []
        for t, x in enumerate(states):
            y = self.proj(x)
            out.append(F.softplus(y + self.beta[min(t, n_beta - 1)].to(y.dtype)))
        return out

    def forward(self, states: list[Tensor]) -> tuple[Tensor, Tensor]:
        """``states`` = ``[x^(1), ..., x^(k)]`` -> ``(z, gate_mass)``.

        ``gate_mass`` is ``[k, B*S]`` fp32: the normalised gate ``gtilde^(t)`` averaged over
        the channels (and the HC streams) at each position. It sums to 1 over axis 0 by
        construction, and it is the ONLY per-iteration gate quantity kept — the full
        ``[k, B, S, n, C]`` tensor is never materialised.
        """
        gs = self.raw_gates(states)
        # PREFIX RULE (a Poisson draw, 2026-09-14): `states[t]` may hold only the rows
        # still active at iteration t, and those rows are always the FIRST n_t of the
        # depth-sorted batch (n_0 >= n_1 >= ...). A row therefore gates over exactly its
        # own realised passes: the running sum and z touch rows [:n_t] only, and the gate
        # mass of a finished row at a later iteration is exactly 0. With every n_t == B
        # (a fixed depth, or eval) each `cat` below is the identity and the arithmetic is
        # the pre-2026-09-14 form op for op.
        B = int(states[0].shape[0])
        tot = gs[0]
        for g in gs[1:]:
            n = int(g.shape[0])
            tot = tot + g if n == B else torch.cat([tot[:n] + g, tot[n:]], dim=0)
        tot = tot + self.eps
        z = None
        mass: list[Tensor] = []
        for g, x in zip(gs, states):
            n = int(g.shape[0])
            gt = g / tot[:n]
            zt = gt * x
            if z is None:
                z = zt
            elif n == B:
                z = z + zt
            else:
                z = torch.cat([z[:n] + zt, z[n:]], dim=0)
            # [n, S, ...] -> [B*S]: mean over every axis but batch and position; the rows
            # this iteration did not run are 0.
            m = gt.float().flatten(0, 1).flatten(1).mean(dim=1)
            if n != B:
                m = torch.cat([m, m.new_zeros((B - n) * int(x.shape[1]))], dim=0)
            mass.append(m)
        return z, torch.stack(mass, dim=0)


# tul.code_grade: the base of the per-step generators the graded sampler draws from. Its
# own streams, never the global ones — a graded step must leave every later draw of the
# training run exactly where it found it (`_tul_code_grade` saves and restores both).
_GRADE_SEED: int = 0x6EADE

def span_ce_index(labels: Tensor, layout: SlotLayout):
    """The scatter index that turns per-TOKEN CE into per-SPAN CE on a packed TUL row.

    Returns ``(gid, keep_tok, lab, n_groups)``:

    * ``gid`` ``[B, L]`` int64 — the flat bin ``b * (S + 1) + bag_id`` of every position,
      forced to bin 0 wherever the position is not scored. Unscored positions contribute
      an exactly-zero CE, so the dump into bin 0 is harmless and bin 0 is never read
      (slot ``s`` reads bag ``s + 1``).
    * ``keep_tok`` ``[B, L]`` bool — a real TOKEN with a real label. Slot positions are
      excluded: their emit label carries no loss on this family (`emit_weight: 0.0`).
    * ``lab`` — ``labels`` with ``-100`` clamped to 0, safe to hand to ``cross_entropy``
      because ``keep_tok`` zeroes those entries afterwards.
    * ``n_groups`` = ``S + 1``, the bag axis including the pad/dump bin.

    ONE home, because two readers need the identical arithmetic and a drift between them
    would be invisible: ``_tul_code_span_scorer`` (Explorative Modeling's
    ``code_xm_select="coda"``) and ``_tul_fan_oracle`` (LXTUL's oracle-over-stream).
    """
    B = labels.shape[0]
    S = int(layout.slot_valid.shape[1])
    G = S + 1
    keep_tok = (labels >= 0) & (~layout.slot_mask)
    lab = labels.clamp_min(0)
    gid = torch.arange(B, device=labels.device).view(B, 1) * G + layout.bag_id.clamp(0, S)
    gid = torch.where(keep_tok, gid, torch.zeros_like(gid))
    return gid, keep_tok, lab, G


def accumulate_span_ce(xh: Tensor, w_head: Tensor, gid: Tensor, keep_tok: Tensor,
                       lab: Tensor, n_groups: int) -> Tensor:
    """``[B, n_groups]`` SUMMED token CE per bag, from a coda state and the tied head.

    The logits are built ONE ROW at a time and never materialised for the whole batch:
    ``[L, V]`` at ``V`` ~ 49k is 200 MB in fp32 and ``[B, L, V]`` is not affordable
    inside an eval forward that already holds the loop's trajectory.
    """
    B = lab.shape[0]
    out = torch.zeros(B * n_groups, device=lab.device, dtype=torch.float32)
    for b in range(B):
        logits = (xh[b].to(w_head.dtype) @ w_head.t()).float()          # [L, V]
        ce = F.cross_entropy(logits, lab[b], reduction="none") * keep_tok[b]
        out.index_add_(0, gid[b], ce)
    return out.view(B, n_groups)


def span_token_counts(gid: Tensor, keep_tok: Tensor, n_groups: int) -> Tensor:
    """``[B, n_groups]`` count of SCORED tokens per bag — the weights a CE average needs."""
    B = gid.shape[0]
    out = torch.zeros(B * n_groups, device=gid.device, dtype=torch.float32)
    out.index_add_(0, gid.reshape(-1), keep_tok.reshape(-1).float())
    return out.view(B, n_groups)


def slot_cell_relation(n_slots: int, m_cells: int, device, reach: int = 0
                       ) -> tuple[Tensor, Tensor]:
    """The Thought Register's CELL relation, in one place (``tul.slot_cells``).

    The compact axis is ``n_slots * m_cells`` cells, slot-major (index ``s*M + i``).
    Returns ``(blk, same)``, both ``[1, 1, S*M, S*M]`` bool:

    * ``blk`` — cell ``i`` of slot ``k`` reads every cell of its OWN slot (including the
      ones AFTER it) and every cell of slots ``< k``, narrowed to ``reach`` slots back
      when ``reach > 0``. Plain causal on the FLATTENED axis is a different relation and a
      wrong one: it would leave cell 0 permanently blind to its siblings, which is the
      opposite of a register.
    * ``same`` — slot-local: a cell reads its own slot's cells and nothing else. This is
      what the core's layers ``1..n-1`` run under a reach budget, where the whole per-pass
      cross-cell allowance is spent in layer 0.

    ONE builder, two callers: ``_tul_core``'s per-pass core stage and ``_tul_cond_apply``'s
    think-once stack. A second copy of this mask is how the stack and the loop would drift
    into two different relations without a test noticing.

    At ``m_cells == 1`` ``blk`` is exactly plain causal over the ``n_slots`` positions —
    which is why the stack at M = 1 can keep running the block's own causal attention with
    no mask at all and stay bit-identical to the placement before the register existed.

    **How it is DELIVERED, and why that is not ``tg_allow`` (fixed 2026-09-13, before any
    GPU step of any register arm).** ``blk`` is a SUPERSET of flattened causal:
    ``slot(p) >= slot(q)`` is implied by ``p >= q``. ``tg_allow`` / ``tg_comp_allow`` are
    ANDed into a relation that is ALREADY causal and can only NARROW, so handing ``blk``
    to them executed as plain flattened causal at ``reach == 0`` — a cell never read a
    LATER cell of its own slot, in the loop or in the think-once stack, and the mask was a
    no-op there. Both callers therefore pass this mask as ``tg_relation``, the ONE kwarg
    that REPLACES a branch's causal term instead of narrowing it
    (``attention._tg_relation_guard``, ``attention._tg_slot_attention``,
    ``attention._window_fallback``). Replacing is safe across slots precisely because
    ``blk`` is block-causal there; within a slot it widens to all M cells, which is the
    documented relation. Pinned two-sided and NUMERICALLY — the register's loss differs
    from plain flattened causal and equals an independently written all-true-within-slot
    mask — in ``tests/test_tul_slot_register.py`` and ``tests/test_tul_cond4_strict.py``.

    **What is still CAUSAL on the cell axis, by decision.** The CCA causal conv, its
    ``W_v_prev`` value shift and (on a core that carries one) the GLA retention branch are
    position-wise / recurrent operators, not attention: they read backwards along the
    flattened axis and have no mask to widen. They are left alone. Every position they
    reach is a cell of the same or an earlier slot, which ``blk`` already allows, so they
    never leak; they simply give cell 0 no sibling context of its own. Widening them would
    mean an acausal conv, a different mechanism from this relation.
    """
    sm = n_slots * m_cells
    sl = torch.arange(sm, device=device) // m_cells            # slot id per cell
    si, sj = sl.unsqueeze(1), sl.unsqueeze(0)
    blk = si >= sj
    if reach > 0:
        blk = blk & (sj >= si - reach)
    return blk.view(1, 1, sm, sm), (si == sj).view(1, 1, sm, sm)


class MORPHTransformer(nn.Module):

    # Operating-point capture for the core-map Jacobian probe
    # (morph/training/core_jacobian.py). `None` — the default — makes every capture site
    # a Python-level no-op, so the forward is bit-identical when the probe is off. Set to
    # a list to collect one dict per core-loop iteration.
    _jac_capture: list | None = None

    # Progressive-loss counters written by `_tul_core` when `tul.progressive_p > 0`
    # (0-dim GPU tensors; the trainer turns them into `loop/prog_*`). None on every other
    # model, which is what makes the trainer's read a no-op there.
    _loop_prog: dict | None = None
    _loop_prog_k = None          # [B, S] int — the per-slot no-grad prefix length drawn

    # Per-pass MUX terms written by `_forward_tul` when `tul.mux_every_pass` is on and the
    # model is training (`mux_pass_terms` a float, the rest detached 0-dim tensors; the
    # trainer turns them into `loop/mux_pass_*`). None on every other model.
    _loop_mux: dict | None = None

    # Per-pass gradient-conditioning readouts written by `_tul_core` when `tul.grad_pass`
    # is on and the model is training: `own_pass_t{t}` (the local target's value at pass t)
    # and `gp_rel_t{t}` (the injected term's norm over the state's). Detached 0-dim
    # tensors; the trainer turns them into `loop/*`. None on every other model.
    _loop_gradpass: dict | None = None

    # Per-pass loop-carry readouts written by `_tul_core` when `tul.loop_carry` is on:
    # `rms_t{t}` (the carry's per-cell RMS after pass t, mean over valid slots),
    # `gate_mean_t{t}` (the `gate` mode's sigmoid, mean) and `inject_ratio_t{t}` (the
    # injected term's RMS over the carrier's — 1.0 by construction, the check that the
    # RMS match is live). Detached 0-dim tensors; `_forward_tul` folds them into `groups`
    # as `carry_*` and the trainer logs them under `carry/`. None on every other model.
    _loop_carry_stats: dict | None = None

    # `tul.loop_carry`'s test hook: attach a list and `_tul_core` appends
    # ``{"read": r_k(t), "carry_prev": c_k(t-1), "carry": c_k(t), "gate": the sigmoid or
    # None}`` (all detached) once per pass. None by default — a Python-level branch, so
    # the shipped graph never sees it.
    _carry_capture: list | None = None

    def __init__(self, cfg: MORPHConfig):
        super().__init__()
        self.cfg = cfg
        if bool(cfg.depth_fixed) and int(cfg.mean_depth) != int(cfg.max_depth):
            raise ValueError(
                f"model.depth_fixed needs mean_depth == max_depth (got {cfg.mean_depth} / "
                f"{cfg.max_depth}): eval runs mean_depth for every row and train runs "
                f"max_depth for every row, and a fixed depth means the same number.")
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
        # THE STRICT GEOMETRY (tul.tg_geometry). A Python-level constant read at trace
        # time, exactly like `_tg_restrict`: False — every model before this key — builds
        # and threads the same masks it always did, bit-identically.
        self._tg_strict = (cfg.tul is not None and cfg.tul.tg_geometry == "strict")
        if cfg.tg_scoped_kernels and not self._tg_restrict:
            raise ValueError(
                "model.tg_scoped_kernels=true requires tul.tg_restrict=true: outside TG "
                "restriction use model.use_kernels for the full fused path instead.")
        if self._tg_restrict and cfg.use_kernels:
            raise ValueError(
                "model.tul.tg_restrict=true requires model.use_kernels=false "
                "(docs/tul-tg-spec.md §2/§6): the TG arms run eager only — the fused "
                "window/CSA/HCA kernels do not know about the restriction, and a "
                "silent unmasked kernel path is forbidden.")

        # ── The cross-span budget mask (model.span_mask) ─────────────────────
        # Construction-time only, like the TG block above: it decides which attention
        # variant is BUILT and what `_forward_single` threads into every block. Every
        # refusal below is a path whose cross-span route the mask would not reach —
        # running any of them would report a budget measured through a leak.
        if cfg.span_mask not in ("off", "row", "span"):
            raise ValueError(
                f"model.span_mask must be 'off', 'row' or 'span', got {cfg.span_mask!r}")
        self._span_mask = cfg.span_mask != "off"
        self._span_rule = cfg.span_rule if cfg.span_mask == "span" else None
        if self._span_mask:
            if cfg.span_mask == "span" and cfg.span_rule is None:
                raise ValueError(
                    "model.span_mask='span' needs a resolved BoundaryRule in "
                    "MORPHConfig.span_rule (morph/training/train.py builds it from the "
                    "tokenizer). Without it there is no cut and the arm would silently "
                    "be the 'row' control.")
            if cfg.span_mask == "row" and cfg.span_rule is not None:
                raise ValueError(
                    "model.span_mask='row' is the ONE-SPAN-PER-ROW control and must not "
                    "carry a BoundaryRule; the rule would not be applied and the config "
                    "would promise a cut the forward does not make.")
            if self._tg_restrict:
                raise ValueError(
                    "model.span_mask with tul.tg_restrict: both drive the same attention "
                    "kwargs from different relations (spans vs slots). Pick one.")
            if cfg.use_kernels:
                raise ValueError(
                    "model.span_mask requires model.use_kernels=false: the fused "
                    "window/CSA/HCA kernels do not know about the span relation, and a "
                    "silent unmasked kernel path is forbidden.")
            if cfg.core_impl != "parcae":
                raise NotImplementedError(
                    "model.span_mask requires model.core_impl='parcae'. A MORPH core's "
                    "compressed branch attends POOLED BLOCKS of positions; a 128- or "
                    "16-position block straddles several spans (mean span ~20 tokens on "
                    "web text), so there is no block-level same-span restriction that "
                    "leaves the branch alive. The Parcae core's dense softmax takes the "
                    "relation directly.")
            if cfg.retention:
                raise NotImplementedError(
                    "model.span_mask with model.retention=true: the GLA branch carries a "
                    "recurrent state ACROSS positions and the span reset mask is not "
                    "derived here. Set model.retention=false (the panel recipe does).")
            if cfg.mtp_heads != 1:
                raise NotImplementedError(
                    "model.span_mask with mtp_heads > 1: an MTP head predicts token t+j, "
                    "which crosses a span boundary for j large enough, and its label "
                    "stream is not cut here.")
            if cfg.fm is not None or cfg.scse_enabled:
                raise NotImplementedError(
                    "model.span_mask has no defined interaction with the FM planner or "
                    "SCSE; both redefine the loop carrier or the plan path.")

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
            # span_mask builds the SAME attention variant as tg_restrict — a
            # compressed branch that attends POSITIONS, no pooled compressor, no
            # indexer — and then drives it from the span ids (see attention.py).
            tg_restrict=self._tg_restrict or self._span_mask,
            tg_span_gate=(bool(cfg.tul.tg_span_gate) if cfg.tul is not None
                          else False),
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
        if cfg.injection_channels == "ctx":
            self.injection = DiagonalInjection(self._ctx_start, self._ctx_end,
                                               use_B=cfg.injection_B)
        elif cfg.injection_channels == "all":
            _dec = torch.full((d,), float(cfg.injection_all_decay))
            if cfg.injection_all_dt is None:
                _dt = 1.0 - _dec
                _dec[self._ctx_start:self._ctx_end] = 0.447
                _dt[self._ctx_start:self._ctx_end] = 1.0
                _desc = (f"ctx legacy 0.447/1.0; rest decay {cfg.injection_all_decay}, "
                         f"dt {1 - cfg.injection_all_decay:.3f}")
            else:
                _dt = torch.full((d,), float(cfg.injection_all_dt))
                _desc = f"every dim decay {cfg.injection_all_decay}, dt {cfg.injection_all_dt}"
            self.injection = DiagonalInjection(0, d, init_decay_vec=_dec, init_dt_vec=_dt,
                                               use_B=cfg.injection_B)
            print(f"  CARRY: diagonal injection on ALL {d} dims ({_desc}"
                  f"{'; learned B (identity init)' if cfg.injection_B else ''})")
        else:
            raise ValueError(f"model.injection_channels must be 'ctx' or 'all', "
                             f"got {cfg.injection_channels!r}")

        # ── Core (shared across loop iterations — MortarLinear for CMS pruning)
        # `core_impl` is read ONCE, here. `self._core_is_parcae` is a Python bool that is
        # never rebound, so every branch on it in the forward resolves at trace time and
        # a "morph" model's graph — and its RNG stream, and every weight — is what it was
        # before this knob existed (tests/test_parcae_core.py pins the gradient hash).
        if cfg.core_impl not in ("morph", "parcae"):
            raise ValueError(f"model.core_impl must be 'morph' or 'parcae', "
                             f"got {cfg.core_impl!r}")
        self._core_is_parcae = cfg.core_impl == "parcae"
        if not self._core_is_parcae:
            self.core = nn.ModuleList([
                _make_block(cfg.n_prelude + i, core_attn_kw)
                for i in range(cfg.n_core)
            ])
        else:
            from .parcae_core import ParcaeCoreBlock, parcae_core_d_head
            # Refusals, not silent drops. Each of these is a mechanism the Parcae block
            # cannot carry; running it anyway would report a "one-factor swap" that had
            # quietly lost a second factor.
            if cfg.retention and "core" in tuple(cfg.retention_sections):
                raise NotImplementedError(
                    "model.core_impl='parcae' with retention on the core section: the "
                    "Parcae block carries no GLA branch. Drop 'core' from "
                    "model.retention_sections or use core_impl='morph'.")
            if cfg.scse_enabled or cfg.core_init_scale > 0.0:
                raise NotImplementedError(
                    "model.core_impl='parcae' with SCSE / core_init_scale is not "
                    "defined: both redefine the loop carrier, which this swap does not "
                    "cover.")
            if cfg.core_hca_compress_ratio is not None:
                raise ValueError(
                    "model.core_hca_compress_ratio has no meaning under "
                    "core_impl='parcae' — the Parcae core has no pooled compressor and "
                    "no HCA branch to re-block. Remove the key.")
            if self._tg_restrict:
                raise NotImplementedError(
                    "model.core_impl='parcae' with tul.tg_restrict: the TG restriction "
                    "is a build flag on MORPHAttention's compressed branch and the "
                    "Parcae core has no such branch, so the core's cross-span mask "
                    "would silently vanish while prelude/coda kept theirs.")
            _pd_head = parcae_core_d_head(d, cfg.n_heads, cfg.compression,
                                          cfg.parcae_core_d_head)
            self.core = nn.ModuleList([
                ParcaeCoreBlock(
                    d_model=d, d_ff=d_ff, n_heads=cfg.n_heads, d_head=_pd_head,
                    max_seq_len=cfg.max_seq_len, context_len=cfg.context_len,
                    dropout=cfg.dropout, residual_blocks=max(1, cfg.n_core),
                )
                for _ in range(cfg.n_core)
            ])
            _n_pc = sum(p.numel() for b in self.core for p in b.parameters())
            print(f"  CORE = PARCAE: {cfg.n_core} plain pre-norm blocks "
                  f"(dense causal softmax, {cfg.n_heads} heads x d_head {_pd_head}, "
                  f"SwiGLU d_ff {d_ff}, single-stream residual) -> {_n_pc:,} params "
                  f"({_n_pc / 1e6:.2f}M), bf16 (never ternarised), dense nn.Linear "
                  f"(never pruned / carved / routed / packed). Carrier: stream mean in, "
                  f"broadcast to {cfg.hc_streams} streams out, once per pass.", flush=True)

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
        # The trainer's step counter, for `tul.code_grade` (spec §17.2): the graded term
        # runs on every `code_grade_every`-th step and seeds its own generators from this.
        # A buffer for the reason above, and read ONLY inside `_tul_code_grade`, which is
        # `torch.compiler.disable`d — so no `.item()` ever lands in a captured graph.
        self.register_buffer("code_grade_step", torch.zeros((), dtype=torch.long),
                             persistent=False)
        self._in_code_grade = False      # reentrancy guard: the grader runs the forward
        # tul.code_target_ref (spec §17.1): the FROZEN reference copy of the VAE stage.
        # Deliberately NOT a registered submodule. Every walk in this tree enumerates
        # modules or parameters — the ternary QAT pass, the embedding QAT, the CMS
        # prune / carve / route walk, the optimizer, the gradient probes — and a
        # registered 270M-parameter twin would silently double all of them. It is built
        # and loaded by `tul_code_ref_snapshot` and persisted under its own checkpoint
        # key (`save_checkpoint`), never inside `model`.
        self.__dict__["_code_ref"] = None


        # ── Retention branch (#230) ────────────────────────────────────
        # Attach AFTER all base modules → GLA's RNG draws are a tail, so the base model is
        # byte-identical to the baseline whether retention is on or off. With the branch-gate
        # near 0 at init, retention-on ≈ baseline at step 0, and the ablation isolates exactly
        # the retention branch (no confound from a different random init of the rest of the net).
        self._retention_layers = tuple(cfg.retention_layers)
        ret_sections = tuple(cfg.retention_sections)
        self._core_has_retention = cfg.retention and "core" in ret_sections and any(
            i in self._retention_layers for i in range(cfg.n_core))
        if cfg.retention:
            from .gla import GatedLinearAttention
            rheads = cfg.retention_heads or cfg.n_heads
            if cfg.retention_write_shift:
                print("  GLA WRITE-SHIFT ON: next-latent alignment (k_{t-1}, v_t) — FWA eq 2.3")
            for sname, section in (("prelude", self.prelude), ("core", self.core),
                                   ("coda", self.coda)):
                if sname not in ret_sections:
                    continue
                for si, blk in enumerate(section):
                    if si in self._retention_layers:
                        blk.attach_retention(
                            GatedLinearAttention(
                                d, rheads,
                                mode="kernel" if cfg.use_kernels else "chunked",
                                chunk=cfg.retention_chunk,
                                gate_logit_bias=cfg.retention_gate_bias,
                                write_shift=cfg.retention_write_shift),
                            RMSNorm(d), gate_init=cfg.retention_gate_init)

        # ── Multi-token prediction heads (cfg.mtp_heads > 1; arc E8) ───────
        # Identity init draws no RNG, so a model with heads shares every base weight with
        # the same-seed model without them (tests/test_mtp_heads.py).
        if int(cfg.mtp_heads) < 1:
            raise ValueError(f"model.mtp_heads must be >= 1, got {cfg.mtp_heads}")
        self.mtp = None
        if int(cfg.mtp_heads) > 1:
            self.mtp = nn.ModuleList([_MTPHead(d) for _ in range(int(cfg.mtp_heads) - 1)])
            print(f"  MTP: {int(cfg.mtp_heads) - 1} parallel lookahead heads on the coda "
                  f"readout (targets t+2..t+{int(cfg.mtp_heads)}), weight {cfg.mtp_weight}")

        # ── LoopMTP (arXiv 2608.03624): the gated aggregator and the soft MTP target ──
        # Both inits are DETERMINISTIC (zeros / identity — zero RNG draws), so a LoopMTP
        # model's base weights are byte-identical to the same-seed ladder rung and the arm
        # differs by the mechanism alone. Off (core_readout "last" AND loopmtp_weight 0)
        # builds nothing and `_loopmtp_states` stays False, so `_core_region` keeps not one
        # extra op.
        if cfg.core_readout not in ("last", "gated"):
            raise ValueError(f"model.core_readout must be 'last' or 'gated', got "
                             f"{cfg.core_readout!r}")
        if cfg.loopmtp_proj not in ("linear", "none"):
            raise ValueError(f"model.loopmtp_proj must be 'linear' or 'none', got "
                             f"{cfg.loopmtp_proj!r}")
        if float(cfg.loopmtp_weight) < 0.0:
            raise ValueError(f"model.loopmtp_weight must be >= 0, got {cfg.loopmtp_weight}")
        if float(cfg.loopmtp_ponder_weight) < 0.0:
            raise ValueError(f"model.loopmtp_ponder_weight must be >= 0, got "
                             f"{cfg.loopmtp_ponder_weight}")
        _lm_gated = str(cfg.core_readout) == "gated"
        _lm_align = float(cfg.loopmtp_weight) > 0.0
        if float(cfg.loopmtp_ponder_weight) > 0.0 and not _lm_gated:
            raise ValueError(
                "model.loopmtp_ponder_weight > 0 needs model.core_readout='gated': the "
                "ponder regulariser (arXiv 2608.03624, KL of the per-iteration gate "
                "distribution against uniform) is a term ON the aggregator's gate, and "
                "there is no gate to regularise under the 'last' read-out.")
        self.loopmtp_gate = None
        self.loopmtp_proj = None
        self._loopmtp_states = bool(_lm_gated or _lm_align)
        if self._loopmtp_states:
            # T is the LONGEST loop a row can run: `beta` needs one bias per iteration
            # that can exist. Under `depth_fixed` that is the one depth every row runs
            # (mean == max); under the Poisson draw (allowed since 2026-09-14) a row at
            # depth d_i gates over its own d_i states and is aligned on its own d_i
            # horizons — the prefix rule in `_core_region`, `_LoopMTPGate.forward`,
            # `_loopmtp_align_maps` and `_loopmtp_ponder`. Eval runs mean_depth for every
            # row, so `beta[mean_depth:]` is trained only by the rows that drew past it.
            _T = int(cfg.max_depth)
            if bool(cfg.depth_fixed) and int(cfg.mean_depth) != int(cfg.max_depth):
                raise ValueError("model.depth_fixed needs mean_depth == max_depth")
            if int(cfg.bptt_depth) < _T:
                raise ValueError(
                    f"LoopMTP needs full BPTT: bptt_depth ({cfg.bptt_depth}) must be >= "
                    f"max_depth {_T}. The paper backpropagates through all T "
                    f"iterations; a truncated window would leave the early iterations' "
                    f"alignment terms with no path to the core weights.")
            if int(cfg.n_core) == 0:
                raise ValueError("LoopMTP needs a core loop (model.n_core > 0).")
            if cfg.tul is not None:
                raise ValueError(
                    "LoopMTP is defined on the PLAIN looped model only. On a TUL model the "
                    "core loop is either the slot loop (`_tul_core`, a different function) "
                    "or the paid loop over a packed row whose slot positions carry no "
                    "`u_{i+t}` label — neither has a defined Eq-13 target, so this raises "
                    "instead of silently scoring pad positions.")
            if bool(cfg.scse_enabled):
                raise ValueError(
                    "LoopMTP is not defined under SCSE: the loop carrier there is the "
                    "DEVIATION from a fixed anchor, so the per-iteration states the "
                    "aggregator would gate and the alignment loss would score are not the "
                    "states the paper's x^(t) names.")
            # `nn.Linear.reset_parameters` DRAWS from the global RNG before the
            # zero/identity overwrite, so building these here would shift every weight
            # constructed after this point. Snapshot and restore the stream around the
            # build: a LoopMTP model then shares every base weight with the same-seed
            # ladder rung byte for byte, and the arm differs by the mechanism alone
            # (tests/test_loopmtp.py::test_knobs_off_and_on_share_every_base_weight).
            _lm_rs = torch.get_rng_state()
            if _lm_gated:
                self.loopmtp_gate = _LoopMTPGate(d, _T, eps=float(cfg.loopmtp_gate_eps))
            if _lm_align and str(cfg.loopmtp_proj) == "linear":
                # RMSNorm -> Linear(d, d) at identity init, exactly `_MTPHead`'s body; at
                # init it is a pure rescale, and cosine is scale-invariant, so the term
                # starts at the same value the paper's projection-free form gives.
                self.loopmtp_proj = _MTPHead(d)
                self.loopmtp_proj.proj._ternary_exclude = True
            torch.set_rng_state(_lm_rs)
            print(f"  LoopMTP: T={_T} ({'fixed' if cfg.depth_fixed else 'Poisson mean ' + str(cfg.mean_depth)}) readout={cfg.core_readout} "
                  f"lambda_align={cfg.loopmtp_weight} free_first={cfg.loopmtp_free_first} "
                  f"proj={cfg.loopmtp_proj} lambda_ponder={cfg.loopmtp_ponder_weight}")

        if cfg.core_gain_direction not in ("power", "random"):
            raise ValueError(f"model.core_gain_direction must be 'power' or 'random', got "
                             f"{cfg.core_gain_direction!r}")
        self._core_aux: dict | None = None      # arc E10 loss terms, stashed by _core_region
        self._core_gain_dir: Tensor | None = None   # the power-iterated probe direction

        # ── TUL slot parameters (docs/tul-spec.md §3.1-§3.4) ───────────────
        # Constructed LAST, after retention, for the same reason: all three inits are
        # DETERMINISTIC (zeros / identity — zero RNG draws), so a TUL model's base weights
        # are byte-identical to a baseline built with the same seed, and the arms differ by
        # the mechanism alone. E_slot is re-initialised from the live embedding table at the
        # activation step (Block Transformer §3.7) — see TULSlots.init_at_activation.
        # `tul.mux_readout='full'` reads the Hyper-Connection stream axis. On a plain
        # residual there is no such axis and the two readouts would be the same object, so
        # the config is refused here rather than silently meaning nothing for a whole run.
        # HONEST NOTE: `_is_hc` is set unconditionally True above — HC-Cayley is the SOLE
        # residual on this tree — so this branch cannot fire today and no test exercises it.
        # It is the contract for a second residual mode, and the guard that CAN fire is the
        # dim check inside `_readout_per_stream`, which a test does exercise.
        if cfg.tul is not None and cfg.tul.mux_readout == "full" and not self._is_hc:
            raise ValueError(
                "tul.mux_readout='full' needs a Hyper-Connection residual carrier "
                f"(model.residual gives {self._residual_mode!r}, which has no stream axis): "
                "the per-stream readout and the mean readout would be the same tensor.")
        if cfg.tul is not None and cfg.tul.mux_stage_own_iters > 0:
            _kmax = int(cfg.tul.slot_max_depth or cfg.max_depth)
            if cfg.tul.mux_stage_own_iters > _kmax:
                raise ValueError(
                    f"tul.mux_stage_own_iters={cfg.tul.mux_stage_own_iters} exceeds the loop's "
                    f"max depth {_kmax} (tul.slot_max_depth / model.max_depth)")
        # Arc E10 loop terms on a TUL model (2026-09-07). The fixed-point term is defined on
        # every core loop: `_core_region` (the plain model and the paid loop) and `_tul_core`
        # (the slot loop, the same finishing-slice term per slot). The gain penalty was
        # refuted (E10c) and stays plain-loop only; it raises HERE, not at step 1.
        if cfg.tul is not None:
            if cfg.core_gain_lambda > 0.0:
                raise RuntimeError("model.core_gain_lambda > 0 acts on the plain loop only "
                                   "(arc E10c, refuted); the slot path has its own hinge "
                                   "(model.slot_gain_lambda)")
            _fixed = int(cfg.tul.slot_depth_fixed)
            _kmax_slot = int(cfg.tul.slot_max_depth or cfg.max_depth)
            if _fixed < 0 or _fixed > _kmax_slot:
                raise ValueError(
                    f"tul.slot_depth_fixed={_fixed} must lie in [0, slot_max_depth={_kmax_slot}] "
                    f"(0 = the Poisson draw)")
        if (cfg.tul is not None and cfg.fm is not None
                and cfg.tul.tg_geometry == "strict"):
            raise NotImplementedError(
                "tul.tg_geometry='strict' with an FM planner (cfg.fm): the planner replaces "
                "the core loop, so 'the slot loop is the only cross-span channel' names a "
                "loop that is not there. Pick one.")
        if cfg.tul is not None and cfg.fm is not None and cfg.tul.tokens_through_core:
            raise ValueError(
                "tul.tokens_through_core=true with an FM planner (cfg.fm): the planner replaces "
                "the core loop and writes its plans through W_prefix; the paid loop runs the "
                "core over every position and has no projection. Pick one (tul_fm1.yaml sets "
                "tokens_through_core: false).")
        # W_prefix only where something writes through it (TULSlots.__init__). The paid
        # loop and `tul.loop_reads_tokens` both leave a cell's looped state AT its own
        # position, so neither has a projection to write through; the FM planner writes
        # its plans through one, so it always gets it.
        _wants_prefix = cfg.tul is not None and (
            (cfg.fm is not None)
            or not (cfg.tul.tokens_through_core or cfg.tul.loop_reads_tokens))
        self.tul: TULSlots | None = (
            TULSlots(d, cfg.tul, with_prefix=_wants_prefix)
            if cfg.tul is not None else None)
        # The gate is built AFTER TULSlots for the same reason and with the same
        # discipline: every one of its inits is a deterministic zero/one, so building it
        # advances the RNG stream by nothing and arm TUL-gate's base weights are
        # byte-identical to arm A1's (docs/tul-gate-spec.md §9 invariant 1).
        _gc = cfg.tul.gate if cfg.tul is not None else None
        self.tul_gate: TULGate | None = TULGate(d, _gc) if _gc is not None else None
        # The reread (TULConfig.reread): the looping slot cross-attends the frozen prelude
        # token states each pass. Private-generator init, W_o zero: the model's RNG stream
        # and every other weight are untouched, and step 0 is the no-reread forward.
        self.tul_reread: TULReread | None = None
        if cfg.tul is not None and cfg.tul.reread:
            if cfg.n_core == 0:
                raise ValueError("tul.reread=true needs a core loop (n_core > 0): there is "
                                 "no pass in which to re-read.")
            self.tul_reread = TULReread(d, cfg.tul.reread_heads)
        # Gradient-conditioned passes (TULConfig.grad_pass; Marino et al. 2018, IODINE).
        # ONE zero matrix, no RNG draw, so an arm with the knob on holds byte-identical
        # weights to the ruler everywhere else and its step 0 is the ruler's forward.
        self.tul_grad_pass: TULGradPass | None = None
        if cfg.tul is not None and cfg.tul.grad_pass:
            if cfg.n_core == 0:
                raise ValueError(
                    "tul.grad_pass needs a core loop (model.n_core > 0): there is no pass "
                    "to condition on a gradient.")
            self.tul_grad_pass = TULGradPass(d, cfg.tul.grad_pass_scale,
                                             cfg.tul.grad_pass_norm)

        # ── The energy the gradient feature is taken OF (TULConfig.grad_pass_energy;
        #    morph/model/tul_egrad.py) ───────────────────────────────────────────────
        # `own_mux` (the default) builds NOTHING: the energy is `_tul_mux_loss(target="own")`
        # and the tree is the one from before this existed. The two modules below are
        # TRAINING-ONLY scorers, RNG-neutral (private generators) and `_ternary_exclude` on
        # every leaf, so the QAT scope, the prune, the carve, the router and the packer all
        # walk past them.
        self.tul_egrad: nn.Module | None = None
        _ge = "own_mux" if cfg.tul is None else str(cfg.tul.grad_pass_energy)
        if _ge not in ("own_mux", "recon", "disc", "critic"):
            raise ValueError(
                f"tul.grad_pass_energy must be 'own_mux', 'recon', 'disc' or 'critic', "
                f"got {_ge!r}")
        if cfg.tul is not None and _ge != "own_mux":
            if not cfg.tul.grad_pass:
                raise ValueError(
                    f"tul.grad_pass_energy={_ge!r} without tul.grad_pass: the energy exists "
                    "only to be differentiated into the pass's input. Set grad_pass: true "
                    "or leave the energy at 'own_mux'.")
            if cfg.tul.tokens_through_core:
                raise NotImplementedError(
                    f"tul.grad_pass_energy={_ge!r} has no defined meaning under the paid "
                    "loop (tokens_through_core): there is no per-slot looped state to "
                    "score. The paid loop already refuses tul.grad_pass in _tul_core.")
            if _ge == "critic":
                # The within-context improvement critic. Same construction shape as `disc`
                # — a scalar head on [z, ctx], private init stream, no RNG draw from the
                # global one — and a different LABEL: a pairwise comparison of two
                # candidate slot states through a REPLAY of the real coda
                # (`_tul_critic_loss`). The full-axis coda requirement is the same as
                # `disc`'s and is already checked in `TULConfig.__post_init__`; the two
                # below need the MODEL's shape.
                if cfg.n_core == 0:
                    raise ValueError(
                        "tul.grad_pass_energy='critic' needs a core loop "
                        "(model.n_core > 0): its label compares the state after pass t "
                        "with the state after pass t-1, and a coreless model has no "
                        "passes to compare.")
                if cfg.tul.gate is not None:
                    raise NotImplementedError(
                        "tul.grad_pass_energy='critic' with tul.gate is not defined: the "
                        "gate rewrites h_slots with a decoded budget before "
                        "`prefix_project`, so the replay would write an un-conditioned "
                        "candidate into a cell the shipped forward fills with a "
                        "conditioned one, and the two CEs would not be comparable.")
                self.tul_egrad = CriticEnergy(d, int(cfg.tul.egrad_disc_hidden or d))
                # `critic_every` counts TRAINING FORWARDS, not optimiser steps. Under
                # gradient accumulation those differ, and the arm's panel runs
                # accumulation 1 — named in the pre-registration rather than hidden here.
                self._critic_calls = 0
            elif _ge == "recon":
                self.tul_egrad = ReconEnergy(
                    d_model=d,
                    n_heads=int(cfg.tul.egrad_heads or cfg.n_heads),
                    d_ff=int(cfg.d_ff),
                    n_layers=int(cfg.tul.egrad_layers),
                    max_tokens=int(cfg.tul.egrad_max_tokens or cfg.tul.bound_span_cap),
                    soft_labels=bool(cfg.tul.egrad_soft_labels),
                    soft_mix=float(cfg.tul.egrad_soft_mix),
                )
            else:
                if not (cfg.tul.coda_sees_slots and cfg.tul.coda_token_cut == 0):
                    raise NotImplementedError(
                        "tul.grad_pass_energy='disc' needs the FULL-AXIS coda "
                        "(coda_sees_slots=true, coda_token_cut=0): its label is the coda's "
                        "per-token CE indexed by `layout.bag_id`, and the gathered coda of "
                        "arm A4 / arm CW runs on a different index space that "
                        "`slot_outcome_labels` does not re-derive.")
                self.tul_egrad = DiscEnergy(d, int(cfg.tul.egrad_disc_hidden or d))
        if cfg.tul is not None and cfg.tul.reinject_seed_every_pass:
            # See TULConfig.reinject_seed_every_pass: `_tul_core` already hands the slot's
            # prelude-entry state `e` to EVERY pass through `_apply_core_step`'s opening
            # `self.injection(h_in, e_in)` and the per-layer x0/bigram terms gathered at the
            # slot positions. A second additive copy of the seed is a duplicate path, so
            # this refuses rather than shipping one.
            raise NotImplementedError(
                "tul.reinject_seed_every_pass is a NO-OP by construction and therefore "
                "refused: the slot seed already reaches every pass. `_tul_core` binds "
                "`_e_arg = e` (the prelude's output at the slot position, i.e. `E_slot + "
                "W_sent . embed(t_last)` after the prelude) ONCE and passes it to every "
                "`_core_step`, where `_apply_core_step` opens with "
                "`self.injection(h_in, e_in)` — a DiagonalInjection of that same `e` at "
                "every pass — and then adds the per-core-layer x0/bigram injection terms, "
                "themselves gathered from the slot positions. Adding a second copy would "
                "be a duplicate path no measurement could separate from a change in the "
                "injection's gain. If you want a STRONGER seed, change the injection, not "
                "this knob.")

        # ── The span decoder (TULConfig.spandec; morph/model/tul_spandec.py) ──────
        # Built here, beside the other slot-loop readers, and RNG-neutral: every real draw
        # comes from a private generator, so this arm's base weights are byte-identical to
        # its ruler's and the two differ by the mechanism alone.
        self.tul_spandec: SpanDecoder | None = None
        if cfg.tul is not None and cfg.tul.spandec:
            if cfg.tul.gate is not None:
                raise NotImplementedError(
                    "tul.spandec with tul.gate is not defined: the gate conditions h_slots "
                    "on a decoded budget AFTER the local losses are taken, so the decoder "
                    "and the coda would read two different slot states. Pick one.")
            self.tul_spandec = SpanDecoder(
                d_model=d,
                n_heads=int(cfg.tul.spandec_heads or cfg.n_heads),
                d_ff=int(cfg.d_ff),
                n_layers=int(cfg.tul.spandec_layers),
                max_tokens=int(cfg.tul.spandec_max_tokens or cfg.tul.bound_span_cap),
                horizon=int(cfg.tul.spandec_horizon),
                # The per-pass target's own position table (TULConfig.spandec_per_pass):
                # `pass_horizon_max` blocks of `pass_tokens`, a DIFFERENT block geometry
                # from the exit target's, so it cannot share `pos`. Zero-init, so an arm
                # with the knob on still starts from the ruler's forward.
                pass_positions=(int(cfg.tul.spandec_pass_horizon_max)
                                * int(cfg.tul.spandec_pass_tokens)
                                if cfg.tul.spandec_per_pass else 0),
                # The register's READER (TULConfig.spandec_reads_cells): a per-layer
                # cross-attention onto the slot's M cells. Its output projection is
                # zero-init and its weights come from a THIRD private stream, so an arm
                # with the knob on is byte-identical to `slot-register-m4` at step 0 and
                # differs by one mechanism afterwards. Refused at slot_cells == 1.
                reads_cells=bool(cfg.tul.spandec_reads_cells),
                # WHICH span the decoder decodes (TULConfig.spandec_target_offset). 1 is
                # the shipped next-span target; k decodes span s+k and ONLY that span.
                target_offset=int(cfg.tul.spandec_target_offset),
            )
            if cfg.tul.spandec_per_pass and cfg.n_core == 0:
                raise ValueError(
                    "tul.spandec_per_pass needs a core loop (model.n_core > 0): it grades "
                    "the state after EVERY pass, and a coreless TUL model has no passes.")

        # ── Parallel span decoding from the coda (TULConfig.coda_span_heads) ──────
        # J offset heads on the coda's FINAL state at each slot's emitting position. The
        # `_MTPHead` construction (RMSNorm + a [d, d] linear at identity init) is
        # deterministic, so building these draws no RNG and an arm's base weights stay
        # byte-identical to its ruler's. Every refusal that depends only on the config
        # lives in `TULConfig.__post_init__`; the two below need the MODEL's shape.
        self.coda_span: nn.ModuleList | None = None
        if cfg.tul is not None and int(cfg.tul.coda_span_heads) > 0:
            if cfg.tul.prefix_k < 1:
                raise ValueError(
                    "tul.coda_span_heads needs tul.prefix_k >= 1: the heads read the coda "
                    "at `slot_index + prefix_k - 1`, the slot's last prefix cell.")
            if cfg.tul.gate is not None:
                raise NotImplementedError(
                    "tul.coda_span_heads with tul.gate is not defined: the gate rewrites "
                    "h_slots with a decoded budget before `prefix_project`, so the cell the "
                    "heads read would carry the budget-conditioned state and the arm's "
                    "reading would mix two mechanisms.")
            # RNG-NEUTRAL, and it has to be said out loud: `_MTPHead`'s weights are
            # deterministic (identity), but `nn.Linear.reset_parameters` still DRAWS from
            # the global stream before the identity overwrites it, which would shift every
            # weight constructed after this point and make the arm differ from its ruler by
            # more than the mechanism. The `_slot_gain_penalty` precedent: put the stream
            # back. Verified by `tests/test_tul_coda_span.py`, which asserts every shared
            # parameter is byte-identical between an on-model and an off-model.
            _rng = torch.get_rng_state()
            self.coda_span = nn.ModuleList(
                [_MTPHead(d) for _ in range(int(cfg.tul.coda_span_heads))])
            torch.set_rng_state(_rng)
        # ── The core-token auxiliary (TULConfig.core_token_aux) ───────────────────
        # Builds NOTHING: it is a second forward through parameters that already exist, so
        # an arm with the knob on holds byte-identical weights to its ruler and its only
        # cost is compute. The refusals that depend on the config alone live in
        # `TULConfig.__post_init__`; the three below need the MODEL's shape.
        if cfg.tul is not None and cfg.tul.core_token_aux:
            if cfg.n_core == 0:
                raise ValueError(
                    "tul.core_token_aux needs a core loop (model.n_core > 0): the arm's "
                    "whole content is sending the token positions through the core, and a "
                    "coreless model would run a second copy of the prelude's output "
                    "through the coda for nothing.")
            # `model.core_gain_lambda` needs no refusal here: it is ALREADY refused on
            # every TUL model above (arc E10c), which is what keeps `_core_region`'s
            # persistent `_core_gain_dir` power-iteration buffer out of the aux forward.
            if getattr(cfg, "fm", None) is not None:
                raise NotImplementedError(
                    "tul.core_token_aux with an FM planner (tul.fm / cfg.fm) is not "
                    "defined: the planner REPLACES the core loop (n_core == 0 is a build "
                    "precondition there), so there is no core for the tokens to be sent "
                    "through and no shared weights for the aux CE to train.")

        # ── The loop reads tokens (TULConfig.loop_reads_tokens) ───────────────────
        # Builds NOTHING: it runs `_core_region` — weights that already exist — over the
        # packed row instead of over the gathered slot cells, so an arm with the knob on
        # holds byte-identical weights to its ruler. The config-only refusals live in
        # `TULConfig.__post_init__`; the three below need the MODEL's shape.
        if cfg.tul is not None and cfg.tul.loop_reads_tokens:
            if cfg.n_core == 0:
                raise ValueError(
                    "tul.loop_reads_tokens needs a core loop (model.n_core > 0): the arm's "
                    "whole content is which positions the core runs over, and a coreless "
                    "model runs it over none.")
            if getattr(cfg, "fm", None) is not None:
                raise NotImplementedError(
                    "tul.loop_reads_tokens with an FM planner (tul.fm / cfg.fm) is not "
                    "defined: the planner REPLACES the core loop (n_core == 0 is a build "
                    "precondition there), so there is no core for the tokens to run "
                    "through.")
            if cfg.scse_enabled:
                raise NotImplementedError(
                    "tul.loop_reads_tokens under SCSE is not defined: SCSE's core is "
                    "source-free and its carrier is the DEVIATION, and nothing specifies "
                    "what a token position's deviation from its own anchor means here.")

        # ── The Thought Register (TULConfig.slot_cells) ───────────────────────────
        # `W_o` zero-init and `P_cell` zeros, so step 0 is the ruler's forward exactly; the
        # three real draws come from a PRIVATE generator, so a register model's BASE
        # weights are byte-identical to its ruler's and the arm differs by the mechanism
        # alone (the `W_sent` precedent). M == 1 builds nothing.
        self.tul_register: TULSlotRegister | None = None
        if cfg.tul is not None and cfg.tul.slot_cells > 1:
            if cfg.n_core == 0:
                raise ValueError(
                    "tul.slot_cells>1 needs a core loop (model.n_core > 0): the register's "
                    "claim is that M cells give the PASSES something to relate, and there "
                    "are no passes.")
            if cfg.scse_enabled:
                raise NotImplementedError(
                    "tul.slot_cells>1 under SCSE is not defined: the carrier is the "
                    "DEVIATION, so 'cell i of slot k' names a deviation and not a state.")
            if getattr(cfg, "fm", None) is not None:
                raise NotImplementedError(
                    "tul.slot_cells>1 with an FM planner (tul.fm / cfg.fm): the planner "
                    "REPLACES the core loop and writes ONE plan per slot through W_prefix.")
            self.tul_register = TULSlotRegister(d, cfg.tul.slot_cells,
                                                cfg.tul.slot_cell_init == "distinct")

        # ── LXTUL: the fan's exit mixture (TULConfig.fan_k; morph/model/tul_fan.py) ──
        # `fan_k: 0` builds nothing and the attribute stays None, so every branch that
        # reads it below is a Python-level constant that traces out and the forward is
        # the one from before this key. The fan's K STREAMS are the register built just
        # above (`slot_cells == fan_k`, enforced in TULConfig); THIS module is the one
        # thing the register did not have — the exit selector that turns K streams into
        # the ONE state every reader and the ordinary single-source `prefix_project`
        # take. At `fan_mix: "mean"` it owns no parameter and draws no RNG; at
        # `"softmax"` its ONE d->1 linear is zero-init and RNG-neutral, so a softmax
        # arm's step-0 forward equals the mean control's exactly.
        self.tul_fan: TULFanMix | None = None
        if cfg.tul is not None and cfg.tul.fan_k > 0:
            self.tul_fan = TULFanMix(d, cfg.tul.fan_k, cfg.tul.fan_mix)
        # `tul.fan_repel_mode: "epi"` — the epiplexity diversity term's FROZEN random
        # reservoir (morph/model/tul_fan.py). Buffers, never parameters; a private
        # generator, so the base weights are byte-identical to the cosine arm's. `"cos"`
        # (the default) builds nothing and the forward is the one from before the key.
        self.tul_fan_epi: FanReservoir | None = None
        if self.tul_fan is not None and cfg.tul.fan_repel_mode == "epi":
            self.tul_fan_epi = FanReservoir(d, cfg.tul.fan_epi_features, seed=0)

        # ── The discrete thought (TULConfig.vq_codes; morph/model/tul_vq.py) ──────
        # K codes per span instead of one continuous vector, lifted into the K prefix
        # cells the coda already reads. RNG-neutral (private generator, global stream
        # snapshotted), so a VQ model's base weights are byte-identical to its ruler's.
        # `vq_codes: 0` builds nothing and draws nothing.
        # ── TUL-Code (TULConfig.code; morph/model/tul_code.py; docs/tul-code-spec.md) ──
        # The slot holds the CODE of the span it precedes. E makes it from that span at
        # train; the core body samples it at eval. `code: false` builds nothing and the
        # forward is bit-identical to the strict ruler (tests/test_tul_code.py, C1).
        # RNG-neutral: every module below draws from a private generator.
        self.tul_code_enc: TULCodeEncoder | None = None
        self.tul_code_proj: TULCodeProj | None = None
        self.tul_code_time: TULCodeTime | None = None
        self.tul_code_head: TULCodeHead | None = None
        # `code_phase` is a Python int the TRAINER sets per step (1 define the code, 2 learn
        # to guess it, 3 rollout). A trace-time branch, on purpose: a multiply-by-zero gate
        # would pay the thinker in phase 1 and the sampler in phases 1-2 (spec §6). Two
        # recompiles per run, at the two switches.
        self.code_phase: int = 2
        self._code_fm_scale: float = 1.0
        self._code_last_passes: int = 0
        # The statistic a TRUTH code carries into the coda at train: `z + noise·ε` has RMS
        # sqrt(1 + noise²) unless `code_noise_renorm` puts it back to 1. Eval's encoder mode
        # and the generator's closed-slot tape feed z at THIS scale, so the coda sees the
        # truth code at the statistic it was trained on (2026-09-15: the bare-z eval read
        # 1.25 nats where the trained statistic reads 0.35 on tul-code-20k).
        self._code_truth_scale: float = 1.0
        if cfg.tul is not None and cfg.tul.code and cfg.tul.code_noise > 0.0 \
                and not cfg.tul.code_noise_renorm:
            self._code_truth_scale = float((1.0 + cfg.tul.code_noise ** 2) ** 0.5)
        if cfg.tul is not None and cfg.tul.code:
            if cfg.n_core == 0:
                raise ValueError("tul.code needs a core body (model.n_core > 0): the core "
                                 "blocks ARE the velocity field.")
            if cfg.fm is not None:
                raise NotImplementedError(
                    "tul.code with an FM planner (cfg.fm): two samplers for one cell.")
            if cfg.core_init_scale > 0.0 or cfg.scse_enabled:
                raise NotImplementedError(
                    "tul.code with core_init_scale > 0 or SCSE: the thinker's entry state is "
                    "z_t itself (spec §3.5), so the entry map and the deviation carry have "
                    "no place on this path.")
            if str(cfg.tul.core_stage_cond) != "none":
                raise NotImplementedError(
                    f"tul.code with tul.core_stage_cond={cfg.tul.core_stage_cond!r}: the "
                    f"code path carries its own time conditioning (TULCodeTime).")
            _M = int(cfg.tul.prefix_k)
            self.tul_code_enc = TULCodeEncoder(d, _M)
            self.tul_code_time = TULCodeTime(d, t_embed_scale=cfg.tul.code_t_embed_scale)
            # LCTUL-D (spec §16): the code is N = M·G symbols; the thinker is a masked
            # denoiser (`tul_code_sym_head`) over symbol embeddings (`tul_code_sym`, row C
            # = MASK) and there is NO velocity head. The symbol embedding is drawn at unit
            # per-component scale (the flow thinker's entry state is a unit-RMS z_t) from
            # a PRIVATE generator with the global stream restored (the TULSlotRegister
            # precedent), so a discrete model's base weights equal its ruler's.
            self.tul_code_vq: TULThoughtVQ | None = None
            self.tul_code_sym: nn.Embedding | None = None
            self.tul_code_sym_head: TULCodeSymHead | None = None
            self.tul_code_head: TULCodeHead | None = None
            _N = _M
            if cfg.tul.code_discrete:
                _C = int(cfg.tul.code_vq_codebook)
                _N = _M * int(cfg.tul.code_vq_groups)
                self.tul_code_vq = TULThoughtVQ(
                    d, codes=_M, codebook=_C, dim=int(cfg.tul.code_vq_dim),
                    groups=int(cfg.tul.code_vq_groups), beta=float(cfg.tul.code_vq_beta),
                    reset_after=0)
                _rng0 = torch.random.get_rng_state()
                _g = torch.Generator(device="cpu").manual_seed(0xC0DE + 2)
                self.tul_code_sym = nn.Embedding(_C + 1, d)
                with torch.no_grad():
                    self.tul_code_sym.weight.copy_(torch.empty(_C + 1, d).normal_(
                        mean=0.0, std=1.0, generator=_g))
                torch.random.set_rng_state(_rng0)
                self.tul_code_sym.weight._ternary_exclude = True
                self.tul_code_sym_head = TULCodeSymHead(d, _C)
                self._code_mask_id = _C
                self._code_n_sym = _N
            else:
                self.tul_code_head = TULCodeHead(d)
            # Per-cell embedding on the NOISY copies (which cell of the slot is being
            # denoised: at t≈0 the state is noise and carries no cell identity) and one
            # marker on the CLEAN copies. Both zero-init, no RNG draw. On a discrete model
            # the "cell" axis is the N symbol positions.
            self.tul_code_cell = nn.Parameter(torch.zeros(_N, d))
            self.tul_code_clean = nn.Parameter(torch.zeros(d))
            # CFG null seed, built ONLY when the null condition is trained (`code_cfg_drop`
            # > 0), so a pre-CFG checkpoint still loads into a non-CFG model.
            self.tul_code_null = (nn.Parameter(torch.zeros(d))
                                  if cfg.tul.code_cfg_drop > 0.0 else None)
            # The unit the flow / denoiser loss is reported in: the CFM null floor, or for
            # a discrete code the UNIFORM floor N·log C (the ELBO of a head that knows
            # nothing), so `code_fm_rel` reads 1.0 at the zero-init head on both paths.
            self._code_fm_scale = (float(_N) * math.log(float(cfg.tul.code_vq_codebook))
                                   if cfg.tul.code_discrete
                                   else cfm_null_floor(d, _M, cfg.tul.code_source_std))
            # W_prefix is BUILT (the parameter set matches the ruler, C1) and never applied
            # on a code model: the cells are scattered directly (spec §3.2).
            if self.tul is not None and self.tul.W_prefix is not None:
                self.tul.W_prefix.requires_grad_(False)
        if cfg.tul is not None and cfg.tul.code_target:
            # ── the code target (tul.code_target; spec §17) ──────────────────────
            # E is a frozen TARGET: built here so its tensors load from the VAE stage's
            # checkpoint, and never trained by this arm (requires_grad off at build;
            # `training.train_only` may not re-enable it — the arm's config lists no
            # `tul_code_enc.` prefix). The projection is the loop's ONLY write into the
            # coda; `W_prefix` is built and inert, the `tul.code` precedent.
            if cfg.n_core == 0:
                raise ValueError("tul.code_target needs a slot loop (model.n_core > 0): the "
                                 "projection reads the loop's exit state.")
            if cfg.fm is not None:
                raise NotImplementedError(
                    "tul.code_target with an FM planner (cfg.fm): the planner replaces the "
                    "slot loop, so there is no looped exit state to project.")
            _M = int(cfg.tul.prefix_k)
            self.tul_code_enc = TULCodeEncoder(d, _M)
            for _p in self.tul_code_enc.parameters():
                _p.requires_grad_(False)
            self.tul_code_proj = TULCodeProj(d, _M)
            if self.tul is not None and self.tul.W_prefix is not None:
                self.tul.W_prefix.requires_grad_(False)
        self.tul_vq: TULThoughtVQ | None = None
        if cfg.tul is not None and cfg.tul.vq_codes > 0:
            if cfg.n_core == 0:
                raise ValueError(
                    "tul.vq_codes needs a core loop (model.n_core > 0): the quantizer sits "
                    "on the loop's EXIT state and a coreless TUL model has no exit.")
            if cfg.scse_enabled:
                raise NotImplementedError(
                    "tul.vq_codes under SCSE is not defined: the carrier is the DEVIATION, "
                    "so a code would be a symbol for a deviation and not for a thought.")
            if getattr(cfg, "fm", None) is not None:
                raise NotImplementedError(
                    "tul.vq_codes with an FM planner (cfg.fm): the planner REPLACES the "
                    "core loop and writes ONE detached plan per slot through W_prefix, so "
                    "there is no looped exit state to quantize and no gradient for the STE "
                    "to carry. Pick one.")
            self.tul_vq = TULThoughtVQ(
                d, codes=cfg.tul.vq_codes, codebook=cfg.tul.vq_codebook,
                dim=cfg.tul.vq_dim, groups=cfg.tul.vq_groups, beta=cfg.tul.vq_beta,
                reset_after=cfg.tul.vq_reset_after)
        # ── C1: the row-centered exit (tul.center_exit; morph/model/tul.py) ───────
        # Built here, beside the other slot-loop levers, and RNG-NEUTRAL by construction:
        # the only parameter is `b_center`, drawn by `torch.zeros`, so a center arm's base
        # weights are byte-identical to its ruler's whatever the build order.
        #
        # `cfg.tul.center_exit` already refuses `tokens_through_core` and
        # `loop_reads_tokens` in `TULConfig.__post_init__` (config-only refusals live
        # there, per the rule at the head of this block). The two below need the MODEL's
        # shape and cannot be seen from the config.
        self.tul_center: TULCenterExit | None = None
        if cfg.tul is not None and cfg.tul.center_exit:
            if getattr(cfg, "fm", None) is not None:
                raise NotImplementedError(
                    "tul.center_exit with an FM planner (cfg.fm): the planner REPLACES the "
                    "core loop and DETACHES its plan before W_prefix, so there is no looped "
                    "exit state and centering could not reach the loop it is aimed at. "
                    "Pick one.")
            if cfg.n_core == 0:
                raise ValueError(
                    "tul.center_exit needs a core loop (model.n_core > 0): the lever centers "
                    "the LOOP's exit state and there are no passes.")
            self.tul_center = TULCenterExit(
                d, self._n_streams if self._is_hc else 0)

        # ── C2: the within-row contrastive objective (tul.row_contrast_lambda) ────
        # Built beside the register, and the POSITION does not matter, which is the
        # point: `nn.Linear` draws on the GLOBAL RNG stream before its weight is
        # overwritten, so `TULRowContrast` snapshots that stream and restores it. A
        # contrast arm's base weights are byte-identical to its ruler's and no module
        # built after this one is shifted.
        self.tul_contrast: TULRowContrast | None = None
        if cfg.tul is not None and cfg.tul.row_contrast_lambda > 0.0:
            if getattr(cfg, "fm", None) is not None:
                raise NotImplementedError(
                    "tul.row_contrast_lambda > 0 with an FM planner (cfg.fm): the planner "
                    "REPLACES the core loop and detaches its plan, so the term would "
                    "shape nothing the loop does. Pick one.")
            if cfg.n_core == 0:
                raise ValueError(
                    "tul.row_contrast_lambda > 0 needs a core loop (model.n_core > 0): the "
                    "term exists to make the LOOP write distinct states and there are no "
                    "passes.")
            self.tul_contrast = TULRowContrast(d, cfg.tul.row_contrast_tau)

        # ── LoopMTP horizon alignment (tul.horizon_weight; morph/model/tul.py) ─────
        # `nn.Linear` at IDENTITY init (`torch.eye`, deterministic, no RNG draw): step 0
        # compares the pass's RAW readout to the horizon target, and every other weight
        # of an arm with this on is byte-identical to its ruler's at the same seed.
        # TRAINING-ONLY (the oracle_z / spandec_per_pass precedent): it never appears in
        # the deployed forward, so it carries `_ternary_exclude` like those decoders.
        self.tul_horizon_proj: nn.Linear | None = None
        if cfg.tul is not None and cfg.tul.horizon_weight > 0.0:
            if cfg.n_core == 0:
                raise ValueError(
                    "tul.horizon_weight > 0 needs a core loop (model.n_core > 0): the "
                    "term grades the LOOP's per-pass states and there are no passes.")
            if cfg.fm is not None:
                raise NotImplementedError(
                    "tul.horizon_weight > 0 with an FM planner (cfg.fm): the planner "
                    "REPLACES the core loop and detaches its plan, so there is no "
                    "per-pass trajectory to align. Pick one.")
            self.tul_horizon_proj = nn.Linear(d, d, bias=False)
            with torch.no_grad():
                self.tul_horizon_proj.weight.copy_(torch.eye(d))
            self.tul_horizon_proj._ternary_exclude = True

        # ── LoopMTP gated readout (tul.pass_readout="gated"; morph/model/tul.py) ───
        # NOT training-only: substituted for `h_slots` at the same seam `cond_layers`,
        # `center_exit`, the register's mean and VQ already share, so it is part of the
        # REAL forward at train and eval. `slot_depth_fixed` is already validated > 0 by
        # `TULConfig.__post_init__` whenever this knob is "gated".
        self.tul_pass_gate: TULPassGate | None = None
        if cfg.tul is not None and cfg.tul.pass_readout == "gated":
            if cfg.n_core == 0:
                raise ValueError(
                    "tul.pass_readout='gated' needs a core loop (model.n_core > 0): the "
                    "gate combines the LOOP's per-pass states and there are no passes.")
            if cfg.fm is not None:
                raise NotImplementedError(
                    "tul.pass_readout='gated' with an FM planner (cfg.fm): the planner "
                    "REPLACES the core loop and detaches its plan, so there is no "
                    "per-pass trajectory to gate over. Pick one.")
            self.tul_pass_gate = TULPassGate(d, int(cfg.tul.slot_depth_fixed))

        # ── The slot chain (TULConfig.slot_chain) ─────────────────────────────────
        # Zero-init, no RNG draw: step 0 is the ruler's forward bit for bit.
        # ── the loop carry (tul.loop_carry; morph/model/tul_carry.py) ────────────
        # `none` — every other model — leaves this None, and every branch that reads it
        # in the forward is then a Python-level constant that traces out: no capture is
        # requested from core layer 0, no state is allocated and `_core_step` is the
        # function it was before this key. At `sum` the module owns NO parameter; at
        # `gate` it owns ONE zero-init `d x 2d` and draws no RNG, so a carry model shares
        # every base weight with its same-seed ruler.
        self.tul_carry: TULLoopCarry | None = None
        if cfg.tul is not None and cfg.tul.loop_carry != "none":
            if cfg.n_core == 0:
                raise ValueError(
                    "tul.loop_carry needs a core loop (model.n_core > 0): the carry is "
                    "written from a pass's cross-cell read and re-injected at the next "
                    "pass, and there are no passes.")
            if cfg.scse_enabled:
                raise NotImplementedError(
                    "tul.loop_carry under SCSE is not defined: the loop carrier there is "
                    "the DEVIATION from a fixed anchor h*, and `_core_step` is run "
                    "source-free — the per-layer injection the carry would ride is "
                    "skipped entirely (spec D3), and an RMS match against a deviation is "
                    "a match against a quantity that is meant to shrink.")
            self.tul_carry = TULLoopCarry(d, cfg.tul.loop_carry)

        self.tul_chain: TULSlotChain | None = None
        if cfg.tul is not None and cfg.tul.slot_chain:
            if cfg.n_core == 0:
                raise ValueError(
                    "tul.slot_chain needs a core loop (model.n_core > 0): the chain runs "
                    "once per pass, and there are no passes.")
            if cfg.scse_enabled:
                raise NotImplementedError(
                    "tul.slot_chain under SCSE is not defined: the loop carrier is the "
                    "DEVIATION, so the chain would forward a deviation, not a slot state.")
            self.tul_chain = TULSlotChain(d, cfg.tul.slot_chain_detach)

        # Progressive loss (tul.progressive_p, Bansal et al. 2022). Builds NOTHING — it is
        # a per-slot detach pattern inside `_tul_core` — so the only construction-time work
        # is refusing the two model shapes on which the per-slot prefix has no meaning.
        if cfg.tul is not None and cfg.tul.progressive_p > 0.0:
            if cfg.n_core == 0:
                raise ValueError(
                    "tul.progressive_p needs a core loop (model.n_core > 0): there are no "
                    "passes to split into a no-grad prefix and a grad tail.")
            if cfg.scse_enabled:
                raise NotImplementedError(
                    "tul.progressive_p under SCSE is not defined: the loop carrier is the "
                    "DEVIATION and the anchor h* is built once OUTSIDE the loop and stays "
                    "live, so detaching a prefix would still leak gradient into h* and the "
                    "prefix would not be gradient-free.")

        # ── Core-stage conditioning (faithful DiffusionBlocks, morph/model/iter_cond.py)
        # Built AFTER TULSlots/tul_gate for the same RNG-neutrality reason: "none" (the
        # default) constructs nothing, draws no RNG, and every OTHER arm's base weights
        # stay byte-identical to a build from before this module existed.
        # `_core_stage_cond_mode` is a Python-level constant read once at construction —
        # every branch on it below traces out, so the "none" graph is unchanged.
        self._core_stage_cond_mode: str = cfg.tul.core_stage_cond if cfg.tul is not None else "none"
        self.tul_stage_cond: CoreStageConditioning | None = None
        self._db1_sampler: DB1Sampler | None = None
        if self._core_stage_cond_mode != "none":
            if cfg.n_core <= 0:
                raise ValueError(
                    "tul.core_stage_cond requires model.n_core > 0 — there are no core "
                    "layers to condition (docs: morph/model/iter_cond.py).")
            self.tul_stage_cond = CoreStageConditioning(
                cfg.n_core, d, cond_dim=cfg.tul.db1_cond_dim)
            self._db1_sampler = DB1Sampler(
                sigma_min=cfg.tul.db1_sigma_min, sigma_max=cfg.tul.db1_sigma_max,
                p_mean=cfg.tul.db1_p_mean, p_std=cfg.tul.db1_p_std,
                sigma_data=cfg.tul.db1_sigma_data)

        # ── GRT recurrence gate (morph/model/recur_gate.py, gate-ladder G1/G2) ──
        # Built after stage-cond, before FM, same RNG-neutrality contract: "none" (the
        # default) constructs nothing and every other arm's weights are byte-identical.
        self.tul_recur_gate: RecurrenceGate | None = None
        if cfg.tul is not None and cfg.tul.recur_gate == "grt":
            if cfg.n_core <= 0:
                raise ValueError(
                    "tul.recur_gate requires model.n_core > 0 — there is no recurrence "
                    "to gate.")
            if cfg.scse_enabled:
                raise NotImplementedError(
                    "tul.recur_gate under SCSE is not defined: the loop carrier is the "
                    "DEVIATION and a convex blend of deviations is not a convex blend of "
                    "states. Build it when an arm needs it.")
            self.tul_recur_gate = RecurrenceGate(
                d, tau=cfg.tul.recur_gate_tau, bias_init=cfg.tul.recur_gate_bias,
                noise=cfg.tul.recur_gate_noise)

        # ── FM1 planner (morph/model/tul_fm.py) ────────────────────────────
        # Built LAST so a non-FM model's weights are byte-identical to today's: every
        # parameter drawn below advances the global RNG, so any earlier placement would
        # change the baseline's own initialisation.
        self.fm_planner = None
        self._fm_schedule = None
        self._fm_loss_scale = 1.0
        if cfg.fm is not None:
            self._build_fm(cfg, d)

        # ── Think-once conditioning stack (arm R7, tul.cond_layers) ─────────────────
        # `cond_layers` NON-SHARED MORPHBlocks that run ONCE over the compact slot
        # sequence after the core loop and before the coda reads z. They are ordinary
        # blocks (same constructor as the coda, same attention kwargs as the core) so a
        # `cond_layers: 4` model has exactly the parameter count of an `n_coda + 4`
        # model — that equality is what makes R7 (cond4 + coda4) vs R8 (coda8) a fair
        # question about WHERE four layers pay: on ~50 slot positions or on every token.
        # Built after the FM planner and only when > 0, so every other arm draws no RNG
        # here and keeps byte-identical weights (the retention/TULSlots contract).
        self.tul_cond: nn.ModuleList | None = None
        if cfg.tul is not None and cfg.tul.cond_layers > 0:
            if cfg.fm is not None:
                raise NotImplementedError(
                    "tul.cond_layers with an FM planner is not defined: the planner "
                    "already replaces the slot loop and detaches its plan. Pick one.")
            if cfg.tul.tokens_through_core:
                raise NotImplementedError(
                    "tul.cond_layers with tul.tokens_through_core (A2) is not defined: A2 "
                    "has no compact slot sequence for the stack to run over.")
            if cfg.tul.db_loop:
                raise NotImplementedError(
                    "tul.cond_layers with tul.db_loop is not defined: the db local losses "
                    "supervise the loop's per-iteration states, which the stack would "
                    "no longer be the state the coda reads.")
            self.tul_cond = nn.ModuleList([
                _make_block(n_total + i, core_attn_kw) for i in range(cfg.tul.cond_layers)
            ])

        # ── SCSE Stage 1 loop entry ────────────────────────────────────────
        # Built LAST, for the same reason TULSlots is: `_CloneInit` draws no RNG at all,
        # and `_SCSEInit` draws its Linear init AFTER every other parameter, so a baseline
        # model and an SCSE model built from the same seed share byte-identical weights
        # everywhere except this projection. The arms then differ by the mechanism alone.
        if cfg.core_state_init not in ("prelude", "noise"):
            raise ValueError(f"model.core_state_init must be 'prelude' or 'noise', "
                             f"got {cfg.core_state_init!r}")
        if cfg.core_state_init == "noise":
            if cfg.injection_channels != "all":
                raise ValueError(
                    "model.core_state_init='noise' requires model.injection_channels='all': "
                    "with the ctx-only carry the other carrier dims would never receive e and "
                    "stay noise for the whole loop.")
            if cfg.core_init_scale > 0.0 or cfg.scse_enabled:
                raise ValueError("model.core_state_init='noise' is not defined with "
                                 "core_init_scale > 0 or scse_enabled (both are h_0 = e entries).")
        self.core_init: nn.Module = (
            _SCSEInit(d, cfg.core_init_scale) if cfg.core_init_scale > 0.0
            else _NoiseInit(cfg.core_state_init_std) if cfg.core_state_init == "noise"
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
        if cfg.slot_cot_clip < 0.0:
            raise ValueError(f"model.slot_cot_clip must be >= 0 (0 = off), got {cfg.slot_cot_clip}")
        # The slot-loop levers (slot_cot_clip, slot_state_renorm, slot_gain_lambda) act inside
        # `_tul_core` and nowhere else. base.yaml carries the measured constraint ON for every
        # slot-loop model, so a model WITHOUT a slot loop (no TUL, or the paid loop's
        # tokens_through_core, or the FM planner) must build; it says so once, loudly, instead
        # of pretending the knob did something (lab/divergence/BREAK-GLASS-IN-CASE-OF-
        # DIVERGENCE-THE-SLOT-LOOP-GAIN-CONSTRAINT.md). A TUL model with n_core == 0 asked for
        # a lever on a loop it does not have: that is a contradiction and raises.
        _levers = {k: getattr(cfg, k) for k in ("slot_cot_clip", "slot_state_renorm", "slot_gain_lambda")
                   if getattr(cfg, k)}
        _why = ("no TUL block" if cfg.tul is None else
                "n_core=0 (a coreless TUL model has no loop)" if cfg.n_core == 0 else
                "tokens_through_core (the paid loop runs _core_region, not _tul_core)"
                if cfg.tul.tokens_through_core else
                # `loop_reads_tokens` is the SAME miss for the same reason: it runs
                # `_core_region` over the whole packed row and never enters `_tul_core`,
                # so a config that carries base.yaml's constraint knobs (as
                # tul_slot_spandec_strict_tokloop.yaml does, slot_gain_lambda 100 and
                # slot_cot_clip 4.0) gets NOTHING from them. Found 2026-09-13 by an
                # independent review of the knob's first commit.
                "loop_reads_tokens (the core stage runs _core_region, not _tul_core)"
                if cfg.tul.loop_reads_tokens else
                "an FM planner replaces the slot loop" if cfg.fm is not None else None)
        if _levers and _why is not None:
            print(f"  [slot-levers] {_levers} INERT on this model: {_why}. They act only inside "
                  "the slot loop (_tul_core).", flush=True)
        # An INERT lever cannot conflict with SCSE: the shipped paid loop builds with base.yaml's
        # constraint knobs on and scse off or on, and never reaches _tul_core.
        if cfg.slot_state_renorm and cfg.scse_enabled and _why is None:
            raise NotImplementedError(
                "model.slot_state_renorm under SCSE is not defined: the carrier is the DEVIATION "
                "and pinning its norm pins the wrong quantity (the db_loop precedent).")
        if cfg.slot_gain_lambda < 0.0 or cfg.slot_gain_eps <= 0.0:
            raise ValueError("model.slot_gain_lambda must be >= 0 and slot_gain_eps > 0")
        if cfg.slot_gain_lambda > 0.0 and cfg.scse_enabled and _why is None:
            raise NotImplementedError("model.slot_gain_lambda under SCSE is not defined (deviation carrier)")
        self.scse: _SCSE | None = (
            _SCSE(d, step_scale=cfg.scse_step_scale, anchor_scale=cfg.scse_anchor_scale,
                  init_scale=cfg.scse_init_scale, eps=cfg.scse_eps, kappa=cfg.scse_kappa,
                  input_mode=cfg.scse_input_mode, delta_clip=cfg.scse_delta_clip)
            if cfg.scse_enabled else None)

        # ── Per-pass low-rank deltas on the shared core (tul.pass_lora_rank) ───────
        # Bae et al. 2024. Attached LAST, and to the CORE blocks only: prelude and coda
        # run once and have no pass index. `PassLoRA` draws from a private generator and
        # B is exactly zero, so a model built with the knob has byte-identical base
        # weights to one without it AND an identical forward until training moves B
        # (tests/test_tul_pass_lora.py). The number of passes is the loop's own maximum
        # depth, so `iter_idx` can never index past the stack.
        self.pass_lora_n_passes = 0
        if cfg.tul is not None and cfg.tul.pass_lora_rank > 0:
            if cfg.n_core == 0:
                raise ValueError(
                    "tul.pass_lora_rank needs a core loop (model.n_core > 0): there are "
                    "no shared blocks and no passes to specialise.")
            if cfg.fm is not None:
                raise NotImplementedError(
                    "tul.pass_lora_rank with an FM planner is not defined: the planner "
                    "replaces the slot loop, so there is no pass index to key on.")
            self.pass_lora_n_passes = int(cfg.tul.slot_max_depth or cfg.max_depth)
            _lora_t = tuple(cfg.tul.pass_lora_targets)
            for _blk in self.core:
                _blk.attach_pass_lora(PassLoRA(
                    d, self.pass_lora_n_passes, int(cfg.tul.pass_lora_rank), _lora_t))
            _n_lora = sum(p.numel() for b in self.core
                          for p in b.pass_lora.parameters())
            print(f"  TUL PASS-LORA ON: rank={cfg.tul.pass_lora_rank} "
                  f"targets={_lora_t} passes={self.pass_lora_n_passes} "
                  f"blocks={cfg.n_core} -> {_n_lora:,} params ({_n_lora / 1e6:.2f}M), bf16, not "
                  f"ternarised, not pruned (Bae et al. 2024)", flush=True)

        # Master kernel switch → drives the fused-Triton-vs-eager-reference
        # dispatch in the attention kernels (process-global flag). Set at build
        # so the choice is captured in the run; the fused-CE branch in forward()
        # reads self.cfg.use_kernels directly.
        from morph.kernels.triton._eager_flag import set_force_eager
        set_force_eager(not cfg.use_kernels and not cfg.tg_scoped_kernels)
        if cfg.tg_scoped_kernels:
            print("  TG SCOPED KERNELS ON: HC-Cayley + CCA prologue + core window fused; "
                  "TG-restricted attention branches eager by construction; CE chunked")

        # Static-region CUDA graphs (MORPH_STATIC_GRAPHS): plain dict attr — holds the
        # graphed front/back callables + capture shapes. Deliberately NOT a submodule.
        self._static_graphs: dict = {}

        if cfg.retention_carry_mode == "acausal_final":
            print("  WARNING: retention_carry='acausal_final' — the cross-iteration GLA "
                  "carry feeds the WHOLE-SEQUENCE final state into loop iteration 2+, so "
                  "every position sees the future (runtime-invariants §5 violation; the "
                  "leak that faked the l2cap depth-earning). Load/diagnose pre-2026-08-31 "
                  "checkpoints only; NEVER train new models with it.")
        n_params = sum(p.numel() for p in self.parameters())
        _res = self._residual_mode + (f"(n={self._n_streams})" if self._is_hc else "")
        print(f"MORPHTransformer: {n_params/1e6:.1f}M params, "
              f"loop {cfg.n_prelude}:{cfg.n_core}×{cfg.mean_depth}:{cfg.n_coda} "
              f"(kernels={'fused' if cfg.use_kernels else ('EAGER+TGSCOPED' if cfg.tg_scoped_kernels else 'EAGER')}, "
              f"residual={_res})")

    # ── Helpers ───────────────────────────────────────────────────────

    def _sample_depths(self, B: int, device: torch.device) -> Tensor:
        if bool(self.cfg.depth_fixed):
            return torch.full((B,), int(self.cfg.max_depth), device=device, dtype=torch.long)
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
                         ret_state=None, iter_idx=0, inj_terms=None, source_free=False,
                         stage_cond=None, attn_kw=None):
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

        ``stage_cond`` (faithful DiffusionBlocks, morph/model/iter_cond.py):
        ``[B, cond_dim]`` or ``None``. ``None`` is bit-identical to before this
        parameter existed — every pre-existing call site passes nothing, so the
        default keeps the graph unchanged. When given, EVERY core layer's input is
        AdaLN-modulated by ``self.tul_stage_cond`` before that layer runs (zero-init,
        so this is a no-op until training moves the gate weights).
        """
        np_ = self.cfg.n_prelude
        mlp_kw = {"iter_idx": iter_idx}
        # ── The stream boundary of the Parcae core swap (model.core_impl) ────────────
        # `_core_is_parcae` is a Python bool fixed at construction, so this is a
        # trace-time branch and a "morph" model's graph is byte-for-byte what it was.
        # A ParcaeCoreBlock has a plain single-stream residual, so the [B,S,n,C] carrier
        # is collapsed ONCE per pass by the stream MEAN — the same reduction `_readout`
        # and `TULSlots.unpack` already use to read this carrier — and the pass's output
        # is broadcast back over the n streams at the end. The prelude and the coda keep
        # their Cayley carrier untouched, which is what makes the swap one factor.
        # `e_in` is collapsed too: `DiagonalInjection` reads it elementwise against `h`.
        _parcae = self._core_is_parcae
        if _parcae:
            h_in = h_in.mean(dim=2)
            if e_in is not None:
                e_in = e_in.mean(dim=2)
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
            if stage_cond is not None:
                h_injected = self.tul_stage_cond.modulate(h_injected, stage_cond, i)
            # Retention carry only for the designated core layer(s); others get None.
            is_ret = ret_cap is not None and (i in self._retention_layers)
            rs_arg = ret_state if is_ret else None
            rc_arg = ret_cap if is_ret else None
            # `attn_kw` is normally ONE dict shared by every core layer. `tul.loop_reach`
            # hands a per-LAYER tuple instead, because a pass's cross-cell reach is the
            # COMPOSITION over the n_core layers: at reach w applied to every layer a pass
            # would carry w * n_core, not w. A tuple is read positionally; a dict keeps the
            # old behaviour for every other caller, bit-identically.
            _akw = attn_kw[i] if isinstance(attn_kw, (list, tuple)) else attn_kw
            h_injected = layer(h_injected, mlp_kwargs=mlp_kw,
                               ret_state=rs_arg, ret_capture=rc_arg,
                               attn_kwargs=_akw, pass_idx=iter_idx)
        new_ret = ret_cap.get("state") if ret_cap is not None else None
        if _parcae:
            # Broadcast the single-stream pass output back over the n streams. `.expand`
            # then `.contiguous()`: every caller of this method treats the return as an
            # ordinary carrier (torch.where, advanced indexing, tensor hooks, checkpoint
            # inputs), and a materialised tensor is the one that cannot surprise any of
            # them. On the slot loop the carrier is [B, 64, 4, C] — the copy is noise.
            h_injected = h_injected.unsqueeze(2).expand(
                -1, -1, self._n_streams, -1).contiguous()
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
        if self._span_mask:
            raise RuntimeError(
                "MORPH_STATIC_GRAPHS with model.span_mask: the captured FRONT/BACK "
                "regions take input_ids only and would replay every batch under the "
                "span relation of the capture batch. Unset MORPH_STATIC_GRAPHS.")
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

    def _span_context(self, input_ids: Tensor):
        """``(block_kwargs, core_kwargs, bigram_cut)`` for ``model.span_mask``.

        Built ONCE per forward from the token ids alone, then threaded into the prelude,
        the core and the coda exactly the way the TG masks are — there is no per-block
        recomputation and no second rule. ``(None, None, None)`` on a model built with
        ``span_mask: "off"``, which keeps every op below untouched.

        ``span_mask: "row"`` passes ``rule=None``: one span per row, so the relation is
        plain causal, the segment reset never fires and the bigram cut is only the row's
        first position — i.e. exactly the unrestricted model, reached through the SAME
        code as the restricted one. That is what makes the budget pair a one-factor pair.

        The cut itself is ``BoundaryRule.cut``, a numpy state machine, so this costs one
        device->host copy of the ids per forward (50 KB at the panel shape) plus ~1 ms of
        CPU. It runs in the model rather than in the loader so that EVERY entry point —
        training, eval, `lab/divergence/core_depth_sweep.py` — is masked by construction,
        instead of only the ones somebody remembered to pass a mask to.

        The core's dict carries ``tg_allow`` alone: a `ParcaeCoreBlock` has no conv, no
        value shift and no pooled branch, so the softmax relation IS its whole
        cross-position path, and `_core_region` already sorts `tg_allow` into
        active-set order.
        """
        if not self._span_mask:
            return None, None, None
        ids_np = input_ids.detach().to("cpu", torch.int64).numpy()
        span_id = torch.from_numpy(span_ids_from_ids(ids_np, self._span_rule)).to(
            input_ids.device)
        allow = span_allow_mask(span_id)                       # [B, 1, S, S] bool
        cut = span_start_mask(span_id)                         # [B, S] bool
        block_kw = {"tg_allow": allow, "tg_comp_allow": allow, "tg_seg": span_id}
        return block_kw, {"tg_allow": allow}, cut

    def _front_region(self, input_ids: Tensor, attn_kwargs: dict | None = None,
                      bigram_cut: Tensor | None = None
                      ) -> tuple[Tensor, Tensor, Tensor | None]:
        """bag0 FRONT region: embed+dropout+bigram → _front_tail. Fixed shapes, no
        recurrence, one RNG site (embed_drop) + prelude MLP dropouts → graphable.

        ``attn_kwargs`` / ``bigram_cut``: ``model.span_mask``'s per-forward span masks
        (`_span_context`). None on every other path → bit-identical to before."""
        x = self.embed_drop(self.embed(input_ids))
        bigram_emb = self.embed.get_bigram(input_ids, bigram_cut)
        x, x0 = self._front_tail(x, input_ids, bigram_emb, None, attn_kwargs=attn_kwargs)
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

    def _core_gain_penalty(self, core_step, args, rs_a, t: int, akw, lam: float) -> dict:
        """Directional finite-difference gain hinge on the plain loop (arc E10b).

        Same probe as :meth:`_slot_gain_penalty` (two extra applications of the SAME core
        step at the detached operating point, RNG stream put back around each) with ONE
        change: under ``core_gain_direction == "power"`` the direction ``v`` (shape = one
        sample's state, unit norm, shared across the batch) is a persistent buffer updated
        by a power step from the probe's own response, so after a few steps ``g`` reads the
        map's top singular value rather than its typical gain. ``d = core_gain_eps *
        ||h_b|| * v`` per sample; ``g_b = ||f(h_b + d_b) - f(h_b)|| / ||d_b||``; penalty
        ``lam * mean relu(g - core_gain_target)^2``.
        """
        hp = args[0].detach()
        e_d = args[1].detach() if torch.is_tensor(args[1]) else args[1]
        inj_d = args[2].detach() if torch.is_tensor(args[2]) else args[2]
        rs_d = rs_a.detach() if torch.is_tensor(rs_a) else rs_a
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state() if hp.is_cuda else None

        def _restore():
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)

        shp = tuple(hp.shape[1:])
        v = self._core_gain_dir
        if (self.cfg.core_gain_direction != "power" or v is None or tuple(v.shape) != shp
                or v.device != hp.device):
            v = torch.randn(shp, device=hp.device, dtype=torch.float32)
            v = v / (v.norm() + 1e-6)
        hn = hp.float().flatten(1).norm(dim=1)                                   # [n]
        scale = (float(self.cfg.core_gain_eps) * hn).to(hp.dtype)
        d = v.to(hp.dtype).unsqueeze(0) * scale.view(-1, *([1] * (hp.dim() - 1)))
        try:
            _restore()
            f0, _ = core_step(hp, e_d, inj_d, ret_state=rs_d, iter_idx=t, attn_kw=akw)
            # Within-step power iterations (core_gain_power_iters): refine the direction at
            # THIS input, no grad (the reading, not the direction, carries the penalty's grad).
            for _ in range(int(self.cfg.core_gain_power_iters)):
                _restore()
                with torch.no_grad():
                    fk, _ = core_step(hp + d, e_d, inj_d, ret_state=rs_d, iter_idx=t,
                                      attn_kw=akw)
                    nv = (fk - f0).float().mean(0)
                    v = nv / (nv.norm() + 1e-6)
                    d = v.to(hp.dtype).unsqueeze(0) * scale.view(-1, *([1] * (hp.dim() - 1)))
            _restore()
            f1, _ = core_step(hp + d, e_d, inj_d, ret_state=rs_d, iter_idx=t, attn_kw=akw)
        finally:
            _restore()
        diff = (f1 - f0).float()
        num = diff.flatten(1).norm(dim=1)
        den = d.float().flatten(1).norm(dim=1) + 1e-6
        gain = num / den                                                         # [n]
        hinge = torch.relu(gain - float(self.cfg.core_gain_target))
        pen = lam * (hinge * hinge).mean()
        if self.cfg.core_gain_direction == "power":
            with torch.no_grad():
                nv = diff.mean(0)
                self._core_gain_dir = (nv / (nv.norm() + 1e-6)).detach()
        return {"gain": gain.detach().mean(), "gain_max": gain.detach().max(), "penalty": pen}

    def _apply_core_aux(self, out: dict) -> None:
        """Add the arc E10 loop terms stashed by ``_core_region`` to ``out`` and the loss."""
        aux = self._core_aux
        self._core_aux = None
        if not aux:
            return
        for k, v in aux.items():
            out[k] = v
        if any(k.endswith("_weighted") for k in aux):
            # The CE BEFORE any core-loop term is added, so a reader can recover the number
            # every arm is compared on exactly (not by subtracting in float afterwards).
            # `_mtp_apply` sets the same key first when it runs, hence setdefault.
            out.setdefault("ce_main", out["loss"])
        if "fp_weighted" in aux:
            out["loss"] = out["loss"] + aux["fp_weighted"]
        if "pass_res_weighted" in aux:
            out["loss"] = out["loss"] + aux["pass_res_weighted"]
        if "core_gain_weighted" in aux:
            out["loss"] = out["loss"] + aux["core_gain_weighted"]
        if "loopmtp_weighted" in aux:
            out["loss"] = out["loss"] + aux["loopmtp_weighted"]
        if "loopmtp_ponder_weighted" in aux:
            out["loss"] = out["loss"] + aux["loopmtp_ponder_weighted"]

    def _loopmtp_align(self, states: list[Tensor], labels: Tensor) -> Tensor:
        """LoopMTP Eq 12-13: the soft multi-token target on the loop's own iterates.

        ``states[k]`` is the carrier AFTER core iteration ``k + 1``. Iteration ``t``
        (1-based) is aligned to the DETACHED tied output embedding of the token ``t`` steps
        ahead of position ``i``::

            L_align^(t) = mean over valid i of  1 - cos(proj(x_i^(t)), sg[E_{u_{i+t}}])

        MORPH's ``labels[i]`` is already ``u_{i+1}``, so ``u_{i+t}`` is
        ``labels[i + t - 1]`` — the SAME shift ``_mtp_apply`` uses for its head ``j = t``.
        The tail ``t - 1`` positions of a row have no such token and are padded to -100;
        so is every position whose label was already -100 (row padding). The paper's
        ``1/(S - t)`` normaliser is exactly the count of those valid positions when the row
        carries no padding, and this form stays right when it does.

        Three things are load-bearing:

        * ``lm_weight().detach()`` — the memory rule (auxiliary heads must not train the
          tied embedding table). The paper's ``sg[E]`` says the same thing.
        * the stream reduction ``mean(dim=2)`` is the SAME one ``_readout`` applies, so the
          quantity scored is the single-stream state the coda's read-out sees.
        * ``self.loopmtp_proj`` is ONE shared projection, not one per iteration, so the
          iterates themselves must differ to satisfy T different targets.

        Returns the mean over the supervised iterations (Eq 13's ``1/(T-1)``), a scalar.
        """
        maps = self._loopmtp_align_maps(states, labels)
        if not maps:
            if bool(self.cfg.depth_fixed):
                raise RuntimeError(
                    "LoopMTP alignment has no supervised iteration: loopmtp_free_first is "
                    "on and the loop ran a single iteration. Use loopmtp_free_first=false "
                    "at T=1, or a depth >= 2.")
            # A Poisson batch whose every row drew depth 1: nothing to align this step.
            return states[0].new_zeros((), dtype=torch.float32)
        terms = [(m * v).sum() / v.sum().clamp(min=1).to(m.dtype) for _, m, v in maps]
        return torch.stack(terms).mean()

    def _loopmtp_align_maps(self, states: list[Tensor],
                            labels: Tensor) -> list[tuple[int, Tensor, Tensor]]:
        """The per-position terms :meth:`_loopmtp_align` averages, one entry per SUPERVISED
        iteration: ``(t, (1 - cos) map [B, S], valid mask [B, S])`` with ``t`` 1-based.

        ONE home for the indexing: the loss, the per-iteration read-out in
        ``lab/divergence/loopmtp_iteration_probe.py`` and the tests all read it, so an
        off-by-one in the horizon cannot exist in the loss and be absent from the probe.
        """
        if labels.ndim != 2:
            raise RuntimeError(
                "LoopMTP is undefined with 3-D TST bag labels: a bagged position carries "
                "s token targets and Eq 13 aligns the state to ONE embedding row.")
        w = self.embed.lm_weight().detach()                      # [V, d] — sg[E]
        start = 1 if bool(self.cfg.loopmtp_free_first) else 0
        out: list[tuple[int, Tensor, Tensor]] = []
        for k in range(start, len(states)):
            h = states[k]
            # PREFIX RULE: `states[k]` may hold only the first n_k rows of a depth-sorted
            # batch (the rows that ran iteration k + 1); `labels` is in the SAME order and
            # is cut to them, so a row is aligned on exactly its own realised horizons.
            # Every n_k == B (fixed depth, eval) leaves this a no-op.
            n_k = int(h.shape[0])
            labels_k = labels if n_k == int(labels.shape[0]) else labels[:n_k]
            if h.dim() == 4:                                     # HC carrier [B,S,n,C]
                h = h.mean(dim=2)
            if self.loopmtp_proj is not None:
                h = self.loopmtp_proj(h)
            # iteration t = k + 1 (1-based) targets u_{i+t} = labels[i + t - 1] = labels[i + k]
            lab = labels_k if k == 0 else F.pad(labels_k[:, k:], (0, k), value=-100)
            valid = (lab >= 0)
            tgt = F.embedding(lab.clamp(min=0), w)               # [B, S, d]
            cos = F.cosine_similarity(h.float(), tgt.float(), dim=-1)
            out.append((k + 1, 1.0 - cos, valid))
        return out

    @staticmethod
    def _loopmtp_ponder(gates: Tensor) -> Tensor:
        """LoopMTP's ponder regulariser: KL(per-iteration gate mass || uniform).

        ``gates`` is the ``[T, B*S]`` gate mass :meth:`_LoopMTPGate.forward` returns:
        ``gtilde_i^(t)`` averaged over the channels (and, on the HC carrier, the streams —
        the gate is per stream and per channel, and the paper's ``G`` is "the gate this
        position gave iteration t" averaged over everything that is not an iteration).
        Returns ``mean_i sum_t G_i(t) log(T * G_i(t))``, a scalar >= 0, zero iff the gate is
        uniform at every position.
        """
        if gates.dim() != 2:
            raise RuntimeError(
                f"LoopMTP ponder reads the [T, B*S] gate mass _LoopMTPGate returns, got "
                f"shape {tuple(gates.shape)}")
        # A finished row's later-iteration mass is exactly 0 (`_LoopMTPGate.forward`), so
        # the count of non-zero entries per position is that row's realised depth and the
        # uniform it is compared with is over THOSE passes. At a fixed depth every entry is
        # positive (softplus), the count is T everywhere and this is the original form.
        ran = gates > 0
        n_i = ran.sum(dim=0, keepdim=True).clamp(min=1).to(gates.dtype)
        gi = gates.clamp(min=1.0e-8)
        gi = gi / gi.sum(dim=0, keepdim=True)
        kl = gi * (gi * n_i).log()
        return torch.where(ran, kl, torch.zeros_like(kl)).sum(dim=0).mean()

    def _mtp_apply(self, out: dict, x: Tensor, labels: Tensor | None,
                   w_full: Tensor | None) -> None:
        """Multi-token prediction on the readout ``x`` (``[B, T, d]``; arc E8).

        Adds ``ce_mtp_j`` (j = 2..k, the CE of head j against labels shifted by j-1, tail
        -100), ``mtp_weighted`` (= mtp_weight * their sum, ALREADY added to ``out["loss"]``)
        and ``ce_main`` to ``out``; with ``labels is None`` (the eager generation / sweep
        path) it adds ``mtp_logits`` (a list of ``[B, T, V]``) instead. ``w_full`` given ⇒
        the fused chunked CE (training / kernel eval); None ⇒ full logits (eager).
        """
        if labels is None:
            if w_full is not None:
                return
            out["mtp_logits"] = [self.embed.attend(h(x)) for h in self.mtp]
            return
        if labels.ndim != 2:
            raise RuntimeError("model.mtp_heads > 1 is undefined with 3-D TST bag labels")
        out.setdefault("ce_main", out["loss"])
        total = None
        for j, head in enumerate(self.mtp, start=2):
            lab = F.pad(labels[:, j - 1:], (0, j - 1), value=-100)
            hx = head(x)
            if w_full is not None:
                ce_j = fused_linear_cross_entropy(
                    hx.reshape(-1, hx.shape[-1]), w_full, lab.reshape(-1),
                    ignore_index=-100, chunk_size=self.cfg.ce_chunk_size)
            else:
                ce_j = F.cross_entropy(
                    self.embed.attend(hx).reshape(-1, self.cfg.vocab_size),
                    lab.reshape(-1), ignore_index=-100)
            out[f"ce_mtp_{j}"] = ce_j
            total = ce_j if total is None else total + ce_j
        out["mtp_weighted"] = float(self.cfg.mtp_weight) * total
        out["loss"] = out["loss"] + out["mtp_weighted"]

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

    def _readout_per_stream(self, x: Tensor) -> Tensor:
        """``tul.mux_readout='full'``: the readout applied to EVERY Hyper-Connection stream,
        averaged afterwards. ``[B, S, n, C] -> [B, S, C]``.

        This IS "the tied head applied to each stream and the logits averaged". Both steps
        after the streams are linear in ``x`` — ``lm_mixer`` is a per-channel-group scale
        plus a bias-free ``nn.Linear``, and the head is ``z @ lm_weight().t()`` — so
        ``mean_n (z_n @ W')`` equals ``(mean_n z_n) @ W'`` exactly, and returning the mean of
        the per-stream readouts costs ONE ``[B, S, V]`` matmul rather than ``n`` of them.

        Why this definition and not "the head applied to the stream SUM": the sum is
        provably the same object as the mean. ``final_norm`` is an RMSNorm, which is
        scale-invariant, and everything between the streams and it is linear, so
        ``_readout(n * mean) == _readout(mean)`` to the last bit. The sum could not be a
        different arm. The per-stream form is also the one consistent with
        :meth:`TULSlots.prefix_project`, the reader that actually feeds the coda: it applies
        its shared ``[C, C]`` map to each stream separately and hands the coda all four,
        while ``_readout`` collapses them first. The only difference between the two
        readouts is therefore WHERE the RMS normalisation sits — one per stream, or one
        after the mean — which is exactly the quantity finding F2 measured: on
        ``slot-unpack-free`` the loop's update survives ``h.mean(dim=2)`` at 0.139 of its
        per-stream norm and the entry at 0.972, so the unweighted mean lets one stream's
        magnitude decide what the head sees.

        It does NOT undo an exact antipodal cancellation across streams (per-stream
        normalisation rescales, it does not re-phase), and it adds no parameters. The
        learned ``[n·C -> C]`` projection the audit names as the other fix does add them and
        is a different arm.
        """
        if x.dim() != 4:
            raise RuntimeError(
                "tul.mux_readout='full' needs a Hyper-Connection carrier [B, S, n, C]; "
                f"got a {x.dim()}-d tensor. Construction refuses this config on a model "
                "without a stream axis, so reaching here is a wiring bug.")
        return self.final_norm(self.lm_mixer(x)).mean(dim=2)

    def prelude_states(self, input_ids: Tensor, apply_input_norm: bool = True,
                       layout: "SlotLayout | None" = None) -> Tensor:
        """``[B, L, d_model]`` FEATURE READ-OUT after the prelude. Adds no behaviour.

        Written for TUL-FM P1 (``lab/tulfm/``), which trains a separate planner on a
        FROZEN backbone and needs the backbone's states over the context. Nothing in the
        training or inference path calls it, so every existing path stays bit-identical:
        this is a new public entry point that reuses :meth:`_front_region` (``layout is
        None``) or :meth:`_tul_front` (``layout`` given), the same boundary norm
        ``_core_region`` / ``_tul_core`` applies, and the same stream reduction
        :meth:`_readout` applies.

        With ``apply_input_norm=True`` on an ``n_core == 0`` model (arm A3), the returned
        tensor is EXACTLY what the coda consumes: ``_core_region`` reduces to
        ``self.input_norm(x)`` there. On a model with a core it is the state the core
        loop starts from, before any iteration.

        ``layout`` (TUL-FM P1's TG-restrict backbones): runs the TUL prelude over the
        packed ``[B, L_total]`` sequence (token AND slot positions) instead of the plain
        one. Requires a model built with ``MORPHConfig(tul=...)``. Under ``tg_restrict``
        this builds the SAME ``tg_allow`` / ``tg_slot_mask`` / GLA-reset-mask
        ``_forward_tul`` builds, for the same reason :meth:`_forward_tul` builds them
        ONCE per forward rather than never: a bare ``_tul_front(..., attn_kwargs=None)``
        call would silently hand ``_tg_slot_attention`` a ``None`` slot mask, which its
        own docstring defines as "every position is a slot" — a WRONG answer (every token
        position would be treated as attendable in the compressed branch), not a raise.
        Returns the FULL packed-position tensor (input_norm(prelude) at every position,
        exactly :meth:`_tul_core`'s own ``xn = self.input_norm(x)``, spec's own "tokens
        keep it, the n_core==0 seed path"); the caller restricts to TOKEN positions
        (``~layout.slot_mask``) itself, exactly as the ``layout is None`` path is already
        a token-only tensor with no slot positions to strip.

        The HC carrier is reduced by the mean over streams — the same scale-preserving
        readout :meth:`_readout` uses — so the result is single-stream ``[B, L, d_model]``
        whatever ``hc_streams`` is.

        Raises in ``train()`` mode: embed/MLP dropout would make the "frozen features"
        stochastic, and a silently-noisy feature is worse than a missing one.
        """
        if self._span_mask:
            raise NotImplementedError(
                "prelude_states does not build model.span_mask's span masks, so it would\n"
                "run the prelude UNRESTRICTED on a budget arm. Use forward().")
        if self.training:
            raise RuntimeError(
                "prelude_states() is a frozen-feature read-out and must run in eval mode "
                "(dropout would make the features stochastic). Call model.eval() first.")
        if layout is not None:
            if self.tul is None:
                raise RuntimeError(
                    "prelude_states(layout=...) requires a model built with "
                    "MORPHConfig(tul=...); this model has no TUL parameters "
                    "(E_slot / E_mask / W_prefix).")
            tc = self.cfg.tul
            if layout.prefix_k != tc.prefix_k:
                raise ValueError(
                    f"layout prefix_k {layout.prefix_k} != model {tc.prefix_k}")
            tg_attn_kwargs = tg_reset = None
            if self._tg_strict:
                # The SAME prelude relation `_forward_tul` builds (tul.tg_geometry), for
                # the same reason: a probe that read an unrestricted prelude off a strict
                # arm would report states the run never computed.
                _seg = tg_segment_ids(layout)
                _pre_allow = tg_strict_allow(layout, "prelude")
                tg_attn_kwargs = {"tg_allow": _pre_allow,
                                  "tg_slot_mask": layout.slot_mask,
                                  "tg_comp_allow": _pre_allow, "tg_seg": _seg}
                tg_reset = tg_reset_from_ids(_seg)
            elif self._tg_restrict and tc.tg_restrict_scope == "all":
                # scope "coda": the prelude is global, exactly as in _forward_tul
                tg_allow = tg_allow_mask(layout, soft_prev_span=tc.tg_soft_prev_span)
                tg_attn_kwargs = {"tg_allow": tg_allow, "tg_slot_mask": layout.slot_mask}
                if tc.tg_span_comp:
                    # E-SAC: per-span pooled compressed branch (attention.py
                    # _tg_span_attention). Built once per forward like tg_allow.
                    _tok_sel = ~layout.slot_mask
                    tg_attn_kwargs["tg_span"] = {
                        "bag_id": layout.bag_id, "token_sel": _tok_sel,
                        "span_end": boundary_token_index(
                            layout.bag_id, _tok_sel, layout.max_slots)}
                tg_reset = tg_reset_mask(layout)
            x, _x0, _bigram = self._tul_front(input_ids, layout,
                                              attn_kwargs=tg_attn_kwargs,
                                              ret_reset_mask=tg_reset)
        else:
            x, _x0, _bigram = self._front_region(input_ids)
        if apply_input_norm:
            x = self.input_norm(x)
        if self._is_hc:
            x = x.mean(dim=2)
        return x

    def _core_region(self, x: Tensor, x0: Tensor, bigram_emb,
                     input_ids: Tensor | None = None,
                     attn_kwargs: dict | None = None,
                     jac_active: Tensor | None = None,
                     labels: Tensor | None = None) -> Tensor:
        """CORE region: input_norm → the Poisson-depth core loop → the looped carrier.

        Pure code motion out of ``_forward_single`` (the ``_front_region`` /
        ``_back_region`` precedent): the ops and their order are IDENTICAL to the old
        inline block, so every non-TUL path is bit-identical. It is a method so the paid
        TUL loop (``tokens_through_core``, docs/tul-paid-loop-recipe.md) can run the SAME
        per-sample core over a sequence that happens to contain slot positions, instead
        of forking a second implementation of the loop. ``jac_active`` (``[B, L]`` bool,
        optional) is read ONLY by the Jacobian probe capture: positions that carry no
        information (the packed row's tail-pad slot positions) are excluded from the
        probe's active set. None → every position is active, and nothing else changes.

        ``labels`` is read ONLY by LoopMTP's Eq-13 alignment term (arXiv 2608.03624), and
        only in ``training``; every other caller leaves it None and the forward is
        untouched. It is a parameter rather than a stash because the term is a per-iteration
        loss on states that exist only inside this function."""
        B = x.shape[0]
        # LoopMTP: a Python-level constant set at build, so with both knobs off every
        # branch below traces out and the loop is bit-identical to the pre-LoopMTP tree.
        _lmtp = self._loopmtp_states
        _lm_states: list[Tensor] = []
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

            def _core_step(h_in, e_in, inj_terms, ret_state=None, iter_idx=0,
                           attn_kw=None):
                # Thin closure → the bound `_apply_core_step` method (single source of truth so
                # the σ_max probe / diagnostics exercise the EXACT training core map). Kept as a
                # closure so `checkpoint(_core_step, ...)` and the eager/no_grad call sites below
                # are unchanged; np_ (= cfg.n_prelude) is now recomputed inside the method.
                # ids/x0_terms/bg are None here: the injection is precomputed (inj_terms) and
                # threaded as a checkpoint input so the recompute reuses it.
                if _scse is None:
                    return self._apply_core_step(h_in, e_in, None, None, None,
                                                 ret_state=ret_state, iter_idx=iter_idx,
                                                 inj_terms=inj_terms, attn_kw=attn_kw)
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
                    ret_state=ret_state, iter_idx=iter_idx, inj_terms=None, source_free=True,
                    attn_kw=attn_kw)
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
                jac_s = None if jac_active is None else jac_active[perm]
                # Under SCSE `e_s` carries the ANCHOR, not the source: the loop's second
                # positional argument is what `_core_step` forwards to `recurrent_input`,
                # and SCSE's recurrence never sees `e` again after Delta_0 and h* are built.
                e_s = (h_star if _scse is not None else e)[perm]
                # ids_s / bg_s / x0_s gathers are gone: the injection is precomputed
                # (inj_core_terms) and only IT needs sorting into active-set order. This also
                # drops 3 gather kernels/step (input_ids, bigram, x0-stack) from the hot loop.
                inj_s = _inj_none if _scse is not None else inj_core_terms[:, perm]
                # A2s (tokens_through_core + tg_restrict) and the core-token auxiliary
                # (tul.core_token_aux): the per-sample TG masks must follow the SAME
                # active-set permutation as the carrier, or a sorted sample attends under
                # another sample's mask. Sorted ONCE here; sliced [:n_active] per iteration
                # below, exactly like h_s / inj_s. `attn_kwargs=None` (every non-TG caller)
                # leaves `tg_s` empty and the loop bit-identical.
                #
                # Every entry is a PER-SAMPLE tensor with batch as dim 0, which is what
                # makes `[perm]` and `[:n_active]` correct. A key outside `_CORE_TG_KEYS`
                # is not permutable that way (`tg_span` is a dict of three tensors, one of
                # them a derived index), so it RAISES rather than being dropped — a silently
                # ignored mask is an unrestricted core wearing a restricted arm's name.
                if attn_kwargs:
                    _bad_tg = sorted(set(attn_kwargs) - _CORE_TG_KEYS)
                    if _bad_tg:
                        raise NotImplementedError(
                            f"_core_region does not thread {_bad_tg} through the active-set "
                            f"sort; it handles {sorted(_CORE_TG_KEYS)}. Raises rather than "
                            f"running the core with the mask silently dropped.")
                    tg_s = {k: v[perm] for k, v in attn_kwargs.items() if v is not None}
                else:
                    tg_s = {}

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
            track_ret = (self._core_has_retention
                         and self.cfg.retention_carry_mode == "acausal_final")
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
            # Arc E10 loss terms (training only; 0 = off, and every line below traces out).
            self._core_aux = None
            _fp_lam = float(self.cfg.core_fixed_point_lambda) if self.training else 0.0
            _cg_lam = float(self.cfg.core_gain_lambda) if self.training else 0.0
            _fp_terms: list[Tensor] = []
            _cg = None
            _t_cg = -1
            if _cg_lam > 0.0 and n_grad_iters > 0:
                if _scse is not None:
                    raise RuntimeError("model.core_gain_lambda > 0 is not defined under SCSE")
                _rs_cpu = torch.get_rng_state()
                _t_cg = n_nograd + int(torch.randint(n_grad_iters, (1,)).item())
                torch.set_rng_state(_rs_cpu)
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
                akw = {k: v[:n_active] for k, v in tg_s.items()} or None
                # Jacobian probe capture — see the twin in `_tul_core`. None by default,
                # so this branch traces out and the forward stays bit-identical.
                if self._jac_capture is not None:
                    self._jac_capture.append({
                        "h": h_a.detach(), "e": args[1].detach(), "inj": args[2].detach(),
                        "ret_state": None if rs_a is None else rs_a.detach(),
                        "iter_idx": t,
                        "active": (h_a.new_ones(h_a.shape[:2], dtype=torch.bool)
                                   if jac_s is None else jac_s[:n_active]),
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
                        h_new, rs_new = _core_step(*args, ret_state=rs_a, iter_idx=t,
                                                   attn_kw=akw)
                elif do_ckpt:
                    h_new, rs_new = checkpoint(_core_step, *args, ret_state=rs_a, iter_idx=t,
                                               attn_kw=akw, use_reentrant=False)
                else:
                    # eval, OR a grad-iter we chose not to checkpoint (activations retained).
                    h_new, rs_new = _core_step(*args, ret_state=rs_a, iter_idx=t,
                                               attn_kw=akw)

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

                if _lmtp:
                    # x^(t+1) of Eq 9/13, after the gain governor (there is none on a
                    # LoopMTP arm) and before the frozen-suffix concat. `h_new` holds the
                    # rows still active at this iteration, which are the FIRST n_active of
                    # the depth-sorted batch (the prefix rule): at a fixed depth that is
                    # the whole batch; under the Poisson draw a row appears in exactly its
                    # own d_i states. Kept in SORTED order; the gate, the alignment maps
                    # and the ponder term all read that order and the exit un-permutes once.
                    _lm_states.append(h_new)

                if _fp_lam > 0.0 and t >= n_nograd:
                    # Samples that FINISH at this iteration are the tail of the active prefix
                    # (sorted by depth, descending): active now, not at t+1.
                    n_next = active_counts[t + 1] if t + 1 < total_iters else 0
                    if n_next < n_active:
                        _fn = h_new[n_next:n_active].float().flatten(1)
                        _fo = h_a[n_next:n_active].float().flatten(1)
                        # Under SCSE the carrier is the deviation; the step is the same
                        # (h* is fixed) but the denominator is the ABSOLUTE state h* + Delta,
                        # so the term is the same quantity as on the plain carrier.
                        _fd = _fn if _scse is None else \
                            _fn + e_s[n_next:n_active].float().flatten(1)
                        _fp_terms.append((_fn - _fo).pow(2).sum(1) / (_fd.pow(2).sum(1) + 1e-6))
                if _cg_lam > 0.0 and t == _t_cg:
                    _cg = self._core_gain_penalty(_core_step, args, rs_a, t, akw, _cg_lam)

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
            if _fp_terms or _cg is not None:
                aux: dict = {}
                if _fp_terms:
                    _fp = torch.cat(_fp_terms).mean()
                    aux["fixed_point"] = _fp.detach()
                    aux["fp_weighted"] = _fp_lam * _fp
                if _cg is not None:
                    aux["core_gain_est"] = _cg["gain"]
                    aux["core_gain_max"] = _cg["gain_max"]
                    aux["core_gain_weighted"] = _cg["penalty"]
                self._core_aux = aux

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

            if _lmtp and _lm_states:
                # `_lm_states` is empty only at a FORCED depth of 0 (the plain readout's
                # `--depths 0,...` rung), where the loop never runs. There is no iterate to
                # gate and no iterate to align, so the aggregator is the identity on the
                # entry carrier — the same tensor the "last" read-out hands the coda there.
                # The states are prefixes of the depth-SORTED batch; the gate and the
                # alignment run in that order (each is row-wise, so the result is the same
                # per row) and only the outputs are put back in batch order: z once through
                # inv_perm, the [T, B*S] gate mass by rows. With every prefix the full
                # batch this is the pre-2026-09-14 arithmetic per row.
                _lm_full = all(int(st.shape[0]) == B for st in _lm_states)
                _lm_mass = None
                if self.loopmtp_gate is not None:
                    # Eq 9-11: the coda reads the gated mix of ALL iterates, not the last.
                    z_s, _lm_mass_s = self.loopmtp_gate(_lm_states)
                    x = z_s[inv_perm]
                    _S = int(z_s.shape[1])
                    _lm_mass = (_lm_mass_s.view(-1, B, _S)[:, inv_perm]
                                .reshape(_lm_mass_s.shape[0], B * _S))
                if getattr(self, "_loopmtp_capture", False):
                    # Read-out hook for lab/divergence/loopmtp_iteration_probe.py and the
                    # tests. An instance flag, default absent -> getattr False -> a
                    # Python-level no-op, so the training forward is unchanged. Do NOT set
                    # it during training: it pins T detached carriers per forward. The
                    # iterates are batch-ordered only when every row ran every iteration
                    # (eval, or a fixed depth); a partial Poisson batch stores None for
                    # them and keeps the batch-ordered gate mass, which is defined per row.
                    self._loopmtp_iterates = ([st[inv_perm].detach() for st in _lm_states]
                                              if _lm_full else None)
                    self._loopmtp_gate_mass = (None if _lm_mass is None
                                               else _lm_mass.detach())
                if self.training and labels is not None:
                    # Both terms are TRAINING ONLY (the `core_fixed_point_lambda`
                    # precedent), so val CE stays the plain next-token number every arm in
                    # the ladder is compared on.
                    _aux_lm = self._core_aux if self._core_aux is not None else {}
                    _lm_lam = float(self.cfg.loopmtp_weight)
                    if _lm_lam > 0.0:
                        # Labels in the states' (sorted) order; the maps cut them to each
                        # iteration's prefix.
                        _lm_al = self._loopmtp_align(_lm_states,
                                                     labels if _lm_full else labels[perm])
                        _aux_lm["loopmtp_align"] = _lm_al.detach()
                        _aux_lm["loopmtp_weighted"] = _lm_lam * _lm_al
                    if _lm_mass is not None:
                        # Always computed on a gated arm, weighted into the loss only when
                        # lambda_ponder > 0. It is the gate-collapse instrument: 0 means a
                        # uniform gate, log(T) means the aggregator picked one iterate, and
                        # without it a gated run has no live signal that the aggregator
                        # threw the loop away.
                        _lm_pd = self._loopmtp_ponder(_lm_mass)
                        _aux_lm["loopmtp_ponder"] = _lm_pd.detach()
                        _lm_plam = float(self.cfg.loopmtp_ponder_weight)
                        if _lm_plam > 0.0:
                            _aux_lm["loopmtp_ponder_weighted"] = _lm_plam * _lm_pd
                    if _aux_lm:
                        self._core_aux = _aux_lm
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
        fixed = int(tc.slot_depth_fixed)
        if fixed > 0:
            # The k-fixed panel (2026-09-07): every valid slot loops exactly `fixed` times,
            # in training AND at eval, so the forced-depth sweep reads a model that never
            # saw another depth. bptt_depth < fixed truncates uniformly (Parcae's form).
            d = torch.full(shape, fixed, device=device, dtype=torch.long)
        elif self.training:
            d = torch.poisson(torch.full(shape, float(mean_d), device=device)).long()
            d = d.clamp(min=1, max=max_d)
        else:
            d = torch.full(shape, int(mean_d), device=device, dtype=torch.long)
        return self._pad_slot_depths(d, layout)

    @staticmethod
    def _pad_slot_depths(d: Tensor, layout: SlotLayout) -> Tensor:
        """Pad slots loop exactly once. ONE home for the rule both depth sources obey."""
        return torch.where(layout.slot_valid, d, torch.ones_like(d))

    def _slot_depth_override(self, layout: SlotLayout, slot_depths: Tensor,
                             device) -> Tensor:
        """``[B, max_slots]`` EVAL-ONLY per-slot depth table, validated.

        The `slot_layout` rule: a per-forward DATA argument, never a config knob and never
        a training path. `None` at the call site is a Python-level branch, so a model that
        never passes one traces the graph from before this existed
        (tests/test_slot_depth_isolation.py pins the loss bit-for-bit).

        It OVERRIDES `tul.slot_depth_fixed` and `tul.slot_mean_depth` — that is its job:
        `lab/divergence/slot_depth_isolation.py` needs slot 7 at depth 1 while every other
        slot runs at 6, which no scalar knob can say. The pad rule is unchanged (a pad
        slot loops once whatever the table says), and the value range is the same one
        `_sample_slot_depths` clamps the Poisson draw to, but here it RAISES instead of
        clamping: a silently clamped instrument reads a depth it did not ask for.
        """
        if self.training:
            raise RuntimeError(
                "slot_depths is EVAL ONLY: training draws the per-slot Poisson depth, and "
                "forcing it would change the map the optimiser sees. Call under "
                "model.eval().")
        shape = layout.slot_index.shape
        if tuple(slot_depths.shape) != tuple(shape):
            raise ValueError(
                f"slot_depths must be [B, max_slots] = {tuple(shape)}, got "
                f"{tuple(slot_depths.shape)}")
        if slot_depths.dtype not in (torch.long, torch.int32, torch.int16, torch.int8):
            raise ValueError(f"slot_depths must be an integer tensor, got "
                             f"{slot_depths.dtype}")
        d = slot_depths.to(device=device, dtype=torch.long)
        max_d = int(self.cfg.tul.slot_max_depth or self.cfg.max_depth)
        lo, hi = int(d.min().item()), int(d.max().item())
        if lo < 1 or hi > max_d:
            raise ValueError(
                f"slot_depths must lie in [1, slot_max_depth={max_d}], got [{lo}, {hi}]. "
                "A depth outside the trained range is not a measurement of this model.")
        return self._pad_slot_depths(d, layout)

    def _loop_cot_ref_hook(self, g: Tensor) -> None:
        """Record the per-row norm of the cotangent arriving at the slot loop's EXIT state.

        Registered on the loop's final carrier when `slot_cot_clip > 0`. Autograd reaches
        this tensor before any iteration's output, so the reference is in place when the
        iteration hooks fire. Returns None: the exit cotangent is never clipped.
        """
        self._loop_cot_ref = g.detach().float().flatten(1).norm(dim=1)

    def _loop_cot_hook(self, t: int, record: bool, g: Tensor) -> Tensor | None:
        """The cotangent arriving at slot-loop iteration `t`'s output, in the backward.

        Two jobs, both per step:
          * `record` (the onset-capture probe, `_probe_cot`): the norm goes to
            `_loop_cot[t]`, read by the trainer's pre-clip probe as `loop/cot_norm_t{t}`.
            Recording alone returns None, so autograd keeps the gradient it computed
            (tests/test_onset_capture.py proves the grads are bit-identical).
          * clip-through-time (`cfg.slot_cot_clip` > 0): the gradient is rescaled per row
            to at most `slot_cot_clip` times that row's exit cotangent, and the post-clip
            norm and the fraction of rows the clip shrank go to `_loop_cot_post[t]` /
            `_loop_cot_bind[t]` when recording. A row the cap does not bind is multiplied
            by exactly 1.0, so an unbinding cap is bit-identical to no cap
            (tests/test_slot_cot_clip.py).
        """
        if record:
            self._loop_cot[int(t)] = g.detach().float().norm()
        r = float(self.cfg.slot_cot_clip)
        if r <= 0.0:
            return None
        ref = self._loop_cot_ref
        if ref is None:
            # The exit state did not receive a gradient (no loss reached the loop's
            # output). There is nothing to clip against; leave the gradient alone.
            return None
        rows = g.detach().float().flatten(1).norm(dim=1)
        scale = torch.clamp(r * ref / (rows + 1e-12), max=1.0)
        if record:
            self._loop_cot_rows[int(t)] = rows
            self._loop_cot_post[int(t)] = (rows * scale).norm()
            self._loop_cot_bind[int(t)] = (scale < 1.0).float().mean()
        return g * scale.view(-1, *([1] * (g.dim() - 1))).to(g.dtype)

    def _tul_core(self, x: Tensor, x0: Tensor, bigram_emb, layout: SlotLayout,
                  halt: bool = False, input_ids: Tensor | None = None,
                  slot_depths: Tensor | None = None):
        """Gather slots → masked per-slot depth loop → looped states (spec §3.3).

        Returns ``(xn, h_slots, depths, g_traj, db_traj, gain_reg, mep_keep)``:
        ``xn = input_norm(prelude)`` for the
        whole carrier (token positions keep it — the ``n_core == 0`` seed path, BLT
        Eq. 9), ``h_slots`` ``[B, max_slots, …]`` the looped state of each slot, the
        realised per-slot ``depths``, and — when the gate is built — ``g_traj``
        ``[B, max_slots, T]``, the gate output after EVERY iteration
        (docs/tul-gate-spec.md §4). ``g_traj`` is a RETURN VALUE, never a side channel:
        the ``ret_capture`` lesson is that a side channel is not checkpoint-safe. So is
        ``mep_keep`` (``tul.mux_every_pass`` and ``tul.mux_stage_all``, which share one
        trajectory and one set of masks; ``None`` everywhere else): the per-pass
        supervision mask belongs with the trajectory it indexes, and rebuilding it in the
        caller would duplicate the conditions (``progressive_p``, ``self.training``) that
        decide it.

        ``input_ids`` is read by ONE mechanism, ``tul.grad_pass``: the own-span loss it
        differentiates needs the row's token ids. Keyword, defaulting to ``None``, because
        every other caller (the probes, the tests, ``tul_forward_ablated``) passes four
        positional arguments and must stay unchanged; ``grad_pass`` with ``None`` RAISES
        rather than silently running the loop with the feature switched off.

        ``slot_depths`` ``[B, max_slots]`` is the EVAL-ONLY per-slot depth table
        (:meth:`_slot_depth_override`): it replaces the Poisson draw, ``slot_mean_depth``
        and ``slot_depth_fixed` for THIS forward only. ``None`` — every trainer call and
        every existing probe — is a Python-level branch that traces the graph from before
        this parameter existed. It exists because
        ``lab/divergence/slot_depth_isolation.py`` asks what ONE slot's passes are worth,
        which no scalar depth knob can express.

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
        # ── THE THOUGHT REGISTER (tul.slot_cells) ─────────────────────────────────
        # M == 1 — every model before this key — binds `layout` to itself and `_m_cells`
        # to 1, so every branch below traces exactly the graph it traced before.
        #
        # At M > 1 the compact sequence is S*M CELLS, slot-major (index s*M + i), and
        # `layout` is REBOUND for the rest of this method to a CELL-LEVEL view: the same
        # `slot_mask` / `bag_id` (they index the packed row and are unchanged), with
        # `slot_index` holding each cell's own row position and `slot_valid` expanded. That
        # rebinding is the whole change — every `layout.slot_valid` mask, the gain hinge,
        # the fixed-point term, the probes and the per-pass bookkeeping below then run on
        # cells with no further edit, which is the only version of this that a reviewer can
        # check. The mechanisms that would read ONE state per slot (the chain, the reread,
        # the energy feature, the per-pass targets) are refused at construction, so a
        # cell-level layout never reaches them.
        _m_cells = int(self.cfg.tul.slot_cells)
        _n_slots = layout.slot_index.shape[1]
        _layout_slots = layout          # the PER-SLOT view, kept for the register and the
                                        # per-slot depth draw; `layout` becomes per-CELL
        if _m_cells > 1:
            _off = torch.arange(_m_cells, device=layout.slot_index.device)
            layout = SlotLayout(
                slot_mask=layout.slot_mask, bag_id=layout.bag_id,
                slot_index=(layout.slot_index.unsqueeze(-1) + _off).reshape(
                    B, _n_slots * _m_cells),
                slot_valid=layout.slot_valid.repeat_interleave(_m_cells, dim=1),
                prefix_k=layout.prefix_k)
        gidx, gvalid = layout.slot_index, layout.slot_valid

        xn = self.input_norm(x)
        e = gather_valid(xn, gidx, gvalid)                            # [B, S, n, C]
        if self.tul_register is not None:
            # The seed pull-apart, added to the gathered prelude state BEFORE `core_init`.
            # Single-stream (the register pools token states, which have no stream axis of
            # their own) and broadcast into the Hyper-Connection carrier the way every
            # other injection is. `W_o` is zero-init, so this line is an exact no-op at
            # step 0 and the arm starts at its ruler.
            _reg = self.tul_register(xn.mean(dim=2) if self._is_hc else xn, _layout_slots)
            e = e + (_reg.unsqueeze(2) if self._is_hc else _reg).to(e.dtype)

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
            return xn, e, depths, None, None, None, None

        _scse = self.scse           # Python-level constant → every branch below traces out
        # ── DB-shaped loop (arm L3): detached carry, per-iteration local supervision ──
        # See TULConfig.db_loop. A Python-level constant read once, so every branch on it
        # below traces out and the db_loop=False graph is unchanged.
        _db = bool(self.cfg.tul.db_loop)
        if _db and _scse is not None:
            raise NotImplementedError(
                "tul.db_loop under SCSE is not defined: the carry is the DEVIATION and "
                "detaching it detaches h* reconstruction. Build it when an arm needs it.")
        # Staged targets (TULConfig.mux_stage_own_iters): the same trajectory list, with the
        # carry LIVE. Under SCSE the carried state is the deviation, and a readout of it is
        # not a readout of the slot — the same reason db_loop raises.
        _stage = int(self.cfg.tul.mux_stage_own_iters) > 0
        if _stage and _scse is not None:
            raise NotImplementedError(
                "tul.mux_stage_own_iters under SCSE is not defined: the trajectory the "
                "stage supervises is the DEVIATION, not the slot state.")
        # Per-pass MUX (TULConfig.mux_every_pass): the SAME live-carry trajectory the staged
        # target uses, kept for EVERY pass, plus the per-pass keep mask the loss needs. A
        # Python-level constant, and TRAINING ONLY — an eval forward keeps the single
        # final-state term, which is what leaves `core_depth_sweep.py`'s forced-depth
        # `mux_local` column identical to the ruler's.
        # `tul.mux_stage_all` (arm slot-mnext-staged-all) needs exactly the same two things
        # — the live trajectory and the per-pass keep masks — for its own-span terms, so it
        # turns the SAME collection on rather than building a second one. It is refused
        # without `mux_stage_own_iters > 0`, so `_stage` (and the SCSE raise above) already
        # covers it.
        _stage_all = _stage and bool(self.cfg.tul.mux_stage_all)
        _mep_cfg = bool(self.cfg.tul.mux_every_pass)
        if _mep_cfg and _scse is not None:
            raise NotImplementedError(
                "tul.mux_every_pass under SCSE is not defined: the carried state is the "
                "DEVIATION, and a MUX readout of a deviation is not a readout of the slot "
                "(the same reason db_loop and mux_stage_own_iters raise here).")
        _mep = (_mep_cfg or _stage_all) and self.training
        # Gradient-conditioned passes (TULConfig.grad_pass). A Python-level constant read
        # once: `None` — every other model — traces the graph from before this existed.
        # ON at eval too: the feature is part of the MAP, so a forced-depth sweep must see
        # the same function the trainer ran.
        # The slot chain (tul.slot_chain): a Python-level constant read once — `None`,
        # every other model, traces the graph from before this existed. ON at eval too: the
        # chain is part of the MAP, so a forced-depth sweep must see the function the
        # trainer ran (the `grad_pass` rule).
        _chain = self.tul_chain
        _gp = self.tul_grad_pass
        if _gp is not None:
            if _scse is not None:
                raise NotImplementedError(
                    "tul.grad_pass under SCSE is not defined: the loop carries the "
                    "DEVIATION, and the gradient of an own-span readout of a deviation is "
                    "not the gradient at the slot state the loss is defined on (the same "
                    "reason db_loop and mux_stage_own_iters raise here).")
            if input_ids is None:
                raise RuntimeError(
                    "tul.grad_pass needs `input_ids` in _tul_core: the own-span loss it "
                    "differentiates is built from the row's token ids. This caller passed "
                    "none — pass `input_ids=` rather than running the loop with the "
                    "feature silently switched off.")
        # The arm's readouts. TRAINING and grad-enabled only, for the `_probe_write`
        # reason: a no-grad forward that runs between the backward and the trainer's read
        # must not overwrite the step's values with its own.
        _gp_write = _gp is not None and self.training and torch.is_grad_enabled()
        _gp_stats: dict = {}
        if _gp_write:
            self._loop_gradpass = None
        # The `disc` critic's context: the slot's PRELUDE-ENTRY state, mean over the
        # Hyper-Connection streams. Constant across passes by construction, so the critic
        # scores "how far has this state moved from where it began, and did that help"
        # rather than re-reading the span. None for every other energy.
        # The `critic` energy scores the same two inputs, for the same reason, so it takes
        # the same context — the slot's prelude-entry state, constant across passes, which
        # is what makes "how far has this state moved from where it began, and did that
        # help" a well-posed question and keeps both scorer inputs causal for the slot.
        _eg_ctx = None
        if isinstance(self.tul_egrad, (DiscEnergy, CriticEnergy)):
            _eg_ctx = (e.mean(dim=2) if self._is_hc else e).detach()
        # The per-pass residual bound (TULConfig.pass_residual_lambda). A Python-level
        # constant: 0.0 builds nothing and the graph is the one from before this existed.
        # Twin of `_fp_terms` below, and deliberately NOT the same term: the fixed-point
        # term charges each slot's LAST pass, this one charges EVERY gradient pass.
        _pr_lam = float(self.cfg.tul.pass_residual_lambda) if self.training else 0.0
        _pr_terms: list[Tensor] = []
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
        # Trajectory for the local losses: OUTER-graph states, one per iteration, returned
        # (never a side channel — the g_traj / ret_capture lesson). _db_traj[0] is the seed
        # state; entry t is the post-update state after iteration t-1.
        # The oracle-z teacher (tul.oracle_z) needs the SAME live-carry trajectory, so it
        # turns the SAME collection on rather than building a second one. Training only —
        # an eval forward keeps `db_traj` None here and the sweep reads the ruler's columns.
        _oz = bool(self.cfg.tul.oracle_z) and self.training and self.tul_spandec is not None
        # The per-pass planning target (tul.spandec_per_pass) reads the SAME trajectory the
        # oracle reads, for the same reason: it grades the state after every pass. Training
        # only, so an eval forward keeps `db_traj` None and the sweep reads ruler columns.
        _pp = (bool(self.cfg.tul.spandec_per_pass) and self.training
               and self.tul_spandec is not None)
        # The `critic` energy reads the SAME trajectory, and for the closest reason of any
        # of them: its label is a comparison of `_db_traj[t-1]` against `_db_traj[t]`
        # through a replay of the real coda. Training only, so an eval forward keeps
        # `db_traj` None and the forced-depth sweep pays nothing for it. The states are
        # detached at the label site, never here — the trajectory is the same live-carry
        # list every other per-pass reader uses.
        _cr = isinstance(self.tul_egrad, CriticEnergy) and self.training
        # `tul.prefix_source="trajectory"` reads the SAME live-carry trajectory, and is the
        # ONE reader of it that is NOT training-only: the cells it fills are what the coda
        # sees, so an eval forward — the forced-depth sweep, `worth_profile`,
        # `slot_z_optimize`, generation — must build the identical list or it would score a
        # different model from the one that trained. Under SCSE the carry is the DEVIATION,
        # so a per-pass cell would hold a deviation and not a slot state: refused at
        # construction (MORPHTransformer.__init__), asserted here.
        # `entry_exit` reads index 0 of the SAME list — the entry state `core_init(e)` the
        # information control hands the coda beside the exit — so it turns the same
        # collection on rather than stashing the entry somewhere else.
        _traj_src = self.cfg.tul.prefix_source in ("trajectory", "entry_exit")
        if _traj_src and _scse is not None:
            raise NotImplementedError(
                "tul.prefix_source='trajectory'/'entry_exit' under SCSE is not defined: "
                "the loop carries "
                "the DEVIATION, so cell k would hold Delta_k and not the slot state the "
                "coda is meant to read (the db_loop / mux_stage_own_iters precedent).")
        # LoopMTP's horizon alignment (tul.horizon_weight) reads the SAME trajectory, for
        # the same reason spandec_per_pass/oracle_z do: it grades the state after every
        # pass. Training only, so an eval forward keeps `db_traj` None and the sweep reads
        # ruler columns. The gated readout (tul.pass_readout="gated") is NOT training
        # only — it replaces `h_slots` in the real forward — so it turns the SAME
        # collection on unconditionally, the `_traj_src` precedent.
        _hz_loss = bool(self.cfg.tul.horizon_weight > 0.0) and self.training
        _hz_gate = self.cfg.tul.pass_readout == "gated"
        # The code target (tul.code_target) reads the SAME trajectory for its per-pass
        # cosine to the frozen code (`code_target_cos_l{t}`, no_grad, train only): the
        # depth instrument of that arm. An eval forward keeps `db_traj` None and the
        # forced-depth sweep reads the ruler's columns.
        _ct = self.tul_code_proj is not None and self.training
        # LXTUL's fan (tul.fan_k) reads the SAME trajectory, and — like
        # `prefix_source='trajectory'` and unlike every training-only reader above — it
        # reads it at EVAL too. Two reasons: the repulsion term is charged per PASS, and
        # the arm's headline instruments (`fan/stream_cos_t{t}`, `fan/stream_rank_t{t}`)
        # are the collapse SHAPE across passes, which a val forward has to be able to see
        # or the arm has no read at all (the standing
        # `depth-summing-instruments-hide-pass-trades` rule). SCSE is already refused at
        # `slot_cells > 1`, so no deviation carry can reach this list.
        _fan = self.tul_fan is not None
        if (_hz_loss or _hz_gate) and _scse is not None:
            raise NotImplementedError(
                "tul.horizon_weight>0 / tul.pass_readout='gated' under SCSE is not "
                "defined: the loop carries the DEVIATION, so db_traj[t] would hold "
                "Delta_t and not the slot state the horizon target / gate need to read "
                "(the prefix_source='trajectory' precedent).")
        _db_traj: list[Tensor] | None = (
            [h] if (_db or _stage or _mep or _oz or _pp or _cr or _traj_src
                    or _hz_loss or _hz_gate or _ct or _fan) else None)
        # Per-pass MUX: entry t-1 is the mask for `_db_traj[t]` — the slots whose realised
        # depth REACHES pass t and whose pass t carries gradient (a progressive prefix pass
        # is excluded: it is detached, so a term there would train nothing and still be
        # averaged into the loss). None everywhere else, which is the signal `_forward_tul`
        # branches on.
        _mep_keep: list[Tensor] | None = [] if _mep else None
        # slot_state_renorm: the per-slot norm the state ENTERED with is the norm it keeps.
        # Detached (a constant target); pads enter at 0 and stay there.
        _renorm = bool(self.cfg.slot_state_renorm)
        _n0 = h.detach().flatten(2).float().norm(dim=2) if _renorm else None      # [B, S]
        _gain_lambda = float(self.cfg.slot_gain_lambda)
        _gain_on = _gain_lambda > 0.0 and torch.is_grad_enabled() and self.training
        # The terminal fixed-point term (base.yaml `core_fixed_point_lambda`, the twin of
        # `_core_region`'s): ||h_T - h_{T-1}||^2 / ||h_T||^2 on every VALID slot at its LAST
        # iteration, grad iterations only, training only. Stashed in `_core_aux` and consumed
        # by `_forward_tul` exactly as the paid loop's is. Under SCSE the denominator is the
        # absolute state h* + Delta. `halt` (eval only) never reaches it.
        _fp_lam = float(self.cfg.core_fixed_point_lambda) if self.training else 0.0
        _fp_terms: list[Tensor] = []
        _gain_reg: dict | None = None

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
            if slot_depths is not None:
                raise ValueError(
                    "halt=True and slot_depths are two answers to the same question: the "
                    "gate decides the depth under halt, so a forced table would be "
                    "ignored. Pass one or the other.")
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
            depths = (self._sample_slot_depths(layout, x.device) if slot_depths is None
                      else self._slot_depth_override(layout, slot_depths, x.device))
            if _m_cells > 1:
                # ONE depth per SPAN, not per cell: the M cells of a slot are one register
                # and they iterate together. The draw above runs on the S*M axis and cell
                # 0's value is taken for the whole slot, so the arm consumes more of the RNG
                # stream than a per-slot draw would — which costs nothing here (the register
                # is a new arm with no bit-identity partner) and keeps ONE sampler in the
                # tree. A forced table must already be constant within a slot; the
                # instruments that build one (`slot_depth_isolation.py`) work per SLOT.
                _dv = depths.view(B, _n_slots, _m_cells)
                if slot_depths is not None and not bool((_dv == _dv[:, :, :1]).all()):
                    raise ValueError(
                        "slot_depths differs across the M cells of a slot; the register "
                        "loops a slot's cells together, so the table must be constant "
                        "within a slot. Raises rather than silently using cell 0's value.")
                depths = _dv[:, :, :1].expand(B, _n_slots, _m_cells).reshape(
                    B, _n_slots * _m_cells).contiguous()
            total_iters = int(depths.max().item())
        # db_loop: the truncated-BPTT window is meaningless (no gradient crosses an
        # iteration boundary by construction), and a no_grad iteration would silently
        # drop that iteration's LOCAL loss — so every iteration carries grad.
        n_nograd = 0 if _db else max(0, total_iters - self.cfg.bptt_depth)
        n_grad_iters = total_iters - n_nograd
        # ── progressive loss (TULConfig.progressive_p; Bansal et al. 2022) ──────────
        # A PER-SLOT no-grad prefix on top of the global truncated-BPTT window. `_prog` is a
        # Python-level constant (p, self.training and grad-enabled are all known at trace
        # time), so at p = 0 — every other arm — nothing below this exists and the graph is
        # the one from before the knob. The draw consumes the global RNG stream on purpose:
        # it is a training mechanism, not an instrument, and it must vary per step.
        _prog_p = float(self.cfg.tul.progressive_p)
        _prog = _prog_p > 0.0 and self.training and torch.is_grad_enabled() and not halt
        _pk = None
        if _prog:
            _sel = torch.rand(depths.shape, device=depths.device) < _prog_p
            # k uniform in [1, T_i - 1]: floor(u * (T_i - 1)) is 0 .. T_i - 2, so k never
            # reaches T_i and a slot's LAST pass always carries gradient (the terminal
            # fixed-point term and the exit state therefore always sit on a grad pass).
            _u = torch.rand(depths.shape, device=depths.device, dtype=torch.float32)
            _pk = 1 + (_u * (depths - 1).clamp(min=1).float()).long()
            _pk = torch.where(_sel & (depths >= 2) & layout.slot_valid,
                              _pk, torch.zeros_like(_pk))
            # Logged by the trainer (`loop/prog_*`); 0-dim GPU tensors, no host sync here.
            _valid_f = layout.slot_valid.float()
            _n_valid = _valid_f.sum().clamp(min=1.0)
            self._loop_prog = {
                "prog_frac": ((_pk > 0).float() * _valid_f).sum() / _n_valid,
                "prog_nograd_passes": (_pk.float() * _valid_f).sum() / _n_valid,
                "prog_grad_depth": ((depths - _pk).float() * _valid_f).sum() / _n_valid,
            }
            # The draw itself, [B, S]: the ONE place a test or a probe can read WHICH
            # passes were cut. Kept out of `_loop_prog` because the trainer reduces every
            # entry of that dict to a float.
            self._loop_prog_k = _pk.detach()
        # The regularised iteration: one grad iteration per step, drawn from the global
        # stream and the stream put back, so the draw is free of side effects on the run.
        _t_gain = -1
        _gain_all = bool(self.cfg.slot_gain_all_iters)
        if _gain_on and n_grad_iters > 0 and not _gain_all:
            _rs = torch.get_rng_state()
            _t_gain = n_nograd + int(torch.randint(n_grad_iters, (1,)).item())
            torch.set_rng_state(_rs)
        _gain_terms: list[dict] = []
        _ck = self.cfg.ckpt_grad_iters
        n_ckpt = n_grad_iters if _ck < 0 else max(0, min(_ck, n_grad_iters))

        track_ret = (self._core_has_retention
                         and self.cfg.retention_carry_mode == "acausal_final")
        if track_ret:
            _rh = self.cfg.retention_heads or self.cfg.n_heads
            _rdh = self.cfg.d_model // _rh
            ret_state = h.new_zeros(h.shape[0], _rh, _rdh, _rdh, dtype=torch.float32)
        else:
            ret_state = None

        # The reread (tul.reread): K/V from the frozen prelude token states, built ONCE;
        # every pass starts by adding the slot's read of them to its state. Inside
        # `_core_step` on purpose: the hinge's gain probe and the checkpointed step both
        # see the map WITH the read.
        _rr = self.tul_reread
        if _rr is not None:
            if _scse is not None or _db:
                raise NotImplementedError(
                    "tul.reread under SCSE / tul.db_loop is not defined (the loop state "
                    "there is a deviation, not the slot state the read would query).")
            _rr_k, _rr_v, _rr_allow = _rr.prepare(
                xn.mean(dim=2) if self._is_hc else xn, layout, self.cfg.tul.reread_scope)

        # ── depth as reach (tul.loop_reach) ───────────────────────────────────
        # A Python-level constant read once: 0 — every model before this key — builds no
        # mask and hands `_apply_core_step` the `attn_kw=None` it has always had, so the
        # graph is the one from before this existed.
        #
        # THE BUDGET IS PER PASS, not per layer, and that is the whole point: one pass is
        # ONE Jacobi step of `h_{t+1}[k] = f(h_t[k-w .. k])`, so `f` may read its
        # neighbours ONCE. The n_core layers are the internals of `f`. Applying reach w at
        # every layer would carry `w * n_core` per pass (measured on the fixture: reach 2
        # over two core layers moved a slot four cells away at pass 1), and the law
        # "a slot m spans back first moves at pass ceil(m/w)" would be false.
        #
        # So EVERY cross-cell route is spent in core layer 0 and closed in layers 1..n-1:
        #   layer 0     window branch `tg_allow` = causal AND j >= i - w;
        #               compressed branch the SAME relation through `tg_comp_allow` (under
        #               tg_restrict it is `_tg_slot_attention`'s DENSE form at the compact
        #               shape, a full causal route and not a pooled one);
        #   layers 1..  the same two relations at reach 0 — a cell reads ITSELF only. The
        #               window branch's XSA excludes the self token, so its row is empty
        #               there and `out_win` is 0 (SDPA returns 0 for an all -inf row,
        #               morph/model/CLAUDE.md); the compressed branch keeps j == i, so the
        #               cell still sees its own value;
        #   every layer the CCA causal conv and its W_v_prev value shift reset PER CELL. A
        #               kernel-4 two-stage conv reaches six cells back per BLOCK, a
        #               distance no window relation can express, so it is cut rather than
        #               budgeted.
        # This is a large change to what the core's later layers ARE, and it is named here
        # and in the arm's config rather than discovered from a K-curve: the reach arm's
        # core mixes cells once per pass and is position-local for the rest of it.
        #
        # Rate cost, also named: the core's window branch and its CCA prologue leave the
        # fused path here (extra_mask and seg are eager-only), which `tg_scoped_kernels`
        # keeps fused on the strict control. At 64 cells the tensors are small, but it is a
        # real second difference and the smoke's tok/s decides.
        _reach = int(self.cfg.tul.loop_reach)
        _core_akw = None
        if _m_cells > 1:
            # ── the register's IN-LOOP relation ───────────────────────────────────
            # Cell i of slot k is ALLOWED every cell of slots < k (as today, and as
            # narrowed by `loop_reach`) AND every cell of its OWN slot k.
            #
            # IT IS DELIVERED AS `tg_relation`, NOT `tg_allow` (fixed 2026-09-13, before
            # any GPU step of any register arm). `tg_allow`/`tg_comp_allow` are ANDed into
            # an already-causal relation and can only NARROW, and this mask is a SUPERSET
            # of flattened causal, so handing it to them executed as plain flattened
            # causal at `loop_reach 0`: a cell never read a LATER cell of its own slot,
            # which is the opposite of a register. `tg_relation` REPLACES the branch's
            # causal term on both halves of the layer (window + compressed). Replacing is
            # safe across slots because the mask is block-causal there; within a slot it
            # widens to all M cells. `slot_cell_relation`'s docstring is the one home.
            #
            # The CCA conv and the value shift are left alone, unlike the `loop_reach`
            # arm's: they are CAUSAL on the flattened axis, so every position they reach
            # is a cell of the same or an earlier slot, which this relation already allows.
            # They are position-wise operators with no mask to widen — cell 0 gets no
            # sibling context THROUGH THEM, only through attention. A `tg_seg` reset here
            # would cut the conv at slot boundaries, a restriction today's 64-cell core
            # does not have either.
            if _reach > 0 and _scse is not None:
                raise NotImplementedError(
                    "tul.loop_reach with tul.slot_cells>1 under SCSE is not defined.")
            # ONE builder, shared with the think-once stack (`_tul_cond_apply`): a second
            # copy of this mask is how the two would drift into different relations.
            _mask0, _same = slot_cell_relation(_n_slots, _m_cells, x.device, _reach)
            # With a reach budget the later core layers are position-local exactly as the
            # reach arm's are, except that "position" is the SLOT: a cell keeps its own
            # slot's cells, which is what makes the register a register and not M
            # independent loops.
            _kw0 = {"tg_relation": _mask0}
            _kwr = {"tg_relation": _same}
            _core_akw = tuple([_kw0] + [(_kwr if _reach > 0 else _kw0)
                                        for _ in range(n_core - 1)])
        elif _reach > 0:
            if _scse is not None:
                raise NotImplementedError(
                    "tul.loop_reach under SCSE is not defined: the compact sequence there "
                    "carries the DEVIATION and the core is source-free, so 'slot k reads "
                    "slots k-w..k' names a state this loop does not hold.")
            _S = gidx.shape[1]
            _ri = torch.arange(_S, device=x.device)
            _ii, _jj = _ri.unsqueeze(1), _ri.unsqueeze(0)
            _seg_cell = _ri.unsqueeze(0).expand(B, _S)

            def _reach_kw(w: int) -> dict:
                m = ((_ii >= _jj) & (_jj >= _ii - w)).view(1, 1, _S, _S)
                return {"tg_allow": m, "tg_comp_allow": m, "tg_seg": _seg_cell}
            _core_akw = tuple([_reach_kw(_reach)]
                              + [_reach_kw(0) for _ in range(n_core - 1)])

        # ── the loop carry (tul.loop_carry; morph/model/tul_carry.py) ──────────
        # `None` on every other model: a Python-level constant, so nothing below this
        # exists in the traced graph and `_core_step` is the function it was.
        #
        # WHERE THE READ IS CAPTURED, and why THAT tensor. `_carry_cap` is handed to CORE
        # LAYER 0 ALONE, inside the reach kwargs built just above, and travels untouched
        # through `_apply_core_step` -> `MORPHBlock.forward(attn_kwargs=...)` ->
        # `MORPHAttention` -> `_CCABase._gate_combine_up`, which writes
        # `W_up(g_win * out_win)` into it: the WINDOW branch's own gated contribution to
        # that layer's attention output, in d_model space. Under `loop_reach w >= 1` the
        # window branch's XSA excludes the self token, so at layer 0 its rows hold cells
        # k-w .. k-1 and nothing else — the read is PURELY cross-cell, which is what
        # makes it `r_k(t)`. Layers 1..n-1 run at reach 0 (a cell reads itself only), so
        # capturing layer 0 captures every cross-cell route of the pass.
        #
        # It is RETURNED from `_core_step`, not read off the dict, because the pass may
        # run inside `torch.utils.checkpoint`: a side-channel tensor is not
        # checkpoint-safe (MORPHBlock.forward's `ret_capture` docstring says so, and
        # `ret_state` is returned for the same reason). The dict is the transport INSIDE
        # one call; the tuple is the contract across the boundary.
        _carry = self.tul_carry
        _carry_cap: dict = {}
        if _carry is not None:
            _core_akw = (({**_core_akw[0], "tg_win_capture": _carry_cap},)
                         + tuple(_core_akw[1:]))

        def _core_step(h_in, e_in, inj_terms, ret_state=None, iter_idx=0, stage_cond=None,
                       carry=None, want_carry=False):
            if _rr is not None:
                h_in = self._apply_injection(
                    h_in, _rr.read(h_in, _rr_k, _rr_v, _rr_allow, layout.slot_valid))
            if carry is not None:
                # THE RE-INJECTION, at the site `TULReread` adds its term and for the
                # same reason: the gain hinge's probe and the checkpointed step must both
                # see the map WITH the read. `inject_term` scales the carry to the
                # per-cell RMS of THIS tensor — the carrier it is added to — so an
                # unbounded accumulation adds one carrier-RMS worth of its DIRECTION and
                # never more (morph/model/tul_carry.py). The ratio is stashed detached
                # for `carry/inject_ratio_t{t}`; a detached VALUE is checkpoint-safe where
                # a graph node is not.
                _h_pre = h_in
                h_in = self._apply_injection(h_in, _carry.inject_term(carry, h_in))
                with torch.no_grad():
                    # Per-cell RMS(what the add actually MOVED the carrier) / RMS(the
                    # carrier), averaged over the cells that got a term. Read off
                    # `h_in - _h_pre` — the two carriers this function goes on to use —
                    # and NOT off the term, so an injection that is computed and then
                    # dropped reads 0 here instead of a reassuring 1.0. It read 1.0 in
                    # the 2026-09-19 sabotage pass until this line was written this way.
                    #
                    # Cells whose carry is exactly zero — every pad, and cell 0 of every
                    # row, whose window row under a reach budget is empty — are excluded:
                    # they add nothing and would drag a ratio that is 1.0 BY
                    # CONSTRUCTION below 1 for a reason that is not the match.
                    _rt = (h_in - _h_pre).detach().float().flatten(2).pow(2).mean(
                        -1).sqrt()                                            # [B, S]
                    _rh = _h_pre.detach().float().flatten(2).pow(2).mean(-1).sqrt()
                    _msk = _rt > 0
                    _carry_cap["ratio"] = (
                        (_rt[_msk] / _rh[_msk].clamp_min(1e-12)).mean() if bool(_msk.any())
                        else _rt.new_zeros(())).detach()
            if _scse is None:
                _h_out, _rs = self._apply_core_step(
                    h_in, e_in, None, None, None,
                    ret_state=ret_state, iter_idx=iter_idx,
                    inj_terms=inj_terms, stage_cond=stage_cond, attn_kw=_core_akw)
                if want_carry:
                    return _h_out, _rs, _carry_cap["win"]
                return _h_out, _rs
            # ── SCSE, Eqs. 3-5 ──────────────────────────────────────────────────────
            # `h_in` IS Delta_t; `e_in` carries h*. Signature unchanged so the three call
            # sites (no_grad / checkpoint / eager) and the truncated-BPTT window they
            # implement are untouched.
            # `_rec` is bound once and handed to BOTH calls — see the twin in `_tul_core`.
            _rec = _scse.recurrent_input(h_in, e_in)
            g_out, new_ret = self._apply_core_step(
                _rec, None, None, None, None,
                ret_state=ret_state, iter_idx=iter_idx, inj_terms=None, source_free=True,
                stage_cond=stage_cond)
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
        # Onset capture (lab/experiments/planned/2026-09-03-tul-onset-capture.md). Three
        # more per-iteration readings, each a plain Python flag read at trace time so the
        # default graph is untouched:
        #   delta_mean — the MEAN over rows of ‖h_new − h‖/‖h‖ (the max above is one row);
        #   eff_rank   — entropy effective rank of the active slot states after each
        #                iteration (the attractor's dimension), only when `_probe_rank`;
        #   _loop_cot  — the cotangent norm ARRIVING at each grad iteration's output during
        #                backward, via a tensor hook, only when `_probe_cot`. The hook
        #                returns None so the gradient itself is untouched (tests/
        #                test_onset_capture.py proves the grads are bit-identical).
        _pr_dmean: list[Tensor] = []
        _pr_rank: list[Tensor] = []
        _probe_rank = _probe and bool(getattr(self, "_probe_rank", False))
        _probe_cot = _probe and bool(getattr(self, "_probe_cot", False))
        # The readings belong to the TRAINING forward. A no-grad forward that runs between
        # the backward and the trainer's read (the Jacobian probe's capture forward) must
        # neither erase the cotangent dict nor overwrite `_loop_probe` with its own values.
        _probe_write = _probe and torch.is_grad_enabled()
        if _probe_write:
            self._loop_cot = {}
        # Clip-through-time (cfg.slot_cot_clip, the hooks' docstrings). A Python-level
        # constant: 0.0 registers nothing and the graph is the one before this existed.
        _cot_clip = float(self.cfg.slot_cot_clip)
        _cot_hooks = (_probe_cot or _cot_clip > 0.0) and torch.is_grad_enabled()
        if _cot_hooks:
            self._loop_cot_ref = None
            self._loop_cot_rows = {}
            self._loop_cot_post = {}
            self._loop_cot_bind = {}
        # "iter" mode (faithful DiffusionBlocks, morph/model/iter_cond.py): every
        # core-layer application in the T-loop gets an AdaLN-Zero signal for WHICH
        # iteration it is. Python-level constant — `_iter_mode=False` (every model
        # except a `core_stage_cond="iter"` build) makes `_sc` permanently None below
        # and the loop is bit-identical to before this existed.
        _iter_mode = self._core_stage_cond_mode == "iter"
        # ── the loop carry's per-forward state (tul.loop_carry) ───────────────
        # c_k(0) = 0, reset every forward, one [B, S, C] tensor. It is NOT a buffer and
        # nothing about it survives the call, so the forced-depth probes, the per-slot
        # depth draw and the fixed-point term (still on the carrier) see nothing new.
        # Pass 0 is handed `carry=None`, not a zero tensor: the spec's add is at the
        # entry of pass t+1, and it also keeps the t=0 graph free of a term that is
        # identically zero.
        _carry_state = None
        _carry_stats: dict[str, Tensor] = {}
        if _carry is not None:
            _carry_state = h.new_zeros(h.shape[0], h.shape[1], h.shape[-1])
        for t in range(total_iters):
            active = alive if halt else (depths > t)               # [B, S]
            _sc = self.tul_stage_cond.stage_embed(iter_stage_value(t, x.device)) \
                if _iter_mode else None
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
            # db_loop: the CARRY is detached — iteration t's graph reaches the seed only
            # through the live e/injection (ONE core application), never through h. The
            # retention state is detached below for the same reason.
            _h_in = h.detach() if _db else h
            # progressive: this slot's pass t is inside its private no-grad prefix.
            _pfx = (_pk > t) if _prog else None
            _pv = None if _pfx is None else _pfx.view(*_pfx.shape, *([1] * (h.dim() - 2)))
            # ── gradient-conditioned pass (tul.grad_pass) ──────────────────────────
            # OUTSIDE `_core_step` on purpose, and this is the checkpointing decision:
            # `ckpt_grad_iters` wraps `_core_step` in `torch.utils.checkpoint`, whose
            # recompute re-runs that function under a fresh grad context — running an
            # inner `autograd.grad` there is not a shape this tree has ever executed. So
            # the feature is built HERE, once per pass, and only the (already computed)
            # tensor crosses the checkpoint boundary as part of `_h_in`. `ckpt_grad_iters`
            # therefore keeps working unchanged and needs no override.
            #
            # Consequences, stated where they are taken:
            #  * the gain hinge below receives the INJECTED `_h_in`, so it probes the map
            #    at the operating point the run actually reached; its finite difference
            #    still sees the feature as a CONSTANT (the feature is a function of `h`,
            #    not of the probe's perturbation), which is the same "exogenous input"
            #    reading IODINE takes, and it means the hinge does not bound the feature's
            #    own contribution to the gain.
            #  * the Jacobian capture above records the PRE-feature `h`, and the loop
            #    probes below read `in_norm` off `h` while `out_norm` comes from the
            #    post-feature step — so on THIS arm `core_gain` includes whatever the
            #    feature adds, which is the honest reading of "what one pass does" and is
            #    NOT comparable pass-for-pass with the ruler's.
            #  * at a no-grad iteration (`t < n_nograd`) the term is still built; the step
            #    that consumes it runs under `torch.no_grad()`, so no edge to `W_g`
            #    survives and the truncated-BPTT window is what it was.
            # ── the slot chain (tul.slot_chain) ───────────────────────────────────
            # Slot k takes W(z_{k-1}) off the CURRENT carry — the previous slot's exit
            # state whenever that slot's depth is already spent, its live state otherwise.
            # Causal by construction (the shift only ever looks one slot back, and the
            # packer orders slots by row position), and the recurrence depth is the pass
            # count, not the 64 slots: pass t's chain reads a state pass t-1 produced.
            # Placed before the grad-pass feature so both compose, and INSIDE what the gain
            # hinge probes — the chain is part of the map the constraint bounds.
            if _chain is not None:
                _h_in = self._apply_injection(_h_in, _chain(_h_in, layout.slot_valid))
            if _gp is not None:
                _gpm = active & layout.slot_valid
                _gp_g, _gp_l = self._egrad_feature(_h_in, input_ids, layout, _gpm,
                                                   _eg_ctx)
                _gp_term = _gp(_gp_g, _gpm)
                if _gp_write:
                    # 0-dim detached tensors, still on GPU: the local target's value at this
                    # pass, and how far the feature actually moves the state the core sees
                    # (a term the optimiser has left at zero would read 0 forever). The
                    # trainer's float() is the only sync, and it happens after the step.
                    with torch.no_grad():
                        _gp_stats[f"own_pass_t{t}"] = _gp_l
                        _gp_stats[f"gp_rel_t{t}"] = (
                            _gp_term.float().norm() / (_h_in.float().norm() + 1e-6)).detach()
                _h_in = self._apply_injection(_h_in, _gp_term)
            # The carry enters as an ARGUMENT, never as a closure variable: under
            # `checkpoint(use_reentrant=False)` the recompute re-runs `_core_step` during
            # backward, when a nonlocal would already hold c(T-1) instead of c(t-1). An
            # argument is saved with the call and replayed with the right value.
            _cy = _carry_state if (_carry is not None and t > 0) else None
            _want_carry = _carry is not None
            if t < n_nograd:
                with torch.no_grad():
                    _step_out = _core_step(_h_in, _e_arg, _inj_arg, ret_state=ret_state,
                                           iter_idx=t, stage_cond=_sc,
                                           carry=_cy, want_carry=_want_carry)
            elif do_ckpt:
                _step_out = checkpoint(_core_step, _h_in, _e_arg, _inj_arg,
                                       ret_state=ret_state, iter_idx=t, stage_cond=_sc,
                                       carry=_cy, want_carry=_want_carry,
                                       use_reentrant=False)
            else:
                _step_out = _core_step(_h_in, _e_arg, _inj_arg, ret_state=ret_state,
                                       iter_idx=t, stage_cond=_sc,
                                       carry=_cy, want_carry=_want_carry)
            if _want_carry:
                h_new, rs_new, _read = _step_out
            else:
                h_new, rs_new = _step_out
            if _prog:
                # THE cut, and the only one. Detaching a prefix pass's OUTPUT means no
                # cotangent ever enters at that position, so the pass contributes no
                # weight gradient through that slot and nothing flows further back along
                # its trajectory — Bansal's contract. Detaching the INPUT as well was
                # tried and removed: for t >= 1 it is redundant (the previous pass's
                # output detach already makes the carry a constant at that slot, and the
                # prefix is a contiguous initial run, so induction covers every t >= 1),
                # and at t == 0 it would ALSO cut a GRAD slot's own pass-0 read of a
                # neighbouring slot's entry state, starving the prelude of gradient the
                # progressive loss never asked to remove. The slots share one sequence:
                # a prefix slot still serves K/V to grad slots, and that read carries
                # gradient because it belongs to a grad pass.
                h_new = torch.where(_pv, h_new.detach(), h_new)

            if _carry is not None:
                # c_k(t) from c_k(t-1) and THIS pass's read, on the slots whose pass at t
                # actually ran: a finished slot's carrier is frozen by the `torch.where`
                # at the foot of the loop and its carry freezes with it, and a pad slot
                # (never valid) keeps the zero it started at, so its injected term stays
                # exactly zero instead of a 0/0.
                #
                # HERE, immediately after the step and its progressive detach, and before
                # the gain hinge: the hinge re-runs `_core_step` at a detached operating
                # point and would otherwise overwrite `_carry_cap["ratio"]` with its own
                # reading of the same quantity.
                _c_prev = _carry_state
                _c_new, _c_gate = _carry.update(_read, _carry_state)
                if _prog:
                    # The same cut the carrier takes: at a slot's no-grad prefix pass no
                    # cotangent may enter, and the carry is state that crosses passes.
                    _c_new = torch.where(_pfx.unsqueeze(-1), _c_new.detach(), _c_new)
                _cm = (active & layout.slot_valid).unsqueeze(-1)
                _carry_state = torch.where(_cm, _c_new, _carry_state)
                if self._carry_capture is not None:
                    # The test hook, and it is the ONLY way a test can see `r_k(t)`: the
                    # read is consumed inside the loop and never returned. `None` by
                    # default (a Python-level branch that traces out), a list when
                    # `tests/test_tul_loop_carry.py` attaches one. Detached, so attaching
                    # it cannot change a gradient.
                    self._carry_capture.append({
                        "read": _read.detach(), "carry_prev": _c_prev.detach(),
                        "carry": _carry_state.detach(),
                        "gate": None if _c_gate is None else _c_gate.detach()})
                with torch.no_grad():
                    _cv = layout.slot_valid
                    _n_valid = _cv.sum().clamp(min=1)
                    _carry_stats[f"rms_t{t + 1}"] = (
                        (_carry_state.float().pow(2).mean(-1).sqrt() * _cv).sum()
                        / _n_valid).detach()
                    if _c_gate is not None:
                        _carry_stats[f"gate_mean_t{t + 1}"] = (
                            (_c_gate.float().mean(-1) * _cv).sum() / _n_valid).detach()
                    if "ratio" in _carry_cap:
                        # The injection that OPENED this pass used c(t), so the reading
                        # belongs to pass t, not t+1. 1.0 by construction (the RMS match)
                        # — it is the check that the match is live, not a free reading.
                        _carry_stats[f"inject_ratio_t{t}"] = _carry_cap.pop("ratio")

            if t == _t_gain or (_gain_on and _gain_all and t >= n_nograd):
                # The hinge must read the map on the slots whose pass at t CARRIES
                # gradient: its penalty is added to the loss and shapes the core weights,
                # and applying it at prefix positions would constrain the map exactly
                # where the training objective has been cut away. Under `_prog` the drawn
                # prefix can, rarely, leave no grad slot active at the sampled iteration —
                # then there is nothing to measure and the term is skipped (the RNG
                # discipline inside `_slot_gain_penalty` makes skipping side-effect free).
                _gm = active & layout.slot_valid
                if _prog:
                    _gm = _gm & ~_pfx
                if (not _prog) or bool(_gm.any()):
                    _gain_terms.append(self._slot_gain_penalty(
                        _core_step, _h_in, _e_arg, _inj_arg, ret_state, t, _sc,
                        _gm, _gain_lambda, carry=_cy))
            if _renorm:
                # Direction preserved, per-slot norm pinned to the entry norm. Runs on the
                # raw step output, BEFORE the gain governor and the recurrence gate, so it is
                # part of the map every later reader (gate, probes, exit) sees.
                _hn = h_new.flatten(2).float().norm(dim=2)                       # [B, S]
                _rs_ = (_n0 / (_hn + 1e-6)).to(h_new.dtype)
                h_new = h_new * _rs_.view(*_rs_.shape, *([1] * (h_new.dim() - 2)))

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

            # ── GRT recurrence gate (Eq. 4 blend; morph/model/recur_gate.py) ───
            # Part of the core MAP, so it must live under the SAME grad context as the
            # step it blends: at a no_grad iteration the blend runs under no_grad too,
            # or the gate MLP would accumulate gradient through iterations the
            # truncated-BPTT window excludes. (Moot at the panel's full BPTT, exact
            # anywhere else.) Placed BEFORE the halting readout and the loop probes so
            # both see the state the loop actually carries. Inputs are h (pre-step
            # state) and _e_arg (the prelude entry): STATE + PRELUDE only — the
            # cond-zero constraint forbids the iteration index here.
            if self.tul_recur_gate is not None:
                if t < n_nograd:
                    with torch.no_grad():
                        g_r = self.tul_recur_gate(h, _e_arg)
                        h_new = g_r * h + (1.0 - g_r) * h_new
                else:
                    g_r = self.tul_recur_gate(h, _e_arg)
                    h_new = g_r * h + (1.0 - g_r) * h_new

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
                    _pr_dmean.append((((h_new - h) * _am).flatten(1).float().norm(dim=1)
                                      / (_hi + 1e-6)).mean().detach())
                    _pr_ret.append((rs_new if (track_ret and rs_new is not None)
                                    else h.new_zeros(())).float().norm().detach())
                    if _probe_rank:
                        # Stream-reduced state of the ACTIVE slots, [n, C]; entropy
                        # effective rank of its singular values (1 = every active slot
                        # holds the same vector, n = an orthogonal set). Uncentred: the
                        # question is the dimension the states span, not their spread.
                        _z = (h_new.mean(dim=2) if self._is_hc else h_new)[active].float()
                        if _z.shape[0] > 1:
                            _sv = torch.linalg.svdvals(_z)
                            _p = _sv / (_sv.sum() + 1e-12)
                            _er = torch.exp(-(_p * torch.log(_p + 1e-12)).sum())
                        else:
                            _er = h.new_ones(()).float()
                        _pr_rank.append(_er.detach())
            if _cot_hooks and h_new.requires_grad:
                h_new.register_hook(functools.partial(self._loop_cot_hook, t, _probe_cot))
            if _pr_lam > 0.0 and t >= n_nograd and not halt:
                # ||h_{t+1} - h_t||^2 / ||h_t||^2 at EVERY gradient pass, on the slots whose
                # pass at t carries gradient. A prefix pass is excluded for the same reason
                # the gain hinge excludes it: the term shapes the core weights, and applying
                # it where the objective has been cut away constrains the map exactly where
                # nothing trains it.
                _prm = active & layout.slot_valid
                if _prog:
                    _prm = _prm & ~_pfx
                if bool(_prm.any()):
                    _pn = h_new.flatten(2).float()
                    _po = h.flatten(2).float()
                    _pd = _po if _scse is None else _po + h_star.flatten(2).float()
                    _pr = (_pn - _po).pow(2).sum(-1) / (_pd.pow(2).sum(-1) + 1e-6)
                    _pr_terms.append(_pr[_prm])
            if _fp_lam > 0.0 and t >= n_nograd and not halt:
                _fin = active & layout.slot_valid & ~(depths > t + 1)        # finish here
                if _prog:
                    # Redundant by construction (k_i <= T_i - 1 keeps every slot's LAST
                    # pass in the grad window) and kept explicit so the invariant is
                    # stated where it is relied on, not only where the draw is made.
                    _fin = _fin & ~_pfx
                if bool(_fin.any()):
                    _fn = h_new.flatten(2).float()
                    _fo = h.flatten(2).float()
                    _fd = _fn if _scse is None else _fn + h_star.flatten(2).float()
                    _rel = (_fn - _fo).pow(2).sum(-1) / (_fd.pow(2).sum(-1) + 1e-6)   # [B, S]
                    _fp_terms.append(_rel[_fin])

            h = torch.where(active.view(*active.shape, *([1] * (h.dim() - 2))), h_new, h)
            if _db_traj is not None:
                _db_traj.append(h)
            if _mep:
                # `active` is `depths > t`, so this is "the slot's depth reaches pass t+1",
                # ANDed with validity (a pad enters at 0 and is "active" at t = 0) and,
                # under the progressive draw, with "this pass carries gradient".
                _mk = active & layout.slot_valid
                _mep_keep.append(_mk & ~_pfx if _prog else _mk)
            if track_ret and rs_new is not None:
                ret_state = rs_new.detach() if _db else rs_new

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
        if _cot_hooks and _cot_clip > 0.0 and h.requires_grad:
            # The reference for every iteration's clip: the cotangent that reaches the
            # loop's exit carrier. Registered on the carrier itself (not on the last
            # iteration's h_new, whose gradient is masked to the slots still active there).
            h.register_hook(self._loop_cot_ref_hook)
        if _probe_write:
            # [T] each, still on GPU and detached. The trainer reads them once per step.
            self._loop_probe = {
                "core_gain": torch.stack(_pr_gain) if _pr_gain else None,
                "ret_state_norm": torch.stack(_pr_ret) if _pr_ret else None,
                "in_norm": torch.stack(_pr_in) if _pr_in else None,
                "out_norm": torch.stack(_pr_out) if _pr_out else None,
                "delta_ratio": torch.stack(_pr_delta) if _pr_delta else None,
                "clip_bind": torch.stack(_pr_bind) if _pr_bind else None,
                "delta_mean": torch.stack(_pr_dmean) if _pr_dmean else None,
                "eff_rank": torch.stack(_pr_rank) if _pr_rank else None,
            }
        if _gp_write:
            self._loop_gradpass = _gp_stats
        if _carry is not None:
            # 0-dim detached tensors, still on GPU; `_forward_tul` folds them into
            # `groups` as `carry_*` and train.py logs them under `carry/`. Stashed rather
            # than returned for the `_loop_gradpass` reason: this function's return tuple
            # is unpacked positionally by the probes and the tests.
            self._loop_carry_stats = _carry_stats
        # The `disc` critic needs the SAME context at its own training site, which sits
        # after the coda (its label is the coda's CE). Stashed rather than returned for the
        # `_loop_gradpass` reason: it is a detached tensor built OUTSIDE every checkpointed
        # region, and `_tul_core`'s return tuple is already unpacked positionally by the
        # probes and the tests.
        self._tul_egrad_ctx = _eg_ctx
        g_traj = torch.stack(g_list, dim=-1) if g_list else None   # [B, S, T]
        if _scse is not None:
            # Eq. 5 tail: h_T = h* + Delta_T (invariant S6). The deviation lives ONLY inside
            # this function; `_forward_tul` scatters an absolute carrier exactly as today.
            h = h_star + h
        _aux: dict = {}
        if _fp_terms:
            _fp = torch.cat(_fp_terms).mean()
            _aux["fixed_point"] = _fp.detach()
            _aux["fp_weighted"] = _fp_lam * _fp
        if _pr_terms:
            _pr_all = torch.cat(_pr_terms).mean()
            _aux["pass_residual"] = _pr_all.detach()
            _aux["pass_res_weighted"] = _pr_lam * _pr_all
        if _aux:
            self._core_aux = _aux
        if _gain_terms:
            # One sampled iteration: its dict as before. Every iteration: the hinges SUM
            # (each iteration's map is held under the target), the gains report mean / max.
            _gain_reg = {
                "gain": torch.stack([g["gain"] for g in _gain_terms]).mean(),
                "gain_max": torch.stack([g["gain_max"] for g in _gain_terms]).max(),
                "penalty": torch.stack([g["penalty"] for g in _gain_terms]).sum(),
                "n_iters": float(len(_gain_terms)),
            }
        return xn, h, depths, g_traj, _db_traj, _gain_reg, _mep_keep

    def _slot_gain_penalty(self, core_step, h_in, e_arg, inj_arg, ret_state, t, stage_cond,
                           mask, lam: float, carry=None) -> dict:
        """Hinge penalty on the slot map's typical gain at the live operating point.

        g = ||f(h + d) - f(h)|| / ||d|| over the active real slots, d a Gaussian direction
        scaled per slot to `slot_gain_eps` of that slot's norm; penalty
        lam * relu(g - slot_gain_target)^2. Two extra applications of the SAME core step
        (same weights, same iteration index, same retention state) at the DETACHED operating
        point, so the gradient shapes the map at h and nothing upstream of it.

        RNG discipline (the Jacobian-probe lesson, `_jacobian_probe`): the direction, and the
        dropout masks of both applications, are drawn from the global stream with the stream
        put back afterwards — the two applications see identical masks (the difference is
        the map's, not dropout's), and every later draw of the run is what it would have been
        with the penalty off. bf16 is enough here: the two outputs differ by ~eps of their
        magnitude and rounding noise adds in quadrature over ~1e5 elements (measured against
        the fp32 power-iteration probe in tests/test_slot_gain_reg.py).
        """
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state() if h_in.is_cuda else None
        def _restore():
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
        hp = h_in.detach()
        # The map's gain in h at a FIXED source: the injection source and the retention
        # state are detached too, so the penalty shapes the core's weights and nothing
        # upstream of the loop (tests/test_slot_gain_reg.py: prelude grads bit-identical).
        e_arg = e_arg.detach() if torch.is_tensor(e_arg) else e_arg
        inj_arg = inj_arg.detach() if torch.is_tensor(inj_arg) else inj_arg
        ret_state = ret_state.detach() if torch.is_tensor(ret_state) else ret_state
        # `tul.loop_carry`: the carry is part of the map the run applied at this pass, so
        # the probe must see it — detached with the other sources, so the hinge shapes the
        # core's weights and nothing upstream of the loop. None on every other arm.
        carry = carry.detach() if torch.is_tensor(carry) else carry
        m = mask.view(*mask.shape, *([1] * (hp.dim() - 2))).to(hp.dtype)
        v = torch.randn_like(hp) * m
        hn = hp.flatten(2).float().norm(dim=2)                                    # [B, S]
        vn = v.flatten(2).float().norm(dim=2)
        scale = (float(self.cfg.slot_gain_eps) * hn / (vn + 1e-6)).to(hp.dtype)
        d = v * scale.view(*scale.shape, *([1] * (hp.dim() - 2)))
        try:
            _restore()
            f0, _ = core_step(hp, e_arg, inj_arg, ret_state=ret_state, iter_idx=t,
                              stage_cond=stage_cond, carry=carry)
            _restore()
            f1, _ = core_step(hp + d, e_arg, inj_arg, ret_state=ret_state, iter_idx=t,
                              stage_cond=stage_cond, carry=carry)
        finally:
            _restore()
        num = ((f1 - f0) * m).float().flatten(1).norm(dim=1)                       # [B]
        den = d.float().flatten(1).norm(dim=1) + 1e-6
        gain = (num / den)                                                         # [B]
        hinge = torch.relu(gain - float(self.cfg.slot_gain_target))
        pen = lam * (hinge * hinge).mean()
        return {"gain": gain.detach().mean(), "gain_max": gain.detach().max(), "penalty": pen}

    # ── TUL-Code (docs/tul-code-spec.md) ─────────────────────────────────────────

    def _tul_code_seed(self, x: Tensor, x0: Tensor, bigram_emb, layout: SlotLayout):
        """``(xn, e, inj)``: the normalised prelude, the seed at every slot position
        ``[B, S, (n,) C]`` and the loop-invariant injection stack ``[n_core, B, S, C]`` —
        exactly what ``_tul_core`` hands its first pass (lines 3914-3925 / 4134-4139)."""
        np_, n_core = self.cfg.n_prelude, self.cfg.n_core
        gidx, gvalid = layout.slot_index, layout.slot_valid
        xn = self.input_norm(x)
        e = gather_valid(xn, gidx, gvalid)
        x0_s = gather_valid(x0, gidx, gvalid)
        bg_s = gather_valid(bigram_emb, gidx, gvalid) if bigram_emb is not None else None
        inj = torch.stack(
            [self._build_injection_term(np_ + i, self.x0_injects[np_ + i].precompute(x0_s),
                                        None, bg_s, e.dtype)
             for i in range(n_core)], dim=0)
        return xn, e, inj

    def _tul_code_thinker(self, z_noisy: Tensor, z_clean: Tensor, t: Tensor, e: Tensor,
                          inj: Tensor, layout: SlotLayout, seed_detach: bool) -> Tensor:
        """ONE core pass over the doubled slot sequence -> velocity ``[B, S, M, C]`` fp32.

        ``z_noisy`` / ``z_clean`` ``[B, S, M, C]`` fp32 (the clean copies are the tape and
        are context only; the caller detaches them), ``t`` ``[B, S]``. Layout, relation and
        the order's leak argument: ``morph/model/tul_code.py`` module docstring. The seed
        ``e`` and the injection stack enter both copies; the time embedding and the
        per-cell embedding enter the noisy copies only, the clean marker the clean ones.
        """
        B, S, M, C = z_noisy.shape
        dt = e.dtype
        if seed_detach:
            e, inj = e.detach(), inj.detach()
        cells = torch.cat([z_noisy, z_clean], dim=2).to(dt)            # [B, S, 2M, C], noisy FIRST
        h_in = cells.reshape(B, S * 2 * M, C)
        e_rep = e.repeat_interleave(2 * M, dim=1)                       # [B, 2SM, (n,) C]
        inj_rep = inj.repeat_interleave(2 * M, dim=2)                   # [n_core, B, 2SM, C]
        add = torch.zeros(B, S, 2 * M, C, dtype=dt, device=h_in.device)
        add[:, :, :M] = (self.tul_code_time(t).to(dt).unsqueeze(2)
                         + self.tul_code_cell.to(dt).view(1, 1, M, C))
        add[:, :, M:] = self.tul_code_clean.to(dt).view(1, 1, 1, C)
        inj_rep = inj_rep + add.reshape(1, B, S * 2 * M, C)
        if self._is_hc:
            h_in = h_in.unsqueeze(2).expand(-1, -1, self._n_streams, -1).contiguous()
        rel = code_thinker_relation(S, M, h_in.device)
        h_out, _ = self._apply_core_step(h_in, e_rep, None, None, None, ret_state=None,
                                         iter_idx=0, inj_terms=inj_rep,
                                         attn_kw={"tg_relation": rel})
        if self._is_hc:
            h_out = h_out.mean(dim=2)
        h_noisy = h_out.reshape(B, S, 2 * M, C)[:, :, :M]
        return self.tul_code_head(h_noisy).float()

    def _tul_code_null_condition(self, e: Tensor, inj: Tensor, z_tape: Tensor,
                                 rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """The CFG null condition on the rows where ``rows`` ``[B]`` is true: the seed is
        ``tul_code_null`` at every slot, the injection stack is zero, the clean tape is
        zero. Everything the thinker conditions on (`_tul_code_thinker`: ``e``, ``inj``,
        the clean copies) goes through here, so a null-conditioned row's velocity cannot
        depend on the past (tested)."""
        if self.tul_code_null is None:
            raise ValueError("the null condition needs tul.code_cfg_drop > 0 (no tul_code_null)")
        B = rows.shape[0]
        r_e = rows.view(B, *([1] * (e.dim() - 1)))
        null = self.tul_code_null.to(e.dtype).view(*([1] * (e.dim() - 1)), -1).expand_as(e)
        e_n = torch.where(r_e, null, e)
        inj_n = inj * (~rows).view(1, B, 1, 1).to(inj.dtype)
        if z_tape.dtype == torch.long:
            # A discrete tape: the null context is MASK at every symbol (the embedding the
            # denoiser already knows as "no information").
            tape_n = torch.where(rows.view(B, *([1] * (z_tape.dim() - 1))),
                                 torch.full_like(z_tape, int(self._code_mask_id)), z_tape)
        else:
            tape_n = z_tape * (~rows).view(B, 1, 1, 1).to(z_tape.dtype)
        return e_n, inj_n, tape_n

    # ── LCTUL-D: the masked denoiser over a discrete code (spec §16) ──────────────

    def _tul_code_thinker_discrete(self, idx_noisy: Tensor, idx_clean: Tensor, t: Tensor,
                                   e: Tensor, inj: Tensor, layout: SlotLayout,
                                   seed_detach: bool) -> Tensor:
        """ONE core pass over the doubled slot sequence of SYMBOL positions -> logits
        ``[B, S, N, C_book]`` fp32 at the noisy copies.

        ``idx_noisy`` / ``idx_clean`` ``[B, S, N]`` long (``mask_id`` = MASK), ``t``
        ``[B, S]`` = the slot's masked fraction. Same layout, relation and injection
        routes as :meth:`_tul_code_thinker` with ``M := N``: the symbol (or MASK)
        embedding is the carrier at every copy, the time embedding and the per-position
        embedding enter the noisy copies, the clean marker the clean ones.
        """
        B, S, N = idx_noisy.shape
        dt = e.dtype
        if seed_detach:
            e, inj = e.detach(), inj.detach()
        emb = self.tul_code_sym.weight.to(dt)
        cells = torch.cat([emb[idx_noisy], emb[idx_clean]], dim=2)          # [B, S, 2N, C]
        h_in = cells.reshape(B, S * 2 * N, -1)
        e_rep = e.repeat_interleave(2 * N, dim=1)
        inj_rep = inj.repeat_interleave(2 * N, dim=2)
        add = torch.zeros(B, S, 2 * N, h_in.shape[-1], dtype=dt, device=h_in.device)
        add[:, :, :N] = (self.tul_code_time(t).to(dt).unsqueeze(2)
                         + self.tul_code_cell.to(dt).view(1, 1, N, -1))
        add[:, :, N:] = self.tul_code_clean.to(dt).view(1, 1, 1, -1)
        inj_rep = inj_rep + add.reshape(1, B, S * 2 * N, -1)
        if self._is_hc:
            h_in = h_in.unsqueeze(2).expand(-1, -1, self._n_streams, -1).contiguous()
        rel = code_thinker_relation(S, N, h_in.device)
        h_out, _ = self._apply_core_step(h_in, e_rep, None, None, None, ret_state=None,
                                         iter_idx=0, inj_terms=inj_rep,
                                         attn_kw={"tg_relation": rel})
        if self._is_hc:
            h_out = h_out.mean(dim=2)
        h_noisy = h_out.reshape(B, S, 2 * N, -1)[:, :, :N]
        return self.tul_code_sym_head(h_noisy)

    def _tul_code_lift(self, idx: Tensor, ok: Tensor) -> Tensor:
        """Symbols ``[B, S, N]`` long -> the cells the coda reads, ``[B, S, M, C]`` fp32:
        the quantiser's lift, RMS-normalised per cell, zero where ``ok`` is false. The
        SAME statistic a truth code has on this path (the training forward reads
        ``code_rmsnorm`` of the quantiser's cells), so a sample and a truth code are told
        apart by content only."""
        B, S, N = idx.shape
        vq = self.tul_code_vq
        cells = vq.lift(idx.reshape(B, S, vq.k, vq.g))
        return code_rmsnorm(cells) * ok.view(B, S, 1, 1).float()

    def _tul_code_sample_idx(self, idx_tape: Tensor, ok: Tensor, e: Tensor, inj: Tensor,
                             layout: SlotLayout, k: int, generator=None,
                             guidance: float | None = None) -> Tensor:
        """``k`` unmasking rounds (:func:`maskgit_sample`) for every slot in parallel with
        ``idx_tape`` ``[B, S, N]`` long as the clean context -> ``[B, S, N]`` long, ``0``
        where ``ok`` is false. Guidance ``w != 1``: ``l = l_u + w (l_c − l_u)`` on the
        logits, the null pass on every row. Records the pass count."""
        tc = self.cfg.tul
        w = float(tc.code_cfg_scale if guidance is None else guidance)
        mask_id = int(self._code_mask_id)
        if w != 1.0:
            e_u, inj_u, tape_u = self._tul_code_null_condition(
                e, inj, idx_tape, torch.ones(ok.shape[0], dtype=torch.bool, device=ok.device))

            def _logits(idx: Tensor, t: Tensor) -> Tensor:
                l_c = self._tul_code_thinker_discrete(idx, idx_tape, t, e, inj, layout, False)
                l_u = self._tul_code_thinker_discrete(idx, tape_u, t, e_u, inj_u, layout, False)
                return l_u + w * (l_c - l_u)
        else:
            def _logits(idx: Tensor, t: Tensor) -> Tensor:
                return self._tul_code_thinker_discrete(idx, idx_tape, t, e, inj, layout, False)

        return maskgit_sample(_logits, ok, int(self._code_n_sym), mask_id, int(k),
                              tc.code_mask_schedule, generator=generator)

    def _tul_code_sample(self, z_tape: Tensor, ok: Tensor, e: Tensor, inj: Tensor,
                         layout: SlotLayout, k: int, generator=None,
                         guidance: float | None = None, z0: Tensor | None = None) -> Tensor:
        """Sample every slot's code in parallel with ``z_tape`` (``[B, S, M, C]`` fp32) as
        the clean context: ``k`` Euler steps from ``z_0 ~ N(0, source_std²)`` (or the given
        ``z0``). Returns the endpoint, fp32, unnormalised, zero where ``ok`` is false.
        Records the pass count.

        On a discrete code (``tul.code_discrete``) ``z_tape`` is the SYMBOL tape
        ``[B, S, N]`` long, ``k`` is the number of unmasking rounds, and the return is the
        sampled symbols LIFTED to cells (``[B, S, M, C]`` fp32, already at the coda's
        statistic: :meth:`_tul_code_lift`); ``z0`` has no meaning there and is refused."""
        tc = self.cfg.tul
        if tc.code_discrete:
            if z0 is not None:
                raise ValueError("a discrete code has no source draw: z0 is not accepted")
            if z_tape.dtype != torch.long:
                raise TypeError("a discrete code's tape is the [B, S, N] long symbol tensor")
            idx = self._tul_code_sample_idx(z_tape, ok, e, inj, layout, k, generator, guidance)
            return self._tul_code_lift(idx, ok)
        if z0 is None:
            z0 = torch.randn(z_tape.shape, device=z_tape.device, dtype=torch.float32,
                             generator=generator) * float(tc.code_source_std)
        else:
            z0 = z0.float()

        w = float(tc.code_cfg_scale if guidance is None else guidance)
        if w != 1.0:
            # Classifier-free guidance: v = v_u + w (v_c − v_u), the null pass on every row.
            e_u, inj_u, tape_u = self._tul_code_null_condition(
                e, inj, z_tape, torch.ones(ok.shape[0], dtype=torch.bool, device=ok.device))

            def _vel(z: Tensor, t: Tensor) -> Tensor:
                v_c = self._tul_code_thinker(z, z_tape, t, e, inj, layout, seed_detach=False)
                v_u = self._tul_code_thinker(z, tape_u, t, e_u, inj_u, layout, seed_detach=False)
                return v_u + w * (v_c - v_u)
        else:
            def _vel(z: Tensor, t: Tensor) -> Tensor:
                return self._tul_code_thinker(z, z_tape, t, e, inj, layout, seed_detach=False)

        zk = euler_sample(_vel, z0, int(k))
        return zk * ok.view(*ok.shape, 1, 1).float()

    def _tul_code_sample_rolled(self, ok: Tensor, e: Tensor, inj: Tensor,
                                layout: SlotLayout, k: int, generator=None) -> Tensor:
        """The generation regime: the tape is SAMPLED slot by slot, each slot reading the
        sampled codes before it. ``S·k`` core passes. Eval instrument only (`val/ce_k8_rolled`)."""
        B, S = ok.shape
        M, C = int(self.cfg.tul.prefix_k), self.cfg.d_model
        if self.cfg.tul.code_discrete:
            tape_i = torch.full((B, S, int(self._code_n_sym)), int(self._code_mask_id),
                                dtype=torch.long, device=ok.device)
            for s in range(S):
                ik = self._tul_code_sample_idx(tape_i, ok, e, inj, layout, k, generator)
                tape_i[:, s] = ik[:, s]
            return self._tul_code_lift(tape_i, ok)
        tape = torch.zeros(B, S, M, C, device=ok.device, dtype=torch.float32)
        for s in range(S):
            zk = self._tul_code_sample(tape, ok, e, inj, layout, k, generator=generator)
            tape[:, s] = code_rmsnorm(zk[:, s]) * ok[:, s].view(B, 1, 1).float()
        return tape

    def _tul_code_core(self, x: Tensor, x0: Tensor, bigram_emb, layout: SlotLayout,
                       code_mode: str | None, code_steps: int | None, plan_mode: str,
                       code_seed: int | None = None,
                       code_given: Tensor | None = None,
                       code_given_mask: Tensor | None = None,
                       span_scorer=None):
        """The code branch of :meth:`_forward_tul` (spec §4).

        ``span_scorer(cells) -> [B, S]``: the coda's CE on slot s's NEXT span when it reads
        ``cells``; supplied by :meth:`_forward_tul` for ``code_xm_select="coda"`` (training
        only; no grad).

        Returns ``(xn, cells [B, S, M, C], fm_loss | None, stats, h_slots [B, S, C],
        depths)``. ``cells`` is what the coda reads (after the plan ablation); ``h_slots``
        is their mean, for the readers that take one state per slot; ``depths`` is ones,
        a metric placeholder — no slot loop ran.

        Train (``self.code_phase``): 1 — cells = E's code + noise, no thinker pass;
        2 — plus the flow loss (one thinker pass on the interpolant); 3 — plus a fraction
        ``code_rollout_p`` of the valid slots hand the coda a SAMPLED code (no grad).
        Eval (``code_mode``): "encoder" | "sampled" (default, ``code_steps`` or
        ``code_infer_steps``) | "rolled".
        """
        tc = self.cfg.tul
        xn, e, inj = self._tul_code_seed(x, x0, bigram_emb, layout)
        xs = xn.mean(dim=2) if self._is_hc else xn
        z, ok = self.tul_code_enc(xs, layout)                          # [B, S, M, C], [B, S]
        B, S = ok.shape
        if span_scorer is not None:
            span_scorer.bind(xn)
        okf = ok.view(B, S, 1, 1).to(z.dtype)
        stats: dict = {}
        fm_loss = None
        self._code_last_passes = 0
        self._code_sigreg_loss = None
        self._code_vq_out = None
        idx_flat: Tensor | None = None
        if tc.code_discrete:
            # LCTUL-D: E's pooled vector -> N symbols (TULThoughtVQ, cosine codebook, the
            # VQ-VAE terms in `_code_vq_out`) -> the lifted cells, RMS-normalised, are what
            # the coda reads, with the straight-through gradient into E. Rate control at
            # train: `code_sub_p` of the VALID symbols are replaced by a uniform random
            # symbol (LaDiR's token substitution), gradient cut at those symbols; the
            # denoiser's target is always the encoder's own symbols.
            vq = self.tul_code_vq
            z_pool = z.float().mean(dim=2)                                 # [B, S, C]
            sub_index = None
            if self.training and tc.code_sub_p > 0.0:
                draw = torch.rand(B, S, vq.k, vq.g, device=z.device) < tc.code_sub_p
                draw = draw & ok.view(B, S, 1, 1)
                rnd = torch.randint(0, vq.c, (B, S, vq.k, vq.g), device=z.device)
                sub_index = torch.where(draw, rnd, torch.full_like(rnd, -1))
                stats["code_sub_frac"] = float(draw.float().sum()
                                               / (ok.float().sum() * vq.k * vq.g).clamp_min(1))
            cells_q, _, vq_out = vq(z_pool.to(z.dtype), ok, sub_index=sub_index)
            self._code_vq_out = vq_out
            idx_flat = vq_out["index"].clamp_min(0).reshape(B, S, -1)       # [B, S, N]
            z = code_rmsnorm(cells_q.float()).to(z.dtype) * okf
        if self.training:
            phase = int(self.code_phase)
            if tc.code_sigreg_lambda > 0.0:
                # LeJEPA's collapse guard on the CODE: SIGReg per cell index over the valid
                # slots (morph/model/sigreg.py). The gradient reaches E through z directly
                # (not through the detached target), so it acts however code_target_lambda
                # is set; with lambda 1.0 this is LeJEPA's objective on the code.
                _zs = z.float()
                _terms = [sigreg_epps_pulley(_zs[:, :, m][ok], num_slices=tc.sigreg_slices)
                          for m in range(_zs.shape[2])]
                self._code_sigreg_loss = torch.stack(_terms).mean()
                stats["code_sigreg"] = float(self._code_sigreg_loss.detach())
            z_coda = z
            if tc.code_noise > 0.0:
                z_coda = z_coda + tc.code_noise * torch.randn_like(z_coda)
                if tc.code_noise_renorm:
                    z_coda = code_rmsnorm(z_coda).to(z.dtype)
                z_coda = z_coda * okf
            # ── Explorative Modeling (arXiv 2607.27372, Forward XM) ────────────────
            # K independent samples per slot; the one nearest the data is kept, and BOTH
            # losses train on it: the flow pair uses ITS z_0 (the paper's rule — the
            # standard loss on the selected generation's seed), and the phase-3 rollout
            # hands the coda that sample rather than a fresh draw. Selection is no-grad.
            # `code_xm_mode="noise"` is the paper's Diffusion/Flow hybrid instead: no
            # generation, K corruption noises at ONE drawn t, the pair with the lowest flow
            # loss trains (below, inside the flow block).
            xm_z0 = xm_hat = None
            if phase >= 2 and tc.code_xm_k > 1 and tc.code_xm_mode == "sample":
                K = int(tc.code_xm_k)
                with torch.no_grad():
                    z_ref = z.detach().float()
                    z0s = torch.randn((K,) + tuple(z_ref.shape), device=z.device,
                                      dtype=torch.float32) * float(tc.code_source_std)
                    cands = torch.stack([
                        self._tul_code_sample(z_ref, ok, e, inj, layout,
                                              tc.code_rollout_steps, z0=z0s[j])
                        for j in range(K)])                                  # [K, B, S, M, C]
                    cands_n = code_rmsnorm(cands)
                    if tc.code_xm_select == "coda":
                        if span_scorer is None:
                            raise RuntimeError(
                                "tul.code_xm_select='coda' needs the coda scorer from "
                                "_forward_tul (labels required); none was supplied.")
                        score = torch.stack([span_scorer(cands_n[j].to(z.dtype))
                                             for j in range(K)])              # [K, B, S]
                    else:
                        score = (cands_n - z_ref).pow(2).sum((-1, -2))       # [K, B, S]
                    best = score.argmin(dim=0)                                # [B, S]
                    idx = best.view(1, B, S, 1, 1).expand(1, B, S, *z_ref.shape[2:])
                    xm_z0 = z0s.gather(0, idx).squeeze(0)
                    xm_hat = cands.gather(0, idx).squeeze(0)
                    okf_ = ok.float()
                    n_ok = okf_.sum().clamp_min(1.0)
                    stats["code_xm_score_mean"] = float((score.mean(0) * okf_).sum() / n_ok)
                    stats["code_xm_score_best"] = float((score.min(0).values * okf_).sum() / n_ok)
                self._code_last_passes = K * tc.code_rollout_steps
            if phase >= 3 and tc.code_rollout_p > 0.0:
                sel = (torch.rand(B, S, device=z.device) < tc.code_rollout_p) & ok
                z_hat = xm_hat if xm_hat is not None else self._tul_code_sample(
                    idx_flat if tc.code_discrete else z.detach().float(),
                    ok, e, inj, layout, tc.code_rollout_steps)
                z_coda = torch.where(sel.view(B, S, 1, 1), code_rmsnorm(z_hat).to(z.dtype),
                                     z_coda)
                stats["code_rollout_frac"] = float(sel.float().sum() / ok.float().sum().clamp_min(1))
            if phase >= 2 and tc.code_discrete:
                # The masked-diffusion ELBO (spec §16): t ~ U(0,1) per slot, each symbol
                # masked with probability t, CE on the masked symbols weighted 1/t; in
                # nats per span it upper-bounds −log p(code | past). Reported through the
                # same `code_fm_*` keys as the flow loss, in units of the uniform floor
                # N·log C, so the trainer's `fm=` column and the watchers read both paths.
                t = torch.rand(B, S, device=z.device, dtype=torch.float32).clamp_min(1e-3)
                mask = mdm_mask(t, int(self._code_n_sym))
                mask_id = int(self._code_mask_id)
                idx_noisy = torch.where(mask, torch.full_like(idx_flat, mask_id), idx_flat)
                tape_t = idx_flat
                _tape_passes = 0
                if tc.code_tape_rollout_p > 0.0:
                    with torch.no_grad():
                        tape_hat = self._tul_code_sample_idx(idx_flat, ok, e, inj, layout,
                                                             tc.code_rollout_steps)
                    rows_r = torch.rand(B, device=z.device) < tc.code_tape_rollout_p
                    tape_t = torch.where(rows_r.view(B, 1, 1), tape_hat, idx_flat)
                    stats["code_tape_rollout_frac"] = float(rows_r.float().mean())
                    _tape_passes = int(tc.code_rollout_steps)
                e_t, inj_t = e, inj
                if tc.code_cfg_drop > 0.0:
                    rows = torch.rand(B, device=z.device) < tc.code_cfg_drop
                    e_t, inj_t, tape_t = self._tul_code_null_condition(e, inj, tape_t, rows)
                    stats["code_cfg_drop_frac"] = float(rows.float().mean())
                logits = self._tul_code_thinker_discrete(idx_noisy, tape_t, t, e_t, inj_t,
                                                         layout, tc.code_seed_detach)
                loss_raw, per_slot = mdm_loss(logits, idx_flat, mask, t, ok)
                fm_loss = loss_raw / float(self._code_fm_scale)
                with torch.no_grad():
                    stats["code_fm_raw"] = float(loss_raw)
                    stats["code_mdm_nats"] = float(loss_raw)
                    stats["code_fm_null"] = float(self._code_fm_scale)
                    stats["code_fm_rel"] = float(fm_loss)
                    stats["code_mask_frac"] = float(
                        (mask.float().mean(-1) * ok.float()).sum() / ok.float().sum().clamp_min(1))
                    for b in range(4):
                        m = ok & (t >= b / 4.0) & (t < (b + 1) / 4.0)
                        if bool(m.any()):
                            stats[f"code_fm_band{b}_rel"] = float(
                                per_slot[m].mean() / float(self._code_fm_scale))
                _roll = (tc.code_rollout_steps if phase >= 3 else 0)
                self._code_last_passes = 1 + _roll + _tape_passes
            elif phase >= 2:
                t = torch.rand(B, S, device=z.device, dtype=torch.float32)
                # The target: E's code, detached (C4) unless `code_target_lambda` lets the
                # flow loss's gradient reach E at that weight (value unchanged). The clean
                # TAPE stays detached in every case: it is context, not target.
                lam = float(tc.code_target_lambda)
                z_tape_t = z.detach().float()
                z_tgt = (z.float() * lam + z_tape_t * (1.0 - lam)) if lam > 0.0 else z_tape_t
                _tape_passes = 0
                if tc.code_tape_rollout_p > 0.0:
                    # LaDiR's reasoning-model stage 2 (arXiv 2510.04573 §3.3): the thinker's
                    # CONTEXT is its own generated tape, the target stays the oracle code.
                    # v1 draws every slot's sample in ONE parallel run conditioned on the
                    # truth tape (S x k sequential passes would be the paper's exact form);
                    # the selected rows then read that sampled tape at every earlier slot.
                    with torch.no_grad():
                        tape_hat = code_rmsnorm(self._tul_code_sample(
                            z_tape_t, ok, e, inj, layout, tc.code_rollout_steps)) * okf
                    rows_r = torch.rand(B, device=z.device) < tc.code_tape_rollout_p
                    z_tape_t = torch.where(rows_r.view(B, 1, 1, 1), tape_hat.to(z_tape_t.dtype),
                                           z_tape_t)
                    stats["code_tape_rollout_frac"] = float(rows_r.float().mean())
                    _tape_passes = int(tc.code_rollout_steps)
                e_t, inj_t = e, inj
                if tc.code_cfg_drop > 0.0:
                    rows = torch.rand(B, device=z.device) < tc.code_cfg_drop
                    e_t, inj_t, z_tape_t = self._tul_code_null_condition(e, inj, z_tape_t, rows)
                    stats["code_cfg_drop_frac"] = float(rows.float().mean())
                if phase >= 2 and tc.code_xm_k > 1 and tc.code_xm_mode == "noise":
                    # Explorative Modeling, the paper's Diffusion/Flow hybrid (App. C): the
                    # same target, t and condition, K corruption noises; each candidate is
                    # one velocity prediction scored by the flow loss itself; the lowest
                    # pair is re-run with grad (the memory-saving mode). K no-grad passes.
                    K = int(tc.code_xm_k)
                    with torch.no_grad():
                        z0s = torch.randn((K,) + tuple(z_tgt.shape), device=z.device,
                                          dtype=torch.float32) * float(tc.code_source_std)
                        score = []
                        for j in range(K):
                            _, z_t_j, v_tgt_j = cfm_pair(z_tgt, tc.code_source_std, t, z0=z0s[j])
                            v_hat_j = self._tul_code_thinker(z_t_j, z_tape_t, t, e_t, inj_t,
                                                             layout, tc.code_seed_detach)
                            score.append((v_hat_j - v_tgt_j).pow(2).sum((-1, -2)))
                        score = torch.stack(score)                            # [K, B, S]
                        best = score.argmin(dim=0)                            # [B, S]
                        idx = best.view(1, B, S, 1, 1).expand(1, B, S, *z_tgt.shape[2:])
                        xm_z0 = z0s.gather(0, idx).squeeze(0)
                        okf_ = ok.float()
                        n_ok_ = okf_.sum().clamp_min(1.0)
                        stats["code_xm_score_mean"] = float((score.mean(0) * okf_).sum() / n_ok_)
                        stats["code_xm_score_best"] = float(
                            (score.min(0).values * okf_).sum() / n_ok_)
                z0, z_t, v_tgt = cfm_pair(z_tgt, tc.code_source_std, t, z0=xm_z0)
                v_hat = self._tul_code_thinker(z_t, z_tape_t, t, e_t, inj_t, layout,
                                               tc.code_seed_detach)
                per_slot = (v_hat - v_tgt).pow(2).sum((-1, -2))                 # [B, S]
                null_slot = v_tgt.pow(2).sum((-1, -2))
                n_ok = ok.float().sum().clamp_min(1.0)
                loss_raw = (per_slot * ok.float()).sum() / n_ok
                fm_loss = loss_raw / float(self._code_fm_scale)
                with torch.no_grad():
                    null = (null_slot * ok.float()).sum() / n_ok
                    stats["code_fm_raw"] = float(loss_raw)
                    stats["code_fm_null"] = float(null)
                    stats["code_fm_rel"] = float(loss_raw / null.clamp_min(1e-12))
                    for b in range(4):
                        m = ok & (t >= b / 4.0) & (t < (b + 1) / 4.0)
                        if bool(m.any()):
                            stats[f"code_fm_band{b}_rel"] = float(
                                per_slot[m].sum() / null_slot[m].sum().clamp_min(1e-12))
                _g = 2 if tc.code_cfg_scale != 1.0 else 1
                _roll = (_g * tc.code_rollout_steps if phase >= 3 else 0)
                if tc.code_xm_k > 1 and tc.code_xm_mode == "sample":
                    self._code_last_passes = 1 + int(tc.code_xm_k) * tc.code_rollout_steps
                elif tc.code_xm_k > 1:                                  # noise search
                    self._code_last_passes = 1 + int(tc.code_xm_k) + _roll
                else:
                    self._code_last_passes = 1 + _roll
                self._code_last_passes += _tape_passes
            stats["code_phase"] = float(phase)
        else:
            mode = code_mode or "sampled"
            k = int(code_steps or tc.code_infer_steps)
            gen = torch.Generator(device="cpu" if z.device.type == "mps" else z.device)
            # One fixed stream per eval (seed 0) so val/loss is reproducible; the K-sample
            # marginal (morph/training/code_eval.py) passes code_seed = 0 .. K-1.
            gen.manual_seed(int(code_seed) if code_seed is not None else 0)
            if mode == "encoder":
                z_coda = z * self._code_truth_scale
            elif mode == "sampled":
                z_hat = self._tul_code_sample(idx_flat if tc.code_discrete else z.float(),
                                              ok, e, inj, layout, k, generator=gen)
                z_coda = code_rmsnorm(z_hat).to(z.dtype) * okf
                self._code_last_passes = k * (2 if tc.code_cfg_scale != 1.0 else 1)
            elif mode == "rolled":
                z_hat = self._tul_code_sample_rolled(ok, e, inj, layout, k, generator=gen)
                z_coda = z_hat.to(z.dtype)
                self._code_last_passes = k * S
            elif mode == "generate":
                # GENERATION (spec §7, as built): a slot whose next span is complete holds
                # E's code of that span, re-encoded from the text already written; the OPEN
                # slot (the span being written, `slot_valid & ~ok`) holds the cells handed
                # in through `code_given` when the caller has them, else ONE sample at k
                # steps with the encoded tape as context. The generator caches the sample so
                # a span is written from one code (morph/inference/tul_generate.py).
                z_coda = z * self._code_truth_scale
                _open = layout.slot_valid & ~ok
                if code_given_mask is not None:
                    if code_given is None or code_given.shape != z.shape \
                            or code_given_mask.shape != ok.shape:
                        raise ValueError(
                            f"code_given {None if code_given is None else tuple(code_given.shape)} / "
                            f"code_given_mask {tuple(code_given_mask.shape)} must be "
                            f"[B, S, M, C] = {tuple(z.shape)} and [B, S] = {tuple(ok.shape)}")
                    _use = code_given_mask & _open
                    z_coda = torch.where(_use.view(B, S, 1, 1), code_given.to(z.dtype), z_coda)
                    _open = _open & ~code_given_mask
                if bool(_open.any()):
                    # `code_seed` given ⇒ the open slot's z_0 comes from the seeded stream
                    # (the generator derives it from its token seed and the open slot, so a
                    # seeded generation is reproducible end to end); None ⇒ the global RNG.
                    z_hat = self._tul_code_sample(idx_flat if tc.code_discrete else z.float(),
                                                  _open, e, inj, layout, k,
                                                  generator=gen if code_seed is not None else None)
                    z_coda = torch.where(_open.view(B, S, 1, 1), code_rmsnorm(z_hat).to(z.dtype),
                                         z_coda)
                    self._code_last_passes = k
            else:
                raise ValueError(
                    f"code_mode must be encoder|sampled|rolled|generate, got {mode!r}")
            if mode != "generate" and code_given is not None:
                raise ValueError("code_given is read under code_mode='generate' only.")
            stats["code_steps"] = float(k if mode != "encoder" else 0)
        cells = self._tul_plan_ablate(z_coda, layout, plan_mode)
        h_slots = cells.mean(dim=2)
        depths = torch.ones_like(layout.slot_index)
        return xn, cells, fm_loss, stats, h_slots, depths

    def _tul_code_span_scorer(self, x0: Tensor, bigram_emb, input_ids: Tensor,
                              labels: Tensor, layout: SlotLayout, tg_attn_kwargs,
                              tg_reset, L: int):
        """Explorative Modeling's ``code_xm_select="coda"`` criterion: a no-grad closure
        ``score(cells) -> [B, S]`` = the coda's summed token CE on slot s's NEXT span (bag
        s+1) when the cells are written into the prefix positions of the CURRENT token
        states. The coda runs exactly as the training call does (same injections, same
        TG restriction, the strict cell keep) with the Bowman dropout OFF, so the
        selection reads the coda, not a dropout draw. ``xn`` is bound on the first call
        (the code core produces it after the encoder runs)."""
        tc = self.cfg.tul
        # ONE home for the token-CE -> span-CE scatter (`span_ce_index`), shared with
        # LXTUL's oracle-over-stream. Bit-identical to the inline version this replaced.
        gid, keep_tok, lab, G = span_ce_index(labels, layout)
        pos = self.tul.prefix_positions(layout, L)
        w_head = self.embed.lm_weight()
        state = {"xn": None}

        @torch.no_grad()
        def score(cells: Tensor) -> Tensor:
            xn = state["xn"]
            if xn is None:
                raise RuntimeError("span scorer called before the code core bound xn")
            _B, _S, _M, _C = cells.shape
            values = cells.reshape(_B, _S * _M, _C).to(xn.dtype)
            if self._is_hc:
                values = values.unsqueeze(2).expand(-1, -1, self._n_streams, -1).contiguous()
            x_coda = scatter_positions(xn, pos, values)
            keep = None
            if tc.coda_token_input == "embed" or self._tg_strict:
                keep = slot_cell_inject_keep(layout, x_coda.dtype)
            xh = self._back_region(x_coda, x0, bigram_emb, input_ids, inject_keep=keep,
                                   attn_kwargs=tg_attn_kwargs, ret_reset_mask=tg_reset)
            span_ce = accumulate_span_ce(xh, w_head, gid, keep_tok, lab, G)
            return span_ce[:, 1:]                       # slot s reads bag s+1: [B, S]

        score.bind = lambda xn: state.__setitem__("xn", xn)
        return score

    @torch.compiler.disable
    @torch.no_grad()
    def _tul_fan_oracle(self, cells: Tensor, xh: Tensor, base: Tensor, x0: Tensor,
                        bigram_emb, input_ids: Tensor, labels: Tensor,
                        layout: SlotLayout, L: int, keep, coda_kw, tg_reset,
                        stats: dict) -> None:
        """LXTUL's ORACLE-OVER-STREAM (``tul.fan_k``) — the arm's falsifier, eval only.

        PLR Figure 4 and Parallel-TTS's coverage@N, in the units MORPH is scored in. For
        each of the K streams the coda is re-run with THAT STREAM ALONE written into the
        prefix cells, through the SAME ``TULSlots.prefix_project`` the shipped forward
        writes through, and the summed CE of every span is recorded. The three numbers
        that come out, on identical rows and identical tokens:

        ``fan/single_ce``  stream 0 alone — the K = 1 reading inside a K-stream model.
        ``fan/mixed_ce``   the shipped write, read off the coda state ``xh`` this forward
                           already produced, so it is not a second definition of the CE.
        ``fan/oracle_ce``  the per-span MINIMUM over the K streams, picked after the fact.

        THE PREDICTION THIS DECIDES (survey, "The falsifying prediction"): if
        ``oracle_ce`` does not beat ``single_ce`` by more than a width control's own CE
        gain over the ruler, the K streams are K copies with a gate on top and the width
        branch closes. Stated this way, and not on the mixture's CE, because PLR's
        oracle-ceiling result says their width did NOT raise the ceiling — it only closed
        the gap to it, and a better gate would then buy CE without buying exploration.

        ``fan/oracle_pick0`` is the selection-bias reading beside it: the fraction of
        spans whose best stream IS stream 0. At 1/K the argmin is uninformative noise; at
        1.0 the other streams never win and the fan is one stream plus decoration.

        COST: K extra ``_back_region`` passes per eval batch, no backward, ``no_grad``,
        and compiled code is left out of it (``torch.compiler.disable``) because the
        loop's trip count is ``K`` and the logit matmul is per row. Never called on a
        training step, and never under a plan ablation (the caller checks), because a
        shuffled or zeroed write makes "which stream is best" meaningless.
        """
        k = int(cells.shape[2])
        gid, keep_tok, lab, g_bins = span_ce_index(labels, layout)
        w_head = self.embed.lm_weight()
        n_tok = span_token_counts(gid, keep_tok, g_bins)[:, 1:]        # [B, S]
        per_stream = []
        for i in range(k):
            values, pos = self.tul.prefix_project(cells[:, :, i], layout, L)
            x_i = scatter_positions(base, pos, values)
            xh_i = self._back_region(x_i, x0, bigram_emb, input_ids, inject_keep=keep,
                                     attn_kwargs=coda_kw, ret_reset_mask=tg_reset)
            per_stream.append(accumulate_span_ce(xh_i, w_head, gid, keep_tok,
                                                 lab, g_bins)[:, 1:])
        ce = torch.stack(per_stream, dim=-1)                           # [B, S, K]
        mixed = accumulate_span_ce(xh, w_head, gid, keep_tok, lab, g_bins)[:, 1:]
        # A span is scored only when its slot is real AND the span has scored tokens: a
        # tail-pad slot and a slot whose next span fell off the row would otherwise enter
        # the minimum at CE 0 and drag the oracle to an artefact.
        ok = layout.slot_valid & (n_tok > 0)
        denom = n_tok[ok].sum().clamp_min(1.0)
        best, arg = ce.min(dim=-1)
        stats["oracle_ce"] = float(best[ok].sum() / denom)
        stats["single_ce"] = float(ce[..., 0][ok].sum() / denom)
        stats["mixed_ce"] = float(mixed[ok].sum() / denom)
        stats["oracle_gap"] = stats["single_ce"] - stats["oracle_ce"]
        stats["oracle_pick0"] = float((arg[ok] == 0).float().mean()) if bool(ok.any()) else 0.0
        stats["oracle_n_spans"] = float(ok.sum())
        stats["oracle_n_tokens"] = float(n_tok[ok].sum())
        for i in range(k):
            # Each stream's own token-weighted CE, on the SAME spans. `stream_ce_k0` is
            # `single_ce` by definition and the pair is asserted in tests/test_tul_fan.py;
            # the spread across i is the direct "are these K copies" reading, and the
            # oracle can only sit at or below the smallest of them.
            stats[f"stream_ce_k{i}"] = float(ce[..., i][ok].sum() / denom)

    def _tul_db1_precheck(self, what: str) -> None:
        """Shared guards for :meth:`_tul_core_db1` and :meth:`_tul_core_db1_ladder`.

        Every raise here is a combination the mission left undefined rather than a bug:
        no gate readout exists for a state produced by ONE (or K sigma-stepped)
        conditioned application(s) instead of a T-iteration trajectory, SCSE's carry is
        the DEVIATION (the same reason ``db_loop`` raises under SCSE), and
        ``core_gain_clip`` governs the T-iteration carrier's growth — a single EDM-
        preconditioned pass has no such carrier to clip.
        """
        if self._core_stage_cond_mode != "sigma":
            raise RuntimeError(
                f"{what} requires a model built with tul.core_stage_cond='sigma' "
                f"(got {self._core_stage_cond_mode!r}).")
        if self.scse is not None:
            raise NotImplementedError(
                f"{what} under SCSE is not defined: the carry is the DEVIATION and the "
                f"EDM noising target here is the ABSOLUTE slot state (see db_loop's twin "
                f"raise in _tul_core).")
        if self.tul_gate is not None:
            raise NotImplementedError(
                f"{what} has no defined gate interaction: §4 reads the span length off "
                f"the core's per-iteration trajectory and {what} builds no such "
                f"trajectory (one — or K sigma-stepped — conditioned applications, not "
                f"a T-iteration loop).")
        if self.cfg.core_gain_clip > 0.0:
            raise NotImplementedError(
                f"{what} has no defined interaction with model.core_gain_clip: the clip "
                f"governs the T-iteration carrier's realised growth, which {what} does "
                f"not have (EDM preconditioning is the analogous control here). Set "
                f"core_gain_clip=0.0.")

    def _tul_core_db1(self, x: Tensor, x0: Tensor, bigram_emb, layout: SlotLayout):
        """ONE σ-conditioned core application — the faithful DiffusionBlocks training
        step (arXiv 2506.14202 App. B "recurrent-depth architectures"; morph/model/
        iter_cond.py). Replaces the T-iteration loop for ONE training step.

        Design (flagged per the mission's request — every judgment call named):

        * **The "clean" reference is the slot SEED state** ``h_0 = core_init(e)``, i.e.
          exactly what ``_tul_core`` calls ``h`` before its first iteration — NOT a
          separately-observed target. Unlike image diffusion (where ``y`` is the real
          training example), MORPH's downstream task supplies supervision only through
          the coda's CE/mux loss on the DECODED state, so there is no directly-observed
          "clean state" to regress an L2 loss against; this mirrors MORPH's own prior
          finding (``.agents/notes/rejected/feature/2026-08-21-diffusionblocks-
          verdict.md`` "Post-audit addendum": a "ce" objective escapes L2 regression-
          to-mean where an L2-to-embedding objective does not). ``h_0`` doubles as
          BOTH the noised variable ``z`` (``z_σ = h_0 + σ·ε``) AND the clean
          conditioning input ``x`` the paper's ``D_θ(z_σ, x, σ)`` reads through the
          UNCHANGED ``DiagonalInjection``/x0-injection path — the same split
          ``_tul_core`` already makes between its evolving carry ``h`` and its
          loop-invariant source ``e``.
        * **σ is sampled per BATCH SAMPLE, not per slot.** The mission text says "per
          slot"; TUL's ``[B, S, n, C]`` carrier would support a per-(B,S) σ, but this
          v1 samples ``[B]`` (broadcast over every slot in a row) — the paper itself
          never conditions sub-batch, and per-slot σ raises open questions (would a
          slot's OWN depth/Poisson draw also need to correlate with its σ range?) that
          are unmeasured. Flagged, not resolved.
        * **The core layer stack runs EXACTLY ONCE** — one ``_apply_core_step`` call,
          not a T-iteration loop — which is the entire point (paper: "reducing
          computational cost by factor K").
        * Downstream (coda, mux/CE loss) is UNCHANGED: the returned ``h`` lands exactly
          where ``_tul_core``'s ``h_slots`` does today, so ``_forward_tul``'s existing
          ``elif db_traj is None:`` branch supervises it with the SAME weighted-CE /
          mux machinery every other arm uses — no new loss code (mission spec).
        """
        self._tul_db1_precheck("tul_step_mode='db1'")
        np_, n_core = self.cfg.n_prelude, self.cfg.n_core
        gidx, gvalid = layout.slot_index, layout.slot_valid

        xn = self.input_norm(x)
        e = gather_valid(xn, gidx, gvalid)                            # [B, S, n, C]
        h0 = self.core_init(e)                                        # the "clean" seed

        B = x.shape[0]
        sampler = self._db1_sampler
        sigma = sampler.sample(B, device=h0.device)                   # [B]
        eps = torch.randn(h0.shape, device=h0.device, dtype=torch.float32)
        _bview = sigma.view(-1, *([1] * (h0.dim() - 1)))
        z_sigma = h0.float() + _bview * eps

        c_skip, c_out, c_in, c_noise = sampler.precond.coeffs(sigma)
        _cs = c_skip.view(-1, *([1] * (h0.dim() - 1)))
        _co = c_out.view(-1, *([1] * (h0.dim() - 1)))
        _ci = c_in.view(-1, *([1] * (h0.dim() - 1)))
        net_in = (_ci * z_sigma).to(h0.dtype)

        x0_s = gather_valid(x0, gidx, gvalid)
        bg_s = gather_valid(bigram_emb, gidx, gvalid) if bigram_emb is not None else None
        inj = torch.stack(
            [self._build_injection_term(np_ + i, self.x0_injects[np_ + i].precompute(x0_s),
                                        None, bg_s, net_in.dtype)
             for i in range(n_core)], dim=0)

        stage_cond = self.tul_stage_cond.stage_embed(c_noise)         # [B, cond_dim]
        f_out, _ = self._apply_core_step(net_in, e, None, None, None,
                                         ret_state=None, iter_idx=0, inj_terms=inj,
                                         stage_cond=stage_cond)
        h = (_cs * z_sigma + _co * f_out.float()).to(h0.dtype)
        depths = torch.ones_like(gidx)     # metric only — ONE conditioned application
        return xn, h, depths, None, None

    def _tul_core_db1_ladder(self, x: Tensor, x0: Tensor, bigram_emb, layout: SlotLayout,
                             k_steps: int | None = None):
        """K conditioned applications, σ stepping σ_max → σ_min — the eval/inference
        counterpart of :meth:`_tul_core_db1` (paper App. B: "maintaining the original
        K-iteration inference procedure").

        Judgment call (flagged): the ladder's INITIAL noised state uses the SAME
        ``z_σ = h_0 + σ·ε`` construction training used (at ``σ = σ_max``), rather than
        pure noise with no seed signal (the paper's literal image-generation Eq. 5
        ``z_0 = σ_max·ε``). TUL's seed already carries real per-span content the model
        must condition on; discarding it at eval would create a train/eval σ_max
        mismatch (training never sees a σ_max sample with the seed's signal entirely
        absent) as well as throwing away the one input inference actually has. The
        noise draw uses a FIXED generator (seed 0) so the ladder is deterministic
        (mission spec / test (d)) rather than reading the ambient RNG stream.
        """
        self._tul_db1_precheck("the db1 Euler-ladder eval")
        np_, n_core = self.cfg.n_prelude, self.cfg.n_core
        gidx, gvalid = layout.slot_index, layout.slot_valid

        xn = self.input_norm(x)
        e = gather_valid(xn, gidx, gvalid)
        h0 = self.core_init(e)
        B = x.shape[0]

        sampler = self._db1_sampler
        K = int(k_steps or self.cfg.tul.db1_ladder_steps or self.cfg.mean_depth)
        sigmas = sampler.ladder(K)                                     # [K] descending

        gen = torch.Generator(device="cpu" if h0.device.type == "mps" else h0.device)
        gen.manual_seed(0)
        eps = torch.randn(h0.shape, device=h0.device, dtype=torch.float32, generator=gen)
        z = h0.float() + float(sigmas[0]) * eps

        x0_s = gather_valid(x0, gidx, gvalid)
        bg_s = gather_valid(bigram_emb, gidx, gvalid) if bigram_emb is not None else None
        inj = torch.stack(
            [self._build_injection_term(np_ + i, self.x0_injects[np_ + i].precompute(x0_s),
                                        None, bg_s, h0.dtype)
             for i in range(n_core)], dim=0)

        d_hat = None
        for k in range(K):
            s = sigmas[k].to(h0.device).expand(B)
            c_skip, c_out, c_in, c_noise = sampler.precond.coeffs(s)
            _cs = c_skip.view(-1, *([1] * (h0.dim() - 1)))
            _co = c_out.view(-1, *([1] * (h0.dim() - 1)))
            _ci = c_in.view(-1, *([1] * (h0.dim() - 1)))
            net_in = (_ci * z).to(h0.dtype)
            stage_cond = self.tul_stage_cond.stage_embed(c_noise)
            f_out, _ = self._apply_core_step(net_in, e, None, None, None,
                                             ret_state=None, iter_idx=k, inj_terms=inj,
                                             stage_cond=stage_cond)
            d_hat = _cs * z + _co * f_out.float()
            if k < K - 1:
                next_s = sigmas[k + 1].to(h0.device).expand(B)
                z = euler_step(z, d_hat, s, next_s)
        h = d_hat.to(h0.dtype)
        depths = torch.full_like(gidx, K)
        return xn, h, depths, None, None

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

    @staticmethod
    def _mux_reference_terms(pos_valid: Tensor, alpha: Tensor, tgt_slot: Tensor,
                             safe_ids: Tensor, n_sup: Tensor, S: int,
                             vocab: int) -> tuple[float, float]:
        """``(H(target), CE(target, marginal))`` — the two reference points for mux_rel.

        Neither needs the model: both are pure functions of the multiplexed targets, so
        the honesty null costs one extra pass over ``[B, L]`` and NOT a second
        ``[B, S, V]`` forward.

        H(target) uses the identity ``H = -sum_j alpha_j log m_{v(j)}`` where ``m_v`` is
        the AGGREGATED mass of vocab item ``v`` in that span. Aggregation matters: Eq. 2
        sums one-hots, so a subword appearing twice in a span carries the sum of its two
        alphas, and ``-sum_j alpha_j log alpha_j`` would be a different (larger) number.
        The aggregation is done with a compact ``unique`` over ``(row, slot, token)``
        keys rather than a dense ``[B, S, V]`` buffer.

        The null predictor is the batch MARGINAL ``pbar = mean_i target_i`` — the
        unigram baseline. It is strictly tighter than uniform (a test asserts it) and,
        unlike a batch-mean of the MODEL's predictions, it does not move as the model
        trains, so ``mux_rel`` is a stationary reference rather than a moving one.
        """
        m = pos_valid.reshape(-1)
        if not bool(m.any()):
            return 0.0, 0.0
        a = alpha.reshape(-1)[m].double()
        ids = safe_ids.reshape(-1)[m]
        B = pos_valid.shape[0]
        row = (torch.arange(B, device=alpha.device)
               .unsqueeze(1).expand_as(pos_valid).reshape(-1)[m])
        key = (row * S + tgt_slot.reshape(-1)[m]) * vocab + ids
        _u, inv = torch.unique(key, return_inverse=True)
        mass = torch.zeros(int(inv.max()) + 1, device=a.device, dtype=a.dtype)
        mass.scatter_add_(0, inv, a)
        ent = -(a * mass[inv].clamp_min(1e-30).log()).sum() / n_sup

        pbar = torch.zeros(vocab, device=a.device, dtype=a.dtype)
        pbar.scatter_add_(0, ids, a)
        pbar = pbar / n_sup.double()
        ce_null = -(a * pbar[ids].clamp_min(1e-30).log()).sum() / n_sup
        return float(ent), float(ce_null)

    def _tul_mux_loss(self, h_slots: Tensor, input_ids: Tensor,
                      layout: SlotLayout, stats: dict | None = None,
                      slot_keep: Tensor | None = None,
                      target: str | None = None) -> Tensor:
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
        # `mux_readout` (F2): "mean" is `_readout`, the stream mean BEFORE lm_mixer and
        # final_norm — the shipped path. "full" normalises each stream first and averages
        # after, which is the tied head applied per stream with the logits averaged. A
        # Python-level constant on a config field: the branch traces out.
        z = (self._readout_per_stream(h_slots) if tc.mux_readout == "full"
             else self._readout(h_slots))                     # [B, S, C]
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
        # `target` overrides the config's target for ONE call (the staged local loss
        # supervises an intermediate state toward "own" and the final one toward
        # `tc.mux_target`); None keeps the config's, so every existing call is unchanged.
        pos_valid, alpha, tgt_slot, sup = mux_span_targets(
            input_ids, layout, tc.mux_rho,
            target=tc.mux_target if target is None else target)
        if slot_keep is not None:
            # db_loop: supervise only the slots whose state is FRESH at this iteration
            # (depths ≥ t). A frozen slot's state equals its final one; supervising it
            # again at every later iteration would over-weight shallow slots. [B, S] bool,
            # ANDed into both the per-position weights and the normaliser.
            sup = sup & slot_keep
            keep_pos = slot_keep.gather(1, tgt_slot)          # [B, L]
            pos_valid = pos_valid & keep_pos
        B = input_ids.shape[0]
        V = logp.shape[-1]
        # logp[b, tgt_slot[b,p], input_ids[b,p]] without a [B, L, V] gather.
        # Invalid positions gather id 0, NOT their own id: a slot position's own id
        # is slot_id, whose logp is the masked -inf, and 0 * -inf = NaN (caught by
        # test_mux_loss_decomposes_as_ce_plus_weighted_term before it ever ran).
        safe_ids = torch.where(pos_valid, input_ids, torch.zeros_like(input_ids))
        lp = logp.reshape(B, -1).gather(1, tgt_slot * V + safe_ids)    # [B, L]
        ce = -torch.where(pos_valid, alpha * lp, torch.zeros_like(lp)).sum()
        n_sup = sup.sum().to(ce.dtype).clamp(min=1.0)
        loss = ce / n_sup
        if stats is not None:
            # THE HONESTY NULL, the fm_rel shape. The optimised quantity is the weighted
            # CE, which equals the paper's Eq. 4 KL up to the target's entropy — a
            # constant in the parameters, so the GRADIENT is the paper's exactly. But a
            # CE that only ever falls to the target entropy looks like progress it is
            # not, so report all three: the KL (0 at a perfect predictor), the entropy
            # floor, and mux_rel against a null predictor that knows only the corpus
            # marginal. mux_rel == 1.0 exactly at that predictor (tested).
            with torch.no_grad():
                ent, ce_null = self._mux_reference_terms(
                    pos_valid, alpha, tgt_slot, safe_ids, n_sup,
                    layout.slot_index.shape[1], V)
            stats["mux_ce"] = float(loss.detach())
            stats["mux_entropy"] = ent
            stats["mux_kl"] = float(loss.detach()) - ent
            stats["mux_null"] = ce_null
            stats["mux_rel"] = float(loss.detach()) / max(ce_null, 1e-12)
            stats["mux_n_supervised"] = float(n_sup)
        return loss

    def _tul_code_target_write(self, h_slots: Tensor, xn: Tensor, db_traj, depths: Tensor,
                               layout: SlotLayout, L: int, plan_mode: str,
                               code_mode: str | None, code_given: Tensor | None,
                               code_given_mask: Tensor | None,
                               input_ids: Tensor | None = None):
        """``tul.code_target`` (spec §17): the cells the coda reads, and the term.

        Returns ``(values, cells, z, loss, stats, grade_loss, grade_stats)``: ``z`` is the
        frozen encoder's code of THIS row's spans (``out["code_z"]`` at eval, what the
        graded sampler reads off a candidate row), ``grade_loss`` / ``grade_stats`` are
        :meth:`_tul_code_grade`'s (spec §17.2; ``None`` and ``{}`` off a graded step).
        ``values`` ``[B, S·M, (n,) C]`` ready for
        :func:`scatter_positions` at :meth:`TULSlots.prefix_positions`; ``cells``
        ``[B, S, M, C]`` what the coda reads (after the oracle switch, ``code_given`` and
        the plan ablation), returned as ``out["code_cells"]`` at eval; ``loss`` the
        regression of the PREDICTED cells onto the frozen encoder's code, ``2 (1 − cos)``
        per valid cell; ``stats`` the readings.

        THE TARGET. E runs under ``no_grad`` on the token positions of this forward's own
        prelude output (``xn``), exactly as the VAE stage ran it, and its code is unit-RMS
        with slots that have no next span at exactly 0 — the projection is masked by the
        SAME ``ok``, so the coda reads 0 where it was trained to read 0.

        THE READ. ``code_target_detach`` decides whether the coda's CE reaches the loop
        through the cells (LaDiR's decoder never trains on a generated latent: detached).
        ``code_mode="encoder"`` hands the coda E's own code (the ceiling, ``val/ce_tf``);
        ``code_given`` overrides the given slots (the generator's cache, the probes' SHUF /
        ZERO / ORACLE conditions). The plan ablation runs on the CELLS.

        THE INSTRUMENT. With ``db_traj`` (training only) every pass's state is projected
        and read against the same code, no_grad: ``code_target_cos_l0`` is the entry state
        ``core_init(e)``, ``code_target_cos_l{t}`` the state after pass ``t`` over the slots
        whose realised depth reaches it. Whether the passes MOVE toward the code is the
        arm's depth question, and this is where it is read.
        """
        tc = self.cfg.tul
        ref = self.__dict__.get("_code_ref")
        with torch.no_grad():
            if ref is None:
                xs = xn.mean(dim=2) if self._is_hc else xn
                z_tgt, ok = self.tul_code_enc(xs, layout)                 # [B, S, M, C], [B, S]
            else:
                # tul.code_target_ref: E is frozen but its INPUT is not — it pools the
                # prelude's states of the next span, and on any arm where the prelude or
                # the embeddings train those states drift and E's codes collapse (measured
                # 2026-09-17: own cosine 0.61 -> 0.99, shuffled 0.56 -> 0.98 by step 3000).
                # So the whole front comes from the frozen twin, built with the twin's OWN
                # TG kwargs (`instruments-must-use-the-models-tg-kwargs`: a bare
                # `_tul_front` scored strict arms from an unrestricted prelude once
                # already). `input_ids` is required here and the config check guarantees it.
                if input_ids is None:
                    raise RuntimeError(
                        "tul.code_target_ref needs input_ids at the code-target seam: the "
                        "reference recomputes the front itself and cannot reuse `xn`.")
                _fkw, _freset, _, _ = ref._tul_tg_kwargs(layout)
                _rx, _, _ = ref._tul_front(input_ids, layout, attn_kwargs=_fkw,
                                           ret_reset_mask=_freset)
                _rxn = ref.input_norm(_rx)
                z_tgt, ok = ref.tul_code_enc(
                    _rxn.mean(dim=2) if ref._is_hc else _rxn, layout)
        # THE OPEN SLOT (2026-09-17, measured). `ok` is `code_target_valid`: False for a
        # row's LAST valid slot, which has no next span IN THE LAYOUT. At GENERATION that
        # slot is the OPEN one — the span the coda is about to write — so masking the
        # projection by `ok` handed the coda a ZERO cell for every generated span, and
        # `tul_generate.py` cached the zero. The 10k semantic probe on `tul-code-target`
        # read OWN, SHUF and ZERO byte-identical (cos_true 0.1291 for all three, paired
        # +0.0000 [+0.0000, +0.0000]) while ORACLE, injected through the SAME `code_given`
        # route, read 0.559: the injection path worked and the cell was a zero vector.
        # So the PREDICTION is masked by `slot_valid` at eval (every valid slot, the open
        # one included) and stays masked by `ok` at TRAIN: the tail slot has no target
        # there, and feeding the train-side coda a cell nothing grades would put an
        # ungraded input in the reader's path. The LOSS and the cosines below index with
        # `ok` in both modes, so no term ever grades a slot with no target.
        pred = self.tul_code_proj(self._readout(h_slots),
                                  ok if self.training else layout.slot_valid)
        if tc.code_target_loss == "infonce":
            loss, cos_mean, n, acc = code_target_infonce(pred, z_tgt, ok, tc.code_target_tau)
            mse, _, _ = code_target_regression(pred.detach(), z_tgt, ok)
            stats = {"code_target_mse": float(mse), "code_target_cos": float(cos_mean),
                     "code_target_n": float(n), "code_target_acc": float(acc)}
        else:
            loss, cos_mean, n = code_target_regression(pred, z_tgt, ok)
            stats = {"code_target_mse": float(loss.detach()), "code_target_cos": float(cos_mean),
                     "code_target_n": float(n)}
        # own minus this is what the cell knows about ITS span (the generic floor)
        stats["code_target_cos_shuf"] = float(code_target_shuffled_cos(pred.detach(), z_tgt, ok))
        if db_traj is not None and self.training:
            with torch.no_grad():
                for t, ht in enumerate(db_traj):
                    keep = ok if t == 0 else (ok & (depths >= t))
                    pt = self.tul_code_proj(self._readout(ht), keep)
                    _, c_t, _n = code_target_regression(pt, z_tgt, keep)
                    stats[f"code_target_cos_l{t}"] = float(c_t)
        cells = pred.detach() if tc.code_target_detach else pred
        if code_mode == "encoder":
            cells = z_tgt.to(cells.dtype)
        if code_given is not None:
            if code_given_mask is None:
                raise ValueError("code_given needs code_given_mask")
            B_, S_ = ok.shape
            gm = code_given_mask.view(B_, S_, 1, 1)
            cells = torch.where(gm, code_given.to(cells.dtype), cells)
        cells = self._tul_plan_ablate(cells, layout, plan_mode)
        B, S, M, C = cells.shape
        values = cells.reshape(B, S * M, C).to(xn.dtype)
        if self._is_hc:
            values = values.unsqueeze(2).expand(-1, -1, self._n_streams, -1).contiguous()
        # The graded-continuation term (tul.code_grade, spec §17.2) lives at the SAME seam:
        # it needs the live `pred`, the frozen code, `ok`, the trajectory and the ids, and
        # they all meet here and nowhere else.
        grade_loss, grade_stats = None, {}
        if self.cfg.tul.code_grade and input_ids is not None:
            grade_loss, grade_stats = self._tul_code_grade(
                pred, z_tgt, ok, layout, input_ids, db_traj, depths)
        return values, cells, z_tgt, loss, stats, grade_loss, grade_stats

    # ── the frozen reference copy (tul.code_target_ref; spec §17.1) ──────────────────

    @property
    def code_ref(self):
        """The frozen VAE-stage twin, or ``None``. Read it, never assign it."""
        return self.__dict__.get("_code_ref")

    def tul_code_ref_snapshot(self, state: dict | None = None) -> str:
        """Build the frozen reference copy. Returns "resume" or "snapshot".

        Called by the trainer ONCE, AFTER quantisation and AFTER
        ``training.init_from`` / ``training.resume`` have loaded weights — so the twin is
        an exact copy of whatever the live model was seeded with, parametrisations
        included. With ``state`` (a resume: the checkpoint carries the reference it was
        trained against) that state is loaded STRICTLY into the twin instead, because a
        resumed run must keep measuring against the SAME VAE stage: re-snapshotting the
        live weights at step N would silently move the target mid-run, which is the exact
        failure this mechanism exists to prevent.

        The twin is frozen (`requires_grad` False everywhere), in eval mode, and holds no
        reference of its own.
        """
        if self.cfg.tul is None or not self.cfg.tul.code_target_ref:
            raise RuntimeError(
                "tul_code_ref_snapshot on a model built without tul.code_target_ref: "
                "nothing reads the reference, so building it would be dead weight.")
        self.__dict__["_code_ref"] = None            # never deep-copy a copy
        ref = copy.deepcopy(self)
        ref.__dict__["_code_ref"] = None
        how = "snapshot"
        if state is not None:
            ref.load_state_dict(state, strict=True)
            how = "resume"
        for prm in ref.parameters():
            prm.requires_grad_(False)
        ref.eval()
        self.__dict__["_code_ref"] = ref
        return how

    def tul_code_ref_state(self):
        """The reference's ``state_dict`` for the checkpoint, or ``None``."""
        ref = self.__dict__.get("_code_ref")
        return None if ref is None else ref.state_dict()

    def _code_read_model(self):
        """WHICH model computes the VAE-stage readings: the frozen twin when there is one.

        One accessor, so the target encoder, the sampler's coda and the grader's coda can
        never disagree about which weights they are the VAE stage of.
        """
        return self.__dict__.get("_code_ref") or self

    # ── the graded-continuation target (tul.code_grade; spec §17.2) ──────────────────

    def _code_grade_sub_layout(self, layout: SlotLayout, rows: Tensor, k: int) -> SlotLayout:
        """The layout of ``rows``, each repeated ``k`` times — the candidate batch's."""
        def _rep(t):
            return None if t is None else t[rows].repeat_interleave(k, dim=0)
        return SlotLayout(slot_mask=_rep(layout.slot_mask), bag_id=_rep(layout.bag_id),
                          slot_index=_rep(layout.slot_index),
                          slot_valid=_rep(layout.slot_valid), prefix_k=layout.prefix_k,
                          span_len=_rep(layout.span_len),
                          len_supervised=_rep(layout.len_supervised))

    def _code_grade_forward(self, ids: Tensor, lay: SlotLayout, cells: Tensor):
        """One eval forward with every valid slot's cell GIVEN — the sampler's and the
        grader's single read path. Returns ``(coda_state [N, L, C], code_z [N, S, M, C])``.

        The cells are handed in through ``code_mode="generate"`` / ``code_given``, the same
        route the generator and ``code_semantic_probe.py`` take, so nothing here is a second
        implementation of the write. ``coda_state_only`` keeps the ``[N, L, V]`` logits from
        ever being built.
        """
        mdl = self._code_read_model()
        out = mdl._forward_single(ids, None, 0, None, lay,
                                  _code_mode="generate", _code_given=cells,
                                  _code_given_mask=lay.slot_valid,
                                  _coda_state_only=True)
        return out["coda_state"], out["code_z"]

    def _code_grade_logits(self, h: Tensor, src: Tensor, live: Tensor) -> Tensor:
        """Logits at the LIVE source positions only: ``h`` ``[N, L, C]``, ``src`` / ``live``
        ``[N, S]`` -> ``[n_live, V]`` fp32 with the structural slot id at ``-inf``.

        Never ``embed.attend`` over the whole row: at ``L`` 1024 and ``V`` 49169 that is
        806 MB of fp32 per decode step, and this reads at most one position per slot.
        """
        r, c = live.nonzero(as_tuple=True)
        # the REFERENCE's tied head when there is one: the coda that produced `h` is the
        # reference's, and MORPH's head is the input embedding table
        # (`morph-lm-head-is-weight-tied`), so reading it off the live embeddings would
        # decode a frozen coda's state through a table that has moved.
        lg = self._code_read_model().embed.attend(h[r, src[r, c]]).float()
        return lg.index_fill(-1, torch.tensor([self.cfg.tul.slot_id], device=lg.device),
                             float("-inf"))

    def _code_grade_score(self, ids: Tensor, lay: SlotLayout, cells: Tensor, ref: Tensor,
                          first: Tensor, src0: Tensor, live: Tensor, lens: Tensor,
                          n_tok: int) -> tuple[Tensor, Tensor]:
        """The grader: mean log-probability per token of ``ref``'s spans under ``cells``.

        ``ref`` ``[N, S, J]`` the tokens to score (the candidate, or the TRUE span for the
        `code_grade_true` instrument), ``live`` ``[N, S]`` which slots to score, ``lens``
        ``[N, S]`` how many tokens each scored span has. Token ``j`` is scored at the
        position that PREDICTS it: the boundary token of span ``s`` for ``j = 0`` (the only
        trained emit head at ``emit_weight: 0``, and cell-blind under the strict geometry),
        and candidate token ``j-1``'s own position after that.

        Returns ``(grade [N, S] mean log-prob per token, code_z [N, S, M, C])``.
        """
        h, code_z = self._code_grade_forward(ids, lay, cells)
        total = torch.zeros_like(lens, dtype=torch.float32)
        for j in range(n_tok):
            step_live = live & (lens > j)
            if not bool(step_live.any()):
                break
            src = src0 if j == 0 else first + (j - 1)
            lp = torch.log_softmax(self._code_grade_logits(h, src, step_live), dim=-1)
            r_i, s_i = step_live.nonzero(as_tuple=True)          # the SAME order as `lp`
            tgt = ref[r_i, s_i, j]
            total[r_i, s_i] = total[r_i, s_i] + lp.gather(1, tgt.view(-1, 1)).squeeze(1)
        return total / lens.clamp_min(1).float(), code_z

    def _code_grade_cells(self, z_true: Tensor, parity: int, n_slots: int) -> Tensor:
        """The grader's cells. ``coda_zero``: everything zero. ``coda_past``: the frozen
        encoder's TRUE codes with the slots of this ``parity`` zeroed — those are the slots
        being graded on this pass, and cell ``s`` holds the code OF span ``s+1``, which is
        exactly what the grader must not see. Every OTHER cell is a past span's code, so the
        grade is context aware. One pass cannot both zero cell ``s`` (to grade span ``s+1``)
        and keep it (as context for span ``s+2``); hence the two parities.
        """
        if self.cfg.tul.code_grade_grader == "coda_zero":
            return torch.zeros_like(z_true)
        keep = (torch.arange(n_slots, device=z_true.device) % 2) != parity
        return z_true * keep.view(1, n_slots, 1, 1).to(z_true.dtype)

    @torch.compiler.disable()
    def _tul_code_grade(self, pred: Tensor, z_tgt: Tensor, ok: Tensor, layout: SlotLayout,
                        input_ids: Tensor, db_traj, depths: Tensor):
        """``tul.code_grade`` (spec §17.2) — the loop proposes, a blind grader ranks.

        Returns ``(loss, stats)``, ``(None, {})`` on a step that is not graded. Everything
        up to the target runs under ``no_grad`` in EVAL mode with the module's training flag
        and both RNG states restored afterwards; the only live tensor in the term is
        ``pred``, so the gradient reaches the loop and nothing else.

        THE PARALLEL-SPAN DECODE, which is the whole reason this is affordable. Under
        ``tg_geometry="strict"`` the prelude is span-local and a coda token reads its own
        span plus EARLIER prefix cells (:func:`tg_strict_allow`), with the conv, the value
        shift and the retention carry reset per segment. Substituting candidate tokens into
        span ``s+1`` therefore changes the model's output ONLY at span ``s+1``'s own
        positions, so ONE forward advances a candidate token for EVERY eligible span of a
        row at once and the spans cannot contaminate each other. The graded-slot count is
        free; the cost levers are ``code_grade_k``, ``code_grade_tokens`` and
        ``code_grade_rows``, amortised by ``code_grade_every``.

        WHAT THE CANDIDATE IS. It fills the TRUE span's token window. A slot whose next span
        is longer than ``code_grade_tokens`` is NOT graded: ``E`` pools the whole span, so a
        partly substituted span would carry the truth into the target. The boundary rule is
        never re-run — closing the candidate at its own boundary would move every slot
        position after it and force a repack per candidate per decode step.

        WHICH WEIGHTS. With `tul.code_target_ref` every VAE-stage reading here — the coda
        that samples, the coda that grades, the tied head both read through, and `E` on the
        candidate rows — comes from the frozen twin (`_code_read_model`). The live model
        contributes the predicted cell and nothing else.

        WHY THE GRADER IS NOT THE PROPOSING CODA. If a candidate were scored under the cell
        that generated it, the argmax would be that cell's own greedy decode: the target
        would be a fixed point of the current cell and the term would teach nothing. That is
        ``disc``'s 2026-09-12 failure in a new costume.
        """
        tc = self.cfg.tul
        stats: dict[str, float] = {}
        if self._in_code_grade or not self.training:
            return None, stats
        step = int(self.code_grade_step)
        if step % int(tc.code_grade_every) != 0:
            return None, stats
        B, S, M, C = pred.shape
        dev = pred.device
        J, K = int(tc.code_grade_tokens), int(tc.code_grade_k)
        R = min(int(tc.code_grade_rows), B)
        pk = int(layout.prefix_k)

        # ── eligible slots: a next span of 2..J tokens, and a code to compare against ──
        s_ar = torch.arange(S, device=dev)
        n_tok = ((layout.bag_id.unsqueeze(1) == s_ar.view(1, S, 1))
                 & (~layout.slot_mask).unsqueeze(1)).sum(-1)                     # [B, S]
        nxt_len = torch.zeros_like(n_tok)
        nxt_len[:, :S - 1] = n_tok[:, 1:]
        elig = ok & (nxt_len >= 2) & (nxt_len <= J)
        g_cpu = torch.Generator(device="cpu").manual_seed((_GRADE_SEED + step) % (2 ** 31))
        rows = torch.randperm(B, generator=g_cpu)[:R].to(dev)
        gm = elig[rows]                                                          # [R, S]
        stats["code_grade_slot_frac"] = (float(gm.sum())
                                         / max(float(ok[rows].sum()), 1.0))
        stats["code_grade_n"] = float(gm.sum())
        if int(gm.sum()) == 0:
            return (pred * 0.0).sum(), stats

        lens_r = nxt_len[rows]
        first_r = layout.slot_index[rows] + pk        # span s+1's FIRST token position
        src0_r = layout.slot_index[rows] - 1          # span s's boundary token position
        # A real check, not a decorative one: if the packer ever stops putting a slot's
        # cells immediately after its span, every position below is off by k and the
        # grader would silently score the wrong tokens.
        _r, _s = gm.nonzero(as_tuple=True)
        _p = first_r[_r, _s]
        if bool((layout.bag_id[rows][_r, _p] != (_s + 1)).any()) or \
                bool(layout.slot_mask[rows][_r, _p].any()) or \
                bool(layout.slot_mask[rows][_r, src0_r[_r, _s]].any()):
            raise RuntimeError(
                "tul.code_grade: slot_index + prefix_k is not the first TOKEN of the next "
                "span (or slot_index - 1 is not the boundary token). The packer's layout "
                "and the grader disagree; nothing below would score the right positions.")

        N = R * K
        lay_c = self._code_grade_sub_layout(layout, rows, K)
        ids_c = input_ids[rows].repeat_interleave(K, 0).clone()
        cells_prop = pred.detach()[rows].repeat_interleave(K, 0)
        z_true_c = z_tgt.detach()[rows].repeat_interleave(K, 0)
        gm_c = gm.repeat_interleave(K, 0)
        lens_c = lens_r.repeat_interleave(K, 0)
        first_c = first_r.repeat_interleave(K, 0)
        src0_c = src0_r.repeat_interleave(K, 0)
        row_ix = torch.arange(N, device=dev).view(N, 1).expand(N, S)
        cand = torch.zeros(N, S, J, dtype=torch.long, device=dev)

        was_training = self.training
        rng_cpu = torch.random.get_rng_state()
        rng_dev = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None
        self._in_code_grade = True
        self.eval()
        try:
            g_dev = torch.Generator(device=dev).manual_seed(
                (_GRADE_SEED * 3 + step) % (2 ** 31))
            with torch.no_grad():
                # ── the proposal: K candidates, every eligible span of the rows at once ──
                for j in range(J):
                    live = gm_c & (lens_c > j)
                    if not bool(live.any()):
                        break
                    h, _ = self._code_grade_forward(ids_c, lay_c, cells_prop)
                    src = src0_c if j == 0 else first_c + (j - 1)
                    lg = self._code_grade_logits(h, src, live) / float(tc.code_grade_temp)
                    draw = torch.multinomial(torch.softmax(lg, dim=-1), 1,
                                             generator=g_dev).squeeze(1)
                    r_i, s_i = live.nonzero(as_tuple=True)
                    ids_c[r_i, first_c[r_i, s_i] + j] = draw
                    cand[r_i, s_i, j] = draw

                # ── the grade: the frozen coda, blind to the span it is grading ─────────
                grade = torch.full((N, S), -1e9, device=dev, dtype=torch.float32)
                z_cand = None
                parities = (0,) if tc.code_grade_grader == "coda_zero" else (0, 1)
                for par in parities:
                    cz = self._code_grade_cells(z_true_c, par, S)
                    live_p = gm_c if tc.code_grade_grader == "coda_zero" else \
                        (gm_c & ((s_ar % 2) == par).view(1, S))
                    g_p, z_p = self._code_grade_score(ids_c, lay_c, cz, cand, first_c,
                                                      src0_c, live_p, lens_c, J)
                    grade = torch.where(live_p, g_p, grade)
                    if z_cand is None:
                        z_cand = z_p         # E is a function of the PRELUDE, not the cells
                # the TRUE span under the SAME grader: does it rank the truth above its own
                # samples? (R rows, no candidate copies — two more passes at 1/K the width)
                lay_t = self._code_grade_sub_layout(layout, rows, 1)
                ids_t = input_ids[rows]
                z_true_t = z_tgt.detach()[rows]
                ref_t = torch.zeros(R, S, J, dtype=torch.long, device=dev)
                _rt, _st = gm.nonzero(as_tuple=True)
                for j in range(J):
                    _sel = lens_r[_rt, _st] > j
                    ref_t[_rt[_sel], _st[_sel], j] = ids_t[_rt[_sel],
                                                           first_r[_rt, _st][_sel] + j]
                g_true = torch.full((R, S), float("nan"), device=dev, dtype=torch.float32)
                for par in parities:
                    ct = self._code_grade_cells(z_true_t, par, S)
                    live_p = gm if tc.code_grade_grader == "coda_zero" else \
                        (gm & ((s_ar % 2) == par).view(1, S))
                    g_t, _ = self._code_grade_score(ids_t, lay_t, ct, ref_t, first_r,
                                                    src0_r, live_p, lens_r, J)
                    g_true = torch.where(live_p, g_t, g_true)

                # ── the diversity guard, then best and worst ────────────────────────────
                d2 = code_grade_distinct2(cand, lens_c)
                degen = gm_c & (d2 < float(tc.code_grade_min_distinct2))
                grade = torch.where(degen, torch.full_like(grade, -1e9), grade)
                gr = grade.view(R, K, S)
                dg = degen.view(R, K, S)
                keep = gm & ~dg.all(dim=1)
                best = gr.argmax(dim=1)
                # The worst REAL candidate. A degenerate was just set to -1e9, so a plain
                # argmin picks it whenever one exists, and both readings it feeds become
                # meaningless: `code_grade_worst` is a mean over slots, so ONE sentinel
                # drags it to -1e9/n (measured 2026-09-18 on tul-code-grade-l2: n=45,
                # degen=0.02, worst=-22222230 = (-1e9 + 44*-7)/45), and
                # `code_grade_cos_worst_true` becomes the cosine to a DEGENERATE code,
                # which is how cosW 0.185 came out above cosB 0.152 on that step. `best` is
                # an argmax so it was never affected, and neither was the loss: `pref`
                # takes z_cand and `best`, never z_worst.
                worst = torch.where(dg, torch.full_like(gr, 1e9), gr).argmin(dim=1)
                z_cand = z_cand.view(R, K, S, M, C)
                _g = best.view(R, 1, S, 1, 1).expand(R, 1, S, M, C)
                z_best = z_cand.gather(1, _g).squeeze(1)
                z_worst = z_cand.gather(1, worst.view(R, 1, S, 1, 1)
                                        .expand(R, 1, S, M, C)).squeeze(1)
                if int(keep.sum()):
                    _k3 = keep.unsqueeze(1).expand(R, K, S)
                    _live = _k3 & ~dg
                    z_t_r = z_tgt.detach()[rows].float()
                    stats["code_grade_best"] = float(
                        gr.gather(1, best.unsqueeze(1)).squeeze(1)[keep].mean())
                    stats["code_grade_worst"] = float(
                        gr.gather(1, worst.unsqueeze(1)).squeeze(1)[keep].mean())
                    stats["code_grade_mean"] = float(gr[_live].mean())
                    stats["code_grade_true"] = float(g_true[keep].mean())
                    # The grader's sanity reading: where the real continuation lands
                    # among the samples, as a FRACTION of the non-degenerate candidates it
                    # beats. A rank and not a difference, because the difference is only
                    # meaningful while the grader is not the sampler. If a candidate were
                    # scored under the cell that drew it, Jensen would settle it against
                    # the truth (E_p[log p] = -H at or above the truth's -CE) whatever the
                    # grader was worth. Here the cell the sampler used is zeroed, so the
                    # two distributions differ and the truth can and does win: the
                    # 2026-09-17 Spark smoke reads rank 0.89 at the VAE checkpoint. A rank
                    # collapsing toward 0 is this grader drifting into the sampler.
                    _beat = ((gr < g_true.unsqueeze(1)) & ~dg).sum(dim=1).float()
                    _cnt = (~dg).sum(dim=1).clamp_min(1).float()
                    stats["code_grade_true_rank"] = float((_beat / _cnt)[keep].mean())
                    stats["code_grade_degen"] = float(dg[_k3].float().mean())
                    stats["code_grade_cos_best_true"] = float(
                        ((z_best.float() * z_t_r).sum(-1) / float(C))[keep].mean())
                    stats["code_grade_cos_worst_true"] = float(
                        ((z_worst.float() * z_t_r).sum(-1) / float(C))[keep].mean())
                stats["code_grade_n"] = float(keep.sum())
        finally:
            self.train(was_training)
            self._in_code_grade = False
            torch.random.set_rng_state(rng_cpu)
            if rng_dev is not None:
                torch.cuda.set_rng_state(rng_dev, dev)

        if int(keep.sum()) == 0:
            return (pred * 0.0).sum(), stats
        pred_r = pred[rows]
        if tc.code_grade_loss == "pref":
            loss, margin = code_grade_pref_loss(pred_r, z_cand, best, keep,
                                                tc.code_grade_tau)
            stats["code_grade_margin"] = float(margin)
            with torch.no_grad():
                _, cos_b, _ = code_target_regression(pred_r.detach(), z_best, keep)
        else:
            loss, cos_b, _n = code_target_regression(pred_r, z_best, keep)
        stats["code_grade_cos_best"] = float(cos_b)
        if db_traj is not None:
            # The depth question, read against the COMPUTED target: does pass t move the
            # state toward the winner's code? Beside `code_target_cos_l{t}`, which reads the
            # same passes against the TRUTH's code.
            with torch.no_grad():
                for t, ht in enumerate(db_traj):
                    kp = keep if t == 0 else (keep & (depths[rows] >= t))
                    pt = self.tul_code_proj(self._readout(ht[rows]), kp)
                    _, c_t, _n = code_target_regression(pt, z_best, kp)
                    stats[f"code_grade_cos_l{t}"] = float(c_t)
        return loss, stats

    def _tul_oracle_z_loss(self, db_traj, depths: Tensor, input_ids: Tensor,
                           layout: SlotLayout, stats: dict | None = None) -> Tensor:
        """``tul.oracle_z`` — a per-PASS target the loop is asked to match (train only).

        THIS IS THE RULE-BREAKING ARM, and the docstring says so where the code is. The
        standing rule (root ``CLAUDE.md``; LCM T3/4, CoCoMix §6b, BT §4.2) is never to
        regress onto the slot state. Eleven arms have read a per-pass K-curve of zero, and
        the open question is whether the passes CANNOT descend a useful objective or
        whether nothing has ever told them what each pass is FOR. The cheapest way to ask
        is to hand each pass a target that a one-step optimiser could match, and see
        whether a per-pass K-curve appears. If it does not, the answer is about the map and
        not about the supervision. Wolfe decides whether it ever ships.

        THE TRAJECTORY. ``z*_0`` is ``db_traj[0]`` — ``core_init(e)``, exactly the state
        pass 1 receives — detached and in fp32. Each step takes the gradient of the SPAN
        DECODER's next-span CE with respect to ``z*_{t-1}`` and moves the state by
        ``oracle_z_lr * ||z*_{t-1}||`` along ``-g/||g||``, per slot, so a step's size is
        relative to the state it starts from and ``oracle_z_lr`` reads as a fraction. Every
        step runs under ``no_grad`` except its own inner ``autograd.grad``, taken with
        ``create_graph=False`` against a leaf copy — so the trajectory has no ``grad_fn``
        and nothing here trains the decoder. ``autograd.grad`` never writes ``.grad``, so
        the decoder's parameters are untouched even though the inner backward passes
        through them (``tests/test_tul_oracle_z.py`` asserts the decoder's gradients are
        EQUAL with and without this term).

        THE TERM. ``oracle_z_weight * mean_t ||h_t - z*_t||^2 / d`` over the slots whose
        REALISED depth reaches pass ``t``. A slot that stopped at pass 3 has a frozen
        ``h_t`` for every later ``t``, and matching a frozen state to a moving target would
        supervise the ``torch.where`` and not the map.

        COST, arithmetic and not a guess. ``T`` forward+backward passes of a ``[B, S, J,
        V]`` readout at ``J = tul.oracle_z_max_tokens``. One readout is 38.7 GFLOP per
        token of J at B 6, S 64, C 1024, V 49169, so J 8 and T 6 is roughly 5.6 TFLOP
        against a ~50 TFLOP step (~11 %). At the decoder's own J of 32 it would be ~45 %
        and the arm would miss the queue's rate floor — which is why
        ``oracle_z_max_tokens`` exists as its own knob.
        """
        tc = self.cfg.tul
        dec = self.tul_spandec
        assert dec is not None and db_traj is not None
        T = min(int(tc.oracle_z_steps), len(db_traj) - 1)
        if T < 1:
            return db_traj[0].new_zeros(())
        J = min(int(tc.oracle_z_max_tokens), dec.per_span_tokens)
        ids, valid = next_span_slots(input_ids, layout, J)
        # The tied table, detached at BOTH ends exactly as `_tul_spandec_loss` reads the
        # decoder's input side: the oracle must not reshape the table its own target is
        # made of, and it trains nothing at all.
        w_tied = self.embed.lm_weight().detach()
        lab = torch.where(valid, ids, torch.full_like(ids, -100))
        lr = float(tc.oracle_z_lr)

        z = db_traj[0].detach().float()
        star: list[Tensor] = []
        losses: list[float] = []
        for _t in range(T):
            with torch.enable_grad():
                zr = z.detach().requires_grad_(True)
                st = dec.decode(self._readout(zr), ids, valid, w_tied)
                loss_t = fused_linear_cross_entropy(
                    st.reshape(-1, st.shape[-1]), w_tied, lab.reshape(-1),
                    ignore_index=-100, chunk_size=self.cfg.ce_chunk_size,
                    mask_token_id=tc.slot_id)
                g, = torch.autograd.grad(loss_t, zr, create_graph=False)
            with torch.no_grad():
                losses.append(float(loss_t.detach()))
                gn = g.flatten(2).norm(dim=2)                            # [B, S]
                zn = z.flatten(2).norm(dim=2)
                sc = (lr * zn / (gn + 1e-12)).view(*gn.shape, *([1] * (z.dim() - 2)))
                z = (z - g * sc).detach()
            star.append(z)

        d = float(db_traj[0].shape[-1])
        terms: list[Tensor] = []
        for t in range(1, T + 1):
            keep = (depths >= t) & layout.slot_valid                     # [B, S]
            if not bool(keep.any()):
                continue
            diff = (db_traj[t].float() - star[t - 1]).flatten(2).pow(2).sum(-1) / d
            terms.append(diff[keep].mean())
        if not terms:
            return db_traj[0].new_zeros(())
        out = torch.stack(terms).mean()
        if stats is not None:
            stats["oracle_z_mse"] = float(out.detach())
            stats["oracle_z_steps_used"] = float(T)
            # The oracle's OWN readout per step: the honesty instrument. If these do not
            # fall, the trajectory is not a descent and the term is teaching noise.
            for t, v in enumerate(losses):
                stats[f"oracle_z_l{t}"] = v
        return out

    def _tul_spandec_loss(self, h_slots: Tensor, input_ids: Tensor,
                          layout: SlotLayout, stats: dict | None = None,
                          cells: Tensor | None = None) -> Tensor:
        """Span-decoder local loss: decode the WHOLE next span from the slot's exit state.

        The MUX head (:meth:`_tul_mux_loss`) scores ``z`` against an ORDER-FREE geometric
        bag of the next span's tokens, so its optimum is that span's weighted unigram
        marginal. This term instead runs a small causal decoder over
        ``[z, t_0 .. t_{J-2}]`` and charges ``-log p(t_j | z, t_{<j})`` at every token of
        the span, so the gradient reaches ``z`` from every token and ``z`` has to carry
        what the span's CONTINUATION needs, not what its marginal needs. Design and
        contracts: ``morph/model/tul_spandec.py``.

        ``z`` is read BEFORE :meth:`TULSlots.prefix_project`, through the same
        ``_readout`` stream mean the MUX head, ``slot_z_optimize.py`` and the gradient
        probe read — so this term grades exactly the state every earlier instrument
        measured, and ``worth_profile``'s prefix-write ablation stays the right companion
        reading.

        The tied head is read through ``tul.mux_detach_head`` (default true), for the
        reason ``TULConfig.mux_detach_head`` records: ``embed.lm_weight()`` IS the input
        embedding table, and arm v1a diverged at step 2800 with the detach off. The SAME
        detached table supplies the decoder's input embeddings.

        ``cells`` ``[B, S, M, *carrier, C]`` is the Thought Register's M looped cells
        (``tul.slot_cells``), handed in by the caller at the SAME seam ``h_slots`` is read
        — after the think-once stack, before the eval-only plan ablation — so the memory
        the decoder cross-attends to under ``tul.spandec_reads_cells`` is cell for cell the
        object ``TULSlots.prefix_project`` writes into the coda's prefix positions. Each
        cell goes through the SAME ``_readout`` as ``z``, so the decoder's two inputs live
        in one space. ``None`` on every model without the register, and required when the
        decoder was built with ``reads_cells``.

        ``dec.target_offset`` (``tul.spandec_target_offset``) picks WHICH span is decoded:
        1 is the next span; k is span s+k and only that span, with the last k−1 slots of a
        row masked to ``ignore_index`` because the span they would be graded on is not in
        the row.

        Cost, stated because it is not free. The readout is
        ``[B, S, J, V]`` — 2.4 GB fp32 at B=6, S=64, J=32, V=49169 — so it goes through
        :func:`fused_linear_cross_entropy`, which never materialises it. That kernel
        always accumulates a ``[V, d]`` fp32 ``grad_w`` (201 MB at V=49169, d=1024) and
        saves it for the backward, and with a DETACHED head that accumulator is computed
        and thrown away. It is the price of having one chunked-CE implementation in the
        tree rather than two.
        """
        tc = self.cfg.tul
        dec = self.tul_spandec
        assert dec is not None
        z = self._readout(h_slots)                                    # [B, S, C]
        mem = None
        if dec.reads_cells:
            if cells is None:
                raise RuntimeError(
                    "tul.spandec_reads_cells is on but the span-decoder loss was handed no "
                    "register cells. The knob is refused at tul.slot_cells == 1, so the "
                    "only way here is a caller that dropped the `cells` argument.")
            _B, _S, _M = cells.shape[0], cells.shape[1], cells.shape[2]
            # ONE `_readout` per cell, through the same stream mean / lm_mixer / final_norm
            # `z` takes — `_readout` reduces the CARRIER axis, which sits after the cell
            # axis, so the cells are flattened onto the slot axis for the call and put back.
            mem = self._readout(cells.reshape(_B, _S * _M, *cells.shape[3:]))
            mem = mem.reshape(_B, _S, _M, mem.shape[-1])              # [B, S, M, C]
        w_tied = self.embed.lm_weight()                               # [V, C]
        w_head = w_tied.detach() if tc.mux_detach_head else w_tied
        ids, valid = horizon_span_slots(input_ids, layout, dec.per_span_tokens,
                                        dec.horizon, start=dec.target_offset)
        # THE ASYMMETRY, and it is deliberate. The OUTPUT head follows `mux_detach_head`,
        # because that knob's whole subject is "may an auxiliary head train the tied
        # table" and the answer must not depend on which auxiliary is asking. The INPUT
        # embedding read is ALWAYS detached: the MUX has no input-side read, so there is no
        # precedent to follow, and an undetached one would let the decoder reshape the
        # table that the slot's own seed (`E_slot` + a bag-mean OF that table) is built
        # from — the feedback loop `TULConfig.mux_detach_head` records. `SpanDecoder.tok_in`
        # is the learnable map that lets the decoder adapt without writing into the table.
        st = dec.decode(z, ids, valid, w_tied.detach(), mem=mem)       # [B, S, J, C]
        C = st.shape[-1]
        lab = torch.where(valid, ids, torch.full_like(ids, -100))
        loss = fused_linear_cross_entropy(
            st.reshape(-1, C), w_head, lab.reshape(-1), ignore_index=-100,
            chunk_size=self.cfg.ce_chunk_size, mask_token_id=tc.slot_id)
        if stats is not None:
            # THE DECODE-CHEAP READOUT. `spandec_ce` is a per-TOKEN conditional CE over the
            # next span, so it is directly comparable with the model's own token CE — which
            # is the whole point: it says how many nats of the span the thought alone (plus
            # the span's own prefix) buys, at the decoder's cost rather than the coda's.
            # `spandec_ce_h` is the term actually optimised — a per-TOKEN conditional CE
            # over whatever horizon the arm runs. `spandec_ce` is the H = 1 PART of it, so
            # a horizon arm's sweep column stays comparable with every earlier arm's and
            # with the model's own token CE.
            stats["spandec_ce_h"] = float(loss.detach())
            stats["spandec_horizon"] = float(dec.horizon)
            # WHICH span the column above is a CE over. At offset > 1 a `spandec_ce` is
            # NOT comparable with any earlier arm's: it grades a span two or three
            # boundaries away, which is a harder job than the next span. The offset is
            # logged beside it so a scorer can never read the two as the same number.
            stats["spandec_target_offset"] = float(dec.target_offset)
            stats["spandec_n_tokens"] = float(valid.sum())
            if dec.horizon == 1:
                stats["spandec_ce"] = float(loss.detach())
            elif not self.training:
                # A SECOND chunked CE over block 0 alone. Eval only: it is a full extra
                # [B, S, J, V] readout, the sweep and the val pass are where the column is
                # read, and paying for it every training step would cost the arm its rate.
                # A training run therefore logs `spandec_ce_h` and no `spandec_ce`.
                _lab0 = lab.clone()
                _lab0[:, :, dec.per_span_tokens:] = -100
                stats["spandec_ce"] = float(fused_linear_cross_entropy(
                    st.reshape(-1, C), w_head, _lab0.reshape(-1), ignore_index=-100,
                    chunk_size=self.cfg.ce_chunk_size,
                    mask_token_id=tc.slot_id).detach())
                stats["spandec_n_tokens_h1"] = float(valid[:, :, :dec.per_span_tokens].sum())
        return loss

    def _spandec_pass_decode(self, z: Tensor, ids: Tensor, valid: Tensor,
                             emb: Tensor) -> Tensor:
        """One per-pass decode, as a named function so it can be CHECKPOINTED.

        `torch.utils.checkpoint.checkpoint` needs a callable whose arguments are the
        tensors it must re-supply in backward; a bound method keeps the recompute
        readable in a traceback and lets a test call the same code path the
        checkpointed forward runs. The per-pass position table is read here rather
        than passed, because it is a parameter of the decoder, not an input.
        """
        dec = self.tul_spandec
        assert dec is not None
        return dec.decode(z, ids, valid, emb, pos=dec.pos_pass)

    def _tul_spandec_per_pass_loss(self, db_traj, depths: Tensor, input_ids: Tensor,
                                   layout: SlotLayout, stats: dict | None = None) -> Tensor:
        """``tul.spandec_per_pass`` — one planning target per PASS, growing by a span a pass.

        THE OBJECTIVE. The state after pass ``t`` is decoded into spans ``s+1 .. s+H_t``,
        ``H_t = min(t, tul.spandec_pass_horizon_max)``, through the SAME
        :class:`~morph.model.tul_spandec.SpanDecoder` the exit state is graded by. Pass 1
        is asked for the next span, pass 2 for the next two, and so on, so a pass can only
        improve on its predecessor's target by extending the plan one span further. The
        EXIT term is untouched: it stays the shipped ``H = 1`` "next thought" loss, which
        is what keeps ``z`` the thing the coda has to decode.

        WHY THIS AND NOT REACHABILITY. The strict panel's amendment 3 (2026-09-12) found
        the first per-pass K-curve that moved — ``tg_coda_prefix_reach: prev`` plus
        ``loop_reach: 1``, token K1-K6 +0.0163 at CE parity — by BLINDING the coda, so the
        loop had to carry history. Wolfe: "z has to hold the history when it should hold
        the present next thought that needs decoding. Our objectives are still poor." Here
        the coda keeps its full reach and the depth is asked for by the target instead.

        THE MASK, the oracle's rule. A slot is graded at pass ``t`` only when its REALISED
        depth reaches ``t``. A slot that stopped at pass 3 carries a frozen state for every
        later pass, and grading it again would supervise the ``torch.where`` carry and
        over-weight shallow slots.

        THE REDUCTION. One CE per pass (a mean over that pass's graded target tokens), then
        a plain mean over the passes — passes weigh EQUALLY. A token-weighted mean would
        make the term mostly about the deepest pass, which carries ``pass_horizon_max``
        times pass 1's tokens.

        THE GRADIENT, and this is the point. Nothing is detached: the term reaches pass
        ``t``'s core application through ``db_traj[t]``, and through the live carry it
        reaches every EARLIER pass as well. The decoder trains on it too (the parameters
        are shared with the exit target), which is deliberate — one reader, one notion of
        what a decodable plan is.

        THE POSITION TABLE is :attr:`SpanDecoder.pos_pass`, not :attr:`SpanDecoder.pos`.
        The per-pass sequence is ``H_t`` blocks of ``spandec_pass_tokens``; the exit
        sequence is one block of ``spandec_max_tokens``. Row 8 means "span s+2, token 0"
        here and "span s+1, token 8" there, and one parameter cannot be both.

        COST, arithmetic. ``sum_{t=1..6} t * 8 = 168`` decoded positions per slot per step
        at the panel's settings, against the ``H = 3`` exit target's 96 and the shipped
        target's 32 — 21.0 decoder block-passes per real token plus the exit term's 4.0.
        """
        tc = self.cfg.tul
        dec = self.tul_spandec
        assert dec is not None and db_traj is not None
        T = len(db_traj) - 1
        if T < 1:
            return db_traj[0].new_zeros(())
        if dec.pos_pass is None:
            raise RuntimeError(
                "tul.spandec_per_pass ran on a decoder built with no per-pass position "
                "table: SpanDecoder(pass_positions=0). The table is sized at construction "
                "from spandec_pass_horizon_max * spandec_pass_tokens.")
        J1 = int(tc.spandec_pass_tokens)
        cap = int(tc.spandec_pass_horizon_max)
        w_tied = self.embed.lm_weight()                               # [V, C]
        w_head = w_tied.detach() if tc.mux_detach_head else w_tied
        terms: list[Tensor] = []
        n_tokens = 0.0
        for t in range(1, T + 1):
            keep = (depths >= t) & layout.slot_valid                  # [B, S]
            if not bool(keep.any()):
                continue
            H = min(t, cap)
            ids, valid = horizon_span_slots(input_ids, layout, J1, H)  # [B, S, H*J1]
            valid = valid & keep.unsqueeze(-1)
            z = self._readout(db_traj[t])                             # [B, S, C]
            # The SAME detach asymmetry `_tul_spandec_loss` documents: the OUTPUT head
            # follows `mux_detach_head`, the decoder's INPUT read of the tied table is
            # always detached.
            #
            # RECOMPUTED IN BACKWARD, and the arm does not fit without it. Measured on a
            # GB10 at the panel budget (seq 1024, batch 6, 64 cells, T = 8 passes,
            # `pass_horizon_max` 6, `pass_tokens` 8): the un-checkpointed term costs
            # 12.60 GB of the 25.55 GB step against `slot-spandec-strict`'s 12.95 GB, and
            # the 5090 died in the MAIN token CE one step after a 25.50 GB step 0. The
            # cost is linear in DECODED POSITIONS (111.3 KB each, 101,376 of them), not in
            # the number of CE calls: the decoder's own saved activations are 11.0 GB of
            # it — RMSNorm's fp32 upcast 5.64, the SwiGLU 3.09, qkv 1.16, proj 0.39 — and
            # the T fused-CE `grad_w` accumulators are only 1.5. So the DECODE is
            # checkpointed and the CE is NOT: recomputing the decoder costs one extra
            # 2-block forward (4.4 TFLOP/step) and saves 9.09 GB measured in isolation
            # (12.87 -> 3.78), while recomputing the CE as well would save a further 0.76
            # and cost a second vocab pass (~33 TFLOP/step, more than the model).
            # Amendment 2 of lab/experiments/planned/2026-09-12-arc-objective-arms.md.
            # The objective is untouched: `checkpoint` recomputes, it does not re-weight.
            st = checkpoint(self._spandec_pass_decode, z, ids, valid, w_tied.detach(),
                            use_reentrant=False)
            C = st.shape[-1]
            lab = torch.where(valid, ids, torch.full_like(ids, -100))
            loss_t = fused_linear_cross_entropy(
                st.reshape(-1, C), w_head, lab.reshape(-1), ignore_index=-100,
                chunk_size=self.cfg.ce_chunk_size, mask_token_id=tc.slot_id)
            terms.append(loss_t)
            if stats is not None:
                stats[f"spandec_pass_t{t}"] = float(loss_t.detach())
                stats[f"spandec_pass_h{t}"] = float(H)
                n_tokens += float(valid.sum())
        if not terms:
            return db_traj[0].new_zeros(())
        out = torch.stack(terms).mean()
        if stats is not None:
            stats["spandec_pass_ce"] = float(out.detach())
            stats["spandec_pass_terms"] = float(len(terms))
            stats["spandec_pass_n_tokens"] = n_tokens
        return out

    def _tul_horizon_loss(self, db_traj, input_ids: Tensor, layout: SlotLayout,
                          stats: dict | None = None) -> Tensor:
        """``tul.horizon_weight`` — LoopMTP's per-pass horizon alignment (Eq 12-14).

        Pass ``t`` (``t = 1..T``, ``T = tul.slot_depth_fixed``) is scored against the
        DETACHED, mean-pooled tied-embedding representation of span ``i+t`` — a
        cosine-similarity loss, never a cross-entropy, so this needs no decoder at all.
        Pass 1 is skipped when ``tul.horizon_free_first`` (the default, LoopMTP's own
        choice): it is left free to serve as the substrate later passes read from.

        THE MASK is :func:`span_slots`' own rule at ``shift=t``: a slot is graded at
        pass ``t`` only when span ``i+t`` EXISTS and is COMPLETE. Rows too short to have
        a span ``i+t`` at all contribute nothing to that pass's term (not zero-padded
        into it), and a ``t`` at or past the row's slot budget is skipped outright — the
        row has no slot that could ever be graded there.

        THE GRADIENT reaches pass ``t``'s core application through ``db_traj[t]`` and,
        through the live carry, every earlier pass too. Nothing here trains
        ``embed`` (the target embeddings are detached) or any later reader of ``z`` — it
        shapes the LOOP alone, through ``tul_horizon_proj``.
        """
        tc = self.cfg.tul
        assert self.tul_horizon_proj is not None and db_traj is not None
        T = len(db_traj) - 1
        S = layout.slot_index.shape[1]
        J = int(tc.horizon_tokens or tc.bound_span_cap)
        w_tied = self.embed.lm_weight().detach()
        start_t = 2 if tc.horizon_free_first else 1
        terms: list[Tensor] = []
        n_tokens = 0.0
        for t in range(start_t, T + 1):
            if t >= S:
                # Slot s is graded on span s+t; at or past the row's slot budget no slot
                # could ever have one. `span_spans` itself raises here rather than
                # returning an empty label set — skip the pass instead of hitting that.
                continue
            ids, valid = span_slots(input_ids, layout, J, shift=t)      # [B, S, J]
            keep = valid.any(dim=-1) & layout.slot_valid                # [B, S]
            if not bool(keep.any()):
                continue
            e = F.embedding(ids, w_tied)                                # [B, S, J, C]
            m = valid.unsqueeze(-1).to(e.dtype)
            tgt = (e * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)      # [B, S, C]
            z = self.tul_horizon_proj(self._readout(db_traj[t]).float())
            cos = F.cosine_similarity(z, tgt.float(), dim=-1)           # [B, S]
            loss_t = (1.0 - cos)[keep].mean()
            terms.append(loss_t)
            if stats is not None:
                stats[f"horizon_t{t}"] = float(loss_t.detach())
                n_tokens += float(valid.sum())
        if not terms:
            return db_traj[0].new_zeros(())
        out = torch.stack(terms).mean()
        if stats is not None:
            stats["horizon_ce"] = float(out.detach())
            stats["horizon_terms"] = float(len(terms))
            stats["horizon_n_tokens"] = n_tokens
        return out

    def _tul_coda_span_loss(self, xh: Tensor, input_ids: Tensor, layout: SlotLayout,
                            stats: dict | None = None) -> Tensor:
        """``tul.coda_span_heads`` — decode the next span from the CODA, all offsets at once.

        Wolfe, 2026-09-12: "try parallel token decoding from the coda. Perhaps the coda
        needing to spit out a lot of the span or all the span at once changes the
        behavior." Every span-decoder arm so far grades ``z`` through a SEPARATE reader the
        token CE never touches. This grades the reader the model ships.

        WHAT RUNS. ``xh`` is the coda readout (``_back_region``: coda blocks, the
        Hyper-Connection stream mean, ``lm_mixer``, ``final_norm``) at every packed
        position. At each slot's emitting position it is put through ``J =
        coda_span_heads`` parallel offset heads — the ``_MTPHead`` construction, RMSNorm
        plus a ``[d, d]`` linear at IDENTITY init, so at step 0 every head predicts exactly
        what the next-token head at that position predicts — and each head's state is read
        through the tied table. Head ``j`` is scored against token ``j`` of the NEXT span
        (``next_span_slots``). NON-AUTOREGRESSIVE: no teacher forcing, no token path, all J
        offsets from one state. That is the whole question — the span decoder's conditional
        path is what made ``z`` carry the span, and this asks whether the coda alone can be
        made to.

        WHERE IT READS (``tul.coda_span_source``):

        * ``"cell"`` — ``slot_index[s] + prefix_k - 1``, the slot's LAST prefix cell. Under
          the strict geometry that cell carries the looped state and nothing else, so the
          gradient reaches the loop's write through ``TULSlots.prefix_project``. It is also
          the position whose own emit label is the next span's first token and which
          carries NO loss at ``emit_weight: 0.0``, so head 1 is the first term ever to
          train it.
        * ``"token"`` — the boundary TOKEN position (``boundary_token_index``), the
          position ``emit_source="token"`` generates from. **Its coda state has never seen
          its own slot's z**: the boundary token sits BEFORE that slot's cells and the coda
          is causal. The heads then reach the loop only through EARLIER slots' writes. It
          is a control, and the docstring says so because a reader would otherwise assume
          both sources grade the same thing.

        THE TIED HEAD follows ``tul.mux_detach_head`` (default true), the rule every
        auxiliary head on this tree follows: ``embed.lm_weight()`` IS the input embedding
        table and arm v1a diverged at step 2800 with the detach off. The model's own MTP
        heads (``model.mtp_heads``) read it UNDETACHED because they are the shipped
        next-token objective; these are not.

        THE READOUT is ONE ``fused_linear_cross_entropy`` call over ``[B*S*J, d]``, not J
        calls: the kernel allocates a ``[V, d]`` fp32 ``grad_w`` per call (201 MB at
        V=49169, d=1024) and J calls would pay it J times. The ``[B*S*J, V]`` logits are
        never materialised — they would be 604 MB fp32 at B 6, S 64, J 8, V 49169; the
        kernel walks the vocabulary in ``ce_chunk_size`` chunks instead.
        """
        tc = self.cfg.tul
        heads = self.coda_span
        assert heads is not None
        J = len(heads)
        B, S = layout.slot_index.shape
        L = xh.shape[1]
        ids, valid = next_span_slots(input_ids, layout, J)            # [B, S, J]
        if tc.coda_span_source == "cell":
            pos = layout.slot_index + (int(layout.prefix_k) - 1)
            ok = layout.slot_valid
        else:
            # `boundary_token_index` returns [B, S+1] with -1 where a bag owns no token
            # position (a pad slot, or the dump bin). Clamp for the gather and mask with
            # the same test, never with the clamped index.
            bt = boundary_token_index(layout.bag_id, ~layout.slot_mask, S)[:, :S]
            ok = layout.slot_valid & (bt >= 0)
            pos = bt
        pos = pos.clamp(0, L - 1)
        st = gather_positions(xh, pos)                                # [B, S, C]
        hs = torch.stack([h(st) for h in heads], dim=2)               # [B, S, J, C]
        C = hs.shape[-1]
        lab = torch.where(valid & ok.unsqueeze(-1), ids,
                          torch.full_like(ids, -100))
        w_tied = self.embed.lm_weight()
        w_head = w_tied.detach() if tc.mux_detach_head else w_tied
        loss = fused_linear_cross_entropy(
            hs.reshape(-1, C), w_head, lab.reshape(-1), ignore_index=-100,
            chunk_size=self.cfg.ce_chunk_size, mask_token_id=tc.slot_id)
        if stats is not None:
            # A per-TOKEN conditional-free CE over the next span, directly comparable with
            # the model's own token CE and with `spandec_ce` — except that this reader has
            # NO token path, so it is an upper bound on what one state can say about the
            # span in parallel, not a competitor to the decoder's number.
            stats["coda_span_ce"] = float(loss.detach())
            stats["coda_span_heads"] = float(J)
            stats["coda_span_n_tokens"] = float((lab != -100).sum())
        return loss

    @staticmethod
    def _core_token_aux_kwargs(layout: SlotLayout) -> dict:
        """The attention relation the CORE-TOKEN AUXILIARY runs its core under.

        ``causal AND (same span OR j is a slot cell)`` on the window branch
        (:func:`tg_allow_mask`), the same slot-column restriction on the compressed branch
        (``tg_slot_mask``; under that relation the slot columns are plain causal, so a
        separate ``tg_comp_allow`` would be the identical mask and is not built), and the
        :func:`tg_segment_ids` reset on the CCA conv and its ``W_v_prev`` value shift.

        A SEPARATE METHOD so the leak test can probe the SHIPPED relation instead of
        rebuilding it. The first version of `tests/test_tul_core_token_aux.py` built these
        kwargs itself, and a sabotage that widened the real ones to plain causal was
        therefore MISSED (2026-09-12). One home, one probe.
        """
        return {"tg_allow": tg_allow_mask(layout),
                "tg_slot_mask": layout.slot_mask,
                "tg_seg": tg_segment_ids(layout)}

    def _tul_core_token_aux(self, x: Tensor, x0: Tensor, bigram_emb, input_ids: Tensor,
                            labels: Tensor, layout: SlotLayout, coda_kw: dict | None,
                            ret_reset_mask: Tensor | None,
                            stats: dict | None = None) -> Tensor:
        """``tul.core_token_aux`` — the TOKEN CE through the core, TRAINING ONLY.

        THE FACT. In the slot loop the six shared core blocks are trained by the SLOT
        losses alone: ~51 valid cells per row, one exit target each, while the token CE
        reaches the core at ~1 % of the prelude's gradient (the 2026-09-10 per-pass
        cotangent probe). The PLAIN looped model trains the same six blocks on 1,024
        next-token targets per row and earns 0.185 nats of depth. Twelve slot arms since
        2026-09-04 read a token K1-K6 inside [-0.0001, +0.0033]. Wolfe, 2026-09-12: "train
        the core on the token CE as well (tokens through the core for gradient only, slots
        still the only cross-span channel at inference)."

        WHAT RUNS. A SECOND pass over the SAME prelude output ``x``: every position —
        tokens and slot cells together, the paid loop's shape — goes through
        :meth:`_core_region` (the per-sample Poisson-depth core the plain model and the
        paid loop already run; no second core is written here), then through
        :meth:`_back_region`, and is scored by :meth:`_tul_group_losses` — ONE weighted CE
        with the §5 weights, i.e. token labels only at ``emit_weight: 0.0``, exactly the
        paid loop's reduction. The returned tensor is added to the loss as
        ``core_token_aux_weighted``, which ``train.py`` subtracts so ``train/loss`` stays
        the SHIPPED path's CE.

        WHAT DOES NOT CHANGE. The shipped forward. Eval, the forced-depth sweep,
        ``worth_profile``, ``slot_z_optimize`` and inference all run the slot loop with
        tokens OUTSIDE the core. This method is guarded on ``self.training`` at its call
        site, so an eval forward never builds it and the sweep reads the ruler's columns.

        THE GEOMETRY, and it is the whole reason this is not just "the paid loop again".
        Tokens in the core must not become a cross-span channel the core can lean on, or
        the arm buys its CE by re-opening the bypass ``tg_geometry="strict"`` exists to
        cut, and the slot loop stops being the only thing that crosses a boundary. So the
        aux core runs under :func:`tg_allow_mask` — causal AND (same span OR ``j`` is a
        slot cell) — on the window branch, the same slot-column restriction on the
        compressed branch, and the :func:`tg_segment_ids` reset on the CCA conv and its
        ``W_v_prev`` value shift. A token reads its own span's tokens and reaches every
        EARLIER span only through a slot cell, which is the reachability the slot cells
        themselves have inside the loop. The aux coda takes the SHIPPED coda relation
        (``coda_kw``) and the same zeroed slot-cell injections
        (:func:`slot_cell_inject_keep`), so a cell there still carries only what the core
        put in it.

        THE DEPTH IS ITS OWN DRAW. The slot loop draws a per-SLOT depth ``[B, S]``;
        ``_core_region`` draws a per-SAMPLE one ``[B]``. Every reduction of the first to
        the second distorts the distribution — the max over ~51 Poisson(6) draws capped at
        8 is 8 almost surely — so the aux takes the plain model's own per-sample Poisson
        draw instead. Training the core the way the PLAIN model trains it is the arm's
        entire claim.

        SIDE EFFECTS, all suppressed rather than left to luck:

        * **RNG.** The whole call is wrapped in a save/restore of the CPU (and CUDA)
          generator state, the :meth:`_slot_gain_penalty` precedent. The aux consumes
          nothing from the run's stream, so ``loss - core_token_aux_weighted`` equals the
          off-model's loss BIT FOR BIT whatever else runs after it.
        * **``self._core_aux``.** ``_core_region`` stashes the terminal fixed-point term
          there and ``_forward_tul`` consumes it at the end. The slot loop has already
          written its own, so the aux's is saved out and RESTORED: the aux's fixed-point
          value is reported as ``core_token_aux_fp`` and is NOT added to the loss. Adding
          it would apply ``model.core_fixed_point_lambda`` twice per step and make the
          ``fixed_point`` series incomparable with every arm that came before.
        * **``self._jac_capture``.** Disabled for the duration. A probe that recorded both
          the slot map's operating points and the aux core's would be measuring two
          different maps under one name.
        """
        core_kw = self._core_token_aux_kwargs(layout)
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state() if x.is_cuda else None
        saved_aux, saved_jac = self._core_aux, self._jac_capture
        self._jac_capture = None
        try:
            xc = self._core_region(x, x0, bigram_emb, input_ids, attn_kwargs=core_kw)
            keep = slot_cell_inject_keep(layout, xc.dtype)
            xh = self._back_region(xc, x0, bigram_emb, input_ids, inject_keep=keep,
                                   attn_kwargs=coda_kw, ret_reset_mask=ret_reset_mask)
            groups = self._tul_group_losses(xh, labels, layout, want_groups=False)
            aux_fp = self._core_aux or {}
        finally:
            self._core_aux, self._jac_capture = saved_aux, saved_jac
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state(rng_cuda)
        loss = groups["loss"]
        if stats is not None:
            # `core_token_aux_ce` against the model's own `ce_main` is the reading: the
            # same tokens, the same coda, the same weights — the ONLY difference is that
            # these states went through the core. `core_token_aux_fp` is the aux core's
            # fixed-point ratio, reported and NOT charged (see the docstring).
            stats["core_token_aux_ce"] = float(loss.detach())
            stats["core_token_aux_n"] = float(groups["n_targets"].detach())
            if "fixed_point" in aux_fp:
                stats["core_token_aux_fp"] = float(aux_fp["fixed_point"])
        return loss

    def _own_span_grad(self, h: Tensor, input_ids: Tensor, layout: SlotLayout,
                       mask: Tensor) -> tuple[Tensor, Tensor]:
        """``(dL_own/dz, L_own)`` at the CURRENT slot state, both DETACHED.

        The gradient is the IODINE feature; the loss beside it is the arm's own readout —
        ``loop/own_pass_t{t}``, the trajectory of the local target THROUGH the loop, which
        is what says whether the passes are descending the objective they are handed.

        ``L_own`` is :meth:`_tul_mux_loss` at ``target="own"``: the slot's own span read
        through the tied head. That target is CAUSAL for the slot — ``mux_span_targets``
        gives slot ``i`` the tokens of span ``i``, and slot ``i`` sits after every one of
        them — so a generator can compute the same feature at inference with no lookahead.

        The gradient is taken with respect to a DETACHED copy of ``h`` and with
        ``create_graph=False``, so:

        * no outer gradient flows through the feature (the returned tensor has no
          ``grad_fn``; the token CE never differentiates through this inner backward), and
        * the inner graph is freed by ``autograd.grad`` itself, so the cost is one extra
          ``[B, S, V]`` readout per pass and NOT one retained per pass.

        The alternative, ``create_graph=True``, is the second-order learned-optimiser
        objective (Andrychowicz et al. 2016). It is not built: it would retain one inner
        graph per pass and put a Hessian-vector product in every training step, and IODINE
        (§3.1) reports the detached feature is what works.

        The reduction over Hyper-Connection streams follows the readout the loss actually
        used, and the two readouts differ:

        * ``tul.mux_readout='mean'`` (the default and the arm's setting): :meth:`_readout`
          starts with ``x.mean(dim=2)``, so the gradient arriving at each of the ``n``
          streams is the SAME vector scaled by ``1/n``. The mean over streams recovers its
          direction and the RMS normalisation in :class:`TULGradPass` removes the constant.
        * ``tul.mux_readout='full'`` (finding F2): each stream is normalised separately, so
          the per-stream gradients genuinely DIFFER. The mean is then a summary of them,
          not a recovery of one vector — a defensible single-stream feature, and a weaker
          claim. Stated because it is the only thing about this reduction that changes
          between the two readouts; a test covers it.

        Either way the caller broadcasts ONE ``[B, S, C]`` term to every stream through
        :meth:`_apply_injection`, exactly as every other injection into this carrier does.

        ``mask`` is the ``[B, S]`` set of slots whose pass is live at this iteration; it is
        the ``slot_keep`` of the own loss, so a frozen or pad slot contributes nothing to
        the normaliser and reads a zero feature.
        """
        with torch.enable_grad():
            h_d = h.detach().requires_grad_(True)
            loss = self._tul_mux_loss(h_d, input_ids, layout, slot_keep=mask, target="own")
            g = torch.autograd.grad(loss, h_d, create_graph=False, allow_unused=True)[0]
        if g is None:
            # No supervised slot in this batch at this pass (every span empty under the
            # mask). Not an error: the feature is zero, exactly as it is for a pad slot.
            g = torch.zeros_like(h)
        g = g.detach()
        return (g.mean(dim=2) if self._is_hc else g), loss.detach()

    def _egrad_feature(self, h: Tensor, input_ids: Tensor, layout: SlotLayout,
                       mask: Tensor, ctx: Tensor | None) -> tuple[Tensor, Tensor]:
        """``(dE/dz, E)`` at the CURRENT slot state, both DETACHED — the dispatch for
        ``tul.grad_pass_energy``.

        ``own_mux`` delegates to :meth:`_own_span_grad` unchanged, so the shipped arm's
        forward is untouched. The other two energies live in ``morph/model/tul_egrad.py``
        and share every property that makes the feature a FEATURE:

        * the gradient is taken with respect to a DETACHED copy of ``h`` and with
          ``create_graph=False``, so the outer graph never differentiates through this
          inner backward and no second-order term exists;
        * the energy MODULE's parameters are in this inner graph but receive nothing from
          it (``autograd.grad`` accumulates into no ``.grad``), and their own training loss
          is taken separately on a stop-gradient copy of ``z`` — so the energy cannot shape
          ``z`` except through ``W_g``;
        * the reduction over Hyper-Connection streams is the mean, and the caller
          broadcasts one ``[B, S, C]`` term back to every stream, exactly as every other
          injection into this carrier does.

        ``ctx`` is the ``disc`` critic's context — the slot's prelude-entry state, mean over
        streams — and is ``None`` for the other two energies.
        """
        eg = self.tul_egrad
        if eg is None:
            return self._own_span_grad(h, input_ids, layout, mask)
        tc = self.cfg.tul
        with torch.enable_grad():
            h_d = h.detach().requires_grad_(True)
            z = self._readout(h_d) if tc.mux_readout == "mean" \
                else self._readout_per_stream(h_d)
            if isinstance(eg, ReconEnergy):
                w_tied = self.embed.lm_weight()
                w_head = w_tied.detach() if tc.mux_detach_head else w_tied
                loss = eg.loss(z, input_ids, layout, w_head, w_tied.detach(),
                               self.cfg.ce_chunk_size, tc.slot_id, slot_keep=mask)
            else:
                assert ctx is not None
                loss = eg.energy(z, ctx, mask)
            g = torch.autograd.grad(loss, h_d, create_graph=False, allow_unused=True)[0]
        if g is None:
            # Nothing supervised at this pass (every span empty under the mask). The
            # feature is zero, exactly as it is for a pad slot.
            g = torch.zeros_like(h)
        g = g.detach()
        return (g.mean(dim=2) if self._is_hc else g), loss.detach()

    @torch.no_grad()
    def _critic_replay_ce(self, cand: Tensor, exit_h: Tensor, base: Tensor, x0: Tensor,
                          bigram_emb, input_ids: Tensor, labels: Tensor,
                          layout: SlotLayout, keep: Tensor | None, coda_kw: dict | None,
                          ret_reset_mask: Tensor | None, groups: int
                          ) -> tuple[Tensor, Tensor]:
        """``(mean_ce [B, S], scored [B, S])`` — the REAL coda's next-span CE per slot when
        ``cand`` is the state written into each slot's prefix cells.

        THE MEASUREMENT, and nothing else. ``no_grad`` on the decorator: no gradient
        reaches the loop, ``tul.W_prefix``, the coda, the tied table or the span decoder
        from this. ``autograd.grad`` is not used either — there is no graph at all — so the
        claim "the label trains nothing" is structural rather than argued.

        The write goes through the SAME :meth:`TULSlots.prefix_project` and the same
        ``scatter_positions`` the shipped forward uses, and the coda is replayed with the
        SAME ``keep`` (so the token-state dropout draw is the shipped one, reused, not a
        second draw), the same allow relation and the same retention reset. So the only
        difference between two replays is the candidate state, which is what makes the
        pairwise label a within-context comparison.

        ``groups`` (``tul.critic_replay_groups``) is the CONFOUND CONTROL, and the
        confound is real. Under ``tg_coda_prefix_reach: all`` a token of span ``s+1`` reads
        EVERY earlier slot's cells, so a replay that substitutes every slot at once
        attributes span ``s+1``'s CE change to slot ``s`` while every earlier slot's
        substitution also moved it. At ``groups = G`` the substitution is split over ``G``
        replays, each touching slots ``s = g (mod G)`` and leaving every other slot at its
        EXIT state, so the nearest confounder sits ``G`` spans back — at ``G`` times the
        cost. ``G = 1`` is the cheap default and ACCEPTS the noise; the arm's
        pre-registration says so.
        """
        L = input_ids.shape[1]
        w_head = self.embed.lm_weight().detach()
        chunk = self.cfg.ce_chunk_size
        S = layout.slot_index.shape[1]

        # The shipped coda also REPLACES a dropped token's state with `E_mask`
        # (`TULSlots.apply_token_dropout`), not only its injections. `keep` records that
        # draw, so the replay reconstructs the SAME substitution from it and draws no new
        # mask — otherwise the label would be measured on a coda the run never ran.
        # `keep` is 0 at a slot CELL too under the strict geometry (`slot_cell_inject_keep`
        # multiplies into it), hence the `~slot_mask`: a cell must keep the candidate write.
        drop = None
        if keep is not None:
            drop = (keep.reshape(keep.shape[0], keep.shape[1]) == 0) & (~layout.slot_mask)
            if not bool(drop.any()):
                drop = None
        mask_vec = self.tul.E_mask

        def _one(h_in: Tensor) -> tuple[Tensor, Tensor]:
            values, pos = self.tul.prefix_project(h_in, layout, L)
            xc = scatter_positions(base, pos, values)
            if drop is not None:
                xc = torch.where(drop.view(*drop.shape, *([1] * (xc.dim() - 2))),
                                 mask_vec.to(xc.dtype), xc)
            xh = self._back_region(xc, x0, bigram_emb, input_ids, inject_keep=keep,
                                   attn_kwargs=coda_kw, ret_reset_mask=ret_reset_mask)
            _y, scored, ce = slot_outcome_labels(xh, labels, layout, w_head, chunk)
            return ce, scored

        if groups <= 1:
            return _one(cand)
        ce_out = cand.new_zeros(cand.shape[0], S, dtype=torch.float32)
        sc_out = torch.zeros_like(layout.slot_valid)
        sel_all = torch.arange(S, device=cand.device)
        for g in range(groups):
            sel = (sel_all % groups) == g                              # [S]
            selv = sel.view(1, S, *([1] * (cand.dim() - 2)))
            ce_g, sc_g = _one(torch.where(selv, cand, exit_h))
            ce_out = torch.where(sel.unsqueeze(0), ce_g, ce_out)
            sc_out = torch.where(sel.unsqueeze(0), sc_g, sc_out)
        return ce_out, sc_out

    def _tul_critic_loss(self, db_traj, depths: Tensor, h_slots: Tensor, ctx: Tensor,
                         base: Tensor, x0: Tensor, bigram_emb, input_ids: Tensor,
                         labels: Tensor, layout: SlotLayout, keep: Tensor | None,
                         coda_kw: dict | None, ret_reset_mask: Tensor | None,
                         stats: dict) -> Tensor | None:
        """``tul.grad_pass_energy='critic'`` — the critic's OWN training loss.

        THE LABEL IS WITHIN CONTEXT. ``disc``'s label is "is this slot's next span below
        the BATCH MEDIAN CE?", which mostly reads how predictable the next span happens to
        be. Wolfe, 2026-09-12: "a within-context critic that scores whether the state after
        pass t beats the state after pass t-1 on the same coda loss." Here two candidate
        states for the SAME slot are each written into that slot's prefix cells, the REAL
        coda is replayed, and the next span's mean token CE is measured. Everything the two
        candidates share — the row, the context, the span, the decoder — cancels.

        THE CANDIDATES, three states and two pairs:

        * ``h_{t-1}`` and ``h_t`` for a per-slot ``t`` drawn uniformly in
          ``[1, realised depth]``. "Did this pass help?"
        * ``h_t`` and ``h_t + eps * rms(h_t) * n``, ``n ~ N(0, I)`` and
          ``eps = tul.critic_eps``. "Which way is up from here?" Without it the critic only
          ever sees states the loop already produces and has no reason to be smooth
          anywhere else, which is exactly where its gradient is read.

        THE LOSS is :meth:`CriticEnergy.pairwise` on each pair — pairwise logistic on the
        SCORE DIFFERENCE, weighted by the measured CE gap, so a pair the coda cannot tell
        apart teaches nothing — averaged over the two pairs. ``z`` is DETACHED at every
        site below: the critic trains its own parameters and nothing else, and the only
        route from it into the loop is the detached feature crossing ``W_g``.

        THE READINGS. ``critic_agree`` is the arm's honesty instrument, the twin of
        ``egrad_auc``: the CE-weighted fraction of pairs the critic already ranks
        correctly. A critic at 0.5 means the energy carries nothing and the arm is its
        ruler with an extra injection channel. ``critic_gap_traj`` is the MEAN
        ``CE_{t-1} - CE_t`` over the scored slots — the measured worth of one pass, which
        every earlier reading has put near zero, so it is a number worth watching in its
        own right.

        COST. ``3 * G`` coda forwards per step (``G = tul.critic_replay_groups``), no
        backward through any of them, plus one no-grad per-token readout each. The whole
        label computation sits inside an RNG save/restore, so the perturbation draw
        consumes nothing from the run's stream.

        ``None`` when the batch has no scored slot, or on a step ``tul.critic_every`` skips.
        """
        eg = self.tul_egrad
        assert isinstance(eg, CriticEnergy) and db_traj is not None
        T = len(db_traj) - 1
        if T < 1:
            return None
        tc = self.cfg.tul
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state() if h_slots.is_cuda else None
        try:
            with torch.no_grad():
                valid = layout.slot_valid
                # t uniform in [1, depth]: `depths` is the REALISED per-slot depth, so
                # `db_traj[t]` is a state that pass actually produced and `db_traj[t-1]`
                # the one it started from. A pad slot has depth 1 and is masked out below.
                u = torch.rand(depths.shape, device=depths.device, dtype=torch.float32)
                d1 = depths.clamp(min=1)
                # u < 1, so floor(u * d) is 0 .. d-1 and t lands in [1, d]: `db_traj[t]`
                # is a state that pass actually produced and `db_traj[t-1]` the one it
                # started from. The clamp to T covers a batch whose loop ran shallower
                # than a slot's recorded depth (it cannot, and a silent index error here
                # would be worse than a redundant clamp).
                t_idx = torch.minimum(1 + (u * d1.float()).long(), d1).clamp(1, T)
                exit_h = h_slots.detach()
                h_a = torch.zeros_like(exit_h)                       # h_{t-1}
                h_b = torch.zeros_like(exit_h)                       # h_t
                for j in range(1, T + 1):
                    sel = (t_idx == j) & valid
                    selv = sel.view(*sel.shape, *([1] * (exit_h.dim() - 2)))
                    h_a = torch.where(selv, db_traj[j - 1].detach(), h_a)
                    h_b = torch.where(selv, db_traj[j].detach(), h_b)
                # A slot whose draw landed outside [1, T] (none by construction) and every
                # pad slot keeps the exit state, so its replay is the ruler's and its pair
                # carries a zero gap — it is masked out of the loss anyway.
                unset = ~valid
                unsetv = unset.view(*unset.shape, *([1] * (exit_h.dim() - 2)))
                h_a = torch.where(unsetv, exit_h, h_a)
                h_b = torch.where(unsetv, exit_h, h_b)
                # The perturbation pair, around the SAME h_t. `rms` is per slot over the
                # whole carrier, so the step size is relative to the state it starts from
                # and `critic_eps` reads as a fraction.
                rms = h_b.float().flatten(2).pow(2).mean(-1).sqrt()          # [B, S]
                noise = torch.randn_like(h_b.float())
                noise = noise / noise.flatten(2).pow(2).mean(-1).sqrt().view(
                    *rms.shape, *([1] * (h_b.dim() - 2))).clamp(min=1e-6)
                h_p = h_b + (float(tc.critic_eps) * rms).view(
                    *rms.shape, *([1] * (h_b.dim() - 2))).to(h_b.dtype) * noise.to(h_b.dtype)

                G = int(tc.critic_replay_groups)
                rep = dict(base=base, x0=x0, bigram_emb=bigram_emb, input_ids=input_ids,
                           labels=labels, layout=layout, keep=keep, coda_kw=coda_kw,
                           ret_reset_mask=ret_reset_mask, groups=G)
                ce_a, sc_a = self._critic_replay_ce(h_a, exit_h, **rep)
                ce_b, sc_b = self._critic_replay_ce(h_b, exit_h, **rep)
                ce_p, sc_p = self._critic_replay_ce(h_p, exit_h, **rep)
                keep_traj = valid & sc_a & sc_b
                keep_pert = valid & sc_b & sc_p
            if not bool(keep_traj.any()) and not bool(keep_pert.any()):
                return None
            z_a = self._readout(h_a).detach()
            z_b = self._readout(h_b).detach()
            z_p = self._readout(h_p).detach()
            c = ctx.detach()
            l_t, n_t, ag_t = eg.pairwise(z_b, z_a, c, ce_b, ce_a, keep_traj)
            l_p, n_p, ag_p = eg.pairwise(z_b, z_p, c, ce_b, ce_p, keep_pert)
            loss = 0.5 * (l_t + l_p)
        finally:
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state(rng_cuda)
        stats["critic_train"] = float(loss.detach())
        stats["critic_agree"] = float(0.5 * (ag_t + ag_p).detach())
        stats["critic_agree_traj"] = float(ag_t.detach())
        stats["critic_agree_pert"] = float(ag_p.detach())
        stats["critic_n_traj"] = float(n_t.detach())
        stats["critic_n_pert"] = float(n_p.detach())
        # The measured worth of ONE pass, through the real coda: CE(h_{t-1}) - CE(h_t).
        # Positive means the pass helped. Every earlier reading of the loop's per-pass
        # value has put this near zero; it is the number this arm is really about.
        _w = keep_traj.to(ce_a.dtype)
        stats["critic_gap_traj"] = float(
            ((ce_a - ce_b) * _w).sum() / _w.sum().clamp(min=1.0))
        _wp = keep_pert.to(ce_p.dtype)
        stats["critic_gap_pert"] = float(
            ((ce_p - ce_b) * _wp).sum() / _wp.sum().clamp(min=1.0))
        stats["critic_replays"] = float(3 * int(tc.critic_replay_groups))
        return loss

    def _egrad_train_loss(self, h_slots: Tensor, input_ids: Tensor, layout: SlotLayout,
                          ctx: Tensor | None, xh: Tensor, labels: Tensor,
                          stats: dict) -> Tensor | None:
        """The energy module's OWN training loss, on a STOP-GRADIENT copy of ``z``.

        Returns ``None`` when there is no energy module (``grad_pass_energy='own_mux'``).
        ``z`` is detached here and nowhere else, which is the single statement of the
        contract "the energy cannot shape the loop except through ``W_g``";
        ``tests/test_tul_egrad.py`` proves it by autograd rather than by reading this.

        The energy is fitted at the loop's EXIT state and then read at every pass, so the
        feature at pass 0 is an extrapolation. That is a deliberate cost trade — fitting at
        every pass would double the arm's extra FLOPs — and it is named in the arm's
        pre-registration under "Not verified" rather than hidden here.

        ``recon`` reconstructs the slot's own span from the detached exit ``z``.
        ``disc`` is a BCE classifier against the MEASURED outcome: the coda's mean token CE
        over the slot's NEXT span, thresholded at the batch median, which is exactly the
        label the Step-0 linear probe fits.
        """
        eg = self.tul_egrad
        if eg is None:
            return None
        if isinstance(eg, CriticEnergy):
            # The critic's own loss is built at its OWN site (`_tul_critic_loss`), because
            # its label needs the coda's inputs — `base`, the dropout `keep`, the allow
            # relation — and a REPLAY of `_back_region`, none of which this signature has.
            # It is exposed as `critic` / `critic_weighted`, not as `egrad`, so the two
            # scorers' series never share a name. Returning None here leaves the `egrad`
            # block in `_forward_tul` adding nothing.
            return None
        tc = self.cfg.tul
        z = (self._readout(h_slots) if tc.mux_readout == "mean"
             else self._readout_per_stream(h_slots)).detach()
        if isinstance(eg, ReconEnergy):
            w_tied = self.embed.lm_weight()
            w_head = w_tied.detach() if tc.mux_detach_head else w_tied
            loss = eg.loss(z, input_ids, layout, w_head, w_tied.detach(),
                           self.cfg.ce_chunk_size, tc.slot_id,
                           slot_keep=layout.slot_valid)
            stats["egrad_train"] = float(loss.detach())
            return loss
        assert ctx is not None
        y, scored, _ce = slot_outcome_labels(xh.detach(), labels, layout,
                                             self.embed.lm_weight().detach(),
                                             self.cfg.ce_chunk_size)
        loss = eg.bce(z, ctx.detach(), y, scored)
        stats["egrad_train"] = float(loss.detach())
        stats["egrad_pos_frac"] = float((y * scored.to(y.dtype)).sum()
                                        / scored.sum().clamp(min=1))
        stats["egrad_auc"] = float(DiscEnergy.train_auc(z, ctx.detach(), y, scored,
                                                        eg.score))
        return loss

    def _tul_row_contrast_loss(self, h_slots: Tensor, x: Tensor, layout: SlotLayout,
                               stats: dict | None = None) -> Tensor:
        """``tul.row_contrast_lambda`` -- can a row's next spans be told apart by the
        states that are supposed to forecast them?

        THE DEFECT IT IS BUILT ON. A row's written slot states sit at effective rank 5.76
        in 1024 dimensions with mean pairwise cosine 0.71 on the strict ruler. SIGReg and
        the row centering attack that GEOMETRY directly; this term attacks the JOB
        instead. It hands the row's slots a retrieval task that indistinguishable states
        cannot do, and reports the score.

        THE ANCHOR, THE TARGET AND THE NEGATIVES.

        * anchor -- slot ``i``'s exit state, read at the SAME seam the MUX and the span
          decoder read, through the SAME ``_readout`` stream mean, then through the term's
          own ``W_contrast``. LIVE: the gradient reaches the loop, the seed and the core.
        * target -- span ``i+1``'s pooled PRELUDE token states, i.e. the mean over that
          span's token positions of ``_readout(x)`` where ``x`` is the prelude output THIS
          forward already built under ``_tul_tg_kwargs``'s relation. ``x`` is DETACHED
          before the readout, so the target path trains ``W_contrast`` and nothing else --
          not the prelude, not ``lm_mixer``, not the embedding table (the
          ``mux_detach_head`` rule: ``embed.lm_weight()`` IS the input embedding table and
          an auxiliary that writes into it is what made arm v1a diverge at step 2800).
        * negatives -- the OTHER valid slots of the SAME row, and nothing else. Two rows
          of a batch never mix, and a pad slot is never a key or an anchor.

        WHY THE PRELUDE'S STATES AND NOT THE SPAN DECODER'S TOKEN IDS. The decoder already
        grades ``z`` token by token; a second token-level term would be the same
        supervision at a different temperature. This one grades ``z`` against a
        REPRESENTATION of the span and scores it by RANKING within the row, which is what
        makes ``row_contrast_acc`` a distinctness reading rather than a second CE.

        No reader is added and no reader is changed: the term is a loss only. The state the
        coda gets is the one it would have got, including under
        :class:`~morph.model.tul.TULCenterExit` (the centering runs upstream of this seam,
        so the two levers compose and this term scores the centered state).
        """
        con = self.tul_contrast
        assert con is not None
        z = self._readout(h_slots)                                      # [B, S, C]
        # Detach BEFORE the readout: `_readout` ends in `lm_mixer` + `final_norm`, both
        # trained modules, and detaching after would train them from the target path.
        tok = self._readout(x.detach())                                 # [B, L, C]
        pool, ok = next_span_pool(tok, layout)                          # [B, S, C], [B, S]
        loss, acc, n_rows = con(z, pool, ok)
        if stats is not None:
            # `row_contrast_acc` is the number to read: 1/n_valid is chance, and the term
            # itself is log(n_valid) at chance. `row_contrast_n_anchors` is the count the
            # two are averaged over, so a reading cannot be mistaken for a full batch when
            # most rows were dropped for having a single anchor.
            stats["row_contrast_acc"] = float(acc)
            stats["row_contrast_n_rows"] = float(n_rows)
            stats["row_contrast_n_anchors"] = float(ok.sum())
        return loss

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

    def _tul_tg_kwargs(self, layout: SlotLayout) -> tuple[dict | None, Tensor | None,
                                                         dict | None, Tensor | None]:
        """``(front_kw, front_reset, coda_kw, coda_reset)`` — the TG restriction of ONE forward.

        The ONE home of the relation the prelude and the coda run under
        (docs/tul-tg-spec.md §§1-4). :meth:`_forward_tul` reads it, and so must every
        instrument that rebuilds the front outside the forward
        (``lab/divergence/spandec_horizon_grid.py``, ``core_token_aux_probe.py``,
        :meth:`tul_slot_state_probe`): a bare ``_tul_front(input_ids, layout)`` on a
        ``tg_geometry="strict"`` or ``tg_restrict_scope="all"`` model runs the prelude
        UNRESTRICTED and every state downstream of it is off-distribution. The horizon
        grid did exactly that on 2026-09-13 and read the depth effect with the wrong sign
        (`tests/test_spandec_horizon_grid.py`, the strict twin of the exit-column test).

        All four are ``None`` on a ``tg_restrict=false`` model (bit-identical, spec T4).
        ``front_kw`` is ``coda_kw`` under ``restrict`` at scope ``all``, ``None`` at scope
        ``coda``, and a DIFFERENT dict under ``strict`` (the prelude is same-span only).
        """
        tc = self.cfg.tul
        # tg_attn_kwargs feeds the window branch's extra_mask (tg_allow) and the
        # compressed branch's slot mask; tg_reset feeds the GLA segment reset. Both
        # None on a tg_restrict=false model (bit-identical, spec T4).
        tg_attn_kwargs = tg_reset = None
        _strict_front_kw = None
        if self._tg_strict:
            # ── STRICT (tul.tg_geometry) ──────────────────────────────────────────
            # The prelude and the coda get DIFFERENT relations, so they get different
            # kwarg dicts — the one place in this forward where `_front_kw` is not
            # `tg_attn_kwargs`. `tg_allow` narrows the window branch and `tg_comp_allow`
            # the compressed one (both branches, one relation: the F1 defect class). The
            # conv and the value shift are reset at every segment — a span's tokens, its
            # own cells, the next span's tokens — and the retention carry is reset on the
            # same partition rather than on `bag_id`, which does not separate a span from
            # its own cells. `tg_strict_allow` carries the relation and the reasoning.
            _seg = tg_segment_ids(layout)
            _pre_allow = tg_strict_allow(layout, "prelude")
            _coda_allow = tg_strict_allow(layout, "coda",
                                          coda_prefix_reach=tc.tg_coda_prefix_reach)
            _strict_front_kw = {"tg_allow": _pre_allow, "tg_slot_mask": layout.slot_mask,
                                "tg_comp_allow": _pre_allow, "tg_seg": _seg}
            tg_attn_kwargs = {"tg_allow": _coda_allow, "tg_slot_mask": layout.slot_mask,
                              "tg_comp_allow": _coda_allow, "tg_seg": _seg}
            tg_reset = tg_reset_from_ids(_seg)
        elif self._tg_restrict:
            # tg_restrict_scope="coda" (TULConfig): the mask reaches the coda only and
            # carries the coda rule (a slot cell attends slot cells only, so it stays z).
            # "all" is the shipped TG path, bit-identical.
            _coda_scope = tc.tg_restrict_scope == "coda"
            tg_allow = tg_allow_mask(layout, soft_prev_span=tc.tg_soft_prev_span,
                                     slot_queries_slots_only=_coda_scope)
            tg_attn_kwargs = {"tg_allow": tg_allow, "tg_slot_mask": layout.slot_mask}
            if _coda_scope:
                # The conv / value-shift reset (attention.segment_causal_conv): a span's
                # tokens, its slot cells and the next span's tokens are three segments.
                tg_attn_kwargs["tg_seg"] = tg_segment_ids(layout)
            if tc.tg_span_comp:
                # E-SAC: per-span pooled compressed branch (attention.py
                # _tg_span_attention). Built once per forward like tg_allow.
                _tok_sel = ~layout.slot_mask
                tg_attn_kwargs["tg_span"] = {
                    "bag_id": layout.bag_id, "token_sel": _tok_sel,
                    "span_end": boundary_token_index(
                        layout.bag_id, _tok_sel, layout.max_slots)}
            tg_reset = tg_reset_mask(layout)

        if self._tg_strict:
            _front_kw, _front_reset = _strict_front_kw, tg_reset
        else:
            _front_kw = tg_attn_kwargs if tc.tg_restrict_scope == "all" else None
            _front_reset = tg_reset if tc.tg_restrict_scope == "all" else None
        return _front_kw, _front_reset, tg_attn_kwargs, tg_reset

    def _forward_tul(self, input_ids: Tensor, labels: Tensor | None,
                     layout: SlotLayout, plan_nats: bool, halt: bool = False,
                     plan_mode: str = "normal",
                     tul_step_mode: str | None = None,
                     slot_depths: Tensor | None = None,
                     code_mode: str | None = None,
                     code_steps: int | None = None,
                     code_seed: int | None = None,
                     code_given: Tensor | None = None,
                     code_given_mask: Tensor | None = None,
                     coda_state_only: bool = False) -> dict:
        """The TUL forward (docs/tul-spec.md §3). One shared position axis.

        ``coda_state_only`` (eval only, ``labels=None``): return the coda's readout as
        ``out["coda_state"]`` ``[B, L, C]`` and compute NO logits. The graded-continuation
        sampler (``tul.code_grade``, spec §17.2) reads ~64 positions of a 1024-position row
        per decode step; materialising ``[B, L, V]`` first would cost 806 MB per step on a
        card with about 1 GB of slack.

        ``code_mode`` / ``code_steps`` (TUL-Code, eval only): which cells the coda reads —
        ``"encoder"`` (the ground-truth codes, `val/ce_tf`), ``"sampled"`` (k Euler steps
        with the encoder's codes as the tape, the default at eval), ``"rolled"`` (the tape
        sampled slot by slot, the generation regime). ``None`` is the shipped path.

        ``tul_step_mode`` (faithful DiffusionBlocks, morph/model/iter_cond.py) is a
        per-forward DATA argument, the ``slot_layout`` pattern: ``None`` (default) is
        BIT-IDENTICAL to before this parameter existed. ``"db1"`` selects the one-pass
        training step (:meth:`_tul_core_db1`) in place of the T-iteration loop; it is
        a TRAINING-time selector only — at eval (``not self.training``) a model built
        with ``tul.core_stage_cond="sigma"`` runs the deterministic Euler-ladder
        (:meth:`_tul_core_db1_ladder`) unless ``tul_step_mode='bptt'`` explicitly opts
        into the plain ``_tul_core`` loop (the forced-depth sweep's path), matching the
        paper's "trained single-pass, sampled with the original K-iteration procedure".
        """
        # Arc E10 loop terms: the fixed-point term lives in `_core_region`, which the paid
        # loop runs unchanged, so it is stashed there and consumed at the end of this
        # forward. A slot-loop or gain-penalty model never gets here (rejected at build).
        self._core_aux = None
        if self.tul is None:
            raise RuntimeError(
                "forward(slot_layout=...) requires a model built with MORPHConfig(tul=...); "
                "this model has no TUL parameters (E_slot / E_mask / W_prefix)."
            )
        tc = self.cfg.tul
        if tc.code:
            if slot_depths is not None:
                raise NotImplementedError(
                    "slot_depths on a code model: no slot loop runs, so there is no per-slot "
                    "depth to force. The eval dial is code_steps.")
            if tul_step_mode is not None:
                raise NotImplementedError(
                    "tul_step_mode on a code model: the thinker is one velocity pass at "
                    "train and code_steps Euler passes at eval; there is no db1/bptt choice.")
            if self.training and (code_mode is not None or code_steps is not None
                                  or code_seed is not None):
                raise ValueError(
                    "code_mode / code_steps are EVAL-ONLY: at train the phase decides what "
                    "the coda reads (spec §6).")
            if plan_mode == "wrong_seed":
                raise NotImplementedError(
                    "plan_mode='wrong_seed' on a code model: the seed feeds the thinker, not "
                    "the cells, so the reading would mean something else (spec §8).")
        elif tc.code_target:
            if code_steps is not None or code_seed is not None:
                raise NotImplementedError(
                    "code_steps / code_seed on a code-target model: the cells are the slot "
                    "loop's projection and there is no sampler; the eval dial is slot_depths.")
            if code_mode not in (None, "encoder", "generate"):
                raise NotImplementedError(
                    f"code_mode={code_mode!r} on a code-target model: only 'encoder' (E's "
                    f"own code in the cells, the ceiling) and 'generate' (with code_given) "
                    f"exist here.")
            if self.training and (code_mode is not None or code_given is not None):
                raise ValueError(
                    "code_mode / code_given are EVAL-ONLY on a code-target model: at train "
                    "the coda reads the projection's cells.")
        elif (code_mode is not None or code_steps is not None or code_given is not None
              or code_seed is not None):
            raise ValueError("code_mode / code_steps / code_given need a model built with "
                             "tul.code=true.")
        if coda_state_only and (labels is not None or self.training):
            raise ValueError(
                "coda_state_only is EVAL-ONLY and takes labels=None: it replaces the logits "
                "with the coda readout, so there is nothing to score.")
        _code_cells_out = None
        _code_z_out = None
        code_target_loss, code_target_stats = None, {}
        code_grade_loss, code_grade_stats = None, {}
        if slot_depths is not None:
            # The SAME rule tul_step_mode='db1' states above: every branch of this forward
            # that never reaches `_tul_core` would ignore the table in silence, so each is
            # named and raises. A sigma-conditioned model at eval takes the Euler ladder
            # unless the caller asks for 'bptt', so that combination is refused too.
            if tc.tokens_through_core:
                raise NotImplementedError(
                    "slot_depths has no meaning on the paid loop (tul.tokens_through_core): "
                    "it runs the ordinary per-SAMPLE core over every position and has no "
                    "per-slot depth. Force model.cfg.mean_depth instead.")
            if self.fm_planner is not None:
                raise NotImplementedError(
                    "slot_depths has no meaning with an FM planner (tul.fm): the planner "
                    "replaces the core loop, so there are no per-slot passes to force.")
            if tul_step_mode == "db1":
                raise NotImplementedError(
                    "slot_depths with tul_step_mode='db1': the db1 step is ONE core "
                    "application by construction, so a depth table would be ignored.")
            if (not self.training and self._core_stage_cond_mode == "sigma"
                    and tul_step_mode != "bptt"):
                raise NotImplementedError(
                    "slot_depths on a sigma-conditioned model at eval: that path runs the "
                    "Euler ladder (`_tul_core_db1_ladder`), whose step count is "
                    "tul.db1_ladder_steps and not a per-slot table. Pass "
                    "tul_step_mode='bptt' to force the plain loop first.")
        if tul_step_mode not in (None, "bptt", "db1"):
            raise ValueError(
                f"tul_step_mode must be one of None, 'bptt', 'db1', got {tul_step_mode!r}")
        if tul_step_mode == "db1":
            if tc.tokens_through_core:
                raise NotImplementedError(
                    "tul_step_mode='db1' has no defined interaction with "
                    "tul.tokens_through_core (A2): A2 takes an earlier branch in "
                    "_forward_tul that never reaches the db1 core — a silent ignore, "
                    "so this raises instead.")
            if self.fm_planner is not None:
                raise NotImplementedError(
                    "tul_step_mode='db1' has no defined interaction with an FM planner "
                    "(tul.fm): the FM branch takes an earlier branch in _forward_tul "
                    "that never reaches the db1 core — a silent ignore, so this raises "
                    "instead.")
            if halt:
                raise NotImplementedError(
                    "tul_step_mode='db1' is a TRAINING step; halt=True is EVAL-only "
                    "(docs/tul-gate-spec.md §7) and the two have no defined interaction.")
        if halt and self._core_stage_cond_mode == "sigma":
            raise NotImplementedError(
                "halt=True (arm TUL-halt) has no defined interaction with "
                "tul.core_stage_cond='sigma': eval on a sigma-conditioned model runs the "
                "deterministic Euler ladder, which drives no gate and reads no halting "
                "decision.")
        if layout.prefix_k != tc.prefix_k:
            raise ValueError(f"layout prefix_k {layout.prefix_k} != model {tc.prefix_k}")
        B, L = input_ids.shape
        if tc.coda_token_cut >= L:
            raise ValueError(
                f"tul.coda_token_cut={tc.coda_token_cut} >= seq_len {L} "
                f"(.agents/notes/implemented/architecture/2026-08-18-tul-compaction-window.md): every token position would be "
                f"dropped from the coda, leaving nothing to predict. Lower the cut."
            )

        # ── TG restriction (docs/tul-tg-spec.md §§1-4) — built ONCE per forward, in
        # `_tul_tg_kwargs`, the ONE home every instrument that rebuilds the front
        # must read (2026-09-13: the horizon grid rebuilt it bare and scored a
        # strict model from an unrestricted prelude).
        _front_kw, _front_reset, tg_attn_kwargs, tg_reset = self._tul_tg_kwargs(layout)
        x, x0, bigram_emb = self._tul_front(input_ids, layout,
                                            attn_kwargs=_front_kw,
                                            ret_reset_mask=_front_reset)

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
        # A2s (2026-09-02): tg_restrict × tokens_through_core is now DEFINED — the
        # restriction means the same thing in the core as in the prelude/coda: the
        # window branch carries tg_allow (same-span-or-slot) and the compressed
        # branch carries tg_slot_mask (slot K/V only). _core_region threads the
        # masks through the active-set sort. Prereg:
        # lab/experiments/planned/2026-09-02-a2s-restricted-paid-loop.md.
        # The coda's base carrier, kept for the `critic` energy's replay: the tensor
        # `prefix_project`'s write is scattered INTO, so a replay that swaps the write for
        # a candidate state differs from the shipped coda in that write and nothing else.
        # None on the paid loop and the FM planner, both of which the critic refuses.
        _critic_base = None
        code_fm_loss, code_stats = None, {}
        # `tul.prefix_source="trajectory"`'s PAD cells at their row positions, or None on
        # every other model. Bound here rather than in the slot-loop branch because the
        # coda's key set is narrowed after the branch dispatch, beside `all_slots`.
        _pad_pos = None
        # LXTUL's fan (tul.fan_k): the repulsion term and the arm's per-pass statistics.
        # Bound here, beside `_pad_pos`, because the fold that adds the term to the loss
        # and the eval-only oracle both sit AFTER the branch dispatch, and only the
        # slot-loop branch can produce them (`fan_k` inherits the register's refusal of
        # `tokens_through_core`, the FM planner and the code core). `None` / `{}` on every
        # other path is the signal those folds branch on.
        fan_repel_loss = None
        fan_stats: dict[str, float] = {}
        _fan_cells = None
        if tc.tokens_through_core:
            # Arm A2 (slots-as-memory): tokens AND slots run the ordinary per-SAMPLE core.
            # RESOLVED SPEC AMBIGUITY — §7.1's A2 row says "Poisson/slot" in the depth
            # column but "uniform depth" in the isolates column. A2 must differ from A0 by
            # the presence of slots ALONE (it isolates C2), so it reuses today's core
            # region unchanged; a per-position Poisson depth would change two things at once.
            if plan_mode != "normal":
                raise ValueError(
                    f"plan_mode={plan_mode!r} is a slot-loop / FM-planner ablation: the paid "
                    f"loop has no separate plan tensor to zero or shuffle (the slot IS a "
                    f"looped position). Raises rather than reporting a zero-by-construction "
                    f"plan_worth.")
            # Tail-pad slot positions (slot_mask at the dump bin) carry E_slot alone and no
            # label; they are looped like every other position but excluded from the
            # Jacobian probe's active set (a [B, L] mask, read only under capture).
            pad_pos = layout.slot_mask & (layout.bag_id == layout.max_slots)
            x_coda = self._core_region(x, x0, bigram_emb, input_ids,
                                       attn_kwargs=tg_attn_kwargs, jac_active=~pad_pos)
            depths, g_traj, mux_loss, sigreg_loss, gain_reg = None, None, None, None, None
            fm_y = fm_geom = fm_ctx = None
            mux_stats = {}
            spandec_loss, spandec_stats, _egrad_src = None, {}, None
            oracle_z_loss, oracle_z_stats = None, {}
            spandec_pass_loss, spandec_pass_stats = None, {}
            horizon_loss, horizon_stats = None, {}
            rcon_loss, rcon_stats = None, {}
            db_traj = mep_keep = None
            # tul.vq_codes is refused with the paid loop at construction (there is no
            # prefix write to lift a code into), so this is always None here; it exists
            # so the loss block below reads ONE name on every branch.
            _vq_out = None
        elif tc.loop_reads_tokens:
            # ── THE LOOP READS TOKENS (tul.loop_reads_tokens, 2026-09-13) ───────────
            # The SHIPPED core stage over EVERY position — tokens and slot cells in ONE
            # sequence, `_core_region`'s per-SAMPLE Poisson depth — under the SPAN
            # relation `causal AND (same span OR j is a slot cell)`. No second core is
            # written: this is the same `_core_region` the plain model, the paid loop and
            # `tul.core_token_aux` run, and the same `_core_token_aux_kwargs` relation the
            # aux runs under, which is why that method is a named method with ONE home.
            #
            # WHAT IS DIFFERENT FROM THE PAID LOOP, in one line: the paid loop's core is
            # UNRESTRICTED, so a token reads every earlier token directly and the cells
            # carry nothing anybody needs. Here a token reaches an earlier span ONLY
            # through a slot cell, at inference as in training, so the cells are still the
            # whole cross-span channel and the loop is what fills them.
            #
            # NO PREFIX WRITE. A cell's looped state is already AT its own position when
            # the core returns, so `prefix_project` would scatter a second copy of a
            # tensor that is already there and `W_prefix` would sit between the loop and
            # the coda for no reason. `z` — what the span decoder and the MUX grade — is
            # `gather_valid` at the slot's FIRST cell, the same seam every other arm
            # reads, so `spandec_ce` stays comparable across the family.
            if plan_mode != "normal":
                raise ValueError(
                    f"plan_mode={plan_mode!r} with tul.loop_reads_tokens: there is no "
                    f"separate plan tensor to zero or shuffle — the cell IS a looped "
                    f"position, exactly as on the paid loop. Raises rather than reporting "
                    f"a plan_worth that is zero by construction.")
            if halt:
                raise NotImplementedError(
                    "halt=True with tul.loop_reads_tokens: the gate stops a PER-SLOT loop "
                    "and this mode runs one per-sample depth over the whole row.")
            if slot_depths is not None:
                raise NotImplementedError(
                    "slot_depths with tul.loop_reads_tokens: the core here is per-SAMPLE, "
                    "so a per-slot depth table would be ignored. Force model.cfg.mean_depth "
                    "instead (the tokens_through_core rule; lab/divergence/_build.py's "
                    "DepthLever and core_depth_sweep.py already do).")
            pad_pos = layout.slot_mask & (layout.bag_id == layout.max_slots)
            x_coda = self._core_region(x, x0, bigram_emb, input_ids,
                                       attn_kwargs=self._core_token_aux_kwargs(layout),
                                       jac_active=~pad_pos)
            h_slots = gather_valid(x_coda, layout.slot_index, layout.slot_valid)
            depths, g_traj, gain_reg = None, None, None
            db_traj = mep_keep = None
            fm_y = fm_geom = fm_ctx = None
            oracle_z_loss, oracle_z_stats = None, {}
            spandec_pass_loss, spandec_pass_stats = None, {}
            horizon_loss, horizon_stats = None, {}
            mux_stats = {}
            mux_loss = (self._tul_mux_loss(h_slots, input_ids, layout, stats=mux_stats)
                        if tc.mux_beta > 0.0 else None)
            spandec_stats = {}
            spandec_loss = (self._tul_spandec_loss(h_slots, input_ids, layout,
                                                   stats=spandec_stats)
                            if self.tul_spandec is not None else None)
            sigreg_loss = (self._tul_sigreg_loss(h_slots, layout)
                           if tc.sigreg_lambda > 0.0 else None)
            # C2 runs here unchanged: this arm HAS a per-slot state at a reader seam, and
            # the term reads it exactly as the span decoder above does. C1 does NOT --
            # `TULConfig` refuses `center_exit` with `loop_reads_tokens`, because this arm
            # writes nothing through `prefix_project` and centering `h_slots` would move
            # the local losses while the coda kept reading the uncentered state.
            rcon_stats = {}
            rcon_loss = (self._tul_row_contrast_loss(h_slots, x, layout,
                                                     stats=rcon_stats)
                         if self.tul_contrast is not None else None)
            _egrad_src = None
            _vq_out = None            # refused with tul.loop_reads_tokens (no write)
        elif self.fm_planner is not None:
            # FM1 (morph/model/tul_fm.py). The planner replaces the core loop; the plan
            # is DETACHED before it reaches W_prefix, so the coda's CE never touches the
            # ladder and there is no BPTT through an iterated map.
            xn, h_slots, fm_y, fm_geom, fm_ctx = self._tul_fm_core(x, layout)
            depths, g_traj, mux_loss, sigreg_loss, gain_reg = None, None, None, None, None
            mux_stats = {}
            spandec_loss, spandec_stats, _egrad_src = None, {}, None
            oracle_z_loss, oracle_z_stats = None, {}
            spandec_pass_loss, spandec_pass_stats = None, {}
            horizon_loss, horizon_stats = None, {}
            _vq_out = None            # refused with an FM planner (no looped exit state)
            rcon_loss, rcon_stats = None, {}
            h_slots = self._tul_plan_ablate(h_slots, layout, plan_mode)
            values, pos = self.tul.prefix_project(h_slots, layout, L)
            x_coda = scatter_positions(xn, pos, values)
        elif tc.code:
            # ── TUL-Code (docs/tul-code-spec.md §4): the cells ARE the code ───────────
            fm_y = fm_geom = fm_ctx = None
            _scorer = None
            if (self.training and tc.code_xm_k > 1 and tc.code_xm_select == "coda"
                    and labels is not None):
                _scorer = self._tul_code_span_scorer(x0, bigram_emb, input_ids, labels, layout,
                                                     tg_attn_kwargs, tg_reset, L)
            xn, _cells, code_fm_loss, code_stats, h_slots, depths = self._tul_code_core(
                x, x0, bigram_emb, layout, code_mode=code_mode, code_steps=code_steps,
                plan_mode=plan_mode, code_seed=code_seed, code_given=code_given,
                code_given_mask=code_given_mask, span_scorer=_scorer)
            _code_cells_out = _cells
            g_traj = db_traj = gain_reg = mep_keep = None
            mux_loss, sigreg_loss, mux_stats = None, None, {}
            spandec_loss, spandec_stats, _egrad_src = None, {}, None
            oracle_z_loss, oracle_z_stats = None, {}
            spandec_pass_loss, spandec_pass_stats = None, {}
            horizon_loss, horizon_stats = None, {}
            _vq_out = getattr(self, "_code_vq_out", None)   # LCTUL-D's quantiser terms
            rcon_loss, rcon_stats = None, {}
            # The M cells go 1:1 into the M prefix positions, broadcast over the HC
            # streams, through NO projection (W_prefix is built and inert).
            _B, _S, _M, _C = _cells.shape
            values = _cells.reshape(_B, _S * _M, _C)
            if self._is_hc:
                values = values.unsqueeze(2).expand(-1, -1, self._n_streams, -1).contiguous()
            pos = self.tul.prefix_positions(layout, L)
            base = xn
            _critic_base = base
            x_coda = scatter_positions(xn, pos, values)
        else:
            fm_y = fm_geom = fm_ctx = None
            # ── faithful DiffusionBlocks dispatch (morph/model/iter_cond.py) ────────
            # "db1" is a TRAINING selector; the Euler ladder auto-fires at eval on a
            # sigma-conditioned model regardless of tul_step_mode (docstring above).
            # Both are scoped to THIS branch only — the tokens_through_core/fm_planner
            # branches above already raise on tul_step_mode="db1" rather than silently
            # ignoring it (see the guards at the top of this function).
            if tul_step_mode == "db1":
                xn, h_slots, depths, g_traj, db_traj = self._tul_core_db1(
                    x, x0, bigram_emb, layout)
                gain_reg = mep_keep = None   # one application: no iterated map, no passes
            elif (not self.training and self._core_stage_cond_mode == "sigma"
                  and tul_step_mode != "bptt"):
                # An explicit "bptt" at eval opts OUT of the auto-ladder and runs the
                # plain _tul_core loop (conditioning unused) — this is what the bptt
                # half of a step_mix arm trains, and the forced-depth sweep needs to
                # measure it separately from the sigma/Euler path.
                xn, h_slots, depths, g_traj, db_traj = self._tul_core_db1_ladder(
                    x, x0, bigram_emb, layout)
                gain_reg = mep_keep = None   # eval-only ladder: no penalty, no passes
            else:
                xn, h_slots, depths, g_traj, db_traj, gain_reg, mep_keep = self._tul_core(
                    x, x0, bigram_emb, layout, halt=halt, input_ids=input_ids,
                    slot_depths=slot_depths)
            # ── the Thought Register (tul.slot_cells) ─────────────────────────────
            # `_tul_core` returns the compact CELL axis, [B, S*M, …]. M == 1 — every model
            # before the knob — leaves `_reg_cells` None and this block traces out.
            #
            # `h_slots` becomes the MEAN of a slot's M cells, and that is a decision with a
            # reason: every reader between here and the write (the MUX, the span decoder,
            # SIGReg, the energy) takes ONE state per slot, and handing them the mean keeps
            # each of those mechanisms the SHIPPED one, so the register arm differs from
            # its ruler by the register alone. Giving the span decoder the M cells as a
            # memory to cross-attend to is a second mechanism and is the named follow-up,
            # not this arm. The mean's gradient still reaches every cell.
            # The CODA reads the M cells individually — that is the point of the arm — and
            # it reads them through the ordinary `prefix_project` write, 1:1.
            #
            # Think-once conditioning (arm R7, `tul.cond_layers`) runs BEFORE that mean,
            # on the compact axis straight out of `_tul_core` — the S*M CELL axis on a
            # register model, the S slot axis on every other one — because the stack has
            # to be the reader of what the loop wrote. On the mean it would be the reader
            # of neither of the register's two readers: the coda's 1:1 prefix write takes
            # cell i and the span decoder takes the cells' mean, and with the stack after
            # the mean the write read the RAW loop cells while only the decoder saw the
            # stack. At `slot_cells: 1` the register block below is a no-op, so this is
            # the placement the stack has always had, bit-identical
            # (tests/test_tul_cond4_strict.py pins it against the pre-change source).
            # Everything downstream — the mux local loss, the span decoder, SIGReg, the
            # gate budget, the plan ablations, prefix_project — reads the stack's output.
            _m = int(tc.slot_cells)
            _S = layout.slot_index.shape[1]
            if self.tul_cond is not None:
                h_slots = self._tul_cond_apply(h_slots, n_slots=_S, m_cells=_m)
            # ── C1: the row-centered exit (tul.center_exit) ────────────────────────
            # Placed HERE, on the compact CELL axis, after the loop and after the
            # think-once stack, and BEFORE the register's mean. Two reasons and both are
            # load-bearing:
            #   * this is the last point that is upstream of EVERY reader. The MUX, the
            #     span decoder, SIGReg, the energy and `prefix_project` all take the state
            #     from below this line, so ONE edit centers what all five see. (The gate's
            #     budget, `detach_z` and the eval plan ablation sit further down and are
            #     applied to the centered state, unchanged.)
            #   * on the CELL axis the centering is per CELL INDEX for free: cell i of
            #     every valid slot against cell i of every other. Below the mean the
            #     register's within-slot axis has already been collapsed and that reading
            #     could not be had.
            # `self.tul_center is None` on every model without the knob, so the line
            # traces out and the forward is bit-identical.
            if self.tul_center is not None:
                h_slots = self.tul_center(h_slots, layout.slot_valid, m_cells=_m)
            _reg_cells = None
            if _m > 1:
                _reg_cells = h_slots.reshape(h_slots.shape[0], _S, _m, *h_slots.shape[2:])
                if self.tul_fan is None:
                    h_slots = _reg_cells.mean(dim=2)
                else:
                    # ── LXTUL: the fan's exit mixture (tul.fan_mix) ───────────────
                    # THE SAME SEAM the register's mean sits at, and for the same reason:
                    # every reader between here and the write (the MUX, the span decoder,
                    # SIGReg, the energy) takes ONE state per slot, so mixing HERE makes
                    # all of them grade the state the coda actually gets. At
                    # `fan_mix: "mean"` the mixture IS `cells.mean(dim=2)` — the register's
                    # own read — so the mean arm differs from the register by its WRITE
                    # alone (single-source, below), which is the width control the
                    # `trajectory-prefix-is-width-plus-pad-artefact` note demands.
                    h_slots, _fan_w = self.tul_fan(_reg_cells)
                    fan_stats["mix_entropy"] = float(
                        TULFanMix.entropy(_fan_w, layout.slot_valid).detach())
                    fan_stats["mix_w_max"] = float(
                        _fan_w[layout.slot_valid].amax(dim=-1).mean().detach()
                        if bool(layout.slot_valid.any()) else 0.0)
                    _fan_cells = _reg_cells
                    # ── the repulsion (tul.fan_repel_lambda, tul.fan_repel_passes) ──
                    # Read off the SAME live-carry trajectory every per-pass reader uses,
                    # on the CELL axis `_tul_core` carries. The penalised passes are the
                    # FIRST ones (PLR Theorem 4.4: the collapse is exponential in depth,
                    # so pass 1 fights `L^2` and pass 6 fights `L^12`); every OTHER pass,
                    # the seed included, is still measured and reported as
                    # `fan/stream_cos_t{t}` under no_grad. Training only — at eval the
                    # cosines are instruments and there is no term.
                    if db_traj is not None:
                        if self.tul_fan_epi is None:
                            _rp = fan_repel_term(db_traj, layout.slot_valid, _m,
                                                 int(tc.fan_repel_passes), stats=fan_stats)
                        else:
                            # `fan_repel_mode: "epi"`: the cosines stay INSTRUMENTS (every
                            # pass, no gradient) and the charged term is minus the
                            # epiplexity of the streams' deviations (`fan_epi_t{t}`).
                            with torch.no_grad():
                                fan_repel_term(db_traj, layout.slot_valid, _m,
                                               int(tc.fan_repel_passes), stats=fan_stats)
                            _rp = fan_epi_term(db_traj, layout.slot_valid, _m,
                                               int(tc.fan_repel_passes), self.tul_fan_epi,
                                               float(tc.fan_epi_ridge), float(tc.fan_epi_eta),
                                               stats=fan_stats)
                        if self.training and tc.fan_repel_lambda > 0.0:
                            fan_repel_loss = _rp
                        if not self.training:
                            # The per-pass RANK, eval only: an eigendecomposition-free
                            # trace formula per slot (`eval-probes-must-not-stall-the-gpu`),
                            # but still K Gram products per pass, so it is not paid on a
                            # training step. Bounded by K; the register read 1.24 of 4.
                            for _t in range(len(db_traj)):
                                fan_stats[f"stream_rank_t{_t}"] = fan_stream_rank(
                                    db_traj[_t], layout.slot_valid, _m)
                depths = depths.reshape(depths.shape[0], _S, _m)[:, :, 0].contiguous()
            # ── the discrete thought (tul.vq_codes; morph/model/tul_vq.py) ─────────
            # THE SAME SEAM the register's mean sits at, and for the same reason: every
            # reader between here and the write (the MUX, the span decoder, SIGReg, the
            # energy) takes ONE state per slot, so replacing `h_slots` with the DEQUANTIZED
            # thought here makes all of them grade the thought the coda actually gets. The
            # K lifted codes are held aside and go 1:1 into the K prefix cells below,
            # exactly as the register's cells do (`prefix_k` is refused unless it equals
            # `vq_codes`). `vq_codes: 0` leaves `self.tul_vq` None and this block traces
            # out — every model before the knob is bit-identical.
            _vq_cells, _vq_out = None, None
            if self.tul_vq is not None:
                _vq_cells, h_slots, _vq_out = self.tul_vq(h_slots, layout.slot_valid)
            # ── the pass-gated readout (tul.pass_readout="gated"; LoopMTP Eq 9-11) ──
            # THE SAME SEAM the register's mean and VQ sit at: "last" — every model
            # before this knob — leaves `h_slots` exactly what `_tul_core` returned and
            # this block traces out, bit-identical. "gated" replaces it with a
            # content-conditional mixture of EVERY realised pass's state, so every
            # reader between here and the write grades and writes the SAME mixed
            # thought instead of only the last pass's.
            if self.tul_pass_gate is not None:
                assert db_traj is not None, (
                    "tul.pass_readout='gated' reached the readout with no trajectory: "
                    "_tul_core must have failed to turn on its collection.")
                h_slots = self.tul_pass_gate(db_traj[1:])
            mux_stats: dict = {}
            if tc.mux_beta <= 0.0:
                mux_loss = None
            elif tc.mux_stage_own_iters > 0:
                # ── staged targets (arc E3; tul.mux_stage_all, arm slot-mnext-staged-all) ─
                # Two jobs in sequence on ONE live-carry trajectory: the state after
                # iteration k toward the span the slot terminates (memory), for the slots
                # whose depth reaches k — or, under `mux_stage_all`, that same memory term
                # at EVERY pass from k to T-1, averaged; and the final state toward the
                # forecast. Mean of the two sides, so mux_beta keeps its meaning. Stats come
                # from the FINAL (forecast) term so mux_rel / mux_kl stay comparable with
                # the unstaged arm; the own term's loss is reported beside them, and under
                # `mux_stage_all` `mux_stage_own` is the MEAN of its terms and
                # `mux_stage_own_n` the count from the FIRST one (pass k, the largest keep
                # set — every later pass supervises a subset of it).
                k = int(tc.mux_stage_own_iters)
                assert db_traj is not None
                own_stats: dict = {}
                if mep_keep is not None and len(db_traj) - 1 > k:
                    # ── tul.mux_stage_all: the own term at EVERY non-final pass ──────
                    # The toy study's winning attachment. Passes k .. T-1 (the FINAL state
                    # is the forecast's, never the memory's), each on the slots whose
                    # realised depth reaches that pass — `mep_keep[j-1]` is the mask for
                    # `db_traj[j]`, the same list `mux_every_pass` uses, so there is ONE
                    # trajectory and ONE set of masks in the tree. Averaged, so the own
                    # side keeps weight 0.5 whatever the batch's depth draw was. `mep_keep`
                    # is None outside training, so an eval forward takes the single-k
                    # branch below and the sweep's two final-state columns are unchanged.
                    idxs = list(range(k, len(db_traj) - 1))
                    own_terms = [self._tul_mux_loss(
                                     db_traj[j], input_ids, layout,
                                     stats=(own_stats if j == idxs[0] else None),
                                     slot_keep=mep_keep[j - 1], target="own")
                                 for j in idxs]
                    own_loss = torch.stack(own_terms).mean()
                    mux_stats["mux_stage_terms"] = float(len(own_terms))
                    # `loop/mux_stage_*` for the trainer's pre-clip probe (the `_loop_prog`
                    # route `mux_every_pass` writes `loop/mux_pass_*` through). Detached
                    # 0-dim tensors: the float() sync happens in the trainer, on the steps
                    # it logs, never inside the forward.
                    self._loop_mux = {"mux_stage_terms": float(len(own_terms))}
                    for j, t_ in zip(idxs, own_terms):
                        self._loop_mux[f"mux_stage_t{j}"] = t_.detach()
                else:
                    # The loop runs to the batch's deepest slot; a batch whose deepest slot
                    # stops short of k supervises nothing on the own term (keep is empty and
                    # the loss is 0), so the index is clamped, never raised on (construction
                    # already refused a k past the configured max depth).
                    own_loss = self._tul_mux_loss(db_traj[min(k, len(db_traj) - 1)],
                                                  input_ids, layout, stats=own_stats,
                                                  slot_keep=(depths >= k), target="own")
                nxt_loss = self._tul_mux_loss(h_slots, input_ids, layout, stats=mux_stats)
                mux_loss = 0.5 * (own_loss + nxt_loss)
                mux_stats["mux_stage_own"] = float(own_loss.detach())
                mux_stats["mux_stage_own_n"] = own_stats.get("mux_n_supervised", 0.0)
                mux_stats["mux_stage_next"] = float(nxt_loss.detach())
                if not self.training:
                    # The depth sweep's columns: BOTH targets read from the FINAL state at
                    # the forced depth (core_depth_sweep.py), so a stage arm's memory and
                    # forecast earning are measured on the same state the reader gets.
                    fin: dict = {}
                    fin_own = self._tul_mux_loss(h_slots, input_ids, layout, stats=fin,
                                                 target="own")
                    mux_stats["mux_local_own_final"] = float(fin_own.detach())
                    mux_stats["mux_n_supervised_own"] = fin.get("mux_n_supervised", 0.0)
                    mux_stats["mux_local_next_final"] = float(nxt_loss.detach())
                    mux_stats["mux_n_supervised_next"] = mux_stats.get("mux_n_supervised", 0.0)
            elif mep_keep is not None:
                # ── the MUX on EVERY pass (tul.mux_every_pass, arm slot-mnext-mux-every-pass) ──
                # The configured target on the state after each pass of a LIVE carry (no
                # detach anywhere — `_tul_core` keeps the outer-graph states), for the slots
                # whose realised depth reaches that pass, PLUS the final state for every
                # valid slot exactly as the ruler supervises it. Uniform weights summing to
                # 1, so `mux_beta` means what it means on every other arm; stats come from
                # the FINAL term so `mux_local` / `mux_rel` / `mux_kl` stay comparable.
                # `mep_keep` is None outside training, so an eval forward takes the branch
                # below and the forced-depth sweep reads the same column as the ruler.
                terms = [self._tul_mux_loss(db_traj[j], input_ids, layout,
                                            slot_keep=mep_keep[j - 1])
                         for j in range(1, len(db_traj))]
                terms.append(self._tul_mux_loss(h_slots, input_ids, layout,
                                                stats=mux_stats))
                mux_loss = torch.stack(terms).mean()
                mux_stats["mux_pass_terms"] = float(len(terms))
                # `loop/mux_pass_*` for the trainer's pre-clip probe (the `_loop_prog`
                # route). Detached 0-dim tensors: the float() sync happens in the trainer,
                # on the steps it logs, never inside the forward.
                self._loop_mux = {"mux_pass_terms": float(len(terms))}
                for j, t_ in enumerate(terms[:-1], start=1):
                    self._loop_mux[f"mux_pass_t{j}"] = t_.detach()
                self._loop_mux["mux_pass_final"] = terms[-1].detach()
            elif db_traj is None or not tc.db_loop:
                mux_loss = self._tul_mux_loss(h_slots, input_ids, layout, stats=mux_stats)
            else:
                # ── db_loop local losses ──────────────────────────────────────────
                # Evenly spaced supervised iterations, always including the seed (t=0)
                # and the final state; each state's grad reaches core application t and
                # the live seed injection ONLY (the carry is detached in _tul_core).
                # Weights sum to 1 so mux_beta means the same thing as in gl1b; stats
                # come from the FINAL state so mux_rel stays comparable across arms.
                T_states = len(db_traj)                                  # T+1 entries
                n_pick = max(2, min(tc.db_mux_iters, T_states))
                idxs = sorted({round(i * (T_states - 1) / (n_pick - 1))
                               for i in range(n_pick)})
                terms = []
                for t_i in idxs:
                    # Seed and FINAL states supervise every valid slot (a frozen slot's
                    # final state keeps its graph through the where-carry, so the grad
                    # still reaches that slot's last application). Intermediate picks
                    # supervise only slots still fresh at that iteration.
                    keep = None if t_i in (0, idxs[-1]) else (depths >= t_i)
                    st = mux_stats if t_i == idxs[-1] else None
                    terms.append(self._tul_mux_loss(db_traj[t_i], input_ids, layout,
                                                    stats=st, slot_keep=keep))
                mux_loss = torch.stack(terms).mean()
                mux_stats["mux_db_n_iters"] = float(len(idxs))
            # ── the span decoder (tul.spandec) ────────────────────────────────
            # Read at the SAME seam as the MUX: the loop's exit state, before the gate's
            # budget conditioning, before `detach_z` (refused with this knob) and before
            # the eval-only plan ablation. So the state the decoder grades is the state
            # the coda reads.
            spandec_stats: dict = {}
            # `cells` is the register's M looped cells (None on every other model), handed
            # in so `tul.spandec_reads_cells` can cross-attend to exactly what
            # `prefix_project` below writes into the coda. Read at the same seam as
            # `h_slots`: after the think-once stack, before the eval-only plan ablation.
            spandec_loss = (self._tul_spandec_loss(h_slots, input_ids, layout,
                                                   stats=spandec_stats, cells=_reg_cells)
                            if self.tul_spandec is not None else None)
            # ── C2: the within-row contrastive term (tul.row_contrast_lambda) ──
            # Read at the SAME seam as the MUX and the span decoder: the loop's exit
            # state, before the gate's budget, before `detach_z` and before the eval-only
            # plan ablation. `x` is the prelude output THIS forward already computed
            # under `_front_kw` -- no second front is built, so the term cannot score a
            # strict model from an unrestricted prelude (the 2026-09-13 horizon-grid
            # defect; `_tul_tg_kwargs` is the ONE home).
            rcon_stats: dict = {}
            rcon_loss = (self._tul_row_contrast_loss(h_slots, x, layout,
                                                     stats=rcon_stats)
                         if self.tul_contrast is not None else None)
            # ── the oracle-z per-pass teacher (tul.oracle_z) ──────────────────
            # Read on the SAME trajectory `mux_every_pass` and the staged target use, and
            # built here because it needs the realised per-slot depths beside it. Training
            # only and `spandec`-only, both enforced at construction; `db_traj` is None on
            # an eval forward, so a forced-depth sweep never pays for it.
            oracle_z_loss, oracle_z_stats = None, {}
            if tc.oracle_z and self.training and db_traj is not None:
                oracle_z_loss = self._tul_oracle_z_loss(db_traj, depths, input_ids, layout,
                                                        stats=oracle_z_stats)
            # ── the per-pass planning target (tul.spandec_per_pass) ───────────
            # The same trajectory, the same realised depths, the same training-only rule.
            # It does NOT replace the exit term above: the exit stays the shipped H = 1
            # "next thought" and this adds one growing target per pass.
            spandec_pass_loss, spandec_pass_stats = None, {}
            if tc.spandec_per_pass and self.training and db_traj is not None:
                spandec_pass_loss = self._tul_spandec_per_pass_loss(
                    db_traj, depths, input_ids, layout, stats=spandec_pass_stats)
            # ── LoopMTP's horizon-indexed alignment (tul.horizon_weight) ──────
            # The same trajectory, training-only, needs no decoder (a cosine loss
            # against a detached tied-embedding target, not a cross-entropy).
            horizon_loss, horizon_stats = None, {}
            if tc.horizon_weight > 0.0 and self.training and db_traj is not None:
                horizon_loss = self._tul_horizon_loss(
                    db_traj, input_ids, layout, stats=horizon_stats)
            # The energy module trains on THIS state, detached, at the same seam every
            # other reader of z uses — but its loss is built at the END of the forward,
            # because the `disc` critic's label is the coda's own CE over the next span.
            _egrad_src = h_slots
            sigreg_loss = (self._tul_sigreg_loss(h_slots, layout)
                           if tc.sigreg_lambda > 0.0 else None)
            if self.tul_gate is not None:
                budget_ids = self._tul_budget_ids(layout, depths, g_traj)
                h_slots = self.tul_gate.apply_budget(h_slots, budget_ids)
            # "Frozen z" (arms R7d/R8d, tul.detach_z): the coda reads the thought with
            # stop-gradient, so the token CE never shapes the loop or the conditioning
            # stack — they learn from the mux local loss ALONE (computed above, on the
            # live state). A Python-level constant: False traces the identical graph.
            if tc.detach_z:
                h_slots = h_slots.detach()
            # Same eval-only ablation the FM branch takes, applied at the same seam.
            # `normal` returns its input unchanged, so the training forward is
            # bit-identical to a model with no ablation code at all.
            #
            # `tul.prefix_source` (2026-09-13): at "exit" — every model before the knob —
            # `_cells` stays None, `prefix_project` takes its old single-source path and
            # this block is the one that has always been here. Otherwise each cell gets its
            # own source state and the ablation is applied to the STACK, with the exit cell
            # read back off it so a `shuffle` draws ONE permutation, not two.
            _cells = _pad_cells = _pad_pos = None
            _ct_values = None
            if self.tul_code_proj is not None:
                # ── the code target (tul.code_target; spec §17) ─────────────────
                # The projection's cells replace the `prefix_project` write. The
                # regression, the per-pass readings, the oracle switch and the plan
                # ablation all live in one method so the cells the coda reads and the
                # cells the loss grades are built at one seam.
                (_ct_values, _code_cells_out, _code_z_out, code_target_loss,
                 code_target_stats, code_grade_loss, code_grade_stats
                 ) = self._tul_code_target_write(h_slots, xn, db_traj, depths, layout, L,
                                                 plan_mode, code_mode, code_given,
                                                 code_given_mask, input_ids)
                h_slots = self._tul_plan_ablate(h_slots, layout, plan_mode)
            elif _reg_cells is not None and self.tul_fan is None:
                # The register's M cells go 1:1 into the M prefix cells (`prefix_k` is
                # refused unless it equals `slot_cells`). The ablation runs on the STACK
                # and the exit mean is read back off it, so a `shuffle` draws ONE
                # permutation and the coda's cells and the reported `h_slots` agree.
                #
                # A FAN model deliberately does NOT come here. Its K streams were already
                # mixed to ONE state at the register's mean seam above, and that state
                # falls through to the single-source `prefix_project` in the `else` below
                # — the strict ruler's write, at the ruler's prefix width. The register's
                # 1:1 write would reintroduce exactly the 4x-wider readout that made its
                # own 0.022 nat win unreadable.
                _cells = self._tul_plan_ablate(_reg_cells, layout, plan_mode)
                h_slots = _cells.mean(dim=2)
            elif _vq_cells is not None:
                # The K lifted codes go 1:1 into the K prefix cells. The ablation runs on
                # the STACK and the dequantized mean is read back off it, so a `shuffle`
                # draws ONE permutation and the coda's cells and the reported `h_slots`
                # agree — the register's contract, at the same seam.
                _cells = self._tul_plan_ablate(_vq_cells, layout, plan_mode)
                h_slots = _cells.mean(dim=2)
            elif tc.prefix_source != "exit":
                _cells, _pad_cells, _pad_pos = self._tul_prefix_cells(
                    h_slots, db_traj, depths, layout)
                _cells = self._tul_plan_ablate(_cells, layout, plan_mode)
                h_slots = _cells[:, :, -1]
            else:
                h_slots = self._tul_plan_ablate(h_slots, layout, plan_mode)
            if _ct_values is not None:
                values, pos = _ct_values, self.tul.prefix_positions(layout, L)
            else:
                values, pos = self.tul.prefix_project(h_slots, layout, L, cells=_cells)
            if _pad_cells is not None:
                # A PAD cell's carrier is EXACTLY zero, `E_pass` included. Zeroing the
                # SOURCE state is not enough: `prefix_project` adds the per-cell embedding
                # AFTER the projection, so a pad would otherwise carry `E_pass[k]` — a
                # learned constant that says how deep the slot went. Cutting it out of the
                # coda's key set does not cut it out of the CCA causal conv or the
                # `W_v_prev` value shift, which run inside a slot's own segment and carry
                # cell k into cell k+1..K-1 including the EXIT cell. Measured before the
                # fix: `E_pass.grad.abs().sum()` 7407 on a fixture whose cells are almost
                # all pads, against 0.25 on the exit_repeat twin.
                _pv = (~_pad_cells).reshape(values.shape[0], -1,
                                            *([1] * (values.dim() - 2)))
                values = values * _pv.to(values.dtype)
            # coda_token_input (TULConfig): "prelude" = xn, the prelude's OUTPUT at every
            # position (the shipped §3.4 path); "embed" = the prelude's INPUT carrier,
            # x0 (the embedding after embed dropout) expanded to the HC streams and put
            # through the same input_norm — a token carries only itself into the coda.
            if tc.coda_token_input == "embed":
                base = x0.unsqueeze(2).expand_as(xn) if self._is_hc else x0
                base = self.input_norm(base)
            else:
                base = xn
            _critic_base = base
            x_coda = scatter_positions(base, pos, values)

        x_coda, keep = self.tul.apply_token_dropout(x_coda, layout, self.training)
        if tc.coda_token_input == "embed" or self._tg_strict:
            # The arm's contract: in the coda a slot cell carries z and NOTHING else. The
            # per-layer coda injections at the slot cells (x0 = the seed, bigram = the
            # span's bag mean) would otherwise hand the coda span content beside z.
            #
            # Under `tul.tg_geometry="strict"` this is not a second knob but PART of the
            # geometry, and it is load-bearing: the strict coda lets a later span's tokens
            # read earlier prefix cells, so a cell that still carried its own span's
            # bag-mean would hand that span's token identities forward with no pass of the
            # loop in between — exactly the bypass strict exists to cut. It is also what
            # makes `plan_mode="zero"` and `plan_mode="all_slots"` the SAME ablation on a
            # strict arm (tests/test_tul_strict_geometry.py).
            _slot_keep = slot_cell_inject_keep(layout, x_coda.dtype)
            keep = _slot_keep if keep is None else keep * _slot_keep
        if tc.bcast and not tc.tokens_through_core:
            # The unpack: z_{i} through the offset-indexed linears, ADDED to span i+1's
            # token inputs (every stream). After the dropout on purpose: the dropped token
            # loses itself, not the thought it is decoding.
            x_coda = self._apply_injection(x_coda, self.tul.unpack(h_slots, layout))

        # `all_slots` (the route split of the 2026-09-11 budget result): the prefix write is
        # already zeroed in `_tul_plan_ablate`; this cuts the cells' own injections and
        # their read of their span, so the coda's slot cells carry nothing at all.
        _coda_kw = tg_attn_kwargs
        if plan_mode == "all_slots":
            _coda_kw, keep = self._tul_all_slots_coda(x_coda, layout, tc, keep,
                                                      tg_attn_kwargs)
        if _pad_pos is not None:
            # The trajectory arm's PAD cells leave the coda's key set. AFTER `all_slots`
            # so the two narrowings compose rather than one replacing the other.
            _coda_kw = self._tul_pad_cell_narrow(_coda_kw, _pad_pos)

        out: dict = {"logits": None}
        _skip_coda = (self.tul_code_proj is not None and tc.code_target_skip_coda
                      and self.training and labels is not None)
        if _skip_coda:
            # The code-ONLY arm (tul.code_target_skip_coda): at train the forward ends at
            # the projection. No coda, no token CE; `groups["loss"]` starts at an exact 0
            # so the folds below (the slot-loop constraint, the code term) are the whole
            # loss, and train.py's subtraction of `code_target_weighted` reports a
            # train/loss of exactly that constraint. The eval forward takes the ordinary
            # branch and the frozen-at-VAE coda reads the cells for the val instrument.
            xh = None
            groups = {"loss": code_target_loss.new_zeros(())}
            coda_positions = 0
        elif tc.coda_sees_slots and tc.coda_token_cut == 0:
            xh = self._back_region(x_coda, x0, bigram_emb, input_ids, inject_keep=keep,
                                   attn_kwargs=_coda_kw, ret_reset_mask=tg_reset)
            groups = (self._tul_group_losses(xh, labels, layout, want_groups=not self.training)
                      if labels is not None else None)
            coda_positions = L
        else:
            # Arm A4 (coda_sees_slots=False) and/or arm CW (coda_token_cut>0, spec
            # .agents/notes/implemented/architecture/2026-08-18-tul-compaction-window.md) — ONE gather whose drop_mask is the
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

        if gain_reg is not None and groups is not None:
            # The slot map's typical-gain penalty (model.slot_gain_lambda). Same contract as
            # sigreg: `gain_reg_weighted` is subtracted from every reported model loss so a
            # penalised arm stays comparable to its control; `gain_est` is the live reading.
            groups = dict(groups)
            groups["gain_est"] = gain_reg["gain"]
            groups["gain_est_max"] = gain_reg["gain_max"]
            if "n_iters" in gain_reg:
                groups["gain_n_iters"] = gain_reg["gain"].new_tensor(gain_reg["n_iters"])
            groups["gain_reg_weighted"] = gain_reg["penalty"].detach()
            groups["loss"] = groups["loss"] + gain_reg["penalty"]

        if mux_loss is not None and groups is not None:
            groups = dict(groups)
            groups["mux_local"] = mux_loss.detach()
            # UNDETACHED handle, for tul_mux_grad_share only (the gate's `loss_tokens`
            # precedent). Every logging site reads `mux_local`, which stays detached.
            groups["mux_local_live"] = mux_loss
            for _k, _v in mux_stats.items():
                groups[_k] = mux_loss.new_tensor(_v)
            # The WEIGHTED term, exposed so train.py can report train/loss and
            # val loss as the MODEL's CE (the spectral-penalty precedent: an
            # auxiliary term inside train/loss makes the arm incomparable to its
            # control and lets the ppl divergence guard fire on the objective).
            _mw = tc.mux_beta * self.mux_gate * mux_loss
            groups["mux_weighted"] = _mw.detach()
            groups["loss"] = groups["loss"] + _mw

        if spandec_loss is not None and groups is not None:
            # Same contract as `mux_weighted` and `sigreg_weighted`: the WEIGHTED term is
            # exposed so train.py can subtract it and keep train/loss and the val loss on
            # the MODEL's CE — an auxiliary inside the reported loss makes the arm
            # incomparable to its control and fires the ppl divergence guard on the
            # objective (the spectral-penalty precedent).
            groups = dict(groups)
            groups["spandec"] = spandec_loss.detach()
            for _k, _v in spandec_stats.items():
                groups[_k] = spandec_loss.new_tensor(_v)
            _dw = tc.spandec_weight * spandec_loss
            groups["spandec_weighted"] = _dw.detach()
            groups["loss"] = groups["loss"] + _dw
        if code_target_loss is not None and groups is not None:
            # The code target (tul.code_target). Same contract as `spandec_weighted`: the
            # WEIGHTED term is exposed so train.py subtracts it and keeps train/loss and
            # the val loss on the MODEL's CE.
            groups = dict(groups)
            groups["code_target"] = code_target_loss.detach()
            for _k, _v in code_target_stats.items():
                groups[_k] = code_target_loss.new_tensor(_v)
            _tw = tc.code_target_weight * code_target_loss
            groups["code_target_weighted"] = _tw.detach()
            groups["loss"] = groups["loss"] + _tw
        if code_grade_loss is not None and groups is not None:
            # The graded-continuation term (tul.code_grade; spec §17.2). Same contract as
            # `code_target_weighted`: the WEIGHTED term is exposed so train.py subtracts it
            # and train/loss stays the MODEL's CE.
            groups = dict(groups)
            groups["code_grade"] = code_grade_loss.detach()
            for _k, _v in code_grade_stats.items():
                groups[_k] = code_grade_loss.new_tensor(_v)
            _gw = tc.code_grade_weight * code_grade_loss
            groups["code_grade_weighted"] = _gw.detach()
            groups["loss"] = groups["loss"] + _gw
        if tc.code and groups is not None:
            # TUL-Code: the flow term, same contract as `spandec_weighted` (train.py
            # subtracts it so train/loss and the val loss stay the model's CE).
            groups = dict(groups)
            for _k, _v in code_stats.items():
                groups[_k] = groups["loss"].new_tensor(float(_v))
            if code_fm_loss is not None:
                groups["code_fm"] = code_fm_loss.detach()
                _cw = tc.code_fm_weight * code_fm_loss
                groups["code_fm_weighted"] = _cw.detach()
                groups["loss"] = groups["loss"] + _cw
            _csr = getattr(self, "_code_sigreg_loss", None)
            if _csr is not None:
                # Same contract: the weighted term is exposed so train.py subtracts it
                # and train/loss stays the model's CE.
                _sw = tc.code_sigreg_lambda * _csr
                groups["code_sigreg_weighted"] = _sw.detach()
                groups["loss"] = groups["loss"] + _sw

        if _vq_out is not None and groups is not None:
            # The discrete thought's two VQ-VAE terms (tul.vq_codes). Same contract as
            # `spandec_weighted`: the WEIGHTED term is exposed so train.py subtracts it and
            # train/loss stays the MODEL's CE — an auxiliary inside the reported loss makes
            # the arm incomparable to its control and fires the ppl divergence guard on the
            # objective (the spectral-penalty precedent).
            #
            # `vq_perplexity` is the number to read first: the codebook usage of the batch,
            # in (1, vq_codebook]. At 1 every span picked the same symbol and the channel
            # carries nothing, whatever the CE says.
            groups = dict(groups)
            groups["vq"] = _vq_out["loss"].detach()
            groups["vq_commit"] = _vq_out["commit"].detach()
            groups["vq_codebook_loss"] = _vq_out["codebook"].detach()
            for _k in ("vq_perplexity", "vq_used", "vq_n_codes", "vq_codebook_size"):
                groups[_k] = _vq_out["loss"].new_tensor(_vq_out[_k])
            _vw = (tc.code_vq_weight if tc.code else tc.vq_weight) * _vq_out["loss"]
            groups["vq_weighted"] = _vw.detach()
            groups["loss"] = groups["loss"] + _vw
        if rcon_loss is not None and groups is not None:
            # Same contract as `spandec_weighted` and `sigreg_weighted`: the WEIGHTED term
            # is exposed so train.py can subtract it and keep train/loss and the val loss
            # on the MODEL's CE -- an auxiliary inside the reported loss makes the arm
            # incomparable to its control and fires the ppl divergence guard on the
            # objective (the spectral-penalty precedent).
            #
            # `row_contrast` is the raw term and it has an ABSOLUTE reference: it reads
            # log(n_valid) and `row_contrast_acc` reads 1/n_valid when the row's slot
            # states are indistinguishable. `row_contrast_n_rows` is how many rows of the
            # batch carried it (a row needs two anchors).
            groups = dict(groups)
            groups["row_contrast"] = rcon_loss.detach()
            for _k, _v in rcon_stats.items():
                groups[_k] = rcon_loss.new_tensor(_v)
            _rw = tc.row_contrast_lambda * rcon_loss
            groups["row_contrast_weighted"] = _rw.detach()
            groups["loss"] = groups["loss"] + _rw

        # ── LXTUL: the fan's oracle instrument and its repulsion (tul.fan_k) ──────
        # The oracle is EVAL ONLY and costs K extra coda passes, so it is built here, at
        # the end of the forward, for the reason the critic's label is: it replays
        # `_back_region`, which does not exist until the shipped coda has run, and it
        # needs the exact carrier that coda read (`base`, the shipped `keep`, the shipped
        # allow relation) so the replay differs in the WRITE and nothing else.
        if (self.tul_fan is not None and _fan_cells is not None and not self.training
                and labels is not None and groups is not None and xh is not None
                and plan_mode == "normal"):
            self._tul_fan_oracle(_fan_cells, xh, base, x0, bigram_emb, input_ids, labels,
                                 layout, L, keep, _coda_kw, tg_reset, fan_stats)
        if fan_repel_loss is not None and groups is not None:
            # Same contract as `row_contrast_weighted` / `spandec_weighted`: the WEIGHTED
            # term is exposed so train.py subtracts it and train/loss stays the MODEL's
            # CE. `fan_repel` is the raw mean pairwise cosine over the penalised passes
            # and has an ABSOLUTE reference — 1.0 is four identical streams, 0.0 four
            # orthogonal ones, -1/(K-1) the simplex floor. Read it beside
            # `fan/stream_cos_t{t}`, which reports EVERY pass including the ones the term
            # never touches.
            groups = dict(groups)
            groups["fan_repel"] = fan_repel_loss.detach()
            _fw = tc.fan_repel_lambda * fan_repel_loss
            groups["fan_repel_weighted"] = _fw.detach()
            groups["loss"] = groups["loss"] + _fw
        if self.tul_carry is not None and groups is not None and self._loop_carry_stats:
            # `tul.loop_carry`'s per-pass readings. A VARIABLE number of keys (the
            # batch's realised max depth decides how many), so it is a scan and not a
            # tuple — the `fan_*` contract. They carry NO loss term: the carry is a
            # forward mechanism, not a regulariser, so nothing here is subtracted from
            # train/loss.
            groups = dict(groups)
            for _k, _v in self._loop_carry_stats.items():
                groups[f"carry_{_k}"] = _v.detach().to(groups["loss"].dtype)

        if fan_stats and groups is not None:
            # Every fan reading travels as a `fan_*` key so train.py can scan for the
            # prefix: the per-pass cosines and ranks are a VARIABLE number of keys (the
            # batch's realised max depth decides how many), which a fixed tuple cannot
            # carry. train.py logs them under `fan/`.
            groups = dict(groups)
            for _k, _v in fan_stats.items():
                groups[f"fan_{_k}"] = groups["loss"].new_tensor(_v)

        if spandec_pass_loss is not None and groups is not None:
            # Same contract as `spandec_weighted`: the WEIGHTED term is exposed so train.py
            # can subtract it and keep train/loss and the val loss on the MODEL's CE.
            groups = dict(groups)
            groups["spandec_pass"] = spandec_pass_loss.detach()
            for _k, _v in spandec_pass_stats.items():
                groups[_k] = spandec_pass_loss.new_tensor(_v)
            _pw = tc.spandec_pass_weight * spandec_pass_loss
            groups["spandec_pass_weighted"] = _pw.detach()
            groups["loss"] = groups["loss"] + _pw

        if horizon_loss is not None and groups is not None:
            # Same contract as `spandec_weighted`: the WEIGHTED term is exposed so
            # train.py can subtract it and keep train/loss and the val loss on the
            # MODEL's CE.
            groups = dict(groups)
            groups["horizon"] = horizon_loss.detach()
            for _k, _v in horizon_stats.items():
                groups[_k] = horizon_loss.new_tensor(_v)
            _hw = tc.horizon_weight * horizon_loss
            groups["horizon_weighted"] = _hw.detach()
            groups["loss"] = groups["loss"] + _hw

        if self.coda_span is not None and groups is not None:
            # The parallel-decode heads (tul.coda_span_heads). Built HERE and not beside the
            # other local losses because they read the CODA's output, which does not exist
            # until `_back_region` has run. Construction refuses every shape in which `xh`
            # is not the full packed axis (the paid loop, arm A4, arm CW), so the slot
            # positions this indexes are always the ones `prefix_project` wrote.
            _cs_stats: dict = {}
            _cs = self._tul_coda_span_loss(xh, input_ids, layout, stats=_cs_stats)
            groups = dict(groups)
            groups["coda_span"] = _cs.detach()
            for _k, _v in _cs_stats.items():
                groups[_k] = _cs.new_tensor(_v)
            _cw = tc.coda_span_weight * _cs
            groups["coda_span_weighted"] = _cw.detach()
            groups["loss"] = groups["loss"] + _cw

        if tc.core_token_aux and self.training and groups is not None:
            # The core-token auxiliary (`_tul_core_token_aux`). Built HERE, at the END of
            # the forward, for two reasons and both are load-bearing: the shipped path's
            # per-slot depth draw must sit at the SAME position in the RNG stream it has on
            # the ruler (the aux runs after it and puts the stream back), and
            # `_core_region` overwrites `self._core_aux`, which the slot loop has already
            # filled and `_apply_core_aux` consumes below. `self.training` is the eval
            # guard: the sweep, `worth_profile` and inference never pay for this.
            _ca_stats: dict = {}
            _ca = self._tul_core_token_aux(x, x0, bigram_emb, input_ids, labels, layout,
                                           tg_attn_kwargs, tg_reset, stats=_ca_stats)
            groups = dict(groups)
            groups["core_token_aux"] = _ca.detach()
            for _k, _v in _ca_stats.items():
                groups[_k] = _ca.new_tensor(_v)
            _caw = tc.core_token_aux_weight * _ca
            groups["core_token_aux_weighted"] = _caw.detach()
            groups["loss"] = groups["loss"] + _caw

        if oracle_z_loss is not None and groups is not None:
            # Same contract as `spandec_weighted`: the WEIGHTED term is exposed so train.py
            # can subtract it and keep train/loss and the val loss on the MODEL's CE.
            groups = dict(groups)
            groups["oracle_z"] = oracle_z_loss.detach()
            for _k, _v in oracle_z_stats.items():
                groups[_k] = oracle_z_loss.new_tensor(_v)
            _ow = tc.oracle_z_weight * oracle_z_loss
            groups["oracle_z_weighted"] = _ow.detach()
            groups["loss"] = groups["loss"] + _ow

        if (isinstance(self.tul_egrad, CriticEnergy) and self.training
                and groups is not None and db_traj is not None):
            # The within-context critic's OWN training loss. Built HERE and not beside the
            # other local losses because its label REPLAYS `_back_region`, which does not
            # exist until the shipped coda has run, and it needs the exact carrier that
            # coda read (`_critic_base`, the shipped dropout `keep`, the shipped allow
            # relation). Every replay is no_grad, so nothing here trains the loop, the
            # coda, `W_prefix` or the decoder. `critic_every` skips the LABEL on a step,
            # never the energy: the feature is read at every pass of every step regardless,
            # which is what keeps the map the same function on every step.
            self._critic_calls += 1
            _cr_stats: dict = {}
            _cr = None
            if (self._critic_calls - 1) % int(tc.critic_every) == 0:
                _cr = self._tul_critic_loss(
                    db_traj, depths, _egrad_src, self._tul_egrad_ctx, _critic_base, x0,
                    bigram_emb, input_ids, labels, layout, keep, _coda_kw, tg_reset,
                    _cr_stats)
            if _cr is not None:
                groups = dict(groups)
                groups["critic"] = _cr.detach()
                for _k, _v in _cr_stats.items():
                    groups[_k] = _cr.new_tensor(_v)
                _crw = tc.critic_weight * _cr
                groups["critic_weighted"] = _crw.detach()
                groups["loss"] = groups["loss"] + _crw

        if _egrad_src is not None and groups is not None and self.tul_egrad is not None:
            # The energy module's OWN training loss (`tul.grad_pass_energy` 'recon' /
            # 'disc'). Same contract as `mux_weighted` / `spandec_weighted`: the WEIGHTED
            # term is exposed so train.py can subtract it and keep train/loss and the val
            # loss on the MODEL's CE. `z` is detached inside `_egrad_train_loss`, so this
            # term reaches the energy's parameters and nothing else.
            _eg_stats: dict = {}
            _eg = self._egrad_train_loss(_egrad_src, input_ids, layout,
                                         getattr(self, "_tul_egrad_ctx", None),
                                         xh, labels, _eg_stats)
            if _eg is not None:
                groups = dict(groups)
                groups["egrad"] = _eg.detach()
                for _k, _v in _eg_stats.items():
                    groups[_k] = _eg.new_tensor(_v)
                _ew = tc.egrad_weight * _eg
                groups["egrad_weighted"] = _ew.detach()
                groups["loss"] = groups["loss"] + _ew

        if groups is not None:
            out.update(groups)
            if self.mtp is not None:
                # Arc E8 heads on the TUL coda (the parallel-decode arm of the k=12 panel):
                # the heads read the coda state at TOKEN positions only, in token order, so
                # "j-1 labels ahead" means j-1 TOKENS ahead and never crosses a slot.
                if not (tc.coda_sees_slots and tc.coda_token_cut == 0):
                    raise NotImplementedError(
                        "model.mtp_heads > 1 with a gathered coda (coda_sees_slots=False "
                        "or coda_token_cut > 0) is not defined: the head labels are built "
                        "from the full-axis layout")
                x_tok, lab_tok = self._tul_token_compact(xh, labels, layout)
                self._mtp_apply(out, x_tok, lab_tok, self.embed.lm_weight())
            if self._core_aux is not None:
                # The paid loop's fixed-point term (stashed by `_core_region`): added to the
                # loss and exposed as `fixed_point` / `fp_weighted`, the same keys and the
                # same train.py subtraction as the plain path.
                self._apply_core_aux(out)
        elif coda_state_only:
            # The graded-continuation sampler (tul.code_grade): it applies `embed.attend`
            # itself, at the handful of positions it reads. No logits are built.
            self._core_aux = None
            out["coda_state"] = xh
        else:
            self._core_aux = None
            # Generation (labels=None): full logits, with the structural slot id masked
            # out of the head (spec §3.1 / invariant 4 — "masked … at generation").
            # index_fill is out-of-place, so this is safe under grad as well as no_grad.
            out["logits"] = self.embed.attend(xh).index_fill(
                -1, torch.tensor([tc.slot_id], device=xh.device), float("-inf"))
            if self.mtp is not None:
                # The heads' logits on the COMPACT token axis ([B, n_max, V], token order,
                # a row's ragged tail past its token count is unscored garbage); the same
                # gather as the training path, so head j at compact index i is the
                # prediction for the token j-1 TOKENS ahead of token i.
                if not (tc.coda_sees_slots and tc.coda_token_cut == 0):
                    raise NotImplementedError(
                        "model.mtp_heads > 1 with a gathered coda is not defined")
                x_tok, _ = self._tul_token_compact(
                    xh, torch.zeros_like(layout.slot_mask, dtype=torch.long), layout)
                self._mtp_apply(out, x_tok, None, None)
            if self.tul_gate is not None:
                # The generator needs the model's OWN budget for each slot: how many
                # tokens the plan it just built covers (§8). It is the same tensor the
                # coda was conditioned on, never a second, separately-decoded one.
                out["gate_k"] = budget_ids
        out["layer_passes"] = self._tul_layer_passes(layout, depths, coda_positions)
        out["n_tokens"] = (~layout.slot_mask).sum()
        if _code_cells_out is not None and not self.training:
            # TUL-Code: the cells the coda read, ``[B, S, M, C]`` — the generator caches
            # the open span's sampled cells from here (spec §7).
            out["code_cells"] = _code_cells_out.detach()
        if _code_z_out is not None and not self.training:
            # The frozen encoder's code of THIS row's spans, ``[B, S, M, C]``. The graded
            # sampler (tul.code_grade) reads it off a CANDIDATE row to get E(candidate);
            # `code_cells` above is what the coda READ, which under `code_given` is not it.
            out["code_z"] = _code_z_out.detach()
        return out

    @staticmethod
    def _tul_token_compact(xh: Tensor, labels: Tensor, layout: SlotLayout) -> tuple[Tensor, Tensor]:
        """The coda readout and labels at TOKEN positions only, in token order, ``[B, n_max, …]``.

        Rows carry different token counts (the packer fixes the ROW length, not the split
        between tokens and slots: a row with more spans has more slots and fewer tokens), so
        the compaction gathers to the largest count and marks each row's ragged tail with
        label -100. Token order is preserved (:func:`compact_index`); slot positions and
        their emit labels are dropped; ``labels[b, i]`` on the result is the next TOKEN after
        token ``i`` of row ``b``, which is what a lookahead head shifts, and the -100 tail
        propagates through that shift so no head is scored across a row's end.
        """
        B, L = layout.slot_mask.shape
        n_tok = (~layout.slot_mask).sum(dim=1)                                   # [B]
        n_max = int(n_tok.max())
        cidx = compact_index(layout.slot_mask)[:, :n_max]                        # tail → L
        valid = torch.arange(n_max, device=cidx.device).unsqueeze(0) < n_tok.unsqueeze(1)
        cidx = cidx.clamp(max=L - 1)                                             # no dump row needed
        x_tok = gather_positions(xh, cidx)
        lab_tok = torch.where(valid, gather_positions(labels, cidx), labels.new_full((), -100))
        return x_tok, lab_tok

    def _tul_prefix_cells(self, h_slots: Tensor, db_traj: list[Tensor] | None,
                          depths: Tensor, layout: SlotLayout
                          ) -> tuple[Tensor, Tensor, Tensor]:
        """``tul.prefix_source`` != "exit": one SOURCE STATE per coda cell.

        Returns ``(cells, pad_cell, pad_pos)``:

        * ``cells`` ``[B, S, K, …, C]`` — the state cell ``k`` of slot ``s`` is written
          from, before :meth:`TULSlots.prefix_project` puts it through ``W_prefix[k]``;
        * ``pad_cell`` ``[B, S, K]`` bool — True where that cell is a PAD (no pass reached
          it), its ``cells`` entry already zeroed;
        * ``pad_pos`` ``[B, L]`` bool — the same flags at their ROW positions, which is
          what :meth:`_tul_pad_cell_narrow` cuts out of the coda's key set.

        THE RULE (``trajectory``), stated once and tested in
        ``tests/test_tul_prefix_source.py``:

        ===============  ==============================  ==========================
        cell             source                          written iff
        ===============  ==============================  ==========================
        ``k < K − 1``    ``h_{k+1}`` (after pass k+1)     realised depth ``≥ k + 2``
        ``k = K − 1``    ``h_depth`` — the EXIT           always (a valid slot)
        ===============  ==============================  ==========================

        So a slot of realised depth ``d`` writes exactly ``min(d, K)`` non-pad cells, the
        EXIT is present at every forced depth and always in the SAME cell, and no cell
        ever duplicates the exit (``d = k + 1`` would put the exit in cell ``k``, and the
        ``≥ k + 2`` test is what excludes it). A slot of depth 1 therefore writes ONE
        cell, the last.

        ``exit_repeat`` writes the exit into every cell and pads nothing — the matched-count,
        matched-parameter control for the content question.

        ``entry_exit`` writes the ENTRY state ``z_1 = core_init(e)`` into cell 0 and the
        EXIT into every other cell, and pads nothing. It is the BINDING control, for an
        information reason rather than a capacity one: the Lean result in
        ``.agents/notes/proposed/architecture/2026-09-13-information-view-of-the-slot-loop.md``
        gives ``I((z_1..z_T); Y) = I(z_1; Y)`` and ``I(z_T; Y) <= I(z_1; Y)``, so the whole
        trajectory carries exactly the entry's information and the exit can only have lost
        some of it. A trajectory prefix that beats an exit-repeat prefix may therefore be
        recovering what the EXIT threw away, which is not a loop gain. ``entry_exit`` minus
        ``exit_repeat`` is ONE cell's content; ``trajectory`` minus ``entry_exit`` is
        "cells 1..K-2 carry passes 2..K-1 instead of exit copies".

        ``pad_cell`` and ``pad_pos`` are returned as ``None`` in every mode that pads
        nothing, so the caller's masking and zeroing are Python-level branches that trace
        out rather than all-False tensor work on the hot path.

        ``db_traj`` entry ``t`` is the carrier AFTER iteration ``t−1`` with finished slots
        already frozen (``_tul_core``'s ``h = where(active, h_new, h)``), so a finished
        slot's later entries hold its exit; the ``≥ k + 2`` test is what keeps those out.
        A batch whose deepest slot stops short of ``K − 1`` leaves the list short — every
        cell past it is a pad for every slot in the batch, so the index is clamped and the
        value masked, never raised on.

        ``plan_mode="shuffle"``: the caller ablates ``cells`` and leaves ``pad_cell``
        alone, so the pad PATTERN stays with the position while the content moves. That is
        what shuffle is for — destroy the correspondence, change nothing else — and it is
        recorded here rather than discovered from a worth profile.
        """
        tc = self.cfg.tul
        K = tc.prefix_k
        B, S = layout.slot_index.shape
        valid = layout.slot_valid                                  # [B, S]
        if tc.prefix_source == "exit_repeat":
            return h_slots.unsqueeze(2).expand(B, S, K, *h_slots.shape[2:]), None, None
        if tc.prefix_source == "entry_exit":
            if db_traj is None:
                raise RuntimeError(
                    "tul.prefix_source='entry_exit' reached the write with no trajectory: "
                    "`_tul_core` returns one whenever the knob is set, so this is a core "
                    "stage that never ran it (tul_step_mode='db1' or the Euler ladder).")
            # `db_traj[0]` IS the entry: `_tul_core` seeds the list with `core_init(e)`
            # before the first pass, and nothing else writes index 0.
            ee = torch.stack([db_traj[0]] + [h_slots] * (K - 1), dim=2)
            return ee, None, None
        if tc.prefix_source == "trajectory":
            if db_traj is None:
                raise RuntimeError(
                    "tul.prefix_source='trajectory' reached the write with no trajectory: "
                    "`_tul_core` returns one whenever the knob is set, so this is a core "
                    "stage that never ran it (tul_step_mode='db1' or the Euler ladder). "
                    "Raises rather than writing the exit into every cell under the "
                    "trajectory arm's name.")
            zeros = torch.zeros_like(h_slots)
            src: list[Tensor] = []
            pads: list[Tensor] = []
            for k in range(K - 1):
                written = valid & (depths >= k + 2)                 # [B, S]
                st = db_traj[k + 1] if k + 1 < len(db_traj) else zeros
                _v = written.view(B, S, *([1] * (h_slots.dim() - 2)))
                src.append(torch.where(_v, st, zeros))
                pads.append(~written)
            src.append(h_slots)                                     # the EXIT cell
            pads.append(~valid)
            cells = torch.stack(src, dim=2)                         # [B, S, K, …, C]
            pad_cell = torch.stack(pads, dim=2)                     # [B, S, K]
        else:
            raise ValueError(f"_tul_prefix_cells called at prefix_source={tc.prefix_source!r}")
        # The same flags at ROW positions. Invalid slots address the dump row exactly as
        # `prefix_project` sends them there, so a tail-pad slot's physical cells are not
        # touched here — under the strict/restrict coda they already carry the dump bin's
        # `bag_id` and no token may read them.
        L = layout.l_total
        offs = torch.arange(K, device=h_slots.device)
        pos = layout.slot_index.unsqueeze(-1) + offs                # [B, S, K]
        pos = torch.where(valid.unsqueeze(-1), pos, torch.full_like(pos, L))
        pad_pos = torch.zeros(B, L + 1, dtype=torch.bool, device=h_slots.device)
        pad_pos.scatter_(1, pos.reshape(B, S * K),
                         (pad_cell & valid.unsqueeze(-1)).reshape(B, S * K))
        return cells, pad_cell, pad_pos[:, :L]

    @staticmethod
    def _tul_pad_cell_narrow(kw: dict | None, pad_pos: Tensor) -> dict:
        """Remove the PAD cells of a ``prefix_source="trajectory"`` row from the key set.

        A pad cell carries a zero carrier and no label, but zero is CONTENT to attention:
        without this a later token would attend every pad and read the depth draw's own
        pattern. So the coda's allow relation is narrowed on the KEY axis, both branches
        (``tg_allow`` and ``tg_comp_allow`` — one relation, two branches, the F1 defect
        class).

        The DIAGONAL is kept. A pad cell's own query row would otherwise be empty on the
        compressed branch (which keeps ``j == i``), and an all-``-inf`` softmax row is 0
        under SDPA and NaN under an explicit one (morph/model/CLAUDE.md). Nothing reads a
        pad cell's output — it is masked out of every other query's keys and carries no
        label — so keeping its self-edge costs nothing and removes a NaN class.
        """
        if kw is None or not any(kw.get(k) is not None for k in ("tg_allow", "tg_comp_allow")):
            raise RuntimeError(
                "tul.prefix_source='trajectory' needs a coda allow relation to cut its pad "
                "cells out of; this forward built none. TULConfig refuses the combination "
                "at construction, so reaching here means the coda kwargs were rebuilt "
                "without it (`_tul_tg_kwargs` is the ONE home).")
        L = pad_pos.shape[1]
        eye = torch.eye(L, dtype=torch.bool, device=pad_pos.device).view(1, 1, L, L)
        key_ok = (~pad_pos).view(pad_pos.shape[0], 1, 1, L) | eye
        out = dict(kw)
        for _k in ("tg_allow", "tg_comp_allow"):
            if out.get(_k) is not None:
                out[_k] = out[_k] & key_ok
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
        if mode in ("zero", "all_slots"):
            # `all_slots` zeroes the prefix write exactly as `zero` does; the REST of it —
            # the coda's per-layer injections at the slot cells and the cells' own read of
            # their span — is cut in `_forward_tul._tul_all_slots_coda`, because it lives
            # in the coda call and not in the state this method holds.
            return torch.zeros_like(h_slots)
        if mode != "shuffle":
            raise ValueError(
                f"plan_mode must be normal|zero|shuffle|all_slots, got {mode!r}")
        B, S = layout.slot_valid.shape
        # Pads sort last (score 2.0 > any uniform draw), so real slots are permuted
        # among the real slot POSITIONS only — SlotLayout guarantees pads are last.
        r = torch.rand(B, S, device=h_slots.device)
        r = torch.where(layout.slot_valid, r, torch.full_like(r, 2.0))
        perm = r.argsort(dim=1)
        idx = perm.reshape(B, S, *([1] * (h_slots.dim() - 2))).expand_as(h_slots)
        return h_slots.gather(1, idx)

    def _tul_all_slots_coda(self, x_coda: Tensor, layout: SlotLayout, tc: TULConfig,
                            keep: Tensor | None, tg_attn_kwargs: dict | None
                            ) -> tuple[dict, Tensor]:
        """``plan_mode="all_slots"``: make the coda's slot cells carry NOTHING.

        ``zero`` ablates ONE route — the prefix write. The cross-span budget
        (``lab/experiments/failures/2026-09-11-arc-span-budget.md``) priced that route at
        0.093 nats against a 0.399-nat budget and named the rest: under ``tg_restrict`` the
        allow relation is "same span OR any slot position", so a slot CELL in the coda
        re-summarises its own span from the coda's token states and every later token reads
        that summary. Zeroing the write does not touch it.

        Three tensors reach a slot cell in the coda, and this cuts all three:

        1. the carrier value written by :meth:`TULSlots.prefix_project` — zeroed by
           :meth:`_tul_plan_ablate` at ``all_slots``;
        2. the per-layer additive injection at that position (``x0_injects`` carries the
           slot's SEED, ``E_slot`` + the span bag-mean, and the bigram term carries the
           span's bag-mean of bigram embeddings — both are added at EVERY coda layer) —
           zeroed here through ``inject_keep``, the same lever the token-state dropout and
           ``coda_token_input="embed"`` already use;
        3. the cell's own attention over its span's token states — cut here by rebuilding
           ``tg_allow`` with ``slot_queries_slots_only=True``, the relation
           ``tul.tg_restrict_scope="coda"`` already builds: a slot cell's query may attend
           slot cells only, and every slot cell now carries nothing.

        NOT cut, so the reading is a LOWER bound on what the cells carry: the CCA causal
        conv and its ``W_v_prev`` value shift still read the positions before a slot cell.
        Cutting them needs a segment reset (``tg_seg``), and that reset also fires between
        one span's tokens and the next span's, which would change the TOKEN positions'
        conv and put an operator change inside a paired CE difference. Named rather than
        silently included.

        Eval-only, and refused wherever the three routes above are not the whole story.
        """
        if not self._tg_restrict:
            raise NotImplementedError(
                "plan_mode='all_slots' is defined only under tul.tg_restrict: without the "
                "mask a coda slot cell attends EVERY earlier position, so 'the cell carries "
                "nothing' would need an allow relation this model never trained with. Run "
                "it on a mask arm (tul_slot_mux_mask_norm_match and its lineage).")
        if self._tg_strict:
            # Under strict the coda ALREADY cuts routes 2 and 3 on every forward: the
            # cell's per-layer injections are zeroed in `_forward_tul` and its query
            # reaches itself alone. Rebuilding `tg_allow` here would WIDEN the relation
            # (slot_queries_slots_only lets a cell read every earlier cell), so the strict
            # kwargs are kept and only the injection cut is re-applied — which the caller
            # has already applied, making this the identity. `all_slots` and `zero` are
            # therefore the same ablation on a strict arm, and that equality is the
            # instrument's own check that the geometry is doing what it claims.
            _sk = slot_cell_inject_keep(layout, x_coda.dtype)
            return dict(tg_attn_kwargs or {}), (_sk if keep is None else keep * _sk)
        if tc.tg_restrict_scope != "all":
            raise NotImplementedError(
                f"plan_mode='all_slots' with tul.tg_restrict_scope={tc.tg_restrict_scope!r} "
                "is not defined: at scope 'coda' the cell already attends slot cells only, "
                "and the prelude is global, so the cell's content comes from a route this "
                "ablation does not model.")
        if tc.tokens_through_core:
            raise NotImplementedError(
                "plan_mode='all_slots' has no meaning on the paid loop: a slot IS a looped "
                "position there and there is no separate write to ablate.")
        if not (tc.coda_sees_slots and tc.coda_token_cut == 0):
            raise NotImplementedError(
                "plan_mode='all_slots' needs the FULL-L coda (coda_sees_slots=true, "
                "coda_token_cut=0): on a gathered coda the allow relation is not re-derived "
                "for the gathered index space.")
        if tc.bcast:
            raise NotImplementedError(
                "plan_mode='all_slots' with tul.bcast is not defined: the unpack adds z to "
                "the TOKEN inputs of the next span, a fourth route that is not a slot cell.")
        allow = tg_allow_mask(layout, soft_prev_span=tc.tg_soft_prev_span,
                              slot_queries_slots_only=True)
        kw = dict(tg_attn_kwargs or {})
        kw["tg_allow"] = allow
        kw["tg_slot_mask"] = layout.slot_mask
        _sk = slot_cell_inject_keep(layout, x_coda.dtype)
        return kw, (_sk if keep is None else keep * _sk)

    def tul_forward_ablated(self, input_ids: Tensor, labels: Tensor | None,
                            layout: SlotLayout, plan_mode: str = "normal",
                            tul_step_mode: str | None = None,
                            slot_depths: Tensor | None = None,
                            code_mode: str | None = None,
                            code_steps: int | None = None,
                            code_seed: int | None = None) -> dict:
        """Eval-only forward with the slot state ablated. Works on ANY TUL arm.

        ``normal`` — the shipped path.
        ``zero``   — the slot values written into the coda are zeroed. Removes the
                     plan's content AND the fact that a plan is there.
        ``shuffle``— whole slots permuted WITHIN a row. Removes only the correspondence
                     between a slot and its span, which is what makes it the
                     span-SPECIFICITY number. Report the shuffle COST, never a
                     specificity fraction (docs/tul-fm-probing.md §4 rule 1).
        ``all_slots`` — THE ROUTE SPLIT (2026-09-11). ``zero`` plus the two routes it
                     leaves open: the coda's per-layer injections at the slot cells and the
                     cells' own attention over their span. Under ``tg_restrict`` this makes
                     the slot channel carry nothing at all, so the paired cost is the WHOLE
                     slot channel rather than the prefix write alone — the difference
                     between the two is what the prelude carries through the cell without
                     the loop. See :meth:`_tul_all_slots_coda`, including the one route it
                     does NOT cut. Mask arms only; it raises elsewhere.

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

        ``slot_depths`` ``[B, max_slots]`` forces the per-slot loop depth for THIS
        forward (:meth:`_slot_depth_override`), which is what
        ``lab/divergence/slot_depth_isolation.py`` runs one slot at a time. ``None`` is
        the shipped path.

        A separate entry point rather than a forward flag, for the reason
        :meth:`tul_forward_with_plan_nats` gives: the training path must not carry a
        branch that decides how much work to do.
        """
        if self.tul is None:
            raise RuntimeError(
                "tul_forward_ablated needs a model built with MORPHConfig(tul=...)")
        if plan_mode != "wrong_seed":
            return self._forward_single(input_ids, labels, 0, None, layout,
                                        _plan_mode=plan_mode,
                                        tul_step_mode=tul_step_mode,
                                        _slot_depths=slot_depths,
                                        _code_mode=code_mode, _code_steps=code_steps,
                                        _code_seed=code_seed)
        if self.tul_code_enc is not None:
            raise NotImplementedError(
                "plan_mode='wrong_seed' on a code model: the seed feeds the thinker, not the "
                "cells (spec §8).")
        tc = self.cfg.tul
        orig = tc.slot_seed
        alt = "bag_mean" if orig != "bag_mean" else "e_slot"
        tc.slot_seed = alt
        try:
            return self._forward_single(input_ids, labels, 0, None, layout,
                                        tul_step_mode=tul_step_mode,
                                        _slot_depths=slot_depths)
        finally:
            tc.slot_seed = orig

    def tul_fm_forward(self, input_ids: Tensor, labels: Tensor | None,
                       layout: SlotLayout, plan_mode: str = "normal") -> dict:
        """FM1's name for :meth:`tul_forward_ablated`, kept so the FM1 gates and the
        trainer's FM eval block keep pointing at the same call."""
        if self.fm_planner is None:
            raise RuntimeError("tul_fm_forward needs a model built with MORPHConfig(fm=...)")
        return self.tul_forward_ablated(input_ids, labels, layout, plan_mode)

    def _euc_embed_leaf(self) -> Tensor:
        """The euclidean embedding table's LEAF parameter.

        Not ``embed.hybrid.euc_embed.weight``: under the int6 embedding QAT that every
        real arm runs, ``.weight`` is a ``torch.nn.utils.parametrize`` PROPERTY that
        recomputes a fresh non-leaf tensor on every access, so a tensor grabbed after the
        forward is not the one the forward used and ``autograd.grad(..., allow_unused=
        True)`` silently returns None. That read as "the auxiliary touches nothing",
        which is the opposite of what the probe exists to detect.
        """
        for name, prm in self.embed.named_parameters():
            if "euc_embed" in name:
                return prm
        raise RuntimeError("no euclidean embedding parameter found")

    def tul_mux_grad_share(self, input_ids: Tensor, labels: Tensor,
                           layout: SlotLayout) -> dict:
        """Eval-only: how much of the tied embedding table's gradient the MUX term owns.

        THE FM2 SCAR, made observable. FM2's emit CE could not reach the planner's
        WEIGHTS — a test proved it — and degraded the planner anyway, by reshaping the
        shared prelude features that defined its targets (copy_gap 0.47 -> 0.26). An
        auxiliary loss on a WEIGHT-TIED head is the same hazard one level up: with
        ``mux_detach_head: false`` the MUX gradient lands directly on the embedding
        table that the token CE also owns and that every span target is built from.

        Reports ``||g_mux|| / (||g_mux|| + ||g_ce||)`` on the euclidean embedding table.
        Near 0 means MUX is a passenger; approaching or above 0.5 means the auxiliary is
        steering the representation the main objective and its own targets are made of.
        """
        if self.tul is None or self.cfg.tul.mux_beta <= 0.0:
            raise RuntimeError("tul_mux_grad_share needs a model with tul.mux_beta > 0")
        # SLICE TO 2 ROWS. This is the one eval instrument that runs a forward WITH a
        # backward graph, and in eval mode the core loop's checkpointing is off
        # (do_ckpt = self.training and ...), so a full-BPTT looped arm retains every
        # iteration's activations. At batch 6 that OOM'd the 5090 at step-0 eval
        # (tul-l1, 2026-08-29: 25.6 GiB in use, died on a 146 MiB alloc). A gradient
        # NORM RATIO does not need the full batch.
        import dataclasses as _dc
        k = min(2, input_ids.shape[0])
        input_ids, labels = input_ids[:k], labels[:k]
        layout = _dc.replace(layout, **{f.name: getattr(layout, f.name)[:k]
                                        for f in _dc.fields(layout)
                                        if isinstance(getattr(layout, f.name), Tensor)})
        was = self.training
        self.eval()
        try:
            # enable_grad explicitly: the trainer's eval loop is @torch.no_grad, and
            # this is the one eval instrument that NEEDS a backward graph.
            with torch.enable_grad():
                out = self(input_ids, labels, slot_layout=layout)
                w = self._euc_embed_leaf()
                ce = out["loss"] - out["mux_weighted"]   # mux_weighted is detached
                g_ce = torch.autograd.grad(ce, w, retain_graph=True,
                                           allow_unused=True)[0]
                mux_w = self.cfg.tul.mux_beta * self.mux_gate * out["mux_local_live"]
                g_mx = torch.autograd.grad(mux_w, w, retain_graph=False,
                                           allow_unused=True)[0]
        finally:
            self.train(was)
        n_ce = 0.0 if g_ce is None else float(g_ce.norm())
        n_mx = 0.0 if g_mx is None else float(g_mx.norm())
        tot = n_ce + n_mx
        return {"mux_embed_grad_share": (n_mx / tot) if tot > 0 else 0.0,
                "mux_embed_grad_norm": n_mx, "ce_embed_grad_norm": n_ce}

    @torch.no_grad()
    def tul_attn_lift_probe(self, input_ids: Tensor, layout: SlotLayout) -> dict:
        """Eval-only: MUX §8.3's reasoning attention lift, on the WINDOW branch.

        "The slot is the only route" is architecture; "the token actually looks at it"
        is behaviour, and only the second one predicts whether the gist is carrying
        anything. See ``morph/model/attn_lift.py`` for exactly which branch is counted
        and why the compressed branch is not.

        Eager only — it swaps the window-branch reference implementation for one forward.
        Works on the unrestricted control too (same reference path), so the two arms are
        comparable on this metric.
        """
        if self.tul is None:
            raise RuntimeError("tul_attn_lift_probe needs MORPHConfig(tul=...)")
        from morph.model.attn_lift import AttnLiftStats, capture_attn_lift

        stats = AttnLiftStats()
        with capture_attn_lift(layout, stats):
            self(input_ids, labels=None, slot_layout=layout)
        return stats.summary()

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

        _fkw, _freset, _ckw, _creset = self._tul_tg_kwargs(layout)
        x, x0, bigram = self._tul_front(input_ids, layout, attn_kwargs=_fkw,
                                        ret_reset_mask=_freset)
        if self.tul_code_enc is not None:
            # TUL-Code: the written cells are E's codes (spec §8 `val/code_eff_rank`), read
            # over the S*M cells of a row in C dims — the SAME computation as the slot
            # family's `val/slot_eff_rank`, so the two numbers compare (slot family 5.7-7.3).
            # Both names are returned: `code_*` is the spec's, `slot_*` keeps the panel's
            # dashboards pointed at the same quantity.
            _xn = self.input_norm(x)
            _xs = _xn.mean(dim=2) if self._is_hc else _xn
            _z, _ok = self.tul_code_enc(_xs, layout)
            _Bc, _Sc, _Mc, _Cc = _z.shape
            _zc = _z.float().reshape(_Bc, _Sc * _Mc, _Cc).cpu()
            _vc = _ok.repeat_interleave(_Mc, dim=1).cpu()
            _rows = _zc[_vc]
            _er = effective_rank(_zc, _vc)
            _pc = mean_pairwise_cos(_zc, _vc)
            return {"code_eff_rank": _er, "code_pairwise_cos": _pc,
                    "slot_eff_rank": _er, "slot_pairwise_cos": _pc,
                    "slot_norm_mean": float(_rows.norm(dim=-1).mean()) if _rows.numel() else 0.0,
                    "slot_component_std": float(_rows.std()) if _rows.numel() > 1 else 0.0,
                    "slot_component_mean": float(_rows.mean()) if _rows.numel() else 0.0}
        _xn, h_slots, _d, _g, *_ = self._tul_core(x, x0, bigram, layout,
                                                 input_ids=input_ids)
        # ── the think-once stack (tul.cond_layers) ───────────────────────────────
        # This probe's contract is "the WRITTEN slot states, read at the point the coda
        # reads them". On a `cond_layers` model the coda reads the STACK's output, so the
        # stack has to run here or the headline rank would be the loop's exit and the arm's
        # own instrument would be blind to its own mechanism. `tul_cond is None` on every
        # other arm, so no queued run's reading moves.
        if self.tul_cond is not None:
            h_slots = self._tul_cond_apply(
                h_slots, n_slots=layout.slot_index.shape[1],
                m_cells=int(self.cfg.tul.slot_cells))
        # ── the discrete thought (tul.vq_codes) ──────────────────────────────────
        # THE INSTRUMENT'S DEFINITION ON A VQ MODEL, stated because it is a choice: the
        # rank is read over the K LIFTED CODES, which are exactly what the coda's prefix
        # cells hold. Reading the continuous exit state instead would report the rank of a
        # tensor no reader on this arm ever sees, and reading the dequantized MEAN would
        # report one number per slot and hide the whole mechanism. `slot_eff_rank` is then
        # over all S*K cells of a row (the register's convention at the same shape) and
        # `slot_cell_eff_rank` is the rank WITHIN a slot across its own K codes — which is
        # bounded by K and by the number of DISTINCT codes the slot picked, so it is the
        # direct reading of "did the quantizer give the thought rank by construction".
        _vq_k = 0
        if self.tul_vq is not None:
            _vq_k = int(self.tul_vq.k)
            _cells, _dq, _ = self.tul_vq(h_slots, layout.slot_valid)
            h_slots = _cells.reshape(_cells.shape[0], -1, *_cells.shape[3:])
        # ── C1 (tul.center_exit) ─────────────────────────────────────────────────
        # Same contract as the `cond_layers` line above, and for the same reason: this
        # probe reports "the WRITTEN slot states, read at the point the coda reads them",
        # and on a center arm the coda reads the CENTERED state. Without this line the
        # arm's headline instrument would report the loop's raw exit and be blind to the
        # only thing the arm does. `tul_center is None` on every other model, so no queued
        # run's reading moves.
        if self.tul_center is not None:
            h_slots = self.tul_center(h_slots, layout.slot_valid,
                                      m_cells=int(self.cfg.tul.slot_cells))
        z = self._readout(h_slots).float()                     # [B, S, C] or [B, S*M, C]
        valid = layout.slot_valid
        # ── the Thought Register (tul.slot_cells) ────────────────────────────────
        # `_tul_core` returns the CELL axis, so on a register model `z` is [B, S*M, C] and
        # `valid` is still [B, S]. The headline numbers stay defined the same way — the
        # rank of a ROW's written states — and are now over all S*M cells, which is the
        # quantity the arm exists to move: the ruler reads 6.34 in 1024 dimensions with
        # pairwise cosine 0.74, so "the slots of a row are near copies" is the defect.
        # `slot_cell_eff_rank` / `slot_cell_pairwise_cos` are the SECOND reading and the
        # one no earlier arm could have: the rank WITHIN a slot, across its own M cells.
        # A register whose cells collapse onto each other reads ~1 there and has bought
        # nothing, whatever the row number says.
        _m = _vq_k if _vq_k else int(self.cfg.tul.slot_cells)
        within: dict[str, float] = {}
        if _m > 1:
            # Vectorised on the DEVICE, no eigendecomposition. The first version moved
            # the cells to the CPU and ran one `eigvalsh` per (row, slot): 6 x 64 per batch
            # x 20 eval batches, eight CPU threads, the GPU at 9 % for ~3.7 min per val
            # (slot-register-m4, 2026-09-13, killed at step 600). The [M, C] cells of a
            # slot have a [C, C] covariance whose NONZERO eigenvalues are those of the
            # [M, M] Gram X Xᵀ/(M-1), so the participation ratio (Σλ)²/Σλ² is
            # tr(G)² / ‖G‖_F², a trace formula; the pairwise cosine is the mean of the
            # normalised Gram's off-diagonal. Both in float64 to match `effective_rank`.
            _S = valid.shape[1]
            valid = valid.repeat_interleave(_m, dim=1)
            _zc = z.reshape(z.shape[0], _S, _m, z.shape[-1])          # [B, S, M, C]
            _cells = _zc[layout.slot_valid].double()                  # [N, M, C] valid slots
            if _cells.shape[0] > 0:
                # ONE home for this arithmetic: `morph.model.tul_fan.fan_stream_stats`,
                # shared with LXTUL's per-pass `fan/stream_rank_t{t}` so the register's
                # headline reading and the fan's cannot drift apart. Bit-identical to the
                # inline trace formula it replaced (float64, centered for the rank, raw
                # for the cosine).
                _er_slot, _cos_slot = fan_stream_stats(_cells)
                within = {"slot_cell_eff_rank": float(_er_slot.mean()),
                          "slot_cell_pairwise_cos": float(_cos_slot.mean()),
                          "slot_cells": float(_m)}
            else:
                within = {"slot_cell_eff_rank": 0.0, "slot_cell_pairwise_cos": 0.0,
                          "slot_cells": float(_m)}
        rows = z[valid]
        # eigvalsh goes through cusolver on CUDA, and cusolverDnCreate failed with
        # INTERNAL_ERROR at GL1b's first eval (2026-08-29 smoke) once the mux terms and
        # attn-lift hooks shared eval memory — the same call had run clean in GL1. The
        # covariance is [C, C] = 8 MB; the CPU eigendecomposition costs milliseconds and
        # removes the cusolver surface from this eval-only probe entirely.
        z_cpu, valid_cpu = z.cpu(), valid.cpu()
        return {
            "slot_eff_rank": effective_rank(z_cpu, valid_cpu),
            "slot_pairwise_cos": mean_pairwise_cos(z_cpu, valid_cpu),
            "slot_norm_mean": float(rows.norm(dim=-1).mean()),
            # The scale SIGReg's statistic actually sees. `_readout` ends in RMSNorm, so
            # this sits at ~1 by construction and no standardisation is applied before
            # the statistic — standardising would make the loss vacuous (see
            # morph/model/sigreg.py). Logged so a change to the readout cannot silently
            # move the target the regulariser is chasing.
            "slot_component_std": float(rows.std()),
            "slot_component_mean": float(rows.mean()),
            **within,
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
        ``coda_token_cut>0`` — .agents/notes/implemented/architecture/2026-08-18-tul-compaction-window.md)."""
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
        with every slot — .agents/notes/implemented/architecture/2026-08-18-tul-compaction-window.md). The gather itself does not
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
        IDENTICAL across every arm CW variant (.agents/notes/implemented/architecture/2026-08-18-tul-compaction-window.md): which
        positions the CODA sees is decided after this point, never before it. Shared by
        :meth:`_forward_tul` and :meth:`tul_forward_cw_arms` so the (expensive) core loop
        runs once per input, not once per arm.
        """
        tc = self.cfg.tul
        if tc.tokens_through_core:
            raise NotImplementedError(
                "tul.tokens_through_core (arm A2) has no defined interaction with arm CW "
                "(.agents/notes/implemented/architecture/2026-08-18-tul-compaction-window.md) — it is not specified, so this "
                "raises rather than silently picking a behaviour."
            )
        _fkw, _freset, _ckw, _creset = self._tul_tg_kwargs(layout)
        x, x0, bigram_emb = self._tul_front(input_ids, layout, attn_kwargs=_fkw,
                                            ret_reset_mask=_freset)
        xn, h_slots, depths, g_traj, *_ = self._tul_core(x, x0, bigram_emb, layout,
                                                         input_ids=input_ids)
        if self.tul_gate is not None:
            h_slots = self.tul_gate.apply_budget(
                h_slots, self._tul_budget_ids(layout, depths, g_traj))
        values, pos = self.tul.prefix_project(h_slots, layout, layout.l_total)
        x_coda = scatter_positions(xn, pos, values)
        x_coda, keep = self.tul.apply_token_dropout(x_coda, layout, self.training)
        return x_coda, x0, bigram_emb, keep, depths

    def tul_forward_cw_arms(self, input_ids: Tensor, labels: Tensor, layout: SlotLayout,
                            cut: int, seed: int = 0) -> dict[str, dict]:
        """Eval-only: score CW0/CW1/CW2/CW3 in one pass (.agents/notes/implemented/architecture/2026-08-18-tul-compaction-window.md).

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
                f"(.agents/notes/implemented/architecture/2026-08-18-tul-compaction-window.md)."
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

    def _tul_cond_apply(self, h_slots: Tensor, n_slots: int = 0,
                        m_cells: int = 1) -> Tensor:
        """Run the think-once conditioning stack ONCE over the compact slot sequence.

        ``h_slots`` is ``[B, S*M, n, C]`` (HC carrier) straight out of ``_tul_core``,
        BEFORE the register's mean. That placement is the arm: the stack has to be the
        reader of whatever the loop wrote, and on a register model BOTH readers are
        per-cell — the coda's 1:1 prefix write reads cell ``i``, and the span decoder
        reads the cells' mean. A stack that ran on the mean would be the reader of
        neither.

        ``m_cells == 1`` (every model before ``tul.slot_cells``): the blocks are called
        exactly as the core calls its blocks on this sequence — causal attention among
        slots, no injection term, NO MASK. ``slot_cell_relation`` at M = 1 IS that plain
        causal relation, so passing it would change nothing but the kernel the attention
        takes; the no-mask call is kept so this placement is bit-identical to the one
        before the register existed.

        ``m_cells > 1``: the compact axis is ``S*M`` cells and the stack runs under the
        SAME relation the register loops with — own slot full, earlier slots causal, from
        the ONE builder ``slot_cell_relation``, delivered through the SAME ``tg_relation``
        kwarg the core stage uses so the two cannot execute differently. Plain causal on
        the flattened axis would make cell 0 blind to its siblings, and the stack would be
        a different mechanism from the loop it conditions. ``tul.loop_reach`` is refused
        with a stack (``TULConfig``), so no reach budget reaches this relation.

        Pad slots sit at the tail of the compact sequence and the relation never lets a
        valid cell read past its own slot, so a pad cell is never a key of a valid one;
        the pad's own output is dropped downstream (``prefix_project`` sends an invalid
        slot to the dump row). No checkpointing: ``S*M`` is far shorter than the token
        stream (spec §3.3).
        """
        akw = None
        if m_cells > 1:
            allow, _same = slot_cell_relation(n_slots, m_cells, h_slots.device)
            akw = {"tg_relation": allow}
        for layer in self.tul_cond:
            h_slots = layer(h_slots, attn_kwargs=akw)
        return h_slots

    def _tul_layer_passes(self, layout: SlotLayout, depths: Tensor | None,
                          coda_positions: int) -> Tensor:
        """Total layer-passes in this batch (spec §2: 10.3 vs 44 at OWT span 19.2).

        prelude and coda run on every position; the core runs ``depth`` times on each
        REAL slot (or, for arm A2, on every position at the sampled per-sample depth);
        the think-once conditioning stack runs once per REAL slot, and once per CELL of a
        real slot at ``tul.slot_cells > 1``.
        The caller divides by ``n_tokens`` to get the headline number.
        """
        cfg = self.cfg
        L = layout.l_total
        B = layout.slot_mask.shape[0]
        passes = torch.tensor(float(cfg.n_prelude * L * B + cfg.n_coda * coda_positions * B),
                              device=layout.slot_mask.device)
        if self.tul_code_enc is not None:
            # TUL-Code: the thinker runs `_code_last_passes` core passes over the doubled
            # slot sequence (2·M positions per slot, pads included — a fixed shape), set by
            # `_tul_code_core` on this forward: 0 in phase 1, 1 in phases 2-3 at train, k
            # at eval ("sampled"), S·k under "rolled". Spec §12.
            _S = layout.slot_index.shape[1]
            passes = passes + float(cfg.n_core * 2 * int(cfg.tul.prefix_k) * _S * B
                                    * int(self._code_last_passes))
        elif depths is None:                                 # arm A2: core over all positions
            passes = passes + float(cfg.n_core * L * B * cfg.mean_depth)
        else:
            passes = passes + cfg.n_core * (depths * layout.slot_valid).sum()
        if self.tul_cond is not None:
            # Once per REAL slot — and once per CELL of a real slot on a register model,
            # where the stack's compact axis is S*M, not S.
            passes = passes + (len(self.tul_cond) * int(cfg.tul.slot_cells)
                               * layout.slot_valid.sum())
        if self.tul_spandec is not None:
            # The span decoder runs its blocks over EVERY slot cell of the fixed-shape
            # [B, S, J] grid, pads included — a fixed shape is what keeps the compiler out
            # of a data-dependent branch, and the cost is paid whether or not a cell is
            # supervised. Counting the valid slots only would under-report the arm's real
            # price by the pad fraction, which is exactly the number this metric exists to
            # keep honest.
            S = layout.slot_index.shape[1]
            passes = passes + float(len(self.tul_spandec.blocks) * B * S
                                    * self.tul_spandec.max_tokens)
            if self.cfg.tul.spandec_per_pass and self.training and depths is not None:
                # The per-pass target decodes `min(t, cap) * pass_tokens` positions at every
                # pass t up to the batch's realised maximum depth — 168 positions per slot
                # at cap 6 / pass_tokens 8 / depth 6, against the exit term's 32. Kept as
                # tensor arithmetic so the metric costs no host sync, and gated on training
                # because an eval forward never builds the trajectory the term reads.
                _cap = int(self.cfg.tul.spandec_pass_horizon_max)
                _pt = int(self.cfg.tul.spandec_pass_tokens)
                _max_d = int(self.cfg.tul.slot_max_depth or self.cfg.max_depth)
                _t = torch.arange(1, _max_d + 1, device=depths.device)
                _n = (_t.clamp(max=_cap) * (_t <= depths.max()).long()).sum() * _pt
                passes = passes + len(self.tul_spandec.blocks) * B * S * _n
        # `self.coda_span` adds NO block pass: each head is one RMSNorm and one [d, d]
        # matmul on [B, S, d] (3.2 GFLOP at the panel shape, against a ~50 TFLOP step). Its
        # real cost is the J x S readout rows, which this metric does not count for the
        # token CE either.
        return passes

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, input_ids: Tensor, labels: Tensor | None = None,
                bag_size: int = 0, seq_lens: Tensor | None = None,
                slot_layout: SlotLayout | None = None,
                tul_step_mode: str | None = None,
                slot_depths: Tensor | None = None,
                code_mode: str | None = None, code_steps: int | None = None,
                code_seed: int | None = None,
                code_given: Tensor | None = None,
                code_given_mask: Tensor | None = None) -> dict:
        """``slot_depths`` ``[B, max_slots]``: the EVAL-ONLY per-slot depth table, the
        ``slot_layout`` pattern — ``None`` is bit-identical to before it existed. See
        :meth:`_slot_depth_override`.

        ``code_*`` (TUL-Code, eval only; docs/tul-code-spec.md §7): ``code_mode``
        "encoder" | "sampled" | "rolled" | "generate", ``code_steps`` the sampler's k,
        and under "generate" ``code_given`` ``[B, S, M, C]`` + ``code_given_mask``
        ``[B, S]`` hand the forward already-sampled cells (the generator's cache for the
        open span) so a span is written from ONE sample. ``None`` everywhere is the
        shipped path."""
        return self._forward_single(input_ids, labels, bag_size, seq_lens, slot_layout,
                                    tul_step_mode=tul_step_mode,
                                    _slot_depths=slot_depths,
                                    _code_mode=code_mode, _code_steps=code_steps,
                                    _code_seed=code_seed,
                                    _code_given=code_given,
                                    _code_given_mask=code_given_mask)

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
                        _plan_mode: str = "normal",
                        tul_step_mode: str | None = None,
                        _slot_depths: Tensor | None = None,
                        _code_mode: str | None = None,
                        _code_steps: int | None = None,
                        _code_seed: int | None = None,
                        _code_given: Tensor | None = None,
                        _code_given_mask: Tensor | None = None,
                        _coda_state_only: bool = False) -> dict:
        if self._span_mask and slot_layout is not None:
            raise NotImplementedError(
                "model.span_mask with a slot_layout: the TUL forward is a different "
                "region chain and does not thread the span masks. The budget arms run "
                "tul.activate_at: never.")
        if slot_layout is not None:
            if bag_size > 0:
                raise ValueError(
                    "slot_layout and bag_size are mutually exclusive: TUL activates AT the "
                    "TST switch (spec §5), and val/gen always run TUL on with bag_size 0 "
                    "(invariant 6)."
                )
            return self._forward_tul(input_ids, labels, slot_layout, _plan_nats,
                                     halt=_halt, plan_mode=_plan_mode,
                                     tul_step_mode=tul_step_mode,
                                     slot_depths=_slot_depths,
                                     code_mode=_code_mode, code_steps=_code_steps,
                                     code_seed=_code_seed,
                                     code_given=_code_given,
                                     code_given_mask=_code_given_mask,
                                     coda_state_only=_coda_state_only)
        if _coda_state_only:
            raise ValueError(
                "coda_state_only requires slot_layout: it is the graded-continuation "
                "sampler's read path (tul.code_grade) and exists on the TUL forward only.")
        if _slot_depths is not None:
            raise ValueError(
                "slot_depths requires slot_layout: it forces the depth of the SLOT loop, "
                "and the plain path has no slots. Force model.cfg.mean_depth instead.")
        if tul_step_mode is not None:
            raise ValueError(
                "tul_step_mode requires slot_layout (faithful DiffusionBlocks conditions "
                "the TUL slot loop only; there is no core loop to condition here).")
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
        # The span masks, built ONCE per forward from the ids (see `_span_context`).
        _span_kw, _span_core_kw, _span_cut = self._span_context(input_ids)
        if self._span_mask and s > 0:
            raise NotImplementedError(
                "model.span_mask with TST bagging (bag_size > 0): a bagged position is "
                "the mean of s tokens, which can straddle a span boundary, so the span "
                "relation is not defined on the bagged axis. The budget arms run with "
                "training.tst_bag_size 0.")
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
                x, x0, bigram_emb = self._front_region(input_ids, _span_kw, _span_cut)

        # `labels` is read only by LoopMTP's Eq-13 term (training only, and only when
        # model.loopmtp_weight > 0); on every other model it is an unread argument and the
        # core is bit-identical.
        x = self._core_region(x, x0, bigram_emb, input_ids, attn_kwargs=_span_core_kw,
                              labels=labels)

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
            x = self._back_region(x, x0, bigram_emb, input_ids,
                                  attn_kwargs=_span_kw)

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
            if self.mtp is not None:
                self._mtp_apply(out, x, labels, w_full)
            if self._core_aux is not None:
                self._apply_core_aux(out)
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
            if self.mtp is not None:
                self._mtp_apply(out, x, labels, None)
            if self._core_aux is not None:
                if labels is not None:
                    self._apply_core_aux(out)
                else:
                    self._core_aux = None   # never let a label-less forward's stash leak

        return out
