"""KV-cached incremental generation for the STRICT TUL slot-loop model.

``generate_tul`` (``tul_generate.py``) recomputes prelude -> slot loop -> coda over the
whole grown row at every token. This module produces the same tokens and the same
per-step log-probs by computing each position ONCE and caching what later positions read.
It is an inference-only path: it adds no branch to the training forward and reuses the
model's own modules (embeddings, injections, Hyper-Connection residuals, MLPs, the CCA
prologue and gate) for every per-position op. Only the attention READ is re-implemented,
against a cache, with the relation the full forward uses.

WHY THE STRICT GEOMETRY MAKES THIS EXACT (``tul.tg_geometry="strict"``; the relation is
``tul_layout.tg_strict_allow``, the partition ``tg_segment_ids``):

* PRELUDE. A position reads only positions of its own span (``bag_id`` equal, causal),
  and the CCA conv and ``W_v_prev`` value shift are reset at every segment
  (``2*bag_id + slot_mask``). So a token's prelude state depends on its own span's earlier
  tokens only. Cache per prelude layer: the open span's K/V and its conv history; both are
  dropped when the span closes. A slot's seed is the prelude output at its FIRST prefix
  cell, which reads its own span's tokens and itself.
* SLOT LOOP. Runs on the compact slot axis, causal, at the eval depth for every valid slot
  (``mean_depth``; pads sit after every valid slot). Slot ``s`` at pass ``t``, layer ``l``
  reads slots ``<= s`` at the SAME pass and layer, so appending a slot changes no earlier
  slot's state. Cache per (pass, layer): every earlier slot's K/V plus the conv history of
  the compact sequence. The ``tul.code_enum_k`` rollouts are the batch axis (rollout-major,
  ``repeat_along_batch``); each adds its own code after every pass.
* CODA. A token reads its own span's tokens and the prefix cells of EVERY earlier slot
  (``tg_coda_prefix_reach="all"``); a prefix cell reads only itself (and its sibling
  cell through the segment conv). Cache per coda layer: every prefix cell's K/V
  (permanent) plus the open span's token K/V and conv history (dropped at the boundary).
  The per-layer coda injections are zero at the cells (``slot_cell_inject_keep``).
* THE READ. A label-free forward of a ``code_enum_k`` model returns the per-span
  sequential Bayes mixture over the K rollouts (``rollout_mixture``): the weights at a
  position are the softmax over rollouts of the summed log-probs each rollout gave the
  span's earlier tokens. Kept as a running per-span sum.

LXTUL (``K = 1``, ``M = tul.slot_cells`` looped cells per slot; added 2026-10-10, tracker
item C1c; Agent Note ``.agents/notes/proposed/feature/2026-10-10-lxtul-cached-decode.md``):

* The M cells sit at the slot's first M prefix positions. Their prelude reads the span's
  tokens, the earlier sibling cells and itself (one conv segment). The register adds its
  seed term (``TULSlotRegister`` over the span's token rows, cached per span); the
  router's context is the first cell's prelude output (``_lsel_begin``).
* The loop cache holds every cell of every earlier slot per (pass, layer), on the compact
  cell axis ``s*M + i`` (the RoPE position and the conv order). A cell reads every cell of
  earlier slots and every cell of its own slot, later siblings included
  (:func:`loop_relation`). After each pass: the cell norm, then the router's pick and,
  while the depth continues, every cell reset to the winner (``_lsel_pass`` at eval).
* The write keeps the final winner only (losers exactly 0, ``_fan_route_cells``), cell i
  through ``W_prefix[i]``; all ``prefix_k`` cells stay coda keys.
* The read is the one-rollout slot-masked log-softmax, mixed with the pointer head
  (``TULPointer.mixed_logprobs``): the pointer cache holds every earlier token's key and
  the token that followed it, row-wide (it is not span-local).

Everything outside this list is REFUSED at construction (``_check_supported``): a model
whose forward carries a mechanism this file does not reproduce raises instead of
generating from a different function.

Positions: RoPE (CoPE) uses the ABSOLUTE row position in the prelude and coda and the
compact slot index in the loop, exactly as the full forward's ``cos_cached[:S]`` does.
The window branch keeps its ``window_size`` limit and its XSA self-exclusion; the slot
branch keeps its per-head sink logit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import parametrize

import morph.model.attention as _attn_mod
from morph.inference.sampling import sample_next
from morph.inference.tul_generate import TulRowBuilder
from morph.kernels.triton.fused_cca_conv import fused_cca_conv
from morph.kernels.triton.fused_cca_prologue import fused_cca_prologue
from morph.model.attention import segment_causal_conv
from morph.model.tul_fan import _cell_readout
from morph.model.tul_fan_route import reset_to_winner
from morph.model.tul_layout import BoundaryRule, SlotLayout, TulLayoutSpec
from morph.model.tul_pointer import _rms as _ptr_rms

__all__ = ["TulCachedDecoder", "generate_tul_cached"]


# ─────────────────────────────────────────────────────────────────────────────
# What this file reproduces, checked once per decoder
# ─────────────────────────────────────────────────────────────────────────────

# Every optional submodule whose presence changes the label-free eval forward of a slot-
# loop model. Each must be absent: this decoder does not reproduce it.
_MUST_BE_NONE = (
    "fm_planner", "scse", "tul_stage_cond", "tul_cond", "tul_recur_gate", "tul_center",
    "tul_code_enc", "tul_code_proj", "tul_code_time", "tul_code_head", "tul_code_sym",
    "tul_code_cell", "tul_code_clean", "tul_code_null", "tul_code_sym_head",
    "tul_gate", "tul_grad_pass", "tul_gram", "tul_carry", "tul_loop_denoise",
    "tul_pass_gate", "tul_reread", "tul_chain", "tul_code_vq", "tul_vq",
    "mtp",
    # arm B (tul.code_policy_k, 2026-09-29): a per-slot policy choice inside ONE rollout;
    # the loop caches here hold one code per rollout, never a per-slot pick.
    "tul_code_policy",
    # The fan's other routes (a learned gate over the written cells, the factor fan, the
    # pseudo-snap write) and the LX exploration knobs' hypothesis merge: each changes
    # what the coda reads. Only the latent-selected loop's router pick is reproduced.
    "tul_fan_router", "tul_fan_opf", "tul_pseudo_snap", "tul_hyp_merge_gate",
    "tul_hyp_score_head", "tul_latent_pre_head",
)


def loop_cell_count(model, cells: int | None = None) -> int:
    """The number of looped cells per slot ``M`` the decoders run (``tul.slot_cells``; 1 on
    every model without the Thought Register).

    ``cells`` is the decoder's argument for the variable cell count (phase-2 item B: train
    with a sampled M, sweep M at inference). Today every part that reads a cell does so by
    its INDEX (the register's query ``Q_i`` and ``P_cell[i]``, ``W_prefix[i]``, the router's
    per-cell ``bias``), so a forward at any other M is undefined and is REFUSED here rather
    than guessed. The caches, masks and loops below are sized by ``M``, not by the config."""
    m = int(model.cfg.tul.slot_cells)
    if cells is not None and int(cells) != m:
        raise NotImplementedError(
            f"cells={cells}: this model defines its forward at M={m} only (the register's "
            f"per-cell queries, W_prefix[i] and the router's bias are indexed by cell; "
            f"the index-free design is tracker item B1)")
    return m


def _check_supported(model) -> None:
    """RAISE unless ``model``'s label-free forward is the one this decoder reproduces."""
    cfg = model.cfg
    tc = cfg.tul
    if tc is None or model.tul is None:
        raise ValueError("generate_tul_cached needs a TUL model (cfg.tul).")

    def need(cond: bool, what: str) -> None:
        if not cond:
            raise NotImplementedError(f"generate_tul_cached does not reproduce: {what}")

    need(bool(model._tg_strict), "a geometry other than tul.tg_geometry='strict'")
    need(tc.tg_coda_prefix_reach == "all", f"tg_coda_prefix_reach={tc.tg_coda_prefix_reach!r}")
    need(not tc.tokens_through_core, "the paid loop (tul.tokens_through_core)")
    need(not getattr(tc, "loop_reads_tokens", False), "tul.loop_reads_tokens")
    need(not tc.code and not tc.code_target, "TUL-Code / code_target")
    rollouts = model.tul_code_enum is not None
    need(not rollouts or model._code_enum_k > 1, "tul.code_enum_k == 1 with a code module")
    # The fan (fan_k > 0): reproduced ONLY as LXTUL builds it. M = fan_k = slot_cells =
    # prefix_k looped cells per slot (the Thought Register), the write-all route
    # (fan_mix 'all', one cell 1:1 into each prefix cell) and, when the latent-selected
    # loop is on, the router's pick and reset after every pass and the winner-only write.
    # LX-Fan (fan_k > 0 under code_enum_k: K rollouts x M cells) is not built.
    if int(tc.fan_k) > 0 or int(tc.slot_cells) > 1:
        need(not rollouts, "the fan under tul.code_enum_k (LX-Fan's K rollouts x M cells)")
        need(model.tul_fan is not None and model.tul_register is not None
             and int(tc.fan_k) == int(tc.slot_cells) == int(tc.prefix_k),
             "the fan or the Thought Register outside LXTUL's shape (needs fan_k == "
             "slot_cells == prefix_k with both built)")
        need(str(tc.fan_mix) == "all", f"tul.fan_mix={tc.fan_mix!r} (only 'all')")
        need(model._lsel_mode in ("off", "joint", "detached") and not model._lsel_read_all,
             f"tul.fan_loop_select={model._lsel_mode!r} / fan_lsel_read "
             f"{tc.fan_lsel_read!r} (only the winner read)")
        need(not model._fan_lineage, "tul.fan_lineage (the per-stream cross-slot relation)")
        need(int(model._fan_history_streams) == 0, "tul.fan_history_streams")
        need(not model._fan_trigger_every_pass, "tul.fan_trigger_every_pass")
        need(not model._fan_head_grader, "tul.fan_all_wta_grader='head'")
    # Transitively covered by the fan line above (tul.prefix_per_cell > 1 is refused at
    # construction without fan_k > 0 and fan_mix='all'), named explicitly so the message
    # says what is unsupported rather than relying on a reader to trace it back here.
    need(int(tc.prefix_per_cell) == 1, "tul.prefix_per_cell > 1 (the wider write)")
    need(tc.latent_set_loss == "none" and not tc.hyp_score_head and tc.hyp_merge == "none",
         "the LX exploration knobs (latent_set_loss / hyp_score_head / hyp_merge)")
    need(tc.prefix_source == "exit", f"tul.prefix_source={tc.prefix_source!r}")
    need(tc.slot_seed in ("boundary", "bag_mean"), f"tul.slot_seed={tc.slot_seed!r}")
    need(tc.coda_token_input == "prelude", f"tul.coda_token_input={tc.coda_token_input!r}")
    need(bool(tc.coda_sees_slots) and int(tc.coda_token_cut) == 0,
         "a gathered coda (coda_sees_slots false or coda_token_cut > 0)")
    need(not tc.bcast, "tul.bcast (the unpack term)")
    need(int(tc.loop_reach) == 0, "tul.loop_reach")
    # Added after the build: the coda token reach (2026-09-26) widens a coda token's read to
    # the previous spans' TOKENS, which this decoder drops at every boundary.
    need(int(tc.tg_coda_token_reach) == 0, "tul.tg_coda_token_reach (previous-span token read)")
    need(int(tc.xhc_streams) == 0, "tul.xhc_streams (the 16-stream xHC slot loop)")
    # Transitively covered by the fan_k==0 line above (both arms are LXTUL, a fan model),
    # named so the refusal says what is missing: `_block` below builds its own attention
    # sublayer and would drop the centering.
    need(getattr(tc, "loop_attn_center", "off") == "off",
         "tul.loop_attn_center (the centered slot-loop attention)")
    need(getattr(tc, "loop_attn_hc", "cayley") == "cayley",
         "tul.loop_attn_hc (the uniform slot-loop attention residual)")
    if model.tul_pointer is not None:
        # The pointer is cached over every earlier token (`_PointerCache`); its two
        # variants that read more than a token's own final state are not.
        need(not model.tul_pointer.cell_key, "tul.pointer_cell_key (the cell key term)")
        need(not model.tul_pointer.coverage, "tul.pointer_coverage_lambda (the coverage term)")
    need(not tc.slot_source_once, "tul.slot_source_once")
    need(not tc.tg_span_comp, "tul.tg_span_comp")
    need(not tc.center_bag_mean, "tul.center_bag_mean")
    need(not bool(cfg.slot_state_renorm), "model.slot_state_renorm")
    need(float(cfg.core_gain_clip) == 0.0, "model.core_gain_clip")
    need(float(getattr(model, "_fan_seed_noise", 0.0)) == 0.0, "tul.fan_seed_noise")
    need(model._core_stage_cond_mode == "none", "tul.core_stage_cond")
    need(not model._core_is_parcae, "model.core_impl='parcae'")
    need(not model._span_mask, "model.span_mask")
    need(bool(model._is_hc), "a non-Hyper-Connection carrier")
    need(all(i < cfg.n_prelude for i in model._ve_layer_map),
         "value embeddings outside the prelude")
    need(type(model.core_init).__name__ == "_CloneInit",
         f"core_init {type(model.core_init).__name__} (only h_0 = e)")
    need(getattr(model.tul, "E_pass", None) is None, "tul E_pass")
    need(model.tul.W_prefix is not None, "a model without W_prefix")
    for name in _MUST_BE_NONE:
        need(getattr(model, name, None) is None, f"model.{name}")
    for blk in list(model.prelude) + list(model.core) + list(model.coda):
        need(blk.retention is None, "a GLA retention branch")
        need(blk.pass_lora is None, "tul.pass_lora_rank")
        need(bool(blk.attention._impl.tg_restrict), "an attention block without tg_restrict")
        need(blk.attention._impl.tg_span_gate_w is None, "tul.tg_span_gate")
    if bool(tc.slot_depth_fixed):
        need(int(tc.slot_depth_fixed) <= int(tc.slot_max_depth or cfg.max_depth),
             "slot_depth_fixed above max depth")


# ─────────────────────────────────────────────────────────────────────────────
# Cache containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _KV:
    """Keys/values of ONE attention call-site, in position order.

    ``k``/``v`` are post-prologue ``[B, H, N, D]`` (RoPE applied, GQA-expanded) — exactly
    the tensors the full forward's window and slot branches read. ``pos`` ``[N]`` is the
    position each key sits at (absolute row position, or the compact slot index in the
    loop) and ``slot`` ``[N]`` marks slot COLUMNS (what the slot branch may read)."""

    k: Tensor | None = None
    v: Tensor | None = None
    pos: Tensor | None = None
    slot: Tensor | None = None

    def append(self, k: Tensor, v: Tensor, pos: Tensor, slot: Tensor) -> None:
        if self.k is None:
            self.k, self.v, self.pos, self.slot = k, v, pos, slot
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
            self.pos = torch.cat([self.pos, pos])
            self.slot = torch.cat([self.slot, slot])

    def extend(self, other: "_KV") -> None:
        if other.k is not None:
            self.append(other.k, other.v, other.pos, other.slot)


@dataclass
class _ConvHist:
    """The conv / value-shift history of ONE call-site's current segment.

    ``lat`` ``[B, <=2(k-1), dq+dk]``: the raw ``q_lat || k_lat`` rows of the last positions
    of the segment (the two stacked kernel-k causal convs reach ``2(k-1)`` back).
    ``vprev`` ``[B, 1, v_half]``: ``W_v_prev(x)`` of the segment's last position, which is
    the value shift's input for the next one. Both None at a segment start."""

    lat: Tensor | None = None
    vprev: Tensor | None = None


@dataclass
class _Region:
    """Per-layer caches of one region: ``kv[l]`` + ``hist[l]``."""

    n: int
    kv: list[_KV] = field(default_factory=list)
    hist: list[_ConvHist] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.kv = [_KV() for _ in range(self.n)]
        self.hist = [_ConvHist() for _ in range(self.n)]


# ─────────────────────────────────────────────────────────────────────────────
# The one attention site against a cache
# ─────────────────────────────────────────────────────────────────────────────


def _cca_rows(impl, x: Tensor, pos: Tensor, lat_prev: Tensor | None,
              vprev_prev: Tensor | None, seg_mode: bool):
    """The CCA projection of the NEW rows ``x`` ``[B, n, C]`` at positions ``pos`` ``[n]``.

    Mirrors ``_CCABase._cca_project`` (the tg_restrict branch of the attention impl) row
    for row: the fused input GEMM when the forward uses it, the conv over the segment's
    earlier raw latents ``lat_prev`` ``[B, h, dq+dk]`` plus the new rows (the segment-reset
    conv in the prelude/coda, the plain causal conv on the loop's compact axis, the fused
    q||k pair when the forward takes it), the value shift from ``vprev_prev`` ``[B, 1, vh]``
    (``W_v_prev`` of the segment's previous position; ``None`` = segment start), and the
    fused prologue with the cos/sin rows of ``pos``.

    ``None`` history is a segment start. Returns ``q, k, v, q_lat, gate_pre, win,
    v_prev_raw``: ``win`` is the padded conv window (its last ``2(k-1)`` rows are the next
    call's ``lat_prev``) and ``v_prev_raw[:, -1:]`` the next call's ``vprev_prev``."""
    cca = impl.cca
    B, n, _ = x.shape
    dq = cca.latent_q_dim
    gate_pre = None
    if _attn_mod._FUSED_ATTN_PROJ:
        _y, (q_lat, k_lat, v_curr, v_prev_raw, gate_pre) = _attn_mod._fused_x_proj(
            x, impl._fuse_mods)
    else:
        q_lat = cca.W_down_q(x)
        k_lat = cca.W_down_k(x)
        v_curr = cca.W_v_curr(x)
        v_prev_raw = cca.W_v_prev(x)
    kk = cca.conv_q_dw.kernel_size[0]
    need = 2 * (kk - 1)
    lat_new = torch.cat([q_lat, k_lat], dim=-1)                       # [B, n, dq+dk]
    win = lat_new if lat_prev is None else torch.cat([lat_prev, lat_new], dim=1)
    # Left-pad with zero rows to the conv's full reach: a tap on a zero row adds exactly
    # what a tap cut by the segment mask (or the full forward's own left pad) adds, 0,
    # and the conv functions need a sequence at least one kernel long.
    if win.shape[1] < need + n:
        win = F.pad(win, (0, 0, need + n - win.shape[1], 0))
    if seg_mode:
        # The window holds ONE segment (history reset at every segment start), so the
        # segment ids are constant and the conv's left zero-pad is the segment cut.
        seg = torch.zeros(B, win.shape[1], dtype=torch.long, device=x.device)
        wq, wk = win[..., :dq], win[..., dq:]
        q_conv = segment_causal_conv(
            wq.transpose(1, 2), cca.conv_q_dw.weight.to(wq.dtype),
            cca.conv_q_gp.weight.to(wq.dtype), seg).transpose(1, 2)[:, -n:]
        k_conv = segment_causal_conv(
            wk.transpose(1, 2), cca.conv_k_dw.weight.to(wk.dtype),
            cca.conv_k_gp.weight.to(wk.dtype), seg).transpose(1, 2)[:, -n:]
    elif _attn_mod._FUSED_ATTN_PROJ and _attn_mod._FUSED_ATTN_QKCONV:
        w_dw = torch.cat([cca.conv_q_dw.weight, cca.conv_k_dw.weight], dim=0)
        w_gp = torch.cat([cca.conv_q_gp.weight, cca.conv_k_gp.weight], dim=0)
        pair = fused_cca_conv(win.transpose(1, 2), w_dw.to(win.dtype), w_gp.to(win.dtype),
                              cca.n_heads + cca.n_kv_heads, kk)
        q_conv = pair[:, :dq].transpose(1, 2)[:, -n:]
        k_conv = pair[:, dq:].transpose(1, 2)[:, -n:]
    else:
        q_conv = cca._causal_conv(win[..., :dq].transpose(1, 2), cca.conv_q_dw,
                                  cca.conv_q_gp).transpose(1, 2)[:, -n:]
        k_conv = cca._causal_conv(win[..., dq:].transpose(1, 2), cca.conv_k_dw,
                                  cca.conv_k_gp).transpose(1, 2)[:, -n:]
    # Value shift: row i reads W_v_prev(x_{i-1}) of the same segment, 0 at its start.
    first = torch.zeros_like(v_prev_raw[:, :1]) if vprev_prev is None else vprev_prev
    v_prev = torch.cat([first, v_prev_raw[:, :-1]], dim=1)
    cos = cca.rope.cos_cached[:, :, pos]
    sin = cca.rope.sin_cached[:, :, pos]
    q, k, v = fused_cca_prologue(
        q_lat, k_lat, q_conv, k_conv, v_curr, v_prev,
        cca.q_norm.weight, cca.k_norm.weight, cca.temp, cos, sin,
        cca.n_heads, cca.n_kv_heads, cca.d_head, n_skip_rope=0, eps=cca.q_norm.eps)
    return q, k, v, q_lat, gate_pre, win, v_prev_raw


def _project_new(impl, x: Tensor, pos: Tensor, hist: _ConvHist, seg_mode: bool):
    """:func:`_cca_rows` against a growing ``_ConvHist``, which it updates in place.
    Returns ``q, k, v, q_lat, gate_pre``."""
    q, k, v, q_lat, gate_pre, win, v_prev_raw = _cca_rows(
        impl, x, pos, hist.lat, hist.vprev, seg_mode)
    need = 2 * (impl.cca.conv_q_dw.kernel_size[0] - 1)
    hist.lat = win[:, win.shape[1] - need:]
    hist.vprev = v_prev_raw[:, -1:]
    return q, k, v, q_lat, gate_pre


def _branches(impl, q: Tensor, k: Tensor, v: Tensor, win_mask: Tensor,
              k_s: Tensor | None, v_s: Tensor | None, s_allow: Tensor | None):
    """``(out_win, out_comp)`` of queries ``q`` ``[B, H, n, D]`` given the masks.

    Window branch = ``_window_fallback``'s SDPA over a float bias (``win_mask`` ``[n, N]``
    already holds the window size, causality and XSA). A row with no key gets exactly 0,
    what the full forward's SDPA returns for an all ``-inf`` row (morph/model/CLAUDE.md).
    Slot branch = ``_tg_slot_attention``'s fp32 score, per-head sink logit and softmax over
    the slot columns ``k_s``/``v_s`` under ``s_allow`` ``[n, M]``; ``None`` = no slot
    column at all, whose output is exactly 0 (all weight on the zero-valued sink)."""
    cca = impl.cca
    scale = cca.d_head ** -0.5
    B, H, n, _ = q.shape
    # The bias in the query's dtype: CPU SDPA reads an fp32 mask under fp64 queries wrongly
    # once there are more than 16 keys (measured, torch 2.11), which broke the fp64
    # reference the tolerances are argued from. In fp32 the cast is a no-op.
    bias = torch.where(win_mask, 0.0, float("-inf")).to(q.dtype)[None, None]
    out_win = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale)
    has = win_mask.any(dim=-1)
    out_win = torch.where(has.view(1, 1, -1, 1), out_win, torch.zeros_like(out_win))
    if k_s is None:
        return out_win, torch.zeros_like(q)
    scores = torch.einsum("bhid,bhjd->bhij", q.float(), k_s.float()) * scale
    scores = scores.masked_fill(~s_allow.view(1, 1, n, -1), float("-inf"))
    sink = cca.sink_logits.view(1, H, 1, 1).to(scores.dtype).expand(B, H, n, 1)
    scores = torch.cat([scores, sink], dim=-1)
    weights = torch.softmax(scores, dim=-1).to(q.dtype)
    return out_win, torch.einsum("bhij,bhjd->bhid", weights[..., :-1], v_s)


def loop_relation(qpos: Tensor, kpos: Tensor, cells: int) -> Tensor:
    """``[n, N]`` the slot loop's relation on the compact CELL axis (index ``s*M + i``):
    a cell reads every cell of earlier slots and every cell of its OWN slot, later siblings
    included (``transformer.slot_cell_relation`` at reach 0, delivered as ``tg_relation``,
    which REPLACES the attention's causal term). At ``cells == 1`` it is plain causal."""
    if cells == 1:
        return qpos.view(-1, 1) >= kpos.view(1, -1)
    return (qpos.view(-1, 1) // cells) >= (kpos.view(1, -1) // cells)


def _attend(impl, q: Tensor, keys: _KV, qpos: Tensor, self_only: bool, cells: int = 0):
    """``(out_win, out_comp)`` for queries at ``qpos`` over the key set ``keys``.

    ``keys`` is the full set the query's relation can reach (the caller assembles it per
    region); the masks built here add what every region shares: causal order, the window
    size, XSA self-exclusion on the window branch, the slot-column restriction on the slot
    branch, and ``self_only`` (a strict coda prefix cell reads itself alone). ``cells > 0``
    is the slot loop's cell relation (:func:`loop_relation`) in place of causal order; its
    window keeps ``|dist| < window_size`` (``attention._window_fallback`` under a relation),
    which reaches a later sibling."""
    dist = qpos.view(-1, 1) - keys.pos.view(1, -1)
    if self_only:
        allow = dist == 0
    elif cells:
        allow = loop_relation(qpos, keys.pos, cells)
    else:
        allow = dist >= 0
    win_mask = allow & (dist.abs() < impl.cca.window_size) & (dist != 0)  # [n, N]
    cols = keys.slot.nonzero().flatten()
    if cols.numel() == 0:
        return _branches(impl, q, keys.k, keys.v, win_mask, None, None, None)
    return _branches(impl, q, keys.k, keys.v, win_mask, keys.k[:, :, cols],
                     keys.v[:, :, cols], allow[:, cols])


def _site(x: Tensor, *, impl, pos: Tensor, is_slot: Tensor, hist: _ConvHist, seg_mode: bool,
          store: _KV, key_sets: list[_KV], self_only: bool = False, cells: int = 0) -> Tensor:
    """One attention application on NEW rows: project, append to ``store``, attend over
    the union of ``key_sets`` (``store`` is expected to be one of them), gate + up.
    ``cells``: the slot loop's cell relation (:func:`_attend`)."""
    q, k, v, q_lat, gate_pre = _project_new(impl, x, pos, hist, seg_mode)
    store.append(k, v, pos, is_slot)
    if len(key_sets) == 1:
        keys = key_sets[0]
    else:
        live = [ks for ks in key_sets if ks.k is not None]
        keys = _KV(k=torch.cat([ks.k for ks in live], dim=2),
                   v=torch.cat([ks.v for ks in live], dim=2),
                   pos=torch.cat([ks.pos for ks in live]),
                   slot=torch.cat([ks.slot for ks in live]))
    out_win, out_comp = _attend(impl, q, keys, pos, self_only, cells)
    return impl.cca._gate_combine_up(x, out_comp, out_win, q_lat=q_lat, gate_pre=gate_pre)


def _span_mean(rows: list[Tensor]) -> Tensor:
    """``[1, 1, C]`` mean of a span's per-token rows, as ``tul.bag_mean`` forms it (a
    ones-row GEMM over the span, then divided by the count)."""
    x = torch.cat(rows, dim=1)                                           # [1, n, C]
    ones = torch.ones(1, 1, x.shape[1], dtype=x.dtype, device=x.device)
    return torch.bmm(ones, x) / float(x.shape[1])


def _block(block, h: Tensor, attn_call, iter_idx: int) -> Tensor:
    """``MORPHBlock.forward`` with the attention sublayer replaced by ``attn_call``.

    The block's own norms, Hyper-Connection residuals, MLP and dropout, in its order
    (``_check_supported`` refuses retention and pass-LoRA, the two things skipped)."""
    def _attn_fn(x: Tensor) -> Tensor:
        return block.drop(attn_call(block.norm_attn(x)))

    def _mlp_fn(x: Tensor) -> Tensor:
        return block.drop(block.mlp(block.norm_mlp(x), iter_idx=iter_idx))

    h = block.mrr_attn(h, _attn_fn)
    return block.mrr_mlp(h, _mlp_fn)


# ─────────────────────────────────────────────────────────────────────────────
# Shared per-slot / per-token pieces of LXTUL (the cached and the graphed decoder)
# ─────────────────────────────────────────────────────────────────────────────


def register_term(model, xs: Tensor, valid: Tensor | None = None) -> Tensor:
    """``[1, M, C]``: the Thought Register's seed term for ONE slot (``TULSlotRegister``).

    ``xs`` ``[1, n, C]`` are the span's token rows the full forward pools: the prelude
    output after ``input_norm``, stream mean. ``valid`` ``[n]`` marks the live rows of a
    fixed-capacity buffer (``None``: all). The model's own module runs on a one-slot layout
    whose span is exactly those rows, so the pooling is the forward's, not a copy."""
    n = xs.shape[1]
    dev = xs.device
    bag = (torch.zeros(1, n, dtype=torch.long, device=dev) if valid is None
           else (~valid).long().view(1, n))                  # a dead row joins bag 1
    lay = SlotLayout(slot_mask=torch.zeros(1, n, dtype=torch.bool, device=dev), bag_id=bag,
                     slot_index=torch.zeros(1, 1, dtype=torch.long, device=dev),
                     slot_valid=torch.ones(1, 1, dtype=torch.bool, device=dev),
                     prefix_k=int(model.cfg.tul.prefix_k))
    return model.tul_register(xs, lay)


def lsel_step(model, h: Tensor, ctx: Tensor, m: int, reset: bool) -> tuple[Tensor, Tensor]:
    """One pass's latent-selected-loop step for ONE slot (``_lsel_pass`` at eval, where
    the loop follows the router): the router scores the ``m`` cells of ``h`` ``[1, m, *, C]``
    against ``ctx`` ``[1, 1, C]``, picks one, and (``reset``: the slot's depth continues)
    every cell takes the winner's state (``reset_to_winner``). Returns ``(h, pick [1, 1])``."""
    cr = _cell_readout(h.view(1, 1, m, *h.shape[2:]))
    pick = model.tul_fan_lsel_router.select(model.tul_fan_lsel_router.scores(cr, ctx))[0]
    if reset:
        h = reset_to_winner(h, pick, torch.ones(1, 1, dtype=torch.bool, device=h.device), m)
    return h, pick


def prefix_write(model, h: Tensor, pick: Tensor | None, kp: int) -> Tensor:
    """``[R, kp, *, C]`` the slot's prefix-cell values (``TULSlots.prefix_project``) from the
    loop's exit ``h`` ``[R, m, *, C]``: one cell (``m == 1``) through every ``W_prefix[k]``;
    ``m == kp`` cells 1:1, after the winner route (``_fan_route_cells``: losers exactly 0)
    when ``pick`` is given."""
    w = model.tul.W_prefix.to(h.dtype)
    m = h.shape[1]
    if m == 1:
        return torch.stack([torch.matmul(h[:, 0], w[k]) for k in range(kp)], dim=1)
    if pick is not None:
        # one_hot(pick) as a comparison: the same exact 0 / 1, and no value check that
        # could sync with the host inside a CUDA graph.
        oh = (torch.arange(m, device=h.device) == pick.view(-1, 1)).to(h.dtype)   # [1, m]
        h = h * oh.view(1, m, *([1] * (h.dim() - 2)))
    return torch.stack([torch.matmul(h[:, k], w[k]) for k in range(kp)], dim=1)


def pointer_keys(model, h: Tensor) -> tuple[Tensor, Tensor]:
    """``(q, k)`` ``[H, dh]`` fp32 of one token's final hidden state ``h`` ``[1, C]``
    (``TULPointer.attend``: per-head RMS-normed projections)."""
    p = model.tul_pointer
    q = _ptr_rms(p.q(h).view(p.h, p.dh)).float()
    k = _ptr_rms(p.k(h).view(p.h, p.dh)).float()
    return q, k


def pointer_mix(model, h: Tensor, lm: Tensor, q: Tensor, keys: Tensor, cand: Tensor,
                valid: Tensor | None = None) -> Tensor:
    """``[V]`` the model + pointer mixture at one token (``TULPointer.mixed_logprobs``).

    ``h`` ``[1, C]`` the token's final hidden state, ``lm`` ``[V]`` the model's slot-masked
    log-softmax, ``q`` ``[H, dh]`` its query; ``keys`` ``[H, N, dh]`` every EARLIER token's
    key in token order and ``cand`` ``[N]`` the token that followed each (the next token's
    id: every earlier token has one); ``valid`` ``[N]`` the live entries of a fixed-capacity
    buffer. The null key's abstain mass goes back to the model."""
    p = model.tul_pointer
    scale = (p.log_scale_bias.exp() / math.sqrt(p.dh)).float()                     # [H]
    s = torch.einsum("hd,hnd->hn", q, keys) * scale[:, None]
    if valid is not None:
        s = s.masked_fill(~valid.view(1, -1), float("-inf"))
    s_null = (q * _ptr_rms(p.null_k_bias).float()).sum(-1) * scale                  # [H]
    pr = torch.softmax(torch.cat([s, s_null[:, None]], -1), -1)
    alpha, null = pr[:, :-1], pr[:, -1]
    a = torch.softmax(p.gate_norm_head(h).float(), -1)[0]                          # [1+H]
    w0 = a[0] + (a[1:] * null).sum()
    beta = (alpha * a[1:, None]).sum(0)                                            # [N]
    if valid is not None:
        beta = beta * valid.to(beta.dtype)
    p_ptr = torch.zeros_like(lm).scatter_add_(0, cand.clamp_min(0), beta)
    return torch.log(w0 * lm.exp() + p_ptr)


# ─────────────────────────────────────────────────────────────────────────────
# The decoder
# ─────────────────────────────────────────────────────────────────────────────


class TulCachedDecoder:
    """Incremental state of ONE row. Feed it tokens in row order with the slot decisions
    ``TulRowBuilder.append`` made; read the log-probs it produced.

    Two model families share this state machine (``_check_supported`` admits nothing else):

    * the K-rollout model (``tul.code_enum_k = K > 1``, one looped cell): the rollouts are
      the batch axis and the read is the per-span Bayes mixture;
    * LXTUL (``K = 1``): ``M = tul.slot_cells`` looped cells per slot on the cache's
      position axis, the register's seed, the per-pass cell norm, the router's pick and
      reset, the winner-only write, the one-rollout read and the pointer head. ``M = 1``
      with no fan is the plain one-cell slot loop.

    The builder stays the ONE source of the layout: this class never decides a boundary,
    it only mirrors the positions the builder assigned (so a divergence from the loader's
    layout is impossible here by construction, and ``generate_tul``'s parity test covers
    the builder)."""

    def __init__(self, model, spec: TulLayoutSpec, cells: int | None = None):
        _check_supported(model)
        self.m = model
        self.spec = spec
        cfg, tc = model.cfg, model.cfg.tul
        if spec.prefix_k != tc.prefix_k:
            raise ValueError(f"spec prefix_k {spec.prefix_k} != model {tc.prefix_k}")
        self.K = int(model._code_enum_k) if model.tul_code_enum is not None else 1
        self.M = loop_cell_count(model, cells)
        self.lsel = model._lsel_mode != "off"
        self.depth = int(tc.slot_depth_fixed) if int(tc.slot_depth_fixed) > 0 else int(
            tc.slot_mean_depth or cfg.mean_depth)
        self.np, self.nc, self.nd = cfg.n_prelude, cfg.n_core, cfg.n_coda
        self.slot_id = int(tc.slot_id)
        self.dev = next(model.parameters()).device
        self.max_pos = int(model.prelude[0].attention._impl.cca.rope.cos_cached.shape[2])
        self.w_head: Tensor | None = None
        self._slot_col_idx = torch.tensor([self.slot_id], device=self.dev)
        self.pre = _Region(self.np)
        self.coda_span = _Region(self.nd)
        self.coda_cells = [_KV() for _ in range(self.nd)]
        self.core = [_Region(self.nc) for _ in range(self.depth)]
        self.n_slots = 0
        self.prev_id = 0                   # the row's previous input id (bigram key)
        self.span_bigram: list[Tensor] = []
        self.span_emb: list[Tensor] = []
        self.span_xn: list[Tensor] = []    # the register's pooled rows (register only)
        self.n_ve = len(model._ve_layer_map)
        self.span_ve: list[list[Tensor]] = [[] for _ in range(self.n_ve)]
        self.seed_mode = str(tc.slot_seed)
        # The pointer's cache: every earlier token's key and the token that followed it
        # (row-wide, never dropped), plus the newest token's key, whose follower is the
        # NEXT fed token.
        self.ptr = model.tul_pointer is not None
        self.ptr_keys: Tensor | None = None       # [H, N, dh] fp32
        self.ptr_cand: Tensor | None = None       # [N] long
        self.ptr_pending: Tensor | None = None    # [H, 1, dh]
        self.C: Tensor | None = None        # [K] fp64 per-span evidence sum
        self.last_logp: Tensor | None = None   # [K, V] of the open span's last token
        self.logits: Tensor | None = None   # [V] log-probs of the last token
        self.cell_logits: Tensor | None = None  # [V] at the last prefix cell (emit "slot")

    # -- per-position pieces ------------------------------------------------------
    def _front_inputs(self, tok: int):
        m = self.m
        ids = torch.tensor([[tok]], dtype=torch.long, device=self.dev)
        emb = m.embed(ids)                                           # [1, 1, C]
        x = m.embed_drop(emb)
        bg = m.embed.get_bigram(torch.tensor([[self.prev_id, tok]], dtype=torch.long,
                                             device=self.dev))[:, 1:]
        # Value embeddings (prelude only): the per-token signal `_tul_front` hands the
        # prelude as `ve_bagged` (unchanged at token positions).
        ve = [m.value_embeds[k].precompute(m.value_embed_tables[k](ids))
              for k in range(self.n_ve)]
        return emb, x, bg, ve

    def _prelude(self, x: Tensor, x0: Tensor, bg: Tensor, ve: list[Tensor], pos: Tensor,
                 is_slot: Tensor, cell: bool) -> Tensor:
        m = self.m
        h = x.unsqueeze(2).expand(x.shape[0], x.shape[1], m._n_streams,
                                  x.shape[-1]).contiguous()
        for i, blk in enumerate(m.prelude):
            term = m._build_injection_term(i, m.x0_injects[i].precompute(x0), None, bg, h.dtype,
                                           ve_bagged=ve if self.n_ve else None)
            h = m._apply_injection(h, term)
            impl = blk.attention._impl
            if cell:
                # A slot's cells: their own conv segment, each reading its span's tokens,
                # its earlier sibling cells and itself (same bag, causal).
                tmp = _KV()
                call = partial(_site, impl=impl, pos=pos, is_slot=is_slot, hist=_ConvHist(),
                               seg_mode=True, store=tmp, key_sets=[self.pre.kv[i], tmp])
            else:
                call = partial(_site, impl=impl, pos=pos, is_slot=is_slot,
                               hist=self.pre.hist[i], seg_mode=True, store=self.pre.kv[i],
                               key_sets=[self.pre.kv[i]])
            h = _block(blk, h, call, 0)
        return m.input_norm(h)

    def _coda(self, xc: Tensor, x0: Tensor, bg: Tensor, pos: Tensor, is_slot: Tensor,
              keep: float, cells: bool) -> Tensor:
        m = self.m
        base_i = self.np + self.nc
        for i, blk in enumerate(m.coda):
            gi = base_i + i
            term = m._build_injection_term(gi, m.x0_injects[gi].precompute(x0), None, bg,
                                           xc.dtype)
            term = term * torch.full((1, 1, 1), keep, dtype=term.dtype, device=term.device)
            xc = m._apply_injection(xc, term)
            impl = blk.attention._impl
            if cells:
                tmp = _KV()
                call = partial(_site, impl=impl, pos=pos, is_slot=is_slot, hist=_ConvHist(),
                               seg_mode=True, store=tmp, key_sets=[tmp], self_only=True)
                xc = _block(blk, xc, call, 0)
                self.coda_cells[i].extend(tmp)
            else:
                call = partial(_site, impl=impl, pos=pos, is_slot=is_slot,
                               hist=self.coda_span.hist[i], seg_mode=True,
                               store=self.coda_span.kv[i],
                               key_sets=[self.coda_cells[i], self.coda_span.kv[i]])
                xc = _block(blk, xc, call, 0)
        return m._readout(xc)

    def _rep(self, t: Tensor) -> Tensor:
        return t.repeat(self.K, *([1] * (t.dim() - 1))) if self.K > 1 else t

    def _logp(self, xh: Tensor) -> Tensor:
        """``[K, V]`` fp32 slot-masked log-softmax per rollout (``_enum_mixture_logprobs``)."""
        logits = (xh[:, -1] @ self.w_head.T).index_fill(-1, self._slot_col_idx, float("-inf"))
        return torch.log_softmax(logits.float(), dim=-1)

    def _mix(self, logp: Tensor, C: Tensor) -> Tensor:
        logw = torch.log_softmax(C, dim=0).float()                   # [K]
        acc = None
        for k in range(self.K):
            term = logp[k].float() + logw[k]
            acc = term if acc is None else torch.logaddexp(acc, term)
        return acc

    def _pointer(self, xh: Tensor, lm: Tensor, tok: int) -> Tensor:
        """The pointer mixture at the token just fed (``lm`` its model log-probs), then its
        key joins the cache; the previous token's key gets its follower, this token."""
        h = xh[:, -1]                                                     # [1, C]
        q, k = pointer_keys(self.m, h)
        if self.ptr_pending is not None:
            c = torch.tensor([tok], dtype=torch.long, device=self.dev)
            if self.ptr_keys is None:
                self.ptr_keys, self.ptr_cand = self.ptr_pending, c
            else:
                self.ptr_keys = torch.cat([self.ptr_keys, self.ptr_pending], dim=1)
                self.ptr_cand = torch.cat([self.ptr_cand, c])
        self.ptr_pending = k.unsqueeze(1)
        keys = (self.ptr_keys if self.ptr_keys is not None
                else k.new_zeros(k.shape[0], 0, k.shape[1]))
        cand = (self.ptr_cand if self.ptr_cand is not None
                else torch.zeros(0, dtype=torch.long, device=self.dev))
        return pointer_mix(self.m, h, lm, q, keys, cand)

    # -- the two events ----------------------------------------------------------
    def feed_token(self, tok: int, pos: int) -> None:
        """One TOKEN at row position ``pos``: prelude (span-local), coda, the read."""
        if pos >= self.max_pos:
            raise ValueError(f"row position {pos} exceeds the RoPE cache {self.max_pos}")
        emb, x, bg, ve = self._front_inputs(tok)
        self.span_bigram.append(bg)
        self.span_emb.append(emb)
        for k in range(self.n_ve):
            self.span_ve[k].append(ve[k])
        p = torch.tensor([pos], dtype=torch.long, device=self.dev)
        no = torch.zeros(1, dtype=torch.bool, device=self.dev)
        xn = self._prelude(x, x, bg, ve, p, no, cell=False)
        if self.m.tul_register is not None:
            self.span_xn.append(xn.mean(dim=2))
        xh = self._coda(self._rep(xn), self._rep(x), self._rep(bg), p, no, 1.0, cells=False)
        logp = self._logp(xh)
        if self.K > 1:
            if self.C is None:
                self.C = torch.zeros(self.K, dtype=torch.float64, device=self.dev)
            elif self.last_logp is not None:
                # The previous position is a token of THIS span: its label (this token)
                # is evidence for every later position of the span (``evidence_labels``).
                self.C = self.C + self.last_logp[:, tok].double()
            self.logits = self._mix(logp, self.C)
            self.last_logp = logp
        else:
            self.logits = self._pointer(xh, logp[0], tok) if self.ptr else logp[0]
        self.prev_id = tok

    def feed_slot(self, first_pos: int, want_cell_logits: bool) -> None:
        """The slot the builder just inserted after the span's last token, its
        ``prefix_k`` cells at ``first_pos ..``: seed, prelude of the ``M`` looped cells,
        the loop for the new slot (cached per pass and layer), the prefix write and the
        coda of the cells. Then the span-local caches are dropped."""
        m, tc = self.m, self.m.cfg.tul
        s = self.n_slots
        Kp, Mc = tc.prefix_k, self.M
        if first_pos + Kp - 1 >= self.max_pos:
            raise ValueError(f"row position {first_pos + Kp - 1} exceeds the RoPE cache")
        dtype = self.span_emb[-1].dtype
        # ── the seed (``TULSlots.slot_input``) and the bigram's span bag mean ────────
        bag = torch.tensor([[s]], dtype=torch.long, device=self.dev)
        if self.seed_mode == "boundary":           # E_slot + W_sent . embed(t_last)
            seed = m.tul.W_sent(self.span_emb[-1].to(dtype)) + m.tul._e_slot_term(bag, dtype)
        else:                                      # "bag_mean": E_slot + span mean
            seed = _span_mean(self.span_emb) + m.tul._e_slot_term(bag, dtype)
        x_cell = m.embed_drop(seed)
        bg_cell = _span_mean(self.span_bigram)
        ve_cell = [_span_mean(self.span_ve[k]) for k in range(self.n_ve)]
        # ── the prelude of the looped cells (every prefix position carries the seed) ──
        pm = torch.arange(first_pos, first_pos + Mc, dtype=torch.long, device=self.dev)
        ym = torch.ones(Mc, dtype=torch.bool, device=self.dev)
        x_m, bg_m = x_cell.expand(1, Mc, -1), bg_cell.expand(1, Mc, -1)
        ve_m = [v.expand(1, Mc, -1) for v in ve_cell]
        e = self._prelude(x_m, x_m, bg_m, ve_m, pm, ym, cell=True)      # [1, M, n, C]
        ctx = e[:, :1].mean(dim=2) if self.lsel else None                # [1, 1, C]
        if m.tul_register is not None:
            e = e + register_term(m, torch.cat(self.span_xn, dim=1)).unsqueeze(2).to(e.dtype)
        # ── the slot loop for slot s: K rollouts on the batch axis, M cells on the
        #    compact cell axis (positions s*M + i) ──────────────────────────────────────
        eK = self._rep(e)
        x0K, bgK = self._rep(x_m), self._rep(bg_m)
        h = m.core_init(eK)
        np_ = self.np
        inj = torch.stack(
            [m._build_injection_term(np_ + i, m.x0_injects[np_ + i].precompute(x0K), None,
                                     bgK, h.dtype) for i in range(self.nc)], dim=0)
        sp = torch.arange(s * Mc, s * Mc + Mc, dtype=torch.long, device=self.dev)
        valid = torch.ones(self.K, 1, dtype=torch.bool, device=self.dev)
        pick = None
        for t in range(self.depth):
            reg = self.core[t]
            hi = m.injection(h, eK)
            for i, blk in enumerate(m.core):
                hi = m._apply_injection(hi, inj[i])
                impl = blk.attention._impl
                call = partial(_site, impl=impl, pos=sp, is_slot=ym, hist=reg.hist[i],
                               seg_mode=False, store=reg.kv[i], key_sets=[reg.kv[i]],
                               cells=Mc)
                hi = _block(blk, hi, call, t)
            h = hi
            if self.K > 1:
                h = m._apply_injection(hi, m.tul_code_enum.term(hi, valid, self.K))
            if m.tul_cell_norm is not None:
                # tul.slot_cell_pass_norm: the full forward norms every carried cell after
                # every pass, after the code term (`_tul_core` -> `_slot_cell_norm`).
                h = m._cell_norm_apply(h)
            if self.lsel:
                h, pick = lsel_step(m, h, ctx, Mc, reset=t + 1 < self.depth)
        # ── the prefix write and the coda of the cells (self-only, injections zeroed) ─
        values = prefix_write(m, h, pick, Kp)
        pc = torch.arange(first_pos, first_pos + Kp, dtype=torch.long, device=self.dev)
        yk = torch.ones(Kp, dtype=torch.bool, device=self.dev)
        x0c = self._rep(x_cell.expand(1, Kp, -1))
        bgc = self._rep(bg_cell.expand(1, Kp, -1))
        xh = self._coda(values.to(eK.dtype), x0c, bgc, pc, yk, 0.0, cells=True)
        if want_cell_logits:
            lp = self._logp(xh)
            self.cell_logits = self._mix(lp, self.C) if self.K > 1 else lp[0]
        else:
            self.cell_logits = None
        # ── the span is closed: drop everything span-local ─────────────────────────
        self.n_slots += 1
        self.pre.reset()
        self.coda_span.reset()
        self.span_bigram = []
        self.span_emb = []
        self.span_xn = []
        self.span_ve = [[] for _ in range(self.n_ve)]
        self.C = None
        self.last_logp = None
        self.prev_id = self.slot_id


@torch.no_grad()
def generate_tul_cached(
    model,
    prompt_ids: list[int] | Tensor,
    rule: BoundaryRule,
    spec: TulLayoutSpec,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
    device=None,
    emit_source: str = "slot",
    return_logits: bool = False,
    top_p: float = 0.0,
    cells: int | None = None,
):
    """``generate_tul`` with the KV cache: same arguments (no ``halt``: the gate is
    refused; ``top_p`` is the eager generator's nucleus knob, default off), same sampling
    step, same returned ``(token_ids, builder)``. With ``return_logits`` a third value, the
    ``[V]`` log-prob row each step sampled from. ``cells``: the looped cell count M
    (:func:`loop_cell_count`; ``None`` = the model's)."""
    if emit_source not in ("slot", "token"):
        raise ValueError(f"emit_source must be 'slot' or 'token', got {emit_source!r}")
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    gen = None
    if seed is not None:
        gen = torch.Generator(device=str(device)).manual_seed(seed)
    if isinstance(prompt_ids, Tensor):
        prompt_ids = prompt_ids.flatten().tolist()
    if not prompt_ids:
        raise ValueError("generate_tul_cached needs at least one prompt token")
    builder = TulRowBuilder(rule=rule, spec=spec)
    dec = TulCachedDecoder(model, spec, cells)
    emitted: list[int] = []
    rows: list[Tensor] = []
    try:
        with parametrize.cached():
            # The QAT parametrizations (ternary MLP weights, int6 embedding) are the same
            # tensors on every call within one generation; cache them once instead of per
            # position. `lm_weight()` is likewise one tensor for the whole call.
            dec.w_head = model.embed.lm_weight()

            def _feed(tok: int) -> None:
                pos = len(builder.ids)
                cut = builder.append(int(tok))
                dec.feed_token(int(tok), pos)
                if cut:
                    dec.feed_slot(builder.slot_first[-1], want_cell_logits=(
                        emit_source == "slot"))

            for t in prompt_ids:
                _feed(int(t))
            for step in range(max_new_tokens):
                ends_in_slot = builder.slot_mask[-1]
                logits = (dec.cell_logits if (ends_in_slot and emit_source == "slot")
                          else dec.logits)
                if return_logits:
                    rows.append(logits.detach().clone())
                nxt = sample_next(logits, temperature, top_k, gen, top_p)
                emitted.append(nxt)
                if step + 1 < max_new_tokens:
                    _feed(nxt)
                else:
                    builder.append(int(nxt))      # the layout of the last token, no forward
    finally:
        if was_training:
            model.train()
    if return_logits:
        return emitted, builder, rows
    return emitted, builder
