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

Everything outside this list is REFUSED at construction (``_check_supported``): a model
whose forward carries a mechanism this file does not reproduce raises instead of
generating from a different function.

Positions: RoPE (CoPE) uses the ABSOLUTE row position in the prelude and coda and the
compact slot index in the loop, exactly as the full forward's ``cos_cached[:S]`` does.
The window branch keeps its ``window_size`` limit and its XSA self-exclusion; the slot
branch keeps its per-head sink logit.
"""

from __future__ import annotations

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
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec

__all__ = ["TulCachedDecoder", "generate_tul_cached"]


# ─────────────────────────────────────────────────────────────────────────────
# What this file reproduces, checked once per decoder
# ─────────────────────────────────────────────────────────────────────────────

# Every optional submodule whose presence changes the label-free eval forward of a slot-
# loop model. Each must be absent: this decoder does not reproduce it.
_MUST_BE_NONE = (
    "fm_planner", "scse", "tul_stage_cond", "tul_cond", "tul_recur_gate", "tul_center",
    "tul_code_enc", "tul_code_proj", "tul_code_time", "tul_code_head", "tul_code_sym",
    "tul_fan", "tul_gate", "tul_grad_pass", "tul_gram", "tul_carry", "tul_loop_denoise",
    "tul_pass_gate", "tul_reread", "tul_chain", "tul_register", "tul_code_vq", "tul_vq",
    "mtp",
    # arm B (tul.code_policy_k, 2026-09-29): a per-slot policy choice inside ONE rollout;
    # the loop caches here hold one code per rollout, never a per-slot pick.
    "tul_code_policy",
)


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
    # LX-Fan (fan_k > 0 under code_enum_k, 2026-09-26): the loop caches here hold ONE cell
    # per slot per rollout and the coda cell cache one write per slot; M write-all cells
    # per slot were not built. Named before the register line so the refusal says so.
    need(int(tc.fan_k) == 0, "the fan (tul.fan_k > 0, incl. LX-Fan's K rollouts x M cells)")
    need(int(tc.slot_cells) == 1, "the Thought Register (tul.slot_cells > 1)")
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
    need(model.tul_code_enum is not None and model._code_enum_k > 1,
         "a model without tul.code_enum_k > 1 (the plain one-rollout read)")
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
    bias = torch.where(win_mask, 0.0, float("-inf"))[None, None]
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


def _attend(impl, q: Tensor, keys: _KV, qpos: Tensor, self_only: bool):
    """``(out_win, out_comp)`` for queries at ``qpos`` over the key set ``keys``.

    ``keys`` is the full set the query's relation can reach (the caller assembles it per
    region); the masks built here add what every region shares: causal order, the window
    size, XSA self-exclusion on the window branch, the slot-column restriction on the slot
    branch, and ``self_only`` (a strict coda prefix cell reads itself alone)."""
    dist = qpos.view(-1, 1) - keys.pos.view(1, -1)
    allow = (dist == 0) if self_only else (dist >= 0)
    win_mask = allow & (dist < impl.cca.window_size) & (dist != 0)       # [n, N]
    cols = keys.slot.nonzero().flatten()
    if cols.numel() == 0:
        return _branches(impl, q, keys.k, keys.v, win_mask, None, None, None)
    return _branches(impl, q, keys.k, keys.v, win_mask, keys.k[:, :, cols],
                     keys.v[:, :, cols], allow[:, cols])


def _site(x: Tensor, *, impl, pos: Tensor, is_slot: Tensor, hist: _ConvHist, seg_mode: bool,
          store: _KV, key_sets: list[_KV], self_only: bool = False) -> Tensor:
    """One attention application on NEW rows: project, append to ``store``, attend over
    the union of ``key_sets`` (``store`` is expected to be one of them), gate + up."""
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
    out_win, out_comp = _attend(impl, q, keys, pos, self_only)
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
# The decoder
# ─────────────────────────────────────────────────────────────────────────────


class TulCachedDecoder:
    """Incremental state of ONE row. Feed it tokens in row order with the slot decisions
    ``TulRowBuilder.append`` made; read the mixture log-probs it produced.

    The builder stays the ONE source of the layout: this class never decides a boundary,
    it only mirrors the positions the builder assigned (so a divergence from the loader's
    layout is impossible here by construction, and ``generate_tul``'s parity test covers
    the builder)."""

    def __init__(self, model, spec: TulLayoutSpec):
        _check_supported(model)
        self.m = model
        self.spec = spec
        cfg, tc = model.cfg, model.cfg.tul
        if spec.prefix_k != tc.prefix_k:
            raise ValueError(f"spec prefix_k {spec.prefix_k} != model {tc.prefix_k}")
        self.K = int(model._code_enum_k)
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
        self.n_ve = len(model._ve_layer_map)
        self.span_ve: list[list[Tensor]] = [[] for _ in range(self.n_ve)]
        self.seed_mode = str(tc.slot_seed)
        self.C: Tensor | None = None        # [K] fp64 per-span evidence sum
        self.last_logp: Tensor | None = None   # [K, V] of the open span's last token
        self.logits: Tensor | None = None   # [V] mixture log-probs of the last token
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
        return t.repeat(self.K, *([1] * (t.dim() - 1)))

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

    # -- the two events ----------------------------------------------------------
    def feed_token(self, tok: int, pos: int) -> None:
        """One TOKEN at row position ``pos``: prelude (span-local), coda, mixture read."""
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
        xh = self._coda(self._rep(xn), self._rep(x), self._rep(bg), p, no, 1.0, cells=False)
        logp = self._logp(xh)
        if self.C is None:
            self.C = torch.zeros(self.K, dtype=torch.float64, device=self.dev)
        elif self.last_logp is not None:
            # The previous position is a token of THIS span: its label (this token) is
            # evidence for every later position of the span (``evidence_labels``).
            self.C = self.C + self.last_logp[:, tok].double()
        self.logits = self._mix(logp, self.C)
        self.last_logp = logp
        self.prev_id = tok

    def feed_slot(self, first_pos: int, want_cell_logits: bool) -> None:
        """The slot the builder just inserted after the span's last token, its
        ``prefix_k`` cells at ``first_pos ..``: seed, prelude of the first cell, the loop
        for the new slot (K rollouts, cached per pass and layer), the prefix write and the
        coda of the cells. Then the span-local caches are dropped."""
        m, tc = self.m, self.m.cfg.tul
        s = self.n_slots
        Kp = tc.prefix_k
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
        p1 = torch.tensor([first_pos], dtype=torch.long, device=self.dev)
        yes = torch.ones(1, dtype=torch.bool, device=self.dev)
        ve_cell = [_span_mean(self.span_ve[k]) for k in range(self.n_ve)]
        e = self._prelude(x_cell, x_cell, bg_cell, ve_cell, p1, yes, cell=True)   # [1,1,n,C]
        # ── the slot loop for slot s, K rollouts on the batch axis ──────────────────
        eK = self._rep(e)
        x0K, bgK = self._rep(x_cell), self._rep(bg_cell)
        h = m.core_init(eK)
        np_ = self.np
        inj = torch.stack(
            [m._build_injection_term(np_ + i, m.x0_injects[np_ + i].precompute(x0K), None,
                                     bgK, h.dtype) for i in range(self.nc)], dim=0)
        sp = torch.tensor([s], dtype=torch.long, device=self.dev)
        valid = torch.ones(self.K, 1, dtype=torch.bool, device=self.dev)
        for t in range(self.depth):
            reg = self.core[t]
            hi = m.injection(h, eK)
            for i, blk in enumerate(m.core):
                hi = m._apply_injection(hi, inj[i])
                impl = blk.attention._impl
                call = partial(_site, impl=impl, pos=sp, is_slot=yes, hist=reg.hist[i],
                               seg_mode=False, store=reg.kv[i], key_sets=[reg.kv[i]])
                hi = _block(blk, hi, call, t)
            h = m._apply_injection(hi, m.tul_code_enum.term(hi, valid, self.K))
        # ── the prefix write: cell k = h W_prefix[k] (``TULSlots.prefix_project``) ────
        w = m.tul.W_prefix.to(h.dtype)
        C = h.shape[-1]
        hm = h.reshape(self.K, 1, -1, C)
        proj = torch.stack([torch.matmul(hm, w[k]) for k in range(Kp)], dim=2)
        values = proj.reshape(self.K, Kp, *h.shape[2:-1], C)
        # ── the coda of the cells (self-only relation, injections zeroed) ─────────────
        pc = torch.arange(first_pos, first_pos + Kp, dtype=torch.long, device=self.dev)
        yk = torch.ones(Kp, dtype=torch.bool, device=self.dev)
        x0c = self._rep(x_cell.expand(1, Kp, -1))
        bgc = self._rep(bg_cell.expand(1, Kp, -1))
        xh = self._coda(values.to(eK.dtype), x0c, bgc, pc, yk, 0.0, cells=True)
        self.cell_logits = (self._mix(self._logp(xh), self.C) if want_cell_logits
                            else None)
        # ── the span is closed: drop everything span-local ─────────────────────────
        self.n_slots += 1
        self.pre.reset()
        self.coda_span.reset()
        self.span_bigram = []
        self.span_emb = []
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
):
    """``generate_tul`` with the KV cache: same arguments (no ``halt``: the gate is
    refused), same sampling step, same returned ``(token_ids, builder)``. With
    ``return_logits`` a third value, the ``[V]`` log-prob row each step sampled from."""
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
    dec = TulCachedDecoder(model, spec)
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
                nxt = sample_next(logits, temperature, top_k, gen)
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
