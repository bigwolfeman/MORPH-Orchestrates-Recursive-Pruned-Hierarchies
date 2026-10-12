"""Run only the live cells of each slot-loop pass (``model.slot_compact``).

The LXTUL slot loop (``MORPHTransformer._tul_core``) runs every pass over all ``S * M`` cells
and freezes the cells whose slot has finished with ``torch.where(active, h_new, h)``. About
40 % of that work is thrown away (audit 2026-10-08). The finished cells still matter: a later
slot's attention reads every cell of every earlier slot at every later pass.

The two keys of this module change that, in two steps that can be measured apart:

``model.slot_depth_stratified`` (a different JOINT law of the depths, same marginal)
    Per row, the ``S`` slot depths are a SYSTEMATIC sample of the clamped Poisson law
    ``D = clamp(Poisson(mean), 1, max)``: one ``U ~ Uniform[0, 1)`` per row, slot ``k`` of
    the sorted sample gets the quantile at ``(k + U) / S``, and a uniform random permutation
    spreads the sample over the row's slots. A slot sees ``(k + U) / S`` with ``k`` uniform,
    which is ``Uniform[0, 1)``, so EVERY slot's marginal law is exactly the clamped Poisson
    one. What changes is the joint law: a row can no longer draw many deep slots at once.

    The capacity bound (proved here, pinned in tests/test_slot_compact.py). Let
    ``G_t = S * P(D <= t)`` be the float64 number the draw compares against. A slot of the
    sorted sample is deeper than ``t`` iff ``x_k > G_t``, where ``x_k = fl(k + U)`` lies in
    ``[k, k + 1]``. So ``x_k > G_t`` needs ``k + 1 > G_t``, i.e. ``k >= floor(G_t)`` when
    ``G_t`` is not an integer and ``k >= G_t`` when it is: at most ``S - floor(G_t)`` slots
    of a row are deeper than ``t``, whatever the rounding of ``x_k``. Pad slots then get depth
    1 (``_pad_slot_depths``), which only lowers the count, and ``t = 0`` gives ``S``. So the
    number of ACTIVE slots of a row at pass ``t`` (pads are active at ``t = 0`` only) is at
    most ``C_t = S - floor(G_t)``: 64, 63, 61, 55, 46, 36, 26, 17 at ``S = 64``, mean 6,
    max 8.

``model.slot_compact`` (the execution, at a fixed joint law)
    ``"full"``: every pass still runs every cell, but a frozen cell serves the attention of
    each core layer from a per-layer CACHE of that layer's attention input (``norm(x_bar)``),
    written at the cell's last ACTIVE pass. Its own output is discarded, as before. This is
    the reference form of the new function: the slot loop before this key recomputed a
    frozen cell's keys and values from its frozen carrier and the CURRENT state of the slots
    before it; the cache holds the ones its last live pass computed.
    ``"gather"``: the same function, computed on the ``C_t * M`` rows of the pass only. The
    active cells are moved to the front of the row (``compact_order``), the core step runs on
    the first ``C_t * M`` rows (row-wise ops: injection, HC, norms, MLP; attention queries),
    the attention's keys and values are computed over every cell from the cache with this
    pass's rows written in, and the strict attention kernel reads each query's ORIGINAL
    position (``tg_strict_attention``'s ``qpos``), so the cell relation and the window are
    those of the full row. The rows past the active ones are inactive cells (frozen, finished
    or pad): real, finite states whose outputs are discarded and never written to a cache.

Rows move with ``_RowTake`` (a gather of the pass's ``n`` positions) and ``_RowPut`` (a
scatter of ``n`` rows into the carrier). The ``n`` positions are distinct, so the backward of
each is the mirror move: one write per row, no atomics, no accumulation, deterministic. Only
the ``n`` rows of a pass are read or written, never the whole row.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

SLOT_COMPACT_MODES = ("off", "full", "gather")


def clamped_poisson_cdf(mean: float, max_d: int) -> list[float]:
    """``[P(D <= t) for t = 0 .. max_d]``, ``D = clamp(Poisson(mean), 1, max_d)``, float64."""
    out, acc, term = [], 0.0, math.exp(-float(mean))
    for t in range(max_d + 1):
        if t > 0:
            term *= float(mean) / t
        acc += term
        out.append(0.0 if t == 0 else (1.0 if t >= max_d else acc))
    return out


def stratified_thresholds(mean: float, max_d: int, n_slots: int) -> list[float]:
    """``G_t = n_slots * P(D <= t)`` for ``t = 1 .. max_d - 1``: what the draw compares
    ``k + U`` against, and what :func:`slot_depth_capacity` bounds from (one home)."""
    cdf = clamped_poisson_cdf(mean, max_d)
    return [n_slots * cdf[t] for t in range(1, max_d)]


def slot_depth_capacity(mean: float, max_d: int, n_slots: int) -> tuple[int, ...]:
    """``C_t``, ``t = 0 .. max_d - 1``: the most slots of a row a stratified draw leaves
    active at pass ``t`` (module docstring for the proof)."""
    g = [0.0] + stratified_thresholds(mean, max_d, n_slots)
    return tuple(n_slots - int(math.floor(x)) for x in g)


_THRESHOLDS: dict[tuple, Tensor] = {}


def stratified_slot_depths(n_rows: int, n_slots: int, mean: float, max_d: int,
                           device) -> Tensor:
    """``[n_rows, n_slots]`` int64 depths in ``[1, max_d]``: per row a systematic sample of
    the clamped Poisson law, randomly permuted over the slots (module docstring). Draws two
    float64 tensors from the global device stream (``U`` and the permutation keys). The
    thresholds are copied to the device once per shape (``_THRESHOLDS``): a host copy inside
    a captured training step is refused by CUDA graph capture, and the first training step
    of a graph-step run is eager."""
    key = (int(n_slots), float(mean), int(max_d), str(torch.device(device)))
    g = _THRESHOLDS.get(key)
    if g is None:
        g = _THRESHOLDS[key] = torch.tensor(stratified_thresholds(mean, max_d, n_slots),
                                            dtype=torch.float64, device=device)
    u = torch.rand(n_rows, 1, dtype=torch.float64, device=device)
    x = torch.arange(n_slots, dtype=torch.float64, device=device) + u          # [R, S] sorted
    d = 1 + (x.unsqueeze(-1) > g).sum(-1)                                      # [R, S]
    perm = torch.rand(n_rows, n_slots, dtype=torch.float64, device=device).argsort(dim=1)
    return d.gather(1, perm)


def compact_order(active: Tensor) -> Tensor:
    """``perm`` for ``active`` ``[B, N]`` bool: ``perm[b, r]`` is the row position at rank
    ``r`` (active rows first, then the rest, each in position order). int64, deterministic
    (a stable sort)."""
    return torch.argsort((~active).to(torch.int8), dim=1, stable=True)


def _index_view(idx: Tensor, like: Tensor, dim: int, bdim: int) -> Tensor:
    """``idx`` ``[B, n]`` broadcast to ``like``'s shape with ``dim`` cut to ``n``, B at
    ``bdim`` and n at ``dim``."""
    shape = [1] * like.dim()
    shape[bdim], shape[dim] = idx.shape[0], idx.shape[1]
    out = list(like.shape)
    out[dim] = idx.shape[1]
    return idx.view(shape).expand(out)


class _RowTake(torch.autograd.Function):
    """``out = x.gather(dim, idx)`` with ``idx`` ``[B, n]`` distinct positions per row (the first
    ``n`` ranks of a ``compact_order``). The backward writes each gradient row back at its
    position into zeros: ``scatter_`` of distinct indices, one write per row, no atomics, no
    accumulation, deterministic and graph-capturable. Only the ``n`` rows are touched on the
    way forward."""

    @staticmethod
    def forward(ctx, x: Tensor, idx: Tensor, dim: int, bdim: int) -> Tensor:
        ctx.save_for_backward(idx)
        ctx.meta = (dim, bdim, x.shape)
        return x.gather(dim, _index_view(idx, x, dim, bdim))

    @staticmethod
    def backward(ctx, g: Tensor):
        (idx,) = ctx.saved_tensors
        dim, bdim, shape = ctx.meta
        gx = g.new_zeros(shape).scatter_(dim, _index_view(idx, g, dim, bdim), g)
        return gx, None, None, None


class _RowPut(torch.autograd.Function):
    """``out = fill`` with the rows at ``idx`` (``[B, n]`` distinct positions) replaced by
    ``src``. The backward is the mirror of ``_RowTake``: ``src`` gets the gradient rows at
    ``idx``, ``fill`` the gradient with those rows zeroed. Distinct indices, so exact and
    deterministic."""

    @staticmethod
    def forward(ctx, src: Tensor, fill: Tensor, idx: Tensor, dim: int, bdim: int) -> Tensor:
        ctx.save_for_backward(idx)
        ctx.meta = (dim, bdim)
        return fill.scatter(dim, _index_view(idx, src, dim, bdim), src)

    @staticmethod
    def backward(ctx, g: Tensor):
        (idx,) = ctx.saved_tensors
        dim, bdim = ctx.meta
        iv = _index_view(idx, g, dim, bdim)
        return g.gather(dim, iv), g.scatter(dim, iv, 0.0), None, None, None


def take_rows(x: Tensor, perm: Tensor, n: int, dim: int = 1, bdim: int = 0) -> Tensor:
    """The first ``n`` rows of ``x`` in ``perm`` order (``compact_order``)."""
    return _RowTake.apply(x, perm[:, :n], dim, bdim)


def put_rows(x_c: Tensor, perm: Tensor, fill: Tensor, keep: Tensor, dim: int = 1,
             bdim: int = 0, fill_c: Tensor | None = None) -> Tensor:
    """``x_c`` (rows in ``perm`` order, the first ``n``) written back at their positions
    where ``keep`` ``[B, N]`` is True, ``fill`` everywhere else. ``keep`` must lie inside the
    first ``n`` ranks (the caller's capacity bound): a kept row past them keeps ``fill``.
    ``fill_c`` is ``take_rows(fill, perm, n)`` when the caller already holds it."""
    n = x_c.shape[dim]
    idx = perm[:, :n]
    if fill_c is None:
        fill_c = _RowTake.apply(fill, idx, dim, bdim)
    kshape = [1] * x_c.dim()
    kshape[bdim], kshape[dim] = keep.shape[0], n
    keep_c = keep.gather(1, idx).view(kshape)
    src = torch.where(keep_c, x_c.to(fill.dtype), fill_c)
    return _RowPut.apply(src, fill, idx, dim, bdim)


def compact_attn_input(xa: Tensor, payload: tuple) -> Tensor:
    """One core layer's attention input over EVERY cell (``model.slot_compact``).

    ``xa`` is the layer's ``norm(x_bar)`` on the rows this pass computes (all ``N`` cells in
    ``"full"`` mode, the first ``n`` ranks in ``"gather"`` mode); ``payload`` is
    ``(cache, keep, perm)``: the layer's cache ``[B, N, C]`` (None at the first pass),
    the cells whose pass this is (``active``), and the row order (None in ``"full"`` mode).
    The result is ``xa`` at the active cells and the cache elsewhere; it is also the cache
    the next pass reads, so a frozen cell keeps the input of its last active pass.
    """
    cache, keep, perm = payload
    if cache is None:
        # pass 0: every cell is active (valid slots draw depth >= 1, pads loop once), so
        # the zeros are never read; they only give `where` its shape
        cache = xa.new_zeros(keep.shape[0], keep.shape[1], xa.shape[-1])
    if perm is None:
        return torch.where(keep.unsqueeze(-1), xa.to(cache.dtype), cache)
    return put_rows(xa, perm, cache, keep, dim=1)


def validate_slot_compact(cfg) -> None:
    """Refuse every configuration the two keys were not built or tested for (``cfg`` is a
    ``MORPHConfig``). Called once at model build."""
    strat = bool(cfg.slot_depth_stratified)
    mode = str(cfg.slot_compact)
    if mode not in SLOT_COMPACT_MODES:
        raise ValueError(f"model.slot_compact must be one of {SLOT_COMPACT_MODES}, got {mode!r}")
    if not strat and mode == "off":
        return
    tc = cfg.tul
    if (tc is None or cfg.n_core == 0 or tc.tokens_through_core
            or getattr(tc, "loop_reads_tokens", False) or getattr(cfg, "fm", None) is not None):
        raise ValueError("model.slot_depth_stratified / model.slot_compact act on the slot loop "
                         "(_tul_core) only; this model has none.")
    if strat and int(tc.slot_depth_fixed) > 0:
        raise ValueError("model.slot_depth_stratified with tul.slot_depth_fixed > 0: a fixed "
                         "depth has nothing to stratify.")
    if mode == "off":
        return
    _need = {
        "model.slot_depth_stratified=true (the capacity bound C_t)": strat,
        "tul.slot_cells > 1 (the core's cell index, MODE_CELLS)": int(tc.slot_cells) > 1,
        "model.tg_fused_attention=true (the kernel reads query positions)":
            bool(cfg.tg_fused_attention),
        "model.graph_safe=true with its 'passes' part (a fixed pass count)":
            bool(cfg.graph_safe) and "passes" in cfg.graph_safe_parts,
        "model.ckpt_grad_iters=0 (a checkpointed pass would recompute the cache)":
            int(cfg.ckpt_grad_iters) == 0,
        "model.slot_gain_no_ckpt=true or no hinge (the hinge's applications not checkpointed)":
            float(cfg.slot_gain_lambda) <= 0.0 or bool(cfg.slot_gain_no_ckpt),
        "model.slot_gain_reuse_f0=false": not bool(cfg.slot_gain_reuse_f0),
        "model.scse_enabled=false": not bool(cfg.scse_enabled),
        "model.core_impl='morph'": str(cfg.core_impl) == "morph",
        "model.core_depth_state=false": not bool(cfg.core_depth_state),
        "no retention branch in the core":
            not (bool(cfg.retention) and "core" in tuple(cfg.retention_sections)),
    }
    _off = {"loop_reach": 0, "loop_carry": "none", "reread": False, "pass_lora_rank": 0,
            "xhc_streams": 0, "code_enum_k": 1, "code_policy_k": 0, "slot_chain": False,
            "grad_pass": False, "loop_denoise": False, "gram": False,
            "core_stage_cond": "none", "loop_attn_center": "off", "progressive_p": 0.0,
            "slot_source_once": False}
    for k, v in _off.items():
        _need[f"tul.{k}={v!r}"] = getattr(tc, k) == v
    _miss = [k for k, ok in _need.items() if not ok]
    if _miss:
        raise NotImplementedError(
            f"model.slot_compact={mode!r} is built and tested for the LXTUL graph-step recipe "
            f"only; it needs: {', '.join(_miss)}.")
