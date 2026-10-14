"""``model.block_custom_ops``: the block's Triton autograd Functions as ``torch.library`` ops.

WHY. The HC region kernels, the CCA conv and the row-tiled CCA prologue are
``torch.autograd.Function``s behind ``kernel_fence`` (``torch.compiler.disable``). Under
``torch.compile`` every fence cuts the block into a compiled piece per gap, so inductor never
sees the casts, SiLU, dropout and norms that sit between two kernels as one graph. A
``torch.library.custom_op`` with ``register_autograd`` and ``register_fake`` is opaque to
inductor (it never traces the Triton body) and is NOT a graph break, so a block compiles as ONE
graph with the kernels as opaque nodes in it. The strict attention is already such an op
(``tg_strict_attention.py``); ``split_wgrad.py`` is the other precedent.

SAME KERNELS. Every op body is the old Function's body, moved into a shared ``*_impl`` function
that both call (``fused_hyper_connection._hcr_*_impl``, ``fused_cca_conv._cca_conv_*_impl``,
``fused_seg_causal_conv._seg_conv_*_impl``, ``fused_cca_prologue._rows_*_impl``), so eager results are bit-identical to the Functions
(``tests/test_block_custom_ops.py``). What inductor fuses AROUND the ops may round differently.

THE HC LINK. ``_HCRegionEntry`` returns ``h.view_as(h)`` as ``h_link`` and ``_HCRegionExit``
hands its output grad to the entry backward as ``h_link``'s grad: the entry backward needs
gout (the Hres grad is ``sum_c gout . h``, and the carrier grad folds gout in one pass). A custom
op may not return an alias of its input, and a plain ``h`` passed to both nodes would sum the two
grads outside the kernel and lose the fold. So the entry op also returns ``link``, a fresh
UNINITIALISED tensor of ``h``'s shape (``torch.empty``: no kernel, no read, no write), and the
exit op takes it as an input it never reads. Its only job is the autograd edge: the exit backward
returns gout as ``link``'s grad, which is exactly the entry backward's ``g_link``. The wrappers
carry ``(h, link)`` as the ``h_link`` the model code passes from the entry to the exit. The cost
is the allocation (carrier-sized, live from the entry to the exit).

NOT COVERED. ``hc_pre_map`` / ``hc_post`` / the fold variants (the non-region HC path), the
non-tiled CCA prologue and the retention / window kernels keep their fences.
``HyperConnectionResidual`` and ``MORPHTransformer`` refuse the key where they cannot honour it.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .fused_cca_conv import _cca_conv_bwd_impl, _cca_conv_fwd_impl
from .fused_cca_prologue import _rows_bwd_impl, _rows_fwd_impl
from .fused_seg_causal_conv import _seg_conv_bwd_impl, _seg_conv_fwd_impl
from .fused_hyper_connection import (
    _hc_region_active, _hcr_entry_bwd_impl, _hcr_entry_fwd_impl, _hcr_exit_bwd_impl,
    _hcr_exit_fwd_impl, hc_region_entry, hc_region_exit)

__all__ = ["cca_conv_op", "cca_prologue_rows_op", "hc_region_entry_ops", "hc_region_exit_ops",
           "seg_conv_op"]


# ===========================================================================
# CCA conv
# ===========================================================================

@torch.library.custom_op("morph::cca_conv_fwd", mutates_args=())
def _cca_conv_fwd(x: Tensor, w_dw2: Tensor, w_gp: Tensor, CG: int, K: int, flat: bool = False
                  ) -> tuple[Tensor, Tensor]:
    y_gp, y_dw, _, _ = _cca_conv_fwd_impl(x, w_dw2, w_gp, CG, K)
    return y_gp, y_dw


@_cca_conv_fwd.register_fake
def _cca_conv_fwd_fake(x, w_dw2, w_gp, CG, K, flat=False):
    return torch.empty_like(x, memory_format=torch.contiguous_format), torch.empty_like(
        x, memory_format=torch.contiguous_format)


@torch.library.custom_op("morph::cca_conv_bwd", mutates_args=())
def _cca_conv_bwd(grad_out: Tensor, x: Tensor, y_dw: Tensor, w_dw2: Tensor, w_gp: Tensor,
                  CG: int, K: int, flat: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    return _cca_conv_bwd_impl(grad_out, x, y_dw, w_dw2, w_gp, CG, K, flat)


@_cca_conv_bwd.register_fake
def _cca_conv_bwd_fake(grad_out, x, y_dw, w_dw2, w_gp, CG, K, flat=False):
    return (torch.empty_like(x, memory_format=torch.contiguous_format),
            torch.empty_like(w_dw2, memory_format=torch.contiguous_format),
            torch.empty_like(w_gp, memory_format=torch.contiguous_format))


def _cca_conv_setup(ctx, inputs, output):
    x, w_dw2, w_gp, CG, K, flat = inputs
    ctx.save_for_backward(x, output[1], w_dw2, w_gp)
    ctx.meta = (CG, K, flat)
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[1])          # y_dw: the backward's saved intermediate


def _cca_conv_backward(ctx, g_y, g_ydw):
    x, y_dw, w_dw2, w_gp = ctx.saved_tensors
    dx, dw_dw, dw_gp = _cca_conv_bwd(g_y, x, y_dw, w_dw2, w_gp, *ctx.meta)
    return dx, dw_dw, dw_gp, None, None, None


_cca_conv_fwd.register_autograd(_cca_conv_backward, setup_context=_cca_conv_setup)


def cca_conv_op(x_BCS: Tensor, w_dw: Tensor, w_gp: Tensor, CG: int, K: int,
                flat: bool = False) -> Tensor:
    """``_FusedCCAConv.apply`` as an op. The operands are made contiguous HERE, outside the op,
    so inductor fuses the copy with the producer (the ``[B, S, C]`` -> ``[B, C, S]`` transpose)
    and the op saves the very tensor it read."""
    x = x_BCS.contiguous()
    w_dw2 = w_dw.reshape(w_dw.shape[0], K).contiguous()
    return _cca_conv_fwd(x, w_dw2, w_gp.contiguous(), CG, K, flat)[0]


# ===========================================================================
# Segment-reset CCA conv (the strict geometry's prelude and coda)
# ===========================================================================

@torch.library.custom_op("morph::seg_conv_fwd", mutates_args=())
def _seg_conv_fwd(x: Tensor, w_dw2: Tensor, w_gp: Tensor, seg: Tensor, CG: int, K: int
                  ) -> tuple[Tensor, Tensor]:
    y_gp, y_dw, _, _, _ = _seg_conv_fwd_impl(x, w_dw2, w_gp, seg, CG, K)
    return y_gp, y_dw


@_seg_conv_fwd.register_fake
def _seg_conv_fwd_fake(x, w_dw2, w_gp, seg, CG, K):
    return (torch.empty_like(x, memory_format=torch.contiguous_format),
            torch.empty_like(x, memory_format=torch.contiguous_format))


@torch.library.custom_op("morph::seg_conv_bwd", mutates_args=())
def _seg_conv_bwd(grad_out: Tensor, x: Tensor, y_dw: Tensor, w_dw2: Tensor, w_gp: Tensor,
                  seg: Tensor, CG: int, K: int, need_x: bool, need_dw: bool, need_gp: bool
                  ) -> tuple[Tensor, Tensor, Tensor]:
    grads = _seg_conv_bwd_impl(grad_out, x, y_dw, w_dw2, w_gp, seg, CG, K, need_x, need_dw,
                               need_gp)
    return tuple(g if g is not None else x.new_empty(0) for g in grads)    # None -> empty


@_seg_conv_bwd.register_fake
def _seg_conv_bwd_fake(grad_out, x, y_dw, w_dw2, w_gp, seg, CG, K, need_x, need_dw, need_gp):
    return (torch.empty_like(x, memory_format=torch.contiguous_format) if need_x
            else x.new_empty(0),
            torch.empty_like(w_dw2, memory_format=torch.contiguous_format) if need_dw
            else x.new_empty(0),
            torch.empty_like(w_gp, memory_format=torch.contiguous_format) if need_gp
            else x.new_empty(0))


def _seg_conv_setup(ctx, inputs, output):
    x, w_dw2, w_gp, seg, CG, K = inputs
    ctx.save_for_backward(x, output[1], w_dw2, w_gp, seg)
    ctx.meta = (CG, K)
    ctx.need = tuple(ctx.needs_input_grad[:3])
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[1])


def _seg_conv_backward(ctx, g_y, g_ydw):
    x, y_dw, w_dw2, w_gp, seg = ctx.saved_tensors
    need_x, need_dw, need_gp = ctx.need
    dx, dw_dw, dw_gp = _seg_conv_bwd(g_y, x, y_dw, w_dw2, w_gp, seg, *ctx.meta, need_x, need_dw,
                                     need_gp)
    return (dx if need_x else None, dw_dw if need_dw else None, dw_gp if need_gp else None,
            None, None, None)


_seg_conv_fwd.register_autograd(_seg_conv_backward, setup_context=_seg_conv_setup)


def seg_conv_op(x_BCS: Tensor, w_dw: Tensor, w_gp: Tensor, seg: Tensor) -> Tensor:
    """``_FusedSegCausalConv.apply`` as an op (the arguments of ``fused_segment_causal_conv``;
    the shape checks stay in that function's caller, ``segment_causal_conv``)."""
    x = x_BCS.contiguous()
    w_dw2 = w_dw.reshape(w_dw.shape[0], w_dw.shape[-1]).contiguous()
    return _seg_conv_fwd(x, w_dw2, w_gp.contiguous(), seg, w_gp.shape[1], w_dw.shape[-1])[0]


# ===========================================================================
# CCA prologue (row-tiled)
# ===========================================================================

@torch.library.custom_op("morph::cca_prologue_rows_fwd", mutates_args=())
def _rows_fwd(q_lat: Tensor, k_lat: Tensor, q_conv: Tensor, k_conv: Tensor, v_curr: Tensor,
              v_prev: Tensor, wq: Tensor, wk: Tensor, temp: Tensor, cos: Tensor, sin: Tensor,
              H: int, Hkv: int, D: int, n_rep: int, n_skip_rope: int, eps: float,
              flat: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    return _rows_fwd_impl(q_lat, k_lat, q_conv, k_conv, v_curr, v_prev, wq, wk, temp, cos, sin,
                          H, Hkv, D, n_rep, n_skip_rope, eps)


@_rows_fwd.register_fake
def _rows_fwd_fake(q_lat, k_lat, q_conv, k_conv, v_curr, v_prev, wq, wk, temp, cos, sin,
                   H, Hkv, D, n_rep, n_skip_rope, eps, flat=False):
    B, S, _ = q_lat.shape
    return tuple(q_lat.new_empty(B, H, S, D) for _ in range(3))


@torch.library.custom_op("morph::cca_prologue_rows_bwd", mutates_args=())
def _rows_bwd(grad_q: Tensor, grad_k: Tensor, grad_v: Tensor, q_lat: Tensor, k_lat: Tensor,
              q_conv: Tensor, k_conv: Tensor, wq: Tensor, wk: Tensor, temp: Tensor, cos: Tensor,
              sin: Tensor, H: int, Hkv: int, D: int, n_rep: int, n_skip_rope: int, eps: float,
              vhalf: int, flat: bool = False
              ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _rows_bwd_impl(grad_q, grad_k, grad_v, q_lat, k_lat, q_conv, k_conv, wq, wk, temp,
                          cos, sin, H, Hkv, D, n_rep, n_skip_rope, eps, vhalf, flat)


@_rows_bwd.register_fake
def _rows_bwd_fake(grad_q, grad_k, grad_v, q_lat, k_lat, q_conv, k_conv, wq, wk, temp, cos, sin,
                   H, Hkv, D, n_rep, n_skip_rope, eps, vhalf, flat=False):
    B, S, _ = q_lat.shape
    # the same layouts as the kernel wrapper: each grad takes its input's (empty_like).
    return (torch.empty_like(q_lat), torch.empty_like(k_lat), torch.empty_like(q_conv),
            torch.empty_like(k_conv), q_lat.new_empty(B, S, vhalf), q_lat.new_empty(B, S, vhalf),
            torch.empty_like(wq), torch.empty_like(wk), torch.empty_like(temp))


def _rows_setup(ctx, inputs, output):
    (q_lat, k_lat, q_conv, k_conv, v_curr, v_prev, wq, wk, temp, cos, sin, H, Hkv, D, n_rep,
     n_skip_rope, eps, flat) = inputs
    ctx.save_for_backward(q_lat, k_lat, q_conv, k_conv, wq, wk, temp, cos, sin)
    ctx.dims = (H, Hkv, D, n_rep, n_skip_rope, eps, v_curr.shape[-1], flat)


def _rows_backward(ctx, grad_q, grad_k, grad_v):
    q_lat, k_lat, q_conv, k_conv, wq, wk, temp, cos, sin = ctx.saved_tensors
    grads = _rows_bwd(grad_q, grad_k, grad_v, q_lat, k_lat, q_conv, k_conv, wq, wk, temp, cos,
                      sin, *ctx.dims)
    return (*grads, None, None, None, None, None, None, None, None, None)


_rows_fwd.register_autograd(_rows_backward, setup_context=_rows_setup)


def cca_prologue_rows_op(q_lat, k_lat, q_conv, k_conv, v_curr, v_prev, wq, wk, temp, cos, sin,
                         H, Hkv, D, n_rep, n_skip_rope, eps, flat=False):
    """``_FusedCCAPrologueRows.apply`` as an op (the arguments of ``fused_cca_prologue``'s
    dispatchers: ``cos`` / ``sin`` already ``[S, D]``). ``flat``: model.kv_flat_kernels."""
    return _rows_fwd(q_lat, k_lat, q_conv, k_conv, v_curr, v_prev, wq, wk, temp, cos, sin,
                     H, Hkv, D, n_rep, n_skip_rope, eps, flat)


# ===========================================================================
# HC region entry / exit
# ===========================================================================

@torch.library.custom_op("morph::hc_region_entry_fwd", mutates_args=())
def _entry_fwd(h: Tensor, proj_w: Tensor, proj_b: Tensor, tau: float, alpha: float, eps: float,
               bwd_fused: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    xbar, hres, hpr, raw, rms, h, _, _ = _hcr_entry_fwd_impl(h, proj_w, proj_b, tau, alpha, eps)
    return xbar, hres, hpr, raw, rms, torch.empty_like(h)         # the last: the link (module doc)


@_entry_fwd.register_fake
def _entry_fwd_fake(h, proj_w, proj_b, tau, alpha, eps, bwd_fused=False):
    B, S, N, C = h.shape
    f32 = torch.float32
    return (h.new_empty(B, S, C), h.new_empty(B, S, N, N, dtype=f32),
            h.new_empty(B, S, N, dtype=f32), h.new_empty(B, S, 48, dtype=f32),
            h.new_empty(B, S, 1, dtype=f32), torch.empty_like(h))


@torch.library.custom_op("morph::hc_region_entry_bwd", mutates_args=())
def _entry_bwd(g_xbar: Tensor | None, g_hpr: Tensor | None, g_link: Tensor | None, h: Tensor,
               proj_w: Tensor, proj_b: Tensor, raw: Tensor, rms: Tensor, tau: float,
               alpha: float, bwd_fused: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    return _hcr_entry_bwd_impl(g_xbar, g_hpr, g_link, h, proj_w.contiguous(),
                               proj_b.float().contiguous(), raw, rms, tau, alpha, proj_w.dtype,
                               proj_b.dtype, bwd_fused)


@_entry_bwd.register_fake
def _entry_bwd_fake(g_xbar, g_hpr, g_link, h, proj_w, proj_b, raw, rms, tau, alpha,
                    bwd_fused=False):
    return torch.empty_like(h), torch.empty_like(proj_w), torch.empty_like(proj_b)


def _entry_setup(ctx, inputs, output):
    h, proj_w, proj_b, tau, alpha, eps, bwd_fused = inputs
    _, _, _, raw, rms, _ = output
    ctx.save_for_backward(h, proj_w, proj_b, raw, rms)
    ctx.cfg = (float(tau), float(alpha), bool(bwd_fused))
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(raw, rms)


def _entry_backward(ctx, g_xbar, g_hres, g_hpr, g_raw, g_rms, g_link):
    # g_hres is None (the exit forms no grad for it; the backward kernel computes it).
    h, proj_w, proj_b, raw, rms = ctx.saved_tensors
    gh, grad_w, grad_b = _entry_bwd(g_xbar, g_hpr, g_link, h, proj_w, proj_b, raw, rms, *ctx.cfg)
    return gh, grad_w, grad_b, None, None, None, None


_entry_fwd.register_autograd(_entry_backward, setup_context=_entry_setup)


@torch.library.custom_op("morph::hc_region_exit_fwd", mutates_args=())
def _exit_fwd(hres: Tensor, hpost_row: Tensor, h: Tensor, link: Tensor, y: Tensor,
              term: Tensor | None) -> Tensor:
    # `link` is read by nobody: it is the autograd edge to the entry (module doc).
    return _hcr_exit_fwd_impl(hres, hpost_row, h, y, term)[0]


@_exit_fwd.register_fake
def _exit_fwd_fake(hres, hpost_row, h, link, y, term):
    return torch.empty_like(h, memory_format=torch.contiguous_format)


@torch.library.custom_op("morph::hc_region_exit_bwd", mutates_args=())
def _exit_bwd(grad_out: Tensor, hpost_row: Tensor, y: Tensor, term_dtype: torch.dtype | None
              ) -> tuple[Tensor, Tensor, Tensor]:
    gy, ghpost, gterm = _hcr_exit_bwd_impl(grad_out, hpost_row, y, term_dtype)
    if term_dtype is None:
        gterm = gy.new_empty(0)                 # an op may not return the same tensor twice
    return gy, ghpost, gterm


@_exit_bwd.register_fake
def _exit_bwd_fake(grad_out, hpost_row, y, term_dtype):
    B, S, N, C = grad_out.shape
    gterm = (y.new_empty(0) if term_dtype is None
             else grad_out.new_empty(B, S, C, dtype=term_dtype))
    return torch.empty_like(y), grad_out.new_empty(B, S, N, dtype=torch.float32), gterm


def _exit_setup(ctx, inputs, output):
    hres, hpost_row, h, link, y, term = inputs
    ctx.save_for_backward(hpost_row, y)
    ctx.term_dtype = None if term is None else term.dtype


def _exit_backward(ctx, g_out):
    hpost_row, y = ctx.saved_tensors
    g_out = g_out.contiguous()
    gy, ghpost, gterm = _exit_bwd(g_out, hpost_row, y, ctx.term_dtype)
    # (hres, hpost_row, h, link, y, term): the carrier grad goes to the entry through `link`.
    return (None, ghpost.to(hpost_row.dtype), None, g_out, gy,
            None if ctx.term_dtype is None else gterm)


_exit_fwd.register_autograd(_exit_backward, setup_context=_exit_setup)


def hc_region_entry_ops(h: Tensor, proj_w: Tensor, proj_b: Tensor, tau: float, alpha: float,
                        iters: int, eps: float, carrier_bf16: bool = False,
                        bwd_fused: bool = False):
    """``hc_region_entry`` on the custom ops: ``(x_bar, Hres, Hpost_row, h_link)``. Where the
    region kernels do not apply (``_hc_region_active``) it IS ``hc_region_entry``; otherwise
    ``h_link`` is the pair ``(h, link)`` that :func:`hc_region_exit_ops` takes."""
    if int(iters) != 3:
        raise ValueError("model.hc_region_fused needs hc_cayley_iters=3")
    if not _hc_region_active(h, carrier_bf16):
        return hc_region_entry(h, proj_w, proj_b, tau, alpha, iters, eps, carrier_bf16, bwd_fused)
    h = h.contiguous()
    x_bar, hres, hpr, _, _, link = _entry_fwd(h, proj_w, proj_b, float(tau), float(alpha),
                                              float(eps), bool(bwd_fused))
    return x_bar, hres, hpr, (h, link)


def hc_region_exit_ops(hres: Tensor, hpost_row: Tensor, h_link, y: Tensor,
                       term: Tensor | None = None, carrier_bf16: bool = False) -> Tensor:
    """``hc_region_exit`` on the custom ops; ``h_link`` is what :func:`hc_region_entry_ops`
    returned (a bare tensor means the entry took the fallback, and so does the exit)."""
    if not isinstance(h_link, tuple):
        return hc_region_exit(hres, hpost_row, h_link, y, term, carrier_bf16)
    h, link = h_link
    return _exit_fwd(hres.contiguous(), hpost_row.contiguous(), h, link, y.contiguous(),
                     None if term is None else term.contiguous())
