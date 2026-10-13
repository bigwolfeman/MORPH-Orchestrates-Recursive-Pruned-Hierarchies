"""``model.split_wgrad``: weight gradients of the block linears on a side CUDA stream.

THE IDEA. A linear's backward is two GEMMs on the same ``dy``: the data grad ``dx = dy @ W``
(the next backward op waits for it) and the weight grad ``dW = dy^T @ x`` (nothing in the
backward waits for it; only the optimizer reads it). Autograd runs both on one stream. Here
the weight GEMM, its cast to the parameter's dtype and its sum over the weight's reads run on
a second stream, in parallel with the rest of the backward.

WHAT IT COVERS (``bind_split_wgrad``): every linear whose weight a block reads through
``F.linear`` under bf16 autocast, and whose weight no other code reads under autograd:

  * the ternary MLP linears (``CMSBlockLinear`` whose ``weight`` is a ``TernarySTE`` bound to
    the step cache): the GEMM weight is the cached bf16 ``_wq``;
  * the attention's fused input projection (``attention._fused_x_proj``: the ``_fuse_mods``
    concatenated into one GEMM) and ``W_up`` (plain fp32 parameters cast to bf16 per read).

Prelude, core, coda and the gain hinge all run these modules, so all four are covered.

EXACTNESS. Today each READ of a weight yields one bf16 weight grad ``g_r``; it is cast to the
parameter's dtype and the reads are summed by the autograd engine in the order their backward
nodes run (``((g_1 + g_2) + g_3) ...``), then added to ``p.grad``. Here ``_SplitLinear.backward``
is called in the same order (it is the same read's node), enqueues the same bf16 GEMM
(``dy^T @ x``), the same cast and the same add on ONE stream, so the sum is the same bit for bit.
``finish`` then does what ``AccumulateGrad`` does: ``p.grad = sum`` if there is none, else
``p.grad += sum``. A fused group's gradient is one ``[N, K]`` accumulator and each parameter's
``.grad`` is a row slice of it (the cat's backward is exactly that slice, per element).
A parameter that no read touched keeps ``p.grad = None``.

THE OP IS OPT-IN BY SCOPE (``train_backward_scope``): only the training step's forward arms it;
every other read (probes, eval, ``create_graph``) is the plain path.

EVERY READ OF A COVERED WEIGHT INSIDE THE SCOPE GOES THROUGH THIS OP OR NONE. The covered parameters are not in
the autograd graph at all (the op takes a detached weight), so a read outside the op under
autograd would add its own gradient to ``p.grad`` first and change the summation order.
``bind_split_wgrad`` refuses what it can check at build; the forward paths that read the same
weights without the op run only under ``no_grad`` or without bf16 autocast (``split_active``
says so), where no gradient exists. ``WgradStream.strict`` (tests) raises if a covered
``p.grad`` is already set when a backward joins.

LIFETIMES. The side stream reads ``dy`` and ``x``, which other streams allocated. The op keeps a
reference to both until the streams that allocated them have waited for the side stream; a
reference held is a block the allocator cannot reuse, so no ``record_stream`` is needed (inside
a CUDA graph capture that call would defer every free to the capture's end). The references are
released in groups of ``group_reads`` reads, two groups behind the newest, and the rest at the
join (``WgradStream``).

JOIN. ``_SplitLinear.backward`` queues ``finish`` as an autograd final callback, which the
engine runs after the last node of the backward, on the caller's current stream (the capture
stream inside ``graph_step``), so the join sits inside both hinge-branch bodies of a recording.

The op is a ``torch.library.custom_op`` pair (``morph::split_linear`` and its backward
``morph::split_linear_bwd``, which carries the side-stream work and its Python state). It is
opaque to ``torch.compile``: no graph break, the block stays ONE compiled graph (a graph break
changed which kernels inductor fused and how many dropout seeds the compiled graphs draw, so a
step was no longer the same step; measured 2026-10-12).
"""
from __future__ import annotations

import contextlib
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from morph.kernels.triton.split_gemm import (linear_dgrad, linear_fwd, linear_wgrad, swiglu_bwd,
                                             swiglu_fwd)

from .streams import side_stream

__all__ = ["SplitWeight", "WgradStream", "as_bf16", "bind_split_wgrad", "split_active",
           "train_backward_scope",
           "split_linear", "split_swiglu_mlp"]


def _autocast_bf16() -> bool:
    return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16


class WgradStream:
    """The side stream, the references it keeps alive and the join (module notes).

    References are released in GROUPS of ``group_reads`` reads, two groups behind the newest:
    when a group closes, an event marks the side stream's progress, and the group that closed
    two before it is dropped after the streams that allocated its tensors wait for that event
    (the side stream is ahead by then, so the wait costs nothing). The held bytes stay at
    about three groups' worth instead of the whole backward's."""

    def __init__(self, stream: torch.cuda.Stream, strict: bool = False, group_reads: int = 8,
                 triton: bool = False):
        if group_reads < 1:
            raise ValueError(f"split_wgrad: group_reads must be >= 1, got {group_reads}")
        self.stream = stream
        self.strict = strict                  # tests: raise on a covered p.grad set before a join
        self.group_reads = group_reads
        self.triton = triton                  # model.split_wgrad_triton: Triton GEMMs, no scratch
        # the open group: (dy, x) pairs and the streams that produced them
        self._held: list[tuple[Tensor, Tensor]] = []
        self._streams: dict[int, torch.cuda.Stream] = {}
        # closed groups, oldest first: (side-stream event after the group's GEMMs, refs, streams)
        self._closed: list[tuple[torch.cuda.Event, list, dict]] = []
        self._scratch: dict = {}
        self._touched: dict[int, "SplitWeight"] = {}      # id -> weight, in first-read order
        self.deferred = 0                      # weight GEMMs enqueued, ever (tests)
        self.joins = 0                         # finish calls that had work, ever (tests)
        self.peak_held = 0                     # most (dy, x) pairs alive at once (tests)

    def scratch(self, n: int, k: int, device) -> Tensor:
        """The bf16 ``[n, k]`` GEMM scratch for that shape (made on first ask, at bind)."""
        key = (n, k, torch.device(device))
        if key not in self._scratch:
            self._scratch[key] = torch.empty(n, k, dtype=torch.bfloat16, device=device)
        return self._scratch[key]

    def defer(self, sw: "SplitWeight", dy: Tensor, x: Tensor, ready: torch.cuda.Event) -> None:
        """Enqueue one read's weight grad on the side stream. ``ready`` was recorded on the
        node's stream before the data-grad GEMM, so ``dy`` is complete when the side stream
        passes it and the data-grad GEMM is not waited for."""
        cur = torch.cuda.current_stream()
        self.stream.wait_event(ready)
        with torch.cuda.stream(self.stream):
            if self.triton:
                # one kernel: the GEMM, the bf16 rounding and the copy / fp32 add into the sum
                linear_wgrad(dy.reshape(-1, sw.n_out), x.reshape(-1, sw.n_in), sw.buf, sw.started)
            else:
                # No allocation here: the GEMM lands in the shared scratch, and the fp32 sum in
                # the weight's own persistent buffer (cast on the fly by copy_ / add_, exact).
                torch.mm(dy.reshape(-1, sw.n_out).t(), x.reshape(-1, sw.n_in), out=sw.scratch)
                if sw.started:
                    sw.buf.add_(sw.scratch)
                else:
                    sw.buf.copy_(sw.scratch)
            sw.started = True
        self._held.append((dy, x))
        self._touched.setdefault(id(sw), sw)
        self._streams[cur.cuda_stream] = cur
        self.deferred += 1
        self.peak_held = max(self.peak_held,
                             len(self._held) + sum(len(c[1]) for c in self._closed))
        if len(self._held) >= self.group_reads:
            self._close_group()
        # One callback per read, not per backward: a backward that raised leaves state behind,
        # and the next backward must still join it. ``finish`` is idempotent.
        torch.autograd.Variable._execution_engine.queue_callback(self.finish)

    def _close_group(self) -> None:
        mark = torch.cuda.Event()
        mark.record(self.stream)
        self._closed.append((mark, self._held, self._streams))
        self._held, self._streams = [], {}
        if len(self._closed) > 2:
            ev, refs, streams = self._closed.pop(0)
            for s in streams.values():         # the streams that allocated these tensors
                s.wait_event(ev)
            del refs                            # now the allocator may reuse them

    def finish(self) -> None:
        """Join: the consumer streams wait for the side stream, the gradients are bound, the
        references are dropped. Runs as the engine's final callback."""
        if not self._touched:
            return
        done = torch.cuda.Event()
        done.record(self.stream)
        cur = torch.cuda.current_stream()
        cur.wait_event(done)
        streams = dict(self._streams)
        for _ev, _refs, st in self._closed:
            streams.update(st)
        for s in streams.values():             # a hinge's dys were allocated on its stream
            if s != cur:
                s.wait_event(done)
        for sw in self._touched.values():
            sw.bind_grads(self.strict)
        self._held.clear()
        self._closed.clear()
        self._touched.clear()
        self._streams.clear()
        self.joins += 1


_REGISTRY: dict[int, "SplitWeight"] = {}      # sid -> weight: the ops carry an int, not an object


class SplitWeight:
    """One covered GEMM weight: its shape, its parameters' row ranges in the ``[N, K]`` weight
    grad, how to get the bf16 weight, and the accumulator of the backward in flight.

    ``kind`` says where the bf16 weight comes from (written out, not a closure, so dynamo
    inlines it): ``cached`` the step cache's ``ste._wq``; ``fused`` the cat of ``mods``'
    weights cast to bf16; ``plain`` one module's weight cast to bf16; ``callable`` the
    ``source`` function (tests)."""

    def __init__(self, mgr: WgradStream, name: str, params: Sequence[nn.Parameter],
                 rows: Sequence[tuple[int, int]], n_in: int, *, kind: str,
                 ste=None, mods: tuple = (), source: Callable[[], Tensor] | None = None):
        dtypes = {p.dtype for p in params}
        if len(dtypes) != 1:
            raise ValueError(f"split_wgrad: {name}: parameters of one fused weight differ in "
                             f"dtype {sorted(map(str, dtypes))}")
        if kind not in ("cached", "fused", "plain", "callable"):
            raise ValueError(f"split_wgrad: {name}: unknown weight kind {kind!r}")
        self.mgr, self.name, self.params, self.rows = mgr, name, tuple(params), tuple(rows)
        self.kind, self.ste, self.mods, self.source = kind, ste, tuple(mods), source
        self.n_in = int(n_in)
        self.n_out = int(self.rows[-1][1])
        self.dtype = params[0].dtype
        if self.rows[0][0] != 0 or any(a[1] != b[0] for a, b in zip(self.rows, self.rows[1:])):
            raise ValueError(f"split_wgrad: {name}: row ranges {self.rows} are not contiguous "
                             f"from 0")
        self.sid = len(_REGISTRY)
        _REGISTRY[self.sid] = self
        # Allocated once, here, outside any capture: the fp32 sum of this weight's grad (its
        # parameters' ``.grad`` are views of it) and the bf16 GEMM scratch (one per shape,
        # shared: the side stream runs the reads one after the other).
        self.buf = torch.empty(self.n_out, self.n_in, dtype=self.dtype, device=params[0].device)
        self.scratch = None if mgr.triton else mgr.scratch(self.n_out, self.n_in, params[0].device)
        self.started = False                  # the first read of a backward copies, later ones add

    def weight(self) -> Tensor:
        """The bf16 ``[N, K]`` GEMM weight, with no autograd history."""
        if self.kind == "cached":
            return self.ste._wq
        if self.kind == "fused":
            return torch.cat([m.weight.detach() for m in self.mods], dim=0).to(torch.bfloat16)
        if self.kind == "plain":
            return self.mods[0].weight.detach().to(torch.bfloat16)
        return self.source()

    def __deepcopy__(self, memo: dict) -> None:
        """A copy of a module (the fan's EMA twin) is never bound, as for ``TernaryStepCache``:
        it reads its own weights through the plain path."""
        return None

    def bind_grads(self, strict: bool) -> None:
        """AccumulateGrad's rule on views of ``buf``: ``p.grad`` None -> set (the first backward
        keeps -0.0 as a stolen gradient does); ``p.grad`` already this view (a hinge body that
        zeroed it first) -> the buffer holds the sum, and ``+ 0.0`` gives ``0 + g``; any other
        ``p.grad`` -> add (strict: raise, the read bypassed the op)."""
        self.started = False
        for p, (r0, r1) in zip(self.params, self.rows):
            g = self.buf if len(self.params) == 1 else self.buf[r0:r1]
            if p.grad is None:
                p.grad = g
            elif p.grad.data_ptr() == g.data_ptr() and p.grad.shape == g.shape:
                g.add_(0.0)
            else:
                if strict:
                    raise RuntimeError(f"split_wgrad: {self.name}: p.grad was already set when "
                                       f"the backward joined; a read of this weight bypassed "
                                       f"the split op")
                p.grad.add_(g)


def as_bf16(x: Tensor) -> Tensor:
    """``x`` in the dtype autocast gives the GEMM (a no-op for bf16)."""
    return x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)


_ARMED = 0          # depth of open ``train_backward_scope`` blocks


@contextlib.contextmanager
def train_backward_scope():
    """Arm the op for the forward of a TRAINING step whose backward joins into ``p.grad``
    (the trainer's eager step, the warm-ups, ``graph_step``'s recording and eager bodies).
    A read outside the scope takes the plain path with ordinary autograd: a probe that runs
    ``autograd.grad`` (or ``create_graph=True``) through a covered linear never touches
    ``p.grad`` and never meets an op with no double backward."""
    global _ARMED
    _ARMED += 1
    try:
        yield
    finally:
        _ARMED -= 1


def split_active(sw: SplitWeight | None) -> bool:
    """True iff this read runs through the op: inside ``train_backward_scope``, a gradient
    will flow (grad mode, parameters that train) and the GEMM is the bf16 autocast one the
    op does."""
    return (sw is not None and _ARMED > 0 and torch.is_grad_enabled()
            and sw.params[0].requires_grad and _autocast_bf16())


@torch.library.custom_op("morph::split_linear", mutates_args=())
def _split_linear(x: Tensor, w: Tensor, sid: int) -> Tensor:
    """``F.linear(x, w)``; its backward (below) sends the weight grad to the side stream."""
    if _REGISTRY[sid].mgr.triton:
        return linear_fwd(x.reshape(-1, w.shape[1]), w).reshape(*x.shape[:-1], w.shape[0])
    return F.linear(x, w)


@_split_linear.register_fake
def _(x: Tensor, w: Tensor, sid: int) -> Tensor:
    return x.new_empty(*x.shape[:-1], w.shape[0])


@torch.library.custom_op("morph::split_linear_bwd", mutates_args=())
def _split_linear_bwd(dy: Tensor, x: Tensor, w: Tensor, sid: int) -> Tensor:
    """The data grad ``dy @ w`` on the node's stream; the weight grad ``dy^T @ x`` on the side
    stream. The op is opaque to the compiler: a compiled backward calls it as it would any
    kernel, and it holds ``dy`` and ``x`` alive until the side stream is done with them."""
    sw = _REGISTRY[sid]
    ready = torch.cuda.Event()
    ready.record()
    if sw.mgr.triton:
        dx = linear_dgrad(dy.reshape(-1, sw.n_out), w).reshape(x.shape)
    else:
        dx = dy.reshape(-1, sw.n_out).mm(w).reshape(x.shape)
    sw.mgr.defer(sw, dy, x, ready)
    return dx


@_split_linear_bwd.register_fake
def _(dy: Tensor, x: Tensor, w: Tensor, sid: int) -> Tensor:
    return x.new_empty(x.shape)


def _setup(ctx, inputs, output) -> None:
    x, w, sid = inputs
    ctx.save_for_backward(x, w)
    ctx.sid = sid


def _backward(ctx, dy: Tensor):
    x, w = ctx.saved_tensors
    return torch.ops.morph.split_linear_bwd(dy, x, w, ctx.sid), None, None


_split_linear.register_autograd(_backward, setup_context=_setup)


def split_linear(x: Tensor, sw: SplitWeight) -> Tensor:
    """``F.linear(x, W)`` for the covered weight ``sw``: the forward is the ordinary GEMM and
    the weight grad goes to the side stream. ``x`` must be bf16 (the caller casts it, as
    autocast would) and must carry a gradient: the covered parameter is not an input of the
    op, so a read whose ``x`` has none would drop the weight grad silently."""
    if not x.requires_grad:
        raise RuntimeError(f"split_wgrad: {sw.name} is read by an input that carries no "
                           f"gradient while the weight trains; its weight grad would be lost")
    return torch.ops.morph.split_linear(x, sw.weight(), sw.sid)


# ── model.gemm_epilogue_fusion: the whole SwiGLU MLP as ONE op ────────────────────────────
#
# ``down(silu(gate) * up)`` with ``[gate, up] = gate_up(x)``: the forward writes ``h`` in the
# ``gate_up`` GEMM's epilogue, the backward forms ``d_gate`` and ``d_up`` in the ``down`` data-grad
# GEMM's epilogue. The op returns ``h`` and ``gu`` next to ``y`` only so autograd can save them
# (a custom op saves its inputs and outputs, not its internals); nothing reads them. The saved set
# is what the unfused path saves too (``gu`` for the activation backward, ``h`` for ``down``'s weight
# grad). The backward is the two linears' nodes in the order the unfused path runs them, down
# first, so each weight's reads reach the side stream in the same order.

@torch.library.custom_op("morph::split_swiglu_mlp", mutates_args=())
def _split_swiglu_mlp(x: Tensor, w_gu: Tensor, w_dn: Tensor, sid_gu: int, sid_dn: int
                      ) -> tuple[Tensor, Tensor, Tensor]:
    """``(y, h, gu)``: ``y = down(h)``, ``h = silu(gate) * up``, ``gu = gate_up(x)``."""
    lead = x.shape[:-1]
    gu, h = swiglu_fwd(x.reshape(-1, w_gu.shape[1]), w_gu)
    y = linear_fwd(h, w_dn)
    return (y.reshape(*lead, w_dn.shape[0]), h.reshape(*lead, w_dn.shape[1]),
            gu.reshape(*lead, w_gu.shape[0]))


@_split_swiglu_mlp.register_fake
def _(x: Tensor, w_gu: Tensor, w_dn: Tensor, sid_gu: int, sid_dn: int):
    lead = x.shape[:-1]
    return (x.new_empty(*lead, w_dn.shape[0]), x.new_empty(*lead, w_dn.shape[1]),
            x.new_empty(*lead, w_gu.shape[0]))


@torch.library.custom_op("morph::split_swiglu_mlp_bwd", mutates_args=())
def _split_swiglu_mlp_bwd(dy: Tensor, x: Tensor, h: Tensor, gu: Tensor, w_gu: Tensor,
                          w_dn: Tensor, sid_gu: int, sid_dn: int) -> Tensor:
    """``dx`` on the node's stream; both weight grads on the side stream (``down``'s first)."""
    sw_gu, sw_dn = _REGISTRY[sid_gu], _REGISTRY[sid_dn]
    ready = torch.cuda.Event()
    ready.record()
    dgu = swiglu_bwd(dy.reshape(-1, sw_dn.n_out), w_dn, gu.reshape(-1, sw_gu.n_out))
    sw_dn.mgr.defer(sw_dn, dy, h, ready)
    ready = torch.cuda.Event()
    ready.record()                                   # dgu is complete past this point
    dx = linear_dgrad(dgu, w_gu).reshape(x.shape)
    sw_gu.mgr.defer(sw_gu, dgu, x, ready)
    return dx


@_split_swiglu_mlp_bwd.register_fake
def _(dy: Tensor, x: Tensor, h: Tensor, gu: Tensor, w_gu: Tensor, w_dn: Tensor, sid_gu: int,
      sid_dn: int) -> Tensor:
    return x.new_empty(x.shape)


def _swiglu_setup(ctx, inputs, output) -> None:
    x, w_gu, w_dn, sid_gu, sid_dn = inputs
    _y, h, gu = output
    ctx.save_for_backward(x, h, gu, w_gu, w_dn)
    ctx.sids = (sid_gu, sid_dn)
    ctx.set_materialize_grads(False)                 # h and gu carry no gradient


def _swiglu_backward(ctx, dy: Tensor, _dh, _dgu):
    x, h, gu, w_gu, w_dn = ctx.saved_tensors
    dx = torch.ops.morph.split_swiglu_mlp_bwd(dy, x, h, gu, w_gu, w_dn, *ctx.sids)
    return dx, None, None, None, None


_split_swiglu_mlp.register_autograd(_swiglu_backward, setup_context=_swiglu_setup)


def split_swiglu_mlp(x: Tensor, sw_gu: SplitWeight, sw_dn: SplitWeight) -> Tensor:
    """``down(silu(gate) * up)`` of ``[gate, up] = gate_up(x)`` for the covered weights, with the
    SwiGLU in the two GEMMs' epilogues (``model.gemm_epilogue_fusion``). Same contract as
    ``split_linear``: ``x`` is bf16 and carries a gradient."""
    if not (sw_gu.mgr.triton and sw_dn.mgr.triton):
        raise RuntimeError("model.gemm_epilogue_fusion needs the Triton split GEMMs "
                           "(model.split_wgrad_triton)")
    if sw_gu.n_out != 2 * sw_dn.n_in or sw_gu.n_in != sw_dn.n_out:
        raise ValueError(f"split_swiglu_mlp: {sw_gu.name} ({sw_gu.n_out} x {sw_gu.n_in}) and "
                         f"{sw_dn.name} ({sw_dn.n_out} x {sw_dn.n_in}) are not one SwiGLU MLP")
    if not x.requires_grad:
        raise RuntimeError(f"split_wgrad: {sw_gu.name} is read by an input that carries no "
                           f"gradient while the weight trains; its weight grad would be lost")
    return torch.ops.morph.split_swiglu_mlp(x, sw_gu.weight(), sw_dn.weight(), sw_gu.sid,
                                            sw_dn.sid)[0]


def bind_split_wgrad(model: nn.Module, *, strict: bool = False, triton: bool = False,
                     epilogue: bool = False) -> WgradStream:
    """Cover every linear named in the module notes. Call AFTER ``bind_ternary_step_cache`` and
    BEFORE the compile warm-up. Raises when it covers nothing, when a covered module is not
    one the op reproduces exactly (a bias, a parametrization other than the cached ternary
    STE, a carved or FP8 layer), or when the attention input fusion is off. ``epilogue``
    (``model.gemm_epilogue_fusion``): every covered SwiGLU MLP runs as ONE op with the SwiGLU in
    its GEMMs' epilogues; needs ``triton``, and an MLP with a router or a missing covered linear
    is refused."""
    if epilogue and not triton:
        raise ValueError("model.gemm_epilogue_fusion needs model.split_wgrad_triton=true: the "
                         "SwiGLU epilogues live in the Triton split GEMMs.")
    import torch.nn.utils.parametrize as parametrize

    from . import attention
    from .attention import _CCABase, _CCACSAAttention, _CCAHCAAttention
    from .layers.block_sparse import CMSBlockLinear
    from .ternary_qat import TernarySTE

    mgr = WgradStream(side_stream("wgrad"), strict=strict, triton=triton)
    n = {"mlp": 0, "attn_in": 0, "attn_up": 0}
    fused_mlps = []

    def plain_param(name: str, mod: nn.Module) -> nn.Parameter:
        """The fp32 parameter of a linear that nothing parametrizes (attention projections)."""
        if parametrize.is_parametrized(mod, "weight"):
            raise ValueError(f"model.split_wgrad: {name}.weight is parametrized (ternary or "
                             f"quantised attention); only plain attention projections and the "
                             f"step-cached ternary MLP are covered")
        if getattr(mod, "bias", None) is not None:
            raise ValueError(f"model.split_wgrad: {name} has a bias")
        return mod.weight

    for name, mod in model.named_modules():
        if isinstance(mod, CMSBlockLinear):
            if not mod._dense_mode or mod._mortar or mod._fp8_mode or mod._fp8_tfwd \
                    or mod._ternary_mode or mod._values_ternary_mode:
                raise ValueError(f"model.split_wgrad: {name} is carved, FP8 or per-tile "
                                 f"ternary; only the dense layer is covered")
            if mod.bias is not None:
                raise ValueError(f"model.split_wgrad: {name} has a bias")
            if not parametrize.is_parametrized(mod, "weight"):
                raise ValueError(f"model.split_wgrad: {name}.weight is not ternary; the "
                                 f"recipe's MLP is (model.ternary_step_cache)")
            plist = mod.parametrizations.weight
            ste = plist[0]
            if len(plist) != 1 or not isinstance(ste, TernarySTE) \
                    or "_wq" not in ste._buffers:
                raise ValueError(f"model.split_wgrad: {name}.weight is not a lone step-cached "
                                 f"TernarySTE; set model.ternary_step_cache=true")
            mod._split = SplitWeight(
                mgr, name, [plist.original], [(0, mod.out_features)], mod.in_features,
                kind="cached", ste=ste)
            n["mlp"] += 1
        elif isinstance(mod, (_CCACSAAttention, _CCAHCAAttention)):
            if not attention._FUSED_ATTN_PROJ:
                raise ValueError("model.split_wgrad needs the fused attention input projection "
                                 "(MORPH_FUSED_ATTN_PROJ): the unfused reads are separate "
                                 "GEMMs the op does not cover")
            mods = mod._fuse_mods
            params = [plain_param(f"{name}.fuse[{i}]", m) for i, m in enumerate(mods)]
            rows, r = [], 0
            for m in mods:
                rows.append((r, r + m.out_features))
                r += m.out_features
            mod._split_in = SplitWeight(
                mgr, f"{name}.fused_in", params, rows, mods[0].in_features,
                kind="fused", mods=mods)
            n["attn_in"] += 1
        elif isinstance(mod, _CCABase):
            p = plain_param(f"{name}.W_up", mod.W_up)
            mod._split_up = SplitWeight(
                mgr, f"{name}.W_up", [p], [(0, mod.W_up.out_features)], mod.W_up.in_features,
                kind="plain", mods=(mod.W_up,))
            n["attn_up"] += 1
    if epilogue:
        from .transformer import _SwiGLUMortar
        for name, mod in model.named_modules():
            if isinstance(mod, _SwiGLUMortar):
                if mod.router is not None:
                    raise ValueError(f"model.gemm_epilogue_fusion: {name} has a ReMoE router; "
                                     f"the fused MLP op has no gate between the SwiGLU and "
                                     f"down (route_start must stay beyond the run)")
                if mod.gate_up._cms._split is None or mod.down._cms._split is None:
                    raise ValueError(f"model.gemm_epilogue_fusion: {name} has an MLP linear the "
                                     f"split op does not cover")
                mod._swiglu_fused = True
                fused_mlps.append(name)
        if not fused_mlps:
            raise ValueError("model.gemm_epilogue_fusion: the model has no SwiGLU MLP to fuse")
    if not any(n.values()):
        raise ValueError("model.split_wgrad: the model has no covered linear")
    print(f"  [split-wgrad] ON: {n['mlp']} ternary MLP linears, {n['attn_in']} fused attention "
          f"input projections, {n['attn_up']} W_up; weight grads on a side stream"
          f"{', Triton GEMMs' if triton else ''}"
          f"{f', SwiGLU epilogues in {len(fused_mlps)} MLPs' if epilogue else ''}", flush=True)
    return mgr
