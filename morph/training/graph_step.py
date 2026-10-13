"""``training.graph_step``: the training step as a replayed CUDA graph.

The winner step is host-serial: ~31k launches at ~20 us each, and the GPU waits on the host
(.agents/notes/proposed/architecture/2026-10-07-graph-captured-training-step.md, tasks 1.6 to
1.8). Under this key the trainer records forward, backward, clip, found-inf flag, optimizer
step and the fan's post-step update (``tul_fan_after_step``) ONCE per step kind, and replays
the recording on every later step of that kind. A replay is one ``cudaGraphLaunch``.

Two step KINDS, one graph each, sharing one memory pool: the regular step and the instrument
step (``MORPHTransformer._train_instruments``, every 20th step, the log step). After the
recording NO training step runs eager, the instrument step included: an eager step needs its
own activation memory next to the pool, which holds about one step's activations for the
run. Measured on the 300M winner (2026-10-07): pool 16.6 GiB, and the eager instrument step at
step 20 ran out of memory on the 31.4 GiB card. The graphs never run concurrently, so they
may reuse each other's free blocks; a graph's outputs are read only right after its replay.

VARIANTS (2026-10-10): a host choice a step makes that changes its op sequence and so cannot
live inside one recording. The one in use is the gain hinge's pass (``model.slot_gain_pass_
force``; the hinge runs at ONE loop pass, a Python branch): the trainer passes the model's
``gain_pass_choices()`` as ``variants``, ``draw_gain_pass`` as ``draw_variant`` and
``pin_gain_pass`` as ``set_variant``. ONE graph is recorded per (kind, variant), all in the
same pool; every step, eager or replayed, draws its variant once on the host, and a replay
runs the graph recorded for it. The draw is the eager forward's own draw (same generator, one
uniform per step), so a GraphStep run draws the same passes as an eager run. Why one graph
per pass and not a device-side index into one recording: under ``model.slot_compact`` each
pass runs the hinge on that pass's own row count (a different shape per pass), and the
hinge's dropout masks come from the CUDA stream at its call site, so one shared hinge would
be a different function from the eager one, not the same function picked late.

What a replay relies on, and where each is kept:
  * Inputs at fixed addresses. ``StaticBatch`` holds ids, labels and every ``SlotLayout``
    tensor (``ditto_prev`` included); each step ``copy_``s its batch in before the replay.
  * Parameters, buffers and optimizer state updated IN PLACE and never rebound. Checked:
    every address is recorded at capture and compared before the first replay after any
    eager work (an eager step, eval, a checkpoint), and a moved tensor raises.
  * Gradients: ``zero_grad(set_to_none=True)`` is the body's first statement, so inside the
    capture every gradient is allocated from the graph's pool, written by the backward and
    read by the clip and the optimizer within one replay. No eager step can free them; an
    eager step between replays allocates its own (the trainer's ``zero_grad`` drops the
    references every step, as today). A KEPT kind (``keep_grads``, the trainer's log block
    reads ``p.grad`` after the step) ends its recording with a copy of every gradient into
    ONE set of buffers outside the pool, shared by all graphs (``_grad_bufs``); a replay binds
    ``p.grad`` to them. Kept per graph instead, the 8 instrument graphs of the hinge's 8 passes
    would hold 8 gradient sets (8 x 1.2 GB on the 300M winner).
  * Per-step scalars on the device (``training.capturable_optimizer``):
    ``optimizer.write_step_scalars()`` runs before the replay, outside the graph.
  * One op sequence and fixed shapes (``model.graph_safe``: fixed pass count, no host read).
  * Randomness: the CUDA generator is registered with the graph (philox seed and offset are
    read at replay and the offset advances by the graph's increment), so a replay draws what
    an eager step at that point would. The gain hinge's pass is a VARIANT (above), drawn on
    the host from the model's private generator before each step. The global CPU generator
    is read by nothing in the step: a recording that moves it raises (a host draw inside the
    step would be frozen into the graph).
  * Compiled frames: the capture runs under the ``fail_on_recompile`` stance, so a guard miss
    raises instead of silently recording an eager fallback (the trainer's
    ``eager_on_recompile`` stance would), and the dynamo graph count must not move.
  * Python state: a replay runs no Python, so a module attribute that a step changes (a
    counter, a schedule flag) would freeze. Every scalar attribute of every module is
    compared across each recording; a change raises. A TENSOR attribute the step sets to a
    new object (``_loop_cot_ref``, set by the cotangent-clip hook in the backward) is the
    recording's object, which the replay writes: each replay binds the attributes its
    recording set again (``_Captured.rebind``, 2026-10-10), so a reader sees that replay's
    values and not those of the last recording or eager step.

SHAPES (2026-10-12, ``tul.cell_count``). A batch's SHAPE is a host property of the batch itself,
not a choice the step makes: the fan's cell count M changes the row length and so every tensor
shape in the step. ``shape_of(layout)`` reads it (``layout.prefix_k``); ONE graph is recorded per
(kind, variant, shape), each shape with its own ``StaticBatch``, all in the same pool. At the
recording ``shape_batch(shape, x, y, layout)`` gives the capture batch's rows laid out at each
other shape (the recording runs no kernel, so only the shapes and the layout structure matter,
not the values). With the default ``shape_of=None`` there is one shape and the graph key is
(kind, variant), as before. The roles of ``tul.cell_count_subset: random`` ride in the layout
(``cell_roles``, a device tensor gathered inside the graph), so a different subset replays the
same recording.

Order: every step runs eager, on the capture side stream (the PyTorch recipe's warm-up),
until ``first_capture`` and until every kind has run once. On the next step every (kind,
variant) is recorded on that step's batch, then that step's (kind, variant) is REPLAYED (a
recording executes no kernel). From there on every step is a replay. Every variant's compiled
frames must exist before the recording (``fail_on_recompile``): the trainer's warm-up compile
runs one forward per hinge pass (train.py ``warmup_compile_all_shapes``).

THE HINGE BRANCH (2026-10-10, phase-2 tracker C4). The gain hinge
(``model.slot_gain_lambda``) adds two core applications at the drawn pass, and their backward
is about 11 ms of the lxtul_pointer_fast step. On most steps no row's gain is over the target,
the hinge is exactly 0 and so is its gradient: that backward adds exact zeros. When the
forward reports the hinge (``out["gain_pen"]``, the penalty as it enters the loss, and
``out["gain_active"]``, the device flag "some charged term is not 0"), a recording is FOUR
torch graphs in the pool, run as three launches:

    pre   = zero_grad, forward, loss terms, ``loss / 1``           (torch replay)
    IF gain_active:  A = the backward, exactly the eager one        (one CondBranch launch,
    ELSE:            B = the backward without the penalty's edge     morph/training/graph_cond.py)
    post  = clip, found-inf flag, optimizer step, after_step, keep copy  (torch replay)

B is a backward from the loss chain's OTHER addends (``_cut_roots``: the pass-through adds and
the ``/ 1`` from the loss down to the penalty's add, each other edge a root with the gradient
1 it receives there), so the hinge's nodes are never reached. Why B equals the eager step: the
autograd engine runs the reachable nodes in descending creation order (a node's consumers are
all newer, so by induction the newest unexecuted node is always ready when its turn comes);
leaving the hinge's nodes out keeps every other node's order and every input buffer's
summation order, and drops addends that are exact zeros. Two things can differ: the sign of a
zero gradient element (+0 + -0), and a step on which a hinge-side activation is inf (its zero
gradient times inf is NaN in A, which the found-inf flag would catch; B never computes it).
A is recorded first, with ``retain_graph`` (B walks the same saved tensors), so A allocates
every gradient; B zeroes those buffers in place and accumulates into them, so ``post`` reads
one set either way (``p.grad`` after B is the same tensor objects A left: checked). A module
attribute a backward hook sets (``_loop_cot_ref``, the instrument kind's ``_loop_cot*``) is
the object B set when the recording ends; B ends by copying its values into A's objects and
rebinding A's (``_TensorAttrs``), so a reader sees the step's values whichever body ran. Both
bodies must draw no random numbers (graph_cond.py): the generator cannot be read during a
capture, so the eager steps before the recording measure the backward's draws
(``_eager_bwd_draws``), and a backward that drew (or no eager step) makes the recording raise.
The eager step (warm-up, ``run_eager``) is the plain body. A recording whose forward reports
no hinge is two graphs, pre and the rest, replayed in turn (the same kernels as one graph).
The hinge branch costs a zero fill of every gradient on a B step and a second backward's
recording; A keeps the forward's saved tensors to its end (``retain_graph``), so its
recording cannot reuse their memory for its temporaries (the pool can grow; measured in the
tracker's C4 entry).

Eval and generation do not fit beside the pool either (measured: eval's pointer mixture OOM
at step 25 with the 15.4 GiB pool held, also when it borrowed the pool's free blocks). The
trainer runs them inside ``released()``: the graphs and their pool are freed (after a device
sync), and the next step records every graph again on its own batch and replays. A recording
executes no kernel and records the same op sequence, so this costs recording time only (once
per (kind, variant): the message prints the seconds); the 45-step gate trace is byte-identical
either way.
"""
from __future__ import annotations

import gc
import itertools
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from morph.model.split_wgrad import train_backward_scope
from morph.model.tul_layout import SlotLayout
from morph.training.graph_cond import CondBranch

__all__ = ["GraphStep", "StaticBatch", "graph_step_refusals"]

# SlotLayout's tensor fields, in declaration order. A field added to SlotLayout that is not
# listed here raises in StaticBatch rather than being silently left at its capture value.
_LAYOUT_TENSORS = ("slot_mask", "bag_id", "slot_index", "slot_valid", "span_len",
                   "len_supervised", "ditto_prev", "cell_roles")
_LAYOUT_HOST = ("prefix_k", "stats")


def graph_step_refusals(cfg, *, total_steps: int, curriculum: bool) -> list[str]:
    """Every reason this config cannot run ``training.graph_step`` (empty = it can).

    The key needs the two halves it is built on, and refuses every feature that makes the
    step's op sequence or shapes depend on the batch, reads a device value on the host inside
    the step, or changes the parameter set mid-run."""
    tr, m = cfg.training, cfg.model
    tul = getattr(cfg, "tul", None)
    g = lambda node, k, d=None: getattr(node, k, d) if node is not None else d  # noqa: E731
    out: list[str] = []
    if not bool(g(tr, "capturable_optimizer", False)):
        out.append("training.capturable_optimizer must be true (no GradScaler host read, "
                   "device schedule scalars)")
    if not bool(g(m, "graph_safe", False)):
        out.append("model.graph_safe must be true (fixed pass count, no host-shadow reads)")
    parts = g(m, "graph_safe_parts", None)
    if parts is not None:
        from morph.model.transformer import MORPHConfig
        missing = sorted(set(MORPHConfig.graph_safe_parts) - set(parts))
        if missing:
            out.append(f"model.graph_safe_parts leaves out {missing}: a piece left out reads "
                       f"a host count the replay would freeze")
    if bool(g(m, "ce_compact_rows", False)):
        out.append("model.ce_compact_rows: the compacted row count is data-dependent")
    if curriculum:
        out.append("curriculum: the stage changes the sequence length and grad accumulation")
    if int(g(tr, "grad_accum", 1) or 1) != 1:
        out.append("training.grad_accum must be 1")
    if g(tr, "step_mix", None):
        out.append("training.step_mix: the forward mode changes per step")
    if float(g(tr, "ntp_dropout_p", 0.0) or 0.0) > 0.0:
        out.append("training.ntp_dropout_p: a second, slot-free forward on some steps")
    for k in ("grad_probe_every", "jac_probe_every", "loop_rank_every", "batch_dump_every"):
        if int(g(tr, k, 0) or 0) > 0:
            out.append(f"training.{k}: a probe between the backward and the clip reads the "
                       f"device on the host")
    for k in ("abort_core_share", "abort_block_gain"):
        if float(g(tr, k, 0.0) or 0.0) > 0.0:
            out.append(f"training.{k}: the abort guard needs the per-step probe")
    if float(g(tr, "spectral_penalty_lambda", 0.0) or 0.0) > 0.0 and \
            float(g(tr, "spectral_penalty_cap", 0.0) or 0.0) > 0.0:
        out.append("training.spectral_penalty: the live penalty's value is read on the host")
    if float(g(tr, "spectral_project_cap", 0.0) or 0.0) > 0.0:
        out.append("training.spectral_project_cap: the projection is not part of the body")
    if bool(g(tr, "frozen_eval", False)):
        out.append("training.frozen_eval: the root runs in eval mode, where graph_safe is off")
    if float(g(tul, "code_target_ema", 0.0) or 0.0) > 0.0:
        out.append("tul.code_target_ema: the code reference EMA is not part of the body")
    if bool(g(tul, "code", False)):
        out.append("tul.code: the phase switches retrace the forward")
    # Pruning: an event changes masks or the parameter set; scoring reads grads each step.
    ps = int(g(tr, "prune_start", 10**9))
    cs = int(g(tr, "compact_step", 10**9))
    rs = int(g(getattr(cfg, "routing", None), "route_start", 0) or 0)
    if ps < total_steps:
        out.append(f"training.prune_start={ps} < steps={total_steps}: pruning inside the run")
    if cs < total_steps:
        out.append(f"training.compact_step={cs} < steps={total_steps}: carve inside the run")
    if 0 < rs < total_steps:
        out.append(f"routing.route_start={rs} < steps={total_steps}: routing inside the run")
    return out


class StaticBatch:
    """Fixed device buffers for one batch: ids, labels and every ``SlotLayout`` tensor.

    ``load`` copies a batch in (``copy_``, on the current stream, so it is ordered before
    the replay launched after it) and checks that its shapes, dtypes and present fields are
    the captured ones. The layout's host fields (``prefix_k``) must match; its ``stats`` are
    host-only readings the model does not use, so the static layout carries none."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, layout: Optional[SlotLayout]):
        self.x = torch.empty_like(x)
        self.y = torch.empty_like(y)
        self.layout = None
        if layout is not None:
            extra = set(vars(layout)) - set(_LAYOUT_TENSORS) - set(_LAYOUT_HOST)
            if extra:
                raise RuntimeError(f"StaticBatch: SlotLayout has fields {sorted(extra)} that "
                                   f"no static buffer holds; add them to _LAYOUT_TENSORS")
            kw = {k: (None if getattr(layout, k) is None else torch.empty_like(getattr(layout, k)))
                  for k in _LAYOUT_TENSORS}
            self.layout = SlotLayout(prefix_k=layout.prefix_k, stats=None, **kw)

    @staticmethod
    def _copy(dst: Optional[torch.Tensor], src: Optional[torch.Tensor], what: str) -> None:
        if (dst is None) != (src is None):
            raise RuntimeError(f"StaticBatch: {what} is {'absent' if src is None else 'present'}"
                               f" in this batch but was {'present' if src is None else 'absent'}"
                               f" at capture")
        if dst is None:
            return
        if dst.shape != src.shape or dst.dtype != src.dtype:
            raise RuntimeError(f"StaticBatch: {what} is {tuple(src.shape)} {src.dtype}, the "
                               f"graph was captured on {tuple(dst.shape)} {dst.dtype}")
        dst.copy_(src, non_blocking=True)

    def load(self, x: torch.Tensor, y: torch.Tensor, layout: Optional[SlotLayout]) -> None:
        self._copy(self.x, x, "input_ids")
        self._copy(self.y, y, "labels")
        if (self.layout is None) != (layout is None):
            raise RuntimeError("StaticBatch: a slot layout appeared or vanished after capture")
        if layout is None:
            return
        if layout.prefix_k != self.layout.prefix_k:
            raise RuntimeError(f"StaticBatch: prefix_k {layout.prefix_k} != captured "
                               f"{self.layout.prefix_k}")
        for k in _LAYOUT_TENSORS:
            self._copy(getattr(self.layout, k), getattr(layout, k), f"layout.{k}")


_PASS_THROUGH = ("AddBackward0", "DivBackward0", "DivBackward1")


def _cut_roots(loss: torch.Tensor, cut: torch.Tensor) -> list:
    """The backward roots of ``loss`` with ``cut``'s edge left out (module notes, THE HINGE
    BRANCH): walk the loss chain's pass-through nodes (an add with alpha 1, a division by 1:
    each hands its incoming gradient unchanged to every input) down to the add that takes
    ``cut``; every OTHER input edge met on the way is a root, which receives the gradient 1
    the chain would hand it. A pass-through node with no ``cut`` below stays one root. Raises
    if ``cut`` is not an addend of the chain (or sits under any other node): the cut would
    then change more than the hinge."""
    from torch.autograd.graph import GradientEdge
    want = (cut.grad_fn, cut.output_nr)

    def through(node) -> bool:
        name = node.name()
        if name == "AddBackward0":
            return float(node._saved_alpha) == 1.0
        if name in ("DivBackward0", "DivBackward1"):
            o = node._saved_other
            return float(o) == 1.0 if not torch.is_tensor(o) or o.numel() == 1 else False
        return False

    def expand(node, nr) -> tuple[list, bool]:
        if node is None:
            return [], False
        if node is want[0] and nr == want[1]:
            return [], True
        if through(node):
            parts = [expand(n, k) for n, k in node.next_functions]
            if any(f for _r, f in parts):
                return [r for rs, _f in parts for r in rs], True
        return [GradientEdge(node, nr)], False

    roots, found = expand(loss.grad_fn, loss.output_nr)
    if not found:
        raise RuntimeError("graph_step: out['gain_pen'] is not an addend of the loss chain "
                           "(pass-through adds and '/ 1' only); the hinge branch cannot cut it")
    return roots


def _attr_objs(mods) -> dict:
    """``{(id(module), name): (module, name, object)}`` for every tensor attribute and every
    non-empty dict-of-tensors attribute of ``mods`` (``_TensorAttrs._copyable``; a dict is
    copied, so a later in-place edit of the module's dict does not change the snapshot)."""
    return {(id(m), k): (m, k, c) for m in mods for k, v in vars(m).items()
            if k not in _TensorAttrs._SKIP and (c := _TensorAttrs._copyable(v)) is not None}


def _same_obj(a, b) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return a is b
    return a.keys() == b.keys() and all(a[i] is b[i] for i in a)


class _TensorAttrs:
    """The tensor attributes (and dicts of tensors) of some modules, as one backward left
    them; ``restore()``, inside the next body's recording, copies the values that body left
    into these objects and binds them again (module notes, THE HINGE BRANCH)."""

    _SKIP = ("_parameters", "_buffers", "_modules")

    def __init__(self, mods):
        self.mods = mods
        self.snap = [(m, k, self._copyable(v)) for m in mods for k, v in vars(m).items()
                     if k not in self._SKIP and self._copyable(v) is not None]

    @staticmethod
    def _copyable(v):
        if torch.is_tensor(v):
            return v
        if isinstance(v, dict) and v and all(torch.is_tensor(t) for t in v.values()):
            return dict(v)
        return None

    def restore(self) -> None:
        for m, k, old in self.snap:
            new = vars(m).get(k)
            if new is old or (isinstance(old, dict) and isinstance(new, dict)
                              and all(new.get(i) is t for i, t in old.items())
                              and new.keys() == old.keys()):
                continue
            pairs = ([(old, new)] if torch.is_tensor(old)
                     else [(t, new[i]) for i, t in old.items()]
                     if isinstance(new, dict) and new.keys() == old.keys() else None)
            if pairs is None or any(not torch.is_tensor(b) or a.shape != b.shape
                                    or a.dtype != b.dtype for a, b in pairs):
                raise RuntimeError(f"graph_step: {type(m).__name__}.{k} is set differently by "
                                   f"the hinge branch's two bodies; a replay could not read it")
            for a, b in pairs:
                if a is not b:
                    a.copy_(b)
            setattr(m, k, old if torch.is_tensor(old) else type(new)(old))


@dataclass
class _Captured:
    """One (kind, variant)'s graph and the tensors it writes."""

    graph: torch.cuda.CUDAGraph       # `pre`: zero_grad, forward, loss (module notes)
    rest: Optional[torch.cuda.CUDAGraph]   # backward + post, when there is no hinge branch
    branch: Optional[CondBranch]      # IF gain_active: A ELSE: B (the hinge branch)
    post: Optional[torch.cuda.CUDAGraph]   # clip .. keep copy, after the branch
    bodies: tuple                     # A and B (kept alive: the branch runs their clones)
    rebind: list                      # (module, name, object): the tensor attributes the
                                      # recording set; a replay binds them again
    out: dict
    loss: torch.Tensor
    gnorm: torch.Tensor
    present: Optional[list]           # kept kinds: which params had a gradient (copied to
                                      # the shared `_grad_bufs`); None = gradients not kept
    context: Any                      # the trainer's per-capture key (the phase)
    replays: int = 0


def _close_branches(graphs: dict) -> None:
    """Close every recorded hinge branch. A function so its loop variable dies on return
    (see `GraphStep.released`)."""
    for cap in graphs.values():
        if cap.branch is not None:
            cap.branch.close()


@dataclass
class GraphStepStats:
    eager: int = 0                    # steps run eager (warm-up, or run_eager between replays)
    replayed: int = 0                 # steps run as a replay, the capture step included
    captures: int = 0                 # graphs recorded
    releases: int = 0                 # times the graphs were freed for eager work (eval)

    def eager_share(self) -> float:
        return self.eager / max(self.eager + self.replayed, 1)


class GraphStep:
    """Run each training step eager or as a replay (module notes).

    ``forward(x, y, layout) -> dict`` is the trainer's model call (its ``out`` dict, with
    ``out["loss"]``); ``loss_terms()`` returns the addends the trainer puts on the loss inside
    autocast (the logging-only spectral penalty's exact zero) or None; ``after_step()`` is the
    trainer's post-optimizer work (``tul_fan_after_step``). The body is the eager loop's
    sequence, statement for statement, so a replay is bit-identical to an eager step.

    ``kinds`` and ``set_kind``: the step kinds (the trainer's ``_train_instruments`` flag) and
    the callback that switches the model to one, so that ONE capture step records all of them.
    ``first_capture``: the first step index graphs may be recorded at; until then, and until
    every kind has run eager once (on the capture stream: the PyTorch recipe's warm-up), steps
    run eager. Then every kind is recorded on that step, which is then REPLAYED: after the
    first recording no training step runs eager, because an eager step needs activation memory
    of its own next to the graphs' pool (on the 300M winner the two do not fit on 32 GB:
    measured, an eager instrument step after the regular capture ran out of memory).
    ``keep_grads``: kinds whose gradients a replay binds to ``p.grad`` afterwards (the log
    block's per-region gradient norms read them); each such recording ends by copying them into
    the shared ``_grad_bufs`` (module notes, Gradients).

    ``variants``, ``draw_variant``, ``set_variant`` (module notes, VARIANTS): the host choices
    besides the kind, one graph per (kind, variant). ``draw_variant()`` runs once per training
    step, eager or replayed, and returns one of ``variants``; ``set_variant(v)`` makes the
    following forwards run variant ``v`` (``set_variant(None)`` after every step). Default: one
    variant, ``None``, and no callbacks: the two-graph step from before."""

    def __init__(self, model: nn.Module, optimizer, *, grad_clip: float,
                 forward: Callable[[torch.Tensor, torch.Tensor, Optional[SlotLayout]], dict],
                 kinds: tuple, set_kind: Callable[[Any], None],
                 loss_terms: Optional[Callable[[], Optional[torch.Tensor]]] = None,
                 after_step: Optional[Callable[[], None]] = None,
                 first_capture: int = 3, keep_grads: tuple = (True,),
                 compiled: bool = True, variants: tuple = (None,),
                 draw_variant: Optional[Callable[[], Any]] = None,
                 set_variant: Optional[Callable[[Any], None]] = None,
                 shapes: tuple = (None,),
                 shape_of: Optional[Callable[[Optional[SlotLayout]], Any]] = None,
                 shape_batch: Optional[Callable[[Any, torch.Tensor, torch.Tensor,
                                                 Optional[SlotLayout]], tuple]] = None):
        if not getattr(optimizer, "capturable", False):
            raise ValueError("GraphStep needs a capturable optimizer "
                             "(training.capturable_optimizer)")
        variants = tuple(variants)
        if not variants:
            raise ValueError("GraphStep: variants must hold at least one value")
        if len(variants) > 1 and (draw_variant is None or set_variant is None):
            raise ValueError(f"GraphStep: {len(variants)} variants need draw_variant and "
                             f"set_variant (which one a step runs is a host draw)")
        if variants != (None,) and set_variant is None:
            raise ValueError("GraphStep: a variant other than None needs set_variant")
        shapes = tuple(shapes)
        if not shapes:
            raise ValueError("GraphStep: shapes must hold at least one value")
        if shape_of is None and shapes != (None,):
            raise ValueError("GraphStep: shapes other than (None,) need shape_of")
        if len(shapes) > 1 and shape_batch is None:
            raise ValueError(f"GraphStep: {len(shapes)} shapes need shape_batch (the capture "
                             f"batch laid out at each other shape)")
        self.shapes = shapes
        self.shape_of = shape_of
        self.shape_batch = shape_batch
        self.model = model
        self.root = getattr(model, "_orig_mod", model)
        self.optimizer = optimizer
        self.params = list(model.parameters())
        self.grad_clip = float(grad_clip)
        self.forward = forward
        self.kinds = tuple(kinds)
        self.set_kind = set_kind
        self.loss_terms = loss_terms
        self.after_step = after_step
        self.first_capture = int(first_capture)
        self.keep_grads = tuple(keep_grads)
        self.compiled = bool(compiled)
        self.variants = variants
        self.draw_variant = draw_variant
        self.set_variant = set_variant
        self.stream = torch.cuda.Stream()
        # The graphs' private pool (a fresh one after every `released`).
        self.mempool = torch.cuda.MemPool()
        self.static: dict[Any, StaticBatch] = {}   # keyed by shape
        self.graphs: dict[Any, _Captured] = {}    # keyed (kind, variant), or (.., shape)
        self.stats = GraphStepStats()
        self._warm: set = set()               # kinds that have run eager on the stream
        self._addresses: Optional[list] = None
        self._dirty = True                    # eager work since the last address check
        # The kept kinds' gradients, ONE set for every graph (module notes, Gradients).
        # Allocated outside the pool before a recording, freed with the graphs.
        self._grad_bufs: Optional[list] = None
        # Whether an eager step's backward moved the CUDA generator (None: no eager step
        # yet). The hinge branch needs False (module notes, THE HINGE BRANCH).
        self._eager_bwd_draws: Optional[bool] = None

    # ── the shape (module notes, SHAPES) ────────────────────────────────────────────────
    def _shape(self, layout):
        sh = self.shape_of(layout) if self.shape_of is not None else None
        if sh not in self.shapes:
            raise RuntimeError(f"graph_step: the batch has shape {sh!r}, not one of the "
                               f"recorded {self.shapes}")
        return sh

    def _key(self, kind, variant, shape):
        """The graph key: (kind, variant), plus the shape when shapes are in use."""
        return (kind, variant) if self.shape_of is None else (kind, variant, shape)

    # ── the variant (module notes, VARIANTS) ────────────────────────────────────────────
    def _next_variant(self):
        """This step's variant: drawn once per training step (eager or replayed) and set on
        the model, so the forwards of the step run it. Released by ``_end_variant``."""
        v = self.draw_variant() if self.draw_variant is not None else self.variants[0]
        if v not in self.variants:
            raise RuntimeError(f"graph_step: the step drew variant {v!r}, not one of the "
                               f"recorded {self.variants}")
        if self.set_variant is not None:
            self.set_variant(v)
        return v

    def _end_variant(self) -> None:
        if self.set_variant is not None:
            self.set_variant(None)

    # ── the step body: the trainer's eager sequence (train.py, fwd .. opt) ──────────────
    def _body(self, x, y, layout, keep: bool = False):
        """The step. ``keep`` (a kept kind's RECORDING only): end by copying every gradient
        into the shared ``_grad_bufs``, bits unchanged, so the replay can bind them."""
        out, loss = self._forward_loss(x, y, layout)
        # The CUDA generator's offset across the backward (eager, so it is readable): a
        # backward that draws cannot be a hinge-branch body (module notes).
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        off0 = gen.get_offset()
        # `/ 1`: the trainer's `scaler.scale(loss / _ga)` at grad_accum 1 with the disabled
        # scaler, kept so the backward starts from the same node.
        (loss / 1).backward()
        self._eager_bwd_draws = bool(self._eager_bwd_draws) or gen.get_offset() != off0
        gnorm = self._post(keep)
        return out, loss, gnorm

    def _forward_loss(self, x, y, layout):
        """The body up to the loss: zero_grad, the forward under autocast, the loss terms."""
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16), train_backward_scope():
            out = self.forward(x, y, layout)
            loss = out["loss"]
            if self.loss_terms is not None:
                extra = self.loss_terms()
                if extra is not None:
                    loss = loss + extra.to(loss.dtype)
        return out, loss

    def _post(self, keep: bool):
        """The body after the backward: clip, found-inf flag, optimizer step, after_step and
        (``keep``) the copy of every gradient into ``_grad_bufs``."""
        gnorm = nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.mark_found_inf(gnorm)
        self.optimizer.step()
        if self.after_step is not None:
            self.after_step()
        if keep:
            have = [(b, p.grad) for b, p in zip(self._grad_bufs, self.params)
                    if p.grad is not None]
            if have:
                torch._foreach_copy_([b for b, _ in have], [g for _, g in have])
        return gnorm

    @staticmethod
    def _detached(out: dict) -> dict:
        return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}

    # ── the persistent state a replay reads and writes in place ─────────────────────────
    def _hidden_modules(self):
        """Modules held in a plain attribute (not registered), e.g. the fan's EMA twin
        ``_fan_target``: their tensors are updated by the graph too."""
        seen, out = set(), []
        for mod in self.root.modules():
            for k, v in vars(mod).items():
                if isinstance(v, nn.Module) and id(v) not in seen:
                    seen.add(id(v))
                    out.append((f"{type(mod).__name__}.{k}", v))
        return out

    def _persistent(self) -> list[tuple[str, int]]:
        ts: list[tuple[str, torch.Tensor]] = []
        for owner, mod in [("", self.root)] + self._hidden_modules():
            ts += [(f"{owner}:{n}", t) for n, t in mod.named_parameters()]
            ts += [(f"{owner}:{n}", t) for n, t in mod.named_buffers()]
        opt = self.optimizer
        for i, p in enumerate(self.params):
            for k, v in opt.state.get(p, {}).items():
                if torch.is_tensor(v):
                    ts.append((f"opt.state[{i}].{k}", v))
        for gi, cb in enumerate(getattr(opt, "_cap", [])):
            ts += [(f"opt._cap[{gi}].{k}", v) for k, v in cb.items() if torch.is_tensor(v)]
        if torch.is_tensor(getattr(opt, "found_inf", None)):
            ts.append(("opt.found_inf", opt.found_inf))
        return [(n, t.data_ptr()) for n, t in ts]

    def _check_addresses(self) -> None:
        now = self._persistent()
        if now != self._addresses:
            a, b = dict(self._addresses), dict(now)
            moved = [n for n in a.keys() | b.keys() if a.get(n) != b.get(n)]
            raise RuntimeError(f"graph_step: {len(moved)} persistent tensor(s) were rebound "
                               f"or (de)allocated since capture; the graph would read the old "
                               f"storage. First: {sorted(moved)[:8]}")
        self._dirty = False

    def _python_state(self) -> dict:
        st = {}
        for owner, mod in [("", self.root)] + self._hidden_modules():
            for n, sub in mod.named_modules():
                for k, v in vars(sub).items():
                    if isinstance(v, (bool, int, float, str)) and not k.startswith("__"):
                        st[f"{owner}:{n}.{k}"] = repr(v)      # repr: a NaN equals itself
        return st

    @staticmethod
    def _dynamo_graphs() -> int:
        from torch._dynamo.utils import counters
        return int(counters["stats"]["unique_graphs"])

    @contextmanager
    def released(self):
        """Free the graphs and their pool for eager work that does not fit beside them (eval,
        generation); the next ``step`` records them again. The caller drops its own references
        to the graphs' outputs (``out``, ``loss``, the grad norm) first. Measured 2026-10-07 on the winner:
        with the 15.4 GiB pool held, eval's pointer mixture ran out of memory, in the default
        pool and also when it borrowed the pool's free blocks (fragmented: a 1.4 GiB request
        grew the pool instead). A re-recording executes no kernel and records the same
        sequence, so the training math does not change; it costs the recording time."""
        # The last replay may still be running: its graph and pool must outlive it (freeing
        # them under it was an illegal memory access, measured on the winner at step 25).
        torch.cuda.synchronize()
        # In a helper, never inline: a generator's locals live until it ends, and a loop
        # variable left bound to the last `_Captured` keeps its graphs, and so the whole pool,
        # alive across the yield (measured 2026-10-11: 18.5 GiB of free private-pool segments
        # that `empty_cache` could not return, and the eval's pointer mixture ran out of memory).
        _close_branches(self.graphs)
        self.graphs = {}
        self.static = {}
        self._addresses = None
        for p in self.params:                     # a replay's bound gradient buffers
            p.grad = None
        self._grad_bufs = None                    # the kept gradients, 1.2 GB on the winner
        self.mempool = torch.cuda.MemPool()       # the old pool dies with its last graph
        gc.collect()
        torch.cuda.empty_cache()
        try:
            yield
        finally:
            gc.collect()
            torch.cuda.empty_cache()              # the eager work's cache, for the recording
            self.stats.releases += 1

    def mark_dirty(self) -> None:
        """The trainer ran eager GPU work outside a step (eval, a checkpoint, generation):
        re-check every persistent address before the next replay."""
        self._dirty = True

    # ── the two ways to run a step ──────────────────────────────────────────────────────
    def step(self, step: int, kind: Any, x, y, layout, context: Any = None):
        """Run training step ``step`` of ``kind``. Returns ``(out, loss, gnorm, mode)``;
        mode is "eager", "capture" (every (kind, variant) recorded, then this step replayed)
        or "replay". The step's variant is drawn first, once, whichever way it runs.
        ``context`` is any hashable value the recording bakes in (the phase); a replay under a
        different one raises."""
        if kind not in self.kinds:
            raise ValueError(f"graph_step: kind {kind!r} is not one of {self.kinds}")
        v = self._next_variant()
        try:
            shape = self._shape(layout)
            mode = "replay"
            if not self.graphs:
                if step < self.first_capture or set(self.kinds) - self._warm:
                    self._warm.add(kind)
                    return (*self._eager(x, y, layout), "eager")
                self._capture_all(kind, v, shape, x, y, layout, context)
                mode = "capture"
            return (*self._replay(self.graphs[self._key(kind, v, shape)], kind, v, shape,
                                  x, y, layout, context), mode)
        finally:
            self._end_variant()

    def _capture_all(self, kind, variant, shape, x, y, layout, context):
        """Record every (kind, variant, shape)'s step on this step's batch (laid out at each
        shape by ``shape_batch``). Recording executes no kernel; the caller replays this step's
        (kind, variant, shape) right after."""
        from torch.compiler import set_stance
        t0 = time.perf_counter()
        # The eager warm-up steps leave their allocator cache behind (28 GB reserved on the
        # speed base); the recordings' pool and the driver's own allocations for the
        # instantiated graphs (cuGraphInstantiate) need that room. Values are untouched.
        gc.collect()
        torch.cuda.empty_cache()
        self.static = {}
        for sh in self.shapes:
            bx, by, bl = (x, y, layout) if sh == shape else self.shape_batch(sh, x, y, layout)
            self.static[sh] = StaticBatch(bx, by, bl)
            self.static[sh].load(bx, by, bl)
        # Lazy optimizer state must exist BEFORE a recording (its zero fill would otherwise
        # be recorded and replayed every step); every kind has stepped once, so it does.
        self.optimizer.init_state()
        if self._grad_bufs is None and any(k in self.keep_grads for k in self.kinds):
            self._grad_bufs = [torch.empty_like(p) for p in self.params]
        n_graphs = self._dynamo_graphs()
        cpu_rng = torch.get_rng_state()
        try:
            for k in self.kinds:
                self.set_kind(k)
                keep = k in self.keep_grads
                for var, sh in itertools.product(self.variants, self.shapes):
                    if self.set_variant is not None:
                        self.set_variant(var)
                    sb = self.static[sh]
                    py_before = self._python_state()
                    with set_stance("fail_on_recompile"):
                        rec = self._record(sb.x, sb.y, sb.layout, keep)
                    py_after = self._python_state()
                    changed = sorted(n for n in py_before.keys() | py_after.keys()
                                     if py_before.get(n) != py_after.get(n))
                    if changed:
                        raise RuntimeError(f"graph_step: a training step of kind {k!r} "
                                           f"(variant {var!r}, shape {sh!r}) changes module "
                                           f"attributes a replay would freeze: {changed[:12]}")
                    if not torch.equal(torch.get_rng_state(), cpu_rng):
                        raise RuntimeError("graph_step: the step consumes the CPU generator (a "
                                           "draw with no restore); a replay cannot reproduce "
                                           "that")
                    # A kept kind's gradients were copied into `_grad_bufs` by the recording;
                    # its own buffers, like every other kind's, go back to the pool so the
                    # next recording may reuse them.
                    present = [p.grad is not None for p in self.params] if keep else None
                    self.optimizer.zero_grad(set_to_none=True)
                    c_out, c_loss, c_gnorm = rec.pop("out"), rec.pop("loss"), rec.pop("gnorm")
                    self.graphs[self._key(k, var, sh)] = _Captured(
                        **rec, out=self._detached(c_out), loss=c_loss.detach(),
                        gnorm=c_gnorm.detach(), present=present, context=context)
                    del c_out, c_loss, c_gnorm, rec
                    self.stats.captures += 1
        finally:
            self.set_kind(kind)
            if self.set_variant is not None:
                self.set_variant(variant)
        if self.compiled and self._dynamo_graphs() != n_graphs:
            raise RuntimeError(f"graph_step: dynamo compiled {self._dynamo_graphs() - n_graphs} "
                               f"new graph(s) while recording")
        self._addresses = self._persistent()
        self._dirty = False
        n_br = sum(c.branch is not None for c in self.graphs.values())
        print(f"  [graph-step] recorded {len(self.graphs)} graph(s) (kinds {list(self.kinds)} x "
              f"{len(self.variants)} variant(s) x {len(self.shapes)} shape(s); {n_br} with the "
              f"hinge branch) in "
              f"{time.perf_counter() - t0:.1f} s; "
              f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved", flush=True)

    def _graph(self, keep_graph: bool = False):
        """A recording context on the capture stream, in the graphs' pool."""
        g = torch.cuda.CUDAGraph(keep_graph=keep_graph)
        return g, torch.cuda.graph(g, pool=self.mempool.id, stream=self.stream,
                                   capture_error_mode="thread_local")

    def _record(self, sx, sy, sl, keep: bool) -> dict:
        """Record one (kind, variant): ``pre`` and either ``rest`` or the hinge branch and
        ``post`` (module notes, THE HINGE BRANCH). Returns the ``_Captured`` fields.

        ``rebind``: every tensor attribute (or dict of tensors) of the model that the step
        set to a new object while recording, e.g. ``_loop_cot_ref`` (the cotangent-clip
        hook). The replay writes into these objects, so a replay binds them again: a reader
        after the step then sees this replay's values, not those of whichever step (another
        variant's recording, an eager step) set the attribute last."""
        owners = self._attr_owners()
        before = _attr_objs(owners)
        rec = self._record_graphs(sx, sy, sl, keep)
        after = _attr_objs(owners)
        rec["rebind"] = [(m, k, v) for key, (m, k, v) in after.items()
                         if key not in before or not _same_obj(before[key][2], v)]
        return rec

    def _record_graphs(self, sx, sy, sl, keep: bool) -> dict:
        g_pre, ctx = self._graph()
        with ctx:
            out, loss = self._forward_loss(sx, sy, sl)
            l1 = loss / 1
        pen, flag = out.get("gain_pen"), out.get("gain_active")
        if pen is None or flag is None or not pen.requires_grad:
            g_rest, ctx = self._graph()
            with ctx:
                l1.backward()
                gnorm = self._post(keep)
            return dict(graph=g_pre, rest=g_rest, branch=None, post=None, bodies=(),
                        out=out, loss=loss, gnorm=gnorm)
        roots = _cut_roots(l1, pen)
        # A body launched by the branch does not run torch's replay prologue (which sets a
        # graph's philox offsets), so it must draw nothing. The generator cannot be read
        # during a capture; the eager steps before the recording measured the backward.
        if self._eager_bwd_draws is not False:
            raise RuntimeError("graph_step: the hinge branch needs an eager step's backward "
                               "that draws no random numbers before the recording "
                               f"(measured: {self._eager_bwd_draws}); its bodies would reuse "
                               "the forward graph's philox offsets (graph_cond.py)")
        g_a, ctx = self._graph(keep_graph=True)
        with ctx:
            l1.backward(retain_graph=True)
        grads = [p.grad for p in self.params]
        attrs = _TensorAttrs(self._attr_owners())
        g_b, ctx = self._graph(keep_graph=True)
        with ctx:
            live = [g for g in grads if g is not None]
            if live:
                torch._foreach_zero_(live)
            torch.autograd.backward([e for e in roots],
                                    grad_tensors=[torch.ones((), dtype=l1.dtype,
                                                             device=l1.device)] * len(roots))
            attrs.restore()
        moved = [i for i, (p, g) in enumerate(zip(self.params, grads)) if p.grad is not g]
        if moved:
            raise RuntimeError(f"graph_step: the hinge branch's second body rebound "
                               f"{len(moved)} gradient(s) (first: parameter {moved[0]}); "
                               f"the optimizer would read the first body's buffers only")
        del l1
        branch = CondBranch(flag, g_a, g_b)
        g_post, ctx = self._graph()
        try:
            with ctx:
                gnorm = self._post(keep)
        except Exception:
            branch.close()
            raise
        return dict(graph=g_pre, rest=None, branch=branch, post=g_post, bodies=(g_a, g_b),
                    out=out, loss=loss, gnorm=gnorm)

    def _attr_owners(self) -> list:
        return [self.root] + [m for _n, m in self._hidden_modules()]

    def _replay(self, cap: _Captured, kind, variant, shape, x, y, layout, context):
        if context != cap.context:
            raise RuntimeError(f"graph_step: kind {kind!r} was captured under {cap.context!r}, "
                               f"this step runs under {context!r}")
        if self._dirty:
            self._check_addresses()
        self.static[shape].load(x, y, layout)
        cap.graph.replay()
        if cap.branch is None:
            cap.rest.replay()
        else:
            cap.branch.launch(self._launch_stream())
            cap.post.replay()
        for m, k, v in cap.rebind:
            setattr(m, k, v if torch.is_tensor(v) else dict(v))
        cap.replays += 1
        self.stats.replayed += 1
        if cap.present is not None:
            for p, b, has in zip(self.params, self._grad_bufs, cap.present):
                p.grad = b if has else None
        return cap.out, cap.loss, cap.gnorm

    @staticmethod
    def _launch_stream():
        """The stream a torch replay launches on (the caller's current stream)."""
        return torch.cuda.current_stream()

    def _side(self, x, y, layout):
        """The body, eager, on the capture stream. EVERY eager step runs there, not on the
        default stream: autograd caches one AccumulateGrad node per parameter, tagged with
        the stream it was created on, and keeps it while any earlier step's graph is still
        referenced (the trainer's `out`, a model stash). A node from a default-stream step
        makes the captured backward join the legacy stream: cudaErrorStreamCaptureImplicit,
        measured on the tiny winner 2026-10-07."""
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)
        with torch.cuda.stream(self.stream):
            res = self._body(x, y, layout)
        cur.wait_stream(self.stream)
        return res

    def run_eager(self, x, y, layout):
        """An eager training step (the test's step between replays). Draws the step's
        variant like any other step. Leaves every address a replay reads where it was;
        checked at the next replay."""
        self._next_variant()
        try:
            return self._eager(x, y, layout)
        finally:
            self._end_variant()

    def _eager(self, x, y, layout):
        """The eager step under the variant already set (the warm-up, ``run_eager``)."""
        self._dirty = True
        self.stats.eager += 1
        return self._side(x, y, layout)
