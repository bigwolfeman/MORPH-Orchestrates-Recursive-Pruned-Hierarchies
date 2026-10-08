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
    references every step, as today).
  * Per-step scalars on the device (``training.capturable_optimizer``):
    ``optimizer.write_step_scalars()`` runs before the replay, outside the graph.
  * One op sequence and fixed shapes (``model.graph_safe``: fixed pass count, no host read).
  * Randomness: the CUDA generator is registered with the graph (philox seed and offset are
    read at replay and the offset advances by the graph's increment), so a replay draws what
    an eager step at that point would. The gain hinge's pass is drawn on the HOST from the CPU
    generator, which the draw saves and restores, so it is the same pass while the CPU
    generator stays where it was at capture: compared before every replay, raises if not.
  * Compiled frames: the capture runs under the ``fail_on_recompile`` stance, so a guard miss
    raises instead of silently recording an eager fallback (the trainer's
    ``eager_on_recompile`` stance would), and the dynamo graph count must not move.
  * Python state: a replay runs no Python, so a module attribute that a step changes (a
    counter, a schedule flag) would freeze. Every scalar attribute of every module is
    compared across each recording; a change raises.

Order: every step runs eager, on the capture side stream (the PyTorch recipe's warm-up),
until ``first_capture`` and until every kind has run once. On the next step every kind is
recorded on that step's batch, then that step's kind is REPLAYED (a recording executes no
kernel). From there on every step is a replay.

Eval and generation do not fit beside the pool either (measured: eval's pointer mixture OOM
at step 25 with the 15.4 GiB pool held, also when it borrowed the pool's free blocks). The
trainer runs them inside ``released()``: the graphs and their pool are freed (after a device
sync), and the next step records both kinds again on its own batch and replays. A recording
executes no kernel and records the same op sequence, so this costs recording time only; the
45-step gate trace is byte-identical either way.
"""
from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from morph.model.tul_layout import SlotLayout

__all__ = ["GraphStep", "StaticBatch", "graph_step_refusals"]

# SlotLayout's tensor fields, in declaration order. A field added to SlotLayout that is not
# listed here raises in StaticBatch rather than being silently left at its capture value.
_LAYOUT_TENSORS = ("slot_mask", "bag_id", "slot_index", "slot_valid", "span_len",
                   "len_supervised", "ditto_prev")
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


@dataclass
class _Captured:
    """One step kind's graph and the tensors it writes."""

    graph: torch.cuda.CUDAGraph
    out: dict
    loss: torch.Tensor
    gnorm: torch.Tensor
    grads: Optional[list]             # the graph's own gradient buffers, when kept
    cpu_rng: torch.Tensor             # CPU generator state at capture (the hinge's draw)
    context: Any                      # the trainer's per-capture key (the phase)
    replays: int = 0


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
    ``keep_grads``: kinds whose gradient buffers stay referenced, so a replay can bind them to
    ``p.grad`` (the log block's per-region gradient norms read them)."""

    def __init__(self, model: nn.Module, optimizer, *, grad_clip: float,
                 forward: Callable[[torch.Tensor, torch.Tensor, Optional[SlotLayout]], dict],
                 kinds: tuple, set_kind: Callable[[Any], None],
                 loss_terms: Optional[Callable[[], Optional[torch.Tensor]]] = None,
                 after_step: Optional[Callable[[], None]] = None,
                 first_capture: int = 3, keep_grads: tuple = (True,),
                 compiled: bool = True):
        if not getattr(optimizer, "capturable", False):
            raise ValueError("GraphStep needs a capturable optimizer "
                             "(training.capturable_optimizer)")
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
        self.stream = torch.cuda.Stream()
        # The graphs' private pool (a fresh one after every `released`).
        self.mempool = torch.cuda.MemPool()
        self.static: Optional[StaticBatch] = None
        self.graphs: dict[Any, _Captured] = {}
        self.stats = GraphStepStats()
        self._warm: set = set()               # kinds that have run eager on the stream
        self._addresses: Optional[list] = None
        self._dirty = True                    # eager work since the last address check

    # ── the step body: the trainer's eager sequence (train.py, fwd .. opt) ──────────────
    def _body(self, x, y, layout):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.forward(x, y, layout)
            loss = out["loss"]
            if self.loss_terms is not None:
                extra = self.loss_terms()
                if extra is not None:
                    loss = loss + extra.to(loss.dtype)
        # `/ 1`: the trainer's `scaler.scale(loss / _ga)` at grad_accum 1 with the disabled
        # scaler, kept so the backward starts from the same node.
        (loss / 1).backward()
        gnorm = nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.mark_found_inf(gnorm)
        self.optimizer.step()
        if self.after_step is not None:
            self.after_step()
        return out, loss, gnorm

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
        self.graphs = {}
        self.static = None
        self._addresses = None
        for p in self.params:                     # a replay's bound gradient buffers
            p.grad = None
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
        mode is "eager", "capture" (every kind recorded, then this step replayed) or
        "replay". ``context`` is any hashable value the recording bakes in (the phase); a
        replay under a different one raises."""
        if kind not in self.kinds:
            raise ValueError(f"graph_step: kind {kind!r} is not one of {self.kinds}")
        mode = "replay"
        if not self.graphs:
            if step < self.first_capture or set(self.kinds) - self._warm:
                self._warm.add(kind)
                return (*self.run_eager(x, y, layout), "eager")
            self._capture_all(kind, x, y, layout, context)
            mode = "capture"
        return (*self._replay(self.graphs[kind], kind, x, y, layout, context), mode)

    def _capture_all(self, kind, x, y, layout, context):
        """Record every kind's step on this step's batch. Recording executes no kernel; the
        caller replays this step's kind right after."""
        from torch.compiler import set_stance
        self.static = StaticBatch(x, y, layout)
        self.static.load(x, y, layout)
        sx, sy, sl = self.static.x, self.static.y, self.static.layout
        # Lazy optimizer state must exist BEFORE a recording (its zero fill would otherwise
        # be recorded and replayed every step); every kind has stepped once, so it does.
        self.optimizer.init_state()
        n_graphs = self._dynamo_graphs()
        cpu_rng = torch.get_rng_state()
        try:
            for k in self.kinds:
                self.set_kind(k)
                py_before = self._python_state()
                g = torch.cuda.CUDAGraph()
                with set_stance("fail_on_recompile"):
                    with torch.cuda.graph(g, pool=self.mempool.id, stream=self.stream,
                                          capture_error_mode="thread_local"):
                        c_out, c_loss, c_gnorm = self._body(sx, sy, sl)
                py_after = self._python_state()
                changed = sorted(n for n in py_before.keys() | py_after.keys()
                                 if py_before.get(n) != py_after.get(n))
                if changed:
                    raise RuntimeError(f"graph_step: a training step of kind {k!r} changes "
                                       f"module attributes a replay would freeze: "
                                       f"{changed[:12]}")
                if not torch.equal(torch.get_rng_state(), cpu_rng):
                    raise RuntimeError("graph_step: the step consumes the CPU generator (a draw "
                                       "with no restore); a replay cannot reproduce that")
                # The kept kinds' gradient buffers; every other kind's are released to the
                # pool, so the next recording may reuse them.
                grads = [p.grad for p in self.params] if k in self.keep_grads else None
                self.optimizer.zero_grad(set_to_none=True)
                self.graphs[k] = _Captured(graph=g, out=self._detached(c_out),
                                           loss=c_loss.detach(), gnorm=c_gnorm.detach(),
                                           grads=grads, cpu_rng=cpu_rng, context=context)
                del c_out, c_loss, c_gnorm
                self.stats.captures += 1
        finally:
            self.set_kind(kind)
        if self.compiled and self._dynamo_graphs() != n_graphs:
            raise RuntimeError(f"graph_step: dynamo compiled {self._dynamo_graphs() - n_graphs} "
                               f"new graph(s) while recording")
        self._addresses = self._persistent()
        self._dirty = False
        print(f"  [graph-step] recorded {len(self.graphs)} graph(s) (kinds {list(self.kinds)}); "
              f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved", flush=True)

    def _replay(self, cap: _Captured, kind, x, y, layout, context):
        if context != cap.context:
            raise RuntimeError(f"graph_step: kind {kind!r} was captured under {cap.context!r}, "
                               f"this step runs under {context!r}")
        if not torch.equal(torch.get_rng_state(), cap.cpu_rng):
            raise RuntimeError("graph_step: the CPU generator moved since capture, so the gain "
                               "hinge's host draw may pick another pass than the recorded one")
        if self._dirty:
            self._check_addresses()
        self.static.load(x, y, layout)
        cap.graph.replay()
        cap.replays += 1
        self.stats.replayed += 1
        if cap.grads is not None:
            for p, gr in zip(self.params, cap.grads):
                p.grad = gr
        return cap.out, cap.loss, cap.gnorm

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
        """An eager training step (the warm-up, and the test's step between replays).
        Leaves every address a replay reads where it was; checked at the next replay."""
        self._dirty = True
        self.stats.eager += 1
        return self._side(x, y, layout)
