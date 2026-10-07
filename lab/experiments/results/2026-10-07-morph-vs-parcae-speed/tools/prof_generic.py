"""torch.profiler window over training steps of either trainer (2026-10-07 MORPH-vs-Parcae round).

  PROF_TRAINER=morph|parcae, PROF_START (optimizer steps before the window, default 30),
  PROF_N (steps in the window, default 4), PROF_OUT (chrome trace path, .json).
  PROF_RANGES=1 on morph imports prof_patch (R:: ranges around MORPH methods); on parcae it
  wraps the Parcae-LXTUL model methods listed in PARCAE_RANGES.
The window opens at the first forward after PROF_START optimizer steps and closes after the
PROF_N-th optimizer step inside it (wall clock and GPU work of whole steps), then exits.
Also prints the wall time per step over the window measured with cuda sync at both ends.
"""
import functools
import os
import sys
import time

import torch

START = int(os.environ.get("PROF_START", "30"))
N = int(os.environ.get("PROF_N", "4"))
OUT = os.environ["PROF_OUT"]
st = {"opt": 0, "prof": None, "n": 0, "t0": 0.0}


def enter():
    if st["prof"] is None and st["opt"] >= START:
        torch.cuda.synchronize()
        p = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=bool(os.environ.get("PROF_SHAPES")))
        p.__enter__()
        st["prof"] = p
        st["t0"] = time.perf_counter()


def wrap_opt(cls):
    if getattr(cls, "_pg_wrapped", False):
        return
    orig = cls.step

    @functools.wraps(orig)
    def step(self, *a, **k):
        if st["prof"] is not None:
            with torch.profiler.record_function("R::optimizer.step"):
                r = orig(self, *a, **k)
        else:
            r = orig(self, *a, **k)
        st["opt"] += 1
        if st["prof"] is not None:
            st["n"] += 1
            if st["n"] == N:
                torch.cuda.synchronize()
                dt = time.perf_counter() - st["t0"]
                st["prof"].__exit__(None, None, None)
                st["prof"].export_chrome_trace(OUT)
                print(f"[prof] {N} steps, profiled wall {1e3*dt/N:.1f} ms/step -> {OUT}", flush=True)
                sys.stdout.flush()
                os._exit(0)
        return r
    cls.step = step
    cls._pg_wrapped = True


def rf(obj, name, label):
    fn = getattr(obj, name)

    @functools.wraps(fn)
    def w(*a, **k):
        with torch.profiler.record_function(label):
            return fn(*a, **k)
    setattr(obj, name, w)


trainer = os.environ.get("PROF_TRAINER", "morph")
if trainer == "morph":
    if os.environ.get("PROF_RANGES"):
        sys.path.insert(0, os.path.dirname(__file__))
        import prof_patch  # noqa: F401  (wraps MORPH methods with R:: ranges on import)
    from morph.model import transformer as T
    from morph.training import train as TR
    _fwd = T.MORPHTransformer.forward
    _sub = bool(os.environ.get("MORPH_SUB_RANGES"))
    if _sub:
        from morph.model import attention as MA, hyper_connections as HCm, mhc as MH
        rf(MA.MORPHAttention, "forward", "R::attention")
        rf(MA.RMSNorm, "forward", "R::norm")
        from morph.model.layers import norms as NM
        rf(NM.RMSNorm, "forward", "R::norm")
        _hcf = HCm.HyperConnectionResidual.forward

        def _hc(self, *a, **k):
            with torch.profiler.record_function(getattr(self, "_plabel", "R::hc")):
                return _hcf(self, *a, **k)
        HCm.HyperConnectionResidual.forward = _hc

    @functools.wraps(_fwd)
    def fwd(self, *a, **k):
        if _sub and not getattr(self, "_plabeled", False):
            for m in self.modules():
                if isinstance(m, MH.MORPHBlock):
                    m.mrr_attn._plabel, m.mrr_mlp._plabel = "R::hc_attn", "R::hc_mlp"
            self._plabeled = True
        if torch.is_grad_enabled():
            enter()
        return _fwd(self, *a, **k)
    T.MORPHTransformer.forward = fwd
    _oc = TR.create_optimizer

    def _create(*a, **k):
        o = _oc(*a, **k)
        wrap_opt(type(o))
        return o
    TR.create_optimizer = _create
    main = TR.main
else:
    from lxtul import train as PT
    _f = PT._forward

    def _forward(*a, **k):
        enter()
        with torch.profiler.record_function("R::forward"):
            return _f(*a, **k)
    PT._forward = _forward
    _mk = PT.make_optimizer

    def make_optimizer(*a, **k):
        o = _mk(*a, **k)
        wrap_opt(type(o))
        return o
    PT.make_optimizer = make_optimizer
    if os.environ.get("PROF_RANGES"):
        from lxtul import model as LM
        for spec in os.environ.get("PARCAE_RANGES", "").split(","):
            if spec:
                cls_name, meth = spec.split(".")
                rf(getattr(LM, cls_name), meth, f"R::{cls_name}.{meth}")
    if os.environ.get("PARCAE_BLOCK_RANGES"):
        import torch.nn as nn

        class Ranged(nn.Module):
            def __init__(self, inner, label):
                super().__init__()
                self.inner, self.label = inner, label

            def forward(self, *a, **k):
                with torch.profiler.record_function(self.label):
                    return self.inner(*a, **k)
        _build = PT.build

        def build(*a, **k):
            model, mcfg, rt = _build(*a, **k)
            t = model.transformer
            groups = [(t, "prelude"), (t, "core_block"), (t, "coda"), (model, "twin_prelude"),
                      (model, "sd_blocks")]
            for owner, name in groups:
                ml = getattr(owner, name, None) if hasattr(owner, name) or name in getattr(owner, "_modules", {}) else None
                if ml is None:
                    continue
                setattr(owner, name, nn.ModuleList(Ranged(b, f"R::blocks.{name}") for b in ml))
            return model, mcfg, rt
        PT.build = build
    main = PT.main

if __name__ == "__main__":
    main()
