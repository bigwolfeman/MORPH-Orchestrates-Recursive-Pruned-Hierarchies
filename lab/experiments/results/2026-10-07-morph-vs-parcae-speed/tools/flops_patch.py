"""Count matmul/attention FLOPs of ONE training step (forward + backward + recompute), by module.

Scratch tool for the 2026-10-07 MORPH-vs-Parcae speed round. Works on either trainer:
  FLOPS_TRAINER=morph   -> wraps MORPHTransformer.forward and the optimizer step, runs morph.training.train
  FLOPS_TRAINER=parcae  -> wraps lxtul.train._forward and the optimizer step, runs lxtul.train
Env: FLOPS_STEP (optimizer steps to skip first, default 3), FLOPS_OUT (json path).
The counter enters at the first model forward after FLOPS_STEP optimizer steps and exits at the
next optimizer step, then the process exits (os._exit) so no further steps run.
Counts NSTEPS consecutive steps INCLUDING their optimizer steps (Muon's Newton-Schulz is matmul work).
Run with compile OFF: a dispatch mode does not see inside inductor kernels.
"""
import functools
import json
import os
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

STEP = int(os.environ.get("FLOPS_STEP", "3"))
NSTEPS = int(os.environ.get("FLOPS_NSTEPS", "1"))
OUT = os.environ["FLOPS_OUT"]
state = {"opt_steps": 0, "mode": None, "done": False, "counted": 0, "opt_flops": 0}


def enter():
    if state["mode"] is None and state["opt_steps"] >= STEP and not state["done"]:
        torch.cuda.synchronize()
        state["mode"] = FlopCounterMode(display=False, depth=None)
        state["mode"].__enter__()


def exit_and_dump():
    m = state["mode"]
    torch.cuda.synchronize()
    m.__exit__(None, None, None)
    counts = m.get_flop_counts()
    out = {"total": m.get_total_flops(), "nsteps": NSTEPS, "optimizer": state["opt_flops"],
           "by_module": {k: {str(op): int(v) for op, v in d.items()} for k, d in counts.items()}}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[flops] total {out['total']/1e12:.3f} TFLOP/step -> {OUT}", flush=True)
    sys.stdout.flush()
    os._exit(0)


def wrap_opt(cls):
    if getattr(cls, "_flops_wrapped", False):
        return
    orig = cls.step

    @functools.wraps(orig)
    def step(self, *a, **k):
        f0 = state["mode"].get_total_flops() if state["mode"] is not None else 0
        r = orig(self, *a, **k)
        if state["mode"] is not None:
            state["opt_flops"] += state["mode"].get_total_flops() - f0
        state["opt_steps"] += 1
        if state["mode"] is not None:
            state["counted"] += 1
            if state["counted"] == NSTEPS:
                exit_and_dump()
        return r
    cls.step = step
    cls._flops_wrapped = True


trainer = os.environ.get("FLOPS_TRAINER", "morph")
if trainer == "morph":
    from morph.model import transformer as T
    from morph.training import train as TR
    _fwd = T.MORPHTransformer.forward

    @functools.wraps(_fwd)
    def fwd(self, *a, **k):
        if torch.is_grad_enabled():
            enter()
        return _fwd(self, *a, **k)
    T.MORPHTransformer.forward = fwd
    _orig_create = TR.create_optimizer

    def _create(*a, **k):
        opt = _orig_create(*a, **k)
        wrap_opt(type(opt))
        return opt
    TR.create_optimizer = _create
    if __name__ == "__main__":
        TR.main()
else:
    from lxtul import train as PT
    from lxtul import attention as LA
    import torch.nn.functional as F

    def _dense_flex(q, k, v, block_mask=None, **kw):
        # FlopCounterMode cannot dispatch flex_attention: count a dense causal SDPA of the same
        # shapes instead (an UPPER bound on the flex kernel's FLOPs; values differ, shapes do not).
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    LA._flex = _dense_flex
    _f = PT._forward

    def _forward(*a, **k):
        enter()
        return _f(*a, **k)
    PT._forward = _forward
    _mk = PT.make_optimizer

    def make_optimizer(*a, **k):
        opt = _mk(*a, **k)
        wrap_opt(type(opt))
        return opt
    PT.make_optimizer = make_optimizer
    if __name__ == "__main__":
        PT.main()
