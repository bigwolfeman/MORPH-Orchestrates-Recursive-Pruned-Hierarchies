"""The ONE model build the offline divergence probes share.

`jac_ladder.py` and `optstate_probe.py` both need a MORPHTransformer whose parameter
names and shapes match a checkpoint exactly. They used to each carry their own copy of
this; the copy in `jac_ladder.py` is the one that shipped the QAT comment below, and the
first version of that script proved the comment by loading no MLP weight at all. One
copy, one place to fix.
"""
from __future__ import annotations

import os
import sys

import torch
from hydra import compose, initialize_config_dir

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from morph.model.transformer import MORPHTransformer          # noqa: E402
from morph.training.fm_setup import build_fm_runtime          # noqa: E402
from morph.training.quant_setup import apply_quantization     # noqa: E402
from morph.training.train import build_morph_config           # noqa: E402
from morph.training.tul_setup import build_tul_runtime        # noqa: E402

__all__ = ["build_cfg", "build_model", "DepthLever", "ROOT"]

ROOT = _ROOT


def build_cfg(config_name: str, overrides: list[str]):
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.join(_ROOT, "morph", "configs")):
        return compose(config_name=config_name, overrides=overrides)


def build_model(cfg, device: str = "cuda"):
    """Model + tul runtime, quantised, on `device`. NOT loaded from any checkpoint.

    QAT BEFORE any load. The core MLP's ternary STE is registered as a weight
    PARAMETRIZATION, so an unquantised model's key is `..._cms.weight` while the
    checkpoint's is `..._cms.parametrizations.weight.original`. Skipping this and loading
    with strict=False drops every MLP tensor in silence and leaves them at random init —
    measured, on the first version of `jac_ladder.py`: the cap sweep reported that NO core
    linear exceeded 2.0 while the run's own log had sigma_max at 3.30.
    """
    tul_rt = build_tul_runtime(cfg)
    fm_rt = build_fm_runtime(cfg, tul_rt)
    model = MORPHTransformer(build_morph_config(cfg, tul=tul_rt.model_cfg if tul_rt else None,
                                                fm=fm_rt))
    model = model.to(torch.device(device))
    apply_quantization(model, cfg)
    return model, tul_rt


class DepthLever:
    """The ONE eval-time forced-depth knob, chosen by arm.

    Two arms, two knobs, and a probe that sets the wrong one measures nothing while
    printing a full table:

    * slot-loop arms (A0/A1/A3, l2*) loop the core on SLOT positions; eval depth is
      ``model.cfg.tul.slot_mean_depth`` (with ``slot_max_depth`` raised to match).
    * A2 (``tul.tokens_through_core``) runs the ordinary per-sample core over the whole
      packed row; eval depth is ``model.cfg.mean_depth`` (the ``else`` of
      ``_core_region``'s ``if self.training``), and the slot knobs are inert.

    ``a2_depth_sweep.py`` and ``future_leak_probe.py`` both mutate the knob in place
    and restore it; this is that logic in one home.
    """

    def __init__(self, model, tul_rt, fallback_max_depth: int):
        self.model = model
        self.a2 = bool(tul_rt is not None and tul_rt.model_cfg.tokens_through_core)
        tc = model.cfg.tul
        if self.a2:
            self.name = "model.cfg.mean_depth"
            self._orig = (int(model.cfg.mean_depth),)
        else:
            if tc is None:
                raise ValueError(
                    "DepthLever: this model has no TUL config, so there is no slot loop to "
                    "force a depth on. A notul arm's lever is model.cfg.mean_depth via "
                    "token_depth_sweep.py; this helper serves the packed-row probes only.")
            self.name = "model.cfg.tul.slot_mean_depth"
            self._orig = (int(tc.slot_mean_depth), int(tc.slot_max_depth))
        self._fallback_max = int(fallback_max_depth)

    def set(self, depth: int) -> None:
        if self.a2:
            self.model.cfg.mean_depth = int(depth)
        else:
            tc = self.model.cfg.tul
            tc.slot_mean_depth = int(depth)
            tc.slot_max_depth = max(int(depth), self._orig[1] or self._fallback_max)

    def restore(self) -> None:
        if self.a2:
            self.model.cfg.mean_depth = self._orig[0]
        else:
            tc = self.model.cfg.tul
            tc.slot_mean_depth, tc.slot_max_depth = self._orig
