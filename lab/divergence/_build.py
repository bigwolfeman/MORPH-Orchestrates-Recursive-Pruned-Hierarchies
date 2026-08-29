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

__all__ = ["build_cfg", "build_model", "ROOT"]

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
