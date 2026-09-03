"""``dmorph:`` accepts exactly its known keys and resolves its derived numbers
(morph/training/dmorph_setup.py); the shipped dmorph configs compose."""

from __future__ import annotations

import glob
import math

import pytest
from omegaconf import OmegaConf

from morph.training.dmorph_setup import (KNOWN_DMORPH_KEYS, build_dmorph_runtime,
                                         reject_unknown_dmorph_keys)


class _Tul:
    pass


def _cfg(n_core=0, **dm):
    return OmegaConf.create({
        "model": {"d_model": 1024, "n_prelude": 6, "n_core": n_core, "n_coda": 6},
        "dmorph": {"enabled": True, **dm},
    })


@pytest.mark.parametrize("bad", ["stp_lambda", "xattn", "bcast", "bridge", "objective", "arm_"])
def test_unknown_dmorph_key_raises(bad):
    with pytest.raises(ValueError, match=bad):
        build_dmorph_runtime(_cfg(**{bad: True}), _Tul())


def test_disabled_returns_none_and_known_keys_pass():
    reject_unknown_dmorph_keys({k: 0 for k in KNOWN_DMORPH_KEYS})
    c = _cfg()
    c.dmorph.enabled = False
    assert build_dmorph_runtime(c, _Tul()) is None
    assert build_dmorph_runtime(OmegaConf.create({"model": {"n_core": 0}}), None) is None


def test_requires_tul_and_a_flat_stack_that_the_blocks_divide():
    with pytest.raises(ValueError, match="TUL active"):
        build_dmorph_runtime(_cfg(), None)
    with pytest.raises(ValueError, match="n_core=0"):
        build_dmorph_runtime(_cfg(n_core=6), _Tul())
    with pytest.raises(ValueError, match="must divide"):
        build_dmorph_runtime(_cfg(n_blocks=5), _Tul())
    with pytest.raises(ValueError, match="arm"):
        build_dmorph_runtime(_cfg(arm="edm"), _Tul())


def test_matched_and_auto_resolve_from_d_model_and_land_in_the_manifest():
    rt = build_dmorph_runtime(_cfg(source_std="matched", in_gain="auto"), _Tul())
    assert rt.model_cfg.source_std == pytest.approx(1.0 / math.sqrt(1024))
    assert rt.model_cfg.in_gain == pytest.approx(32.0)
    assert rt.manifest["source_std_note"] == "MATCHED"
    assert rt.manifest["null_floor"] == pytest.approx(2.0)
    assert rt.manifest["layers_per_block"] == 3
    assert rt.manifest["target_detached"] is True
    rt2 = build_dmorph_runtime(_cfg(source_std=1.0, arm="hs", sigreg_lambda=0.02), _Tul())
    assert rt2.manifest["source_std_note"] == "MISMATCHED"
    assert rt2.manifest["target_detached"] is False


def test_the_shipped_dmorph_configs_use_only_known_keys_and_compose():
    from hydra import compose, initialize_config_dir
    import os
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "morph", "configs")
    seen = 0
    for path in sorted(glob.glob(os.path.join(root, "*.yaml"))):
        raw = OmegaConf.load(path)
        dc = raw.get("dmorph", None)
        if dc is None or not hasattr(dc, "keys"):
            continue
        reject_unknown_dmorph_keys(dc)
        seen += 1
    assert seen >= 5   # base + ctl + tok + hs + smoke
    with initialize_config_dir(version_base=None, config_dir=root):
        for name, enabled, arm in (("dmorph_ctl", False, "tok"), ("dmorph_tok", True, "tok"),
                                   ("dmorph_hs", True, "hs"), ("dmorph_smoke", True, "tok")):
            cfg = compose(config_name=name)
            assert bool(cfg.dmorph.enabled) is enabled, name
            assert str(cfg.dmorph.arm) == arm, name
            assert int(cfg.model.n_core) == 0, name
            assert (int(cfg.model.n_prelude) + int(cfg.model.n_coda)) % int(cfg.dmorph.n_blocks) == 0
            assert str(cfg.tul.activate_at) == "0.0", name
            assert int(cfg.training.ademamix_t_beta3) > 0, "pin the beta3 horizon (rule 5)"
