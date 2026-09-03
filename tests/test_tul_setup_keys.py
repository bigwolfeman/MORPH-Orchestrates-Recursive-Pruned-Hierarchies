"""`tul:` accepts exactly its known keys (runtime-invariants §6b).

A retired arm key in an old override file (`tul.gate: true`, `tul.tg_restrict: true`,
`tul.coda_sees_slots: false`, …) must never run the shipped paid loop under a name that
promises something else. The check runs BEFORE the tokenizer is touched, so it is cheap and
CPU-only.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from morph.training.tul_setup import (
    KNOWN_TUL_KEYS,
    build_boundary_rule,
    build_tul_runtime,
    reject_unknown_tul_keys,
)


def _cfg(**tul):
    return OmegaConf.create({"model": {"vocab_size": 64}, "data": {"tokenizer": "x", "seq_len": 32},
                             "tul": tul})


@pytest.mark.parametrize("bad", ["gate", "tg_restrict", "coda_sees_slots", "tokens_through_core",
                                 "per_slot_embed", "fixed_stride", "stp_lambda", "slot_seeed"])
def test_unknown_tul_config_key_raises(bad):
    with pytest.raises(ValueError, match=bad):
        build_tul_runtime(_cfg(activate_at=0.0, **{bad: True}))
    with pytest.raises(ValueError, match=bad):
        build_boundary_rule(_cfg(**{bad: True}))


def test_known_keys_pass_the_check_and_never_still_returns_none():
    reject_unknown_tul_keys({k: 0 for k in KNOWN_TUL_KEYS})   # must not raise
    assert build_tul_runtime(_cfg(activate_at="never", prefix_k=2)) is None


def test_the_shipped_configs_use_only_known_keys():
    """Every `tul:` block in morph/configs must compose against the known set."""
    import glob
    for path in sorted(glob.glob("morph/configs/*.yaml")):
        raw = OmegaConf.load(path)
        tc = raw.get("tul", None)
        if tc is None or not hasattr(tc, "keys"):
            continue
        reject_unknown_tul_keys(tc)
