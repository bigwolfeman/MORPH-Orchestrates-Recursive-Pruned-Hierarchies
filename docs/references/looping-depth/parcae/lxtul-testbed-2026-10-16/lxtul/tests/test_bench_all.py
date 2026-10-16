"""CPU tests of the bench4 harness (lxtul/bench_all.py, summarize_bench4.py): the arm table composes, the
speed-key arms use only real LXTULConfig keys and construct, batch 12 stacks two cached batches, and the
summary table reads a folder of results.

    PYTHONPATH=.:$MORPH_ROOT python -m pytest -q lxtul/tests/test_bench_all.py
"""
from __future__ import annotations

import dataclasses
import json

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import lxtul.bench_all as ba
from lxtul.model import LXTULConfig
from lxtul.tests.test_compact import _batch, _build, cpu_reference_kernels  # noqa: F401  (autouse fixture)


@pytest.mark.parametrize("arm", list(ba.ARMS))
def test_every_arm_composes_with_real_keys(arm):
    hydra_arm, extra = ba.ARMS[arm]
    with initialize_config_dir(ba.CONF_DIR, version_base=None):
        cfg = compose("train", overrides=[f"arm={hydra_arm}", "run.name=t", *extra])
    fields = {f.name for f in dataclasses.fields(LXTULConfig)}
    if "lxtul" in cfg.arm:
        assert set(OmegaConf.to_container(cfg.arm.lxtul)) <= fields
    assert cfg.arm.kind in ("gpt", "plain", "lxtul", "spiral")
    for o in extra:                                    # every override changed a real key
        key, val = o.split("=")
        assert OmegaConf.select(cfg, key) == (val == "true")


def test_speed_key_arms_construct_and_run_a_forward():
    from omegaconf import OmegaConf as OC
    for name in ("lxtul_pointer_fast", "lxtul_pointer_fast_nockpt"):
        keys = OC.to_container(OC.load(f"{ba.CONF_DIR}/arm/{name}.yaml").lxtul, resolve=True)
        m = _build(**keys)
        assert m.tul.compact_slots and m.tul.static_depth and m.tul.cot_clip_fn and m.tul.ce_rows
        assert m.tul.spandec_packed and m.tul.pointer_heads == 4
        x, y, lay = _batch(2)
        out = m(x, y, lay)
        assert torch.isfinite(out["loss"])


def test_cat_layout_stacks_rows():
    a, b = _batch(2), _batch(3)
    la = a[2]
    lb = b[2]
    lay = ba._cat_layout(la, lb)
    for f in ba.CELL_FIELDS:
        assert torch.equal(getattr(lay, f), torch.cat([getattr(la, f), getattr(lb, f)]))
    assert lay.prefix_k == la.prefix_k and lay.slot_mask.shape[0] == 6


def test_summary_reads_a_folder(tmp_path):
    import lxtul.summarize_bench4 as sb
    def cell(arm, tok_s, ms, tf):
        return {"arm": arm, "batch": 6, "params": 1,
                "train": {"status": "ok", "median_ms": ms, "tok_s": tok_s, "peak_alloc_gib": 5.0,
                          "phases_ms": {"forward": 1, "backward": 2, "clip": 0.1, "optimizer": 0.5, "ema": 0, "total": 3.6},
                          "flops": {"tflop_fwd_bwd": tf, "tflop_optimizer": 1.0}, "achieved_tflops": (tf + 1.0) / (ms / 1e3)},
                "infer": {"status": "ok", "median_ms": ms / 3, "tok_s": 3 * tok_s, "peak_alloc_gib": 2.0,
                          "flops": {"tflop_per_token": 1e-9 * tf}}}
    for arm, tok, ms, tf in (("gpt8", 80000.0, 76.8, 6.0), ("flat14", 50000.0, 122.9, 10.0), ("minimal", 40000.0, 153.6, 10.0)):
        (tmp_path / f"{arm}_b6.json").write_text(json.dumps(cell(arm, tok, ms, tf)))
    (tmp_path / "minimal_b12.json").write_text(json.dumps({"arm": "minimal", "batch": 12, "train": {"status": "OOM", "peak_alloc_gib": 31.0}}))
    md = sb.render(tmp_path)
    assert "| minimal |" in md and "OOM" in md and "0.50" in md      # minimal / gpt8 tok/s = 0.50
    assert "0.80" in md                                              # minimal / flat14 = 40000 / 50000
