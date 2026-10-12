"""Eval cost (2026-10-12): ``training.eval_keep_graphs`` and ``training.eval_diag_every``.

  a. ``tul_attn_lift_probe(lean=True)`` stops at the coda readout, so the ``[B, L, V]`` logits
     and the pointer mixture over the vocabulary are never built, and the counted statistics
     are the same numbers as the full forward's (bit-equal here).
  b. ``evaluate(lean_probe=True)`` logs the same ``val/`` keys with the same values.
  c. ``evaluate(diag=False)`` drops exactly the ablation / probe keys and leaves the val
     loss and every other reading unchanged.
  d. The two keys exist in base.yaml with the shipped defaults (off / every eval).

CPU, fp32, the ``tests/test_tul_gl1b.py`` fixtures plus the output-only pointer head.
"""
from __future__ import annotations

import pathlib

import pytest
import torch

import morph.model.transformer as transformer_mod
from morph.training.train import evaluate
from test_tul_gl1b import _batch, _model, _tul

ABLATION_KEYS = ("val/plan_worth_zero", "val/plan_worth_shuffle", "val/plan_worth_wrong_seed",
                 "val/slot_eff_rank", "val/slot_pairwise_cos")


def _m():
    return _model(seed=1, tul=_tul(mux_beta=0.0, pointer_heads=2)).eval()


def _eval(m, **kw) -> dict:
    x, y, lay, _ = _batch(B=2)
    extra: dict = {}
    torch.manual_seed(0)       # the shuffle probe draws its permutation from the global stream
    loss, _ppl = evaluate(m, torch.device("cpu"), iter([(x, y, lay)] * 2), n_batches=2,
                          tul=True, extra=extra, **kw)
    return {"val/loss": loss, **extra}


def test_lean_lift_probe_is_equal_and_builds_no_logits(monkeypatch):
    m = _m()
    x, _y, lay, _ = _batch(B=2)
    full = m.tul_attn_lift_probe(x, lay)
    calls: list[str] = []
    ptr = m.tul_pointer
    real_mixed = ptr.mixed_logprobs
    monkeypatch.setattr(ptr, "mixed_logprobs",
                        lambda *a, **k: calls.append("mixed") or real_mixed(*a, **k))
    real_attend = m.embed.attend
    monkeypatch.setattr(m.embed, "attend",
                        lambda *a, **k: calls.append("attend") or real_attend(*a, **k))
    m.tul_attn_lift_probe(x, lay)
    assert calls == ["attend", "mixed"], "the full probe no longer builds logits; the test is inert"
    calls.clear()
    lean = m.tul_attn_lift_probe(x, lay, lean=True)
    assert calls == [], f"lean probe built logits: {calls}"
    assert full.keys() == lean.keys() and len(full) > 0
    for k in full:
        assert (full[k] == lean[k]) or (full[k] != full[k] and lean[k] != lean[k]), k


def test_evaluate_lean_probe_logs_the_same_values():
    m = _m()
    a, b = _eval(m), _eval(m, lean_probe=True)
    assert a.keys() == b.keys()
    assert any(k.startswith("val/attn_lift") or "lift" in k for k in a), sorted(a)
    for k in a:
        assert a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]), (k, a[k], b[k])


def test_evaluate_diag_false_drops_only_the_ablation_block():
    m = _m()
    on, off = _eval(m), _eval(m, diag=False)
    for k in ABLATION_KEYS:
        assert k in on and k not in off, k
    dropped = set(on) - set(off)
    assert dropped and not (set(off) - set(on))
    assert any("lift" in k for k in dropped)
    for k in off:
        assert on[k] == off[k], (k, on[k], off[k])


def test_base_yaml_defaults():
    import yaml
    t = yaml.safe_load(
        (pathlib.Path(__file__).parents[1] / "morph/configs/base.yaml").read_text())["training"]
    assert t["eval_keep_graphs"] is False and t["eval_diag_every"] == 1
