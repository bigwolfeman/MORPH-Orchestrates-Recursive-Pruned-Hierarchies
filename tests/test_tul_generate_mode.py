"""``forward(..., generate=True)``: the generation forward skips the loss heads, not a logit.

Tracker item C1e (2026-10-10). The label-free eval forward of a slot-loop model runs every
head the val pass reads: the span decoder (a 2-layer decoder and a full-vocab CE over every
slot, the cause of the eager generator's OOM on 2026-10-07), the parallel span head,
NextLat, the MUX, the fan's diversity terms and per-pass readings, SIGReg, the row
contrast. Generation reads none of them. ``generate=True`` skips them at their call sites;
the val pass and the offline instruments call the forward without it.

What each test pins:
  1. The logits are BIT-equal with and without the mode (the LXTUL + pointer fixture: 4
     cells, latent-selected loop, cell norm, span decoder, epivol diversity, pointer).
  2. The span decoder and the diversity terms do not run under the mode, and do run without
     it (call counts), so 1 is not passing because nothing was skipped.
  3. ``generate_tul`` / ``generate_tul_batch`` use the mode, and their tokens and the rows
     they sample from are identical with and without it, greedy and sampled.
  4. Refusals: labels, train mode, no slot layout, and the TUL forwards the mode does not
     gate (the paid loop).

Sabotage checks (run and reverted; session report): (a) the mode also skips the pointer mix
-> test 1 fails; (b) the span decoder's gate is removed -> test 2 fails.

CPU, fp32, the tests/test_loop_attn_center.py LXTUL fixture.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import morph.inference.tul_generate as eager_mod
import morph.model.transformer as tr_mod
from morph.inference.tul_generate import generate_tul, generate_tul_batch
from morph.model.transformer import MORPHTransformer
from morph.model.tul_layout import BoundaryRule
from test_loop_attn_center import _lx
from test_tul_fan import _batch
from test_tul_lx_credit import M
from test_tul_strict_geometry import _spec, _tiny, _tul

M = int(M)
V = 64


def _live_lx() -> MORPHTransformer:
    """LXTUL + pointer with every route live: the pointer gate shares the mass with the
    model, the cell norm's gain is off its init, the bigram and x0 injections are on, and
    the register's zero-init output ``W_o`` / ``P_cell`` are drawn (else its term is 0)."""
    m = _lx(pointer_heads=4, fan_repel_lambda=0.1, fan_repel_mode="epivol")
    with torch.no_grad():
        m.tul_pointer.gate_norm_head.bias.zero_()
        g = torch.Generator().manual_seed(3)
        reg = m.tul_register
        reg.W_o.weight.copy_(torch.randn(reg.W_o.weight.shape, generator=g) * 0.2)
        reg.P_cell.copy_(torch.randn(reg.P_cell.shape, generator=g) * 0.2)
        m.tul_cell_norm.weight.copy_(
            torch.empty_like(m.tul_cell_norm.weight).uniform_(0.5, 1.5, generator=g))
        m.embed.bigram.lambdas.fill_(0.5)
        for inj in m.x0_injects:
            inj.log_scale.fill_(0.3)
    return m.eval()


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[0, 10, 11, 20, 30]] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


PROMPT = [5, 6, 7, 8, 10, 12, 13, 14, 15, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]


def test_logits_bit_equal_with_and_without_the_mode():
    m = _live_lx()
    _ids, inp, _lab, layout = _batch(M)
    with torch.no_grad():
        a = m(inp, slot_layout=layout)["logits"]
        b = m(inp, slot_layout=layout, generate=True)["logits"]
    assert torch.equal(a, b)
    assert m.tul_pointer is not None and m.tul_spandec is not None


def test_the_heads_are_skipped(monkeypatch):
    m = _live_lx()
    _ids, inp, _lab, layout = _batch(M)
    calls = {"spandec": 0, "epi": 0, "vol": 0}

    def _count(name, real):
        def f(*a, **k):
            calls[name] += 1
            return real(*a, **k)
        return f

    monkeypatch.setattr(m, "_tul_spandec_loss", _count("spandec", m._tul_spandec_loss))
    monkeypatch.setattr(tr_mod, "fan_epi_term", _count("epi", tr_mod.fan_epi_term))
    monkeypatch.setattr(tr_mod, "fan_vol_term", _count("vol", tr_mod.fan_vol_term))
    with torch.no_grad():
        m(inp, slot_layout=layout, generate=True)
    assert calls == {"spandec": 0, "epi": 0, "vol": 0}, calls
    with torch.no_grad():
        m(inp, slot_layout=layout)
    assert calls["spandec"] == 1 and calls["epi"] >= 1 and calls["vol"] >= 1, calls


def _record(monkeypatch) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    real = eager_mod.sample_next

    def _spy(logits, temperature, top_k, generator, top_p=0.0):
        rows.append(logits.detach().float().clone())
        return real(logits, temperature, top_k, generator, top_p)

    monkeypatch.setattr(eager_mod, "sample_next", _spy)
    return rows


@pytest.mark.parametrize("temp", [0.0, 1.0], ids=["greedy", "sampled"])
def test_generators_use_the_mode_and_emit_the_same_tokens(monkeypatch, temp):
    m = _live_lx()
    spec = _spec(prefix_k=M, max_slots=64, seq_len=512)
    seen: list[bool] = []
    real_fwd = m.forward

    def _fwd(*a, **k):
        seen.append(bool(k.get("generate", False)))
        return real_fwd(*a, **k)

    monkeypatch.setattr(m, "forward", _fwd)       # nn.Module.__call__ reads self.forward
    out = {}
    for mode in (True, False):
        if not mode:
            monkeypatch.setattr(eager_mod, "_generation_kwargs", lambda model: {})
        rows = _record(monkeypatch)
        seen.clear()
        a, ba = generate_tul(m, PROMPT, _rule(), spec, max_new_tokens=24, temperature=temp,
                             seed=7, emit_source="token", device="cpu")
        bb, _bld = generate_tul_batch(m, [PROMPT, PROMPT[:9]], _rule(), spec,
                                      max_new_tokens=12, temperature=temp, seeds=[7, 8],
                                      emit_source="token", device="cpu")
        out[mode] = (a, ba.ids, bb, rows)
        assert ba.n_slots >= 3
        assert seen and all(x == mode for x in seen), (mode, seen[:4])
    assert out[True][0] == out[False][0] and out[True][1] == out[False][1]
    assert out[True][2] == out[False][2]
    assert all(torch.equal(x, y) for x, y in zip(out[True][3], out[False][3]))


def test_the_eager_generator_passes_the_mode():
    m = _live_lx()
    assert eager_mod._generation_kwargs(m) == {"generate": True}


def test_refusals():
    m = _live_lx()
    _ids, inp, lab, layout = _batch(M)
    with torch.no_grad():
        with pytest.raises(ValueError, match="label-free"):
            m(inp, labels=lab, slot_layout=layout, generate=True)
        with pytest.raises(ValueError, match="requires slot_layout"):
            m(inp[:, :8], generate=True)
    m.train()
    with pytest.raises(ValueError, match="label-free"):
        m(inp, slot_layout=layout, generate=True)
    torch.manual_seed(0)
    paid = MORPHTransformer(_tiny(tul=_tul(tokens_through_core=True))).eval()
    assert not paid.tul_generate_supported
    assert eager_mod._generation_kwargs(paid) == {}
    with torch.no_grad(), pytest.raises(NotImplementedError, match="slot-loop"):
        paid(inp, slot_layout=layout, generate=True)
