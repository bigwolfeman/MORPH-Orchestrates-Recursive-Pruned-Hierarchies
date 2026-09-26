"""`generate_tul_cached` against `generate_tul` — the KV cache is the same function.

`morph/inference/tul_generate_cached.py` computes every position of a strict TUL row once
and caches what later positions read; `generate_tul` recomputes the whole grown row at
every token. On a tiny strict, `code_enum_k = 4` model in fp32 on CPU the two must emit
the SAME tokens and sample from log-prob rows that agree to 1e-5 at every step, for:

* a prompt that crosses several boundaries and one that ends mid-span;
* both emit sources (`token`: the boundary token's row; `slot`: the last prefix cell's);
* both slot seeds the decoder reproduces (`boundary`, `bag_mean`);
* greedy AND sampled decoding (the same seeded generator draws the same tokens only if
  the rows match);
* a generation long enough that the slot count passes `window_size`, so the loop's
  window limit and the coda's absolute-position window both bind.

The per-step rows are read by wrapping the ONE sampling step both generators call
(`sample_next`), so what is compared is exactly the row each generator sampled from.

`test_sabotage_*` re-breaks one cache piece and asserts the comparison FAILS, so the
tolerance above cannot pass a decoder that is wrong.

CPU, fp32, tiny config (`tests/test_tul_strict_geometry.py::_tiny`).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import morph.inference.tul_generate as eager_mod
import morph.inference.tul_generate_cached as cached_mod
from morph.inference.tul_generate import generate_tul
from morph.inference.tul_generate_cached import generate_tul_cached
import morph.inference.tul_generate_graphed as graphed_mod
from morph.inference.tul_generate_graphed import generate_tul_graphed
from morph.model.transformer import MORPHTransformer
from morph.model.tul_layout import BoundaryRule
from test_tul_strict_geometry import _spec, _tiny, _tul

V = 64
TOL = 1e-5

PROMPT_CROSS = [5, 6, 7, 8, 10, 12, 13, 14, 15, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
PROMPT_MID = [5, 6, 7, 8, 10, 12, 13]


def _rule() -> BoundaryRule:
    # Several boundary ids and a short cap, so even a greedy loop on one token cuts a
    # span at least every 8 tokens: every generation below crosses many boundaries.
    lut = np.zeros(V, dtype=bool)
    lut[[0, 10, 11, 20, 30]] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _model(slot_seed: str = "boundary", seed: int = 1234, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tc = _tul(tg_geometry="strict", spandec=False, spandec_parallel=True, code_enum_k=4,
              plast_weight=1.0, slot_seed=slot_seed, **kw)
    m = MORPHTransformer(_tiny(tul=tc, d_ff=96, dropout=0.0))
    with torch.no_grad():
        # Zero-init scales would switch whole routes off and let a wrong cache pass: the
        # bigram lambdas, the x0 / value-embed injection scales and the enum code ratio's
        # carrier are all made live here.
        m.embed.bigram.lambdas.fill_(0.5)
        for inj in m.x0_injects:
            inj.log_scale.fill_(0.3)
        for ve in m.value_embeds:
            for p in ve.parameters():
                if p.dim() == 0:
                    p.fill_(0.3)
    return m.eval().float()


def _record(monkeypatch, module) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    real = module.sample_next

    def _spy(logits, temperature, top_k, generator):
        rows.append(logits.detach().float().clone())
        return real(logits, temperature, top_k, generator)

    monkeypatch.setattr(module, "sample_next", _spy)
    return rows


def _run_both(monkeypatch, m, prompt, n, emit, temp, spec=None):
    spec = spec or _spec(max_slots=200, seq_len=512)
    rule = _rule()
    er = _record(monkeypatch, eager_mod)
    a, ba = generate_tul(m, prompt, rule, spec, max_new_tokens=n, temperature=temp, seed=7,
                         emit_source=emit, device="cpu")
    cr = _record(monkeypatch, cached_mod)
    b, bb = generate_tul_cached(m, prompt, rule, spec, max_new_tokens=n, temperature=temp,
                                seed=7, emit_source=emit, device="cpu")
    return a, ba, er, b, bb, cr


def _run_static(monkeypatch, m, prompt, n, emit, temp, spec=None):
    """The fixed-capacity decoder (``tul_generate_graphed``) with ``use_graphs=False``: the
    same step functions the CUDA graphs capture, run eagerly on CPU."""
    spec = spec or _spec(max_slots=200, seq_len=512)
    sr = _record(monkeypatch, graphed_mod)
    b, bb = generate_tul_graphed(m, prompt, _rule(), spec, max_new_tokens=n,
                                 temperature=temp, seed=7, emit_source=emit, device="cpu",
                                 use_graphs=False)
    return b, bb, sr


def _max_row_diff(er, cr) -> float:
    assert len(er) == len(cr)
    worst = 0.0
    for x, y in zip(er, cr):
        fx, fy = torch.isfinite(x), torch.isfinite(y)
        assert torch.equal(fx, fy), "the -inf pattern (the masked slot id) differs"
        worst = max(worst, float((x[fx] - y[fy]).abs().max()))
    return worst


@pytest.mark.parametrize("slot_seed", ["boundary", "bag_mean"])
@pytest.mark.parametrize("emit", ["token", "slot"])
@pytest.mark.parametrize("prompt", [PROMPT_CROSS, PROMPT_MID], ids=["cross", "midspan"])
@pytest.mark.parametrize("temp", [0.0, 1.0], ids=["greedy", "sampled"])
def test_cached_matches_eager(monkeypatch, slot_seed, emit, prompt, temp):
    m = _model(slot_seed)
    a, ba, er, b, bb, cr = _run_both(monkeypatch, m, prompt, 48, emit, temp)
    assert a == b
    assert ba.ids == bb.ids and ba.slot_first == bb.slot_first
    assert ba.n_slots >= 3, "the generation must cross boundaries to test the slot path"
    assert _max_row_diff(er, cr) < TOL


@pytest.mark.parametrize("slot_seed", ["boundary", "bag_mean"])
@pytest.mark.parametrize("emit", ["token", "slot"])
@pytest.mark.parametrize("prompt", [PROMPT_CROSS, PROMPT_MID], ids=["cross", "midspan"])
def test_static_buffers_match_eager(monkeypatch, slot_seed, emit, prompt):
    """The CUDA-graph decoder's step functions (fixed capacity, device-side masks) against
    the eager generator, greedy and sampled."""
    m = _model(slot_seed)
    for temp in (0.0, 1.0):
        a, ba, er, _b, _bb, _cr = _run_both(monkeypatch, m, prompt, 48, emit, temp)
        s, bs, sr = _run_static(monkeypatch, m, prompt, 48, emit, temp)
        assert a == s
        assert ba.ids == bs.ids and ba.n_slots >= 3
        assert _max_row_diff(er, sr) < TOL


def test_static_long_generation_passes_the_window(monkeypatch):
    m = _model("boundary")
    a, ba, er, _b, _bb, _cr = _run_both(monkeypatch, m, PROMPT_CROSS, 110, "token", 1.0)
    s, bs, sr = _run_static(monkeypatch, m, PROMPT_CROSS, 110, "token", 1.0)
    assert ba.n_slots > m.cfg.window_size
    assert a == s
    assert _max_row_diff(er, sr) < TOL


def test_long_generation_passes_the_window(monkeypatch):
    """More slots than `window_size` (16): the loop's compact-axis window and the coda's
    absolute-position window both cut keys here, and the coda's slot branch (no window)
    still reads every earlier cell."""
    m = _model("boundary")
    a, ba, er, b, bb, cr = _run_both(monkeypatch, m, PROMPT_CROSS, 110, "token", 1.0)
    assert ba.n_slots > m.cfg.window_size, ba.n_slots
    assert a == b
    assert _max_row_diff(er, cr) < TOL


def test_refuses_a_model_it_does_not_reproduce():
    torch.manual_seed(0)
    tc = _tul(tg_geometry="restrict", spandec=False, spandec_parallel=True, code_enum_k=4,
              plast_weight=1.0)
    m = MORPHTransformer(_tiny(tul=tc, d_ff=96, dropout=0.0)).eval()
    with pytest.raises(NotImplementedError, match="strict"):
        generate_tul_cached(m, PROMPT_MID, _rule(), _spec(), max_new_tokens=2,
                            temperature=0.0, device="cpu")


# ── sabotage: the comparison above must FAIL on a decoder that is wrong ──────────────


def _sabotaged_diff(monkeypatch, m) -> tuple[bool, float]:
    a, _ba, er, b, _bb, cr = _run_both(monkeypatch, m, PROMPT_CROSS, 32, "token", 0.0)
    n = min(len(er), len(cr))
    return a == b, _max_row_diff(er[:n], cr[:n])


def test_sabotage_skipping_the_conv_reset_is_caught(monkeypatch):
    """The span-local conv / value-shift history is NOT dropped at a boundary: the first
    tokens of every new span read the previous span through the conv."""
    m = _model("boundary")
    real_reset = cached_mod._Region.reset

    def _keep_hist(self):
        old = getattr(self, "hist", None)
        real_reset(self)
        if old:
            self.hist = old

    monkeypatch.setattr(cached_mod._Region, "reset", _keep_hist)
    same, diff = _sabotaged_diff(monkeypatch, m)
    assert diff > 1e-3, diff


def test_sabotage_dropping_the_code_term_is_caught(monkeypatch):
    """The K rollouts collapse to one code: the loop's per-pass code term is skipped."""
    m = _model("boundary")

    def _zero_term(h, valid, n):
        return torch.zeros(h.shape[0], h.shape[1], h.shape[-1], dtype=h.dtype)

    orig = m.tul_code_enum.term
    real_init = cached_mod.TulCachedDecoder.__init__

    def _init(self, model, spec):
        real_init(self, model, spec)
        # Sabotage only the cached decoder's view: the eager forward keeps the real term.
        self.m = _Proxy(model, _zero_term)

    class _Proxy:
        def __init__(self, inner, term):
            self._inner = inner
            self.tul_code_enum = type("E", (), {"term": staticmethod(term)})()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(cached_mod.TulCachedDecoder, "__init__", _init)
    same, diff = _sabotaged_diff(monkeypatch, m)
    assert m.tul_code_enum.term == orig
    assert diff > 1e-3, diff
