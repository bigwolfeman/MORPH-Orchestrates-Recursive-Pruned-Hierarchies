"""The KV-cached decoder on LXTUL (tracker item C1c, 2026-10-10): the cache is the same function.

`morph/inference/tul_generate_cached.py` now reproduces the LXTUL label-free forward: M = 4
looped cells per slot (the Thought Register's seed, the 4-cell loop cache with sibling
reads, the per-pass cell RMSNorm, the router's pick and reset after every pass, the
winner-only write), the one-rollout read and the pointer head with a cache over every
earlier token. On the tiny LXTUL + pointer fixture in fp32 on CPU the cached decoder must
emit the SAME tokens as `generate_tul` and sample from log-prob rows that agree to 1e-5:

* both emit sources, a prompt that crosses boundaries and one that ends mid-span, greedy
  and sampled (a seeded sampler draws the same tokens only if the rows match);
* without the pointer; the plain one-cell slot loop (M = 1, no code rollouts); the
  ungraded fan (M = 4, every cell written, no router);
* a generation whose cell axis and token axis both pass `window_size`.

Tolerance, argued from fp64 (`test_fp64_reference_is_exact`): with the model and both
generators in fp64 END TO END (the tree's explicit fp32 casts in attention, the router and
the pointer turned into no-ops for that test) the worst row difference over every fixture,
emit source and decode is 3e-14 (measured 2026-10-10), so the two paths compute the same
function and what is left in fp32 is summation order (one row's GEMM against a row slice
of an [L, C] GEMM, attention over a key subset). Measured fp32 worst: 3.3e-6 without the
pointer, 2.2e-5 with it (the pointer adds an fp32 log of a mixture of probabilities). TOL
is 1e-4, about 5x the fp32 worst; every sabotage below moves a row by more than 1e-2.

`test_sabotage_*` re-breaks one piece of the cached path and asserts the comparison
FAILS: skip the reset, skip the cell norm, write a loser, drop the register's term, read
the pointer's follower one token late, run the loop causal (no later sibling). `cells` other
than the model's M is refused.

CPU, fp32, the tests/test_loop_attn_center.py LXTUL fixture (via test_tul_generate_mode).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import morph.inference.tul_generate as eager_mod
import morph.inference.tul_generate_cached as cached_mod
import morph.inference.tul_generate_graphed as graphed_mod
from morph.inference.tul_generate import generate_tul
from morph.inference.tul_generate_cached import generate_tul_cached
from morph.inference.tul_generate_graphed import generate_tul_graphed
from morph.model.transformer import MORPHTransformer
from morph.model.tul_layout import BoundaryRule
from test_tul_fan_lsel import _build, _nowta_kw
from test_tul_generate_mode import _live_lx
from test_tul_lx_credit import M
from test_tul_strict_geometry import _spec, _tiny, _tul

M = int(M)
V = 64
TOL = 1e-4           # fp32 rows; see the module docstring
TOL64 = 1e-10        # fp64 end to end

PROMPT_CROSS = [5, 6, 7, 8, 10, 12, 13, 14, 15, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
PROMPT_MID = [5, 6, 7, 8, 10, 12, 13]


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[0, 10, 11, 20, 30]] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _live(m: MORPHTransformer) -> MORPHTransformer:
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
        for inj in m.x0_injects:
            inj.log_scale.fill_(0.3)
    return m.eval().float()


def _no_pointer() -> MORPHTransformer:
    from test_loop_attn_center import _lx
    m = _lx()
    assert m.tul_pointer is None
    return _live(m)


def _one_cell() -> MORPHTransformer:
    """The plain strict slot loop: one cell, no code rollouts, prefix_k 2."""
    torch.manual_seed(1234)
    m = MORPHTransformer(_tiny(tul=_tul(tg_geometry="strict", spandec=False,
                                        plast_weight=1.0, slot_seed="boundary"),
                               d_ff=96, dropout=0.0))
    assert m.tul_code_enum is None and m.tul_register is None
    return _live(m)


def _ungraded_fan() -> MORPHTransformer:
    m = _build(**_nowta_kw())
    assert m._lsel_mode == "off" and m.tul_fan is not None
    return _live(m)


def _record(monkeypatch, module) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    real = module.sample_next

    def _spy(logits, temperature, top_k, generator, top_p=0.0):
        rows.append(logits.detach().float().clone())
        return real(logits, temperature, top_k, generator, top_p)

    monkeypatch.setattr(module, "sample_next", _spy)
    return rows


def _run_both(monkeypatch, m, prompt, n, emit, temp):
    # max_slots * M cells must fit the tiny RoPE cache (256) on the eager loop axis
    spec = _spec(prefix_k=int(m.cfg.tul.prefix_k), max_slots=60, seq_len=512)
    er = _record(monkeypatch, eager_mod)
    a, ba = generate_tul(m, prompt, _rule(), spec, max_new_tokens=n, temperature=temp, seed=7,
                         emit_source=emit, device="cpu")
    cr = _record(monkeypatch, cached_mod)
    b, bb = generate_tul_cached(m, prompt, _rule(), spec, max_new_tokens=n, temperature=temp,
                                seed=7, emit_source=emit, device="cpu")
    return a, ba, er, b, bb, cr


def _run_static(monkeypatch, m, prompt, n, emit, temp, device="cpu", use_graphs=False):
    """The fixed-capacity decoder (``tul_generate_graphed``): its step functions run
    eagerly (``use_graphs=False``, the CPU path) or replayed as CUDA graphs."""
    spec = _spec(prefix_k=int(m.cfg.tul.prefix_k), max_slots=60, seq_len=512)
    sr = _record(monkeypatch, graphed_mod)
    b, bb = generate_tul_graphed(m, prompt, _rule(), spec, max_new_tokens=n,
                                 temperature=temp, seed=7, emit_source=emit, device=device,
                                 use_graphs=use_graphs)
    return b, bb, sr


def _max_row_diff(er, cr) -> float:
    """Worst entry difference of the two generators' rows AS LOG-PROBS. The eager generator
    samples from raw logits on a model without the pointer (``embed.attend``, slot masked)
    and from log-probs with it; the cached one always from log-probs. Sampling is
    invariant to the row's constant, so both are normalised before comparing."""
    assert len(er) == len(cr)
    worst = 0.0
    for x, y in zip(er, cr):
        x, y = torch.log_softmax(x.double(), -1), torch.log_softmax(y.double(), -1)
        fx, fy = torch.isfinite(x), torch.isfinite(y)
        assert torch.equal(fx, fy), "the -inf pattern (the masked slot id) differs"
        worst = max(worst, float((x[fx] - y[fy]).abs().max()))
    return worst


@pytest.mark.parametrize("emit", ["token", "slot"])
@pytest.mark.parametrize("prompt", [PROMPT_CROSS, PROMPT_MID], ids=["cross", "midspan"])
@pytest.mark.parametrize("temp", [0.0, 1.0], ids=["greedy", "sampled"])
def test_lxtul_pointer_cached_matches_eager(monkeypatch, emit, prompt, temp):
    m = _live_lx()
    a, ba, er, b, bb, cr = _run_both(monkeypatch, m, prompt, 40, emit, temp)
    assert a == b
    assert ba.ids == bb.ids and ba.slot_first == bb.slot_first
    assert ba.n_slots >= 3, "the generation must cross boundaries to test the slot path"
    assert _max_row_diff(er, cr) < TOL


@pytest.mark.parametrize("build", [_no_pointer, _one_cell, _ungraded_fan],
                         ids=["lxtul_no_pointer", "one_cell", "ungraded_fan"])
@pytest.mark.parametrize("temp", [0.0, 1.0], ids=["greedy", "sampled"])
def test_other_supported_models_match_eager(monkeypatch, build, temp):
    m = build()
    for emit in ("token", "slot"):
        a, ba, er, b, bb, cr = _run_both(monkeypatch, m, PROMPT_CROSS, 40, emit, temp)
        assert a == b and ba.ids == bb.ids and ba.n_slots >= 3
        assert _max_row_diff(er, cr) < TOL


@pytest.mark.parametrize("build", [_live_lx, _no_pointer, _one_cell, _ungraded_fan],
                         ids=["lxtul_pointer", "lxtul_no_pointer", "one_cell", "ungraded_fan"])
@pytest.mark.parametrize("emit", ["token", "slot"])
def test_static_buffers_match_eager(monkeypatch, build, emit):
    """The CUDA-graph decoder's step functions (fixed capacity, device-side masks, the
    pointer's validity row) against the eager generator, greedy and sampled, on a
    generation long enough to pass the window."""
    m = build()
    for temp, n in ((0.0, 40), (1.0, 90)):
        a, ba, er, _b, _bb, _cr = _run_both(monkeypatch, m, PROMPT_CROSS, n, emit, temp)
        s, bs, sr = _run_static(monkeypatch, m, PROMPT_CROSS, n, emit, temp)
        assert a == s and ba.ids == bs.ids and ba.n_slots >= 3
        assert _max_row_diff(er, sr) < TOL


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("emit", ["token", "slot"])
def test_cuda_graphs_match_eager(monkeypatch, emit):
    """The captured graphs on the GPU, fp32, against the eager generator on the GPU."""
    m = _live_lx().cuda()
    for temp in (0.0, 1.0):
        spec = _spec(prefix_k=M, max_slots=60, seq_len=512)
        er = _record(monkeypatch, eager_mod)
        a, ba = generate_tul(m, PROMPT_CROSS, _rule(), spec, max_new_tokens=60,
                             temperature=temp, seed=7, emit_source=emit, device="cuda")
        s, bs, sr = _run_static(monkeypatch, m, PROMPT_CROSS, 60, emit, temp, device="cuda",
                                use_graphs=True)
        assert a == s and ba.ids == bs.ids and ba.n_slots >= 3
        assert _max_row_diff([r.cpu() for r in er], [r.cpu() for r in sr]) < TOL


def test_long_generation_passes_the_window(monkeypatch):
    """More cells than `window_size` (16) on the loop's compact axis and more tokens than it
    on the row: both windows cut keys, the slot branches still read every earlier cell."""
    m = _live_lx()
    a, ba, er, b, bb, cr = _run_both(monkeypatch, m, PROMPT_CROSS, 90, "token", 1.0)
    assert ba.n_slots * M > m.cfg.window_size and len(ba.ids) > 3 * m.cfg.window_size
    assert a == b
    assert _max_row_diff(er, cr) < TOL


def _fp64_everywhere(monkeypatch) -> None:
    """Make a ``.double()`` model stay fp64: the tree casts to fp32 explicitly (``.float()``,
    ``.to(torch.float32)``, ``dtype=torch.float32`` buffers) in attention, the router and the
    pointer; under this patch those are no-ops on fp64 tensors. Test-scoped (monkeypatch)."""
    f32, to, zeros = torch.Tensor.float, torch.Tensor.to, torch.zeros

    def _float(self, *a, **k):
        return self if self.dtype == torch.float64 else f32(self, *a, **k)

    def _to(self, *a, **k):
        want32 = (a and a[0] is torch.float32) or k.get("dtype") is torch.float32
        return self if self.dtype == torch.float64 and want32 else to(self, *a, **k)

    def _zeros(*a, **k):
        if k.get("dtype") is torch.float32:
            k["dtype"] = torch.float64
        return zeros(*a, **k)

    monkeypatch.setattr(torch.Tensor, "float", _float)
    monkeypatch.setattr(torch.Tensor, "to", _to)
    monkeypatch.setattr(torch, "zeros", _zeros)


@pytest.mark.parametrize("build", [_live_lx, _one_cell], ids=["lxtul_pointer", "one_cell"])
def test_fp64_reference_is_exact(monkeypatch, build):
    m = build().double()
    _fp64_everywhere(monkeypatch)
    for emit in ("token", "slot"):
        a, ba, er, b, bb, cr = _run_both(monkeypatch, m, PROMPT_CROSS, 40, emit, 1.0)
        assert a == b and ba.n_slots >= 3
        assert all(r.dtype == torch.float64 for r in er + cr)
        assert _max_row_diff(er, cr) < TOL64


def test_cells_other_than_the_models_are_refused():
    m = _live_lx()
    spec = _spec(prefix_k=M, max_slots=60, seq_len=512)
    out, _b = generate_tul_cached(m, PROMPT_MID, _rule(), spec, max_new_tokens=2,
                                  temperature=0.0, device="cpu", cells=M)
    assert len(out) == 2
    for c in (1, 2, 8):
        with pytest.raises(NotImplementedError, match="B1"):
            generate_tul_cached(m, PROMPT_MID, _rule(), spec, max_new_tokens=2,
                                temperature=0.0, device="cpu", cells=c)
        with pytest.raises(NotImplementedError, match="B1"):
            generate_tul_graphed(m, PROMPT_MID, _rule(), spec, max_new_tokens=2,
                                 temperature=0.0, device="cpu", use_graphs=False, cells=c)


@pytest.mark.parametrize("kw,what", [(dict(pointer_coverage_lambda=1.0), "coverage"),
                                     (dict(pointer_cell_key=True), "cell key")])
def test_refuses_pointer_variants(kw, what):
    from test_loop_attn_center import _lx
    m = _live(_lx(pointer_heads=2, **kw))
    with pytest.raises(NotImplementedError, match=what):
        generate_tul_cached(m, PROMPT_MID, _rule(), _spec(prefix_k=M), max_new_tokens=2,
                            temperature=0.0, device="cpu")


# ── sabotage: the comparison above must FAIL on a decoder that is wrong ──────────────


def _sabotaged_diff(monkeypatch, m) -> tuple[bool, float]:
    a, _ba, er, b, _bb, cr = _run_both(monkeypatch, m, PROMPT_CROSS, 32, "token", 0.0)
    n = min(len(er), len(cr))
    return a == b, _max_row_diff(er[:n], cr[:n])


class _Proxy:
    """The model as the CACHED decoder sees it, with named attributes replaced; the eager
    forward keeps the real model."""

    def __init__(self, inner, **over):
        self._inner = inner
        self.__dict__.update(over)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _proxy_decoder(monkeypatch, **over) -> None:
    real_init = cached_mod.TulCachedDecoder.__init__

    def _init(self, model, spec, cells=None):
        real_init(self, model, spec, cells)
        self.m = _Proxy(model, **over)

    monkeypatch.setattr(cached_mod.TulCachedDecoder, "__init__", _init)


def test_sabotage_skipping_the_reset_is_caught(monkeypatch):
    m = _live_lx()
    real = cached_mod.lsel_step
    monkeypatch.setattr(cached_mod, "lsel_step",
                        lambda model, h, ctx, mm, reset: real(model, h, ctx, mm, False))
    _same, diff = _sabotaged_diff(monkeypatch, m)
    assert diff > 1e-3, diff


def test_sabotage_skipping_the_cell_norm_is_caught(monkeypatch):
    m = _live_lx()
    _proxy_decoder(monkeypatch, _cell_norm_apply=lambda h: h)
    _same, diff = _sabotaged_diff(monkeypatch, m)
    assert diff > 1e-3, diff


def test_sabotage_writing_a_loser_is_caught(monkeypatch):
    m = _live_lx()
    real = cached_mod.prefix_write
    monkeypatch.setattr(cached_mod, "prefix_write",
                        lambda model, h, pick, kp: real(model, h, (pick + 1) % M, kp))
    _same, diff = _sabotaged_diff(monkeypatch, m)
    assert diff > 1e-3, diff


def test_sabotage_dropping_the_register_term_is_caught(monkeypatch):
    m = _live_lx()
    monkeypatch.setattr(cached_mod, "register_term",
                        lambda model, xs, valid=None: torch.zeros(1, M, xs.shape[-1]))
    _same, diff = _sabotaged_diff(monkeypatch, m)
    assert diff > 1e-3, diff


def test_sabotage_a_late_pointer_follower_is_caught(monkeypatch):
    """Every cached key's follower is shifted one token late (the token after the true one)."""
    m = _live_lx()
    real = cached_mod.pointer_mix

    def _late(model, h, lm, q, keys, cand, valid=None):
        late = torch.cat([cand[1:], cand[-1:]]) if cand.numel() else cand
        return real(model, h, lm, q, keys, late, valid)

    monkeypatch.setattr(cached_mod, "pointer_mix", _late)
    _same, diff = _sabotaged_diff(monkeypatch, m)
    assert diff > 1e-3, diff


def test_sabotage_a_causal_cell_loop_is_caught(monkeypatch):
    """The loop relation as plain causal on the cell axis: no cell reads a later sibling."""
    m = _live_lx()
    real = cached_mod.loop_relation
    monkeypatch.setattr(cached_mod, "loop_relation",
                        lambda qpos, kpos, cells: real(qpos, kpos, 1))
    _same, diff = _sabotaged_diff(monkeypatch, m)
    assert diff > 1e-3, diff


def _static_sabotaged_diff(monkeypatch, m) -> float:
    a, _ba, er, *_ = _run_both(monkeypatch, m, PROMPT_CROSS, 32, "token", 0.0)
    s, _bs, sr = _run_static(monkeypatch, m, PROMPT_CROSS, 32, "token", 0.0)
    n = min(len(er), len(sr))
    return _max_row_diff(er[:n], sr[:n])


def test_sabotage_static_skipping_the_reset_is_caught(monkeypatch):
    m = _live_lx()
    real = graphed_mod.lsel_step
    monkeypatch.setattr(graphed_mod, "lsel_step",
                        lambda model, h, ctx, mm, reset: real(model, h, ctx, mm, False))
    diff = _static_sabotaged_diff(monkeypatch, m)
    assert diff > 1e-3, diff


def test_sabotage_static_pointer_reads_its_own_key_is_caught(monkeypatch):
    """Every buffer entry counts as valid: a query also reads the newest key, whose
    follower is not fed yet, and every unwritten (zero) entry of the buffer."""
    m = _live_lx()
    real = graphed_mod.pointer_mix

    def _all_valid(model, h, lm, q, keys, cand, valid=None):
        return real(model, h, lm, q, keys, cand, torch.ones_like(valid))

    monkeypatch.setattr(graphed_mod, "pointer_mix", _all_valid)
    diff = _static_sabotaged_diff(monkeypatch, m)
    assert diff > 1e-3, diff
