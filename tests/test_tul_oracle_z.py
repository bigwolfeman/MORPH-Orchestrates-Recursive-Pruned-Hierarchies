"""`tul.oracle_z` — the per-PASS teacher, and the rule it breaks.

The standing rule (root `CLAUDE.md`; LCM T3/4, CoCoMix §6b, BT §4.2) is NEVER to regress
onto the slot state. This knob does exactly that. It exists because eleven arms have read
a per-pass K-curve of zero and the open question is whether the loop's passes CANNOT
descend a useful objective or whether nothing has ever told them what each pass is FOR.
An oracle trajectory a one-step optimiser could match is the cheapest way to ask. It is a
TEST, not a shipped design, and nothing composes it by default.

What this file pins:

* **Off is nothing.** `oracle_z: false` never calls the term — proved by monkeypatching
  `_tul_oracle_z_loss` to raise — and leaves no key in the output.
* **The term acts.** On, it is positive, it is in `loss`, and its weighted twin is exposed
  the way `spandec_weighted` is so `train.py` can keep train/loss on the model's CE.
* **The decoder is NOT trained by it.** Every decoder parameter's gradient is EQUAL with
  and without the term, bit for bit, while core parameters' gradients differ. The inner
  `autograd.grad` passes THROUGH the decoder but never writes `.grad`, and the trajectory
  carries no `grad_fn`.
* **The oracle actually descends.** Its own readout falls over its T steps on a fixed
  batch. If it did not, the per-pass target would be teaching noise.

CPU only, fp32, tiny config — the `tests/test_tul_strict_geometry.py` fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=3, max_depth=4, bptt_depth=4,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                mux_beta=0.0, spandec=True, spandec_layers=1, spandec_max_tokens=8,
                slot_depth_fixed=3, slot_max_depth=4)
    base.update(kw)
    return TULConfig(**base)


def _batch(B: int = 2, n: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::8] = DOT
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _model(seed: int = 99, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**tul_kw)))
    return m.train().float()


def _run(m: MORPHTransformer):
    inp, lab, layout, _ = _batch()
    torch.manual_seed(3)
    return m(inp, labels=lab, slot_layout=layout)


# ── off is nothing ───────────────────────────────────────────────────────────

def test_off_never_calls_the_term(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("_tul_oracle_z_loss ran on a model with tul.oracle_z=false")
    monkeypatch.setattr(MORPHTransformer, "_tul_oracle_z_loss", boom)
    out = _run(_model())
    assert "oracle_z" not in out and "oracle_z_weighted" not in out


def test_off_keeps_the_trajectory_unbuilt():
    """The collection itself must stay off, or an oracle-off arm pays its memory."""
    m = _model()
    inp, _lab, layout, _ = _batch()
    x, x0, bg = m._tul_front(inp, layout)
    out = m._tul_core(x, x0, bg, layout, input_ids=inp)
    assert out[4] is None, "db_traj is built on a model that asked for nothing"


def test_on_builds_the_trajectory():
    m = _model(oracle_z=True, oracle_z_steps=3)
    inp, _lab, layout, _ = _batch()
    x, x0, bg = m._tul_front(inp, layout)
    out = m._tul_core(x, x0, bg, layout, input_ids=inp)
    assert out[4] is not None and len(out[4]) >= 4      # seed + 3 passes


def test_eval_pays_nothing(monkeypatch):
    """A forced-depth sweep must read the ruler's columns, not the oracle's cost."""
    def boom(*a, **k):
        raise AssertionError("the oracle ran on an EVAL forward")
    monkeypatch.setattr(MORPHTransformer, "_tul_oracle_z_loss", boom)
    m = _model(oracle_z=True).eval()
    inp, lab, layout, _ = _batch()
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    assert "oracle_z" not in out


# ── the term acts ────────────────────────────────────────────────────────────

def test_the_term_is_positive_and_enters_the_loss():
    m = _model(oracle_z=True, oracle_z_steps=3, oracle_z_weight=2.0)
    out = _run(m)
    assert float(out["oracle_z"]) > 0.0
    assert abs(float(out["oracle_z_weighted"]) - 2.0 * float(out["oracle_z"])) < 1e-5
    assert float(out["oracle_z_steps_used"]) == 3.0
    m2 = _model(oracle_z=True, oracle_z_steps=3, oracle_z_weight=2.0)
    out2 = _run(m2)
    # the same model twice: the term is deterministic given the batch and the depths
    assert float(out["oracle_z"]) == float(out2["oracle_z"])


@pytest.mark.parametrize("lr", [0.2, 0.1, 0.05, 0.02])
def test_the_oracle_descends_its_own_readout(lr):
    m = _model(oracle_z=True, oracle_z_steps=4, oracle_z_lr=lr, slot_depth_fixed=4)
    out = _run(m)
    T = int(out["oracle_z_steps_used"])
    assert T == 4, "fixture: the loop must run as many passes as the oracle has steps"
    ls = [float(out[f"oracle_z_l{t}"]) for t in range(T)]
    assert all(b < a for a, b in zip(ls, ls[1:])), (
        f"the oracle's own decoder loss did not fall monotonically over its steps: {ls} — "
        "then the trajectory is not a descent and the per-pass target is noise")


# ── the decoder is NOT trained by it ─────────────────────────────────────────

def _grads(m: MORPHTransformer):
    out = _run(m)
    out["loss"].backward()
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in m.named_parameters()}


def test_the_decoder_gradients_are_identical_with_and_without_the_term():
    """The contract: the oracle READS the decoder and never trains it.

    Two models built from the same seed, so their parameters are byte-identical; the only
    difference is the extra term in the loss. Every `tul_spandec.*` gradient must match
    bit for bit, and at least one core gradient must NOT.
    """
    g_off = _grads(_model())
    g_on = _grads(_model(oracle_z=True, oracle_z_steps=3))
    dec = [n for n in g_off if n.startswith("tul_spandec.")]
    assert dec, "fixture: the span decoder must be built"
    for n in dec:
        a, b = g_off[n], g_on[n]
        assert (a is None) == (b is None), n
        if a is not None:
            assert torch.equal(a, b), (
                f"the oracle trained the span decoder through {n}: max |delta| "
                f"{(a - b).abs().max().item():.3e}")
    core = [n for n in g_off if n.startswith("core.") and g_off[n] is not None]
    assert any(not torch.equal(g_off[n], g_on[n]) for n in core), (
        "the oracle term changed NO core gradient — it reaches the loop through nothing")


def test_the_term_carries_no_edge_to_the_decoder():
    """Autograd's own answer, beside the numerical one above.

    The term is built directly here rather than read off the forward's output, because
    the output exposes it DETACHED (the `spandec_weighted` contract) and a detached tensor
    cannot be asked what it is connected to.
    """
    m = _model(oracle_z=True, oracle_z_steps=2)
    inp, _lab, layout, _ = _batch()
    torch.manual_seed(3)
    x, x0, bg = m._tul_front(inp, layout)
    _xn, _h, depths, _g, db_traj, _gr, _mk = m._tul_core(x, x0, bg, layout, input_ids=inp)
    term = m._tul_oracle_z_loss(db_traj, depths, inp, layout)
    assert float(term.detach()) > 0.0
    dec = [p for n, p in m.named_parameters() if n.startswith("tul_spandec.")]
    g = torch.autograd.grad(term, dec, allow_unused=True, retain_graph=True)
    assert all(x is None for x in g), \
        "the oracle term has a live edge to a span-decoder parameter"
    # and it DOES reach the core, or the assertion above is vacuous
    core = [p for n, p in m.named_parameters() if n.startswith("core.")]
    gc = torch.autograd.grad(term, core, allow_unused=True)
    assert any(x is not None and float(x.abs().sum()) > 0 for x in gc), \
        "the oracle term reaches no core parameter at all"


# ── the refusals ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw,match", [
    (dict(oracle_z=True, spandec=False, spandec_layers=2, spandec_max_tokens=0),
     "requires tul.spandec"),
    (dict(oracle_z=True, oracle_z_steps=0), "oracle_z_steps"),
    (dict(oracle_z=True, oracle_z_lr=0.0), "oracle_z_lr"),
    (dict(oracle_z=True, oracle_z_weight=0.0), "oracle_z_weight"),
    (dict(oracle_z=True, oracle_z_max_tokens=1), "oracle_z_max_tokens"),
    (dict(oracle_z=True, detach_z=True), "detach_z"),
    (dict(oracle_z=True, db_loop=True), "db_loop"),
    (dict(oracle_z_steps=3), "silently ignored"),
])
def test_oracle_z_refusals(kw, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _tul(**kw)
