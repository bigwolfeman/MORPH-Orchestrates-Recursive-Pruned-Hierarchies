"""`model.core_impl` — the Parcae-core swap (morph/model/parcae_core.py).

One test per contract of the arm `slot-mnext-parcae-core`
(`lab/experiments/planned/2026-09-10-arc-slot-mnext-parcae-core.md`):

* `core_impl: morph` is the tree as it was — a pinned gradient hash, and inertness of
  the knob when it is not set;
* `core_impl: parcae` builds, runs the SLOT loop and the PLAIN forward, and the backward
  reaches every Parcae weight;
* the ternary QAT walk skips the Parcae core and ONLY the Parcae core, under the widest
  scope there is;
* the prune / carve walk cannot see it at all;
* the loop's diagonal injection really acts on each pass (fixture sensitivity: neutralise
  it and the loss moves);
* the block refuses, rather than silently drops, every mechanism it cannot carry.

CPU only, tiny config, no tokenizer — the `tests/test_tul_forward.py` fixtures.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
import torch.nn as nn

from morph.model.parcae_core import ParcaeCoreBlock, parcae_core_d_head
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=2, max_depth=3, bptt_depth=2,
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
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _spec(**kw) -> TulLayoutSpec:
    base = dict(seq_len=32, prefix_k=2, max_slots=5, slot_id=4)
    base.update(kw)
    return TulLayoutSpec(**base)


def _batch(spec, B=2, n=90, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == spec.slot_id] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, tokens_through_core=False)
    base.update(kw)
    return TULConfig(**base)


def _model(seed=1234, tul=None, **cfg_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(tul=tul, **cfg_kw))


def _grad_hash(model: MORPHTransformer) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        g = p.grad
        h.update(b"\x00" if g is None else g.detach().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def _plain_step(model: MORPHTransformer, seed=99):
    x = torch.randint(0, V, (2, 32), generator=torch.Generator().manual_seed(7))
    y = torch.randint(0, V, (2, 32), generator=torch.Generator().manual_seed(8))
    model.train()
    torch.manual_seed(seed)
    out = model(x, labels=y)
    out["loss"].backward()
    return out["loss"].detach().clone()


# ── core_impl: morph is the tree as it was ───────────────────────────────────

# The gradient hash of the tiny plain model at `core_impl` default, recorded on this
# tree (torch 2.x CPU, fp32 eager, `use_kernels=False`, ONE CPU thread). It exists so
# that a later edit to `_apply_core_step`, to the config dataclass ordering or to the
# core build cannot move a "morph" model without somebody noticing. If torch itself
# changes a kernel this will move too — before repinning it, prove the delta is NOT from
# `core_impl` by checking out the parent commit and re-running this same function there.
# The step runs under `torch.set_num_threads(1)`: a CPU matmul's reduction order, and so
# the low bits of every gradient, depend on the thread count (the first pin, taken under
# the default thread count, failed under OMP_NUM_THREADS=1 and =2 on 2026-09-10).
_MORPH_GRAD_HASH = "5d2647ba503ba10ea558397ec1c2ba35d21d1c6c1655f087f1268ba0e799901e"


def test_morph_core_gradient_hash_is_pinned():
    m = _model()
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _plain_step(m)
    finally:
        torch.set_num_threads(n_threads)
    assert _grad_hash(m) == _MORPH_GRAD_HASH, (
        "the default (core_impl: morph) model's gradients moved; see the note above "
        "_MORPH_GRAD_HASH before repinning")


def test_core_impl_knob_is_inert_when_unset():
    """Setting `core_impl='morph'` explicitly is the same model, bit for bit."""
    losses, grads, params = [], [], []
    for kw in ({}, {"core_impl": "morph"}):
        m = _model(**kw)
        assert m._core_is_parcae is False
        losses.append(_plain_step(m))
        params.append(torch.cat([p.detach().flatten() for _, p in sorted(m.named_parameters())]))
        grads.append(torch.cat([p.grad.flatten() for _, p in sorted(m.named_parameters())
                                if p.grad is not None]))
    assert torch.equal(losses[0], losses[1])
    assert torch.equal(params[0], params[1]), "the knob perturbed the init RNG"
    assert torch.equal(grads[0], grads[1])


def test_core_impl_rejects_an_unknown_value():
    with pytest.raises(ValueError, match="core_impl"):
        _model(core_impl="parcea")


# ── core_impl: parcae builds and runs both forwards ──────────────────────────

def test_parcae_core_is_built_from_parcae_blocks():
    m = _model(core_impl="parcae")
    assert m._core_is_parcae is True
    assert len(m.core) == 2
    assert all(isinstance(b, ParcaeCoreBlock) for b in m.core)
    # No hyper-connection residual anywhere in the core, and no MORTAR MLP.
    from morph.model.sparsity import MortarLinear
    for name, mod in m.core.named_modules():
        assert "HyperConnection" not in type(mod).__name__, name
        assert not isinstance(mod, MortarLinear), name
    # The prelude and the coda are untouched MORPH blocks.
    from morph.model.mhc import MORPHBlock
    assert all(isinstance(b, MORPHBlock) for b in m.prelude)
    assert all(isinstance(b, MORPHBlock) for b in m.coda)
    n_core = sum(p.numel() for p in m.core.parameters())
    n_all = sum(p.numel() for p in m.parameters())
    print(f"\n[parcae] tiny model: {n_all:,} params total, core {n_core:,} "
          f"({100 * n_core / n_all:.1f}%), d_head "
          f"{parcae_core_d_head(64, 2, 2, None)}")


def test_parcae_plain_forward_and_backward():
    """`slot_layout=None` — the depth sweep and the init probe both use this path."""
    m = _model(core_impl="parcae")
    loss = _plain_step(m)
    assert torch.isfinite(loss)
    missing = [n for n, p in m.core.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"
    assert any(p.grad.abs().sum() > 0 for p in m.core.parameters())


def test_parcae_slot_loop_forward_and_backward():
    x, y, layout, _ = _batch(_spec())
    m = _model(tul=_tul(), core_impl="parcae")
    m.train()
    out = m(x, labels=y, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    missing = [n for n, p in m.core.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"
    assert any(p.grad.abs().sum() > 0 for p in m.core.parameters())


def test_parcae_core_returns_the_carrier_shape_the_loop_expects():
    """The stream collapse/expand is invisible to every caller of `_apply_core_step`."""
    m = _model(core_impl="parcae")
    h = torch.randn(2, 16, m.cfg.hc_streams, m.cfg.d_model)
    e = torch.randn_like(h)
    inj = torch.zeros(m.cfg.n_core, 2, 16, m.cfg.d_model)
    out, ret = m._apply_core_step(h, e, None, None, None, inj_terms=inj)
    assert out.shape == h.shape and ret is None
    # Single stream in the core means every stream of the return is the same tensor.
    assert torch.equal(out[:, :, 0], out[:, :, m.cfg.hc_streams - 1])


# ── precision: the Parcae core, and only it, escapes the ternary walk ────────

def test_ternary_qat_skips_the_parcae_core_and_nothing_else():
    from morph.model.ternary_qat import apply_ternary_qat

    m = _model(core_impl="parcae")
    # "full" is the widest scope there is — backbone + attention + embeddings.
    man = apply_ternary_qat(m, scope="full", scale_mode="norm_match")
    names = set(man["module_names"])
    assert not any(n.startswith("core.") for n in names), \
        f"the Parcae core was ternarised: {sorted(n for n in names if n.startswith('core.'))}"
    # Everything else the scope asks for is still there.
    assert any(n.startswith("prelude.") for n in names)
    assert any(n.startswith("coda.") for n in names)
    assert man["n_modules_ternary"] > 0
    # And the registry itself: no core weight carries a parametrization.
    import torch.nn.utils.parametrize as parametrize
    for name, mod in m.core.named_modules():
        assert not parametrize.is_parametrized(mod, "weight"), f"core.{name} is parametrized"


def test_ternary_qat_still_reaches_a_morph_core():
    """The negative control for the test above — the exclusion is not a blanket one."""
    from morph.model.ternary_qat import apply_ternary_qat

    m = _model()
    man = apply_ternary_qat(m, scope="full", scale_mode="norm_match")
    assert any(n.startswith("core.") for n in man["module_names"]), \
        "a MORPH core must still ternarise, or this suite proves nothing"


def test_prune_and_carve_walks_cannot_see_the_parcae_core():
    from morph.training.pruning import _find_cms_layers as _cms_layers, _find_mortar_layers as _mortar_layers

    m = _model(core_impl="parcae")
    for walk in (_cms_layers, _mortar_layers):
        found = [n for n, _ in walk(m) if n.startswith("core.")]
        assert not found, f"{walk.__name__} reached the Parcae core: {found}"
    # Negative control: the same walks DO reach a MORPH core.
    mm = _model()
    assert any(n.startswith("core.") for n, _ in _mortar_layers(mm))
    assert any(n.startswith("core.") for n, _ in _cms_layers(mm))


# ── the injection acts on every pass ─────────────────────────────────────────

def test_injection_acts_on_the_parcae_loop():
    """Fixture sensitivity: neutralise the diagonal carry and the slot loss moves.

    The Parcae block itself has no source term — Parcae's recurrence is
    `h_t = A h_{t-1} + B e + R(h_{t-1}, e)` and `A`/`B` is `DiagonalInjection`, applied by
    `_apply_core_step` outside the block. If the loss did not move here, the swapped core
    would be running an autonomous map and the arm would be measuring something else.
    """
    x, y, layout, _ = _batch(_spec())
    m = _model(tul=_tul(), core_impl="parcae",
               injection_channels="all", injection_all_dt=0.8, injection_B=True)
    m.eval()
    with torch.no_grad():
        base = m(x, labels=y, slot_layout=layout)["loss"].clone()
        # A -> 1 (pass the state through) and dt -> ~0 (inject nothing): the identity.
        m.injection.log_A.zero_()
        m.injection.log_dt.fill_(-30.0)
        cut = m(x, labels=y, slot_layout=layout)["loss"].clone()
    assert not torch.equal(base, cut), \
        "zeroing the injection left the loss untouched — the carry is not acting"
    assert torch.isfinite(cut)


# ── refusals, not silent drops ───────────────────────────────────────────────

def _blk() -> ParcaeCoreBlock:
    torch.manual_seed(0)
    return ParcaeCoreBlock(d_model=32, d_ff=64, n_heads=2, d_head=16,
                           max_seq_len=64, context_len=64)


@pytest.mark.parametrize("kwargs", [
    {"attn_kwargs": {"tg_allow": torch.zeros(1)}},
    {"next_inject_term": torch.zeros(1, 4, 32)},
    {"ret_state": torch.zeros(1)},
    {"ret_capture": {}},
    {"ret_reset_mask": torch.zeros(1, 4, dtype=torch.bool)},
])
def test_parcae_block_refuses_what_it_cannot_carry(kwargs):
    with pytest.raises(NotImplementedError):
        _blk()(torch.randn(1, 4, 32), **kwargs)


def test_parcae_block_refuses_a_multi_stream_carrier():
    with pytest.raises(ValueError, match="single-stream"):
        _blk()(torch.randn(1, 4, 4, 32))


def test_parcae_block_runs_the_plain_call():
    b = _blk()
    h = torch.randn(1, 4, 32)
    out = b(h, mlp_kwargs={"iter_idx": 1}, pass_idx=1)
    assert out.shape == h.shape and torch.isfinite(out).all()


@pytest.mark.parametrize("kw,match", [
    (dict(core_hca_compress_ratio=16), "core_hca_compress_ratio"),
    (dict(retention=True, retention_sections=("core",)), "retention"),
    (dict(core_init_scale=0.5), "SCSE"),
])
def test_parcae_build_refuses_incompatible_knobs(kw, match):
    with pytest.raises((NotImplementedError, ValueError), match=match):
        _model(core_impl="parcae", **kw)


def test_parcae_d_head_default_matches_morph_attention():
    """The default head width is MORPHAttention's own, so the swap is param-matched."""
    assert parcae_core_d_head(1024, 8, 2, None) == 1024 // (2 * 8)
    assert parcae_core_d_head(1024, 8, 2, 128) == 128
    with pytest.raises(ValueError):
        parcae_core_d_head(1024, 8, 2, 15)


def test_parcae_core_linears_are_plain_and_marked():
    m = _model(core_impl="parcae")
    lins = [(n, mod) for n, mod in m.core.named_modules() if isinstance(mod, nn.Linear)]
    assert lins
    for n, mod in lins:
        assert type(mod) is nn.Linear, n
        assert getattr(mod, "_ternary_exclude", False) is True, n
