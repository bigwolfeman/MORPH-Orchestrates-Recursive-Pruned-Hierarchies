"""Contract for SCSE Stage 1 — the core loop's initial deviation (arXiv:2607.27656).

Plan: `.agents/notes/proposed/architecture/2026-08-24-scse-source-centered-core-loop.md`.

Three things have to be true or the port is unsafe, and each one has failed somewhere in
this codebase's history:

1. **OFF is bit-identical.** `core_init_scale = 0.0` must build NO parameters, draw NO RNG,
   and produce the same tensors as `h = e.clone()` did. A "flag that is off" which still
   shifts the RNG stream would silently decorrelate every arm from its control -- MORPH
   decorrelates within 11 steps of any perturbation.
2. **ON actually moves Delta_0 off zero.** The entire structural argument for the port is
   that `Delta_0 = 0` makes the whole trajectory the propagated forcing response. A port
   that leaves `Delta_0 = 0` changes nothing while looking like it changed something.
3. **`G_theta(0) = 0` numerically.** Acceptance criterion 3 of the plan. The Stage 3
   zero-deviation mask is only sound if a zero carrier through the REAL core returns zero.
   A code audit found no additive output offset; the plan states in writing that the audit
   is necessary and not sufficient, and that only a forward pass proves it. This is that
   forward pass.

CPU only, tiny config, no tokenizer -- the fixture style of test_core_jacobian.py.
"""

from __future__ import annotations

import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer, _CloneInit, _SCSEInit

V = 64


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


def _build(seed: int, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(**kw))


# ── 1. OFF is bit-identical, and costs no RNG ───────────────────────────────────────────

def test_off_builds_no_parameters_and_draws_no_rng():
    """The default model must be byte-identical to one built before core_init existed.

    Checked by building at the same seed with the scale explicitly 0.0 and comparing every
    parameter bitwise. If `_CloneInit` ever gains a parameter this fails immediately.
    """
    a = _build(0)
    b = _build(0, core_init_scale=0.0)
    assert isinstance(a.core_init, _CloneInit)
    assert list(a.core_init.parameters()) == [], "the OFF path must hold no parameters"

    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert pa.keys() == pb.keys()
    for k in pa:
        assert torch.equal(pa[k], pb[k]), f"{k} differs between default and explicit 0.0"


def test_off_leaves_the_rng_stream_where_the_baseline_left_it():
    """Building the OFF model must consume exactly as much RNG as the old code did.

    The old code had no module here at all, so the post-build RNG state is the invariant:
    a model built with the SCSE projection must consume MORE, and the OFF model exactly
    the same. Both halves are asserted so the test fails if `_SCSEInit` is ever made
    RNG-free (which would make arms silently comparable when they are not).
    """
    torch.manual_seed(0)
    MORPHTransformer(_tiny())
    off_state = torch.random.get_rng_state()

    torch.manual_seed(0)
    MORPHTransformer(_tiny(core_init_scale=0.0))
    assert torch.equal(torch.random.get_rng_state(), off_state)

    torch.manual_seed(0)
    MORPHTransformer(_tiny(core_init_scale=0.1))
    assert not torch.equal(torch.random.get_rng_state(), off_state), (
        "the SCSE projection must draw RNG; if it stops, every arm built after it "
        "silently shares a stream with its control")


def test_off_forward_equals_the_old_clone_behaviour():
    """`h_0 = e` exactly, which is what `h = e.clone()` produced."""
    m = _build(0).eval()
    e = torch.randn(2, 5, 64)
    assert torch.equal(m.core_init(e), e)
    assert m.core_init(e) is not e, "must be a copy, not an alias into the carrier"


def test_off_model_logits_are_unchanged_by_the_refactor():
    """Two OFF models at one seed give bitwise-equal logits on the real forward path.

    This is the end-to-end version of the claim: the call-site change from `e.clone()` to
    `self.core_init(e)` moved no numbers.
    """
    ids = torch.randint(0, V, (2, 16))
    a, b = _build(0).eval(), _build(0, core_init_scale=0.0).eval()
    with torch.no_grad():
        la, lb = a(ids)["logits"], b(ids)["logits"]
    assert torch.equal(la, lb)


# ── 2. ON moves Delta_0 off zero ────────────────────────────────────────────────────────

def test_on_makes_the_initial_deviation_nonzero():
    """The whole point of Stage 1. `Delta_0 = h_0 - e` must be a real displacement."""
    m = _build(0, core_init_scale=0.1).eval()
    assert isinstance(m.core_init, _SCSEInit)
    e = torch.randn(2, 5, 64)
    with torch.no_grad():
        d = m.core_init(e) - e
    rel = float(d.norm() / e.norm())
    assert rel > 1e-3, f"Delta_0 is effectively zero (rel={rel:.2e}); Stage 1 did nothing"


def test_on_scale_controls_the_deviation_size():
    """`Delta_0` must scale linearly with `core_init_scale` at a fixed projection.

    Catches a scale that is read but never applied -- the deviation would then be the same
    at 0.1 and 0.5 and no sweep over it could mean anything.
    """
    e = torch.randn(2, 5, 64)
    m1 = _build(0, core_init_scale=0.1).eval()
    m5 = _build(0, core_init_scale=0.5).eval()
    with torch.no_grad():
        d1 = (m1.core_init(e) - e).norm()
        d5 = (m5.core_init(e) - e).norm()
    assert torch.allclose(d5, d1 * 5.0, rtol=1e-4), f"{float(d1)=} {float(d5)=}"


def test_on_changes_the_model_output():
    """A Delta_0 that never reaches the loss is a decoration. This proves it propagates."""
    ids = torch.randint(0, V, (2, 16))
    off, on = _build(0).eval(), _build(0, core_init_scale=0.1).eval()
    with torch.no_grad():
        lo, ln = off(ids)["logits"], on(ids)["logits"]
    assert not torch.allclose(lo, ln), "Stage 1 did not reach the output"


# ── 3. the owed gate: G_theta(0) = 0 through the REAL core ──────────────────────────────

def test_zero_carrier_through_the_real_core_returns_zero():
    """Acceptance criterion 3 of the SCSE plan, and it is a NUMERICAL check by design.

    The plan's Stage 3 audit read every additive parameter in the core and argued each one
    is either a softmax logit or a linear mixing coefficient, so a zero carrier stays zero.
    That audit is necessary and NOT sufficient -- it cannot see a term the reader missed.
    Only a forward pass on a zero carrier proves it, and Stage 3's zero-deviation mask is
    unsound without it.

    The core map is called through `_apply_core_step`, which is the exact training-path
    `f_theta`, with the source injections zeroed: `G_theta` is the core's own transition,
    not the transition plus the input drive.
    """
    m = _build(0, core_init_scale=0.1).eval()
    d, n = m.cfg.d_model, m.cfg.hc_streams
    B, S = 2, 6

    # The carrier is [B, S, n_streams, C] -- HyperConnection keeps n streams of width
    # d_model -- and the injection term is [n_core, B, S, C]. Captured from a live
    # forward rather than assumed; a wrong shape here would make the gate vacuous.
    zero_h = torch.zeros(B, S, n, d)
    zero_e = torch.zeros(B, S, n, d)
    zero_inj = torch.zeros(m.cfg.n_core, B, S, d)

    with torch.no_grad():
        out, _ = m._apply_core_step(zero_h, zero_e, None, None, None,
                                    ret_state=None, iter_idx=0, inj_terms=zero_inj)

    peak = float(out.abs().max())
    assert peak < 1e-6, (
        f"G_theta(0) != 0: a zero carrier came back with max |out| = {peak:.3e}. "
        f"The core carries an additive offset the Stage 3 code audit did not find, and "
        f"the zero-deviation mask MUST NOT be enabled until it is located.")


def test_the_zero_carrier_gate_would_catch_an_injected_offset():
    """The gate above must FAIL when the property it checks is broken.

    Without this, a test that passes proves only that it ran. An additive output bias is
    planted in the last core block and the same assertion is required to fire.
    """
    m = _build(0, core_init_scale=0.1).eval()
    d, n = m.cfg.d_model, m.cfg.hc_streams
    B, S = 2, 6

    blk = m.core[-1]
    handle = blk.register_forward_hook(
        lambda mod, inp, out: (out[0] + 1.0, *out[1:]) if isinstance(out, tuple)
        else out + 1.0)
    try:
        with torch.no_grad():
            out, _ = m._apply_core_step(torch.zeros(B, S, n, d), torch.zeros(B, S, n, d),
                                        None, None, None, ret_state=None, iter_idx=0,
                                        inj_terms=torch.zeros(m.cfg.n_core, B, S, d))
        assert float(out.abs().max()) > 1e-6, (
            "the planted offset did not reach the output, so this sabotage does not "
            "exercise the gate and the gate above is unproven")
    finally:
        handle.remove()


# ── the same gate, on the REAL model ────────────────────────────────────────────────────

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@requires_cuda
def test_zero_carrier_returns_zero_on_the_REAL_model():
    """The tiny fixture above runs `retention=False`. The shipped model does not.

    The Stage 3 audit was written about the 286.1M `tul_a1` model with GLA retention, six
    core blocks and ternary QAT live. A gate that only ever ran on a 0.3M model with
    retention off would not have tested the thing the audit claimed. This runs it on the
    real configuration, in fp32 AND under the bf16 autocast the training path actually
    uses, because a term that cancels in one precision need not cancel in the other.

    Measured 2026-08-25: peak |out| = 0.000e+00 in both, exactly.
    """
    import sys
    sys.path.insert(0, "lab/divergence")
    from drift_probe import build  # noqa: E402

    _cfg, model, _x, _y, _layout = build(
        "tul_a1", ["training.batch_size=2", "model.use_kernels=false"])
    model.eval()
    root = getattr(model, "_orig_mod", model)
    c = root.cfg
    B, S = 2, 8
    zh = torch.zeros(B, S, c.hc_streams, c.d_model, device="cuda")
    ze = torch.zeros_like(zh)
    zi = torch.zeros(c.n_core, B, S, c.d_model, device="cuda")

    for label, ctx in (("fp32", torch.autocast("cuda", enabled=False)),
                       ("bf16", torch.autocast("cuda", dtype=torch.bfloat16))):
        with torch.no_grad(), ctx:
            out, _ = root._apply_core_step(zh, ze, None, None, None, ret_state=None,
                                           iter_idx=0, inj_terms=zi)
        peak = float(out.abs().max())
        assert peak < 1e-6, (
            f"G_theta(0) != 0 on the REAL model in {label}: peak |out| = {peak:.3e}. "
            f"The zero-deviation mask (Stage 3) MUST NOT be enabled.")
