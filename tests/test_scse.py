"""Contract for the FULL SCSE method. Spec: ``docs/scse-spec.md`` (arXiv:2607.27656).

One test per invariant in spec section 5, named after it. These are deliberately written to
fail when the code is broken rather than to confirm shapes: the previous SCSE round shipped
a `b_rel` scorer whose "HELD" verdict was carried by a tie-break, and an experiment that
tested a configuration the paper never reports. Each test below names the specific way the
port goes wrong silently.

CPU only, tiny config, no tokenizer -- the fixture style of test_tul_forward.py. The one
CUDA test is marked and skipped without a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer, _SCSE
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


def _build(seed: int = 0, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(**kw))


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _batch(B=2, n=90, seed=0):
    spec = TulLayoutSpec(seq_len=32, prefix_k=2, max_slots=5, slot_id=4)
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == spec.slot_id] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _count_injection_calls(model: MORPHTransformer) -> list[int]:
    """Install a counter on the module that injects the SOURCE into the core loop."""
    n = [0]

    def hook(*_a, **_k):
        n[0] += 1

    model.injection.register_forward_hook(hook)
    return n


# ── S1: OFF is bitwise identical to master ──────────────────────────────────────────────

def test_scse_off_builds_nothing():
    m = _build(0)
    assert m.scse is None, "the default model must not build SCSE"
    assert not [k for k, _ in m.named_parameters() if k.startswith("scse.")]


def test_scse_construction_does_not_move_the_rng():
    """S1. An ON model and an OFF model built at the same seed must share every other weight.

    MORPH decorrelates within ~11 steps of any perturbation, so a switch that shifts the
    init RNG stream silently decorrelates the arm from its control and every CE comparison
    afterwards is measuring the seed, not the mechanism. `_SCSE` is built LAST for exactly
    this reason; this test is what stops someone moving it.
    """
    off, on = _build(0), _build(0, scse_enabled=True)
    p_off, p_on = dict(off.named_parameters()), dict(on.named_parameters())
    extra = set(p_on) - set(p_off)
    assert extra == {"scse.anchor_proj.weight", "scse.init_proj.weight"}, extra
    for k in p_off:
        assert torch.equal(p_off[k], p_on[k]), f"{k} moved when SCSE was switched on"


def test_scse_off_logits_are_unchanged():
    """S1, the other half: OFF must produce the same tensors, not merely the same weights."""
    x = torch.randint(0, V, (2, 32))
    outs = []
    for kw in ({}, {"scse_enabled": False}):
        m = _build(0, **kw)
        m.eval()
        with torch.no_grad():
            outs.append(m(x)["logits"])
    assert torch.equal(outs[0], outs[1])


# ── S2: the anchor is built ONCE and held fixed ─────────────────────────────────────────

def test_anchor_is_built_exactly_once_per_forward():
    """S2. The paper is explicit that `anchor_proj` runs once BEFORE the loop, so that it
    "defines the fixed reference point" rather than being regenerated per iteration. A port
    that rebuilds it inside the loop is a different method that would still train.

    The loop runs up to max_depth=3 iterations, so a per-iteration anchor would count >= 2.
    """
    m = _build(0, scse_enabled=True)
    m.eval()
    n = [0]
    m.scse.anchor_proj.register_forward_hook(lambda *a, **k: n.__setitem__(0, n[0] + 1))
    with torch.no_grad():
        m(torch.randint(0, V, (2, 32)))
    assert n[0] == 1, f"anchor_proj ran {n[0]} times; the anchor must be built once"


# ── S3: the core receives the deviation ONLY ────────────────────────────────────────────

def test_core_is_source_free_under_scse():
    """S3. `DiagonalInjection` is the source path into the core loop. Under SCSE it must not
    run at all (spec D3) -- the source enters through the anchor instead. If it still runs,
    the model is the paper's additive-source BASELINE wearing SCSE's parameter names, which
    is precisely the failure that would make an arm look like a fair test and not be one.
    """
    x = torch.randint(0, V, (2, 32))

    base = _build(0)
    base.eval()
    n_base = _count_injection_calls(base)
    with torch.no_grad():
        base(x)
    assert n_base[0] > 0, "the baseline must inject the source every iteration"

    on = _build(0, scse_enabled=True)
    on.eval()
    n_on = _count_injection_calls(on)
    with torch.no_grad():
        on(x)
    assert n_on[0] == 0, f"SCSE core ran the source injection {n_on[0]} times"


def test_core_gets_no_x0_or_bigram_term_under_scse():
    """S3, second source path: the per-core-layer x0/bigram injection must also be gone, and
    its parameters must therefore receive NO gradient."""
    m = _build(0, scse_enabled=True)
    m.train()
    out = m(torch.randint(0, V, (2, 32)), labels=torch.randint(0, V, (2, 32)))
    out["loss"].backward()
    np_, n_core = m.cfg.n_prelude, m.cfg.n_core
    for i in range(n_core):
        for name, p in m.x0_injects[np_ + i].named_parameters():
            assert p.grad is None, (
                f"x0_injects[{np_ + i}].{name} got a gradient: the core is still being "
                f"fed the source, so this is not SCSE")
    # and the prelude/coda injections MUST still be live -- the signal is not lost, it just
    # stops being re-injected at core depth.
    assert any(p.grad is not None for p in m.x0_injects[0].parameters())


# ── S4: Delta = 0 is an exact fixed point ───────────────────────────────────────────────

def _force_zero_initial_deviation(m: MORPHTransformer) -> None:
    """Make Delta_0 identically zero: init_scale*W_init == anchor_scale*W_anchor."""
    s = m.scse
    with torch.no_grad():
        s.init_proj.weight.copy_(s.anchor_proj.weight * (s.anchor_scale / s.init_scale))


def test_zero_deviation_is_a_fixed_point():
    """S4. This is the paper's 294.37 failure row and the single way this port kills a model:
    with Delta_0 == 0 the core never updates and PPL becomes independent of depth. The
    behaviour is CORRECT (it is the anchor's one-step fixed point); the test pins it so it is
    never discovered by accident in a training curve.

    NOTE, established by sabotage on 2026-08-25: this test does NOT test the mask. Disabling
    the mask entirely leaves it passing, because MORPH's core is zero-preserving, so
    `Delta_1 = 0 + s*G(0) = 0` by the core alone. That is the paper's own point -- "the
    source-centered, zero-preserving core is the primary reparameterization. The mask
    supplies the exact pointwise boundary condition even if the underlying core is not
    zero-preserving." The mask is tested separately, below.
    """
    m = _build(0, scse_enabled=True)
    _force_zero_initial_deviation(m)
    m.eval()
    m._jac_capture = []
    with torch.no_grad():
        m(torch.randint(0, V, (2, 32)))
    caps = m._jac_capture
    m._jac_capture = None
    assert len(caps) >= 2, "expected at least two loop iterations to be captured"
    for c in caps:
        assert c["scse"] is True
        peak = float(c["h"].abs().max())
        assert peak == 0.0, f"iter {c['iter_idx']}: |Delta| = {peak:.3e}, must be exactly 0"


def test_live_model_is_not_at_the_fixed_point():
    """S4's twin, and the one that actually protects the run: an ordinarily initialised model
    must have Delta_0 far off zero, or it trains as a depth-independent feedforward net."""
    m = _build(0, scse_enabled=True)
    m.eval()
    m._jac_capture = []
    with torch.no_grad():
        m(torch.randint(0, V, (2, 32)))
    caps, m._jac_capture = m._jac_capture, None
    d0 = caps[0]["h"]
    assert float(d0.abs().max()) > 1e-4, "Delta_0 is at the frozen fixed point"
    moved = float((caps[1]["h"] - caps[0]["h"]).abs().max())
    assert moved > 0.0, "the deviation did not move: the mask is stuck off"


# ── S5: the mask is PER EXAMPLE ─────────────────────────────────────────────────────────

def test_mask_is_per_example_not_per_position():
    """S5. Listing 1 reduces over every axis except the batch axis and the paper says "The
    per-example mask". A per-POSITION mask is a different method: it would freeze individual
    tokens whose deviation happened to be small, which SCSE never does.
    """
    s = _SCSE(8, step_scale=0.5, anchor_scale=0.1, init_scale=0.1, eps=1e-8, kappa=0.0)
    d = torch.zeros(3, 5, 4, 8)
    d[1, 0, 0, 0] = 1.0          # row 1 active by ONE element
    d[2] = 7.0                   # row 2 fully active
    m = s.gate(d)
    assert m.shape == (3, 1, 1, 1), f"mask shape {tuple(m.shape)} is not per-example"
    assert m.flatten().tolist() == [0.0, 1.0, 1.0]


def test_mask_threshold_is_the_squared_frobenius_norm():
    """S5. `eps` is compared against ||Delta||_F^2, not against ||Delta||_F or a mean."""
    s = _SCSE(4, step_scale=0.5, anchor_scale=0.1, init_scale=0.1, eps=1e-8, kappa=0.0)
    d = torch.zeros(2, 1, 1, 4)
    d[0, 0, 0, 0] = 5e-5         # squared = 2.5e-9  < 1e-8  -> frozen
    d[1, 0, 0, 0] = 2e-4         # squared = 4.0e-8  > 1e-8  -> active
    assert s.gate(d).flatten().tolist() == [0.0, 1.0]


def test_mask_norm_survives_bf16():
    """S5 + D4. In bf16 the sum of squares must still resolve the 1e-8 threshold; accumulating
    in the carrier dtype would round a just-active row down to frozen."""
    s = _SCSE(4, step_scale=0.5, anchor_scale=0.1, init_scale=0.1, eps=1e-8, kappa=0.0)
    d = torch.zeros(1, 64, 4, 4, dtype=torch.bfloat16)
    d[0, 0, 0, 0] = torch.tensor(2e-4, dtype=torch.bfloat16)
    assert s.gate(d).flatten().tolist() == [1.0]


# ── S6 / D8: the reconstruction identity ────────────────────────────────────────────────

def test_h_star_plus_delta0_equals_H0_of_e():
    """S6 + D8. `Delta_0` is formed directly rather than as the literal `H_0(e) - h*`, so the
    identity `h* + Delta_0 == H_0(e)` is the thing that proves the shortcut is the same map.
    Checked in float64 so a real algebra error cannot hide behind float tolerance.
    """
    s = _SCSE(16, step_scale=0.5, anchor_scale=0.1, init_scale=0.1,
              eps=1e-8, kappa=0.0).double()
    e = torch.randn(2, 3, 4, 16, dtype=torch.float64)
    h_star, d0 = s.entry(e)
    H0 = e + s.init_scale * s.init_proj(e)
    assert torch.allclose(h_star + d0, H0, rtol=0, atol=1e-12)
    # and h* is genuinely the paper's anchor, not something else
    assert torch.allclose(h_star, e + s.anchor_scale * s.anchor_proj(e), rtol=0, atol=1e-12)


# ── S8: TUL pad slots enter at Delta = 0 ────────────────────────────────────────────────

def test_pad_slots_enter_at_zero():
    """S8. `gather_valid` zeroes pad slots. Both projections are bias-free (spec D2) so a pad
    gets h* = 0 and Delta_0 = 0 exactly. A bias would give padding a forward effect, which is
    a silent correctness bug: the same text would score differently at a different pad count.
    """
    s = _SCSE(16, step_scale=0.5, anchor_scale=0.1, init_scale=0.1, eps=1e-8, kappa=0.0)
    z = torch.zeros(2, 3, 4, 16)
    h_star, d0 = s.entry(z)
    assert float(h_star.abs().max()) == 0.0
    assert float(d0.abs().max()) == 0.0


# ── S9: the now-dead injection parameters do not break the optimizer ────────────────────

def test_optimizer_step_runs_with_dead_injection_params():
    """S9. SCSE leaves `injection` and the core `x0_injects` with no gradient. A real
    optimizer step must still run: a port that trains for one step and then dies on
    `NoneType` in the update is not shippable, and this is cheaper to learn here.
    """
    m = _build(0, scse_enabled=True)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    out = m(torch.randint(0, V, (2, 32)), labels=torch.randint(0, V, (2, 32)))
    out["loss"].backward()
    assert m.injection.log_A.grad is None, "DiagonalInjection should be dead under SCSE"
    before = m.scse.anchor_proj.weight.detach().clone()
    opt.step()
    assert m.scse.anchor_proj.weight.grad is not None, "the anchor must be LEARNED"
    assert not torch.equal(before, m.scse.anchor_proj.weight), "the anchor did not update"


# ── S10: both loop bodies are ported ────────────────────────────────────────────────────

def test_tul_slot_path_uses_scse():
    """S10. The arms run `--config-name tul_a1`, which goes through `_tul_core`, NOT
    `_core_region`. Porting only the token path would produce an arm that reports SCSE in its
    config and runs the baseline recurrence.
    """
    x, y, layout, _ = _batch()
    tul = TULConfig(prefix_k=2, slot_id=4)

    base = _build(1, tul=tul)
    base.eval()
    n_base = _count_injection_calls(base)
    with torch.no_grad():
        base(x, labels=y, slot_layout=layout)
    assert n_base[0] > 0

    on = _build(1, tul=tul, scse_enabled=True)
    on.eval()
    n_on = _count_injection_calls(on)
    with torch.no_grad():
        out = on(x, labels=y, slot_layout=layout)
    assert n_on[0] == 0, f"the TUL slot path injected the source {n_on[0]} times"
    assert torch.isfinite(out["loss"])


def test_tul_and_token_paths_both_reach_the_anchor():
    """S10. Both loop bodies must build the anchor exactly once (S2 on each path)."""
    x, y, layout, _ = _batch()
    m = _build(1, tul=TULConfig(prefix_k=2, slot_id=4), scse_enabled=True)
    m.eval()
    n = [0]
    m.scse.anchor_proj.register_forward_hook(lambda *a, **k: n.__setitem__(0, n[0] + 1))
    with torch.no_grad():
        m(x, labels=y, slot_layout=layout)
    assert n[0] == 1, f"TUL path built the anchor {n[0]} times"


# ── construction guards ─────────────────────────────────────────────────────────────────

def test_scse_and_stage1_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build(0, scse_enabled=True, core_init_scale=0.1)


def test_scse_rejects_the_core_gain_governor():
    with pytest.raises(ValueError, match="core_gain_clip"):
        _build(0, scse_enabled=True, core_gain_clip=1.5)


def test_scse_rejects_a_coreless_model():
    with pytest.raises(ValueError, match="n_core"):
        _build(0, scse_enabled=True, n_core=0)


def test_kappa_zero_builds_no_cond_proj():
    """Listing 1's caption: "Set cond_proj=None, kappa=0, and leak=0 for SCSE"."""
    m = _build(0, scse_enabled=True)
    assert m.scse.cond_proj is None
    on = _build(0, scse_enabled=True, scse_kappa=0.05)
    assert on.scse.cond_proj is not None


# ── S7: the forcing bias is exactly zero on the REAL model ──────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the 5090")
def test_forcing_bias_is_zero_on_the_real_model():
    """S7. `b_t(e) = T_t(0; e)` must be EXACTLY zero on the shipped 286M configuration, in
    fp32 and under the bf16 autocast training actually uses. This is the property the whole
    method is built on, and the tiny CPU fixture above does not exercise ternary QAT, GLA
    retention or six core blocks.
    """
    import sys
    sys.path.insert(0, "lab/divergence")
    from drift_probe import build  # noqa: E402

    _cfg, model, _x, _y, _layout = build(
        # base.yaml ships the slot-loop gain constraint ON; under SCSE the carrier is the
        # DEVIATION and the constraint RAISES by contract, so the SCSE arm turns it off.
        "tul_a1", ["training.batch_size=2", "model.use_kernels=false",
                   "model.scse_enabled=true", "model.slot_gain_lambda=0",
                   "model.slot_cot_clip=0"])
    model.eval()
    root = getattr(model, "_orig_mod", model)
    c = root.cfg
    assert root.scse is not None, "the SCSE arm did not build the module"
    B, S = 2, 8
    zero_delta = torch.zeros(B, S, c.hc_streams, c.d_model, device="cuda")
    anchor = torch.randn_like(zero_delta)          # h* is arbitrary; b_t must not depend on it

    for label, ctx in (("fp32", torch.autocast("cuda", enabled=False)),
                       ("bf16", torch.autocast("cuda", dtype=torch.bfloat16))):
        with torch.no_grad(), ctx:
            g_out, _ = root._apply_core_step(
                root.scse.recurrent_input(zero_delta, anchor), None, None, None, None,
                ret_state=None, iter_idx=0, inj_terms=None, source_free=True)
            nxt = root.scse.update(zero_delta, g_out)
        # Assert the CORE's own output first. Checking only the masked step would be
        # near-tautological: `gate(0) = 0` zeroes the product whatever the core returned, so
        # that assertion can fail only on a non-finite value. This is the real claim.
        assert float(g_out.abs().max()) == 0.0, (
            f"G_theta(0) != 0 in {label}: peak |stack(0)| = {float(g_out.abs().max()):.3e}")
        assert float(nxt.abs().max()) == 0.0, (
            f"b_t != 0 in {label}: the anchor is not a one-step fixed point")


# ── S6: the loop exit reconstructs the ABSOLUTE state ───────────────────────────────────

def test_core_region_reconstructs_the_absolute_state():
    """S6, token path. With `step_scale = 0` the loop leaves Delta at Delta_0, so the region
    MUST return `h* + Delta_0 == H_0(e)`.

    This is the one failure mode nothing else in this file catches: deleting the
    reconstruction hands the DEVIATION to the coda -- a tensor about 20x smaller that still
    trains, still gives a finite loss, and is silently a different model.
    """
    from morph.model.tul import gather_valid  # noqa: F401  (kept next to its twin below)

    m = _build(0, scse_enabled=True, scse_step_scale=0.0)
    m.eval()
    c = m.cfg
    x = torch.randn(2, 8, c.hc_streams, c.d_model)
    with torch.no_grad():
        out = m._core_region(x, x, None)
        e = m.input_norm(x)
        h_star, d0 = m.scse.entry(e)
    assert torch.allclose(out, h_star + d0, rtol=0, atol=1e-6), "S6 broken on the token path"
    assert not torch.allclose(out, d0, rtol=0, atol=1e-4), "the region returned the deviation"


def test_tul_core_reconstructs_the_absolute_state():
    """S6, TUL slot path -- the one the arms actually run."""
    from morph.model.tul import gather_valid

    x, y, layout, _ = _batch()
    m = _build(1, tul=TULConfig(prefix_k=2, slot_id=4),
               scse_enabled=True, scse_step_scale=0.0)
    m.eval()
    c = m.cfg
    carrier = torch.randn(x.shape[0], x.shape[1], c.hc_streams, c.d_model)
    with torch.no_grad():
        _xn, h, _d, _g, *_ = m._tul_core(carrier, carrier, None, layout)
        e = gather_valid(m.input_norm(carrier), layout.slot_index, layout.slot_valid)
        h_star, d0 = m.scse.entry(e)
    assert torch.allclose(h, h_star + d0, rtol=0, atol=1e-6), "S6 broken on the TUL path"
    assert not torch.allclose(h, d0, rtol=0, atol=1e-4), "the slot path returned the deviation"


def test_mask_freezes_a_below_threshold_deviation():
    """S4, the part the fixed-point test cannot see: the MASK itself.

    A deviation that is non-zero but below the threshold must produce EXACTLY no update,
    even though `G(Delta) != 0` there. This matters more than it looks: RMSNorm divides by
    the RMS, so a near-zero deviation is not mapped to a near-zero output -- it is amplified
    by roughly `1/sqrt(eps_norm)`. Without the mask, a deviation that decays into the
    threshold region gets blown back out of it.

    Constructed by forcing Delta_0 = 0 and then perturbing `init_proj` by 1e-9, which puts
    ||Delta_0||_F^2 far below `scse_eps = 1e-8` while leaving it non-zero.
    """
    m = _build(0, scse_enabled=True)
    _force_zero_initial_deviation(m)
    with torch.no_grad():
        m.scse.init_proj.weight.add_(1e-9)
    m.eval()
    m._jac_capture = []
    with torch.no_grad():
        m(torch.randint(0, V, (2, 32)))
    caps, m._jac_capture = m._jac_capture, None

    d0 = caps[0]["h"]
    nsq = d0.float().pow(2).sum(dim=tuple(range(1, d0.dim())))
    assert float(nsq.max()) > 0.0, "the perturbation did not take: Delta_0 is exactly zero"
    assert float(nsq.max()) < 1e-8, f"Delta_0 is not below threshold: ||.||^2 = {nsq.max():.3e}"
    for c in caps[1:]:
        assert torch.equal(c["h"], d0), (
            f"iter {c['iter_idx']}: a sub-threshold deviation moved, so the mask is not "
            f"being applied (max move {float((c['h'] - d0).abs().max()):.3e})")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the 5090")
def test_real_model_loop_is_source_free_and_anchored():
    """S3 + S10 on the SHIPPED configuration, not the tiny CPU fixture.

    The CPU tests prove the recurrence on a 0.3M model with retention off, no ternary QAT
    and two core blocks. This runs one real `tul_a1` forward (286M, GLA retention, six core
    blocks, TUL slot path) and checks the two things that decide whether the arm is SCSE at
    all: the source injection never fires inside the loop, and the anchor is built once.
    """
    import sys
    sys.path.insert(0, "lab/divergence")
    from drift_probe import build  # noqa: E402

    _cfg, model, x, y, layout = build(
        # base.yaml ships the slot-loop gain constraint ON; under SCSE the carrier is the
        # DEVIATION and the constraint RAISES by contract, so the SCSE arm turns it off.
        "tul_a1", ["training.batch_size=2", "model.use_kernels=false",
                   "model.scse_enabled=true", "model.slot_gain_lambda=0",
                   "model.slot_cot_clip=0"])
    root = getattr(model, "_orig_mod", model)
    root.eval()
    n_inj, n_anchor = [0], [0]
    root.injection.register_forward_hook(lambda *a, **k: n_inj.__setitem__(0, n_inj[0] + 1))
    root.scse.anchor_proj.register_forward_hook(
        lambda *a, **k: n_anchor.__setitem__(0, n_anchor[0] + 1))
    with torch.no_grad():
        out = model(x, labels=y, slot_layout=layout)
    assert n_inj[0] == 0, f"the REAL loop injected the source {n_inj[0]} times"
    assert n_anchor[0] == 1, f"the REAL loop built the anchor {n_anchor[0]} times"
    assert torch.isfinite(out["loss"]), "the real SCSE forward produced a non-finite loss"


# ── the recurrence FORM itself (audit 2026-08-25) ───────────────────────────────────────

def test_update_subtracts_the_deviation_from_the_stack_output():
    """`G(D) = stack(D) - D`, NOT `stack(D)`. This is the bug the first port shipped.

    MORPH's core blocks carry their own residual, so `D + s*stack(D)` applies the residual
    twice and gains `(1+s)` per iteration on top of whatever the update does -- measured at
    1.414x per iteration against the corrected form's 0.923x. The paper's `G_theta` has no
    top-level identity: if it did, its own tuned-adapter formula would reach `1.5^48` at the
    T = 48 it evaluates at.

    Pinned in closed form so the subtraction cannot be quietly removed again.
    """
    s = _SCSE(8, step_scale=0.5, anchor_scale=0.1, init_scale=0.1, eps=1e-8, kappa=0.0)
    d = torch.randn(2, 3, 4, 8) * 3.0        # well above the mask threshold
    stack_out = torch.randn(2, 3, 4, 8)
    got = s.update(d, stack_out)
    assert torch.allclose(got, d + 0.5 * (stack_out - d), rtol=0, atol=1e-6)
    assert not torch.allclose(got, d + 0.5 * stack_out, rtol=0, atol=1e-4), (
        "update() is computing the DOUBLED form D + s*stack(D)")
    # s = 1 must recover MORPH's own core map in deviation coordinates. That is the check
    # that the damping interpretation of `s` is real and not a rationalisation.
    s1 = _SCSE(8, step_scale=1.0, anchor_scale=0.1, init_scale=0.1, eps=1e-8, kappa=0.0)
    assert torch.allclose(s1.update(d, stack_out), stack_out, rtol=0, atol=1e-6)


# ── gaps found by the 2026-08-25 audit: three sabotages that survived the whole suite ───

def test_the_core_receives_the_BARE_deviation():
    """Audit sabotage A: `recurrent_input` returning `delta + h_star` passed all 25 tests.

    The old S3 tests counted `DiagonalInjection` calls and `x0_injects` gradients -- MORPH's
    LEGACY source paths -- and nothing looked at the tensor actually handed to the blocks.
    The anchor is threaded through the loop as the second argument for the kappa path, so
    smuggling it into the recurrent input was invisible. This reads the real tensor.
    """
    m = _build(0, scse_enabled=True)
    m.eval()
    seen: list[torch.Tensor] = []
    m.core[0].register_forward_pre_hook(
        lambda _mod, args: seen.append(args[0].detach().clone()))
    m._jac_capture = []
    with torch.no_grad():
        m(torch.randint(0, V, (2, 32)))
    caps, m._jac_capture = m._jac_capture, None
    assert len(seen) >= len(caps) >= 2, f"{len(seen)} block inputs, {len(caps)} captures"
    for c, inp in zip(caps, seen):
        assert torch.equal(inp, c["h"]), (
            f"iter {c['iter_idx']}: the first core block received something that is NOT the "
            f"bare deviation (max diff {float((inp - c['h']).abs().max()):.3e})")


def test_mask_freezes_a_below_threshold_deviation_on_the_TUL_path():
    """Audit sabotage B: deleting the gate from `_tul_core` ALONE passed all 25 tests.

    The only in-loop mask test ran the token path, and the arms run `tul_a1`, which goes
    through `_tul_core`. The shipped path was unprotected. (The recurrence is now a single
    `_SCSE.update`, so there is one place to break rather than three -- but the path still
    needs its own test, because the loop body chooses whether to call it.)
    """
    x, y, layout, _ = _batch()
    m = _build(1, tul=TULConfig(prefix_k=2, slot_id=4), scse_enabled=True)
    _force_zero_initial_deviation(m)
    with torch.no_grad():
        m.scse.init_proj.weight.add_(1e-9)
    m.eval()
    m._jac_capture = []
    with torch.no_grad():
        m(x, labels=y, slot_layout=layout)
    caps, m._jac_capture = m._jac_capture, None
    d0 = caps[0]["h"]
    nsq = d0.float().pow(2).sum(dim=tuple(range(1, d0.dim())))
    assert float(nsq.max()) > 0.0, "the perturbation did not take"
    assert float(nsq.max()) < 1e-8, f"not below threshold: {float(nsq.max()):.3e}"
    for c in caps[1:]:
        assert torch.equal(c["h"], d0), (
            f"iter {c['iter_idx']}: a sub-threshold deviation moved on the TUL path")


def test_h_star_is_aligned_with_the_ORIGINAL_batch_order():
    """Audit sabotage C: `x = h_star[perm] + x` passed the entire 384-test suite.

    Every other test evaluates with uniform depths, where the depth-sort permutation is the
    identity and the bug is invisible. In training the depths are Poisson, and this class of
    error silently adds another SAMPLE'S anchor to each deviation. Depths are pinned here
    rather than sampled so the permutation is guaranteed non-trivial.
    """
    m = _build(0, scse_enabled=True, scse_step_scale=0.0)
    m.eval()
    pinned = torch.tensor([1, 3, 2, 3])
    m._sample_depths = lambda B, dev: pinned.to(dev)     # non-identity sort order
    m.training = True                                    # take the sampled-depth branch
    c = m.cfg
    torch.manual_seed(7)
    x = torch.randn(4, 8, c.hc_streams, c.d_model)
    with torch.no_grad():
        out = m._core_region(x, x, None)
        e = m.input_norm(x)
        h_star, d0 = m.scse.entry(e)
    # step_scale = 0 freezes Delta at Delta_0 for EVERY sample whatever its depth, so the
    # region must return h* + Delta_0 row for row.
    assert torch.allclose(out, h_star + d0, rtol=0, atol=1e-5), (
        f"anchor/deviation batch misalignment: max diff "
        f"{float((out - (h_star + d0)).abs().max()):.3e}")


def test_the_core_receives_the_BARE_deviation_on_the_TUL_path():
    """Re-audit sabotage: feeding `h_in + e_in` to the blocks in `_tul_core` ALONE passed all
    29 tests. S13 hooks `core[0]` but drives only the token path, and the arms run
    `tul_a1` -> `_tul_core`. This is the TUL twin, exactly as S14 is the TUL twin of the mask
    test. Same lesson twice: a guard on the token path is not a guard on the shipped path.
    """
    x, y, layout, _ = _batch()
    m = _build(1, tul=TULConfig(prefix_k=2, slot_id=4), scse_enabled=True)
    m.eval()
    seen: list[torch.Tensor] = []
    m.core[0].register_forward_pre_hook(
        lambda _mod, args: seen.append(args[0].detach().clone()))
    m._jac_capture = []
    with torch.no_grad():
        m(x, labels=y, slot_layout=layout)
    caps, m._jac_capture = m._jac_capture, None
    assert len(seen) >= len(caps) >= 2, f"{len(seen)} block inputs, {len(caps)} captures"
    for c, inp in zip(caps, seen):
        assert torch.equal(inp, c["h"]), (
            f"iter {c['iter_idx']}: the TUL slot path fed the core something that is NOT the "
            f"bare deviation (max diff {float((inp - c['h']).abs().max()):.3e})")


def test_update_gradients_are_correct_not_merely_present():
    """Re-audit finding: `stack_out - delta.detach()` leaves the FORWARD bit-identical and the
    BACKWARD wrong, and passed every test. Every other test here asserts forward values or
    gradient EXISTENCE, never gradient VALUES.

    Closed in closed form rather than by autograd comparison, because the derivative of
    `D + m*s*(S - D)` is exact and known: with the mask active, `d/dD = (1 - s)` and
    `d/dS = s`. A detached `D` inside the update makes the first `1` instead of `1 - s`.
    Run in float64 so the equality is a real check and not a tolerance.
    """
    s = _SCSE(4, step_scale=0.5, anchor_scale=0.1, init_scale=0.1,
              eps=1e-8, kappa=0.0).double()
    d = (torch.randn(2, 3, 4, 4, dtype=torch.float64) * 3.0).requires_grad_(True)
    so = torch.randn(2, 3, 4, 4, dtype=torch.float64).requires_grad_(True)
    s.update(d, so).sum().backward()
    assert torch.allclose(d.grad, torch.full_like(d.grad, 1.0 - 0.5), rtol=0, atol=1e-12), (
        "d(update)/d(delta) is not (1 - s): the deviation is detached or double-counted")
    assert torch.allclose(so.grad, torch.full_like(so.grad, 0.5), rtol=0, atol=1e-12), (
        "d(update)/d(stack_out) is not s")
