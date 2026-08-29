"""GL1 gates — the gist-loop baseline: mask + gradient-carrying tap write + NO loop.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_tul_gl1.py -v

Decision note: ``.agents/notes/proposed/architecture/2026-08-29-gist-loop.md``.

Two properties carry this arm and both are gated here:

1. **The write carries gradient.** Nothing on the slot path is detached. A later span's
   cross-entropy must reach the boundary tap ``W_sent`` and the earlier span's
   embeddings, and it must do so THROUGH a slot. Every other arm in the campaign had to
   protect against exactly this gradient because it unrolled an iterated map; with
   ``n_core == 0`` there is no iterated map, so the thing that was fatal elsewhere is
   the objective here.
2. **The slot is the only route.** Sever the slot channel and that gradient must be
   EXACTLY zero — spec §7 test T3, whose machinery is imported from
   ``test_tg_restrict.py`` rather than copied, so the two arms cannot drift apart.

The T3 tests below run with ``retention=True`` (the arm's real setting), where the
existing T3 test runs with it off. GLA is a recurrent scan and would be a second,
mask-invisible channel between spans if ``tg_reset_mask`` did not segment it — so
running the falsifier with retention ON is the version that actually protects GL1.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from test_tg_restrict import (          # noqa: E402  (pytest puts tests/ on sys.path)
    _finite_logit_sum,
    _find_probe_positions,
    _severed_forward,
)

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10
SLOT_ID = 4


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _spec(**kw) -> TulLayoutSpec:
    base = dict(seq_len=48, prefix_k=2, max_slots=12, slot_id=SLOT_ID)
    base.update(kw)
    return TulLayoutSpec(**base)


def _batch(B: int = 2, n: int = 90, seed: int = 0, spec=None):
    spec = spec or _spec()
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == SLOT_ID] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _tul(**kw) -> TULConfig:
    """The GL1 arm's own TUL settings."""
    base = dict(prefix_k=2, slot_id=SLOT_ID, slot_seed="boundary", tg_restrict=True,
                emit_weight=0.0, plast_weight=1.0, token_state_dropout=0.0,
                sigreg_lambda=0.02, sigreg_slices=64, eval_ablations=True)
    base.update(kw)
    return TULConfig(**base)


def _cfg(tul=None, n_core=0, retention=True, **kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256,
        context_len=256, n_prelude=2, n_core=n_core, n_coda=2, mean_depth=2, max_depth=2,
        bptt_depth=1, channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=8, bigram_hash_vocab=V,
        use_kernels=False, hc_use_kernel=False, dropout=0.0,
        retention=retention, retention_layers=(1,), retention_chunk=8,
        retention_carry=True,
        tul=tul if tul is not None else _tul(),
    )
    base.update(kw)
    return MORPHConfig(**base)


def _model(seed: int = 0, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_cfg(**kw))


# ── 1. the coreless slot path exists at all ──────────────────────────────────

def test_the_coreless_slot_path_runs_under_the_restriction():
    """Before this arm, ``n_core == 0`` with a slot layout raised 'stack expects a
    non-empty TensorList' on the per-core-layer injection stack. That was the whole
    reason the mask+tap+no-loop cell had never been run."""
    x, y, lay, _ = _batch()
    m = _model()
    m.train()
    out = m(x, y, slot_layout=lay)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert float(m.tul.W_sent.weight.grad.abs().sum()) > 0


def test_the_slot_state_is_the_prelude_output_at_the_slot_position():
    """No loop means the written state IS ``input_norm(prelude)`` gathered at the slot —
    the same boundary norm the coreless TOKEN path applies. If a loop ever comes back,
    this stops holding and the arm is no longer the cell it claims to be."""
    x, _y, lay, _ = _batch()
    m = _model()
    m.eval()
    with torch.no_grad():
        xf, x0, bg = m._tul_front(x, lay)
        xn, h_slots, depths, g_traj = m._tul_core(xf, x0, bg, lay)
        want = m.input_norm(xf)
    assert torch.equal(xn, want)
    assert g_traj is None and int(depths.abs().sum()) == 0, \
        "a coreless arm reported loop iterations"
    for b in range(x.shape[0]):
        for i in torch.nonzero(lay.slot_valid[b]).flatten().tolist():
            pos = int(lay.slot_index[b, i])
            assert torch.equal(h_slots[b, i], xn[b, pos])


def test_the_loop_only_configurations_refuse_at_n_core_zero():
    from morph.model.tul import TULGateConfig
    m = _model()
    x, y, lay, _ = _batch()
    with pytest.raises(RuntimeError, match="halt=True needs a core loop"):
        m.tul_forward_halt(x, y, lay)
    gated = _tul(gate=TULGateConfig(k_max=8, k_decode_max=8))
    mg = _model(tul=gated)
    with pytest.raises(NotImplementedError, match="tul.gate has no defined meaning"):
        mg(x, y, slot_layout=lay)


# ── 2. the gradient-carrying write ───────────────────────────────────────────

def test_the_slot_state_carries_gradient_and_nothing_on_the_path_is_detached():
    x, _y, lay, _ = _batch()
    m = _model()
    m.train()
    xf, x0, bg = m._tul_front(x, lay)
    _xn, h_slots, _d, _g = m._tul_core(xf, x0, bg, lay)
    assert h_slots.requires_grad, (
        "the slot state is detached — GL1's entire mechanism is that a later span's CE "
        "backpropagates through the write into the tap and the prelude")
    g = torch.autograd.grad(h_slots.sum(), m.tul.W_sent.weight, allow_unused=True)[0]
    assert g is not None and float(g.abs().sum()) > 0, \
        "the boundary tap is not in the slot state's graph"


def test_later_span_ce_reaches_the_tap_and_the_earlier_span_through_a_slot():
    """The positive control for T3 below. Under the mask, a later span's logits must
    still depend on an earlier span's embeddings — the slot is the route, not a wall."""
    spec = _spec(seq_len=64, max_slots=12)
    x, _y, lay, _ = _batch(B=1, n=200, seed=3, spec=spec)
    u, t = _find_probe_positions(lay)
    assert u is not None, "the layout did not produce three well-separated spans"

    m = _model(seed=7)
    m.eval()
    captured = {}

    def _hook(_mod, _inp, out):
        out.retain_grad()
        captured["embed_out"] = out

    h = m.embed.register_forward_hook(_hook)
    try:
        out = m(x, labels=None, slot_layout=lay)
    finally:
        h.remove()
    loss = _finite_logit_sum(out["logits"], t, SLOT_ID)

    g_tap = torch.autograd.grad(loss, m.tul.W_sent.weight, retain_graph=True)[0]
    assert float(g_tap.abs().sum()) > 0, \
        "a later span's logits do not depend on the boundary tap at all"
    loss.backward()
    g_u = captured["embed_out"].grad[0, u, :]
    assert float(g_u.abs().sum()) > 0, (
        f"span-2 logits carry no gradient to span-0 token {u}; the slot channel is not "
        f"passing information forward")


def test_severing_the_slot_channel_gives_exactly_zero_grad_with_retention_on():
    """T3, at ``n_core == 0`` and with retention ON — the arm's real setting.

    GLA is a recurrent scan, so without ``tg_reset_mask`` segmenting it, it would carry
    span 0 into span 2 invisibly to the attention mask and GL1's central claim ("the
    slot is the only route") would be false. The shipped T3 test runs with retention
    OFF and cannot see that.
    """
    spec = _spec(seq_len=64, max_slots=12)
    x, _y, lay, _ = _batch(B=1, n=200, seed=3, spec=spec)
    u, t = _find_probe_positions(lay)
    assert u is not None

    m = _model(seed=7)
    m.eval()
    embed_out, logits = _severed_forward(m, x, lay)
    _finite_logit_sum(logits, t, SLOT_ID).backward()
    g = embed_out.grad[0, u, :]
    assert torch.all(g == 0), (
        f"with the slot channel severed, span-2 logits STILL depend on span-0 token {u} "
        f"(max |grad| {float(g.abs().max()):.3e}). Under retention the GLA scan is a "
        f"second route between spans; if this fires, tg_reset_mask is not segmenting it "
        f"and GL1's 'slots are the only route' claim is FALSE.")


def test_the_severing_probe_is_not_vacuous_at_n_core_zero():
    """Same probe, restriction OFF: it must FIND gradient, or the test above proves
    nothing but that the hook broke the graph."""
    spec = _spec(seq_len=64, max_slots=12)
    x, _y, lay, _ = _batch(B=1, n=200, seed=3, spec=spec)
    u, t = _find_probe_positions(lay)
    m = _model(seed=7, tul=_tul(tg_restrict=False))
    m.eval()
    captured = {}

    def _hook(_mod, _inp, out):
        out.retain_grad()
        captured["e"] = out

    h = m.embed.register_forward_hook(_hook)
    try:
        out = m(x, labels=None, slot_layout=lay)
    finally:
        h.remove()
    _finite_logit_sum(out["logits"], t, SLOT_ID).backward()
    assert float(captured["e"].grad[0, u, :].abs().sum()) > 0


# ── 3. SIGReg on the written slot states ─────────────────────────────────────

def test_sigreg_fires_on_the_written_slot_states_and_enters_the_loss():
    x, y, lay, _ = _batch()
    m = _model()
    m.train()
    out = m(x, y, slot_layout=lay)
    assert "sigreg" in out and float(out["sigreg"]) > 0
    assert float(out["sigreg_weighted"]) == pytest.approx(0.02 * float(out["sigreg"]),
                                                          rel=1e-6)
    # and it reaches the tap: the regulariser must be able to move the WRITE
    xf, x0, bg = m._tul_front(x, lay)
    _xn, h_slots, _d, _g = m._tul_core(xf, x0, bg, lay)
    sig = m._tul_sigreg_loss(h_slots, lay)
    g = torch.autograd.grad(sig, m.tul.W_sent.weight, allow_unused=True)[0]
    assert g is not None and float(g.abs().sum()) > 0, \
        "SIGReg cannot reach the write it is supposed to diversify"


def test_sigreg_lambda_zero_leaves_the_loss_exactly_the_token_ce():
    x, y, lay, _ = _batch()
    m = _model(tul=_tul(sigreg_lambda=0.0))
    m.train()
    out = m(x, y, slot_layout=lay)
    assert "sigreg" not in out and "sigreg_weighted" not in out
    assert float(out["n_targets"]) == pytest.approx(float(out["n_tokens"]))


def test_the_slot_states_arrive_at_the_scale_sigreg_asks_for():
    """WHY NO STANDARDISATION IS APPLIED, as a measurement.

    SIGReg tests each 1-D projection against N(0,1), i.e. it wants per-component std 1.
    ``_readout`` ends in RMSNorm, so the states already arrive there — standardising on
    top would make the loss vacuous (morph/model/sigreg.py says so in its own
    docstring). If a change to the readout ever moves this, the arm's regulariser is
    chasing a target it cannot reach and this test says so.
    """
    x, _y, lay, _ = _batch()
    m = _model()
    m.eval()
    p = m.tul_slot_state_probe(x, lay)
    # Band, not a point: this tiny model reads 0.999 while the SHIPPED d=1024 model
    # reads 0.830 at init (RMSNorm's eps floor bites while the lm_mixer activations are
    # still small). Both are a scale SIGReg can close through final_norm.weight, which
    # is what "no standardisation needed" means. Outside this band the config note is
    # wrong and the regulariser is chasing a target it cannot reach.
    assert 0.7 <= p["slot_component_std"] <= 1.3, (
        f"slot states arrive at per-component std {p['slot_component_std']:.4f}; SIGReg "
        f"targets 1.0 and the config's no-standardisation note is now wrong")
    assert 0.7 <= p["slot_norm_mean"] / 64 ** 0.5 <= 1.3


def test_sigreg_discriminates_homogeneous_slot_states():
    """The disease it is aimed at: TG4b's writes were near-identical across slots."""
    from morph.model.sigreg import sigreg_epps_pulley
    import torch.nn.functional as F
    d, n = 64, 300
    g = torch.Generator().manual_seed(0)
    iso = torch.randn(n, d, generator=g)
    one = F.normalize(torch.randn(d, generator=g), dim=-1)
    homo = one[None].expand(n, d) * (d ** 0.5) + 0.05 * torch.randn(n, d, generator=g)
    torch.manual_seed(1)
    s_iso = float(sigreg_epps_pulley(iso, num_slices=256))
    torch.manual_seed(1)
    s_homo = float(sigreg_epps_pulley(homo, num_slices=256))
    assert s_homo > 5.0 * s_iso, f"isotropic {s_iso:.3f} vs homogeneous {s_homo:.3f}"


def test_the_slot_geometry_probe_reports_the_homogeneity_dial():
    x, _y, lay, _ = _batch()
    m = _model()
    m.eval()
    p = m.tul_slot_state_probe(x, lay)
    for k in ("slot_eff_rank", "slot_pairwise_cos", "slot_norm_mean",
              "slot_component_std", "slot_component_mean"):
        assert k in p, k
    assert 0.0 < p["slot_eff_rank"] <= 64.0
    assert -1.0 <= p["slot_pairwise_cos"] <= 1.0


# ── 4. the eval instruments ──────────────────────────────────────────────────

def test_the_three_ablations_all_move_the_ce_and_differ_from_each_other():
    x, y, lay, _ = _batch()
    m = _model()
    m.eval()
    with torch.no_grad():
        m.tul.W_prefix.normal_(0.0, 0.3)
        m.tul.W_sent.weight.normal_(0.0, 0.3)
    ce = {k: float(m.tul_forward_ablated(x, y, lay, plan_mode=k)["ce_tokens"].detach())
          for k in ("normal", "zero", "shuffle", "wrong_seed")}
    for k in ("zero", "shuffle", "wrong_seed"):
        assert abs(ce[k] - ce["normal"]) > 1e-5, f"{k} did not move ce_tokens: {ce}"
    assert len({round(v, 8) for v in ce.values()}) == 4, f"conditions collapsed: {ce}"


def test_the_wrong_seed_probe_restores_the_arms_own_seed():
    """It swaps ``tul.slot_seed`` for the duration of ONE forward. If it leaked, every
    later step would silently train a different arm."""
    x, y, lay, _ = _batch()
    m = _model()
    m.eval()
    assert m.cfg.tul.slot_seed == "boundary"
    m.tul_forward_ablated(x, y, lay, plan_mode="wrong_seed")
    assert m.cfg.tul.slot_seed == "boundary", "the wrong-seed probe leaked its swap"
    # and it really is a DIFFERENT computation, not a no-op
    a = float(m.tul_forward_ablated(x, y, lay, "normal")["ce_tokens"].detach())
    b = float(m.tul_forward_ablated(x, y, lay, "wrong_seed")["ce_tokens"].detach())
    assert a != b


def test_normal_mode_is_the_untouched_training_forward():
    x, y, lay, _ = _batch()
    m = _model()
    m.eval()
    a = m(x, y, slot_layout=lay)
    b = m.tul_forward_ablated(x, y, lay, plan_mode="normal")
    # ce_tokens is deterministic; `loss` is not, because SIGReg draws its M directions
    # from the GLOBAL RNG on every call (sigreg_epps_pulley(step=None)). Comparing the
    # CE is the claim that matters — the ablation seam must not touch the CE path — and
    # the loss is compared under a matched seed so the SIGReg draw is the same one.
    assert float(a["ce_tokens"].detach()) == pytest.approx(
        float(b["ce_tokens"].detach()), rel=1e-12)
    torch.manual_seed(5)
    la = float(m(x, y, slot_layout=lay)["loss"].detach())
    torch.manual_seed(5)
    lb = float(m.tul_forward_ablated(x, y, lay, plan_mode="normal")["loss"].detach())
    assert la == pytest.approx(lb, rel=1e-12)
    with pytest.raises(ValueError, match="normal|zero|shuffle"):
        m.tul_forward_ablated(x, y, lay, plan_mode="nonsense")


def test_first_token_metrics_are_present_at_eval():
    """``first_tok_counterfactual`` is half of GL1's gate; ``emit_weight=0`` must not
    remove it, because the group CEs are eval METRICS, not training terms."""
    x, y, lay, _ = _batch()
    m = _model()
    m.eval()
    out = m(x, y, slot_layout=lay)
    for k in ("ce_first_tok", "ce_first_tok_plain", "first_tok_counterfactual",
              "ce_tokens"):
        assert k in out, k
    assert float(out["first_tok_counterfactual"].detach()) == pytest.approx(
        float(out["ce_plast"].detach()) - float(out["ce_emit"].detach()), rel=1e-6)


def test_the_token_ce_covers_every_token_position():
    """``plast_weight: 1.0`` is a deliberate departure from tg2's 0.0 — at 0.0 the
    t_last tokens leave training entirely."""
    x, y, lay, _ = _batch()
    m = _model()
    m.train()
    assert float(m(x, y, slot_layout=lay)["n_targets"]) == pytest.approx(
        float(m(x, y, slot_layout=lay)["n_tokens"]))
    m0 = _model(tul=_tul(plast_weight=0.0))
    o0 = m0(x, y, slot_layout=lay)
    assert float(o0["n_targets"]) < float(o0["n_tokens"])


# ── 5. the arm and its matched control ───────────────────────────────────────

def test_gl1_and_its_control_differ_by_the_mask_alone():
    from hydra import compose, initialize_config_dir
    import os
    cd = os.path.abspath("morph/configs")
    with initialize_config_dir(version_base=None, config_dir=cd):
        a = compose(config_name="tul_gl1")
    with initialize_config_dir(version_base=None, config_dir=cd):
        b = compose(config_name="tul_gl1", overrides=["tul.tg_restrict=false"])
    from omegaconf import OmegaConf
    da = OmegaConf.to_container(a, resolve=True)
    db = OmegaConf.to_container(b, resolve=True)

    def _flat(d, pre=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(_flat(v, f"{pre}{k}."))
            else:
                out[f"{pre}{k}"] = v
        return out

    fa, fb = _flat(da), _flat(db)
    diff = {k for k in fa if fa[k] != fb.get(k)}
    assert diff == {"tul.tg_restrict"}, f"the control differs by more than the mask: {diff}"
    assert da["model"]["n_core"] == 0
    assert da["tul"]["slot_seed"] == "boundary"
    assert da["tul"]["sigreg_lambda"] == 0.02
    assert da["tul"]["eval_ablations"] is True
    assert da["fm"]["enabled"] is False
    assert da["model"]["use_kernels"] is False       # tg_restrict is eager-only


def test_the_control_is_config_matched_but_NOT_parameter_matched():
    """A measured caveat, pinned so nobody later claims a matched-parameter control.

    ``tg_restrict`` does not merely add a mask: it REPLACES the HCA dense-compressed
    attention branch with ``_tg_slot_attention``, so the restricted model never builds
    the compressor. Measured on the shipped configs: the unrestricted control carries
    60 extra tensors, 2.037 M parameters, +0.98 %. GL1 carries none the control lacks.

    This is pre-existing for every TG arm and the decision note's "0.17-0.42 nats vs
    unrestricted at matched steps" reference already lives with it — but "matched
    steps" is not "matched parameters" and the gap has to be priced, not assumed away.
    """
    from morph.training.train import build_morph_config
    from morph.training.tul_setup import build_tul_runtime
    from hydra import compose, initialize_config_dir
    import os
    cd = os.path.abspath("morph/configs")
    counts = {}
    for tag, ov in (("gl1", []), ("ctrl", ["tul.tg_restrict=false"])):
        with initialize_config_dir(version_base=None, config_dir=cd):
            c = compose(config_name="tul_gl1",
                        overrides=ov + ["model.hc_use_kernel=false"])
        rt = build_tul_runtime(c)
        mm = MORPHTransformer(build_morph_config(c, tul=rt.model_cfg))
        counts[tag] = {n: p.numel() for n, p in mm.named_parameters()}
        del mm
    extra = {k for k in counts["ctrl"] if k not in counts["gl1"]}
    assert not [k for k in counts["gl1"] if k not in counts["ctrl"]],         "the restricted arm gained parameters the control lacks; that is new"
    assert extra, "tg_restrict no longer changes the parameter set — update the caveat"
    n_extra = sum(counts["ctrl"][k] for k in extra)
    frac = n_extra / sum(counts["gl1"].values())
    assert 0.005 < frac < 0.02, f"the control/GL1 parameter gap moved to {frac:.2%}"
    assert all(("compressor" in k or "indexer" in k or "comp_norm" in k)
               and ".attention._impl." in k for k in extra), \
        f"the gap is no longer only the HCA compressed branch: {sorted(extra)[:5]}"
