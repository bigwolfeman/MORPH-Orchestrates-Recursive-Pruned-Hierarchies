"""FM1 gates — the flow-matching planner inside the live model.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_tul_fm1.py -v

The load-bearing test is :func:`test_coda_ce_never_reaches_the_planner`. Everything else
in this arc rests on it: if the coda's cross-entropy could backpropagate through the
Euler ladder, FM1 would be BPTT through an iterated map — the exact mechanism behind
every takeover the 2026-08 campaign measured — and the arm's central claim would be
false while the loss curve looked fine.

Tiny dims, CPU, real settings where they matter: the causality gate runs with retention
ON, because the previous generation of anti-leak tests in this project all ran with it
off and therefore protected nothing
(``.agents/notes/implemented/bug-fix/2026-08-23-retention-carry-breaks-causality.md``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from morph.model.fm_planner import effective_rank, pool_targets
from morph.model.sigreg import sigreg_epps_pulley
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_fm import (
    FMArmConfig,
    copy_gap_scores,
    fm_geometry,
    fm_sigreg_loss,
    fm_span_targets,
)
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10
SLOT_ID = 4


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _batch(B: int = 2, n: int = 90, seed: int = 0, max_slots: int = 12):
    """A packed TUL batch with pad slots present (max_slots above the span count)."""
    spec = TulLayoutSpec(seq_len=48, prefix_k=2, max_slots=max_slots, slot_id=SLOT_ID)
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == SLOT_ID] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _fm_arm(**kw) -> FMArmConfig:
    base = dict(d_p=32, n_layers=2, n_heads=4, d_ff=64, cond_dim=32, sigreg_slices=64,
                source_std=1.0 / 8.0, max_slots=12, l_total=72)
    base.update(kw)
    return FMArmConfig(**base)


_DEFAULT = object()


def _cfg(fm=None, tul=_DEFAULT, n_core=0, **kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128,
        context_len=128, n_prelude=2, n_core=n_core, n_coda=1, mean_depth=2, max_depth=2,
        bptt_depth=1, channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=8, bigram_hash_vocab=V,
        use_kernels=False, hc_use_kernel=False, dropout=0.0,
        retention=True, retention_layers=(1,), retention_chunk=8, retention_carry=True,
        tul=(TULConfig(prefix_k=2, slot_id=SLOT_ID, token_state_dropout=0.0,
                       emit_weight=0.0, plast_weight=1.0) if tul is _DEFAULT else tul),
        fm=fm,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _model(fm=None, seed: int = 0, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_cfg(fm=fm, **kw))


# ── 1. THE DETACH ────────────────────────────────────────────────────────────

def test_coda_ce_never_reaches_the_planner():
    """NO planner parameter may appear in the token-CE graph. The arc rests on this.

    ``autograd.grad(..., allow_unused=True)`` returns None for a parameter that is not
    in the graph AT ALL, which is a stronger statement than "its gradient is small" or
    even "its gradient is zero" — a zero gradient can be a numerical accident.

    Three positive controls in the same test, so the assertion cannot pass by the CE
    graph being empty or the plan being ignored:
      * the TOTAL loss DOES reach every planner parameter (the fm term works);
      * the CE graph DOES reach ``W_prefix`` (the coda can learn to read the plan);
      * the CE graph DOES reach ``E_slot`` and the embedding table.
    """
    x, y, lay, _ = _batch()
    m = _model(_fm_arm())
    m.train()
    out = m(x, y, slot_layout=lay)

    planner_params = list(m.fm_planner.parameters())
    assert len(planner_params) > 10
    g_ce = torch.autograd.grad(out["loss_tokens_only"], planner_params,
                               allow_unused=True, retain_graph=True)
    in_graph = [n for (n, _p), t in zip(m.fm_planner.named_parameters(), g_ce)
                if t is not None]
    assert in_graph == [], (
        f"{len(in_graph)} planner parameters are in the coda-CE graph: {in_graph[:5]}. "
        f"The plan is not detached and FM1 is BPTT through an iterated map.")

    g_tot = torch.autograd.grad(out["loss"], planner_params, allow_unused=True,
                                retain_graph=True)
    assert all(t is not None for t in g_tot), \
        "the fm term does not reach the planner at all — the arm trains nothing"

    for name, prm in (("W_prefix", m.tul.W_prefix), ("E_slot", m.tul.E_slot),
                      ("embed", m.embed.hybrid.euc_embed.weight)):
        t = torch.autograd.grad(out["loss_tokens_only"], prm, allow_unused=True,
                                retain_graph=True)[0]
        assert t is not None and float(t.abs().sum()) > 0, \
            f"the coda CE does not reach {name}; the CE graph is broken, not just short"


def test_the_generated_plan_carries_no_graph_at_all():
    """The ladder itself must run outside autograd, not merely be detached afterwards.

    A ``detach()`` on a tensor that was built WITH a graph still pays for building it —
    six planner passes of stored activations per forward. This asserts the graph is
    never created.
    """
    x, _y, lay, _ = _batch()
    m = _model(_fm_arm())
    m.train()
    xf, _x0, _bg = m._tul_front(x, lay)
    _xn, h_slots, y_t, geom, ctx = m._tul_fm_core(xf, lay)
    assert not h_slots.requires_grad, "the plan carries a graph into the coda"
    assert ctx.requires_grad, (
        "the carrier handed to the planner is already detached upstream — then SIGReg "
        "has no path into the backbone and the targets are never shaped")
    assert y_t.requires_grad, \
        "the pooled targets are detached — SIGReg would have no gradient path"


def test_the_plan_still_changes_the_ce():
    """Detached is not the same as inert. If zeroing the plan cost nothing, W_prefix
    would have nothing to learn and the arm would be measuring nothing."""
    x, y, lay, _ = _batch()
    m = _model(_fm_arm())
    m.eval()
    ce = {k: float(m.tul_fm_forward(x, y, lay, plan_mode=k)["ce_tokens"].detach())
          for k in ("normal", "zero", "shuffle")}
    assert ce["zero"] != ce["normal"] and ce["shuffle"] != ce["normal"], ce
    with pytest.raises(ValueError, match="normal|zero|shuffle"):
        m.tul_fm_forward(x, y, lay, plan_mode="nonsense")


def test_the_planner_unsticks_after_the_zero_init_first_step():
    """``out`` and every AdaLN gate are zero-init, so ONLY the output head has gradient
    at step 0. That is the DiT discipline, not a bug — but it must unstick, and a test
    that only checked step 0 would happily pass on a planner that never trains."""
    x, y, lay, _ = _batch()
    m = _model(_fm_arm())
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    live = []
    for _ in range(3):
        out = m(x, y, slot_layout=lay)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        live.append(sum(1 for p in m.fm_planner.parameters()
                        if p.grad is not None and float(p.grad.abs().sum()) > 0))
        opt.step()
    n_total = len(list(m.fm_planner.parameters()))
    assert live[0] == 2, f"expected only out.weight/out.bias at step 0, got {live[0]}"
    assert live[-1] == n_total, f"planner never unstuck: {live} of {n_total}"


# ── 2. SIGReg ────────────────────────────────────────────────────────────────

def _sphere(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_sigreg_scores_a_collapsed_batch_far_worse_than_an_isotropic_one():
    d, n = 64, 400
    valid = torch.ones(1, n, dtype=torch.bool)
    iso = _sphere(n, d, 0)[None]
    one = F.normalize(torch.randn(d, generator=torch.Generator().manual_seed(1)), dim=-1)
    collapsed = F.normalize(one[None].expand(n, d)
                            + 0.01 * torch.randn(n, d,
                                                 generator=torch.Generator().manual_seed(2)),
                            dim=-1)[None]

    s_iso = float(fm_sigreg_loss(iso, valid, num_slices=256))
    s_col = float(fm_sigreg_loss(collapsed, valid, num_slices=256))
    assert s_col > 5.0 * s_iso, \
        f"SIGReg does not discriminate collapse: isotropic {s_iso:.3f} vs collapsed {s_col:.3f}"


def test_sigreg_gradients_are_finite_and_bounded_on_both():
    """Theorem 4 of the paper is bounded loss, gradient AND curvature for ANY input.
    A collapsed batch is the adversarial case, so check it there too."""
    d, n = 64, 400
    valid = torch.ones(1, n, dtype=torch.bool)
    one = F.normalize(torch.randn(d, generator=torch.Generator().manual_seed(1)), dim=-1)
    for name, z in (("isotropic", _sphere(n, d, 0)[None]),
                    ("collapsed", F.normalize(one[None].expand(n, d).contiguous()
                                              + 1e-2, dim=-1)[None])):
        z = z.clone().requires_grad_(True)
        loss = fm_sigreg_loss(z, valid, num_slices=256)
        assert torch.isfinite(loss), name
        g = torch.autograd.grad(loss, z)[0]
        assert torch.isfinite(g).all(), f"{name}: non-finite SIGReg gradient"
        per_row = g.reshape(-1, d).norm(dim=-1)
        assert float(per_row.max()) < 1e3, \
            f"{name}: per-row SIGReg gradient norm {float(per_row.max()):.3e} is not bounded"


def test_sigreg_scale_fix_is_what_makes_the_statistic_achievable():
    """The reason :func:`fm_sigreg_loss` multiplies by sqrt(d) before the statistic.

    SIGReg asks each 1-D projection to be N(0,1), i.e. E||z||^2 = d. Unit-L2 targets have
    E||z||^2 = 1. Run on them directly the statistic is enormous and irreducible; run on
    sqrt(d)*y it is the achievable question "are these uniform on the sphere", which by
    Poincare's lemma converges to the Gaussian marginal as d grows.
    """
    d, n = 256, 600
    y = _sphere(n, d, 3)
    raw = float(sigreg_epps_pulley(y, num_slices=256))
    scaled = float(sigreg_epps_pulley(y * math.sqrt(d), num_slices=256))
    gauss = float(sigreg_epps_pulley(
        torch.randn(n, d, generator=torch.Generator().manual_seed(4)), num_slices=256))
    assert raw > 20.0 * scaled, \
        f"unit-norm targets are not the pathological case they should be: {raw} vs {scaled}"
    assert scaled < 5.0 * gauss, (
        f"sqrt(d)-scaled sphere ({scaled:.3f}) is not close to a true Gaussian "
        f"({gauss:.3f}); the Poincare argument does not hold at d={d}")


def test_sigreg_ignores_pad_slots():
    d = 32
    valid = torch.zeros(1, 40, dtype=torch.bool)
    valid[:, :20] = True
    y = torch.zeros(1, 40, d)
    y[:, :20] = _sphere(20, d, 5)
    # The direction draw comes from the GLOBAL RNG (sigreg_epps_pulley(step=None)),
    # so the two calls must be seeded identically or they differ for a reason that has
    # nothing to do with pad slots.
    torch.manual_seed(99)
    a = float(fm_sigreg_loss(y, valid, num_slices=128))
    y2 = y.clone()
    y2[:, 20:] = 7.0                       # garbage in the pads
    torch.manual_seed(99)
    b = float(fm_sigreg_loss(y2, valid, num_slices=128))
    assert a == pytest.approx(b, rel=1e-12), "pad slots reached the SIGReg statistic"


# ── 3. bit-compat: silencing FM1 restores the skeleton ───────────────────────

def test_silencing_fm1_leaves_exactly_the_token_ce():
    """``fm_weight=0`` and ``sigreg_lambda=0`` must leave the total loss EXACTLY the
    token CE — not approximately, and with no sigreg term built at all."""
    x, y, lay, _ = _batch()
    m = _model(_fm_arm(fm_weight=0.0, sigreg_lambda=0.0))
    m.eval()
    out = m.tul_fm_forward(x, y, lay)
    assert float(out["fm_weighted"]) == 0.0
    assert "fm_sigreg" not in out, "sigreg_lambda=0 must build no sigreg term at all"
    assert float(out["loss"].detach()) == pytest.approx(
        float(out["loss_tokens_only"].detach()), rel=1e-12)


def test_the_fm_path_touches_only_the_slot_prefix_positions():
    """The structural half of bit-compat: the plan is injected at the prefix positions
    and NOWHERE else.

    This is the assertion a 'compare against a plain skeleton' test is trying to make,
    and it can actually be made here — a plain skeleton does not exist at ``n_core == 0``
    (``_tul_core`` has no coreless path), so comparing against one would mean comparing
    against something this arm cannot build.
    """
    x, _y, lay, _ = _batch()
    m = _model(_fm_arm())
    m.eval()
    with torch.no_grad():
        m.tul.W_prefix.normal_(0.0, 0.3)
        for p in m.fm_planner.parameters():
            p.add_(0.1 * torch.randn_like(p))
        xf, _x0, _bg = m._tul_front(x, lay)
        xn, h_slots, _y_t, _geom, _c = m._tul_fm_core(xf, lay)
        values, pos = m.tul.prefix_project(h_slots, lay, lay.l_total)
        from morph.model.tul import scatter_positions
        x_coda = scatter_positions(xn, pos, values)

    B, S = lay.slot_valid.shape
    K = lay.prefix_k
    touched = torch.zeros(B, lay.l_total, dtype=torch.bool)
    for b in range(B):
        for i in torch.nonzero(lay.slot_valid[b]).flatten().tolist():
            base = int(lay.slot_index[b, i])
            touched[b, base:base + K] = True
    assert bool(touched.any())
    diff = (x_coda - xn).abs().flatten(2).amax(-1) > 0        # [B, L]
    assert not bool((diff & ~touched).any()), (
        f"the FM path changed {int((diff & ~touched).sum())} positions outside the slot "
        f"prefixes — it is not confined to the plan channel")
    # ... and it really did write into EVERY one of them, otherwise the confinement
    # assertion above would pass on a path that writes nothing at all.
    missed = int((touched & ~diff).sum())
    assert missed == 0, f"{missed} of {int(touched.sum())} prefix positions were not written"


def test_a_model_without_fm_is_untouched():
    """``fm=None`` builds no planner, and every FM entry point refuses rather than
    silently doing something."""
    m = _model(fm=None, n_core=2)
    assert m.fm_planner is None
    x, y, lay, _ = _batch()
    with pytest.raises(RuntimeError, match="MORPHConfig\\(fm="):
        m.tul_fm_forward(x, y, lay)
    with pytest.raises(RuntimeError, match="MORPHConfig\\(fm="):
        m.fm_eval_probe(x, lay)
    out = m(x, y, slot_layout=lay)
    assert "fm" not in out and "fm_sigreg" not in out


def test_fm_refuses_the_configurations_it_has_no_meaning_for():
    with pytest.raises(ValueError, match="n_core == 0"):
        MORPHTransformer(_cfg(fm=_fm_arm(), n_core=2))
    with pytest.raises(ValueError, match="requires tul"):
        MORPHTransformer(_cfg(fm=_fm_arm(), tul=None))


# ── 4. causality ─────────────────────────────────────────────────────────────

def test_plans_are_causal_in_the_span_axis():
    """Perturbing a token AFTER slot i's span end must leave slot i's plan unchanged.

    Runs with retention ON and ``retention_carry`` ON — the arm's real setting. FM1 has
    ``n_core == 0``, so the carry defect (which lives in the CORE loop, from iteration 2
    onward) cannot fire; this pins that claim instead of assuming it.

    The perturbed token is swapped for another NON-boundary id, and the layout is
    asserted identical between the two runs, so a changed plan could only mean a leak
    and never "the spans moved".
    """
    spec = TulLayoutSpec(seq_len=48, prefix_k=2, max_slots=12, slot_id=SLOT_ID)
    rng = np.random.default_rng(7)
    ids = rng.integers(12, V, size=(1, 90))       # 12.. avoids DOT(10) and 11
    ids[:, ::6] = DOT
    bad = ids.copy()
    raw_probe = 25                                 # a plain token, mid-row
    assert bad[0, raw_probe] not in (DOT, 11, 0, SLOT_ID)
    bad[0, raw_probe] = 13

    xa, _ya, la, _ = slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)
    xb, _yb, lb, _ = slot_layout_from_ids(bad.astype(np.int64), _rule(), spec)
    assert torch.equal(la.slot_index, lb.slot_index) and torch.equal(la.slot_valid,
                                                                     lb.slot_valid)
    # The packer inserts slot positions, so the raw index is NOT the packed index.
    # Read the perturbed PACKED position off the diff instead of computing it.
    diff = (xa[0] != xb[0]).nonzero().flatten()
    assert diff.numel() == 1, f"the perturbation moved {diff.numel()} positions"
    probe = int(diff[0])

    m = _model(_fm_arm())
    m.eval()
    with torch.no_grad():                          # wake the zero-init head
        for p in m.fm_planner.parameters():
            p.add_(0.05 * torch.randn_like(p))

    def _plans(xx, ll):
        torch.manual_seed(3)                       # same ladder noise both sides
        xf, _x0, _bg = m._tul_front(xx, ll)
        _xn, h, _y, geom, _c = m._tul_fm_core(xf, ll)
        return h[:, :, 0, :], geom

    za, geom = _plans(xa, la)
    zb, _ = _plans(xb, lb)

    # which slots END before the perturbed position? their plans must be bit-identical
    ends = geom.slot_end[0]
    before = (ends < probe) & geom.valid[0]
    after = (ends >= probe) & geom.valid[0]
    assert int(before.sum()) >= 2 and int(after.sum()) >= 1, \
        f"probe position is degenerate: {int(before.sum())} before / {int(after.sum())} after"
    assert torch.equal(za[0, before], zb[0, before]), \
        "a plan changed when a token AFTER its span end changed — the ladder leaks"
    assert not torch.equal(za[0, after], zb[0, after]), \
        "no plan changed at all; the probe is inert and proves nothing"


# ── 5. targets, geometry and the eval instruments ────────────────────────────

def test_layout_pooling_agrees_with_the_contiguous_range_pooling():
    """``fm_span_targets`` pools from the layout's OWN bag map; ``pool_targets`` pools a
    contiguous index range. They must agree on every packed row — and if the packer's
    padding ever changes, this is the test that notices."""
    x, _y, lay, _ = _batch()
    geom = fm_geometry(lay)
    torch.manual_seed(11)
    h = torch.randn(x.shape[0], lay.l_total, 24)
    a = fm_span_targets(h, lay, geom)
    b = pool_targets(h, geom) * geom.valid[..., None].float()
    assert torch.allclose(a, b, atol=1e-5), \
        f"bag-map pooling and range pooling disagree by {float((a-b).abs().max()):.3e}"


def test_targets_are_unit_norm_and_the_last_real_slot_is_never_a_query():
    x, _y, lay, _ = _batch()
    geom = fm_geometry(lay)
    torch.manual_seed(12)
    h = torch.randn(x.shape[0], lay.l_total, 24)
    y = fm_span_targets(h, lay, geom)
    n = int(geom.valid.sum())
    assert n > 4
    assert torch.allclose(y[geom.valid].norm(dim=-1), torch.ones(n), atol=1e-5)
    assert torch.equal(y[~geom.valid], torch.zeros_like(y[~geom.valid]))
    # slot i is a query only if slot i+1 is real, so each row loses exactly its last one
    for b in range(x.shape[0]):
        n_slots = int(lay.slot_valid[b].sum())
        assert int(geom.valid[b].sum()) == n_slots - 1
        assert not bool(geom.valid[b, n_slots - 1])


def test_slot_end_is_the_last_token_before_the_slot():
    x, _y, lay, _ = _batch()
    geom = fm_geometry(lay)
    for b in range(x.shape[0]):
        for i in torch.nonzero(geom.valid[b]).flatten().tolist():
            e = int(geom.slot_end[b, i])
            assert e == int(lay.slot_index[b, i]) - 1
            assert not bool(lay.slot_mask[b, e]), "slot_end landed on a slot position"
            assert bool(lay.slot_mask[b, e + 1]), "the slot does not follow slot_end"


def test_copy_gap_recovers_a_known_answer():
    """The instrument itself, on data whose answer is arithmetic.

    A PERFECT planner (plan == target) scores top-1 1.0. A pure COPY planner (plan ==
    the previous span's target) scores exactly what the copy baseline scores, because it
    IS the copy baseline — that identity is the test.
    """
    torch.manual_seed(13)
    B, S, d = 3, 8, 16
    y = F.normalize(torch.randn(B, S, d), dim=-1)
    valid = torch.ones(B, S, dtype=torch.bool)

    perfect = copy_gap_scores(y.clone(), y, valid)
    assert perfect["plan_top1"] == 1.0
    assert perfect["n_queries"] == B * (S - 1)
    assert perfect["n_candidates"] == pytest.approx(S - 1.0)   # y_{i-1} excluded
    assert perfect["copy_gap"] == pytest.approx(1.0 - perfect["copy_top1"])

    copycat = copy_gap_scores(torch.roll(y, 1, dims=1), y, valid)
    assert copycat["plan_top1"] == pytest.approx(copycat["copy_top1"]), \
        "a plan that IS the copy baseline did not score as the copy baseline"
    assert copycat["copy_gap"] == pytest.approx(0.0)


def test_copy_gap_excludes_the_self_match_candidate():
    """Without the exclusion the copy baseline retrieves ITSELF and reads ~1.0, which is
    how a baseline gets dismissed as broken instead of being taken seriously."""
    torch.manual_seed(14)
    B, S, d = 2, 6, 16
    y = F.normalize(torch.randn(B, S, d), dim=-1)
    valid = torch.ones(B, S, dtype=torch.bool)
    r = copy_gap_scores(y.clone(), y, valid)
    assert r["n_candidates"] == pytest.approx(S - 1.0), \
        "the previous-span candidate is still in the pool"
    # with it left in, the copy baseline would be perfect; it must not be
    assert r["copy_top1"] < 0.5


def test_plan_worth_helpers_move_when_the_plan_carries_signal():
    """zero and shuffle must both differ from normal, and from each other, once the plan
    actually carries slot-specific content. Built by hand: give W_prefix a real map and
    the plans distinct directions."""
    x, y, lay, _ = _batch()
    m = _model(_fm_arm())
    m.eval()
    with torch.no_grad():
        m.tul.W_prefix.normal_(0.0, 0.3)
        for p in m.fm_planner.parameters():
            p.add_(0.1 * torch.randn_like(p))
    ce = {k: float(m.tul_fm_forward(x, y, lay, plan_mode=k)["ce_tokens"].detach())
          for k in ("normal", "zero", "shuffle")}
    assert abs(ce["zero"] - ce["normal"]) > 1e-4, ce
    assert abs(ce["shuffle"] - ce["normal"]) > 1e-4, ce
    assert abs(ce["shuffle"] - ce["zero"]) > 1e-6, ce


def test_shuffle_permutes_within_a_row_and_keeps_pads_out():
    x, _y, lay, _ = _batch()
    m = _model(_fm_arm())
    B, S = lay.slot_valid.shape
    h = torch.arange(B * S, dtype=torch.float32).reshape(B, S, 1).expand(B, S, 5).clone()
    torch.manual_seed(2)
    out = m._tul_plan_ablate(h, lay, "shuffle")
    for b in range(B):
        nv = int(lay.slot_valid[b].sum())
        got = set(out[b, :nv, 0].tolist())
        want = set(h[b, :nv, 0].tolist())
        assert got == want, f"row {b}: shuffle left the row or dropped a slot"
        assert out[b, :nv, 0].tolist() != h[b, :nv, 0].tolist() or nv < 2


def test_fm_eval_probe_returns_the_full_instrument_set():
    x, _y, lay, _ = _batch()
    m = _model(_fm_arm())
    m.eval()
    p = m.fm_eval_probe(x, lay)
    for k in ("plan_top1", "copy_top1", "copy_gap", "chance", "n_queries",
              "target_eff_rank", "target_pairwise_cos", "target_norm_mean"):
        assert k in p, k
    assert p["n_queries"] > 0
    assert p["target_norm_mean"] == pytest.approx(1.0, abs=1e-4)
    assert p["copy_gap"] == pytest.approx(p["plan_top1"] - p["copy_top1"])


def test_the_total_loss_is_the_sum_of_its_three_reported_terms():
    """A composite loss whose parts do not add up is a reporting bug that survives every
    other test in this file."""
    x, y, lay, _ = _batch()
    arm = _fm_arm(fm_weight=0.7, sigreg_lambda=0.05)
    m = _model(arm)
    m.train()
    out = m(x, y, slot_layout=lay)
    total = (float(out["loss_tokens_only"].detach()) + float(out["fm_weighted"])
             + float(out["fm_sigreg_weighted"]))
    assert float(out["loss"].detach()) == pytest.approx(total, rel=1e-6)
    assert float(out["fm_weighted"]) == pytest.approx(0.7 * float(out["fm"]), rel=1e-6)
    assert float(out["fm_sigreg_weighted"]) == pytest.approx(
        0.05 * float(out["fm_sigreg"]), rel=1e-6)


def test_the_token_ce_covers_every_token_position():
    """``emit_weight=0`` / ``plast_weight=1`` must mean exactly 'ordinary token CE over
    token positions'. If plast_weight were 0 too, ~1 token in 20 would silently vanish
    from training — and they are the tokens a plan is supposed to help with."""
    x, y, lay, _ = _batch()
    m = _model(_fm_arm())
    m.train()
    out = m(x, y, slot_layout=lay)
    assert float(out["n_targets"]) == pytest.approx(float(out["n_tokens"]))

    tul0 = TULConfig(prefix_k=2, slot_id=SLOT_ID, token_state_dropout=0.0,
                     emit_weight=0.0, plast_weight=0.0)
    m0 = _model(_fm_arm(), tul=tul0)
    out0 = m0(x, y, slot_layout=lay)
    assert float(out0["n_targets"]) < float(out0["n_tokens"]), \
        "plast_weight=0 should DROP the t_last tokens; if it does not, the note is wrong"


def test_target_effective_rank_is_reported_on_real_targets():
    x, _y, lay, _ = _batch()
    geom = fm_geometry(lay)
    torch.manual_seed(15)
    h = torch.randn(x.shape[0], lay.l_total, 32)
    y = fm_span_targets(h, lay, geom)
    r = effective_rank(y, geom.valid)
    assert 1.0 < r <= 32.0, r


# ── 9. ARM FM2 (emit CE as the reader-trainer) ───────────────────────────────

def test_emit_ce_reaches_the_reader_not_the_planner():
    """FM2's safety property: with emit_weight=0.5 the WEIGHTED CE now scores the slot
    positions themselves — and it must STILL not reach a single planner parameter (z is
    detached; there is no core loop). Positive controls: the same weighted CE must reach
    W_prefix, E_slot, and the embedding table (the reading machinery emit exists to
    train), and the emit labels must actually be in the loss (n_targets grows vs the
    emit_weight=0 model on the identical batch)."""
    x, y, lay, _ = _batch()
    tul2 = TULConfig(prefix_k=2, slot_id=SLOT_ID, token_state_dropout=0.0,
                     emit_weight=0.5, plast_weight=0.5)
    m = _model(_fm_arm(), tul=tul2)
    m.train()
    out = m(x, y, slot_layout=lay)

    m0 = _model(_fm_arm())     # the FM1 weighting (emit 0.0 / plast 1.0)
    out0 = m0(x, y, slot_layout=lay)
    n_slot_labels = int((lay.slot_mask & (y != -100)).sum())
    assert n_slot_labels > 0, "fixture carries no emit labels; test is toothless"
    assert float(out["n_targets"]) > float(out0["n_targets"]) - 1e-6, \
        "emit_weight=0.5 did not add the slot positions to the scored set"

    planner_params = list(m.fm_planner.parameters())
    g_ce = torch.autograd.grad(out["loss_tokens_only"], planner_params,
                               allow_unused=True, retain_graph=True)
    leaked = [n for (n, _p), t in zip(m.fm_planner.named_parameters(), g_ce)
              if t is not None]
    assert leaked == [], (
        f"{len(leaked)} planner parameters entered the CE graph under emit_weight=0.5: "
        f"{leaked[:5]} — the reader-trainer is backpropagating into the writer.")

    for name, prm in (("W_prefix", m.tul.W_prefix), ("E_slot", m.tul.E_slot),
                      ("embed", m.embed.hybrid.euc_embed.weight)):
        t = torch.autograd.grad(out["loss_tokens_only"], prm, allow_unused=True,
                                retain_graph=True)[0]
        assert t is not None and float(t.abs().sum()) > 0, \
            f"emit CE does not reach {name} — the reading machinery gets no gradient"
