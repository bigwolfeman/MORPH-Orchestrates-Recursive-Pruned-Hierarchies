"""TUL-FM Phase 1 gates — targets, masks, causality, the ladder, and end-to-end signal.

Everything here runs on CPU at tiny dims. Run it with the GPU untouched:

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_tulfm_p1.py -v

Each test is written so it FAILS when the thing it names is broken:

* the target tests compare against an independently computed pooled vector, not against
  a shape or a type;
* the causality tests have a POSITIVE control in the same test — a perturbation that must
  change the output — so "bit-identical" cannot pass by the output being constant;
* the end-to-end test trains the real planner through the real ladder on held-out toy
  data and requires retrieval to beat an untrained control by a wide, stated margin.

The causality tests run the backbone with the A3 arm's REAL settings — retention on,
``retention_carry`` on — because the previous generation of anti-leak tests in this
project all ran with retention OFF and therefore protected nothing
(``.agents/notes/proposed/bug-fix/2026-08-23-retention-carry-breaks-causality.md``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lab.tulfm.fm_planner import (
    FMPlanner,
    FMPlannerConfig,
    SpanGeometry,
    band_edges,
    band_of_sigma,
    build_masks,
    build_schedule,
    effective_rank,
    fm_loss,
    generate_plans,
    mean_pairwise_cos,
    pool_targets,
    segment_rows,
)
from lab.tulfm.retrieval_probe import retrieval_scores, row_index_of_valid
from morph.model.diffusion_blocks import SIGMA_MAX, SIGMA_MIN
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul_layout import BoundaryRule

V = 48
BOUNDARY_ID = 7          # the ONLY boundary id in these tests (plus EOS)
EOS_ID = 0


# ── fixtures / helpers ───────────────────────────────────────────────────────

def _rule(min_span: int = 4, span_cap: int = 32) -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[BOUNDARY_ID] = True
    lut[EOS_ID] = True
    return BoundaryRule(is_boundary=lut, min_span=min_span, span_cap=span_cap,
                        eos_id=EOS_ID)


def _ids_with_boundaries_at(positions: list[int], length: int,
                            filler: int = 11) -> torch.Tensor:
    """A single row whose ONLY boundary tokens sit at ``positions``."""
    row = np.full(length, filler, dtype=np.int64)
    for p in positions:
        row[p] = BOUNDARY_ID
    return torch.from_numpy(row)[None]


def _planner(d_ctx: int, max_slots: int, max_ctx_len: int, seed: int = 0,
             d_p: int = 64, n_layers: int = 2, n_heads: int = 4, d_ff: int = 128,
             cond_dim: int = 32) -> FMPlanner:
    torch.manual_seed(seed)
    return FMPlanner(FMPlannerConfig(
        d_ctx=d_ctx, d_p=d_p, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
        cond_dim=cond_dim, max_slots=max_slots, max_ctx_len=max_ctx_len)).eval()


def _wake_up(planner: FMPlanner, std: float = 0.05) -> FMPlanner:
    """Give the zero-init output head and the zero-init AdaLN gates real weights.

    At construction ``out`` and every ``AdaLNGate.to_mod`` are zero, so ``D̂ = c_skip·z``
    IGNORES the context entirely. A causality test on that model would pass no matter what
    the mask did. Every causality test below therefore wakes the planner up first, and
    carries a positive control that fails if this did not take.
    """
    torch.manual_seed(1234)
    with torch.no_grad():
        planner.out.weight.normal_(0.0, std)
        planner.out.bias.normal_(0.0, std)
        for layer in planner.layers:
            for gate in (layer.ada_self, layer.ada_cross, layer.ada_mlp):
                gate.to_mod.weight.normal_(0.0, std)
                gate.to_mod.bias.normal_(0.0, std)
    return planner


def _backbone(**kw) -> MORPHTransformer:
    """A tiny MORPH with the A3 arm's shape: ``n_core = 0``, retention ON, carry ON."""
    base = dict(
        d_model=32, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128,
        context_len=128, n_prelude=2, n_core=0, n_coda=1, mean_depth=2, max_depth=2,
        bptt_depth=1, channel_dims=(16, 8, 8), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=8, bigram_hash_vocab=V,
        use_kernels=False, hc_use_kernel=False, dropout=0.0,
        retention=True, retention_layers=(1,), retention_chunk=8, retention_carry=True,
    )
    base.update(kw)
    torch.manual_seed(0)
    return MORPHTransformer(MORPHConfig(**base)).eval()


# ── 1. span geometry ─────────────────────────────────────────────────────────

def test_geometry_matches_a_hand_computed_span_layout():
    """Slot i ends at boundary p_i; its target is the CLOSED span [p_i+1, p_{i+1}]."""
    ids = _ids_with_boundaries_at([5, 11, 19, 27], length=40)
    g = segment_rows(ids, _rule(), max_slots=16)

    # 4 closed spans -> 3 slots. The open tail after position 27 is never a target.
    assert g.n_spans_total == 4
    assert g.n_slots_valid == 3
    assert g.valid[0].tolist()[:4] == [True, True, True, False]
    assert g.slot_end[0, :3].tolist() == [5, 11, 19]
    assert g.tgt_start[0, :3].tolist() == [6, 12, 20]
    assert g.tgt_end[0, :3].tolist() == [11, 19, 27]
    assert g.dropped_fraction == pytest.approx(1.0 - 3 / 4)


def test_the_min_span_rule_actually_suppresses_a_boundary():
    """A boundary closer than min_span must NOT create a span — otherwise segment_rows
    is not running THE rule and every downstream number is on different data."""
    ids = _ids_with_boundaries_at([2, 9, 20], length=32)   # position 2 is span_len 3 < 4
    g = segment_rows(ids, _rule(min_span=4), max_slots=16)
    assert g.slot_end[0, :1].tolist() == [9], "min_span=4 should have suppressed pos 2"
    assert g.n_spans_total == 2


def test_the_max_slots_budget_is_reported_and_not_silent():
    ids = _ids_with_boundaries_at([5, 11, 17, 23, 29, 35], length=48)
    g = segment_rows(ids, _rule(), max_slots=2)
    assert g.n_slots_valid == 2
    assert g.n_dropped_budget == 3          # 5 available slots, 2 kept
    assert g.dropped_fraction > 0.6


# ── 2. targets ───────────────────────────────────────────────────────────────

def test_targets_are_unit_norm_and_are_the_next_span_mean():
    ids = _ids_with_boundaries_at([5, 11, 19], length=28)
    g = segment_rows(ids, _rule(), max_slots=8)
    torch.manual_seed(3)
    h = torch.randn(1, 28, 16)
    y = pool_targets(h, g)

    assert torch.allclose(y[0, :2].norm(dim=-1), torch.ones(2), atol=1e-6), \
        "SCAR: targets must be UNIT L2 norm (not per-component-std / SliceScaler)"
    assert torch.equal(y[0, 2:], torch.zeros_like(y[0, 2:])), \
        "invalid slots must be exactly zero"

    for i, (lo, hi) in enumerate([(6, 11), (12, 19)]):
        want = F.normalize(h[0, lo:hi + 1].mean(0), dim=-1)
        assert torch.allclose(y[0, i], want, atol=1e-6)


def test_the_target_is_the_NEXT_span_not_the_current_one():
    """The whole arc rests on this. A target pooled from span i would be readable from
    the conditioning and the objective would be autoencoding, not prediction."""
    ids = _ids_with_boundaries_at([5, 11, 19], length=28)
    g = segment_rows(ids, _rule(), max_slots=8)
    torch.manual_seed(4)
    h = torch.randn(1, 28, 16)
    y = pool_targets(h, g)

    cur = F.normalize(h[0, 0:6].mean(0), dim=-1)     # span 0, ending at slot 0
    nxt = F.normalize(h[0, 6:12].mean(0), dim=-1)    # span 1, the target
    assert float(y[0, 0] @ nxt) == pytest.approx(1.0, abs=1e-6)
    assert abs(float(y[0, 0] @ cur)) < 0.9, "target looks like the CURRENT span"


# ── 3. masks ─────────────────────────────────────────────────────────────────

def test_cross_mask_admits_exactly_the_positions_up_to_e_i():
    ids = _ids_with_boundaries_at([5, 11, 19], length=28)
    g = segment_rows(ids, _rule(), max_slots=8)
    self_mask, cross = build_masks(g)
    for i, e in enumerate([5, 11]):
        row = cross[0, i]
        assert bool(row[:e + 1].all()) and not bool(row[e + 1:].any()), \
            f"slot {i} must see exactly positions <= {e}"
    # slot self-attention is causal in slot order over valid slots
    assert self_mask[0, 0, 1].item() is False
    assert self_mask[0, 1, 0].item() is True


def test_every_query_row_has_at_least_one_key():
    """A fully-masked softmax row is NaN, and a NaN in a padding slot poisons the whole
    loss through the reduction. Padding rows must stay defined."""
    ids = _ids_with_boundaries_at([5, 11], length=28)
    g = segment_rows(ids, _rule(), max_slots=8)
    self_mask, cross = build_masks(g)
    assert bool(cross.any(dim=-1).all())
    assert bool(self_mask.any(dim=-1).all())


# ── 4. the frozen-feature hook ───────────────────────────────────────────────

def test_prelude_states_shape_and_the_eval_guard():
    m = _backbone()
    ids = torch.randint(1, V, (2, 24))
    h = m.prelude_states(ids)
    assert h.shape == (2, 24, 32)
    a = m.prelude_states(ids)
    assert torch.equal(h, a), "frozen features must be deterministic"

    m.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        m.prelude_states(ids)


def test_prelude_states_does_not_change_the_ordinary_forward():
    """The hook adds a read-out; it must not perturb anything the training path does."""
    m = _backbone()
    ids = torch.randint(1, V, (2, 24))
    before = m(ids)["logits"].clone()
    _ = m.prelude_states(ids)
    after = m(ids)["logits"]
    assert torch.equal(before, after)


# ── 5. causality — the leak gates (real backbone settings) ───────────────────

def _leak_setup(perturb_at: int):
    """Build a row, perturb ONE non-boundary token, and return both frozen feature maps.

    The perturbed token is swapped for another NON-boundary id so the span layout is
    provably identical between the two runs — otherwise a changed D̂ could mean "the
    spans moved", not "the future leaked".
    """
    m = _backbone()
    ids = _ids_with_boundaries_at([7, 15, 23], length=32, filler=11)
    bad = ids.clone()
    bad[0, perturb_at] = 13                     # another non-boundary id
    rule = _rule()
    g_a = segment_rows(ids, rule, max_slots=8)
    g_b = segment_rows(bad, rule, max_slots=8)
    assert torch.equal(g_a.slot_end, g_b.slot_end), "perturbation moved the spans"
    assert torch.equal(g_a.valid, g_b.valid)
    h_a = m.prelude_states(ids)
    h_b = m.prelude_states(bad)
    return g_a, h_a, h_b


def _dhat(planner, h, g, sigma_val: float = 1.0):
    """D̂ at a FIXED z and σ, so the only free input is the context."""
    self_mask, cross_mask = build_masks(g)
    torch.manual_seed(99)
    z = torch.randn(*g.valid.shape, h.shape[-1])
    sigma = torch.full(g.valid.shape, sigma_val)
    ctx = planner.encode_ctx(h)
    with torch.no_grad():
        return planner.denoise(z, sigma, ctx, g, self_mask, cross_mask)


def test_the_frozen_prelude_is_causal_at_positions_up_to_e_i():
    """If this fails it is a REAL FINDING about the backbone, not about the planner.

    A3 has ``n_core == 0``, so the retention_carry defect (which lives in the CORE loop,
    from iteration 2 onward) cannot fire — the prelude's GLA runs once and chunk-causally.
    This test pins that claim instead of assuming it.
    """
    g, h_a, h_b = _leak_setup(perturb_at=20)     # after e_1 = 15
    e1 = int(g.slot_end[0, 1])
    delta = (h_a[0, :e1 + 1] - h_b[0, :e1 + 1]).abs().max().item()
    assert delta == 0.0, (
        f"FROZEN PRELUDE LEAKS THE FUTURE: perturbing token 20 moved prelude states at "
        f"positions <= {e1} by {delta:.3e}. Every P1 number would be contaminated.")
    tail = (h_a[0, 21:] - h_b[0, 21:]).abs().max().item()
    assert tail > 0.0, "the perturbation did nothing at all; the probe is inert"


def test_dhat_is_bit_identical_when_a_token_after_e_i_changes():
    g, h_a, h_b = _leak_setup(perturb_at=20)     # strictly after e_1 = 15
    p = _wake_up(_planner(d_ctx=32, max_slots=8, max_ctx_len=32))
    a = _dhat(p, h_a, g)
    b = _dhat(p, h_b, g)
    assert torch.equal(a[0, 1], b[0, 1]), \
        "slot 1 saw a token after its own span end — the cross mask leaks"
    assert torch.equal(a[0, 0], b[0, 0]), "slot 0 leaked too"


def test_dhat_CHANGES_when_a_token_at_or_before_e_i_changes():
    """The positive control. Without it the test above passes on a planner that ignores
    its context entirely (which is exactly what the zero-init head does)."""
    g, h_a, h_b = _leak_setup(perturb_at=10)     # inside span 1, before e_1 = 15
    p = _wake_up(_planner(d_ctx=32, max_slots=8, max_ctx_len=32))
    a = _dhat(p, h_a, g)
    b = _dhat(p, h_b, g)
    assert not torch.equal(a[0, 1], b[0, 1]), \
        "perturbing a VISIBLE token did nothing — the conditioning path is dead"
    assert (a[0, 1] - b[0, 1]).abs().max().item() > 1e-6


def test_generated_plan_is_bit_identical_when_the_future_changes():
    g, h_a, h_b = _leak_setup(perturb_at=20)
    p = _wake_up(_planner(d_ctx=32, max_slots=8, max_ctx_len=32))
    sch = build_schedule()
    ga = torch.Generator().manual_seed(5)
    gb = torch.Generator().manual_seed(5)
    za = generate_plans(p, h_a, g, sch, n_steps=6, generator=ga)
    zb = generate_plans(p, h_b, g, sch, n_steps=6, generator=gb)
    assert torch.equal(za[0, 1], zb[0, 1])
    assert za[0, 1].abs().max().item() > 0.0, "plan is all-zero; equality is vacuous"


def test_loss_is_bit_identical_when_tokens_after_the_TARGET_span_change():
    """The honest form of the loss half of the leak gate.

    Perturbing a token in ``(e_i, e_{i+1}]`` legitimately changes slot i's LOSS, because
    that range IS the target — the target is a function of the future by construction.
    The leak claim is about positions after the target span. Perturb there.
    """
    g, h_a, h_b = _leak_setup(perturb_at=28)     # after e_2 = 23, the last target span
    p = _wake_up(_planner(d_ctx=32, max_slots=8, max_ctx_len=32))
    sch = build_schedule()
    la, _ = fm_loss(p, h_a, g, sch, generator=torch.Generator().manual_seed(7))
    lb, _ = fm_loss(p, h_b, g, sch, generator=torch.Generator().manual_seed(7))
    assert torch.equal(la, lb), f"loss moved by {abs(float(la) - float(lb)):.3e}"


def test_the_loss_DOES_move_when_a_visible_token_changes():
    g, h_a, h_b = _leak_setup(perturb_at=10)
    p = _wake_up(_planner(d_ctx=32, max_slots=8, max_ctx_len=32))
    sch = build_schedule()
    la, _ = fm_loss(p, h_a, g, sch, generator=torch.Generator().manual_seed(7))
    lb, _ = fm_loss(p, h_b, g, sch, generator=torch.Generator().manual_seed(7))
    assert not torch.equal(la, lb)


# ── 6. σ machinery (the audited pieces, used the way P1 uses them) ───────────

def test_sampled_sigma_stays_in_range_and_the_bands_are_equiprobable():
    sch = build_schedule()
    s = sch.sample_sigma(0, 60000, torch.device("cpu"),
                         generator=torch.Generator().manual_seed(0))
    assert float(s.min()) >= SIGMA_MIN and float(s.max()) <= SIGMA_MAX
    edges = band_edges(sch, 6)
    assert bool((edges[1:] > edges[:-1]).all()), "band edges must ascend"
    assert float(edges[0]) == pytest.approx(SIGMA_MIN, rel=1e-6)
    assert float(edges[-1]) == pytest.approx(SIGMA_MAX, rel=1e-6)
    counts = torch.bincount(band_of_sigma(s, edges), minlength=6).float() / s.numel()
    assert bool((counts - 1 / 6).abs().max() < 0.02), \
        f"bands are not equi-probability: {counts.tolist()}"


def test_the_inference_ladder_descends_and_starts_where_the_sampler_starts():
    sch = build_schedule()
    lad = sch.inference_sigmas(6)
    assert bool((lad[:-1] > lad[1:]).all()), "euler_step reads its sign from the descent"
    assert float(lad[0]) == pytest.approx(SIGMA_MAX, rel=1e-6)
    assert float(lad[-1]) == pytest.approx(SIGMA_MIN, rel=1e-6)


def test_a_perfect_denoiser_would_give_zero_loss_and_a_null_one_would_not():
    """Pin the objective's own arithmetic: EDM weighting cannot make ‖D̂−y‖² vanish
    for a wrong D̂, and it must vanish for the right one."""
    ids = _ids_with_boundaries_at([5, 11, 19], length=28)
    g = segment_rows(ids, _rule(), max_slots=8)
    torch.manual_seed(6)
    h = torch.randn(1, 28, 32)
    y = pool_targets(h, g)
    p = _planner(d_ctx=32, max_slots=8, max_ctx_len=28)

    class _Perfect(torch.nn.Module):
        precond = p.precond

        def encode_ctx(self, hh):
            return p.encode_ctx(hh)

        def denoise(self, z, sigma, ctx, geom, sm, cm):
            return y

    loss, _ = fm_loss(_Perfect(), h, g, build_schedule(),
                      generator=torch.Generator().manual_seed(1), y=y)
    assert float(loss) == pytest.approx(0.0, abs=1e-10)

    loss0, _ = fm_loss(p, h, g, build_schedule(),
                       generator=torch.Generator().manual_seed(1), y=y)
    assert float(loss0.detach()) > 1e-3, "the untrained planner's loss is suspiciously zero"


def test_the_null_floor_equals_the_loss_of_the_untrained_planner():
    """``rel_loss`` must be EXACTLY 1.0 at init, because the zero-init head makes the
    planner literally the null denoiser. That is what turns the per-band curves into
    "did this band learn anything" instead of "how big is w(sigma) here"."""
    ids = _ids_with_boundaries_at([5, 11, 19], length=28)
    g = segment_rows(ids, _rule(), max_slots=8)
    torch.manual_seed(12)
    h = torch.randn(1, 28, 32)
    p = _planner(d_ctx=32, max_slots=8, max_ctx_len=28)
    sch = build_schedule()
    edges = band_edges(sch, 6)
    loss, st = fm_loss(p, h, g, sch, generator=torch.Generator().manual_seed(2),
                       edges=edges)
    assert st["rel_loss"] == pytest.approx(1.0, abs=1e-6)
    for b in range(6):
        if st[f"band{b}/n"] > 0:
            assert st[f"band{b}/rel"] == pytest.approx(1.0, abs=1e-5)

    # And the floor is NOT a constant the model can never beat: a perfect denoiser
    # drives rel to 0.
    y = pool_targets(h, g)

    class _Perfect(torch.nn.Module):
        precond = p.precond

        def encode_ctx(self, hh):
            return p.encode_ctx(hh)

        def denoise(self, z, sigma, ctx, geom, sm, cm):
            return y

    _, st2 = fm_loss(_Perfect(), h, g, sch,
                     generator=torch.Generator().manual_seed(2), edges=edges, y=y)
    assert st2["rel_loss"] < 1e-9


def test_the_low_sigma_band_carries_far_more_loss_than_the_high_one():
    """Pin the unit-norm / sigma_data imbalance as a MEASURED fact, not a docstring claim.

    With ‖y‖ = 1 and sigma_data = 0.5 the null floor is ~d in the lowest band and ~4 in
    the highest. If this ratio ever collapses toward 1, someone changed the target scale
    or sigma_data and the module docstring's analysis is stale.
    """
    ids = _ids_with_boundaries_at(list(range(5, 200, 6)), length=256)
    g = segment_rows(ids, _rule(), max_slots=64)
    torch.manual_seed(13)
    h = torch.randn(1, 256, 64)
    p = _planner(d_ctx=64, max_slots=64, max_ctx_len=256)
    sch = build_schedule()
    edges = band_edges(sch, 6)
    lo, hi = [], []
    for seed in range(12):
        _, st = fm_loss(p, h, g, sch, generator=torch.Generator().manual_seed(seed),
                        edges=edges)
        if st["band0/n"] > 0:
            lo.append(st["band0/null"])
        if st["band5/n"] > 0:
            hi.append(st["band5/null"])
    ratio = (sum(lo) / len(lo)) / (sum(hi) / len(hi))
    assert ratio > 5.0, (
        f"low/high null-floor ratio is {ratio:.2f}; the docstring says it should be "
        f"large (~d·sigma_data²/‖y‖²). Re-derive it before trusting the band curves.")


def test_denoise_at_init_is_exactly_the_c_skip_passthrough():
    """Zero-init ``out`` ⇒ F_θ ≡ 0 ⇒ D̂ = c_skip(σ)·z. This is what makes the untrained
    control in the probe a DEFINED floor rather than a lucky draw."""
    ids = _ids_with_boundaries_at([5, 11, 19], length=28)
    g = segment_rows(ids, _rule(), max_slots=8)
    torch.manual_seed(8)
    h = torch.randn(1, 28, 32)
    p = _planner(d_ctx=32, max_slots=8, max_ctx_len=28)
    sm, cm = build_masks(g)
    z = torch.randn(1, 8, 32)
    sigma = torch.full((1, 8), 0.7)
    d_hat = p.denoise(z, sigma, p.encode_ctx(h), g, sm, cm)
    c_skip, _, _, _ = p.precond.coeffs(sigma)
    assert torch.allclose(d_hat, c_skip[..., None] * z, atol=1e-6)


# ── 7. retrieval scoring and the collapse guard ──────────────────────────────

def test_retrieval_scores_are_right_at_the_two_extremes():
    torch.manual_seed(0)
    y = F.normalize(torch.randn(4, 5, 8), dim=-1)
    valid = torch.ones(4, 5, dtype=torch.bool)

    perfect = retrieval_scores(y.clone(), y, valid)
    assert perfect["top1"] == 1.0 and perfect["mrr"] == 1.0
    assert perfect["n_candidates"] == 20.0 and perfect["chance"] == pytest.approx(0.05)

    # One constant query for every slot: whichever target it happens to like best wins
    # for ALL queries, so exactly one query is right — the chance rate, never more.
    tied = retrieval_scores(torch.ones(4, 5, 8), y, valid)
    assert tied["top1"] <= tied["chance"] + 1e-9, \
        f"a constant query beat chance: {tied['top1']}"

    # Collapsed TARGETS: every similarity is equal, so ties must count AGAINST us
    # (rank N, not rank 1). Without that, a collapsed target space would read as a
    # perfect retrieval score — the exact way this probe could lie.
    one = F.normalize(torch.randn(8), dim=-1)
    collapsed = one[None, None].expand(4, 5, 8).contiguous()
    deg = retrieval_scores(collapsed.clone(), collapsed, valid)
    assert deg["top1"] == 0.0 and deg["median_rank"] == 20.0, \
        f"ties did not count against us: {deg}"

    torch.manual_seed(1)
    rand = retrieval_scores(torch.randn(4, 5, 8), y, valid)
    assert rand["top1"] <= 0.25, f"random queries scored top1={rand['top1']}"


def test_retrieval_uses_only_valid_slots():
    torch.manual_seed(0)
    y = F.normalize(torch.randn(2, 6, 8), dim=-1)
    valid = torch.zeros(2, 6, dtype=torch.bool)
    valid[:, :3] = True
    assert retrieval_scores(y.clone(), y, valid)["n_candidates"] == 6.0


def test_within_row_scope_deletes_the_document_cue():
    """A planner that only identifies the ROW must beat batch-wide chance and NOT beat
    within-row chance. That gap is the reason both numbers are reported, and it is the
    confound the pre-registered gate is written against."""
    torch.manual_seed(0)
    B, S, d = 4, 5, 16
    row_dir = F.normalize(torch.randn(B, 1, d), dim=-1)
    y = F.normalize(row_dir + 0.05 * torch.randn(B, S, d), dim=-1)
    valid = torch.ones(B, S, dtype=torch.bool)
    rows = row_index_of_valid(valid)

    # A "document-only" planner: it reproduces the row's direction and nothing else.
    zhat = row_dir.expand(B, S, d).contiguous()
    wide = retrieval_scores(zhat, y, valid)
    narrow = retrieval_scores(zhat, y, valid, row_of=rows)

    assert wide["n_candidates"] == 20.0 and narrow["n_candidates"] == 5.0
    assert wide["top5"] > 0.9, \
        f"the document cue is not even visible batch-wide: {wide}"
    assert narrow["top1"] <= narrow["chance"] + 1e-9, (
        f"within-row scope did NOT delete the document cue: top1={narrow['top1']:.3f} "
        f"vs chance {narrow['chance']:.3f}")


def test_the_collapse_guard_needs_BOTH_numbers_to_see_a_collapse():
    """Effective rank alone cannot see a tight cluster; the cosine number can.

    This test exists because the obvious single-number guard is wrong: 200 near-copies of
    one vector plus 1 % jitter still vary along every axis, so the centered participation
    ratio reads HIGH on a target space that has collapsed.
    """
    torch.manual_seed(0)
    valid = torch.ones(1, 200, dtype=torch.bool)

    iso = F.normalize(torch.randn(1, 200, 32), dim=-1)
    assert effective_rank(iso, valid) > 20.0
    assert abs(mean_pairwise_cos(iso, valid)) < 0.05

    d = F.normalize(torch.randn(32), dim=-1)
    tight = F.normalize(d[None, None].expand(1, 200, 32)
                        + 0.01 * torch.randn(1, 200, 32), dim=-1)
    assert mean_pairwise_cos(tight, valid) > 0.95, \
        "the cosine guard missed a collapsed target space"

    exact = d[None, None].expand(1, 200, 32).contiguous()
    assert effective_rank(exact, valid) <= 1.5   # 0 or 1, per fp residue
    assert mean_pairwise_cos(exact, valid) == pytest.approx(1.0, abs=1e-5)


# ── 8. the freeze ────────────────────────────────────────────────────────────

def test_a_training_step_leaves_the_backbone_without_gradients():
    m = _backbone()
    for p_ in m.parameters():
        p_.requires_grad_(False)
    ids = _ids_with_boundaries_at([7, 15, 23], length=32)
    g = segment_rows(ids, _rule(), max_slots=8)
    with torch.no_grad():
        h = m.prelude_states(ids).float()

    p = _planner(d_ctx=32, max_slots=8, max_ctx_len=32)
    loss, _ = fm_loss(p, h, g, build_schedule(),
                      generator=torch.Generator().manual_seed(0))
    loss.backward()

    assert all(q.grad is None for q in m.parameters()), "gradient reached the backbone"
    assert any(q.grad is not None and q.grad.abs().sum() > 0 for q in p.parameters()), \
        "no planner parameter received gradient"


def test_the_shipped_planner_dims_land_in_the_declared_param_band():
    """The config declares 15-25 M and the trainer asserts it; pin the number here so a
    dim change cannot pass review by also editing the band."""
    p = FMPlanner(FMPlannerConfig(d_ctx=1024, d_p=512, n_layers=4, n_heads=8, d_ff=1408,
                                  cond_dim=256, max_slots=256, max_ctx_len=1024))
    n = p.n_params()
    assert 15e6 <= n <= 25e6, f"planner is {n/1e6:.2f}M params"
    # And pin that the band is not vacuously wide: d_ff 2048 falls OUT of it, which is
    # exactly why the shipped config says 1408.
    big = FMPlanner(FMPlannerConfig(d_ctx=1024, d_p=512, n_layers=4, n_heads=8, d_ff=2048,
                                    cond_dim=256, max_slots=256, max_ctx_len=1024))
    assert big.n_params() > 25e6


# ── 9. end to end: can this pipeline pass signal at all? ─────────────────────

def _toy_batch(B: int, L: int, d: int, span: int, a_mat: torch.Tensor,
               gen: torch.Generator) -> tuple[torch.Tensor, SpanGeometry, torch.Tensor]:
    """Toy data whose target is a deterministic LINEAR function of VISIBLE context.

    ``y_i = normalize(A · h[e_i])`` — the last position slot i is allowed to see. It is
    reachable only through the cross-attention mask, so a planner that ignores the
    context, attends to the wrong place, or reads the wrong row cannot solve it. The
    targets are near-orthogonal random directions, so retrieval is not confounded by
    neighbouring targets looking alike.
    """
    h = torch.randn(B, L, d, generator=gen)
    ends = list(range(span - 1, L, span))[:-1]          # slot ends; last span is open
    S = len(ends)
    e = torch.tensor(ends, dtype=torch.long)[None].expand(B, S).contiguous()
    geom = SpanGeometry(
        slot_end=e, tgt_start=e + 1, tgt_end=(e + span).clamp_max(L - 1),
        valid=torch.ones(B, S, dtype=torch.bool),
        n_spans_total=B * (S + 1), n_slots_valid=B * S, n_dropped_budget=0, seq_len=L,
    )
    y = F.normalize(h.gather(1, e[..., None].expand(B, S, d)) @ a_mat.T, dim=-1)
    return h, geom, y



def test_end_to_end_the_pipeline_can_pass_signal():
    """Target → conditioning → ladder → probe, all of it, on data with a known answer.

    Trains on FRESH toy batches every step and scores a HELD-OUT batch, so the bar is
    generalisation of the map, not memorisation of one batch. The untrained control
    scores the same held-out batch through the same ladder.
    """
    B, L, d, span, steps = 6, 48, 32, 6, 400
    gen = torch.Generator().manual_seed(0)
    a_mat = torch.randn(d, d, generator=gen) / d ** 0.5

    p = _planner(d_ctx=d, max_slots=L // span, max_ctx_len=L, seed=0,
                 d_p=96, n_layers=2, n_heads=4, d_ff=192, cond_dim=32)
    p.train()
    untrained = _planner(d_ctx=d, max_slots=L // span, max_ctx_len=L, seed=17,
                         d_p=96, n_layers=2, n_heads=4, d_ff=192, cond_dim=32)

    sch = build_schedule()
    opt = torch.optim.AdamW(p.parameters(), lr=3e-3, weight_decay=0.0)
    for step in range(steps):
        h, geom, y = _toy_batch(B, L, d, span, a_mat, gen)
        loss, _ = fm_loss(p, h, geom, sch, generator=gen, y=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(p.parameters(), 1.0)
        opt.step()

    p.eval()
    held = torch.Generator().manual_seed(9999)
    h, geom, y = _toy_batch(B, L, d, span, a_mat, held)
    z = generate_plans(p, h, geom, sch, n_steps=6,
                       generator=torch.Generator().manual_seed(3))
    z0 = generate_plans(untrained, h, geom, sch, n_steps=6,
                        generator=torch.Generator().manual_seed(3))
    zs = generate_plans(p, h.roll(1, dims=0), geom, sch, n_steps=6,
                        generator=torch.Generator().manual_seed(3))

    trained = retrieval_scores(z, y, geom.valid)
    control = retrieval_scores(z0, y, geom.valid)
    shuffled = retrieval_scores(zs, y, geom.valid)
    chance = trained["chance"]

    assert trained["top1"] > 0.9, (
        f"pipeline cannot pass signal: top1={trained['top1']:.3f} on a target that IS a "
        f"linear function of visible context (chance={chance:.4f})")
    assert control["top1"] < 5 * chance, \
        f"untrained control is not at chance: {control['top1']:.3f} vs {chance:.4f}"
    assert shuffled["top1"] < 0.5 * trained["top1"], (
        f"shuffled context scored {shuffled['top1']:.3f} against trained "
        f"{trained['top1']:.3f} — the planner is not reading THIS row")
    assert effective_rank(y, geom.valid) > 10.0, \
        "toy targets collapsed; the retrieval number would be meaningless"
