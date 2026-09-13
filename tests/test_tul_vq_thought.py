"""`tul.vq_codes` — the DISCRETE thought: K codes per span, not one continuous vector.

THE DEFECT. A row's written slot states sit at effective rank 5.7598 in 1024 dimensions
with mean pairwise cosine 0.7104 (`slot-spandec-strict`; 5.7-7.3 / 0.72-0.77 across the
slot family). The coda reads TOKENS well and this channel badly, and a token is a DISCRETE
symbol out of a large alphabet. So make the thought the same kind of object.

WHAT THIS FILE HAS TO PROVE:

1. OFF IS NOTHING. `vq_codes: 0` builds no module, draws no RNG, adds no state-dict key and
   leaves the forward bit-identical. Proved OUTSIDE this file by running the pre-work tree
   (`f89256d`) and this one on four fixtures — loss, logit sum, grad sum, state-dict key
   count, parameter count and the whole slot probe match to the last printed digit; the
   numbers are pinned here so a regression shows up in the suite.
2. THE WRITE IS THE CODES. The coda's prefix cell k holds exactly `W_vq_out[k] @ e_n` for
   the code the quantizer recorded at position k, scattered through the shipped
   `prefix_project` / `scatter_positions` path. Read off the real coda input with a spy, not
   re-derived from the module.
3. THE STE CARRIES THE GRADIENT. A CORE parameter reads a nonzero gradient through the VQ
   path, and detaching the estimator drives it to exactly zero. That is the whole
   mechanism: on a VQ model the straight-through edge is the ONLY route the token CE and
   the span decoder have into the loop.
4. THE TWO VQ-VAE TERMS ARE IN THE LOSS, positive, and exposed as a weighted twin the way
   `spandec_weighted` is, so `train.py` subtracts them and `train/loss` stays the CE.
5. THE DECODER GRADES THE DEQUANTIZED THOUGHT — the MEAN of the K lifted cells — and not
   the continuous exit state the coda never sees. Spied on the real call.
6. PADS. A pad slot's index is -1, its cells are exactly zero, it is out of both loss terms
   and out of the usage histogram, and no VQ parameter reads a NaN gradient. (This is where
   the register's own NaN bug lived.)
7. THE INSTRUMENTS. `vq_perplexity` is in (1, C], `vq_used` counts distinct codes, and the
   slot-state probe reads the LIFTED CELLS so `slot_cell_eff_rank` is the within-slot rank
   across the K codes.
8. EVERY REFUSAL.

CPU only, fp32, `use_kernels=False`, tiny config.

Record: lab/experiments/planned/2026-09-13-arc-discrete-thought-vq.md
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids
from morph.model.tul_vq import TULThoughtVQ

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


def _ids(B: int = 2, n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::8] = DOT
    return ids.astype(np.int64)


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                mux_beta=0.0, spandec=True, spandec_layers=1, spandec_max_tokens=8,
                tg_restrict=True, tg_restrict_scope="all", tg_geometry="strict")
    base.update(kw)
    return TULConfig(**base)


def _batch(K: int = 2, seed: int = 0):
    spec = TulLayoutSpec(seq_len=64, prefix_k=max(K, 2), max_slots=10, slot_id=4)
    ids = _ids(seed=seed)
    inp, lab, layout, _ = slot_layout_from_ids(ids, _rule(), spec)
    return ids, inp, lab, layout


def _model(K: int = 0, seed: int = 99, codebook: int = 32, dim: int = 16, **tul_kw):
    """`K == 0` is the ruler at prefix_k 2; `K > 0` builds the quantizer at prefix_k == K."""
    kw = dict(prefix_k=max(K, 2))
    if K:
        kw.update(vq_codes=K, vq_codebook=codebook, vq_dim=dim)
    kw.update(tul_kw)
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**kw)))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.train().float()


def _fwd(m, inp, lab, layout):
    return m(torch.as_tensor(inp), labels=torch.as_tensor(lab), slot_layout=layout)


# ── 1. OFF IS NOTHING ────────────────────────────────────────────────────────

def test_off_is_the_default_and_builds_no_quantizer():
    m = _model(0)
    assert m.cfg.tul.vq_codes == 0
    assert m.tul_vq is None
    assert not any("tul_vq" in k for k in m.state_dict())


def test_off_state_is_bit_identical_to_the_pre_work_tree():
    """The pins from the cross-commit run (`f89256d` vs this tree, four fixtures).

    The comparison itself cannot live in pytest — it needs two checkouts — so what lives
    here is the RESULT, to the last printed digit. If a later change to this module moves
    any of these, the OFF-state claim is broken and the suite says so.
    """
    pins = {
        # (prefix_k, extra tul kwargs) -> (loss, grad_sum, state-dict keys, params)
        (2, ()): (9.9752750397, -13.439471, 221, 516509),
        (4, ()): (9.9581413269, -14.849805, 221, 524701),
        (4, (("slot_cells", 4),)): (9.9072341919, -13.706003, 226, 537501),
    }
    for (k, extra), (want_loss, want_g, want_keys, want_p) in pins.items():
        _i, inp, lab, layout = _batch(k)
        m = _model(0, prefix_k=k, **dict(extra))
        torch.manual_seed(7)
        out = _fwd(m, inp, lab, layout)
        out["loss"].backward()
        g = sum(float(p.grad.double().sum()) for p in m.parameters() if p.grad is not None)
        assert float(out["loss"]) == pytest.approx(want_loss, abs=1e-9)
        assert g == pytest.approx(want_g, abs=1e-5)
        assert len(m.state_dict()) == want_keys
        assert sum(p.numel() for p in m.parameters()) == want_p


def test_the_quantizer_adds_exactly_three_tensors_and_moves_no_base_weight():
    base = dict(_model(0, prefix_k=4).state_dict())
    vq = dict(_model(4).state_dict())
    new = sorted(set(vq) - set(base))
    assert new == ["tul_vq.W_vq.weight", "tul_vq.W_vq_out", "tul_vq.vq_E"], new
    assert not set(base) - set(vq)
    # `code_age` is a live counter, not a weight: persistent=False keeps it out.
    assert "tul_vq.code_age" not in vq
    for k, v in base.items():
        assert torch.equal(v, vq[k]), f"{k} moved when the quantizer was built"


def test_building_the_quantizer_moves_no_global_rng_state():
    torch.manual_seed(3)
    before = torch.random.get_rng_state().clone()
    TULThoughtVQ(64, codes=4, codebook=32, dim=16, groups=1, beta=0.25)
    assert torch.equal(before, torch.random.get_rng_state())


def test_the_quantizers_own_weights_do_not_depend_on_the_ambient_rng():
    torch.manual_seed(1)
    a = TULThoughtVQ(64, codes=4, codebook=32, dim=16, groups=1, beta=0.25)
    torch.manual_seed(12345)
    b = TULThoughtVQ(64, codes=4, codebook=32, dim=16, groups=1, beta=0.25)
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), n


# ── 2. THE WRITE IS THE CODES ────────────────────────────────────────────────

def _coda_input_spy(m):
    """Capture the coda's carrier (`_back_region`'s `x`) on the real forward."""
    seen = {}
    real = m._back_region

    def spy(x, *a, **kw):
        seen["x"] = x.detach().clone()
        return real(x, *a, **kw)

    m._back_region = spy
    return seen


def test_the_coda_prefix_cells_are_the_k_lifted_codes():
    """Cell k of a valid slot == `W_prefix[k] applied to W_vq_out[k] @ e_n`.

    Read off the REAL coda input (spy) and compared against a value rebuilt from the
    recorded code INDEX — so it pins the whole chain: the head, the argmax, the lift, the
    1:1 cell assignment and `prefix_project`'s per-cell `W_prefix[k]`.
    """
    K = 4
    _i, inp, lab, layout = _batch(K)
    m = _model(K)
    seen = _coda_input_spy(m)
    cap = {}
    real = m.tul_vq.forward

    def spy_vq(z, valid):
        cells, deq, out = real(z, valid)
        cap["cells"], cap["out"] = cells.detach().clone(), out
        return cells, deq, out

    m.tul_vq.forward = spy_vq
    _fwd(m, inp, lab, layout)
    x = seen["x"]                                   # [B, L, n, C], post token dropout (p=0)
    cells, out = cap["cells"], cap["out"]
    idx = out["index"]                              # [B, S, n, K, G]
    e = m.tul_vq.vq_E.detach()
    e_n = e / (e.norm(dim=-1, keepdim=True) + m.tul_vq.eps)
    B, S = layout.slot_valid.shape
    checked = 0
    for b in range(B):
        for s in range(S):
            if not bool(layout.slot_valid[b, s]):
                continue
            pos = int(layout.slot_index[b, s])
            for k in range(K):
                # (a) the written coda position holds cell k through W_prefix[k].
                want = torch.matmul(cells[b, s, k], m.tul.W_prefix[k].detach())
                assert torch.allclose(x[b, pos + k], want, atol=1e-5), (b, s, k)
                # (b) cell k IS the lift of the recorded code at position k.
                q = e_n[idx[b, s, :, k, 0]]                    # [n, d_g] (G == 1)
                lifted = torch.matmul(q, m.tul_vq.W_vq_out[k].detach())
                assert torch.allclose(cells[b, s, k], lifted, atol=1e-5), (b, s, k)
                checked += 1
    assert checked > 0


def test_prefix_k_must_equal_vq_codes():
    with pytest.raises(ValueError, match="needs tul.prefix_k"):
        _tul(prefix_k=2, vq_codes=4)
    with pytest.raises(ValueError, match="needs tul.prefix_k"):
        _tul(prefix_k=8, vq_codes=4)
    _tul(prefix_k=4, vq_codes=4)          # the legal one builds


def test_a_pad_slots_cells_are_exactly_zero_and_its_index_is_minus_one():
    K = 4
    _i, inp, lab, layout = _batch(K)
    m = _model(K)
    cap = {}
    real = m.tul_vq.forward

    def spy_vq(z, valid):
        cells, deq, out = real(z, valid)
        cap["cells"], cap["out"] = cells.detach().clone(), out
        return cells, deq, out

    m.tul_vq.forward = spy_vq
    _fwd(m, inp, lab, layout)
    pad = ~layout.slot_valid
    assert bool(pad.any()), "the fixture must contain a pad slot"
    assert float(cap["cells"][pad].abs().sum()) == 0.0
    idx = cap["out"]["index"]
    assert int(idx[pad].max()) == -1 and int(idx[pad].min()) == -1
    assert int(idx[layout.slot_valid].min()) >= 0


# ── 3. THE STE CARRIES THE GRADIENT ──────────────────────────────────────────

def _core_grad(m, inp, lab, layout, detach_ste: bool):
    """Total |grad| on the CORE's parameters, with the STE optionally cut."""
    if detach_ste:
        real = m.tul_vq.forward

        def sab(z, valid):
            cells, deq, out = real(z.detach(), valid)
            return cells, deq, out

        m.tul_vq.forward = sab
    torch.manual_seed(7)
    out = _fwd(m, inp, lab, layout)
    out["loss"].backward()
    return sum(float(p.grad.abs().sum()) for n, p in m.named_parameters()
               if n.startswith("core.") and p.grad is not None)


def test_the_ste_passes_gradient_to_the_loop_and_detaching_it_kills_every_core_grad():
    """The estimator IS the mechanism, so it gets a two-sided test.

    AT `vq_weight: 0`, and that is the whole point of the setting here. The two VQ-VAE
    terms are a SECOND gradient edge into the loop — the commitment term reaches `u_n`,
    hence `W_vq`, hence `z` — so at the shipped weight a detached estimator still leaves
    the core a nonzero gradient and this test would pass on a broken model (measured:
    sabotage S2 MISSED against the first version of it). With the VQ terms weighted 0 the
    only remaining route from the loss to the loop is the straight-through edge, and
    cutting it must drive the core's gradient to EXACTLY zero.
    """
    K = 4
    _i, inp, lab, layout = _batch(K)
    live = _core_grad(_model(K, vq_weight=0.0), inp, lab, layout, detach_ste=False)
    dead = _core_grad(_model(K, vq_weight=0.0), inp, lab, layout, detach_ste=True)
    assert live > 1e-4, live
    assert dead == 0.0, dead


def test_the_cells_value_is_the_code_and_their_gradient_is_the_encoders():
    """The STE at the source, on the bare module: the FORWARD value is the code's lift and
    the BACKWARD reaches the encoder. `q = q_e.detach()` breaks the second half while
    leaving the first intact, which is exactly the shape of a sabotage a value-only test
    cannot see."""
    vq = TULThoughtVQ(16, codes=2, codebook=8, dim=8, groups=1, beta=0.25)
    z = torch.randn(2, 3, 16, requires_grad=True)
    valid = torch.ones(2, 3, dtype=torch.bool)
    cells, _deq, out = vq(z, valid)
    e = vq.vq_E.detach()
    e_n = e / (e.norm(dim=-1, keepdim=True) + vq.eps)
    for b in range(2):
        for s in range(3):
            for k in range(2):
                want = torch.matmul(e_n[out["index"][b, s, k, 0]], vq.W_vq_out[k].detach())
                assert torch.allclose(cells[b, s, k], want, atol=1e-6)
    cells.sum().backward()
    assert vq.W_vq.weight.grad is not None
    assert float(vq.W_vq.weight.grad.abs().sum()) > 0.0
    assert z.grad is not None and float(z.grad.abs().sum()) > 0.0


def test_the_two_terms_carry_the_two_stop_gradients_they_are_named_for():
    """Two-sided, because the VALUES of the two terms are identical on the unit sphere and
    only the stop-gradients tell them apart (sabotage S4 dropped one and nothing noticed).

      codebook `||sg[u] - e||^2`   moves the CODEBOOK and must not move the encoder
      commit   `||u - sg[e]||^2`   moves the ENCODER and must not move the codebook
    """
    for term, moves, frozen in (("codebook", "vq_E", "W_vq"),
                                ("commit", "W_vq", "vq_E")):
        vq = TULThoughtVQ(16, codes=2, codebook=8, dim=8, groups=1, beta=0.25)
        z = torch.randn(2, 3, 16)
        _c, _d, out = vq(z, torch.ones(2, 3, dtype=torch.bool))
        out[term].backward()
        g = dict(vq.named_parameters())
        mv = g[moves] if moves == "vq_E" else g["W_vq.weight"]
        fz = g[frozen] if frozen == "vq_E" else g["W_vq.weight"]
        assert mv.grad is not None and float(mv.grad.abs().sum()) > 0.0, (term, moves)
        assert fz.grad is None or float(fz.grad.abs().sum()) == 0.0, (term, frozen)


def test_every_quantizer_parameter_receives_a_finite_gradient():
    K = 4
    _i, inp, lab, layout = _batch(K)
    m = _model(K)
    out = _fwd(m, inp, lab, layout)
    out["loss"].backward()
    for n, p in m.tul_vq.named_parameters():
        assert p.grad is not None, n
        assert torch.isfinite(p.grad).all(), n
        assert float(p.grad.abs().sum()) > 0.0, n
    bad = [n for n, p in m.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert bad == []


# ── 4. THE TWO VQ-VAE TERMS ──────────────────────────────────────────────────

def test_the_commitment_and_codebook_terms_are_positive_and_in_the_loss():
    K = 4
    _i, inp, lab, layout = _batch(K)
    m = _model(K)
    out = _fwd(m, inp, lab, layout)
    for k in ("vq", "vq_weighted", "vq_commit", "vq_codebook_loss"):
        assert k in out, k
        assert float(out[k]) > 0.0, k
    beta = m.cfg.tul.vq_beta
    assert float(out["vq"]) == pytest.approx(
        float(out["vq_codebook_loss"]) + beta * float(out["vq_commit"]), rel=1e-6)
    assert float(out["vq_weighted"]) == pytest.approx(
        m.cfg.tul.vq_weight * float(out["vq"]), rel=1e-6)


def test_the_weighted_twin_is_detached_and_train_py_subtracts_it():
    """`spandec_weighted`'s contract: the twin is a DETACHED float train.py subtracts, so
    train/loss and the val loss stay the model's CE and an arm stays comparable to its
    control (the spectral-penalty precedent). Two halves: the twin carries no graph, and
    the key is in train.py's subtraction lists."""
    import inspect

    from morph.training import train as train_mod
    K = 4
    _i, inp, lab, layout = _batch(K)
    m = _model(K, vq_weight=3.0)
    out = _fwd(m, inp, lab, layout)
    assert float(out["vq_weighted"]) == pytest.approx(3.0 * float(out["vq"]), rel=1e-6)
    for k in ("vq", "vq_weighted", "vq_commit", "vq_codebook_loss"):
        assert not out[k].requires_grad, k
    assert out["loss"].requires_grad
    src = inspect.getsource(train_mod)
    assert src.count('"vq_weighted"') >= 3, "train.py must subtract vq_weighted"


def test_vq_weight_scales_the_terms_contribution_and_nothing_else():
    K = 4
    _i, inp, lab, layout = _batch(K)
    a = _fwd(_model(K, vq_weight=1.0), inp, lab, layout)
    b = _fwd(_model(K, vq_weight=5.0), inp, lab, layout)
    assert float(a["vq"]) == pytest.approx(float(b["vq"]), rel=1e-9)
    assert float(b["loss"]) - float(a["loss"]) == pytest.approx(
        4.0 * float(a["vq"]), rel=1e-5)


# ── 5. THE DECODER GRADES THE DEQUANTIZED THOUGHT ────────────────────────────

def test_the_span_decoder_grades_the_mean_of_the_k_lifted_codes():
    """Spied on the real `_tul_spandec_loss` call, not re-derived."""
    K = 4
    _i, inp, lab, layout = _batch(K)
    m = _model(K)
    cap = {}
    real_vq = m.tul_vq.forward

    def spy_vq(z, valid):
        cells, deq, out = real_vq(z, valid)
        cap["cells"] = cells.detach().clone()
        cap["exit"] = z.detach().clone()
        return cells, deq, out

    m.tul_vq.forward = spy_vq
    real_dec = m._tul_spandec_loss

    def spy_dec(h_slots, *a, **kw):
        cap["graded"] = h_slots.detach().clone()
        return real_dec(h_slots, *a, **kw)

    m._tul_spandec_loss = spy_dec
    _fwd(m, inp, lab, layout)
    assert torch.allclose(cap["graded"], cap["cells"].mean(dim=2), atol=1e-6)
    # And NOT the continuous exit state the coda never sees.
    assert not torch.allclose(cap["graded"], cap["exit"], atol=1e-3)


# ── 6. THE INSTRUMENTS ───────────────────────────────────────────────────────

def test_perplexity_is_in_the_open_unit_to_codebook_range_and_counts_used_codes():
    K, C = 4, 32
    _i, inp, lab, layout = _batch(K)
    m = _model(K, codebook=C)
    out = _fwd(m, inp, lab, layout)
    ppl, used = float(out["vq_perplexity"]), float(out["vq_used"])
    assert 1.0 < ppl <= C, ppl
    assert 1.0 <= used <= C
    assert ppl <= used + 1e-4, (ppl, used)      # perplexity can never exceed the support


def test_the_usage_histogram_counts_valid_slots_only():
    """A pad's encoder output is the zero vector, so its similarity row is all zeros and
    `argmax` hands it code 0 — a fake assignment that would pile onto one code and read as
    a collapsing codebook. Recomputed from the recorded INDEX, which is masked at pads by
    separate code, so the two cannot be wrong the same way. (Sabotage S5 dropped the mask
    and the range assertions above did not notice.)"""
    K, C = 4, 32
    _i, inp, lab, layout = _batch(K)
    m = _model(K, codebook=C)
    cap = {}
    real = m.tul_vq.forward

    def spy(z, valid):
        cells, deq, out = real(z, valid)
        cap["out"] = out
        return cells, deq, out

    m.tul_vq.forward = spy
    out = _fwd(m, inp, lab, layout)
    idx = cap["out"]["index"]
    assert bool((idx < 0).any()), "the fixture must contain a pad slot"
    live = idx[idx >= 0]
    counts = torch.bincount(live, minlength=C).double()
    p = counts / counts.sum()
    want_ppl = float(torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum()))
    assert float(out["vq_used"]) == float(int((counts > 0).sum()))
    assert float(out["vq_perplexity"]) == pytest.approx(want_ppl, rel=1e-5)
    # And it is NOT the number a pad-contaminated histogram would give.
    all_counts = torch.bincount(idx.reshape(-1).clamp(min=0), minlength=C).double()
    q = all_counts / all_counts.sum()
    bad = float(torch.exp(-(q * torch.log(q.clamp(min=1e-12))).sum()))
    assert abs(bad - want_ppl) > 1e-3, (bad, want_ppl)


def test_a_collapsed_codebook_reads_perplexity_one():
    """The instrument has to be able to say the bad thing, or it says nothing."""
    vq = TULThoughtVQ(8, codes=2, codebook=16, dim=4, groups=1, beta=0.25)
    with torch.no_grad():
        vq.vq_E.copy_(torch.ones_like(vq.vq_E))   # every row the same direction
    z = torch.randn(2, 5, 8)
    valid = torch.ones(2, 5, dtype=torch.bool)
    _c, _d, out = vq(z, valid)
    assert out["vq_perplexity"] == pytest.approx(1.0, abs=1e-5)
    assert out["vq_used"] == 1.0


def test_the_probe_reads_the_lifted_cells_and_reports_the_within_slot_rank():
    K = 4
    _i, inp, lab, layout = _batch(K)
    m = _model(K).eval()
    pr = m.tul_slot_state_probe(torch.as_tensor(inp), layout)
    assert pr["slot_cells"] == float(K)
    assert "slot_cell_eff_rank" in pr and "slot_cell_pairwise_cos" in pr
    assert 1.0 <= pr["slot_cell_eff_rank"] <= K + 1e-6
    # The headline rank is over the S*K LIFTED cells, so it must move off the ruler's
    # single-state reading at the same fixture.
    ruler = _model(0, prefix_k=K).eval().tul_slot_state_probe(torch.as_tensor(inp), layout)
    assert "slot_cell_eff_rank" not in ruler
    assert pr["slot_eff_rank"] != ruler["slot_eff_rank"]


@pytest.mark.parametrize("mode", ["normal", "zero", "shuffle", "all_slots"])
def test_a_vq_model_runs_every_worth_profile_mode(mode):
    K = 4
    _i, inp, lab, layout = _batch(K)
    m = _model(K).eval()
    with torch.no_grad():
        res = m.tul_forward_ablated(inp, lab, layout, plan_mode=mode)
    assert torch.isfinite(res["loss"])


def test_zero_and_all_slots_agree_under_strict_with_a_quantizer():
    """Under strict geometry the other cross-span routes are already cut on every forward,
    so `all_slots` IS `zero`. The check exists because the VQ arm adds a new write path and
    a leak there would show up exactly here."""
    K = 4
    _i, inp, lab, layout = _batch(K)
    m = _model(K).eval()
    with torch.no_grad():
        a = m.tul_forward_ablated(inp, lab, layout, plan_mode="zero")["loss"]
        b = m.tul_forward_ablated(inp, lab, layout, plan_mode="all_slots")["loss"]
    assert torch.equal(a, b)


def test_the_depth_lever_still_moves_a_vq_model():
    """A K-curve on this arm has to be a K-curve: forced depth must change the loss."""
    K = 4
    _i, inp, lab, layout = _batch(K)
    m = _model(K).eval()
    seen = []
    for d in (1, 2, 4):
        m.cfg.tul.slot_mean_depth = d
        with torch.no_grad():
            seen.append(float(m.tul_forward_ablated(inp, lab, layout)["loss"]))
    assert len(set(seen)) == 3, f"the depth lever did nothing: {seen}"


# ── 7. PRODUCT QUANTIZATION AND THE DEAD-CODE RESET ──────────────────────────

def test_groups_split_a_cell_into_g_symbols():
    vq = TULThoughtVQ(64, codes=4, codebook=32, dim=16, groups=4, beta=0.25)
    assert vq.d_g == 4
    z = torch.randn(2, 5, 64)
    valid = torch.ones(2, 5, dtype=torch.bool)
    cells, deq, out = vq(z, valid)
    assert out["index"].shape == (2, 5, 4, 4)          # [B, S, K, G]
    assert out["vq_n_codes"] == 16.0
    assert cells.shape == (2, 5, 4, 64)
    assert torch.allclose(deq, cells.mean(dim=2))


def test_the_dead_code_reset_revives_an_unused_code_and_only_in_training():
    vq = TULThoughtVQ(8, codes=2, codebook=8, dim=4, groups=1, beta=0.25, reset_after=1)
    z = torch.randn(2, 6, 8)
    valid = torch.ones(2, 6, dtype=torch.bool)
    vq.train()
    before = vq.vq_E.detach().clone()
    _c, _d, a = vq(z, valid)
    assert a["vq_used"] < 8.0, "the fixture must leave a dead code for the reset to hit"
    moved = (~torch.isclose(before, vq.vq_E.detach())).any(dim=-1)
    assert bool(moved.any()), "no codebook row was re-seeded"
    # Every re-seeded row is now a unit vector (it is a copy of a normalised encoder
    # output), and the reset draws NO random number.
    for i in moved.nonzero(as_tuple=True)[0]:
        assert float(vq.vq_E[i].norm()) == pytest.approx(1.0, abs=1e-4)
    # Eval never moves the codebook: the instruments run many forwards with no step.
    vq.eval()
    frozen = vq.vq_E.detach().clone()
    for _ in range(3):
        vq(z, valid)
    assert torch.equal(frozen, vq.vq_E.detach())


def test_the_assignment_is_made_in_fp32_whatever_the_carrier_dtype():
    """The discrete choice must not depend on the rounding.

    Under autocast the carrier is bf16, and an `argmax` over cosine similarities in bf16 can
    flip between two close codes for numerical reasons alone — a run's code assignments
    would then be a property of the arithmetic, not of the state. The quantizer pins
    everything downstream of the `W_vq` matmul to fp32, so a bf16 carrier gives the SAME
    assignment as its fp32 twin on the same input and the two loss terms come back fp32.
    """
    vq = TULThoughtVQ(16, codes=2, codebook=8, dim=8, groups=1, beta=0.25)
    z = torch.randn(2, 4, 16)
    valid = torch.ones(2, 4, dtype=torch.bool)
    _c32, d32, o32 = vq(z, valid)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cb, db, ob = vq(z.to(torch.bfloat16), valid)
    assert torch.equal(o32["index"], ob["index"])
    assert o32["loss"].dtype == torch.float32 and ob["loss"].dtype == torch.float32
    assert float(ob["vq_perplexity"]) == pytest.approx(float(o32["vq_perplexity"]), rel=1e-5)
    # The lift itself follows the carrier, as every other Linear in the model does.
    assert cb.dtype == torch.bfloat16 and db.dtype == torch.bfloat16
    assert torch.isfinite(cb.float()).all()
    assert torch.allclose(db.float(), d32, atol=5e-2)


def test_the_reset_is_off_by_default_and_never_moves_the_codebook():
    vq = TULThoughtVQ(8, codes=2, codebook=8, dim=4, groups=1, beta=0.25).train()
    z = torch.randn(2, 6, 8)
    valid = torch.ones(2, 6, dtype=torch.bool)
    before = vq.vq_E.detach().clone()
    for _ in range(3):
        vq(z, valid)
    assert torch.equal(before, vq.vq_E.detach())


# ── 8. REFUSALS ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw", [
    dict(slot_cells=4),
    dict(prefix_source="trajectory"),
    dict(prefix_source="exit_repeat"),
    dict(tokens_through_core=True),
    dict(loop_reads_tokens=True),
    dict(detach_z=True),
    dict(db_loop=True),
    dict(slot_chain=True),
    dict(grad_pass=True),
    dict(oracle_z=True),
    dict(spandec_per_pass=True),
    dict(mux_every_pass=True),
    dict(mux_stage_all=True),
    dict(mux_stage_own_iters=2),
    dict(coda_span_heads=2),
    dict(coda_sees_slots=False),
    dict(coda_token_cut=4),
])
def test_every_conflicting_mechanism_raises(kw):
    with pytest.raises((NotImplementedError, ValueError)):
        _tul(prefix_k=4, vq_codes=4, **kw)


def test_the_gate_raises_with_a_quantizer():
    from morph.model.tul import TULGateConfig
    with pytest.raises(NotImplementedError, match="tul.vq_codes with tul.gate"):
        _tul(prefix_k=4, vq_codes=4, gate=TULGateConfig(k_max=32))


@pytest.mark.parametrize("kw", [
    dict(vq_codebook=256), dict(vq_dim=8), dict(vq_groups=2),
    dict(vq_beta=0.5), dict(vq_weight=2.0), dict(vq_reset_after=5),
])
def test_a_vq_knob_without_the_quantizer_raises(kw):
    with pytest.raises(ValueError, match="set with tul.vq_codes=0"):
        _tul(**kw)


def test_one_code_position_is_refused():
    with pytest.raises(ValueError, match="which is a lookup table"):
        _tul(prefix_k=1, vq_codes=1)


def test_bad_shapes_raise_at_construction():
    # d_model 100 is not divisible by 8 and vq_dim is 0
    with pytest.raises(ValueError, match="does not divide d_model"):
        TULThoughtVQ(100, codes=8, codebook=32, dim=0, groups=1, beta=0.25)
    with pytest.raises(ValueError, match="must divide the code width"):
        TULThoughtVQ(64, codes=4, codebook=32, dim=6, groups=4, beta=0.25)
    with pytest.raises(ValueError, match="needs codes >= 2"):
        TULThoughtVQ(64, codes=1, codebook=32, dim=8, groups=1, beta=0.25)
    with pytest.raises(ValueError, match="needs codebook >= 2"):
        TULThoughtVQ(64, codes=4, codebook=1, dim=8, groups=1, beta=0.25)


def test_a_coreless_vq_model_raises_at_build():
    with pytest.raises(ValueError, match="needs a core loop"):
        MORPHTransformer(_tiny(n_core=0, tul=_tul(prefix_k=4, vq_codes=4, vq_codebook=32,
                                                  vq_dim=16)))


def test_unknown_vq_keys_are_known_to_tul_setup():
    from morph.training.tul_setup import KNOWN_TUL_KEYS
    for k in ("vq_codes", "vq_codebook", "vq_dim", "vq_groups", "vq_beta", "vq_weight",
              "vq_reset_after"):
        assert k in KNOWN_TUL_KEYS, k


# ── 9. THE SHIPPED CONFIGS ───────────────────────────────────────────────────

def test_the_shipped_configs_compose_and_carry_the_knobs():
    import os

    from hydra import compose, initialize_config_dir

    from morph.training.tul_setup import build_tul_runtime
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    want = {"tul_slot_spandec_strict_vq8": 8, "tul_slot_spandec_strict_vq4": 4}
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.join(root, "morph", "configs")):
        for name, k in want.items():
            cfg = compose(config_name=name, overrides=[])
            rt = build_tul_runtime(cfg)
            mc = rt.model_cfg
            assert mc.vq_codes == k and mc.prefix_k == k
            assert mc.vq_codebook == 512 and mc.vq_groups == 1
            assert mc.vq_beta == 0.25 and mc.vq_weight == 1.0
            assert mc.vq_reset_after == 0
            assert mc.vq_dim == 0            # -> d_model // K
            assert mc.tg_geometry == "strict" and mc.tg_coda_prefix_reach == "all"
            assert mc.spandec and mc.mux_beta == 0.0
            assert mc.slot_cells == 1 and mc.prefix_source == "exit"
            assert rt.data_cfg.spec_for(cfg.data.seq_len).l_total == 1024 + k * 64
