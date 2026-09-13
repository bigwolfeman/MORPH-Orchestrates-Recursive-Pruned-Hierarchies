"""`lab/divergence/slot_rank_anatomy.py` — the contracts its readings depend on.

Four things can make this instrument print plausible numbers that mean nothing:

1. the participation ratio is not the one `val/slot_eff_rank` uses, so "5.76" is
   reproduced by coincidence or not at all;
2. the raw/centered split is vacuous — if both columns always moved together the whole
   shared-offset question could not be answered;
3. a stage is not the state it claims to be (the exit is a re-run that drifted, the write
   is not what the coda reads);
4. the front is rebuilt WITHOUT the model's own relation, so every state on a
   `tg_geometry="strict"` arm is off-distribution. That is the 2026-09-13 defect in
   `spandec_horizon_grid.py`, which read the depth effect with the wrong sign.

One test per row. CPU only, fp32, tiny config — the `tests/test_tul_strict_geometry.py`
fixtures.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

from morph.model.fm_planner import effective_rank, mean_pairwise_cos
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig, gather_valid
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "lab", "divergence"))
from slot_rank_anatomy import (  # noqa: E402
    capture_batch,
    mean_pairwise_cos as anat_cos,
    participation_ratio,
    rank_stats,
    residual_fraction,
    spectrum,
)

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=2, max_depth=3, bptt_depth=3,
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
    base = dict(prefix_k=2, slot_id=4, tg_restrict=True, tg_restrict_scope="all",
                emit_weight=0.0, token_state_dropout=0.0, mux_beta=0.0)
    base.update(kw)
    return TULConfig(**base)


def _model(seed: int = 1234, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**tul_kw)))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.eval().float()


def _pack(B: int = 2, n: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::8] = DOT
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
    inp, lab, layout, _stats = slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)
    return inp, lab, layout


# ── (a) the participation ratio is the model's ───────────────────────────────

def test_participation_ratio_on_a_known_spectrum():
    """A matrix built with singular values ``s`` must read ``(Σs²)²/Σs⁴``."""
    torch.manual_seed(0)
    n, d = 6, 16
    u = torch.linalg.qr(torch.randn(n, 4, dtype=torch.float64))[0]      # [n, 4], UᵀU = I
    v = torch.linalg.qr(torch.randn(d, 4, dtype=torch.float64))[0].T    # [4, d], VVᵀ = I
    lam = torch.tensor([4.0, 2.0, 1.0, 1.0], dtype=torch.float64)       # eigenvalues wanted
    m = u @ torch.diag(lam.sqrt()) @ v
    want = float((lam.sum() ** 2) / (lam * lam).sum())                  # 64 / 22
    assert abs(want - 64.0 / 22.0) < 1e-12
    assert abs(participation_ratio(spectrum(m)) - want) < 1e-8
    assert abs(rank_stats(m)["eff_rank_raw"] - want) < 1e-8


def test_the_centered_rank_is_the_models_effective_rank():
    """`eff_rank_centered` IS `morph/model/fm_planner.py::effective_rank`, the quantity
    the trainer logs as `val/slot_eff_rank`. A different definition would make the
    reproduction check meaningless."""
    torch.manual_seed(1)
    x = torch.randn(40, 24) @ torch.randn(24, 64) + 3.0
    ones = torch.ones(1, 40, dtype=torch.bool)
    mine = rank_stats(x)
    assert abs(mine["eff_rank_centered"] - effective_rank(x.unsqueeze(0), ones)) < 1e-3
    assert abs(mine["cos_raw"] - mean_pairwise_cos(x.unsqueeze(0), ones)) < 1e-5
    assert abs(anat_cos(x) - mean_pairwise_cos(x.unsqueeze(0), ones)) < 1e-5


# ── (b) the raw / centered split is not vacuous ───────────────────────────────

def test_a_shared_offset_reads_rank_one_raw_and_full_rank_centered():
    """The whole question this instrument exists for, on synthetic data with a KNOWN
    answer: one common direction plus a small isotropic residual must read
    ``eff_rank_raw`` ~1 while ``eff_rank_centered`` reads the residual's own spread. If
    the two columns moved together, 'shared offset' and 'true collapse' could not be told
    apart.

    The centered number is NOT n−1: the participation ratio of a sample covariance built
    from n=32 isotropic draws in d=64 dimensions is ~20 (Marchenko-Pastur spread), not 31,
    so the bound is stated against the measured isotropic control in the same test rather
    than against the algebraic rank."""
    torch.manual_seed(2)
    n, d = 32, 64
    offset = torch.randn(d) * 10.0
    resid = torch.randn(n, d)
    x = offset.unsqueeze(0) + resid * 0.01
    st = rank_stats(x)
    iso = rank_stats(resid)                       # the SAME residual, no offset
    assert st["eff_rank_raw"] < 1.01, st["eff_rank_raw"]
    assert st["eff_rank_centered"] > 15.0, st["eff_rank_centered"]
    assert st["eff_rank_centered"] > 15.0 * st["eff_rank_raw"]
    # the offset changed the RAW geometry and left the centered one alone
    assert abs(st["eff_rank_centered"] - iso["eff_rank_centered"]) < 0.5
    assert st["cos_raw"] > 0.999 and abs(st["cos_centered"]) < 0.2
    assert st["row_mean_norm_ratio"] > 0.999
    assert iso["eff_rank_raw"] > 15.0 and iso["row_mean_norm_ratio"] < 0.3


def test_residual_fraction_sees_a_new_direction():
    """`resid_prev` is 0 inside the previous span and ~1 orthogonal to it."""
    torch.manual_seed(3)
    d = 32
    basis = torch.linalg.qr(torch.randn(d, d))[0]
    prev = torch.randn(20, 4) @ basis[:, :4].T            # spans dims 0..3
    inside = torch.randn(20, 4) @ basis[:, :4].T
    outside = torch.randn(20, 4) @ basis[:, 8:12].T
    assert residual_fraction(prev, inside, k=4) < 1e-5
    assert residual_fraction(prev, outside, k=4) > 0.999


# ── (c) the stages are the model's own states ────────────────────────────────

def test_the_exit_stage_is_the_models_own_h_slots():
    m = _model(tg_geometry="strict")
    inp, _lab, layout = _pack()
    packed, valid = capture_batch(m, inp, layout, depth=int(m.cfg.mean_depth))
    fkw, freset, _ckw, _creset = m._tul_tg_kwargs(layout)
    with torch.no_grad():
        x, x0, bg = m._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
        h = m._tul_core(x, x0, bg, layout, input_ids=inp)[1]
    assert torch.equal(packed["sX_exit"]["raw"], h.mean(dim=2).float())
    assert torch.equal(packed["sX_exit"]["readout"], m._readout(h).float())
    assert valid["sX_exit"].shape == layout.slot_valid.shape


def test_the_write_stage_is_what_the_coda_reads():
    """`sW_write` must equal the values `prefix_project` hands the coda on the SHIPPED
    forward — spied there, not re-derived."""
    m = _model(tg_geometry="strict")
    inp, _lab, layout = _pack()
    packed, valid = capture_batch(m, inp, layout, depth=int(m.cfg.mean_depth))
    seen: list[torch.Tensor] = []
    real = m.tul.prefix_project

    def spy(h_slots, lay, l_total, cells=None):
        out = real(h_slots, lay, l_total, cells=cells)
        seen.append(out[0])
        return out

    m.tul.prefix_project = spy
    try:
        with torch.no_grad():
            m.tul_forward_ablated(inp, None, layout, plan_mode="normal")
    finally:
        m.tul.prefix_project = real
    assert len(seen) == 1
    assert torch.equal(packed["sW_write"]["raw"], seen[0].mean(dim=2).float())
    k = int(m.cfg.tul.prefix_k)
    assert valid["sW_write"].shape[1] == layout.slot_valid.shape[1] * k


def test_the_seed_and_entry_stages_are_distinct_and_shaped_right():
    m = _model(tg_geometry="strict")
    inp, _lab, layout = _pack()
    d = int(m.cfg.mean_depth)
    packed, _valid = capture_batch(m, inp, layout, depth=d)
    want = ["s0_seed", "s1_entry"] + [f"pass{i}" for i in range(1, d + 1)] \
        + ["sX_exit", "sW_write"]
    assert list(packed.keys()) == want
    b, s = layout.slot_valid.shape
    for name in want[:-1]:
        assert packed[name]["raw"].shape == (b, s, m.cfg.d_model), name
    # the seed is PRE-prelude: it must not already be the entry state
    assert not torch.equal(packed["s0_seed"]["raw"], packed["s1_entry"]["raw"])
    # and the loop must move the state at all on this fixture, or the stage table is
    # measuring one repeated vector and every pass row would be the same by accident
    assert not torch.equal(packed["s1_entry"]["raw"], packed["pass1"]["raw"])


# ── (d) the instrument honours the model's relation ──────────────────────────

def test_a_strict_model_and_its_restrict_twin_give_different_entry_states():
    """Same weights, same tokens, ONE difference: `tg_geometry`. Under `strict` the
    prelude is same-span only; under `restrict` a token may read any earlier slot cell.
    If `capture_batch` rebuilt the front without `_tul_tg_kwargs` the two entry states
    would be IDENTICAL — that is exactly the defect this guards."""
    strict = _model(tg_geometry="strict")
    restrict = _model(tg_geometry="restrict")
    sa, sb = strict.state_dict(), restrict.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"the twin is not weight-identical at {k}"
    inp, _lab, layout = _pack()
    assert strict._tg_strict and not restrict._tg_strict
    a, _ = capture_batch(strict, inp, layout, depth=int(strict.cfg.mean_depth))
    b, _ = capture_batch(restrict, inp, layout, depth=int(restrict.cfg.mean_depth))
    diff = (a["s1_entry"]["raw"] - b["s1_entry"]["raw"]).abs().max()
    assert float(diff) > 1e-4, (
        "the strict model's loop entry equals its restrict twin's: the instrument is "
        "rebuilding the prelude without the model's own relation")


def test_the_strict_entry_is_not_the_bare_front_entry():
    """The 2026-09-13 defect in its exact shape: a `_tul_front(inp, layout)` with no
    kwargs runs the prelude UNRESTRICTED. The instrument's entry must not equal it."""
    m = _model(tg_geometry="strict")
    inp, _lab, layout = _pack()
    got, _ = capture_batch(m, inp, layout, depth=int(m.cfg.mean_depth))
    with torch.no_grad():
        x, x0, bg = m._tul_front(inp, layout)                 # the bare call, no relation
        bare = m.core_init(gather_valid(m.input_norm(x), layout.slot_index,
                                        layout.slot_valid))
    assert bare.shape == (layout.slot_valid.shape[0], layout.slot_valid.shape[1],
                          4, m.cfg.d_model)
    diff = (got["s1_entry"]["raw"] - bare.mean(dim=2).float()).abs().max()
    assert float(diff) > 1e-4, (
        "the instrument's entry state equals the UNRESTRICTED one — `_tul_tg_kwargs` is "
        "not reaching `_tul_front`")


def test_the_forced_depth_table_reproduces_the_unforced_exit():
    """The whole pass-by-pass reading rests on 'forced depth d == trajectory entry d'.
    `capture_batch` asserts it every batch; this pins that the assertion can FIRE."""
    m = _model(tg_geometry="strict")
    inp, _lab, layout = _pack()
    capture_batch(m, inp, layout, depth=int(m.cfg.mean_depth))       # passes
    with pytest.raises(RuntimeError, match="not reading the trajectory"):
        capture_batch(m, inp, layout, depth=int(m.cfg.mean_depth) + 1, check_exit=True)
