"""`tul.spandec_per_pass` — one planning target per PASS, growing by a span a pass.

Arm `slot-spandec-strict-perpass`, 2026-09-12. Amendment 3 of the strict panel found the
first per-pass K-curve that moved (`tg_coda_prefix_reach: prev` + `loop_reach: 1`, token
K1-K6 +0.0163 at CE parity) and it did it by BLINDING the coda, so the loop had to carry
history. Wolfe: "z has to hold the history when it should hold the present next thought
that needs decoding. Our objectives are still poor." This knob asks for the depth from the
OBJECTIVE instead: pass `t` is graded on spans `s+1 .. s+min(t, cap)`, through the SAME
span decoder, while the exit term stays the shipped H = 1 "next thought" and the coda keeps
its full reach.

What this file pins, one test per invariant:

* **Off is nothing.** The term is never called, the trajectory is never collected, and no
  position table is built.
* **On is PURELY ADDITIVE.** Every shared parameter is byte-identical to an off-model built
  from the same seed (the extra position table is zero-init and the model's RNG stream is
  untouched), and `loss - spandec_pass_weighted` equals the off-model's loss BIT FOR BIT.
* **The targets are the right spans.** At pass `t`, block `h` holds span `s+h`'s tokens —
  checked against a Python oracle that walks `bag_id` itself, not against a second call of
  the function under test.
* **The horizon is `min(t, cap)`.**
* **The depth mask is exact.** A slot of realised depth `d` is graded at passes 1..d and at
  no other pass.
* **The gradient reaches the core through pass 1 AND through pass 3**, individually.

Three source-level sabotages were run against this file on 2026-09-12 (grade the wrong span
set; drop the depth mask; detach the trajectory) and each was caught — see
`lab/experiments/planned/2026-09-12-arc-objective-arms.md`.

The Hydra compose-and-build test for `tul_slot_spandec_strict_perpass.yaml` lives in
`tests/test_tul_objective_arms.py` with this batch's other arms — one place where every
2026-09-12 objective arm is composed, mapped through `build_tul_runtime`, built at the
PANEL's real budgets and run.

CPU only, fp32, tiny config — the `tests/test_tul_oracle_z.py` fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import morph.model.transformer as tfm
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
                slot_depth_fixed=4, slot_max_depth=4)
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


def _core(m: MORPHTransformer, inp, layout):
    """The loop's trajectory and the realised per-slot depths, on the real forward."""
    x, x0, bg = m._tul_front(inp, layout)
    _xn, _h, depths, _g, db_traj, _gr, _mk = m._tul_core(x, x0, bg, layout, input_ids=inp)
    return db_traj, depths


def _span_tokens(layout, ids, b: int) -> list[list[int]]:
    """Span index -> its token ids, IN ORDER, walked off the layout by hand.

    The independent oracle. It reads `bag_id` and `slot_mask` and nothing else, so a test
    written against it cannot be satisfied by `span_slots` agreeing with itself.
    """
    S = int(layout.slot_index.shape[1])
    out: list[list[int]] = [[] for _ in range(S)]
    for p in range(ids.shape[1]):
        if bool(layout.slot_mask[b, p]):
            continue
        k = int(layout.bag_id[b, p])
        if 0 <= k < S:
            out[k].append(int(ids[b, p]))
    return out


def _capture(monkeypatch) -> list[torch.Tensor]:
    """Record the LABEL tensor of every fused-CE call, call the real kernel through."""
    seen: list[torch.Tensor] = []
    real = tfm.fused_linear_cross_entropy

    def spy(x, w, labels, *a, **k):
        seen.append(labels.detach().clone())
        return real(x, w, labels, *a, **k)

    monkeypatch.setattr(tfm, "fused_linear_cross_entropy", spy)
    return seen


# ── off is nothing ───────────────────────────────────────────────────────────

def test_off_never_calls_the_term(monkeypatch):
    def boom(*a, **k):
        raise AssertionError(
            "_tul_spandec_per_pass_loss ran on a model with tul.spandec_per_pass=false")
    monkeypatch.setattr(MORPHTransformer, "_tul_spandec_per_pass_loss", boom)
    out = _run(_model())
    assert "spandec_pass" not in out and "spandec_pass_weighted" not in out


def test_off_builds_no_position_table_and_no_trajectory():
    m = _model()
    assert m.tul_spandec.pos_pass is None
    assert "tul_spandec.pos_pass" not in dict(m.named_parameters())
    inp, _lab, layout, _ = _batch()
    assert _core(m, inp, layout)[0] is None, \
        "db_traj is collected on a model that asked for nothing"


def test_on_builds_the_trajectory_and_the_table():
    m = _model(spandec_per_pass=True, spandec_pass_horizon_max=3, spandec_pass_tokens=4)
    assert m.tul_spandec.pos_pass is not None
    assert tuple(m.tul_spandec.pos_pass.shape) == (12, m.cfg.d_model)
    assert float(m.tul_spandec.pos_pass.detach().abs().sum()) == 0.0, \
        "the table must be zero-init"
    inp, _lab, layout, _ = _batch()
    traj, _d = _core(m, inp, layout)
    assert traj is not None and len(traj) == 5          # seed + 4 fixed passes


def test_eval_pays_nothing(monkeypatch):
    """A forced-depth sweep must read the ruler's columns, not this arm's cost."""
    def boom(*a, **k):
        raise AssertionError("the per-pass target ran on an EVAL forward")
    monkeypatch.setattr(MORPHTransformer, "_tul_spandec_per_pass_loss", boom)
    m = _model(spandec_per_pass=True).eval()
    inp, lab, layout, _ = _batch()
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    assert "spandec_pass" not in out


# ── on is purely additive ────────────────────────────────────────────────────

def test_on_shifts_no_weight_and_adds_only_its_own_term():
    """The two halves of "bit-identical when off", measured rather than read.

    1. Every parameter an off-model has is byte-identical in the on-model, so the extra
       position table cost the model nothing (it is zero-init and draws no RNG).
    2. `loss - spandec_pass_weighted` equals the off-model's loss BIT FOR BIT, so the term
       is added and changes nothing about the forward that produced the CE.
    """
    off, on = _model(), _model(spandec_per_pass=True)
    d_off, d_on = dict(off.named_parameters()), dict(on.named_parameters())
    assert sorted(set(d_off) - set(d_on)) == []
    assert sorted(set(d_on) - set(d_off)) == ["tul_spandec.pos_pass"]
    for n, p in d_off.items():
        assert torch.equal(p, d_on[n]), f"building the per-pass table moved {n}"
    o_off, o_on = _run(off), _run(on)
    assert torch.equal(o_on["loss"] - o_on["spandec_pass_weighted"], o_off["loss"])


def test_the_term_is_positive_and_carries_its_weight():
    m = _model(spandec_per_pass=True, spandec_pass_weight=2.5)
    out = _run(m)
    assert float(out["spandec_pass"]) > 0.0
    assert abs(float(out["spandec_pass_weighted"])
               - 2.5 * float(out["spandec_pass"])) < 1e-5
    assert float(out["spandec_pass_terms"]) == 4.0       # slot_depth_fixed=4


# ── the targets are the right spans ──────────────────────────────────────────

def _check_targets(m: MORPHTransformer, monkeypatch) -> None:
    """At pass t, block h of the graded sequence holds span s+h's tokens. Raises if not."""
    tc = m.cfg.tul
    J1, cap = int(tc.spandec_pass_tokens), int(tc.spandec_pass_horizon_max)
    inp, _lab, layout, _ = _batch()
    torch.manual_seed(3)
    traj, depths = _core(m, inp, layout)
    seen = _capture(monkeypatch)
    m._tul_spandec_per_pass_loss(traj, depths, inp, layout)
    T = len(traj) - 1
    assert len(seen) == T, f"expected one CE per pass, got {len(seen)} for {T} passes"
    S = int(layout.slot_index.shape[1])
    B = inp.shape[0]
    n_checked = 0
    for t in range(1, T + 1):
        H = min(t, cap)
        lab = seen[t - 1].reshape(B, S, H * J1)
        assert lab.shape[2] == H * J1, f"pass {t}: horizon {lab.shape[2] // J1} != {H}"
        for b in range(B):
            spans = _span_tokens(layout, inp, b)
            for s in range(S):
                graded = bool(layout.slot_valid[b, s]) and int(depths[b, s]) >= t
                for h in range(1, H + 1):
                    for j in range(J1):
                        got = int(lab[b, s, (h - 1) * J1 + j])
                        k = s + h
                        ok = (graded and k < S and bool(layout.slot_valid[b, k])
                              and j < len(spans[k]))
                        want = spans[k][j] if ok else -100
                        assert got == want, (
                            f"pass {t} slot {s} block {h} offset {j}: label {got} != "
                            f"{want} (span {k})")
                        n_checked += 1
    assert n_checked > 0


def test_every_pass_grades_the_right_spans(monkeypatch):
    _check_targets(_model(spandec_per_pass=True, spandec_pass_horizon_max=3,
                          spandec_pass_tokens=4), monkeypatch)


def test_the_horizon_at_pass_t_is_min_t_cap(monkeypatch):
    """The growth law, read straight off the graded width, at a cap the fixture reaches."""
    m = _model(spandec_per_pass=True, spandec_pass_horizon_max=2, spandec_pass_tokens=4)
    inp, _lab, layout, _ = _batch()
    torch.manual_seed(3)
    traj, depths = _core(m, inp, layout)
    seen = _capture(monkeypatch)
    m._tul_spandec_per_pass_loss(traj, depths, inp, layout)
    B, S = inp.shape[0], int(layout.slot_index.shape[1])
    widths = [int(s.numel() // (B * S)) // 4 for s in seen]
    assert widths == [1, 2, 2, 2], f"horizons {widths} != min(t, 2) for t = 1..4"


# ── the depth mask is exact ──────────────────────────────────────────────────

def _check_depth_mask(m: MORPHTransformer, monkeypatch) -> None:
    """A slot of realised depth d is graded at passes 1..d and NOWHERE else."""
    inp, _lab, layout, _ = _batch()
    torch.manual_seed(3)
    traj, _d = _core(m, inp, layout)
    B, S = inp.shape[0], int(layout.slot_index.shape[1])
    # A HAND-BUILT depth per slot, so the mask is read against a number this test chose
    # rather than against the Poisson draw the forward happened to make.
    depths = torch.arange(1, S + 1).clamp(max=len(traj) - 1).unsqueeze(0).repeat(B, 1)
    seen = _capture(monkeypatch)
    m._tul_spandec_per_pass_loss(traj, depths, inp, layout)
    T = len(traj) - 1
    for t in range(1, T + 1):
        lab = seen[t - 1].reshape(B, S, -1)
        for b in range(B):
            for s in range(S):
                live = (lab[b, s] != -100).any().item()
                if int(depths[b, s]) < t:
                    assert not live, (
                        f"slot {s} (depth {int(depths[b, s])}) was graded at pass {t}")
                elif bool(layout.slot_valid[b, s]) and s + 1 < S \
                        and bool(layout.slot_valid[b, s + 1]):
                    assert live, (
                        f"slot {s} (depth {int(depths[b, s])}) was NOT graded at pass {t}")


def test_a_slot_is_graded_at_exactly_the_passes_its_depth_reaches(monkeypatch):
    _check_depth_mask(_model(spandec_per_pass=True, spandec_pass_horizon_max=3,
                             spandec_pass_tokens=4), monkeypatch)


# ── the gradient reaches the core, pass by pass ──────────────────────────────

def _grad_through_pass(m: MORPHTransformer, t_live: int) -> float:
    """Total core-parameter gradient when ONLY pass `t_live`'s state carries gradient."""
    inp, _lab, layout, _ = _batch()
    torch.manual_seed(3)
    traj, depths = _core(m, inp, layout)
    cut = [traj[0]] + [h if i == t_live else h.detach()
                       for i, h in enumerate(traj[1:], start=1)]
    term = m._tul_spandec_per_pass_loss(cut, depths, inp, layout)
    core = [p for n, p in m.named_parameters() if n.startswith("core.")]
    g = torch.autograd.grad(term, core, allow_unused=True)
    return sum(float(x.abs().sum()) for x in g if x is not None)


@pytest.mark.parametrize("t", [1, 3])
def test_the_gradient_reaches_the_core_through_that_pass(t):
    """The arm's mechanism: every pass's state is on the path to the core's weights.

    With only pass `t` live and every other pass detached, a non-zero core gradient can
    only have come through pass `t`'s own state. Run at t = 1 and t = 3, the two the
    pre-registration names.
    """
    m = _model(spandec_per_pass=True, spandec_pass_horizon_max=3, spandec_pass_tokens=4)
    assert _grad_through_pass(m, t) > 0.0


def test_detaching_the_whole_trajectory_kills_the_core_gradient():
    """The null the test above is measured against."""
    m = _model(spandec_per_pass=True, spandec_pass_horizon_max=3, spandec_pass_tokens=4)
    inp, _lab, layout, _ = _batch()
    torch.manual_seed(3)
    traj, depths = _core(m, inp, layout)
    term = m._tul_spandec_per_pass_loss([h.detach() for h in traj], depths, inp, layout)
    core = [p for n, p in m.named_parameters() if n.startswith("core.")]
    g = torch.autograd.grad(term, core, allow_unused=True)
    assert all(x is None or float(x.abs().sum()) == 0.0 for x in g)


def test_the_decoder_is_trained_by_the_term():
    """Deliberate, and the opposite of `tul.oracle_z`: ONE reader, one notion of a plan."""
    m = _model(spandec_per_pass=True, spandec_pass_horizon_max=3, spandec_pass_tokens=4)
    inp, _lab, layout, _ = _batch()
    torch.manual_seed(3)
    traj, depths = _core(m, inp, layout)
    term = m._tul_spandec_per_pass_loss(traj, depths, inp, layout)
    dec = [p for n, p in m.named_parameters() if n.startswith("tul_spandec.")]
    g = torch.autograd.grad(term, dec, allow_unused=True)
    assert any(x is not None and float(x.abs().sum()) > 0 for x in g)


# ── the refusals ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw,match", [
    (dict(spandec_per_pass=True, spandec=False, spandec_layers=2, spandec_max_tokens=0),
     "requires tul.spandec"),
    (dict(spandec_per_pass=True, spandec_pass_horizon_max=0), "pass_horizon_max"),
    (dict(spandec_per_pass=True, spandec_pass_tokens=1), "pass_tokens"),
    (dict(spandec_per_pass=True, spandec_pass_weight=0.0), "pass_weight"),
    (dict(spandec_per_pass=True, spandec_horizon=3), "two-factor"),
    (dict(spandec_per_pass=True, oracle_z=True), "oracle_z"),
    (dict(spandec_per_pass=True, db_loop=True), "db_loop"),
    (dict(spandec=False, spandec_layers=2, spandec_max_tokens=0, spandec_pass_tokens=4),
     "silently ignored"),
])
def test_per_pass_refusals(kw, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _tul(**kw)


def test_per_pass_needs_a_core_loop():
    with pytest.raises(ValueError, match="needs a core loop"):
        MORPHTransformer(_tiny(n_core=0, tul=_tul(spandec_per_pass=True)))
