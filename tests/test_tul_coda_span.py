"""`tul.coda_span_heads` — decode the next span from the CODA, all offsets at once.

Arm `slot-spandec-strict-codaspan` / `-codaspan32`, 2026-09-12. Wolfe: "try parallel token
decoding from the coda. Perhaps the coda needing to spit out a lot of the span or all the
span at once changes the behavior." Every span-decoder arm so far grades `z` through a
SEPARATE reader the token CE never touches; this one grades the reader the model ships, at
the slot's last prefix cell, which under the strict geometry carries the looped state and
nothing else.

What this file pins, one test per invariant:

* **Off is nothing.** No module, no parameter, no key in the output, and the term is never
  called.
* **On is PURELY ADDITIVE and RNG-NEUTRAL.** Every shared parameter is byte-identical to an
  off-model built from the same seed — `nn.Linear.reset_parameters` draws before the
  identity init overwrites it, so the construction puts the global stream back — and
  `loss - coda_span_weighted` equals the off-model's loss BIT FOR BIT.
* **Head `j` at slot `k` is scored against span `k+1`'s token `j`**, checked against a
  Python oracle that walks `bag_id` itself.
* **The masking is exact.** A pad slot, a slot with no following span, and every offset
  past a span's end carry `ignore_index`.
* **The read position is the LAST prefix cell** — proved by the gradient with respect to
  the coda readout being non-zero at exactly those positions and nowhere else.
* **The gradient reaches the coda AND the loop's write** (`tul.W_prefix`).

Three source-level sabotages were run against this file on 2026-09-12 (shift the head
offsets; read the slot's OWN span; drop the validity mask) and each was caught — see
`lab/experiments/planned/2026-09-12-arc-objective-arms.md`.

The Hydra compose-and-build test for both codaspan configs lives in
`tests/test_tul_objective_arms.py`, at the panel's real budgets (J 8 AND J 32).

CPU only, fp32, tiny config — the `tests/test_tul_oracle_z.py` fixtures.
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


def _span_tokens(layout, ids, b: int) -> list[list[int]]:
    """Span index -> its token ids, IN ORDER, walked off `bag_id` by hand.

    The independent oracle: it never calls `next_span_slots`, so a test written against it
    cannot be satisfied by the target builder agreeing with itself.
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


def _labels(m: MORPHTransformer, monkeypatch, xh=None):
    """The ONE label tensor the heads' fused CE is called with, plus the fixture."""
    import morph.model.transformer as tfm
    inp, _lab, layout, _ = _batch()
    if xh is None:
        xh = torch.randn(inp.shape[0], layout.slot_mask.shape[1], m.cfg.d_model)
    seen: list[torch.Tensor] = []
    real = tfm.fused_linear_cross_entropy

    def spy(x, w, labels, *a, **k):
        seen.append(labels.detach().clone())
        return real(x, w, labels, *a, **k)

    monkeypatch.setattr(tfm, "fused_linear_cross_entropy", spy)
    m._tul_coda_span_loss(xh, inp, layout)
    assert len(seen) == 1, f"the heads must be ONE chunked CE call, got {len(seen)}"
    J = len(m.coda_span)
    S = int(layout.slot_index.shape[1])
    return seen[0].reshape(inp.shape[0], S, J), inp, layout


# ── off is nothing ───────────────────────────────────────────────────────────

def test_off_builds_nothing_and_never_runs():
    m = _model()
    assert m.coda_span is None
    assert not [n for n in dict(m.named_parameters()) if n.startswith("coda_span")]
    out = _run(m)
    assert "coda_span" not in out and "coda_span_weighted" not in out


def test_on_builds_exactly_j_heads():
    m = _model(coda_span_heads=5)
    assert len(m.coda_span) == 5
    for h in m.coda_span:
        # identity init: at step 0 a head reproduces the position's own next-token head
        assert torch.equal(h.proj.weight, torch.eye(m.cfg.d_model))


# ── on is purely additive and RNG-neutral ────────────────────────────────────

def test_on_shifts_no_weight_and_adds_only_its_own_term():
    """`nn.Linear.reset_parameters` DRAWS before the identity overwrites it.

    Without the save/restore in `MORPHTransformer.__init__` every weight built after the
    heads would move, and the arm would differ from its ruler by more than the mechanism.
    """
    off, on = _model(), _model(coda_span_heads=4)
    d_off, d_on = dict(off.named_parameters()), dict(on.named_parameters())
    assert sorted(set(d_off) - set(d_on)) == []
    assert all(n.startswith("coda_span.") for n in set(d_on) - set(d_off))
    for n, p in d_off.items():
        assert torch.equal(p, d_on[n]), f"building the coda heads moved {n}"
    o_off, o_on = _run(off), _run(on)
    assert torch.equal(o_on["loss"] - o_on["coda_span_weighted"], o_off["loss"])


def test_the_term_is_positive_and_carries_its_weight():
    m = _model(coda_span_heads=4, coda_span_weight=0.25)
    out = _run(m)
    assert float(out["coda_span"]) > 0.0
    assert abs(float(out["coda_span_weighted"]) - 0.25 * float(out["coda_span"])) < 1e-5
    assert float(out["coda_span_heads"]) == 4.0
    assert float(out["coda_span_n_tokens"]) > 0.0


# ── head j reads span k+1's token j ──────────────────────────────────────────

def _check_targets(m: MORPHTransformer, monkeypatch) -> None:
    lab, inp, layout = _labels(m, monkeypatch)
    B, S, J = lab.shape
    n_real = 0
    for b in range(B):
        spans = _span_tokens(layout, inp, b)
        for s in range(S):
            for j in range(J):
                got = int(lab[b, s, j])
                ok = (bool(layout.slot_valid[b, s]) and s + 1 < S
                      and bool(layout.slot_valid[b, s + 1]) and j < len(spans[s + 1]))
                want = spans[s + 1][j] if ok else -100
                assert got == want, (
                    f"head {j} at slot {s} (row {b}): label {got} != {want}")
                n_real += int(ok)
    assert n_real > 0, "fixture: no head was supervised at all"


def test_head_j_is_scored_against_the_next_spans_token_j(monkeypatch):
    _check_targets(_model(coda_span_heads=4), monkeypatch)


def test_the_masking_is_exact(monkeypatch):
    """A pad slot, a last slot, and every offset past a span's end are ignore_index."""
    # J = 12 on a fixture whose spans are ~8 tokens, so the "past the end" branch is
    # exercised rather than asserted vacuously.
    m = _model(coda_span_heads=12)
    lab, inp, layout = _labels(m, monkeypatch)
    B, S, J = lab.shape
    seen_pad = seen_short = 0
    for b in range(B):
        spans = _span_tokens(layout, inp, b)
        for s in range(S):
            if not bool(layout.slot_valid[b, s]):
                assert bool((lab[b, s] == -100).all()), f"pad slot {s} was supervised"
                seen_pad += 1
                continue
            if s + 1 >= S or not bool(layout.slot_valid[b, s + 1]):
                assert bool((lab[b, s] == -100).all()), \
                    f"slot {s} has no next span and was supervised"
                continue
            n = len(spans[s + 1])
            if n < J:
                assert bool((lab[b, s, n:] == -100).all()), \
                    f"slot {s}: offsets past the span's {n} tokens were supervised"
                seen_short += 1
    assert seen_pad > 0 and seen_short > 0, (
        "fixture: the batch must contain a pad slot AND a span shorter than J")


# ── the read position ────────────────────────────────────────────────────────

def _read_positions(m: MORPHTransformer) -> tuple[set, set]:
    """(positions the term actually reads, positions it should read) for one batch.

    "Actually reads" is measured, not asserted: the gradient of the term with respect to
    the coda readout is non-zero exactly where the term consumed it.
    """
    inp, _lab, layout, _ = _batch()
    L = layout.slot_mask.shape[1]
    xh = torch.randn(inp.shape[0], L, m.cfg.d_model, requires_grad=True)
    term = m._tul_coda_span_loss(xh, inp, layout)
    g, = torch.autograd.grad(term, xh)
    got = {(int(b), int(p)) for b, p in (g.abs().sum(-1) > 0).nonzero().tolist()}
    want = set()
    for b in range(inp.shape[0]):
        for s in range(int(layout.slot_index.shape[1])):
            if not bool(layout.slot_valid[b, s]):
                continue
            if s + 1 >= layout.slot_index.shape[1] or not bool(layout.slot_valid[b, s + 1]):
                continue                          # nothing supervised -> no gradient
            want.add((b, int(layout.slot_index[b, s]) + layout.prefix_k - 1))
    return got, want


def test_the_heads_read_the_slots_last_prefix_cell():
    got, want = _read_positions(_model(coda_span_heads=4))
    assert got == want, (
        f"the heads read {sorted(got - want)[:5]} they should not and missed "
        f"{sorted(want - got)[:5]}")


def test_the_token_source_reads_the_boundary_token_instead():
    """The CONTROL, and the thing about it that has to be said out loud.

    `coda_span_source: token` reads the position `emit_source="token"` generates from. That
    token sits BEFORE its own slot's prefix cells and the coda is causal, so its state has
    never seen its own slot's z. The heads then reach the loop only through EARLIER slots'
    writes. This test pins WHERE it reads; the consequence is documented, not measured.
    """
    from morph.model.tul import boundary_token_index
    m = _model(coda_span_heads=4, coda_span_source="token")
    inp, _lab, layout, _ = _batch()
    L = layout.slot_mask.shape[1]
    xh = torch.randn(inp.shape[0], L, m.cfg.d_model, requires_grad=True)
    term = m._tul_coda_span_loss(xh, inp, layout)
    g, = torch.autograd.grad(term, xh)
    got = {(int(b), int(p)) for b, p in (g.abs().sum(-1) > 0).nonzero().tolist()}
    S = int(layout.slot_index.shape[1])
    bt = boundary_token_index(layout.bag_id, ~layout.slot_mask, S)[:, :S]
    cells = {(b, int(layout.slot_index[b, s]) + layout.prefix_k - 1)
             for b in range(inp.shape[0]) for s in range(S)}
    assert got, "the token source read nothing"
    assert not (got & cells), "the token source read a prefix CELL"
    for b, p in got:
        assert p in [int(v) for v in bt[b]], f"({b}, {p}) is not a boundary token position"


# ── the gradient reaches the coda and the loop's write ───────────────────────

def test_the_gradient_reaches_the_coda_and_the_loops_write(monkeypatch):
    """The arm's whole mechanism, on the REAL forward and isolated to this term.

    `out["coda_span"]` is exposed detached (the `spandec_weighted` contract), so the live
    tensor is captured on the way past instead.
    """
    real = MORPHTransformer._tul_coda_span_loss

    def spy(self, xh, ids, layout, stats=None):
        t = real(self, xh, ids, layout, stats)
        self._cs_term = t
        return t

    monkeypatch.setattr(MORPHTransformer, "_tul_coda_span_loss", spy)
    m = _model(coda_span_heads=4)
    _run(m)
    names = ["tul.W_prefix"] + [n for n, _ in m.named_parameters() if n.startswith("coda.")]
    params = [dict(m.named_parameters())[n] for n in names]
    g = torch.autograd.grad(m._cs_term, params, allow_unused=True, retain_graph=True)
    assert g[0] is not None and float(g[0].abs().sum()) > 0.0, \
        "the heads' term reaches no loop write (tul.W_prefix)"
    assert any(x is not None and float(x.abs().sum()) > 0 for x in g[1:]), \
        "the heads' term reaches no coda parameter"
    heads = [p for n, p in m.named_parameters() if n.startswith("coda_span.")]
    gh = torch.autograd.grad(m._cs_term, heads, allow_unused=True)
    assert any(x is not None and float(x.abs().sum()) > 0 for x in gh)


# ── the refusals ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw,match", [
    (dict(coda_span_heads=-1), "must be >= 0"),
    (dict(coda_span_heads=4, coda_span_weight=0.0), "coda_span_weight"),
    (dict(coda_span_heads=64), "bound_span_cap"),
    (dict(coda_span_heads=4, tokens_through_core=True, spandec=False, spandec_layers=2,
          spandec_max_tokens=0, mux_beta=0.0), "SLOT-LOOP lever"),
    (dict(coda_span_heads=4, coda_sees_slots=False), "FULL-AXIS coda"),
    (dict(coda_span_heads=4, detach_z=True, spandec=False, spandec_layers=2,
          spandec_max_tokens=0), "detach_z"),
    (dict(coda_span_source="nope"), "coda_span_source"),
    (dict(coda_span_weight=2.0), "silently ignored"),
])
def test_coda_span_refusals(kw, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _tul(**kw)
