"""``tul.cell_count`` — a variable cell count M on the write-all fan (2026-10-12).

Design: .agents/notes/proposed/architecture/2026-10-12-fan-variable-cell-count-design.md.
Two changes to its build brief (Wolfe, 2026-10-12): the draw law is general
(``categorical`` beside ``poisson``) and ``tul.cell_count_subset: random`` trains every role.

WHAT THIS FILE HAS TO PROVE
  1. THE RE-LAYOUT KEEPS THE ROW. Re-laying a packed row out to M cells keeps every token, label,
     boundary and span; each slot's cells sit at ``first' .. first' + M - 1`` and the emit label
     on the LAST one; a round trip to M and back is the original row, byte for byte; the
     adversarial rows (all spans at the cap, all spans at the minimum, the slot budget) fit
     ``L(M)`` and a row that does not fit raises instead of losing a token.
  2. THE LAW. Counter-based (batch ``i`` draws from ``[seed, 0xCE11, i]``, independent of any
     global stream, replayed after a resume), frequencies inside 3 sigma, roles ascending and
     unique; the Poisson masses are the ones the note prints; a law that is not a distribution
     raises; a knob set under the wrong law raises.
  3. THE MODEL.
       off          the fixed fan is the tree before the key (pinned to master's numbers).
       fixed 4      key on with every batch at M = fan_k: loss, logits and every gradient
                    ``torch.equal`` to key off.
       prefix       M = 2 on the 4-role model is a ``fan_k = 2`` model whose per-cell tensors
                    (``Q``, ``P_cell``, ``W_prefix``, and every other tensor with a cell axis)
                    are rows 0, 1: loss ``torch.equal``.
       random       M = 2 with roles {1, 3} is the ``fan_k = 2`` model with rows [1, 3].
       M = 1        runs the register path and has no diversity term; finite loss and grads.
       extend       M = 8 runs at eval under ``cyclic`` and ``fresh``; training refuses M > fan_k.

Sabotage checks, run and reverted (session report, not committed):
  (a) ``slot_first`` shifted by one in ``relayout_cells`` -> test_relayout_roundtrip and
      test_relayout_content fail.
  (b) ``cell_l_total`` with ``S_lo + 1`` -> the adversarial 35-slot row overflows and raises.
  (c) ``select_cell_rows`` indexes ``table[1:m + 1]`` -> test_cellcount_fixed4 and
      test_cellcount_prefix_model fail.
  (d) ``select_cell_rows`` ignores ``roles`` (indexes by position) -> test_cellcount_random_
      subset_model fails.
  (e) the draw seeded from the global numpy stream -> test_cellcount_draw fails.
CPU, fp32, the ``tests/test_tul_fan.py`` fixtures.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import pytest
import torch

from morph.model.tul import TULConfig, select_cell_rows
from morph.model.tul_layout import (
    CellCountLaw,
    TulLayoutSpec,
    cell_l_total,
    cell_slots_lower_bound,
    pack_tul_row,
    poisson_cell_probs,
    relayout_batch,
    relayout_cells,
)
from test_tul_fan import DOT, V, _batch, _model, _rule, _spec

SPAN_CAP = 32
SPEC = TulLayoutSpec(seq_len=1024, prefix_k=4, max_slots=64, slot_id=4)     # L_ref 1280


# ── 1. the re-layout ──────────────────────────────────────────────────────────────

def _stream(n: int, seed: int, p_boundary: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=n)
    ids[ids == 4] = 5
    ids[rng.random(n) < p_boundary] = DOT
    return ids.astype(np.int64)


def _rows(n_rows: int, seed0: int = 0) -> list[dict]:
    out = []
    for i in range(n_rows):
        ids = _stream(1400, seed0 + i, (0.0, 0.01, 0.05, 0.12, 0.2, 0.3)[i % 6])
        arrays, _n, _st = pack_tul_row(ids, _rule(), SPEC)
        out.append(arrays)
    return out


def _adversarial_rows() -> dict[str, dict]:
    """Rows at the packer's extremes: spans all at the cap (fewest slots), all at min_span
    (the slot budget), and the cap with no boundary token anywhere."""
    rows = {}
    cap = np.full(1400, 5, dtype=np.int64)                 # no boundary token: cap-forced cuts
    rows["cap"] = pack_tul_row(cap, _rule(), SPEC)[0]
    mn = np.full(1400, 5, dtype=np.int64)
    mn[3::4] = DOT                                         # a boundary every 4 tokens
    rows["min_span"] = pack_tul_row(mn, _rule(), SPEC)[0]
    return rows


def test_relayout_roundtrip():
    """M and back is the original row, every field, byte for byte."""
    for row in _rows(24) + list(_adversarial_rows().values()):
        for m in (1, 2, 3, 5, 8, 16):
            there = relayout_cells(row, m, SPEC, SPAN_CAP)
            back = relayout_cells(there, 4, SPEC, SPAN_CAP, m_from=m)
            for k, v in row.items():
                assert np.array_equal(v, back[k]), (m, k)


def test_relayout_identity_at_the_reference_width():
    for row in _rows(6):
        assert relayout_cells(row, 4, SPEC, SPAN_CAP) is row


def test_relayout_content():
    for row in _rows(18) + list(_adversarial_rows().values()):
        K = 4
        tok = np.flatnonzero(~row["slot_mask"])
        n = int(row["slot_valid"].sum())
        first = row["slot_index"][:n]
        for m in (1, 2, 3, 4, 5, 8, 16, 32):
            new = relayout_cells(row, m, SPEC, SPAN_CAP)
            assert new["input_ids"].shape == (cell_l_total(SPEC, m, SPAN_CAP),)
            ntok = np.flatnonzero(~new["slot_mask"])
            assert ntok.size == tok.size
            # tokens, labels, span membership: same, in order
            assert np.array_equal(new["input_ids"][ntok], row["input_ids"][tok])
            assert np.array_equal(new["labels"][ntok], row["labels"][tok])
            assert np.array_equal(new["bag_id"][ntok], row["bag_id"][tok])
            nfirst = new["slot_index"][:n]
            assert np.array_equal(nfirst, first + (m - K) * np.arange(n))
            assert np.array_equal(new["slot_valid"], row["slot_valid"])
            for s in range(n):
                cells = np.arange(nfirst[s], nfirst[s] + m)
                assert new["slot_mask"][cells].all()
                assert (new["input_ids"][cells] == SPEC.slot_id).all()
                assert (new["bag_id"][cells] == s).all()
                assert (new["labels"][cells[:-1]] == -100).all()
                # the emit label sits on the LAST cell and is the old row's emit label
                assert new["labels"][cells[-1]] == row["labels"][first[s] + K - 1]
            # everything not a token and not a cell is a tail pad
            used = np.zeros_like(new["slot_mask"])
            used[ntok] = True
            for s in range(n):
                used[nfirst[s]:nfirst[s] + m] = True
            assert (new["labels"][~used] == -100).all() and new["slot_mask"][~used].all()
            assert (new["bag_id"][~used] == SPEC.max_slots).all()


def test_relayout_budget_is_the_proven_one():
    # (1280 - 4 - 32) / 36 = 34.6 -> 35 slots, at these shapes (the note's S_lo)
    assert cell_slots_lower_bound(SPEC, SPAN_CAP) == 35
    assert [cell_l_total(SPEC, m, SPAN_CAP) for m in (1, 2, 3, 4, 8)] == [
        1175, 1210, 1245, 1280, 1536]
    # the cap row is the tight one: it needs exactly S_lo slots and fills L(1) exactly
    cap = _adversarial_rows()["cap"]
    assert int(cap["slot_valid"].sum()) == 35
    new = relayout_cells(cap, 1, SPEC, SPAN_CAP)
    assert int(np.flatnonzero(~new["slot_mask"] | (new["bag_id"] < SPEC.max_slots)).max()) \
        == cell_l_total(SPEC, 1, SPAN_CAP) - 1


def test_relayout_refuses_a_row_that_does_not_fit():
    """Never drops a token: a row with fewer slots than the proof allows raises at m < K."""
    row = {k: v.copy() for k, v in _adversarial_rows()["cap"].items()}
    # a forged row: every position a token and no slot, so nothing shifts down at m = 1
    row["slot_mask"][:] = False
    row["slot_valid"][:] = False
    row["slot_index"][:] = 0
    with pytest.raises(ValueError, match="never truncated"):
        relayout_cells(row, 1, SPEC, SPAN_CAP)
    # the same row grows without trouble: more cells per slot never loses a position
    assert relayout_cells(row, 8, SPEC, SPAN_CAP)["input_ids"].shape == (
        cell_l_total(SPEC, 8, SPAN_CAP),)


def test_relayout_refuses_ditto_rows():
    row = dict(_rows(1)[0])
    row["ditto_prev"] = np.full_like(row["labels"], -1)
    with pytest.raises(ValueError, match="DITTO"):
        relayout_cells(row, 2, SPEC, SPAN_CAP)


def test_relayout_batch_matches_rows_and_carries_roles():
    from morph.model.tul_layout import SlotLayout
    rows = _rows(3)
    x = torch.from_numpy(np.stack([r["input_ids"] for r in rows]))
    y = torch.from_numpy(np.stack([r["labels"] for r in rows]))
    lay = SlotLayout.from_rows(rows, 4)
    bx, by, bl = relayout_batch(x, y, lay, 2, SPEC, SPAN_CAP, np.array([1, 3]))
    assert bl.prefix_k == 2 and bx.shape == (3, cell_l_total(SPEC, 2, SPAN_CAP))
    assert bl.cell_roles.tolist() == [[1, 3]] * 3
    for r, row in enumerate(rows):
        ref = relayout_cells(row, 2, SPEC, SPAN_CAP)
        assert np.array_equal(bx[r].numpy(), ref["input_ids"])
        assert np.array_equal(by[r].numpy(), ref["labels"])
        assert np.array_equal(bl.slot_index[r].numpy(), ref["slot_index"])
    # at the reference width only the roles are attached
    _x, _y, same = relayout_batch(x, y, lay, 4, SPEC, SPAN_CAP, np.arange(4))
    assert same.prefix_k == 4 and torch.equal(same.slot_index, lay.slot_index)
    assert same.cell_roles.shape == (3, 4)


# ── 2. the law ────────────────────────────────────────────────────────────────────

def test_poisson_masses_are_the_notes():
    p = poisson_cell_probs(3.5, 1, 4)
    assert sum(p) == pytest.approx(1.0, abs=1e-12)
    assert [round(x, 3) for x in p] == [0.136, 0.185, 0.216, 0.463]
    assert poisson_cell_probs(2.0, 2, 2) == (0.0, 1.0)
    q = poisson_cell_probs(2.0, 1, 3)
    assert len(q) == 3 and sum(q) == pytest.approx(1.0, abs=1e-12)


def test_cellcount_draw():
    law = CellCountLaw((0.1, 0.2, 0.3, 0.4), subset="random", seed=5)
    again = CellCountLaw((0.1, 0.2, 0.3, 0.4), subset="random", seed=5)
    np.random.seed(123)
    first = [law.draw(i) for i in range(2000)]
    np.random.seed(999)                                  # a global stream the draw must not read
    # a "resume": a new object, batches 1000.. replayed, equals the original's 1000..
    for i in range(1000, 2000):
        m, roles = again.draw(i)
        assert m == first[i][0] and np.array_equal(roles, first[i][1])
    ms = np.array([m for m, _ in first], dtype=np.int64)
    big = np.array([law.draw(i)[0] for i in range(100_000)])
    for m, p in enumerate(law.probs, start=1):
        f = float((big == m).mean())
        assert abs(f - p) < 3 * math.sqrt(p * (1 - p) / big.size), (m, f, p)
    assert ms.min() >= 1 and ms.max() <= 4
    for m, roles in first[:300]:
        assert roles.shape == (m,) and np.all(np.diff(roles) > 0) and roles.max() < 4
    # random subsets are uniform over roles: every role appears about m/4 of the time
    cnt = np.zeros(4)
    for m, roles in first:
        cnt[roles] += 1.0 / max(m, 1)
    assert cnt.min() > 0.8 * cnt.mean()
    # a different seed gives a different sequence
    other = [CellCountLaw((0.1, 0.2, 0.3, 0.4), seed=6).draw(i)[0] for i in range(200)]
    assert other != [m for m, _ in first[:200]]


def test_cellcount_prefix_subset_is_the_prefix():
    law = CellCountLaw((0.25,) * 4, subset="prefix", seed=0)
    for i in range(50):
        m, roles = law.draw(i)
        assert roles.tolist() == list(range(m))


def test_cellcount_law_refusals():
    with pytest.raises(ValueError, match="sum to 1"):
        CellCountLaw((0.5, 0.2, 0.2, 0.2))
    with pytest.raises(ValueError, match="subset"):
        CellCountLaw((1.0,), subset="sorted")
    base = dict(fan_k=4, slot_cells=4, prefix_k=4, fan_mix="all", fan_all_wta_lambda=0.0)
    with pytest.raises(ValueError, match="silently ignored"):
        TULConfig(**base, cell_count_probs=(0.25,) * 4)                   # law fixed
    with pytest.raises(ValueError, match="other law"):
        TULConfig(**base, cell_count="poisson", cell_count_probs=(0.25,) * 4)
    with pytest.raises(ValueError, match="other law"):
        TULConfig(**base, cell_count="categorical", cell_count_probs=(0.25,) * 4,
                  cell_count_mean=2.0)
    with pytest.raises(ValueError, match="sum to 1"):
        TULConfig(**base, cell_count="categorical", cell_count_probs=(0.5, 0.2, 0.2, 0.2))
    with pytest.raises(ValueError, match="4 entries"):
        TULConfig(**base, cell_count="categorical", cell_count_probs=(0.5, 0.5))
    with pytest.raises(ValueError, match="needs tul.cell_count_probs"):
        TULConfig(**base, cell_count="categorical")
    with pytest.raises(ValueError, match="fan_k"):
        TULConfig(prefix_k=2, cell_count="poisson")
    with pytest.raises(ValueError, match="max"):
        TULConfig(**base, cell_count="poisson", cell_count_min=3, cell_count_max=2)
    with pytest.raises(ValueError, match="cell_count_eval"):
        TULConfig(**base, cell_count="poisson", cell_count_eval=5)
    with pytest.raises(NotImplementedError, match="fan_mix"):
        TULConfig(fan_k=4, slot_cells=4, prefix_k=4, fan_mix="mean", cell_count="poisson")
    with pytest.raises(NotImplementedError, match="ditto_rows"):
        TULConfig(**base, cell_count="poisson", ditto_rows=1)
    with pytest.raises(NotImplementedError, match="fan_seed_noise"):
        TULConfig(**base, cell_count="poisson", fan_seed_noise=0.1)
    # the accepted forms
    TULConfig(**base, cell_count="poisson", cell_count_mean=3.5, cell_count_max=4)
    TULConfig(**base, cell_count="categorical", cell_count_probs=(0.0, 0.0, 0.25, 0.75),
              cell_count_subset="random")


def test_select_cell_rows():
    t = torch.arange(12.0).view(4, 3)
    assert torch.equal(select_cell_rows(t, 2, None), t[:2])
    assert torch.equal(select_cell_rows(t, 4, None), t)
    roles = torch.tensor([[1, 3], [1, 3]])
    assert torch.equal(select_cell_rows(t, 2, roles), t[[1, 3]])
    with pytest.raises(ValueError, match="cyclic"):
        select_cell_rows(t, 6, None)
    assert torch.equal(select_cell_rows(t, 6, None, "cyclic"), t[[0, 1, 2, 3, 0, 1]])
    fresh = select_cell_rows(t, 6, None, "fresh", lambda n: torch.full((n, 3), -1.0))
    assert torch.equal(fresh[:4], t) and (fresh[4:] == -1).all()


# ── 3. the model ──────────────────────────────────────────────────────────────────

# The recipe's fan at the tiny width: write-all, epivol diversity, the set-reading span decoder,
# pointer heads, the cell pass norm (nowta_pointer_fast_spdcells).
_FAN = dict(fan_k=4, slot_cells=4, prefix_k=4, fan_mix="all", fan_all_wta_lambda=0.0,
            fan_repel_lambda=0.1, fan_repel_mode="epivol", spandec=True,
            spandec_reads_cells=True, spandec_layers=1, pointer_heads=2,
            slot_cell_pass_norm="rms")
_VAR = dict(cell_count="categorical", cell_count_probs=(0.25, 0.25, 0.25, 0.25))


def _perturb(m, scale: float = 0.05, w_prefix: bool = True):
    """Non-trivial values in the zero-init cell tensors and (``w_prefix``) a per-role
    perturbation of ``W_prefix`` (identity at init), so a mis-indexed role is visible."""
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("W_o.weight") or n.endswith("P_cell"):
                g = torch.Generator().manual_seed(zlib.crc32(n.encode()) % 1000 + 1)
                p.copy_(torch.randn(p.shape, generator=g) * scale)
            if w_prefix and n.endswith("W_prefix"):
                g = torch.Generator().manual_seed(zlib.crc32(n.encode()) % 1000 + 2)
                p.add_(torch.randn(p.shape, generator=g) * 0.02)
    return m


def _var_batch(m_cells: int, roles=None):
    ids, inp, lab, layout = _batch(4)
    return relayout_batch(inp, lab, layout, m_cells, _spec(4), SPAN_CAP, roles)


def _loss_grads(m, inp, lab, layout, seed: int = 7):
    m.train()
    m.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    res = m(inp, labels=lab, slot_layout=layout)
    res["loss"].backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    return res, grads


def test_cellcount_off_is_pinned_to_master():
    """Key off: the tiny write-all fan, numbers measured on master 41be5b52
    (/home/wolfe/morph-scratch/abq/cellcount-build/pin.py)."""
    _ids, inp, lab, layout = _batch(4)
    m = _perturb(_model(**_FAN), w_prefix=False)
    res, grads = _loss_grads(m, inp, lab, layout)
    assert float(res["loss"].detach()) == 9.578686714172363
    assert sum(float(g.double().abs().sum()) for g in grads.values()) == 2661.064937990642
    m.eval()
    with torch.no_grad():
        lg = m(inp, labels=None, slot_layout=layout)["logits"]
    assert float(torch.nan_to_num(lg.double(), nan=0.0, posinf=0.0, neginf=0.0).sum()) \
        == -64649.21856123209
    assert len(m.state_dict()) == 240


def test_cellcount_fixed4():
    """Key on, every batch at M = fan_k: forward and backward equal key off, bit for bit."""
    _ids, inp, lab, layout = _batch(4)
    off = _perturb(_model(**_FAN))
    on = _perturb(_model(**_FAN, **_VAR))
    assert set(off.state_dict()) == set(on.state_dict())
    on.load_state_dict(off.state_dict())
    r0, g0 = _loss_grads(off, inp, lab, layout)
    r1, g1 = _loss_grads(on, inp, lab, layout)
    assert torch.equal(r0["loss"], r1["loss"])
    assert g0.keys() == g1.keys()
    for n in g0:
        assert torch.equal(g0[n], g1[n]), n
    off.eval(), on.eval()
    with torch.no_grad():
        assert torch.equal(off(inp, labels=None, slot_layout=layout)["logits"],
                           on(inp, labels=None, slot_layout=layout)["logits"])


def _load_rows(big, small, rows):
    """Load ``big``'s tensors into ``small`` (a fixed fan with ``len(rows)`` cells): equal shapes
    copy, a tensor with a cell axis takes ``rows``. Returns the names that had a cell axis, so a
    caller sees exactly which tensors are indexed by cell."""
    sd_big, sd_small = big.state_dict(), small.state_dict()
    assert sd_big.keys() == sd_small.keys()
    cellful, new = [], {}
    for n, t in sd_big.items():
        if t.shape == sd_small[n].shape:
            new[n] = t.clone()
        else:
            assert t.shape[0] == 4 and sd_small[n].shape[0] == len(rows) \
                and t.shape[1:] == sd_small[n].shape[1:], (n, t.shape, sd_small[n].shape)
            new[n] = t[list(rows)].clone()
            cellful.append(n)
    small.load_state_dict(new)
    return cellful


def _two_cell_pair(roles):
    big = _perturb(_model(**_FAN, **_VAR))
    small = _perturb(_model(**{**_FAN, "fan_k": 2, "slot_cells": 2, "prefix_k": 2}))
    cellful = _load_rows(big, small, roles)
    return big, small, cellful


@pytest.mark.parametrize("roles", [(0, 1)])
def test_cellcount_prefix_model(roles):
    big, small, cellful = _two_cell_pair(roles)
    # exactly the role tables carry a cell axis: nothing else is indexed by cell position
    assert sorted(n.split(".")[-1] for n in cellful) == ["P_cell", "Q", "W_prefix"], cellful
    inp, lab, layout = _var_batch(2)                      # no roles: cells are roles 0, 1
    assert layout.prefix_k == 2 and layout.cell_roles is None
    r_big, g_big = _loss_grads(big, inp, lab, layout)
    r_small, g_small = _loss_grads(small, inp, lab, layout)
    assert torch.equal(r_big["loss"], r_small["loss"])
    # the gradient of a role table lands on the rows that were used and nowhere else
    for n in cellful:
        gb, gs = g_big[n], g_small[n]
        assert torch.equal(gb[list(roles)], gs), n
        rest = [r for r in range(4) if r not in roles]
        assert (gb[rest] == 0).all(), n
    big.eval(), small.eval()
    with torch.no_grad():
        assert torch.equal(big(inp, labels=None, slot_layout=layout)["logits"],
                           small(inp, labels=None, slot_layout=layout)["logits"])


def test_cellcount_random_subset_model():
    """M = 2 with roles {1, 3}: the fan_k = 2 model whose role rows are [1, 3]."""
    roles = (1, 3)
    big, small, cellful = _two_cell_pair(roles)
    inp, lab, layout = _var_batch(2, np.array(roles))
    assert layout.cell_roles.tolist() == [list(roles)] * layout.slot_mask.shape[0]
    r_big, g_big = _loss_grads(big, inp, lab, layout)
    r_small, g_small = _loss_grads(small, inp, lab, layout)
    assert torch.equal(r_big["loss"], r_small["loss"])
    for n in cellful:
        assert torch.equal(g_big[n][list(roles)], g_small[n]), n
        assert (g_big[n][[0, 2]] == 0).all(), n
    # and the roles matter: the prefix roles {0, 1} give a different loss on the same tokens
    inp2, lab2, layout2 = _var_batch(2, np.array([0, 1]))
    r_other, _ = _loss_grads(big, inp2, lab2, layout2)
    assert not torch.equal(r_big["loss"], r_other["loss"])
    big.eval(), small.eval()
    with torch.no_grad():
        assert torch.equal(big(inp, labels=None, slot_layout=layout)["logits"],
                           small(inp, labels=None, slot_layout=layout)["logits"])


@pytest.mark.parametrize("m_cells", [1, 2, 3])
def test_cellcount_below_four_runs_and_trains_every_role_it_uses(m_cells):
    big = _perturb(_model(**_FAN, **_VAR))
    inp, lab, layout = _var_batch(m_cells)
    res, grads = _loss_grads(big, inp, lab, layout)
    assert torch.isfinite(res["loss"])
    for n, g in grads.items():
        assert torch.isfinite(g).all(), n
    q = grads["tul_register.Q"]
    assert (q[:m_cells].abs().sum(dim=1) > 0).all() and (q[m_cells:] == 0).all()
    wp = grads["tul.W_prefix"]
    assert (wp[:m_cells].flatten(1).abs().sum(dim=1) > 0).all() and (wp[m_cells:] == 0).all()


def test_cellcount_m1_has_no_diversity_term():
    big = _perturb(_model(**_FAN, **_VAR))
    inp, lab, layout = _var_batch(1)
    res, _g = _loss_grads(big, inp, lab, layout)
    assert float(res.get("fan_repel_weighted", 0.0)) == 0.0
    inp4, lab4, layout4 = _var_batch(4)
    res4, _g = _loss_grads(big, inp4, lab4, layout4)
    assert float(res4["fan_repel_weighted"]) != 0.0


def test_cell_extend():
    big = _perturb(_model(**_FAN, **_VAR))
    _ids, inp, lab, layout = _batch(4)
    ext = {}
    for m_cells in (6, 8):
        ext[m_cells] = relayout_batch(inp, lab, layout, m_cells, _spec(4), SPAN_CAP)
    big.eval()
    with torch.no_grad():
        for m_cells, (x, y, lay) in ext.items():
            with pytest.raises(ValueError, match="cell_extend"):
                big(x, labels=y, slot_layout=lay)                 # default: refuse
        outs = {}
        for rule in ("cyclic", "fresh"):
            big.cell_extend = rule
            for m_cells, (x, y, lay) in ext.items():
                o = big(x, labels=y, slot_layout=lay)
                assert torch.isfinite(o["loss"]), (rule, m_cells)
                outs[(rule, m_cells)] = float(o["loss"])
        # the two rules are different models for the cells past the table
        assert outs[("cyclic", 8)] != outs[("fresh", 8)]
        big.cell_extend = "cyclic"
        # cyclic at M = 8: cell i is role i mod 4 -> the same logits as a manual role table
        x, y, lay = ext[8]
        a = big(x, labels=y, slot_layout=lay)["loss"]
        big.cell_extend = "refuse"
    big.train()
    with pytest.raises(ValueError, match="cell_extend"):
        big.cell_extend = "cyclic"                                # eval-mode setting only
    x, y, lay = ext[6]
    with pytest.raises(ValueError, match="training never extends"):
        big(x, labels=y, slot_layout=lay)
    assert float(a) == outs[("cyclic", 8)]
    big.eval()
    with pytest.raises(ValueError, match="cell_extend must be"):
        big.cell_extend = "wrap"


@pytest.mark.parametrize("m_cells", [1, 2, 3, 4])
def test_cellcount_eval_forward_reports_per_cell_streams(m_cells):
    """The eval forward (the oracle replays one stream at a time) runs at every M and reports
    `fan_stream_ce_k{i}` for the cells that exist, only."""
    big = _perturb(_model(**_FAN, **_VAR)).eval()
    inp, lab, layout = _var_batch(m_cells)
    with torch.no_grad():
        out = big(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    ks = sorted(k for k in out if k.startswith("fan_stream_ce_k"))
    assert ks == [f"fan_stream_ce_k{i}" for i in range(m_cells)], ks
    # the diversity readings exist only where a pair of cells does
    assert any(k.startswith("fan_vol_t") for k in out) == (m_cells > 1)


# ── 4. the config wiring ──────────────────────────────────────────────────────────

def test_cellcount_configs_compose_into_the_runtime(monkeypatch):
    from test_graph_step import _compose
    cfg, rt = _compose(monkeypatch, "nowta_pointer_fast_spdcells_speed_bf16c")
    assert rt.model_cfg.cell_count == "fixed" and rt.data_cfg.cell_law is None
    assert rt.cell_shapes == (None,) and rt.manifest["cell_count_law"] is None
    # Poisson 3.5 clamped to [1, 4], prefix roles
    cfg, rt = _compose(monkeypatch, "nowta_pointer_fast_spdcells_speed_bf16c_cellp3")
    law = rt.data_cfg.cell_law
    assert [round(p, 3) for p in law.probs] == [0.136, 0.185, 0.216, 0.463]
    assert law.subset == "prefix" and rt.cell_shapes == (1, 2, 3, 4)
    assert rt.manifest["cell_count"] == "poisson" and rt.manifest["cell_count_law"] == list(law.probs)
    assert rt.model_cfg.fan_k == 4 and rt.data_cfg.prefix_k == 4
    # the val loader draws nothing and lays out at the reference width
    assert rt.val_data_cfg.cell_law is None and rt.val_data_cfg.eval_cells == 0
    # categorical [0, 0, .25, .75], random roles: only M in {3, 4} get a graph
    cfg, rt = _compose(monkeypatch, "nowta_pointer_fast_spdcells_speed_bf16c_cellg375")
    law = rt.data_cfg.cell_law
    assert law.probs == (0.0, 0.0, 0.25, 0.75) and law.subset == "random"
    assert rt.cell_shapes == (3, 4)
    assert rt.manifest["cell_count_subset"] == "random"
    # cell_count_eval sets the val loader's fixed M
    cfg, rt = _compose(monkeypatch, "nowta_pointer_fast_spdcells_speed_bf16c_cellp3",
                       "+tul.cell_count_eval=2")
    assert rt.val_data_cfg.eval_cells == 2 and rt.val_data_cfg.cell_law is None
    cfg, rt = _compose(monkeypatch, "nowta_pointer_fast_spdcells_speed_bf16c_cellp3",
                       "+tul.cell_count_eval=4")
    assert rt.val_data_cfg.eval_cells == 0           # the reference width needs no re-layout


@pytest.mark.parametrize("m_cells", [1, 3])
def test_cellcount_eval_instruments_run_below_fan_k(m_cells):
    """The trainer's eval block at a cell_count_eval below fan_k: the plan ablations
    (zero / shuffle / all_slots / wrong_seed) and the slot-state probe."""
    big = _perturb(_model(**_FAN, **_VAR)).eval()
    inp, lab, layout = _var_batch(m_cells)
    with torch.no_grad():
        base = float(big(inp, labels=lab, slot_layout=layout)["loss"])
        for mode in ("zero", "shuffle", "all_slots", "wrong_seed"):
            out = big.tul_forward_ablated(inp, lab, layout, plan_mode=mode)
            assert torch.isfinite(out["loss"]), mode
        # a plan that is wiped costs the coda something on this batch
        assert float(big.tul_forward_ablated(inp, lab, layout, plan_mode="zero")["loss"]) != base
        probe = big.tul_slot_state_probe(inp, layout)
    if m_cells > 1:
        assert probe["slot_cells"] == float(m_cells)
    else:
        assert "slot_cells" not in probe
