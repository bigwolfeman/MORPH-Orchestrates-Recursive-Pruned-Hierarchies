"""``tul.ditto_rows`` — DITTO pseudo-repetition rows against self-reinforcing loops (2026-10-07).

morph/model/tul_ditto.py (the loss), tul_layout.py (`pack_ditto_row`, `pack_tul_batch`,
`SlotLayout.ditto_prev`), transformer.py (`_tul_group_losses`), tul.py (`_check_ditto`),
tul_setup.py (keys, val config, manifest), data.py (the train loader),
morph/configs/lxtul_pointer_ditto.yaml.

What each test pins:
  1. The loss is eq. 1 of Xu et al. 2022, zero at p_n = lambda * p_{n-1}, and its gradient
     never reaches the previous copy's probability (the stop-gradient).
  2. A DITTO row is the real prefix, then one span repeated; every marked position points one
     copy back (same input, same label, exactly one span of tokens earlier) and every target
     in copy 1 or later is marked.
  3. The batch packer: ditto_rows 0 is the old packer (no field, no draw); with DITTO rows it
     consumes exactly the tokens the ordinary batch consumes and leaves the other rows alone.
  4. On the LXTUL register model, with and without the pointer: the term in the training
     loss equals `ditto_loss` on the eval path's per-token log-probs, DITTO positions carry
     no CE, and the config alone (no DITTO rows in the layout) is the tree.
  5. Refusals, known keys, the Hydra path (train config on, val config off), compose diff.

Sabotage checks, run and reverted (session report, not committed):
  (a) the stop-gradient removed (`p` in place of `p.detach()` for p_prev) -> test 1 fails.
  (b) DITTO positions left in the CE (`_lab_ok` not masked) -> test 4 fails.
  (c) the copy offset off by one in `pack_ditto_row` (m from start + S) -> test 2 fails.
CPU, fp32, the LXTUL fixtures (M = 4 cells, strict geometry).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.tul import TULConfig
from morph.model.tul_ditto import ditto_loss
from morph.model.tul_layout import pack_ditto_row, pack_tul_batch
from test_loop_attn_center import _CONFIG_DIR, MASTER_PIN, _compose_leaves, _lx, _pin_run
from test_tul_fan import _ids, _rule, _spec
from test_tul_lx_credit import M

M = int(M)
LAM = 0.5


# ── 1. the loss ──────────────────────────────────────────────────────────────────


def test_loss_is_eq1_and_stops_the_gradient_at_the_previous_copy():
    g = torch.Generator().manual_seed(0)
    lp = (-torch.rand(2, 12, generator=g) * 3).requires_grad_(True)
    prev = torch.full((2, 12), -1, dtype=torch.long)
    prev[0, 6:9], prev[1, 9:12] = torch.tensor([2, 3, 4]), torch.tensor([5, 6, 7])
    term, ratio, pos = ditto_loss(lp, prev, LAM)
    p = lp.detach().exp()
    ref = [-np.log(1 - abs(float(p[b, i]) - LAM * float(p[b, int(prev[b, i])])))
           for b in range(2) for i in range(12) if prev[b, i] >= 0]
    assert abs(float(term.detach()) - np.mean(ref)) < 1e-6
    assert int(pos.sum()) == 6
    assert abs(float(ratio) - float(p[pos].sum() / p[0, 2:5].sum().add(p[1, 5:8].sum()))) < 1e-6
    term.backward()
    pointed = torch.zeros_like(pos)
    pointed[0, 2:5], pointed[1, 5:8] = True, True
    assert torch.all(lp.grad[pointed & ~pos] == 0), lp.grad          # sg(p_{n-1})
    assert torch.all(lp.grad[pos] != 0)
    assert torch.all(lp.grad[~pos & ~pointed] == 0)
    # zero exactly at p_n = lambda * p_{n-1}
    lp2 = lp.detach().clone()
    lp2[0, 6:9] = lp2[0, 2:5] + np.log(LAM)
    lp2[1, 9:12] = lp2[1, 5:8] + np.log(LAM)
    assert float(ditto_loss(lp2, prev, LAM)[0]) < 1e-6


# ── 2. the row ───────────────────────────────────────────────────────────────────


def _row(seed=0, n=200):
    ids = _ids(B=1, n=n, seed=seed)[0]
    return ids, pack_ditto_row(ids, _rule(), _spec(M), np.random.default_rng(seed))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_row_is_prefix_then_one_span_repeated_and_every_copy_points_back(seed):
    ids, a = _row(seed)
    assert a is not None
    sm, inp, lab, prev = a["slot_mask"], a["input_ids"], a["labels"], a["ditto_prev"]
    tok = np.flatnonzero(~sm)
    seq = inp[tok]                                        # the row's tokens in order
    tix = np.cumsum(~sm) - 1                              # token index of each position
    d = np.flatnonzero(prev >= 0)
    assert d.size > 0 and np.all(~sm[d]) and np.all(~sm[prev[d]]) and np.all(prev[d] < d)
    S = int(tix[d[0]] - tix[prev[d[0]]])
    assert np.all(tix[d] - tix[prev[d]] == S)             # exactly one span back
    assert np.array_equal(inp[d], inp[prev[d]]) and np.array_equal(lab[d], lab[prev[d]])
    first = int(tix[d[0]])                                # first marked token
    # copy 1 starts where the row first leaves the real text; the first marked token is
    # the one that predicts copy 1's first token (found independently of the marks)
    c1 = int(np.flatnonzero(seq[:ids.size] != ids[:seq.size])[0])
    assert first == c1 - 1, (first, c1)
    start = first + 1 - S                                 # where copy 0 starts
    assert start >= 1 and np.array_equal(seq[:start + S], ids[:start + S])   # real prefix
    span = seq[start:start + S]
    reps = seq[start:]
    assert np.array_equal(reps, np.resize(span, reps.size))          # then the span, repeated
    # every target in copy >= 1 is marked, nothing before it
    assert d.size == tok.size - first
    assert np.all(prev[tok[:first]] == -1)


def test_row_without_a_candidate_is_none():
    ids = np.zeros(200, dtype=np.int64)                  # every span holds an EOS
    assert pack_ditto_row(ids, _rule(), _spec(M), np.random.default_rng(0)) is None


# ── 3. the batch ─────────────────────────────────────────────────────────────────


def _buf(B=3, seed=0):
    return _ids(B=1, n=B * 400, seed=seed)[0].tolist()


def test_batch_without_ditto_is_the_old_packer_and_ditto_consumes_the_same_tokens():
    spec, rule = _spec(M), _rule()
    b0, b1, b2 = _buf(), _buf(), _buf()
    x0, y0, l0 = pack_tul_batch(b0, rule, spec, 3)
    x1, y1, l1 = pack_tul_batch(b1, rule, spec, 3, ditto_rows=0)
    assert torch.equal(x0, x1) and torch.equal(y0, y1) and l1.ditto_prev is None
    x2, y2, l2 = pack_tul_batch(b2, rule, spec, 3, ditto_rows=1,
                                ditto_rng=np.random.default_rng(0))
    assert b2 == b0                                        # same stream position
    assert torch.equal(x2[1:], x0[1:]) and torch.equal(y2[1:], y0[1:])
    assert not torch.equal(x2[0], x0[0])
    assert l2.ditto_prev.shape == x2.shape
    assert bool((l2.ditto_prev[0] >= 0).any()) and bool((l2.ditto_prev[1:] == -1).all())
    assert l2.stats["ditto_rows_built"] == 1.0
    lt = l2.to("cpu")
    assert torch.equal(lt.ditto_prev, l2.ditto_prev)
    assert torch.equal(l2.repeat_rows(2).ditto_prev[3], l2.ditto_prev[0])
    with pytest.raises(ValueError, match="ditto_rows"):
        pack_tul_batch(_buf(), rule, spec, 3, ditto_rows=3, ditto_rng=np.random.default_rng(0))


# ── 4. the model ─────────────────────────────────────────────────────────────────


def _ditto_batch():
    x, y, lay = pack_tul_batch(_buf(B=2, seed=5), _rule(), _spec(M), 2, ditto_rows=1,
                               ditto_rng=np.random.default_rng(1))
    assert bool((lay.ditto_prev >= 0).any())
    return x, y, lay


def _no_ditto(lay):
    import dataclasses
    return dataclasses.replace(lay, ditto_prev=None)


def _eval_lp(m, x, y, lay):
    m.eval()
    with torch.no_grad():
        lg = m(x, labels=None, slot_layout=_no_ditto(lay))["logits"].float()
    return torch.log_softmax(lg, -1).gather(-1, y.clamp_min(0)[..., None])[..., 0]


@pytest.mark.parametrize("kw", [dict(pointer_heads=2), {}])
def test_model_term_matches_the_eval_path_and_ditto_positions_carry_no_ce(kw):
    x, y, lay = _ditto_batch()
    m = _lx(ditto_rows=1, **kw)
    if m.tul_pointer is not None:
        with torch.no_grad():
            m.tul_pointer.gate_norm_head.bias.zero_()
    lp = _eval_lp(m, x, y, lay)
    ref, _r, pos = ditto_loss(lp, lay.ditto_prev, LAM)
    # the group loss alone: the forward's other terms read the labels too
    groups, orig = [], m._tul_group_losses
    m._tul_group_losses = lambda *a, **k: groups.append(orig(*a, **k)) or groups[-1]
    with torch.no_grad():
        o = m(x, labels=y, slot_layout=lay)
        y_masked = y.clone()
        y_masked[pos] = -100
        m(x, labels=y_masked, slot_layout=_no_ditto(lay))
    g, g_ce = groups
    assert "ditto" in o and "ditto" in g, sorted(o)
    assert abs(float(g["ditto"]) - float(ref)) < 1e-4, (float(g["ditto"]), float(ref))
    ce = float(g["loss"]) - float(g["ditto_weighted"])
    assert abs(ce - float(g_ce["loss"])) < 1e-4, (ce, float(g_ce["loss"]))
    assert int(g["ditto_n"]) == int(pos.sum())


def test_config_alone_is_the_tree_and_a_live_step_trains():
    assert _pin_run(_lx(ditto_rows=1)) == MASTER_PIN
    x, y, lay = _ditto_batch()
    m = _lx(ditto_rows=1, pointer_heads=2)
    m.train()
    o = m(x, labels=y, slot_layout=lay)
    o["loss"].backward()
    assert float(o["ditto"]) > 0
    assert m.tul_pointer.gate_norm_head.weight.grad.abs().sum() > 0


# ── 5. config ────────────────────────────────────────────────────────────────────


def test_config_refusals():
    with pytest.raises(ValueError, match="ditto_rows must be >= 0"):
        TULConfig(ditto_rows=-1)
    with pytest.raises(ValueError, match="ditto_lambda must be in"):
        TULConfig(ditto_rows=1, ditto_lambda=1.5)
    with pytest.raises(ValueError, match="code_enum_k"):
        TULConfig(ditto_rows=1, code_enum_k=2)


def test_hydra_path_val_off_manifest_and_compose_diff(monkeypatch):
    import transformers
    from hydra import compose, initialize_config_dir

    from morph.training import tul_setup
    from test_slot_gain_tail import _MISSING
    from test_tul_strict_geometry import _StubTok, _rule as _strict_rule

    tul_setup.reject_unknown_tul_keys({"ditto_rows": 1, "ditto_lambda": 0.5})
    monkeypatch.setattr(transformers, "AutoTokenizer", _StubTok)
    monkeypatch.setattr(tul_setup, "build_boundary_rule",
                        lambda cfg, cache_dir="": (_strict_rule(), _strict_rule().is_boundary,
                                                   0, ("\n",)))
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        cfg = compose(config_name="lxtul_pointer_ditto")
    rt = tul_setup.build_tul_runtime(cfg)
    assert rt.model_cfg.ditto_rows == 3 and rt.model_cfg.ditto_lambda == 0.5
    assert rt.data_cfg.ditto_rows == 3 and rt.val_data_cfg.ditto_rows == 0
    assert rt.manifest["ditto_rows"] == 3 and rt.manifest["ditto_lambda"] == 0.5
    ft, dt = _compose_leaves("lxtul_pointer_ft"), _compose_leaves("lxtul_pointer_ditto")
    diff = {k for k in ft.keys() | dt.keys() if ft.get(k, _MISSING) != dt.get(k, _MISSING)}
    assert diff == {"tul.ditto_rows", "tul.ditto_lambda", "wandb.name"}, sorted(diff)
