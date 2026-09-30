"""``tul.fan_all_wta_grader`` (2026-09-29, arm hwta): the write-all fan's WTA term
GRADED BY THE PARALLEL SPAN HEAD instead of by M extra coda passes.

Files: morph/model/tul.py (the field, `_check_fan_head_grader` near the top of
`__post_init__`, the lifted fan refusal in `_check_spandec_parallel`),
morph/model/transformer.py (`fan_head_wta_targets`, `_tul_fan_head_wta`, the
`_tul_fan_all` early return, the `fan_head_wta_weighted` fold, the oracle's
`head_table` instruments), morph/training/tul_setup.py (key, build, manifest, banner),
morph/training/train.py (both subtraction tuples, the val routing),
morph/configs/tul_slot_spandec_strict_fan4_all_fp01_{hwta,nowta}.yaml.
Note: .agents/notes/proposed/architecture/2026-09-29-fan-head-graded-wta.md

What each test pins:
  * "coda" (the default) is the tree: a2's pinned loss / eval logits / eval loss / grad
    sum (measured before this key, `tests/test_tul_lx_credit.py::FAN_PINS`), and an
    explicit "coda" equals the default bit for bit under dropout 0.1.
  * "head" runs exactly ONE coda pass at train, the same count as the no-WTA fan; a2 runs
    three (the batched pick pass, the grad winner pass, the deployed pass).
  * The head's NLL_i reads cell i ALONE (perturb cell j, NLL_i unchanged) and equals an
    independent per-cell reference built from `_readout(cells[:, :, i])`.
  * The WTA weights are (1-eps) on the argmin, eps/(M-1) on every other cell (hand check).
  * The gradient reaches the head and the loop; the head's TARGET never reaches the
    coda's input (labels changed, same seed: the coda input and the cells are equal).
  * The target is the coda table's own tokens, the dump-bin slot of a row the packer
    ended at max_slots included.
  * Real dropout > 0: finite, repeatable.
  * Every refusal fires.
  * Both configs compose, differ from a2 by the stated keys, reach TULConfig and train.
  * The val loss is the model's CE (evaluate() subtracts `fan_head_wta_weighted`; the real
    eval loss minus the term equals a2's on the same weights; both train.py tuples list
    the key), and the val instruments exist and match a hand recomputation.

CPU, fp32, the `tests/test_tul_fan.py` / `tests/test_tul_lxfan.py` strict fixtures.
"""
from __future__ import annotations

import ast
import pathlib

import pytest
import torch

from morph.model.transformer import (MORPHTransformer, accumulate_span_ce,
                                     fan_head_wta_targets, scatter_positions,
                                     span_ce_index, span_token_counts)
from morph.model.tul import TULConfig
from morph.model.tul_layout import TulLayoutSpec, slot_layout_from_ids
from test_tul_fan import _batch, _ids, _rule, _tiny
from test_tul_lx_credit import FAN_PINS, M, _fan_pin_run
from test_tul_lxfan import _build, _fan_kw, _Spy


def _head_kw(**kw) -> dict:
    """a2's fan (write-all, WTA 1.0, eps 0.05, epivol) + a2's teacher-forced span decoder,
    graded by the live parallel head."""
    base = dict(spandec=True, spandec_parallel=True, fan_all_wta_grader="head")
    base.update(kw)
    return _fan_kw(M, **base)


def _capture_cells(m: MORPHTransformer, inp, lab, layout, seed: int = 5):
    """One forward with a spy on `_tul_fan_head_wta`: the exact (cells, labels, layout)
    the model graded, detached."""
    spy = _Spy(m, "_tul_fan_head_wta")
    torch.manual_seed(seed)
    out = m(inp, labels=lab, slot_layout=layout)
    assert len(spy.calls) == 1
    cells = spy.calls[0][0][0].detach().clone()
    del m._tul_fan_head_wta                               # drop the instance-level spy
    return out, cells, spy


# ── 1. "coda" is the tree ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_coda_default_and_explicit_are_a2s_pins(dropout):
    """a2's pins were measured before this key existed; the default AND an explicit
    "coda" reproduce them exactly (loss, eval logit sum, eval loss, grad sum, n grads)."""
    a = _build(model_kw={"dropout": dropout}, **_fan_kw(M))
    b = _build(model_kw={"dropout": dropout}, **_fan_kw(M, fan_all_wta_grader="coda"))
    assert a.cfg.tul.fan_all_wta_grader == "coda" and not a._fan_head_grader
    assert _fan_pin_run(a) == FAN_PINS[("a2", dropout)]
    assert _fan_pin_run(b) == FAN_PINS[("a2", dropout)]


def test_explicit_coda_is_bit_identical_to_the_default_under_dropout():
    _ids0, inp, lab, layout = _batch(M)
    a = _build(model_kw={"dropout": 0.1}, **_fan_kw(M)).train()
    b = _build(model_kw={"dropout": 0.1}, **_fan_kw(M, fan_all_wta_grader="coda")).train()
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k, v in sa.items():
        assert torch.equal(v, sb[k]), k
    torch.manual_seed(9)
    oa = a(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(9)
    ob = b(inp, labels=lab, slot_layout=layout)
    assert oa.keys() == ob.keys()
    for key, v in oa.items():
        if torch.is_tensor(v):
            assert torch.equal(v, ob[key]), key
    assert not any("head_wta" in k for k in oa)
    oa["loss"].backward()
    ob["loss"].backward()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert (pa.grad is None) == (pb.grad is None), na
        if pa.grad is not None:
            assert torch.equal(pa.grad, pb.grad), na


# ── 2. "head" runs no extra coda pass ───────────────────────────────────────────────


def test_head_runs_one_coda_pass_like_the_no_wta_fan_and_a2_runs_three():
    _ids0, inp, lab, layout = _batch(M)
    counts = {}
    for name, kw in (("head", _head_kw()),
                     ("nowta", _fan_kw(M, spandec=True, fan_all_wta_lambda=0.0)),
                     ("a2", _fan_kw(M, spandec=True))):
        m = _build(**kw).train()
        spy = _Spy(m, "_back_region")
        torch.manual_seed(3)
        out = m(inp, labels=lab, slot_layout=layout)
        out["loss"].backward()
        counts[name] = len(spy.calls)
    assert counts["head"] == 1 and counts["nowta"] == 1, counts
    assert counts["a2"] == 3, counts


# ── 3. NLL_i reads cell i alone ─────────────────────────────────────────────────────


def test_head_nll_of_cell_i_is_a_function_of_cell_i_alone():
    _ids0, inp, lab, layout = _batch(M)
    m = _build(**_head_kw()).eval()
    with torch.no_grad():
        _out, cells, _spy = _capture_cells(m, inp, lab, layout)
        loss0, (t0, sup, ntok, ndrop) = m._tul_fan_head_wta(cells, lab, layout, {})
        assert int(ndrop) == 0 and bool(sup.any())
        # Independent reference: cell i through `_readout` ALONE, the head's own reader.
        head = m.tul_spandec_par
        ids, valid, _nd = fan_head_wta_targets(lab, layout, head.max_tokens)
        w = m.embed.lm_weight().detach()
        for i in range(M):
            zi = m._readout(cells[:, :, i])
            lp, _n, sp = head.rollout_logp(zi.unsqueeze(0), ids, valid, w,
                                           mask_token_id=m.cfg.tul.slot_id)
            assert torch.equal(sp, sup)
            torch.testing.assert_close(t0[..., i][sup], -lp[0], rtol=1e-5, atol=1e-4)
        # Perturb cell j = 2 only: NLL_i for i != 2 is bit-identical, NLL_2 moves.
        cells2 = cells.clone()
        torch.manual_seed(0)
        cells2[:, :, 2] += 0.5 * torch.randn_like(cells2[:, :, 2])
        _l2, (t2, sup2, _n2, _d2) = m._tul_fan_head_wta(cells2, lab, layout, {})
    assert torch.equal(sup2, sup)
    for i in range(M):
        if i == 2:
            assert not torch.allclose(t2[..., i][sup], t0[..., i][sup])
        else:
            assert torch.equal(t2[..., i], t0[..., i]), i


# ── 4. the relaxed WTA weights ──────────────────────────────────────────────────────


@pytest.mark.parametrize("eps", [0.05, 0.2])
def test_relaxed_wta_weights_are_one_minus_eps_and_eps_over_m_minus_one(eps):
    _ids0, inp, lab, layout = _batch(M)
    m = _build(**_head_kw(fan_select_eps=eps)).eval()
    with torch.no_grad():
        _out, cells, _spy = _capture_cells(m, inp, lab, layout)
        stats: dict = {}
        loss, (t, sup, ntok, _nd) = m._tul_fan_head_wta(cells, lab, layout, stats)
    nll = t[sup].double()                                   # [N, M] summed NLL per slot
    n = float(ntok[sup].sum())
    win = nll.argmin(dim=-1)
    hand = 0.0
    for r in range(nll.shape[0]):
        for i in range(M):
            wgt = (1.0 - eps) if i == int(win[r]) else eps / (M - 1)
            hand += wgt * float(nll[r, i])
    hand /= n
    assert float(loss) == pytest.approx(hand, rel=1e-6)
    assert float(stats["head_wta_ce"]) == pytest.approx(hand, rel=1e-6)
    assert float(stats["head_wta_winner_nll"]) == pytest.approx(
        float(nll.min(dim=-1).values.sum()) / n, rel=1e-6)
    assert float(stats["head_wta_mean_nll"]) == pytest.approx(
        float(nll.mean(dim=-1).sum()) / n, rel=1e-6)
    shares = [float(stats[f"head_wta_share_k{i}"]) for i in range(M)]
    assert sum(shares) == pytest.approx(1.0, abs=1e-6)
    for i in range(M):
        assert shares[i] == pytest.approx(float((win == i).double().mean()), abs=1e-6)


# ── 5. gradient reaches the head and the loop; the target never reaches the coda ────


def test_gradient_reaches_every_cell_and_the_head():
    _ids0, inp, lab, layout = _batch(M)
    m = _build(**_head_kw()).train()
    _out, cells, _spy = _capture_cells(m, inp, lab, layout)
    m.zero_grad(set_to_none=True)
    c = cells.clone().requires_grad_(True)
    loss, (_t, sup, _n, _d) = m._tul_fan_head_wta(c, lab, layout, {})
    loss.backward()
    for i in range(M):
        # eps > 0: every cell of every scored slot carries a weight, so every one moves.
        assert float(c.grad[:, :, i][sup].abs().sum()) > 0.0, i
        assert float(c.grad[:, :, i][~sup].abs().sum()) == 0.0, i
    hg = [p.grad for p in m.tul_spandec_par.parameters() if p.grad is not None]
    assert hg and all(torch.isfinite(g).all() for g in hg)
    assert sum(float(g.abs().sum()) for g in hg) > 0.0
    # From the cells, the term reaches the readout (`lm_mixer`, `final_norm`) and the head
    # and nothing else: the tied table is the head's OUTPUT and is detached there.
    moved = {n for n, p in m.named_parameters()
             if p.grad is not None and float(p.grad.abs().sum()) > 0.0}
    assert moved and all(n.startswith(("tul_spandec_par.", "lm_mixer.", "final_norm."))
                         for n in moved), sorted(moved)


def test_the_live_term_reaches_the_loop_through_the_cells():
    """End to end: the WTA term the forward folds has a gradient into the register (the
    loop's cell seeds) and the core, not only into the head."""
    _ids0, inp, lab, layout = _batch(M)
    m = _build(**_head_kw()).train()
    spy = _Spy(m, "_tul_fan_head_wta")
    torch.manual_seed(5)
    m(inp, labels=lab, slot_layout=layout)
    live = spy.outs[0][0]
    loop_params = [p for n, p in m.named_parameters()
                   if n.startswith(("tul_register.", "core.")) and p.requires_grad]
    grads = torch.autograd.grad(live, loop_params, allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in grads)


def test_the_head_target_never_reaches_the_coda_input_or_the_cells():
    """Change the LABELS (the head's target) at every scored position, same inputs, same
    seed, dropout 0: the cells the head grades and the carrier the coda reads are
    bit-identical, and only the head's term moves."""
    _ids0, inp, lab, layout = _batch(M)
    lab2 = lab.clone()
    scored = (lab2 >= 0) & (~layout.slot_mask)
    lab2[scored] = 5 + (lab2[scored] - 5 + 7) % (64 - 5)     # a different real token id
    assert bool((lab2 != lab)[scored].all()) and not bool((lab2 == 4).any())
    runs = []
    for lb in (lab, lab2):
        m = _build(**_head_kw()).train()
        spy_b = _Spy(m, "_back_region")
        spy_h = _Spy(m, "_tul_fan_head_wta")
        torch.manual_seed(11)
        out = m(inp, labels=lb, slot_layout=layout)
        runs.append((spy_b.calls[0][0][0].detach(), spy_h.calls[0][0][0].detach(),
                     float(out["fan_head_wta_ce"])))
    assert torch.equal(runs[0][0], runs[1][0]), "the coda's input moved with the target"
    assert torch.equal(runs[0][1], runs[1][1]), "the cells moved with the target"
    assert runs[0][2] != runs[1][2], "non-vacuous: the head's term must read the target"


# ── 6. the target is the coda table's own tokens ────────────────────────────────────


@pytest.mark.parametrize("max_slots", [10, 3])
def test_targets_are_the_coda_tables_tokens_including_the_dump_bin(max_slots):
    """`fan_head_wta_targets` against `span_ce_index` + `span_token_counts`, position by
    position. max_slots 3 ends every row at the slot budget, so the LAST slot's next span
    is the dump bin, which the coda table scores and `span_slots` would not."""
    ids = _ids(B=2, n=200, seed=1)
    spec = TulLayoutSpec(seq_len=64, prefix_k=4, max_slots=max_slots, slot_id=4)
    _inp, lab, layout, _ = slot_layout_from_ids(ids, _rule(), spec)
    J = 32
    tid, valid, n_drop = fan_head_wta_targets(lab, layout, J)
    assert int(n_drop) == 0
    gid, keep_tok, lab_c, G = span_ce_index(lab, layout)
    n_tok = span_token_counts(gid, keep_tok, G)[:, 1:]
    expect = torch.where(layout.slot_valid, n_tok, torch.zeros_like(n_tok))
    assert torch.equal(valid.sum(-1).to(expect.dtype), expect)
    S = layout.slot_valid.shape[1]
    dump_scored = False
    for b in range(lab.shape[0]):
        for s in range(S):
            if not bool(layout.slot_valid[b, s]):
                continue
            pos = torch.nonzero((gid[b] == b * G + s + 1) & keep_tok[b]).flatten()
            want = lab_c[b, pos]
            got = tid[b, s][valid[b, s]]
            assert torch.equal(got, want), (b, s)
            if s + 1 == S and pos.numel() > 0:
                dump_scored = True
    if max_slots == 3:
        assert dump_scored, "precondition: a row must end at max_slots with a scored tail"


def test_the_model_grades_the_dump_bin_slot_and_the_val_check_passes():
    """Model level: on rows the packer ended at max_slots, the head's table (the one the
    WTA term is built from) scores the LAST slot's next span, and the val oracle's
    identical-tokens check (which RAISES on any slot or token-count mismatch) passes."""
    ids = _ids(B=2, n=200, seed=1)
    spec = TulLayoutSpec(seq_len=64, prefix_k=4, max_slots=3, slot_id=4)
    inp, lab, layout, _ = slot_layout_from_ids(ids, _rule(), spec)
    m = _build(**_head_kw()).eval()
    spy = _Spy(m, "_tul_fan_head_wta")
    with torch.no_grad():
        out = m.tul_forward_with_plan_nats(inp, lab, layout)
    _loss, (_t, sup, _n, n_drop) = spy.outs[0]
    assert int(n_drop) == 0
    assert bool(sup[:, -1].any()), "the dump-bin slot must be scored by the head"
    assert "fan_head_coda_agree" in out


# ── 7. real dropout ─────────────────────────────────────────────────────────────────


def test_real_dropout_train_step_is_finite_and_repeatable():
    _ids0, inp, lab, layout = _batch(M)
    res = []
    for _ in range(2):
        m = _build(model_kw={"dropout": 0.1}, **_head_kw()).train()
        torch.manual_seed(21)
        out = m(inp, labels=lab, slot_layout=layout)
        out["loss"].backward()
        g = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
        assert torch.isfinite(out["loss"]) and all(torch.isfinite(v).all() for v in g.values())
        assert float(out["fan_head_wta_weighted"]) > 0.0
        res.append((out["loss"].detach(), g))
    assert torch.equal(res[0][0], res[1][0])
    assert res[0][1].keys() == res[1][1].keys()
    for k in res[0][1]:
        assert torch.equal(res[0][1][k], res[1][1][k]), k


# ── 8. refusals ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kw,exc,match", [
    (dict(fan_all_wta_grader="bogus"), ValueError, "must be 'coda' or 'head'"),
    (dict(fan_mix="softmax"), NotImplementedError, "fan_mix='softmax'"),
    (dict(fan_all_wta_lambda=0.0), NotImplementedError, "fan_all_wta_lambda=0"),
    (dict(code_enum_k=4), NotImplementedError, "code_enum_k=4"),
    (dict(spandec_parallel=False), NotImplementedError, "spandec_parallel=false"),
    (dict(spandec_parallel_detach=True), NotImplementedError, "spandec_parallel_detach=true"),
    (dict(spandec_parallel_k=4), NotImplementedError, "spandec_parallel_k=4"),
    (dict(spandec_parallel_span_cap=8), NotImplementedError, "spandec_parallel_span_cap=8"),
    (dict(spandec_target_offset=2), NotImplementedError, "spandec_target_offset=2"),
    (dict(spandec_max_tokens=16), NotImplementedError, "head J=16"),
    (dict(spandec_parallel_weight=0.5), NotImplementedError, "spandec_parallel_weight=0.5"),
    (dict(fan_all_wta_winner="latent"), NotImplementedError, "fan_all_wta_winner='latent'"),
    (dict(fan_all_wta_grad_rollouts="map"), NotImplementedError, "fan_all_wta_grad_rollouts"),
    (dict(fan_history_streams=1), NotImplementedError, "fan_history_streams=1"),
])
def test_head_grader_refusals(kw, exc, match):
    with pytest.raises(exc, match=match):
        TULConfig(**_head_kw(**kw))


def test_head_grader_needs_a_fan_and_the_head_stays_refused_on_the_coda_grader():
    with pytest.raises(NotImplementedError, match="fan_k=0"):
        TULConfig(fan_all_wta_grader="head")
    # The fan's refusal of the parallel head is lifted for the head grader ONLY.
    with pytest.raises(NotImplementedError, match="tul.fan_k"):
        TULConfig(**_fan_kw(M, spandec_parallel=True))
    TULConfig(**_head_kw())                                  # the legal arm builds


# ── 9. configs ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,stated,wb", [
    ("tul_slot_spandec_strict_fan4_all_fp01_hwta",
     {"tul.spandec_parallel": True, "tul.spandec_parallel_detach": False,
      "tul.fan_all_wta_grader": "head"},
     "slot-spandec-strict-fan4-all-fp01-hwta"),
    ("tul_slot_spandec_strict_fan4_all_fp01_nowta",
     {"tul.fan_all_wta_lambda": 0.0},
     "slot-spandec-strict-fan4-all-fp01-nowta"),
])
def test_configs_compose_differ_from_a2_by_the_stated_keys_and_train(name, stated, wb,
                                                                      monkeypatch):
    from omegaconf import OmegaConf
    from test_slot_gain_tail import _leaves, _MISSING
    from test_tul_strict_geometry import _runtime

    from morph.training.train import build_morph_config

    parent = "tul_slot_spandec_strict_fan4_all_fp01"
    cfg, rt = _runtime(name, monkeypatch)
    pcfg, prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
    changed = {k for k, v in stated.items() if p.get(k, _MISSING) != v} | {"wandb.name"}
    assert diff == changed, sorted(diff)
    assert c["wandb.name"] == wb and c["training.steps"] == p["training.steps"] == 5000
    for k, v in stated.items():
        assert getattr(rt.model_cfg, k.split(".", 1)[1]) == v, k
    assert prt.model_cfg.fan_all_wta_grader == "coda" and prt.model_cfg.fan_all_wta_lambda == 1.0
    mc = build_morph_config(cfg, tul=rt.model_cfg)
    assert mc.tul.fan_all_wta_grader == rt.model_cfg.fan_all_wta_grader
    assert mc.core_fixed_point_lambda == 0.1
    # Train one step on the tiny model with the composed tul block.
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=rt.model_cfg, d_ff=96,
                               core_fixed_point_lambda=mc.core_fixed_point_lambda)
                         ).train().float()
    assert (m.tul_spandec_par is not None) == (name.endswith("_hwta"))
    _ids0, inp, lab, layout = _batch(M)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert ("fan_head_wta_weighted" in out) == name.endswith("_hwta")
    assert "fan_wta_weighted" not in out


# ── 10. the val loss is the model's CE; the val instruments ─────────────────────────


class _AuxStub(torch.nn.Module):
    """The val path's view of a head-grader model: a loss that includes the term."""

    def tul_forward_with_plan_nats(self, x, y, layout):
        return {"loss": torch.tensor(6.0), "fan_head_wta_weighted": torch.tensor(1.5),
                "fan_head_wta_ce": torch.tensor(1.5), "fan_head_coda_agree": torch.tensor(0.5),
                "fan_head_pick_regret": torch.tensor(0.01),
                "fan_rand_pick_regret": torch.tensor(0.02),
                "ce_tokens": 4.5, "layer_passes": 8.0, "n_tokens": 4.0}


class _Layout:
    stats: dict = {}

    def to(self, device):
        return self


def test_evaluate_subtracts_the_head_term_and_routes_the_readings():
    from morph.training.train import evaluate
    x = torch.zeros(1, 4, dtype=torch.long)
    extra: dict = {}
    avg, _ppl = evaluate(_AuxStub(), torch.device("cpu"), iter([(x, x, _Layout())] * 2),
                         n_batches=2, tul=True, extra=extra)
    assert avg == pytest.approx(4.5)
    assert extra["fan/head_coda_agree"] == pytest.approx(0.5)
    assert extra["fan/head_pick_regret"] == pytest.approx(0.01)
    assert extra["fan/rand_pick_regret"] == pytest.approx(0.02)
    # train emits `fan_head_wta_*` too, so val routes them away from the train series
    assert extra["val/fan_head_wta_ce"] == pytest.approx(1.5)
    assert "fan/head_wta_ce" not in extra


def _tuple_after(src: str, var: str) -> set[str]:
    """The string constants of the tuple a `for <var> in (...)` loop in train.py walks."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id == var and isinstance(node.iter, ast.Tuple)):
            return {e.value for e in node.iter.elts if isinstance(e, ast.Constant)}
    raise AssertionError(f"no `for {var} in (...)` loop in train.py")


def test_both_train_py_subtraction_tuples_list_the_key():
    """A SOURCE check (not an execution check): the val loop's `_aux2` tuple and the train
    step's `_ak` tuple both carry `fan_head_wta_weighted`, so train/loss and val loss are
    the model's CE. The val tuple is exercised end to end by the evaluate() test above;
    the train step's is not unit-testable without running the trainer."""
    src = pathlib.Path("morph/training/train.py").read_text()
    assert "fan_head_wta_weighted" in _tuple_after(src, "_aux2")
    assert "fan_head_wta_weighted" in _tuple_after(src, "_ak")


def test_real_eval_loss_minus_the_term_is_a2s_loss_on_the_same_weights():
    """The head arm's base weights ARE a2's (the head is built RNG-neutral); at eval the
    only thing the head grader adds to the loss is `fan_head_wta_weighted`."""
    _ids0, inp, lab, layout = _batch(M)
    h = _build(**_head_kw()).eval()
    a = _build(**_fan_kw(M, spandec=True)).eval()
    sh = {k: v for k, v in h.state_dict().items() if not k.startswith("tul_spandec_par.")}
    sa = a.state_dict()
    assert sh.keys() == sa.keys()
    for k in sa:
        assert torch.equal(sh[k], sa[k]), k
    with torch.no_grad():
        oh = h.tul_forward_with_plan_nats(inp, lab, layout)
        oa = a.tul_forward_with_plan_nats(inp, lab, layout)
    assert float(oh["fan_head_wta_weighted"]) > 0.0
    assert "fan_head_wta_weighted" not in oa and "fan_wta_weighted" not in oa
    assert float(oh["ce_tokens"]) == float(oa["ce_tokens"])
    torch.testing.assert_close(oh["loss"] - oh["fan_head_wta_weighted"], oa["loss"],
                               rtol=0.0, atol=2e-6)


def test_val_instruments_exist_and_match_a_hand_recomputation():
    _ids0, inp, lab, layout = _batch(M)
    m = _build(**_head_kw()).eval()
    spy = _Spy(m, "_tul_fan_oracle")
    with torch.no_grad():
        out = m.tul_forward_with_plan_nats(inp, lab, layout)
    agree = float(out["fan_head_coda_agree"])
    reg_h = float(out["fan_head_pick_regret"])
    reg_r = float(out["fan_rand_pick_regret"])
    assert 0.0 <= agree <= 1.0 and reg_h >= 0.0 and reg_r >= 0.0
    # Hand recomputation from the oracle's own inputs: one coda replay per cell, the
    # summed span CE table, then per-TOKEN regrets divided ONCE by the token total.
    a, k = spy.calls[0]
    cells, xh, base, x0, bigram, ids_, labels, lay, L, keep, coda_kw, tg_reset, _st = a
    head_nll, head_sup, _hn, _hd = k["head_table"]
    gid, keep_tok, lab_c, G = span_ce_index(labels, lay)
    w_head = m.embed.lm_weight()
    per = []
    with torch.no_grad():
        for i in range(M):
            values, pos = m._tul_fan_stream_write(cells, i, lay, L)
            xi = scatter_positions(base, pos, values)
            xhi = m._back_region(xi, x0, bigram, ids_, inject_keep=keep, attn_kwargs=coda_kw,
                                 ret_reset_mask=tg_reset)
            per.append(accumulate_span_ce(xhi, w_head, gid, keep_tok, lab_c, G)[:, 1:])
    ce = torch.stack(per, dim=-1).double()
    n_tok = span_token_counts(gid, keep_tok, G)[:, 1:]
    ok = lay.slot_valid & (n_tok > 0)
    assert torch.equal(head_sup, ok)
    best, arg = ce.min(dim=-1)
    h_arg = head_nll.argmin(dim=-1)
    denom = float(n_tok[ok].sum())
    assert agree == pytest.approx(float((h_arg[ok] == arg[ok]).double().mean()), abs=1e-6)
    want_h = float((ce.gather(-1, h_arg.unsqueeze(-1)).squeeze(-1) - best)[ok].sum()) / denom
    want_r = float((ce.mean(dim=-1) - best)[ok].sum()) / denom
    assert reg_h == pytest.approx(want_h, rel=1e-4, abs=1e-6)
    assert reg_r == pytest.approx(want_r, rel=1e-4, abs=1e-6)
    assert reg_r > 0.0, "non-vacuous: the cells must differ for the random floor to exist"
