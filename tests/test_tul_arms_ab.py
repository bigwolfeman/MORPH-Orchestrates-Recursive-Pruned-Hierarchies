"""The slot-channel arms of 2026-09-26: (a) a WIDER channel, (b) a LOOSER coda geometry.

fp01 (`tul_slot_spandec_strict_e4probe_fp01`) trails the plain model by 0.265 nats at 5k
and 0.333 at 10k (lab/experiments/failures/2026-09-26-lxtul-fp01-vs-plain-10k.md). Under
strict geometry the slot loop is the only cross-span channel, so two arms test the channel:

* (a1) `tul_slot_spandec_strict_e4probe_fp01_pk4`: fp01 + `tul.prefix_k: 4`.
* (a2) `tul_slot_spandec_strict_fan4_all_fp01`: fan4 write-all + fp01's fixed-point term.
  The fan is REFUSED on fp01's code_enum / parallel head, and a test pins that refusal.
* (b)  `tul_slot_spandec_strict_e4probe_fp01_reach1`: fp01 + `tul.tg_coda_token_reach: 1`,
  a coda token also reads the PREVIOUS span's tokens directly.

WHAT THIS FILE PROVES about (b), by PERTURBATION, not by reading the mask:

1. REACH 0 IS THE TREE. The relation equals a brute-force reference of the documented
   strict relation; the reach-0 forward never reaches the new code (a patch that raises
   inside it is never hit); loss / logits / every grad are `torch.equal` to a model that
   never set the key; and the HEAD pins below were measured on a `git archive` of
   7cb6783 (the tree before this key), where the same numbers came out.
2. THE SLOT CHANNEL CAN BE REMOVED. `plan_mode="zero"` zeroes the loop's prefix write.
   At reach 0 an edit one or two spans back then moves span s by EXACTLY 0, and with the
   write ON the same edit moves it. So the ablation removes the slot channel and nothing
   else is left.
3. ONE LAYER'S RELATION. With the channel removed and ONE coda layer, reach r: an edit
   r+1 spans back moves span s by exactly 0, an edit r spans back moves it. This is what
   the conv / value-shift partition must hold: a key's conv features never cross into
   another segment, so s-2 cannot ride in on a key of s-1.
4. THE REAL RECEPTIVE FIELD. Layers compose. With the channel removed, reach 1 and n
   coda layers, an edit m spans back moves span s iff m <= n. Measured at n = 1, 2, 3.
5. THE OPEN TAIL (dump bin) reads the last real span and not the one before it.
6. SABOTAGES, each caught: the new mask term dropped; the coda conv / value-shift
   reset dropped; the dump-bin ordinal dropped.

CPU, fp32, `use_kernels=False`, the `tests/test_tul_strict_geometry.py` fixtures. fp32
carries the exact zeros this file asserts.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

import morph.model.transformer as transformer_mod
import morph.model.tul_layout as tul_layout_mod
from morph.model.transformer import MORPHTransformer
from morph.model.tul_layout import (
    TulLayoutSpec,
    slot_layout_from_ids,
    strict_span_ordinal,
    tg_strict_allow,
)
from test_tul_lxtul_e import _D_FF, _tc
from test_tul_strict_geometry import _edit, _pack, _rule, _runtime, _spec, _tiny, _tul

# A window that holds a whole previous span of the fixture (8-9 tokens + 2 cells), the
# way fp01's window_size 256 holds a span_cap-32 span. The relation, not the window, is
# then what bounds the reach.
_WIN = 32


def _model(n_coda: int = 2, seed: int = 1234, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(tg_geometry="strict", **tul_kw), n_coda=n_coda,
                               window_size=_WIN))
    # `BigramEmbedding.lambdas` is zero-init: without this the bigram route is off and an
    # id-perturbation test would not exercise it (the strict-geometry file's lesson).
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.eval().float()


def _logits(m: MORPHTransformer, ids: np.ndarray, plan_mode: str):
    inp, _lab, layout, _ = slot_layout_from_ids(ids, _rule(), _spec())
    with torch.no_grad():
        if plan_mode == "normal":
            out = m(inp, labels=None, slot_layout=layout)
        else:
            out = m.tul_forward_ablated(inp, None, layout, plan_mode=plan_mode)
    return out["logits"], layout


def _span_delta(m: MORPHTransformer, edit_span: int, read_span: int, row: int = 0,
                plan_mode: str = "zero") -> tuple[float, float]:
    """``(max |dlogit| over read_span's TOKENS, max |dlogit| over edit_span's tokens)``
    when one token of ``edit_span`` changes. ``read_span`` is a span ORDINAL
    (:func:`strict_span_ordinal`), so the open tail is addressable."""
    ids, _inp, _lab, layout = _pack()
    edited = _edit(ids, layout, row, edit_span)
    a, lay = _logits(m, ids, plan_mode)
    b, _ = _logits(m, edited, plan_mode)
    span = strict_span_ordinal(lay)[row]
    tok = ~lay.slot_mask[row]
    # The slot_id column is -inf everywhere and -inf - -inf is NaN: compare by inequality.
    d = (a[row] - b[row]).abs().nan_to_num(0.0)
    d = torch.where(a[row] != b[row], d, torch.zeros_like(d))
    read = tok & (span == read_span)
    own = tok & (span == edit_span)
    assert bool(read.any()) and bool(own.any()), "fixture: both spans must hold tokens"
    return float(d[read].max()), float(d[own].max())


# ── 1. the relation, and REACH 0 IS THE TREE ────────────────────────────────

def _reference(layout, prefix_reach: str, token_reach: int) -> torch.Tensor:
    """Brute force of the documented strict coda relation, one pair at a time."""
    B, L = layout.bag_id.shape
    ms = layout.max_slots
    out = torch.zeros(B, 1, L, L, dtype=torch.bool)
    for b in range(B):
        bag = layout.bag_id[b].tolist()
        sm = layout.slot_mask[b].tolist()
        n_valid = int(layout.slot_valid[b].sum())
        ordn = [n_valid if x == ms else x for x in bag]
        for i in range(L):
            for j in range(i + 1):
                if sm[i]:
                    ok = i == j
                else:
                    cell = sm[j] and (bag[j] < bag[i] if prefix_reach == "all"
                                      else (bag[j] == bag[i] - 1 and bag[i] < ms))
                    prev_tok = (not sm[j]) and 1 <= ordn[i] - ordn[j] <= token_reach
                    ok = bag[i] == bag[j] or cell or prev_tok
                out[b, 0, i, j] = ok
    return out


@pytest.mark.parametrize("prefix_reach", ["all", "prev"])
@pytest.mark.parametrize("token_reach", [0, 1, 2])
def test_the_coda_relation_matches_a_brute_force_reference(prefix_reach, token_reach):
    _ids, _inp, _lab, layout = _pack()
    got = tg_strict_allow(layout, "coda", coda_prefix_reach=prefix_reach,
                          coda_token_reach=token_reach)
    assert torch.equal(got, _reference(layout, prefix_reach, token_reach))


def test_reach0_relation_is_the_default_call_and_reach1_is_wider_by_tokens_only():
    _ids, _inp, _lab, layout = _pack()
    r0 = tg_strict_allow(layout, "coda")
    assert torch.equal(r0, tg_strict_allow(layout, "coda", coda_token_reach=0))
    r1 = tg_strict_allow(layout, "coda", coda_token_reach=1)
    added = r1 & ~r0
    assert bool(added.any()) and not bool((r0 & ~r1).any()), "reach 1 must only widen"
    sm = layout.slot_mask
    # every added pair is (token query, token key): no cell gains or loses anything, so
    # the compressed branch (slot columns only) is unchanged
    assert not bool((added & sm.view(*sm.shape, 1).unsqueeze(1)).any())
    assert not bool((added & sm.view(sm.shape[0], 1, 1, -1)).any())
    # the prelude takes no token reach at all
    with pytest.raises(ValueError, match="CODA relation"):
        tg_strict_allow(layout, "prelude", coda_token_reach=1)
    with pytest.raises(ValueError, match=">= 0"):
        tg_strict_allow(layout, "coda", coda_token_reach=-1)


def test_the_open_tail_is_the_span_after_the_last_slot():
    _ids, _inp, _lab, layout = _pack()
    span = strict_span_ordinal(layout)
    for b in range(layout.bag_id.shape[0]):
        n = int(layout.slot_valid[b].sum())
        tail = (layout.bag_id[b] == layout.max_slots) & ~layout.slot_mask[b]
        assert bool(tail.any()), "fixture: the row must have an open tail"
        assert bool((span[b][tail] == n).all())
        real = layout.bag_id[b] < layout.max_slots
        assert torch.equal(span[b][real], layout.bag_id[b][real])


# The tree before `tul.tg_coda_token_reach` (7cb6783), measured 2026-09-26 on a `git
# archive` of that commit with the same fixtures, seed and single CPU thread. The
# worktree reproduced loss, eval logits and all 174 / 190 gradient tensors `torch.equal`.
HEAD_STRICT = (5.045138835906982, 842.074198674527, 977.2346582540736, 174)
HEAD_FP01ISH = (9.599393844604492, -52828.743996977806, 1523.114804623198, 190)


def _pin_run(mk) -> tuple[float, float, float, int, dict, torch.Tensor]:
    torch.manual_seed(1234)
    m = MORPHTransformer(mk()).train().float()
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    _ids, inp, lab, lay = _pack()
    torch.manual_seed(99)
    o = m(inp, labels=lab, slot_layout=lay)
    o["loss"].backward()
    m.eval()
    with torch.no_grad():
        lg = m(inp, labels=None, slot_layout=lay)["logits"]
    grads = {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}
    ls = float(torch.nan_to_num(lg.double(), nan=0.0, posinf=0.0, neginf=0.0).sum())
    gs = float(sum(g.double().abs().sum() for g in grads.values()))
    return float(o["loss"].detach()), ls, gs, len(grads), grads, lg


@pytest.mark.parametrize("name,mk,pin", [
    ("strict", lambda **kw: _tiny(tul=_tul(tg_geometry="strict", **kw)), HEAD_STRICT),
    ("fp01ish", lambda **kw: _tiny(tul=_tc(**kw), d_ff=_D_FF), HEAD_FP01ISH),
])
def test_reach0_is_bit_identical_to_the_tree(name, mk, pin, monkeypatch):
    base = _pin_run(mk)
    assert base[:4] == pin, f"{name}: the default forward moved off the tree's pin"

    def _boom(*_a, **_k):
        raise AssertionError("reach 0 reached the new token-reach term")
    monkeypatch.setattr(tul_layout_mod, "strict_span_ordinal", _boom)
    zero = _pin_run(lambda: mk(tg_coda_token_reach=0))
    assert zero[:4] == pin
    assert base[4].keys() == zero[4].keys()
    assert all(torch.equal(base[4][k], zero[4][k]) for k in base[4])
    assert torch.equal(base[5].nan_to_num(), zero[5].nan_to_num())
    # two-sided: the patch point is live, so reach 1 must hit it
    with pytest.raises(AssertionError, match="new token-reach term"):
        _pin_run(lambda: mk(tg_coda_token_reach=1))


def test_reach1_changes_the_forward_and_adds_no_parameter():
    m0, m1 = _model(), _model(tg_coda_token_reach=1)
    assert list(m0.state_dict()) == list(m1.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(m0.state_dict().values(),
                                                 m1.state_dict().values()))
    ids, *_ = _pack()
    a, _ = _logits(m0, ids, "normal")
    b, _ = _logits(m1, ids, "normal")
    assert not torch.equal(a.nan_to_num(), b.nan_to_num()), "the knob is inert"


# ── 2. the ablation removes the slot channel ────────────────────────────────

@pytest.mark.parametrize("m_back", [1, 2])
def test_plan_zero_removes_the_slot_channel_at_reach0(m_back):
    s = 5
    m = _model(n_coda=2)
    d_read, d_own = _span_delta(m, s - m_back, s, plan_mode="zero")
    assert d_own > 0, "the edit moved nothing: the fixture is inert"
    assert d_read == 0.0, f"reach 0, write zeroed: span {s - m_back} moved span {s}"
    d_on, _ = _span_delta(m, s - m_back, s, plan_mode="normal")
    assert d_on > 0, "with the write ON the loop must carry the edit (two-sided)"


def test_plan_zero_removes_the_slot_channel_on_the_fp01_shape():
    """The same verification on the code_enum_k 4 + parallel-head model the arm runs."""
    torch.manual_seed(1234)
    m = MORPHTransformer(_tiny(tul=_tc(), d_ff=_D_FF, window_size=_WIN)).eval().float()
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    d_read, d_own = _span_delta(m, 3, 5, plan_mode="zero")
    assert d_own > 0 and d_read == 0.0
    assert _span_delta(m, 3, 5, plan_mode="normal")[0] > 0


# ── 3. one coda layer's relation ────────────────────────────────────────────

@pytest.mark.parametrize("reach", [1, 2])
def test_one_coda_layer_reads_exactly_reach_spans_back(reach):
    s = 5
    m = _model(n_coda=1, tg_coda_token_reach=reach)
    far, own = _span_delta(m, s - reach - 1, s)
    assert own > 0 and far == 0.0, (
        f"LEAK: one coda layer at reach {reach} moved span {s} from span "
        f"{s - reach - 1} by {far:.3e} with the slot channel removed")
    near, _ = _span_delta(m, s - reach, s)
    assert near > 0, f"reach {reach}: span {s - reach} did not reach span {s}"


def test_one_coda_layer_on_the_fp01_shape():
    torch.manual_seed(1234)
    m = MORPHTransformer(_tiny(tul=_tc(tg_coda_token_reach=1), d_ff=_D_FF, n_coda=1,
                               window_size=_WIN)).eval().float()
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    assert _span_delta(m, 3, 5)[0] == 0.0
    assert _span_delta(m, 4, 5)[0] > 0


# ── 4. the real receptive field ─────────────────────────────────────────────

def receptive_field(n_coda: int, reach: int = 1, s: int = 6) -> dict[int, float]:
    """``{m: max |dlogit| of span s}`` for an edit m spans back, slot channel removed."""
    m = _model(n_coda=n_coda, tg_coda_token_reach=reach)
    return {k: _span_delta(m, s - k, s)[0] for k in range(1, s)}


@pytest.mark.parametrize("n_coda", [1, 2, 3])
def test_the_token_path_reaches_reach_times_n_coda_spans(n_coda):
    rf = receptive_field(n_coda)
    moved = {k for k, v in rf.items() if v > 0}
    assert moved == set(range(1, n_coda + 1)), (
        f"n_coda={n_coda}: spans back that move span 6 = {sorted(moved)}, expected "
        f"1..{n_coda}; readings {rf}")


# ── 5. the open tail ────────────────────────────────────────────────────────

def test_the_open_tail_reads_the_last_real_span_and_not_the_one_before():
    m = _model(n_coda=1, tg_coda_token_reach=1)
    _ids, _inp, _lab, layout = _pack()
    n = int(layout.slot_valid[0].sum())
    assert _span_delta(m, n - 1, n)[0] > 0
    assert _span_delta(m, n - 2, n)[0] == 0.0


# ── 6. sabotages ────────────────────────────────────────────────────────────

def test_sabotage_new_mask_dropped_is_caught(monkeypatch):
    real = tg_strict_allow

    def fake(layout, stage, coda_prefix_reach="all", coda_token_reach=0):
        return real(layout, stage, coda_prefix_reach=coda_prefix_reach)
    monkeypatch.setattr(transformer_mod, "tg_strict_allow", fake)
    m = _model(n_coda=1, tg_coda_token_reach=1)
    near, own = _span_delta(m, 4, 5)
    assert own > 0 and near == 0.0, "the positive side of the reach test would still pass"


def test_sabotage_coda_conv_reset_dropped_is_caught(monkeypatch):
    """Drop the CODA's segment reset only (the prelude keeps it): the conv and W_v_prev
    at span s-1's head then read span s-2's tail, and s-2 reaches span s in ONE layer."""
    real = MORPHTransformer._tul_tg_kwargs

    def fake(self, layout):
        fk, fr, ck, cr = real(self, layout)
        ck = dict(ck)
        ck["tg_seg"] = torch.zeros_like(ck["tg_seg"])
        return fk, fr, ck, cr
    monkeypatch.setattr(MORPHTransformer, "_tul_tg_kwargs", fake)
    m = _model(n_coda=1, tg_coda_token_reach=1)
    far, own = _span_delta(m, 3, 5)
    assert own > 0 and far > 0.0, "the one-layer leak test would not see the conv route"


def test_sabotage_open_tail_ordinal_dropped_is_caught(monkeypatch):
    monkeypatch.setattr(tul_layout_mod, "strict_span_ordinal", lambda layout: layout.bag_id)
    m = _model(n_coda=1, tg_coda_token_reach=1)
    _ids, _inp, _lab, layout = _pack()
    n = int(layout.slot_valid[0].sum())
    assert _span_delta(m, n - 1, n)[0] == 0.0, "the open-tail test would still pass"


# ── the refusals ────────────────────────────────────────────────────────────

def test_token_reach_is_refused_outside_strict_and_negative():
    with pytest.raises(ValueError, match="tg_geometry='strict' knob"):
        _tul(tg_coda_token_reach=1)
    with pytest.raises(ValueError, match=">= 0"):
        _tul(tg_geometry="strict", tg_coda_token_reach=-1)
    # composes with the prev prefix reach: a different, independent disjunct
    _tul(tg_geometry="strict", tg_coda_prefix_reach="prev", tg_coda_token_reach=1)


def test_token_reach_is_refused_with_the_graded_target():
    tc = _tul(tg_geometry="strict")
    with pytest.raises(NotImplementedError, match="code_grade"):
        dataclasses.replace(tc, tg_coda_token_reach=1, code_grade=True, code_target=True)


# ── the arms: Hydra compose, key diff, build, one train step ────────────────

def _diff(name: str, parent: str, monkeypatch):
    from omegaconf import OmegaConf
    from test_slot_gain_tail import _MISSING, _leaves
    cfg, rt = _runtime(name, monkeypatch)
    pcfg, prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    return cfg, rt, prt, c, {k for k in c.keys() | p.keys()
                             if c.get(k, _MISSING) != p.get(k, _MISSING)}


def _one_step(cfg, tc, prefix_k: int):
    from morph.training.train import build_morph_config
    mc = build_morph_config(cfg, tul=tc)
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=tc, d_ff=_D_FF,
                               core_fixed_point_lambda=mc.core_fixed_point_lambda)).train()
    m = m.float()
    ids = np.random.default_rng(0).integers(5, 64, size=(2, 160)).astype(np.int64)
    ids[:, ::8] = 10
    inp, lab, layout, _ = slot_layout_from_ids(ids, _rule(),
                                               TulLayoutSpec(seq_len=64, prefix_k=prefix_k,
                                                             max_slots=10, slot_id=4))
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.tul.parameters())
    return mc


def test_arm_b_reach1_differs_from_fp01_by_the_stated_key(monkeypatch):
    cfg, rt, prt, c, diff = _diff("tul_slot_spandec_strict_e4probe_fp01_reach1",
                                  "tul_slot_spandec_strict_e4probe_fp01", monkeypatch)
    assert diff == {"tul.tg_coda_token_reach", "wandb.name"}, sorted(diff)
    assert rt.model_cfg.tg_coda_token_reach == 1 and prt.model_cfg.tg_coda_token_reach == 0
    assert rt.manifest["tg_coda_token_reach"] == 1
    mc = _one_step(cfg, rt.model_cfg, prefix_k=2)
    assert mc.core_fixed_point_lambda == 0.1 and mc.n_coda == 4


def test_arm_a1_pk4_differs_from_fp01_by_the_stated_key(monkeypatch):
    cfg, rt, _prt, c, diff = _diff("tul_slot_spandec_strict_e4probe_fp01_pk4",
                                   "tul_slot_spandec_strict_e4probe_fp01", monkeypatch)
    assert diff == {"tul.prefix_k", "wandb.name"}, sorted(diff)
    tc = rt.model_cfg
    assert tc.prefix_k == 4 and tc.code_enum_k == 4 and tc.spandec_parallel
    spec = TulLayoutSpec(seq_len=int(cfg.data.seq_len), prefix_k=tc.prefix_k,
                         max_slots=int(cfg.tul.max_slots))
    assert spec.l_total == 1024 + 4 * 64 == 1280
    _one_step(cfg, tc, prefix_k=4)


def test_arm_a2_fan4_all_fp01_differs_from_fan4_all_by_the_fp_term(monkeypatch):
    cfg, rt, _prt, c, diff = _diff("tul_slot_spandec_strict_fan4_all_fp01",
                                   "tul_slot_spandec_strict_fan4_all", monkeypatch)
    assert diff == {"model.core_fixed_point_lambda", "wandb.name"}, sorted(diff)
    tc = rt.model_cfg
    assert tc.fan_k == 4 and tc.fan_mix == "all" and tc.prefix_k == 4
    mc = _one_step(cfg, tc, prefix_k=4)
    assert mc.core_fixed_point_lambda == 0.1


def test_the_fan_is_refused_on_fp01_which_is_why_a2_is_the_fan_recipe(monkeypatch):
    _cfg, rt = _runtime("tul_slot_spandec_strict_e4probe_fp01", monkeypatch)
    tc = rt.model_cfg
    with pytest.raises(NotImplementedError, match="fan"):
        dataclasses.replace(tc, fan_k=4, slot_cells=4, prefix_k=4, fan_mix="all")
    # and each of fp01's two LXTUL-E features refuses it on its own
    with pytest.raises(NotImplementedError, match="code_enum_k > 1 with tul.fan_k"):
        dataclasses.replace(tc, spandec_parallel=False, spandec_parallel_detach=False,
                            fan_k=4, slot_cells=4, prefix_k=4, fan_mix="all")
    with pytest.raises(NotImplementedError, match="spandec_parallel with tul.fan_k"):
        dataclasses.replace(tc, code_enum_k=1, fan_k=4, slot_cells=4, prefix_k=4,
                            fan_mix="all")
