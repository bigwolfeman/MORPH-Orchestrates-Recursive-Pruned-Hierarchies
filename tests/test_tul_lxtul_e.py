"""LXTUL-E Stage 1: the enumerated loop code (`tul.code_enum_k`) and the parallel head
in place of the teacher-forced span decoder.

Files: morph/model/tul_code_enum.py (the codes and the per-pass term),
morph/model/rollout_mixture.py (the ONE copy of the mixture math),
morph/model/transformer.py (`_tul_core`'s re-injection, `_enum_mix_losses`,
`_enum_mixture_logprobs`, `_tul_spandec_par_loss(n_rollouts=)`),
morph/configs/tul_slot_spandec_strict_e{1,4}.yaml.
Note: .agents/notes/proposed/architecture/2026-09-23-provable-loop-contribution.md

What each test pins:
  * e1 is the ruler with ONE module swapped (the decoder out, the head in) and nothing
    else: weights, logits, every coda CE key and the coda's train-mode output at nonzero
    dropout are bit-identical;
  * at K = 4 the code is re-added at the end of EVERY pass, after f, sized by rms(f(h)):
    a linear toy core's exit equals a hand-rolled recurrence, and dropping the code after
    pass 1 changes the exit;
  * forced depth (the `slot_depths` table and `slot_mean_depth`) reaches the term;
  * the codes are mean-free, unit-RMS and equidistant with no learnable scale;
  * the K rollouts share every dropout mask and the depth draw at nonzero dropout;
  * identical codes give the K = 1 loss;
  * the prelude runs once, on the base batch;
  * the coda's loss is the brute-force per-span logsumexp, the deploy read's CE equals it,
    and the parallel head reads each rollout under the exact mixture;
  * the configs compose and build; the refusals hold.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from morph.model.rollout_mixture import evidence_labels, log_mean_exp
from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_code_enum import TULCodeEnum, helmert_simplex
from morph.model.tul_spandec_parallel import mixture_span_nll
from test_tul_strict_geometry import _pack, _runtime, _tiny, _tul

_D_FF = 96
K = 4


def _tc(k: int = K, spandec: bool = False, parallel: bool = True, **kw) -> TULConfig:
    base = dict(tg_geometry="strict", spandec=spandec, spandec_parallel=parallel,
                code_enum_k=k, plast_weight=1.0)
    base.update(kw)
    return _tul(**base)


def _model(k: int = K, seed: int = 1234, dropout: float = 0.0, spandec: bool = False,
           parallel: bool = True, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tc(k, spandec, parallel, **kw), d_ff=_D_FF,
                               dropout=dropout))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.float()


class _Spy:
    """Wrap a bound method, record its args and outputs, call through."""

    def __init__(self, obj, name):
        self.calls: list[tuple] = []
        self.outs: list = []
        fn = getattr(obj, name)

        def _w(*a, **k):
            r = fn(*a, **k)
            self.calls.append((a, k))
            self.outs.append(r)
            return r
        setattr(obj, name, _w)


def _table(layout, d: int) -> torch.Tensor:
    return torch.full(layout.slot_index.shape, d, dtype=torch.long)


# ── e1 is the ruler with the head in place of the decoder ───────────────────────────


def test_e1_is_the_ruler_with_only_the_decoder_swapped_for_the_head():
    """WHAT DIFFERS, exactly: the module `tul_spandec` (absent on e1) and
    `tul_spandec_par` (absent on the ruler), and the loss term `spandec` vs `par`. Every
    shared weight, the logits, every coda CE key, and the coda's train-mode output at
    dropout 0.1 are bit-identical."""
    _ids, inp, lab, layout = _pack()
    r = _model(k=1, spandec=True, parallel=False)
    e = _model(k=1, spandec=False, parallel=True)
    assert e.tul_code_enum is None and e._code_enum_k == 0
    sr, se = r.state_dict(), e.state_dict()
    only_r = {k.split(".")[0] for k in set(sr) - set(se)}
    only_e = {k.split(".")[0] for k in set(se) - set(sr)}
    assert only_r == {"tul_spandec"} and only_e == {"tul_spandec_par"}
    for k in set(sr) & set(se):
        assert torch.equal(sr[k], se[k]), k
    r.eval()
    e.eval()
    with torch.no_grad():
        assert torch.equal(r(inp, slot_layout=layout)["logits"],
                           e(inp, slot_layout=layout)["logits"])
        orr = r(inp, labels=lab, slot_layout=layout)
        oe = e(inp, labels=lab, slot_layout=layout)
    for k in ("ce_tokens", "ce_main", "ce_plast", "ce_emit", "n_main", "n_plast",
              "n_targets"):
        assert torch.equal(torch.as_tensor(orr[k]), torch.as_tensor(oe[k])), k
    assert "spandec" in orr and "par" not in orr and "par" in oe and "spandec" not in oe
    torch.testing.assert_close(orr["loss"] - orr["spandec_weighted"],
                               oe["loss"] - oe["par_weighted"], rtol=0, atol=1e-6)
    # train mode, NONZERO dropout: the coda's readout is bit-identical
    xs = []
    for m in (_model(k=1, spandec=True, parallel=False, dropout=0.1),
              _model(k=1, spandec=False, parallel=True, dropout=0.1)):
        m.train()
        spy = _Spy(m, "_tul_group_losses")
        torch.manual_seed(5)
        m(inp, labels=lab, slot_layout=layout)
        xs.append(spy.calls[0][0][0])
    assert torch.equal(xs[0], xs[1])


# ── the code reaches every pass ─────────────────────────────────────────────────────


def test_linear_toy_core_exit_is_the_hand_rolled_recurrence_every_pass():
    """Core map replaced by h -> lam * h. The model's exit must equal
    h_{t+1} = f + r * rms(f) * u_k, f = lam * h_t, run for the forced depth from the
    captured entry state, per rollout. Then the same with the code dropped after pass 1:
    the exit changes, and the per-pass separation grows as the recurrence says."""
    _ids, inp, lab, layout = _pack()
    lam, D = 0.9, 3
    m = _model().eval()
    m._apply_core_step = lambda h_in, *a, **k: (lam * h_in, None)
    m._jac_capture = []
    core = _Spy(m, "_tul_core")
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout, slot_depths=_table(layout, D))
    h_exit = core.outs[0][1]                                        # [K*B, S, n, C]
    h = m._jac_capture[0]["h"]                                      # the entry, pass 0
    B0 = h.shape[0] // K
    assert torch.equal(h[:B0], h[B0:2 * B0])                        # the entry has no code
    enum = m.tul_code_enum
    u = enum.directions().detach()
    valid = layout.slot_valid.repeat(K, 1)
    code = u.repeat_interleave(B0, 0)[:, None, None, :]               # [K*B, 1, 1, C]
    ref = h.clone()
    seps = []
    for _t in range(D):
        f = lam * ref
        rms = f.float().flatten(2).pow(2).mean(-1).sqrt()[..., None, None]
        ref = f + enum.ratio * rms * code * valid[..., None, None]
        d = (ref[:B0] - ref[B0:2 * B0])[layout.slot_valid]
        seps.append(float(d.pow(2).mean().sqrt()))
    v = valid
    torch.testing.assert_close(h_exit[v], ref[v], rtol=1e-5, atol=1e-6)
    assert all(b > a for a, b in zip(seps, seps[1:])), seps
    # the code only at pass 1: the exit moves
    m2 = _model().eval()
    m2._apply_core_step = lambda h_in, *a, **k: (lam * h_in, None)
    n = {"c": 0}
    real = m2.tul_code_enum.term

    def _once(h_, valid_, n_r):
        n["c"] += 1
        t_ = real(h_, valid_, n_r)
        return t_ if n["c"] == 1 else torch.zeros_like(t_)
    m2.tul_code_enum.term = _once
    core2 = _Spy(m2, "_tul_core")
    with torch.no_grad():
        m2(inp, labels=lab, slot_layout=layout, slot_depths=_table(layout, D))
    assert n["c"] == D
    gap = (core2.outs[0][1][v] - h_exit[v]).abs().max()
    assert gap > 1e-3, float(gap)


def test_the_code_reaches_every_pass_of_the_real_core_and_rollouts_differ_only_after():
    """On the real (tiny) core: at every pass's START the K rollouts' states are equal
    at pass 1 and different from pass 2 on (the code of the previous pass is inside),
    and the term is called once per pass."""
    _ids, inp, lab, layout = _pack()
    m = _model().eval()
    m._jac_capture = []
    spy = _Spy(m.tul_code_enum, "term")
    D = 3
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout, slot_depths=_table(layout, D))
    assert len(spy.calls) == D
    B0 = layout.slot_valid.shape[0]
    v = layout.slot_valid
    for t, cap in enumerate(m._jac_capture):
        h = cap["h"]
        diff = (h[:B0] - h[B0:2 * B0])[v].abs().max()
        if t == 0:
            assert diff == 0
        else:
            assert diff > 1e-4, (t, float(diff))


def test_forced_depth_reaches_the_reinjection():
    """Both depth dials an instrument uses: the `slot_depths` table (per slot) and
    `tul.slot_mean_depth` (core_depth_sweep.py). The term runs once per pass, labelled or
    not."""
    _ids, inp, lab, layout = _pack()
    m = _model().eval()
    spy = _Spy(m.tul_code_enum, "term")
    with torch.no_grad():
        for d in (1, 2, 3):
            spy.calls.clear()
            m(inp, labels=lab, slot_layout=layout, slot_depths=_table(layout, d))
            assert len(spy.calls) == d
            spy.calls.clear()
            m.cfg.tul.slot_mean_depth = d
            m(inp, slot_layout=layout)
            assert len(spy.calls) == d
        m.cfg.tul.slot_mean_depth = 0


# ── the codes ────────────────────────────────────────────────────────────────────────


def test_codes_are_mean_free_unit_rms_equidistant_and_carry_no_scale():
    enum = TULCodeEnum(64, K, 0.1)
    assert [n for n, _ in enum.named_parameters()] == ["basis"]
    assert list(enum.state_dict()) == ["basis"]
    for scale in (1.0, 7.3, 1e-3):
        with torch.no_grad():
            enum.basis.mul_(scale)
        u = enum.directions()
        torch.testing.assert_close(u.sum(0), torch.zeros(64), rtol=0, atol=1e-5)
        torch.testing.assert_close(u.pow(2).mean(1).sqrt(), torch.ones(K), rtol=0,
                                   atol=1e-5)
        dd = torch.cdist(u, u)[~torch.eye(K, dtype=torch.bool)]
        torch.testing.assert_close(dd, torch.full_like(dd, math.sqrt(2 * 64 * K / (K - 1))),
                                   rtol=1e-5, atol=0)
    u0 = enum.directions()
    with torch.no_grad():
        enum.basis.mul_(5.0)
    torch.testing.assert_close(enum.directions(), u0, rtol=0, atol=1e-5)
    # a gradient step moves the orientation and keeps every property
    enum.basis.grad = torch.randn_like(enum.basis)
    torch.optim.SGD(enum.parameters(), lr=0.5).step()
    u1 = enum.directions()
    assert (u1 - u0).abs().max() > 1e-3
    torch.testing.assert_close(u1.sum(0), torch.zeros(64), rtol=0, atol=1e-5)
    s = helmert_simplex(K)
    torch.testing.assert_close(s.sum(0), torch.zeros(K - 1, dtype=torch.float64))


def test_the_term_sizes_by_a_detached_rms_and_zeroes_pads():
    enum = TULCodeEnum(16, K, 0.1)
    h = torch.randn(K * 2, 3, 4, 16, requires_grad=True)
    valid = torch.tensor([[True, True, False]]).repeat(K * 2, 1)
    t = enum.term(h, valid, K)
    (g,) = torch.autograd.grad(t.sum(), [h], allow_unused=True)
    assert g is None, "the RMS must be detached: the code's size is not a function of h"
    assert torch.equal(t[:, 2], torch.zeros_like(t[:, 2]))
    rms = h.detach().flatten(2).pow(2).mean(-1).sqrt()
    torch.testing.assert_close(t.pow(2).mean(-1).sqrt()[:, :2], 0.1 * rms[:, :2])
    with pytest.raises(RuntimeError, match="rollout batch"):
        enum.term(h, valid, 1)


def test_the_code_basis_takes_no_weight_decay():
    from morph.training.optimizer import _split_by_decay
    _d, _nd, dn, ndn = _split_by_decay(_model())
    assert "tul_code_enum.basis" in ndn and "tul_code_enum.basis" not in dn


# ── rollouts differ in the code alone ───────────────────────────────────────────────


def test_rollouts_share_every_dropout_mask_and_the_depth_draw():
    """dropout 0.1, token-state dropout 0.15, TRAIN mode, Poisson depth, codes zeroed:
    the K rollouts must then be the SAME computation, bit for bit."""
    _ids, inp, lab, layout = _pack()
    m = _model(dropout=0.1, token_state_dropout=0.15).train()
    assert m._n_rollout_dropout > 0
    m.tul_code_enum.directions = lambda: torch.zeros(K, m.cfg.d_model)
    core = _Spy(m, "_tul_core")
    coda = _Spy(m, "_enum_mix_losses")
    torch.manual_seed(3)
    out = m(inp, labels=lab, slot_layout=layout)
    B0 = inp.shape[0]
    h_slots, depths = core.outs[0][1], core.outs[0][2]
    xh = coda.calls[0][0][0]
    for k in range(1, K):
        assert torch.equal(depths[:B0], depths[k * B0:(k + 1) * B0])
        assert torch.equal(h_slots[:B0], h_slots[k * B0:(k + 1) * B0]), k
        assert torch.equal(xh[:B0], xh[k * B0:(k + 1) * B0]), k
    assert float(out["enum_width_gain"]) == pytest.approx(0.0, abs=1e-6)


def test_identical_codes_give_the_k1_loss():
    _ids, inp, lab, layout = _pack()
    e4 = _model()
    e1 = _model(k=1)
    missing, unexpected = e1.load_state_dict(e4.state_dict(), strict=False)
    assert missing == [] and unexpected == ["tul_code_enum.basis"]
    for k, v in e1.state_dict().items():
        assert torch.equal(v, e4.state_dict()[k]), k        # RNG-neutral build
    e4.tul_code_enum.directions = lambda: torch.zeros(K, e4.cfg.d_model)
    for mode in ("eval", "train"):
        getattr(e4, mode)()
        getattr(e1, mode)()
        torch.manual_seed(11)
        o4 = e4(inp, labels=lab, slot_layout=layout)
        torch.manual_seed(11)
        o1 = e1(inp, labels=lab, slot_layout=layout)
        torch.testing.assert_close(o4["loss"], o1["loss"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(o4["par_ce"], o1["par_ce"], rtol=1e-5, atol=1e-5)
        if mode == "eval":
            torch.testing.assert_close(o4["ce_tokens"], o1["ce_tokens"], rtol=1e-5,
                                       atol=1e-5)


def test_the_prelude_runs_once_on_the_base_batch():
    _ids, inp, lab, layout = _pack()
    m = _model().train()
    seen: dict[str, list[int]] = {"prelude": [], "core": [], "coda": []}
    for name in seen:
        for blk in getattr(m, name):
            blk.register_forward_hook(
                lambda mod, a, o, _n=name: seen[_n].append(int(a[0].shape[0])))
    m(inp, labels=lab, slot_layout=layout)
    B0 = inp.shape[0]
    assert seen["prelude"] == [B0] * len(m.prelude)
    assert seen["core"] and set(seen["core"]) == {K * B0}
    assert seen["coda"] == [K * B0] * len(m.coda)


# ── the readers ──────────────────────────────────────────────────────────────────────


def _brute_rollout_lp(m, xh, lab):
    """[K, B, L] log p of each label under each rollout, from full logits."""
    B0, L = lab.shape
    logits = xh.float() @ m.embed.lm_weight().float().T
    logits[..., m.cfg.tul.slot_id] = float("-inf")
    lp = torch.log_softmax(logits, -1).gather(
        -1, lab.clamp(min=0).repeat(K, 1).unsqueeze(-1)).squeeze(-1)
    return lp.view(K, B0, L)


def test_coda_mixture_is_the_brute_force_per_span_logsumexp_and_the_deploy_read():
    _ids, inp, lab, layout = _pack()
    m = _model().eval()
    coda = _Spy(m, "_enum_mix_losses")
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
        logits = m(inp, slot_layout=layout)["logits"]
    xh = coda.calls[0][0][0]
    lp = _brute_rollout_lp(m, xh, lab)
    row_w, _p, _z = m._tul_half_weights(lab, layout)
    w = (row_w.view_as(lab) * (lab != -100)).float()
    num = 0.0
    for b in range(lab.shape[0]):
        for g in layout.bag_id[b].unique().tolist():
            sel = (layout.bag_id[b] == g) & (w[b] > 0)
            if sel.any():
                S = (lp[:, b][:, sel] * w[b][sel]).sum(-1)                   # [K]
                num = num + (torch.logsumexp(S, 0) - math.log(K))
    brute = -num / w.sum()
    torch.testing.assert_close(out["enum_ce_mix"], brute, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out["ce_tokens"], out["enum_ce_mix"], rtol=1e-5, atol=1e-5)
    # the deploy read: a proper distribution per position whose CE is that mixture
    assert logits.shape == (lab.shape[0], lab.shape[1], m.cfg.vocab_size)
    torch.testing.assert_close(torch.logsumexp(logits, -1),
                               torch.zeros(lab.shape), rtol=0, atol=1e-5)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.clamp(min=0).reshape(-1),
                         reduction="none").view_as(lab)
    torch.testing.assert_close((ce * w).sum() / w.sum(), brute, rtol=1e-5, atol=1e-5)
    # and it is not the mean of the rollouts' CEs (the mixture is not a mean)
    mean_ce = -(lp * w).sum((1, 2)).mean() / w.sum()
    assert float(mean_ce.detach() - brute.detach()) > 1e-6


def test_parallel_head_reads_each_rollout_exit_under_the_exact_mixture():
    _ids, inp, lab, layout = _pack()
    m = _model().eval()
    assert m.tul_spandec_par.codes is None                     # no code at the head input
    spy = _Spy(m, "_tul_spandec_par_loss")
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
        h_slots = spy.calls[0][0][0]
        head = m.tul_spandec_par
        B0 = inp.shape[0]
        z = m._readout(h_slots).view(K, B0, *h_slots.shape[1:2], -1)
        ids, valid = head.targets(inp, layout)
        w = m.embed.lm_weight().detach()
        per = [head.rollout_logp(z[k:k + 1], ids, valid, w, mask_token_id=m.cfg.tul.slot_id)
               for k in range(K)]
        lps = torch.cat([p[0] for p in per], 0)                      # [K, M]
        want = mixture_span_nll(lps, per[0][1])
        mean_prod = -(lps.mean(0)).sum() / per[0][1]
    torch.testing.assert_close(out["par_ce"], want, rtol=1e-5, atol=1e-5)
    assert float(out["par_k"]) == K
    assert float(mean_prod - want) > 1e-6


def test_evidence_labels_are_the_packers_labels_and_never_a_span_end():
    _ids, inp, lab, layout = _pack(B=3, seed=4)
    ev, mask = evidence_labels(inp, layout)
    assert torch.equal(ev[mask], lab[mask])
    tok = ~layout.slot_mask
    assert not (mask & ~tok).any()
    # a span's last token (its label is the NEXT span's first token) is never evidence
    _w, p_idx, _z = MORPHTransformer._tul_half_weights(_model(), lab, layout)
    flat = mask.reshape(-1)
    real = p_idx[p_idx < flat.numel()]
    assert not flat[real].any()


def test_log_mean_exp_is_the_one_mixture_primitive():
    S = torch.randn(4, 7)
    torch.testing.assert_close(log_mean_exp(S), torch.logsumexp(S, 0) - math.log(4))
    one = torch.randn(1, 5)
    assert torch.equal(log_mean_exp(one), one[0])


# ── configs and refusals ────────────────────────────────────────────────────────────


def test_the_stage1_configs_compose_build_and_differ_by_one_factor(monkeypatch):
    from omegaconf import OmegaConf
    cfgs = {}
    for name, k, wb in (("tul_slot_spandec_strict", None, "slot-spandec-strict"),
                        ("tul_slot_spandec_strict_e1", 1, "lxtul-e1"),
                        ("tul_slot_spandec_strict_e4", 4, "lxtul-e4")):
        cfg, rt = _runtime(name, monkeypatch)
        tc = rt.model_cfg
        assert cfg.wandb.name == wb and tc.tg_geometry == "strict"
        tr = cfg.training
        assert tr.get("init_from") in (None, "") and not tr.get("frozen_eval", False)
        assert int(tr.steps) == 5000 and int(tr.batch_size) == 6 and int(tr.seed) == 1
        if k is not None:
            assert not tc.spandec and tc.spandec_parallel and tc.spandec_parallel_k == 1
            assert tc.code_enum_k == k and tc.code_enum_ratio == 0.1
            torch.manual_seed(7)
            m = MORPHTransformer(_tiny(tul=tc, d_ff=_D_FF)).eval().float()
            assert (m.tul_code_enum is None) == (k == 1)
            _ids0, inp, lab, layout = _pack()
            with torch.no_grad():
                out = m(inp, labels=lab, slot_layout=layout)
            assert torch.isfinite(out["par_ce"]) and torch.isfinite(out["loss"])
        cfgs[wb] = OmegaConf.to_container(cfg, resolve=True)
    r, e1, e4 = cfgs["slot-spandec-strict"], cfgs["lxtul-e1"], cfgs["lxtul-e4"]
    for c in (r, e1, e4):
        c["wandb"].pop("name")
    assert e4["tul"].pop("code_enum_k") == 4
    assert e4 == e1, "e4 differs from e1 in something other than K and the name"
    assert e1["tul"].pop("spandec") is False and r["tul"].pop("spandec") is True
    assert e1["tul"].pop("spandec_parallel") is True
    r["tul"].pop("spandec_parallel", None)
    assert e1 == r, "e1 differs from the ruler in something other than the reader"


@pytest.mark.parametrize("kw,exc,match", [
    (dict(code_enum_k=1, code_enum_ratio=0.2), ValueError, "code_enum_ratio"),
    (dict(code_enum_k=0), ValueError, "code_enum_k"),
    (dict(code_enum_ratio=0.0), ValueError, "code_enum_ratio"),
    (dict(gram=True, gram_objective="iw", gram_beta=0.1), NotImplementedError, "gram"),
    (dict(fan_k=4, slot_cells=4), (ValueError, NotImplementedError), None),
    (dict(spandec_parallel_k=4), NotImplementedError, "spandec_parallel_k"),
    (dict(plast_weight=0.5), NotImplementedError, "plast_weight"),
    (dict(emit_weight=0.5), NotImplementedError, "emit_weight"),
])
def test_refusals(kw, exc, match):
    with pytest.raises(exc, match=match):
        _tc(**kw)


def test_reads_cells_is_refused():
    with pytest.raises((ValueError, NotImplementedError)):
        _tc(spandec=True, parallel=False, spandec_reads_cells=True, slot_cells=4)


def test_a_base_batch_loop_call_raises():
    _ids, inp, lab, layout = _pack()
    m = _model().eval()
    with torch.no_grad():
        _fk, _fr, _ck, _cr = m._tul_tg_kwargs(layout)
        x, x0, bg = m._tul_front(inp, layout, attn_kwargs=_fk, ret_reset_mask=_fr)
        with pytest.raises(RuntimeError, match="rollout batch"):
            m._tul_core(x, x0, bg, layout, input_ids=inp)


@pytest.mark.parametrize("mode", ["zero", "all_slots"])
def test_zero_plan_makes_every_rollout_the_same_read(mode):
    """With the cells zeroed the code has no route into the coda: every rollout's coda
    output is bit-identical (a code leaking into the coda or the prelude would break it)."""
    _ids, inp, lab, layout = _pack()
    m = _model().eval()
    coda = _Spy(m, "_enum_mix_losses")
    with torch.no_grad():
        out = m.tul_forward_ablated(inp, lab, layout, plan_mode=mode)
    xh = coda.calls[0][0][0]
    B0 = inp.shape[0]
    for k in range(1, K):
        assert torch.equal(xh[:B0], xh[k * B0:(k + 1) * B0]), k
    assert float(out["enum_width_gain"]) == pytest.approx(0.0, abs=1e-6)


def test_shuffle_ablation_permutes_every_rollout_of_a_row_alike():
    """The trainer's eval ablation (`eval_ablations`) runs `shuffle`. With the codes
    zeroed, a shared permutation leaves the K rollouts identical; a per-row draw would not."""
    _ids, inp, lab, layout = _pack(B=3, seed=2)
    m = _model().eval()
    m.tul_code_enum.directions = lambda: torch.zeros(K, m.cfg.d_model)
    torch.manual_seed(0)
    with torch.no_grad():
        normal = m(inp, labels=lab, slot_layout=layout)
        coda = _Spy(m, "_enum_mix_losses")
        out = m.tul_forward_ablated(inp, lab, layout, plan_mode="shuffle")
    xh = coda.calls[0][0][0]
    B0 = inp.shape[0]
    for k in range(1, K):
        assert torch.equal(xh[:B0], xh[k * B0:(k + 1) * B0]), k
    assert float(out["ce_tokens"]) != float(normal["ce_tokens"])     # the shuffle acted


def test_the_prelude_output_is_the_k1_models():
    """The loop's input `x` (the prelude output) of e4 equals e1's at shared weights: no
    code reaches the front."""
    _ids, inp, lab, layout = _pack()
    e4, e1 = _model().eval(), _model(k=1).eval()
    e1.load_state_dict(e4.state_dict(), strict=False)
    xs = []
    for m in (e4, e1):
        spy = _Spy(m, "_tul_core")
        with torch.no_grad():
            m(inp, labels=lab, slot_layout=layout)
        xs.append(spy.calls[0][0][0])
    assert torch.equal(xs[0], xs[1])


def test_np_and_torch_rng_untouched_by_the_code_build():
    st_np = np.random.get_state()[1].copy()
    torch.manual_seed(0)
    a = torch.rand(3)
    torch.manual_seed(0)
    TULCodeEnum(64, K, 0.1)
    b = torch.rand(3)
    assert torch.equal(a, b)
    assert np.array_equal(np.random.get_state()[1], st_np)


def test_the_trainers_eval_probes_run_on_the_rollout_batch():
    """`eval_ablations` (on in the ruler recipe) runs `tul_slot_state_probe` and
    `tul_attn_lift_probe` at every val pass. The first reads rollout 0's states (one code,
    comparable with e1); the second averages the coda's lift over every rollout."""
    _ids, inp, lab, layout = _pack()
    m = _model().eval()
    with torch.no_grad():
        st = m.tul_slot_state_probe(inp, layout)
        lift = m.tul_attn_lift_probe(inp, layout)
    assert np.isfinite(st["slot_eff_rank"]) and st["slot_eff_rank"] > 0
    assert lift and any(np.isfinite(v) for v in lift.values())
