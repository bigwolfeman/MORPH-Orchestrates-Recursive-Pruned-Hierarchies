"""LX-Fan: LXTUL-E's enumerated loop code (`tul.code_enum_k`) on top of the write-all fan
(`tul.fan_k`, `fan_mix: all`), 2026-09-26.

Files: morph/model/tul.py (`_check_code_enum`, `_check_spandec_parallel`, the fan's eps
rule), morph/model/transformer.py (`_tul_core`'s register on the base rows,
`_tul_fan_all` at WTA 0, the epi ridge under the code, `_tul_fan_oracle(n_rollouts=)`,
`_enum_exit_stats` on the cells), morph/inference/tul_generate_cached.py (the refusal),
morph/configs/tul_slot_spandec_strict_lxfan{4,6}_fp01.yaml.
Note: .agents/notes/proposed/architecture/2026-09-26-lx-fan.md

What each test pins:
  * OFF IS THE TREE, both ways. The fan with no code (a2's fan: WTA 1.0, epivol, M = 4
    and 6) and LX with no fan (fp01's LX, probe head detached and live) give the loss,
    the eval logits, the eval loss and every gradient measured on 36a9823, the tree
    before this change (exact float equality of the loss and of the sums; the full
    tensors were compared `torch.equal` on the same tree by
    /home/wolfe/morph-scratch/lxfan/pin_compare.py, 2026-09-26).
  * THE CODE IS THE ONLY DIFFERENCE: with the codes zeroed LX-Fan is its fan partner (the
    same fan at WTA 0 and no code), loss and oracle, train and eval.
  * ROLLOUT k's CODE REACHES EVERY CELL AT EVERY PASS: on a linear toy core the cell-axis
    exit is the hand-rolled per-CELL recurrence h <- f + r * rms_cell(f) * u_k; on the
    real core every valid cell of every cell index separates from pass 2 on and not at
    the entry.
  * THE MIXTURE IS EXACT: the token loss is the brute-force per-span logsumexp over the
    K rollouts computed from the coda's own output, and the deploy read's CE equals it;
    the fan oracle's `mixed_ce` is the same per-span mixture on its spans.
  * THE REGISTER RUNS ONCE on the base rows; WTA 0 runs NO extra coda pass; the epi
    ridge under the code is the base fit on the rollout-mean deviations.
  * THE EAGER GENERATOR runs on LX-Fan at M = 4 and 6 and repeats under a seed.
  * REFUSALS: gone for write-all + WTA 0 + no probe head only; every other fan mode, WTA
    > 0, seed noise, the bare register and the probe head stay refused; the cached
    decoder refuses (tests/test_tul_generate_cached.py).
  * THE CONFIGS compose, differ from their parents by the stated keys, build and train a
    step on the tiny model.

CPU, fp32, the `tests/test_tul_fan.py` fixtures (strict geometry, dropout 0).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import morph.model.transformer as transformer_mod
from morph.model.transformer import MORPHTransformer
from morph.model.tul_fan import epi_score, ridge_map
from morph.model.tul_layout import TulLayoutSpec, slot_layout_from_ids
from test_tul_fan import _batch, _rule, _tiny, _tul

_D_FF = 96
K = 4

# fp01's loop constraint (fixed-point term 0.1, row hinge 0.98, tail hinge 100 @ 1.1): the
# pins run the shared paths (the hinge's two `_core_step` calls, the terminal term) on the
# K-fold batch as well as on the base one.
_FP01 = dict(core_fixed_point_lambda=0.1, slot_gain_lambda=100.0, slot_gain_target=0.98,
             slot_gain_tail_lambda=100.0, slot_gain_tail_target=1.1)


def _build(seed: int = 1234, model_kw: dict | None = None, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**tul_kw), d_ff=_D_FF, **{**_FP01, **(model_kw or {})}))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.float()


def _fan_kw(m: int = 4, **kw) -> dict:
    """a2's fan: write-all, WTA 1.0 at eps 0.05, epivol 0.1 over 2 passes."""
    base = dict(fan_k=m, slot_cells=m, prefix_k=m, fan_mix="all", fan_select_eps=0.05,
                fan_all_wta_lambda=1.0, fan_repel_lambda=0.1, fan_repel_mode="epivol",
                plast_weight=1.0)
    base.update(kw)
    return base


def _lx_kw(**kw) -> dict:
    """fp01's LX: K = 4 codes, the detached parallel head, no teacher-forced decoder."""
    base = dict(prefix_k=2, code_enum_k=K, spandec=False, spandec_parallel=True,
                spandec_parallel_detach=True, plast_weight=1.0)
    base.update(kw)
    return base


def _lxfan_kw(m: int = 4, **kw) -> dict:
    """The LX-Fan configs' tul block on the tiny model: the fan's write-all read and its
    epivol term, no WTA, no parallel head, K = 4 codes."""
    base = _fan_kw(m, fan_all_wta_lambda=0.0, code_enum_k=K)
    base.update(kw)
    return base


def _pin_run(m: MORPHTransformer, prefix_k: int):
    """``(train loss, eval logits, eval labelled loss, grads)`` — one seeded train step's
    loss and every gradient, then the label-free and labelled eval forwards."""
    _ids, inp, lab, lay = _batch(prefix_k)
    m.train()
    torch.manual_seed(99)
    o = m(inp, labels=lab, slot_layout=lay)
    o["loss"].backward()
    grads = {k: p.grad.detach().clone() for k, p in m.named_parameters()
             if p.grad is not None}
    m.eval()
    with torch.no_grad():
        lg = m(inp, labels=None, slot_layout=lay)["logits"]
        ev = m(inp, labels=lab, slot_layout=lay)["loss"]
    return o["loss"].detach(), lg, ev.detach(), grads


def _pin_scalars(res) -> tuple[float, float, float, float, int]:
    loss, lg, ev, grads = res
    ls = float(torch.nan_to_num(lg.double(), nan=0.0, posinf=0.0, neginf=0.0).sum())
    gs = float(sum(g.double().abs().sum() for g in grads.values()))
    return float(loss), ls, float(ev), gs, len(grads)


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


# ── 1. OFF IS THE TREE ──────────────────────────────────────────────────────────────

# Measured 2026-09-26 on the UNMODIFIED worktree at 36a9823 (the tree before LX-Fan) with
# these fixtures and one CPU thread: (train loss, eval logit sum, eval loss, sum |grad|,
# n grads). /home/wolfe/morph-scratch/lxfan/pin_head.py wrote them; pin_compare.py then
# found every tensor `torch.equal` on the changed tree.
HEAD_PINS = {
    "fan4": (10.31538200378418, 1358.1454057991505, 5.132460117340088, 1436.3044085161928, 179),
    "fan6": (10.335302352905273, 1519.400069154045, 5.132365703582764, 1426.90597463816, 179),
    "lx": (9.606481552124023, -52828.743996977806, 9.570661544799805, 1201.495515583244, 190),
    "lx_live": (9.606481552124023, -52828.743996977806, 9.570661544799805,
                1523.0509139084759, 190),
}
_PIN_CASES = {
    "fan4": (lambda: _fan_kw(4), 4),
    "fan6": (lambda: _fan_kw(6), 6),
    "lx": (lambda: _lx_kw(), 2),
    "lx_live": (lambda: _lx_kw(spandec_parallel_detach=False), 2),
}


@pytest.mark.parametrize("name", sorted(HEAD_PINS))
def test_without_the_other_half_each_forward_is_the_tree(name):
    kw, pk = _PIN_CASES[name]
    got = _pin_scalars(_pin_run(_build(**kw()), pk))
    assert got == HEAD_PINS[name], f"{name}: moved off the 36a9823 pin: {got}"


def test_identical_codes_give_the_fan_partner():
    """LX-Fan with every code zeroed is its fan partner (the same fan, WTA 0, no code):
    the K rollouts are then one computation, the mixture of K equal atoms is that atom,
    and the register computed once and tiled equals the register on the full batch.
    The gain hinge is OFF here: it draws one random direction per ROW, so on the K-fold
    batch it draws K times as many (LX's own behaviour) and would differ by the draw."""
    _ids, inp, lab, layout = _batch(4)
    nohinge = dict(slot_gain_lambda=0.0, slot_gain_tail_lambda=0.0)
    lxf = _build(model_kw=nohinge, **_lxfan_kw(4))
    fan = _build(model_kw=nohinge, **_fan_kw(4, fan_all_wta_lambda=0.0))
    missing, unexpected = fan.load_state_dict(lxf.state_dict(), strict=False)
    assert missing == [] and unexpected == ["tul_code_enum.basis"]
    for k, v in fan.state_dict().items():
        assert torch.equal(v, lxf.state_dict()[k]), k        # RNG-neutral build
    lxf.tul_code_enum.directions = lambda: torch.zeros(K, lxf.cfg.d_model)
    for mode in ("train", "eval"):
        getattr(lxf, mode)()
        getattr(fan, mode)()
        torch.manual_seed(11)
        a = lxf(inp, labels=lab, slot_layout=layout)
        torch.manual_seed(11)
        b = fan(inp, labels=lab, slot_layout=layout)
        torch.testing.assert_close(a["loss"], b["loss"], rtol=1e-5, atol=1e-5)
        for key in (("fan_repel",) if mode == "train" else ()) + ("fan_vol_t1", "fan_epi_t1"):
            torch.testing.assert_close(a[key], b[key], rtol=1e-4, atol=1e-5)
        if mode == "eval":
            for key in ("fan_oracle_ce", "fan_single_ce", "fan_mixed_ce", "fan_stream_ce_k3"):
                torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=1e-5)
            with torch.no_grad():
                la = lxf(inp, slot_layout=layout)["logits"]
                lb = torch.log_softmax(fan(inp, slot_layout=layout)["logits"].float(), -1)
            fin = torch.isfinite(lb)
            torch.testing.assert_close(la[fin], lb[fin], rtol=1e-5, atol=1e-5)


# ── 2. the code reaches every cell ─────────────────────────────────────────────────


def _cell_table(layout, m_cells: int, d: int) -> torch.Tensor:
    """A forced per-CELL depth table (the register's `_tul_core` runs on the cell axis)."""
    B, S = layout.slot_index.shape
    return torch.full((B, S * m_cells), d, dtype=torch.long)


@pytest.mark.parametrize("m_cells", [4, 6])
def test_linear_toy_core_exit_is_the_per_cell_recurrence(m_cells):
    """Core map replaced by h -> lam * h. The exit on the CELL axis must equal
    h_{t+1} = f + r * rms_cell(f) * u_k, f = lam * h_t, run for the forced depth from the
    captured entry, per rollout and per CELL (rms over one cell's streams and channels)."""
    _ids, inp, lab, layout = _batch(m_cells)
    lam, D = 0.9, 3
    m = _build(**_lxfan_kw(m_cells)).eval()
    m._apply_core_step = lambda h_in, *a, **k: (lam * h_in, None)
    m._jac_capture = []
    core = _Spy(m, "_tul_core")
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout,
          slot_depths=_cell_table(layout, m_cells, D))
    h_exit = core.outs[0][1]                                    # [K*B, S*M, n, C]
    h = m._jac_capture[0]["h"]                                  # the entry, pass 0
    B0 = h.shape[0] // K
    for k in range(1, K):
        assert torch.equal(h[:B0], h[k * B0:(k + 1) * B0])     # the entry has no code
    enum = m.tul_code_enum
    u = enum.directions().detach()
    valid = layout.slot_valid.repeat_interleave(m_cells, dim=1).repeat(K, 1)   # per cell
    code = u.repeat_interleave(B0, 0)[:, None, None, :]           # [K*B, 1, 1, C]
    ref = h.clone()
    for _t in range(D):
        f = lam * ref
        rms = f.float().flatten(2).pow(2).mean(-1).sqrt()[..., None, None]     # per cell
        ref = f + enum.ratio * rms * code * valid[..., None, None]
    assert h_exit.shape[1] == layout.slot_index.shape[1] * m_cells
    torch.testing.assert_close(h_exit[valid], ref[valid], rtol=1e-5, atol=1e-6)


def test_the_code_separates_every_cell_index_from_pass_two_on():
    """Real (tiny) core: at every pass's START the K rollouts are equal at pass 1 and
    differ from pass 2 on at EVERY valid cell of EVERY cell index (a code that reached
    cell 0 only, or one slot in M, fails here); the term is called once per pass."""
    M = 4
    _ids, inp, lab, layout = _batch(M)
    m = _build(**_lxfan_kw(M)).eval()
    m._jac_capture = []
    spy = _Spy(m.tul_code_enum, "term")
    D = 3
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout, slot_depths=_cell_table(layout, M, D))
    assert len(spy.calls) == D
    B0, S = layout.slot_valid.shape
    v = layout.slot_valid
    for t, cap in enumerate(m._jac_capture):
        h = cap["h"].view(K, B0, S, M, -1)
        for k in range(1, K):
            d = (h[k] - h[0]).abs().amax(-1)                      # [B0, S, M]
            if t == 0:
                assert float(d.max()) == 0.0
            else:
                assert float(d[v].min()) > 1e-5, (t, k, float(d[v].min()))


# ── 3. the mixture ─────────────────────────────────────────────────────────────────


def _brute_rollout_lp(m, xh, lab, mask_slot: bool = True):
    """[K, B, L] log p of each label under each rollout, from full logits. ``mask_slot``
    False is the fan instruments' convention: `accumulate_span_ce` scores the plain tied
    head and leaves the slot id its logit (the token loss masks it)."""
    B0, L = lab.shape
    logits = xh.detach().float() @ m.embed.lm_weight().detach().float().T
    if mask_slot:
        logits[..., m.cfg.tul.slot_id] = float("-inf")
    lp = torch.log_softmax(logits, -1).gather(
        -1, lab.clamp(min=0).repeat(K, 1).unsqueeze(-1)).squeeze(-1)
    return lp.view(K, B0, L)


def test_the_token_loss_is_the_exact_per_span_mixture_and_the_deploy_read():
    """Hand check: for every span of every row, S_k = sum of log p_k over its scored
    tokens, the span term is log mean_k exp(S_k), the loss minus their sum over the token
    count. The model's loss, the deploy read's CE and this agree; the mean of the
    rollouts' CEs does not."""
    _ids, inp, lab, layout = _batch(4)
    m = _build(**_lxfan_kw(4)).eval()
    coda = _Spy(m, "_enum_mix_losses")
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
        logits = m(inp, slot_layout=layout)["logits"]
    xh = coda.calls[0][0][0]
    assert xh.shape[0] == K * inp.shape[0]
    lp = _brute_rollout_lp(m, xh, lab)
    row_w, _p, _z = m._tul_half_weights(lab, layout)
    w = (row_w.view_as(lab) * (lab != -100)).float()
    num, n_spans = 0.0, 0
    for b in range(lab.shape[0]):
        for g in layout.bag_id[b].unique().tolist():
            sel = (layout.bag_id[b] == g) & (w[b] > 0)
            if sel.any():
                S = (lp[:, b][:, sel] * w[b][sel]).sum(-1)                   # [K]
                num = num + (torch.logsumexp(S, 0) - math.log(K))
                n_spans += 1
    assert n_spans >= 6
    brute = -num / w.sum()
    torch.testing.assert_close(out["enum_ce_mix"], brute, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out["ce_tokens"], brute, rtol=1e-5, atol=1e-5)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.clamp(min=0).reshape(-1),
                         reduction="none").view_as(lab)
    torch.testing.assert_close((ce * w).sum() / w.sum(), brute, rtol=1e-5, atol=1e-5)
    mean_ce = -(lp * w).sum((1, 2)).mean() / w.sum()
    assert float(mean_ce - brute) > 1e-6


def test_the_fan_oracle_reads_per_span_mixtures_over_the_rollouts():
    """`fan/mixed_ce` on LX-Fan is the per-span mixture over the K rollouts of the shipped
    write's span CE, on the oracle's spans (a valid slot whose next span has scored
    tokens), recomputed here by brute force from the coda's output; the per-rollout mean
    is a different, larger number. Scored on the UNMASKED head, the fan instruments' own
    convention since they were built (`accumulate_span_ce`); the token loss masks the
    slot id, so on the same spans the two differ by the slot id's mass."""
    _ids, inp, lab, layout = _batch(4)
    m = _build(**_lxfan_kw(4)).eval()
    coda = _Spy(m, "_enum_mix_losses")
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    xh = coda.calls[0][0][0]
    lp = _brute_rollout_lp(m, xh, lab, mask_slot=False)              # [K, B, L]
    tok = (~layout.slot_mask) & (lab != -100)
    num, mean_num, n_tok = 0.0, 0.0, 0
    for b in range(lab.shape[0]):
        for s in range(layout.slot_valid.shape[1]):
            sel = tok[b] & (layout.bag_id[b] == s + 1)
            if not (bool(layout.slot_valid[b, s]) and bool(sel.any())):
                continue
            S = lp[:, b][:, sel].sum(-1)                                     # [K]
            num = num - (torch.logsumexp(S, 0) - math.log(K))
            mean_num = mean_num - S.mean()
            n_tok += int(sel.sum())
    assert n_tok > 0 and float(out["fan_oracle_n_tokens"]) == n_tok
    torch.testing.assert_close(out["fan_mixed_ce"], num / n_tok, rtol=1e-5, atol=1e-5)
    assert float(mean_num / n_tok - out["fan_mixed_ce"]) > 1e-6
    # the oracle picks per span over streams of mixtures: never above any one stream
    for i in range(4):
        assert float(out["fan_oracle_ce"]) <= float(out[f"fan_stream_ce_k{i}"]) + 1e-6


# ── 4. what runs, and how often ────────────────────────────────────────────────────


def test_the_register_pools_once_on_the_base_rows():
    _ids, inp, lab, layout = _batch(4)
    m = _build(**_lxfan_kw(4)).train()
    seen: list[int] = []
    m.tul_register.register_forward_hook(lambda mod, a, o: seen.append(int(a[0].shape[0])))
    m(inp, labels=lab, slot_layout=layout)
    assert seen == [inp.shape[0]], seen


def test_wta_zero_runs_no_extra_coda_pass_and_the_eps_knob_is_refused_there():
    _ids, inp, lab, layout = _batch(4)
    counts = {}
    for name, kw in (("lxfan", _lxfan_kw(4)), ("fan_wta0", _fan_kw(4, fan_all_wta_lambda=0.0)),
                     ("fan_wta1", _fan_kw(4))):
        m = _build(**kw).train()
        spy = _Spy(m, "_back_region")
        out = m(inp, labels=lab, slot_layout=layout)
        counts[name] = len(spy.calls)
        assert ("fan_wta_ce" in out) == (name == "fan_wta1"), name
    assert counts == {"lxfan": 1, "fan_wta0": 1, "fan_wta1": 3}, counts
    with pytest.raises(ValueError, match="fan_select_eps"):
        _tul(**_fan_kw(4, fan_all_wta_lambda=0.0, fan_select_eps=0.1))


def test_the_epi_ridge_under_the_code_is_the_base_fit_on_rollout_mean_deviations():
    """K duplicated seeds at ridge lam * K give the SAME readout W as the base seeds at
    lam on the rollout-mean deviations, so the epi score is that of the mean; at plain lam
    it is not. The model hands the term lam * K under the code and lam without it."""
    g = torch.Generator().manual_seed(0)
    N0, F_, C, lam, eta = 12, 5, 7, 3.0, 30.0
    H = torch.randn(N0, F_, generator=g)
    D = torch.randn(K, N0, C, generator=g)
    a_dup = ridge_map(H.repeat(K, 1), lam * K)
    a_base = ridge_map(H, lam)
    # fp32 tolerance: `ridge_map` standardises its input in the input's dtype before the
    # double solve, and the std over K copies rounds differently from the std over one.
    torch.testing.assert_close(a_dup @ D.reshape(K * N0, C).double(),
                               a_base @ D.mean(0).double(), rtol=1e-5, atol=1e-6)
    s_dup = epi_score(D.reshape(K * N0, C), a_dup, eta)
    torch.testing.assert_close(s_dup, epi_score(D.mean(0), a_base, eta), rtol=1e-5, atol=1e-6)
    assert abs(float(epi_score(D.reshape(K * N0, C), ridge_map(H.repeat(K, 1), lam), eta))
               - float(s_dup)) > 1e-3
    # the model's own call
    _ids, inp, lab, layout = _batch(4)
    seen: list[float] = []
    real = transformer_mod.fan_epi_term

    def _spy(*a, **k):
        seen.append(float(a[5]))
        return real(*a, **k)
    transformer_mod.fan_epi_term = _spy
    try:
        for kw in (_lxfan_kw(4), _fan_kw(4)):
            m = _build(**kw).train()
            m(inp, labels=lab, slot_layout=layout)
    finally:
        transformer_mod.fan_epi_term = real
    ridge = float(_tul(**_fan_kw(4)).fan_epi_ridge)
    assert seen == [ridge * K, ridge], seen


def test_plan_ablations_and_the_eval_probes_run_on_lx_fan():
    """`plan_mode` zero makes every rollout the same read of the cells (they are all
    blank), shuffle permutes a row's slots alike in every rollout, and the trainer's
    eval probes (`eval_ablations`) run on the rollout batch."""
    _ids, inp, lab, layout = _batch(4)
    m = _build(**_lxfan_kw(4)).eval()
    with torch.no_grad():
        z = m.tul_forward_ablated(inp, lab, layout, plan_mode="zero")
        torch.manual_seed(2)
        s = m.tul_forward_ablated(inp, lab, layout, plan_mode="shuffle")
        n = m(inp, labels=lab, slot_layout=layout)
        st = m.tul_slot_state_probe(inp, layout)
        lift = m.tul_attn_lift_probe(inp, layout)
    assert float(z["enum_width_gain"]) == pytest.approx(0.0, abs=1e-6)
    assert float(n["enum_width_gain"]) > 0.0
    assert torch.isfinite(s["loss"]) and float(s["loss"]) != float(n["loss"])
    assert np.isfinite(st["slot_eff_rank"]) and st["slot_eff_rank"] > 0
    assert lift and any(np.isfinite(v) for v in lift.values())


@pytest.mark.parametrize("m_cells", [4, 6])
def test_eager_generation_runs_and_a_seeded_run_repeats(m_cells):
    """The eager generator (the one decoder that does not refuse LX-Fan) reads the
    label-free forward, i.e. the per-span Bayes read over the K rollouts of M cells; a
    short prompt holds no complete next span. It must run and repeat under a seed."""
    from morph.inference.tul_generate import generate_tul
    from test_tul_fan import _spec
    m = _build(**_lxfan_kw(m_cells)).eval()
    runs = [generate_tul(m, [5, 9, 12, 3], _rule(), _spec(m_cells), max_new_tokens=40,
                         temperature=1.0, seed=7, emit_source="token")[0] for _ in range(2)]
    assert len(runs[0]) == 40 and runs[0] == runs[1]
    assert all(0 <= t < m.cfg.vocab_size for t in runs[0])


# ── 5. refusals ────────────────────────────────────────────────────────────────────


def test_the_supported_combination_builds_at_four_and_six_cells():
    for mc in (4, 6):
        tc = _tul(**_lxfan_kw(mc))
        assert tc.code_enum_k == K and tc.fan_k == mc and tc.prefix_k == mc
        assert tc.fan_mix == "all" and tc.fan_all_wta_lambda == 0.0
        m = MORPHTransformer(_tiny(tul=tc, d_ff=_D_FF))
        assert m.tul_code_enum is not None and m.tul_fan is not None
        assert m.tul_spandec_par is None and m.tul_register is not None


@pytest.mark.parametrize("kw,exc,match", [
    (dict(fan_mix="mean", prefix_k=4, fan_all_wta_lambda=1.0), NotImplementedError,
     "fan_mix='mean'"),
    (dict(fan_mix="softmax", fan_all_wta_lambda=1.0), NotImplementedError,
     "fan_mix='softmax'"),
    (dict(fan_mix="select", fan_all_wta_lambda=1.0), NotImplementedError,
     "fan_mix='select'"),
    (dict(fan_all_wta_lambda=1.0), NotImplementedError, "fan_all_wta_lambda=1.0"),
    (dict(fan_seed_noise=0.5), NotImplementedError, "fan_seed_noise"),
    (dict(spandec_parallel=True), NotImplementedError, "spandec_parallel with tul.fan_k"),
    (dict(spandec_parallel=True, spandec_parallel_detach=True), NotImplementedError,
     "spandec_parallel with tul.fan_k"),
])
def test_everything_but_write_all_at_wta_zero_stays_refused(kw, exc, match):
    with pytest.raises(exc, match=match):
        _tul(**_lxfan_kw(4, **kw))


def test_the_bare_register_is_still_refused_under_the_code():
    with pytest.raises(NotImplementedError, match="Thought Register"):
        _tul(slot_cells=4, prefix_k=4, code_enum_k=K, plast_weight=1.0)


# ── 6. the configs ─────────────────────────────────────────────────────────────────


def _diff(name: str, parent: str, monkeypatch):
    from omegaconf import OmegaConf
    from test_slot_gain_tail import _MISSING, _leaves
    from test_tul_strict_geometry import _runtime
    cfg, rt = _runtime(name, monkeypatch)
    pcfg, _prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    return cfg, rt, {k for k in c.keys() | p.keys()
                     if c.get(k, _MISSING) != p.get(k, _MISSING)}


def _one_step(cfg, tc, prefix_k: int):
    from morph.training.train import build_morph_config
    mc = build_morph_config(cfg, tul=tc)
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=tc, d_ff=_D_FF,
                               core_fixed_point_lambda=mc.core_fixed_point_lambda,
                               slot_gain_lambda=mc.slot_gain_lambda,
                               slot_gain_target=mc.slot_gain_target,
                               slot_gain_tail_lambda=mc.slot_gain_tail_lambda,
                               slot_gain_tail_target=mc.slot_gain_tail_target)).train()
    m = m.float()
    ids = np.random.default_rng(0).integers(5, 64, size=(2, 160)).astype(np.int64)
    ids[:, ::8] = 10
    inp, lab, layout, _ = slot_layout_from_ids(
        ids, _rule(), TulLayoutSpec(seq_len=64, prefix_k=prefix_k, max_slots=10, slot_id=4))
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert "enum_ce_mix" in out and "fan_repel" in out and "fan_wta_ce" not in out
    assert float(m.tul_code_enum.basis.grad.abs().sum()) > 0
    assert float(m.tul_register.W_o.weight.grad.abs().sum()) > 0
    return mc


def test_lxfan4_is_fp01_plus_the_fan_minus_the_probe_head(monkeypatch):
    cfg, rt, diff = _diff("tul_slot_spandec_strict_lxfan4_fp01",
                          "tul_slot_spandec_strict_e4probe_fp01", monkeypatch)
    assert diff == {"tul.prefix_k", "tul.fan_k", "tul.fan_mix", "tul.slot_cell_init",
                    "tul.fan_all_wta_lambda", "tul.fan_repel_mode", "tul.fan_repel_lambda",
                    "tul.fan_repel_passes", "tul.fan_epi_features", "tul.fan_epi_ridge",
                    "tul.fan_epi_eta", "tul.spandec_parallel", "tul.spandec_parallel_detach",
                    "wandb.name"}, sorted(diff)
    tc = rt.model_cfg
    assert (tc.code_enum_k, tc.fan_k, tc.slot_cells, tc.prefix_k) == (4, 4, 4, 4)
    assert tc.fan_mix == "all" and tc.fan_all_wta_lambda == 0.0 and not tc.spandec
    assert not tc.spandec_parallel and tc.fan_repel_mode == "epivol"
    assert int(cfg.training.steps) == 5000 and cfg.wandb.name == "lxtul-lxfan4-fp01"
    mc = _one_step(cfg, tc, prefix_k=4)
    assert mc.core_fixed_point_lambda == 0.1 and mc.slot_gain_target == 0.98
    assert mc.slot_gain_tail_lambda == 100.0


def test_lxfan6_is_lxfan4_at_six_cells(monkeypatch):
    cfg, rt, diff = _diff("tul_slot_spandec_strict_lxfan6_fp01",
                          "tul_slot_spandec_strict_lxfan4_fp01", monkeypatch)
    assert diff == {"tul.fan_k", "tul.prefix_k", "wandb.name"}, sorted(diff)
    tc = rt.model_cfg
    assert (tc.code_enum_k, tc.fan_k, tc.slot_cells, tc.prefix_k) == (4, 6, 6, 6)
    assert int(cfg.training.steps) == 5000 and cfg.wandb.name == "lxtul-lxfan6-fp01"
    spec = TulLayoutSpec(seq_len=int(cfg.data.seq_len), prefix_k=tc.prefix_k,
                         max_slots=int(cfg.tul.max_slots))
    assert spec.l_total == 1024 + 6 * 64
    _one_step(cfg, tc, prefix_k=6)


def test_lxfan4_against_a2_differs_by_the_code_and_fp01s_keys(monkeypatch):
    _cfg, _rt, diff = _diff("tul_slot_spandec_strict_lxfan4_fp01",
                            "tul_slot_spandec_strict_fan4_all_fp01", monkeypatch)
    assert diff == {"tul.code_enum_k", "tul.fan_all_wta_lambda", "tul.spandec",
                    "tul.spandec_parallel", "tul.spandec_parallel_detach",
                    "tul.fan_select_eps", "model.slot_gain_target",
                    "model.slot_gain_tail_lambda", "wandb.name"}, sorted(diff)


def test_dataclass_replace_keeps_the_refusals():
    """`dataclasses.replace` re-runs `__post_init__`: fp01's LX cannot be turned into LX-Fan
    while its probe head is on."""
    tc = _tul(**_lx_kw())
    with pytest.raises(NotImplementedError, match="spandec_parallel with tul.fan_k"):
        dataclasses.replace(tc, **{k: v for k, v in _lxfan_kw(4).items()
                                   if k != "code_enum_k"})
