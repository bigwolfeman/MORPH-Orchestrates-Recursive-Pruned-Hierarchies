"""LX credit arms (2026-09-26): how the training token loss credits LX's K code rollouts,
and (below) the write-all fan's winner-take-all kept under the codes.

Files: morph/model/rollout_mixture.py (`hard_credit_weights`, `hard_credit_span_sum`),
morph/model/transformer.py (`_enum_mix_losses`'s hard branch, `enum_win_entropy`),
morph/model/tul.py (`code_enum_credit`, `code_enum_hard_eps`, `_check_code_enum`),
morph/training/train.py (`enum_hard_weighted` in both `*_weighted` lists),
morph/configs/tul_slot_spandec_strict_e4probe_fp01_hard.yaml.
Note: .agents/notes/proposed/architecture/2026-09-26-lx-hard-credit.md

What each test pins (HARD CREDIT, arm c):
  * SOFT IS THE TREE: `code_enum_credit="soft"` gives the loss, the eval logits, the eval
    loss and every gradient measured on e951498 (the tree before the key), at dropout 0
    and 0.1 (full tensors compared `torch.equal` by
    /home/wolfe/morph-scratch/credit/pin_compare.py).
  * THE RELAXED ONE-HOT, BY HAND: known per-span sums give the known objective, the
    known credit (ties to the lowest rollout) and a gradient into S equal to the credit.
  * THE GRADIENT, ON THE MODEL: into the coda output of rollout k at a scored position of
    span g it is exactly c_k(g) times that rollout's plain CE gradient, with c built from
    an independently computed per-span CE table (winner 1 - eps, others eps / (K - 1)).
  * EVAL IS UNTOUCHED: soft and hard models with the same weights give identical eval
    logits and every identical eval key.
  * THE TRAIN FORWARD IS THE SAME FORWARD: every key but the loss is identical; the
    mixture NLL `enum_ce_mix` is the soft model's; loss - `enum_hard_weighted` is the soft
    loss (train/loss contract); `enum_hard_obj` is the hand objective.
  * THE WIN STATISTICS: `enum_code_win{k}` and `enum_win_entropy` equal the hand counts.
  * REFUSALS and THE CONFIG.

CPU, fp32, the `tests/test_tul_fan.py` fixtures (strict geometry) with fp01's constraint.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from morph.model.rollout_mixture import hard_credit_span_sum, hard_credit_weights
from morph.model.transformer import MORPHTransformer
from morph.model.tul_layout import TulLayoutSpec, slot_layout_from_ids
from test_tul_fan import _batch, _rule, _tiny, _tul
from test_tul_lxfan import _D_FF, _FP01, K, _build, _lx_kw, _Spy

EPS = 0.05


def _hard_kw(**kw) -> dict:
    return _lx_kw(code_enum_credit="hard", code_enum_hard_eps=EPS, **kw)


# ── 1. SOFT IS THE TREE ─────────────────────────────────────────────────────────────

# Measured 2026-09-26 on the UNMODIFIED worktree at e951498 by
# /home/wolfe/morph-scratch/credit/pin_head.py (one CPU thread): (train loss, eval logit
# sum, eval loss, sum |grad|, n grads) of fp01's LX. The dropout-0 row is also
# tests/test_tul_lxfan.py's "lx" pin, measured on 36a9823.
SOFT_PINS = {
    0.0: (9.606481552124023, -52828.743996977806, 9.570661544799805, 1201.495515583244, 190),
    0.1: (9.521562576293945, -52828.743996977806, 9.570661544799805, 1187.999528134106, 190),
}


def _pin_run(m: MORPHTransformer):
    _ids, inp, lab, lay = _batch(2)
    m.train()
    torch.manual_seed(99)
    o = m(inp, labels=lab, slot_layout=lay)
    o["loss"].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    m.eval()
    with torch.no_grad():
        lg = m(inp, labels=None, slot_layout=lay)["logits"]
        ev = m(inp, labels=lab, slot_layout=lay)["loss"]
    ls = float(torch.nan_to_num(lg.double(), nan=0.0, posinf=0.0, neginf=0.0).sum())
    gs = float(sum(g.double().abs().sum() for g in grads))
    return float(o["loss"]), ls, float(ev), gs, len(grads)


@pytest.mark.parametrize("dropout", sorted(SOFT_PINS))
def test_soft_credit_is_the_tree(dropout):
    m = _build(model_kw={"dropout": dropout}, **_lx_kw(code_enum_credit="soft"))
    assert m._enum_hard_eps is None
    assert _pin_run(m) == SOFT_PINS[dropout]


# ── 2. the relaxed one-hot, by hand ────────────────────────────────────────────────


def test_hard_credit_by_hand():
    """K = 4, three spans. Span 0: rollout 0 is best (S = -1). Span 1: rollouts 1 and 2
    tie at S = -1, the lowest (1) wins. Span 2 is unscored and contributes nothing."""
    S = torch.tensor([[-1.0, -5.0, -7.0],
                      [-2.0, -1.0, -7.0],
                      [-3.0, -1.0, -7.0],
                      [-4.0, -9.0, -7.0]], requires_grad=True)
    scored = torch.tensor([True, True, False])
    lo, hi = EPS / 3, 1.0 - EPS
    want_c = torch.tensor([[hi, lo, hi], [lo, hi, lo], [lo, lo, lo], [lo, lo, lo]])
    torch.testing.assert_close(hard_credit_weights(S, EPS), want_c, rtol=0, atol=0)
    obj = hard_credit_span_sum(S, scored, EPS)
    # span 0: 0.95 * -1 + (0.05/3) * (-2 - 3 - 4) = -1.1 ; span 1: -0.95 + (0.05/3) * -15
    assert float(obj) == pytest.approx(-1.1 + -1.2, abs=1e-6)
    obj.backward()
    want_g = want_c.clone()
    want_g[:, 2] = 0.0
    torch.testing.assert_close(S.grad, want_g, rtol=0, atol=0)
    assert torch.allclose(hard_credit_weights(S, EPS).sum(0), torch.ones(3))


def test_eps_at_the_top_is_the_uniform_credit_and_zero_is_pure_wta():
    S = torch.randn(4, 5)
    torch.testing.assert_close(hard_credit_weights(S, 0.75), torch.full((4, 5), 0.25))
    c0 = hard_credit_weights(S, 0.0)
    assert torch.equal(c0, F.one_hot(S.argmax(0), 4).t().float())


# ── 3. the gradient on the model ───────────────────────────────────────────────────


def _captured(m: MORPHTransformer):
    """One train forward; the args `_enum_mix_losses` received (the K-fold coda output
    and the expanded labels / ids / layout)."""
    _ids, inp, lab, lay = _batch(2)
    m.train()
    spy = _Spy(m, "_enum_mix_losses")
    torch.manual_seed(3)
    m(inp, labels=lab, slot_layout=lay)
    (xh, labels, input_ids, layout, k), kw = spy.calls[0]
    assert k == K and kw == {"want_groups": False}
    return xh.detach(), labels, input_ids, layout


def _ref_table(m: MORPHTransformer, xh: torch.Tensor, labels, layout):
    """An INDEPENDENT per-rollout, per-span log-likelihood table from the coda output:
    plain log-softmax with the slot id masked, scored = labelled token positions (emit
    weight 0, plast weight 1 under LX), groups = (row, bag_id) by Python loops."""
    BK, L, _C = xh.shape
    B = BK // K
    logits = xh.float() @ m.embed.lm_weight().float().t()
    logits[..., m.cfg.tul.slot_id] = float("-inf")
    lp_all = torch.log_softmax(logits, dim=-1)
    lab = labels[:B]
    scored = (lab != -100) & ~layout.slot_mask[:B]
    groups: dict[tuple[int, int], list[int]] = {}
    for b in range(B):
        for j in range(L):
            if bool(scored[b, j]):
                groups.setdefault((b, int(layout.bag_id[b, j])), []).append(j)
    keys = sorted(groups)
    lp = torch.stack([lp_all[k * B:(k + 1) * B].gather(
        -1, lab.clamp_min(0).unsqueeze(-1)).squeeze(-1) for k in range(K)])      # [K, B, L]
    return lp, keys, groups, int(scored.sum())


def test_hard_gradient_is_the_relaxed_one_hot_times_each_rollouts_ce_gradient():
    m = _build(**_hard_kw())
    xh0, labels, input_ids, layout = _captured(m)
    B = xh0.shape[0] // K
    # the model's hard objective and its gradient into the coda output
    xh = xh0.clone().requires_grad_(True)
    out = m._enum_mix_losses(xh, labels, input_ids, layout, K, want_groups=False)
    out["loss"].backward()
    g_hard = xh.grad.view(K, B, *xh.shape[1:])
    # the independent table: every rollout's plain CE gradient (credit 1 everywhere) ...
    xr = xh0.clone().requires_grad_(True)
    lp, keys, groups, n_sc = _ref_table(m, xr, labels, layout)
    assert float(out["n_targets"]) == n_sc
    S = torch.stack([torch.stack([lp[k, b, groups[(b, g)]].sum() for (b, g) in keys])
                     for k in range(K)])                                          # [K, G]
    (-S.sum() / n_sc).backward()
    g_ce = xr.grad.view(K, B, *xr.shape[1:])
    # ... and the credit by hand: argmin CE (argmax S), the lowest k on a tie
    c_pos = torch.zeros(K, B, xh.shape[1])
    n_win = [0] * K
    for gi, (b, g) in enumerate(keys):
        col = [float(S[k, gi]) for k in range(K)]
        best = max(range(K), key=lambda k: (col[k], -k))
        n_win[best] += 1
        for k in range(K):
            c_pos[k, b, groups[(b, g)]] = (1.0 - EPS) if k == best else EPS / (K - 1)
    assert 0 < max(n_win) < len(keys), "degenerate case: one rollout wins every span"
    want = c_pos.unsqueeze(-1) * g_ce
    torch.testing.assert_close(g_hard, want, rtol=2e-5, atol=1e-9)
    # a non-winner's gradient is eps/(K-1) of its CE gradient, and it is NOT zero
    lose = c_pos == EPS / (K - 1)
    assert bool(lose.any()) and float(g_hard[lose].abs().sum()) > 0
    # the objective's value: sum_g sum_k c_k CE_k(g) / n
    c_g = torch.tensor([[(1.0 - EPS) if k == max(range(K), key=lambda kk: (float(S[kk, gi]),
                                                                          -kk))
                         else EPS / (K - 1) for gi in range(len(keys))] for k in range(K)])
    want_obj = float(-(c_g * S.detach()).sum() / n_sc)
    assert float(out["enum_hard_obj"]) == pytest.approx(want_obj, rel=1e-5)
    # the win statistics
    for k in range(K):
        assert float(out[f"enum_code_win{k}"]) == pytest.approx(n_win[k] / len(keys))
    frac = np.array(n_win) / len(keys)
    ent = -sum(f * math.log(f) for f in frac if f > 0) / math.log(K)
    assert float(out["enum_win_entropy"]) == pytest.approx(ent, rel=1e-5)


def test_identical_codes_tie_every_span_and_rollout_zero_takes_the_winner_share():
    """Zeroed codes make the K rollouts one computation: every span ties, the lowest
    rollout wins it, and rollout 0 gets 1 - eps of the gradient, each other eps / (K-1)."""
    m = _build(**_hard_kw())
    m.tul_code_enum.directions = lambda: torch.zeros(K, m.cfg.d_model)
    xh0, labels, input_ids, layout = _captured(m)
    B = xh0.shape[0] // K
    assert torch.equal(xh0[:B], xh0[B:2 * B])
    xh = xh0.clone().requires_grad_(True)
    out = m._enum_mix_losses(xh, labels, input_ids, layout, K, want_groups=False)
    out["loss"].backward()
    assert float(out["enum_code_win0"]) == 1.0 and float(out["enum_win_entropy"]) == 0.0
    g = xh.grad.view(K, B, *xh.shape[1:])
    for k in range(1, K):
        torch.testing.assert_close(g[k] * ((1.0 - EPS) / (EPS / (K - 1))), g[0],
                                   rtol=1e-5, atol=1e-9)


# ── 4. eval is untouched; the train forward is the same forward ─────────────────────


def test_eval_is_identical_between_soft_and_hard():
    soft = _build(model_kw={"dropout": 0.1}, **_lx_kw())
    hard = _build(model_kw={"dropout": 0.1}, **_hard_kw())
    for k, v in soft.state_dict().items():
        assert torch.equal(v, hard.state_dict()[k]), k     # the knob builds nothing
    assert set(soft.state_dict()) == set(hard.state_dict())
    _ids, inp, lab, lay = _batch(2)
    soft.eval()
    hard.eval()
    with torch.no_grad():
        assert torch.equal(soft(inp, slot_layout=lay)["logits"],
                           hard(inp, slot_layout=lay)["logits"])
        a = soft(inp, labels=lab, slot_layout=lay)
        b = hard(inp, labels=lab, slot_layout=lay)
    assert set(a) == set(b) and "enum_hard_obj" not in b and "enum_hard_weighted" not in b
    for k, v in a.items():
        if torch.is_tensor(v):
            assert torch.equal(v, b[k]), k


def test_train_forward_is_the_same_forward_and_train_loss_is_the_mixture():
    soft = _build(model_kw={"dropout": 0.1}, **_lx_kw())
    hard = _build(model_kw={"dropout": 0.1}, **_hard_kw())
    _ids, inp, lab, lay = _batch(2)
    soft.train()
    hard.train()
    torch.manual_seed(5)
    a = soft(inp, labels=lab, slot_layout=lay)
    torch.manual_seed(5)
    b = hard(inp, labels=lab, slot_layout=lay)
    assert set(b) - set(a) == {"enum_hard_obj", "enum_hard_weighted"}
    for k, v in a.items():
        if k not in ("loss", "ce_main") and torch.is_tensor(v):
            assert torch.equal(v, b[k]), k      # fp_weighted, gain terms, par_*, enum_ce_mix
    assert float(b["enum_hard_weighted"]) != 0.0
    # train.py subtracts `enum_hard_weighted`: train/loss is the soft model's loss. The
    # train-side `ce_main` is "the loss before the core-loop terms" (`_apply_core_aux`),
    # which already carries every `groups` fold (gain hinge, probe head); the hard excess
    # is one more of them, and it subtracts the same way.
    for key in ("loss", "ce_main"):
        torch.testing.assert_close(b[key] - b["enum_hard_weighted"], a[key],
                                   rtol=0, atol=2e-6)
    # the probe head is detached and reads the mixture: its gradients do not see the credit
    a["loss"].backward()
    b["loss"].backward()
    for (na, pa), (nb, pb) in zip(soft.named_parameters(), hard.named_parameters()):
        if na.startswith("tul_spandec_par."):
            assert torch.equal(pa.grad, pb.grad), na
    ga = soft.tul_code_enum.basis.grad
    assert not torch.equal(ga, hard.tul_code_enum.basis.grad)


def test_the_val_loss_subtracts_the_hard_excess():
    from morph.training.train import evaluate

    class _Stub(torch.nn.Module):
        def tul_forward_with_plan_nats(self, x, y, layout):
            return {"loss": torch.tensor(5.0), "enum_hard_weighted": torch.tensor(0.75),
                    "ce_tokens": 4.25, "layer_passes": 8.0, "n_tokens": 4.0}

    class _Layout:
        stats: dict = {}

        def to(self, device):
            return self

    x = torch.zeros(1, 4, dtype=torch.long)
    avg, _ppl = evaluate(_Stub(), torch.device("cpu"), iter([(x, x, _Layout())] * 2),
                         n_batches=2, tul=True, extra={})
    assert avg == pytest.approx(4.25)


def test_train_logging_subtracts_the_hard_excess():
    """train.py's train-side `*_weighted` tuple carries the key (the loop that builds
    train/loss is inline in `main`, so the list is read from its source)."""
    import inspect

    import morph.training.train as tr
    src = inspect.getsource(tr)
    i = src.index('"fp_weighted", "core_gain_weighted", "egrad_weighted"')
    j = src.index("_lv = _lv - float(out[_ak])", i)
    assert '"enum_hard_weighted"' in src[i:j]


# ── 5. refusals ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kw,exc,match", [
    (dict(code_enum_credit="mcl"), ValueError, "must be 'soft' or 'hard'"),
    (dict(code_enum_hard_eps=0.1), ValueError, "silently ignored"),
    (dict(code_enum_credit="hard", code_enum_hard_eps=0.8), ValueError, r"\(K-1\)/K"),
    (dict(code_enum_credit="hard", code_enum_hard_eps=-0.1), ValueError, r"\(K-1\)/K"),
    (dict(code_enum_credit="hard", spandec_parallel_detach=False), NotImplementedError,
     "LIVE parallel head"),
])
def test_refusals(kw, exc, match):
    with pytest.raises(exc, match=match):
        _tul(**_lx_kw(**kw))


def test_hard_credit_needs_rollouts():
    with pytest.raises(ValueError, match="one rollout"):
        _tul(code_enum_credit="hard", plast_weight=1.0)


# ── 6. the config ──────────────────────────────────────────────────────────────────


def test_fp01_hard_is_fp01_plus_the_two_keys(monkeypatch):
    from test_tul_lxfan import _diff

    from morph.training.train import build_morph_config
    cfg, rt, diff = _diff("tul_slot_spandec_strict_e4probe_fp01_hard",
                          "tul_slot_spandec_strict_e4probe_fp01", monkeypatch)
    assert diff == {"tul.code_enum_credit", "tul.code_enum_hard_eps", "wandb.name"}, diff
    tc = rt.model_cfg
    assert tc.code_enum_credit == "hard" and tc.code_enum_hard_eps == 0.05
    assert tc.code_enum_k == 4 and tc.spandec_parallel and tc.spandec_parallel_detach
    assert int(cfg.training.steps) == 5000 and cfg.wandb.name == "lxtul-e4probe-fp01-hard"
    mc = build_morph_config(cfg, tul=tc)
    assert mc.core_fixed_point_lambda == 0.1
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=tc, d_ff=_D_FF, **_FP01)).train().float()
    assert m._enum_hard_eps == 0.05
    ids = np.random.default_rng(0).integers(5, 64, size=(2, 160)).astype(np.int64)
    ids[:, ::8] = 10
    inp, lab, layout, _ = slot_layout_from_ids(
        ids, _rule(), TulLayoutSpec(seq_len=64, prefix_k=tc.prefix_k, max_slots=10, slot_id=4))
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and "enum_hard_obj" in out
    assert float(m.tul_code_enum.basis.grad.abs().sum()) > 0


# ═════════════════════════════════════════════════════════════════════════════════════
# LX-Fan WITH a2's WINNER-TAKE-ALL OVER THE CELLS (arm b)
#
# Files: morph/model/transformer.py (`_tul_fan_all(n_rollouts=)`: the per-stream no-grad
# table passes on the rollout batch), morph/model/tul.py (the WTA refusal under the code lifted),
# morph/configs/tul_slot_spandec_strict_lxfan4_wta_fp01.yaml.
# Note: .agents/notes/proposed/architecture/2026-09-26-lx-fan.md (2026-09-26 amendment)
#
# What each test pins:
#   * OFF IS THE TREE: lxfan4 (WTA 0) and a2's fan (WTA 1.0, no code) at dropout 0.1 give
#     the e951498 numbers (dropout 0: tests/test_tul_lxfan.py's pins; full tensors of
#     both, at both dropouts, `torch.equal` by pin_compare.py).
#   * THE PASS COUNT: one no-grad table pass per stream on the rollout batch (K*B0 rows),
#     one grad pass, the model's own coda (a2: one batched M*B pass, then the same two).
#   * THE WINNER IS PER (ROLLOUT, SLOT): the table equals an independent per-stream sequential
#     reference on the rollout batch, and at eps 0 the grad pass's CE is the table's
#     per-row minimum (the winner of each rollout's cells, not one winner per span).
#   * THE MASKS: in the table passes every coda dropout gives row (k, i, b) the mask of
#     (i, b): equal over the K rollouts, different over streams and rows (checked on
#     content, codes zeroed). With the codes zeroed the K rollouts' tables are then
#     identical under dropout.
#   * THE CONTRACT: loss - `fan_wta_weighted` is the WTA-0 model's loss.
#   * THE CONFIG composes, differs from lxfan4 by a2's two keys, builds and trains a step.
# ═════════════════════════════════════════════════════════════════════════════════════

from morph.model.rollout_dropout import RolloutSharedDropout  # noqa: E402
from test_tul_lxfan import _fan_kw, _lxfan_kw  # noqa: E402

M = 4

# e951498, pin_head.py (one CPU thread), dropout 0.1: (train loss, eval logit sum, eval
# loss, sum |grad|, n grads). The dropout-0 rows of lxfan4 and a2's fan are in
# tests/test_tul_lxfan.py / measured here as 'lxfan4@0.0' and 'fan4_wta@0.0'.
FAN_PINS = {
    ("lxfan4", 0.0): (5.077616214752197, -64528.78620353341, 5.130616188049316,
                      710.712064732762, 180),
    ("lxfan4", 0.1): (5.025187015533447, -64528.78620353341, 5.130616188049316,
                      708.4147908984702, 180),
    ("a2", 0.0): (10.31538200378418, 1358.1454057991505, 5.132460117340088,
                  1436.3044085161928, 179),
    ("a2", 0.1): (10.261831283569336, 1358.1454057991505, 5.132460117340088,
                  1443.9956181086357, 179),
}


def _wta_kw(**kw) -> dict:
    """lxfan4 + a2's WTA (lambda 1.0, eps 0.05)."""
    return _lxfan_kw(M, **{"fan_all_wta_lambda": 1.0, "fan_select_eps": 0.05, **kw})


def _fan_pin_run(m: MORPHTransformer):
    _ids, inp, lab, lay = _batch(M)
    m.train()
    torch.manual_seed(99)
    o = m(inp, labels=lab, slot_layout=lay)
    o["loss"].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    m.eval()
    with torch.no_grad():
        lg = m(inp, labels=None, slot_layout=lay)["logits"]
        ev = m(inp, labels=lab, slot_layout=lay)["loss"]
    ls = float(torch.nan_to_num(lg.double(), nan=0.0, posinf=0.0, neginf=0.0).sum())
    gs = float(sum(g.double().abs().sum() for g in grads))
    return float(o["loss"].detach()), ls, float(ev), gs, len(grads)


@pytest.mark.parametrize("name,dropout", sorted(FAN_PINS))
def test_lxfan4_and_a2_are_the_tree(name, dropout):
    kw = _lxfan_kw(M) if name == "lxfan4" else _fan_kw(M)
    m = _build(model_kw={"dropout": dropout}, **kw)
    assert _fan_pin_run(m) == FAN_PINS[(name, dropout)]


def test_wta_under_the_code_runs_one_table_pass_per_stream_then_a2s_two():
    """a2 runs ONE batched no-grad pass over its M streams (M*B rows); under the code the
    table is one no-grad pass PER STREAM on the rollout batch (K*B0 rows each), then the
    winner's grad pass and the model's coda."""
    _ids, inp, lab, layout = _batch(M)
    B0 = inp.shape[0]
    m = _build(**_wta_kw()).train()
    spy = _Spy(m, "_back_region")
    out = m(inp, labels=lab, slot_layout=layout)
    rows = [int(a[0].shape[0]) for a, _k in spy.calls]
    assert rows == [K * B0] * (M + 2), rows
    a2 = _build(**_fan_kw(M)).train()
    spy2 = _Spy(a2, "_back_region")
    a2(inp, labels=lab, slot_layout=layout)
    assert [int(a[0].shape[0]) for a, _k in spy2.calls] == [M * B0, B0, B0]
    assert "fan_wta_ce" in out and "enum_ce_mix" in out and "fan_wta_weighted" in out
    out["loss"].backward()
    assert float(m.tul_code_enum.basis.grad.abs().sum()) > 0


def _table(m: MORPHTransformer, inp, lab, layout, seed: int = 5):
    m._fan_all_ce_capture = []
    torch.manual_seed(seed)
    out = m(inp, labels=lab, slot_layout=layout)
    cap = m._fan_all_ce_capture[0]
    m._fan_all_ce_capture = None
    return out, cap


def test_the_table_is_the_per_stream_table_and_the_winner_is_per_rollout(
        monkeypatch):
    """At dropout 0 the table equals M sequential per-stream passes rebuilt from the
    captured inputs with the model's own write / coda / CE (an independent loop), and at
    eps 0 the grad pass writes, in every (rollout, slot), THAT row's argmin of the table:
    the cells of each rollout compete, there is no winner shared across rollouts.
    (`fan_wta_ce` is NOT the table's minimum, in a2 either: the table writes stream i in
    EVERY slot, the grad pass writes each slot's own winner, and a coda token reads the
    earlier slots' cells too. Measured on a2 at eps 0: 5.2355 vs 5.2236.)"""
    import morph.model.transformer as tmod
    from morph.model.transformer import accumulate_span_ce, scatter_positions, span_token_counts
    _ids, inp, lab, layout = _batch(M)
    m = _build(**_wta_kw(fan_select_eps=0.0)).train()
    chosen: list[torch.Tensor] = []
    real = tmod.select_streams
    monkeypatch.setattr(tmod, "select_streams",
                        lambda cells, choice: (chosen.append(choice.clone()),
                                               real(cells, choice))[1])
    _out, cap = _table(m, inp, lab, layout)
    ce = cap["ce"]
    B0 = inp.shape[0]
    assert ce.shape[0] == K * B0 and ce.shape[-1] == M and len(chosen) == 1
    ref = []
    with torch.no_grad():
        for i in range(M):
            values, pos = m._tul_fan_stream_write(cap["cells_d"], i, cap["layout"], cap["L"])
            x_i = scatter_positions(cap["base_d"], pos, values)
            xh_i = m._back_region(x_i, cap["x0"], cap["bigram_emb"], cap["input_ids"],
                                  inject_keep=cap["keep"], attn_kwargs=cap["coda_kw"],
                                  ret_reset_mask=cap["tg_reset"])
            ref.append(accumulate_span_ce(xh_i, cap["w_head"], cap["gid"], cap["keep_tok"],
                                          cap["lab"], cap["g_bins"])[:, 1:])
    ref = torch.stack(ref, dim=-1)
    torch.testing.assert_close(ce, ref, rtol=1e-4, atol=1e-4)
    n_tok = span_token_counts(cap["gid"], cap["keep_tok"], cap["g_bins"])[:, 1:]
    ok = cap["layout"].slot_valid & (n_tok > 0)
    assert torch.equal(chosen[0][ok], ce.argmin(-1)[ok])
    # the winners differ across rollouts somewhere (else this could not tell the two apart)
    win = ce.argmin(-1).view(K, B0, -1)
    okb = ok.view(K, B0, -1)[0]
    assert bool((win[:, okb] != win[0:1, okb]).any())


def test_the_table_passes_tie_each_rows_rollouts_and_nothing_else():
    """Every rollout-shared dropout in the coda, over the M no-grad table passes (one per
    stream, K*B0 rows each). With the codes zeroed, the K rollouts of (stream i, row b)
    carry IDENTICAL input, and every other pair of rows differs. The mask must follow the
    content: rows with equal input draw the equal keep pattern (the K rollouts of one
    (i, b)), rows with different input draw different ones. (A check on row positions
    alone cannot fail: the dropout tiles by position whatever a pass holds; what the
    grouping decides is WHICH content shares a draw.)"""
    _ids, inp, lab, layout = _batch(M)
    m = _build(model_kw={"dropout": 0.3}, **_wta_kw())
    m.tul_code_enum.directions = lambda: torch.zeros(K, m.cfg.d_model)
    m.train()
    B0 = inp.shape[0]
    seen: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def _hook(mod, a, o):
        x = a[0]
        if not torch.is_grad_enabled():                   # the no-grad table passes only
            keep = torch.where(x == 0, -1, (o != 0).to(torch.int64))
            seen.setdefault(id(mod), []).append((x.detach().flatten(1).clone(),
                                                 keep.flatten(1)))

    drops = [mod for mod in m.coda.modules() if isinstance(mod, RolloutSharedDropout)]
    assert drops, "the coda of a code model runs rollout-shared dropout"
    for d in drops:
        d.register_forward_hook(_hook)
    torch.manual_seed(5)
    m(inp, labels=lab, slot_layout=layout)
    assert seen, "no dropout ran in the table passes"
    n_rows = K * M * B0
    for calls in list(seen.values())[:2]:              # two dropout sites
        x = torch.cat([c[0] for c in calls])            # every table row of this site
        keep = torch.cat([c[1] for c in calls])
        assert x.shape[0] == n_rows, x.shape
        assert bool((keep == 0).any()), "no element dropped: dropout did not act"
        same = torch.tensor([[torch.equal(x[r], x[q]) for q in range(n_rows)]
                             for r in range(n_rows)])
        # K identical copies per (i, b): exactly M * B0 content classes of size K
        assert int(same.sum()) == n_rows * K, int(same.sum())
        for r in range(n_rows):
            for q in range(r + 1, n_rows):
                both = (keep[r] >= 0) & (keep[q] >= 0)
                if bool(same[r, q]):
                    assert torch.equal(keep[r][both], keep[q][both]), (r, q)
                else:
                    assert not torch.equal(keep[r][both], keep[q][both]), (r, q)


def test_with_the_codes_zeroed_the_rollouts_tables_are_identical_under_dropout():
    _ids, inp, lab, layout = _batch(M)
    B0 = inp.shape[0]
    m = _build(model_kw={"dropout": 0.3}, **_wta_kw())
    m.tul_code_enum.directions = lambda: torch.zeros(K, m.cfg.d_model)
    m.train()
    _out, cap = _table(m, inp, lab, layout)
    ce = cap["ce"].view(K, B0, *cap["ce"].shape[1:])
    for k in range(1, K):
        assert torch.equal(ce[k], ce[0]), k
    m0 = _build(model_kw={"dropout": 0.0}, **_wta_kw())
    m0.tul_code_enum.directions = lambda: torch.zeros(K, m0.cfg.d_model)
    m0.train()
    _o0, cap0 = _table(m0, inp, lab, layout)
    assert not torch.equal(cap0["ce"], cap["ce"]), "dropout did not act in the table passes"


def test_train_loss_is_the_wta_zero_models_loss():
    """The WTA term is folded as `fan_wta_weighted` (a2's contract): subtracting it gives
    the WTA-0 model's loss on the same weights (token-state dropout 0 and eps 0, so the
    WTA pass draws no RNG that would move a later mask)."""
    _ids, inp, lab, layout = _batch(M)
    a = _build(**_lxfan_kw(M, token_state_dropout=0.0)).train()
    b = _build(**_wta_kw(token_state_dropout=0.0, fan_select_eps=0.0)).train()
    torch.manual_seed(5)
    oa = a(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(5)
    ob = b(inp, labels=lab, slot_layout=layout)
    assert float(ob["fan_wta_weighted"]) > 0.0
    torch.testing.assert_close(ob["loss"] - ob["fan_wta_weighted"], oa["loss"],
                               rtol=0, atol=2e-6)
    assert torch.equal(ob["enum_ce_mix"], oa["enum_ce_mix"])


def test_what_the_fan_still_refuses_under_the_code():
    with pytest.raises(NotImplementedError, match="fan_seed_noise"):
        _tul(**_wta_kw(fan_seed_noise=0.5))
    with pytest.raises(NotImplementedError, match="fan_mix='select'"):
        _tul(**_lxfan_kw(M, fan_mix="select", fan_all_wta_lambda=1.0))


def test_lxfan4_wta_is_lxfan4_plus_a2s_two_keys(monkeypatch):
    from test_tul_lxfan import _diff, _one_step
    cfg, rt, diff = _diff("tul_slot_spandec_strict_lxfan4_wta_fp01",
                          "tul_slot_spandec_strict_lxfan4_fp01", monkeypatch)
    assert diff == {"tul.fan_all_wta_lambda", "tul.fan_select_eps", "wandb.name"}, diff
    tc = rt.model_cfg
    assert tc.fan_all_wta_lambda == 1.0 and tc.fan_select_eps == 0.05
    assert (tc.code_enum_k, tc.fan_k, tc.prefix_k) == (4, 4, 4) and tc.fan_mix == "all"
    assert int(cfg.training.steps) == 5000 and cfg.wandb.name == "lxtul-lxfan4-wta-fp01"
    # a2's two values, exactly
    _c2, rt2, _d2 = _diff("tul_slot_spandec_strict_fan4_all_fp01",
                          "tul_slot_spandec_strict_lxfan4_wta_fp01", monkeypatch)
    assert (rt2.model_cfg.fan_all_wta_lambda, rt2.model_cfg.fan_select_eps) == (1.0, 0.05)
    with pytest.raises(AssertionError, match="fan_wta_ce"):
        _one_step(cfg, tc, prefix_k=4)          # lxfan's step asserts NO wta term


# ═════════════════════════════════════════════════════════════════════════════════════
# THE MEMORY WORK (2026-09-26): what autograd keeps for the backward
#
# Three changes, each bit-identical (the pins above and pin_compare.py) and each cutting
# saved activation memory on the rollout batch:
#   * `_slot_gain_penalty` checkpoints its two core applications on every model (it did
#     under plan C only): the backward replays them (tests/test_slot_gain_tail.py counts
#     the replays);
#   * the fixed-point term and the pass residual gather their slots BEFORE the fp32 casts;
#   * `_tul_core` gathers the slots from the BASE rows and tiles the gathered states.
# The test below pins the bytes on the tiny model so a revert of any one of them fails.
# ═════════════════════════════════════════════════════════════════════════════════════


def _saved_bytes(m: MORPHTransformer, prefix_k: int) -> int:
    """Distinct non-parameter storages autograd saves in one training forward, alive at
    its end (the credit memory table's rule, /home/wolfe/morph-scratch/credit/saved_bytes.py),
    plus the tensor arguments `checkpoint` holds for its replays."""
    import weakref

    import morph.model.transformer as tmod
    _ids, inp, lab, lay = _batch(prefix_k)
    params = {p.untyped_storage().data_ptr() for p in m.parameters()}
    refs: list = []

    def _keep(t):
        if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() not in params:
            refs.append(weakref.ref(t))

    def _pack(t):
        _keep(t)
        return t

    real = tmod.checkpoint

    def _ckpt(fn, *a, **k):
        for v in list(a) + list(k.values()):
            _keep(v)
        return real(fn, *a, **k)

    tmod.checkpoint = _ckpt
    try:
        m.train()
        torch.manual_seed(5)
        with torch.autograd.graph.saved_tensors_hooks(_pack, lambda t: t):
            out = m(inp, labels=lab, slot_layout=lay)
    finally:
        tmod.checkpoint = real
    seen, tot = set(), 0
    for r in refs:
        t = r()
        if t is None:
            continue
        st = t.untyped_storage()
        if st.data_ptr() not in seen:
            seen.add(st.data_ptr())
            tot += st.nbytes()
    del out
    return tot


# Measured 2026-09-26 on this tree (after), and with the three changes reverted (the
# e951498 code for them). One CPU thread, fp32, the fixtures above.
SAVED_BYTES = {"lxfan4": 15093942, "lx": 9041216}


@pytest.mark.parametrize("name", sorted(SAVED_BYTES))
def test_saved_bytes_do_not_regrow(name):
    kw, pk = ((_lxfan_kw(M), M) if name == "lxfan4" else (_lx_kw(), 2))
    got = _saved_bytes(_build(**kw), pk)
    print(f"SAVED_BYTES {name} {got}")
    assert SAVED_BYTES[name] is not None, got
    assert got <= SAVED_BYTES[name], (name, got, SAVED_BYTES[name])
