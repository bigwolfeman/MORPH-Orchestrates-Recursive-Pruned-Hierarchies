"""Stage 1 of the staged latent pretraining (``tul.latent_pre_target``, 2026-10-02).

Files: morph/model/tul_latent_pre.py (targets, losses, readings), morph/model/tul.py (the
keys, ``_check_latent_pre``), morph/model/transformer.py (``_forward_latent_pre``,
``_latent_pre_target``, ``tul_latent_pre_attach_ref``, the dispatch in ``_forward_tul``,
the eval trajectory flag in ``_tul_core``), morph/training/latent_pre_ref.py (the frozen
plain model's loader), morph/training/train.py (the val branch, both subtraction tuples),
morph/training/tul_setup.py, the five stage-1 configs.
Note: .agents/notes/proposed/architecture/2026-10-02-staged-latent-pretraining.md

What each test pins:
  * The target is FROZEN: the plain model gets no gradient, is not a parameter of the live
    model, and it and its targets are unchanged after an optimizer step.
  * The target of slot s is span s+1's: perturbing span s+1 moves slot s's target (and,
    for the span-local prelude target, NOTHING else); perturbing span s+2 leaves it alone.
    Both plain targets equal an independent recomputation on the unpadded span / row.
  * Retrieval chance is 1/candidates from the actual counts; a perfect predictor reads 1.0.
  * L2, InfoNCE and the relaxed WTA equal hand computations.
  * No coda parameter gets a gradient in stage 1; the eval forward runs no coda either and
    reports every pass.
  * Refusals; the loader's strict load and its refusals; the configs compose, reach
    MORPHConfig and train; train.py subtracts `latent_pre_weighted` in both tuples and
    evaluates a stage-1 model without a token CE.

CPU, fp32, the tests/test_tul_fan.py / tests/test_tul_lxfan.py strict fixtures.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_latent_pre import (latent_infonce, latent_l2, latent_wta,
                                        next_span_grid, plain_final_targets,
                                        plain_prelude_targets, retrieval_top1)
from morph.model.tul_latent_pre import PLAIN_TARGET_FNS, standardise
from morph.training.latent_pre_ref import (VAL_DOC_OFFSET, calibrate_latent_pre_ref,
                                           check_calibration_docs)
from test_tul_fan import V, _batch, _tiny
from test_tul_fan_opf import _tuple_after
from test_tul_lxfan import _build, _fan_kw

_REF = dict(latent_pre_ref_config="notul_panel_norm_match_s1r",
            latent_pre_ref_ckpt="/nonexistent/step_5000.pt")
CONFIGS = {
    "tul_latent_pre_prelude_l2": ("plain_prelude", "l2", "stage1-plain-prelude-l2"),
    "tul_latent_pre_prelude_nce": ("plain_prelude", "infonce", "stage1-plain-prelude-infonce"),
    "tul_latent_pre_prelude_wta4": ("plain_prelude", "wta", "stage1-plain-prelude-wta4"),
    "tul_latent_pre_final_l2": ("plain_final", "l2", "stage1-plain-final-l2"),
    "tul_latent_pre_ema_l2": ("ema_prelude", "l2", "stage1-ema-prelude-l2"),
}
_CODA_PREFIXES = ("coda.", "lm_mixer.", "final_norm.", "tul.W_prefix", "tul.E_mask",
                  "tul_spandec")


def _lp_kw(target="plain_prelude", loss="l2", **kw) -> dict:
    """The stage-1 tul block on the tiny strict model: single cell, prefix_k 2."""
    base = dict(prefix_k=2, spandec=False, latent_pre_target=target, latent_pre_loss=loss,
                **(_REF if target != "ema_prelude" else {"latent_pre_target_norm": "ln"}))
    base.update(kw)
    return base


def _wta_kw(**kw) -> dict:
    return _fan_kw(4, fan_all_wta_lambda=0.0, fan_repel_lambda=0.0, fan_repel_mode="cos",
                   **_lp_kw(loss="wta", prefix_k=4, **kw))


def _ref(seed: int = 5) -> MORPHTransformer:
    torch.manual_seed(seed)
    # top_k 32 >= every CSA block count these fixtures reach (84 // 4 = 21): the padding
    # invariance `assert_padding_invariant` requires (the run's shapes meet it too).
    return MORPHTransformer(_tiny(d_ff=96, top_k=32)).float().eval()


def _cal_batches(prefix_k: int, n: int = 3, seed0: int = 100) -> list:
    """Calibration batches for the tiny fixture: distinct id draws, the fixture layout."""
    return [_batch(prefix_k, seed=seed0 + i)[1:] for i in range(n)]


def _model(kw: dict, ref: bool = True) -> MORPHTransformer:
    m = _build(**kw)
    if m._latent_pre_mode in ("plain_prelude", "plain_final") and ref:
        r = _ref()
        m.tul_latent_pre_attach_ref(r)
        if m.cfg.tul.latent_pre_target_norm == "standard":
            calibrate_latent_pre_ref(r, m._latent_pre_mode, _cal_batches(_pk(kw)), "cpu")
    if m._latent_pre_mode == "ema_prelude":
        assert m.tul_fan_target_build()
    return m


def _pk(kw: dict) -> int:
    return int(kw["prefix_k"])


# ── 1. off is nothing ────────────────────────────────────────────────────────────────


def test_off_builds_nothing():
    m = _build(prefix_k=2)
    assert m._latent_pre_mode == "off" and m.tul_latent_pre_head is None
    assert m.latent_pre_ref is None
    assert not any(k.startswith("tul_latent_pre") for k in m.state_dict())


# ── 2. the target is frozen ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("target", ["plain_prelude", "plain_final"])
def test_the_plain_target_is_frozen_through_an_optimizer_step(target):
    kw = _lp_kw(target)
    m = _model(kw).train()
    ref = m.latent_pre_ref
    assert all(not p.requires_grad for p in ref.parameters()) and not ref.training
    live = {id(p) for p in m.parameters()}
    assert not live & {id(p) for p in ref.parameters()}, "the ref is a live parameter"
    assert not any(n.startswith("_latent_pre_ref") for n, _ in m.named_modules())
    before = {k: v.clone() for k, v in ref.state_dict().items()}
    mu0, sg0 = ref.latent_pre_mu.clone(), ref.latent_pre_sigma.clone()
    _ids, inp, lab, lay = _batch(_pk(kw))
    fn = plain_prelude_targets if target == "plain_prelude" else plain_final_targets
    z0, _ok = fn(ref, inp, lab, lay)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    out = m(inp, labels=lab, slot_layout=lay)
    out["loss"].backward()
    assert all(p.grad is None for p in ref.parameters())
    opt.step()
    for k, v in ref.state_dict().items():
        assert torch.equal(v, before[k]), k
    z1, _ = fn(ref, inp, lab, lay)
    assert torch.equal(z0, z1)
    assert torch.equal(ref.latent_pre_mu, mu0) and torch.equal(ref.latent_pre_sigma, sg0)
    # ...and the live model DID move (the step was not a no-op)
    assert m.tul_latent_pre_head.fc1.weight.grad is not None


def test_a_ref_put_back_in_train_mode_is_refused():
    kw = _lp_kw()
    m = _model(kw).train()
    m.latent_pre_ref.train()
    _ids, inp, lab, lay = _batch(2)
    with pytest.raises(RuntimeError, match="no longer frozen"):
        m(inp, labels=lab, slot_layout=lay)


# ── 3. the target of slot s is span s+1's ────────────────────────────────────────────


def _span_positions(lay, b: int, k: int) -> np.ndarray:
    """Packed positions of span k's tokens in row b (bag_id == k, not a slot position)."""
    bag = lay.bag_id[b].numpy()
    tok = ~lay.slot_mask[b].numpy()
    return np.nonzero((bag == k) & tok)[0]


def _perturb(inp, b: int, pos: int) -> torch.Tensor:
    out = inp.clone()
    out[b, pos] = 5 if int(inp[b, pos]) != 5 else 6
    return out


def test_the_grid_is_the_packers_next_span():
    """Independent of the grid's arithmetic: for every valid slot, the ids equal the input
    ids at the packed positions of span s+1 (bag_id == s+1, token, labelled)."""
    _ids, inp, lab, lay = _batch(2)
    g = next_span_grid(inp, lab, lay)
    B, S, W = g["ids"].shape
    n_checked = 0
    for b in range(B):
        for s in range(S):
            pos = _span_positions(lay, b, s + 1)
            pos = pos[lab[b, pos].numpy() >= 0]
            if bool(g["ok"][b, s]):
                n = int(g["valid"][b, s].sum())
                assert n == len(pos) and n > 0
                assert torch.equal(g["ids"][b, s, :n], inp[b, pos])
                assert int(g["last"][b, s]) == int(pos[-1])
                n_checked += 1
            else:
                assert not bool(g["valid"][b, s].any())
    assert n_checked >= 6


def test_prelude_target_is_span_local_and_aligned():
    _ids, inp, lab, lay = _batch(2)
    ref = _ref()
    z, ok = plain_prelude_targets(ref, inp, lab, lay)
    b, s = 0, 2
    assert bool(ok[b, s]) and bool(ok[b, s + 1])
    # (i) equals the plain model's front on span s+1 ALONE, unpadded
    pos = _span_positions(lay, b, s + 1)
    with torch.no_grad():
        x, _, _ = ref._front_region(inp[b, pos].unsqueeze(0))
    want = F.layer_norm(x.float().mean(dim=2).mean(dim=1), (x.shape[-1],))[0]
    torch.testing.assert_close(z[b, s], want, rtol=1e-5, atol=1e-5)
    # (ii) perturb span s+1: slot s moves, NO other slot of either row does
    z1, _ = plain_prelude_targets(ref, _perturb(inp, b, int(pos[1])), lab, lay)
    moved = (z1 - z).abs().amax(dim=-1) > 1e-6
    assert moved[b, s] and int(moved.sum()) == 1, moved
    # (iii) perturb span s+2: slot s does not move (slot s+1 does)
    pos2 = _span_positions(lay, b, s + 2)
    z2, _ = plain_prelude_targets(ref, _perturb(inp, b, int(pos2[1])), lab, lay)
    moved2 = (z2 - z).abs().amax(dim=-1) > 1e-6
    assert not moved2[b, s] and moved2[b, s + 1] and int(moved2.sum()) == 1


def test_final_target_reads_the_plain_row_at_the_next_spans_last_token():
    _ids, inp, lab, lay = _batch(2)
    ref = _ref()
    z, ok = plain_final_targets(ref, inp, lab, lay)
    b, s = 1, 3
    assert bool(ok[b, s])
    # (i) equals the plain model on the row's TOKENS alone (no slot ids, no padding),
    # read at the plain index of span s+1's last token
    tok = ~lay.slot_mask[b]
    row = inp[b][tok].unsqueeze(0)
    last_pos = int(_span_positions(lay, b, s + 1)[-1])
    plain_idx = int(tok[:last_pos].sum())
    with torch.no_grad():
        x, x0, bg = ref._front_region(row)
        h = ref._back_region(ref._core_region(x, x0, bg, row), x0, bg, row)
    want = F.layer_norm(h[0, plain_idx].float(), (h.shape[-1],))
    torch.testing.assert_close(z[b, s], want, rtol=1e-4, atol=1e-4)
    # (ii) perturb span s+1: slot s moves, slots < s (and the other row) do not
    pos = _span_positions(lay, b, s + 1)
    z1, _ = plain_final_targets(ref, _perturb(inp, b, int(pos[0])), lab, lay)
    moved = (z1 - z).abs().amax(dim=-1) > 1e-6
    assert moved[b, s] and not moved[b, :s].any() and not moved[1 - b].any()
    # (iii) perturb span s+2: slot s does not move
    pos2 = _span_positions(lay, b, s + 2)
    z2, _ = plain_final_targets(ref, _perturb(inp, b, int(pos2[0])), lab, lay)
    moved2 = (z2 - z).abs().amax(dim=-1) > 1e-6
    assert not moved2[b, : s + 1].any() and moved2[b, s + 1]


def test_the_predictor_sees_no_token_of_the_next_span():
    """The note's acceptance test (strict geometry): slot s's prediction at the exit does
    not move when a token of span s+1 changes, and DOES move when a token of span s does
    (non-vacuous)."""
    m = _model(_lp_kw()).eval()
    preds = []
    orig = m._latent_pre_predict
    m._latent_pre_predict = lambda *a: (lambda r: (preds.append(r), r)[1])(orig(*a))
    _ids, inp, lab, lay = _batch(2)
    b, s = 0, 2

    def _exit(ids):
        preds.clear()
        with torch.no_grad():
            m(ids, labels=lab, slot_layout=lay)
        return preds[0][b, :, 0]                           # the exit call comes first

    p0 = _exit(inp)
    p_next = _exit(_perturb(inp, b, int(_span_positions(lay, b, s + 1)[1])))
    assert torch.equal(p_next[: s + 1], p0[: s + 1])
    p_own = _exit(_perturb(inp, b, int(_span_positions(lay, b, s)[1])))
    assert (p_own[s] - p0[s]).abs().max() > 1e-6


def test_a_padding_dependent_ref_is_refused():
    """At top_k < n_blocks the CSA selection depends on right-padding (the reason for the
    guard): the tiny default (top_k 8) on the 84-position packed row is refused."""
    _ids, inp, lab, lay = _batch(2)
    torch.manual_seed(5)
    small = MORPHTransformer(_tiny(d_ff=96)).float().eval()
    with pytest.raises(RuntimeError, match="right-padding"):
        plain_final_targets(small, inp, lab, lay)


# ── 4. retrieval and its chance ──────────────────────────────────────────────────────


def _rand_target(B=3, S=7, C=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = F.layer_norm(torch.randn(B, S, C, generator=g), (C,))
    ok = torch.zeros(B, S, dtype=torch.bool)
    ok[0, :5] = True
    ok[1, :2] = True
    ok[2, 1:7] = True
    return z, ok


def test_retrieval_chance_is_one_over_the_actual_candidates_and_perfect_is_one():
    z, ok = _rand_target()
    r = retrieval_top1(z, z, ok)
    assert float(r["retr_same"]) == 1.0 and float(r["retr_all"]) == 1.0
    n_row = ok.sum(dim=1).float()                            # 5, 2, 6
    want_same = float((n_row * (1.0 / n_row)).sum() / ok.sum())  # mean over queries of 1/n_row
    assert float(r["chance_same"]) == pytest.approx(want_same)
    assert float(r["chance_same"]) == pytest.approx((5 / 5 + 2 / 2 + 6 / 6) / 13)
    assert float(r["chance_all"]) == pytest.approx(1 / 13)
    # a predictor that names its NEXT slot's target in the row never hits its own
    r2 = retrieval_top1(torch.roll(z, 1, dims=1), z, ok)
    assert float(r2["retr_same"]) < 0.5
    # M cells: one cell exact, the rest noise -> best-of-M 1.0, its chance from counts
    g = torch.Generator().manual_seed(1)
    cells = torch.randn(3, 7, 4, 16, generator=g)
    cells[:, :, 2] = z
    r3 = retrieval_top1(cells, z, ok)
    assert float(r3["retr_best_same"]) == 1.0 and float(r3["retr_best_all"]) == 1.0
    assert float(r3["chance_best_all"]) == pytest.approx(1 - (1 - 1 / 13) ** 4)


# ── 5. the losses equal hand computations ────────────────────────────────────────────


def test_l2_infonce_and_wta_equal_hand_computations():
    z, ok = _rand_target()
    g = torch.Generator().manual_seed(3)
    pred = torch.randn(3, 7, 16, generator=g)
    idx = [(b, s) for b in range(3) for s in range(7) if ok[b, s]]
    # L2
    l2 = sum(float(((pred[b, s] - z[b, s]) ** 2).mean()) for b, s in idx) / len(idx)
    assert float(latent_l2(pred, z, ok)) == pytest.approx(l2, rel=1e-6)
    # InfoNCE: every ok slot of the batch is a candidate (same row AND other rows)
    tau = 0.1
    tot = 0.0
    for b, s in idx:
        p = F.normalize(pred[b, s], dim=0)
        logits = torch.tensor([float(p @ F.normalize(z[bb, ss], dim=0)) / tau
                               for bb, ss in idx])
        own = idx.index((b, s))
        tot += float(-(logits[own] - torch.logsumexp(logits, 0)))
    assert float(latent_infonce(pred, z, ok, tau)) == pytest.approx(tot / len(idx), rel=1e-5)
    # relaxed WTA over M = 4 cells
    cells = torch.randn(3, 7, 4, 16, generator=g)
    eps = 0.05
    tot = 0.0
    for b, s in idx:
        d = [float(((cells[b, s, i] - z[b, s]) ** 2).mean()) for i in range(4)]
        w = int(np.argmin(d))
        tot += (1 - eps) * d[w] + eps / 3 * sum(d[i] for i in range(4) if i != w)
    loss, winner = latent_wta(cells, z, ok, eps)
    assert float(loss) == pytest.approx(tot / len(idx), rel=1e-6)
    assert winner.shape == (3, 7)


# ── 6. no coda at train or eval ──────────────────────────────────────────────────────


@pytest.mark.parametrize("kw", [_lp_kw(), _lp_kw(loss="infonce"), _wta_kw(),
                                _lp_kw("plain_final"), _lp_kw("ema_prelude")],
                         ids=["prelude-l2", "prelude-nce", "prelude-wta", "final-l2", "ema-l2"])
def test_no_coda_parameter_gets_a_gradient_and_no_coda_runs(kw):
    m = _model(kw).train()
    calls = []
    orig = m._back_region
    m._back_region = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    _ids, inp, lab, lay = _batch(_pk(kw))
    out = m(inp, labels=lab, slot_layout=lay)
    out["loss"].backward()
    assert not calls, "the stage-1 TRAIN forward ran the coda"
    got = {n for n, p in m.named_parameters() if p.grad is not None}
    coda = sorted(n for n in got if n.startswith(_CODA_PREFIXES))
    assert not coda, coda
    # non-vacuous: the head, the prelude and the loop DO train
    for want in ("tul_latent_pre_head.fc1.weight", "prelude.0.", "core.0."):
        assert any(n.startswith(want) for n in got), want
    assert "ce_tokens" not in out and "logits" in out and out["logits"] is None
    m.eval()
    with torch.no_grad():
        ev = m(inp, labels=lab, slot_layout=lay)
    assert not calls, "the stage-1 EVAL forward ran the coda"
    T = int(m.cfg.tul.slot_mean_depth or m.cfg.mean_depth)
    for t in range(T + 1):
        for k in ("retr_same", "retr_all", "r2", "pred_rank"):
            assert f"latent_pre_{k}_t{t}" in ev, f"{k}_t{t}"
    for k in ("chance_same", "chance_all", "target_rank", "retr_same_exit"):
        assert f"latent_pre_{k}" in ev
    if m.cfg.tul.latent_pre_loss == "wta":
        assert "latent_pre_retr_best_same_t1" in ev and "latent_pre_cell_spread_t1" in ev
    # at eval the depth is deterministic, so the last pass IS the exit
    assert float(ev[f"latent_pre_retr_same_t{T}"]) == float(ev["latent_pre_retr_same_exit"])


def test_the_loss_is_the_latent_term_plus_the_loop_constraint():
    kw = _lp_kw()
    m = _model(kw).train()
    _ids, inp, lab, lay = _batch(2)
    out = m(inp, labels=lab, slot_layout=lay)
    parts = float(out["latent_pre_weighted"]) + float(out.get("gain_reg_weighted", 0.0)) \
        + float(out.get("fp_weighted", 0.0))
    assert float(out["loss"]) == pytest.approx(parts, rel=1e-6)
    assert "ce_main" not in out


def test_the_ema_control_steps_its_twin():
    m = _model(_lp_kw("ema_prelude")).train()
    twin = m.__dict__["_fan_target"]
    before = {k: v.clone() for k, v in twin.state_dict().items()}
    _ids, inp, lab, lay = _batch(2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    m(inp, labels=lab, slot_layout=lay)["loss"].backward()
    opt.step()
    m.tul_fan_after_step()
    assert any(not torch.equal(v, before[k]) for k, v in twin.state_dict().items()
               if v.is_floating_point())


# ── 7. refusals ──────────────────────────────────────────────────────────────────────


def test_refusals():
    with pytest.raises(ValueError, match="latent_pre_target='off'"):
        TULConfig(latent_pre_loss="infonce")
    with pytest.raises(ValueError, match="frozen plain"):
        TULConfig(**{**_lp_kw(), "latent_pre_ref_ckpt": ""}, tg_geometry="strict")
    with pytest.raises(ValueError, match="ema_prelude"):
        TULConfig(**{**_lp_kw("ema_prelude"), **_REF}, tg_geometry="strict")
    with pytest.raises(NotImplementedError, match="write-all fan"):
        TULConfig(**_lp_kw(loss="wta"), tg_geometry="strict")
    with pytest.raises(NotImplementedError, match="ONE prediction"):
        TULConfig(**{**_wta_kw(), "latent_pre_loss": "l2", "latent_pre_eps": 0.05},
                  tg_geometry="strict")
    with pytest.raises(NotImplementedError, match="spandec=True"):
        TULConfig(**{**_lp_kw(), "spandec": True}, tg_geometry="strict")
    with pytest.raises(NotImplementedError, match="strict"):
        TULConfig(**_lp_kw())
    with pytest.raises(ValueError, match="fan_all_wta_lambda"):
        TULConfig(**{**_wta_kw(), "fan_all_wta_lambda": 1.0}, tg_geometry="strict")
    with pytest.raises(ValueError, match="non-InfoNCE"):
        TULConfig(**_lp_kw(latent_pre_tau=0.2), tg_geometry="strict")


def test_forward_refusals():
    kw = _lp_kw()
    m = _model(kw).eval()
    _ids, inp, lab, lay = _batch(2)
    with torch.no_grad():
        with pytest.raises(ValueError, match="label-free"):
            m(inp, slot_layout=lay)
        with pytest.raises(NotImplementedError, match="plan_mode"):
            m.tul_forward_ablated(inp, lab, lay, plan_mode="zero")
    bare = _model(kw, ref=False).train()
    with pytest.raises(RuntimeError, match="never attached"):
        bare(inp, labels=lab, slot_layout=lay)
    with pytest.raises(ValueError, match="PLAIN"):
        bare.tul_latent_pre_attach_ref(_build(prefix_k=2))
    with pytest.raises(RuntimeError, match="only the plain_"):
        _model(_lp_kw("ema_prelude")).tul_latent_pre_attach_ref(_ref())


# ── 8. the loader ────────────────────────────────────────────────────────────────────


_TINY_MODEL = dict(d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256,
                   context_len=256, n_prelude=2, n_core=2, n_coda=2, mean_depth=2,
                   max_depth=3, bptt_depth=3, channel_dims=[32, 20, 12], compression=2,
                   csa_compress_ratio=4, hca_compress_ratio=8, top_k=8, window_size=16,
                   retention=False, bigram_hash_vocab=V, use_kernels=False,
                   hc_use_kernel=False, d_ff=96)


def _tiny_plain_cfg():
    from omegaconf import OmegaConf

    from morph.training.latent_pre_ref import compose_ref_config
    cfg = compose_ref_config("notul_panel_norm_match_s1r")
    for k, v in _TINY_MODEL.items():
        OmegaConf.update(cfg, f"model.{k}", v, force_add=True)
    return cfg


def test_loader_loads_strictly_quantised_and_refuses(tmp_path):
    from omegaconf import OmegaConf

    from morph.training.latent_pre_ref import assert_not_live, load_latent_pre_ref
    from morph.training.quant_setup import apply_quantization
    from morph.training.train import build_morph_config
    cfg = _tiny_plain_cfg()
    assert bool(cfg.training.ternary), "the plain run is ternary: the load must quantise"
    torch.manual_seed(11)
    src = MORPHTransformer(build_morph_config(cfg, tul=None))
    apply_quantization(src, cfg)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(0.01 * torch.randn_like(p))
    sd = src.state_dict()
    path = tmp_path / "step_5000.pt"
    torch.save({"model": sd, "step": 5000}, path)
    rng0 = torch.random.get_rng_state()
    ref = load_latent_pre_ref(cfg, str(path), "cpu", 64, "tiny")
    assert torch.equal(torch.random.get_rng_state(), rng0), "the build moved the CPU RNG"
    got = ref.state_dict()
    assert got.keys() == sd.keys()
    for k, v in sd.items():
        assert torch.equal(got[k], v), k
    assert not ref.training and all(not p.requires_grad for p in ref.parameters())
    # refusals: missing file, a missing tensor, a d_model mismatch, a TUL config, live dir
    with pytest.raises(FileNotFoundError):
        load_latent_pre_ref(cfg, str(tmp_path / "nope.pt"), "cpu", 64, "tiny")
    bad = dict(sd)
    bad.pop(sorted(bad)[3])
    torch.save({"model": bad}, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError, match="1 missing"):
        load_latent_pre_ref(cfg, str(tmp_path / "bad.pt"), "cpu", 64, "tiny")
    with pytest.raises(ValueError, match="d_model"):
        load_latent_pre_ref(cfg, str(path), "cpu", 128, "tiny")
    tcfg = cfg.copy()
    OmegaConf.update(tcfg, "tul.activate_at", 0.0, force_add=True)
    with pytest.raises(ValueError, match="PLAIN"):
        load_latent_pre_ref(tcfg, str(path), "cpu", 64, "tiny")
    with pytest.raises(ValueError, match="live copy"):
        assert_not_live(str(path), str(tmp_path))
    assert_not_live(str(path), str(tmp_path / "other_run"))


# ── 9. configs and train.py ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_configs_compose_reach_the_model_and_train(name, monkeypatch):
    import dataclasses

    from omegaconf import OmegaConf
    from test_slot_gain_tail import _MISSING, _leaves
    from test_tul_strict_geometry import _runtime

    from morph.training.train import build_morph_config
    target, loss, wb = CONFIGS[name]
    cfg, rt = _runtime(name, monkeypatch)
    bcfg, brt = _runtime("tul_latent_pre_base", monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(bcfg, resolve=True))
    diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
    want = {"wandb.name"} | {
        "tul_latent_pre_prelude_l2": set(),
        "tul_latent_pre_prelude_nce": {"tul.latent_pre_loss", "tul.latent_pre_tau"},
        "tul_latent_pre_prelude_wta4": {"tul.latent_pre_loss", "tul.latent_pre_eps",
                                        "tul.prefix_k", "tul.fan_k", "tul.slot_cell_init",
                                        "tul.fan_mix", "tul.fan_all_wta_lambda"},
        "tul_latent_pre_final_l2": {"tul.latent_pre_target"},
        "tul_latent_pre_ema_l2": {"tul.latent_pre_target", "tul.latent_pre_ref_config",
                                  "tul.latent_pre_ref_ckpt", "tul.latent_pre_target_norm"},
    }[name]
    assert diff == want, sorted(diff)
    assert c["wandb.name"] == wb
    assert c["training.steps"] == 3000 and c["training.seed"] == 1
    assert c["training.gen_every"] == 0 and c["model.core_fixed_point_lambda"] == 0.1
    mc = build_morph_config(cfg, tul=rt.model_cfg)
    assert mc.tul.latent_pre_target == target and mc.tul.latent_pre_loss == loss
    assert mc.tul.tg_geometry == "strict" and not mc.tul.spandec
    if loss == "wta":
        assert mc.tul.fan_k == 4 and mc.tul.slot_cells == 4 and mc.tul.prefix_k == 4
    if target != "ema_prelude":
        assert mc.tul.latent_pre_ref_config == "notul_panel_norm_match_s1r"
        assert mc.tul.latent_pre_ref_ckpt.endswith("plain-panel-nm-ctrl-s1r/step_5000.pt")
    for k in ("latent_pre_target", "latent_pre_loss", "latent_pre_tau", "latent_pre_eps",
              "latent_pre_hidden", "latent_pre_ref_config", "latent_pre_ref_ckpt",
              "latent_pre_target_norm", "latent_pre_cal_batches",
              "latent_pre_cal_doc_offset"):
        assert rt.manifest[k] == getattr(mc.tul, k), k
    assert mc.tul.latent_pre_target_norm == ("ln" if target == "ema_prelude" else "standard")
    if target != "ema_prelude":
        assert mc.tul.latent_pre_cal_batches == 64
        assert mc.tul.latent_pre_cal_doc_offset == 40_000
    bmc = build_morph_config(bcfg, tul=brt.model_cfg)
    tdiff = {f.name for f in dataclasses.fields(mc.tul)
             if getattr(mc.tul, f.name) != getattr(bmc.tul, f.name)}
    print(f"[latent-pre] {name}: TULConfig diff vs the base = {sorted(tdiff)}")
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=rt.model_cfg, d_ff=96,
                               core_fixed_point_lambda=mc.core_fixed_point_lambda)).train()
    m = m.float()
    if target != "ema_prelude":
        r = _ref()
        m.tul_latent_pre_attach_ref(r)
        calibrate_latent_pre_ref(r, target, _cal_batches(rt.model_cfg.prefix_k), "cpu")
    else:
        assert m.tul_fan_target_build()
    _ids0, inp, lab, layout = _batch(rt.model_cfg.prefix_k)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and "latent_pre_weighted" in out


def test_both_train_py_subtraction_tuples_list_the_key():
    src = pathlib.Path("morph/training/train.py").read_text()
    assert "latent_pre_weighted" in _tuple_after(src, "_aux2")
    assert "latent_pre_weighted" in _tuple_after(src, "_ak")


class _Layout:
    stats: dict = {}

    def to(self, device):
        return self


class _Stage1Stub(torch.nn.Module):
    class cfg:
        class tul:
            latent_pre_target = "plain_prelude"

    def tul_forward_with_plan_nats(self, x, y, layout):
        return {"loss": torch.tensor(3.5), "latent_pre": torch.tensor(1.25),
                "latent_pre_weighted": torch.tensor(1.25),
                "latent_pre_retr_same_t1": torch.tensor(0.3),
                "latent_pre_chance_same": torch.tensor(0.1), "gain_est": torch.tensor(0.9)}


def test_evaluate_scores_a_stage1_model_without_a_token_ce():
    from morph.training.train import evaluate
    x = torch.zeros(1, 4, dtype=torch.long)
    extra: dict = {}
    avg, _ppl = evaluate(_Stage1Stub(), torch.device("cpu"), iter([(x, x, _Layout())] * 2),
                         n_batches=2, tul=True, extra=extra)
    assert avg == pytest.approx(1.25)
    assert extra["val/latent_pre_retr_same_t1"] == pytest.approx(0.3)
    assert extra["val/latent_pre_chance_same"] == pytest.approx(0.1)
    assert extra["val/gain_est"] == pytest.approx(0.9)
    assert "val/ce_tokens" not in extra


# ── 10. the fixed standardisation (tul.latent_pre_target_norm: standard) ─────────────


@pytest.mark.parametrize("mode", ["plain_prelude", "plain_final"])
def test_calibration_is_deterministic_and_standardises_its_own_set(mode):
    """Two calibrations of the same frozen model on the same batches give bit-identical
    statistics; the calibration set's standardised targets have per-coordinate mean 0
    (|mean| < 1e-4) and population std 1 (|std - 1| < 1e-3) on every coordinate the floor
    did not touch, and std < 1 on the floored ones. The buffers are not persistent."""
    batches = _cal_batches(2, n=4)
    r1, r2 = _ref().requires_grad_(False), _ref().requires_grad_(False)
    rng0 = torch.random.get_rng_state()
    st1 = calibrate_latent_pre_ref(r1, mode, batches, "cpu")
    st2 = calibrate_latent_pre_ref(r2, mode, batches, "cpu")
    assert torch.equal(torch.random.get_rng_state(), rng0), "calibration moved the CPU RNG"
    assert torch.equal(r1.latent_pre_mu, r2.latent_pre_mu)
    assert torch.equal(r1.latent_pre_sigma, r2.latent_pre_sigma)
    assert st1 == st2
    assert "latent_pre_mu" not in r1.state_dict() and "latent_pre_sigma" not in r1.state_dict()
    fn = PLAIN_TARGET_FNS[mode]
    rows = []
    for x, y, lay in batches:
        z, ok = fn(r1, x, y, lay)
        rows.append(standardise(z, ok, r1.latent_pre_mu, r1.latent_pre_sigma)[ok])
    zs = torch.cat(rows).double()
    assert zs.shape[0] == st1["n"] and st1["n"] >= 20
    floor = 1e-3 * float(r1.latent_pre_sigma.median())
    live = r1.latent_pre_sigma > floor * (1 + 1e-6)
    assert int((~live).sum()) == st1["n_floored"]
    assert float(zs.mean(0).abs().max()) < 1e-4
    sd = zs.std(0, unbiased=False)
    assert float((sd[live] - 1).abs().max()) < 1e-3
    assert bool((sd[~live] <= 1 + 1e-3).all())
    # the readings are what the loss sees: the standardised spread is far wider than the LN
    assert st1["pr_standard"] > 0 and st1["sigma_min"] >= floor * (1 - 1e-6)


def test_the_model_reads_the_standardised_target_and_refuses_an_uncalibrated_ref():
    kw = _lp_kw()
    m = _model(kw).train()
    ref = m.latent_pre_ref
    _ids, inp, lab, lay = _batch(2)
    z, ok = m._latent_pre_target(inp, lab, lay, None, None, None)
    raw, ok2 = plain_prelude_targets(ref, inp, lab, lay)
    assert torch.equal(ok, ok2)
    torch.testing.assert_close(z, standardise(raw, ok, ref.latent_pre_mu, ref.latent_pre_sigma))
    assert not torch.allclose(z, raw)
    bare = _build(**kw).train()
    bare.tul_latent_pre_attach_ref(_ref())
    with pytest.raises(RuntimeError, match="carries no calibration"):
        bare(inp, labels=lab, slot_layout=lay)
    # ln: the raw LayerNormed target, no calibration needed
    m_ln = _model(_lp_kw(latent_pre_target_norm="ln")).train()
    z_ln, _ = m_ln._latent_pre_target(inp, lab, lay, None, None, None)
    torch.testing.assert_close(z_ln, plain_prelude_targets(m_ln.latent_pre_ref, inp, lab, lay)[0])


def test_target_norm_refusals():
    with pytest.raises(NotImplementedError, match="DRIFTS"):
        TULConfig(**{**_lp_kw("ema_prelude"), "latent_pre_target_norm": "standard"},
                  tg_geometry="strict")
    with pytest.raises(ValueError, match="only 'standard' calibrates"):
        TULConfig(**_lp_kw(latent_pre_target_norm="ln", latent_pre_cal_batches=8),
                  tg_geometry="strict")
    with pytest.raises(ValueError, match="latent_pre_target_norm must be one of"):
        TULConfig(**_lp_kw(latent_pre_target_norm="zscore"), tg_geometry="strict")
    with pytest.raises(ValueError, match="latent_pre_target='off'"):
        TULConfig(latent_pre_target_norm="ln")
    # the document ranges: into val, or a run that reaches the calibration documents
    with pytest.raises(ValueError, match="into the val"):
        check_calibration_docs(49_900, 200, 200_000, 10_000)
    with pytest.raises(ValueError, match="reaching the calibration"):
        check_calibration_docs(40_000, 400, 1_000_000, 40_000 * 2_500)
    info = check_calibration_docs(40_000, 400, 1_000_000, 3000 * 6 * 1024)
    assert info["doc_end"] == 40_400 and info["est_run_docs"] == 7372


def test_val_doc_offset_is_train_py_val_skip():
    src = pathlib.Path("morph/training/train.py").read_text()
    i = src.index("def _make_val_loader")
    assert f"skip_samples={VAL_DOC_OFFSET:_}" in src[i:i + 600]


def test_calibration_draw_leaves_the_training_stream_untouched():
    """The REAL draw (a separate process, the run's data config). Reference: the run's
    first two training batches drawn by a fresh loader in a CLEAN process (document 0, the
    trainer's own call). Then, in this process, the run's training loader serves batch 0, a
    calibration is drawn, and the loader serves batch 1: both bit-identical to the
    reference. The calibration's batches are other documents (from 40000) and the draw
    moved no RNG of this process."""
    from hydra import compose, initialize_config_dir

    from morph.training.data import create_dataloader
    from morph.training.latent_pre_ref import (_CFG_DIR, calibration_batches,
                                               draw_in_subprocess)
    from morph.training.tul_setup import build_tul_runtime
    with initialize_config_dir(version_base=None, config_dir=_CFG_DIR):
        cfg = compose(config_name="tul_latent_pre_prelude_l2")
    rt = build_tul_runtime(cfg)          # the REAL tokenizer and boundary rule
    args = (str(cfg.data.tokenizer), str(cfg.data.dataset), int(cfg.data.seq_len), 1)

    def _eq(a, b):
        return (torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                and torch.equal(a[2].slot_mask, b[2].slot_mask)
                and torch.equal(a[2].bag_id, b[2].bag_id))

    c0, c1 = draw_in_subprocess(*args, rt.data_cfg, 0, 2)["batches"]
    it = create_dataloader(*args, split="train", tul=rt.data_cfg)
    b0 = next(it)
    rng0 = torch.random.get_rng_state()
    cal, info = calibration_batches(*args, rt.data_cfg, 40_000, 2, run_tokens=1000)
    assert torch.equal(torch.random.get_rng_state(), rng0)
    b1 = next(it)
    assert _eq(b0, c0) and _eq(b1, c1)
    assert len(cal) == 2 and not torch.equal(cal[0][0], c0[0])
    assert info["doc_first"] == 40_000 and 40_000 < info["doc_end"] < VAL_DOC_OFFSET
