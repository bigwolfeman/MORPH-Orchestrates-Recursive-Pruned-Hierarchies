"""``tul.pointer_heads`` — the output-only pointer / copy head (2026-10-06).

morph/model/tul_pointer.py; wired in transformer.py (build LAST, `_tul_group_losses`, the
eval logits), tul.py (`_check_pointer`), tul_setup.py (key, manifest), the cached generator's
refusal, morph/configs/lxtul_pointer.yaml.

What each test pins:
  1. Off (absent or explicit 0) is the tree: tests/test_loop_attn_center.py's MASTER_PIN.
  2. The build is RNG-neutral: every tensor but `tul_pointer.*` equals the same-seed model
     without the key; every pointer leaf is never-ternary.
  3. With the gate forced onto the model (heads off), the training loss equals the model
     without the key (the per-token path reproduces the fused weighted CE).
  4. Training and eval agree: at eval, `ce_tokens` (the mixture at the true targets, labels
     path) equals the CE of the label-free eval logits at the same token positions.
  5. The eval logits are causal (future inputs never move an earlier token's log-probs) and
     normalised; a live training step reaches the head's weights.
  6. Config refusals, Hydra path, manifest, the config's compose diff, the cached refusal.

Sabotage checks, run and reverted (session report, not committed):
  (a) candidates j <= i in `attend` -> test 5 fails (causality: the head sees the target;
      the train and eval paths both leak the same way, so test 4 still agrees).
  (b) the abstain (null) mass is dropped instead of returned to the model (the first
      draft's bug: the mixture summed to less than 1) -> tests 4 and 5 fail.
CPU, fp32, the LXTUL fixtures (M = 4 cells, strict geometry).
"""
from __future__ import annotations

import pathlib

import pytest
import torch

from morph.model.tul import TULConfig
from test_loop_attn_center import _CONFIG_DIR, MASTER_PIN, _compose_leaves, _lx, _pin_run
from test_tul_fan import _batch
from test_tul_lx_credit import M

M = int(M)


def _live(m):
    with torch.no_grad():
        m.tul_pointer.gate_norm_head.bias.zero_()          # model and heads share the mass
    return m


@pytest.mark.parametrize("kw", [{}, dict(pointer_heads=0)])
def test_off_is_the_tree(kw):
    m = _lx(**kw)
    assert m.tul_pointer is None
    assert _pin_run(m) == MASTER_PIN


def test_build_is_rng_neutral_and_never_ternary():
    off, on = _lx(), _lx(pointer_heads=2)
    so, sn = off.state_dict(), on.state_dict()
    extra = set(sn) - set(so)
    assert extra and all(k.startswith("tul_pointer.") for k in extra), sorted(extra)
    assert set(so) <= set(sn)
    for k in so:
        assert torch.equal(so[k], sn[k]), k
    assert all(getattr(mod, "_ternary_exclude", False) for mod in on.tul_pointer.modules())


def test_heads_off_reproduces_the_fused_loss():
    _ids, inp, lab, layout = _batch(M)
    a, b = _lx(), _lx(pointer_heads=2)
    with torch.no_grad():
        b.tul_pointer.gate_norm_head.bias.copy_(torch.tensor([0.0, -1e4, -1e4]))
    la, lb = [], []
    for m, out in ((a, la), (b, lb)):
        m.train()
        torch.manual_seed(7)
        out.append(float(m(inp, labels=lab, slot_layout=layout)["loss"]))
    assert abs(la[0] - lb[0]) < 1e-4, (la, lb)


def _token_ce_from_logits(lg, lab, layout):
    tok = (~layout.slot_mask) & (lab >= 0)
    nll = torch.nn.functional.cross_entropy(lg.float().flatten(0, 1), lab.flatten().clamp_min(0),
                                            reduction="none").view(lab.shape)
    return float((nll * tok).sum() / tok.sum())


def test_training_target_and_eval_logits_agree():
    m = _live(_lx(pointer_heads=2))
    _ids, inp, lab, layout = _batch(M)
    m.eval()
    with torch.no_grad():
        o = m(inp, labels=lab, slot_layout=layout)
        lg = m(inp, labels=None, slot_layout=layout)["logits"]
    ce_eval = _token_ce_from_logits(lg, lab, layout)
    assert abs(float(o["ce_tokens"]) - ce_eval) < 1e-4, (float(o["ce_tokens"]), ce_eval)
    # cross_entropy renormalises; agreement needs the mixture to be normalised already
    p = lg.float().exp().sum(-1)[(~layout.slot_mask) & (lab >= 0)]
    assert torch.allclose(p, torch.ones_like(p), atol=1e-4)
    assert abs(float(o["ce_tokens"]) - float(o["ce_tokens_model"])) > 1e-4   # the head is live


def test_eval_logits_are_causal_normalised_and_the_head_trains():
    m = _live(_lx(pointer_heads=2))
    ids, inp, lab, layout = _batch(M)
    m.eval()
    with torch.no_grad():
        lg = m(inp, labels=None, slot_layout=layout)["logits"]
    tok = (~layout.slot_mask[0]).nonzero().squeeze(1)
    t = int(tok[len(tok) // 2])
    inp2 = inp.clone()
    fut = tok[tok >= t]
    inp2[0, fut] = inp[1, fut]                              # in-vocabulary changes
    with torch.no_grad():
        lg2 = m(inp2, labels=None, slot_layout=layout)["logits"]
    before = tok[tok < t]
    assert torch.equal(lg[0, before], lg2[0, before])
    p = lg[0, tok].float().exp().sum(-1)
    assert torch.allclose(p, torch.ones_like(p), atol=1e-4)
    m.train()
    torch.manual_seed(3)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    for prm in (m.tul_pointer.q.weight, m.tul_pointer.k.weight, m.tul_pointer.gate_norm_head.weight):
        assert prm.grad is not None and float(prm.grad.abs().sum()) > 0


def _cellkey_live(seed=11):
    m = _live(_lx(pointer_heads=2, pointer_cell_key=True))
    with torch.no_grad():
        m.tul_pointer.kc.weight.normal_(0, 0.3, generator=torch.Generator().manual_seed(seed))
    return m


def test_cell_key_zero_init_is_the_head_without_them():
    _ids, inp, lab, layout = _batch(M)
    a, b = _live(_lx(pointer_heads=2)), _live(_lx(pointer_heads=2, pointer_cell_key=True))
    b.load_state_dict(a.state_dict(), strict=False)
    for m in (a, b):
        m.eval()
    with torch.no_grad():
        la = a(inp, labels=None, slot_layout=layout)["logits"]
        lb = b(inp, labels=None, slot_layout=layout)["logits"]
    assert torch.equal(la, lb)


def test_cell_key_train_eval_agree_causal_and_live():
    m = _cellkey_live()
    ids, inp, lab, layout = _batch(M)
    m.eval()
    with torch.no_grad():
        o = m(inp, labels=lab, slot_layout=layout)
        lg = m(inp, labels=None, slot_layout=layout)["logits"]
    assert abs(float(o["ce_tokens"]) - _token_ce_from_logits(lg, lab, layout)) < 1e-4
    with torch.no_grad():
        m.tul_pointer.kc.weight.zero_()
        lg0 = m(inp, labels=None, slot_layout=layout)["logits"]
    assert not torch.allclose(lg, lg0, atol=1e-5)                       # the term is live
    m = _cellkey_live()
    m.eval()
    tok = (~layout.slot_mask[0]).nonzero().squeeze(1)
    t = int(tok[len(tok) // 2])
    inp2 = inp.clone()
    fut = tok[tok >= t]
    inp2[0, fut] = inp[1, fut]
    with torch.no_grad():
        a = m(inp, labels=None, slot_layout=layout)["logits"]
        b = m(inp2, labels=None, slot_layout=layout)["logits"]
    before = tok[tok < t]
    assert torch.equal(a[0, before], b[0, before])


def _cov_ref(alpha):
    """mean_h sum_j min(alpha_hij, sum_{i' < i} alpha_hi'j), by loops (See et al. eq. 12)."""
    B, H, N, _ = alpha.shape
    out = torch.zeros(B, N)
    for b in range(B):
        for i in range(N):
            tot = 0.0
            for h in range(H):
                c = alpha[b, h, :i].sum(0)
                tot += float(torch.minimum(alpha[b, h, i], c).sum())
            out[b, i] = tot / H
    return out


def test_coverage_term_starts_off_and_the_loss_matches_the_paper():
    from morph.model.tul_pointer import TULPointer, candidates_from_ids
    _ids, inp, lab, layout = _batch(M)
    torch.manual_seed(4)
    a = TULPointer(16, 2)
    torch.manual_seed(4)
    b = TULPointer(16, 2, coverage=True)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sb) - set(sa) == {"cov_gain_bias"}
    assert all(torch.equal(sa[k], sb[k]) for k in sa)               # built last: no RNG shift
    h = torch.randn(inp.shape[0], inp.shape[1], 16)
    cand = candidates_from_ids(inp, layout.slot_mask)
    with torch.no_grad():
        al_a, nu_a, _g, cov_a = a.attend(h, cand, layout.slot_mask)
        al_b, nu_b, _g, cov_b = b.attend(h, cand, layout.slot_mask)
        assert cov_a is None and torch.equal(al_a, al_b) and torch.equal(nu_a, nu_b)
        assert torch.allclose(cov_b, _cov_ref(al_b), atol=1e-5)
        assert float(cov_b.max()) <= 1.0 + 1e-6 and float(cov_b.sum()) > 0
        b.cov_gain_bias.fill_(-3.0)
        al_c, _n, _g, cov_c = b.attend(h, cand, layout.slot_mask)
        assert not torch.allclose(al_c, al_b)                         # the score term is live
        assert torch.allclose(cov_c, _cov_ref(al_c), atol=1e-5)       # loss on the final alpha


def _cov_live():
    m = _live(_lx(pointer_heads=2, pointer_coverage_lambda=1.0))
    with torch.no_grad():
        m.tul_pointer.cov_gain_bias.copy_(torch.tensor([-2.0, 1.5]))
    return m


def test_coverage_loss_is_in_the_loss_and_trains_the_gain():
    _ids, inp, lab, layout = _batch(M)
    on = _cov_live()
    on.train()
    torch.manual_seed(7)
    o = on(inp, labels=lab, slot_layout=layout)
    assert float(o["pointer_cov"]) > 0
    assert abs(float(o["pointer_cov_weighted"]) - float(o["pointer_cov"])) < 1e-6   # lambda 1
    o["loss"].backward()
    assert float(on.tul_pointer.cov_gain_bias.grad.abs().sum()) > 0
    # the CE part equals the loss minus the weighted term (the train.py subtraction contract)
    two = _lx(pointer_heads=2, pointer_coverage_lambda=2.0)
    two.load_state_dict(on.state_dict())
    two.train()
    torch.manual_seed(7)
    o2 = two(inp, labels=lab, slot_layout=layout)
    d1 = float(o["loss"]) - float(o["pointer_cov_weighted"])
    d2 = float(o2["loss"]) - float(o2["pointer_cov_weighted"])
    assert abs(d1 - d2) < 1e-5 and abs(float(o2["pointer_cov_weighted"]) - 2 * float(o["pointer_cov"])) < 1e-5


def test_coverage_eval_logits_agree_and_are_causal():
    m = _cov_live()
    ids, inp, lab, layout = _batch(M)
    m.eval()
    with torch.no_grad():
        o = m(inp, labels=lab, slot_layout=layout)
        lg = m(inp, labels=None, slot_layout=layout)["logits"]
    assert abs(float(o["ce_tokens"]) - _token_ce_from_logits(lg, lab, layout)) < 1e-4
    tok = (~layout.slot_mask[0]).nonzero().squeeze(1)
    t = int(tok[len(tok) // 2])
    inp2 = inp.clone()
    fut = tok[tok >= t]
    inp2[0, fut] = inp[1, fut]
    with torch.no_grad():
        lg2 = m(inp2, labels=None, slot_layout=layout)["logits"]
    before = tok[tok < t]
    assert torch.equal(lg[0, before], lg2[0, before])


def test_config_refusals():
    with pytest.raises(ValueError, match="pointer_heads must be >= 0"):
        TULConfig(pointer_heads=-1)
    with pytest.raises(ValueError, match="slot loop"):
        TULConfig(pointer_heads=2, tokens_through_core=True)
    with pytest.raises(ValueError, match="pointer_cell_key needs"):
        TULConfig(pointer_cell_key=True)
    with pytest.raises(ValueError, match="pointer_coverage_lambda needs"):
        TULConfig(pointer_coverage_lambda=1.0)
    with pytest.raises(ValueError, match="must be >= 0"):
        TULConfig(pointer_heads=2, pointer_coverage_lambda=-1.0)


def test_hydra_path_manifest_and_compose_diff(monkeypatch):
    import transformers
    from hydra import compose, initialize_config_dir

    from morph.training import tul_setup
    from test_slot_gain_tail import _MISSING
    from test_tul_strict_geometry import _StubTok, _rule as _strict_rule

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubTok)
    monkeypatch.setattr(tul_setup, "build_boundary_rule",
                        lambda cfg, cache_dir="": (_strict_rule(), _strict_rule().is_boundary,
                                                   0, ("\n",)))
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        cfg = compose(config_name="lxtul_pointer")
    rt = tul_setup.build_tul_runtime(cfg)
    assert rt.model_cfg.pointer_heads == 4 and rt.manifest["pointer_heads"] == 4
    a, b = _compose_leaves("lxtul_pointer"), _compose_leaves("lxtul")
    diff = {k for k in a.keys() | b.keys() if a.get(k, _MISSING) != b.get(k, _MISSING)}
    assert diff == {"tul.pointer_heads", "training.steps", "wandb.name"}, sorted(diff)
    c = _compose_leaves("lxtul_pointer_cellkey")
    diff = {k for k in a.keys() | c.keys() if a.get(k, _MISSING) != c.get(k, _MISSING)}
    assert diff == {"tul.pointer_cell_key", "wandb.name"}, sorted(diff)
    ft, cv = _compose_leaves("lxtul_pointer_ft"), _compose_leaves("lxtul_pointer_cov")
    diff = {k for k in ft.keys() | cv.keys() if ft.get(k, _MISSING) != cv.get(k, _MISSING)}
    assert diff == {"tul.pointer_coverage_lambda", "training.init_from_new_modules",
                    "wandb.name"}, sorted(diff)


def test_cached_generator_refuses_the_pointer():
    src = pathlib.Path("morph/inference/tul_generate_cached.py").read_text()
    assert 'int(getattr(tc, "pointer_heads", 0)) == 0' in src
