"""Arm B, the code policy (`tul.code_policy_k`, 2026-09-29): LX's codes, ONE rollout, one
code per slot chosen by a REINFORCE-trained policy head.

Files: morph/model/tul_code_policy.py (`TULCodePolicy`, `span_mean_ce`,
`policy_objective`), morph/model/tul_code_enum.py (`TULCodeEnum.term_per_slot`, `_size`),
morph/model/transformer.py (the build, `_code_policy_choose`, the per-pass add in
`_tul_core`, `_code_policy_loss`, the fold in `_forward_tul`, `code_policy_pick`),
morph/model/tul.py (`_check_code_policy`), morph/training/tul_setup.py and train.py (the
keys, the subtraction, the logs, the val-only `code_policy_vs_random` pass),
morph/configs/tul_slot_spandec_strict_e1probe_fp01{,_policy4}.yaml.

What each test pins:
  * OFF is byte-identical to the tree before the key (commit cbc4530, archived and run in
    a subprocess): train loss, every grad and the eval logits at dropout 0.1, for the K = 1
    control AND for LX K = 4 (the code-size refactor in `TULCodeEnum`);
  * ON builds RNG-neutrally (every shared weight equals the off model's), draws nothing
    from the global stream at build, and starts uniform;
  * `term_per_slot` with the rollout's code in every slot IS `term`;
  * the coda runs exactly once per forward at train and at eval, and the val-only random
    pass is one more;
  * each slot's exit moves by ITS OWN code's term (depth 1, forced choice), pads by nothing;
  * the reward is minus the mean per-token CE (slot id masked) of bag s+1;
  * REINFORCE reaches the policy and value heads only, the policy input has no graph, and
    the token CE still reaches the codes;
  * one SGD step on the policy loss raises the probability of a code with positive
    advantage and lowers it for a negative one;
  * real dropout > 0 trains, and two same-seed steps agree;
  * every refusal fires; both configs compose, differ from their parents by the stated key,
    reach TULConfig, build and train one step;
  * `code_policy_vs_random` is exactly 0 when the random draw is forced to the argmax, and
    nonzero when it is not;
  * train/loss and the val loss are the model's CE (the subtraction lists).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import morph.model.transformer as transformer_mod
from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_code_enum import TULCodeEnum
from morph.model.tul_code_policy import TULCodePolicy, policy_objective, span_mean_ce
from test_tul_lxtul_e import _D_FF, _Spy, _tc
from test_tul_strict_geometry import _pack, _runtime, _tiny, _tul

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_COMMIT = "cbc4530"          # the tip this key was built on
C = 4


def _model(policy_k: int = C, seed: int = 1234, dropout: float = 0.0, k: int = 1,
           **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tkw = dict(kw)
    if policy_k:
        tkw["code_policy_k"] = policy_k
    m = MORPHTransformer(_tiny(tul=_tc(k, **tkw), d_ff=_D_FF, dropout=dropout))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.float()


def _spread_codes(m: MORPHTransformer, seed: int = 5) -> None:
    """Move the codes' basis and both heads off their init, deterministically, so the codes
    are not the build's simplex, the policy is not uniform and the value head is not 0 (at
    v = 0 the value bias's gradient is EXACTLY 0: the leave-one-out residuals sum to 0)."""
    g = torch.Generator().manual_seed(seed)
    pol = m.tul_code_policy
    with torch.no_grad():
        pol.codes.basis.copy_(torch.randn(pol.codes.basis.shape, generator=g))
        pol.policy.weight.copy_(3.0 * torch.randn(pol.policy.weight.shape, generator=g))
        pol.value.weight.copy_(torch.randn(pol.value.weight.shape, generator=g))
        pol.value.bias.fill_(0.3)


# ── OFF: byte-identical to the tree before the key ──────────────────────────────────

_OLD_SCRIPT = r"""
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[1] + "/tests")
import torch
from test_tul_lxtul_e import _D_FF, _tc
from test_tul_strict_geometry import _pack, _tiny
from morph.model.transformer import MORPHTransformer

def run(k):
    torch.manual_seed(1234)
    m = MORPHTransformer(_tiny(tul=_tc(k), d_ff=_D_FF, dropout=0.1))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    m = m.float().train()
    _ids, inp, lab, layout = _pack()
    torch.manual_seed(77)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    grads = {n: hashlib.sha256(p.grad.detach().numpy().tobytes()).hexdigest()
             for n, p in m.named_parameters() if p.grad is not None}
    m.eval()
    with torch.no_grad():
        lg = m(inp, slot_layout=layout)["logits"]
    return {"loss": float(out["loss"].item()),
            "grads": grads,
            "logits": hashlib.sha256(lg.detach().numpy().tobytes()).hexdigest(),
            "keys": sorted(m.state_dict().keys())}

json.dump({"k1": run(1), "k4": run(4)}, open(sys.argv[2], "w"))
"""


def _run_new(k: int) -> dict:
    m = _model(policy_k=0, k=k, dropout=0.1).train()
    _ids, inp, lab, layout = _pack()
    torch.manual_seed(77)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    grads = {n: hashlib.sha256(p.grad.detach().numpy().tobytes()).hexdigest()
             for n, p in m.named_parameters() if p.grad is not None}
    m.eval()
    with torch.no_grad():
        lg = m(inp, slot_layout=layout)["logits"]
    return {"loss": float(out["loss"].item()), "grads": grads,
            "logits": hashlib.sha256(lg.detach().numpy().tobytes()).hexdigest(),
            "keys": sorted(m.state_dict().keys())}


@pytest.fixture(scope="module")
def old_ref():
    """Archive commit cbc4530's `morph/` and `tests/` (read-only `git archive`) and run the
    SAME forward/backward there in a separate process, so the two `morph` packages never
    meet in one interpreter."""
    tmp = Path(tempfile.mkdtemp(prefix="morph_policy_old_ref_"))
    tar = tmp / "old.tar"
    subprocess.run(["git", "archive", "-o", str(tar), OLD_COMMIT, "morph", "tests"],
                   cwd=REPO_ROOT, check=True, capture_output=True)
    subprocess.run(["tar", "-xf", str(tar)], cwd=tmp, check=True, capture_output=True)
    script = tmp / "_old_ref_script.py"
    script.write_text(_OLD_SCRIPT)
    out = tmp / "out.json"
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONPATH="")
    r = subprocess.run([sys.executable, str(script), str(tmp), str(out)], cwd=tmp,
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    ref = json.loads(out.read_text())
    shutil.rmtree(tmp)
    return ref


@pytest.mark.parametrize("k", [1, 4])
def test_off_is_byte_identical_to_the_tree_before_the_key(old_ref, k):
    """K = 1 is the control arm's model; K = 4 is LX, whose code size moved into
    `TULCodeEnum._size`. Loss, every gradient and the eval logits at dropout 0.1, with no
    seed reset between construction and the forward (so the build draws no extra RNG)."""
    ref, new = old_ref[f"k{k}"], _run_new(k)
    assert new["keys"] == ref["keys"]
    assert new["loss"] == ref["loss"]
    assert new["grads"] == ref["grads"]
    assert new["logits"] == ref["logits"]


# ── ON: the build ────────────────────────────────────────────────────────────────────

def test_on_builds_rng_neutral_uniform_and_never_ternarised():
    s0 = torch.get_rng_state()
    pol = TULCodePolicy(32, C, 0.1)
    assert torch.equal(torch.get_rng_state(), s0), "the build moved the global stream"
    off, on = _model(policy_k=0), _model(policy_k=C)
    so, sn = off.state_dict(), on.state_dict()
    extra = set(sn) - set(so)
    assert set(so) <= set(sn)
    assert extra == {"tul_code_policy.codes.basis", "tul_code_policy.policy.weight",
                     "tul_code_policy.policy.bias", "tul_code_policy.value.weight",
                     "tul_code_policy.value.bias"}, sorted(extra)
    for key in so:
        assert torch.equal(so[key], sn[key]), key
    # uniform at build: zero heads
    for t in (pol.policy.weight, pol.policy.bias, pol.value.weight, pol.value.bias):
        assert torch.equal(t, torch.zeros_like(t))
    # never ternarised, and the basis is in the no-decay group
    from morph.model.ternary_qat import _categorize
    for name, mod in on.tul_code_policy.named_modules():
        assert _categorize(f"tul_code_policy.{name}", mod, set()) is None, name
    from morph.training.optimizer import _split_by_decay
    _d, _nd, _dn, ndn = _split_by_decay(on)
    assert "tul_code_policy.codes.basis" in ndn


def test_term_per_slot_is_the_rollout_term_with_the_rollouts_code_in_every_slot():
    torch.manual_seed(0)
    enum = TULCodeEnum(16, C, 0.1)
    with torch.no_grad():
        enum.basis.normal_()
    B0, S = 2, 5
    h = torch.randn(C * B0, S, 4, 16)
    valid = torch.rand(C * B0, S) > 0.3
    codes = torch.arange(C).repeat_interleave(B0).view(-1, 1).expand(-1, S).contiguous()
    assert torch.equal(enum.term(h, valid, C), enum.term_per_slot(h, valid, codes))


# ── one coda pass ────────────────────────────────────────────────────────────────────

def test_the_coda_runs_once_at_train_and_eval_and_the_random_pass_is_one_more():
    m = _model().train()
    _ids, inp, lab, layout = _pack()
    spy = _Spy(m, "_back_region")
    out = m(inp, labels=lab, slot_layout=layout)
    assert len(spy.calls) == 1
    assert "code_policy_weighted" in out
    m.eval()
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout)
        assert len(spy.calls) == 2
        m.tul_forward_ablated(inp, lab, layout, code_policy_pick="random")
        assert len(spy.calls) == 3


# ── the per-slot code ────────────────────────────────────────────────────────────────

def test_each_slot_moves_by_its_own_code_only_and_pads_do_not_move(monkeypatch):
    """Depth 1 (a forced table), eval, the choice forced through `TULCodePolicy.sample`:
    slot 0 takes code 0, slot 1 code 1, every other slot code 2. The exit with the code
    minus the exit with the term zeroed is `0.1 * rms(f(h)) * u_{c_s}` in every stream of
    every valid slot, and exactly 0 on every pad slot. Moving slot 1 to code 3 leaves
    slot 0's delta bit-identical."""
    m = _model().eval()
    _spread_codes(m)
    pol = m.tul_code_policy
    _ids, inp, lab, layout = _pack()
    S = layout.slot_valid.shape[1]
    depth1 = torch.ones_like(layout.slot_index)

    def table(c1: int) -> torch.Tensor:
        t = torch.full(layout.slot_valid.shape, 2, dtype=torch.long)
        t[:, 0], t[:, 1] = 0, c1
        return t

    def exit_state(tab, zero_term: bool) -> torch.Tensor:
        monkeypatch.setattr(pol, "sample", lambda logits, u: tab.clone())
        if zero_term:
            monkeypatch.setattr(pol.codes, "term_per_slot",
                                lambda h, v, c: torch.zeros(*h.shape[:2], h.shape[-1],
                                                            dtype=h.dtype))
        spy = _Spy(m, "_tul_core")
        with torch.no_grad():
            m.tul_forward_ablated(inp, lab, layout, slot_depths=depth1,
                                  code_policy_pick="sample")
        monkeypatch.undo()
        return spy.outs[0][1]                       # h_slots [B, S, n, C]

    base = exit_state(table(1), zero_term=True)
    on = exit_state(table(1), zero_term=False)
    on3 = exit_state(table(3), zero_term=False)
    u = pol.codes.directions().detach()
    rms = base.float().flatten(2).pow(2).mean(-1).sqrt()          # [B, S]
    valid = layout.slot_valid
    assert bool((~valid).any()), "the fixture must have pad slots"
    for b in range(valid.shape[0]):
        for s in range(S):
            delta = on[b, s] - base[b, s]                           # [n, C]
            if not valid[b, s]:
                assert torch.equal(delta, torch.zeros_like(delta)), (b, s)
                continue
            c = int(table(1)[b, s])
            want = (0.1 * rms[b, s] * u[c]).expand_as(delta)
            torch.testing.assert_close(delta, want, rtol=1e-4, atol=1e-5)
    # slot 0's delta does not depend on slot 1's code at depth 1
    assert torch.equal(on[:, 0], on3[:, 0])
    assert not torch.equal(on[:, 1][valid[:, 1]], on3[:, 1][valid[:, 1]])


# ── the reward ───────────────────────────────────────────────────────────────────────

def test_the_reward_is_the_mean_token_ce_of_the_span_the_slot_feeds(monkeypatch):
    """By hand: logits = xh @ W^T with the slot id's logit removed, CE per token, averaged
    over the positions of bag s+1 that are tokens with a label. `accumulate_span_ce`
    (summed, slot id NOT masked) differs from the model's reward only by that mask; the
    hand check here applies the mask the trained CE applies."""
    m = _model().eval()
    _spread_codes(m)
    _ids, inp, lab, layout = _pack()
    seen = {}
    real = transformer_mod.policy_objective

    def spy(logits, value, codes, reward, mask, *a):
        seen["reward"], seen["mask"] = reward.detach().clone(), mask.clone()
        return real(logits, value, codes, reward, mask, *a)
    monkeypatch.setattr(transformer_mod, "policy_objective", spy)
    coda = _Spy(m, "_back_region")
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout)
    xh = coda.outs[0]
    w = m.embed.lm_weight()
    logits = (xh.float() @ w.float().t())
    logits[..., m.cfg.tul.slot_id] = float("-inf")
    B, L = lab.shape
    ce = F.cross_entropy(logits.reshape(B * L, -1), lab.clamp_min(0).reshape(-1),
                         reduction="none").view(B, L)
    tok = (lab >= 0) & (~layout.slot_mask)
    S = layout.slot_valid.shape[1]
    n_checked = 0
    for b in range(B):
        for s in range(S):
            sel = tok[b] & (layout.bag_id[b] == s + 1)
            has = bool(sel.any())
            assert bool(seen["mask"][b, s]) == (has and bool(layout.slot_valid[b, s])), (b, s)
            if has and layout.slot_valid[b, s]:
                torch.testing.assert_close(seen["reward"][b, s], -ce[b][sel].mean(),
                                           rtol=1e-5, atol=1e-5)
                n_checked += 1
    assert n_checked >= 6
    # the pure helper against a hand-made table
    tok_ce = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    gid = torch.tensor([[0, 1, 1, 2, 0]])          # bin 0 = unscored dump, bags 1 and 2
    keep = torch.tensor([[False, True, True, True, False]])
    mean, has = span_mean_ce(tok_ce, gid, keep, 3)
    assert torch.equal(mean, torch.tensor([[2.5, 4.0]])) and bool(has.all())


# ── gradient routes ──────────────────────────────────────────────────────────────────

def test_reinforce_reaches_only_the_heads_and_the_ce_still_reaches_the_codes(monkeypatch):
    m = _model().train()
    _spread_codes(m)
    _ids, inp, lab, layout = _pack()
    seen = {}
    real_obj = transformer_mod.policy_objective
    real_feat = TULCodePolicy.features

    def obj(*a):
        loss, st = real_obj(*a)
        seen["loss"] = loss
        return loss, st

    def feat(e):
        x = real_feat(e)
        seen["x"] = x
        return x
    monkeypatch.setattr(transformer_mod, "policy_objective", obj)
    monkeypatch.setattr(m.tul_code_policy, "features", feat)
    out = m(inp, labels=lab, slot_layout=layout)
    assert seen["x"].grad_fn is None and not seen["x"].requires_grad
    names, params = zip(*[(n, p) for n, p in m.named_parameters() if p.requires_grad])
    g = torch.autograd.grad(seen["loss"], params, allow_unused=True, retain_graph=True)
    reached = {n for n, gi in zip(names, g) if gi is not None and float(gi.abs().sum()) > 0}
    assert reached == {"tul_code_policy.policy.weight", "tul_code_policy.policy.bias",
                       "tul_code_policy.value.weight", "tul_code_policy.value.bias"}, reached
    out["loss"].backward()
    assert float(m.tul_code_policy.codes.basis.grad.abs().sum()) > 0.0


def test_one_sgd_step_on_the_policy_loss_follows_the_advantage():
    """Three slots, fixed rewards 1, 0, 0 and codes 2, 0, 1, zero value: slot 0's
    leave-one-out advantage is +1, the others' -0.5. One SGD step on the policy term alone
    must RAISE pi(2) at slot 0 and LOWER pi(0) at slot 1."""
    logits = torch.zeros(1, 3, C, requires_grad=True)
    value = torch.zeros(1, 3)
    codes = torch.tensor([[2, 0, 1]])
    reward = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    loss, st = policy_objective(logits, value, codes, reward, mask, 1.0, 0.0, 0.0)
    assert torch.allclose(st["code_policy_adv_mean"], torch.tensor(0.0), atol=1e-7)
    loss.backward()
    with torch.no_grad():
        new = logits - 1.0 * logits.grad
    p0, p1 = torch.softmax(logits.detach(), -1), torch.softmax(new, -1)
    assert float(p1[0, 0, 2]) > float(p0[0, 0, 2])
    assert float(p1[0, 1, 0]) < float(p0[0, 1, 0])
    # the value head's target is the residual r - rbar_{-s}
    _l, st2 = policy_objective(logits.detach(), value, codes, reward, mask, 0.0, 0.0, 1.0)
    want = ((1.0 - 0.0) ** 2 + (0.0 - 0.5) ** 2 + (0.0 - 0.5) ** 2) / 3
    assert math.isclose(float(st2["code_policy_value_mse"]), want, rel_tol=1e-6)


# ── real dropout ─────────────────────────────────────────────────────────────────────

def test_real_dropout_trains_and_two_same_seed_steps_agree():
    outs = []
    for _ in range(2):
        m = _model(dropout=0.1).train()
        _ids, inp, lab, layout = _pack()
        spy = _Spy(m, "_back_region")
        torch.manual_seed(99)
        out = m(inp, labels=lab, slot_layout=layout)
        out["loss"].backward()
        assert len(spy.calls) == 1 and torch.isfinite(out["loss"])
        assert float(m.tul_code_policy.policy.weight.grad.abs().sum()) > 0.0
        outs.append((out["loss"].detach(), m.tul_code_policy.policy.weight.grad.clone()))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


# ── refusals ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw, err", [
    (dict(code_policy_k=1), ValueError),
    (dict(code_policy_k=-2), ValueError),
    (dict(code_policy_k=4, code_policy_eval="random"), ValueError),
    (dict(code_policy_k=4, code_policy_lambda=0.0), ValueError),
    (dict(code_policy_k=4, code_policy_value_lambda=0.0), ValueError),
    (dict(code_policy_k=4, code_policy_ratio=0.0), ValueError),
    (dict(code_policy_k=4, code_policy_entropy=-0.1), ValueError),
    (dict(code_policy_entropy=0.1), ValueError),            # set at k = 0
    (dict(code_policy_eval="sample"), ValueError),          # set at k = 0
    (dict(code_policy_k=4, code_enum_k=4), NotImplementedError),
    (dict(code_policy_k=4, slot_cells=2, prefix_k=2), NotImplementedError),
    (dict(code_policy_k=4, tokens_through_core=True), NotImplementedError),
    (dict(code_policy_k=4, loop_reads_tokens=True), NotImplementedError),
    (dict(code_policy_k=4, code=True), NotImplementedError),
    (dict(code_policy_k=4, code_target=True), NotImplementedError),
    (dict(code_policy_k=4, gram=True), NotImplementedError),
    (dict(code_policy_k=4, core_stage_cond="sigma"), NotImplementedError),
    (dict(code_policy_k=4, vq_codes=2), NotImplementedError),
    (dict(code_policy_k=4, prefix_source="trajectory"), NotImplementedError),
    (dict(code_policy_k=4, pass_readout="gated"), NotImplementedError),
    (dict(code_policy_k=4, coda_sees_slots=False), NotImplementedError),
    (dict(code_policy_k=4, xhc_streams=8), NotImplementedError),
    (dict(code_policy_k=4, spandec_parallel_k=2), NotImplementedError),
    (dict(code_policy_k=4, db_loop=True), NotImplementedError),
    (dict(code_policy_k=4, core_token_aux=True), NotImplementedError),
    (dict(code_policy_k=4, grad_pass=True), NotImplementedError),
    (dict(code_policy_k=4, fan_k=4), NotImplementedError),
    (dict(code_policy_k=4, loop_denoise=True), NotImplementedError),
])
def test_every_refusal_fires_by_name(kw, err):
    base = dict(tg_geometry="strict", spandec=False, spandec_parallel=True, plast_weight=1.0)
    base.update(kw)
    with pytest.raises(err) as ei:
        _tul(**base)
    assert "code_policy" in str(ei.value), str(ei.value)


def test_the_model_refuses_what_the_config_cannot_see_and_picks_are_eval_only():
    torch.manual_seed(0)
    with pytest.raises(NotImplementedError, match="code_policy_k"):
        MORPHTransformer(_tiny(tul=_tc(1, code_policy_k=C), d_ff=_D_FF, mtp_heads=2))
    with pytest.raises(ValueError, match="code_policy_k"):
        MORPHTransformer(_tiny(tul=_tc(1, code_policy_k=C), d_ff=_D_FF, n_core=0))
    _ids, inp, lab, layout = _pack()
    m = _model().train()
    with pytest.raises(ValueError, match="EVAL-ONLY"):
        m.tul_forward_ablated(inp, lab, layout, code_policy_pick="random")
    m.eval()
    with pytest.raises(ValueError, match="argmax"):
        with torch.no_grad():
            m.tul_forward_ablated(inp, lab, layout, code_policy_pick="best")
    off = _model(policy_k=0).eval()
    with pytest.raises(ValueError, match="code_policy_k"):
        with torch.no_grad():
            off.tul_forward_ablated(inp, lab, layout, code_policy_pick="random")


def test_the_kv_cached_generator_refuses_the_policy_model():
    """The cached/graphed generators hold one code per ROLLOUT; a per-slot pick is not
    what they reproduce, so they must raise rather than decode a code-free model."""
    from morph.inference.tul_generate_cached import _check_supported
    with pytest.raises(NotImplementedError, match="generate_tul_cached does not reproduce"):
        _check_supported(_model().eval())


# ── configs ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name, parent, key, name_val", [
    ("tul_slot_spandec_strict_e1probe_fp01", "tul_slot_spandec_strict_e4probe_fp01",
     "tul.code_enum_k", "lxtul-e1probe-fp01"),
    ("tul_slot_spandec_strict_e1probe_fp01_policy4", "tul_slot_spandec_strict_e1probe_fp01",
     "tul.code_policy_k", "lxtul-e1probe-fp01-policy4"),
])
def test_the_configs_differ_by_the_stated_key_and_build(name, parent, key, name_val,
                                                        monkeypatch):
    from omegaconf import OmegaConf
    from test_slot_gain_tail import _MISSING, _leaves

    from morph.training.train import build_morph_config

    cfg, rt = _runtime(name, monkeypatch)
    pcfg, _prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
    assert diff == {key, "wandb.name"}, sorted(diff)
    assert c["wandb.name"] == name_val and int(c["training.steps"]) == 5000
    tc = rt.model_cfg
    assert tc.code_enum_k == 1
    assert tc.code_policy_k == (4 if key == "tul.code_policy_k" else 0)
    assert tc.spandec_parallel and tc.spandec_parallel_detach and tc.tg_geometry == "strict"
    mc = build_morph_config(cfg, tul=tc)
    assert mc.core_fixed_point_lambda == 0.1
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=tc, d_ff=_D_FF,
                               core_fixed_point_lambda=mc.core_fixed_point_lambda)).train()
    assert (m.tul_code_policy is not None) == (tc.code_policy_k > 0)
    assert m.tul_code_enum is None
    _ids, inp, lab, layout = _pack()
    out = m.float()(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert ("code_policy_weighted" in out) == (tc.code_policy_k > 0)


# ── the val instrument and the val loss ─────────────────────────────────────────────

class _Layout:
    stats: dict = {}

    def to(self, device):
        return self


class _PolicyStub(torch.nn.Module):
    """The val path's view of a policy model: a loss that includes the weighted term, and
    a random-code pass whose CE is 0.5 nats worse."""

    tul_code_policy = object()

    def tul_forward_with_plan_nats(self, x, y, layout):
        return {"loss": torch.tensor(5.0), "code_policy_weighted": torch.tensor(1.25),
                "code_policy_entropy": torch.tensor(0.75), "ce_tokens": 3.75,
                "layer_passes": 8.0, "n_tokens": 4.0}

    def tul_forward_ablated(self, x, y, layout, code_policy_pick=None):
        assert code_policy_pick == "random"
        return {"ce_tokens": 4.25}


def test_the_val_loss_subtracts_the_weighted_term_and_logs_the_readings():
    from morph.training.train import evaluate
    x = torch.zeros(1, 4, dtype=torch.long)
    extra: dict = {}
    avg, _ppl = evaluate(_PolicyStub(), torch.device("cpu"), iter([(x, x, _Layout())] * 2),
                         n_batches=2, tul=True, extra=extra)
    assert avg == pytest.approx(3.75)
    assert extra["val/code_policy_entropy"] == pytest.approx(0.75)
    assert extra["val/code_policy_vs_random"] == pytest.approx(0.5)


def test_train_logging_subtracts_the_term_and_logs_the_readings():
    """train.py's train-side `*_weighted` tuple carries the key and the `tul/` scan reads
    the `code_policy*` stats (both are inline in `main`, so they are read from its source,
    the `test_tul_lx_credit.py` precedent)."""
    import inspect

    import morph.training.train as tr
    src = inspect.getsource(tr)
    i = src.index('"fp_weighted", "core_gain_weighted", "egrad_weighted"')
    j = src.index("_lv = _lv - float(out[_ak])", i)
    assert '"code_policy_weighted"' in src[i:j]
    assert ('if _k.startswith("code_policy") and torch.is_tensor(out[_k]):\n'
            '                        log[f"tul/{_k}"]') in src


def test_the_train_loss_minus_the_weighted_term_is_the_loss_without_the_policy_term(
        monkeypatch):
    """Two train forwards from the same seed; the second has the policy objective replaced
    by an exact 0 (the choice, the codes and every draw unchanged). The first's loss minus
    `code_policy_weighted` is the second's loss: the key is exactly what was added."""
    _ids, inp, lab, layout = _pack()
    m = _model(dropout=0.1).train()
    _spread_codes(m)
    torch.manual_seed(3)
    a = m(inp, labels=lab, slot_layout=layout)
    real = transformer_mod.policy_objective

    def zero(*args):
        loss, st = real(*args)
        return loss * 0.0, st
    monkeypatch.setattr(transformer_mod, "policy_objective", zero)
    torch.manual_seed(3)
    b = m(inp, labels=lab, slot_layout=layout)
    assert float(a["code_policy_weighted"]) != 0.0
    torch.testing.assert_close(a["loss"] - a["code_policy_weighted"], b["loss"],
                               rtol=0, atol=1e-6)


def test_vs_random_is_zero_when_the_random_draw_is_the_argmax_and_nonzero_otherwise(
        monkeypatch):
    """The real `evaluate` on a real model. (1) With `random_codes` forced to return the
    policy's own argmax table, the random pass IS the argmax pass: the instrument reads
    exactly 0. (2) With the real uniform draw and spread codes it reads a nonzero CE gap."""
    from morph.training.train import evaluate
    m = _model().eval()
    _spread_codes(m)
    _ids, inp, lab, layout = _pack()
    pol = m.tul_code_policy

    def argmax_table(shape, device):
        return pol._last_argmax.clone()
    real_heads = pol.heads

    def heads(x):
        lg, v = real_heads(x)
        pol._last_argmax = lg.argmax(-1).masked_fill(~layout.slot_valid, 0)
        return lg, v
    monkeypatch.setattr(pol, "heads", heads)
    monkeypatch.setattr(pol, "random_codes", argmax_table)
    extra: dict = {}
    evaluate(m, torch.device("cpu"), iter([(inp, lab, layout)]), n_batches=1, tul=True,
             extra=extra)
    assert extra["val/code_policy_vs_random"] == 0.0
    assert "val/code_policy_entropy" in extra and "val/code_policy_share_k0" in extra
    monkeypatch.undo()
    extra2: dict = {}
    evaluate(m, torch.device("cpu"), iter([(inp, lab, layout)]), n_batches=1, tul=True,
             extra=extra2)
    assert extra2["val/code_policy_vs_random"] != 0.0
