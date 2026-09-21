"""``tul.fan_trigger_every_pass`` — the per-stream trigger, re-injected every pass.

WHAT THIS FILE HAS TO PROVE:

1. OFF IS NOTHING. The default is False, a model built with the key absent and one built
   with it False agree to the last bit on loss, logits and the total gradient.
2. ON AT STEP 0 IS NOTHING IN THE FORWARD. ``W_o`` is zero-init and ``P_cell`` is zeros,
   so the re-injected term is EXACTLY zero and the True arm's loss and logits equal the
   False arm's to the last bit. Its GRADIENT is NOT the False arm's, and that is not a
   defect: a term whose VALUE is zero still has a non-zero derivative, so ``W_o`` and
   ``P_cell`` already collect the re-injection's cotangent at step 0. Measured here
   (1.0633 -> 1.6157 on this fixture) so nobody reads "bit-identical at step 0" as
   covering the backward.
3. THE ADD IS THE TRIGGER, EXACTLY, AND ONLY FROM PASS 2. With ``W_o`` and ``P_cell``
   perturbed, the captured ``post - pre`` at every captured pass equals the register's own
   ``W_o(pooled) + P_cell`` broadcast into the carrier, to the last bit; pass 1 (t = 0)
   gets NO capture, and the state after pass 1 is bit-identical to the False arm's.
4. THE GRADIENT REACHES ``W_o`` AND ``P_cell`` THROUGH PASSES >= 2. Isolated by the depth:
   at a forced depth of 1 the True and False arms have the SAME gradient on both tensors
   (there is no pass 2), and at a forced depth of 3 they differ — so the difference is the
   re-injection's own path and not a seed effect.
5. REFUSAL off a fan arm, naming the key.
6. The key composes through ``tul_setup`` (``KNOWN_TUL_KEYS``, the mapping, the manifest)
   and the shipped config resolves to True.

CPU only, fp32, the ``tests/test_tul_fan.py`` fixtures.
Prereg: lab/experiments/planned/2026-09-21-lxtul-fan4-all-trig.md
"""
from __future__ import annotations

import pytest
import torch

from test_tul_fan import _batch, _model, _tul, _tiny, _finite_logit_sum
from morph.model.transformer import MORPHTransformer


K = 4


def _fan(**kw) -> MORPHTransformer:
    """A fan4-all model, the shipped arm's shape, at the tiny fixture's size."""
    base = dict(fan_k=K, slot_cells=K, prefix_k=K, fan_mix="all")
    base.update(kw)
    return _model(**base)


def _fan_depth(depth: int, **kw) -> MORPHTransformer:
    """The same arm at a CONSTANT loop depth (model.depth_fixed), for the grad isolation."""
    base = dict(fan_k=K, slot_cells=K, prefix_k=K, fan_mix="all")
    base.update(kw)
    torch.manual_seed(1234)
    m = MORPHTransformer(_tiny(tul=_tul(**base), mean_depth=depth, max_depth=depth,
                               bptt_depth=depth, depth_fixed=True))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.float()


def _perturb(m: MORPHTransformer, seed: int = 99) -> None:
    """Move the trigger off its zero-init, so the term it re-injects is not 0."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.tul_register.W_o.weight.copy_(
            torch.empty(m.tul_register.W_o.weight.shape).normal_(0.0, 0.05, generator=g))
        m.tul_register.P_cell.copy_(
            torch.empty(m.tul_register.P_cell.shape).normal_(0.0, 0.05, generator=g))


def _train_forward(m: MORPHTransformer, seed: int = 7):
    _ids, inp, lab, layout = _batch(K)
    torch.manual_seed(seed)
    return m.train()(inp, labels=lab, slot_layout=layout)


def _loss_logits_grad(m: MORPHTransformer, seed: int = 7):
    _ids, inp, lab, layout = _batch(K)
    torch.manual_seed(seed)
    res = m.train()(inp, labels=lab, slot_layout=layout)
    res["loss"].backward()
    g = sum(float(p.grad.double().abs().sum()) for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        lg = m(inp, labels=None, slot_layout=layout)["logits"]
    return float(res["loss"].detach()), _finite_logit_sum(lg), g


# ── 1. off is nothing ────────────────────────────────────────────────────────

def test_default_is_false_and_the_absent_key_is_bit_identical():
    a = _fan()
    assert a.cfg.tul.fan_trigger_every_pass is False
    assert a._fan_trigger_every_pass is False
    b = _fan(fan_trigger_every_pass=False)
    assert _loss_logits_grad(a) == _loss_logits_grad(b)


def test_off_captures_nothing_even_with_the_hook_attached():
    m = _fan()
    m._trigger_capture = []
    _train_forward(m)
    assert m._trigger_capture == []


# ── 2. on at step 0 is nothing either ────────────────────────────────────────

def test_true_at_the_zero_init_equals_false_in_the_forward():
    a = _fan()
    b = _fan(fan_trigger_every_pass=True)
    assert float(a.tul_register.W_o.weight.detach().abs().sum()) == 0.0
    assert float(a.tul_register.P_cell.detach().abs().sum()) == 0.0
    la, lga, _ = _loss_logits_grad(a)
    lb, lgb, _ = _loss_logits_grad(b)
    assert la == lb
    assert lga == lgb


def test_the_zero_term_still_carries_gradient_into_the_trigger():
    """The value is 0; the derivative is not. `W_o` and `P_cell` collect the
    re-injection's cotangent from step 0, which is why the arm can move at all."""
    a, b = _fan(), _fan(fan_trigger_every_pass=True)
    _loss_logits_grad(a)
    _loss_logits_grad(b)
    ga = float(a.tul_register.W_o.weight.grad.abs().sum())
    gb = float(b.tul_register.W_o.weight.grad.abs().sum())
    assert ga > 0.0 and gb > ga


def test_the_term_is_exactly_zero_at_the_zero_init():
    m = _fan(fan_trigger_every_pass=True)
    m._trigger_capture = []
    _train_forward(m)
    assert len(m._trigger_capture) > 0, "the fixture must reach at least two passes"
    for rec in m._trigger_capture:
        assert torch.equal(rec["post"], rec["pre"])


# ── 3. the add is the trigger, exactly, and only from pass 2 ─────────────────

def _register_output(m: MORPHTransformer, inp, layout) -> torch.Tensor:
    """The register's OWN output on the real forward, ``[B, S*M, C]``.

    Captured with a forward hook on ``m.tul_register`` rather than recomputed: a
    recomputation would have to reproduce ``_tul_front``'s TG attention kwargs, and the
    standing rule (`instruments-must-use-the-models-tg-kwargs`) is that a bare front
    scores a strict arm from an unrestricted prelude. The hook reads the tensor the
    forward actually built.
    """
    cap: list[torch.Tensor] = []
    h = m.tul_register.register_forward_hook(lambda _m, _a, out: cap.append(out.detach()))
    try:
        with torch.no_grad():
            m(inp, labels=None, slot_layout=layout)
    finally:
        h.remove()
    assert len(cap) == 1, f"the register ran {len(cap)} times, expected once"
    return cap[0]


def test_the_captured_add_is_the_trigger_and_pass_one_is_untouched():
    _ids, inp, lab, layout = _batch(K)
    m = _fan(fan_trigger_every_pass=True)
    _perturb(m)
    m.eval()
    trig = _register_output(m, inp, layout)              # [B, S*M, C]
    assert float(trig.abs().sum()) > 0.0, "the perturbation must make a non-zero term"

    m._trigger_capture = []
    with torch.no_grad():
        m(inp, labels=None, slot_layout=layout)
    caps = m._trigger_capture
    assert len(caps) >= 1, "the fixture must reach at least two passes"
    # only passes 2..T, never pass 1
    assert min(r["t"] for r in caps) == 1
    want = trig.unsqueeze(2) if m._is_hc else trig       # broadcast on the stream axis
    for rec in caps:
        # `post == pre + term`, NOT `post - pre == term`: the subtraction of two nearly
        # equal floats does not recover the addend bit for bit. Reproducing the exact op
        # is what makes this an equality and not a tolerance.
        assert torch.equal(rec["post"], rec["pre"] + want)
        assert not torch.equal(rec["post"], rec["pre"])


def test_pass_one_state_matches_the_false_arm_and_pass_two_does_not():
    """Read off ``_jac_capture``, which records the carrier ENTERING each pass on the real
    forward: entry ``t`` is the state after pass ``t``. Entries 0 and 1 (the seed and the
    state after pass 1) must agree across the arms; entry 2 must not, because pass 2
    opened with the trigger."""
    _ids, inp, lab, layout = _batch(K)
    caps = {}
    for name, flag in (("off", False), ("on", True)):
        m = _fan_depth(3, fan_trigger_every_pass=flag)
        _perturb(m)
        m.eval()
        m._jac_capture = []
        with torch.no_grad():
            m(inp, labels=None, slot_layout=layout)
        caps[name] = [r["h"] for r in m._jac_capture]
    assert len(caps["off"]) == 3 and len(caps["on"]) == 3
    assert torch.equal(caps["off"][0], caps["on"][0]), "the seed state is the same"
    assert torch.equal(caps["off"][1], caps["on"][1]), "pass 1 is untouched"
    assert not torch.equal(caps["off"][2], caps["on"][2]), "pass 2 got the trigger"


# ── 4. the gradient reaches W_o and P_cell through passes >= 2 ───────────────

def _trigger_grads(depth: int, flag: bool):
    m = _fan_depth(depth, fan_trigger_every_pass=flag)
    _perturb(m)
    _ids, inp, lab, layout = _batch(K)
    torch.manual_seed(7)
    res = m.train()(inp, labels=lab, slot_layout=layout)
    gw, gp = torch.autograd.grad(res["loss"],
                                 [m.tul_register.W_o.weight, m.tul_register.P_cell])
    return gw.double(), gp.double()


def test_depth_one_has_no_second_pass_so_the_knob_changes_no_gradient():
    a_w, a_p = _trigger_grads(1, False)
    b_w, b_p = _trigger_grads(1, True)
    assert torch.equal(a_w, b_w) and torch.equal(a_p, b_p)


def test_depth_three_moves_both_trigger_gradients():
    a_w, a_p = _trigger_grads(3, False)
    b_w, b_p = _trigger_grads(3, True)
    assert float(b_w.abs().sum()) > 0.0 and float(b_p.abs().sum()) > 0.0
    assert not torch.equal(a_w, b_w)
    assert not torch.equal(a_p, b_p)
    # a real change, not a rounding wobble: the relative move is well above fp32 noise
    assert float((b_w - a_w).abs().sum() / a_w.abs().sum()) > 1e-4
    assert float((b_p - a_p).abs().sum() / a_p.abs().sum()) > 1e-4


def _ckpt_model(flag: bool) -> MORPHTransformer:
    """``bptt_depth < max_depth`` puts the grad passes inside ``torch.utils.checkpoint``."""
    torch.manual_seed(1234)
    m = MORPHTransformer(_tiny(
        tul=_tul(fan_k=K, slot_cells=K, prefix_k=K, fan_mix="all",
                 fan_trigger_every_pass=flag),
        mean_depth=3, max_depth=3, bptt_depth=2, depth_fixed=True)).float()
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m


def test_the_gradient_survives_gradient_checkpointing():
    """The term is a positional INPUT of the ``checkpoint`` call, so the backward
    recompute replays it and the gradient still reaches both trigger tensors.

    Scored at the ZERO INIT on purpose: there the forward is identical to the False arm's,
    so the ONLY thing that can move ``W_o``'s gradient is the re-injection's own cotangent
    crossing the checkpoint boundary. A detached term (or one captured from a closure and
    lost on recompute) reads equal here; a live one reads strictly larger."""
    a, b = _ckpt_model(False), _ckpt_model(True)
    _ids, inp, lab, layout = _batch(K)
    out = []
    for m in (a, b):
        torch.manual_seed(7)
        res = m.train()(inp, labels=lab, slot_layout=layout)
        gw, gp = torch.autograd.grad(res["loss"],
                                     [m.tul_register.W_o.weight, m.tul_register.P_cell])
        assert torch.isfinite(gw).all() and torch.isfinite(gp).all()
        out.append((float(res["loss"].detach()), float(gw.abs().sum()),
                    float(gp.abs().sum())))
    assert out[0][0] == out[1][0], "zero-init: the forward must still be identical"
    assert out[1][1] > out[0][1] > 0.0, "W_o must collect the re-injection's cotangent"
    assert out[1][2] > out[0][2] > 0.0, "P_cell must collect it too"


# ── 5. refusal ───────────────────────────────────────────────────────────────

def test_true_off_a_fan_arm_is_refused_by_name():
    with pytest.raises(ValueError, match="tul.fan_trigger_every_pass"):
        _tul(fan_trigger_every_pass=True)
    # and it is accepted on a fan arm
    _tul(fan_k=K, slot_cells=K, prefix_k=K, fan_mix="all", fan_trigger_every_pass=True)


# ── 6. the key composes ──────────────────────────────────────────────────────

def test_the_key_composes_through_tul_setup_and_the_config_resolves():
    import os
    from hydra import compose, initialize_config_dir
    from morph.training.tul_setup import KNOWN_TUL_KEYS

    assert "fan_trigger_every_pass" in KNOWN_TUL_KEYS
    cdir = os.path.abspath("morph/configs")
    with initialize_config_dir(config_dir=cdir, version_base=None):
        cfg = compose(config_name="tul_slot_spandec_strict_fan4_all_trig")
    assert bool(cfg.tul.fan_trigger_every_pass) is True
    assert cfg.tul.fan_mix == "all"
    assert int(cfg.tul.fan_k) == 4
    assert cfg.wandb.name == "slot-spandec-strict-fan4-all-trig"
