"""Per-pass low-rank deltas on the shared core (``tul.pass_lora_rank``).

Bae et al. 2024, "Relaxed Recursive Transformers: Effective Parameter Sharing with
Layer-wise LoRA". The looped core keeps its shared weights and pass ``t`` adds its own
rank-``r`` delta ``y_t = sublayer(x) + B_t (A_t x)`` with ``B`` zero-init. See
``morph/model/mhc.py::PassLoRA`` for what a target covers and what it does not.

One test per contract:

  1. zero-init is bit-identical: with the knob on and untrained, the loss, the logits, the
     base weights and every base gradient are the numbers the model produced without it.
     (Checked off the test tree too, against the tiny CPU model at master ``c429e22``:
     loss 9.1006078720092773 and sha256 ``32b174ef…2853f8`` over all 208 base gradient
     tensors, with the knob off, at rank 8, and at rank 32 on the MLP alone.)
  2. the pass index is really threaded: give ONE pass a non-zero ``B`` and only that
     pass's output moves.
  3. gradient reaches ``A_t`` / ``B_t`` for the passes that RAN and is exactly zero for
     the rows of the stack the loop never reached.
  4. neither weight schedule can see the deltas: ternary QAT does not parametrize them,
     and they are not the module types the prune / carve / packer walk selects.
  5. the knob is refused where a pass index has no meaning, and the plain path
     (``slot_layout=None``) is untouched.

CPU only, tiny config, no tokenizer.
"""

from __future__ import annotations

import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402  (tests/ is on sys.path)

from morph.model.mhc import PassLoRA
from morph.model.transformer import MORPHTransformer

MAX_DEPTH = 4
FIXED = 3            # < MAX_DEPTH, so the last row of the pass stack never runs


def _model(seed: int = 3, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul_kw = {k[4:]: v for k, v in kw.items() if k.startswith("tul_")}
    cfg_kw = {k: v for k, v in kw.items() if not k.startswith("tul_")}
    tul = _tul(tg_restrict=False, sigreg_lambda=0.0, mux_beta=1.0, mux_target="next",
               mux_detach_head=False, **tul_kw)
    base = dict(tul=tul, n_core=2, mean_depth=MAX_DEPTH, max_depth=MAX_DEPTH,
                bptt_depth=MAX_DEPTH, retention=False, dropout=0.1,
                slot_gain_lambda=0.0, core_fixed_point_lambda=1.0, ckpt_grad_iters=0)
    base.update(cfg_kw)
    return MORPHTransformer(_cfg(**base))


def _run(m: MORPHTransformer, *, seed: int = 7, backward: bool = True):
    x, y, layout, _ = _batch()
    m.train()
    torch.manual_seed(seed)
    out = m(x, labels=y, slot_layout=layout)
    if backward:
        out["loss"].backward()
    return out


def _base_state(m: MORPHTransformer):
    """(loss-free) digest of every parameter and gradient that is NOT a pass-LoRA one."""
    w = [p.detach().clone() for n, p in sorted(m.named_parameters())
         if "pass_lora" not in n]
    g = [(p.grad.detach().clone() if p.grad is not None else None)
         for n, p in sorted(m.named_parameters()) if "pass_lora" not in n]
    return w, g


def _eq(a, b) -> bool:
    return len(a) == len(b) and all(
        ((x is None) == (y is None)) and (x is None or torch.equal(x, y))
        for x, y in zip(a, b))


# ── 1. zero-init is bit-identical ────────────────────────────────────────────

@pytest.mark.parametrize("kw", [
    dict(tul_pass_lora_rank=8),
    dict(tul_pass_lora_rank=32, tul_pass_lora_targets=("mlp",)),
    dict(tul_pass_lora_rank=4, tul_pass_lora_targets=("attn",)),
])
def test_zero_init_is_bit_identical_to_the_model_without_the_knob(kw):
    m_off = _model()
    out_off = _run(m_off)
    rng_off = torch.get_rng_state()
    m_on = _model(**kw)
    out_on = _run(m_on)
    assert torch.equal(rng_off, torch.get_rng_state()), "the knob drew from the global RNG"
    assert torch.equal(out_off["loss"], out_on["loss"])
    assert torch.equal(out_off["mux_local"], out_on["mux_local"])
    w_off, g_off = _base_state(m_off)
    w_on, g_on = _base_state(m_on)
    assert _eq(w_off, w_on), "building the deltas moved a base weight"
    assert _eq(g_off, g_on), "the zero-init deltas changed a base gradient"
    # `out["logits"]` is None on a labelled forward (fused CE), so score the logits on a
    # label-free one; the loop and the deltas run identically either way.
    x, _, layout, _ = _batch()
    for m in (m_off, m_on):
        m.eval()
    with torch.no_grad():
        torch.manual_seed(5)
        la = m_off(x, slot_layout=layout)["logits"]
        torch.manual_seed(5)
        lb = m_on(x, slot_layout=layout)["logits"]
    assert torch.equal(la, lb), "the zero-init deltas moved the logits"


def test_a_non_zero_delta_does_change_the_loss():
    """Guards the test above: a mechanism that never acts would pass it for free."""
    m = _model(tul_pass_lora_rank=8)
    base = _run(m, backward=False)["loss"].detach().clone()
    with torch.no_grad():
        for blk in m.core:
            blk.pass_lora.B_mlp.normal_(0.0, 0.1)
    assert not torch.equal(base, _run(m, backward=False)["loss"])


# ── 2. the pass index is threaded ────────────────────────────────────────────

def test_only_the_pass_whose_delta_is_non_zero_moves():
    m = _model(tul_pass_lora_rank=8)
    m.eval()
    blk = m.core[0]
    torch.manual_seed(0)
    x = torch.randn(2, 5, 4, 64)          # [B, S, n_streams, C] carrier
    with torch.no_grad():
        ref = [blk(x, pass_idx=t) for t in range(m.pass_lora_n_passes)]
        assert all(torch.equal(ref[0], r) for r in ref[1:]), \
            "the shared block already differs per pass at zero init"
        blk.pass_lora.B_attn[1].normal_(0.0, 0.5)
        blk.pass_lora.B_mlp[1].normal_(0.0, 0.5)
        got = [blk(x, pass_idx=t) for t in range(m.pass_lora_n_passes)]
    for t, (a, b) in enumerate(zip(ref, got)):
        if t == 1:
            assert not torch.equal(a, b), "pass 1's own delta did nothing"
        else:
            assert torch.equal(a, b), f"pass {t} moved when only pass 1's delta was set"


def test_two_different_passes_give_two_different_states():
    m = _model(tul_pass_lora_rank=8)
    m.eval()
    blk = m.core[0]
    torch.manual_seed(0)
    x = torch.randn(2, 5, 4, 64)
    with torch.no_grad():
        blk.pass_lora.B_attn.normal_(0.0, 0.5)
        blk.pass_lora.B_mlp.normal_(0.0, 0.5)
        outs = [blk(x, pass_idx=t) for t in range(m.pass_lora_n_passes)]
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            assert not torch.equal(outs[i], outs[j]), (i, j)


# ── 3. gradient reaches only the passes that ran ─────────────────────────────

def test_b_receives_gradient_exactly_on_the_passes_that_ran():
    m = _model(tul_pass_lora_rank=8, tul_slot_depth_fixed=FIXED)
    _run(m)
    for blk in m.core:
        for name in ("B_attn", "B_mlp"):
            g = getattr(blk.pass_lora, name).grad
            assert g is not None, name
            for t in range(FIXED):
                assert float(g[t].abs().max()) > 0.0, f"{name} pass {t} got no gradient"
            for t in range(FIXED, m.pass_lora_n_passes):
                assert float(g[t].abs().max()) == 0.0, \
                    f"{name} pass {t} never ran and still received gradient"


def test_a_receives_gradient_once_b_is_non_zero():
    """At B = 0 every dL/dA is zero by the chain rule (the standard LoRA cold start), so
    the A path is only observable after B moves."""
    m = _model(tul_pass_lora_rank=8, tul_slot_depth_fixed=FIXED)
    _run(m)
    for blk in m.core:
        assert float(blk.pass_lora.A_mlp.grad.abs().max()) == 0.0, "B is zero, A must be too"
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        for blk in m.core:
            blk.pass_lora.B_attn.normal_(0.0, 0.1)
            blk.pass_lora.B_mlp.normal_(0.0, 0.1)
    _run(m)
    for blk in m.core:
        for name in ("A_attn", "A_mlp"):
            g = getattr(blk.pass_lora, name).grad
            for t in range(FIXED):
                assert float(g[t].abs().max()) > 0.0, f"{name} pass {t} got no gradient"
            for t in range(FIXED, m.pass_lora_n_passes):
                assert float(g[t].abs().max()) == 0.0, f"{name} pass {t} never ran"


def test_the_prelude_and_coda_carry_no_deltas():
    m = _model(tul_pass_lora_rank=8)
    for group in (m.prelude, m.coda):
        for blk in group:
            assert blk.pass_lora is None
    assert all(blk.pass_lora is not None for blk in m.core)


# ── 4. neither weight schedule can see the deltas ────────────────────────────

def test_ternary_qat_walks_past_the_deltas():
    from torch.nn.utils import parametrize

    from morph.model.ternary_qat import apply_ternary_qat
    m = _model(tul_pass_lora_rank=8)
    manifest = apply_ternary_qat(m, scope="backbone", scale_mode="norm_match")
    assert manifest["n_modules_ternary"] > 0, "the fixture ternarized nothing"
    for blk in m.core:
        assert not parametrize.is_parametrized(blk.pass_lora), \
            "ternary QAT parametrized the per-pass deltas"
        for name in ("A_attn", "B_attn", "A_mlp", "B_mlp"):
            p = getattr(blk.pass_lora, name)
            assert isinstance(p, torch.nn.Parameter) and p.dtype == torch.float32


def test_the_prune_and_carve_walks_cannot_select_the_deltas():
    from morph.training.pruning import _find_cms_layers, _find_mortar_layers
    m = _model(tul_pass_lora_rank=8)
    selected = ({id(mod) for _, mod in _find_cms_layers(m)}
                | {id(mod) for _, mod in _find_mortar_layers(m)})
    assert selected, "the fixture has no prunable layer at all"
    for blk in m.core:
        assert id(blk.pass_lora) not in selected
        for sub in blk.pass_lora.modules():
            assert id(sub) not in selected


# ── 5. refusals and the plain path ───────────────────────────────────────────

def test_paid_loop_refuses_the_knob():
    with pytest.raises(NotImplementedError, match="SLOT-LOOP lever"):
        _tul(pass_lora_rank=8, tokens_through_core=True)


def test_iteration_conditioning_refuses_the_knob():
    with pytest.raises(ValueError, match="core_stage_cond"):
        _tul(pass_lora_rank=8, core_stage_cond="iter")


@pytest.mark.parametrize("bad", [("attn", "attn"), ("down",), ()])
def test_bad_targets_raise(bad):
    with pytest.raises(ValueError, match="pass_lora_targets"):
        _tul(pass_lora_rank=8, pass_lora_targets=bad)


def test_negative_rank_raises():
    with pytest.raises(ValueError, match="pass_lora_rank"):
        _tul(pass_lora_rank=-1)


def test_a_coreless_model_refuses_the_knob():
    with pytest.raises(ValueError, match="needs a core loop"):
        _model(n_core=0, tul_pass_lora_rank=8)


def test_pass_lora_module_validates_its_own_arguments():
    with pytest.raises(ValueError, match="rank >= 1"):
        PassLoRA(64, 4, 0, ("mlp",))
    with pytest.raises(ValueError, match="n_passes >= 1"):
        PassLoRA(64, 0, 8, ("mlp",))
    with pytest.raises(ValueError, match="targets"):
        PassLoRA(64, 4, 8, ("attn", "value"))


def test_the_plain_path_is_untouched():
    """`slot_layout=None` runs `_core_region`, which shares `_apply_core_step`. A
    zero-init delta must leave it bit-identical, and no arm builds the knob with the
    paid loop (which is the only way the plain core would carry one)."""
    x = torch.randint(0, 64, (2, 48))
    y = torch.randint(0, 64, (2, 48))
    outs, grads = [], []
    for kw in ({}, dict(tul_pass_lora_rank=8)):
        m = _model(**kw)
        m.train()
        torch.manual_seed(99)
        out = m(x, labels=y)
        out["loss"].backward()
        outs.append(out["loss"].detach().clone())
        grads.append(torch.cat([q.grad.flatten() for n, q in sorted(m.named_parameters())
                                if q.grad is not None and "pass_lora" not in n
                                and not n.startswith("tul.")]))
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(grads[0], grads[1])


def test_the_stack_is_sized_by_the_loop_max_depth():
    m = _model(tul_pass_lora_rank=8)
    assert m.pass_lora_n_passes == MAX_DEPTH
    assert m.core[0].pass_lora.A_attn.shape == (MAX_DEPTH, 8, 64)
    assert m.core[0].pass_lora.B_attn.shape == (MAX_DEPTH, 64, 8)
    m2 = _model(tul_pass_lora_rank=8, tul_slot_max_depth=2)
    assert m2.pass_lora_n_passes == 2
