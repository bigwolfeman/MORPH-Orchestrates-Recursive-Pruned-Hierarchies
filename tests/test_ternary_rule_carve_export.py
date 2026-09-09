"""The ONE ternary rule (``morph/model/ternary_rule.py``) holds on every path that turns a
latent weight into ``scale * codes``: the dense STE, the carved MORTAR layer, the deploy
packer, and the checkpoint round trip in between.

Why this file exists: ``base.yaml`` ships ``ternary_scale_mode: norm_match`` (2026-09-09).
Before that change the carve (``compact_step`` 29000 in the recipe) re-registered a
hardcoded absmean STE on ``mortar_data`` and the deploy packer refused every mode but
``symmetric`` — a norm-matched recipe would have silently reverted to absmean at the
carve and could not be exported at all. Each test below is one of those seams.

CPU only: ``carve()`` and ``_mortar_effective_data()`` are plain torch; the carved
FORWARD (``_forward_mortar``, the stk Triton kernel) is not run here.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from morph.inference.deploy_quant import (
    PackedTernaryLinear,
    extract_ternary_from_parametrized,
    pack_mortar_ternary,
)
from morph.model.layers.block_sparse import CMSBlockLinear
from morph.model.ternary_qat import TernarySTE, _apply_grouped_ste, _encode_scale_fp16, apply_ternary_qat
from morph.model.ternary_rule import (
    EXPORTABLE_SCALE_MODES,
    scale_mode_code,
    scale_mode_from_code,
    ternary_codes_and_scale,
)
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.training.pruning import PruningSchedule

V = 64
D_MODEL = 512
D_FF = 512
BLOCKING = 128
COMPACT_STEP = 3


# ── the rule itself ──────────────────────────────────────────────────────────

def test_symmetric_rule_reproduces_the_grouped_ste_exactly():
    torch.manual_seed(0)
    w = torch.randn(256, 128) * 0.02
    q, s = ternary_codes_and_scale(w, 0.5, "symmetric", _encode_scale_fp16)
    ste_out = _apply_grouped_ste(w, 0, 0.5, _encode_scale_fp16)
    # The STE returns w + (s*q - w).detach() (the straight-through form), which differs
    # from s*q by float round-off; compare in the STE's own form.
    assert torch.equal(w + (s * q - w).detach(), ste_out), "the shared symmetric rule must be the STE's own math"


def test_norm_match_rule_keeps_the_symmetric_codes_and_the_latent_norm():
    torch.manual_seed(1)
    w = torch.randn(256, 128) * 0.02
    q_sym, s_sym = ternary_codes_and_scale(w, 0.5, "symmetric", _encode_scale_fp16)
    q_nm, s_nm = ternary_codes_and_scale(w, 0.5, "norm_match", _encode_scale_fp16)
    assert torch.equal(q_sym, q_nm)
    assert abs((s_nm * q_nm).norm().item() / w.norm().item() - 1.0) < 1e-3
    assert 0.55 < (s_sym * q_sym).norm().item() / w.norm().item() < 0.75
    assert s_nm > s_sym


def test_scale_mode_codes_round_trip_and_refuse_learnable_modes():
    for i, m in enumerate(EXPORTABLE_SCALE_MODES):
        assert scale_mode_code(m) == i
        assert scale_mode_from_code(float(i)) == m
    with pytest.raises(ValueError):
        scale_mode_code("ttq")
    with pytest.raises(ValueError):
        ternary_codes_and_scale(torch.randn(4, 4), 0.5, "dual")


# ── the dense STE → the packer ───────────────────────────────────────────────

@pytest.mark.parametrize("group", [0, 64])
def test_packer_reproduces_the_norm_match_ste_forward(group):
    torch.manual_seed(2)
    lin = nn.Linear(96, 200, bias=False)          # weight [200, 96]: 3 full groups of 64 + a 8-row tail
    with torch.no_grad():
        lin.weight.mul_(0.02)
    ste = TernarySTE(threshold=0.5, weight_shape=(200, 96), mode="norm_match", group=group,
                     weight_init=lin.weight.detach())
    parametrize.register_parametrization(lin, "weight", ste)
    live = lin.weight.detach()
    codes, scale, meta = extract_ternary_from_parametrized(lin)
    assert meta["mode"] == "norm_match" and meta["threshold"] == 0.5
    assert meta["n_groups"] == (1 if group == 0 else 4)
    packed = PackedTernaryLinear.from_parametrized(lin)
    assert torch.allclose(packed.weight, live, atol=1e-7, rtol=0.0)
    assert abs(packed.weight.norm().item() / lin.parametrizations.weight.original.norm().item() - 1.0) < 1e-3
    x = torch.randn(3, 96)
    assert torch.allclose(packed(x), lin(x), atol=1e-5)


def test_packer_still_refuses_learnable_scales():
    lin = nn.Linear(32, 16, bias=False)
    ste = TernarySTE(threshold=0.5, weight_shape=(32, 16), mode="ttq", weight_init=lin.weight.detach())
    parametrize.register_parametrization(lin, "weight", ste)
    with pytest.raises(NotImplementedError):
        extract_ternary_from_parametrized(lin)


# ── the carve → the carved STE → the checkpoint → the packer ────────────────

def _cfg() -> MORPHConfig:
    return MORPHConfig(
        d_model=D_MODEL, n_heads=4, n_kv_heads=4, d_ff=D_FF, vocab_size=V,
        max_seq_len=32, context_len=32,
        n_prelude=1, n_core=1, n_coda=1, mean_depth=1, max_depth=1, bptt_depth=1,
        channel_dims=(256, 160, 96), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )


def _ternary_model(scale_mode: str, seed: int = 0) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_cfg())
    apply_ternary_qat(m, scope="backbone", threshold=0.5, scale_mode=scale_mode,
                      scale_group="tensor", scale_dtype="fp16")
    return m


def _schedule() -> PruningSchedule:
    return PruningSchedule(prune_start=999_999_999, prune_interval=1, prune_rate=0.03,
                           target_density=0.25, compact_step=COMPACT_STEP,
                           route_start=999_999_999, carve_blocking=BLOCKING)


def _train_to_carve(model: MORPHTransformer, schedule: PruningSchedule) -> None:
    """forward → backward → schedule.step, the trainer's order, up to and including the
    carve step. No forward after the carve (the stk kernel is GPU-only)."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(COMPACT_STEP + 1):
        torch.manual_seed(100 + step)
        ids = torch.randint(0, V, (2, 16))
        model(ids, labels=ids.clone())["loss"].backward()
        schedule.step(model, step)
        opt.step()
        opt.zero_grad(set_to_none=True)


def _cms_layers(model: nn.Module) -> list[tuple[str, CMSBlockLinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, CMSBlockLinear)]


def test_carve_carries_norm_match_into_the_carved_ste_and_persists_it():
    model = _ternary_model("norm_match")
    _train_to_carve(model, _schedule())
    layers = _cms_layers(model)
    assert layers and all(m._mortar for _, m in layers)
    for name, cms in layers:
        assert cms._values_ternary_mode, name
        assert cms._values_scale_mode == "norm_match", name
        assert cms.mortar_ternary.tolist() == [1.0, 0.5, float(scale_mode_code("norm_match"))], name
        eff = cms._mortar_effective_data().detach()
        assert abs(eff.norm().item() / cms.mortar_data.detach().norm().item() - 1.0) < 1e-3, name

    # Round trip through a fresh model: the carved layers rebuild themselves from the
    # state_dict (the CMS load hook) and read the mode back from the buffer.
    state = {k: v.clone() for k, v in model.state_dict().items()}
    fresh = _ternary_model("norm_match", seed=1)
    fresh.load_state_dict(state, strict=False)
    for (name, a), (_, b) in zip(layers, _cms_layers(fresh)):
        assert b._mortar and b._values_scale_mode == "norm_match", name
        assert torch.equal(a._mortar_effective_data().detach(), b._mortar_effective_data().detach()), name


def test_symmetric_carve_is_unchanged_and_an_old_two_element_buffer_loads_as_symmetric():
    model = _ternary_model("symmetric")
    _train_to_carve(model, _schedule())
    layers = _cms_layers(model)
    for name, cms in layers:
        assert cms._values_scale_mode == "symmetric", name
        # The pre-2026-09-09 inline absmean math, written out: bit-identical.
        d = cms.mortar_data
        scale = d.detach().abs().mean().clamp(min=1e-8)
        d_norm = d / scale
        q = torch.sign(d_norm) * (d_norm.abs() > 0.5).to(d.dtype)
        assert torch.equal(cms._mortar_effective_data().detach(), (d + (scale * q - d).detach()).detach()), name

    # A checkpoint written before the mode was persisted holds [flag, threshold].
    state = {k: v.clone() for k, v in model.state_dict().items()}
    for k in list(state):
        if k.endswith("mortar_ternary"):
            state[k] = state[k][:2].clone()
    fresh = _ternary_model("symmetric", seed=1)
    fresh.load_state_dict(state, strict=False)
    for name, cms in _cms_layers(fresh):
        assert cms._mortar and cms._values_ternary_mode, name
        assert cms._values_scale_mode == "symmetric", name
        assert cms.mortar_ternary.numel() == 3, name


def test_carve_refuses_learnable_scales_before_touching_any_layer():
    model = _ternary_model("ttq")
    with pytest.raises(NotImplementedError):
        _train_to_carve(model, _schedule())
    for name, cms in _cms_layers(model):
        assert cms._dense_mode and not cms._mortar, name
        assert parametrize.is_parametrized(cms, "weight"), name


def test_packer_bakes_the_trained_rule_on_a_carved_norm_match_layer():
    model = _ternary_model("norm_match")
    _train_to_carve(model, _schedule())
    name, cms = _cms_layers(model)[0]
    live = cms._mortar_effective_data().detach().to(torch.bfloat16)
    with pytest.raises(ValueError):
        pack_mortar_ternary(cms, threshold=0.3)          # not what it trained at
    with pytest.raises(ValueError):
        pack_mortar_ternary(cms, scale_mode="symmetric")  # not the rule it trained under
    assert cms._values_ternary_mode                       # the refusals left the layer alone
    info = pack_mortar_ternary(cms)
    assert info["scale_mode"] == "norm_match" and info["threshold"] == 0.5
    assert not cms._values_ternary_mode and "mortar_packed" in cms._buffers
    assert "mortar_data" not in cms._parameters
    packed = cms._mortar_effective_data()
    assert packed.dtype == torch.bfloat16 and packed.shape == live.shape
    assert torch.allclose(packed, live, atol=0.0, rtol=1e-2), "bf16 dequant of the same codes and scale"
