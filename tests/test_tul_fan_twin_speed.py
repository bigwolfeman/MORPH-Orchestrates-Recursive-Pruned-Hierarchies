"""The fan's EMA target twin, speed work (graph-step, agent twinfan, 2026-10-08).

Files: morph/model/transformer.py (`_tul_front_lookups`, `_tul_front(lookups=)`,
`_tul_fan_target(lookups=)`, the forward hand-off, `model.fan_lsel_pick_bf16` at the per-pass
pick), morph/model/tul_fan_route.py (`FanTargetFront.run_blocks / compile_blocks_`, the
multi-tensor EMA update, `FanLatentHead.forward_bf16`), morph/training/train.py
(`model.fan_twin_compile`).

What each test pins:
  * The twin's target computed from the live front's table reads is BIT-IDENTICAL to the
    target from fresh reads and to an independent rebuild of the old inline twin front
    (train mode, embedding dropout on: the twin must not see the live dropout).
  * A training forward reads the tables ONCE (the twin reuses the live reads).
  * The multi-tensor EMA update equals the per-tensor `lerp_` bit for bit (CPU; CUDA when
    a GPU is present).
  * `compile_blocks_` routes the twin's forward through the given callables, keeps every
    state_dict name, and the EMA update still reaches what the callables run.
  * The bf16 pick head: the picks that differ from the fp32 head's are near ties only.
  * `model.fan_target_online`: z is the online prelude's pooling, detached, bit for bit,
    and the twin's blocks are never run.
  * The new keys refuse the models they do not apply to.

CPU, fp32, the tests/test_tul_fan.py / tests/test_tul_lxfan.py strict fixtures.
"""
from __future__ import annotations

import pytest
import torch

from morph.model.transformer import MORPHTransformer, span_ce_index
from morph.model.tul_fan_route import lsel_distance, pooled_span_states
from test_tul_fan import _batch
from test_tul_fan_lsel import _lsel
from test_tul_lx_credit import M
from test_tul_lxfan import _build, _fan_kw


def _old_twin_z(m: MORPHTransformer, inp, lab, layout, fkw, freset):
    """The twin's target as the tree before 2026-10-08 computed it: its OWN table reads
    inline (no dropout), the twin's projections, `_front_tail(twin=)`, the pooling."""
    twin = m.__dict__["_fan_target"]
    tok = m.tul.slot_input(m.embed(inp), layout, add_e_slot=True)
    bg = m.embed.get_bigram(inp)
    bg = m.tul.slot_input(bg, layout, add_e_slot=False) if bg is not None else None
    ve = [m.tul.slot_input(twin.value_embeds[k].precompute(m.value_embed_tables[k](inp)),
                           layout, add_e_slot=False) for k in range(len(m._ve_layer_map))]
    xt, _ = m._front_tail(tok, inp, bg, ve or None, attn_kwargs=fkw, ret_reset_mask=freset,
                          twin=twin)
    gid, keep_tok, _lab, g_bins = span_ce_index(lab, layout)
    return pooled_span_states(xt, gid, keep_tok, g_bins)


def test_twin_target_from_the_live_reads_is_the_old_target_bit_for_bit():
    _ids0, inp, lab, layout = _batch(M)
    m = _lsel(model_kw={"dropout": 0.1}).train()       # embed_drop live on the online x
    assert len(m._ve_layer_map) > 0 and m.embed.get_bigram(inp) is not None
    fkw, freset, _, _ = m._tul_tg_kwargs(layout)
    lk = m._tul_front_lookups(inp, layout)
    torch.manual_seed(7)
    x, _, _ = m._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset, lookups=lk)
    with torch.no_grad():
        t_lk = m._tul_fan_target(inp, lab, layout, fkw, freset, x, lookups=lk)
        t_fresh = m._tul_fan_target(inp, lab, layout, fkw, freset, x)
        z_old = _old_twin_z(m, inp, lab, layout, fkw, freset)
    assert torch.equal(t_lk["z"], t_fresh["z"])
    assert torch.equal(t_lk["z"], z_old)
    assert torch.equal(t_lk["ok"], t_fresh["ok"]) and torch.equal(t_lk["zo"], t_fresh["zo"])
    # The online front under the same lookups is the online front of fresh reads (same
    # dropout draw): the hand-off moves nothing on the live side either.
    torch.manual_seed(7)
    x2, _, _ = m._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
    assert torch.equal(x, x2)


def test_a_training_forward_reads_the_tables_once():
    _ids0, inp, lab, layout = _batch(M)
    m = _lsel().train()
    calls = []
    orig = m._tul_front_lookups

    def _spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    m._tul_front_lookups = _spy
    m(inp, labels=lab, slot_layout=layout)
    assert len(calls) == 1


def _perturbed_pair(device: str):
    m = _lsel().to(device)
    twin = m.__dict__["_fan_target"]
    twin.to(device)
    g = torch.Generator(device="cpu").manual_seed(3)
    with torch.no_grad():
        for p in m.prelude.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g).to(device))
    return m, twin


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_multi_tensor_ema_update_is_the_per_tensor_lerp(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    m, twin = _perturbed_pair(device)
    mom = 0.996
    pairs = twin._pairs(m, int(m.cfg.n_prelude))
    want = []
    for tw, lv in pairs:
        if tw.is_floating_point():
            want.append(tw.detach().clone().lerp_(lv.detach().to(tw.dtype), 1.0 - mom))
        else:
            want.append(lv.detach().clone())
    assert any(not torch.equal(w, tw) for w, (tw, _) in zip(want, pairs))
    twin.ema_update_(m, int(m.cfg.n_prelude), mom)
    for w, (tw, _) in zip(want, twin._pairs(m, int(m.cfg.n_prelude))):
        assert torch.equal(tw, w)


class _Counted(torch.nn.Module):
    """A stand-in for the trainer's block compile: forwards to the block, counts calls."""

    def __init__(self, block, log):
        super().__init__()
        self.block, self.log = block, log

    def forward(self, *a, **k):
        self.log.append(1)
        return self.block(*a, **k)


def test_compile_blocks_routes_the_forward_and_keeps_the_state_names():
    _ids0, inp, lab, layout = _batch(M)
    m = _lsel().eval()
    twin = m.__dict__["_fan_target"]
    names = list(twin.state_dict().keys())
    fkw, freset, _, _ = m._tul_tg_kwargs(layout)
    with torch.no_grad():
        x, _, _ = m._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
        z0 = m._tul_fan_target(inp, lab, layout, fkw, freset, x)["z"]
    log: list = []
    twin.compile_blocks_(lambda b: _Counted(b, log))
    assert list(twin.state_dict().keys()) == names
    with torch.no_grad():
        z1 = m._tul_fan_target(inp, lab, layout, fkw, freset, x)["z"]
    assert len(log) == int(m.cfg.n_prelude)
    assert torch.equal(z0, z1)
    # The EMA update moves the parameters the routed callables run.
    with torch.no_grad():
        for p in m.prelude.parameters():
            p.add_(0.05)
    twin.ema_update_(m, int(m.cfg.n_prelude), 0.5)
    with torch.no_grad():
        z2 = m._tul_fan_target(inp, lab, layout, fkw, freset, x)["z"]
    assert not torch.equal(z2, z1)


def test_bf16_pick_head_differs_from_fp32_only_at_near_ties():
    m = _lsel()
    head = m.tul_fan_lsel_head
    g = torch.Generator().manual_seed(11)
    d = int(m.cfg.d_model)
    cells = torch.randn(64, 16, M, d, generator=g)
    z = torch.nn.functional.layer_norm(torch.randn(64, 16, d, generator=g), (d,))
    with torch.no_grad():
        d32 = lsel_distance(head(cells), z)
        d16 = lsel_distance(head.forward_bf16(cells), z)
    p32, p16 = d32.argmin(-1), d16.argmin(-1)
    # bf16 GEMMs with fp32 accumulation: each distance within 1 % of the fp32 one ...
    assert torch.allclose(d16, d32, rtol=1e-2, atol=0.0)
    # ... so a pick can only flip where the fp32 margin is inside that band.
    flip = p32 != p16
    if bool(flip.any()):
        margin = (d32.gather(-1, p16.unsqueeze(-1)) - d32.gather(-1, p32.unsqueeze(-1)))
        assert bool((margin.squeeze(-1)[flip] <= 2e-2 * d32.mean(-1)[flip]).all())
    assert flip.float().mean() < 0.05


def test_online_target_is_the_detached_online_pool_and_skips_the_twin():
    _ids0, inp, lab, layout = _batch(M)
    m = _lsel(model_kw={"fan_target_online": True}, fan_lsel_train_follow="router").train()
    twin = m.__dict__["_fan_target"]
    ran: list = []
    twin.compile_blocks_(lambda b: (lambda *a, **k: (ran.append(1), b(*a, **k))[1]))
    fkw, freset, _, _ = m._tul_tg_kwargs(layout)
    x, _, _ = m._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
    t = m._tul_fan_target(inp, lab, layout, fkw, freset, x)
    gid, keep_tok, _lab, g_bins = span_ce_index(lab, layout)
    want = pooled_span_states(x.detach(), gid, keep_tok, g_bins)
    assert ran == []
    assert torch.equal(t["z"], want) and not t["z"].requires_grad
    assert t["zo"].requires_grad and torch.equal(t["zo"].detach(), want)
    m(inp, labels=lab, slot_layout=layout)                 # a whole training forward
    assert ran == []


def test_the_keys_refuse_models_they_do_not_apply_to():
    with pytest.raises(ValueError, match="fan_div_fast"):
        _build(model_kw={"fan_div_fast": True}, **_fan_kw(M, fan_repel_mode="cos"))
    with pytest.raises(ValueError, match="fan_lsel_pick_bf16"):
        _build(model_kw={"fan_lsel_pick_bf16": True}, **_fan_kw(M))
    with pytest.raises(ValueError, match="fan_target_online"):
        _build(model_kw={"fan_target_online": True}, **_fan_kw(M))
    with pytest.raises(ValueError, match="fan_target_online"):          # teacher drives
        _lsel(model_kw={"fan_target_online": True}, fan_lsel_train_follow="teacher")
    _build(model_kw={"fan_div_fast": True}, **_fan_kw(M))           # epivol: builds
