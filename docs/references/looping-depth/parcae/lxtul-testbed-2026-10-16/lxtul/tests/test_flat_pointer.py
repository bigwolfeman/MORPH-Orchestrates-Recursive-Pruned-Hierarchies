"""The flat GPT arms (gpt8, flat14) and their pointer variants (gpt8_pointer, flat14_pointer).

CPU only. `build()` is called with a tiny shape and `Module.cuda` patched to a no-op. Checks: the
pointer-free gpt arms are today's model (same parameters, same output as a direct construction);
`arm.layers` sets the depth; the pointer wrapper is causal, its mixture is a normalised distribution,
it trains one finite step, and the K-sweep scores a flat model once for every depth.

    PYTHONPATH=.:$MORPH_ROOT python -m pytest -q lxtul/tests/test_flat_pointer.py
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir

import parcae_lm
from lxtul.build import build
from lxtul.evaluate import depth_sweep
from lxtul.plain_pointer import PlainPointer

CONF = str(Path(__file__).resolve().parents[1] / "configs")
SEQ = 48
TINY = ["shape.n_embd=64", "shape.num_attention_heads=2", "shape.num_key_value_heads=2",
        "shape.intermediate_size=128", f"shape.block_size={SEQ}", "run.name=t", "run.wandb_mode=disabled"]


def _build(arm: str, monkeypatch):
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **k: self)
    with initialize_config_dir(CONF, version_base=None):
        cfg = compose("train", overrides=[f"arm={arm}", *TINY])
    torch.manual_seed(0)
    model, mcfg, _ = build(cfg, str(cfg.arm.kind), compile=False)
    return model.eval(), int(mcfg.model.vocab_size), cfg


def _direct_gpt(vocab: int, n_layer: int):
    torch.manual_seed(0)
    gc = parcae_lm.create_config("gpt-small-140m", n_embd=64, num_attention_heads=2, num_key_value_heads=2,
                                 intermediate_size=128, n_layer=n_layer, vocab_size=vocab, block_size=SEQ,
                                 use_fused_head=False)
    return gc.construct_model(gradient_checkpointing=False).eval()


def _fill(m):                         # zero-init weights (output projections) would hide wiring bugs
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2 and not p.abs().any():
                p.normal_(0, 0.05)


@pytest.mark.parametrize("arm,layers", [("gpt8", 8), ("flat14", 14)])
def test_pointer_free_gpt_arms_are_the_plain_construction(arm, layers, monkeypatch):
    m, vocab, _ = _build(arm, monkeypatch)
    ref = _direct_gpt(vocab, layers)
    assert not isinstance(m, PlainPointer) and len(m.transformer.h) == layers
    sd, rd = m.state_dict(), ref.state_dict()
    assert sd.keys() == rd.keys() and all(torch.equal(sd[k], rd[k]) for k in sd)
    x = torch.randint(20, vocab, (2, SEQ), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.equal(m(x, return_logits=True)["logits"], ref(x, return_logits=True)["logits"])


@pytest.mark.parametrize("arm,layers", [("gpt8_pointer", 8), ("flat14_pointer", 14)])
def test_pointer_arm_wraps_the_same_gpt(arm, layers, monkeypatch):
    m, vocab, _ = _build(arm, monkeypatch)
    ref = _direct_gpt(vocab, layers)
    assert isinstance(m, PlainPointer) and len(m.inner.transformer.h) == layers
    sd = m.inner.state_dict()
    assert all(torch.equal(sd[k], v) for k, v in ref.state_dict().items())
    assert m.pointer.h == 4


def _pointer_model(monkeypatch):
    m, vocab, cfg = _build("gpt8_pointer", monkeypatch)
    _fill(m)
    return m, vocab


def test_gpt_pointer_is_causal_and_off_equals_the_plain_loss(monkeypatch):
    m, vocab = _pointer_model(monkeypatch)
    g = torch.Generator().manual_seed(1)
    x = torch.randint(20, vocab, (2, SEQ), generator=g)
    x[:, 30:40] = x[:, 4:14]                                    # a planted repeat the head can use
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    with torch.no_grad():
        base = m.token_nll(x, y)
        x2, y2 = x.clone(), y.clone()
        x2[:, 24:] = (x2[:, 24:] + 1 - 20) % (vocab - 20) + 20
        y2[:, 23:-1] = x2[:, 24:]
        assert (m.token_nll(x2, y2)[:, :23] - base[:, :23]).abs().max() == 0
        m.use_pointer = False
        off = m.token_nll(x, y)
        m.use_pointer = True
        lg = m.inner(x, return_logits=True)["logits"]
        ref = torch.nn.functional.cross_entropy(lg.flatten(0, 1), y.flatten().clamp_min(0),
                                                reduction="none").view(y.shape)
        assert torch.allclose(off, torch.where(y >= 0, ref, 0.0), atol=1e-5)
        assert not torch.equal(off, base)                       # the head changes the score


def test_gpt_pointer_mixture_is_normalised():
    """Small vocab so every target v can be scored: sum_v p(v | prefix) == 1 at each position (the null
    mass returns to the model, so the mixture is a distribution)."""
    V = 64
    m = PlainPointer(_direct_gpt(V, 2), heads=4).eval()
    _fill(m)
    with torch.no_grad():                                       # a sharp, non-trivial head
        m.pointer.gate_norm_head.bias.copy_(torch.tensor([0.0, 1.0, 0.5, -0.5, 0.2]))
        m.pointer.q.weight.normal_(0, 0.5)
        m.pointer.k.weight.normal_(0, 0.5)
    g = torch.Generator().manual_seed(2)
    x = torch.randint(0, V, (1, SEQ), generator=g)
    x[:, 30:40] = x[:, 4:14]
    y = torch.roll(x, -1, 1)
    xs = x.repeat(V, 1)
    for pos in (35, 40):                                        # inside and after the repeat
        ys = y.repeat(V, 1)
        ys[:, pos] = torch.arange(V)                            # row v scores target v at `pos`
        with torch.no_grad():
            p = torch.exp(-m.token_nll(xs, ys)[:, pos])
        assert abs(float(p.sum()) - 1.0) < 1e-4, (pos, float(p.sum()))
        assert float(p.max()) < 1.0


def test_gpt_pointer_trains_one_finite_step(monkeypatch):
    m, vocab = _pointer_model(monkeypatch)
    m.train()
    x = torch.randint(20, vocab, (2, SEQ), generator=torch.Generator().manual_seed(3))
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    out = m(x, labels=y)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    for w in (m.pointer.q.weight, m.pointer.k.weight, m.pointer.gate_norm_head.bias, m.inner.transformer.wte.weight):
        assert w.grad is not None and bool(torch.isfinite(w.grad).all()) and float(w.grad.abs().sum()) > 0
    opt.step()
    assert torch.isfinite(m(x, labels=y)["loss"])


@pytest.mark.parametrize("arm", ["gpt8", "gpt8_pointer"])
def test_depth_sweep_scores_a_flat_model_once(arm, monkeypatch):
    m, vocab, _ = _build(arm, monkeypatch)
    _fill(m)
    x = torch.randint(20, vocab, (2, SEQ), generator=torch.Generator().manual_seed(4))
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    lay = SimpleNamespace(slot_mask=torch.zeros_like(x, dtype=torch.bool))
    sw = depth_sweep(m, "gpt", [(x, y, lay)], [1, 6, 8], slot_id=vocab - 1, device="cpu")
    assert sw["sums"]["1"] == sw["sums"]["6"] == sw["sums"]["8"]
    assert sw["count"] == [SEQ - 1, SEQ - 1]
    with torch.no_grad():
        lg = (m.inner if hasattr(m, "inner") else m)(x, return_logits=True)["logits"].float()
        lg[..., vocab - 1] = float("-inf")
        nll = torch.nn.functional.cross_entropy(lg.flatten(0, 1), y.flatten().clamp_min(0), reduction="none").view(y.shape)
        plain = torch.where(y >= 0, nll, 0.0).sum(1).double().tolist()
    if arm == "gpt8":                                           # no head: sums equal the masked-vocab CE
        assert sw["sums"]["6"] == pytest.approx(plain, rel=1e-4)
    else:                                                       # the mixture changes the score
        assert sw["sums"]["6"] != pytest.approx(plain, rel=1e-4)
