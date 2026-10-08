"""The strict geometry, tested on the forward: causality, and the loop as the ONLY
cross-span channel (lxtul/SPEC.md section 2).

A tiny Parcae-LXTUL with every zero-init weight randomized (Parcae's scaled-zero init
makes attention outputs exactly 0 at step 0, which would pass any mask test). Rows are
packed by MORPH's own packer from synthetic ids. GPU only: FlexAttention.

    PYTHONPATH=.:$MORPH_ROOT python -m pytest -q lxtul/tests/test_geometry.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import lxtul.data  # noqa: F401  (puts MORPH_ROOT on sys.path)
import parcae_lm
from lxtul.model import LXTULConfig, LXTULParcae

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs CUDA")

V, SLOT, EOS, SEQ, K, S = 512, 4, 1, 128, 4, 16


def _batch(seed: int, B: int = 3):
    from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, pack_tul_batch
    lut = np.zeros(V, dtype=bool)
    lut[10:20] = True
    lut[EOS] = True
    rule = BoundaryRule(lut, min_span=4, span_cap=12, eos_id=EOS)
    spec = TulLayoutSpec(seq_len=SEQ, prefix_k=K, max_slots=S, slot_id=SLOT)
    rng = np.random.default_rng(seed)
    buf = rng.integers(20, V, size=B * (spec.l_total + 1) * 2).tolist()
    for i in range(0, len(buf), 7):        # a boundary token every ~7 ids
        buf[i] = int(rng.integers(10, 20))
    x, y, lay = pack_tul_batch(buf, rule, spec, B)
    for f in ("slot_mask", "bag_id", "slot_index", "slot_valid"):
        setattr(lay, f, getattr(lay, f).cuda())
    return x.cuda(), y.cuda(), lay


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=128, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=256, recurrent_embedding_dimension=128,
        recurrent_intermediation_embedding_dimension=256, n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2, n_layers_in_coda=1, vocab_size=V, block_size=SEQ + K * S)
    m = LXTULParcae(cfg, LXTULConfig(slot_id=SLOT, max_slots=S, cells=K, span_cap=12,
                                     spandec_layers=1)).cuda().eval()
    m.init_slot_seed()
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2 and not p.abs().any():
                p.normal_(0, 0.05)
        m.reg_p_cell_embed.normal_(0, 0.05)
    return m


def _hidden(m, x, lay):
    with torch.no_grad():
        return m(x, None, lay)["hidden"].float()


def test_future_tokens_never_move_the_past(model):
    x, _, lay = _batch(1)
    base = _hidden(model, x, lay)
    for r in range(x.shape[0]):
        n = int(lay.slot_valid[r].sum())
        cut = int(lay.slot_index[r, n // 2]) + K
        x2 = x.clone()
        fut = (~lay.slot_mask[r]).nonzero().squeeze(1)
        fut = fut[fut >= cut]
        x2[r, fut] = (x2[r, fut] + 1 - 20) % (V - 20) + 20
        d = (_hidden(model, x2, lay)[r] - base[r]).abs().amax(-1)
        assert d[:cut].max() == 0, f"row {r}: a token after {cut} moved an earlier position"
        assert d[cut:].max() > 0, f"row {r}: the perturbation did nothing (test is vacuous)"


def test_loop_is_the_only_cross_span_channel(model):
    x, _, lay = _batch(2)
    r, s = 0, 4
    tok = (~lay.slot_mask[r]).nonzero().squeeze(1)
    prev = tok[lay.bag_id[r, tok] == s - 1]
    here = tok[lay.bag_id[r, tok] == s]
    x2 = x.clone()
    x2[r, prev] = (x2[r, prev] + 1 - 20) % (V - 20) + 20
    d = (_hidden(model, x2, lay)[r] - _hidden(model, x, lay)[r]).abs().amax(-1)
    assert d[here].max() > 0, "span s does not see span s-1 even through the loop"
    saved = [p.detach().clone() for p in model.W_prefix]
    try:
        with torch.no_grad():
            for p in model.W_prefix:
                p.zero_()
        dz = (_hidden(model, x2, lay)[r] - _hidden(model, x, lay)[r]).abs().amax(-1)
    finally:
        with torch.no_grad():
            for p, v in zip(model.W_prefix, saved):
                p.copy_(v)
    assert dz[here].max() == 0, "span s sees span s-1 with the loop write cut: a side channel"


def test_coda_reads_the_winner_alone(model):
    """The coda's input holds the final winner at its own cell position and exact zeros
    at the three loser positions of every valid slot (SPEC section 4)."""
    x, _, lay = _batch(3)
    seen = {}
    hook = model.transformer.coda[0].register_forward_pre_hook(
        lambda mod, args: seen.__setitem__("x", args[0].detach().float()))
    try:
        with torch.no_grad():
            final = model(x, None, lay)["final"]
    finally:
        hook.remove()
    xc = seen["x"]
    for r in range(x.shape[0]):
        for s in range(int(lay.slot_valid[r].sum())):
            base = int(lay.slot_index[r, s])
            w = int(final[r, s])
            norms = xc[r, base:base + K].norm(dim=-1)
            assert norms[w] > 0, f"row {r} slot {s}: the winner cell {w} is empty"
            assert all(norms[i] == 0 for i in range(K) if i != w), \
                f"row {r} slot {s}: a loser cell is nonzero {norms.tolist()}"


def test_ungraded_fan_writes_every_cell(model):
    """select=none: no winner, every cell of every valid slot reaches the coda (nonzero)."""
    import dataclasses
    x, _, lay = _batch(4)
    saved = model.tul
    model.tul = dataclasses.replace(saved, select="none")
    seen = {}
    hook = model.transformer.coda[0].register_forward_pre_hook(
        lambda mod, args: seen.__setitem__("x", args[0].detach().float()))
    try:
        with torch.no_grad():
            model(x, None, lay)
    finally:
        hook.remove()
        model.tul = saved
    for r in range(x.shape[0]):
        for s in range(int(lay.slot_valid[r].sum())):
            base = int(lay.slot_index[r, s])
            assert (seen["x"][r, base:base + K].norm(dim=-1) > 0).all(), f"row {r} slot {s}"


def test_open_geometry_is_causal_and_tokens_read_earlier_spans(model):
    """geometry=open: still causal; span s-1's tokens reach span s WITHOUT the cell write."""
    import dataclasses
    x, _, lay = _batch(5)
    m = model
    old = m.tul
    m.tul = dataclasses.replace(old, geometry="open")
    saved = [p.detach().clone() for p in m.W_prefix]
    try:
        base = _hidden(m, x, lay)
        r = 0
        tok = (~lay.slot_mask[r]).nonzero().squeeze(1)
        n = int(lay.slot_valid[r].sum())
        cut = int(lay.slot_index[r, n // 2]) + K
        x2 = x.clone()
        fut = tok[tok >= cut]
        x2[r, fut] = (x2[r, fut] + 1 - 20) % (V - 20) + 20
        d = (_hidden(m, x2, lay)[r] - base[r]).abs().amax(-1)
        assert d[:cut].max() == 0, "open geometry broke causality"
        s = 4
        prev, here = tok[lay.bag_id[r, tok] == s - 1], tok[lay.bag_id[r, tok] == s]
        x3 = x.clone()
        x3[r, prev] = (x3[r, prev] + 1 - 20) % (V - 20) + 20
        with torch.no_grad():
            for p in m.W_prefix:
                p.zero_()
        dz = (_hidden(m, x3, lay)[r] - _hidden(m, x, lay)[r]).abs().amax(-1)
        assert dz[here].max() > 0, "open geometry: tokens do not read earlier spans directly"
    finally:
        m.tul = old
        with torch.no_grad():
            for p, v in zip(m.W_prefix, saved):
                p.copy_(v)


def test_copy_cache_nll_is_causal_and_the_cache_is_used():
    """copy_cache: token_nll before position t ignores x[t:]; a planted repeat is cheaper."""
    torch.manual_seed(0)
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=128, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=256, recurrent_embedding_dimension=128,
        recurrent_intermediation_embedding_dimension=256, n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2, n_layers_in_coda=1, vocab_size=V, block_size=SEQ + K * S)
    m = LXTULParcae(cfg, LXTULConfig(slot_id=SLOT, max_slots=S, cells=K, span_cap=12,
                                     spandec_layers=1, copy_cache=True)).cuda().eval()
    m.init_slot_seed()
    with torch.no_grad():
        m.copy_gate_norm_head.bias.copy_(torch.tensor([0.0, 0.0, -2.0]))   # trust the bigram cache
    x, y, lay = _batch(3)

    def nll(xx, yy):
        with torch.no_grad():
            h = m(xx, None, lay)["hidden"]
            return m.token_nll(h, xx, yy, lay.slot_mask).float()

    base = nll(x, y)
    r = 0
    tok = (~lay.slot_mask[r]).nonzero().squeeze(1)
    cut = int(tok[len(tok) // 2])
    x2, y2 = x.clone(), y.clone()
    fut = tok[tok >= cut]
    x2[r, fut] = (x2[r, fut] + 1 - 20) % (V - 20) + 20
    # targets are next-token ids: recompute every label from the changed inputs
    nxt = torch.roll(x2[r], -1)
    keep = y[r] >= 0
    y2[r] = torch.where(keep, y[r], y[r])
    y2[r, tok[:-1]] = torch.where(y[r, tok[:-1]] >= 0, x2[r, tok[1:]], -100)
    d = (nll(x2, y2)[r] - base[r]).abs()
    prev = tok[tok < cut]
    assert d[prev[:-1]].max() == 0, "the cache let a future token move an earlier score"
    # cache is live: a token whose bigram repeats with the same continuation scores cheaper
    m_off = m.tul
    import dataclasses
    m.tul = dataclasses.replace(m_off, copy_cache=False)
    plain = nll(x, y)
    m.tul = m_off
    from lxtul.cache import packed_copy_probs
    # plant a repeat: row 0's tokens 40..60 (token order) copy tokens 5..25, labels follow
    xp, yp = x.clone(), y.clone()
    tk = (~lay.slot_mask[0]).nonzero().squeeze(1)
    xp[0, tk[40:60]] = xp[0, tk[5:25]]
    yp[0, tk[:-1]] = torch.where(y[0, tk[:-1]] >= 0, xp[0, tk[1:]], -100)
    x, y = xp, yp
    base, plain = nll(x, y), None
    m.tul = dataclasses.replace(m_off, copy_cache=False)
    plain = nll(x, y)
    m.tul = m_off
    c = packed_copy_probs(x, y, lay.slot_mask)
    hit = (c["p_bi"] > 0.99) & (y >= 0) & ~lay.slot_mask
    assert int(hit.sum()) > 0
    assert float((plain - base)[hit].mean()) > 0.0
    # a TRAINING forward with labels (every loss term live) runs and its CE is the mixture's
    m.train()
    o = m(x, y, lay)
    o["loss"].backward()
    assert torch.isfinite(o["loss"]) and m.copy_gate_norm_head.weight.grad is not None
    assert float(o["ce"]) != float(o["ce_model"])
    m.eval()


def _small(**kw):
    torch.manual_seed(0)
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=128, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=256, recurrent_embedding_dimension=128,
        recurrent_intermediation_embedding_dimension=256, n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2, n_layers_in_coda=1, vocab_size=V, block_size=SEQ + K * S)
    m = LXTULParcae(cfg, LXTULConfig(slot_id=SLOT, max_slots=S, cells=K, span_cap=12,
                                     spandec_layers=1, **kw)).cuda().eval()
    m.init_slot_seed()
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2 and not p.abs().any():
                p.normal_(0, 0.05)
        m.reg_p_cell_embed.normal_(0, 0.05)
    return m


def test_pointer_nll_is_causal_and_trains():
    m = _small(pointer_heads=2)
    x, y, lay = _batch(4)

    def nll(xx, yy):
        with torch.no_grad():
            return m.token_nll(m(xx, None, lay)["hidden"], xx, yy, lay.slot_mask).float()

    base = nll(x, y)
    r = 0
    tok = (~lay.slot_mask[r]).nonzero().squeeze(1)
    cut = int(tok[len(tok) // 2])
    x2, y2 = x.clone(), y.clone()
    fut = tok[tok >= cut]
    x2[r, fut] = (x2[r, fut] + 1 - 20) % (V - 20) + 20
    y2[r, tok[:-1]] = torch.where(y[r, tok[:-1]] >= 0, x2[r, tok[1:]], -100)
    d = (nll(x2, y2)[r] - base[r]).abs()
    assert d[tok[tok < cut][:-1]].max() == 0, "the pointer let a future token move an earlier score"
    m.train()
    o = m(x, y, lay)
    o["loss"].backward()
    assert torch.isfinite(o["loss"])
    for p in (m.pointer.q.weight, m.pointer.k.weight, m.pointer.gate_norm_head.weight):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0
    m.eval()


def test_identity_reach_reads_earlier_spans_but_not_the_loop():
    m = _small(identity_reach_heads=2)
    x, _, lay = _batch(5)
    with torch.no_grad():
        base0 = m(x, None, lay)
    with torch.no_grad():
        m.ident_reach.o.weight.normal_(0, 0.05)              # make the branch live
        live = m(x, None, lay)
    # the loop's cells do not see the branch (it acts after the loop, on token positions only)
    assert torch.equal(base0["final"], live["final"])                # same winners
    sm = lay.slot_mask
    assert torch.equal(base0["hidden"][sm], live["hidden"][sm])       # same cells after the coda
    base = live["hidden"].float()
    # still causal
    r = 0
    tok = (~lay.slot_mask[r]).nonzero().squeeze(1)
    n = int(lay.slot_valid[r].sum())
    cut = int(lay.slot_index[r, n // 2]) + K
    x2 = x.clone()
    fut = tok[tok >= cut]
    x2[r, fut] = (x2[r, fut] + 1 - 20) % (V - 20) + 20
    with torch.no_grad():
        d = (m(x2, None, lay)["hidden"][r].float() - base[r]).abs().amax(-1)
    assert d[:cut].max() == 0, "identity reach broke causality"
    # span s now reads span s-1 directly, with every cell write zeroed
    saved = [p.detach().clone() for p in m.W_prefix]
    try:
        with torch.no_grad():
            for p in m.W_prefix:
                p.zero_()
            s = 4
            prev, here = tok[lay.bag_id[r, tok] == s - 1], tok[lay.bag_id[r, tok] == s]
            x3 = x.clone()
            x3[r, prev] = (x3[r, prev] + 1 - 20) % (V - 20) + 20
            dz = (m(x3, None, lay)["hidden"][r] - m(x, None, lay)["hidden"][r]).abs().amax(-1)
        assert dz[here].max() > 0
    finally:
        with torch.no_grad():
            for p, v in zip(m.W_prefix, saved):
                p.copy_(v)


def test_plain_pointer_is_causal_off_equals_plain_and_trains():
    from lxtul.plain_pointer import PlainPointer
    torch.manual_seed(0)
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=128, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=256, recurrent_embedding_dimension=128,
        recurrent_intermediation_embedding_dimension=256, n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2, n_layers_in_coda=1, vocab_size=V, block_size=SEQ,
        mean_recurrence=2, mean_backprop_depth=2)
    m = PlainPointer(cfg.construct_model(gradient_checkpointing=False), heads=2).cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2 and not p.abs().any():
                p.normal_(0, 0.05)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randint(20, V, (2, SEQ), device="cuda", generator=g)
    x[:, 80:100] = x[:, 10:30]                                  # a planted repeat
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    k = dict(num_steps_pair=torch.tensor([2, 0], device="cuda"))
    # Parcae's recurrent state starts from random noise even in eval: seed every forward
    def nll(xx, yy):
        torch.manual_seed(5)
        return m.token_nll(xx, yy, **k)

    base = nll(x, y)
    x2, y2 = x.clone(), y.clone()
    x2[:, 64:] = (x2[:, 64:] + 1 - 20) % (V - 20) + 20
    y2[:, 63:-1] = x2[:, 64:]
    assert (nll(x2, y2)[:, :63] - base[:, :63]).abs().max() == 0
    m.use_pointer = False
    off = nll(x, y)
    m.use_pointer = True
    torch.manual_seed(5)
    lg = m.inner(x, return_logits=True, **k)["logits"]
    ref = torch.nn.functional.cross_entropy(lg.flatten(0, 1), y.flatten().clamp_min(0),
                                            reduction="none").view(y.shape)
    assert torch.allclose(off, torch.where(y >= 0, ref, 0.0), atol=1e-5)
    assert not torch.equal(off, base)
    m.train()
    m.step = 0
    o = m(x, labels=y)
    o["loss"].backward()
    assert torch.isfinite(o["loss"]) and float(m.pointer.q.weight.grad.abs().sum()) > 0
