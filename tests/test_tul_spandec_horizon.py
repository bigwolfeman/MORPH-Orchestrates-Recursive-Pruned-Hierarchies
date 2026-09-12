"""`tul.spandec_horizon` — grade z on spans s+1 .. s+H, not just s+1.

Why. The measured cross-span budget is NOT front-loaded: a FLAT 0.31 nats at every offset
eight or more tokens into a span, plus a 0.96 spike at the first position
(`lab/experiments/failures/2026-09-11-arc-span-budget.md`). The H = 1 decoder's own worth
profile still decays with offset (`slot-spandec-mask`, 480 rows: all_slots 0.708 at offset
0 down to 0.068 at 16+). A target that stops at the next boundary cannot ask z for
anything past it.

What this file pins:

* **H = 1 is bit-identical.** `horizon_span_slots` at H = 1 returns `next_span_slots`
  tensor for tensor, the decoder builds the same tensors, and a whole forward's loss and
  logits match a model built before the key by construction — checked here as equality
  against `next_span_slots` and against an H = 1 model's own output.
* **H = 3 supervises exactly the tokens of the next THREE spans**, asserted against a
  hand-built layout, index set for index set — not "the count looks right".
* **The mask is the same rule at every block**: a slot is supervised at block h only when
  span s+h exists AND is complete. A slot near the end of a row keeps the blocks it has.
* **`spandec_ce` stays the H = 1 part** so a horizon arm's sweep column is comparable with
  every earlier arm's.

CPU only, fp32, tiny config.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import (
    BoundaryRule,
    SlotLayout,
    TulLayoutSpec,
    slot_layout_from_ids,
)
from morph.model.tul_spandec import horizon_span_slots, next_span_slots, span_slots

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=2, max_depth=3, bptt_depth=3,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                mux_beta=0.0, spandec=True, spandec_layers=1, spandec_max_tokens=8)
    base.update(kw)
    return TULConfig(**base)


def _batch(B: int = 2, n: int = 160, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::8] = DOT
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _model(seed: int = 11, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(tul=_tul(**tul_kw))).eval().float()


# ── H = 1 is the shipped target, tensor for tensor ───────────────────────────

def test_horizon_one_is_next_span_slots():
    inp, _lab, layout, _ = _batch()
    a_ids, a_ok = horizon_span_slots(inp, layout, 8, 1)
    b_ids, b_ok = next_span_slots(inp, layout, 8)
    assert torch.equal(a_ids, b_ids) and torch.equal(a_ok, b_ok)


def test_horizon_one_model_is_the_default_model():
    inp, lab, layout, _ = _batch()
    m1 = _model()
    m2 = _model(spandec_horizon=1)
    assert m1.tul_spandec.max_tokens == m2.tul_spandec.max_tokens == 8
    with torch.no_grad():
        o1 = m1(inp, labels=lab, slot_layout=layout)
        o2 = m2(inp, labels=lab, slot_layout=layout)
    assert torch.equal(o1["ce_main"], o2["ce_main"])
    assert torch.equal(o1["loss"], o2["loss"])
    assert float(o1["spandec_ce"]) == float(o2["spandec_ce"])


def test_a_horizon_decoder_is_longer_and_draws_no_extra_rng():
    """The position table grows with H; it is zero-init, so no weight moves."""
    m1 = _model()
    m3 = _model(spandec_horizon=3)
    assert m3.tul_spandec.max_tokens == 24 and m3.tul_spandec.per_span_tokens == 8
    for (n1, p1), (n3, p3) in zip(m1.named_parameters(), m3.named_parameters()):
        assert n1 == n3
        if n1.endswith("tul_spandec.pos"):
            assert p1.shape[0] == 8 and p3.shape[0] == 24
            assert float(p3.detach().abs().sum()) == 0.0
        else:
            assert torch.equal(p1, p3), f"{n1} moved when the horizon changed"


# ── the target index set, hand-built ─────────────────────────────────────────

def _hand_layout() -> tuple[torch.Tensor, SlotLayout]:
    """B=1, L=16, prefix_k=1, 4 spans of 3 tokens each, each with ONE slot cell.

    positions  0 1 2 | 3 | 4 5 6 | 7 | 8 9 10 | 11 | 12 13 14 | 15
    span         0     s0    1     s1    2       s2     3       s3
    """
    bag = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]])
    sm = torch.tensor([[False, False, False, True] * 4])
    idx = torch.tensor([[3, 7, 11, 15]])
    val = torch.tensor([[True, True, True, True]])
    ids = torch.arange(100, 116).view(1, 16)
    layout = SlotLayout(slot_mask=sm, bag_id=bag, slot_index=idx, slot_valid=val,
                        prefix_k=1)
    return ids, layout


def test_h3_supervises_exactly_the_next_three_spans():
    ids, layout = _hand_layout()
    Jp = 4
    got_ids, got_ok = horizon_span_slots(ids, layout, Jp, 3)
    assert got_ids.shape == (1, 4, 12)
    # slot 0: spans 1, 2, 3 -> token ids [104,105,106], [108,109,110], [112,113,114]
    want = {0: [[104, 105, 106], [108, 109, 110], [112, 113, 114]],
            1: [[108, 109, 110], [112, 113, 114], []],
            2: [[112, 113, 114], [], []],
            3: [[], [], []]}
    for s, blocks in want.items():
        for h, toks in enumerate(blocks):
            lo, hi = h * Jp, (h + 1) * Jp
            ok = got_ok[0, s, lo:hi]
            sel = got_ids[0, s, lo:hi][ok].tolist()
            assert sel == toks, f"slot {s} block {h}: got {sel}, want {toks}"


def test_a_horizon_block_needs_its_span_to_be_complete():
    """Block h of slot s is supervised only when slot s+h exists — the `shift` rule."""
    ids, layout = _hand_layout()
    _g, ok = horizon_span_slots(ids, layout, 4, 3)
    # slot 1 has spans 2 and 3 but no span 4: its third block is entirely masked
    assert not bool(ok[0, 1, 8:].any())
    assert bool(ok[0, 1, :4].any()) and bool(ok[0, 1, 4:8].any())


def test_each_block_equals_span_slots_at_that_shift():
    """The concatenation is the same rule read further ahead, with nothing new in it."""
    inp, _lab, layout, _ = _batch()
    cat_ids, cat_ok = horizon_span_slots(inp, layout, 6, 3)
    for h in range(1, 4):
        sid, sok = span_slots(inp, layout, 6, shift=h)
        lo, hi = (h - 1) * 6, h * 6
        assert torch.equal(cat_ids[:, :, lo:hi], sid)
        assert torch.equal(cat_ok[:, :, lo:hi], sok)


# ── the reported columns ─────────────────────────────────────────────────────

def test_spandec_ce_is_the_h1_part_at_eval_and_ce_h_is_the_optimised_term():
    inp, lab, layout, _ = _batch()
    m1 = _model()
    m3 = _model(spandec_horizon=3)
    with torch.no_grad():
        o1 = m1(inp, labels=lab, slot_layout=layout)
        o3 = m3(inp, labels=lab, slot_layout=layout)
    assert float(o3["spandec_horizon"]) == 3.0
    assert float(o1["spandec_horizon"]) == 1.0
    # the H=1 column exists on both and is a per-token CE, not the H=3 mean
    assert float(o3["spandec_ce"]) != float(o3["spandec_ce_h"])
    assert float(o3["spandec_n_tokens"]) > float(o3["spandec_n_tokens_h1"]) > 0
    # a fresh model's block-0 CE is close to the H=1 model's: same target, same weights
    # for everything but the decoder's own private stream, so only the ballpark is pinned.
    assert abs(float(o3["spandec_ce"]) - float(o1["spandec_ce"])) < 1.0


def test_a_training_step_logs_ce_h_and_skips_the_extra_readout():
    """`spandec_ce` costs a whole second [B, S, J, V] readout, so training omits it."""
    inp, lab, layout, _ = _batch()
    m3 = _model(spandec_horizon=3).train()
    out = m3(inp, labels=lab, slot_layout=layout)
    assert "spandec_ce_h" in out and "spandec_ce" not in out


def test_the_gradient_reaches_z_from_every_block():
    """z must feel the FAR spans, or the horizon is decoration."""
    inp, _lab, layout, _ = _batch()
    m3 = _model(spandec_horizon=3)
    dec = m3.tul_spandec
    ids, valid = horizon_span_slots(inp, layout, dec.per_span_tokens, 3)
    z = torch.randn(inp.shape[0], layout.max_slots, m3.cfg.d_model, requires_grad=True)
    w = m3.embed.lm_weight().detach()
    st = dec.decode(z, ids, valid, w)
    Jp = dec.per_span_tokens
    for h in range(3):
        sel = valid[:, :, h * Jp:(h + 1) * Jp]
        assert bool(sel.any()), f"fixture: block {h} must be supervised somewhere"
        g, = torch.autograd.grad(st[:, :, h * Jp:(h + 1) * Jp][sel].sum(), z,
                                 retain_graph=True)
        assert float(g.abs().sum()) > 0, f"block {h}'s loss does not reach z"


# ── refusals ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw,match", [
    (dict(spandec_horizon=0), "spandec_horizon"),
    (dict(spandec=False, spandec_layers=2, spandec_max_tokens=0, spandec_horizon=3),
     "silently ignored"),
    (dict(spandec_horizon=3, oracle_z=True), "oracle_z"),
])
def test_horizon_refusals(kw, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _tul(**kw)


def test_span_slots_still_refuses_a_negative_shift():
    inp, _lab, layout, _ = _batch()
    with pytest.raises(ValueError, match="shift must be >= 0"):
        span_slots(inp, layout, 4, shift=-1)
