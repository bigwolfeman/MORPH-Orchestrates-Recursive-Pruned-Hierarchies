"""Shared fixtures for the dmorph tests (tests/test_dmorph_*.py).

Tiny flat stack, CPU only (``use_kernels=False``), no tokenizer: 2 prelude + 2 coda layers
(``n_core=0``), so ``n_blocks=2`` gives 2-layer blocks and ``n_blocks=4`` one layer per
block, and retention ON at layer 1 of each section so the GLA branch runs inside the
noisy stream too.
"""

from __future__ import annotations

import numpy as np
import torch

from morph.model.dmorph import DmCtx, DmorphConfig
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10
D = 64


def tiny_cfg(**kw) -> MORPHConfig:
    base = dict(
        d_model=D, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=2, n_core=0, n_coda=2, mean_depth=2, max_depth=3, bptt_depth=2,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=True, retention_layers=(1,), retention_chunk=8,
        bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False, dropout=0.0,
        tul=TULConfig(prefix_k=2, slot_id=4),
    )
    base.update(kw)
    return MORPHConfig(**base)


def dm_cfg(**kw) -> DmorphConfig:
    base = dict(arm="tok", n_blocks=2, source_std=1.0 / np.sqrt(D), in_gain=float(np.sqrt(D)))
    base.update(kw)
    return DmorphConfig(**base)


def rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def spec(**kw) -> TulLayoutSpec:
    base = dict(seq_len=32, prefix_k=2, max_slots=5, slot_id=4)
    base.update(kw)
    return TulLayoutSpec(**base)


def batch(sp=None, B=3, n=90, seed=0):
    sp = sp or spec()
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == sp.slot_id] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), rule(), sp)


def model(dmorph: DmorphConfig | None, seed: int = 1234, **cfg_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(tiny_cfg(dmorph=dmorph, **cfg_kw))


def wake_stream(m: MORPHTransformer, seed: int = 7) -> None:
    """Move the zero-init gates and velocity head off zero.

    At construction the noisy stream is EXACTLY inert (v̂ ≡ 0), which would make any
    leak or routing probe vacuous — the testbed learned this the hard way
    (``tests/test_no_leak.py``: "Zero-init AdaLN makes every block an identity ... Break
    that first, or these tests are vacuous")."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for gate in list(m.dmorph.gates) + [m.dmorph.v_gate]:
            gate.to_mod.weight.normal_(0.0, 0.05, generator=g)
            gate.to_mod.bias.normal_(0.0, 0.05, generator=g)
        m.dmorph.W_v.weight.normal_(0.0, 0.1, generator=g)


@torch.no_grad()
def clean_pass(m: MORPHTransformer, x, layout):
    """The clean forward WITH the K/V capture: ``(xh, caps, ctx)``."""
    n_total = m.cfg.n_prelude + m.cfg.n_core + m.cfg.n_coda
    caps = [dict() for _ in range(n_total)]
    h, x0, bg, ve = m._tul_front(x, layout, attn_caps=caps)
    h = m._core_region(h, x0, bg, x)
    xh = m._back_region(h, x0, bg, x, attn_caps=caps)
    return xh, caps, DmCtx(x0=x0, bigram=bg, input_ids=x, ve_bagged=ve)
