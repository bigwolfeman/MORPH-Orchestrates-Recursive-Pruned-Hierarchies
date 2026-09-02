"""tg_scoped_kernels — scoped fused kernels under TG restriction.

The flag's job: leave the process-global force_eager flag OFF so the
structurally-safe fused kernels (HC-Cayley, CCA prologue, core window) engage
while every TG-restricted attention branch stays eager by construction.
GPU parity/causality/backward live in the 2026-09-01 ladder prereg amendment
(measured: bitwise-causal at 2 probe positions, max logit diff 0.073 bf16,
+52% sps); these CPU tests pin the config contract and the flag's reach.
"""
from __future__ import annotations

import pytest
import torch

from morph.kernels.triton._eager_flag import force_eager, set_force_eager
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig

V = 64


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=2, max_depth=3, bptt_depth=2,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def test_scoped_without_tg_restrict_raises():
    with pytest.raises(ValueError, match="tg_scoped_kernels.*requires.*tg_restrict"):
        MORPHTransformer(_tiny(tg_scoped_kernels=True))


def test_scoped_leaves_global_flag_off():
    """The one thing the flag DOES: constructing a scoped model must leave the
    process-global force_eager False, where the plain TG model sets it True."""
    prior = force_eager()
    try:
        tul = TULConfig(prefix_k=2, slot_id=4, tg_restrict=True)
        torch.manual_seed(0)
        MORPHTransformer(_tiny(tul=tul))
        assert force_eager() is True
        torch.manual_seed(0)
        MORPHTransformer(_tiny(tul=tul, tg_scoped_kernels=True))
        assert force_eager() is False
    finally:
        set_force_eager(prior)


def test_scoped_construction_byte_identity():
    """The flag builds no parameters and draws no RNG."""
    prior = force_eager()
    try:
        tul = TULConfig(prefix_k=2, slot_id=4, tg_restrict=True)
        torch.manual_seed(7)
        a = MORPHTransformer(_tiny(tul=tul))
        torch.manual_seed(7)
        b = MORPHTransformer(_tiny(tul=tul, tg_scoped_kernels=True))
        sa, sb = a.state_dict(), b.state_dict()
        assert set(sa) == set(sb)
        for k in sa:
            assert torch.equal(sa[k], sb[k]), f"weight mismatch at {k}"
    finally:
        set_force_eager(prior)
