"""Process-global kernel-mode invariant (use_kernels ↔ force_eager).

Build-time contract: MORPHTransformer sets the process-global force_eager flag
from cfg.use_kernels. The flag is process-global (last build wins) — documented
in docs/runtime-invariants.md. This test protects the wiring, not a freeze API.
"""
from __future__ import annotations

import torch

from morph.kernels.triton._eager_flag import force_eager, set_force_eager
from morph.model.transformer import MORPHConfig, MORPHTransformer


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=128,
        max_seq_len=64,
        n_prelude=1,
        n_core=1,
        n_coda=1,
        mean_depth=1,
        max_depth=1,
        bptt_depth=1,
        channel_dims=(32, 20, 12),
        retention=False,
        bigram_hash_vocab=0,
        dropout=0.0,
        hc_use_kernel=False,
    )
    base.update(kw)
    return MORPHConfig(**base)


def test_use_kernels_false_sets_force_eager():
    set_force_eager(False)
    try:
        MORPHTransformer(_tiny(use_kernels=False))
        assert force_eager() is True
    finally:
        set_force_eager(False)


def test_use_kernels_true_clears_force_eager():
    set_force_eager(True)
    try:
        MORPHTransformer(_tiny(use_kernels=True))
        assert force_eager() is False
    finally:
        set_force_eager(False)


def test_last_build_wins_process_global_flag():
    """Documented process-global behaviour: a later build overwrites the flag."""
    try:
        MORPHTransformer(_tiny(use_kernels=False))
        assert force_eager() is True
        MORPHTransformer(_tiny(use_kernels=True))
        assert force_eager() is False
    finally:
        set_force_eager(False)


def test_force_eager_routes_window_attention_to_reference():
    """Sanity: force_eager is consulted by a fused entry point (CPU reference)."""
    from morph.kernels.triton import fused_window_attention as fwa

    q = torch.randn(1, 2, 16, 64)
    k = torch.randn(1, 2, 16, 64)
    v = torch.randn(1, 2, 16, 64)
    set_force_eager(True)
    try:
        out = fwa.fused_window_attention(q, k, v, window_size=8, scale=0.125)
        ref = fwa.fused_window_attention_reference(q, k, v, 8, 0, False, 0.125)
        assert torch.equal(out, ref)
    finally:
        set_force_eager(False)
