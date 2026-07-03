"""Incremental decode / KV-cache smoke vs full-forward last position.

Protects that decode_step is wired and agrees with the full forward at the
final prompt position (G1 in ignore/verify_kv_cache.py). Retention is off so
cross-iteration carry non-causality does not widen the tolerance.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from morph.inference.kv_cache import MORPHKVCache, decode_step
from morph.kernels.triton._eager_flag import set_force_eager
from morph.model.transformer import MORPHConfig, MORPHTransformer


def _decode_cfg() -> MORPHConfig:
    return MORPHConfig(
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=128,
        max_seq_len=64,
        context_len=64,
        n_prelude=1,
        n_core=1,
        n_coda=1,
        mean_depth=1,
        max_depth=1,
        bptt_depth=1,
        channel_dims=(32, 20, 12),
        compression=2,
        csa_compress_ratio=4,
        hca_compress_ratio=32,
        top_k=32,
        window_size=32,
        retention=False,
        # decode_step indexes get_bigram(...); vocab=0 returns None and is not handled there.
        bigram_hash_vocab=256,
        use_kernels=False,
        hc_use_kernel=False,
        dropout=0.0,
    )


def test_decode_step_last_position_matches_full_forward():
    set_force_eager(True)
    try:
        torch.manual_seed(0)
        model = MORPHTransformer(_decode_cfg()).eval()
        N = 8
        prompt = torch.randint(0, model.cfg.vocab_size, (1, N))

        with torch.no_grad():
            ref = model(prompt)["logits"][0, -1].float()

            cache = MORPHKVCache()
            cache.csa_pool_len = int(model.cfg.context_len)
            logit = None
            for p in range(N):
                logit = decode_step(model, prompt[:, p], cache)
            assert logit is not None
            dec = logit[0].float()

        # Last position: no future tokens; should match within fp noise.
        max_abs = (ref - dec).abs().max().item()
        rel = max_abs / (ref.abs().max().item() + 1e-9)
        cos = F.cosine_similarity(ref.unsqueeze(0), dec.unsqueeze(0)).item()
        assert cos > 0.99, f"decode vs full-forward cos={cos:.4f} too low"
        assert rel < 5e-2, f"decode vs full-forward rel={rel:.2e} too large"
    finally:
        set_force_eager(False)


def test_decode_step_advances_cache_position():
    set_force_eager(True)
    try:
        torch.manual_seed(1)
        model = MORPHTransformer(_decode_cfg()).eval()
        cache = MORPHKVCache()
        cache.csa_pool_len = int(model.cfg.context_len)
        assert cache.pos == 0
        tok = torch.randint(0, model.cfg.vocab_size, (1,))
        with torch.no_grad():
            logits = decode_step(model, tok, cache)
        assert logits.shape == (1, model.cfg.vocab_size)
        assert cache.pos == 1
        assert torch.isfinite(logits).all()
    finally:
        set_force_eager(False)
