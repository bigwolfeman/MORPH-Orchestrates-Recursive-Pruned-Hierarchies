"""StaticDecodeEngine on the CURRENT plain-model geometry, against the eager kv_cache decode.

The engine (morph/inference/engine.py) was built for a 276M stack whose CSA indexer width
equalled d_head and whose CSA block was 4 tokens of width 32. The 265M plain panel model
has d_indexer 32 against d_head 64, csa_m 8, a 64-wide compressed block and a vocab that
is not a multiple of 16, and the engine refused to build on it (the q_I width assert),
then would have pooled the wrong rows (csa_emit's fixed M=4 / c=32). This file builds a
tiny model with that geometry and checks the engine's logits against
``kv_cache.decode_step`` step by step, teacher-forced on one token stream (so a single
argmax flip cannot hide or fake a state bug), from both entries:

  * ``reset()`` at position 0: the whole stream through the engine (empty window, the
    first CSA block's absent B stream, negative ring rows of the conv history);
  * ``load_from_eager`` at P = 24 (the engine's original entry).

Both run as CUDA-graph replay (``capture()``), the path the decode bench times, at B = 2.
The MLP, x0 and LM-mixer weights are snapped to gamma * {-1, 0, +1} so the engine takes its
ternary GEMV paths, as on the norm_match checkpoint. Two engines: ``exact_weights=True``
(fp32 copies; only reduction order is left, so the tolerance is tight and a state bug
cannot hide under rounding) and the shipped default (bf16 weight copies, a looser
tolerance that covers that rounding).
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="StaticDecodeEngine runs Triton kernels on CUDA")

# Max |log_softmax(engine) - log_softmax(eager)| over every step and stream. Measured on
# this fixture (2026-10-10): exact_weights 3.7e-6 (reset) / 2.8e-6 (convert); shipped bf16
# copies 1.6e-2 / 1.4e-2. Sabotages, each run against TOL_EXACT: the HEAD engine (no fix)
# fails the build on "q_I width != d_head"; csa_emit without the first-block -inf B stream
# reads 4.2e-1 from reset; csa_emit with the block rows swapped 6.8 / 4.4; ring_meta with
# Triton's truncating % on the conv history 3.6 from reset.
TOL_EXACT = 1e-3        # fp32 weight copies: reduction order only
TOL_BF16 = 6e-2         # shipped bf16 weight copies

P_CONVERT = 24          # >= 2 * csa_m: the first CSA block is already eager-emitted
N_TOK = 72              # crosses CSA emits (every 8), HCA emits at 31 and 63, window wrap


def _cfg():
    from morph.model.transformer import MORPHConfig
    return MORPHConfig(
        d_model=512, n_heads=4, n_kv_heads=2, compression=2,   # d_head 64
        d_indexer=32,                                           # != d_head
        vocab_size=1000,                                        # not a multiple of 16
        max_seq_len=256, context_len=256,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=2, max_depth=2, bptt_depth=2,
        channel_dims=(256, 128, 128),
        csa_compress_ratio=8, hca_compress_ratio=32, top_k=16, window_size=32,
        retention=False, n_ve=0, bigram_hash_vocab=4096,
        use_kernels=False, hc_use_kernel=False, dropout=0.0,
    )


def _snap_ternary(w: torch.Tensor) -> None:
    g = w.abs().mean()
    w.copy_(g * torch.round(w / g).clamp(-1, 1))


@torch.no_grad()
def _model():
    from morph.model.transformer import MORPHTransformer
    torch.manual_seed(0)
    model = MORPHTransformer(_cfg()).cuda().eval()
    for blk in list(model.prelude) + list(model.core) + list(model.coda):
        mlp = blk.mlp[0] if hasattr(blk.mlp, "__getitem__") else blk.mlp
        _snap_ternary(mlp.gate_up._cms.weight)
        _snap_ternary(mlp.down._cms.weight)
    for inj in model.x0_injects:
        _snap_ternary(inj.proj.weight)
    _snap_ternary(model.lm_mixer.mix.weight)
    # a non-trivial loop decay and inject scale so those paths carry signal
    for inj in model.x0_injects:
        inj.log_scale.fill_(0.5)
    return model


def _eager_logits(model, toks: torch.Tensor) -> list[torch.Tensor]:
    from morph.inference.kv_cache import MORPHKVCache, decode_step
    cache = MORPHKVCache()
    cache.csa_pool_len = int(model.cfg.context_len)
    return [decode_step(model, toks[:, t], cache).float().clone()
            for t in range(toks.shape[1])]


def _worst(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    w = 0.0
    for x, y in zip(a, b):
        x, y = torch.log_softmax(x.double(), -1), torch.log_softmax(y.double(), -1)
        w = max(w, float((x - y).abs().max()))
    return w


def _engine_vs_eager(model, toks, ref, exact_weights):
    from morph.inference.engine import StaticDecodeEngine
    from morph.inference.kv_cache import MORPHKVCache, decode_step
    B = toks.shape[0]
    eng = StaticDecodeEngine(model, batch_size=B, exact_weights=exact_weights)
    # the geometry this file exists for — fail loudly if the fixture drifts off it
    assert eng.DI == 32 and eng.D == 64 and eng.csa_m == 8
    assert model.cfg.vocab_size % 16 != 0
    eng.capture()

    # entry 1: reset at position 0, every token through the graphs
    eng.reset()
    got = [eng.decode_step(toks[:, t]).float().clone() for t in range(N_TOK)]
    w_reset = _worst(ref, got)

    # entry 2: eager prefill to P_CONVERT, convert, then the graphs
    cache = MORPHKVCache()
    cache.csa_pool_len = int(model.cfg.context_len)
    for t in range(P_CONVERT):
        decode_step(model, toks[:, t], cache)
    eng.load_from_eager(cache)
    got_c = [eng.decode_step(toks[:, t]).float().clone() for t in range(P_CONVERT, N_TOK)]
    w_conv = _worst(ref[P_CONVERT:], got_c)
    return w_reset, w_conv


@torch.no_grad()
def test_engine_matches_eager_on_d_indexer_ne_d_head():
    model = _model()
    g = torch.Generator(device="cuda").manual_seed(1)
    toks = torch.randint(0, model.cfg.vocab_size, (2, N_TOK), device="cuda", generator=g)
    ref = _eager_logits(model, toks)                 # before the engine materialises
    for exact, tol in ((True, TOL_EXACT), (False, TOL_BF16)):
        w_reset, w_conv = _engine_vs_eager(model, toks, ref, exact)
        msg = (f"exact_weights={exact}: worst log-prob diff reset {w_reset:.3e}, "
               f"convert {w_conv:.3e} (tol {tol})")
        print(msg)
        assert w_reset < tol, msg
        assert w_conv < tol, msg
