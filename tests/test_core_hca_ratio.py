"""`core_hca_compress_ratio` re-blocks the CORE's HCA branch and nothing else.

The knob exists because the looped core does not run at the stack's sequence length. Under
TUL the core loops over SLOT positions while prelude and coda run on all of them, and
`GatedPoolCompressor` computes `n_blocks = S // m` — so a token-stream ratio floors to zero
on the slot path and the compressed branch silently produces nothing.
See `.agents/notes/proposed/bug-fix/2026-08-25-hca-compressed-branch-dead-on-slot-path.md`.

CPU only, tiny config, no tokenizer.
"""
from __future__ import annotations

import torch

from morph.model.attention import _CCAHCAAttention
from morph.model.transformer import MORPHConfig, MORPHTransformer


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=64, max_seq_len=128,
        context_len=128, n_prelude=2, n_core=2, n_coda=2, mean_depth=2, max_depth=3,
        bptt_depth=2, channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=32, top_k=8, window_size=16, retention=False,
        bigram_hash_vocab=64, use_kernels=False, hc_use_kernel=False, dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _hca_ratios(model) -> dict[str, list[int]]:
    out = {}
    for name in ("prelude", "core", "coda"):
        out[name] = [blk.attention._impl.compress_ratio
                     for blk in getattr(model, name)
                     if isinstance(blk.attention._impl, _CCAHCAAttention)]
    return out


def test_default_none_leaves_every_section_on_the_stack_ratio():
    r = _hca_ratios(MORPHTransformer(_tiny()))
    assert r == {"prelude": [32], "core": [32], "coda": [32]}, r


def test_the_override_reaches_the_core_and_only_the_core():
    r = _hca_ratios(MORPHTransformer(_tiny(core_hca_compress_ratio=4)))
    assert r["core"] == [4], f"core did not take the override: {r['core']}"
    assert r["prelude"] == [32] and r["coda"] == [32], \
        f"the override leaked outside the core: {r}"


def test_the_compressor_module_takes_the_override_too_not_just_the_wrapper():
    """`_CCAHCAAttention.compress_ratio` and `GatedPoolCompressor.m` must agree.

    They are read in different places — the wrapper's value goes to the attention kernel,
    the compressor's decides `n_blocks`. One without the other is a silent mismatch.
    """
    model = MORPHTransformer(_tiny(core_hca_compress_ratio=4))
    for blk in model.core:
        impl = blk.attention._impl
        if isinstance(impl, _CCAHCAAttention):
            assert impl.compress_ratio == impl.compressor.m == 4
            assert impl.compressor.B_a.shape[0] == 4


def test_the_override_revives_a_branch_that_was_producing_nothing():
    """The whole point: at S below the stack ratio, `out_comp` goes from empty to real."""
    S = 8                                     # below 32, at or above 4
    x = torch.randn(2, S, 64)
    dead = MORPHTransformer(_tiny()).core
    live = MORPHTransformer(_tiny(core_hca_compress_ratio=4)).core
    for blk in dead:
        impl = blk.attention._impl
        if isinstance(impl, _CCAHCAAttention):
            assert impl.compressor(x).shape[1] == 0, \
                "the defect this knob fixes is not reproduced; the test proves nothing"
    n_live = 0
    for blk in live:
        impl = blk.attention._impl
        if isinstance(impl, _CCAHCAAttention):
            c = impl.compressor(x)
            assert c.shape[1] == S // 4 == 2
            assert float(c.detach().abs().sum()) > 0.0, "revived branch still outputs zeros"
            n_live += 1
    assert n_live > 0, "no HCA block in the core; the alternation changed"


def test_setting_the_override_to_the_stack_value_is_the_same_model():
    a = _hca_ratios(MORPHTransformer(_tiny(core_hca_compress_ratio=32)))
    b = _hca_ratios(MORPHTransformer(_tiny()))
    assert a == b


def test_a_forward_still_runs_with_the_override():
    torch.manual_seed(0)
    model = MORPHTransformer(_tiny(core_hca_compress_ratio=4)).eval()
    ids = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        out = model(ids, labels=ids)
    loss = out["loss"] if isinstance(out, dict) else out[0]
    assert torch.isfinite(loss), "the override produced a non-finite loss"
