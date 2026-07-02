"""Math-tile pruning protection (Olympiad seed growth).

Contract: blocks marked by set_math_protection() are NEVER dropped by
prune_step_blocks(), even when their *training* saliency (block_score_ema) is the
lowest in the layer — so continued NLP pretraining can prune NLP-redundant tiles
without destroying the math circuit.
"""

import torch

from morph.model.layers.block_sparse import CMSBlockLinear


def _layer():
    # 512×512, tile 16 → R=C=32; blocking 128 → Rb=Cb=4 (16 execution blocks).
    layer = CMSBlockLinear(512, 512, tile_size=16, density=1.0, bias=False)
    layer.eval()
    return layer


def test_accumulate_math_scores_populates_buffer():
    layer = _layer()
    x = torch.randn(8, 512)
    layer(x).pow(2).mean().backward()
    before = layer.math_score_ema.clone()
    layer.accumulate_math_scores()
    assert (layer.math_score_ema != before).any(), "math_score_ema not updated"
    assert torch.isfinite(layer.math_score_ema).all()
    # block_score_ema (training) must be UNTOUCHED by the math accumulator.
    assert torch.all(layer.block_score_ema == 0)


def test_protected_blocks_survive_aggressive_pruning():
    layer = _layer()
    tpb = 128 // 16  # 8 tiles per block edge

    # Math saliency: blocks (0,0) and (1,1) are the most math-important.
    ms = torch.zeros(32, 32)
    ms[0:tpb, 0:tpb] = 10.0
    ms[tpb:2 * tpb, tpb:2 * tpb] = 9.0
    layer.math_score_ema = ms
    n = layer.set_math_protection(top_frac=2 / 16)  # top 2 of 16 blocks
    assert n == 2, f"expected 2 protected blocks, got {n}"

    # Training saliency: make those SAME blocks the LOWEST — without protection they'd
    # be the first pruned.
    ts = torch.full((32, 32), 100.0)
    ts[0:tpb, 0:tpb] = 1e-3
    ts[tpb:2 * tpb, tpb:2 * tpb] = 2e-3
    layer.block_score_ema = ts

    for _ in range(12):  # drive well past target density
        layer.prune_step_blocks(prune_rate=0.3, target_density=0.25, blocking=128)

    pm = layer._prune_mask  # [32, 32] tile alive-mask
    assert bool(pm[0:tpb, 0:tpb].all()), "protected block (0,0) was pruned!"
    assert bool(pm[tpb:2 * tpb, tpb:2 * tpb].all()), "protected block (1,1) was pruned!"


def test_without_protection_lowest_saliency_block_is_pruned():
    """Control: same low-saliency block IS dropped when not protected."""
    layer = _layer()
    tpb = 8
    ts = torch.full((32, 32), 100.0)
    ts[0:tpb, 0:tpb] = 1e-3  # lowest, and NOT a per-row best (other blocks in row 0 are 100)
    layer.block_score_ema = ts
    for _ in range(12):
        layer.prune_step_blocks(prune_rate=0.3, target_density=0.25, blocking=128)
    pm = layer._prune_mask
    assert not bool(pm[0:tpb, 0:tpb].any()), "lowest-saliency block survived without protection"


def test_freeze_protected_grads_zeros_only_protected():
    layer = _layer()
    tpb = 8
    ms = torch.zeros(32, 32); ms[0:tpb, 0:tpb] = 10.0  # block (0,0) most math-salient
    layer.math_score_ema = ms
    layer.set_math_protection(top_frac=1 / 16)          # protect that one block

    layer(torch.randn(8, 512)).pow(2).mean().backward()
    g = layer._prune_target_weight().grad
    assert torch.any(g[0:128, 0:128] != 0), "sanity: protected region had grad before freeze"
    layer.freeze_protected_grads()
    # Protected block's elementwise region (8 tiles × 16 = rows/cols 0:128) → grad zeroed.
    assert torch.all(g[0:128, 0:128] == 0), "protected grad not zeroed"
    # Somewhere outside must still carry gradient.
    assert torch.any(g[128:, 128:] != 0), "freeze zeroed non-protected grad too"


def test_frozen_protected_weight_does_not_move_under_sgd():
    layer = _layer()
    tpb = 8
    ms = torch.zeros(32, 32); ms[0:tpb, 0:tpb] = 10.0
    layer.math_score_ema = ms
    layer.set_math_protection(top_frac=1 / 16)
    w0 = layer.weight.detach().clone()
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    for _ in range(5):
        opt.zero_grad()
        layer(torch.randn(8, 512)).pow(2).mean().backward()
        layer.freeze_protected_grads()
        opt.step()
    w1 = layer.weight.detach()
    # Protected region unchanged; rest moved.
    assert torch.allclose(w1[0:128, 0:128], w0[0:128, 0:128]), "protected weight drifted"
    assert not torch.allclose(w1[128:, 128:], w0[128:, 128:]), "non-protected weight frozen too"


def test_clear_protection_restores_droppability():
    layer = _layer()
    tpb = 8
    ms = torch.zeros(32, 32); ms[0:tpb, 0:tpb] = 10.0
    layer.math_score_ema = ms
    layer.set_math_protection(top_frac=1 / 16)
    assert layer._protect_block_mask is not None
    layer.clear_math_protection()
    assert layer._protect_block_mask is None
    ts = torch.full((32, 32), 100.0); ts[0:tpb, 0:tpb] = 1e-3
    layer.block_score_ema = ts
    for _ in range(12):
        layer.prune_step_blocks(0.3, 0.25, 128)
    assert not bool(layer._prune_mask[0:tpb, 0:tpb].any()), "block survived after clearing protection"
