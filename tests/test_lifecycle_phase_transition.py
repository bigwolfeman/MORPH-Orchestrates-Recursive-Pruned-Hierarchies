"""Prune → carve → route schedule smoke (flags + density, not quality).

Uses a tiny MortarLinear body with dims divisible by carve blocking (128).
Runs on CPU: post-carve *forward* needs the Triton BCSR path (CUDA), so after
compact we only assert topology flags and activate routing without a carved
forward. That is enough to protect the load-bearing lifecycle wiring.
"""
from __future__ import annotations

import torch

from morph.model.layers.block_sparse import CMSBlockLinear
from morph.model.sparsity import MortarLinear
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.training.pruning import PruningSchedule


def _phase_cfg() -> MORPHConfig:
    # d_model / d_ff must be divisible by carve_blocking=128.
    return MORPHConfig(
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        vocab_size=256,
        max_seq_len=64,
        context_len=64,
        n_prelude=1,
        n_core=1,
        n_coda=1,
        mean_depth=1,
        max_depth=1,
        bptt_depth=1,
        channel_dims=(64, 40, 24),
        compression=2,
        csa_compress_ratio=4,
        hca_compress_ratio=32,
        top_k=32,
        window_size=32,
        retention=False,
        bigram_hash_vocab=0,
        use_kernels=False,
        hc_use_kernel=False,
        dropout=0.0,
    )


def _density(model: torch.nn.Module) -> float:
    layers = [m for m in model.modules() if isinstance(m, CMSBlockLinear)]
    assert layers, "expected CMSBlockLinear layers under MortarLinear"
    alive = sum(int(m._prune_mask.sum()) if m._prune_mask is not None else m.R * m.C
                for m in layers)
    total = sum(m.R * m.C for m in layers)
    return alive / total


def test_prune_compact_route_flags_and_density():
    torch.manual_seed(0)
    model = MORPHTransformer(_phase_cfg())
    assert any(isinstance(m, MortarLinear) for m in model.modules())

    sched = PruningSchedule(
        prune_start=0,
        prune_interval=1,
        prune_rate=0.5,
        target_density=0.25,
        compact_step=2,
        route_start=3,
        n_clusters=4,
        activation_ratio=0.5,
        route_scope="core",
        carve_blocking=128,
    )

    B, T = 2, 16
    dens0 = _density(model)
    assert dens0 == 1.0

    # Steps 0–2: need live grads for saliency; compact fires on step 2 after bwd.
    # Tiny dims hit the per-row block floor before target_density=0.25 — we only
    # require density to fall and the phase flags to fire (lifecycle wiring).
    for step in range(3):
        ids = torch.randint(0, model.cfg.vocab_size, (B, T))
        out = model(ids, labels=ids.clone())
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        stats = sched.step(model, step)
        model.zero_grad(set_to_none=True)
        if step < 2:
            assert not sched.is_compact
            assert stats is not None and stats.get("pruning/prune_step") == 1

    assert sched.is_compact, "compact_step=2 must set is_compact"
    assert _density(model) < dens0, "prune phase must drop density before carve"
    assert any(m.is_compact for m in model.modules() if isinstance(m, MortarLinear))
    assert stats is not None and stats.get("pruning/compacted") == 1
    assert stats.get("_rebuild_optimizer") is True

    # Step 3: route activation does not require a post-carve forward.
    stats = sched.step(model, 3)
    assert sched.is_routed
    assert stats is not None and stats.get("routing/activated") == 1
    assert stats.get("_rebuild_optimizer") is True


def _score_state(model: torch.nn.Module) -> list[torch.Tensor]:
    return [m.block_score_ema.clone() for m in model.modules()
            if isinstance(m, CMSBlockLinear) and getattr(m, "block_score_ema", None) is not None]


def test_scoring_is_skipped_when_prune_can_never_fire():
    """prune_start >= total_steps ⇒ no prune event can consume the saliency EMA, so
    accumulate_scores() must not run (it was 50 ms/step of dead work on dense arms).
    With total_steps unknown (0) the historical every-step scoring is kept."""
    torch.manual_seed(0)
    model = MORPHTransformer(_phase_cfg())
    B, T = 2, 16

    def one_step(sched: PruningSchedule, step: int) -> None:
        ids = torch.randint(0, model.cfg.vocab_size, (B, T))
        out = model(ids, labels=ids.clone())
        out["loss"].backward()
        sched.step(model, step)
        model.zero_grad(set_to_none=True)

    dead = PruningSchedule(prune_start=999_999, compact_step=999_999, route_start=0,
                           total_steps=10)
    assert not dead.scoring_live
    before = _score_state(model)
    one_step(dead, 0)
    after = _score_state(model)
    assert before and all(torch.equal(a, b) for a, b in zip(before, after)), \
        "dead schedule must leave the saliency EMA untouched"

    live = PruningSchedule(prune_start=5, compact_step=999_999, route_start=0, total_steps=10)
    assert live.scoring_live
    one_step(live, 0)
    assert any(not torch.equal(a, b) for a, b in zip(after, _score_state(model))), \
        "live schedule must still accumulate scores before prune_start"

    unknown = PruningSchedule(prune_start=999_999, compact_step=999_999, route_start=0)
    assert unknown.scoring_live, "total_steps=0 keeps the historical every-step scoring"
