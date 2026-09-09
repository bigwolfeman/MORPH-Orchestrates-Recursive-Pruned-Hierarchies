"""PruningSchedule on the density panel's compressed cadence (arc/run_density.sh,
lab/experiments/planned/2026-09-09-arc-density-panel.md), driven on a TINY CPU model.

The density panel scales the production cadence (`prune_start 3000, prune_interval 167,
prune_rate 0.005, target_density 0.25`) down to fit a 5,000-step run (`prune_start 1500,
prune_interval 25, prune_rate 0.03`, `notul_density_half.yaml` / `notul_density_quarter.yaml`).
This test keeps `prune_rate 0.03` (the actual panel value — the number under test) and
shrinks only `prune_start` / `prune_interval` so a fake training loop finishes in about a
second, per `prune_step_blocks`' own rule (`morph/model/layers/block_sparse.py`):

    target_keep = max(Rb, round(target_density * Rb * Cb))
    alive_{k+1} = max(target_keep, min(floor(alive_k * (1 - rate)), alive_k - 1))

Model geometry (d_model=512, d_ff=512, tile_size 16, carve_blocking 128) is chosen so
every CMSBlockLinear's row-floor (Rb, the >=1-alive-block-per-row guarantee) lands
EXACTLY on 25% of its own total blocks (gate_up: Rb=8, Cb=4, total=32; down: Rb=4, Cb=4,
total=16) — the schedule's floor and the configured target coincide, so the model
actually reaches target_density=0.25, not a higher floor. At rate 0.03 both layer types
sit below the 1/rate~=33 threshold where the multiplicative decay ever drops more than
one block per event, so the whole run is the "one block per event" regime the base.yaml
comment describes for the production down-projection: gate_up needs 24 events (32->8),
down needs 12 (16->4); the schedule keeps firing until the SLOWER layer (gate_up)
finishes, so the panel's own event-count arithmetic (~23/~46 events at the real cadence)
is exercised here as an exact, small-integer analogue.

The checkpoint round-trip (test_prune_state_round_trips_through_a_fresh_model) mirrors
`scripts/tul_samples.py::load_ckpt`, the loader `lab/divergence/core_depth_sweep.py` and
`core_anatomy.py` use: build a fresh model, `load_state_dict(state, strict=False)`, and
require no "material" missing/unexpected keys (rope/cache/freqs excluded, same filter
`load_ckpt` applies). Measured here: `_prune_mask` is registered with
`register_buffer(..., persistent=True)` (the default — no `persistent=False` anywhere in
CMSBlockLinear), so it rides `state_dict` exactly; a freshly constructed model that loads
a pruned checkpoint reads back the IDENTICAL tile mask and the IDENTICAL density, not a
1.0-density misreport. Because the corresponding dead weights are ALSO literally zero in
the checkpoint (`apply_prune_mask` re-zeros them every step), logits are bit-identical
regardless of whether the mask itself persisted — so this test asserts on BOTH facts
(mask/density equality AND logit equality) rather than only the one that would still pass
if the mask were silently dropped.
"""

from __future__ import annotations

import math

import torch

from morph.model.layers.block_sparse import CMSBlockLinear
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.training.pruning import PruningSchedule

V = 64
D_MODEL = 512
D_FF = 512
BLOCKING = 128
PRUNE_START = 5
PRUNE_INTERVAL = 1
PRUNE_RATE = 0.03
TARGET_DENSITY = 0.25


def _cfg(**kw) -> MORPHConfig:
    base = dict(
        d_model=D_MODEL, n_heads=4, n_kv_heads=4, d_ff=D_FF, vocab_size=V,
        max_seq_len=32, context_len=32,
        n_prelude=1, n_core=1, n_coda=1, mean_depth=1, max_depth=1, bptt_depth=1,
        channel_dims=(256, 160, 96), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _model(seed: int = 0) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_cfg())
    # Mirror the panel's saliency choice (morph/training/quant_setup.py wires this from
    # training.cms_score_mode == "taylor"; the density-panel arithmetic does not depend
    # on which score_mode is used, only on which blocks are alive, but this keeps the
    # fixture faithful to what the real arms run).
    for mod in m.modules():
        if isinstance(mod, CMSBlockLinear):
            mod.score_mode = "taylor"
    return m


def _schedule(prune_start: int = PRUNE_START) -> PruningSchedule:
    return PruningSchedule(
        prune_start=prune_start,
        prune_interval=PRUNE_INTERVAL,
        prune_rate=PRUNE_RATE,
        target_density=TARGET_DENSITY,
        compact_step=999_999_999,   # never — prune only, no carve (the panel's contract)
        route_start=999_999_999,    # never — no ReMoE
        carve_blocking=BLOCKING,
    )


def _cms_layers(model: MORPHTransformer) -> list[tuple[str, CMSBlockLinear]]:
    return [(n, mod) for n, mod in model.named_modules() if isinstance(mod, CMSBlockLinear)]


def _predict_layer_curve(total: int, rb: int, rate: float, target_density: float):
    """Reproduce prune_step_blocks' own recursion for ONE layer's block count.

    Returns (final_alive, target_keep, n_events). This is the rule under test, copied
    from morph/model/layers/block_sparse.py::prune_step_blocks verbatim (round/floor/
    max/min in the same order), not re-derived — a drift here would silently stop
    testing the real arithmetic.
    """
    target_keep = max(rb, int(round(target_density * total)))
    cur = total
    events = 0
    while cur > target_keep:
        new_alive = math.floor(cur * (1.0 - rate))
        new_alive = max(target_keep, min(new_alive, cur - 1))
        if new_alive == cur:
            break
        cur = new_alive
        events += 1
    return cur, target_keep, events


def _run_fake_training_loop(model, schedule, n_steps: int, seed_base: int = 1000):
    """forward -> loss.backward() -> schedule.step() -> optimizer.step() -> zero_grad().

    Same ordering as morph/training/train.py's real loop (pruning.py's docstring:
    schedule.step MUST run between backward() and zero_grad()). Returns the per-step
    prune-event stats (None on a no-event step) and the aggregate density series.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    events = []
    density_series = []
    for step in range(n_steps):
        torch.manual_seed(seed_base + step)
        inp = torch.randint(0, V, (2, 16))
        lab = torch.randint(0, V, (2, 16))
        out = model(inp, labels=lab)
        out["loss"].backward()
        stats = schedule.step(model, step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        events.append((step, stats))
        density_series.append(schedule._current_density(schedule._layer_cache))
    return events, density_series


def test_density_is_one_before_prune_start():
    model = _model()
    schedule = _schedule(prune_start=PRUNE_START)
    events, density_series = _run_fake_training_loop(model, schedule, n_steps=PRUNE_START)
    assert all(stats is None for _step, stats in events), (
        "no prune event should fire before prune_start"
    )
    assert all(d == 1.0 for d in density_series), density_series
    for _name, layer in _cms_layers(model):
        assert layer.prune_density() == 1.0


def test_prune_falls_monotonically_and_reaches_target_in_the_predicted_events():
    model = _model()
    schedule = _schedule()
    layers = _cms_layers(model)

    # Predict from the actual built geometry, not a hand count: gate_up and down have
    # different (Rb, Cb) — the schedule's overall density keeps firing until the SLOWER
    # layer type (more events to its floor) reaches its own target_keep.
    curves = {}
    for name, layer in layers:
        rb = layer.out_features // BLOCKING
        cb = layer.in_features // BLOCKING
        total = rb * cb
        curves[name] = _predict_layer_curve(total, rb, PRUNE_RATE, TARGET_DENSITY)
    predicted_events = max(c[2] for c in curves.values())
    predicted_final_step = PRUNE_START + predicted_events - 1

    n_steps = predicted_final_step + 3   # a few steps of margin past predicted completion
    events, density_series = _run_fake_training_loop(model, schedule, n_steps=n_steps)

    # Monotonic non-increasing.
    for a, b in zip(density_series, density_series[1:]):
        assert b <= a + 1e-12, (a, b)

    prune_steps = [step for step, stats in events if stats and stats.get("pruning/prune_step")]
    assert len(prune_steps) == predicted_events, (prune_steps, predicted_events)
    assert prune_steps[-1] == predicted_final_step, (prune_steps[-1], predicted_final_step)

    final_density = density_series[-1]
    assert final_density == TARGET_DENSITY, final_density
    # No event after the predicted final step (schedule stops firing once every layer's
    # own target_keep is hit — the aggregate density stays flat, not undershooting).
    assert density_series[predicted_final_step] == TARGET_DENSITY
    assert all(d == TARGET_DENSITY for d in density_series[predicted_final_step:])


def test_current_density_matches_layer_prune_density_readout():
    model = _model()
    schedule = _schedule()
    _run_fake_training_loop(model, schedule, n_steps=PRUNE_START + 30)

    layers = schedule._layer_cache
    alive = 0
    total = 0
    for _name, layer in layers:
        r, c = layer.R, layer.C
        alive += int(layer._prune_mask.sum().item())
        total += r * c
        # Uniform geometry (every CMS layer of a given kind is the same shape here) means
        # every layer independently converges to density 0.25 exactly.
        assert layer.prune_density() == TARGET_DENSITY, (_name, layer.prune_density())
    manual_density = alive / total
    assert schedule._current_density(layers) == manual_density
    assert manual_density == TARGET_DENSITY


def test_prune_step_prints_the_density_line(capsys):
    model = _model()
    schedule = _schedule()
    _run_fake_training_loop(model, schedule, n_steps=PRUNE_START + 2)
    out = capsys.readouterr().out
    assert f"[prune] step {PRUNE_START}:" in out
    assert "density=" in out
    assert f"(target {TARGET_DENSITY})" in out


def test_prune_state_round_trips_through_a_fresh_model():
    """The `lab/divergence` loaders' semantics (scripts/tul_samples.py::load_ckpt):
    build a fresh model, `load_state_dict(state, strict=False)`, require no material
    missing/unexpected keys (rope/cache/freqs excluded).

    Measured (not assumed): `_prune_mask` persists through state_dict — the fresh
    model's density readout equals the source model's exactly, not a 1.0 misreport.
    The dead weights are ALSO literally zero in the checkpoint (apply_prune_mask
    re-zeros them every step), so eval-mode logits are bit-identical independent of
    whether the mask itself persisted. Both facts are asserted, not just the one
    (logit equality) that a silently-dropped mask would still satisfy.
    """
    model = _model(seed=3)
    schedule = _schedule()
    _run_fake_training_loop(model, schedule, n_steps=PRUNE_START + 30)
    model.eval()

    state = {k: v.clone() for k, v in model.state_dict().items()}

    fresh = _model(seed=99)   # different init RNG — must be fully overwritten by the load
    fresh.eval()
    missing, unexpected = fresh.load_state_dict(state, strict=False)
    mat_missing = [k for k in missing
                   if not any(s in k.lower() for s in ("rope", "cache", "freqs"))]
    mat_unexpected = [k for k in unexpected if "rope" not in k.lower()]
    assert mat_missing == [], mat_missing
    assert mat_unexpected == [], mat_unexpected

    # Fact 1: the mask (and therefore the density readout) round-trips exactly.
    orig_layers = dict(_cms_layers(model))
    fresh_layers = dict(_cms_layers(fresh))
    assert orig_layers.keys() == fresh_layers.keys()
    for name, orig_layer in orig_layers.items():
        fresh_layer = fresh_layers[name]
        assert torch.equal(orig_layer._prune_mask, fresh_layer._prune_mask), name
        assert orig_layer.prune_density() == fresh_layer.prune_density() == TARGET_DENSITY, name

    # Fact 2: eval-mode logits are bit-identical on a fixed input.
    torch.manual_seed(42)
    fixed_input = torch.randint(0, V, (2, 16))
    with torch.no_grad():
        torch.manual_seed(7)
        logits_orig = model(fixed_input)["logits"]
        torch.manual_seed(7)
        logits_fresh = fresh(fixed_input)["logits"]
    assert torch.equal(logits_orig, logits_fresh)
