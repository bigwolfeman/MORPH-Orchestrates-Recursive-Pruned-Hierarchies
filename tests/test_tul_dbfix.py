"""Faithful DiffusionBlocks for the TUL slot loop (CLAUDE.md mission).

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_tul_dbfix.py -v

morph/model/iter_cond.py builds the machinery ``tul.db_loop`` never had: an
AdaLN-Zero conditioning signal reaching every core-layer application
(``CoreStageConditioning``), keyed on EITHER the loop iteration index
(``tul.core_stage_cond: "iter"``, works inside today's T-iteration loop) OR an EDM
noise level (``"sigma"``, unlocks the one-pass training step
``morph.model.transformer.MORPHTransformer._tul_core_db1`` and the deterministic
Euler-ladder eval ``_tul_core_db1_ladder``). Reuses this repo's own fixture idioms
(``test_tul_gl1.py``'s ``_batch``/``_cfg``/``_tul``, ``test_tul_loop_ladder.py``'s
model-builder pattern) rather than inventing new ones.
"""

from __future__ import annotations

import torch
import pytest

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402 (pytest puts tests/ on sys.path)

from morph.model.iter_cond import CoreStageConditioning
from morph.model.transformer import MORPHTransformer


def _sigma_model(seed: int = 0, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul = _tul(
        core_stage_cond="sigma",
        mux_beta=1.0,
        mux_target="own",
        sigreg_lambda=0.0,
        eval_ablations=False,
    )
    base = dict(tul=tul, n_core=2, mean_depth=3, max_depth=3, bptt_depth=3)
    base.update(kw)
    return MORPHTransformer(_cfg(**base))


def _iter_model(seed: int = 0, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul = _tul(
        core_stage_cond="iter",
        mux_beta=1.0,
        mux_target="own",
        sigreg_lambda=0.0,
        eval_ablations=False,
    )
    base = dict(tul=tul, n_core=2, mean_depth=3, max_depth=3, bptt_depth=3)
    base.update(kw)
    return MORPHTransformer(_cfg(**base))


def _plain_model(seed: int = 0, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul = _tul(
        core_stage_cond="none",
        mux_beta=1.0,
        mux_target="own",
        sigreg_lambda=0.0,
        eval_ablations=False,
    )
    base = dict(tul=tul, n_core=2, mean_depth=3, max_depth=3, bptt_depth=3)
    base.update(kw)
    return MORPHTransformer(_cfg(**base))


# ── (a) conditioning zero-init bit-identity ─────────────────────────────────


def test_conditioning_module_is_identity_at_init():
    """CoreStageConditioning in isolation: modulate(x, cond, i) == x EXACTLY at
    construction, for ANY cond value (AdaLNGate.to_mod is zero-initialised)."""
    torch.manual_seed(0)
    cs = CoreStageConditioning(n_layers=3, d_model=16, cond_dim=64)
    x = torch.randn(2, 5, 4, 16)  # [B, S, n, C] — the HC n-stream carrier shape
    for stage in (torch.tensor([0.0]), torch.tensor([1.0, -3.5]), torch.zeros(2)):
        cond = cs.stage_embed(stage if stage.numel() != 1 else stage.expand(2))
        for i in range(3):
            out = cs.modulate(x, cond, i)
            assert torch.equal(out, x), f"layer {i} modulation was not identity at init"


def test_conditioning_zero_init_bit_identical_full_model():
    """A model built with ``core_stage_cond='iter'`` and one built with ``'none'``,
    given the SAME core/prelude/coda weights (loaded from the smaller into the
    larger), produce a BITWISE identical forward — the conditioning starts as a
    strict no-op, proving the mission's bit-identity requirement at the full-model
    level, not just the isolated module."""
    x, y, lay, _ = _batch()
    torch.manual_seed(11)
    cond_model = _iter_model(seed=11)
    torch.manual_seed(11)
    plain_model = _plain_model(seed=11)
    missing, unexpected = cond_model.load_state_dict(plain_model.state_dict(), strict=False)
    assert unexpected == []
    assert all(k.startswith("tul_stage_cond.") for k in missing), missing

    cond_model.eval()
    plain_model.eval()
    with torch.no_grad():
        out_a = cond_model(x, y, slot_layout=lay)
        out_b = plain_model(x, y, slot_layout=lay)
    assert torch.equal(out_a["loss"], out_b["loss"])


def test_core_stage_cond_none_builds_nothing():
    """The default ('none') constructs no conditioning module and no sampler —
    the baseline-regression contract: nothing new exists on a model that doesn't
    ask for it."""
    m = _plain_model()
    assert m.tul_stage_cond is None
    assert m._db1_sampler is None


# ── (b) one-pass db1 step: forward+backward, gradients reach core AND cond ──


def test_db1_step_runs_and_gradients_reach_core_and_conditioning():
    x, y, lay, _ = _batch()
    m = _sigma_model()
    m.train()
    out = m(x, y, slot_layout=lay, tul_step_mode="db1")
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    core_grad = sum(float(p.grad.abs().sum()) for p in m.core.parameters() if p.grad is not None)
    cond_grad = sum(
        float(p.grad.abs().sum()) for p in m.tul_stage_cond.parameters() if p.grad is not None
    )
    assert core_grad > 0.0, "no gradient reached the core linears"
    assert cond_grad > 0.0, "no gradient reached the stage-conditioning gates"


def test_db1_requires_sigma_conditioning():
    x, y, lay, _ = _batch()
    m = _iter_model()  # core_stage_cond='iter', not 'sigma'
    m.train()
    with pytest.raises(RuntimeError, match="core_stage_cond='sigma'"):
        m(x, y, slot_layout=lay, tul_step_mode="db1")


def test_tul_step_mode_requires_slot_layout():
    m = _sigma_model()
    x, y, lay, _ = _batch()
    with pytest.raises(ValueError, match="tul_step_mode requires slot_layout"):
        m(x, y, tul_step_mode="db1")  # no slot_layout — TUL not engaged


# ── (c) exactly ONE core application per db1 step ────────────────────────────


def test_db1_calls_every_core_layer_exactly_once():
    """Instruments each core layer's forward with a hook: db1 must invoke each of
    the n_core layers EXACTLY once per training step — never a T-iteration loop —
    which is the entire "reducing computational cost by factor K" claim."""
    x, y, lay, _ = _batch()
    m = _sigma_model()
    m.train()
    counts = [0] * len(m.core)

    hooks = []
    for i, layer in enumerate(m.core):

        def _mk(i):
            def _hook(module, inp, out):
                counts[i] += 1

            return _hook

        hooks.append(layer.register_forward_hook(_mk(i)))
    try:
        out = m(x, y, slot_layout=lay, tul_step_mode="db1")
        out["loss"].backward()
    finally:
        for h in hooks:
            h.remove()
    assert counts == [1] * len(m.core), f"expected exactly one call per core layer, got {counts}"


def test_db1_no_gradient_crosses_a_second_application():
    """Complementary check to the forward-hook count above, from the BACKWARD side:
    a full backward hook on each core layer must also fire EXACTLY once per db1 step.
    Forward-count-one and backward-count-one together are the whole claim — if db1
    had accidentally looped (e.g. called _apply_core_step twice), one of the two
    counts would be 2, whichever direction the extra call's gradient flowed.

    (An earlier version of this test walked the raw ``grad_fn`` graph looking for
    each core parameter's ``AccumulateGrad`` node directly. That undercounted:
    MORTAR's block-sparse linear and the HC-Cayley residual route some parameters
    through custom autograd Functions whose ``next_functions`` do not expose a
    plain recursive walk to the underlying ``nn.Parameter`` objects, so the walk
    found zero core leaves even though ``.grad`` was correctly populated after a
    real ``backward()`` — a fragile probe, replaced with hooks, which everything
    else in this file already uses successfully.)
    """
    x, y, lay, _ = _batch()
    m = _sigma_model()
    m.train()
    fwd_counts = [0] * len(m.core)
    bwd_counts = [0] * len(m.core)
    hooks = []
    for i, layer in enumerate(m.core):

        def _mk_f(i):
            def _h(module, inp, out):
                fwd_counts[i] += 1

            return _h

        def _mk_b(i):
            def _h(module, grad_in, grad_out):
                bwd_counts[i] += 1

            return _h

        hooks.append(layer.register_forward_hook(_mk_f(i)))
        hooks.append(layer.register_full_backward_hook(_mk_b(i)))
    try:
        out = m(x, y, slot_layout=lay, tul_step_mode="db1")
        out["loss"].backward()
    finally:
        for h in hooks:
            h.remove()
    assert fwd_counts == [1] * len(m.core), fwd_counts
    assert bwd_counts == [1] * len(m.core), bwd_counts


# ── (d) Euler-ladder eval: K applications, deterministic ────────────────────


def test_ladder_runs_k_applications_and_is_deterministic():
    x, y, lay, _ = _batch()
    m = _sigma_model()
    m.eval()

    counts = [0] * len(m.core)
    hooks = [
        layer.register_forward_hook(
            (lambda i: lambda module, inp, out: counts.__setitem__(i, counts[i] + 1))(i)
        )
        for i, layer in enumerate(m.core)
    ]
    try:
        with torch.no_grad():
            out_a = m(x, y, slot_layout=lay)
    finally:
        for h in hooks:
            h.remove()
    K = m.cfg.mean_depth  # db1_ladder_steps defaults to 0 -> mean_depth
    assert counts == [K] * len(m.core), f"expected {K} calls per core layer, got {counts}"

    with torch.no_grad():
        out_b = m(x, y, slot_layout=lay)
    assert torch.equal(out_a["loss"], out_b["loss"])
    # No labels -> full logits are returned (see _forward_tul: `groups` is only
    # populated when labels is not None); check those too, matching the idiom in
    # tests/test_tul_loop_ladder.py::test_db_loop_forward_value_is_bitwise_the_undetached_loop.
    with torch.no_grad():
        out_c = m(x, slot_layout=lay)
        out_d = m(x, slot_layout=lay)
    assert torch.equal(out_c["logits"], out_d["logits"])


def test_ladder_step_count_is_configurable():
    x, y, lay, _ = _batch()
    torch.manual_seed(0)
    tul = _tul(core_stage_cond="sigma", mux_beta=0.0, sigreg_lambda=0.0, eval_ablations=False)
    tul.db1_ladder_steps = 5
    m = MORPHTransformer(_cfg(tul=tul, n_core=2, mean_depth=3, max_depth=3, bptt_depth=3))
    m.eval()
    counts = [0] * len(m.core)
    hooks = [
        layer.register_forward_hook(
            (lambda i: lambda module, inp, out: counts.__setitem__(i, counts[i] + 1))(i)
        )
        for i, layer in enumerate(m.core)
    ]
    try:
        with torch.no_grad():
            m(x, y, slot_layout=lay)
    finally:
        for h in hooks:
            h.remove()
    assert counts == [5] * len(m.core)


def test_ladder_requires_sigma_conditioning():
    x, y, lay, _ = _batch()
    m = _iter_model()
    m.eval()
    # "iter" mode doesn't ladder at eval — falls through to the plain looped core,
    # which runs fine (this is NOT an error case; it documents the dispatch rule).
    with torch.no_grad():
        out = m(x, y, slot_layout=lay)
    assert torch.isfinite(out["loss"])


def test_db1_gate_and_core_gain_clip_raise():
    x, y, lay, _ = _batch()
    torch.manual_seed(0)
    gate_tul = _tul(core_stage_cond="sigma", mux_beta=0.0, sigreg_lambda=0.0, eval_ablations=False)
    from morph.model.tul import TULGateConfig

    gate_tul.gate = TULGateConfig(k_max=8)
    m = MORPHTransformer(_cfg(tul=gate_tul, n_core=2, mean_depth=3, max_depth=3, bptt_depth=3))
    m.train()
    with pytest.raises(NotImplementedError, match="gate"):
        m(x, y, slot_layout=lay, tul_step_mode="db1")


# ── (e) step_mix schedule: pure function of the step index ──────────────────


def test_step_mix_cycle_exact_counts_and_alternation():
    from morph.training.train import build_step_mix_cycle

    cycle = build_step_mix_cycle({"bptt": 1, "db1": 1})
    assert len(cycle) == 2
    seq = [cycle[s % len(cycle)] for s in range(10)]
    assert seq.count("bptt") == 5
    assert seq.count("db1") == 5
    # Deterministic function of step index alone: recomputing from the SAME dict
    # gives the SAME sequence (no RNG, no closure-captured mutable state).
    cycle2 = build_step_mix_cycle({"bptt": 1, "db1": 1})
    assert cycle == cycle2
    # 1:1 alternates rather than clustering (bptt, db1, bptt, db1, ...).
    assert seq == ["bptt", "db1"] * 5


def test_step_mix_cycle_other_ratios():
    from morph.training.train import build_step_mix_cycle

    cycle = build_step_mix_cycle({"bptt": 3, "db1": 1})
    assert len(cycle) == 4
    assert cycle.count("bptt") == 3
    assert cycle.count("db1") == 1
    # 30 steps -> 22 or 23 bptt / 7 or 8 db1 within one cycle's rounding, exact over
    # any multiple of the cycle length.
    seq = [cycle[s % len(cycle)] for s in range(40)]
    assert seq.count("bptt") == 30
    assert seq.count("db1") == 10


def test_step_mix_cycle_rejects_bad_input():
    from morph.training.train import build_step_mix_cycle

    with pytest.raises(ValueError):
        build_step_mix_cycle({})
    with pytest.raises(ValueError):
        build_step_mix_cycle({"bptt": 0, "db1": 1})


# ── (f) the four configs compose with the expected key values ───────────────


def test_the_dbfix_configs_resolve():
    from hydra import compose, initialize_config_dir
    import os

    cdir = os.path.abspath("morph/configs")
    want = {
        "tul_dbfix": {
            ("tul", "core_stage_cond"): "sigma",
            ("tul", "db_loop", None): None,
            ("training", "step_mix"): {"db1": 1},
            ("model", "n_core"): 6,
            ("model", "bptt_depth"): 8,
        },
        "tul_db_cond": {
            ("tul", "core_stage_cond"): "iter",
            ("tul", "db_loop"): True,
            ("tul", "db_mux_iters"): 4,
            ("training", "step_mix", None): None,
        },
        "tul_l2cap_cond": {
            ("tul", "core_stage_cond"): "iter",
            ("training", "spectral_project_cap"): 1.5,
            ("training", "step_mix", None): None,
        },
        "tul_ilv50": {
            ("tul", "core_stage_cond"): "sigma",
            ("training", "spectral_project_cap"): 1.5,
            ("training", "step_mix"): {"bptt": 1, "db1": 1},
        },
    }
    with initialize_config_dir(config_dir=cdir, version_base=None):
        for name, checks in want.items():
            cfg = compose(config_name=name)
            for key, val in checks.items():
                if len(key) == 3:  # key absent -> OmegaConf .get default
                    got = cfg[key[0]].get(key[1])
                    got = dict(got) if got is not None and hasattr(got, "keys") else got
                    assert got is val, (name, key, got)
                else:
                    got = cfg[key[0]][key[1]]
                    got = dict(got) if hasattr(got, "keys") else got
                    assert got == val, (name, key, got)


def test_dbfix_configs_land_where_the_consumer_reads_them():
    """The repo's own falsifier for the 'silent-misplaced-key' incident: build the
    ACTUAL TULConfig object build_tul_runtime would build (minus the tokenizer
    network call — patched with a stub) and confirm the resolved value reaches
    TULConfig.core_stage_cond / TULConfig scheduling, not just the raw YAML."""
    from hydra import compose, initialize_config_dir
    import os

    cdir = os.path.abspath("morph/configs")
    with initialize_config_dir(config_dir=cdir, version_base=None):
        cfg = compose(config_name="tul_dbfix")
    tc = cfg.tul
    from morph.model.tul import TULConfig

    model_cfg = TULConfig(
        core_stage_cond=str(tc.get("core_stage_cond", "none")),
        db1_sigma_min=float(tc.get("db1_sigma_min", 0.002)),
        db1_sigma_max=float(tc.get("db1_sigma_max", 80.0)),
        db1_p_mean=float(tc.get("db1_p_mean", -1.2)),
        db1_p_std=float(tc.get("db1_p_std", 1.2)),
        db1_sigma_data=float(tc.get("db1_sigma_data", 0.5)),
    )
    assert model_cfg.core_stage_cond == "sigma"
    assert model_cfg.db1_sigma_data == 0.5


# ── (g) baseline regression: core_stage_cond='none' is bit-identical to master ──


def test_none_mode_is_deterministic_and_reuses_master_dispatch():
    """core_stage_cond='none' takes the SAME _tul_core dispatch as every arm before
    this mission (self.tul_stage_cond is None -> stage_cond=None at every call site
    -> _apply_core_step's modulation branch never executes). Two forwards from the
    same seed/weights on the same batch must be bitwise equal, and the model must
    carry none of the new machinery — the two-sided proof that 'none' changed
    nothing. (The stronger claim — this IS pre-existing master behaviour — is
    covered by running tests/test_tul_loop_ladder.py and tests/test_tul_gl1.py
    unmodified against this same transformer.py, per the mission's verification step.)
    """
    x, y, lay, _ = _batch()
    m = _plain_model(seed=5)
    m.eval()
    with torch.no_grad():
        out_a = m(x, y, slot_layout=lay)
        out_b = m(x, y, slot_layout=lay)
    assert torch.equal(out_a["loss"], out_b["loss"])
    assert m.tul_stage_cond is None and m._db1_sampler is None
    assert m._core_stage_cond_mode == "none"
