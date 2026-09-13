"""The split point and the frozen-model contract of `lab/divergence/slot_z_optimize.py`.

The probe's whole claim rests on ONE structural fact: the tensor it substitutes is the
tensor the coda reads. Three checks, on the tiny CPU slot-loop model:

(a) **The split point is right.** Recording z and substituting it back UNCHANGED must
    reproduce the trained forward's token CE bit for bit. A test that only ever sees a
    passing reproduction cannot tell a correct split from a vacuous one, so the same
    substitution is also run with the PRE-loop state (h_0, the loop's entry) and must
    NOT reproduce it.
(b) **The optimisation works on valid slots only.** After 20 Adam steps the token CE is
    lower than the loop's, and the z of every pad / invalid slot is untouched.
(c) **The model is frozen.** No parameter carries a gradient after the optimisation.

CPU, tiny config, no tokenizer (the `test_tul_forward.py` recipe).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10
DEPTH = 2

_LAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "lab", "divergence")


def _probe_module():
    """Import the probe without running its argparse main (`lab/` is not a package)."""
    if _LAB not in sys.path:
        sys.path.insert(0, _LAB)
    spec = importlib.util.spec_from_file_location(
        "slot_z_optimize", os.path.join(_LAB, "slot_z_optimize.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _tiny(tul: TULConfig) -> MORPHConfig:
    return MORPHConfig(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=DEPTH, max_depth=3, bptt_depth=DEPTH,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0, tul=tul)


def _fixture(seed: int = 1234):
    """A frozen tiny slot-loop model in eval mode + one packed batch WITH pad slots."""
    spec = TulLayoutSpec(seq_len=32, prefix_k=2, max_slots=8, slot_id=4)
    tul = TULConfig(prefix_k=2, slot_id=4, slot_depth_fixed=DEPTH,
                    emit_weight=0.0, plast_weight=1.0, token_state_dropout=0.15)
    torch.manual_seed(seed)
    model = MORPHTransformer(_tiny(tul))
    model.eval()
    model.requires_grad_(False)
    rng = np.random.default_rng(0)
    ids = rng.integers(5, V, size=(2, 90))
    ids[ids == spec.slot_id] = 5
    ids[:, ::7] = DOT
    x, y, layout, _ = slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)
    assert (~layout.slot_valid).any(), "the fixture must contain pad slots"
    return model, x, y, layout


def test_recorded_z_reproduces_the_trained_forward_bit_for_bit():
    """(a) The split point. Exact on the loop's z, and NOT exact on the pre-loop state."""
    mod = _probe_module()
    model, x, y, layout = _fixture()
    with mod.ZSplit(model, amp=False) as split:
        out, z_loop, h0 = split.record(x, y, layout, want_groups=True)
        ce_loop = split.token_ce.clone()
        assert split.st["core0"] == DEPTH, split.st["core0"]

        split.replay(x, y, layout, z_loop)
        ce_rep = split.token_ce.clone()
        assert torch.equal(ce_loop, ce_rep), (
            f"the recorded z does not reproduce the forward: {ce_loop.item()!r} vs "
            f"{ce_rep.item()!r} — the split point is not where the probe assumes")

        # The deliberate WRONG split: h_0, the loop's ENTRY state, substituted where the
        # loop's OUTPUT belongs. It must move the CE, or check (a) proves nothing.
        assert not torch.equal(h0, z_loop), "the loop did not move the state at all"
        split.replay(x, y, layout, h0)
        ce_h0 = split.token_ce.clone()
        assert not torch.equal(ce_loop, ce_h0), (
            "substituting the PRE-loop state reproduced the forward's CE, so this test "
            "cannot tell a right split point from a wrong one")


def test_prefix_project_receives_the_substituted_tensor():
    """The identity the probe asserts on every replay is real, not decorative."""
    mod = _probe_module()
    model, x, y, layout = _fixture()
    seen = []
    with mod.ZSplit(model, amp=False) as split:
        _out, z_loop, _h0 = split.record(x, y, layout)
        real = split._real["prefix"]

        def spy(h_slots, lay, l_total, cells=None):
            seen.append(h_slots)
            return real(h_slots, lay, l_total, cells=cells)
        split._real["prefix"] = spy
        z = z_loop.clone()
        split.replay(x, y, layout, z)
    assert len(seen) == 1 and seen[0] is z


def test_a_transformed_z_is_refused():
    """A model that changes h_slots between the core and the write must not pass silently."""
    mod = _probe_module()
    model, x, y, layout = _fixture()
    with mod.ZSplit(model, amp=False) as split:
        _out, z_loop, _h0 = split.record(x, y, layout)
        real = split._real["prefix"]
        split._real["prefix"] = lambda h, lay, lt, cells=None: real(h, lay, lt, cells=cells)
        # simulate a stage between _tul_core and prefix_project
        orig = model._tul_plan_ablate
        model._tul_plan_ablate = lambda h, lay, mode: h * 1.0
        try:
            with pytest.raises(RuntimeError, match="did not receive the substituted z"):
                split.replay(x, y, layout, z_loop)
        finally:
            model._tul_plan_ablate = orig


def test_optimising_z_lowers_the_token_ce_and_freezes_pad_slots():
    """(b) 20 Adam steps beat the loop's z; pad slots do not move."""
    mod = _probe_module()
    model, x, y, layout = _fixture()
    with mod.ZSplit(model, amp=False) as split:
        _out, z_loop, _h0 = split.record(x, y, layout)
        ce_loop = float(split.token_ce)
        zf, curve = mod.optimise_z(split, x, y, layout, z_loop, 0.05, 20,
                                   layout.slot_valid, every=5)
        split.replay(x, y, layout, zf)
        ce_opt = float(split.token_ce)
    assert ce_opt < ce_loop, f"ce_zopt {ce_opt} !< ce_loop {ce_loop}; curve {curve}"
    pad = ~layout.slot_valid
    assert torch.equal(zf[pad], z_loop[pad]), "a pad slot's z moved"
    assert not torch.equal(zf[layout.slot_valid], z_loop[layout.slot_valid])


def test_no_parameter_carries_a_gradient_after_the_optimisation():
    """(c) The model is frozen: the only leaf the CE reaches is z."""
    mod = _probe_module()
    model, x, y, layout = _fixture()
    with mod.ZSplit(model, amp=False) as split:
        _out, z_loop, _h0 = split.record(x, y, layout)
        mod.optimise_z(split, x, y, layout, z_loop, 0.05, 5, layout.slot_valid, every=5)
    bad = [n for n, p in model.named_parameters() if p.grad is not None]
    assert not bad, f"frozen parameters carry a gradient: {bad[:5]}"
    assert not any(p.requires_grad for p in model.parameters())


def test_buckets_partition_the_token_positions():
    """The head / tail split is a partition of the token positions, nothing dropped."""
    mod = _probe_module()
    _model, _x, _y, layout = _fixture()
    b = mod.buckets(layout, 4)
    tok = ~layout.slot_mask
    union = torch.zeros_like(tok)
    for m in b.values():
        assert not (union & m).any(), "the buckets overlap"
        union |= m
    assert torch.equal(union, tok), "the buckets do not cover the token positions"
    assert b["head"].any() and b["tail"].any() and b["before_slot0"].any()
