"""The per-pass gradient bookkeeping of `lab/divergence/slot_gradient_probe.py`.

The probe's claim is that a shared core weight's leaf gradient splits exactly into the
loop passes that produced it. Two mechanisms are involved and both are checked here:

* the TAP (a `parametrize` parametrization whose backward banks the incoming gradient
  under the pass index its forward ran at) — `sum_t tap_t == leaf.grad`;
* the CROSS-CHECK (`dW_t = g_t^T x_t` from module forward/tensor hooks) — a second,
  independent derivation of the same per-pass quantity on the layers whose module
  forward actually runs.

A test that only ever sees a passing check cannot tell a correct bookkeeping from a
check that always passes, so the cross-check is also run against a deliberately wrong
outer product and must fail.

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
DEPTH = 3

_LAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "lab", "divergence")


def _probe_module():
    """Import the probe without running its argparse main.

    `lab/` is not a package; the probe adds its own directory to `sys.path` for
    `_build` / `_rows`, so the test does the same before loading it.
    """
    if _LAB not in sys.path:
        sys.path.insert(0, _LAB)
    spec = importlib.util.spec_from_file_location(
        "slot_gradient_probe", os.path.join(_LAB, "slot_gradient_probe.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _spec() -> TulLayoutSpec:
    return TulLayoutSpec(seq_len=32, prefix_k=2, max_slots=5, slot_id=4)


def _model(seed: int = 7) -> MORPHTransformer:
    tul = TULConfig(prefix_k=2, slot_id=4, mux_beta=1.0, tokens_through_core=False,
                    slot_depth_fixed=DEPTH, slot_max_depth=DEPTH)
    cfg = MORPHConfig(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=2, max_depth=4, bptt_depth=8,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0, ckpt_grad_iters=0, slot_gain_lambda=0.0, slot_cot_clip=4.0,
        core_fixed_point_lambda=1.0, tul=tul)
    torch.manual_seed(seed)
    m = MORPHTransformer(cfg)
    m.train()
    m._probe_loop = True
    m._probe_cot = True
    return m


def _batch():
    rng = np.random.default_rng(3)
    ids = rng.integers(5, V, size=(2, 90))
    ids[ids == 4] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), _spec())[:3]


class Rig:
    """The probe's instrumentation on a tiny model: taps, hooks and one backward."""

    def __init__(self, probe, model, sources=("s",)):
        self.probe, self.model = probe, model
        self.targets = probe.weight_modules(model)
        self.names = [n for n, _ in self.targets]
        self.bank = probe.Bank(self.names, list(sources), DEPTH)
        self.state = {"pass": -1, "source": None, "stray": [], "crosscheck": False,
                      "want_x": True}
        self.leaves = probe.install_taps(model, self.targets, self.state, self.bank)
        self.xlayers = probe.hookable_linears(model)
        self.xstore = {n: [None] * DEPTH for n, _ in self.xlayers}
        self.handles = probe.hook_crosscheck(model, self.xlayers, self.state, self.xstore)
        self.handles.append(model.core[0].register_forward_pre_hook(
            lambda _m, _a: self.state.__setitem__("pass", self.state["pass"] + 1)))

    def run(self, source="s", term="loss"):
        inp, labels, layout = _batch()
        self.state["pass"] = -1
        out = self.model(inp, labels=labels, slot_layout=layout)
        self.model.zero_grad(set_to_none=True)
        self.model._loop_cot = {}
        self.state["source"] = source
        self.state["crosscheck"] = True
        out[term].backward()
        self.state["source"] = None
        self.state["crosscheck"] = False
        return out

    def leaf_grads(self):
        return {n: (self.leaves[n].grad.detach().to(torch.float32)
                    if self.leaves[n].grad is not None else None)
                for n in self.names}

    def close(self):
        for h in self.handles:
            h.remove()


def test_per_pass_taps_sum_to_the_leaf_gradient():
    """`sum_t tap_t == leaf.grad` on EVERY tapped core weight, to 1e-4 relative.

    This is the probe's own self-check, on a model small enough to be exact in fp32. It
    is what proves the tap covered every access: a weight the tap missed would have a
    leaf gradient with nothing banked against it.
    """
    probe = _probe_module()
    model = _model()
    rig = Rig(probe, model)
    try:
        rig.run()
        stats = probe.pass_stats(rig.bank.b["s"], rig.names, DEPTH, rig.leaf_grads(), 1e-4)
    finally:
        rig.close()
    sc = stats["selfcheck"]
    assert rig.names, "no core weights were tapped"
    assert sc["weights_checked"] >= len(rig.names) - 4, (
        f"only {sc['weights_checked']} of {len(rig.names)} weights carry a gradient; "
        f"inert: {sc['weights_without_gradient']}")
    assert sc["pass"], (f"self-check failed: max rel err {sc['max_rel_err']:.3e} on "
                        f"{sc['worst_weight']}")
    assert not rig.state["stray"], rig.state["stray"][:3]
    assert rig.state["pass"] + 1 == DEPTH


def test_gTx_crosscheck_matches_the_tap_on_every_pass():
    """`g_t^T x_t` from the module hooks equals the tap's per-pass gradient.

    Two independent derivations of the same per-pass quantity. The attention projections
    do not call their `nn.Linear` forward, so this covers a subset — the MLP and two of
    the CCA linears — which is exactly why the tap and not the hooks is the measurement.
    """
    probe = _probe_module()
    model = _model()
    rig = Rig(probe, model)
    try:
        rig.run()
        checked, worst = 0, 0.0
        for n, _m in rig.xlayers:
            parts = rig.xstore[n]
            if any(p is None for p in parts):
                continue
            for t, p in enumerate(parts):
                ref = rig.bank.b["s"]["core." + n][t]
                worst = max(worst, float((p - ref).norm() / (ref.norm() + 1e-30)))
                checked += 1
    finally:
        rig.close()
    assert checked >= 3 * DEPTH, f"only {checked} (layer, pass) pairs were cross-checked"
    assert worst < 1e-4, f"g^T x disagrees with the tap by {worst:.3e}"


def test_a_wrong_bookkeeping_fails_the_crosscheck():
    """The cross-check must FAIL on a wrong outer product and on a wrong magnitude.

    Run for real with `gm.t() @ xm` replaced by `xm.t() @ gm` (the transpose bug this
    guards against): all 24 (layer, pass) pairs on this model came back SHAPE MISMATCHED,
    because no covered core linear is square. That is asserted below, so the transposed
    form cannot silently agree. A wrong magnitude has no shape tell, so it is checked
    numerically with a 1 % perturbation.
    """
    probe = _probe_module()
    model = _model()
    rig = Rig(probe, model)
    try:
        rig.run()
        pairs, worst = 0, 0.0
        for n, _m in rig.xlayers:
            parts = rig.xstore[n]
            if any(p is None for p in parts):
                continue
            for t, p in enumerate(parts):
                ref = rig.bank.b["s"]["core." + n][t]
                assert p.t().shape != ref.shape, (
                    f"{n} is square: a transposed outer product would not be caught "
                    "by the shape")
                pairs += 1
                worst = max(worst, float((p * 1.01 - ref).norm() / (ref.norm() + 1e-30)))
    finally:
        rig.close()
    assert pairs >= 3 * DEPTH
    assert worst > 1e-3, "a 1 % error in the per-pass gradient passed the cross-check"


def test_a_mixed_up_pass_index_fails_the_crosscheck():
    """Swapping two passes' contributions must break the cross-check.

    The self-check (a sum) is blind to a pass permutation; the cross-check is not, and
    this is the test that says so.
    """
    probe = _probe_module()
    model = _model()
    rig = Rig(probe, model)
    try:
        rig.run()
        worst = 0.0
        for n, _m in rig.xlayers:
            parts = rig.xstore[n]
            if any(p is None for p in parts):
                continue
            ref = rig.bank.b["s"]["core." + n]
            worst = max(worst, float((parts[0] - ref[DEPTH - 1]).norm()
                                     / (ref[DEPTH - 1].norm() + 1e-30)))
    finally:
        rig.close()
    assert worst > 1e-2, "pass 1 and pass 3 are indistinguishable on this model"


def test_loop_cotangent_has_one_entry_per_pass():
    """`_loop_cot` carries exactly `depth` entries after one backward.

    The keys come from `_tul_core`'s OWN loop index, not from the probe's counter, so
    this pins the pass count independently of the `core[0]` pre-hook.
    """
    probe = _probe_module()
    model = _model()
    rig = Rig(probe, model)
    try:
        rig.run()
    finally:
        rig.close()
    assert sorted(model._loop_cot) == list(range(DEPTH)), model._loop_cot
    assert all(float(v) > 0.0 for v in model._loop_cot.values())


def test_mux_only_backward_reaches_every_pass():
    """The MUX term alone must bank a gradient at every pass, and still self-check.

    The probe's point is that two loss sources may reach different passes; if the MUX
    backward banked nothing the per-source table would silently be the CE's twice.
    """
    probe = _probe_module()
    model = _model()
    rig = Rig(probe, model, sources=("m",))
    try:
        rig.run(source="m", term="mux_local_live")
        stats = probe.pass_stats(rig.bank.b["m"], rig.names, DEPTH, rig.leaf_grads(), 1e-4)
    finally:
        rig.close()
    assert sorted(model._loop_cot) == list(range(DEPTH))
    assert stats["selfcheck"]["pass"], stats["selfcheck"]
    assert all(v > 0.0 for v in stats["per_pass_norm"]), stats["per_pass_norm"]


def test_carved_layer_is_refused():
    """A MORTAR-carved core MLP has no dense `weight` leaf; the probe must raise."""
    probe = _probe_module()
    model = _model()
    from morph.model.layers.block_sparse import CMSBlockLinear
    for mod in model.core.modules():
        if isinstance(mod, CMSBlockLinear):
            mod._dense_mode = False
            break
    with pytest.raises(RuntimeError, match="CARVED"):
        probe.weight_modules(model)
