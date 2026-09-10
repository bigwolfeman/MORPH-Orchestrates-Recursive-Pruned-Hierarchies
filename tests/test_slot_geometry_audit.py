"""The instrument contract of `lab/divergence/slot_geometry_audit.py`.

The audit's numbers are only worth reading if two structural claims hold:

(a) **The hooks see the whole core.** Every core block's attention branch, MLP branch and
    both HC residual writes must be recorded at EVERY pass, and the pass counter must run
    1..depth. A probe that silently misses a block would report a muted core because it
    only looked at half of one.
(b) **The captured pass-t state is the object the coda reads.** The state captured after
    the LAST pass must be the tensor `prefix_project` receives, and substituting it must
    reproduce the trained forward's token CE bit for bit. The deliberate wrong split —
    the ENTRY state (K0) where the exit belongs — must NOT reproduce it, or (b) proves
    nothing.

Plus the two patches that could silently do nothing: the injection cut must actually
change the exit state, and `_apply_injection` must be restored on exit.

CPU, tiny config, no tokenizer (the `test_tul_forward.py` recipe, as
`test_slot_z_optimize.py` uses it).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10
DEPTH = 2
N_CORE = 2

_LAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "lab", "divergence")


def _mod(name: str):
    """Import a probe without running its argparse main (`lab/` is not a package)."""
    if _LAB not in sys.path:
        sys.path.insert(0, _LAB)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_LAB, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _tiny(tul: TULConfig) -> MORPHConfig:
    return MORPHConfig(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=N_CORE, n_coda=1, mean_depth=DEPTH, max_depth=3,
        bptt_depth=DEPTH, channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
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


def _record(aud, zopt, model, x, y, layout, cut=None):
    """One instrumented record. Returns (hooks, acc, geom, z_loop, h0, ce)."""
    acc, geom = aud.Acc(), {}
    with aud.Split(model, amp=False) as split, \
            aud.GeometryHooks(model, N_CORE, acc, geom) as gh:
        gh.ctx.update(sec="slot", rows=layout.slot_valid,
                      S=int(layout.slot_index.shape[1]), geom=True, inj_cut_from=cut)
        gh.ctx["pass"] = 0
        out = split.record(x, y, layout, want_groups=False)
        gh.ctx.update(sec=None, geom=False)
        ce = split.token_ce.clone()
        _o, z_loop, h0 = out
        states = [t.clone() for t in gh.states]
    return acc, geom, states, z_loop, h0, ce


def test_hooks_cover_every_core_block_at_every_pass():
    """(a) coverage. Four readings per (block, pass), and the pass counter runs 1..DEPTH."""
    aud, zopt = _mod("slot_geometry_audit"), _mod("slot_z_optimize")
    model, x, y, layout = _fixture()
    acc, geom, states, _z, _h0, _ce = _record(aud, zopt, model, x, y, layout)
    res = acc.out()
    missing = [k for p in range(1, DEPTH + 1) for i in range(N_CORE)
               for k in (f"slot/branch_out_attn/b{i}/p{p}", f"slot/branch_out_mlp/b{i}/p{p}",
                         f"slot/d_attn/b{i}/p{p}", f"slot/d_mlp/b{i}/p{p}",
                         f"slot/attn_comp/b{i}/p{p}", f"slot/attn_win/b{i}/p{p}")
               if k not in res]
    assert not missing, f"the hooks missed {len(missing)} readings: {missing[:6]}"
    assert len(states) == DEPTH, f"captured {len(states)} pass states, expected {DEPTH}"
    # a reading of exactly zero everywhere would pass the key check and mean nothing
    for i in range(N_CORE):
        assert res[f"slot/branch_out_attn/b{i}/p1"] > 0.0
        assert res[f"slot/branch_out_mlp/b{i}/p1"] > 0.0
    # the geometry table exists for every block and names the compressed branch's shape
    for i in range(N_CORE):
        g = geom[f"slot/b{i}"]
        assert "win_entropy" in g and "comp_out_norm" in g and "win_keys_row0" in g
    # XSA excludes the self token, so query 0 of the window branch has NO key at all
    assert geom["slot/b0"]["win_keys_row0"] == 0


def test_captured_exit_state_is_the_tensor_the_coda_reads():
    """(b) the split. K_depth reproduces the forward bit for bit; K0 does not."""
    aud, zopt = _mod("slot_geometry_audit"), _mod("slot_z_optimize")
    model, x, y, layout = _fixture()
    acc, geom = aud.Acc(), {}
    with aud.Split(model, amp=False) as split, \
            aud.GeometryHooks(model, N_CORE, acc, geom) as gh:
        gh.ctx.update(sec="slot", rows=layout.slot_valid,
                      S=int(layout.slot_index.shape[1]), geom=False, inj_cut_from=None)
        gh.ctx["pass"] = 0
        _out, z_loop, h0 = split.record(x, y, layout, want_groups=False)
        gh.ctx.update(sec=None)
        ce_loop = split.token_ce.clone()
        valid = layout.slot_valid

        # the captured last-pass state IS the returned h_slots on every valid slot
        assert torch.equal(gh.states[-1][valid], z_loop[valid]), (
            "the captured exit state differs from the tensor prefix_project receives")

        def sub(z):
            zz = torch.where(valid.view(*valid.shape, *([1] * (z.dim() - 2))), z, z_loop)
            split.replay(x, y, layout, zz, want_groups=False)
            return split.token_ce.clone()

        assert torch.equal(sub(gh.states[-1]), ce_loop), (
            "substituting the captured exit state does not reproduce the forward's CE")
        # the deliberate WRONG split: the entry state where the exit belongs
        assert not torch.equal(h0[valid], z_loop[valid]), "the loop did not move the state"
        assert not torch.equal(sub(h0), ce_loop), (
            "K0 reproduced the forward's CE, so this test cannot tell a right split "
            "point from a wrong one")


def test_the_injection_cut_changes_the_exit_state_and_is_reverted():
    """The counterfactual patch is real, and both patches are restored on exit."""
    aud, zopt = _mod("slot_geometry_audit"), _mod("slot_z_optimize")
    model, x, y, layout = _fixture()
    before = type(model)._apply_injection
    _a, _g, st_on, _z, _h, ce_on = _record(aud, zopt, model, x, y, layout, cut=None)
    _a, _g, st_off, _z2, _h2, ce_off = _record(aud, zopt, model, x, y, layout, cut=2)
    valid = layout.slot_valid
    assert not torch.equal(st_on[-1][valid], st_off[-1][valid]), (
        "cutting the injection after pass 1 left the exit state unchanged — the patch "
        "did nothing")
    # the patch must be an instance shadow that is removed, not a class edit
    assert "_apply_injection" not in model.__dict__
    assert type(model)._apply_injection is before
    # and the prelude/coda are untouched by the cut: with DEPTH passes the first pass is
    # identical either way
    assert torch.equal(st_on[0][valid], st_off[0][valid]), (
        "the cut changed pass 1, which it must not (it starts at pass 2)")


def test_readout_stats_report_the_loop_share_of_the_coda_cell():
    """`|W_k·(exit − entry)| / |W_k·entry|` is present for every prefix offset."""
    aud, zopt = _mod("slot_geometry_audit"), _mod("slot_z_optimize")
    model, x, y, layout = _fixture()
    _acc, _geom, _st, z_loop, h0, _ce = _record(aud, zopt, model, x, y, layout)
    acc = aud.Acc()
    aud.readout_stats(model, h0, z_loop, layout.slot_valid, acc, {})
    res = acc.out()
    for k in range(int(model.cfg.tul.prefix_k)):
        assert f"readout/Wp{k}_delta_share" in res
        assert res[f"readout/Wp{k}_delta_share"] >= 0.0
    assert 0.0 <= res["readout/stream_mean_ratio_delta"] <= 1.0 + 1e-6
    assert res["readout/delta_norm"] > 0.0


def test_window_weights_reproduce_the_shipped_window_branch():
    """The geometry reader's mask IS `_window_fallback`'s.

    Entropy, top-1 mass and "what does cell 0 attend" are all read off weights the
    shipped path never materialises. If the rebuilt mask drifted from
    `attention._window_fallback`'s, every one of those numbers would describe a
    different function. So: rebuild the weights, apply them to v, and require the
    shipped branch's own output back. A row with no visible key (query 0 under XSA) is
    compared separately — SDPA returns 0 there and the explicit softmax is undefined.

    fp32 on purpose. `morph/model/CLAUDE.md` records that `_window_fallback` hands SDPA an
    fp32 mask whatever the dtype of q, which makes an fp64 comparison silently wrong (a
    measured error of 3.03) and says in place: do not write an fp64 test against this
    function. fp32 is one of the dtypes MORPH actually runs.
    """
    from morph.model.attention import _window_fallback
    aud = _mod("slot_geometry_audit")
    model, _x, _y, _layout = _fixture()
    cca = model.core[0].attention._impl.cca
    B, H, S, D = 2, cca.n_heads, 24, cca.d_head
    g = torch.Generator().manual_seed(7)
    q = torch.randn(B, H, S, D, generator=g, dtype=torch.float32)
    k = torch.randn(B, H, S, D, generator=g, dtype=torch.float32)
    v = torch.randn(B, H, S, D, generator=g, dtype=torch.float32)
    scale = D ** -0.5
    # window_size 8 < S, so the window rule actually BINDS (a window wider than S would
    # make the test pass on plain causal attention and prove nothing).
    w, n_keys = aud.window_weights(q, k, 8, scale)
    # XSA drops the self token, so a window of 8 admits at most 7 keys, and the rule
    # BINDS: without it a query deep in the sequence would see all 23 predecessors.
    assert int(n_keys.min()) == 0 and int(n_keys.max()) == 7, (
        f"the window rule did not bind: key counts {int(n_keys.min())}..{int(n_keys.max())}")
    got = torch.einsum("bhij,bhjd->bhid", w.to(v.dtype), v)
    ref = _window_fallback(q, k, v, 8, q.device, scale)
    live = (n_keys > 0).unsqueeze(-1).expand_as(got)
    err = (got - ref)[live].abs().max()
    assert err < 2e-6, f"the rebuilt window weights do not reproduce the branch: {err}"
    assert ref[:, :, 0].abs().max() == 0.0, "SDPA did not return 0 for the keyless row"
