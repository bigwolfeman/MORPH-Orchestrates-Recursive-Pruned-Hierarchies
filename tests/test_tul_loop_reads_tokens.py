"""`tul.loop_reads_tokens` — the SHIPPED core stage over tokens AND cells, span-restricted.

Inside `_tul_core` the compact sequence holds slot cells and nothing else, so a pass has
nothing new to look at. This mode runs `_core_region` — the same per-sample Poisson-depth
core the plain model earns 0.185 nats on — over EVERY position under

    allow(i, j) = causal AND ( bag_id[i] == bag_id[j] OR slot_mask[j] )

so a token reads its own span's tokens and reaches every earlier span ONLY through a slot
cell. The cells stay the whole cross-span channel at inference, which is what separates
this from the paid loop (`tul.tokens_through_core`), whose core is unrestricted.

WHAT THIS FILE HAS TO PROVE:

1. OFF IS NOTHING — `loop_reads_tokens: false` is the tree before the knob (the shared
   pin in `tests/test_tul_prefix_source.py` covers the numbers; here: no module, no key,
   the method never reached).
2. THE RELATION IS THE STRICT ONE — inside the core no token reads another span's token.
   Two-sided three ways, mirroring `tests/test_tul_core_token_aux.py`: leave the cells in
   and the edit crosses; widen the relation to plain causal and it crosses with the cells
   blanked; and over TWO passes it crosses THROUGH a cell, which is the design.
3. THE EVAL FORWARD IS THE TRAINING FORWARD at the same forced depth (dropout off), so
   the depth sweep and `worth_profile` score the model that trained.
4. THE DEPTH LEVER IS `model.cfg.mean_depth`, and `lab/divergence/_build.uses_sample_depth`
   — the ONE home the sweep reads — says so. A sweep that forces `slot_mean_depth` here
   would print a perfectly plausible curve of zeros.
5. EVERY KNOB THAT NEEDS A PER-SLOT DEPTH OR A TRAJECTORY RAISES, rather than sitting
   inert under an arm name that promises it.

CPU only, fp32, `use_kernels=False`, tiny config.

Record: lab/experiments/planned/2026-09-13-arc-loop-reads-tokens.md
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import morph.model.transformer as tfm
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import (BoundaryRule, TulLayoutSpec, slot_layout_from_ids,
                                    tg_reset_from_ids, tg_segment_ids, tg_strict_allow)

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=3, max_depth=4, bptt_depth=4,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)


def _spec() -> TulLayoutSpec:
    return TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)


def _ids(B: int = 2, n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::8] = DOT
    return ids.astype(np.int64)


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                mux_beta=0.0, spandec=True, spandec_layers=1, spandec_max_tokens=8,
                tg_restrict=True, tg_restrict_scope="all", tg_geometry="strict")
    base.update(kw)
    return TULConfig(**base)


def _batch(B: int = 2, n: int = 120, seed: int = 0):
    return slot_layout_from_ids(_ids(B, n, seed), _rule(), _spec())


def _model(seed: int = 99, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**tul_kw)))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.train().float()


# ── 1. OFF IS NOTHING ────────────────────────────────────────────────────────

def test_off_is_the_default_and_keeps_W_prefix():
    m = _model()
    assert m.cfg.tul.loop_reads_tokens is False
    assert m.tul.W_prefix is not None


def test_on_builds_no_module_and_drops_only_W_prefix():
    """The confound, declared: the arm holds 2.10 M FEWER parameters than its ruler.

    Nothing writes through a projection here — a cell's looped state is already at its own
    position — so `TULSlots(with_prefix=False)`, the paid loop's rule. Every OTHER tensor
    must still be there and byte-identical, or the arm differs by more than its mechanism.
    """
    a, b = _model(), _model(loop_reads_tokens=True)
    assert b.tul.W_prefix is None
    ka, kb = set(a.state_dict()), set(b.state_dict())
    assert ka - kb == {"tul.W_prefix"}, f"unexpected key change: {sorted(ka ^ kb)}"
    for k in sorted(kb):
        assert torch.equal(a.state_dict()[k], b.state_dict()[k]), (
            f"{k} differs — the mode drew RNG or moved a build order")


def test_off_never_reaches_the_new_branch(monkeypatch):
    """A Python-level branch that traces out, proved by making the path explode."""
    def boom(*a, **k):
        raise AssertionError("_core_token_aux_kwargs was reached on an OFF model")
    monkeypatch.setattr(MORPHTransformer, "_core_token_aux_kwargs", staticmethod(boom))
    inp, lab, layout, _ = _batch()
    torch.manual_seed(3)
    _model()(inp, labels=lab, slot_layout=layout)


# ── 2. THE RELATION IS THE STRICT ONE ────────────────────────────────────────

def _edit(ids: np.ndarray, layout, row: int, span: int) -> np.ndarray:
    bag = layout.bag_id[row].numpy()
    tok = ~layout.slot_mask[row].numpy()
    pos = np.flatnonzero((bag == span) & tok)
    assert pos.size >= 3
    p = int(pos[len(pos) // 2])
    out = ids.copy()
    raw = int(tok[:p].sum())
    assert out[row, raw] not in (DOT, 11)
    out[row, raw] = 12 if out[row, raw] != 12 else 13
    return out


def _core_states(m: MORPHTransformer, ids: np.ndarray, cut_cells: bool):
    """The SHIPPED core stage's output at every position, for one batch of ids.

    The prelude runs under the STRICT prelude relation, exactly as `_forward_tul` runs it.
    ``cut_cells`` zeroes the slot cells' carrier and their injection sources, so a cell
    carries the SAME constant in both runs and nothing about the edit.
    """
    inp, _lab, layout, _ = slot_layout_from_ids(ids, _rule(), _spec())
    pre = tg_strict_allow(layout, "prelude")
    with torch.no_grad():
        x, x0, bg = m._tul_front(
            inp, layout,
            attn_kwargs={"tg_allow": pre, "tg_slot_mask": layout.slot_mask,
                         "tg_comp_allow": pre, "tg_seg": tg_segment_ids(layout)},
            ret_reset_mask=tg_reset_from_ids(tg_segment_ids(layout)))
        if cut_cells:
            sm = layout.slot_mask
            x = x * (~sm).view(*sm.shape, *([1] * (x.dim() - 2))).to(x.dtype)
            x0 = x0 * (~sm).view(*sm.shape, *([1] * (x0.dim() - 2))).to(x0.dtype)
            if bg is not None:
                bg = bg * (~sm).view(*sm.shape, *([1] * (bg.dim() - 2))).to(bg.dtype)
        # The SHIPPED relation, not a rebuild of it. `_core_token_aux_kwargs` has ONE
        # home: the first version of the aux test built its own kwargs and a sabotage that
        # widened the real ones to plain causal was MISSED (2026-09-12).
        xc = m._core_region(x, x0, bg, inp,
                            attn_kwargs=m._core_token_aux_kwargs(layout))
    return xc, layout


def _leak(m, cut_cells: bool, row: int = 0, span: int = 0) -> float:
    ids = _ids()
    a, layout = _core_states(m, ids, cut_cells)
    b, _ = _core_states(m, _edit(ids, layout, row, span), cut_cells)
    bag, tok = layout.bag_id[row], ~layout.slot_mask[row]
    d = (a[row] - b[row]).abs().flatten(1)
    assert float(d[tok & (bag == span)].max()) > 0.0, "fixture: the edit moved nothing"
    return float(d[tok & (bag != span)].max())


def _leak_model():
    torch.manual_seed(99)
    m = MORPHTransformer(_tiny(n_core=1, mean_depth=1, max_depth=1,
                               tul=_tul(loop_reads_tokens=True)))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.eval().float()


def test_the_core_lets_no_token_read_another_spans_token():
    assert _leak(_leak_model(), cut_cells=True) == 0.0, (
        "one application of the core moved another span's token state with the slot cells "
        "blanked — a token is reading another span's token directly and the arm is the "
        "paid loop wearing a restricted arm's name")


def test_the_probe_sees_the_route_when_the_cells_are_left_in():
    assert _leak(_leak_model(), cut_cells=False) > 0.0


def test_a_plain_causal_core_leaks_through_the_same_probe(monkeypatch):
    def wide(layout, *a, **k):
        B, L = layout.bag_id.shape
        row = torch.arange(L).unsqueeze(1)
        col = torch.arange(L).unsqueeze(0)
        return (col <= row).view(1, 1, L, L).expand(B, 1, L, L).clone()

    monkeypatch.setattr(tfm, "tg_allow_mask", wide)
    assert _leak(_leak_model(), cut_cells=True) > 0.0, (
        "the probe cannot see a leak even at plain causal — it measures nothing")


def test_the_core_does_cross_a_boundary_through_a_cell_over_two_passes():
    """The DESIGN, as a test, so nobody reads the leak test as 'nothing crosses'."""
    torch.manual_seed(99)
    m = MORPHTransformer(_tiny(n_core=1, mean_depth=2, max_depth=2,
                               tul=_tul(loop_reads_tokens=True)))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    assert _leak(m.eval().float(), cut_cells=True) > 0.0


def test_the_shipped_forward_uses_that_exact_relation():
    """The leak probe builds the core call itself; this pins the FORWARD to the same one."""
    seen: dict = {}
    m = _model(loop_reads_tokens=True)
    real = m._core_region

    def spy(x, x0, bg, ids=None, attn_kwargs=None, jac_active=None):
        seen["kw"] = attn_kwargs
        return real(x, x0, bg, ids, attn_kwargs=attn_kwargs, jac_active=jac_active)

    m._core_region = spy
    inp, lab, layout, _ = _batch()
    torch.manual_seed(3)
    m(inp, labels=lab, slot_layout=layout)
    m._core_region = real
    want = MORPHTransformer._core_token_aux_kwargs(layout)
    assert set(seen["kw"]) == set(want)
    for k in want:
        assert torch.equal(seen["kw"][k], want[k]), f"{k} is not the shipped relation"


# ── 3. EVAL == TRAINING at the same forced depth ─────────────────────────────

def test_the_eval_forward_equals_the_training_forward_at_the_same_depth(monkeypatch):
    """Dropout is 0 on this fixture, so the only legal difference is the DEPTH DRAW.

    `mean_depth == max_depth` does NOT pin it — `_sample_depths` still draws a Poisson and
    clamps, so a training batch is a mix of 1s, 2s and 3s while eval fills a uniform
    `cfg.mean_depth`. Pinning the DRAW is what isolates the forward, and what is left must
    be bit-equal — otherwise the depth sweep scores a model the trainer did not train.
    """
    torch.manual_seed(99)
    m = MORPHTransformer(_tiny(mean_depth=3, max_depth=3,
                               tul=_tul(loop_reads_tokens=True))).float()
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    monkeypatch.setattr(MORPHTransformer, "_sample_depths",
                        lambda self, B, dev: torch.full((B,), 3, dtype=torch.long,
                                                        device=dev))
    inp, lab, layout, _ = _batch()
    m.train()
    torch.manual_seed(3)
    tr = m(inp, labels=lab, slot_layout=layout)
    m.eval()
    with torch.no_grad():
        ev = m(inp, labels=lab, slot_layout=layout)
        ab = m.tul_forward_ablated(inp, lab, layout)
    # `loss` carries the span decoder's weighted term on both paths; compare the model's
    # own CE, which is what `train.py` reports and what every sweep scores.
    for k in ("loss", "spandec"):
        assert torch.equal(tr[k].detach(), ev[k]), (k, tr[k].item(), ev[k].item())
    assert torch.equal(ev["loss"], ab["loss"]), (
        "tul_forward_ablated(normal) is not the shipped forward")


def test_the_span_decoder_still_grades_z_and_z_is_the_cells_own_state():
    """`z` must be the CORE's output AT the slot's first cell, position by position.

    Asserting only "spandec is finite and scored something" is guard theater: an
    independent review moved the gather to `slot_index + 1` and the whole suite stayed
    green (1480 passed). The gather is what makes `spandec_ce` comparable across the slot
    family, so it is checked against the core's own output, indexed directly — not through
    `gather_valid`, which is the function the sabotage would have lived in.
    """
    m = _model(loop_reads_tokens=True)
    inp, lab, layout, _ = _batch()
    seen = {}
    real_core, real_sd = m._core_region, m._tul_spandec_loss

    def core_spy(*a, **kw):
        out = real_core(*a, **kw)
        seen["x_coda"] = out
        return out

    def sd_spy(h_slots, *a, **kw):
        seen["z"] = h_slots
        return real_sd(h_slots, *a, **kw)

    m._core_region, m._tul_spandec_loss = core_spy, sd_spy
    try:
        torch.manual_seed(3)
        res = m(inp, labels=lab, slot_layout=layout)
    finally:
        m._core_region, m._tul_spandec_loss = real_core, real_sd
    assert "spandec" in res and torch.isfinite(res["spandec"])
    assert float(res["spandec_n_tokens"]) > 0
    xc, z = seen["x_coda"], seen["z"]
    assert z is not None and xc is not None, "the spies never fired"
    B, S = layout.slot_index.shape
    for b in range(B):
        for s in range(S):
            if not bool(layout.slot_valid[b, s]):
                continue
            p = int(layout.slot_index[b, s])
            assert bool(layout.slot_mask[b, p]), "slot_index does not point at a cell"
            assert torch.equal(z[b, s], xc[b, p]), (
                f"z[{b},{s}] is not the core's output at cell position {p}")


def test_the_slot_loop_levers_say_they_are_inert_on_this_mode(capsys):
    """base.yaml turns the slot-loop gain constraint on for EVERY model, and it acts only
    inside `_tul_core`. `loop_reads_tokens` runs `_core_region`, so the knobs do nothing —
    exactly as on the paid loop — and the build must say so once, loudly.

    `tul_slot_spandec_strict_tokloop.yaml` inherits `slot_gain_lambda: 100` and
    `slot_cot_clip: 4.0` from the slot-loop config root. Before 2026-09-13 the notice's
    predicate named only `tul is None`, `n_core == 0`, `tokens_through_core` and the FM
    planner, so this arm would have carried two dead knobs in silence. Found by an
    independent review of the knob's first commit.
    """
    levers = dict(slot_gain_lambda=100.0, slot_cot_clip=4.0)

    def build(**tul_kw):
        torch.manual_seed(99)
        MORPHTransformer(_tiny(tul=_tul(**tul_kw), **levers))
        return capsys.readouterr().out

    assert "[slot-levers]" in build(loop_reads_tokens=True), (
        "the tokloop arm carries slot-loop levers that do nothing and says nothing")
    # the twin: the paid loop already reported it, and the SLOT loop must NOT — there the
    # knobs are live and a notice would be a lie. (`spandec` is a slot-loop lever and the
    # paid loop refuses it, so the paid twin is built bare.)
    torch.manual_seed(99)
    MORPHTransformer(_tiny(tul=TULConfig(prefix_k=2, slot_id=4, tokens_through_core=True),
                           **levers))
    assert "[slot-levers]" in capsys.readouterr().out
    assert "[slot-levers]" not in build(), (
        "the slot loop was told its own live constraint is inert")


def test_the_fixed_point_term_is_still_charged_once():
    torch.manual_seed(99)
    m = MORPHTransformer(_tiny(core_fixed_point_lambda=1.0,
                               tul=_tul(loop_reads_tokens=True))).train().float()
    inp, lab, layout, _ = _batch()
    torch.manual_seed(3)
    res = m(inp, labels=lab, slot_layout=layout)
    assert "fixed_point" in res and torch.isfinite(res["fixed_point"])


# ── 4. THE DEPTH LEVER ───────────────────────────────────────────────────────

def test_uses_sample_depth_names_this_mode_and_the_paid_loop():
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "lab", "divergence"))
    from _build import uses_sample_depth
    assert uses_sample_depth(_tul(loop_reads_tokens=True)) is True
    assert uses_sample_depth(TULConfig(tokens_through_core=True)) is True
    assert uses_sample_depth(TULConfig()) is False
    assert uses_sample_depth(_tul()) is False


def test_mean_depth_is_the_lever_and_slot_mean_depth_is_not():
    m = _model(loop_reads_tokens=True).eval()
    inp, lab, layout, _ = _batch()
    seen = []
    for d in (1, 2, 4):
        m.cfg.mean_depth = d
        with torch.no_grad():
            seen.append(float(m.tul_forward_ablated(inp, lab, layout)["loss"]))
    assert len(set(seen)) == 3, f"model.cfg.mean_depth did nothing: {seen}"
    m.cfg.mean_depth = 3
    flat = []
    for d in (1, 2, 4):
        m.cfg.tul.slot_mean_depth = d
        with torch.no_grad():
            flat.append(float(m.tul_forward_ablated(inp, lab, layout)["loss"]))
    assert len(set(flat)) == 1, (
        f"slot_mean_depth moved the forward: {flat}. If that ever becomes true this "
        "mode grew a per-slot depth and `uses_sample_depth` has to be re-decided.")


def test_a_slot_depth_table_raises_rather_than_being_ignored():
    m = _model(loop_reads_tokens=True).eval()
    inp, lab, layout, _ = _batch()
    with pytest.raises(NotImplementedError, match="per-SAMPLE"):
        m.tul_forward_ablated(inp, lab, layout,
                              slot_depths=torch.full_like(layout.slot_index, 2))


def test_a_plan_ablation_raises_rather_than_reporting_a_zero_by_construction():
    m = _model(loop_reads_tokens=True).eval()
    inp, lab, layout, _ = _batch()
    for mode in ("zero", "shuffle", "all_slots"):
        with pytest.raises(ValueError, match="no separate plan tensor"):
            m.tul_forward_ablated(inp, lab, layout, plan_mode=mode)


# ── 5. REFUSALS ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw", [
    dict(tokens_through_core=True),
    dict(core_token_aux=True),
    dict(detach_z=True),
    dict(bcast=True),
    dict(coda_sees_slots=False),
    dict(mux_every_pass=True, mux_beta=1.0),
    dict(oracle_z=True),
    dict(spandec_per_pass=True),
    dict(grad_pass=True),
    dict(slot_chain=True),
    dict(reread=True),
    dict(progressive_p=0.5),
    dict(mux_stage_own_iters=3, mux_beta=1.0),
    dict(slot_depth_fixed=6),
    dict(slot_mean_depth=6),
    dict(loop_reach=1),
    dict(pass_lora_rank=8),
    dict(pass_residual_lambda=0.01),
    dict(db_loop=True),
])
def test_every_knob_that_needs_a_per_slot_loop_raises(kw):
    with pytest.raises(NotImplementedError):
        _tul(loop_reads_tokens=True, **kw)


def test_a_coreless_model_raises_at_build():
    with pytest.raises(ValueError, match="needs a core loop"):
        MORPHTransformer(_tiny(n_core=0, tul=_tul(loop_reads_tokens=True)))


def test_halt_raises():
    from morph.model.tul import TULGateConfig
    with pytest.raises(NotImplementedError, match="loop_reads_tokens"):
        _tul(loop_reads_tokens=True, gate=TULGateConfig())


def test_the_shipped_config_composes_and_carries_the_knob():
    from hydra import compose, initialize_config_dir
    import os
    from morph.training.tul_setup import build_tul_runtime
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.join(root, "morph", "configs")):
        cfg = compose(config_name="tul_slot_spandec_strict_tokloop", overrides=[])
        rt = build_tul_runtime(cfg)
    assert rt.model_cfg.loop_reads_tokens is True
    assert rt.model_cfg.tokens_through_core is False
    assert rt.model_cfg.tg_geometry == "strict"
    assert rt.model_cfg.prefix_k == 2 and rt.model_cfg.prefix_source == "exit"
