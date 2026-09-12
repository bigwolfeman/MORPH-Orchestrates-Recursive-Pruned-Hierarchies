"""`lab/divergence/slot_z_causal_fit.py` — the causal fitted-z diagnostic's contract.

The instrument exists to remove ONE thing from `slot_z_optimize.py`: hindsight. So the
tests that matter are the ones that prove the hindsight is gone and that nothing else
changed.

* **Samples that EQUAL the real continuation reproduce the hindsight fit.** The
  counterfactual row is then the real row byte for byte, and the fitted `z` matches to
  1e-5. This is what says the placement machinery (inputs, labels, the emitting cell) is
  an identity when the sample is the truth.
* **The real continuation never enters the fit.** With samples chosen to differ from the
  real tokens EVERYWHERE, neither the input ids the fit forward sees nor the label tensor
  `_tul_group_losses` receives contains a real next-span token at a graded position. The
  label half is asserted by spying on the loss itself, not by reading the code.
* **Only the graded spans move.** Everything outside them is the real row, byte for byte,
  and `--fit-groups` really does restrict both the replacement and the gradient.
* **The optimiser is `slot_z_optimize`'s.** One variant reproduces
  `slot_z_optimize.optimise_z` bit for bit, so "same parametrisation and optimiser" is a
  checked claim.
* **The per-position CE map is a view of the shipped number**, not a second definition:
  its masked mean equals `_tul_group_losses`' loss on the same positions.
* **The JSON schema and the CI fields**, from `summarise` on a synthetic accumulator.

CPU only, fp32, tiny config, no tokenizer and no checkpoint — the
`tests/test_tul_oracle_z.py` fixtures. The teacher's sampling and the end-to-end run need
a GPU and real checkpoints and are NOT exercised here; `sample_continuations` is covered
only by its slot-id refusal.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids
from morph.model.tul_spandec import next_span_slots

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "lab", "divergence"))
from _next_span import next_span_positions  # noqa: E402
from slot_z_causal_fit import (  # noqa: E402
    ARMS,
    build_variant,
    ce_positions,
    masked,
    optimise_z_multi,
    quiet_aux,
    row_stream,
    scored_mask,
    snapshot,
    summarise,
    token_ordinals,
    use_snapshot,
)
from slot_z_optimize import ZSplit, optimise_z  # noqa: E402

V = 64
DOT = 10
SLOT_ID = 4


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


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=SLOT_ID, emit_weight=0.0, plast_weight=1.0,
                token_state_dropout=0.0, mux_beta=0.0, slot_depth_fixed=3,
                slot_max_depth=4)
    base.update(kw)
    return TULConfig(**base)


def _batch(B: int = 2, n: int = 150, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == SLOT_ID] = 5
    ids[:, ::8] = DOT
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=SLOT_ID)
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _model(seed: int = 99, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**tul_kw)))
    m = m.eval().float()
    m.requires_grad_(False)
    return m


def _geom(layout, inp, labels):
    pos, valid = next_span_positions(layout, labels, inp)
    J = pos.shape[2]
    tok_pos = (layout.slot_index + layout.prefix_k).unsqueeze(-1) + torch.arange(J).view(
        1, 1, J)
    tok_pos = torch.where(valid, tok_pos, torch.zeros_like(tok_pos))
    return pos, valid, tok_pos


def _real_samples(inp, layout, J):
    """The REAL next-span tokens, shaped like a sample draw."""
    ids, _valid = next_span_slots(inp, layout, J)
    return ids


def _fake_samples(inp, layout, J):
    """A draw that differs from the real continuation at EVERY position."""
    real = _real_samples(inp, layout, J)
    fake = (real + 7) % V
    fake = torch.where(fake == SLOT_ID, torch.full_like(fake, SLOT_ID + 1), fake)
    fake = torch.where(fake == real, (real + 13) % V, fake)
    assert not bool((fake == real).any())
    assert not bool((fake == SLOT_ID).any())
    return fake


# ── placement is an identity when the sample IS the truth ────────────────────

def test_real_samples_rebuild_the_real_row():
    inp, lab, layout, _ = _batch()
    pos, valid, tok_pos = _geom(layout, inp, lab)
    s = _real_samples(inp, layout, pos.shape[2])
    cf_ids, cf_lab = build_variant(inp, lab, layout, tok_pos, pos, valid, s,
                                   layout.slot_valid)
    assert torch.equal(cf_ids, inp)
    assert torch.equal(cf_lab, lab)


def test_the_causal_fit_reproduces_the_hindsight_fit_when_the_samples_are_the_truth():
    m = _model()
    inp, lab, layout, _ = _batch()
    pos, valid, tok_pos = _geom(layout, inp, lab)
    mask = scored_mask(pos, valid, layout.slot_valid, inp.shape[1])
    real_lab = masked(lab, mask)
    cf_ids, cf_lab = build_variant(inp, lab, layout, tok_pos, pos, valid,
                                   _real_samples(inp, layout, pos.shape[2]),
                                   layout.slot_valid)
    cf_lab = masked(cf_lab, mask)
    with ZSplit(m, amp=False) as split, quiet_aux(m):
        _out, z_loop, _h0 = split.record(inp, real_lab, layout, want_groups=False)
        snap_real = snapshot(split)
        split.record(cf_ids, cf_lab, layout, want_groups=False)
        snap_cf = snapshot(split)
        z_causal, _ = optimise_z_multi(split, [(snap_cf, cf_ids, cf_lab)], layout,
                                       z_loop, 1e-2, 12, layout.slot_valid)
        z_hind, _ = optimise_z_multi(split, [(snap_real, inp, real_lab)], layout,
                                     z_loop, 1e-2, 12, layout.slot_valid)
    assert torch.allclose(z_causal, z_hind, atol=1e-5), \
        "samples equal to the real continuation did not reproduce the hindsight fit"
    assert not torch.allclose(z_causal, z_loop, atol=1e-5), \
        "the fit moved nothing — the fixture proves nothing"


# ── the real continuation never enters the fit ───────────────────────────────

def test_the_fit_never_sees_a_real_next_span_token():
    m = _model()
    inp, lab, layout, _ = _batch()
    pos, valid, tok_pos = _geom(layout, inp, lab)
    mask = scored_mask(pos, valid, layout.slot_valid, inp.shape[1])
    fake = _fake_samples(inp, layout, pos.shape[2])
    cf_ids, cf_lab = build_variant(inp, lab, layout, tok_pos, pos, valid, fake,
                                   layout.slot_valid)
    cf_lab = masked(cf_lab, mask)

    # the INPUT side: every graded span's token positions carry the sample
    got = cf_ids.gather(1, tok_pos.reshape(inp.shape[0], -1)).reshape(tok_pos.shape)
    assert torch.equal(got[valid], fake[valid])
    real_ids = inp.gather(1, tok_pos.reshape(inp.shape[0], -1)).reshape(tok_pos.shape)
    assert int((got[valid] == real_ids[valid]).sum()) == 0

    # the LABEL side, taken from the loss itself rather than from the tensor we built
    seen: list[torch.Tensor] = []
    with ZSplit(m, amp=False) as split, quiet_aux(m):
        _o, z_loop, _h = split.record(inp, masked(lab, mask), layout, want_groups=False)
        split.record(cf_ids, cf_lab, layout, want_groups=False)
        snap_cf = snapshot(split)
        inner = m._tul_group_losses          # ZSplit's own wrapper
        m._tul_group_losses = lambda x, labels, layout_, want_groups=True: (
            seen.append(labels.detach().clone()) or inner(x, labels, layout_,
                                                          want_groups=want_groups))
        optimise_z_multi(split, [(snap_cf, cf_ids, cf_lab)], layout, z_loop, 1e-2, 5,
                         layout.slot_valid)
        m._tul_group_losses = inner
    assert len(seen) >= 5, "the spy never fired — the fit did not go through the loss"
    for labels_seen in seen:
        at = labels_seen[mask]
        assert int((at == lab[mask]).sum()) == 0, \
            "a REAL next-span token reached the fit objective"
        assert torch.equal(at, cf_lab[mask])


# ── only the graded spans move ───────────────────────────────────────────────

def test_build_variant_leaves_everything_else_byte_identical():
    inp, lab, layout, _ = _batch()
    pos, valid, tok_pos = _geom(layout, inp, lab)
    fake = _fake_samples(inp, layout, pos.shape[2])
    cf_ids, cf_lab = build_variant(inp, lab, layout, tok_pos, pos, valid, fake,
                                   layout.slot_valid)
    B, L = inp.shape
    touched_ids = torch.zeros((B, L), dtype=torch.bool)
    touched_ids.scatter_(1, torch.where(valid, tok_pos, torch.zeros_like(tok_pos))
                         .reshape(B, -1), valid.reshape(B, -1))
    assert torch.equal(cf_ids[~touched_ids], inp[~touched_ids])
    emit = (layout.slot_index + layout.prefix_k - 1).unsqueeze(-1)
    touched_lab = torch.zeros((B, L), dtype=torch.bool)
    for idx in (pos, emit.expand_as(pos[:, :, :1])):
        v = valid[:, :, :idx.shape[2]]
        touched_lab.scatter_(1, torch.where(v, idx, torch.zeros_like(idx)).reshape(B, -1),
                             v.reshape(B, -1))
    assert torch.equal(cf_lab[~touched_lab], lab[~touched_lab])


def test_fit_groups_restrict_the_replacement():
    inp, lab, layout, _ = _batch()
    pos, valid, tok_pos = _geom(layout, inp, lab)
    fake = _fake_samples(inp, layout, pos.shape[2])
    S = pos.shape[1]
    g0 = (torch.arange(S) % 2 == 0).unsqueeze(0).expand_as(layout.slot_valid) \
        & layout.slot_valid
    cf0, _ = build_variant(inp, lab, layout, tok_pos, pos, valid, fake, g0)
    cf_all, _ = build_variant(inp, lab, layout, tok_pos, pos, valid, fake,
                              layout.slot_valid)
    assert not torch.equal(cf0, inp)
    assert not torch.equal(cf0, cf_all), "the group mask changed nothing"
    # every position the group does NOT own is the real row
    own = torch.zeros(inp.shape, dtype=torch.bool)
    k = valid & g0.unsqueeze(-1)
    own.scatter_(1, torch.where(k, tok_pos, torch.zeros_like(tok_pos))
                 .reshape(inp.shape[0], -1), k.reshape(inp.shape[0], -1))
    assert torch.equal(cf0[~own], inp[~own])


def test_scored_masks_of_the_groups_partition_the_scored_set():
    inp, lab, layout, _ = _batch()
    pos, valid, _tp = _geom(layout, inp, lab)
    L, S = inp.shape[1], pos.shape[1]
    allm = scored_mask(pos, valid, layout.slot_valid, L)
    parts = [scored_mask(pos, valid,
                         (torch.arange(S) % 3 == g).unsqueeze(0).expand_as(
                             layout.slot_valid) & layout.slot_valid, L)
             for g in range(3)]
    assert torch.equal(parts[0] | parts[1] | parts[2], allm)
    assert int((parts[0] & parts[1]).sum()) == 0
    assert int(allm.sum()) == int(valid.sum())


def test_a_group_fit_moves_only_its_own_slots():
    m = _model()
    inp, lab, layout, _ = _batch()
    pos, valid, tok_pos = _geom(layout, inp, lab)
    S, L = pos.shape[1], inp.shape[1]
    g0 = (torch.arange(S) % 2 == 0).unsqueeze(0).expand_as(layout.slot_valid) \
        & layout.slot_valid
    gmask = scored_mask(pos, valid, g0, L)
    fake = _fake_samples(inp, layout, pos.shape[2])
    cf_ids, cf_lab = build_variant(inp, lab, layout, tok_pos, pos, valid, fake, g0)
    cf_lab = masked(cf_lab, gmask)
    with ZSplit(m, amp=False) as split, quiet_aux(m):
        _o, z_loop, _h = split.record(inp, masked(lab, gmask), layout, want_groups=False)
        split.record(cf_ids, cf_lab, layout, want_groups=False)
        zg, _ = optimise_z_multi(split, [(snapshot(split), cf_ids, cf_lab)], layout,
                                 z_loop, 1e-2, 8, g0)
    moved = (zg.float() - z_loop.float()).flatten(2).norm(dim=2) > 0
    assert bool((moved & g0).any()), "the group's own slots did not move"
    assert not bool((moved & ~g0).any()), "a slot outside the group moved"


# ── the optimiser is slot_z_optimize's ───────────────────────────────────────

def test_one_variant_reproduces_slot_z_optimize():
    m = _model()
    inp, lab, layout, _ = _batch()
    with ZSplit(m, amp=False) as split, quiet_aux(m):
        _o, z_loop, _h = split.record(inp, lab, layout, want_groups=False)
        snap = snapshot(split)
        a, _ = optimise_z(split, inp, lab, layout, z_loop, 1e-2, 9, layout.slot_valid)
        use_snapshot(split, snap)
        b, _ = optimise_z_multi(split, [(snap, inp, lab)], layout, z_loop, 1e-2, 9,
                                layout.slot_valid)
    assert torch.equal(a, b), "optimise_z_multi is not slot_z_optimize's optimiser"


# ── the CE map is a view of the shipped number ───────────────────────────────

def test_ce_map_reproduces_the_shipped_masked_loss():
    m = _model()
    inp, lab, layout, _ = _batch()
    pos, valid, _tp = _geom(layout, inp, lab)
    mask = scored_mask(pos, valid, layout.slot_valid, inp.shape[1])
    real_lab = masked(lab, mask)
    with ZSplit(m, amp=False) as split, quiet_aux(m):
        _o, z_loop, _h = split.record(inp, real_lab, layout, want_groups=False)
        split.replay(inp, real_lab, layout, z_loop, want_groups=False)
        shipped = float(split.token_ce)
        mp = ce_positions(m, split.xh, lab)
    got = float((mp * mask).sum() / mask.sum())
    assert abs(got - shipped) < 1e-4, f"map {got:.6f} vs shipped {shipped:.6f}"


# ── small maps ───────────────────────────────────────────────────────────────

def test_row_stream_and_ordinals_agree_with_the_packed_row():
    inp, _lab, layout, _ = _batch()
    ords = token_ordinals(layout)
    for b in range(inp.shape[0]):
        st = row_stream(inp, layout, b)
        assert len(st) == int((~layout.slot_mask[b]).sum())
        for p in range(inp.shape[1]):
            if bool(layout.slot_mask[b, p]):
                assert int(ords[b, p]) == -1
            else:
                assert st[int(ords[b, p])] == int(inp[b, p])


def test_the_teacher_may_not_sample_the_structural_slot_id():
    """A structural id at a token position is OOD for the student. Loud, not substituted."""
    import morph.inference.plain_generate as pg
    from slot_z_causal_fit import sample_continuations

    orig = pg.generate_plain_batch
    pg.generate_plain_batch = lambda _m, prompts, **k: [[SLOT_ID, 6] for _ in prompts]
    try:
        with pytest.raises(SystemExit, match="structural slot id"):
            sample_continuations(object(), [5, 6, 7, 8], [2], [2], 1, 1.0, 0, 8, SLOT_ID,
                                 "cpu")
    finally:
        pg.generate_plain_batch = orig


# ── the JSON schema ──────────────────────────────────────────────────────────

def test_summarise_schema_and_ci_fields():
    rng = np.random.default_rng(0)
    n_rows, n_slots, n_off = 6, 4, 9
    cnt = rng.integers(20, 40, size=n_rows).astype(float)
    row_sum = {k: (cnt * rng.uniform(2.0, 4.0, size=n_rows)).tolist() for k in ARMS}
    off_cnt = [rng.integers(1, 5, size=(n_rows, n_off)).astype(float)]
    off_sum = {k: [off_cnt[0] * rng.uniform(2.0, 4.0, size=(n_rows, n_off))] for k in ARMS}
    slot_cnt = rng.integers(1, 9, size=n_slots).astype(float)
    slot_sum = {k: slot_cnt * rng.uniform(2.0, 4.0, size=n_slots) for k in ARMS}
    res = summarise(row_sum, cnt.tolist(), off_sum, off_cnt, slot_sum, slot_cnt,
                    {"sum": 300.0, "n": 100.0}, {"sum": 250.0, "n": 100.0}, n_boot=200)
    assert set(res) == {"n_scored", "teacher_ce", "n_teacher_scored",
                        "ce_heldout_causal", "n_heldout", "arms"}
    assert res["teacher_ce"] == pytest.approx(2.5)
    assert res["ce_heldout_causal"] == pytest.approx(3.0)
    assert set(res["arms"]) == set(ARMS)
    for name in ARMS:
        arm = res["arms"][name]
        assert set(arm) == {"ce", "vs_loop", "by_offset", "by_slot"}
        assert set(arm["vs_loop"]) == {"point", "lo", "hi", "n_units", "n_boot", "level"}
        assert arm["vs_loop"]["n_units"] == n_rows and arm["vs_loop"]["n_boot"] == 200
        assert arm["vs_loop"]["lo"] <= arm["vs_loop"]["hi"]
        assert len(arm["by_slot"]) == n_slots
        assert len(arm["by_offset"]) == n_off
        assert arm["ce"] == pytest.approx(
            float(np.sum(row_sum[name]) / cnt.sum()))
    lo = res["arms"]["loop"]["vs_loop"]
    assert lo["point"] == 0.0 and lo["lo"] == 0.0 and lo["hi"] == 0.0
