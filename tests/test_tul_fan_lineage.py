"""``tul.fan_lineage`` — LXTUL-P rung P4 (partial): K streams as K LINEAGES.

WHAT THIS FILE HAS TO PROVE:

1. THE MASK IS THE DOCUMENTED RELATION, BY VALUE. Across slots a cell reads ONLY its own
   stream index; within a slot the register's all-to-all relation is untouched; slot 0 is
   unchanged; and the result is a strict, non-empty narrowing of the relation above it.
2. OFF IS NOTHING. ``"off"`` is the default and is bit-identical on loss, logits and the
   total gradient.
3. THE NARROWING EXECUTES. Two-sided, the way ``tests/test_tul_slot_register.py`` had to
   prove the register's own relation: the arm's loss DIFFERS from the off arm's, and it
   EQUALS the loss of an off arm whose ``slot_cell_relation`` is monkeypatched to an
   INDEPENDENTLY written lineage mask. A mask that is built and then delivered through a
   kwarg that can only narrow an already-causal relation is a no-op, and that exact bug
   shipped here once (2026-09-13) with every mask test green.
4. REFUSALS, by name: off a fan arm, and on any value other than the two — the message
   for a third value is where the UNBUILT reweighting is explained.
5. The key composes through ``tul_setup`` and the shipped config resolves.

WHAT IS NOT TESTED BECAUSE IT IS NOT BUILT: the filtering posterior (reweighting the
lineages by each stream's span CE after the span is observed). It cannot be causal in
this forward — `_tul_core` advances every slot's pass t together and the per-stream span
CE comes from coda replays AFTER the loop returns. See ``TULConfig.fan_lineage``.

CPU only, fp32, the ``tests/test_tul_fan.py`` fixtures.
Prereg: lab/experiments/planned/2026-09-21-lxtul-fan4-all-lineage.md
"""
from __future__ import annotations

import contextlib

import pytest
import torch

from test_tul_fan import _batch, _model, _tul, _finite_logit_sum
from morph.model.transformer import MORPHTransformer, slot_cell_relation


K = 4


def _fan(**kw) -> MORPHTransformer:
    base = dict(fan_k=K, slot_cells=K, prefix_k=K, fan_mix="all")
    base.update(kw)
    return _model(**base)


def _perturb(m: MORPHTransformer, seed: int = 99) -> None:
    """Move the register's trigger off its zero-init, so the K cells of a slot actually
    differ and a relation that mixes them can be told from one that does not."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.tul_register.W_o.weight.copy_(
            torch.empty(m.tul_register.W_o.weight.shape).normal_(0.0, 0.05, generator=g))
        m.tul_register.P_cell.copy_(
            torch.empty(m.tul_register.P_cell.shape).normal_(0.0, 0.05, generator=g))


def _loss_logits_grad(m: MORPHTransformer, seed: int = 7):
    _ids, inp, lab, layout = _batch(K)
    torch.manual_seed(seed)
    res = m.train()(inp, labels=lab, slot_layout=layout)
    res["loss"].backward()
    g = sum(float(p.grad.double().abs().sum()) for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        lg = m(inp, labels=None, slot_layout=layout)["logits"]
    return float(res["loss"].detach()), _finite_logit_sum(lg), g


# ── 1. the mask, by value ────────────────────────────────────────────────────

def test_the_lineage_mask_is_own_stream_across_slots_and_own_slot_in_full():
    S, M = 4, K
    plain, same = slot_cell_relation(S, M, "cpu")
    lin, same_l = slot_cell_relation(S, M, "cpu", lineage=True)
    assert torch.equal(same, same_l), "the slot-local relation is not the lineage's business"
    p, q = plain[0, 0], lin[0, 0]
    for i in range(S * M):
        for j in range(S * M):
            si, ci = divmod(i, M)
            sj, cj = divmod(j, M)
            want = (si == sj) or (si > sj and ci == cj)
            assert bool(q[i, j]) is bool(want), (
                f"cell {i} (slot {si}, stream {ci}) -> cell {j} (slot {sj}, stream {cj}): "
                f"got {bool(q[i, j])}, want {want}")
    # the two named halves, said again as the statements a reader checks
    assert bool(q[1 * M + 2, 0 * M + 2]), "stream 2 of slot 1 must read stream 2 of slot 0"
    assert not bool(q[1 * M + 2, 0 * M + 3]), "and must NOT read stream 3 of slot 0"
    assert bool(q[1 * M + 2, 1 * M + 3]), "within its own slot it still reads every cell"
    # slot 0 is untouched: it has no earlier slot to narrow
    assert torch.equal(p[:M, :M], q[:M, :M])
    # a strict, non-empty narrowing
    assert bool((q <= p).all()) and bool((p & ~q).any())


# ── 2. off is nothing ────────────────────────────────────────────────────────

def test_default_is_off_and_the_absent_key_is_bit_identical():
    a = _fan()
    assert a.cfg.tul.fan_lineage == "off"
    assert a._fan_lineage is False
    b = _fan(fan_lineage="off")
    assert _loss_logits_grad(a) == _loss_logits_grad(b)


def test_the_builder_default_is_the_relation_from_before_the_key():
    """`lineage=False` must return the exact tensor the one-argument call returned, or
    every model without the key has silently changed relation."""
    a, sa = slot_cell_relation(5, 3, "cpu", 0)
    b, sb = slot_cell_relation(5, 3, "cpu", 0, lineage=False)
    assert torch.equal(a, b) and torch.equal(sa, sb)


# ── 3. the narrowing EXECUTES — two-sided ────────────────────────────────────

@contextlib.contextmanager
def _force_lineage_mask():
    """Hand the core an INDEPENDENTLY written lineage mask, built here from the flattened
    index and nothing else, in place of `slot_cell_relation`'s `blk`."""
    import morph.model.transformer as _T
    real = _T.slot_cell_relation

    def patched(n_slots, m_cells, device, r=0, lineage=False, history_streams=0):
        assert lineage is False, "the control arm must not ask for the real narrowing"
        assert history_streams == 0, (
            "this file never composes tul.fan_history_streams; a nonzero value here "
            "would mean the call site changed without this stub following it")
        _blk, same = real(n_slots, m_cells, device, r)
        sm = n_slots * m_cells
        idx = torch.arange(sm, device=device)
        sl, cl = idx // m_cells, idx % m_cells
        blk = ((sl.unsqueeze(1) == sl.unsqueeze(0))
               | ((sl.unsqueeze(1) > sl.unsqueeze(0))
                  & (cl.unsqueeze(1) == cl.unsqueeze(0)))).view(1, 1, sm, sm)
        return blk, same

    _T.slot_cell_relation = patched
    try:
        yield
    finally:
        _T.slot_cell_relation = real


def _loss(lineage: str, forced: bool = False) -> float:
    _ids, inp, lab, layout = _batch(K)
    m = _fan(fan_lineage=lineage)
    _perturb(m)
    torch.manual_seed(5)
    return float(m.train()(inp, labels=lab, slot_layout=layout)["loss"].detach())


def test_the_arm_differs_from_the_off_arm_and_equals_the_independent_mask():
    off = _loss("off")
    on = _loss("relation")
    assert on != off, (
        "the lineage relation gives the SAME loss as the register's — the narrowing is "
        "not executing (the 2026-09-13 tg_allow bug, one level down)")
    with _force_lineage_mask():
        forced = _loss("off")
    assert forced == on, (
        "the executed relation is not the documented one: an independently written "
        "own-stream-across-slots mask gives a different loss")


def test_a_pad_cell_is_never_a_key_of_a_valid_cell_under_the_lineage_relation():
    """The relation never lets a valid cell read PAST its own slot, and pads sit at the
    tail of the compact sequence — asserted, because the narrowing touches exactly the
    cross-slot half that rule lives in."""
    S, M = 6, K
    lin = slot_cell_relation(S, M, "cpu", lineage=True)[0][0, 0]
    for i in range(S * M):
        for j in range(S * M):
            if bool(lin[i, j]):
                assert j // M <= i // M


# ── 4. refusals ──────────────────────────────────────────────────────────────

def test_refused_off_a_fan_arm_and_on_a_third_value():
    with pytest.raises(ValueError, match="tul.fan_lineage"):
        _tul(fan_lineage="relation")
    with pytest.raises(ValueError, match="circular"):
        _tul(fan_k=K, slot_cells=K, prefix_k=K, fan_mix="all", fan_lineage="resample")
    _tul(fan_k=K, slot_cells=K, prefix_k=K, fan_mix="all", fan_lineage="relation")


# ── 5. the key composes ──────────────────────────────────────────────────────

def test_the_key_composes_through_tul_setup_and_the_config_resolves():
    import os
    from hydra import compose, initialize_config_dir
    from morph.training.tul_setup import KNOWN_TUL_KEYS

    assert "fan_lineage" in KNOWN_TUL_KEYS
    cdir = os.path.abspath("morph/configs")
    with initialize_config_dir(config_dir=cdir, version_base=None):
        cfg = compose(config_name="tul_slot_spandec_strict_fan4_all_lineage")
    assert cfg.tul.fan_lineage == "relation"
    # it composes the NOISE arm, so the seed noise and the dropped term come with it
    assert float(cfg.tul.fan_seed_noise) == 1.0
    assert float(cfg.tul.fan_repel_lambda) == 0.0
    assert cfg.tul.fan_mix == "all" and int(cfg.tul.fan_k) == 4
    assert cfg.wandb.name == "slot-spandec-strict-fan4-all-lineage"
