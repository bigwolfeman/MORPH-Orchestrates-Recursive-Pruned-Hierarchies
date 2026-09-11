"""``plan_mode="all_slots"`` — the worth profile's route split (2026-09-11).

The cross-span budget (``lab/experiments/failures/2026-09-11-arc-span-budget.md``) priced
the slot channel's prefix write at 0.093 nats against a 0.399-nat budget and named what
the ``zero`` ablation leaves open: under ``tg_restrict`` the coda's allow relation is
"same span OR any slot position", so a slot CELL re-summarises its own span from the
coda's token states and every later token reads that summary.

``all_slots`` cuts all three tensors that reach a coda slot cell:

  1. the prefix write (``_tul_plan_ablate``, shared with ``zero``);
  2. the per-layer additive injection at that position (``inject_keep``);
  3. the cell's attention over its own span (``tg_allow`` with
     ``slot_queries_slots_only=True``).

One test per cut, structurally — the arguments ``_back_region`` actually receives — plus
the behavioural check that the three together move the logits beyond what ``zero`` moves
them, plus every refusal. The structural tests are the ones that fail when a cut is
dropped; a CE comparison on an untrained tiny model cannot carry a sign.

CPU only, tiny config, no tokenizer.
"""

from __future__ import annotations

import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402  (tests/ is on sys.path)

from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig


def _mask_model(seed: int = 4, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    base = dict(tg_restrict=True, tg_restrict_scope="all", sigreg_lambda=0.0,
                mux_beta=0.0, token_state_dropout=0.0)
    base.update(tul_kw)
    return MORPHTransformer(_cfg(tul=_tul(**base), n_core=2, mean_depth=2, max_depth=2,
                                 bptt_depth=2, retention=False, dropout=0.0,
                                 core_fixed_point_lambda=0.0))


def _spy(m: MORPHTransformer) -> list[dict]:
    """Record every ``_back_region`` call's ``inject_keep`` and ``attn_kwargs``.

    Drops any previous instance-level wrapper first, so two ``_run`` calls on the same
    model do not chain and write into each other's list."""
    seen: list[dict] = []
    m.__dict__.pop("_back_region", None)
    real = m._back_region

    def wrapped(x, x0, bigram_emb, input_ids=None, inject_keep=None, attn_kwargs=None,
                ret_reset_mask=None):
        seen.append({"x": x.detach().clone(), "keep": inject_keep, "kw": attn_kwargs})
        return real(x, x0, bigram_emb, input_ids, inject_keep, attn_kwargs, ret_reset_mask)

    m._back_region = wrapped                                   # type: ignore[assignment]
    return seen


def _run(m, mode: str):
    x, _y, lay, _ = _batch()
    m.eval()
    seen = _spy(m)
    with torch.no_grad():
        out = m.tul_forward_ablated(x, None, lay, plan_mode=mode)
    return out, seen, lay


# ── cut 1: the prefix write ──────────────────────────────────────────────────

def test_all_slots_zeroes_the_prefix_write_like_zero_does():
    m = _mask_model()
    _o_z, seen_z, lay = _run(m, "zero")
    _o_a, seen_a, _ = _run(m, "all_slots")
    # REAL slot cells only. A tail-pad cell addresses `prefix_project`'s dump row, so it
    # keeps its prelude value under every mode — harmless, because the packer puts pads at
    # the row's tail where no real token attends them causally, and their labels are -100.
    vslot = lay.slot_mask & (lay.bag_id < lay.max_slots)
    assert int(vslot.sum()) > 4
    for s in (seen_z, seen_a):
        assert len(s) == 1
        xs = s[0]["x"]
        assert float(xs[vslot].abs().sum()) == 0.0, "slot cells still carry a written value"
    # and `normal` does NOT zero it (the check cannot pass vacuously)
    _o_n, seen_n, _ = _run(m, "normal")
    assert float(seen_n[0]["x"][vslot].abs().sum()) > 0.0


# ── cut 2: the per-layer injection at slot cells ─────────────────────────────

def test_all_slots_zeroes_the_coda_injection_at_slot_cells_and_zero_does_not():
    m = _mask_model()
    _o_z, seen_z, lay = _run(m, "zero")
    _o_a, seen_a, _ = _run(m, "all_slots")
    assert seen_z[0]["keep"] is None, (
        "`zero` must leave the injections alone — if it stops doing so, every worth "
        "number filed before 2026-09-11 changes meaning")
    keep = seen_a[0]["keep"]
    assert keep is not None and keep.shape == (*lay.slot_mask.shape, 1)
    assert float(keep[lay.slot_mask].abs().sum()) == 0.0
    assert float(keep[~lay.slot_mask].min()) == 1.0


# ── cut 3: the cell's attention over its own span ────────────────────────────

def test_all_slots_restricts_a_slot_query_to_slot_keys():
    m = _mask_model()
    _o_z, seen_z, lay = _run(m, "zero")
    _o_a, seen_a, _ = _run(m, "all_slots")
    slot = lay.slot_mask
    a_z = seen_z[0]["kw"]["tg_allow"][:, 0]                     # [B, L, L]
    a_a = seen_a[0]["kw"]["tg_allow"][:, 0]
    B, L, _ = a_a.shape
    for b in range(B):
        q = slot[b].nonzero().flatten()
        # under all_slots a slot query may allow slot keys ONLY
        assert not bool((a_a[b][q][:, ~slot[b]]).any()), "a slot cell still reads tokens"
        # and under `zero` it DOES read its own span's tokens (two-sided)
        assert bool((a_z[b][q][:, ~slot[b]]).any())
    # token queries are untouched: the ablation must not change what TOKENS may attend
    for b in range(B):
        t = (~slot[b]).nonzero().flatten()
        assert torch.equal(a_a[b][t], a_z[b][t]), "the ablation moved a TOKEN's relation"


# ── the three together do more than `zero` ───────────────────────────────────

def test_all_slots_moves_the_logits_strictly_further_than_zero():
    m = _mask_model()
    o_n, _s, lay = _run(m, "normal")
    o_z, _s, _ = _run(m, "zero")
    o_a, _s, _ = _run(m, "all_slots")
    tok = ~lay.slot_mask
    # The structural slot id's logit is forced to -inf at generation, so it must come out
    # of the comparison: (-inf) - (-inf) is NaN, not 0.
    keep = torch.ones(o_n["logits"].shape[-1], dtype=torch.bool)
    keep[m.cfg.tul.slot_id] = False
    ln, lz, la = (o["logits"][tok][:, keep] for o in (o_n, o_z, o_a))
    assert torch.isfinite(ln).all()
    d_zero = float((lz - ln).abs().mean())
    d_all = float((la - ln).abs().mean())
    assert d_zero > 0.0
    assert d_all > d_zero, (d_all, d_zero)


# ── refusals ─────────────────────────────────────────────────────────────────

def test_all_slots_refuses_where_the_route_model_does_not_hold():
    x, _y, lay, _ = _batch()
    plain = MORPHTransformer(_cfg(tul=_tul(tg_restrict=False, sigreg_lambda=0.0,
                                           mux_beta=0.0), n_core=2, retention=False))
    plain.eval()
    with pytest.raises(NotImplementedError, match="tul.tg_restrict"):
        plain.tul_forward_ablated(x, None, lay, plan_mode="all_slots")
    coda = _mask_model()
    coda.cfg.tul.tg_restrict_scope = "coda"
    coda.eval()
    with pytest.raises(NotImplementedError, match="tg_restrict_scope"):
        coda.tul_forward_ablated(x, None, lay, plan_mode="all_slots")


def test_an_unknown_plan_mode_still_raises():
    m = _mask_model()
    x, _y, lay, _ = _batch()
    m.eval()
    with pytest.raises(ValueError, match="normal|zero|shuffle|all_slots"):
        m.tul_forward_ablated(x, None, lay, plan_mode="all_slot")


def test_plan_ablate_all_slots_matches_zero_on_the_state_itself():
    m = _mask_model()
    h = torch.randn(2, 4, 4, m.cfg.d_model)
    x, _y, lay, _ = _batch()
    a = m._tul_plan_ablate(h, lay, "all_slots")
    b = m._tul_plan_ablate(h, lay, "zero")
    assert torch.equal(a, b) and float(a.abs().sum()) == 0.0
