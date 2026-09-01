"""Write-side ladder — the "content" and "bound" `TULConfig.slot_seed` modes.

Prereg: lab/experiments/planned/2026-09-01-write-side-ladder.md. E2
(lab/experiments/failures/2026-09-01-bound-seed-rank.md) measured that the shared
`E_slot` additive term collapses every slot seed to ~rank-1; "content" (arm W1) is
"bag_mean" with that term dropped, and "bound" (arm W2) is an HRR-style per-offset
rotation binding, also without the term — the E2 probe's "bag_noeslot" /
"bound_noeslot" columns made model paths. CPU only, tiny config, no tokenizer —
same convention as tests/test_slot_seed.py, whose scaffolding this reuses.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig, TULSlots, bag_mean, bound_seed, build_bound_rotations
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=2, max_depth=3, bptt_depth=2,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _rule(span_cap=8) -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=span_cap, eos_id=0)


def _spec(**kw) -> TulLayoutSpec:
    base = dict(seq_len=32, prefix_k=2, max_slots=5, slot_id=4)
    base.update(kw)
    return TulLayoutSpec(**base)


def _batch(spec, rule=None, B=2, n=90, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == spec.slot_id] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), rule or _rule(), spec)


def _model(tul: TULConfig | None, seed=1234, **cfg_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(tul=tul, **cfg_kw))


def _span_positions(layout, b: int, s: int) -> list[int]:
    """TOKEN positions (not the slot itself) belonging to slot ``s`` of row ``b``,
    IN ORDER — the order the "bound" formula's offset assumes."""
    return [p for p in range(layout.l_total)
            if int(layout.bag_id[b, p]) == s and not bool(layout.slot_mask[b, p])]


# ── validation ──────────────────────────────────────────────────────────────────

def test_illegal_slot_seed_still_raises_with_the_full_legal_set():
    with pytest.raises(ValueError, match=r"bag_mean.*e_slot.*boundary.*content.*bound"):
        TULConfig(slot_seed="mean_pool")


def test_content_mode_constructs():
    TULConfig(slot_seed="content")   # must not raise


def test_bound_mode_constructs():
    TULConfig(slot_seed="bound")   # must not raise


def test_bound_span_cap_must_be_positive():
    with pytest.raises(ValueError, match="bound_span_cap"):
        TULConfig(slot_seed="bound", bound_span_cap=0)


def test_center_bag_mean_with_content_raises():
    """center_bag_mean is scoped to slot_seed="bag_mean" only (see TULConfig's
    __post_init__ comment) — "content" must not silently ignore it either."""
    with pytest.raises(ValueError, match="center_bag_mean"):
        TULConfig(slot_seed="content", center_bag_mean=True)


def test_center_bag_mean_with_bound_raises():
    with pytest.raises(ValueError, match="center_bag_mean"):
        TULConfig(slot_seed="bound", center_bag_mean=True)


# ── construction-time parameter dispatch: bound_R ──────────────────────────────

def test_bound_r_built_only_in_bound_mode():
    assert TULSlots(32, TULConfig(slot_seed="bag_mean")).bound_R is None
    assert TULSlots(32, TULConfig(slot_seed="e_slot")).bound_R is None
    assert TULSlots(32, TULConfig(slot_seed="boundary")).bound_R is None
    assert TULSlots(32, TULConfig(slot_seed="content")).bound_R is None
    r = TULSlots(32, TULConfig(slot_seed="bound", bound_span_cap=6)).bound_R
    assert r is not None and r.shape == (6, 32, 32)


def test_bound_r_is_not_a_persistent_buffer():
    """persistent=False: it must move to `.to(device)` (buffers always do — checked
    indirectly via named_buffers below) but never land in state_dict / checkpoints."""
    m = TULSlots(16, TULConfig(slot_seed="bound", bound_span_cap=4))
    assert "bound_R" not in m.state_dict()
    names = dict(m.named_buffers())
    assert "bound_R" in names and names["bound_R"].shape == (4, 16, 16)


def test_bound_rotations_are_orthogonal():
    R = build_bound_rotations(20, 5, seed=17)
    assert R.shape == (5, 20, 20)
    eye = torch.eye(20)
    for k in range(5):
        torch.testing.assert_close(R[k] @ R[k].T, eye, atol=1e-4, rtol=0)
        torch.testing.assert_close(R[k].T @ R[k], eye, atol=1e-4, rtol=0)


# ── "content" mode: bag_mean minus E_slot, everything else identical ───────────

def test_content_and_bag_mean_state_dicts_have_identical_keys():
    a = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="bag_mean"))
    b = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="content"))
    assert set(a.state_dict()) == set(b.state_dict())


def test_content_mode_equals_bag_mean_minus_e_slot():
    bagm = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="bag_mean"))
    cont = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="content"))
    with torch.no_grad():
        bagm.tul.E_slot.normal_(std=1.0)
    cont.load_state_dict(bagm.state_dict())     # identical keys, same weights now

    spec = _spec()
    x, y, layout, _ = _batch(spec)
    with torch.no_grad():
        emb = bagm.embed(x)
        bag_out = bagm.tul.slot_input(emb, layout, add_e_slot=True)
        content_out = cont.tul.slot_input(emb, layout, add_e_slot=True)
    sm = layout.slot_mask.unsqueeze(-1).expand_as(bag_out)
    e_term = cont.tul.E_slot.expand_as(bag_out)
    torch.testing.assert_close((bag_out - e_term)[sm], content_out[sm], atol=1e-5, rtol=0)


def test_content_mode_add_e_slot_false_stays_the_plain_bag_mean():
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="content"))
    spec = _spec()
    x, y, layout, _ = _batch(spec)
    sig = torch.randn(x.shape[0], layout.l_total, 64)
    got = m.tul.slot_input(sig, layout, add_e_slot=False)
    token_sel = (~layout.slot_mask).float()
    bags = bag_mean(sig, layout.bag_id, token_sel, layout.max_slots)
    ref_at_pos = torch.gather(
        bags, 1, layout.bag_id.unsqueeze(-1).expand(*layout.bag_id.shape, sig.shape[-1]))
    ref = torch.where(layout.slot_mask.unsqueeze(-1), ref_at_pos, sig)
    assert torch.equal(got, ref)


def test_content_mode_pad_slot_is_exactly_zero():
    """Unlike bag_mean/e_slot/boundary (which fall back to E_slot), "content" adds
    nothing at all — a dump-bin (no-span) slot position must be exactly 0."""
    spec = _spec(max_slots=2)
    x, y, layout, _ = _batch(spec, n=90)
    dump_id = layout.max_slots
    dump_pos = [(b, p) for b in range(x.shape[0]) for p in range(layout.l_total)
                if bool(layout.slot_mask[b, p]) and int(layout.bag_id[b, p]) == dump_id]
    assert dump_pos, "this test needs at least one dump-bin (tail-pad) position"
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="content"))
    with torch.no_grad():
        m.tul.E_slot.normal_(std=1.0)
        emb = m.embed(x)
        got = m.tul.slot_input(emb, layout, add_e_slot=True)
    for b, p in dump_pos:
        assert torch.allclose(got[b, p], torch.zeros_like(got[b, p]), atol=1e-6)


# ── "bound" mode: pins the vectorized path to the E2 probe's reference math ────

def _reference_bound_seed(signal: torch.Tensor, layout, span_cap: int, seed: int = 17
                          ) -> dict[tuple[int, int], torch.Tensor]:
    """Brute-force, one Python loop over (row, slot) — the E2 probe's exact formula,
    reimplemented independently of morph.model.tul.bound_seed so this test can
    actually catch a bug in the vectorized implementation."""
    d = signal.shape[-1]
    g = torch.Generator().manual_seed(seed)
    R = torch.stack([torch.linalg.qr(torch.randn(d, d, generator=g))[0]
                     for _ in range(span_cap)])
    out: dict[tuple[int, int], torch.Tensor] = {}
    for b in range(signal.shape[0]):
        for s in range(layout.slot_valid.shape[1]):
            if not bool(layout.slot_valid[b, s]):
                continue
            pos = _span_positions(layout, b, s)
            n = min(len(pos), span_cap)
            if n == 0:
                continue
            e = signal[b, pos[:n]]
            out[(b, s)] = torch.einsum("kij,kj->i", R[:n].float(), e) / (n ** 0.5)
    return out


def test_bound_seed_matches_the_probe_reference_math():
    spec = _spec()
    rule = _rule(span_cap=8)
    x, y, layout, _ = _batch(spec, rule=rule)
    d = 24
    torch.manual_seed(9)
    signal = torch.randn(x.shape[0], layout.l_total, d)
    R = build_bound_rotations(d, rule.span_cap, seed=17)
    token_sel = (~layout.slot_mask).float()

    got_bags = bound_seed(signal, layout.bag_id, token_sel, layout.max_slots, R)
    got_at_pos = torch.gather(
        got_bags, 1, layout.bag_id.unsqueeze(-1).expand(*layout.bag_id.shape, d))
    ref = _reference_bound_seed(signal, layout, rule.span_cap)

    checked = 0
    for (b, s), want in ref.items():
        first = int(layout.slot_index[b, s])
        got = got_at_pos[b, first]
        torch.testing.assert_close(got, want, atol=1e-4, rtol=0)
        checked += 1
    assert checked >= 3, "fixture produced too few real spans to be a meaningful check"


def test_bound_mode_slot_input_matches_bound_seed_plus_gather():
    """The shipped `slot_input` dispatch for "bound" must be exactly
    `bound_seed` gathered to positions — no extra term (unlike bag_mean/boundary,
    "bound" adds no E_slot)."""
    spec = _spec()
    rule = _rule(span_cap=8)
    x, y, layout, _ = _batch(spec, rule=rule)
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="bound",
                         bound_span_cap=rule.span_cap))
    with torch.no_grad():
        emb = m.embed(x)
        got = m.tul.slot_input(emb, layout, add_e_slot=True)
        token_sel = (~layout.slot_mask).float()
        bags = bound_seed(emb, layout.bag_id, token_sel, layout.max_slots, m.tul.bound_R)
        ref_at_pos = torch.gather(
            bags, 1, layout.bag_id.unsqueeze(-1).expand(*layout.bag_id.shape, emb.shape[-1]))
        ref = torch.where(layout.slot_mask.unsqueeze(-1), ref_at_pos, emb)
    assert torch.equal(got, ref)


def test_bound_mode_add_e_slot_false_stays_the_plain_bag_mean():
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="bound"))
    spec = _spec()
    x, y, layout, _ = _batch(spec)
    sig = torch.randn(x.shape[0], layout.l_total, 64)
    got = m.tul.slot_input(sig, layout, add_e_slot=False)
    token_sel = (~layout.slot_mask).float()
    bags = bag_mean(sig, layout.bag_id, token_sel, layout.max_slots)
    ref_at_pos = torch.gather(
        bags, 1, layout.bag_id.unsqueeze(-1).expand(*layout.bag_id.shape, sig.shape[-1]))
    ref = torch.where(layout.slot_mask.unsqueeze(-1), ref_at_pos, sig)
    assert torch.equal(got, ref)


def test_bound_mode_pad_slot_is_exactly_zero():
    spec = _spec(max_slots=2)
    x, y, layout, _ = _batch(spec, n=90)
    dump_id = layout.max_slots
    dump_pos = [(b, p) for b in range(x.shape[0]) for p in range(layout.l_total)
                if bool(layout.slot_mask[b, p]) and int(layout.bag_id[b, p]) == dump_id]
    assert dump_pos, "this test needs at least one dump-bin (tail-pad) position"
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="bound"))
    with torch.no_grad():
        emb = m.embed(x)
        got = m.tul.slot_input(emb, layout, add_e_slot=True)
    for b, p in dump_pos:
        assert torch.allclose(got[b, p], torch.zeros_like(got[b, p]), atol=1e-6)


# ── span_cap truncation: offsets beyond the rotation table are DROPPED ─────────

def test_bound_seed_drops_tokens_past_span_cap():
    """A tiny `span_cap` (rotation table size) smaller than real span lengths must
    truncate the sum to the first `span_cap` tokens of the span, dropping the rest
    — not clamp them into the last rotation's slot."""
    spec = _spec()
    rule = _rule(span_cap=8)             # data spans can run up to 8 tokens
    x, y, layout, _ = _batch(spec, rule=rule, n=90)
    small_cap = 3                         # rotation table smaller than the data allows
    d = 24
    torch.manual_seed(11)
    signal = torch.randn(x.shape[0], layout.l_total, d)
    R = build_bound_rotations(d, small_cap, seed=17)
    token_sel = (~layout.slot_mask).float()
    bags = bound_seed(signal, layout.bag_id, token_sel, layout.max_slots, R)
    got_at_pos = torch.gather(
        bags, 1, layout.bag_id.unsqueeze(-1).expand(*layout.bag_id.shape, d))

    truncated_seen = False
    for b in range(x.shape[0]):
        for s in range(spec.max_slots):
            if not bool(layout.slot_valid[b, s]):
                continue
            pos = _span_positions(layout, b, s)
            n = len(pos)
            if n == 0:
                continue
            n_keep = min(n, small_cap)
            if n > small_cap:
                truncated_seen = True
            e = signal[b, pos[:n_keep]]
            want = torch.einsum("kij,kj->i", R[:n_keep].float(), e) / (n_keep ** 0.5)
            first = int(layout.slot_index[b, s])
            torch.testing.assert_close(got_at_pos[b, first], want, atol=1e-4, rtol=0)
    assert truncated_seen, "fixture never exercised the truncation path"


# ── global-RNG neutrality: building a "bound" model must not perturb the stream ─

def test_bound_construction_does_not_perturb_the_global_rng():
    """`build_bound_rotations` draws from a PRIVATE generator. A "boundary" model
    built after a "bound" model must draw byte-identically to one built with
    nothing "bound" built first."""
    torch.manual_seed(77)
    baseline = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="boundary"), seed=77)

    torch.manual_seed(77)
    _ = TULSlots(64, TULConfig(slot_seed="bound", bound_span_cap=32))   # side model
    after_bound = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="boundary"), seed=77)

    sa, sb = baseline.state_dict(), after_bound.state_dict()
    assert set(sa) == set(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"weight mismatch at {k} — bound_R perturbed the RNG stream"


# ── forward+backward on the shipped path ────────────────────────────────────────

@pytest.mark.parametrize("mode", ["content", "bound"])
def test_forward_backward_run(mode):
    spec = _spec()
    x, y, layout, _ = _batch(spec)
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed=mode))
    out = m(x, labels=y, slot_layout=layout)
    loss = out["loss"]
    assert torch.isfinite(loss), f"non-finite loss under slot_seed={mode!r}: {loss}"
    loss.backward()
    grads = [(n, p.grad) for n, p in m.named_parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for _, g in grads), \
        f"non-finite gradient under slot_seed={mode!r}"
