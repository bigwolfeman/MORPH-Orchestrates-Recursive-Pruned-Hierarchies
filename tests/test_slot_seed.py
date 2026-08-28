"""TUL slot-seed contract (arms TG4a/TG4b; lab/divergence/TG-WORKLIST.md A1).

`TULConfig.slot_seed` is a construction-time enum selecting how a slot's TOKEN
embedding input is built (`TULSlots.slot_input`, `add_e_slot=True`):
"bag_mean" (default, master), "e_slot" (TG4a), "boundary" (TG4b). The bigram /
value-embed signals (`add_e_slot=False`) must stay the plain bag-mean in every mode.
CPU only, tiny config, no tokenizer — same convention as test_tul_forward.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig, TULSlots, bag_mean, boundary_token_index
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


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _spec(**kw) -> TulLayoutSpec:
    base = dict(seq_len=32, prefix_k=2, max_slots=5, slot_id=4)
    base.update(kw)
    return TulLayoutSpec(**base)


def _batch(spec, B=2, n=90, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == spec.slot_id] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _model(tul: TULConfig | None, seed=1234, **cfg_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(tul=tul, **cfg_kw))


def _span_positions(layout, b: int, s: int) -> list[int]:
    """TOKEN positions (not the slot itself) belonging to slot ``s`` of row ``b``."""
    return [p for p in range(layout.l_total)
            if int(layout.bag_id[b, p]) == s and not bool(layout.slot_mask[b, p])]


# ── TULConfig.slot_seed: the enum itself ──────────────────────────────────────

def test_slot_seed_default_is_bag_mean():
    assert TULConfig().slot_seed == "bag_mean"


def test_illegal_slot_seed_raises_with_the_legal_set():
    with pytest.raises(ValueError, match=r"bag_mean.*e_slot.*boundary"):
        TULConfig(slot_seed="mean_pool")


def test_center_bag_mean_with_e_slot_raises():
    with pytest.raises(ValueError, match="center_bag_mean"):
        TULConfig(slot_seed="e_slot", center_bag_mean=True)


def test_center_bag_mean_with_boundary_raises():
    with pytest.raises(ValueError, match="center_bag_mean"):
        TULConfig(slot_seed="boundary", center_bag_mean=True)


def test_center_bag_mean_with_bag_mean_is_legal():
    TULConfig(slot_seed="bag_mean", center_bag_mean=True)   # must not raise


# ── construction-time parameter dispatch: W_sent ──────────────────────────────

def test_w_sent_built_only_in_boundary_mode():
    assert TULSlots(32, TULConfig(slot_seed="bag_mean")).W_sent is None
    assert TULSlots(32, TULConfig(slot_seed="e_slot")).W_sent is None
    w = TULSlots(32, TULConfig(slot_seed="boundary")).W_sent
    assert w is not None and isinstance(w, torch.nn.Linear) and w.bias is None
    assert w.weight.shape == (32, 32)


# ── "bag_mean" mode: bit-identical to the pre-slot_seed formula ───────────────

def test_bag_mean_mode_matches_the_original_formula():
    """Spec §3.2, reimplemented inline from the ORIGINAL formula (not by calling
    slot_input): E_slot + mean_j embed(t_j) over the span's TOKEN positions."""
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="bag_mean"))
    spec = _spec()
    x, y, layout, _ = _batch(spec)
    with torch.no_grad():
        m.tul.E_slot.normal_(std=1.0)
        emb = m.embed(x)
        got = m.tul.slot_input(emb, layout, add_e_slot=True)
        token_sel = (~layout.slot_mask).float()
        ref_bags = bag_mean(emb, layout.bag_id, token_sel, layout.max_slots)
        ref_at_pos = torch.gather(
            ref_bags, 1,
            layout.bag_id.unsqueeze(-1).expand(*layout.bag_id.shape, emb.shape[-1]))
        ref_at_pos = ref_at_pos + m.tul.E_slot
        ref = torch.where(layout.slot_mask.unsqueeze(-1), ref_at_pos, emb)
    assert torch.equal(got, ref)


def test_bag_mean_mode_add_e_slot_false_is_the_plain_bag_mean():
    """add_e_slot=False must be unaffected by slot_seed — checked here for the
    default mode as the baseline the e_slot/boundary tests below diff against."""
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="bag_mean"))
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


# ── "e_slot" mode (arm TG4a) ───────────────────────────────────────────────────

def test_e_slot_mode_every_real_slot_equals_e_slot_exactly():
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="e_slot"))
    spec = _spec()
    x, y, layout, _ = _batch(spec)
    with torch.no_grad():
        m.tul.E_slot.normal_(std=1.0)
        emb = m.embed(x)
        got = m.tul.slot_input(emb, layout, add_e_slot=True)
    slot_pos = layout.slot_mask
    want = m.tul.E_slot.expand_as(got)
    assert torch.equal(got[slot_pos], want[slot_pos]), "every slot position must be E_slot exactly"
    tok = ~slot_pos
    assert torch.equal(got[tok], emb[tok]), "token positions must be untouched"


def test_e_slot_mode_add_e_slot_false_stays_the_plain_bag_mean():
    """The non-negotiable constraint: add_e_slot=False callers (bigram/value-embed)
    keep the bag-mean in EVERY mode, including e_slot."""
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="e_slot"))
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


def test_e_slot_mode_does_not_depend_on_the_span_tokens():
    """No bag-mean is computed: perturbing a span's token embedding must not move
    that span's slot value at all."""
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="e_slot"))
    spec = _spec()
    x, y, layout, _ = _batch(spec)
    with torch.no_grad():
        m.tul.E_slot.normal_(std=1.0)
    b = 0
    s = next(s for s in range(spec.max_slots)
             if bool(layout.slot_valid[b, s]) and _span_positions(layout, b, s))
    emb = m.embed(x).clone()
    base = m.tul.slot_input(emb, layout, add_e_slot=True)
    p = _span_positions(layout, b, s)[0]
    perturbed = emb.clone()
    perturbed[b, p] += 100.0
    moved = m.tul.slot_input(perturbed, layout, add_e_slot=True)
    # Compare SLOT positions only — TOKEN positions pass ``signal`` through unchanged
    # (that is the invariant `test_e_slot_mode_every_real_slot_equals_e_slot_exactly`
    # checks separately), so the perturbed token position itself is SUPPOSED to move;
    # what must not move is every slot position.
    sm = layout.slot_mask
    assert torch.equal(base[sm], moved[sm]), "e_slot must be invariant to the span's tokens"


# ── "boundary" mode (arm TG4b) ─────────────────────────────────────────────────

def test_boundary_mode_equals_e_slot_plus_projected_last_token():
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="boundary"))
    spec = _spec()
    x, y, layout, _ = _batch(spec)
    with torch.no_grad():
        m.tul.E_slot.normal_(std=1.0)
        m.tul.W_sent.weight.normal_(std=1.0)
        emb = m.embed(x)
        got = m.tul.slot_input(emb, layout, add_e_slot=True)
    b = 0
    s = next(s for s in range(spec.max_slots)
             if bool(layout.slot_valid[b, s]) and _span_positions(layout, b, s))
    last = max(_span_positions(layout, b, s))
    first = int(layout.slot_index[b, s])
    with torch.no_grad():
        want = m.tul.E_slot + m.tul.W_sent(emb[b, last])
    for k in range(spec.prefix_k):
        assert torch.allclose(got[b, first + k], want, atol=1e-6), (
            f"prefix position {k} must carry E_slot + W_sent(embed(t_last))")


def test_boundary_mode_reacts_to_the_last_token_only():
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="boundary"))
    spec = _spec()
    x, y, layout, _ = _batch(spec)
    with torch.no_grad():
        m.tul.E_slot.normal_(std=1.0)
        m.tul.W_sent.weight.normal_(std=1.0)
    b = 0
    s = next(s for s in range(spec.max_slots)
             if bool(layout.slot_valid[b, s]) and len(_span_positions(layout, b, s)) >= 2)
    span = _span_positions(layout, b, s)
    last, other = max(span), min(span)
    emb = m.embed(x).clone()
    base = m.tul.slot_input(emb, layout, add_e_slot=True)
    first = int(layout.slot_index[b, s])

    perturbed_last = emb.clone()
    perturbed_last[b, last] += 100.0
    moved_last = m.tul.slot_input(perturbed_last, layout, add_e_slot=True)
    assert not torch.allclose(base[b, first], moved_last[b, first]), (
        "perturbing the LAST token of the span must change the slot value")

    perturbed_other = emb.clone()
    perturbed_other[b, other] += 100.0
    moved_other = m.tul.slot_input(perturbed_other, layout, add_e_slot=True)
    assert torch.allclose(base[b, first], moved_other[b, first]), (
        "perturbing a NON-last token of the same span must NOT change the slot value")


def test_boundary_mode_add_e_slot_false_stays_the_plain_bag_mean():
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="boundary"))
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


# ── pad / dump-bin invariant, all three modes ──────────────────────────────────

def test_pad_slot_gets_e_slot_alone_in_every_mode():
    """A PAD slot (slot_valid=False, no span) never appears at any real L position —
    the invariant to check is the DUMP BIN (bag_id == max_slots, slot_mask=True: a
    tail-pad position with no span of its own), which DOES occupy a real position and
    must resolve to E_slot alone in every mode. A small max_slots forces the packer to
    truncate on "row would exceed max_slots" (spec §3.1), leaving real trailing tokens
    in the dump bin — measured: 28/72 positions with max_slots=2 on this buffer.
    """
    spec = _spec(max_slots=2)
    x, y, layout, _ = _batch(spec, n=90)
    dump_id = layout.max_slots
    dump_pos = [(b, p) for b in range(x.shape[0]) for p in range(layout.l_total)
                if bool(layout.slot_mask[b, p]) and int(layout.bag_id[b, p]) == dump_id]
    assert dump_pos, "this test needs at least one dump-bin (tail-pad) position"
    for mode in ("bag_mean", "e_slot", "boundary"):
        m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed=mode))
        with torch.no_grad():
            m.tul.E_slot.normal_(std=1.0)
            if m.tul.W_sent is not None:
                m.tul.W_sent.weight.normal_(std=1.0)
            emb = m.embed(x)
            got = m.tul.slot_input(emb, layout, add_e_slot=True)
        for b, p in dump_pos:
            assert torch.allclose(got[b, p], m.tul.E_slot, atol=1e-6), (
                mode, b, p, "tail-pad slot must get E_slot alone")


# ── boundary_token_index: brute force, hand-built layout ──────────────────────

def _brute_boundary_index(bag_id: torch.Tensor, token_sel: torch.Tensor, n_bags: int
                          ) -> torch.Tensor:
    B, L = bag_id.shape
    out = torch.full((B, n_bags + 1), -1, dtype=torch.long)
    for b in range(B):
        for p in range(L):
            if bool(token_sel[b, p]):
                s = int(bag_id[b, p])
                if s < n_bags and p > int(out[b, s]):
                    out[b, s] = p
    # dump bin (index n_bags) is forced to -1 by contract, regardless of what tokens
    # landed in it — see boundary_token_index's docstring.
    out[:, n_bags] = -1
    return out


def test_boundary_token_index_hand_built():
    bag_id = torch.tensor([[0, 0, 2, 1, 1, 2]])          # 2 = dump bin (n_bags=2)
    sel = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 0.0]])
    got = boundary_token_index(bag_id, sel, n_bags=2)
    ref = _brute_boundary_index(bag_id, sel, n_bags=2)
    assert torch.equal(got, ref)
    assert got.tolist() == [[1, 4, -1]], got.tolist()


def test_boundary_token_index_matches_brute_force_on_a_real_packed_batch():
    spec = _spec()
    x, y, layout, _ = _batch(spec)
    token_sel = (~layout.slot_mask).float()
    got = boundary_token_index(layout.bag_id, token_sel, layout.max_slots)
    ref = _brute_boundary_index(layout.bag_id, token_sel, layout.max_slots)
    assert torch.equal(got, ref)
    # sanity: every REAL slot got a non-negative boundary, every unreached slot index
    # (slot_valid=False) stayed -1 because bag_id never equals that index.
    for b in range(x.shape[0]):
        for s in range(spec.max_slots):
            if bool(layout.slot_valid[b, s]):
                assert int(got[b, s]) >= 0
            else:
                assert int(got[b, s]) == -1


# ── the eval-time flip used by lab/divergence/slot_path_worth.py ──────────────────

def test_seed_bagmean_flip_restores_the_bag_mean_seed():
    """`seed_bagmean` must make an e_slot model's slot input EQUAL a bag_mean model's.

    The worth harness compares arms by ablating the loop, and `loop_off` leaves the slot
    carrying its own INPUT. On an e_slot arm that input is a constant, so the plain
    "no-loop" column measures "loop vs nothing" while a bag_mean arm's measures "loop vs
    a span summary". `seed_bagmean` flips `slot_seed` at eval time so both fall back to
    the same thing. This test is the reason that flip is trusted: it asserts the flipped
    output is bit-equal to the real bag_mean path and that the flip is UNDONE on exit.
    """
    from lab.divergence.slot_path_worth import seed_bagmean

    spec = _spec()
    x, _y, layout, _ = _batch(spec)
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="e_slot"))
    ref = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="bag_mean"))
    with torch.no_grad():
        m.tul.E_slot.normal_(std=1.0)            # a zero E_slot hides a broken flip
    ref.load_state_dict(m.state_dict())          # same weights, different seed mode

    with torch.no_grad():
        emb = m.embed(x)
        as_eslot = m.tul.slot_input(emb, layout, add_e_slot=True)
        want = ref.tul.slot_input(emb, layout, add_e_slot=True)
        with seed_bagmean(m):
            assert m.tul.tul.slot_seed == "bag_mean"
            got = m.tul.slot_input(emb, layout, add_e_slot=True)

    # The flip is the whole point: it must CHANGE the output, and change it to the
    # bag_mean path exactly. An assertion on only one of those passes on a no-op.
    slots = layout.slot_mask.unsqueeze(-1)
    assert not torch.equal(as_eslot[slots.expand_as(as_eslot)],
                           want[slots.expand_as(want)]), "e_slot and bag_mean agree — " \
        "the fixture cannot detect a broken flip"
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert m.tul.tul.slot_seed == "e_slot", "seed_bagmean leaked past its with-block"


def test_seed_bagmean_restores_on_exception():
    """A raising body must not leave the model in bag_mean mode for every later eval."""
    from lab.divergence.slot_path_worth import seed_bagmean

    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="boundary"))
    with pytest.raises(RuntimeError):
        with seed_bagmean(m):
            raise RuntimeError("boom")
    assert m.tul.tul.slot_seed == "boundary"
