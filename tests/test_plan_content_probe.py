"""`lab/divergence/plan_content_probe.py` must be shown to FAIL and to DETECT before it
is trusted on a real checkpoint. CPU only, tiny shapes, no tokenizer, no dataloader.

Three layers:
  1. Pure-tensor unit tests of the extraction/reduction/shuffle machinery.
  2. The two guards (fit/eval disjointness, frozen-parameter leak) actually firing.
  3. The DECIDING test: synthetic z that literally encodes the next span (positive
     control) must give a LARGE PLAN - SHUFFLED; synthetic z that is pure noise
     (negative control) must give a gap near zero. If the probe cannot tell those two
     cases apart, it cannot be trusted on a real checkpoint.
  4. One end-to-end check against a REAL (tiny) MORPHTransformer + TUL forward, so the
     extraction logic is validated against the layout the model itself produces, not
     just against hand-built tensors.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lab.divergence.plan_content_probe import (
    BlindSpanDecoder, SlotExamples, assert_disjoint_batches, assert_frozen,
    capture_prefix_project, cat_examples, extract_slot_examples, param_fingerprint,
    reduce_prefix_values, run_four_conditions, shuffled_other_row,
)
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

DEVICE = torch.device("cpu")


# ── tiny real model, matching tests/test_tul_forward.py's helpers ────────────

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=32, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=1, n_coda=1, mean_depth=2, max_depth=2, bptt_depth=1,
        channel_dims=(16, 10, 6), compression=2, csa_compress_ratio=4,
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
    base = dict(seq_len=48, prefix_k=2, max_slots=6, slot_id=4)
    base.update(kw)
    return TulLayoutSpec(**base)


def _batch(spec, B=3, n=140, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == spec.slot_id] = 5
    ids[:, ::6] = DOT       # boundary every 6 tokens -> spans of ~5-6 tokens
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _model(seed=1234, **cfg_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul = TULConfig(prefix_k=2, slot_id=4)
    return MORPHTransformer(_tiny(tul=tul, **cfg_kw))


# ── 1. reduce_prefix_values ───────────────────────────────────────────────────

def test_reduce_prefix_values_plain_carrier_no_mid_dims():
    B, S, K, C = 2, 3, 2, 5
    values = torch.arange(B * S * K * C, dtype=torch.float32).reshape(B, S * K, C)
    out = reduce_prefix_values(values, S, K)
    assert out.shape == (B, S, K * C)
    # slot 0's K=2 positions are rows 0,1 of the flattened S*K axis; concatenation must
    # preserve them in order (offset 0's C values, then offset 1's C values).
    expected = torch.cat([values[0, 0], values[0, 1]])
    assert torch.equal(out[0, 0], expected)


def test_reduce_prefix_values_hc_carrier_means_over_streams():
    B, S, K, n, C = 1, 2, 2, 4, 3
    values = torch.zeros(B, S * K, n, C)
    values[0, 0] = torch.tensor([[1.0, 1.0, 1.0]] * n)   # slot 0, offset 0: all streams = 1
    values[0, 0, 0] = torch.tensor([5.0, 5.0, 5.0])        # one stream perturbed
    out = reduce_prefix_values(values, S, K)
    assert out.shape == (B, S, K * C)
    # mean over 4 streams of [5,5,5] once and [1,1,1] three times = [2,2,2]
    assert torch.allclose(out[0, 0, :C], torch.tensor([2.0, 2.0, 2.0]))


# ── 2. extract_slot_examples on hand-built layouts ───────────────────────────

def _hand_layout():
    """One row, 3 real slots + 1 pad. Row: [tok tok tok tok] slot0[k0 k1] [tok tok tok
    tok tok] slot1[k0 k1] [tok tok tok tok] slot2[k0 k1] pad[k0 k1].
    bag_id: span0 tokens -> 0, slot0 -> 0; span1 tokens -> 1, slot1 -> 1; span2 tokens
    -> 2, slot2 -> 2. Matches pack_tul_row's convention exactly."""
    L = 4 + 2 + 5 + 2 + 4 + 2 + 2   # = 21
    ids = torch.arange(100, 100 + L).long().unsqueeze(0)
    slot_mask = torch.zeros(1, L, dtype=torch.bool)
    bag_id = torch.zeros(1, L, dtype=torch.long)
    pos = 0
    span_lens = [4, 5, 4]
    for s, sl in enumerate(span_lens):
        bag_id[0, pos:pos + sl] = s
        pos += sl
        slot_mask[0, pos:pos + 2] = True
        bag_id[0, pos:pos + 2] = s
        pos += 2
    # tail pad slot (bag_id dump bin = max_slots = 4)
    slot_mask[0, pos:pos + 2] = True
    bag_id[0, pos:pos + 2] = 4
    slot_index = torch.tensor([[4, 11, 18, 0]])
    slot_valid = torch.tensor([[True, True, True, False]])
    from morph.model.tul_layout import SlotLayout
    layout = SlotLayout(slot_mask=slot_mask, bag_id=bag_id, slot_index=slot_index,
                        slot_valid=slot_valid, prefix_k=2)
    return ids, layout, span_lens


def test_extract_slot_examples_recovers_span_tokens_in_order():
    ids, layout, span_lens = _hand_layout()
    S = layout.slot_index.shape[1]
    z = torch.randn(1, S, 6)
    ex = extract_slot_examples(ids, z, layout, span_cap=8, row_offset=0)
    # 3 valid slots -> slot 2 (last valid) is excluded (no span 3) -> 2 usable examples
    assert ex.z.shape[0] == 2
    assert ex.n_total_valid == 3
    assert ex.n_excluded == 1
    # example 0: slot 0, span_i = span 0's 4 tokens, span_i+1 = span 1's 5 tokens
    span0_tokens = ids[0, :span_lens[0]]
    span1_start = span_lens[0] + 2
    span1_tokens = ids[0, span1_start:span1_start + span_lens[1]]
    assert torch.equal(ex.span_i[0, :span_lens[0]], span0_tokens)
    assert (ex.span_i[0, span_lens[0]:] == -100).all()
    assert torch.equal(ex.span_ip1[0, :span_lens[1]], span1_tokens)
    assert (ex.span_ip1[0, span_lens[1]:] == -100).all()


def test_extract_slot_examples_raises_on_span_longer_than_cap():
    ids, layout, span_lens = _hand_layout()
    S = layout.slot_index.shape[1]
    z = torch.randn(1, S, 6)
    with pytest.raises(RuntimeError, match="span_cap"):
        extract_slot_examples(ids, z, layout, span_cap=3, row_offset=0)   # span0 has 4


def test_cat_examples_concatenates_and_sums_counts():
    ids, layout, _ = _hand_layout()
    S = layout.slot_index.shape[1]
    z = torch.randn(1, S, 6)
    ex1 = extract_slot_examples(ids, z, layout, span_cap=8, row_offset=0)
    ex2 = extract_slot_examples(ids, z, layout, span_cap=8, row_offset=1)
    both = cat_examples([ex1, ex2])
    assert both.z.shape[0] == ex1.z.shape[0] + ex2.z.shape[0]
    assert both.n_excluded == ex1.n_excluded + ex2.n_excluded
    assert set(both.row_id.tolist()) == {0, 1}   # row_offset actually changed the id


# ── 3. shuffled_other_row ─────────────────────────────────────────────────────

def test_shuffled_other_row_never_reuses_the_same_row():
    row_id = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    z = torch.arange(9).float().unsqueeze(-1)
    shuf = shuffled_other_row(z, row_id, seed=0)
    # z's value IS the original index, so we can read back which index each output
    # example was drawn from and check its row differs from the row at that position.
    picked_row = row_id[shuf.squeeze(-1).long()]
    assert torch.all(picked_row != row_id), "shuffle picked a same-row replacement"


def test_shuffled_other_row_requires_two_rows():
    z = torch.randn(4, 3)
    row_id = torch.zeros(4, dtype=torch.long)
    with pytest.raises(RuntimeError, match="2 distinct rows"):
        shuffled_other_row(z, row_id, seed=0)


# ── 4. freeze guards ──────────────────────────────────────────────────────────

def test_assert_frozen_raises_when_a_param_is_left_trainable():
    m = _model()
    m.requires_grad_(False)
    assert_frozen(m)   # passes: everything is frozen
    next(m.parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError, match="require grad"):
        assert_frozen(m)


def test_param_fingerprint_changes_when_a_param_is_mutated():
    m = _model()
    m.requires_grad_(False)
    fp0 = param_fingerprint(m)
    # `param_fingerprint` samples the first `n` params by SORTED NAME, not registration
    # order — mutate the same one it actually reads, `sorted(named_parameters())[0][1]`.
    with torch.no_grad():
        first_param = sorted(m.named_parameters(), key=lambda kv: kv[0])[0][1]
        first_param.add_(1.0)
    fp1 = param_fingerprint(m)
    assert not torch.equal(fp0, fp1), "fingerprint failed to notice a mutated parameter"


def test_assert_disjoint_batches_raises_on_overlap():
    x = torch.randint(0, 50, (2, 10))
    layout_stub = None
    fit = [(x, x, layout_stub)]
    ev = [(x.clone(), x.clone(), layout_stub)]
    with pytest.raises(RuntimeError, match="fit and eval"):
        assert_disjoint_batches(fit, ev)


def test_assert_disjoint_batches_passes_when_disjoint():
    x1 = torch.randint(0, 50, (2, 10))
    x2 = x1 + 1000   # guaranteed different rows
    assert_disjoint_batches([(x1, x1, None)], [(x2, x2, None)])   # must not raise


# ── 5. THE DECIDING TEST: positive vs. negative control ──────────────────────

def _pad(tokens: np.ndarray, span_cap: int) -> np.ndarray:
    out = np.full(span_cap, -100, dtype=np.int64)
    out[: len(tokens)] = tokens
    return out


def _synthetic_world(*, num_span_ids: int, span_cap: int, z_dim: int, seed: int):
    """The FIXED span_id -> (target tokens, embedding) mapping, built ONCE and shared by
    every fit/eval draw below. Regenerating this per draw (i.e. from the SAME seed as the
    row sampling) would make fit and eval score two DIFFERENT, uncorrelated target
    vocabularies — an artifact of the test harness, not a property of the probe, and it
    produces meaningless numbers. A real corpus's span-content mapping does not change
    between the fit and eval slices of one probe run; this mirrors that."""
    vocab = 24
    target_len = 3
    rng = np.random.default_rng(seed)
    span_targets = rng.integers(1, vocab, size=(num_span_ids, target_len))
    emb_table = (rng.normal(size=(num_span_ids, z_dim)) * 3.0).astype(np.float32)
    return span_targets, emb_table, vocab


def _draw_examples(span_targets: np.ndarray, emb_table: np.ndarray, vocab: int, *,
                   n_rows: int, per_row: int, span_cap: int, mode: str, seed: int
                   ) -> SlotExamples:
    """Fabricated (row, slot) examples, bypassing the model entirely.

    mode="positive": z is the FIXED per-span-id embedding (`z = embedding of the
        target`, the spec's own example) — a BIJECTIVE map from z to the target span's
        tokens, so a decoder that actually reads z can recover the target near-exactly.
    mode="negative": z is independent Gaussian noise, uncorrelated with the target.
    """
    num_span_ids, z_dim = emb_table.shape
    rng = np.random.default_rng(seed)
    zs, span_i, span_ip1, row_ids = [], [], [], []
    for r in range(n_rows):
        for _ in range(per_row):
            sid = int(rng.integers(0, num_span_ids))
            sid_summary = int(rng.integers(0, num_span_ids))
            z = (emb_table[sid] if mode == "positive"
                 else rng.normal(size=(z_dim,)).astype(np.float32))
            zs.append(z)
            span_ip1.append(_pad(span_targets[sid], span_cap))
            span_i.append(_pad(span_targets[sid_summary], span_cap))
            row_ids.append(r)
    n = len(row_ids)
    return SlotExamples(
        z=torch.from_numpy(np.stack(zs)).float(),
        span_i=torch.from_numpy(np.stack(span_i)).long(),
        span_ip1=torch.from_numpy(np.stack(span_ip1)).long(),
        row_id=torch.tensor(row_ids, dtype=torch.long),
        n_excluded=0, n_total_valid=n,
    )


_WORLD_KW = dict(num_span_ids=8, span_cap=6, z_dim=16)
_SYN_KW = dict(n_rows=8, per_row=30, span_cap=_WORLD_KW["span_cap"])


def test_positive_control_gives_a_large_plan_minus_shuffled_gap():
    span_targets, emb_table, vocab = _synthetic_world(**_WORLD_KW, seed=100)
    fit = _draw_examples(span_targets, emb_table, vocab, mode="positive", seed=0, **_SYN_KW)
    ev = _draw_examples(span_targets, emb_table, vocab, mode="positive", seed=1, **_SYN_KW)
    res = run_four_conditions(fit, ev, vocab_size=vocab, hidden=64, steps=600, lr=1e-2,
                              batch_size=32, seed=0, device=DEVICE)
    # nats/token is a LOSS (lower = better); the probe's own "PLAN - SHUFFLED" label is computed as SHUFFLED-minus-PLAN so a positive number means "the real plan is more informative" (see plan_content_probe.py's SIGN NOTE).
    gap = res["SHUFFLED"]["nats_per_token"] - res["PLAN"]["nats_per_token"]
    assert gap > 1.0, (
        f"positive control (z bijectively encodes the target) must show a LARGE "
        f"PLAN-SHUFFLED gap; got {gap:.4f}. If this fails, the probe cannot detect a "
        f"real plan even when one is trivially present.")


def test_negative_control_gives_a_near_zero_plan_minus_shuffled_gap():
    span_targets, emb_table, vocab = _synthetic_world(**_WORLD_KW, seed=200)
    fit = _draw_examples(span_targets, emb_table, vocab, mode="negative", seed=2, **_SYN_KW)
    ev = _draw_examples(span_targets, emb_table, vocab, mode="negative", seed=3, **_SYN_KW)
    res = run_four_conditions(fit, ev, vocab_size=vocab, hidden=64, steps=600, lr=1e-2,
                              batch_size=32, seed=0, device=DEVICE)
    # nats/token is a LOSS (lower = better); the probe's own "PLAN - SHUFFLED" label is computed as SHUFFLED-minus-PLAN so a positive number means "the real plan is more informative" (see plan_content_probe.py's SIGN NOTE).
    gap = res["SHUFFLED"]["nats_per_token"] - res["PLAN"]["nats_per_token"]
    assert abs(gap) < 0.30, (
        f"negative control (z is pure noise) must show a gap near zero; got {gap:.4f}. "
        f"If this fails, the probe reports a plan on a checkpoint that has none.")


def test_run_four_conditions_rejects_mismatched_decoder_param_counts():
    """`run_four_conditions` calls `fit_decoder` exactly 4 times, in the fixed order
    PLAN/SUMMARY/SHUFFLED/POSITION. Sabotage ONLY the 4th (POSITION) call to build a
    wider decoder — a genuine cross-condition mismatch, unlike patching
    `BlindSpanDecoder` globally (which would widen all four EQUALLY and never trip the
    assert, since it only compares the four counts to EACH OTHER) — and confirm the
    parameter-count assert actually fires."""
    import lab.divergence.plan_content_probe as m

    span_targets, emb_table, vocab = _synthetic_world(**_WORLD_KW, seed=300)
    fit = _draw_examples(span_targets, emb_table, vocab, mode="negative", seed=4, **_SYN_KW)
    ev = _draw_examples(span_targets, emb_table, vocab, mode="negative", seed=5, **_SYN_KW)

    orig_fit_decoder = m.fit_decoder
    calls = {"n": 0}

    def sabotaged(z, targets, *, vocab_size, hidden, steps, lr, batch_size, seed, device,
                  weight_decay=1e-2):
        calls["n"] += 1
        if calls["n"] == 4:                     # POSITION: 4th call, dict order is fixed
            hidden = hidden * 2
        return orig_fit_decoder(z, targets, vocab_size=vocab_size, hidden=hidden,
                                steps=steps, lr=lr, batch_size=batch_size, seed=seed,
                                device=device, weight_decay=weight_decay)

    m.fit_decoder = sabotaged
    try:
        with pytest.raises(AssertionError, match="parameter count differs"):
            run_four_conditions(fit, ev, vocab_size=vocab, hidden=32, steps=5, lr=1e-3,
                                batch_size=32, seed=0, device=DEVICE)
    finally:
        m.fit_decoder = orig_fit_decoder


# ── 6. end-to-end: real (tiny) model + real TUL forward ──────────────────────

def test_capture_and_extract_against_a_real_tiny_model():
    """Not a statement about content (random init has no real plan) — a wiring check:
    the capture context manager, `reduce_prefix_values`, and `extract_slot_examples`
    must run cleanly against the SAME layout shapes a real forward produces, and the
    captured z's slot axis must line up with `layout.slot_index`'s."""
    spec = _spec()
    model = _model()
    model.requires_grad_(False)
    model.eval()
    x, y, layout, _ = _batch(spec)

    sink: list[torch.Tensor] = []
    with torch.no_grad(), capture_prefix_project(model, sink):
        out = model(x, labels=y, slot_layout=layout)
    assert len(sink) == 1, "prefix_project should be called exactly once per forward"
    values = sink[0]
    S = layout.slot_index.shape[1]
    K = layout.prefix_k
    assert values.shape[0] == x.shape[0]
    assert values.shape[1] == S * K
    z = reduce_prefix_values(values.float(), S, K)
    assert z.shape == (x.shape[0], S, K * values.shape[-1])

    ex = extract_slot_examples(x, z, layout, span_cap=spec.max_slots * 10, row_offset=0)
    assert ex.n_total_valid == int(layout.slot_valid.sum())
    assert ex.z.shape[0] <= ex.n_total_valid
    # every row contributes AT MOST (its valid slots - 1) usable examples
    per_row_valid = layout.slot_valid.sum(dim=1)
    max_usable = int((per_row_valid - 1).clamp(min=0).sum())
    assert ex.z.shape[0] == max_usable
    assert out["loss"] is not None   # forward actually completed end to end


def test_forward_still_produces_the_real_values_capture_is_transparent():
    """The capture wrapper must be a pure passthrough: the coda-facing values scattered
    into the sequence must be identical to what an unpatched forward produces (same
    seed, same inputs) — otherwise the probe would be measuring its own side effect."""
    spec = _spec()
    x, y, layout, _ = _batch(spec)

    m1 = _model()
    m1.requires_grad_(False)
    m1.eval()
    with torch.no_grad():
        out1 = m1(x, labels=y, slot_layout=layout)

    m2 = _model()   # same seed -> identical weights
    m2.requires_grad_(False)
    m2.eval()
    sink: list[torch.Tensor] = []
    with torch.no_grad(), capture_prefix_project(m2, sink):
        out2 = m2(x, labels=y, slot_layout=layout)

    assert torch.equal(out1["loss"], out2["loss"]), (
        "capture_prefix_project changed the forward's output — it must be transparent")
