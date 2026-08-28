"""TG restriction (docs/tul-tg-spec.md) — spec §7 tests T1, T2, T3, plus the kernels
raise (spec §2/§6) and a state-dict-keys-unchanged check for ``tg_restrict=false``.

CPU only, eager (``use_kernels=False``), tiny config, no tokenizer — the same
conventions as ``test_tul_forward.py`` / ``test_tul_gate.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import morph.model.attention as attention_mod
import morph.model.transformer as transformer_mod
from morph.model.gla import GatedLinearAttention
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import (
    BoundaryRule,
    SlotLayout,
    TulLayoutSpec,
    slot_layout_from_ids,
    tg_allow_mask,
    tg_reset_mask,
)

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=2, max_depth=3, bptt_depth=2,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _rule(min_span: int = 4, span_cap: int = 32) -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=min_span, span_cap=span_cap, eos_id=0)


def _spec(**kw) -> TulLayoutSpec:
    base = dict(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
    base.update(kw)
    return TulLayoutSpec(**base)


def _batch(spec, rule, B=2, n=200, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == spec.slot_id] = 5
    ids[:, ::10] = DOT                    # a boundary every 10 tokens -> ~10-token spans
    return slot_layout_from_ids(ids.astype(np.int64), rule, spec)


def _model(tul: TULConfig | None, seed=1234, **cfg_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(tul=tul, **cfg_kw))


# ── T1: tg_allow_mask against a brute-force reference ────────────────────────

def _brute_force_allow(layout: SlotLayout, soft_prev_span: bool) -> np.ndarray:
    bag_id = layout.bag_id.numpy()
    slot_mask = layout.slot_mask.numpy()
    max_slots = layout.max_slots
    B, L = bag_id.shape
    out = np.zeros((B, L, L), dtype=bool)
    for b in range(B):
        for i in range(L):
            for j in range(L):
                if j > i:
                    continue
                allow = bool(bag_id[b, i] == bag_id[b, j]) or bool(slot_mask[b, j])
                if soft_prev_span and bag_id[b, i] < max_slots:
                    allow = allow or bool(bag_id[b, i] == bag_id[b, j] + 1)
                out[b, i, j] = allow
    return out


def _hand_layout() -> SlotLayout:
    # B=1, L=8: span0 = tok{0,1} + slot0@2; span1 = tok{3,4} + slot1@5; tail tok{6,7}
    # (dump bin, id == max_slots == 2).
    bag_id = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]], dtype=torch.long)
    slot_mask = torch.tensor([[False, False, True, False, False, True, False, False]])
    slot_index = torch.tensor([[2, 5]], dtype=torch.long)
    slot_valid = torch.tensor([[True, True]])
    return SlotLayout(slot_mask=slot_mask, bag_id=bag_id, slot_index=slot_index,
                      slot_valid=slot_valid, prefix_k=1)


@pytest.mark.parametrize("soft_prev_span", [False, True])
def test_tg_allow_mask_hand_written_layout(soft_prev_span):
    layout = _hand_layout()
    got = tg_allow_mask(layout, soft_prev_span=soft_prev_span).squeeze(1).numpy()
    want = _brute_force_allow(layout, soft_prev_span)
    assert np.array_equal(got, want), f"soft_prev_span={soft_prev_span}"


@pytest.mark.parametrize("soft_prev_span", [False, True])
def test_tg_allow_mask_random_packed_batch(soft_prev_span):
    spec = _spec()
    rule = _rule()
    _x, _y, layout, _stats = _batch(spec, rule)
    got = tg_allow_mask(layout, soft_prev_span=soft_prev_span).squeeze(1).numpy()
    want = _brute_force_allow(layout, soft_prev_span)
    assert np.array_equal(got, want), f"soft_prev_span={soft_prev_span}"


def test_tg_allow_mask_soft_prev_span_excludes_dump_bin_tail():
    """Documented judgment call: the extra disjunct is gated on the QUERY not being
    in the tail dump bin, so a tail position never gains visibility into the last
    real span through the soft term alone."""
    layout = _hand_layout()
    hard = tg_allow_mask(layout, soft_prev_span=False).squeeze(1)[0]
    soft = tg_allow_mask(layout, soft_prev_span=True).squeeze(1)[0]
    # Query positions 6,7 are the dump bin (bag_id == max_slots == 2); their allowed
    # key set must be UNCHANGED by soft_prev_span (no grant into span 1, bag_id 1).
    assert torch.equal(hard[6], soft[6])
    assert torch.equal(hard[7], soft[7])
    # A real query DOES gain the extra term: position 3/4 (span 1, bag_id 1) may now
    # also see span 0 (bag_id 0 == 1 - 1) through the soft disjunct.
    assert soft[3, 0] and not hard[3, 0]


def test_tg_reset_mask_matches_bag_id_transitions():
    layout = _hand_layout()
    got = tg_reset_mask(layout)[0]
    want = torch.tensor([True, False, False, True, False, False, True, False])
    assert torch.equal(got, want)


# ── T2: GLA reset (segmented scan) vs the per-segment recurrent oracle ───────
#
# The oracle in every test below is the definition of a reset: run the recurrent
# reference INDEPENDENTLY on each segment from a true zero initial state and
# concatenate. Both production paths must match it — `_recurrent(reset_mask=…)`
# exactly (its reset IS a multiply-by-zero), `_chunked(reset_mask=…)` to 1e-5
# (spec T2's gate; fp32 log-space offsets round, exactness is not claimed).


def _per_segment_oracle(gla, q, k, v, log_alpha, reset_positions, S):
    bounds = sorted(set(reset_positions) | {0}) + [S]
    oracle = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        o_seg, _ = gla._recurrent(q[:, s:e], k[:, s:e], v[:, s:e], log_alpha[:, s:e], None)
        oracle.append(o_seg)
    return torch.cat(oracle, dim=1)


def test_gla_recurrent_reset_matches_the_oracle_exactly():
    torch.manual_seed(0)
    B, S, d, H = 2, 20, 16, 2
    gla = GatedLinearAttention(d, H, mode="recurrent")
    x = torch.randn(B, S, d)
    reset_positions = [0, 5, 13]
    reset_mask = torch.zeros(B, S, dtype=torch.bool)
    reset_mask[:, reset_positions] = True

    q, k, v, log_alpha, _r_pre = gla._project(x)
    out_reset, _ = gla._recurrent(q, k, v, log_alpha, None, reset_mask=reset_mask)
    oracle_out = _per_segment_oracle(gla, q, k, v, log_alpha, reset_positions, S)
    assert torch.equal(out_reset, oracle_out)


def test_gla_chunked_reset_matches_recurrent_oracle_per_segment():
    """The regime that BROKE the first (log_alpha-floor) formulation: several resets
    per chunk at span-density spacing, where the chunk-global cumsum dived past the
    -30 overflow clamp and pinned the relative decay (~780% rel err measured). The
    segmented scan must hold 1e-5 here."""
    torch.manual_seed(0)
    B, S, d, H = 2, 96, 16, 2
    gla = GatedLinearAttention(d, H, mode="chunked", chunk=32)
    x = torch.randn(B, S, d)
    reset_positions = list(range(0, S, 7))          # ~4-5 resets per 32-chunk
    reset_mask = torch.zeros(B, S, dtype=torch.bool)
    reset_mask[:, reset_positions] = True

    q, k, v, log_alpha, _r_pre = gla._project(x)
    out_chunked, state_chunked = gla._chunked(q, k, v, log_alpha, None,
                                              reset_mask=reset_mask)
    oracle_out = _per_segment_oracle(gla, q, k, v, log_alpha, reset_positions, S)

    rel_err = (out_chunked - oracle_out).norm() / oracle_out.norm().clamp(min=1e-12)
    assert rel_err.item() <= 1e-5, f"rel err {rel_err.item()}"

    # The state leaving the scan is the LAST segment's alone.
    last = max(p for p in reset_positions if p < S)
    _, state_oracle = gla._recurrent(q[:, last:], k[:, last:], v[:, last:],
                                     log_alpha[:, last:], None)
    srel = (state_chunked - state_oracle).norm() / state_oracle.norm().clamp(min=1e-12)
    assert srel.item() <= 1e-5, f"state rel err {srel.item()}"


def test_gla_chunked_reset_misaligned_with_chunk_boundaries():
    """Resets that never land on a chunk edge, including a chunk with NO reset (the
    carry segment must then flow through untouched) and one with a reset at its
    first position (the carried state must be fully dropped)."""
    torch.manual_seed(1)
    B, S, d, H = 2, 40, 16, 2
    gla = GatedLinearAttention(d, H, mode="chunked", chunk=8)
    x = torch.randn(B, S, d)
    reset_positions = [0, 3, 16, 29]                 # chunk 2 (16) IS an edge; 3, 29 are not
    reset_mask = torch.zeros(B, S, dtype=torch.bool)
    reset_mask[:, reset_positions] = True

    q, k, v, log_alpha, _r_pre = gla._project(x)
    out_chunked, _ = gla._chunked(q, k, v, log_alpha, None, reset_mask=reset_mask)
    oracle_out = _per_segment_oracle(gla, q, k, v, log_alpha, reset_positions, S)
    rel_err = (out_chunked - oracle_out).norm() / oracle_out.norm().clamp(min=1e-12)
    assert rel_err.item() <= 1e-5, f"rel err {rel_err.item()}"


def test_gla_chunked_all_false_reset_mask_matches_none_path():
    """An all-False reset_mask routes through the segmented code with one segment per
    chunk — it must reproduce the reset_mask=None production path to fp32 noise."""
    torch.manual_seed(2)
    B, S, d, H = 2, 33, 16, 2
    gla = GatedLinearAttention(d, H, mode="chunked", chunk=8)
    x = torch.randn(B, S, d)
    q, k, v, log_alpha, _r_pre = gla._project(x)
    out_none, state_none = gla._chunked(q, k, v, log_alpha, None)
    out_false, state_false = gla._chunked(q, k, v, log_alpha, None,
                                          reset_mask=torch.zeros(B, S, dtype=torch.bool))
    assert torch.allclose(out_none, out_false, atol=1e-6, rtol=1e-6)
    assert torch.allclose(state_none, state_false, atol=1e-6, rtol=1e-6)


def test_gla_forward_reset_is_finite_and_kernel_mode_raises():
    torch.manual_seed(0)
    B, S, d, H = 1, 12, 16, 2
    gla = GatedLinearAttention(d, H, mode="chunked", chunk=4)
    x = torch.randn(B, S, d)
    reset_mask = torch.zeros(B, S, dtype=torch.bool)
    reset_mask[:, [0, 4, 8]] = True
    out, _ = gla(x, reset_mask=reset_mask)
    assert torch.isfinite(out).all()

    gla_k = GatedLinearAttention(d, H, mode="kernel", chunk=4)
    with pytest.raises(NotImplementedError, match="reset_mask"):
        gla_k(x, reset_mask=reset_mask)


# ── T6 (spec §2/§6): tg_restrict + use_kernels=true raises at construction ────

def test_tg_restrict_with_kernels_raises():
    with pytest.raises(ValueError, match="use_kernels"):
        _model(TULConfig(prefix_k=2, slot_id=4, tg_restrict=True), use_kernels=True)


def test_tg_soft_prev_span_without_tg_restrict_raises():
    with pytest.raises(ValueError, match="tg_restrict"):
        TULConfig(prefix_k=2, slot_id=4, tg_soft_prev_span=True)


# ── tg_restrict=false: construction is unchanged (state-dict keys, submodules) ──

def test_tg_restrict_false_builds_the_same_submodules():
    m = _model(TULConfig(prefix_k=2, slot_id=4, tg_restrict=False))
    csa = m.prelude[0].attention._impl               # layer 0 -> even -> CSA
    hca = m.prelude[1].attention._impl                # layer 1 -> odd -> HCA
    assert csa.compressor is not None and csa.comp_norm is not None and csa.indexer is not None
    assert hca.compressor is not None and hca.comp_norm is not None


def test_tg_restrict_false_state_dict_keys_unchanged():
    m_default = _model(TULConfig(prefix_k=2, slot_id=4))
    m_explicit = _model(TULConfig(prefix_k=2, slot_id=4, tg_restrict=False))
    assert set(m_default.state_dict().keys()) == set(m_explicit.state_dict().keys())


def test_tg_restrict_true_does_not_build_compressor():
    m = _model(TULConfig(prefix_k=2, slot_id=4, tg_restrict=True), use_kernels=False)
    csa = m.prelude[0].attention._impl
    hca = m.prelude[1].attention._impl
    assert csa.compressor is None and csa.comp_norm is None and csa.indexer is None
    assert hca.compressor is None and hca.comp_norm is None


# ── T3: the severed-channel leak probe (the falsifier) ───────────────────────

def _find_probe_positions(layout: SlotLayout):
    """u: deepest token of span 0 that is >= 4 positions before span 0 ends.
    t: the first token of span 2. Returns (u, t) row indices, or (None, None) if
    the row does not have three well-separated spans."""
    bag_id = layout.bag_id[0]
    slot_mask = layout.slot_mask[0]
    tok0 = ((bag_id == 0) & ~slot_mask).nonzero(as_tuple=True)[0]
    tok2 = ((bag_id == 2) & ~slot_mask).nonzero(as_tuple=True)[0]
    if tok0.numel() == 0 or tok2.numel() == 0:
        return None, None
    span0_end = int(tok0.max())
    candidates = tok0[tok0 <= span0_end - 4]
    if candidates.numel() == 0:
        return None, None
    return int(candidates.min()), int(tok2.min())


def _severed_forward(model: MORPHTransformer, x: torch.Tensor, layout: SlotLayout):
    """Runs one TUL forward with BOTH tg channels severed: the window branch's
    extra_mask has every slot column zeroed (only same-span survives), and the
    compressed branch is force-replaced with a graph-disconnected constant zero —
    the test-only hook spec T3 asks for. Returns the embedding-table's OUTPUT
    tensor (grad-tracked) and the logits.
    """
    orig_allow = transformer_mod.tg_allow_mask

    def severed_allow(layout, soft_prev_span=False):
        allow = orig_allow(layout, soft_prev_span=soft_prev_span)      # [B,1,L,L]
        sm = layout.slot_mask.view(layout.slot_mask.shape[0], 1, 1, -1)
        return allow & ~sm                                             # drop slot columns

    def severed_comp(q, k, v, slot_mask, sink_logits, scale):
        # Graph-disconnected constant: torch.zeros_like carries no grad_fn back to
        # q/k/v, so this branch contributes EXACTLY zero gradient, not just a
        # numerically-small one.
        return torch.zeros_like(v)

    captured = {}

    def _embed_hook(module, inp, out):
        out.retain_grad()
        captured["embed_out"] = out

    handle = model.embed.register_forward_hook(_embed_hook)
    orig_comp = attention_mod._tg_slot_attention
    transformer_mod.tg_allow_mask = severed_allow
    attention_mod._tg_slot_attention = severed_comp
    try:
        out = model(x, labels=None, slot_layout=layout)
    finally:
        transformer_mod.tg_allow_mask = orig_allow
        attention_mod._tg_slot_attention = orig_comp
        handle.remove()
    return captured["embed_out"], out["logits"]


def _finite_logit_sum(logits: torch.Tensor, t: int, slot_id: int) -> torch.Tensor:
    """logits[0, t, :] minus the structurally -inf slot_id column (spec §3.1's
    masked-at-generation head) — avoids summing a -inf into the backward target."""
    vocab_mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    vocab_mask[slot_id] = False
    return logits[0, t, vocab_mask].sum()


def test_severed_slot_channel_gives_exactly_zero_grad():
    spec = _spec()
    rule = _rule()
    x, _y, layout, _stats = _batch(spec, rule, B=1, n=200, seed=3)
    u, t = _find_probe_positions(layout)
    assert u is not None, "test layout did not produce 3 well-separated spans"

    m = _model(TULConfig(prefix_k=spec.prefix_k, slot_id=spec.slot_id, tg_restrict=True),
              seed=7, use_kernels=False, retention=False)
    m.eval()
    embed_out, logits = _severed_forward(m, x, layout)
    _finite_logit_sum(logits, t, spec.slot_id).backward()
    g = embed_out.grad[0, u, :]
    assert torch.all(g == 0), f"grad at u={u} from t={t} is not exactly zero: {g}"


def test_probe_detects_the_leak_on_an_unrestricted_model():
    """Same probe, tg_restrict=false, no hooks: proves the probe is not vacuous —
    it DOES find gradient when the channel is not severed (spec T3's second half)."""
    spec = _spec()
    rule = _rule()
    x, _y, layout, _stats = _batch(spec, rule, B=1, n=200, seed=3)
    u, t = _find_probe_positions(layout)
    assert u is not None, "test layout did not produce 3 well-separated spans"

    m = _model(TULConfig(prefix_k=spec.prefix_k, slot_id=spec.slot_id, tg_restrict=False),
              seed=7, use_kernels=False, retention=False)
    m.eval()
    captured = {}

    def _embed_hook(module, inp, out):
        out.retain_grad()
        captured["embed_out"] = out

    handle = m.embed.register_forward_hook(_embed_hook)
    out = m(x, labels=None, slot_layout=layout)
    handle.remove()
    _finite_logit_sum(out["logits"], t, spec.slot_id).backward()
    g = captured["embed_out"].grad[0, u, :]
    assert torch.any(g != 0), f"probe found NO gradient on the unrestricted model (u={u}, t={t})"


# ── the SHIPPED forward under soft_prev_span (arm TG3) ────────────────────────────
#
# Everything above verifies `tg_allow_mask` against a brute-force reference for BOTH
# soft_prev_span values, and verifies the config validation. Nothing above ever RUNS a
# forward with soft_prev_span=True. That is the gap this campaign has been bitten by
# before — a mask function can be perfectly correct while the forward that consumes it
# has a shape or dtype bug, and every arm runs the shipped path, not the helper.
# TG3 is queued to train, so its forward gets executed here first.

def test_soft_prev_span_forward_and_backward_run():
    """One real forward+backward on the SHIPPED path with soft_prev_span=True."""
    spec = _spec()
    rule = _rule()
    x, y, layout, _stats = _batch(spec, rule, B=2, n=200, seed=5)
    m = _model(TULConfig(prefix_k=2, slot_id=4, tg_restrict=True, tg_soft_prev_span=True))
    out = m(x, labels=y, slot_layout=layout)
    loss = out["loss"]
    assert torch.isfinite(loss), f"non-finite loss under soft_prev_span: {loss}"
    loss.backward()
    grads = [(n, p.grad) for n, p in m.named_parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for _n, g in grads), \
        "non-finite gradient under soft_prev_span"


def test_soft_prev_span_forward_differs_from_the_hard_restriction():
    """The softening must CHANGE the output. Identical logits would mean the flag never
    reached the attention mask, and the arm would silently be a duplicate of TG1 — a
    whole training run spent re-measuring an arm we already have."""
    spec = _spec()
    rule = _rule()
    x, y, layout, _stats = _batch(spec, rule, B=2, n=200, seed=5)
    hard = _model(TULConfig(prefix_k=2, slot_id=4, tg_restrict=True))
    soft = _model(TULConfig(prefix_k=2, slot_id=4, tg_restrict=True, tg_soft_prev_span=True))
    soft.load_state_dict(hard.state_dict())         # identical weights; only the mask differs
    hard.eval()
    soft.eval()
    with torch.no_grad():
        # labels=None: with labels the fused/chunked CE host returns the loss and leaves
        # "logits" None, so this comparison needs the logits-returning path.
        lh = hard(x, labels=None, slot_layout=layout)["logits"]
        ls = soft(x, labels=None, slot_layout=layout)["logits"]
    assert lh is not None and ls is not None, "no logits returned — check the labels path"
    assert not torch.allclose(lh, ls), \
        "soft_prev_span produced identical logits — the flag never reached the mask"
