"""The decode path that the A0-vs-A1 repetition table is measured through.

Two generators exist — `generate_tul` (slot layout) and `generate_plain` (no slots) — and
their whole purpose is to be differenced against each other. Three things must therefore
hold, and each of them broke at least once in the 2026-08-23 campaign:

  1. both call the SAME sampling step, so a decode difference cannot masquerade as a
     model difference;
  2. `generation_metrics` scores a sample with no slot builder, because refusing to is
     how the first version of the table came out with no baseline arm in it;
  3. rep_n honours its window, because rep_n is not comparable across lengths and the
     first table compared a 200-token row against 100-token rows.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.inference.gen_metrics import generation_metrics, ngram_stats
from morph.inference.plain_generate import generate_plain
from morph.inference.sampling import sample_next

V = 32


class _StubModel(torch.nn.Module):
    """Emits a fixed logit row; records how many forwards it saw."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.row = logits
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, ids, **kw):
        self.calls += 1
        B, L = ids.shape
        return {"logits": self.row.view(1, 1, -1).expand(B, L, -1).contiguous()}


def _logits(peak: int, gap: float = 4.0) -> torch.Tensor:
    z = torch.zeros(V)
    z[peak] = gap
    return z


# ── the shared sampling step ────────────────────────────────────────────────────────
def test_greedy_is_argmax_and_ignores_top_k():
    z = _logits(7)
    assert sample_next(z, 0.0, 0, None) == 7
    assert sample_next(z, 0.0, 50, None) == 7


def test_seeded_sampling_is_reproducible_and_seed_actually_changes_the_draw():
    z = torch.randn(V, generator=torch.Generator().manual_seed(3))
    a = [sample_next(z, 1.0, 0, torch.Generator().manual_seed(11)) for _ in range(8)]
    b = [sample_next(z, 1.0, 0, torch.Generator().manual_seed(11)) for _ in range(8)]
    assert a == b, "same seed must give the same draw"
    c = [sample_next(z, 1.0, 0, torch.Generator().manual_seed(12)) for _ in range(8)]
    assert a != c, "a different seed must give a different draw"


def test_top_k_never_returns_a_token_outside_the_top_k():
    # A tautological version of this test (assert the result is an int) would pass on a
    # broken mask, so assert membership in the actual top-k set.
    z = torch.randn(V, generator=torch.Generator().manual_seed(5))
    k = 3
    allowed = set(torch.topk(z, k).indices.tolist())
    g = torch.Generator().manual_seed(0)
    got = {sample_next(z, 1.0, k, g) for _ in range(200)}
    assert got <= allowed, f"top_k={k} leaked {got - allowed}"
    assert len(got) > 1, "200 draws collapsing to one token means the mask ate everything"


def test_temperature_widens_the_draw():
    z = _logits(7, gap=3.0)
    cold = {sample_next(z, 0.05, 0, torch.Generator().manual_seed(i)) for i in range(60)}
    hot = {sample_next(z, 2.0, 0, torch.Generator().manual_seed(i)) for i in range(60)}
    assert cold == {7}
    assert len(hot) > 5, f"t=2.0 produced only {len(hot)} distinct tokens"


# ── both generators go through it ───────────────────────────────────────────────────
def test_plain_generator_uses_the_shared_sampler(monkeypatch):
    seen = []

    def spy(logits, temperature, top_k, generator):
        seen.append((float(temperature), int(top_k)))
        return 3

    monkeypatch.setattr("morph.inference.plain_generate.sample_next", spy)
    out = generate_plain(_StubModel(_logits(7)), [1, 2], max_new_tokens=5,
                         temperature=0.8, top_k=50, seed=0, device=torch.device("cpu"))
    assert out == [3] * 5
    assert seen == [(0.8, 50)] * 5


def test_tul_generator_uses_the_shared_sampler(monkeypatch):
    from morph.model.tul_layout import BoundaryRule, TulLayoutSpec
    from morph.inference.tul_generate import generate_tul

    seen = []

    def spy(logits, temperature, top_k, generator):
        seen.append((float(temperature), int(top_k)))
        return 3

    monkeypatch.setattr("morph.inference.tul_generate.sample_next", spy)
    lut = np.zeros(V, dtype=bool)
    lut[9] = True
    rule = BoundaryRule(is_boundary=lut, min_span=2, span_cap=8, eos_id=0)
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=8, slot_id=4)
    out, _b = generate_tul(_StubModel(_logits(7)), [1, 2], rule, spec,
                           max_new_tokens=5, temperature=0.8, top_k=50, seed=0,
                           device=torch.device("cpu"))
    assert out == [3] * 5
    assert seen == [(0.8, 50)] * 5


def test_plain_generator_recomputes_the_whole_row_every_step():
    # The TUL generator has no KV cache by design (spec §6 v1). If the baseline arm ever
    # gains one, the two arms stop being decoded by the same procedure and the rep4
    # difference between them becomes uninterpretable.
    m = _StubModel(_logits(7))
    generate_plain(m, [1, 2], max_new_tokens=6, temperature=0.0, top_k=0,
                   device=torch.device("cpu"))
    assert m.calls == 6


# ── metrics ─────────────────────────────────────────────────────────────────────────
def test_generation_metrics_scores_a_sample_with_no_slot_builder():
    m = generation_metrics(list(range(50)))
    assert m["rep4"] == 0.0 and m["distinct3"] == 1.0
    assert "n_spans" not in m, "a non-TUL arm has no span geometry to report"
    looped = generation_metrics([1, 2, 3, 4] * 12)
    assert looped["rep4"] > 0.9, f"a pure loop must score near 1, got {looped['rep4']}"


def test_rep4_honours_the_window_so_lengths_are_comparable():
    # Clean prefix then a hard loop: scoring the first 64 tokens must NOT see the loop.
    ids = list(range(64)) + [1, 2, 3, 4] * 16
    assert ngram_stats(ids, 4, window=64)[0] == 0.0
    assert ngram_stats(ids, 4, window=128)[0] > 0.3
    assert generation_metrics(ids, window=64)["rep4"] == 0.0
    assert generation_metrics(ids, window=128)["rep4"] > 0.3


def test_rep4_of_the_same_text_grows_with_the_window():
    # This is the property that made the 128-token table meaningless: rep_n is a function
    # of length, so an arm scored at 200 tokens cannot be ranked against one at 100.
    # Small alphabet on purpose: with 60 symbols a 1024-token row has ~0 colliding
    # 4-grams (60^4 = 13 M), which measures the alphabet, not the length effect.
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 8, size=1024).tolist()
    r_short = ngram_stats(ids, 4, window=128)[0]
    r_long = ngram_stats(ids, 4, window=1024)[0]
    assert r_long > r_short


# ── batched generation must equal the single-row path ───────────────────────────────
class _CausalSumModel(torch.nn.Module):
    """Logits at position t depend on the cumulative sum of ids[:t+1].

    Content-dependent on purpose. A stub that ignores its input would pass a batched-vs-
    single parity test with a cursor bug still in place, which makes the test theatre.
    Here, reading a row's logits at the batch's max length instead of at the row's own
    cursor changes the answer, and so does letting padding into the prefix.
    """

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids, **kw):
        B, L = ids.shape
        cs = ids.cumsum(dim=1)                      # [B, L] causal by construction
        peak = (cs * 7 + torch.arange(L, device=ids.device)) % V
        logits = torch.zeros(B, L, V, device=ids.device)
        logits.scatter_(2, peak.unsqueeze(-1), 5.0)
        return {"logits": logits}


def test_batched_plain_generation_equals_the_single_row_path():
    from morph.inference.plain_generate import generate_plain_batch

    m = _CausalSumModel()
    dev = torch.device("cpu")
    # RAGGED on purpose: different prompt lengths is the case the cursor logic exists for.
    prompts = [[1, 2, 3], [5], [7, 8], [2, 2, 2, 2]]
    batched = generate_plain_batch(m, prompts, max_new_tokens=12, temperature=0.0,
                                   top_k=0, device=dev)
    single = [generate_plain(m, p, max_new_tokens=12, temperature=0.0, top_k=0,
                             device=dev) for p in prompts]
    assert batched == single, f"batched != single\n  {batched}\n  {single}"


def test_batched_plain_generation_is_not_trivially_constant():
    # Guard on the guard: if the stub emitted the same token everywhere, the parity test
    # above would pass on any implementation at all.
    from morph.inference.plain_generate import generate_plain_batch

    out = generate_plain_batch(_CausalSumModel(), [[1, 2, 3], [5]], max_new_tokens=12,
                               temperature=0.0, top_k=0, device=torch.device("cpu"))
    assert len(set(out[0])) > 2, f"row 0 is nearly constant: {out[0]}"
    assert out[0] != out[1], "two different prompts produced identical continuations"


def test_batched_tul_generation_equals_the_single_row_path():
    from morph.model.tul_layout import BoundaryRule, TulLayoutSpec
    from morph.inference.tul_generate import generate_tul, generate_tul_batch

    lut = np.zeros(V, dtype=bool)
    lut[9] = True
    lut[11] = True
    rule = BoundaryRule(is_boundary=lut, min_span=2, span_cap=6, eos_id=0)
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=16, slot_id=4)
    m = _CausalSumModel()
    dev = torch.device("cpu")
    prompts = [[1, 2, 3], [5], [7, 8], [2, 2, 2, 2]]
    batched, blds = generate_tul_batch(m, prompts, rule, spec, max_new_tokens=12,
                                       temperature=0.0, top_k=0, device=dev)
    single = [generate_tul(m, p, rule, spec, max_new_tokens=12, temperature=0.0,
                           top_k=0, device=dev)[0] for p in prompts]
    assert batched == single, f"batched != single\n  {batched}\n  {single}"
    # And the rows really did diverge in length -- otherwise the ragged path is untested.
    assert len({len(b.ids) for b in blds}) > 1, "no length divergence; test has no teeth"


def test_batched_generators_reject_a_seed_list_of_the_wrong_length():
    from morph.inference.plain_generate import generate_plain_batch

    with pytest.raises(ValueError, match="one entry per row"):
        generate_plain_batch(_CausalSumModel(), [[1], [2]], max_new_tokens=2,
                             temperature=1.0, seeds=[0], device=torch.device("cpu"))


# ── emit_source: where a span's FIRST token is read from ────────────────────────────
class _PosStub(torch.nn.Module):
    """Greedy argmax at position l returns (l + 10) % V — the read position, made visible."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids, **kw):
        B, L = ids.shape
        z = torch.zeros(B, L, V)
        pos = (torch.arange(L) + 10) % V
        z[:, torch.arange(L), pos] = 4.0
        return {"logits": z}


def _boundary_setup():
    from morph.model.tul_layout import BoundaryRule, TulLayoutSpec
    lut = np.zeros(V, dtype=bool)
    lut[9] = True
    rule = BoundaryRule(is_boundary=lut, min_span=2, span_cap=8, eos_id=0)
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=8, slot_id=4)
    return rule, spec


def test_emit_source_token_reads_the_boundary_token_position():
    # Prompt [1,2,9] cuts after 9 → row is [1,2,9,slot,slot] (K=2). The slot emit
    # position is index 4, the boundary TOKEN position is index 2. emit_weight=0 arms
    # never train index 4 (lab/divergence/emit_space_probe.py: 0.47–0.58 space mass vs
    # 0.81 at the token position), so "token" must read index 2.
    from morph.inference.tul_generate import generate_tul
    rule, spec = _boundary_setup()
    out_tok, b_tok = generate_tul(_PosStub(), [1, 2, 9], rule, spec, max_new_tokens=2,
                                  temperature=0.0, top_k=0, device=torch.device("cpu"),
                                  emit_source="token")
    out_slot, b_slot = generate_tul(_PosStub(), [1, 2, 9], rule, spec, max_new_tokens=2,
                                    temperature=0.0, top_k=0, device=torch.device("cpu"),
                                    emit_source="slot")
    assert b_tok.n_slots == 1 and b_slot.n_slots == 1, "prompt must have cut one slot"
    assert out_tok[0] == 12, "token mode: read index 2 (boundary token) → (2+10)%V"
    assert out_slot[0] == 14, "slot mode (spec §6 v1, emit_weight>0 arms): read index 4"
    # Mid-span steps are identical in both modes: next read is the row's last position.
    assert out_tok[1] == out_slot[1] == 15


def test_emit_source_default_is_token_and_bad_value_raises():
    """base.yaml ships ``emit_weight: 0.0``: the slot's emit position is never trained,
    so the generator's default MUST read the span's first token from the boundary TOKEN
    position. A silent flip back to "slot" would sample from an untrained head."""
    from morph.inference.tul_generate import generate_tul
    rule, spec = _boundary_setup()
    out_default, _ = generate_tul(_PosStub(), [1, 2, 9], rule, spec, max_new_tokens=1,
                                  temperature=0.0, top_k=0, device=torch.device("cpu"))
    assert out_default == [12], "default must be the token position (emit_weight=0 recipe)"
    with pytest.raises(ValueError, match="emit_source"):
        generate_tul(_PosStub(), [1, 2, 9], rule, spec, max_new_tokens=1,
                     temperature=0.0, top_k=0, device=torch.device("cpu"),
                     emit_source="tok")


def test_emit_source_token_batch_matches_single():
    from morph.inference.tul_generate import generate_tul, generate_tul_batch
    rule, spec = _boundary_setup()
    prompts = [[1, 2, 9], [1, 2], [5, 6, 9]]
    batched, _ = generate_tul_batch(_PosStub(), prompts, max_new_tokens=6,
                                    rule=rule, spec=spec, temperature=0.0, top_k=0,
                                    device=torch.device("cpu"), emit_source="token")
    single = [generate_tul(_PosStub(), p, rule, spec, max_new_tokens=6, temperature=0.0,
                           top_k=0, device=torch.device("cpu"), emit_source="token")[0]
              for p in prompts]
    assert batched == single
