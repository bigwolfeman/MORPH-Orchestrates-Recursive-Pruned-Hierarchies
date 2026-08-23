"""docs/runtime-invariants.md §5 — the causality contract, gated in THIS repo.

    "No module may pool statistics across the sequence axis. Every position's output
     must depend only on positions <= t."

The doc names the gate ("corrupt tokens after position k, assert logits at <= k
unchanged") and points at a test that lives in the Olympiad tree. Nothing in this
repository ran it, and the invariant is currently BROKEN here: `model.retention_carry`
(true in base.yaml and in every TUL arm) carries the GLA state from the END of core
iteration t into position 0 of iteration t+1. That state summarises the whole sequence,
so from the second core iteration onward every position sees the future.

Measured on `tul-a0-acap1/step_20000` with `ignore/perf/future_corruption_probe.py`:
corrupting tokens after k moves the logits at positions <= k by up to 4.08 against a
mean |logit| of 2.41 with the carry ON, and by exactly 0.000 with it OFF. The
teacher-forced val CE of that checkpoint is 3.2952 with the carry and 3.4385 without it
(20 val batches, 81920 tokens): the lookahead is worth 0.1433 nats, which is larger than
the entire TUL-gate result it was used to measure.

See `.agents/notes/proposed/bug-fix/2026-08-23-retention-carry-breaks-causality.md`.
"""

from __future__ import annotations

import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer

V = 64
S = 32
K = 12


def _model(**kw) -> MORPHTransformer:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128,
        context_len=128, n_prelude=1, n_core=2, n_coda=1, mean_depth=3, max_depth=3,
        bptt_depth=2, channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16, bigram_hash_vocab=V,
        use_kernels=False, hc_use_kernel=False, dropout=0.0,
        retention=True, retention_layers=(1,), retention_chunk=8,
    )
    base.update(kw)
    torch.manual_seed(0)
    return MORPHTransformer(MORPHConfig(**base)).eval()


def _future_corruption_delta(model) -> float:
    """max |Δ logit| over positions ≤ K when every token after K is replaced."""
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(5, V, (1, S), generator=g)
    bad = ids.clone()
    bad[0, K + 1:] = torch.randint(5, V, (S - K - 1,), generator=g)
    with torch.no_grad():
        a = model(ids)["logits"][0, : K + 1].float()
        b = model(bad)["logits"][0, : K + 1].float()
    return (a - b).abs().max().item()


def test_the_probe_itself_is_deterministic():
    # Without this, a delta of 0.000 could mean "causal" or "the two runs were the same
    # tensor", and a non-zero delta could be eval-mode randomness rather than lookahead.
    m = _model(retention_carry=False)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(5, V, (1, S), generator=g)
    with torch.no_grad():
        a = m(ids)["logits"].float()
        b = m(ids)["logits"].float()
    assert torch.equal(a, b), "forward is not deterministic in eval; the gate is unusable"
    assert a.abs().max() > 0, "logits are all zero; the probe would pass on anything"


def test_causality_holds_when_the_retention_carry_is_off():
    """The invariant, on the configuration that satisfies it."""
    d = _future_corruption_delta(_model(retention_carry=False))
    assert d == 0.0, f"future corruption moved earlier logits by {d:.3e} with carry OFF"


def test_causality_holds_when_there_is_no_retention_branch_at_all():
    d = _future_corruption_delta(_model(retention=False))
    assert d == 0.0, f"future corruption moved earlier logits by {d:.3e} with no retention"


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN DEFECT, not yet fixed: retention_carry=true (base.yaml and every TUL arm) "
    "carries the whole-sequence GLA state across core-loop iterations, so from iteration "
    "2 onward every position sees the future. Recorded rather than hidden. When the fix "
    "lands, delete the xfail -- this test then guards it."))
def test_causality_holds_when_the_retention_carry_is_on():
    d = _future_corruption_delta(_model(retention_carry=True))
    assert d == 0.0, f"future corruption moved earlier logits by {d:.3e} with carry ON"


def test_the_carry_is_what_breaks_it_and_the_break_is_material():
    """Pin the CAUSE, and pin that it is not a rounding effect.

    A test that only said "carry on differs from carry off" would pass on a 1e-7
    numerical wobble. The measured effect on a trained checkpoint is 4.08 against a mean
    |logit| of 2.41, so require the tiny model's violation to be well clear of noise too.
    """
    on = _future_corruption_delta(_model(retention_carry=True))
    off = _future_corruption_delta(_model(retention_carry=False))
    assert off == 0.0
    assert on > 1e-4, (
        f"expected a material causality violation with the carry on, got {on:.3e}. "
        "If this is now ~0 the defect may be fixed: check the xfail above.")
