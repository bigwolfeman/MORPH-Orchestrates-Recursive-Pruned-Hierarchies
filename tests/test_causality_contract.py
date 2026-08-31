"""docs/runtime-invariants.md §5 — the causality contract, gated in THIS repo.

    "No module may pool statistics across the sequence axis. Every position's output
     must depend only on positions <= t."

The doc names the gate ("corrupt tokens after position k, assert logits at <= k
unchanged"). The invariant was BROKEN here until 2026-08-31: `retention_carry`
(then a bool, true in base.yaml and every TUL arm) carried the GLA state from the END
of core iteration t into position 0 of iteration t+1. That state summarises the whole
sequence, so from the second core iteration onward every position saw the future.
Since the fix the default is `retention_carry="none"` (state reset each iteration,
strictly causal); the old behaviour survives ONLY as the explicit opt-in
`"acausal_final"`, for loading checkpoints trained before the fix.

Measured on `tul-a0-acap1/step_20000` with `ignore/perf/future_corruption_probe.py`:
corrupting tokens after k moves the logits at positions <= k by up to 4.08 against a
mean |logit| of 2.41 with the carry ON, and by exactly 0.000 with it OFF. The
teacher-forced val CE of that checkpoint is 3.2952 with the carry and 3.4385 without it
(20 val batches, 81920 tokens): the lookahead is worth 0.1433 nats, which is larger than
the entire TUL-gate result it was used to measure.

See `.agents/notes/implemented/bug-fix/2026-08-23-retention-carry-breaks-causality.md`
and the audit that forced the fix,
`lab/experiments/successes/2026-08-31-carry-leak-audit.md` (the leak is LEARNED: 0.14
nats on a truncated-BPTT arm, 3.85 nats after 30k full-BPTT steps).
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
    m = _model(retention_carry="none")
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(5, V, (1, S), generator=g)
    with torch.no_grad():
        a = m(ids)["logits"].float()
        b = m(ids)["logits"].float()
    assert torch.equal(a, b), "forward is not deterministic in eval; the gate is unusable"
    assert a.abs().max() > 0, "logits are all zero; the probe would pass on anything"


def test_causality_holds_on_the_default_config():
    """THE GUARD: the default (no retention_carry passed) must be causal.

    If someone re-defaults the config to the acausal carry, this fails."""
    d = _future_corruption_delta(_model())
    assert d == 0.0, f"future corruption moved earlier logits by {d:.3e} on the DEFAULT config"


def test_causality_holds_when_the_retention_carry_is_off():
    """Explicit "none", and the bool-False back-compat spelling."""
    d = _future_corruption_delta(_model(retention_carry="none"))
    assert d == 0.0, f"future corruption moved earlier logits by {d:.3e} with carry='none'"
    d = _future_corruption_delta(_model(retention_carry=False))
    assert d == 0.0, f"future corruption moved earlier logits by {d:.3e} with carry=False"


def test_causality_holds_when_there_is_no_retention_branch_at_all():
    d = _future_corruption_delta(_model(retention=False))
    assert d == 0.0, f"future corruption moved earlier logits by {d:.3e} with no retention"


def test_acausal_final_optin_still_reproduces_the_leak():
    """The opt-in is honest: "acausal_final" (and bool-True back-compat) still leak.

    This is deliberate — the opt-in exists to load/diagnose pre-fix checkpoints, and a
    silently-causal "acausal_final" would decode a different model than was trained. If
    this ever reads ~0 the opt-in has been broken, not fixed."""
    d = _future_corruption_delta(_model(retention_carry="acausal_final"))
    assert d > 1e-4, f"acausal_final no longer leaks ({d:.3e}) — the opt-in is broken"
    d = _future_corruption_delta(_model(retention_carry=True))
    assert d > 1e-4, f"bool True no longer maps to the acausal carry ({d:.3e})"


def test_invalid_carry_mode_raises():
    with pytest.raises(ValueError):
        _model(retention_carry="per_position")(torch.zeros(1, 4, dtype=torch.long))


def test_the_carry_is_what_breaks_it_and_the_break_is_material():
    """Pin the CAUSE, and pin that it is not a rounding effect.

    A test that only said "carry on differs from carry off" would pass on a 1e-7
    numerical wobble. The measured effect on a trained checkpoint is 4.08 against a mean
    |logit| of 2.41, so require the tiny model's violation to be well clear of noise too.
    """
    on = _future_corruption_delta(_model(retention_carry="acausal_final"))
    off = _future_corruption_delta(_model(retention_carry="none"))
    assert off == 0.0
    assert on > 1e-4, (
        f"expected a material causality violation with the carry on, got {on:.3e}. "
        "If this is now ~0 the defect may be fixed: check the xfail above.")


# ── The SHIPPED path: _tul_core with a slot layout ────────────────────────────
# Every TUL arm runs `_tul_core`, not the plain loop — a guard that only covers the
# token path protects nothing (memory: morph-tul-token-path-needs-twin, caught twice
# in one day). Layout held FIXED; only future TOKEN ids are corrupted, so the probe
# tests the model's causality given a layout, not the (separately causal,
# BoundaryRule-state-machine) layout construction.

def _tul_model(**kw):
    import numpy as np
    from morph.model.tul import TULConfig
    from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids
    torch.manual_seed(0)
    m = _model(tul=TULConfig(prefix_k=2, slot_id=4),
               retention=True, retention_layers=(1,), retention_heads=2,
               retention_chunk=8, **kw)
    lut = np.zeros(V, dtype=bool)
    lut[[10, 11]] = True
    lut[0] = True
    rule = BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)
    spec = TulLayoutSpec(seq_len=32, prefix_k=2, max_slots=5, slot_id=4)
    rng = np.random.default_rng(2)
    ids = rng.integers(5, V, size=(1, 90))
    ids[ids == 4] = 5
    ids[:, ::6] = 10
    x, y, layout, _ = slot_layout_from_ids(ids.astype(np.int64), rule, spec)
    return m, x, layout


def _tul_future_corruption_delta(model, x, layout, first_corrupt_tok: int) -> float:
    """max |Δ logit| over EVERY packed position (tokens AND slots) strictly before the
    first corrupted token position, when all token ids from `first_corrupt_tok` on are
    replaced (slot positions and the layout untouched).

    Two probe-blindness traps, both hit while writing this:
    - Comparing token positions only is blind: the carry leak surfaces FIRST at slot
      positions (their core states change from iteration 2), and a token only sees it
      after attending a slot. The contract covers every position, so compare every
      position before the frontier.
    - The frontier must sit past at least one slot (here: after span 1), or none of the
      compared positions can causally depend on a slot and the probe passes on anything.
    """
    tok_pos = (~layout.slot_mask[0]).nonzero().squeeze(1)      # packed token positions
    corrupt = tok_pos[first_corrupt_tok:]
    assert len(corrupt) > 3, "probe needs a real corrupted tail"
    frontier = int(corrupt[0])                                 # first corrupted packed pos
    n_slots_before = int(layout.slot_mask[0, :frontier].sum())
    assert n_slots_before >= 1, "frontier before the first slot — the probe is blind"
    g = torch.Generator().manual_seed(9)
    bad = x.clone()
    bad[0, corrupt] = torch.randint(5, V, (len(corrupt),), generator=g)
    with torch.no_grad():
        a = model(x, None, slot_layout=layout)["logits"][0, :frontier].float()
        b = model(bad, None, slot_layout=layout)["logits"][0, :frontier].float()
    # The slot_id column is -inf at EVERY position by design (the emit mask), and
    # -inf minus -inf is NaN — compare the finite entries and require the same
    # -inf pattern on both sides.
    fa, fb = torch.isfinite(a), torch.isfinite(b)
    assert torch.equal(fa, fb), "the -inf mask pattern itself moved"
    assert fa.any(), "no finite logits to compare; the probe is blind"
    return (a[fa] - b[fb]).abs().max().item()


def test_tul_core_is_causal_on_the_default_config():
    # retention_gate_init=0.0 (sigmoid 0.5, vs the shipped -6 ≈ 0.0025) makes the probe
    # ~200x more sensitive; the assertion is EXACT zero either way, so boosting the gate
    # only strengthens the guard.
    m, x, layout = _tul_model(retention_gate_init=0.0)
    m.eval()
    for c in (13, 19):
        d = _tul_future_corruption_delta(m, x, layout, c)
        assert d == 0.0, (
            f"TUL path: corrupting tokens from token #{c} moved earlier logits by {d:.3e}")


def test_tul_core_acausal_optin_still_reproduces_the_leak():
    m, x, layout = _tul_model(retention_carry="acausal_final", retention_gate_init=0.0)
    m.eval()
    d = _tul_future_corruption_delta(m, x, layout, 13)
    assert d > 1e-4, (
        f"acausal_final on the TUL path no longer leaks ({d:.3e}) — opt-in broken, "
        "or the probe has gone blind (check mean_depth >= 2: one iteration never carries)")
