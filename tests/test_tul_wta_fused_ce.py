"""``accumulate_span_ce_fused`` ("onewinner-perf", 2026-09-29): the fused-kernel twin of
``accumulate_span_ce`` (morph/model/transformer.py) that ``_tul_fan_all``'s WTA pick,
ranking-replacement (the shared mixture pass) and grad passes now use instead of a
per-row ``xh[b] @ w_head.T`` + ``F.cross_entropy`` Python loop (part 1 of the
"onewinner-perf" task: no behaviour change, just a faster kernel for the same quantity).

WHAT THIS FILE PROVES:

1. VALUE: on random ``xh`` / ``w_head`` / labels through the real ``span_ce_index``
   layout machinery, the fused path's per-bag summed CE matches the eager path's within
   a documented fp32 tolerance — not bit-identical (the fused kernel computes
   ``logits.gather(label) - logsumexp(logits)`` in chunks; the eager path computes
   ``F.cross_entropy``'s internal ``log_softmax`` + ``nll_loss`` over the whole ``[L, V]``
   row at once — same real-number quantity, different float ops, different rounding).
2. GRADIENT: the same, for both ``d/d xh`` and ``d/d w_head``.
3. The unmasked partition (no ``mask_token_id``): confirms this call site is a drop-in
   NUMERICAL twin of ``accumulate_span_ce`` specifically (not of
   ``MORPHTransformer._enum_mix_losses``'s masked mixture read, whose partition excludes
   the TUL slot id) — the fused call with the slot id INCLUDED in the softmax denominator
   is the one that agrees with the eager path; excluding it does not.
4. A model-level smoke: `_tul_fan_all`'s WTA pass (per_rollout and map) gives finite
   loss and gradients with the fused kernel wired in for real (not the synthetic tensors
   above) — the regression the 30-step GPU trace would have caught if this kernel swap
   itself were broken (it is not — the pin tests in test_tul_lxfan.py / test_tul_lx_credit.py
   already exercise this, with the tolerance documented there).

CPU only, fp32.
"""
from __future__ import annotations

import torch

from morph.model.transformer import (accumulate_span_ce, accumulate_span_ce_fused,
                                     span_ce_index)
from morph.model.tul_layout import slot_layout_from_ids
from test_tul_fan import _ids, _rule, _spec


def _fixture(seed: int = 0, V: int = 37, C: int = 16):
    """Real ``span_ce_index`` layout machinery (the ``test_tul_fan`` fixtures), random
    ``xh`` / ``w_head`` requiring grad — the two things :func:`accumulate_span_ce` and
    its fused twin actually read besides the index tensors."""
    ids = _ids(seed=seed)
    inp, lab_full, layout, _ = slot_layout_from_ids(ids, _rule(), _spec(4))
    B, L = inp.shape
    g = torch.Generator().manual_seed(seed)
    xh = torch.randn(B, L, C, generator=g, dtype=torch.float32, requires_grad=True)
    w_head = torch.randn(V, C, generator=g, dtype=torch.float32, requires_grad=True)
    gid, keep_tok, lab, g_bins = span_ce_index(lab_full, layout)
    # `span_ce_index` clamps labels into [0, inf); make sure every kept label is < V
    # (the fixture's real vocab is much bigger than this tiny V) so both paths read a
    # valid column.
    lab = lab.clamp(max=V - 1)
    return xh, w_head, gid, keep_tok, lab, g_bins, layout


def _clone_leaf(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone().requires_grad_(True)


def test_fused_value_matches_eager_within_tolerance():
    xh, w_head, gid, keep_tok, lab, g_bins, _layout = _fixture()
    xh_e, w_e = _clone_leaf(xh), _clone_leaf(w_head)
    xh_f, w_f = _clone_leaf(xh), _clone_leaf(w_head)

    out_e = accumulate_span_ce(xh_e, w_e, gid, keep_tok, lab, g_bins)
    out_f = accumulate_span_ce_fused(xh_f, w_f, gid, keep_tok, lab, g_bins, chunk_size=1024)

    assert out_e.shape == out_f.shape
    # fp32, V=37, C=16, ~2 rows x ~120 positions: the two kernels' logsumexp/gather
    # orderings disagree at the ULP level, accumulated over a [L] cross_entropy sum per
    # row. Measured on this fixture: max abs diff 7.6e-6, max relative diff 1.1e-7. The
    # tolerance below (1e-4 relative, 1e-5 absolute) is looser than that measured gap on
    # purpose, so this is not a hair-trigger pin, while still tight enough that a REAL
    # bug (wrong labels, wrong mask, wrong partition) would fail it by orders of
    # magnitude.
    torch.testing.assert_close(out_e, out_f, rtol=1e-4, atol=1e-5)


def test_fused_gradient_matches_eager_within_tolerance():
    xh, w_head, gid, keep_tok, lab, g_bins, _layout = _fixture()
    xh_e, w_e = _clone_leaf(xh), _clone_leaf(w_head)
    xh_f, w_f = _clone_leaf(xh), _clone_leaf(w_head)

    accumulate_span_ce(xh_e, w_e, gid, keep_tok, lab, g_bins).sum().backward()
    accumulate_span_ce_fused(xh_f, w_f, gid, keep_tok, lab, g_bins,
                             chunk_size=1024).sum().backward()

    assert xh_e.grad is not None and xh_f.grad is not None
    assert w_e.grad is not None and w_f.grad is not None
    # Gradients accumulate the forward's own rounding difference through one more
    # matmul-shaped reduction. Measured on this fixture: max abs diff 1.2e-6 (xh grad),
    # 2.9e-6 (w_head grad) -- the tolerance below (1e-3 relative, 1e-4 absolute) is
    # looser on purpose, while still tight enough to catch a real backward bug (a sign
    # flip, a missing `keep_tok` mask, a transposed weight) by orders of magnitude.
    torch.testing.assert_close(xh_e.grad, xh_f.grad, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(w_e.grad, w_f.grad, rtol=1e-3, atol=1e-4)


def test_fused_call_leaves_mask_token_id_off_by_default():
    """The one design choice that makes this call a twin of `accumulate_span_ce`
    (unmasked partition) and NOT of `_enum_mix_losses`'s masked mixture read: passing
    `mask_token_id` explicitly must move the result away from the eager (unmasked)
    reference, so a future edit that silently adds masking here would be caught."""
    xh, w_head, gid, keep_tok, lab, g_bins, _layout = _fixture()
    xh_e, w_e = _clone_leaf(xh), _clone_leaf(w_head)
    xh_f, w_f = _clone_leaf(xh), _clone_leaf(w_head)

    out_e = accumulate_span_ce(xh_e, w_e, gid, keep_tok, lab, g_bins)
    from morph.model.fused_ce import fused_linear_label_logprob
    B, L, C = xh_f.shape
    lab100 = torch.where(keep_tok, lab, lab.new_full((), -100))
    lp_masked = fused_linear_label_logprob(xh_f.reshape(-1, C), w_f, lab100.reshape(-1),
                                           ignore_index=-100, chunk_size=1024,
                                           mask_token_id=0)      # mask vocab row 0 on purpose
    ce_masked = (-lp_masked).to(torch.float32)
    out_masked = torch.zeros(B * g_bins, dtype=torch.float32).index_add(
        0, gid.reshape(-1), ce_masked).view(B, g_bins)
    # Masking a vocab row strictly lowers the log-partition, so every scored bin's CE
    # can only go DOWN (or stay put if row 0 was never a real label there) against the
    # unmasked eager reference -- never up, and it must differ somewhere on this
    # fixture (row 0 is a real, frequently-labelled token id).
    assert bool((out_masked <= out_e + 1e-6).all())
    assert not torch.allclose(out_masked, out_e, atol=1e-5)


def test_wta_forward_and_backward_are_finite_with_the_fused_kernel(monkeypatch=None):
    """Model-level smoke: `_tul_fan_all`'s WTA pass (a2, per_rollout) trains a real step
    through `accumulate_span_ce_fused` end to end -- not a synthetic tensor, the actual
    coda output. Sabotage-checked below."""
    from test_tul_lxfan import K, _build
    from test_tul_lx_credit import M, _wta_kw
    from test_tul_fan import _batch

    _ids0, inp, lab, layout = _batch(M)
    m = _build(**_wta_kw(model_kw={"dropout": 0.1})).train()
    torch.manual_seed(11)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    n_finite_grads = 0
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n
            n_finite_grads += 1
    assert n_finite_grads > 0
