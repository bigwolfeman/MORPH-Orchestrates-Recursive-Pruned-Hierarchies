"""`tul.coda_logit_l2` — spectral decoupling on the coda's token logits (Pezeshki et
al., "Gradient Starvation", arXiv 2011.09468). One knob, five contracts:

  1. the fused kernel (`fused_linear_cross_entropy_logit_l2`) matches a materialised
     reference to fp32 exactness on loss AND on the gradients of both x and w, on an
     odd (padded) vocab and with per-row weights (TUL's half-weight double label);
  2. `tul.coda_logit_l2: 0.0` is bit-identical to the ruler — `_tul_group_losses` never
     even calls the penalty kernel at 0.0, proven with `torch.equal`, not `allclose`;
  3. the penalty reaches `_tul_group_losses`'s returned loss at exactly the weight
     `coda_logit_l2 / 2 * coda_logit_sq` (the `coda_logit_l2_weighted` contract every
     other TUL auxiliary term follows — grep `spandec_weighted` for the precedent);
  4. `tul.coda_logit_l2 > 0` on `_tul_group_losses`'s OTHER branch (`layout=None`, the
     arm-A4 plan-nats gather) raises rather than silently skipping the penalty;
  5. a negative `coda_logit_l2` is refused at `TULConfig` construction.
"""

from __future__ import annotations

import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402  (tests/ is on sys.path)

from morph.model.fused_ce import fused_linear_cross_entropy_logit_l2
from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig

MAX_DEPTH = 3


# ── 1. kernel exactness against a materialised reference ────────────────────────────

def _reference(x, w, labels, logit_l2_lambda, ignore_index=-100, mask_token_id=-1,
               weights=None):
    """Full-logits reference for the SAME quantity the fused kernel computes: CE +
    lambda/2 * mean(row_w * ||z_raw||^2) / n_valid, z_raw the UNMASKED logits."""
    logits_raw = x @ w.t()                                          # [N, V] fp32
    valid = (labels != ignore_index).float()
    row_w = valid if weights is None else weights.float() * valid
    n_valid_f = row_w.sum().clamp(min=1e-6)

    logits_masked = logits_raw
    if mask_token_id >= 0:
        logits_masked = logits_masked.clone()
        logits_masked[:, mask_token_id] = float("-inf")
    lab_safe = labels.clamp(min=0)
    lse = torch.logsumexp(logits_masked, dim=-1)
    tgt = logits_masked.gather(-1, lab_safe.unsqueeze(-1)).squeeze(-1)
    ce = ((lse - tgt) * row_w).sum() / n_valid_f

    sq = logits_raw.pow(2).sum(dim=-1)                              # [N]
    logit_sq_mean = (sq * row_w).sum() / n_valid_f
    penalty = (logit_l2_lambda / 2.0) * logit_sq_mean
    return ce + penalty, logit_sq_mean.detach()


@pytest.mark.parametrize("mask_token_id", [-1, 5])
@pytest.mark.parametrize("use_weights", [False, True])
def test_kernel_matches_reference_loss_and_grad(mask_token_id, use_weights):
    torch.manual_seed(0)
    N, d, V, lam, chunk = 37, 16, 100, 0.37, 13   # odd N, odd V (exercises _pad_vocab),
                                                   # chunk_size not a divisor of N
    x = torch.randn(N, d, dtype=torch.float32, requires_grad=True)
    w = (torch.randn(V, d, dtype=torch.float32) * d ** -0.5).requires_grad_(True)
    labels = torch.randint(0, V, (N,))
    labels[torch.rand(N) < 0.15] = -100          # some ignored rows
    weights = (0.5 + torch.rand(N)) if use_weights else None

    x_k, w_k = x.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True)
    x_r, w_r = x.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True)

    loss_k, sq_k = fused_linear_cross_entropy_logit_l2(
        x_k, w_k, labels, lam, chunk_size=chunk, mask_token_id=mask_token_id,
        weights=weights)
    loss_r, sq_r = _reference(x_r, w_r, labels, lam, mask_token_id=mask_token_id,
                              weights=weights)

    assert abs(loss_k.item() - loss_r.item()) < 1e-5, (loss_k.item(), loss_r.item())
    assert abs(sq_k.item() - sq_r.item()) < 1e-5 * max(abs(sq_r.item()), 1.0)
    assert sq_k.requires_grad is False, "logit_sq_mean must be a detached stat"

    loss_k.backward()
    loss_r.backward()
    gx_rel = (x_k.grad - x_r.grad).abs().max().item() / (x_r.grad.abs().max().item() + 1e-12)
    gw_rel = (w_k.grad - w_r.grad).abs().max().item() / (w_r.grad.abs().max().item() + 1e-12)
    assert gx_rel < 1e-5, gx_rel
    assert gw_rel < 1e-5, gw_rel


def test_kernel_zero_lambda_matches_plain_ce_and_zero_penalty_grad():
    """lambda=0 through the NEW kernel still adds exactly zero — a sanity check on the
    math, not the off-path (which never calls this kernel at all; see test 2 below)."""
    torch.manual_seed(1)
    N, d, V, chunk = 20, 12, 80, 8
    x = torch.randn(N, d, requires_grad=True)
    w = (torch.randn(V, d) * d ** -0.5).requires_grad_(True)
    labels = torch.randint(0, V, (N,))
    loss, sq = fused_linear_cross_entropy_logit_l2(x, w, labels, 0.0, chunk_size=chunk)
    ref, sq_ref = _reference(x, w, labels, 0.0)
    assert abs(loss.item() - ref.item()) < 1e-5
    assert abs(sq.item() - sq_ref.item()) < 1e-5 * max(abs(sq_ref.item()), 1.0)


# ── model-level: off is bit-identical, weight reaches the loss, raises, validates ──

def _model(seed: int = 0, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul = _tul(tg_restrict=False, sigreg_lambda=0.0, mux_beta=1.0, mux_target="next",
               mux_detach_head=False, **kw)
    base = dict(tul=tul, n_core=2, mean_depth=MAX_DEPTH, max_depth=MAX_DEPTH,
                bptt_depth=MAX_DEPTH, retention=False, dropout=0.1)
    return MORPHTransformer(_cfg(**base))


def _run(m: MORPHTransformer, *, seed: int = 7):
    x, y, layout, _ = _batch()
    m.train()
    torch.manual_seed(seed)
    out = m(x, labels=y, slot_layout=layout)
    out["loss"].backward()
    grads = {n: (p.grad.detach().clone() if p.grad is not None else None)
             for n, p in m.named_parameters()}
    return out, grads


def test_off_is_bit_identical_to_no_knob():
    """`coda_logit_l2: 0.0` (explicit) vs the field's own default (implicit 0.0): the
    penalty kernel is never called at 0.0, so loss and EVERY parameter's grad must be
    torch.equal, not merely close."""
    out_default, g_default = _run(_model(seed=3))
    out_zero, g_zero = _run(_model(seed=3, coda_logit_l2=0.0))
    assert torch.equal(out_default["loss"], out_zero["loss"])
    assert "coda_logit_sq" not in out_default and "coda_logit_sq" not in out_zero
    assert g_default.keys() == g_zero.keys()
    for k in g_default:
        a, b = g_default[k], g_zero[k]
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.equal(a, b), f"grad mismatch at {k} with coda_logit_l2=0.0"


def test_penalty_reaches_the_loss_at_the_documented_weight():
    out_r, _ = _run(_model(seed=3))
    out_p, _ = _run(_model(seed=3, coda_logit_l2=0.5))
    assert "coda_logit_sq" in out_p and float(out_p["coda_logit_sq"]) > 0.0
    assert float(out_p["coda_logit_l2_weighted"]) > 0.0
    expected = 0.5 / 2.0 * float(out_p["coda_logit_sq"])
    assert abs(float(out_p["coda_logit_l2_weighted"]) - expected) < 1e-6
    assert torch.allclose(out_p["loss"] - out_p["coda_logit_l2_weighted"], out_r["loss"],
                          atol=1e-5)


def test_penalty_moves_the_lm_head_gradient():
    _, g_r = _run(_model(seed=3))
    _, g_p = _run(_model(seed=3, coda_logit_l2=0.5))
    head_names = [n for n in g_r if "euc_embed" in n or "lor_embed" in n]
    assert head_names, "expected the tied embedding/LM-head parameters to be named here"
    assert any(not torch.equal(g_r[n], g_p[n]) for n in head_names), \
        "the penalty never reached the tied head's gradient"


def test_layout_none_raises_rather_than_skipping_the_penalty():
    """`_tul_group_losses`'s `layout=None` branch (the arm-A4 plan-nats gather) checks
    the knob before touching anything layout-shaped, so a plain dummy [B, L, C] state
    and [B, L] labels exercise the raise without needing a real coda forward."""
    m = _model(seed=3, coda_logit_l2=0.5)
    B, L, C = 2, 6, m.cfg.d_model
    xh = torch.randn(B, L, C)
    labels = torch.randint(0, m.cfg.vocab_size, (B, L))
    with pytest.raises(NotImplementedError, match="coda_logit_l2"):
        m._tul_group_losses(xh, labels, None)


def test_negative_lambda_raises_at_config():
    with pytest.raises(ValueError, match="coda_logit_l2"):
        TULConfig(coda_logit_l2=-0.1)
