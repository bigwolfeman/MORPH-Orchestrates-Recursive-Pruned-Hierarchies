"""Parallel multi-token prediction on the coda readout (arc E8; model.mtp_heads).

Contracts: mtp_heads=1 builds nothing and is the untouched model; the heads' identity init
draws no RNG, so the base weights of a 4-head model equal the 1-head model's; each head's
CE is the CE of the SAME tied LM head against labels shifted by j-1 with the tail ignored;
the reported loss is CE_1 + w * sum CE_j; the lookahead loss alone reaches the core; the
TUL slot path refuses the heads. CPU only, tiny config.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from morph.model.transformer import MORPHConfig, MORPHTransformer

V = 64


def _cfg(**kw) -> MORPHConfig:
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


def _model(seed=7, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_cfg(**kw))
    m.eval()
    return m


def _batch(seed=0, B=2, T=32):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, V, (B, T), generator=g)
    y = torch.randint(0, V, (B, T), generator=g)
    return x, y


def test_one_head_is_the_plain_model():
    m = _model(mtp_heads=1)
    assert m.mtp is None
    x, y = _batch()
    out = m(x, labels=y)
    assert "mtp_weighted" not in out and "ce_mtp_2" not in out


def test_heads_draw_no_rng_and_start_at_identity():
    m1, m4 = _model(mtp_heads=1), _model(mtp_heads=4)
    p1 = dict(m1.named_parameters())
    for n, p in m4.named_parameters():
        if n.startswith("mtp."):
            continue
        assert torch.equal(p, p1[n]), n
    assert len(m4.mtp) == 3
    for h in m4.mtp:
        assert torch.equal(h.proj.weight, torch.eye(64))


def test_head_losses_are_shifted_label_ce_of_the_tied_head():
    m = _model(mtp_heads=4, mtp_weight=0.5)
    x, y = _batch()
    out = m(x, labels=y)
    logits = m(x, labels=None)
    ce1 = F.cross_entropy(logits["logits"].reshape(-1, V), y.reshape(-1), ignore_index=-100)
    assert torch.allclose(out["ce_main"], ce1, atol=1e-5)
    total = 0.0
    for j in (2, 3, 4):
        lab = F.pad(y[:, j - 1:], (0, j - 1), value=-100)
        lj = logits["mtp_logits"][j - 2]
        # The head logits are the tied LM head applied to the head's transform: at identity
        # init every head reproduces the next-token logits up to the extra RMSNorm.
        ref = F.cross_entropy(lj.reshape(-1, V), lab.reshape(-1), ignore_index=-100)
        assert torch.allclose(out[f"ce_mtp_{j}"], ref, atol=1e-5), j
        total = total + ref
    assert torch.allclose(out["mtp_weighted"], 0.5 * total, atol=1e-5)
    assert torch.allclose(out["loss"], ce1 + 0.5 * total, atol=1e-5)


def test_tail_positions_are_ignored():
    """Head j has exactly T-(j-1) supervised positions: a label change in the ignored tail
    must not move its loss, and a change inside the window must."""
    m = _model(mtp_heads=3)
    x, y = _batch(B=1, T=16)
    base = m(x, labels=y)["ce_mtp_3"]
    # Head 3 predicts y[t+2] from position t, so y[0] and y[1] are never its target.
    y2 = y.clone()
    y2[0, 0] = (y2[0, 0] + 1) % V
    y3 = y.clone()
    y3[0, 1] = (y3[0, 1] + 1) % V
    assert torch.allclose(m(x, labels=y2)["ce_mtp_3"], base)
    assert torch.allclose(m(x, labels=y3)["ce_mtp_3"], base)
    y4 = y.clone()
    y4[0, 5] = (y4[0, 5] + 1) % V
    assert not torch.allclose(m(x, labels=y4)["ce_mtp_3"], base)


def test_lookahead_loss_alone_reaches_the_core():
    m = _model(mtp_heads=2)
    m.train()
    x, y = _batch()
    out = m(x, labels=y)
    out["ce_mtp_2"].backward()
    core = [p for n, p in m.named_parameters() if n.startswith("core.") and p.grad is not None]
    assert core and any(p.grad.abs().sum() > 0 for p in core)
    assert m.mtp[0].proj.weight.grad is not None and m.mtp[0].proj.weight.grad.abs().sum() > 0


def test_bag_labels_and_bad_head_count_raise():
    m = _model(mtp_heads=2)
    x, y = _batch()
    with pytest.raises(RuntimeError):
        m(x, labels=y.unsqueeze(-1).expand(-1, -1, 2))
    with pytest.raises(ValueError):
        _model(mtp_heads=0)
