"""Tests for the two seed-model enablement patches (2026-07-01):

1. n_core=0 — a prelude+coda-only model must construct, forward, and backward.
   The entire core-loop section (input_norm/h clone, depth sampling, x0 hoist,
   DiagonalInjection via _apply_core_step) must be skipped: with zero core
   blocks the injection would otherwise still perturb the ctx channel.

2. bigram_hash_vocab=0 — disables the bigram signal entirely: no table, no
   per-layer lambdas, get_bigram() returns None, injection terms carry a zero
   bigram component. (At the default 49152 the table is 12.6M params at d=256 —
   it must be absent, not merely lambda-silenced, for small seed models.)

Both patches must be no-ops for the default configuration; the bigram>0 +
n_core>0 path was verified bit-identical against pre-patch HEAD via a git
worktree at patch time (max |logit diff| = 0.0). These tests guard the new
paths going forward.
"""
import pytest
import torch

from morph.model.transformer import MORPHTransformer, MORPHConfig


def _cfg(**kw):
    base = dict(d_model=64, n_heads=2, n_kv_heads=2, vocab_size=128,
                n_prelude=2, n_core=2, n_coda=2, max_seq_len=64,
                mean_depth=2, max_depth=3, channel_dims=(32, 20, 12),
                use_kernels=False, bigram_hash_vocab=256)
    base.update(kw)
    return MORPHConfig(**base)


def _fwd_bwd(cfg):
    torch.manual_seed(0)
    m = MORPHTransformer(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    out = m(ids, labels=ids.clone())
    assert torch.isfinite(out["loss"]), "loss not finite"
    out["loss"].backward()
    n_grads = sum(1 for p in m.parameters() if p.grad is not None)
    assert n_grads > 0
    return m, out


# ─────────────────────────────────────────────── n_core = 0

def test_n_core_zero_constructs_and_trains():
    m, out = _fwd_bwd(_cfg(n_core=0))
    assert out["logits"] is None or out["logits"].shape[-1] == 128


def test_n_core_zero_eval_forward():
    m, _ = _fwd_bwd(_cfg(n_core=0))
    m.eval()
    with torch.no_grad():
        out = m(torch.randint(0, 128, (1, 16)))
    assert out["logits"].shape == (1, 16, 128)
    assert torch.isfinite(out["logits"]).all()


def test_n_core_zero_state_dict_roundtrip():
    torch.manual_seed(0)
    m = MORPHTransformer(_cfg(n_core=0))
    sd = m.state_dict()
    m2 = MORPHTransformer(_cfg(n_core=0))
    missing, unexpected = m2.load_state_dict(sd, strict=True), None
    # strict load raising would fail the test; reaching here is the assertion.
    assert not any(k.startswith("core.") for k in sd), "no core params expected"


def test_n_core_positive_unaffected():
    _fwd_bwd(_cfg(n_core=2))


# ─────────────────────────────────────────────── bigram disable

def test_bigram_zero_has_no_bigram_params():
    m = MORPHTransformer(_cfg(bigram_hash_vocab=0))
    assert m.embed.bigram is None
    assert not any("bigram" in k for k in m.state_dict()), \
        "bigram params leaked into state_dict"
    assert m.embed.get_bigram(torch.randint(0, 128, (1, 8))) is None


@pytest.mark.parametrize("n_core", [0, 2])
def test_bigram_zero_forward_backward(n_core):
    _fwd_bwd(_cfg(bigram_hash_vocab=0, n_core=n_core))


def test_bigram_zero_matches_lambda_zero_bigram():
    """Function check: a disabled bigram must equal an enabled bigram whose
    lambdas are (their init value) zero — the bigram contribution at init is
    exactly nothing, so logits must match when all shared weights are equal."""
    torch.manual_seed(0)
    m_off = MORPHTransformer(_cfg(bigram_hash_vocab=0)).eval()
    torch.manual_seed(0)
    m_on = MORPHTransformer(_cfg(bigram_hash_vocab=256)).eval()
    # copy the shared (non-bigram) weights from m_off into m_on so the only
    # difference is the (lambda=0, therefore inert) bigram table
    sd = m_on.state_dict()
    sd.update(m_off.state_dict())
    m_on.load_state_dict(sd, strict=True)
    ids = torch.randint(0, 128, (2, 24))
    with torch.no_grad():
        a = m_off(ids)["logits"]
        b = m_on(ids)["logits"]
    assert torch.equal(a, b), \
        f"disabled-bigram != zero-lambda-bigram: max diff {(a - b).abs().max()}"


def test_bigram_zero_tst_path():
    """The TST bagging branch also guards get_bigram()=None."""
    torch.manual_seed(0)
    cfg = _cfg(bigram_hash_vocab=0)
    m = MORPHTransformer(cfg)
    s = 2
    ids = torch.randint(0, 128, (2, 32 * s))
    labels = torch.randint(0, 128, (2, 32, s))
    out = m(ids, labels=labels, bag_size=s)
    assert torch.isfinite(out["loss"])
