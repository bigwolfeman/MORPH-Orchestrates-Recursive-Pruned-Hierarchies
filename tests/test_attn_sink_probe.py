"""Contract tests for the H18 attention-sink probe.

The probe's runtime self-test only fires when a real model runs on a GPU. These are the
CPU checks that the two things it claims are actually true:

  * `window_weights` really is a transcription of the shipped window branch, so the
    probe measures the model's own attention and not a lookalike;
  * `mass_stats` reports concentration and POSITIONALITY the way its docstring says,
    including the case that would make the whole H18 reading wrong — a content-driven
    sink reading as a positional one.

Every assertion is on the numeric contract, not on the shape or the type: each test
fails if the corresponding line of the probe is broken.
"""
import pytest
import torch

from lab.divergence.attn_sink_probe import mass_stats, window_weights
from morph.model.attention import _window_fallback


def _uniform_attn(B, H, S, K):
    return torch.full((B, H, S, K), 1.0 / K)


# ── window_weights is the shipped path ─────────────────────────────────────────────
@pytest.mark.parametrize("window_size", [4, 256])
@pytest.mark.parametrize("n_skip_rope", [0, 2])
def test_window_weights_reproduce_the_shipped_window_output(window_size, n_skip_rope):
    """`A @ v` must equal `_window_fallback(q, k, v)` on every query that has a key.

    This is the check the probe's runtime self-test performs on GPU, done here against
    the module the model actually calls. If someone edits the mask in `_window_fallback`
    and not in the probe, this fails.
    """
    torch.manual_seed(0)
    B, H, S, D = 2, 3, 16, 8
    # fp32 ON PURPOSE, do not "improve" this to fp64. `_window_fallback` builds its bias
    # with `torch.where(mask, 0.0, -inf)`, which is fp32, and hands it to
    # `F.scaled_dot_product_attention`. Measured 2026-08-25: at fp64 q/k/v that dtype
    # mismatch makes SDPA return a silently WRONG result (max error 3.03 against the
    # explicit softmax) while a matched fp64 mask is exact to 4e-16. At every dtype MORPH
    # actually runs — fp32, bf16, fp16 — the fp32 mask is promoted correctly and the
    # matched and mismatched masks agree to the last bit, so there is no shipped bug and
    # nothing in the hot path is worth changing for a case that cannot occur.
    q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
    scale = D ** -0.5
    ref = _window_fallback(q, k, v, window_size, q.device, scale, n_skip_rope)
    a = window_weights(q, k, window_size, scale, n_skip_rope)
    got = torch.einsum("bhij,bhjd->bhid", a.to(v.dtype), v)
    fin = torch.isfinite(ref) & torch.isfinite(got)
    assert fin.any(), "every query row was masked; the case is degenerate"
    assert torch.allclose(ref[fin], got[fin], atol=1e-6), \
        f"max diff {(ref[fin] - got[fin]).abs().max():.3e}"


def test_window_weights_exclude_the_self_token():
    """XSA: `A[i, i]` must be exactly zero for every query with another key."""
    torch.manual_seed(0)
    q, k = (torch.randn(1, 2, 8, 4) for _ in range(2))
    a = window_weights(q, k, 256, 0.5, 0)
    diag = torch.diagonal(a[0, 0], dim1=-2, dim2=-1)[1:]     # row 0 is all-NaN
    assert torch.all(diag == 0.0), f"self-attention leaked: {diag}"


# ── mass_stats reports what it claims ──────────────────────────────────────────────
def test_uniform_attention_gives_a_participation_ratio_of_n_valid():
    """Flat mass over `n` valid keys must read `pr == n` and `top1 == 1/n`, exactly."""
    B, H, S, K, nv = 2, 3, 10, 10, 6
    a = _uniform_attn(B, H, S, K)
    key_pos = torch.arange(K).view(1, 1, K).expand(B, S, K)
    q_valid = torch.zeros(B, S, dtype=torch.bool)
    q_valid[:, 1:nv] = True
    key_valid = torch.zeros(B, K, dtype=torch.bool)
    key_valid[:, :nv] = True
    st = mass_stats(a, key_pos, q_valid, key_valid)
    assert st["pr"] == pytest.approx(nv, abs=1e-9)
    assert st["top1"] == pytest.approx(1.0 / nv, abs=1e-9)


def test_a_single_sink_gives_a_participation_ratio_of_one():
    """All mass on key 3 must read `pr == 1`, `top1 == 1`, `argmax == 3`."""
    B, H, S, K, nv = 2, 3, 10, 10, 6
    a = torch.zeros(B, H, S, K)
    a[..., 3] = 1.0
    key_pos = torch.arange(K).view(1, 1, K).expand(B, S, K)
    q_valid = torch.ones(B, S, dtype=torch.bool)
    key_valid = torch.zeros(B, K, dtype=torch.bool)
    key_valid[:, :nv] = True
    st = mass_stats(a, key_pos, q_valid, key_valid)
    assert st["pr"] == pytest.approx(1.0, abs=1e-9)
    assert st["top1"] == pytest.approx(1.0, abs=1e-9)
    assert st["argmax"] == 3
    assert st["row_agree"] == pytest.approx(1.0)


def test_mass_on_padding_is_dropped_and_the_rest_renormalised():
    """Half the mass sitting on invalid keys must not change the reported shape."""
    B, H, S, K, nv = 1, 1, 4, 8, 4
    a = torch.zeros(B, H, S, K)
    a[..., :nv] = 0.5 / nv          # half the mass spread over the valid keys
    a[..., nv:] = 0.5 / nv          # the other half on padding
    key_pos = torch.arange(K).view(1, 1, K).expand(B, S, K)
    q_valid = torch.ones(B, S, dtype=torch.bool)
    key_valid = torch.zeros(B, K, dtype=torch.bool)
    key_valid[:, :nv] = True
    st = mass_stats(a, key_pos, q_valid, key_valid)
    assert st["pr"] == pytest.approx(nv, abs=1e-9)
    assert sum(st["mass"][nv:]) == pytest.approx(0.0, abs=1e-12)


def test_a_content_driven_sink_does_not_read_as_positional():
    """THE discriminator H18 rests on.

    Each row peaks on a DIFFERENT key. The batch-mean distribution still has an argmax,
    but `row_agree` must be 1/rows, not 1. If this ever reads high, a content-driven
    sink would be reported as a positional one and the H18 verdict would be wrong.
    """
    B, H, S, K = 4, 2, 6, 8
    a = torch.zeros(B, H, S, K)
    for b in range(B):
        a[b, ..., b] = 1.0
    key_pos = torch.arange(K).view(1, 1, K).expand(B, S, K)
    q_valid = torch.ones(B, S, dtype=torch.bool)
    key_valid = torch.ones(B, K, dtype=torch.bool)
    st = mass_stats(a, key_pos, q_valid, key_valid)
    assert st["row_agree"] == pytest.approx(1.0 / B), \
        "rows peaking on different keys read as agreeing — the positional test is broken"
    assert st["pr"] == pytest.approx(B, abs=1e-9)


def test_key_pos_is_honoured_and_not_the_column_index():
    """CSA passes `top_idx`, a permutation. Ignoring it would mis-attribute every mass."""
    B, H, S, K = 1, 1, 3, 4
    a = torch.zeros(B, H, S, K)
    a[..., 0] = 1.0                                   # all mass in COLUMN 0
    key_pos = torch.tensor([[[2, 0, 1, 3]] * S])      # column 0 refers to POSITION 2
    q_valid = torch.ones(B, S, dtype=torch.bool)
    key_valid = torch.ones(B, K, dtype=torch.bool)
    assert mass_stats(a, key_pos, q_valid, key_valid)["argmax"] == 2


def test_no_usable_row_raises_rather_than_reporting_a_flat_distribution():
    B, H, S, K = 2, 1, 4, 4
    a = _uniform_attn(B, H, S, K)
    key_pos = torch.arange(K).view(1, 1, K).expand(B, S, K)
    with pytest.raises(RuntimeError):
        mass_stats(a, key_pos, torch.zeros(B, S, dtype=torch.bool),
                   torch.ones(B, K, dtype=torch.bool))
