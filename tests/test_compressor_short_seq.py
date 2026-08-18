"""GatedPoolCompressor with fewer tokens than one block.

Regression for the crash found on 2026-08-18 while sampling the finished TUL arms:
generation starts from a prompt of a handful of tokens, CSA's block size is 8, and the
two-stream path built a 1-block B stream against a 0-block A stream. Nothing had caught
it because base.yaml ships gen_every: 0, so no run had ever generated on this config.
"""
import pytest
import torch

from morph.model.attention import GatedPoolCompressor


@pytest.mark.parametrize("two_stream", [True, False])
@pytest.mark.parametrize("S", [1, 3, 7])
def test_short_sequence_returns_no_blocks(two_stream, S):
    """S < m must give ZERO compressed blocks, on both streams, not a crash."""
    d_model, c, m = 32, 8, 8
    comp = GatedPoolCompressor(d_model, c, m, two_stream=two_stream)
    out = comp(torch.randn(2, S, d_model))
    assert out.shape == (2, 0, c), f"S={S} two_stream={two_stream} gave {tuple(out.shape)}"


@pytest.mark.parametrize("two_stream", [True, False])
def test_exact_block_boundary_still_compresses(two_stream):
    """The guard must not swallow the first REAL block: S == m gives exactly one."""
    d_model, c, m = 32, 8, 8
    comp = GatedPoolCompressor(d_model, c, m, two_stream=two_stream)
    assert comp(torch.randn(2, m, d_model)).shape == (2, 1, c)
    assert comp(torch.randn(2, 2 * m + 3, d_model)).shape == (2, 2, c)


def test_two_stream_values_unchanged_above_the_guard():
    """The guard is a short-circuit, not a behaviour change: with a full block the
    two-stream output must still be the joint-softmax mix, and the first block's B
    stream must contribute exactly zero (its gates are -inf)."""
    torch.manual_seed(0)
    d_model, c, m = 32, 8, 8
    comp = GatedPoolCompressor(d_model, c, m, two_stream=True)
    x = torch.randn(1, m, d_model)
    out = comp(x)
    # One block only ⇒ B stream is entirely padding ⇒ result is the A-stream softmax.
    C_a = comp.W_aKV(x).reshape(1, 1, m, c)
    Z_a = comp.W_aZ(x).reshape(1, 1, m, c) + comp.B_a
    expect = (torch.softmax(Z_a.float(), dim=2).to(x.dtype) * C_a).sum(dim=2)
    torch.testing.assert_close(out, expect, rtol=1e-4, atol=1e-5)
