"""FWA next-latent write alignment (write_shift): contract tests.

S_t = diag(a_t) S_{t-1} + k_{t-1} v_t^T with k~_1 = 0 (arXiv:2608.27763 eq 2.3).
"""
import torch

from morph.model.gla import GatedLinearAttention


def _mk(mode, shift, seed=3):
    torch.manual_seed(seed)
    return GatedLinearAttention(64, 4, mode=mode, chunk=8, write_shift=shift).eval()


def test_recurrent_chunked_parity_with_shift():
    a, b = _mk("recurrent", True), _mk("chunked", True)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 33, 64)  # non-multiple of chunk
    with torch.no_grad():
        oa, sa = a(x)
        ob, sb = b(x)
    assert torch.allclose(oa, ob, atol=1e-4), (oa - ob).abs().max()
    assert torch.allclose(sa, sb, atol=1e-4)


def test_length_one_writes_nothing():
    # With shift, position 1's write feature is the zero sentinel: the final state
    # of a length-1 sequence must be exactly zero (nothing was ever written).
    m = _mk("recurrent", True)
    with torch.no_grad():
        _, state = m(torch.randn(2, 1, 64))
    assert torch.equal(state, torch.zeros_like(state))


def test_shift_changes_the_function():
    a, b = _mk("recurrent", False), _mk("recurrent", True)
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 16, 64)
    with torch.no_grad():
        oa, _ = a(x)
        ob, _ = b(x)
    assert not torch.allclose(oa, ob, atol=1e-4)


def test_shift_matches_manual_pair_recurrence():
    # Oracle: run the UNSHIFTED module on a key stream shifted by hand.
    m = _mk("recurrent", True)
    ref = _mk("recurrent", False)
    ref.load_state_dict(m.state_dict())
    x = torch.randn(1, 12, 64)
    with torch.no_grad():
        q, k, v, la, _ = m._project(x)
        k_shift = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)
        o_ref, s_ref = ref._recurrent(q, k_shift, v, la, None, reset_mask=None)
        o_ref = ref._readout(x, o_ref, None)
        o, s = m(x)
    assert torch.allclose(o, o_ref, atol=1e-5)
    assert torch.allclose(s, s_ref, atol=1e-5)


def test_reset_boundary_zeroes_cross_segment_key():
    # With a reset at position j, the shifted key there must be zeroed: the
    # suffix after the reset behaves like a fresh sequence.
    m = _mk("recurrent", True)
    x = torch.randn(1, 20, 64)
    j = 8
    reset = torch.zeros(1, 20, dtype=torch.bool)
    reset[0, j] = True
    with torch.no_grad():
        o_full, _ = m(x, reset_mask=reset)
        o_fresh, _ = m(x[:, j:])
    assert torch.allclose(o_full[:, j:], o_fresh, atol=1e-4), \
        (o_full[:, j:] - o_fresh).abs().max()
