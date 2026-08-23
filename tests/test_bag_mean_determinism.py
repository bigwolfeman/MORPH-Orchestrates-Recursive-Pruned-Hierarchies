"""bag_mean must be bit-reproducible, forward and backward.

Why this is a gate and not a nicety: the TUL divergence hunt spent ~30 runs trying to
bisect an onset window on trajectories that were never the same trajectory twice. The
cause was ``index_add_`` float atomics inside ``bag_mean`` — a varying summation order
that ``torch.use_deterministic_algorithms(True)`` does NOT flag, so the standard guard
gave false assurance. Measured on the old version: 20/20 repeats non-identical, 30.7 %
of backward elements different, and a 3.9e-2 max relative gradient error on a full
training step.

These tests FAIL on the ``index_add_`` implementation and pass on the one-hot ``bmm``
one. Determinism here only breaks on CUDA — CPU ``index_add_`` reduces in a fixed
order — so the determinism tests are CUDA-only. The correctness test is not.
"""

import pytest
import torch

from morph.model.tul import bag_mean

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _case(device: str, dtype: torch.dtype = torch.bfloat16, seed: int = 0):
    """A bag map with heavy collision: many token positions per bag is what makes the
    atomic accumulation order visible in the first place."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    B, L, C, n_bags = 4, 512, 256, 16
    signal = torch.randn(B, L, C, generator=g).to(device=device, dtype=dtype).requires_grad_(True)
    bag_id = torch.randint(0, n_bags + 1, (B, L), generator=g).to(device)
    token_sel = (torch.rand(B, L, generator=g) > 0.1).float().to(device)
    return signal, bag_id, token_sel, n_bags


def _fwd_bwd(signal, bag_id, token_sel, n_bags):
    out = bag_mean(signal, bag_id, token_sel, n_bags)
    grad = torch.autograd.grad(out.float().square().sum(), signal)[0]
    return out.detach().clone(), grad.detach().clone()


@requires_cuda
def test_bag_mean_forward_is_bit_reproducible():
    args = _case("cuda")
    ref, _ = _fwd_bwd(*args)
    for i in range(8):
        out, _ = _fwd_bwd(*args)
        assert torch.equal(out, ref), f"forward differs on repeat {i}"


@requires_cuda
def test_bag_mean_backward_is_bit_reproducible():
    args = _case("cuda")
    _, ref = _fwd_bwd(*args)
    for i in range(8):
        _, grad = _fwd_bwd(*args)
        assert torch.equal(grad, ref), f"backward differs on repeat {i}"


@requires_cuda
def test_determinism_survives_a_different_bag_permutation():
    """Determinism must come from the reduction, not from the indices happening to be
    sorted. Shuffling which position lands in which bag must not reintroduce drift."""
    signal, bag_id, token_sel, n_bags = _case("cuda", seed=3)
    perm = torch.randperm(bag_id.shape[1], device=bag_id.device)
    bag_id = bag_id[:, perm]
    args = (signal, bag_id, token_sel, n_bags)
    ref_out, ref_grad = _fwd_bwd(*args)
    for _ in range(4):
        out, grad = _fwd_bwd(*args)
        assert torch.equal(out, ref_out)
        assert torch.equal(grad, ref_grad)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_bag_mean_still_computes_the_span_mean(device):
    """The determinism fix must not change the contract: row b,k is the mean of the
    signal over the TOKEN positions whose bag is k, and the dump row is exactly zero."""
    B, L, C, n_bags = 2, 9, 3, 2
    signal = torch.arange(B * L * C, dtype=torch.float32, device=device).reshape(B, L, C)
    bag_id = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 2, 2, 2], [1, 0, 1, 0, 2, 2, 0, 1, 2]], device=device
    )
    token_sel = torch.ones(B, L, device=device)
    token_sel[0, 1] = 0.0  # a slot position must not pollute its own bag

    out = bag_mean(signal, bag_id, token_sel, n_bags)
    assert out.shape == (B, n_bags + 1, C)

    for b in range(B):
        for k in range(n_bags):
            mask = (bag_id[b] == k) & (token_sel[b] > 0)
            expect = signal[b][mask].mean(dim=0)
            torch.testing.assert_close(out[b, k], expect, rtol=1e-5, atol=1e-5)
    assert torch.equal(out[:, n_bags], torch.zeros(B, C, device=device))


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_empty_bag_is_zero_not_nan(device):
    """clamp(min=1) on the count: a bag no token lands in must give 0, never 0/0."""
    signal = torch.randn(1, 4, 2, device=device)
    bag_id = torch.zeros(1, 4, dtype=torch.long, device=device)
    token_sel = torch.zeros(1, 4, device=device)  # every position masked out
    out = bag_mean(signal, bag_id, token_sel, n_bags=3)
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))
