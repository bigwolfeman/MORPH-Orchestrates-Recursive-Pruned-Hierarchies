"""Equivalence tests for the gathered `_tg_slot_attention` rewrite.

The gathered form restricts scoring to slot columns. A dense column masked to
-inf gets softmax weight exactly 0 and contributes no gradient, so the gathered
form must be the SAME function (up to fp reduction order), not an approximation.
The reference below is the pre-rewrite dense implementation, verbatim.
"""
import pytest
import torch

from morph.model.attention import _tg_slot_attention


def _dense_reference(q, k, v, slot_mask, sink_logits, scale):
    B, H, S, D = q.shape
    device = q.device
    row = torch.arange(S, device=device).unsqueeze(1)
    col = torch.arange(S, device=device).unsqueeze(0)
    causal = (col <= row).unsqueeze(0)
    allow = causal if slot_mask is None else causal & slot_mask.unsqueeze(1)
    scores = torch.einsum("bhid,bhjd->bhij", q.float(), k.float()) * scale
    scores = scores.masked_fill(~allow.unsqueeze(1), float("-inf"))
    sink = sink_logits.view(1, H, 1, 1).to(scores.dtype).expand(B, H, S, 1)
    scores = torch.cat([scores, sink], dim=-1)
    weights = torch.softmax(scores, dim=-1).to(q.dtype)
    return torch.einsum("bhij,bhjd->bhid", weights[..., :S], v)


def _rand_case(seed, B=3, H=4, S=96, D=16, slots_per_row=(11, 5, 0)):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, S, D, generator=g, requires_grad=True)
    k = torch.randn(B, H, S, D, generator=g, requires_grad=True)
    v = torch.randn(B, H, S, D, generator=g, requires_grad=True)
    sink = torch.randn(H, generator=g, requires_grad=True)
    slot_mask = torch.zeros(B, S, dtype=torch.bool)
    for b, n in enumerate(slots_per_row):
        if n:
            pos = torch.randperm(S, generator=g)[:n]
            slot_mask[b, pos] = True
    return q, k, v, sink, slot_mask


def _both(q, k, v, sink, slot_mask, scale=0.25):
    """Run reference and rewrite on cloned leaf tensors; return (outs, grads)."""
    outs, grads = [], []
    for fn in (_dense_reference, _tg_slot_attention):
        qq, kk, vv, ss = (t.detach().clone().requires_grad_(True)
                          for t in (q, k, v, sink))
        out = fn(qq, kk, vv, slot_mask, ss, scale)
        out.square().sum().backward()
        outs.append(out.detach())
        grads.append(tuple(t.grad for t in (qq, kk, vv, ss)))
    return outs, grads


def test_forward_and_grads_match_dense():
    q, k, v, sink, slot_mask = _rand_case(0)
    (ref, new), (gref, gnew) = _both(q, k, v, sink, slot_mask)
    assert torch.allclose(ref, new, atol=1e-5), (ref - new).abs().max()
    for a, b, name in zip(gref, gnew, ("q", "k", "v", "sink")):
        assert torch.allclose(a, b, atol=1e-5), f"grad {name}: {(a-b).abs().max()}"


def test_uneven_and_empty_rows():
    # Row with 0 slots: every query sees only the sink -> output exactly 0 there,
    # and sink gets a real (zero) grad, never None.
    q, k, v, sink, slot_mask = _rand_case(1, slots_per_row=(13, 1, 0))
    (ref, new), (gref, gnew) = _both(q, k, v, sink, slot_mask)
    assert torch.allclose(ref, new, atol=1e-5)
    assert new[2].abs().max() == 0.0
    assert gnew[3] is not None
    for a, b in zip(gref, gnew):
        assert torch.allclose(a, b, atol=1e-5)


def test_all_rows_empty_matches_dense():
    q, k, v, sink, slot_mask = _rand_case(2, slots_per_row=(0, 0, 0))
    (ref, new), (gref, gnew) = _both(q, k, v, sink, slot_mask)
    assert torch.allclose(ref, new, atol=1e-6)
    assert new.abs().max() == 0.0
    for a, b in zip(gref, gnew):
        assert a is not None and b is not None
        assert torch.allclose(a, b, atol=1e-6)


def test_grad_only_reaches_slot_positions():
    q, k, v, sink, slot_mask = _rand_case(3)
    _, (gref, gnew) = _both(q, k, v, sink, slot_mask)
    for g in (gref[1], gnew[1], gref[2], gnew[2]):        # k and v grads
        nonslot = g[~slot_mask.unsqueeze(1).unsqueeze(-1).expand_as(g)]
        assert nonslot.abs().max() == 0.0


def test_none_mask_path_unchanged():
    q, k, v, sink, _ = _rand_case(4, S=24)
    (ref, new), (gref, gnew) = _both(q, k, v, sink, None)
    assert torch.equal(ref, new)                          # same code path: bitwise
    for a, b in zip(gref, gnew):
        assert torch.equal(a, b)


def test_query_before_first_slot_gets_sink_only():
    # First slot at position 50: queries 0..49 must output exactly 0.
    B, H, S, D = 2, 2, 64, 8
    g = torch.Generator().manual_seed(5)
    q = torch.randn(B, H, S, D, generator=g)
    k = torch.randn(B, H, S, D, generator=g)
    v = torch.randn(B, H, S, D, generator=g)
    sink = torch.randn(2, generator=g)
    slot_mask = torch.zeros(B, S, dtype=torch.bool)
    slot_mask[:, 50] = True
    out = _tg_slot_attention(q, k, v, slot_mask, sink, 0.25)
    assert out[:, :, :50].abs().max() == 0.0
    assert out[:, :, 50:].abs().max() > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
