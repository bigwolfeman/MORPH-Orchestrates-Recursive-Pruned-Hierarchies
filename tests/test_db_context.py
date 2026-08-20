"""Two-source concat masks: the no-leak arithmetic (design §2).

These are the tests a target leak would fail. The invariant under test: a noisy
query at position i, which predicts token i+1, must never attend any CLEAN block or
CLEAN key that covers position i+1 (its own target), while it MUST be able to attend
clean position i (the current token).
"""

import torch

from morph.model.db_context import (
    clean_block_causal_mask,
    merged_block_causal_mask,
    two_source_window,
    compressed_two_source_reference,
)
from morph.model.attention import _compressed_causal_mask


def test_clean_block_includes_current_excludes_target():
    S, m = 16, 4
    nb = S // m
    mask = clean_block_causal_mask(S, nb, m, "cpu")  # [S, nb]
    # Block j covers [j*m, (j+1)*m-1]. Query i: visible iff block_end <= i.
    for i in range(S):
        for j in range(nb):
            block_end = (j + 1) * m - 1
            expected = block_end <= i
            assert bool(mask[i, j]) == expected, (i, j, block_end)


def test_clean_block_no_target_leak():
    # The decisive one: for every query i, no VISIBLE clean block contains position i+1.
    S, m = 24, 4
    nb = S // m
    mask = clean_block_causal_mask(S, nb, m, "cpu")
    for i in range(S):
        target = i + 1
        for j in range(nb):
            covers_target = j * m <= target <= (j + 1) * m - 1
            if covers_target:
                assert not bool(mask[i, j]), (
                    f"query {i} can see clean block {j} which covers its target {target}")


def test_current_token_is_reachable_somewhere():
    # Position i (current token) must be reachable by query i via the clean side:
    # either a whole clean block ending exactly at i, or (near i) the window. Here we
    # assert the block side is inclusive at block boundaries.
    S, m = 16, 4
    nb = S // m
    mask = clean_block_causal_mask(S, nb, m, "cpu")
    for j in range(nb):
        block_end = (j + 1) * m - 1
        assert bool(mask[block_end, j]), (
            f"query at block_end {block_end} cannot see its own completed clean block {j}")


def test_merged_mask_noisy_half_matches_baseline():
    S, m = 32, 8
    nb = S // m
    merged = merged_block_causal_mask(S, nb, nb, m, "cpu")   # [S, 2*nb]
    baseline_noisy = _compressed_causal_mask(S, nb, m, "cpu")  # [S, nb]
    assert torch.equal(merged[:, nb:], baseline_noisy), "noisy half must be bit-identical"
    assert torch.equal(merged[:, :nb], clean_block_causal_mask(S, nb, m, "cpu"))


def test_two_source_window_no_target_leak():
    # Give clean key i+1 a huge, unique value; if query i could attend it, the output
    # at i would move toward it. Confirm it does NOT.
    B, H, S, D = 1, 1, 12, 4
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D)
    k_noisy = torch.randn(B, H, S, D)
    v_noisy = torch.randn(B, H, S, D)
    k_clean = torch.randn(B, H, S, D)
    v_clean = torch.zeros(B, H, S, D)
    # Mark each clean position's value with a distinct large signature.
    for j in range(S):
        v_clean[0, 0, j, :] = 100.0 * (j + 1)
    # Make clean keys equal to q so that, if attended, they would dominate the softmax.
    k_clean = q.clone()
    out = two_source_window(q, k_clean, v_clean, k_noisy, v_noisy,
                            window_size=S, scale=D ** -0.5)
    # For query i, the maximum clean value it may legitimately absorb is 100*(i+1)
    # (position i). It must never absorb 100*(i+2) (position i+1, the target).
    for i in range(S - 1):
        target_sig = 100.0 * (i + 2)
        # output magnitude cannot reach the target signature if it is masked out.
        assert out[0, 0, i].abs().max().item() < target_sig, (
            f"query {i} output {out[0,0,i].abs().max().item():.1f} reached target sig {target_sig}")


def test_two_source_window_reachability_and_leak_by_perturbation():
    # Clean, softmax-safe reachability + no-leak test: perturb one clean value row and
    # see which queries' outputs move. Perturbing clean position p must move exactly the
    # queries that may legitimately see it (i >= p, within window) and NEVER query p-1
    # (for which p is the target p = (p-1)+1).
    B, H, S, D = 1, 2, 10, 8
    torch.manual_seed(1)
    q = torch.randn(B, H, S, D)
    k_clean = torch.randn(B, H, S, D)
    v_clean = torch.randn(B, H, S, D)
    k_noisy = torch.randn(B, H, S, D)
    v_noisy = torch.randn(B, H, S, D)
    base = two_source_window(q, k_clean, v_clean, k_noisy, v_noisy, window_size=S, scale=D ** -0.5)

    for p in range(1, S):
        vc = v_clean.clone()
        vc[0, :, p, :] += 50.0                       # perturb clean position p
        out = two_source_window(q, k_clean, vc, k_noisy, v_noisy, window_size=S, scale=D ** -0.5)
        moved = (out - base).abs().amax(dim=-1)[0, 0]  # [S] per-query movement (head 0)
        # No leak: query p-1 (whose target is p) must be unmoved.
        assert moved[p - 1].item() < 1e-6, f"LEAK: perturbing clean {p} moved query {p-1}"
        # Reachability: query p (current token = p) must move.
        assert moved[p].item() > 1e-4, f"query {p} could not reach its current clean token {p}"


def test_compressed_two_source_no_target_leak_by_perturbation():
    # Same leak-by-perturbation invariant for the COMPRESSED (HCA) branch, which the
    # window test above does NOT cover. Perturb the CLEAN block that contains the target
    # position i+1; query i's output must be UNMOVED (its target block is masked out),
    # while perturbing the block that contains only the current token i must move query i.
    B, H, S, D, m = 1, 2, 12, 8, 4
    nb = S // m
    torch.manual_seed(2)
    q = torch.randn(B, H, S, D)
    C_clean = torch.randn(B, nb, D)
    C_noisy = torch.randn(B, nb, D)
    sink = torch.randn(H)
    mask = merged_block_causal_mask(S, nb, nb, m, "cpu")            # [S, 2nb]
    base = compressed_two_source_reference(
        q, torch.cat([C_clean, C_noisy], dim=1), mask, sink, D ** -0.5)

    for i in range(S - 1):
        tgt_block = (i + 1) // m                                    # clean block holding target i+1
        Cc = C_clean.clone()
        Cc[:, tgt_block, :] += 50.0
        out = compressed_two_source_reference(
            q, torch.cat([Cc, C_noisy], dim=1), mask, sink, D ** -0.5)
        moved = (out - base).abs()[0, :, i, :].max().item()
        assert moved < 1e-6, f"LEAK: perturbing clean block {tgt_block} (holds target {i+1}) moved query {i}"
