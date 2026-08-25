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


# ── the looped core at the TUL slot budget ────────────────────────────────────────
#
# The guard above is correct for generation. It is NOT correct for the looped core: the
# core runs on SLOT positions, and if the slot budget is below the compressed ratio the
# branch silently produces nothing for a whole training run.

import os                                                              # noqa: E402

import yaml                                                            # noqa: E402

_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "morph", "configs")


def _yaml(name):
    with open(os.path.join(_CFG, f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def _slot_budget(cfg_name: str) -> tuple[int, int]:
    """`(max_slots, hca_compress_ratio)` as the named arm config resolves them.

    Walked by hand rather than through hydra so the test has no compose-time side
    effects. `tul.max_slots: 0` means the `seq_len // 8` default, which is what
    `morph/training/tul_setup.py` applies.
    """
    base = _yaml("base")
    arm = _yaml(cfg_name)
    short = _yaml("tul_short")
    seq_len = int((arm.get("data") or {}).get("seq_len")
                  or (short.get("data") or {}).get("seq_len")
                  or base["data"]["seq_len"])
    slots = (arm.get("tul") or {}).get("max_slots", base["tul"]["max_slots"])
    return (int(slots) or seq_len // 8), int(base["model"]["hca_compress_ratio"])


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN DEFECT, measured 2026-08-25. tul_a1 runs the looped core on 64 slot "
    "positions while model.hca_compress_ratio is 256, so the three HCA core blocks get "
    "n_blocks = 64 // 256 = 0 and their compressed branch output is identically 0.0000 "
    "while the gate still spends ~0.50 of its mixture on it. Measured with "
    "lab/divergence/attn_sink_probe.py --geometry. See "
    ".agents/notes/proposed/bug-fix/2026-08-25-hca-compressed-branch-dead-on-slot-path.md "
    "and takeover-campaign.md H24. When the fix lands, delete the xfail -- this test "
    "then guards it."))
def test_core_hca_branch_gets_at_least_one_block_at_the_tul_slot_budget():
    """A TUL arm must not run the core with a dead compressed branch."""
    slots, m = _slot_budget("tul_a1")
    assert slots // m >= 1, (
        f"tul_a1 gives the core {slots} slot positions against hca_compress_ratio {m}, "
        f"so n_blocks = {slots // m}: three of six core blocks output exactly zero from "
        f"their compressed branch for the whole run.")


def test_the_deploy_recipe_does_keep_its_core_hca_branch_alive():
    """base.yaml is NOT affected — the defect needs a slot budget under the ratio.

    Kept beside the xfail so a reader cannot mistake the defect for a global one: at
    seq_len 4096 the derived budget is 512 slots and the branch has 2 blocks.
    """
    base = _yaml("base")
    slots = int(base["tul"]["max_slots"]) or int(base["data"]["seq_len"]) // 8
    assert slots // int(base["model"]["hca_compress_ratio"]) >= 1
