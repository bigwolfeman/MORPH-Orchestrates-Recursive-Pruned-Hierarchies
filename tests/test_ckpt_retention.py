"""Retention for periodic checkpoints.

These exist because `checkpoints/morph/` reached 292 GB on 2026-08-25: the rolling ring
rotated, the ordinary `ckpt_every` path did not. Every test below asserts the CONTRACT
(which files survive on disk), not the shape of the return value, so each one fails if
the rotation is removed or inverted.
"""
from __future__ import annotations

import os

import pytest

from morph.training.ckpt_retention import RetentionRing, existing_step_checkpoints


def _touch(d, name, size=16):
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(b"\0" * size)
    return p


def test_ring_keeps_only_the_newest_k_files_on_disk(tmp_path):
    d = str(tmp_path)
    ring = RetentionRing(3, log=lambda _m: None)
    for step in (100, 200, 300, 400, 500):
        ring.add(_touch(d, f"step_{step}.pt"))
    assert sorted(os.listdir(d)) == ["step_300.pt", "step_400.pt", "step_500.pt"]


def test_ring_deletes_the_OLDEST_not_the_newest(tmp_path):
    """The inversion this catches is real: a lexical sort makes step_900 look newer than
    step_1000, and an inverted pop discards the checkpoint a resume needs."""
    d = str(tmp_path)
    ring = RetentionRing(1, log=lambda _m: None)
    ring.add(_touch(d, "step_900.pt"))
    ring.add(_touch(d, "step_1000.pt"))
    assert os.listdir(d) == ["step_1000.pt"], "the newest checkpoint must be the survivor"


def test_keep_zero_means_unbounded_and_deletes_nothing(tmp_path):
    d = str(tmp_path)
    ring = RetentionRing(0, log=lambda _m: None)
    assert not ring.enabled
    for step in (1, 2, 3, 4, 5):
        ring.add(_touch(d, f"step_{step}.pt"))
    assert len(os.listdir(d)) == 5


def test_negative_keep_is_clamped_to_disabled_not_to_a_magnitude(tmp_path):
    """`abs(keep)` would turn -4 into "keep 4" — a silent bound nobody asked for. A
    negative keep must mean DISABLED, so enough files are written here that a ring which
    quietly became keep=4 would delete two of them and fail."""
    d = str(tmp_path)
    ring = RetentionRing(-4, log=lambda _m: None)
    assert ring.keep == 0
    assert not ring.enabled
    for step in (1, 2, 3, 4, 5, 6):
        ring.add(_touch(d, f"step_{step}.pt"))
    assert len(os.listdir(d)) == 6


def test_files_the_ring_never_saw_are_never_deleted(tmp_path):
    """Abort-guard output is the one thing that must never be rotated away."""
    d = str(tmp_path)
    _touch(d, "DIVERGED_step_2040.pt")
    _touch(d, "TAKEOVER_step_1866.pt")
    _touch(d, "wandb_id.txt")
    ring = RetentionRing(1, log=lambda _m: None)
    for step in (100, 200, 300):
        ring.add(_touch(d, f"step_{step}.pt"))
    assert sorted(os.listdir(d)) == [
        "DIVERGED_step_2040.pt", "TAKEOVER_step_1866.pt", "step_300.pt", "wandb_id.txt",
    ]


def test_a_failed_delete_is_reported_and_the_ring_does_not_retry_it(tmp_path):
    """try/except must not swallow: the message is asserted, and the path must leave the
    ring so a permanently-undeletable file cannot stall rotation forever."""
    d = str(tmp_path)
    msgs = []
    ring = RetentionRing(1, log=msgs.append)
    gone = os.path.join(d, "step_100.pt")     # never created -> os.remove raises
    ring.add(gone)
    ring.add(_touch(d, "step_200.pt"))
    assert any("could not remove" in m and "step_100.pt" in m for m in msgs)
    assert gone not in ring.paths
    ring.add(_touch(d, "step_300.pt"))
    assert os.listdir(d) == ["step_300.pt"]


def test_seed_adopts_existing_files_so_a_resume_still_enforces_the_bound(tmp_path):
    """Without seeding, a resumed run rotates only its OWN writes and the directory grows
    past keep_last on every restart — which is how the 292 GB accumulated."""
    d = str(tmp_path)
    for step in (500, 1000, 1500):
        _touch(d, f"step_{step}.pt")
    ring = RetentionRing(2, log=lambda _m: None)
    ring.seed(existing_step_checkpoints(d))
    assert len(os.listdir(d)) == 3, "seeding alone must not delete anything"
    ring.add(_touch(d, "step_2000.pt"))
    assert sorted(os.listdir(d)) == ["step_1500.pt", "step_2000.pt"]


def test_existing_step_checkpoints_sorts_numerically_not_lexically(tmp_path):
    d = str(tmp_path)
    for step in (900, 1000, 90, 10000):
        _touch(d, f"step_{step}.pt")
    assert [os.path.basename(p) for p in existing_step_checkpoints(d)] == [
        "step_90.pt", "step_900.pt", "step_1000.pt", "step_10000.pt",
    ]


def test_existing_step_checkpoints_ignores_roll_and_guard_files(tmp_path):
    d = str(tmp_path)
    _touch(d, "step_100.pt")
    _touch(d, "ROLL_step_200.pt")
    _touch(d, "DIVERGED_step_300.pt")
    _touch(d, "step_400.pt.tmp")
    _touch(d, "wandb_id.txt")
    assert [os.path.basename(p) for p in existing_step_checkpoints(d)] == ["step_100.pt"]


def test_existing_step_checkpoints_on_a_missing_dir_is_empty_not_an_error(tmp_path):
    assert existing_step_checkpoints(str(tmp_path / "not-created-yet")) == []


def test_base_config_ships_a_bounded_default():
    """The whole point is that the DEFAULT is bounded. A run that opts out must say so."""
    import yaml
    with open("morph/configs/base.yaml") as f:
        cfg = yaml.safe_load(f)
    keep = cfg["training"]["ckpt_keep_last"]
    assert isinstance(keep, int) and keep > 0, (
        f"base.yaml must ship bounded checkpoint retention, got ckpt_keep_last={keep!r}")


def test_the_two_rings_are_distinguishable_in_the_log(tmp_path):
    """Both rings share one class, so without a tag every rotation would read `[ckpt]`
    and a roll rotation would be indistinguishable from a periodic one in a run log."""
    d = str(tmp_path)
    msgs = []
    roll = RetentionRing(1, tag="roll", log=msgs.append)
    roll.add(_touch(d, "ROLL_step_100.pt"))
    roll.add(_touch(d, "ROLL_step_200.pt"))
    assert any(m.strip().startswith("[roll]") for m in msgs), msgs
    assert not any("[ckpt]" in m for m in msgs), msgs
