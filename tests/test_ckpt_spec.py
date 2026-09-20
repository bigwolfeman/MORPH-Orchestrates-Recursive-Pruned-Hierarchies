"""``lab/divergence/_build.py::parse_ckpt_spec`` — the ONE ``--ckpt`` parser of the probes.

The fourth part is the arm's own training overrides; dropping it loaded fan4-mean's
checkpoint into the softmax config (2026-09-20). Every probe goes through this function.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab", "divergence"))
from _build import ROOT, parse_ckpt_spec  # noqa: E402


def test_three_parts_and_absolute_path():
    assert parse_ckpt_spec("arm=cfg=/x/step_5000.pt") == ("arm", "cfg", "/x/step_5000.pt", [])


def test_fourth_part_is_a_comma_list_of_overrides():
    lab, cfg, path, ovr = parse_ckpt_spec("arm=cfg=/x/s.pt=tul.fan_mix=mean,tul.fan_repel_lambda=0")
    assert (lab, cfg, path) == ("arm", "cfg", "/x/s.pt")
    assert ovr == ["tul.fan_mix=mean", "tul.fan_repel_lambda=0"]


def test_empty_fourth_part_means_no_overrides():
    assert parse_ckpt_spec("arm=cfg=/x/s.pt=")[3] == []


def test_relative_path_is_taken_against_the_repo_root():
    assert parse_ckpt_spec("arm=cfg=checkpoints/s.pt")[2] == os.path.join(ROOT, "checkpoints/s.pt")


def test_too_few_parts_raise():
    with pytest.raises(ValueError):
        parse_ckpt_spec("arm=cfg")
    with pytest.raises(ValueError):
        parse_ckpt_spec("arm==/x/s.pt")
