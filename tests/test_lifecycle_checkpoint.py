"""Checkpoint resume step semantics (promoted from ignore/gate_checkpoint_next_step.py).

Protects the save/load contract the train loop depends on — not full train-loop
parity:

  - post-step checkpoints resume at the next unseen loop index
  - pre-step transition checkpoints resume at the same loop index
  - legacy checkpoints without next_step do not silently resume at the old index
"""
from __future__ import annotations

import os
import tempfile

import torch
import torch.nn as nn

from morph.training.train import load_checkpoint, save_checkpoint


def _fresh():
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    # Device string is bookkeeping only; state_dict roundtrips on CPU.
    scaler = torch.amp.GradScaler("cuda")
    return model, opt, scaler


def _load_step(path: str) -> int:
    model, _, scaler = _fresh()
    step, opt_state, needs_rebuild = load_checkpoint(
        path, model, scaler, torch.device("cpu"), pruning=None
    )
    assert isinstance(opt_state, dict)
    assert not needs_rebuild
    return step


def test_post_step_checkpoint_resumes_at_next_step():
    with tempfile.TemporaryDirectory(prefix="lifecycle_ckpt_") as tmp:
        model, opt, scaler = _fresh()
        path = os.path.join(tmp, "post_step_7.pt")
        save_checkpoint(path, 7, model, opt, scaler, pruning=None, next_step=8)
        assert _load_step(path) == 8


def test_pre_step_transition_checkpoint_resumes_at_same_step():
    with tempfile.TemporaryDirectory(prefix="lifecycle_ckpt_") as tmp:
        model, opt, scaler = _fresh()
        path = os.path.join(tmp, "pre_step_7.pt")
        save_checkpoint(path, 7, model, opt, scaler, pruning=None, next_step=7)
        assert _load_step(path) == 7


def test_legacy_checkpoint_without_next_step_assumes_post_step():
    with tempfile.TemporaryDirectory(prefix="lifecycle_ckpt_") as tmp:
        model, opt, scaler = _fresh()
        path = os.path.join(tmp, "post_step_7.pt")
        save_checkpoint(path, 7, model, opt, scaler, pruning=None, next_step=8)
        legacy = os.path.join(tmp, "legacy_step_7.pt")
        raw = torch.load(path, map_location="cpu", weights_only=False)
        raw.pop("next_step")
        torch.save(raw, legacy)
        assert _load_step(legacy) == 8
