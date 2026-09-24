"""init_from of a TUL checkpoint keeps the checkpoint's trained ``tul.E_slot``.

Found 2026-09-23 on LXTUL-E Stage 0: ``init_from`` resets the step to 0, the trainer's TUL
activation branch then ran ``TULSlots.init_at_activation`` and overwrote the seed's
trained slot embedding with the embedding-table mean. The "frozen" ruler came back with
exactly one changed tensor. ``train.e_slot_seeded`` decides from the load's missing keys;
both activation sites in ``train.main`` must read it.
"""
from __future__ import annotations

import inspect
import os

import torch

from morph.model.transformer import MORPHTransformer
from morph.training import train as train_mod
from morph.training.train import e_slot_seeded, load_weights_only
from test_checkpoint_compat import _cfg


def _tul_model(seed: int) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_cfg())


def test_a_tul_checkpoint_seeds_e_slot(tmp_path):
    src = _tul_model(0)
    with torch.no_grad():
        src.tul.E_slot.normal_()
    path = os.path.join(tmp_path, "ck.pt")
    torch.save({"model": src.state_dict()}, path)
    dst = _tul_model(1)
    missing, _ = load_weights_only(path, dst, torch.device("cpu"))
    assert e_slot_seeded(missing, dst)
    assert torch.equal(dst.tul.E_slot, src.tul.E_slot)


def test_a_checkpoint_without_e_slot_does_not_seed_it():
    dst = _tul_model(1)
    assert not e_slot_seeded(["tul.E_slot"], dst)
    assert not e_slot_seeded(["_orig_mod.tul.E_slot"], dst)
    assert e_slot_seeded(["tul_spandec_par.pos"], dst)


def test_a_model_without_tul_is_never_seeded():
    class _Plain(torch.nn.Module):
        tul = None
    assert not e_slot_seeded([], _Plain())


def test_both_activation_sites_read_the_seed_flag():
    """Every ``init_at_activation`` call in ``train.main`` sits under an
    ``if _e_slot_from_ckpt:`` / ``else:`` pair, and the flag is set from the init_from
    load. A new activation site that skips the flag fails here."""
    src = inspect.getsource(train_mod.main)
    assert "_e_slot_from_ckpt = e_slot_seeded(_if_missing, model)" in src
    calls = [i for i in range(len(src)) if src.startswith("init_at_activation(", i)]
    assert len(calls) == 2, f"expected 2 activation sites, found {len(calls)}"
    for i in calls:
        before = src[max(0, i - 400):i]
        assert "if _e_slot_from_ckpt:" in before and "else:" in before, \
            "an init_at_activation call is not guarded by the init_from seed flag"
