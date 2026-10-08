"""Checkpoint compatibility for the paid loop (.agents/notes/rejected/architecture/2026-09-02-tul-paid-loop-recipe.md §3).

The slot-loop arms save ``tul.W_prefix`` (their prefix projection). The paid-loop model
(``tul.tokens_through_core``) builds none of it, and ``load_checkpoint`` RAISES on an unexpected key by design (a homeless
tensor is lost state). Every A2 checkpoint under ``checkpoints/morph/`` from before
2026-09-03 carries the key, so the loaders drop exactly that key, loudly, and only for a
model whose TULSlots has no W_prefix — the slot loop and the FM planner still own the
projection and must keep the strict check. CPU only, tiny config.
"""

from __future__ import annotations

import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_fm import FMArmConfig
from morph.training.train import (RETIRED_TUL_KEYS, _canon_ckpt_key, drop_retired_tul_keys,
                                  load_checkpoint, load_weights_only)

V = 64


def _cfg(**kw) -> MORPHConfig:
    base = dict(
        d_model=32, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=64, context_len=64,
        n_prelude=1, n_core=1, n_coda=1, mean_depth=1, max_depth=1, bptt_depth=1,
        channel_dims=(16, 10, 6), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=8, retention=False,
        bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False, dropout=0.0,
        tul=TULConfig(prefix_k=2, slot_id=4, tokens_through_core=True),
    )
    base.update(kw)
    return MORPHConfig(**base)


def _old_arm_state(model: MORPHTransformer) -> dict:
    """A slot-only-arm checkpoint: the shipped tensors plus the retired projection."""
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    d = model.cfg.d_model
    sd["tul.W_prefix"] = torch.eye(d).expand(2, d, d).clone()
    return sd


def test_retired_key_set_is_exactly_the_prefix_projection():
    assert RETIRED_TUL_KEYS == ("tul.W_prefix",)


def test_drop_removes_only_the_retired_key_for_a_paid_loop_model(capsys):
    torch.manual_seed(0)
    m = MORPHTransformer(_cfg())
    state = _old_arm_state(m)
    n_before = len(state)
    dropped = drop_retired_tul_keys(state, m, "old_arm.pt")
    assert dropped == ["tul.W_prefix"]
    assert "tul.W_prefix" not in state and len(state) == n_before - 1
    assert "dropped 1 retired TUL tensor" in capsys.readouterr().out, "the drop must be LOUD"
    # and the surviving state loads with nothing homeless and nothing missing
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not missing and not unexpected


def test_drop_handles_the_compiled_key_convention():
    torch.manual_seed(0)
    m = MORPHTransformer(_cfg())
    state = {"_orig_mod.tul.W_prefix": torch.zeros(2, 32, 32), "tul.E_slot": torch.zeros(32)}
    assert drop_retired_tul_keys(state, m, "x.pt") == ["_orig_mod.tul.W_prefix"]
    assert set(state) == {"tul.E_slot"}


def test_drop_keeps_every_key_on_an_fm_planner_model():
    """The planner still owns W_prefix: dropping it there would lose a trained tensor."""
    torch.manual_seed(0)
    fm = FMArmConfig(d_p=16, n_layers=1, n_heads=2, d_ff=32, cond_dim=16, sigreg_slices=16,
                     source_std=1.0 / 8.0, max_slots=4, l_total=40)
    # The planner keeps the slot-loop write path (W_prefix); the paid loop + a planner RAISES.
    m = MORPHTransformer(_cfg(n_core=0, fm=fm, tul=TULConfig(prefix_k=2, slot_id=4)))
    assert m.tul.W_prefix is not None
    state = {k: v.clone() for k, v in m.state_dict().items()}
    assert "tul.W_prefix" in state
    assert drop_retired_tul_keys(state, m, "fm.pt") == []
    assert "tul.W_prefix" in state


def test_load_weights_only_loads_an_old_arm_checkpoint_with_no_unexpected_key(tmp_path):
    torch.manual_seed(1)
    src = MORPHTransformer(_cfg())
    with torch.no_grad():
        src.tul.E_slot.normal_()
    path = tmp_path / "old_arm.pt"
    torch.save({"model": _old_arm_state(src), "step": 7}, path)
    torch.manual_seed(2)
    dst = MORPHTransformer(_cfg())
    missing, unexpected = load_weights_only(str(path), dst, torch.device("cpu"))
    assert unexpected == [], f"the retired key must not surface as unexpected: {unexpected}"
    assert missing == []
    assert torch.equal(dst.tul.E_slot, src.tul.E_slot)
    for (ka, a), (kb, b) in zip(src.state_dict().items(), dst.state_dict().items()):
        assert ka == kb and torch.equal(a, b), ka


def _resume_into(tmp_path, src: MORPHTransformer, dst: MORPHTransformer):
    path = tmp_path / "resume.pt"
    torch.save({"model": {k: v.clone() for k, v in src.state_dict().items()},
                "optimizer": {}, "step": 7, "next_step": 8}, path)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return load_checkpoint(str(path), dst, scaler, torch.device("cpu"))


def _assert_same_weights(src: MORPHTransformer, dst: MORPHTransformer):
    a = {_canon_ckpt_key(k): v for k, v in src.state_dict().items()}
    b = {_canon_ckpt_key(k): v for k, v in dst.state_dict().items()}
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_resume_aligns_compile_wrappers_on_either_side(tmp_path):
    """An eager checkpoint resumes into a model whose blocks are wrapped by torch.compile
    (`training.compile_blocks`: keys gain `._orig_mod.`), and a wrapped model's checkpoint
    resumes into an eager one. The optimizer-state names come back in the LIVE model's key
    convention, so align_optimizer_state matches them by name. Before 2026-10-08 the first
    case raised with 420 keys without a home on the lxtul_pointer model."""
    torch.manual_seed(1)
    eager_src = MORPHTransformer(_cfg())
    torch.manual_seed(2)
    wrapped_dst = MORPHTransformer(_cfg())
    wrapped_dst.prelude[0] = torch.compile(wrapped_dst.prelude[0])
    assert any("._orig_mod." in k for k in wrapped_dst.state_dict())
    step, _, _, pnames = _resume_into(tmp_path, eager_src, wrapped_dst)
    assert step == 8
    _assert_same_weights(eager_src, wrapped_dst)
    assert pnames == set(wrapped_dst.state_dict().keys())

    torch.manual_seed(3)
    eager_dst = MORPHTransformer(_cfg())
    _resume_into(tmp_path, wrapped_dst, eager_dst)
    _assert_same_weights(wrapped_dst, eager_dst)

