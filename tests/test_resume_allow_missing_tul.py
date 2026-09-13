"""`training.resume_plain_to_tul` — the plain → TUL bootstrap load, and its refusals.

`slot-strict-bootstrap` (morph/configs/tul_slot_strict_bootstrap.yaml) starts the strict
slot ruler from a checkpoint of a PLAIN 5,000-step run. Two classes of tensor cannot
match across that boundary, and they run in OPPOSITE directions:

  MISSING     every `tul.*` / `tul_spandec.*` parameter. The plain seed never had them.
  UNEXPECTED  every `*.attention._impl.{compressor,comp_norm,indexer}.*` tensor of the
              seed. Under `tul.tg_restrict` MORPHAttention builds none of them — the
              compressed branch attends the slot positions directly instead of pooling —
              so the seed's copies have no home and are DROPPED.

The flag allows exactly those two, enumerated from the LIVE model's modules, and refuses
everything else. Every assertion here is two-sided: the allowed classes must load AND the
same probe must refuse a sabotaged key, because a one-sided test passes just as happily on
a classifier that allows everything.

CPU only, fp32, tiny config, no tokenizer — the `tests/test_tul_strict_geometry.py`
fixtures.
"""

from __future__ import annotations

import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.training.train import (
    classify_plain_to_tul_keys,
    load_checkpoint,
    load_weights_only,
    tg_dropped_attention_prefixes,
    tul_owned_prefixes,
)

V = 64
SEED = 1234


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=2, max_depth=3, bptt_depth=3,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _ruler_tul(**kw) -> TULConfig:
    """The arm's TUL recipe: strict geometry + the span decoder (tul_slot_spandec_strict)."""
    base = dict(prefix_k=2, slot_id=4, tokens_through_core=False,
                tg_restrict=True, tg_restrict_scope="all", tg_geometry="strict",
                tg_coda_prefix_reach="all", spandec=True, spandec_layers=2,
                spandec_weight=1.0, spandec_max_tokens=0, mux_beta=0.0,
                mux_detach_head=True, emit_weight=0.0, token_state_dropout=0.15)
    base.update(kw)
    return TULConfig(**base)


def _plain(seed: int = SEED) -> MORPHTransformer:
    """The seed run's model: `notul_panel_norm_match` has no TUL config at all."""
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny()).eval().float()


def _ruler(seed: int = SEED + 1, **tul_kw) -> MORPHTransformer:
    """The arm's model: the strict ruler, whose core also re-blocks its HCA branch."""
    torch.manual_seed(seed)
    return MORPHTransformer(
        _tiny(core_hca_compress_ratio=16, tg_scoped_kernels=True,
              tul=_ruler_tul(**tul_kw))).eval().float()


def _seed_file(tmp_path, model: MORPHTransformer, step: int = 5000, **extra) -> str:
    """A checkpoint written the way `save_checkpoint` writes one."""
    ck = {"step": step, "next_step": step,
          "model": {k: v.clone() for k, v in model.state_dict().items()},
          "optimizer": {"state": {}, "param_groups": []},
          "scaler": torch.amp.GradScaler("cpu", enabled=False).state_dict(),
          "rng_cpu": None, "rng_cuda": None,
          "pruning_compact": False, "pruning_routed": False}
    ck.update(extra)
    p = str(tmp_path / f"step_{step}.pt")
    torch.save(ck, p)
    return p


# ── (a) the two lists are read off the live model, and a plain model owns nothing ──

def test_a_plain_model_owns_no_tul_prefix_and_drops_no_pooled_branch():
    """The two-sided floor: with no TUL config the classifier may allow NOTHING."""
    m = _plain()
    assert tul_owned_prefixes(m) == ()
    assert tg_dropped_attention_prefixes(m) == ()
    ok_m, bad_m, ok_u, bad_u = classify_plain_to_tul_keys(
        ["tul.E_slot"], ["prelude.0.attention._impl.compressor.B_a"], m)
    assert ok_m == [] and bad_m == ["tul.E_slot"]
    assert ok_u == [] and bad_u == ["prelude.0.attention._impl.compressor.B_a"]


def test_the_ruler_enumerates_its_tul_children_and_its_dropped_pooled_branches():
    m = _ruler()
    assert tul_owned_prefixes(m) == ("tul.", "tul_spandec.")
    tg = tg_dropped_attention_prefixes(m)
    assert tg, "a tg_restrict model must report the branches it did not build"
    # every reported prefix names a real attention module that really holds None there
    for pre in tg:
        assert pre.endswith((".compressor.", ".comp_norm.", ".indexer."))
        mod_path, sub = pre[:-1].rsplit(".", 1)
        mod = m.get_submodule(mod_path)
        assert getattr(mod, sub) is None and mod.tg_restrict is True
    # and it covers all three sections, not just the core
    assert any(p.startswith("prelude.") for p in tg)
    assert any(p.startswith("core.") for p in tg)
    assert any(p.startswith("coda.") for p in tg)


def test_the_two_classes_are_exactly_the_difference_between_the_seed_and_the_arm():
    """The load has no third surprise: every homeless key falls in one of the two lists."""
    seed, arm = _plain(), _ruler()
    sk, ak = set(seed.state_dict()), set(arm.state_dict())
    missing, unexpected = sorted(ak - sk), sorted(sk - ak)
    assert missing and unexpected, "the fixture must actually exercise both directions"
    ok_m, bad_m, ok_u, bad_u = classify_plain_to_tul_keys(missing, unexpected, arm)
    assert bad_m == [] and bad_u == []
    assert ok_m == missing and ok_u == unexpected
    # and the shapes of everything they SHARE agree, which is what makes the seed usable
    assert all(seed.state_dict()[k].shape == arm.state_dict()[k].shape for k in sk & ak)


# ── (b) the load itself ──────────────────────────────────────────────────────────

def test_init_from_with_the_flag_loads_the_shared_weights_byte_exact(tmp_path, capsys):
    seed = _plain()
    arm = _ruler()
    fresh = {k: v.clone() for k, v in _ruler().state_dict().items()}   # same build, same seed
    path = _seed_file(tmp_path, seed)

    load_weights_only(path, arm, torch.device("cpu"), allow_plain_to_tul=True)
    got, sd = arm.state_dict(), seed.state_dict()

    shared = set(got) & set(sd)
    assert len(shared) > 100
    for k in shared:
        assert torch.equal(got[k], sd[k]), f"{k} did not come from the seed"
    # the TUL parameters are untouched — still the fresh init of this build
    tul_keys = [k for k in got if k.startswith(("tul.", "tul_spandec."))]
    assert len(tul_keys) >= 19
    for k in tul_keys:
        assert torch.equal(got[k], fresh[k]), f"{k} moved; it must keep its fresh init"

    out = capsys.readouterr().out
    assert "resume_plain_to_tul=True" in out
    assert "tul.E_slot" in out and "compressor" in out, "both lists must be printed IN FULL"


def test_load_checkpoint_with_the_flag_accepts_the_same_seed(tmp_path, capsys):
    """The full-resume door is opened by the same flag and prints the same two lists."""
    arm = _ruler()
    path = _seed_file(tmp_path, _plain())
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    step, _opt, _rebuild, pnames = load_checkpoint(
        path, arm, scaler, torch.device("cpu"), None, allow_plain_to_tul=True)
    assert step == 5000 and _rebuild is False
    assert "prelude.0.attention._impl.compressor.B_a" in pnames
    assert "resume_plain_to_tul=True" in capsys.readouterr().out


# ── (c) the flag OFF is today's behaviour, unchanged ─────────────────────────────

def test_load_checkpoint_without_the_flag_still_refuses_the_plain_seed(tmp_path):
    arm = _ruler()
    path = _seed_file(tmp_path, _plain())
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    with pytest.raises(RuntimeError, match="had no home in the reconstructed model"):
        load_checkpoint(path, arm, scaler, torch.device("cpu"), None)


def test_init_from_without_the_flag_is_byte_identical_to_with_it(tmp_path, capsys):
    """`load_weights_only` never raised on a homeless key and still does not.

    What the flag adds there is the CHECK and the listing, not a different set of loaded
    tensors — so the weights must come out identical either way.
    """
    path = _seed_file(tmp_path, _plain())
    off, on = _ruler(), _ruler()
    load_weights_only(path, off, torch.device("cpu"))
    assert "resume_plain_to_tul" not in capsys.readouterr().out
    load_weights_only(path, on, torch.device("cpu"), allow_plain_to_tul=True)
    a, b = off.state_dict(), on.state_dict()
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), f"{k} differs between flag off and flag on"


# ── (d) the sabotages: everything outside the two classes still raises ───────────

def test_a_missing_NON_tul_tensor_raises_under_the_flag(tmp_path):
    """Sabotage: rename one core tensor in the seed so the arm's copy goes unfilled."""
    seed = _plain()
    sd = {k: v.clone() for k, v in seed.state_dict().items()}
    victim = next(k for k in sd if k.startswith("core.0.") and "compressor" not in k
                  and "indexer" not in k and "comp_norm" not in k)
    sd["core.0.NOT_A_REAL_TENSOR"] = sd.pop(victim)
    ck = {"step": 5000, "next_step": 5000, "model": sd,
          "optimizer": {"state": {}, "param_groups": []},
          "scaler": torch.amp.GradScaler("cpu", enabled=False).state_dict(),
          "rng_cpu": None, "rng_cuda": None}
    path = str(tmp_path / "sabotage.pt")
    torch.save(ck, path)
    with pytest.raises(RuntimeError, match="Refused"):
        load_weights_only(path, _ruler(), torch.device("cpu"), allow_plain_to_tul=True)


def test_an_unexpected_NON_pooled_tensor_raises_under_the_flag(tmp_path):
    seed = _plain()
    sd = {k: v.clone() for k, v in seed.state_dict().items()}
    sd["core.0.mlp.SOMETHING_THE_ARM_DOES_NOT_BUILD"] = torch.zeros(4)
    path = _seed_file(tmp_path, seed)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck["model"] = sd
    torch.save(ck, path)
    with pytest.raises(RuntimeError, match="Refused"):
        load_weights_only(path, _ruler(), torch.device("cpu"), allow_plain_to_tul=True)


def test_a_tul_model_WITHOUT_tg_restrict_may_drop_nothing(tmp_path):
    """Two-sided control: the pooled-branch allowance comes from tg_restrict, not from TUL.

    An unmasked slot arm builds the compressor, so the same seed loads with no unexpected
    key at all — and a fabricated one is refused.
    """
    torch.manual_seed(SEED + 1)
    arm = MORPHTransformer(_tiny(tul=_ruler_tul(tg_restrict=False, tg_restrict_scope="all",
                                                tg_geometry="restrict"))).eval().float()
    assert tg_dropped_attention_prefixes(arm) == ()
    path = _seed_file(tmp_path, _plain())
    load_weights_only(path, arm, torch.device("cpu"), allow_plain_to_tul=True)
    ok_m, bad_m, ok_u, bad_u = classify_plain_to_tul_keys(
        [], ["prelude.0.attention._impl.compressor.B_a"], arm)
    assert bad_u == ["prelude.0.attention._impl.compressor.B_a"]
    assert ok_u == [] and ok_m == [] and bad_m == []


def test_a_prune_mask_buffer_keeps_its_back_compat_exemption():
    """The one pre-existing missing-key exemption must survive the stricter path."""
    ok_m, bad_m, _ok_u, _bad_u = classify_plain_to_tul_keys(
        ["core.0.mlp.gate_up._prune_mask"], [], _ruler())
    assert ok_m == ["core.0.mlp.gate_up._prune_mask"] and bad_m == []
