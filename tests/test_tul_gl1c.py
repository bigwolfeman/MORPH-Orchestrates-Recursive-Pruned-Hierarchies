"""GL1c gates — the warmup-then-mask curriculum's checkpoint-init mechanics.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_tul_gl1c.py -v

GL1c (``morph/configs/tul_gl1c.yaml``) starts a ``tg_restrict: true`` model from the
UNRESTRICTED twin's checkpoint (``checkpoints/morph/gl1-ctrl-s1/step_2000.pt``, a run of
``tul_gl1 + tul.tg_restrict=false``). ``tg_restrict`` REPLACES the HCA/CSA compressed
attention branch (``morph/model/attention.py``: ``if tg_restrict: self.compressor =
None; self.comp_norm = None; self.indexer = None``) rather than building-and-ignoring
it, so the restricted model's parameter set is a SUBSET of the unrestricted
checkpoint's. This file proves two things with evidence, not reading alone:

1. ``load_weights_only`` (morph/training/train.py, the `training.init_from` path) is
   TOLERANT of unexpected (extra) checkpoint tensors and demands only that at least 50%
   of the live model's tensors match — proven directly against ``model.load_state_dict``
   semantics with a small synthetic checkpoint (Part 1), then proven AT REAL SHAPES
   against the actual GL1c seed checkpoint (Part 2, skipped if the file is absent).
2. The optimizer stays FRESH under ``training.init_from`` — the ``elif init_from_path``
   branch in ``train()`` never touches ``optimizer.load_state_dict`` (only the sibling
   ``if resume_path`` branch does), and ``optimizer = create_optimizer(model, cfg)`` runs
   unconditionally before either branch (Part 3, source-structure check + composed-config
   check, since exercising the real branch needs a full trainer run this CPU test does
   not launch).
"""

from __future__ import annotations

import inspect
import os

import pytest
import torch

from morph.model.tul import TULConfig
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.training.train import load_weights_only

CKPT_PATH = "checkpoints/morph/gl1-ctrl-s1/step_2000.pt"

V = 64


# ── Part 1: synthetic subset load, mechanics only (fast, no GPU, no big file) ───────


def _tul(tg_restrict: bool) -> TULConfig:
    return TULConfig(prefix_k=2, slot_id=4, slot_seed="boundary", tg_restrict=tg_restrict,
                     emit_weight=0.0, plast_weight=1.0, token_state_dropout=0.0,
                     sigreg_lambda=0.02, sigreg_slices=64, eval_ablations=True)


def _cfg(tg_restrict: bool) -> MORPHConfig:
    return MORPHConfig(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256,
        context_len=256, n_prelude=2, n_core=0, n_coda=2, mean_depth=2, max_depth=2,
        bptt_depth=1, channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=8, bigram_hash_vocab=V,
        use_kernels=False, hc_use_kernel=False, dropout=0.0,
        retention=True, retention_layers=(1,), retention_chunk=8, retention_carry=True,
        tul=_tul(tg_restrict),
    )


def test_load_weights_only_tolerates_unexpected_keys_with_zero_missing(tmp_path):
    """The unrestricted (donor) model is a strict SUPERSET of the restricted (GL1c)
    model's tensors. Saving the donor and loading it into the restricted model must
    report 0 missing (every restricted tensor exists, same name/shape, in the donor)
    and >0 unexpected (the donor's compressor/comp_norm/indexer tensors the restricted
    model never builds) — and `load_weights_only` must not raise for either."""
    torch.manual_seed(0)
    donor = MORPHTransformer(_cfg(tg_restrict=False))
    ckpt_path = tmp_path / "donor.pt"
    torch.save({"model": donor.state_dict(), "step": 2000}, ckpt_path)

    torch.manual_seed(1)  # different init — a real weight load, not a no-op check
    restricted = MORPHTransformer(_cfg(tg_restrict=True))
    restricted_keys_before = {k: v.clone() for k, v in restricted.state_dict().items()}

    missing, unexpected = load_weights_only(str(ckpt_path), restricted, torch.device("cpu"))

    assert missing == [], f"restricted model has tensors the donor lacks: {missing}"
    assert len(unexpected) > 0, "expected the donor's compressor/indexer tensors to be dropped"
    assert all(("compressor" in k or "comp_norm" in k or "indexer" in k) for k in unexpected), \
        f"unexpected keys are not only the compressed-branch modules: {unexpected[:5]}"

    # And the load actually landed: every restricted tensor changed from its own init
    # to the donor's value (statistically — same-seed collision is not a real concern
    # at these tensor sizes, and every tensor is checked, not a sample).
    after = restricted.state_dict()
    donor_sd = donor.state_dict()
    n_checked = 0
    for k, v_before in restricted_keys_before.items():
        assert torch.equal(after[k], donor_sd[k]), f"{k} did not load the donor's value"
        n_checked += 1
    assert n_checked == len(restricted_keys_before)


def test_load_weights_only_raises_loud_below_the_50pct_match_guard():
    """The ONLY hard guard in load_weights_only: <50% of the live model's tensors
    matched. This is not a "zero missing keys" demand — a majority-mismatched load is
    tolerated right up to that boundary and only then refused, loudly."""
    torch.manual_seed(0)
    tiny_incompatible = MORPHTransformer(_cfg(tg_restrict=True))
    # Corrupt every key name so nothing can match.
    bad_ckpt = {"model": {f"not.a.real.key.{k}": v
                          for k, v in tiny_incompatible.state_dict().items()},
               "step": 0}
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bad.pt")
        torch.save(bad_ckpt, p)
        torch.manual_seed(2)
        victim = MORPHTransformer(_cfg(tg_restrict=True))
        with pytest.raises(RuntimeError, match="Seed/model key structure mismatch"):
            load_weights_only(p, victim, torch.device("cpu"))


# ── Part 2: real-shape load against the actual GL1c seed checkpoint ─────────────────


@pytest.mark.skipif(not os.path.isfile(CKPT_PATH),
                    reason=f"seed checkpoint not present at {CKPT_PATH}")
def test_real_shape_load_zero_missing_and_reports_dropped_count():
    """The real GL1c init: build the restricted model at REAL dims (tul_gl1 config,
    quantization applied BEFORE loading — the QAT scar in lab/divergence/_build.py:
    ternary/embed-int6 parametrization renames `w.weight` to
    `w.parametrizations.weight.original`, so an unquantized model's keys don't match
    the checkpoint's), then run the exact `training.init_from` load path against the
    real gl1-ctrl-s1/step_2000.pt checkpoint (~1.9 GB, loaded ONCE, CPU only).

    tests/test_tul_gl1.py::test_the_control_is_config_matched_but_NOT_parameter_matched
    already measured this SAME config's parameter-set gap (compressor/comp_norm/indexer
    under `.attention._impl.`) at 60 tensors / 2.037 M params / +0.98%. This test is the
    same claim against the checkpoint FILE rather than a freshly-initialised twin model
    — the number it prints is the one that actually matters for the run.
    """
    from lab.divergence._build import build_cfg, build_model

    cfg = build_cfg("tul_gl1", ["model.hc_use_kernel=false"])
    model, _tul_rt = build_model(cfg, device="cpu")
    model_keys = set(model.state_dict().keys())

    missing, unexpected = load_weights_only(CKPT_PATH, model, torch.device("cpu"))
    n_loaded = len(model_keys) - len(missing)

    print(f"\n[real-shape load] model tensors={len(model_keys)} "
          f"matched={n_loaded} missing={len(missing)} unexpected={len(unexpected)}")

    assert missing == [], (
        f"the restricted model has {len(missing)} tensors the unrestricted checkpoint "
        f"lacks — the subset claim is FALSE at real shapes: {missing[:10]}")
    assert n_loaded == len(model_keys), "every restricted-model tensor must match"
    assert len(unexpected) > 0, (
        "expected the donor's compressed-branch tensors to be reported unexpected; "
        "if this is 0 the checkpoint is no longer a superset (config drift — the run's "
        "own banner is the source of truth, this comment is not)")
    assert all(("compressor" in k or "comp_norm" in k or "indexer" in k)
              and ".attention._impl." in k for k in unexpected), \
        f"the dropped tensors are not only the HCA/CSA compressed branch: {sorted(unexpected)[:8]}"
    # Loose band around the measured 60/2.037M figure — real not because the count is
    # expected to drift, but because a change here should fail LOUD with the actual
    # number in the assertion message, not silently pass a wide range.
    assert 50 <= len(unexpected) <= 70, (
        f"unexpected-tensor count moved to {len(unexpected)} (measured elsewhere: 60); "
        f"update the config comment in tul_gl1c.yaml if this is real drift, not a bug")


# ── Part 3: the optimizer is fresh under init_from ───────────────────────────────────


def test_init_from_branch_never_touches_optimizer_state():
    """Source-structure check: `elif init_from_path:` (not a second `if`) means the
    optimizer-restore block above it — the one that calls `optimizer.load_state_dict`
    — is mutually exclusive with the init_from path. `create_optimizer` runs
    unconditionally before both, so init_from always inherits a config-fresh optimizer.
    """
    import morph.training.train as train_mod

    src = inspect.getsource(train_mod.main)
    resume_idx = src.index("if resume_path and os.path.isfile(resume_path):")
    # second occurrence, inside the "build things" section (the first is the earlier
    # ckpt_pnames pre-scan) — find the one that owns the elif init_from_path branch.
    resume_idx = src.index("if resume_path and os.path.isfile(resume_path):", resume_idx + 1)
    elif_idx = src.index("elif init_from_path:", resume_idx)
    create_opt_idx = src.index("optimizer = create_optimizer(model, cfg)")

    branch_block = src[resume_idx:elif_idx]
    init_from_block = src[elif_idx:elif_idx + 400]

    assert create_opt_idx < resume_idx, (
        "create_optimizer must run before the resume/init_from branch so init_from "
        "inherits a config-fresh optimizer")
    assert "optimizer.load_state_dict" in branch_block, (
        "expected the resume branch (not elif init_from_path) to own optimizer restore")
    assert "optimizer.load_state_dict" not in init_from_block, (
        "training.init_from must NOT restore optimizer state — the curriculum's whole "
        "point is a fresh AdEMAMix state under the new (restricted) architecture")
    assert "load_weights_only" in init_from_block


def test_gl1c_config_composes_init_from_with_resume_unset():
    """`resume` takes precedence over `init_from` if both are set (train.py comment);
    GL1c must not set `resume`, or the checkpoint-init path silently changes."""
    from hydra import compose, initialize_config_dir

    cd = os.path.abspath("morph/configs")
    with initialize_config_dir(version_base=None, config_dir=cd):
        cfg = compose(config_name="tul_gl1c")

    assert cfg.training.resume is None, (
        "tul_gl1c sets training.resume — this would silently take precedence over "
        "init_from and load the optimizer/step too")
    assert cfg.training.init_from == CKPT_PATH
    assert cfg.tul.tg_restrict is True
    assert cfg.model.n_core == 0
    assert cfg.training.ademamix_t_beta3 == 4500
    assert cfg.wandb.name == "gl1c"
