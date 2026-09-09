"""training.ternary_scope="backbone_no_core" (arc E20 depthcand-dense-core).

Diagnostic scope: identical module category set to "backbone", but with a path
exclusion so the shared, looped `core.*` blocks (MORPHTransformer.core, the per-pass
map) stay bf16 while prelude/coda backbone linears still ternarize. Tests the
hypothesis that snapping the shared core weights to ternary is what starves the
loop's depth-earning.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.utils.parametrize as parametrize

from morph.model.ternary_qat import apply_ternary_qat
from morph.model.transformer import MORPHConfig, MORPHTransformer


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=128,
        max_seq_len=64,
        n_prelude=1,
        n_core=2,
        n_coda=1,
        mean_depth=1,
        max_depth=1,
        bptt_depth=1,
        channel_dims=(32, 20, 12),
        retention=False,
        bigram_hash_vocab=0,
        use_kernels=False,
        hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def test_backbone_no_core_excludes_only_core_linears():
    torch.manual_seed(0)
    model = MORPHTransformer(_tiny())
    manifest = apply_ternary_qat(
        model,
        scope="backbone_no_core",
        threshold=0.5,
        scale_mode="symmetric",
        scale_group="tensor",
        scale_dtype="fp16",
    )
    assert manifest["n_modules_ternary"] > 0
    assert manifest["n_excluded_by_path"] > 0

    # (a) no Linear/MortarLinear under core.* carries the ternary parametrization.
    core_parametrized = [
        name
        for name, mod in model.named_modules()
        if (name == "core" or name.startswith("core."))
        and parametrize.is_parametrized(mod, "weight")
    ]
    assert core_parametrized == [], f"core module(s) wrongly ternarized: {core_parametrized}"

    # (b) at least one prelude and one coda MLP linear IS parametrized.
    prelude_parametrized = [
        name
        for name, mod in model.named_modules()
        if (name == "prelude" or name.startswith("prelude."))
        and parametrize.is_parametrized(mod, "weight")
    ]
    coda_parametrized = [
        name
        for name, mod in model.named_modules()
        if (name == "coda" or name.startswith("coda."))
        and parametrize.is_parametrized(mod, "weight")
    ]
    assert prelude_parametrized, "no prelude linear ternarized under backbone_no_core"
    assert coda_parametrized, "no coda linear ternarized under backbone_no_core"


def test_backbone_no_core_count_is_strictly_less_than_backbone():
    torch.manual_seed(0)
    model_full = MORPHTransformer(_tiny())
    manifest_full = apply_ternary_qat(
        model_full,
        scope="backbone",
        threshold=0.5,
        scale_mode="symmetric",
        scale_group="tensor",
        scale_dtype="fp16",
    )

    torch.manual_seed(0)
    model_no_core = MORPHTransformer(_tiny())
    manifest_no_core = apply_ternary_qat(
        model_no_core,
        scope="backbone_no_core",
        threshold=0.5,
        scale_mode="symmetric",
        scale_group="tensor",
        scale_dtype="fp16",
    )

    assert manifest_no_core["counts"]["backbone"] < manifest_full["counts"]["backbone"]
    assert manifest_no_core["n_modules_ternary"] < manifest_full["n_modules_ternary"]
    assert manifest_no_core["n_params_ternary"] < manifest_full["n_params_ternary"]


def test_backbone_no_core_forward_and_backward_run():
    torch.manual_seed(0)
    model = MORPHTransformer(_tiny())
    apply_ternary_qat(
        model,
        scope="backbone_no_core",
        threshold=0.5,
        scale_mode="symmetric",
        scale_group="tensor",
        scale_dtype="fp16",
    )
    ids = torch.randint(0, model.cfg.vocab_size, (2, 8))
    out = model(ids, labels=ids.clone())
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()


def test_unknown_scope_still_raises():
    torch.manual_seed(0)
    model = MORPHTransformer(_tiny())
    with pytest.raises(ValueError, match="unknown ternary_scope"):
        apply_ternary_qat(model, scope="not_a_real_scope")
