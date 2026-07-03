"""TST multi-hot CE + ternary QAT wiring smoke (not quality).

TST: bag labels [B,T,s] must produce init loss ~log(V), not the silent
single-hot bug log(V)/s (ignore/gate_tst_mce_wiring.py).

QAT: apply_ternary_qat registers STE so module.weight is ternary-valued in
forward while the smooth shadow remains trainable.
"""
from __future__ import annotations

import math

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
        n_core=1,
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


def test_tst_bag_labels_use_multi_hot_ce_not_truncated_single_hot():
    torch.manual_seed(0)
    cfg = _tiny()
    model = MORPHTransformer(cfg)
    model.train()

    V = cfg.vocab_size
    log_v = math.log(V)
    s = 6
    log_v_over_s = log_v - math.log(s)
    B, T = 2, 8

    ids_bag = torch.randint(0, V, (B, s * T))
    labels_bag = torch.randint(0, V, (B, T, s))
    ids_ntp = torch.randint(0, V, (B, T))
    labels_ntp = torch.randint(0, V, (B, T))

    out_bag = model(ids_bag, labels=labels_bag, bag_size=s)
    out_ntp = model(ids_ntp, labels=labels_ntp, bag_size=0)
    loss_bag = float(out_bag["loss"].detach())
    loss_ntp = float(out_ntp["loss"].detach())

    # Init CE ≈ log(V); the bug sits at log(V)/s.
    assert log_v - 1.5 < loss_bag < log_v + 1.5, f"TST loss {loss_bag} not ~log(V)={log_v}"
    assert abs(loss_bag - log_v_over_s) > 1.5, (
        f"TST loss {loss_bag} near bug value log(V)/s={log_v_over_s:.3f}"
    )
    assert log_v - 1.5 < loss_ntp < log_v + 1.5, f"NTP loss {loss_ntp} not ~log(V)"

    model.zero_grad(set_to_none=True)
    model(ids_bag, labels=labels_bag, bag_size=s)["loss"].backward()
    grad_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if "embed" in n
    )
    assert grad_ok, "MCE backward produced no finite embedding grad"


def test_ternary_qat_wires_ste_on_backbone_weights():
    torch.manual_seed(0)
    model = MORPHTransformer(_tiny())
    manifest = apply_ternary_qat(model, scope="backbone", threshold=0.5,
                                 scale_mode="symmetric", scale_group="tensor",
                                 scale_dtype="fp16")
    assert manifest["n_modules_ternary"] > 0

    # At least one MLP linear is parametrized and exposes ternary codes.
    found = False
    for mod in model.modules():
        if not parametrize.is_parametrized(mod, "weight"):
            continue
        w = mod.weight.detach()
        # codes in {-1,0,+1} × positive scale → at most 3 unique |values| plus 0
        uniq = torch.unique(w)
        assert torch.isfinite(uniq).all()
        # All nonzero magnitudes share one scale (symmetric tensor mode).
        nz = uniq[uniq != 0]
        if nz.numel() == 0:
            continue
        scale = nz.abs().max()
        assert torch.allclose(nz.abs(), scale.expand_as(nz), rtol=0, atol=1e-5)
        found = True
        break
    assert found, "no ternary-parametrized weight found after apply_ternary_qat"

    ids = torch.randint(0, model.cfg.vocab_size, (2, 8))
    loss = model(ids, labels=ids.clone())["loss"]
    assert torch.isfinite(loss)
    loss.backward()
