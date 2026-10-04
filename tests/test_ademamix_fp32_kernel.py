"""`training.ademamix_fused_fp32` (2026-10-04 speed pass): the fp32-state params of
AdEMAMixB1Zero update in one Triton pass (morph/training/ademamix_fp32_kernel.py) instead of
the `_foreach` sequence. The pass must be BIT-IDENTICAL to the sequence: params, both
states and the (gated, in-place) grads, over several steps, for every branch of the update
(eps inside / outside, SNR gate, g_coef, stale-push cap, update clip, weight decay), on
the 32-bit group and on the sub-min_8bit_size tensors of an 8-bit group, with zeros in the
gradient (the sign-0 and nu-0 lanes). The 8-bit params ride the fused kernel either way.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from morph.training.ademamix_b1zero import AdEMAMixB1Zero

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel, CUDA")

SHAPES_32 = [(7,), (33, 65), (1000, 300), (4096, 129)]      # the 32-bit group
SHAPES_8 = [(5, 7), (64,), (4095,), (128, 256)]              # 8-bit group: small ones fall back


def _params(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda s: torch.nn.Parameter(torch.randn(*s, generator=g).cuda())  # noqa: E731
    return [mk(s) for s in SHAPES_32], [mk(s) for s in SHAPES_8]


def _grads(ps, step):
    g = torch.Generator(device="cpu").manual_seed(1000 + step)
    out = []
    for p in ps:
        gr = torch.randn(p.shape, generator=g) * (0.1 + step)
        gr[torch.rand(p.shape, generator=g) < 0.2] = 0.0        # zero lanes
        out.append(gr.cuda())
    return out


def _run(fused_fp32, kw, steps=4):
    p32, p8 = _params(0)
    groups = [{"params": p8, "weight_decay": kw.pop("wd8", 0.1)},
              {"params": p32, "weight_decay": kw.pop("wd32", 0.0), "optim_bits": 32}]
    opt = AdEMAMixB1Zero(groups, lr=3e-3, betas=(0.0, 0.999, 0.999), alpha=8.0,
                         alpha_cap=3.5, t_alpha=10, t_beta3=10, bits=8, fused=True,
                         fused_dynamic_qmap=True, fused_fp32=fused_fp32, **kw)
    ps = p32 + p8
    grads_after = []
    for s in range(steps):
        for p, gr in zip(ps, _grads(ps, s)):
            p.grad = gr
        opt.step()
        grads_after.append([p.grad.clone() for p in ps])
    st = [{k: v.clone() for k, v in opt.state[p].items() if torch.is_tensor(v)} for p in ps]
    return [p.detach().clone() for p in ps], st, grads_after


_BRANCHES = [
    dict(eps_inside=e, g_snr_gate_kappa=k, g_coef=c, stale_push_cap_coord=cap,
         update_clip=clip, wd32=wd)
    for e, k, c, cap, clip, wd in itertools.product(
        (False, True), (0.0, 0.3), (1.0, 0.7), (0.0, 0.5), (0.0, 5.0), (0.0, 0.05))
]


@pytest.mark.parametrize("kw", _BRANCHES[::3] + [_BRANCHES[-1]],
                         ids=lambda kw: "-".join(f"{k}{v}" for k, v in kw.items()))
def test_one_pass_kernel_is_the_foreach_sequence(kw):
    pa, sa, ga = _run(False, dict(kw))
    pb, sb, gb = _run(True, dict(kw))
    for x, y in zip(pa, pb):
        assert torch.equal(x, y)
    for x, y in zip(sa, sb):
        assert x.keys() == y.keys()
        for k in x:
            assert torch.equal(x[k], y[k]), k
    for step_a, step_b in zip(ga, gb):
        for x, y in zip(step_a, step_b):
            assert torch.equal(x, y)


def test_the_lead_arm_branches_take_the_kernel(monkeypatch):
    """The lead arm's knobs (gate 0.3, cap 0.5, clip 5.0, eps outside) route every fp32
    param through the kernel: the `_foreach` sequence is never reached."""
    calls = []
    orig = torch._foreach_addcmul_
    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(torch, "_foreach_addcmul_", spy)
    _run(True, dict(eps_inside=False, g_snr_gate_kappa=0.3, stale_push_cap_coord=0.5,
                    update_clip=5.0), steps=2)
    assert calls == []
    _run(False, dict(eps_inside=False, g_snr_gate_kappa=0.3, stale_push_cap_coord=0.5,
                     update_clip=5.0), steps=2)
    assert len(calls) == 4                # 2 steps x 2 groups with fp32 params


def test_the_key_reaches_the_optimizer():
    import pathlib
    from hydra import compose, initialize_config_dir
    from morph.training.optimizer import create_optimizer
    cdir = str(pathlib.Path(__file__).resolve().parents[1] / "morph" / "configs")
    with initialize_config_dir(config_dir=cdir, version_base=None):
        on = compose(config_name="base", overrides=["training.ademamix_fused_fp32=true",
                                                     "training.optimizer=ademamix_b1zero",
                                                     "training.adam8bit=true"])
        off = compose(config_name="base", overrides=["training.optimizer=ademamix_b1zero",
                                                      "training.adam8bit=true"])
    m = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Embedding(10, 8))
    assert create_optimizer(m, on).fused_fp32 is True
    assert create_optimizer(m, off).fused_fp32 is False
