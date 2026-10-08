"""`training.capturable_optimizer` (graph-captured training step, tasks 1.4 / 1.5): the
optimizer step of the winner config runs with no host read and every per-step scalar on
the device, so it can be recorded as a CUDA graph and replayed.

The contract, pinned here bit for bit against the eager path (GradScaler skip semantics,
host scalars):
  * ONE captured step (clip + found-inf flag + optimizer step) replayed over several steps
    with new grads and a moving lr gives the same params, optimizer state, gated grads and
    step counter as the eager optimizer stepping on the same grads;
  * a step whose grads hold an inf or a NaN changes nothing: params, state and the step
    counter all stay (the eager reference does not call step() at all on that step);
  * the device schedule table is the eager schedule, rounded to fp32, for every step.
"""
from __future__ import annotations

import pathlib

import pytest
import torch
import torch.nn as nn

from morph.training.ademamix_b1zero import AdEMAMixB1Zero

CFG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "morph" / "configs")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels, CUDA")


def _cfg(*overrides):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        return compose(config_name="lxtul_pointer", overrides=list(overrides))


class _Tiny(nn.Module):
    """One tensor per optimizer path of the winner: a fused 8-bit weight (>= 4096), a
    sub-4096 decay weight (fp32 kernel, 8-bit group), and the no-decay 32-bit group
    (bias, norm, embedding)."""

    def __init__(self):
        super().__init__()
        self.big = nn.Linear(96, 64)            # weight 6144 -> fused 8-bit; bias no-decay
        self.small = nn.Linear(64, 32, bias=False)   # 2048 -> fp32 kernel in the 8-bit group
        self.embed = nn.Embedding(50, 16)       # no-decay, 32-bit
        self.norm = nn.LayerNorm(16)


def _build(capturable: bool, *overrides):
    from morph.training.optimizer import create_optimizer
    torch.manual_seed(0)
    m = _Tiny().cuda()
    cfg = _cfg(f"training.capturable_optimizer={str(capturable).lower()}", *overrides)
    return m, create_optimizer(m, cfg)


def _grads(params, step, poison=None):
    g = torch.Generator(device="cpu").manual_seed(1000 + step)
    out = []
    for p in params:
        gr = torch.randn(p.shape, generator=g) * (0.05 + 0.3 * step)
        gr[torch.rand(p.shape, generator=g) < 0.2] = 0.0          # zero lanes
        out.append(gr.cuda())
    if poison is not None:
        out[step % len(out)].view(-1)[3] = poison
    return out


def _snapshot(opt, params):
    st = [{k: v.clone() for k, v in opt.state[p].items() if torch.is_tensor(v)}
          for p in params]
    steps = [g.get("step", 0) for g in opt.state_dict()["param_groups"]]
    return [p.detach().clone() for p in params], st, steps


def _assert_same(a, b, what):
    (pa, sa, ka), (pb, sb, kb) = a, b
    assert ka == kb, f"{what}: step counters {ka} vs {kb}"
    for i, (x, y) in enumerate(zip(pa, pb)):
        assert torch.equal(x, y), f"{what}: param {i}"
    for i, (x, y) in enumerate(zip(sa, sb)):
        assert x.keys() == y.keys(), f"{what}: state keys {i}"
        for k in x:
            assert torch.equal(x[k], y[k]), f"{what}: state {i}.{k}"


# Per step: (lr, poison). lr moves every step (the warmup ramp); an inf and a NaN step.
SCHEDULE = [(1e-3, None), (2e-3, None), (3e-3, None), (4e-3, float("inf")),
            (5e-3, None), (6e-3, None), (7e-3, float("nan")), (8e-3, None), (8e-3, None)]
# Short alpha / beta3 horizons so the schedule crosses both inside the run.
HORIZONS = ("training.ademamix_t_alpha=3", "training.ademamix_t_beta3=5")


def _eager_reference(schedule):
    """GradScaler semantics: per-element non-finite check, clip, step only when finite."""
    m, opt = _build(False, *HORIZONS)
    params = [p for g in opt.param_groups for p in g["params"]]
    snaps, gated = [], []
    for k, (lr, poison) in enumerate(schedule):
        for pg in opt.param_groups:
            pg["lr"] = lr * pg.get("lr_mult", 1.0)
        for p, gr in zip(params, _grads(params, k, poison)):
            p.grad = gr
        found_inf = not all(bool(torch.isfinite(p.grad).all()) for p in params)
        nn.utils.clip_grad_norm_(params, 1.0)
        if not found_inf:
            opt.step()
        snaps.append(_snapshot(opt, params))
        gated.append([p.grad.clone() for p in params])
    return snaps, gated


@cuda
def test_captured_step_matches_eager_and_skips_nonfinite():
    ref, ref_gated = _eager_reference(SCHEDULE)

    m, opt = _build(True, *HORIZONS)
    assert opt.capturable
    params = [p for g in opt.param_groups for p in g["params"]]
    for p in params:                                  # static grad buffers for the graph
        p.grad = torch.zeros_like(p)
    opt.init_state()

    # Warm the Triton specialisations on a throwaway twin (same shapes, same config) on a
    # side stream, so nothing compiles inside the capture.
    _, warm = _build(True, *HORIZONS)
    wparams = [p for g in warm.param_groups for p in g["params"]]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for p in wparams:
            p.grad = torch.randn_like(p)
        warm.mark_found_inf(nn.utils.clip_grad_norm_(wparams, 1.0))
        warm.step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        total = nn.utils.clip_grad_norm_(params, 1.0)
        opt.mark_found_inf(total)
        opt.step()

    for k, (lr, poison) in enumerate(SCHEDULE):
        before = _snapshot(opt, params)
        for pg in opt.param_groups:                   # host writes, outside the graph
            pg["lr"] = lr * pg.get("lr_mult", 1.0)
        opt.write_step_scalars()
        for p, gr in zip(params, _grads(params, k, poison)):
            p.grad.copy_(gr)
        graph.replay()
        torch.cuda.synchronize()
        now = _snapshot(opt, params)
        _assert_same(now, ref[k], f"step {k}")
        if poison is not None:
            assert bool(opt.found_inf.item() == 1.0)
            _assert_same(now, before, f"non-finite step {k} changed the optimizer")
        else:
            assert bool(opt.found_inf.item() == 0.0)
            for i, (x, y) in enumerate(zip(params, ref_gated[k])):
                assert torch.equal(x.grad, y), f"step {k}: gated grad {i}"
    # The counter counted only the finite steps.
    assert now[2] == [len([1 for _, q in SCHEDULE if q is None])] * len(opt.param_groups)


@cuda
def test_first_step_nonfinite_then_zero_state_init_is_exact():
    """The capturable path allocates zero state on the first (skipped) step and always
    dequantises; the eager path has no state until its first executed step and takes the
    init branch then. The bits must agree."""
    schedule = [(1e-3, float("inf")), (2e-3, None), (3e-3, None)]
    ref, _ = _eager_reference(schedule)
    m, opt = _build(True, *HORIZONS)
    params = [p for g in opt.param_groups for p in g["params"]]
    for k, (lr, poison) in enumerate(schedule):
        for pg in opt.param_groups:
            pg["lr"] = lr * pg.get("lr_mult", 1.0)
        opt.write_step_scalars()
        for p, gr in zip(params, _grads(params, k, poison)):
            p.grad = gr
        opt.mark_found_inf(nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        if k > 0:          # the eager reference has no state at all after a skipped step 0
            _assert_same(_snapshot(opt, params), ref[k], f"step {k}")


@cuda
def test_step_refuses_a_stale_lr():
    m, opt = _build(True)
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    for pg in opt.param_groups:
        pg["lr"] = 0.5
    with pytest.raises(RuntimeError, match="write_step_scalars"):
        opt.step()


@cuda
def test_the_key_reaches_the_optimizer():
    assert _build(True)[1].capturable is True
    assert _build(False)[1].capturable is False
    from morph.training.optimizer import create_optimizer
    with pytest.raises(ValueError, match="ademamix_b1zero only"):
        create_optimizer(_Tiny().cuda(), _cfg("training.capturable_optimizer=true",
                                              "training.optimizer=adamw"))


def test_device_table_is_the_eager_schedule_in_fp32():
    """Every step t, past the table's end too (the device index clamps to the last row),
    reads the fp32 rounding of what the eager step computes in double."""
    import numpy as np
    tr = _cfg().training
    group = {"betas": (0.0, float(tr.ademamix_beta2), float(tr.beta3)),
             "alpha": float(tr.ademamix_alpha), "alpha_cap": float(tr.ademamix_alpha_cap),
             "t_alpha": int(tr.ademamix_t_alpha), "t_beta3": int(tr.ademamix_t_beta3),
             "beta3_warmup_start": float(tr.ademamix_beta3_warmup_start),
             "eps": 1e-8, "weight_decay": float(tr.weight_decay)}
    table = AdEMAMixB1Zero._sched_table(AdEMAMixB1Zero, group)
    n = table.shape[0]
    assert n > max(group["t_alpha"], group["t_beta3"])
    b2, eps, wd = group["betas"][1], group["eps"], group["weight_decay"]
    for t in range(1, n + 2000):
        a_t, _, b3_t = AdEMAMixB1Zero._sched(t, group)
        bc2 = 1.0 - b2 ** t
        # Slots 1..9 of the schedule vector, in kernel order (ademamix_b1zero._SCHED_*).
        want = np.array([b2, b3_t, a_t, eps, bc2, wd, 1.0 - b3_t, 1.0 - b2, 1.0 / bc2],
                        dtype=np.float32)
        got = table[min(t, n - 1)].numpy()
        assert np.array_equal(got, want), f"t={t}: {got} vs {want}"


def _run_steps(opt, params, schedule, ks, capturable):
    """The trainer's per-step order on either path, with no graph."""
    for k in ks:
        lr, poison = schedule[k]
        for pg in opt.param_groups:
            pg["lr"] = lr * pg.get("lr_mult", 1.0)
        if capturable:
            opt.write_step_scalars()
        for p, gr in zip(params, _grads(params, k, poison)):
            p.grad = gr
        if capturable:
            opt.mark_found_inf(nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
        else:
            found_inf = not all(bool(torch.isfinite(p.grad).all()) for p in params)
            nn.utils.clip_grad_norm_(params, 1.0)
            if not found_inf:
                opt.step()


@cuda
def test_checkpoint_round_trip_resumes_on_either_path():
    """A capturable checkpoint (taken after a skipped step) resumes on the capturable path
    AND on the eager path, and both continue bit for bit as the uninterrupted eager run:
    the device step counter (finite steps only) travels through state_dict / load."""
    import copy
    schedule = SCHEDULE[:7]                     # the inf step (3) lies before the save
    ref, _ = _eager_reference(schedule)
    m, opt = _build(True, *HORIZONS)
    params = [p for g in opt.param_groups for p in g["params"]]
    _run_steps(opt, params, schedule, range(4), True)
    saved = copy.deepcopy(opt.state_dict())
    weights = [p.detach().clone() for p in params]
    assert [g["step"] for g in saved["param_groups"]] == [3, 3]
    for capturable in (True, False):
        m2, opt2 = _build(capturable, *HORIZONS)
        params2 = [p for g in opt2.param_groups for p in g["params"]]
        with torch.no_grad():
            for p, w in zip(params2, weights):
                p.copy_(w)
        opt2.load_state_dict(copy.deepcopy(saved))
        _run_steps(opt2, params2, schedule, range(4, 7), capturable)
        _assert_same(_snapshot(opt2, params2), ref[6], f"resumed capturable={capturable}")


@cuda
def test_capturable_checkpoint_loads_into_a_scaler_run():
    """A capturable run saves a DISABLED GradScaler, whose state is {}. Loading that into a
    key-off run (enabled scaler) must not raise; the scaler starts at its default. CUDA
    only: without CUDA, GradScaler disables itself and the enabled case cannot be built."""
    import os
    import tempfile
    from morph.training.train import load_checkpoint, save_checkpoint
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with tempfile.TemporaryDirectory(prefix="cap_scaler_") as tmp:
        path = os.path.join(tmp, "cap.pt")
        save_checkpoint(path, 3, model, opt, torch.amp.GradScaler("cuda", enabled=False),
                        pruning=None, next_step=4)
        for enabled in (True, False):
            scaler = torch.amp.GradScaler("cuda", enabled=enabled)
            step, *_ = load_checkpoint(path, nn.Linear(4, 4), scaler, torch.device("cpu"),
                                       pruning=None)
            assert step == 4
            if enabled:
                assert scaler.state_dict()["scale"] == 2.0 ** 16
