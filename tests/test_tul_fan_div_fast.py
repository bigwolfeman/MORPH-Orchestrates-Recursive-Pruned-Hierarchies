"""``model.fan_div_fast`` (graph-step, agent twinfan, 2026-10-08): the fan's epi and vol
diversity terms batched over every (pass, stream) in fp32 (morph/model/tul_fan.py,
``fan_epi_term_fast`` / ``fan_vol_term_fast`` / ``ridge_map_fast`` / ``logdet_i_wwt``).

NOT bit-identical to the tree's forms (``fan_epi_term`` / ``fan_vol_term``, ``fixed=True``:
the graph-safe forms the FAST step runs). What each test pins, against an INDEPENDENT fp64
reference of the same formulas written here, at the FAST step's real shapes (B 6, S 64,
M 4 cells, 4 HC streams, C 1024, F 64 reservoir features, ridge 3, eta 30, 319 of 384
slots valid, passes 1..2 charged):
  * the term's value and its gradient into every trajectory entry are within a bound
    that sits far below the change the bf16 carrier's own rounding makes in the fp64
    term (``_FLOOR_FRAC`` of it), and within a small multiple of the tree form's error;
  * every per-pass reading (``epi_t{t}`` / ``vol_t{t}``, instruments on) likewise;
  * the ridge map and the batched log-det against fp64 directly;
  * the edge cases: fewer than two valid slots scores an exact 0 with a zero gradient;
    pad rows get an exact 0 gradient.

CPU, one thread (fp64 references are cheap at these shapes).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from morph.model.tul_fan import (FanReservoir, _cell_readout, fan_epi_term, fan_epi_term_fast,
                                 fan_vol_term, fan_vol_term_fast, logdet_i_wwt, ridge_map,
                                 ridge_map_fast)

B, S, M, N_STREAMS, C, FEAT = 6, 64, 4, 4, 1024, 64
RIDGE, ETA, PASSES = 3.0, 30.0, 2
N_VALID = 319
# The fast term's error must be below this fraction of the bf16-rounding floor.
_FLOOR_FRAC = 0.05


def _valid(n_valid: int = N_VALID) -> torch.Tensor:
    v = torch.zeros(B * S, dtype=torch.bool)
    v[:n_valid] = True
    g = torch.Generator().manual_seed(5)
    return v[torch.randperm(B * S, generator=g)].view(B, S)


def _traj(n: int = 4, seed: int = 0) -> list[torch.Tensor]:
    """Cell states like the loop's: a per-slot base, per-cell deviations ~0.2 of it that
    drift per pass, per-stream offsets; ``[B, S*M, n_streams, C]`` fp32."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(B, S, 1, 1, C, generator=g)
    out = []
    dev = 0.2 * torch.randn(B, S, M, 1, C, generator=g)
    for _ in range(n):
        dev = dev + 0.05 * torch.randn(B, S, M, 1, C, generator=g)
        st = base + dev + 0.02 * torch.randn(B, S, M, N_STREAMS, C, generator=g)
        out.append(st.reshape(B, S * M, N_STREAMS, C).float())
    return out


def _reservoir() -> FanReservoir:
    return FanReservoir(C, FEAT, seed=0)


# ── the fp64 reference: the same formulas, gathered rows, no shared code ─────────────


def _ref_rows(state: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    z = state.double().reshape(B, S, M, N_STREAMS, C).mean(dim=3)      # [B, S, M, C]
    return z[valid]                                                     # [N, M, C]


def _ref_dev(rows: torch.Tensor) -> torch.Tensor:
    dev = rows - rows.mean(dim=1, keepdim=True)
    scale = rows.norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    return dev / scale


def _ref_ridge(h: torch.Tensor, lam: float) -> torch.Tensor:
    f = h.shape[1]
    hs = (h - h.mean(0)) / h.std(0, unbiased=False).clamp_min(1e-6) / math.sqrt(f)
    a = hs.T @ hs + lam * torch.eye(f, dtype=h.dtype)
    return torch.linalg.solve(a, hs.T)


def _ref_reservoir(res: FanReservoir, x: torch.Tensor) -> torch.Tensor:
    h = F.layer_norm(x, (x.shape[-1],))
    h = F.elu(h @ res.w1.double().T)
    return h @ res.w2.double().T


def _ref_epi_pass(state, valid, a):
    dev = _ref_dev(_ref_rows(state, valid))                            # [N, M, C]
    f = a.shape[0]
    sc = []
    for i in range(M):
        w = a @ (dev[:, i] - dev[:, i].mean(0))
        k = torch.eye(f, dtype=w.dtype) + ETA * (w @ w.T)
        sc.append(0.5 * torch.linalg.slogdet(k)[1] / math.log(2) / f)
    return torch.stack(sc).mean()


def _ref_epi(traj, valid, res):
    with torch.no_grad():
        seed = _ref_rows(traj[0], valid).mean(dim=1)
        a = _ref_ridge(_ref_reservoir(res, seed), RIDGE)
    per = [_ref_epi_pass(traj[t], valid, a) for t in range(len(traj))]
    return -torch.stack(per[1:PASSES + 1]).mean(), per


def _ref_vol_pass(state, valid):
    dev = _ref_dev(_ref_rows(state, valid))
    g = dev @ dev.transpose(1, 2)
    ld = torch.linalg.slogdet(torch.eye(M, dtype=g.dtype) + ETA * g)[1]
    return (0.5 * ld / math.log(2) / (M - 1)).mean()


def _ref_vol(traj, valid):
    per = [_ref_vol_pass(traj[t], valid) for t in range(len(traj))]
    return -torch.stack(per[1:PASSES + 1]).mean(), per


# ── helpers ──────────────────────────────────────────────────────────────────────────


def _term_and_grads(fn, traj, device: str = "cpu"):
    tr = [t.to(device).clone().requires_grad_(True) for t in traj]
    stats: dict = {}
    val = fn(tr, stats)
    gr = torch.autograd.grad(val, tr[1:PASSES + 1])
    return (val.detach().double().cpu(), [g.double().cpu() for g in gr],
            {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in stats.items()})


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def _fast_epi(tr, st, valid, res):
    return fan_epi_term_fast(tr, valid, M, PASSES, res, RIDGE, ETA, stats=st)


def _tree_epi(tr, st, valid, res):
    return fan_epi_term(tr, valid, M, PASSES, res, RIDGE, ETA, stats=st, fixed=True)


def _fast_vol(tr, st, valid):
    return fan_vol_term_fast(tr, valid, M, PASSES, ETA, stats=st)


def _tree_vol(tr, st, valid):
    return fan_vol_term(tr, valid, M, PASSES, ETA, stats=st, fixed=True)


def _ref_and_floor(ref_fn, traj):
    """The fp64 term + grads, and the same with every input rounded to bf16 (the floor)."""
    tr = [t.double().requires_grad_(True) for t in traj]
    val, per = ref_fn(tr)
    gr = torch.autograd.grad(val, tr[1:PASSES + 1])
    tb = [t.to(torch.bfloat16).double().requires_grad_(True) for t in traj]
    vb, perb = ref_fn(tb)
    gb = torch.autograd.grad(vb, tb[1:PASSES + 1])
    return (val.detach(), list(gr), [p.detach() for p in per],
            abs(float(vb.detach() - val.detach())), [_rel(b, a) for b, a in zip(gb, gr)],
            [abs(float(pb - p)) for pb, p in zip(perb, per)])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("term", ["epi", "vol"])
def test_term_and_gradients_against_fp64_at_the_real_shape(term, device):
    """The fp64 reference runs on the CPU; the fast and the tree forms on ``device``
    (CUDA: cuSOLVER / cuBLAS, the kernels the FAST step runs)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(0)
    traj, valid, res = _traj(), _valid(), _reservoir()
    vd, rd = valid.to(device), _reservoir().to(device)
    if term == "epi":
        ref_fn = lambda tr: _ref_epi(tr, valid, res)                  # noqa: E731
        fast = lambda tr, st: _fast_epi(tr, st, vd, rd)               # noqa: E731
        tree = lambda tr, st: _tree_epi(tr, st, vd, rd)               # noqa: E731
    else:
        ref_fn = lambda tr: _ref_vol(tr, valid)                       # noqa: E731
        fast = lambda tr, st: _fast_vol(tr, st, vd)                   # noqa: E731
        tree = lambda tr, st: _tree_vol(tr, st, vd)                   # noqa: E731
    rv, rg, rper, floor_v, floor_g, floor_per = _ref_and_floor(ref_fn, traj)
    fv, fg, fst = _term_and_grads(fast, traj, device)
    tv, tg, tst = _term_and_grads(tree, traj, device)
    err_fast, err_tree = abs(float(fv - rv)), abs(float(tv - rv))
    gerr_fast = [_rel(a, b) for a, b in zip(fg, rg)]
    gerr_tree = [_rel(a, b) for a, b in zip(tg, rg)]
    print(f"\n[{term} {device}] value ref {float(rv):.6f} | err fast {err_fast:.3e} tree {err_tree:.3e} "
          f"bf16 floor {floor_v:.3e}\n[{term}] grad rel err fast {gerr_fast} tree {gerr_tree} "
          f"bf16 floor {floor_g}")
    assert floor_v > 0 and min(floor_g) > 0
    assert err_fast <= _FLOOR_FRAC * floor_v
    assert err_fast <= max(10.0 * err_tree, 1e-6 * abs(float(rv)))
    for ef, et, fl in zip(gerr_fast, gerr_tree, floor_g):
        assert ef <= _FLOOR_FRAC * fl
        assert ef <= max(10.0 * et, 1e-5)
    # Every per-pass reading (instruments on): the same bound on the value.
    for t in range(len(traj)):
        e = abs(float(fst[f"{term}_t{t}"]) - float(rper[t]))
        assert e <= _FLOOR_FRAC * floor_per[t] + 1e-7, (t, e, floor_per[t])
        assert set(fst) == set(tst)
    assert fst["repel_terms"] == tst["repel_terms"] == float(PASSES)


def test_ridge_map_fast_against_fp64():
    torch.manual_seed(1)
    h = torch.randn(B * S, FEAT) * 2.0 + 0.5
    n = torch.tensor(N_VALID)
    a = ridge_map_fast(h, RIDGE, n)
    ref = _ref_ridge(h[:N_VALID].double(), RIDGE)
    assert torch.equal(a[:, N_VALID:], torch.zeros_like(a[:, N_VALID:]))
    err = _rel(a[:, :N_VALID].double(), ref)
    tree = ridge_map(h, RIDGE, n)                         # the tree's fp64 QR form
    tree_err = _rel(tree[:, :N_VALID], ref)       # fp32 standardisation, then fp64
    print(f"\nridge rel err fast {err:.3e} tree {tree_err:.3e}")
    assert err < 1e-5 and err <= 10.0 * tree_err


def test_logdet_and_its_gradient_against_fp64():
    torch.manual_seed(2)
    w = (0.3 * torch.randn(2, M, FEAT, C)).requires_grad_(True)
    ld = logdet_i_wwt(w, ETA)
    g, = torch.autograd.grad(ld.sum(), w)
    w64 = w.detach().double().requires_grad_(True)
    k = torch.eye(FEAT, dtype=torch.float64) + ETA * w64 @ w64.transpose(-1, -2)
    ref = torch.linalg.slogdet(k)[1]
    g64, = torch.autograd.grad(ref.sum(), w64)
    assert _rel(ld.double(), ref) < 1e-6
    assert _rel(g.double(), g64) < 1e-4


def test_fewer_than_two_valid_slots_scores_zero_and_pads_get_zero_gradient():
    traj, res = _traj(3), _reservoir()
    one = torch.zeros(B, S, dtype=torch.bool)
    one[2, 5] = True
    tr = [t.clone().requires_grad_(True) for t in traj]
    v = fan_epi_term_fast(tr, one, M, PASSES, res, RIDGE, ETA)
    g = torch.autograd.grad(v, tr[1])[0]
    assert float(v) == 0.0 and torch.equal(g, torch.zeros_like(g))
    valid = _valid()
    for fn in (lambda tr: fan_epi_term_fast(tr, valid, M, PASSES, res, RIDGE, ETA),
               lambda tr: fan_vol_term_fast(tr, valid, M, PASSES, ETA)):
        tr = [t.clone().requires_grad_(True) for t in traj]
        g = torch.autograd.grad(fn(tr), tr[1])[0].view(B, S, M, N_STREAMS, C)
        assert torch.equal(g[~valid], torch.zeros_like(g[~valid]))
        assert bool((g[valid].abs().sum(dim=(-1, -2, -3)) > 0).all())
