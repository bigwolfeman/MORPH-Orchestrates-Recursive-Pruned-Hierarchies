"""Plan C: expanded hyper-connections (xHC, arXiv 2607.14530) in the slot loop.

Files: morph/model/tul_xhc.py (`XHCResidual`, `xhc_select`, `causal_slot_conv`,
`gram_schmidt_components`), morph/model/transformer.py (the build, `_xhc_expand`,
`_xhc_cells`, the `_forward_tul` exit, the gain hinge's routing replay, the
`_core_region` refusal), morph/model/mhc.py (`MORPHBlock.forward`'s `xhc_valid` /
`xhc_route`), morph/model/tul.py (`TULConfig._check_xhc`), morph/training/tul_setup.py,
morph/configs/tul_slot_spandec_strict_e4probe_fp01_xhc{,_ta}.yaml.
Note: .agents/notes/proposed/architecture/2026-09-26-plan-c-xhc-slot-loop.md

What each test pins:
  * OFF (`xhc_streams: 0`) is bit-identical to the base tree the arm was built on: loss,
    grads, params and eval logits, same seed, `torch.equal` (two subprocesses, one per tree);
  * the sparse routed update equals an independently written DENSE masked reference (an
    N x N mixer that is the identity off the active set), with and without the temporal
    components, and the idle streams are EXACTLY the input;
  * the router is deterministic, stable on ties, and chooses different streams on
    different passes of the same slot;
  * the temporal components are causal on the slot axis (slot j reaches j..j+7 with kernels
    {2, 4, 8} and never j-1) and a pad slot reaches no valid slot;
  * temporal augmentation with a zero write is C1;
  * the entry holds the 4-stream carrier in streams 0-3 and exact zeros in 4-15;
  * the exit writes cell j from streams 4j..4j+3 and nothing else;
  * the gain hinge replays the operating point's routing;
  * both configs compose, differ from fp01 by the stated keys, build, train a step, and
    every new parameter gets a gradient; the refusals hold; a checkpoint round-trips.

C1b (`model.slot_state_renorm` + `model.slot_gain_renorm`, the fix for C1/C2's detonation):
  * `slot_gain_renorm` off is bit-identical to the tree before C1b (`_BASE_C1B`), with the
    renorm off AND on (two subprocesses, `torch.equal` on loss, grads, params, logits);
  * on a tiny xHC model whose core writes are scaled up, today's carrier grows ~10x per
    pass and the renorm holds every slot of the 16-stream carrier at its entry norm;
  * the hinge under `slot_gain_renorm` reads R(f): on the linear map f(h) = 3h it reads
    3 without the knob and the renormed map's ~1 with it, and on the grown model its raw
    instrument equals today's reading bit for bit while its own reading is below target;
  * the knob refuses to be set where it would do nothing; the C1b arms compose, differ from
    C1 / C2 by the two keys, build, and train a step.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest
import torch

from morph.model.hyper_connections import HyperConnectionResidual
from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import slot_layout_from_ids
from morph.model.tul_xhc import (
    XHCResidual,
    causal_slot_conv,
    gram_schmidt_components,
    xhc_select,
)
from test_tul_lxtul_e import _D_FF, _tc
from test_tul_strict_geometry import _ids, _rule, _runtime, _spec, _tiny

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE = "21e9705"          # the commit plan C was built on (see the OFF test's docstring)
_N, _K, _M = 16, 4, 2


def _pack4(B: int = 2, seed: int = 0):
    ids = _ids(B=B, seed=seed)
    inp, lab, layout, _ = slot_layout_from_ids(ids, _rule(), _spec(prefix_k=4))
    return ids, inp, lab, layout


def _xm(kernels=(), seed: int = 1234, dropout: float = 0.0, gain_target: float = 0.9,
        mkw: dict | None = None, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tc = _tc(prefix_k=4, xhc_streams=_N, xhc_temporal_kernels=tuple(kernels), **kw)
    m = MORPHTransformer(_tiny(tul=tc, d_ff=_D_FF, dropout=dropout,
                               core_fixed_point_lambda=0.1, slot_gain_lambda=100.0,
                               slot_gain_target=gain_target, slot_cot_clip=4.0,
                               **(mkw or {})))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.float()


def _xhc_modules(m: MORPHTransformer) -> list[XHCResidual]:
    return [mod for blk in m.core for mod in (blk.mrr_attn, blk.mrr_mlp)]


# ── OFF is the base tree ──────────────────────────────────────────────────────────────

_OFF_SCRIPT = r'''
import sys
import numpy as np
import torch
import morph
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids
import os
assert os.path.realpath(morph.__file__).startswith(os.path.realpath(sys.argv[2])), morph.__file__
V = 64
lut = np.zeros(V, dtype=bool)
lut[[0, 10, 11]] = True
rule = BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)
spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
rng = np.random.default_rng(0)
ids = rng.integers(5, V, size=(2, 120))
ids[ids == 4] = 5
ids[:, ::8] = 10
inp, lab, layout, _ = slot_layout_from_ids(ids.astype(np.int64), rule, spec)
tc = TULConfig(prefix_k=2, slot_id=4, tg_restrict=True, tg_restrict_scope="all",
               emit_weight=0.0, token_state_dropout=0.0, mux_beta=0.0, tg_geometry="strict",
               spandec=False, spandec_parallel=True, code_enum_k=4, plast_weight=1.0)
cfg = MORPHConfig(d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256,
                  context_len=256, n_prelude=2, n_core=2, n_coda=2, mean_depth=2,
                  max_depth=3, bptt_depth=3, channel_dims=(32, 20, 12), compression=2,
                  csa_compress_ratio=4, hca_compress_ratio=8, top_k=8, window_size=16,
                  retention=False, bigram_hash_vocab=V, use_kernels=False,
                  hc_use_kernel=False, dropout=0.1, d_ff=96, core_fixed_point_lambda=0.1,
                  slot_gain_lambda=100.0, slot_cot_clip=4.0, tul=tc)
torch.manual_seed(1234)
m = MORPHTransformer(cfg).float().train()
with torch.no_grad():
    m.embed.bigram.lambdas.fill_(0.5)
torch.manual_seed(5)
out = m(inp, labels=lab, slot_layout=layout)
out["loss"].backward()
res = {"loss": out["loss"].detach(),
       "grads": {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None},
       "params": {n: p.detach().clone() for n, p in m.named_parameters()}}
m.eval()
with torch.no_grad():
    res["logits"] = m(inp, slot_layout=layout)["logits"]
torch.save(res, sys.argv[1])
'''


def _run_off_fixture(tree: str, out_path: str) -> dict:
    env = dict(os.environ, PYTHONPATH=tree, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-c", _OFF_SCRIPT, out_path, tree], cwd=tree, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"fixture failed in {tree}:\n{r.stderr[-3000:]}"
    return torch.load(out_path)


def test_off_is_bit_identical_to_the_base_tree(tmp_path):
    """The fp01 recipe's tiny twin (strict, parallel head, code_enum_k 4, fixed-point 0.1,
    gain hinge 100, cot clip 4, dropout 0.1) run by the BASE tree (`git archive _BASE`) and
    by this tree, same seed: loss, every grad, every param and the eval logits are
    `torch.equal`. `_BASE` is the commit plan C was built on; once the arm is merged and
    other work changes the forward, move `_BASE` to the merge's first parent."""
    ok = subprocess.run(["git", "-C", _REPO, "cat-file", "-e", f"{_BASE}^{{commit}}"],
                        capture_output=True)
    if ok.returncode != 0:
        pytest.skip(f"base commit {_BASE} is not in this clone")
    base = tmp_path / "base"
    base.mkdir()
    arc = subprocess.run(["git", "-C", _REPO, "archive", _BASE, "morph"],
                         capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(base)], input=arc, check=True)
    ref = _run_off_fixture(str(base), str(tmp_path / "base.pt"))
    got = _run_off_fixture(_REPO, str(tmp_path / "head.pt"))
    assert torch.equal(ref["loss"], got["loss"])
    assert torch.equal(ref["logits"], got["logits"])
    assert ref["params"].keys() == got["params"].keys()
    assert ref["grads"].keys() == got["grads"].keys() and len(got["grads"]) > 50
    for n in ref["params"]:
        assert torch.equal(ref["params"][n], got["params"][n]), n
    for n in ref["grads"]:
        assert torch.equal(ref["grads"][n], got["grads"][n]), n


def test_off_builds_nothing():
    torch.manual_seed(1)
    m = MORPHTransformer(_tiny(tul=_tc(), d_ff=_D_FF))
    assert m._xhc_streams == 0 and not m._xhc_temporal
    assert all(isinstance(mod, HyperConnectionResidual)
               for blk in m.core for mod in (blk.mrr_attn, blk.mrr_mlp))
    assert not any(isinstance(mod, XHCResidual) for mod in m.modules())


# ── the routed update against a dense masked reference ──────────────────────────────


def _cayley_ref(A: torch.Tensor, alpha: float) -> torch.Tensor:
    Bm = 0.5 * alpha * (A - A.T)
    eye = torch.eye(A.shape[0], dtype=A.dtype)
    return (eye + Bm) @ torch.linalg.inv(eye - Bm)


def _dense_reference(mod: XHCResidual, h: torch.Tensor, W: torch.Tensor,
                     valid: torch.Tensor | None) -> torch.Tensor:
    """The update written from the module doc in fp64, one position at a time, as a DENSE
    N x N mixer that is the identity off the active set plus a dense write vector that is
    zero off it. Shares no code with the module (true Cayley by inverse, classical
    Gram-Schmidt, an explicit convolution loop, a Python sort for the router)."""
    Bt, S, N, C = h.shape
    k, m_ = mod.k, mod.m
    hd = h.double()
    xbar = torch.zeros(Bt, S, C, dtype=torch.float64)
    plan = {}
    for b in range(Bt):
        for s in range(S):
            X = hd[b, s]
            xf = X.reshape(-1)
            rms = math.sqrt(float(xf.pow(2).mean()) + mod.eps)
            pre = mod.w_pre.double() @ xf / rms + mod.pre_bias.double()
            hp = torch.softmax(pre / mod.tau, 0)
            xbar[b, s] = (hp[:, None] * X).sum(0)
            sc = torch.sigmoid(mod.w_route.double() @ xf / rms + mod.route_bias.double())
            cand = sorted(range(N - m_), key=lambda i: (-float(sc[i]), i))[: k - m_]
            A = list(range(m_)) + [i + m_ for i in cand]
            gate = [1.0] * m_ + [float(sc[i]) for i in cand]
            plan[b, s] = (A, gate)
    y = torch.tanh(xbar @ W.double())                             # the position-wise F
    comps = None
    if mod.r:
        yv = y * (valid.double()[..., None] if valid is not None else 1.0)
        comps = torch.zeros(Bt, S, mod.r, C, dtype=torch.float64)
        for b in range(Bt):
            for s in range(S):
                raw = []
                for w in mod.conv_w:
                    kap = w.shape[-1]
                    c = torch.zeros(C, dtype=torch.float64)
                    for tau_ in range(kap):
                        if s - tau_ >= 0:
                            c = c + w[:, 0, kap - 1 - tau_].double() * yv[b, s - tau_]
                    raw.append(c)
                basis = [yv[b, s]]
                for r_, c in enumerate(raw):
                    u = c.clone()
                    for q in basis:                                   # classical GS
                        nq = float(q @ q)
                        if nq > 0:
                            u = u - (float(c @ q) / nq) * q
                    basis.append(u)
                    comps[b, s, r_] = u
    out = torch.empty_like(hd)
    for b in range(Bt):
        for s in range(S):
            A, gate = plan[b, s]
            XA = hd[b, s, A]
            xaf = XA.reshape(-1)
            rms_a = math.sqrt(float(xaf.pow(2).mean()) + mod.eps)
            act = mod.w_act.double() @ xaf / rms_a + mod.act_bias.double()
            kk = k * k
            Hres = _cayley_ref(act[:kk].reshape(k, k), mod.cayley_alpha)
            Hpost = torch.softmax(act[kk:2 * kk].reshape(k, k) / mod.tau, dim=0).sum(1)
            M = torch.eye(N, dtype=torch.float64)
            wvec = torch.zeros(N, dtype=torch.float64)
            aug = torch.zeros(N, C, dtype=torch.float64)
            for i, ai in enumerate(A):
                M[ai, ai] = 0.0
                for j, aj in enumerate(A):
                    M[ai, aj] = Hres[i, j]
                wvec[ai] = gate[i] * Hpost[i]
                if mod.r:
                    haug = torch.tanh(act[2 * kk:].reshape(k, mod.r))
                    aug[ai] = gate[i] * (haug[i][:, None] * comps[b, s]).sum(0)
            out[b, s] = M @ hd[b, s] + wvec[:, None] * y[b, s] + aug
    return out, plan


def _live_module(kernels=(), C: int = 12, seed: int = 0) -> XHCResidual:
    g = torch.Generator().manual_seed(seed)
    mod = XHCResidual(C, _N, _K, _M, temporal_kernels=kernels, generator=g)
    with torch.no_grad():
        # Weights a trained model could hold: large enough that H_res is far from I, the
        # router's scores are spread, and every aug weight differs from its init.
        for p in mod.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (1.0 / math.sqrt(max(p.shape[-1], 1))))
    return mod


@pytest.mark.parametrize("kernels", [(), (2, 4, 8)])
def test_routed_update_equals_a_dense_masked_reference(kernels):
    torch.manual_seed(3)
    C, Bt, S = 12, 2, 9
    mod = _live_module(kernels, C)
    h = torch.randn(Bt, S, _N, C)
    h[:, :, 10:] *= 0.1                          # uneven stream sizes, as at the entry
    W = torch.randn(C, C) / math.sqrt(C)
    valid = torch.ones(Bt, S, dtype=torch.bool)
    valid[1, 6:] = False                          # trailing pads on row 1
    kw = {"valid": valid} if kernels else {}
    got = mod(h, lambda x: torch.tanh(x @ W), **kw)
    ref, plan = _dense_reference(mod, h, W, valid if kernels else None)
    torch.testing.assert_close(got.double(), ref, rtol=1e-4, atol=1e-5)
    n_idle = 0
    for (b, s), (A, _g) in plan.items():
        idle = [i for i in range(_N) if i not in A]
        n_idle += len(idle)
        assert torch.equal(got[b, s, idle], h[b, s, idle]), (b, s)
        assert not torch.equal(got[b, s, A], h[b, s, A]), (b, s)
    assert n_idle == Bt * S * (_N - _K)
    # the routed set is not the same at every position (the reference would pass a
    # module that always wrote streams 0-3 if every position chose them)
    assert len({tuple(A) for A, _g in plan.values()}) > 1


def test_select_is_stable_on_ties_and_gates_the_routed_write():
    score = torch.tensor([[0.5] * 14, [0.1, 0.9, 0.9, 0.2] + [0.0] * 10])
    idx, gate = xhc_select(score, 2, 2)
    assert idx.tolist() == [[0, 1, 2, 3], [0, 1, 3, 4]]
    assert gate.tolist() == [[1.0, 1.0, 0.5, 0.5], [1.0, 1.0, pytest.approx(0.9),
                                                    pytest.approx(0.9)]]


def test_the_router_gets_a_gradient_through_the_gate():
    mod = _live_module((), C=8, seed=4)
    h = torch.randn(2, 5, _N, 8)
    mod(h, lambda x: x.tanh()).pow(2).sum().backward()
    assert float(mod.w_route.grad.abs().sum()) > 0 and float(mod.route_bias.grad.abs().sum()) > 0


# ── the router across passes ────────────────────────────────────────────────────────


def _route_log(m: MORPHTransformer):
    log: list[tuple[str, torch.Tensor, object]] = []
    for mod in _xhc_modules(m):
        orig = mod.route

        def _w(h, fixed_route=None, _o=orig, _k=mod.route_key):
            r = _o(h, fixed_route)
            log.append((_k, r[0].detach().clone(), fixed_route))
            return r
        mod.route = _w
    return log


def test_the_router_is_deterministic_and_the_choice_moves_across_passes():
    _ids0, inp, _lab, layout = _pack4()
    depths = torch.full(layout.slot_index.shape, 3, dtype=torch.long)
    logs = []
    for _ in range(2):
        m = _xm().eval()
        log = _route_log(m)
        with torch.no_grad():
            m(inp, slot_layout=layout, slot_depths=depths)
        logs.append(log)
    a, b = logs
    assert len(a) == len(b) == 3 * len(_xhc_modules(m))       # one call per module per pass
    for (ka, ia, _), (kb, ib, _) in zip(a, b):
        assert ka == kb and torch.equal(ia, ib)
    valid = layout.slot_valid.repeat(4, 1)                        # code_enum_k = 4 rollouts
    moved = 0
    for key in {k for k, _i, _f in a}:
        per_pass = [i for k, i, _f in a if k == key]              # pass order
        sets0 = per_pass[0].sort(-1).values
        for later in per_pass[1:]:
            diff = (later.sort(-1).values != sets0).any(-1) & valid
            moved += int(diff.sum())
    assert moved > 0, "no slot chose a different stream set on a later pass"


# ── temporal augmentation: causality, pads, and off == C1 ───────────────────────────


def test_temporal_components_are_causal_on_the_slot_axis():
    mod = _live_module((2, 4, 8), C=10, seed=1)
    S, j = 20, 6
    y = torch.randn(2, S, 10)
    valid = torch.ones(2, S, dtype=torch.bool)
    base = mod.augment(y, valid)
    y2 = y.clone()
    y2[:, j] += torch.randn(2, 10)
    pert = mod.augment(y2, valid)
    assert torch.equal(base[:, :j], pert[:, :j]), "slot j reached an earlier slot"
    for s in range(j + 1, j + 8):
        assert not torch.equal(base[:, s], pert[:, s]), f"slot j did not reach slot {s}"
    # kernel 8 alone reaches j+7; the widest reach ends there
    assert not torch.equal(base[:, j + 7, 2], pert[:, j + 7, 2])
    assert torch.equal(base[:, j + 7, :2], pert[:, j + 7, :2])
    assert torch.equal(base[:, j + 8:], pert[:, j + 8:])


def test_a_pad_slot_does_not_change_a_valid_slot():
    mod = _live_module((2, 4, 8), C=10, seed=2)
    S = 16
    y = torch.randn(2, S, 10)
    valid = torch.ones(2, S, dtype=torch.bool)
    valid[0, 5] = False            # a mid-row pad: the mask, not the trailing order, cuts it
    valid[1, 12:] = False
    base = mod.augment(y, valid)
    y2 = y.clone()
    y2[0, 5] += 10.0
    y2[1, 12:] += 10.0
    pert = mod.augment(y2, valid)
    assert torch.equal(base[valid], pert[valid])


def test_causal_slot_conv_is_the_written_sum():
    y = torch.randn(1, 7, 3)
    w = torch.randn(3, 1, 4)
    got = causal_slot_conv(y, w)
    for s in range(7):
        ref = sum(w[:, 0, 3 - t] * y[0, s - t] for t in range(4) if s - t >= 0)
        torch.testing.assert_close(got[0, s], ref)


def test_gram_schmidt_components_are_orthogonal_to_the_base_and_each_other():
    base = torch.randn(5, 8)
    comps = [torch.randn(5, 8) for _ in range(3)]
    out = gram_schmidt_components(base, comps)
    vecs = [base] + [out[:, r] for r in range(3)]
    for a in range(4):
        for b in range(a + 1, 4):
            assert float((vecs[a] * vecs[b]).sum(-1).abs().max()) < 1e-4


def test_temporal_augmentation_with_a_zero_write_is_c1():
    """C2 with the augmented write set to exactly zero (the aug rows of `w_act` and their
    bias at 0, so `tanh` = 0) and every shared xHC weight copied from C1 runs C1's
    forward: loss, logits and the gradient of every C1 parameter are bit-identical."""
    _ids0, inp, lab, layout = _pack4()
    c1 = _xm(())
    c2 = _xm((2, 4, 8))
    kk = 2 * _K * _K
    with torch.no_grad():
        for a, b in zip(_xhc_modules(c1), _xhc_modules(c2)):
            b.w_pre.copy_(a.w_pre)
            b.pre_bias.copy_(a.pre_bias)
            b.w_route.copy_(a.w_route)
            b.route_bias.copy_(a.route_bias)
            b.w_act[:kk].copy_(a.w_act)
            b.act_bias[:kk].copy_(a.act_bias)
            b.w_act[kk:].zero_()
            b.act_bias[kk:].zero_()
    s1, s2 = c1.state_dict(), c2.state_dict()
    for k in s1:
        if s1[k].shape == s2[k].shape:
            assert torch.equal(s1[k], s2[k]), k
    outs = []
    for m in (c1, c2):
        m.train()
        torch.manual_seed(9)
        o = m(inp, labels=lab, slot_layout=layout)
        o["loss"].backward()
        m.eval()
        with torch.no_grad():
            lg = m(inp, slot_layout=layout)["logits"]
        outs.append((o["loss"].detach(), lg))
    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.equal(outs[0][1], outs[1][1])
    p2 = dict(c2.named_parameters())
    for n, p in c1.named_parameters():
        if p.grad is None:
            continue
        g2 = p2[n].grad
        if n.endswith("w_act") or n.endswith("act_bias"):
            g2 = g2[:p.grad.shape[0]]
        assert torch.equal(p.grad, g2), n


# ── entry and exit ──────────────────────────────────────────────────────────────────


def test_the_entry_holds_the_carrier_in_streams_0_to_3_and_zeros_after():
    _ids0, inp, _lab, layout = _pack4()
    m = _xm().eval()
    m._jac_capture = []
    with torch.no_grad():
        m(inp, slot_layout=layout)
    h0, e0 = m._jac_capture[0]["h"], m._jac_capture[0]["e"]
    assert h0.shape[2] == _N and e0.shape[2] == _N
    assert torch.equal(h0[:, :, 4:], torch.zeros_like(h0[:, :, 4:]))
    assert torch.equal(e0[:, :, 4:], torch.zeros_like(e0[:, :, 4:]))
    assert torch.equal(h0[:, :, :4], e0[:, :, :4])                 # core_init = clone
    assert float(h0[:, :, :4].abs().sum()) > 0
    # after one pass the extra streams hold what the loop wrote
    assert float(m._jac_capture[1]["h"][:, :, 4:].abs().sum()) > 0


@pytest.mark.parametrize("stream", range(16))
def test_the_exit_writes_cell_j_from_streams_4j_to_4j_plus_3(stream):
    _ids0, inp, _lab, layout = _pack4()
    m = _xm().eval()
    cap = []
    orig_pp = m.tul.prefix_project

    def _pp(h, lay, L, cells=None):
        r = orig_pp(h, lay, L, cells=cells)
        cap.append(r[0].detach().clone())
        return r
    m.tul.prefix_project = _pp
    orig_core = m._tul_core
    bump = {"on": False}

    def _core(*a, **k):
        r = list(orig_core(*a, **k))
        if bump["on"]:
            h = r[1].clone()
            h[:, :, stream] += 1.0
            r[1] = h
        return tuple(r)
    m._tul_core = _core
    with torch.no_grad():
        m(inp, slot_layout=layout)
        bump["on"] = True
        m(inp, slot_layout=layout)
    a, b = (v.reshape(v.shape[0], -1, 4, *v.shape[2:]) for v in cap)   # [B, S, K, 4, C]
    valid = layout.slot_valid.repeat(4, 1)
    for j in range(4):
        same = torch.equal(a[:, :, j][valid], b[:, :, j][valid])
        assert same == (j != stream // 4), (stream, j)


# ── the gain hinge ──────────────────────────────────────────────────────────────────


def test_the_gain_hinge_replays_the_operating_points_routing():
    """The hinge's two applications: the first RECORDS every module's stream choice and
    the second REPLAYS it. On this fixture the row gain reads 0.87 with the replay and 1.17
    with the router free (a TopK flip across the finite difference), so the reading is
    pinned below 1 as well."""
    _ids0, inp, lab, layout = _pack4()
    m = _xm().train()
    log = _route_log(m)
    torch.manual_seed(5)
    out = m(inp, labels=lab, slot_layout=layout)
    hinge = [(k, i, f) for k, i, f in log if f is not None]
    n_mod = len(_xhc_modules(m))
    assert len(hinge) == 2 * n_mod
    rec, rep = hinge[:n_mod], hinge[n_mod:]
    assert all(not f["replay"] for _k, _i, f in rec) and all(f["replay"] for _k, _i, f in rep)
    for (k0, i0, _f0), (k1, i1, _f1) in zip(rec, rep):
        assert k0 == k1 and torch.equal(i0, i1)
    assert float(out["gain_est"]) < 1.0


# ── configs, build, one step, gradients ─────────────────────────────────────────────


@pytest.mark.parametrize("name,parent,extra", [
    ("tul_slot_spandec_strict_e4probe_fp01_xhc", "tul_slot_spandec_strict_e4probe_fp01",
     {"tul.prefix_k", "tul.xhc_streams", "tul.xhc_active", "tul.xhc_fixed"}),
    ("tul_slot_spandec_strict_e4probe_fp01_xhc_ta", "tul_slot_spandec_strict_e4probe_fp01",
     {"tul.prefix_k", "tul.xhc_streams", "tul.xhc_active", "tul.xhc_fixed",
      "tul.xhc_temporal_kernels"}),
])
def test_the_arms_compose_differ_by_the_stated_keys_and_train_a_step(name, parent, extra,
                                                                       monkeypatch):
    from omegaconf import OmegaConf
    from test_slot_gain_tail import _MISSING, _leaves

    from morph.training.train import build_morph_config

    cfg, rt = _runtime(name, monkeypatch)
    pcfg, _prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
    assert diff == extra | {"wandb.name"}, sorted(diff)
    tc = rt.model_cfg
    ta = name.endswith("_ta")
    assert (tc.prefix_k, tc.xhc_streams, tc.xhc_active, tc.xhc_fixed) == (4, 16, 4, 2)
    assert tc.xhc_temporal_kernels == ((2, 4, 8) if ta else ())
    assert rt.manifest["xhc_streams"] == 16
    mc = build_morph_config(cfg, tul=tc)
    assert mc.hc_streams * tc.prefix_k == tc.xhc_streams
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=tc, d_ff=_D_FF, dropout=0.1,
                               core_fixed_point_lambda=mc.core_fixed_point_lambda,
                               slot_gain_lambda=mc.slot_gain_lambda,
                               slot_cot_clip=mc.slot_cot_clip)).train().float()
    assert m._xhc_temporal == ta
    _ids0, inp, lab, layout = _pack4()
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    new = {n: q for n, q in m.named_parameters() if n.startswith("core.") and ".mrr_" in n}
    # per core block: attn (w_pre, pre_bias, w_route, route_bias, w_act, act_bias) and the
    # same six on the MLP residual, plus its three conv kernels under temporal augmentation
    assert len(new) == 2 * (6 + 6 + (3 if ta else 0)), sorted(new)
    for n, q in new.items():
        assert q.grad is not None and float(q.grad.abs().sum()) > 0, n
    if ta:
        assert any("conv_w" in n for n in new)
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    opt.step()
    opt.zero_grad()
    out2 = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out2["loss"])


# ── refusals, the token path, the checkpoint ────────────────────────────────────────


@pytest.mark.parametrize("kw,exc,match", [
    ({"xhc_streams": 16, "fan_k": 4, "slot_cells": 4, "fan_mix": "all"},
     (NotImplementedError, ValueError), "fan|slot_cells"),
    ({"xhc_streams": 16, "slot_cells": 4}, (NotImplementedError, ValueError), "slot_cells"),
    ({"xhc_streams": 16, "tokens_through_core": True}, (NotImplementedError, ValueError),
     "tokens_through_core"),
    ({"xhc_streams": 16, "code_target": True}, (NotImplementedError, ValueError),
     "code_target"),
    ({"xhc_streams": 16, "prefix_source": "trajectory"}, (NotImplementedError, ValueError),
     "prefix_source"),
    ({"xhc_streams": 16, "xhc_fixed": 4}, ValueError, "xhc_fixed"),
    ({"xhc_streams": 16, "xhc_temporal_kernels": (1, 4)}, ValueError, "kernels"),
    ({"xhc_streams": 0, "xhc_temporal_kernels": (2,)}, ValueError, "silently"),
    ({"xhc_streams": 18}, ValueError, "prefix_k"),
])
def test_refusals(kw, exc, match):
    base = dict(prefix_k=4, slot_id=4, tg_restrict=True, tg_restrict_scope="all",
                emit_weight=0.0, token_state_dropout=0.0, mux_beta=0.0)
    base.update(kw)
    with pytest.raises(exc, match=match):
        TULConfig(**base)


def test_model_level_refusals():
    with pytest.raises(ValueError, match="prefix_k"):
        MORPHTransformer(_tiny(tul=_tc(prefix_k=2, xhc_streams=16), d_ff=_D_FF))
    plain = TULConfig(prefix_k=4, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                      mux_beta=0.0, xhc_streams=16)
    with pytest.raises(NotImplementedError, match="xhc_streams > 0 with model.core_impl"):
        MORPHTransformer(_tiny(tul=plain, d_ff=_D_FF, core_impl="parcae"))


def test_the_token_path_core_refuses():
    """A forward with no slot layout runs the core on the TOKEN carrier (`_core_region`).
    A strict model refuses that earlier (tg_restrict); an unrestricted xHC model reaches
    `_core_region` and must raise there rather than hand the expanded residuals a
    4-stream carrier."""
    torch.manual_seed(3)
    plain = TULConfig(prefix_k=4, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                      mux_beta=0.0, xhc_streams=16)
    m = MORPHTransformer(_tiny(tul=plain, d_ff=_D_FF)).eval().float()
    _ids0, inp, _lab, _layout = _pack4()
    with pytest.raises(RuntimeError, match="runs inside `_tul_core` only"):
        with torch.no_grad():
            m(inp)


def test_a_checkpoint_round_trips(tmp_path):
    _ids0, inp, _lab, layout = _pack4()
    a = _xm((2, 4, 8), seed=11).eval()
    torch.save(a.state_dict(), tmp_path / "x.pt")
    b = _xm((2, 4, 8), seed=99).eval()
    b.load_state_dict(torch.load(tmp_path / "x.pt"))
    with torch.no_grad():
        assert torch.equal(a(inp, slot_layout=layout)["logits"],
                           b(inp, slot_layout=layout)["logits"])
    assert any(k.endswith("mrr_mlp.conv_w.0") for k in a.state_dict())


def test_the_checkpointed_hinge_is_the_same_function(monkeypatch):
    """On an xHC model the hinge's two applications run under `checkpoint` (memory). With
    `checkpoint` replaced by a plain call everywhere, at dropout 0.1, the loss, the hinge's
    penalty and every gradient are bit-identical: the recompute replays the RNG and the
    routing record/replay."""
    import morph.model.transformer as tm
    _ids0, inp, lab, layout = _pack4()
    res = []
    for plain in (False, True):
        if plain:
            monkeypatch.setattr(tm, "checkpoint",
                                lambda f, *a, use_reentrant=None, **k: f(*a, **k))
        # target 0.5 so the hinge is ACTIVE (the fixture's gain reads ~0.87): its
        # gradient then reaches the core weights through the checkpointed calls
        m = _xm((2, 4, 8), dropout=0.1, gain_target=0.5).train()
        torch.manual_seed(5)
        out = m(inp, labels=lab, slot_layout=layout)
        out["loss"].backward()
        res.append((out["loss"].detach(), out["gain_reg_weighted"].detach(),
                    {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}))
    (l0, g0, d0), (l1, g1, d1) = res
    assert float(g0) > 0.0 and torch.equal(l0, l1) and torch.equal(g0, g1)
    assert d0.keys() == d1.keys()
    for n in d0:
        assert torch.equal(d0[n], d1[n]), n


def test_the_plan_ablations_act_on_every_cell():
    """`zero` removes all four cells (the coda then reads exactly what a zero exit gives)
    and `shuffle` moves every cell with its slot: the val pass's `plan_worth_*` rows read
    the whole 16-stream write, not one cell of it."""
    _ids0, inp, _lab, layout = _pack4()
    m = _xm().eval()
    cap = []
    orig_pp = m.tul.prefix_project

    def _pp(h, lay, L, cells=None):
        cap.append(None if cells is None else cells.detach().clone())
        return orig_pp(h, lay, L, cells=cells)
    m.tul.prefix_project = _pp
    with torch.no_grad():
        normal = m(inp, slot_layout=layout)["logits"]
        zero = m.tul_forward_ablated(inp, None, layout, plan_mode="zero")["logits"]
        m.tul_forward_ablated(inp, None, layout, plan_mode="shuffle")
    c_norm, c_zero, c_shuf = cap
    assert c_norm.shape[2:4] == (4, 4)
    assert torch.equal(c_zero, torch.zeros_like(c_zero))
    assert not torch.equal(normal, zero)
    valid = layout.slot_valid.repeat(4, 1)
    moved = (c_shuf != c_norm).flatten(2).any(-1)                  # [B, S] per slot
    assert bool(moved[valid].any())
    # a shuffled slot carries SOME slot's four cells together: its cell stack equals a
    # normal slot's stack of the same row
    for b in range(c_norm.shape[0]):
        rows = c_norm[b][valid[b]].flatten(1)
        for s in torch.nonzero(valid[b]).flatten().tolist():
            assert (rows == c_shuf[b, s].flatten()).all(-1).any(), (b, s)


# ── C1b: the scale pin and the hinge on the applied map ─────────────────────────────

_BASE_C1B = "36a9823"      # master when C1b was built; the tree the knob-off test compares to

_C1B_OFF_SCRIPT = r'''
import sys
import numpy as np
import torch
import morph
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids
import os
assert os.path.realpath(morph.__file__).startswith(os.path.realpath(sys.argv[2])), morph.__file__
V = 64
lut = np.zeros(V, dtype=bool)
lut[[0, 10, 11]] = True
rule = BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)
spec = TulLayoutSpec(seq_len=64, prefix_k=4, max_slots=10, slot_id=4)
rng = np.random.default_rng(0)
ids = rng.integers(5, V, size=(2, 120))
ids[ids == 4] = 5
ids[:, ::8] = 10
inp, lab, layout, _ = slot_layout_from_ids(ids.astype(np.int64), rule, spec)
res = {}
for renorm in (False, True):
    tc = TULConfig(prefix_k=4, slot_id=4, tg_restrict=True, tg_restrict_scope="all",
                   emit_weight=0.0, token_state_dropout=0.0, mux_beta=0.0,
                   tg_geometry="strict", spandec=False, spandec_parallel=True, code_enum_k=4,
                   plast_weight=1.0, xhc_streams=16)
    cfg = MORPHConfig(d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256,
                      context_len=256, n_prelude=2, n_core=2, n_coda=2, mean_depth=2,
                      max_depth=3, bptt_depth=3, channel_dims=(32, 20, 12), compression=2,
                      csa_compress_ratio=4, hca_compress_ratio=8, top_k=8, window_size=16,
                      retention=False, bigram_hash_vocab=V, use_kernels=False,
                      hc_use_kernel=False, dropout=0.1, d_ff=96,
                      core_fixed_point_lambda=0.1, slot_gain_lambda=100.0,
                      slot_gain_target=0.5, slot_gain_tail_lambda=100.0,
                      slot_gain_all_iters=True, slot_cot_clip=4.0,
                      slot_state_renorm=renorm, tul=tc)
    torch.manual_seed(1234)
    m = MORPHTransformer(cfg).float().train()
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    torch.manual_seed(5)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    r = {"loss": out["loss"].detach(), "keys": sorted(out.keys()),
         "gain_est": out["gain_est"].detach(), "pen": out["gain_reg_weighted"].detach(),
         "grads": {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None},
         "params": {n: p.detach().clone() for n, p in m.named_parameters()}}
    m.eval()
    with torch.no_grad():
        r["logits"] = m(inp, slot_layout=layout)["logits"]
    res[renorm] = r
torch.save(res, sys.argv[1])
'''


def test_c1b_knob_off_is_bit_identical_to_the_pre_c1b_tree(tmp_path):
    """The xHC fp01 twin (hinge 100 at target 0.5 on EVERY iteration so it is active, tail
    100, fixed-point 0.1, cot clip 4, dropout 0.1, code_enum_k 4), with the renorm off and
    on, run by the tree before C1b and by this tree with `slot_gain_renorm` at its default:
    loss, output keys, the hinge's reading and penalty, every grad and param and the eval
    logits are `torch.equal`. Covers the `_renorm_to` refactor of the loop's renorm and the
    hinge's new argument."""
    ok = subprocess.run(["git", "-C", _REPO, "cat-file", "-e", f"{_BASE_C1B}^{{commit}}"],
                        capture_output=True)
    if ok.returncode != 0:
        pytest.skip(f"base commit {_BASE_C1B} is not in this clone")
    base = tmp_path / "base"
    base.mkdir()
    arc = subprocess.run(["git", "-C", _REPO, "archive", _BASE_C1B, "morph"],
                         capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(base)], input=arc, check=True)
    outs = []
    for tree, name in ((str(base), "base.pt"), (_REPO, "head.pt")):
        env = dict(os.environ, PYTHONPATH=tree, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        r = subprocess.run([sys.executable, "-c", _C1B_OFF_SCRIPT, str(tmp_path / name), tree],
                           cwd=tree, env=env, capture_output=True, text=True)
        assert r.returncode == 0, f"fixture failed in {tree}:\n{r.stderr[-3000:]}"
        outs.append(torch.load(tmp_path / name))
    ref, got = outs
    for renorm in (False, True):
        a, b = ref[renorm], got[renorm]
        assert float(a["pen"]) > 0.0, "the hinge must be active for the grads to cover it"
        for k in ("loss", "gain_est", "pen", "logits"):
            assert torch.equal(a[k], b[k]), (renorm, k)
        # `enum_win_entropy` (LX, 2026-09-26, tests/test_tul_lx_credit.py) is a detached
        # instrument added after the base: the one key the base cannot have.
        assert a["keys"] == [k for k in b["keys"] if k != "enum_win_entropy"], renorm
        assert a["params"].keys() == b["params"].keys()
        assert a["grads"].keys() == b["grads"].keys() and len(b["grads"]) > 50
        for n in a["params"]:
            assert torch.equal(a["params"][n], b["params"][n]), (renorm, n)
        for n in a["grads"]:
            assert torch.equal(a["grads"][n], b["grads"][n]), (renorm, n)
    # the renorm is live on this fixture: it changes the loss
    assert not torch.equal(got[False]["loss"], got[True]["loss"])


def _grown(renorm: bool, gain_renorm: bool = False, k: int = 1, scale: float = 30.0,
           **mkw) -> MORPHTransformer:
    """A tiny xHC model whose core WRITES are scaled up (the MLP down-projection and the
    attention up-projection of every core block, x `scale`): each pass then adds several
    times the carried state, the operating point C1 reached by pass 1 at step ~3000."""
    m = _xm(k=k, gain_target=0.98,
            mkw=dict(slot_state_renorm=renorm, slot_gain_renorm=gain_renorm,
                     slot_gain_tail_lambda=100.0, **mkw))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.startswith("core.") and (n.endswith("mlp.down._cms.weight")
                                          or n.endswith("cca.W_up.weight")):
                p.mul_(scale)
    return m


def _pass_norms(m: MORPHTransformer, layout, inp, depth: int, k: int):
    """Per-slot carrier norm (all 16 streams) entering each pass, valid slots, [depth, n]."""
    m._jac_capture = []
    depths = torch.full(layout.slot_index.shape, depth, dtype=torch.long)
    with torch.no_grad():
        m(inp, slot_layout=layout, slot_depths=depths)
    cap, m._jac_capture = m._jac_capture, None
    v = layout.slot_valid.repeat(k, 1)
    return cap, torch.stack([c["h"].flatten(2).float().norm(dim=2)[v] for c in cap])


def test_c1b_renorm_pins_the_16_stream_carrier_where_today_grows():
    """Today (renorm off) the grown model's carrier grows ~10x per pass (the scale jump C1
    ran into); with `slot_state_renorm` every slot's 16-stream norm at the entry of passes
    1 and 2 equals its entry norm, and pads stay exactly 0. `code_enum_k` 1 here: the E code
    is added AFTER the renorm and moves the norm by ~1 % (measured 0.992-1.017 at k 4)."""
    _ids0, inp, _lab, layout = _pack4()
    _c, grow = _pass_norms(_grown(False).eval(), layout, inp, 3, 1)
    assert float((grow[1] / grow[0]).min()) > 3.0 and float((grow[2] / grow[0]).min()) > 5.0
    cap, pin = _pass_norms(_grown(True).eval(), layout, inp, 3, 1)
    assert pin.shape[0] == 3
    torch.testing.assert_close(pin[1:], pin[0].expand(2, -1), rtol=1e-5, atol=0.0)
    pad = ~layout.slot_valid
    for c in cap:
        assert torch.equal(c["h"][pad], torch.zeros_like(c["h"][pad]))


def test_c1b_hinge_reads_the_renormed_map_on_a_linear_step(monkeypatch):
    """With the core step replaced by f(h) = 3h: the raw hinge reads 3 (the gain of 3h),
    with `slot_gain_renorm` it reads the gain of R(f)(h) = h * n0 / |h| at |h| = n0, which is
    the tangential part of a random direction (~1), and `gain_est_raw` is 3. Target 1.05, so
    the knob-on penalty is exactly 0 and the knob-off one is 100 * 1.95^2 per pass."""
    _ids0, inp, lab, layout = _pack4()
    res = {}
    for gr in (False, True):
        m = _xm(k=1, gain_target=1.05,
                mkw=dict(slot_state_renorm=True, slot_gain_renorm=gr,
                         slot_gain_all_iters=True)).train()
        monkeypatch.setattr(m, "_apply_core_step",
                            lambda h, *a, **kw: (3.0 * h, kw.get("ret_state")), raising=False)
        torch.manual_seed(5)
        res[gr] = m(inp, labels=lab, slot_layout=layout)
    assert abs(float(res[False]["gain_est"]) - 3.0) < 1e-4
    assert "gain_est_raw" not in res[False]
    assert abs(float(res[True]["gain_est_raw"]) - 3.0) < 1e-4
    # tangential part of d plus an O(eps^2) radial term: 1.00005 measured
    assert 0.95 < float(res[True]["gain_est"]) < 1.01
    assert float(res[True]["gain_slot_max"]) < 1.05
    # the penalty follows the reading: 100 * (3 - 1.05)^2 per hinged pass against 0
    assert float(res[False]["gain_reg_weighted"]) >= 380.0
    assert float(res[True]["gain_reg_weighted"]) == 0.0


def test_c1b_hinge_on_the_grown_model_charges_the_applied_map():
    """The grown model, renorm on in both: today's hinge (knob off) reads the raw step at
    ~2.1 and charges a penalty in the hundreds for a map the loop never applies; with the
    knob its own reading (R(f)) is under the 0.98 target and charges nothing, and its
    `gain_est_raw` is today's reading bit for bit (same two applications, same RNG)."""
    _ids0, inp, lab, layout = _pack4()
    res = {}
    for gr in (False, True):
        m = _grown(True, gr, k=4, slot_gain_all_iters=True).train()
        torch.manual_seed(5)
        res[gr] = m(inp, labels=lab, slot_layout=layout)
    assert float(res[False]["gain_est"]) > 1.5 and float(res[False]["gain_reg_weighted"]) > 100.0
    assert torch.equal(res[True]["gain_est_raw"], res[False]["gain_est"])
    assert float(res[True]["gain_est"]) < 0.98
    assert float(res[True]["gain_reg_weighted"]) == 0.0
    # everything but the hinge term is the same forward
    torch.testing.assert_close(res[True]["loss"] - res[True]["gain_reg_weighted"],
                               res[False]["loss"] - res[False]["gain_reg_weighted"],
                               rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("mkw", [
    dict(slot_gain_renorm=True),                                      # no renorm
    dict(slot_gain_renorm=True, slot_state_renorm=True, slot_gain_lambda=0.0),   # no hinge
])
def test_c1b_knob_refuses_where_it_would_do_nothing(mkw):
    torch.manual_seed(1)
    kw = dict(d_ff=_D_FF, slot_gain_lambda=100.0)
    kw.update(mkw)
    with pytest.raises(ValueError, match="slot_gain_renorm"):
        MORPHTransformer(_tiny(tul=_tc(prefix_k=4, xhc_streams=_N), **kw))


@pytest.mark.parametrize("name,parent", [
    ("tul_slot_spandec_strict_e4probe_fp01_xhc_c1b", "tul_slot_spandec_strict_e4probe_fp01_xhc"),
    ("tul_slot_spandec_strict_e4probe_fp01_xhc_c1b_ta",
     "tul_slot_spandec_strict_e4probe_fp01_xhc_ta"),
])
def test_c1b_arms_compose_differ_by_the_two_keys_and_train_a_step(name, parent, monkeypatch):
    from omegaconf import OmegaConf
    from test_slot_gain_tail import _MISSING, _leaves

    from morph.training.train import build_morph_config

    cfg, rt = _runtime(name, monkeypatch)
    pcfg, _prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
    assert diff == {"model.slot_state_renorm", "model.slot_gain_renorm", "wandb.name"}, diff
    assert c["wandb.name"] == ("lxtul-e4probe-fp01-xhc-c1b-ta" if name.endswith("_ta")
                               else "lxtul-e4probe-fp01-xhc-c1b")
    tc = rt.model_cfg
    mc = build_morph_config(cfg, tul=tc)
    assert mc.slot_state_renorm and mc.slot_gain_renorm
    assert (mc.slot_gain_lambda, mc.slot_gain_target, mc.slot_gain_tail_lambda) == (100.0, 0.98,
                                                                                     100.0)
    assert (tc.xhc_streams, tc.prefix_k) == (16, 4)
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=tc, d_ff=_D_FF, dropout=0.1,
                               core_fixed_point_lambda=mc.core_fixed_point_lambda,
                               slot_gain_lambda=mc.slot_gain_lambda,
                               slot_gain_target=mc.slot_gain_target,
                               slot_gain_tail_lambda=mc.slot_gain_tail_lambda,
                               slot_cot_clip=mc.slot_cot_clip,
                               slot_state_renorm=mc.slot_state_renorm,
                               slot_gain_renorm=mc.slot_gain_renorm)).train().float()
    _ids0, inp, lab, layout = _pack4()
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"]) and "gain_est_raw" in out
    out["loss"].backward()
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    opt.step()
    out2 = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out2["loss"])
