"""The two CE heads of the lxtul_pointer step at GEMM cost (2026-10-08, graph-step speed work).

What each test pins:
  1. `fused_linear_label_logprob(softmax_kernel=True)` (`model.ce_softmax_kernel`): the
     per-row log p(y) and its gradient for ANY upstream g (to x and to w) are at least as
     close to an fp64 reference as the eager body's, at the token head's real shape
     (7680 x 1024, V 49169 padded to 49280, the masked slot id), with ignored rows, g = 0
     rows and negative g. All-ignored input gives zeros. The reference starts from the SAME
     bf16 logits tile the GEMM writes (and runs the gradient GEMMs in fp64 on the same bf16
     operands), so it measures the elementwise bodies, not the GEMM's output rounding.
  2. The pointer mixture (`TULPointer.target_logprob`) on top of it, as `_tul_group_losses`
     runs it: the loss and the gradients to the hidden states, the head and every pointer
     parameter, kernel vs eager vs an fp64 log p_model. A query with no candidate has null
     mass 1, so its mixed log-prob IS the model's.
  3. The span decoder's CE (`_FusedLinearCE`): a head that does not require grad gets no
     grad_w and the loss and grad_x stay bit-identical; `row_cap` (`model.spandec_ce_row_cap`)
     at the real shape (12288 rows, half labelled) is as accurate as all rows against fp64,
     with and without the kernel and with a trained head; it never syncs the host; a batch
     over the cap gives NaN; an all-ignored batch gives 0 and zero gradients.
  4. `MORPHTransformer._spandec_row_cap` is an upper bound on real packer rows, including the
     worst case (every span at the cap) and DITTO rows, and is tight on the worst case.
  5. Both keys reach MORPHConfig.

Sabotages run and reverted (session report): see the cehead report in the vlt thread
`graph-step`.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

from morph.model.fused_ce import fused_linear_cross_entropy, fused_linear_label_logprob

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels, CUDA")

V_REAL, D_REAL, MASK = 49169, 1024, 4


def _err(a, b):
    return float((a.double() - b.double()).abs().max())


# ── 1. the label log-prob ────────────────────────────────────────────────────────────

def _lp_case(n, d=D_REAL, V=V_REAL, frac_valid=0.87, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(n, d, generator=g) * 0.6).to("cuda", torch.bfloat16).float()
    w = (torch.randn(V, d, generator=g) * 0.05).cuda()
    lab = torch.randint(0, V, (n,), generator=g)
    lab[lab == MASK] = MASK + 1
    lab[torch.rand(n, generator=g) > frac_valid] = -100
    up = torch.randn(n, generator=g) * 1e-3          # the upstream gradient, either sign
    up[::11] = 0.0
    return x, w, lab.cuda(), up.cuda()


def _lp_run(x, w, lab, up, kernel):
    xr, wr = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = fused_linear_label_logprob(xr, wr, lab, chunk_size=1024, mask_token_id=MASK,
                                         softmax_kernel=kernel)
    (out * up).sum().backward()
    return out.detach(), xr.grad, wr.grad


def _bf16_logits(xb16, wb16, s, e):
    """The bf16 logits tile the CE's GEMM writes for rows s..e, in fp64 (exact): the same
    GEMM on the same operands, the vocab zero-padded to the multiple of 128 the CE pads to
    (`fused_ce._pad_vocab`; cuBLAS picks its kernel, and so its rounding, by the shape)."""
    from morph.model.fused_ce import _pad_vocab
    wp, V, _ = _pad_vocab(wb16)
    return (xb16[s:e] @ wp.t())[:, :V].double()


def _lp_ref(x, w, lab, up, chunk=1024):
    """fp64 out, grad_x, grad_w from the bf16 logits tile and the bf16 operands."""
    xb16, wb16 = x.to(torch.bfloat16), w.to(torch.bfloat16)
    xb, wb = xb16.double(), wb16.double()
    valid = lab != -100
    ls = lab.clamp(min=0)
    out = torch.zeros(len(lab), dtype=torch.float64, device=x.device)
    gx = torch.zeros_like(xb)
    gw = torch.zeros_like(wb)
    for s in range(0, len(lab), chunk):
        e = min(s + chunk, len(lab))
        lg = _bf16_logits(xb16, wb16, s, e)
        lg[:, MASK] = float("-inf")
        lse = torch.logsumexp(lg, -1)
        out[s:e] = torch.where(valid[s:e], lg.gather(-1, ls[s:e, None])[:, 0] - lse, 0.0)
        gl = -torch.softmax(lg, -1)
        gl[torch.arange(e - s), ls[s:e]] += 1.0
        gl *= (up[s:e].double() * valid[s:e])[:, None]
        gx[s:e] = gl @ wb
        gw += gl.t() @ xb[s:e]
    return out, gx, gw


@cuda
def test_label_logprob_kernel_is_as_accurate_as_eager_at_the_token_head_shape():
    x, w, lab, up = _lp_case(7680)
    ro, rgx, rgw = _lp_ref(x, w, lab, up)
    eo, egx, egw = _lp_run(x, w, lab, up, False)
    ko, kgx, kgw = _lp_run(x, w, lab, up, True)
    assert not torch.equal(kgw, egw)                     # the kernel really ran
    for name, k, e, r, floor in (("out", ko, eo, ro, 2e-6), ("grad_x", kgx, egx, rgx, 1e-9),
                                 ("grad_w", kgw, egw, rgw, 1e-9)):
        ek, ee = _err(k, r), _err(e, r)
        print(f"{name}: kernel err {ek:.3e}  eager err {ee:.3e}  |ref| {float(r.abs().max()):.3e}")
        assert ek <= 1.25 * ee + floor, (name, ek, ee)
    assert torch.equal(ko[lab == -100], torch.zeros_like(ko[lab == -100]))
    assert float(kgw[MASK].abs().max()) == 0.0          # the masked row gets no gradient
    assert float(kgw[V_REAL - 1].abs().max()) > 0.0     # the last real row does (pad edge)


@cuda
def test_label_logprob_kernel_all_ignored_rows():
    x, w, lab, up = _lp_case(700, V=4096)
    lab[:] = -100
    ko, kgx, kgw = _lp_run(x, w, lab, up, True)
    assert torch.equal(ko, torch.zeros_like(ko))
    assert torch.equal(kgx, torch.zeros_like(kgx)) and torch.equal(kgw, torch.zeros_like(kgw))


# ── 2. the pointer mixture on top ────────────────────────────────────────────────────

def _ptr_case(B=2, L=1280, d=D_REAL, V=V_REAL, seed=5):
    """A packed batch: every 20th position a slot position (label -100), labels drawn from a
    small vocabulary so the pointer has real candidates to copy."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    slot_mask = torch.zeros(B, L, dtype=torch.bool)
    slot_mask[:, 19::20] = True
    lab = torch.randint(10, 400, (B, L), generator=g)
    lab[slot_mask] = -100
    lab[:, -1] = -100
    h = (torch.randn(B, L, d, generator=g) * 0.6).to(torch.bfloat16).float()
    w = (torch.randn(V, d, generator=g) * 0.05)
    row_w = torch.ones(B * L)
    row_w[::37] = 0.5                                  # the TUL half weights
    return h.cuda(), w.cuda(), lab.cuda(), slot_mask.cuda(), row_w.cuda()


def _ptr_loss(ptr, h, w, lab, slot_mask, row_w, mode):
    """`_tul_group_losses`' pointer branch: the weighted mean of -log p_mix(y)."""
    B, L, C = h.shape
    flat, labf = h.reshape(-1, C), lab.reshape(-1)
    if mode == "ref":
        # fp64 from the bf16 logits tile (its VALUE) with the exact fp64 linear map's
        # gradient (the straight-through term); the pointer itself runs as in the model
        with torch.autocast("cuda", enabled=False):
            xb16, wb16 = flat.to(torch.bfloat16), w.to(torch.bfloat16)
            exact = xb16.double() @ wb16.double().t()
            tile = torch.cat([_bf16_logits(xb16.detach(), wb16.detach(), s, s + 1024)
                              for s in range(0, flat.shape[0], 1024)])
            lg = exact + (tile - exact.detach())
            lg[:, MASK] = float("-inf")
            lp_model = torch.where(labf != -100, torch.log_softmax(lg, -1).gather(
                -1, labf.clamp(min=0)[:, None])[:, 0], 0.0).float()
    else:
        lp_model = fused_linear_label_logprob(flat, w, labf, chunk_size=1024,
                                              mask_token_id=MASK,
                                              softmax_kernel=(mode == "kernel"))
    lp, _ = ptr.target_logprob(h, lp_model.view(B, L), lab, slot_mask)
    wv = row_w * (labf != -100).to(row_w.dtype)
    return -(lp.reshape(-1) * wv).sum() / wv.sum(), lp, lp_model


@cuda
def test_pointer_mixture_loss_and_grads_kernel_vs_eager_vs_fp64():
    from morph.model.tul_pointer import TULPointer
    torch.manual_seed(0)
    ptr = TULPointer(D_REAL, 4).cuda()
    with torch.no_grad():                              # a gate that actually mixes
        ptr.gate_norm_head.bias.copy_(torch.tensor([0.0, -1.0, -1.5, -2.0, -2.5]))
        ptr.gate_norm_head.weight.normal_(0, 0.01)
    h0, w0, lab, sm, rw = _ptr_case()
    res = {}
    for mode in ("ref", "eager", "kernel"):
        h, w = h0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
        ptr.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, lp, lp_model = _ptr_loss(ptr, h, w, lab, sm, rw, mode)
        loss.backward()
        res[mode] = dict(loss=loss.detach(), h=h.grad, w=w.grad, lp=lp.detach(),
                         lpm=lp_model.detach(),
                         **{n: p.grad.clone() for n, p in ptr.named_parameters()})
    # the pointer is live: the mixture differs from the model on many labelled tokens
    tok = (~sm) & (lab != -100)
    assert float((res["kernel"]["lp"] - res["kernel"]["lpm"].view_as(lab))[tok].abs().max()) > 0.1
    # a query with no candidate (the first token of each row) has null mass 1: lp == lp_model
    for mode in ("eager", "kernel"):
        assert torch.equal(res[mode]["lp"][:, 0], res[mode]["lpm"].view_as(lab)[:, 0].float())
    for key in res["ref"]:
        if key in ("lp", "lpm"):
            continue
        ek, ee = _err(res["kernel"][key], res["ref"][key]), _err(res["eager"][key], res["ref"][key])
        print(f"{key}: kernel err {ek:.3e}  eager err {ee:.3e}")
        assert ek <= 1.25 * ee + 1e-9, (key, ek, ee)


# ── 3. the span decoder's CE: detached head, fixed row cap ──────────────────────────

def _sd_case(S_rows=384, J=32, d=D_REAL, V=V_REAL, seed=9):
    """12288 decoder rows: per slot a valid PREFIX of 0..32 offsets (mean ~16), the rest -100."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = S_rows * J
    x = (torch.randn(n, d, generator=g) * 0.6).to(torch.bfloat16).float()
    w = torch.randn(V, d, generator=g) * 0.05
    lab = torch.randint(0, V, (S_rows, J), generator=g)
    lab[lab == MASK] = MASK + 1
    ln = torch.randint(0, J + 1, (S_rows, 1), generator=g)
    lab[torch.arange(J)[None] >= ln] = -100
    return x.cuda(), w.cuda(), lab.reshape(-1).cuda()


def _sd_run(x, w, lab, kernel, cap, w_grad=False):
    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(w_grad)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = fused_linear_cross_entropy(xr, wr, lab, chunk_size=1024, mask_token_id=MASK,
                                          softmax_kernel=kernel, row_cap=cap)
    loss.backward()
    return loss.detach(), xr.grad, wr.grad


def _sd_ref(x, w, lab, chunk=1024):
    xb16, wb16 = x.to(torch.bfloat16), w.to(torch.bfloat16)
    xb, wb = xb16.double(), wb16.double()
    valid = lab != -100
    n = valid.sum().clamp_min(1).double()
    ls = lab.clamp(min=0)
    loss = torch.zeros((), dtype=torch.float64, device=x.device)
    gx, gw = torch.zeros_like(xb), torch.zeros_like(wb)
    for s in range(0, len(lab), chunk):
        e = min(s + chunk, len(lab))
        lg = _bf16_logits(xb16, wb16, s, e)
        lg[:, MASK] = float("-inf")
        lse = torch.logsumexp(lg, -1)
        loss += ((lse - lg.gather(-1, ls[s:e, None])[:, 0]) * valid[s:e]).sum()
        p = torch.softmax(lg, -1)
        p[torch.arange(e - s), ls[s:e]] -= 1.0
        p *= valid[s:e, None]
        gx[s:e] = p @ wb
        gw += p.t() @ xb[s:e]
    return loss / n, gx / n, gw / n


@cuda
def test_detached_head_skips_grad_w_bit_identically():
    x, w, lab = _sd_case(S_rows=96)
    for kernel in (False, True):
        l1, gx1, gw1 = _sd_run(x, w, lab, kernel, 0, w_grad=False)
        l2, gx2, gw2 = _sd_run(x, w, lab, kernel, 0, w_grad=True)
        assert gw1 is None and gw2 is not None
        assert torch.equal(l1, l2) and torch.equal(gx1, gx2)


@cuda
@pytest.mark.parametrize("kernel", [False, True])
@pytest.mark.parametrize("w_grad", [False, True])
def test_row_cap_is_as_accurate_as_all_rows_at_the_span_decoder_shape(kernel, w_grad):
    x, w, lab = _sd_case()
    n_lab = int((lab != -100).sum())
    cap = 6792                                          # the lxtul bound, B 6
    assert n_lab < cap < lab.numel()
    rl, rgx, rgw = _sd_ref(x, w, lab)
    al, agx, agw = _sd_run(x, w, lab, kernel, 0, w_grad)
    cl, cgx, cgw = _sd_run(x, w, lab, kernel, cap, w_grad)
    pairs = [("loss", cl, al, rl, 1e-6), ("grad_x", cgx, agx, rgx, 1e-10)]
    if w_grad:
        pairs.append(("grad_w", cgw, agw, rgw, 1e-10))
    for name, c, a, r, floor in pairs:
        ec, ea = _err(c, r), _err(a, r)
        print(f"kernel={kernel} {name}: cap err {ec:.3e}  all-rows err {ea:.3e}")
        assert ec <= 1.25 * ea + floor, (name, ec, ea)
    # an unlabelled row gets exactly zero gradient
    assert torch.equal(cgx[lab == -100], torch.zeros_like(cgx[lab == -100]))


@cuda
def test_row_cap_never_syncs_the_host():
    x, w, lab = _sd_case(S_rows=64)
    cap = int((lab != -100).sum()) + 100
    _sd_run(x, w, lab, True, cap)                       # compile the kernels outside the check
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _sd_run(x, w, lab, True, cap)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@cuda
def test_row_cap_overflow_is_nan_and_all_ignored_is_zero():
    x, w, lab = _sd_case(S_rows=64)
    n_lab = int((lab != -100).sum())
    loss, _, _ = _sd_run(x, w, lab, True, n_lab - 1)
    assert torch.isnan(loss)
    loss, gx, _ = _sd_run(x, w, lab, True, n_lab)       # exactly at the cap: finite
    assert torch.isfinite(loss)
    lab0 = torch.full_like(lab, -100)
    loss, gx, _ = _sd_run(x, w, lab0, True, 100)
    assert float(loss) == 0.0 and torch.equal(gx, torch.zeros_like(gx))


# ── 4. the row-cap bound on real packer rows ─────────────────────────────────────────

class _Dec:
    def __init__(self, J=32, horizon=1, target_offset=1):
        self.per_span_tokens, self.horizon, self.target_offset = J, horizon, target_offset


def _pack(stream, B=6, ditto_rows=0):
    from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, pack_tul_batch
    lut = np.zeros(100, dtype=bool)
    lut[1] = True
    rule = BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)
    spec = TulLayoutSpec(seq_len=1024, prefix_k=4, max_slots=64, slot_id=99)
    kw = {}
    if ditto_rows:
        kw = dict(ditto_rows=ditto_rows, ditto_rng=np.random.default_rng(0))
    return pack_tul_batch(list(stream), rule, spec, B, **kw)


@pytest.mark.parametrize("kind", ["no_boundary", "every4", "random", "ditto"])
def test_spandec_row_cap_bounds_real_packer_rows(kind):
    from morph.model.transformer import MORPHTransformer
    from morph.model.tul_spandec import horizon_span_slots
    rng = np.random.default_rng(3)
    n = 6 * 1300
    if kind == "no_boundary":                          # every span forced at the 32 cap
        stream = rng.integers(5, 99, n)
    elif kind == "every4":
        stream = np.where(np.arange(n) % 4 == 3, 1, rng.integers(5, 99, n))
    else:
        stream = np.where(rng.random(n) < 0.06, 1, rng.integers(5, 99, n))
    x, y, lay = _pack(stream.tolist(), ditto_rows=2 if kind == "ditto" else 0)
    B, L = x.shape
    for dec in (_Dec(), _Dec(horizon=3), _Dec(target_offset=2)):
        ids, valid = horizon_span_slots(x, lay, dec.per_span_tokens, dec.horizon,
                                        start=dec.target_offset)
        cap = MORPHTransformer._spandec_row_cap(B, L, lay, dec)
        per_row = valid.reshape(B, -1).sum(1)
        assert int(per_row.sum()) <= cap, (kind, dec.horizon, dec.target_offset)
        assert int(per_row.max()) <= cap // B
        if kind == "no_boundary" and dec.horizon == 1 and dec.target_offset == 1:
            assert cap == 6 * 1132
            assert int(per_row.max()) >= 1080            # within 5 % of the bound (1088)


# ── 5. the keys reach the model ──────────────────────────────────────────────────────

def test_the_keys_reach_the_model():
    from hydra import compose, initialize_config_dir
    from morph.training.train import build_morph_config
    cdir = str(pathlib.Path(__file__).resolve().parents[1] / "morph" / "configs")
    with initialize_config_dir(config_dir=cdir, version_base=None):
        on = compose(config_name="base", overrides=["model.spandec_ce_row_cap=true",
                                                    "model.ce_softmax_kernel=true"])
        off = compose(config_name="base")
    assert build_morph_config(on).spandec_ce_row_cap is True
    assert build_morph_config(on).ce_softmax_kernel is True
    assert build_morph_config(off).spandec_ce_row_cap is False


# ── 6. the graph-captured step with both keys on ─────────────────────────────────────

@cuda
def test_graph_step_replays_the_eager_step_with_both_keys_on(monkeypatch):
    """tests/test_graph_step.py's test 1 (replay == eager, bit for bit, over 12 steps of
    every step kind, batches whose slot counts differ) with both keys on. The row cap must
    really bind at the tiny shapes, so every call's cap is recorded and checked."""
    import test_graph_step as tgs
    from morph.model.transformer import MORPHTransformer
    keys = ("model.ce_softmax_kernel=true", "model.spandec_ce_row_cap=true")
    caps = []
    real = MORPHTransformer._spandec_row_cap

    def spy(B, L, layout, dec):
        caps.append((real(B, L, layout, dec), B * layout.max_slots * dec.max_tokens))
        return caps[-1][0]
    monkeypatch.setattr(MORPHTransformer, "_spandec_row_cap", staticmethod(spy))
    built = []
    real_build = tgs._build

    def build(cfg, rt, seed=0):
        m, opt = real_build(cfg, rt, seed)
        built.append((m.cfg.ce_softmax_kernel, m.cfg.spandec_ce_row_cap))
        return m, opt
    monkeypatch.setattr(tgs, "_build", build)
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        tgs.test_replay_is_the_eager_step_bit_for_bit(monkeypatch, None, keys)
    finally:
        torch.use_deterministic_algorithms(prev)
    assert built and all(b == (True, True) for b in built)
    assert caps and all(0 < c < n for c, n in caps), caps[:3]
