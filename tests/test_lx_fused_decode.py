"""The LXTUL fused decoder (morph/inference/lx_fused_decode.py) is the graphed decoder's function.

Note: .agents/notes/proposed/feature/2026-10-10-lxtul-fused-decode.md. The fused decoder runs
every block on the plain decode engine's kernels and the strict attention through
``lx_attn`` (the kernel, or its PyTorch stand-in ``lx_attn_reference``); this file checks
it against ``tul_generate_graphed.TulGraphDecoder`` (itself checked against the eager
``generate_tul`` in tests/test_tul_generate_lxtul.py):

* TEACHER-FORCED: the same token stream into both decoders, every log-prob row compared
  (token rows, and the cell rows under emit source "slot"), so one flipped argmax can
  neither hide nor fake a state bug. The stream crosses many boundaries, passes the
  window on the loop's cell axis and on the row, and repeats tokens (the pointer copies).
* BATCHED: B = 3 rows (bucket 4, one dummy row) with different streams, every row
  against its own single-row reference; and ``generate_lx_fused_batch`` with prompts of
  different lengths against ``generate_tul_graphed`` per row (greedy tokens equal).
* Both weight modes: ``exact_weights=True`` (fp32 copies, reduction order only) and the
  bf16 copies.
* SABOTAGE: each piece of state the fused decoder adds is broken once and the
  teacher-forced comparison must fail.

The reference is made pure fp32 for the tight tolerance: ``fused_cca_conv`` (the
reference's loop conv) runs ``tl.dot`` on fp32 operands at TF32, and cuDNN's conv1d runs
at TF32 by default, so both are switched off for the reference (``_fp32_reference``).

Fixture: the tests/test_tul_generate_mode.py LXTUL + pointer model at kernel-legal widths
(d_model 256, 4 heads, d_head 32, gate hidden 64; the engine's kernels need d_head a power
of 2 and widths in multiples of 32), MLP / x0 / mixer weights snapped to gamma * {-1, 0, 1}
so the ternary GEMVs run. CUDA only, fp32.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the fused decoder runs Triton kernels on CUDA")

import morph.inference.lx_fused_decode as fmod  # noqa: E402
import morph.model.attention as attn_mod  # noqa: E402
from morph.inference.lx_fused_decode import LxFusedDecoder, generate_lx_fused_batch  # noqa: E402
from morph.inference.tul_generate import TulRowBuilder  # noqa: E402
from morph.inference.tul_generate_graphed import TulGraphDecoder, generate_tul_graphed  # noqa: E402
from morph.model.tul_layout import BoundaryRule  # noqa: E402
from test_loop_attn_center import LXTUL_KW  # noqa: E402
from test_tul_fan_lsel import _lsel  # noqa: E402
from test_tul_strict_geometry import _spec  # noqa: E402

V = 64
M = 4
TOL_EXACT = 1e-4     # measured worst recorded in the report; reduction order only
WIDE = dict(d_model=256, n_heads=4, n_kv_heads=2, channel_dims=(128, 80, 48))


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[0, 10, 11, 20, 30]] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _snap(w: torch.Tensor) -> None:
    g = w.abs().mean()
    w.copy_(g * torch.round(w / g).clamp(-1, 1))


@torch.no_grad()
def _model(pointer: bool = True, bf16_values: bool = False):
    kw = {**LXTUL_KW, "fan_repel_lambda": 0.1, "fan_repel_mode": "epivol"}
    if pointer:
        kw["pointer_heads"] = 4
    m = _lsel("joint", model_kw=WIDE, **kw)
    g = torch.Generator().manual_seed(3)
    if pointer:
        m.tul_pointer.gate_norm_head.bias.zero_()
    reg = m.tul_register
    reg.W_o.weight.copy_(torch.randn(reg.W_o.weight.shape, generator=g) * 0.2)
    reg.P_cell.copy_(torch.randn(reg.P_cell.shape, generator=g) * 0.2)
    m.tul_cell_norm.weight.copy_(
        torch.empty_like(m.tul_cell_norm.weight).uniform_(0.5, 1.5, generator=g))
    m.embed.bigram.lambdas.fill_(0.5)
    for inj in m.x0_injects:
        inj.log_scale.fill_(0.3)
        _snap(inj.proj.weight)
    for blk in list(m.prelude) + list(m.core) + list(m.coda):
        mlp = blk.mlp[0] if hasattr(blk.mlp, "__getitem__") else blk.mlp
        _snap(mlp.gate_up._cms.weight)
        _snap(mlp.down._cms.weight)
    _snap(m.lm_mixer.mix.weight)
    if bf16_values:
        # every parameter on the bf16 grid: the decoder's bf16 weight copies are then
        # the same numbers, and the bf16 mode must reproduce the reference as tightly as
        # the exact mode (only the copies' dtype differs, not the weights)
        for prm in m.parameters():
            prm.copy_(prm.to(torch.bfloat16).float())
    return m.cuda().eval()


@pytest.fixture(scope="module")
def lx():
    return _model()


@pytest.fixture
def _fp32_reference(monkeypatch):
    monkeypatch.setattr(attn_mod, "_FUSED_ATTN_QKCONV", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def _stream(n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=n)
    ids[ids == 4] = 5
    bnd = rng.random(n) < 0.15
    ids[bnd] = rng.choice([10, 11, 20, 30], size=int(bnd.sum()))
    if n >= 40:
        ids[n // 3: n // 3 + 12] = ids[5:17]              # a repeat for the pointer
    return [int(x) for x in ids]


def _spec_lx():
    return _spec(prefix_k=M, max_slots=60, seq_len=512)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(x.double().cpu(), -1)


def _diff(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = _norm(a), _norm(b)
    fx, fy = torch.isfinite(x), torch.isfinite(y)
    assert torch.equal(fx, fy), "the -inf pattern differs"
    return float((x[fx] - y[fy]).abs().max())


@torch.no_grad()
def _ref_rows(m, stream, emit):
    """The graphed decoder fed ``stream``: per token its log-prob row, and per slot (emit
    "slot") the cell row. Returns ``(tok_rows, cell_rows, n_slots)``."""
    spec, rule = _spec_lx(), _rule()
    tok_rows, cell_rows = [], []
    with TulGraphDecoder(m, spec, rule, emit, use_graphs=True) as dec:
        dec.reset()
        b = TulRowBuilder(rule=rule, spec=spec)
        for t in stream:
            pos = len(b.ids)
            cut = b.append(t)
            dec.feed_token(t, pos)
            tok_rows.append(dec.logits_out.clone())
            if cut:
                dec.feed_slot(b.slot_first[-1])
                if emit == "slot":
                    cell_rows.append(dec.cell_logits_out.clone())
        torch.cuda.synchronize()
        return tok_rows, cell_rows, b.n_slots


def _fused_rows(m, streams, emit, dec=None, **kw):
    """The fused decoder fed ``streams`` (one per row): per row, the token rows and the
    cell rows."""
    spec, rule = _spec_lx(), _rule()
    B = len(streams)
    own = dec is None
    if own:
        dec = LxFusedDecoder(m, spec, rule, B, emit, max_tokens=len(streams[0]) + 4, **kw)
    tok_rows = [[] for _ in range(B)]
    cell_rows = [[] for _ in range(B)]
    try:
        dec.reset()
        bs = [TulRowBuilder(rule=rule, spec=spec) for _ in range(B)]
        for k in range(len(streams[0])):
            cut = dec.feed([s[k] for s in streams], bs)
            for b in range(B):
                tok_rows[b].append(dec.logp_tok[b].clone())
            if emit == "slot":
                for b in cut:
                    cell_rows[b].append(dec.logp_cell[b].clone())
        torch.cuda.synchronize()
    finally:
        if own:
            dec.__exit__(None, None, None)
    return tok_rows, cell_rows


def _worst(ref, got) -> float:
    assert len(ref) == len(got)
    return max(_diff(a, b) for a, b in zip(ref, got))


@pytest.mark.parametrize("attn", ["reference", "kernel"])  # K1 or its stand-in; K2 follows
@pytest.mark.parametrize("emit", ["token", "slot"])
def test_teacher_forced_rows_match_the_graphed_decoder(lx, _fp32_reference, emit, attn):
    s = _stream(140, 0)
    rt, rc, n_slots = _ref_rows(lx, s, emit)
    assert n_slots * M > lx.cfg.window_size and n_slots >= 10
    ft, fc = _fused_rows(lx, [s], emit, attn=attn, kv_row=attn)
    worst = _worst(rt, ft[0])
    if emit == "slot":
        assert len(rc) == n_slots
        worst = max(worst, _worst(rc, fc[0]))
    print(f"teacher-forced worst ({emit}, {attn}): {worst:.3e}")
    assert worst < TOL_EXACT, worst


def test_bf16_weight_copies_on_bf16_values_match(_fp32_reference):
    """The bf16-copy mode on a model whose weights are bf16 numbers: the copies lose
    nothing, so the rows must match the reference at the exact tolerance. This pins the
    mode's plumbing; what bf16 rounding costs a real checkpoint is the bench's
    ``lx_fused_bf16`` gate (it is a property of the weights, not of this code)."""
    mb = _model(bf16_values=True)
    s = _stream(100, 1)
    rt, _rc, _n = _ref_rows(mb, s, "token")
    for exact in (True, False):
        ft, _fc = _fused_rows(mb, [s], "token", exact_weights=exact)
        worst = _worst(rt, ft[0])
        print(f"teacher-forced worst (bf16-valued model, exact_weights={exact}): {worst:.3e}")
        assert worst < TOL_EXACT, (exact, worst)


@pytest.mark.parametrize("attn", ["reference", "kernel"])  # K1 or its stand-in; K2 follows
def test_batched_rows_match_their_single_row_reference(lx, _fp32_reference, attn):
    streams = [_stream(120, 10 + b) for b in range(3)]
    ft, _fc = _fused_rows(lx, streams, "token", attn=attn, kv_row=attn)  # B=3: bucket 4
    for b, s in enumerate(streams):
        rt, _rc, _n = _ref_rows(lx, s, "token")
        worst = _worst(rt, ft[b])
        print(f"row {b} worst ({attn}): {worst:.3e}")
        assert worst < TOL_EXACT, (b, worst)


@pytest.mark.parametrize("emit", ["token", "slot"])
def test_greedy_generation_matches_per_row(lx, _fp32_reference, emit):
    """Prompts of different lengths: every row's tokens equal its single-row graphed run."""
    spec, rule = _spec_lx(), _rule()
    prompts = [_stream(20, 30), _stream(13, 31), _stream(27, 32)]
    out, bs = generate_lx_fused_batch(lx, prompts, rule, spec, max_new_tokens=50,
                                      emit_source=emit)
    for b, p in enumerate(prompts):
        ref, rb = generate_tul_graphed(lx, p, rule, spec, max_new_tokens=50,
                                       temperature=0.0, emit_source=emit)
        assert out[b] == ref, b
        assert bs[b].ids == rb.ids and bs[b].slot_first == rb.slot_first, b


# ── sabotage: the teacher-forced comparison must FAIL on a wrong decoder ─────────────


def _sabotaged(lx, patch, streams=None) -> float:
    """Worst row difference of an eager (no graphs) fused decoder with ``patch(dec)``
    applied, against the graphed reference, row by row."""
    streams = streams or [_stream(120, 0)]
    spec, rule = _spec_lx(), _rule()
    dec = LxFusedDecoder(lx, spec, rule, len(streams), "token", use_graphs=False,
                         max_tokens=len(streams[0]) + 4)
    try:
        patch(dec)
        ft, _fc = _fused_rows(lx, streams, "token", dec=dec)
    finally:
        dec.__exit__(None, None, None)
    return max(_worst(_ref_rows(lx, s, "token")[0], ft[b]) for b, s in enumerate(streams))


def _wrap_attn(dec, edit):
    real = dec.lx_attn

    def _f(q, kc, vc, meta, kpos, *a):
        meta, kpos = edit(meta.clone(), kpos)
        return real(q, kc, vc, meta, kpos, *a)

    dec.lx_attn = _f


def test_unsabotaged_eager_decoder_passes(lx, _fp32_reference):
    assert _sabotaged(lx, lambda dec: None) < TOL_EXACT


def test_sabotage_cell_reads_itself_alone_in_the_prelude(lx, _fp32_reference):
    """Prelude cells (lo1 == P on a non-self query) lose their earlier siblings."""
    P = _rule().span_cap

    def edit(meta, kpos):
        cell = meta[:, 1] == P
        meta[cell, 1] = meta[cell, 2] - 1
        return meta, kpos

    assert _sabotaged(lx, lambda dec: _wrap_attn(dec, edit)) > 1e-2


def test_sabotage_ignoring_cell_positions_in_the_coda_window(lx, _fp32_reference):
    def edit(meta, kpos):
        return meta, None

    assert _sabotaged(lx, lambda dec: _wrap_attn(dec, edit)) > 1e-2


def test_sabotage_reading_cache_row_zero_is_caught_in_a_batch(lx, _fp32_reference):
    def edit(meta, kpos):
        meta[:, 0] = 0
        return meta, kpos

    streams = [_stream(100, 10 + b) for b in range(3)]
    assert _sabotaged(lx, lambda dec: _wrap_attn(dec, edit), streams) > 1e-2


def test_sabotage_no_token_history_commit(lx, _fp32_reference, monkeypatch):
    monkeypatch.setattr(fmod, "ring_commit", lambda x, p: None)
    assert _sabotaged(lx, lambda dec: None) > 1e-2


def test_sabotage_no_loop_history_commit(lx, _fp32_reference):
    def patch(dec):
        dec._commit_loop_history = lambda dst, src: None

    assert _sabotaged(lx, patch) > 1e-2


def test_sabotage_skipping_the_reset(lx, _fp32_reference, monkeypatch):
    monkeypatch.setattr(fmod, "reset_to_winner", lambda h, w, r, m: h)
    assert _sabotaged(lx, lambda dec: None) > 1e-2


def test_sabotage_cell_conv_without_its_zero_row(lx, _fp32_reference):
    """The cells' fresh conv segment reads the table's row 0 instead of the zero row."""
    def patch(dec):
        real = dec._cell_xoff

        def _xoff(nb, cnt):
            x = real(nb, cnt)
            return np.where(x == dec.CX_zero, 0, x).astype(np.int32)

        dec._cell_xoff = _xoff

    assert _sabotaged(lx, patch) > 1e-2
