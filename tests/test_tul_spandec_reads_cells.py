"""`tul.spandec_reads_cells` — the span decoder reads the register's M CELLS, not their mean.

THE DEFECT. `tul.slot_cells: M` gives a span M mutable looped cells and then grades them
through `h_slots = _reg_cells.mean(dim=2)` — one vector, one target per span, the same
object the one-cell ruler already had. The register's own note names this as its most
likely reason to read a flat K-curve ("The objective still grades one state per slot ...
Capacity may be necessary and not sufficient") and names this build as the follow-up.

WHAT THIS FILE HAS TO PROVE:

1. OFF IS NOTHING. `spandec_reads_cells: false` builds no cross module, draws no RNG and
   leaves the forward bit-identical. The pin is MEASURED, not argued: the four rows below
   were produced by `lab/divergence/spandec_off_pin.py` on the tree BEFORE this change
   (`f89256d`, via `git stash`) and on the tree after, and they agree to the last printed
   digit.
2. ON AT STEP 0 IS OFF. The cross output projection is zero-init, so the arm starts AT
   `slot-register-m4`: same loss, same `spandec_ce`, same gradient sum, bit for bit. And
   it does NOT stay there — one optimizer step separates the two.
3. THE WEIGHTS ARE UNMOVED. Every tensor an OFF model has, an ON model has with the same
   bytes; the ON model's extra tensors are all under `.cross.`. The cross weights draw
   from a THIRD private generator and the global RNG state is snapshotted and restored
   around their construction, so the knob adds tensors and moves none.
4. THE MEMORY IS THE CODA'S. What the decoder cross-attends to equals, cell by cell, the
   object `TULSlots.prefix_project` writes into the coda's prefix positions — through the
   same `_readout` `z` takes. Held by a spy on both, not by reading the call site.
5. EVERY CELL IS LOAD-BEARING. Blanking cell j of the memory, with `z` HELD FIXED, moves
   the decoder's states — for every j. The two-sided control: with the projection still at
   its zero init, blanking the same cell moves nothing, which is what makes the ON test a
   statement about the cross-attention and not about the mean.
6. THE REFUSALS. `slot_cells == 1`, `spandec: false`, a memory with no reader, a reader
   with no memory, a mis-shaped memory.

CPU only, fp32, `use_kernels=False`, tiny config.

Record: lab/experiments/planned/2026-09-13-arc-register-reader-and-downstream-target.md
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lab.divergence.spandec_off_pin import batch as _pin_batch
from lab.divergence.spandec_off_pin import model as _pin_model
from lab.divergence.spandec_off_pin import run as _pin_run
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids
from morph.model.tul_spandec import SpanDecoder

V = 64
DOT = 10

# Measured 2026-09-13 with `lab/divergence/spandec_off_pin.py`, first on the parent commit
# f89256d (this change's four source files stashed) and then on this tree. Identical to the
# last printed digit on all four fixtures. 2026-09-23: the fourth column (grad sum) is
# re-pinned after `prefix_project` dropped its broadcast matmul (an expanded [B,S,K,C,C]
# weight copy); the forward columns are unchanged, the grad sums moved by <2e-8 relative
# (fp32 summation order of the W_prefix gradient).
OFF_PIN = {
    (1, 0): (9.975275039672852, 588.597041240384, 4.661545753479004,
             -13.439470942660298, 221),
    (1, 1): (9.963083267211914, 584.2413757609356, 4.579278945922852,
             2.661492524187061, 221),
    (4, 0): (9.907234191894531, 556.8746325914599, 4.609741687774658,
             -13.706002625373948, 226),
    (4, 1): (9.939682006835938, 595.9072399611105, 4.568795680999756,
             -1.6188369479138378, 226),
}


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=3, max_depth=4, bptt_depth=4,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)


def _ids(B: int = 2, n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::8] = DOT
    return ids.astype(np.int64)


def _tul(**kw) -> TULConfig:
    # TWO decoder layers, not one, and that is load-bearing. The cross module is built
    # AFTER a block's own Linears, so at one layer nothing downstream of it exists in
    # either init stream and a leak has nowhere to show. Sabotage R1-e (the cross weights
    # drawn off the BLOCK generator) MISSED at one layer and is caught at two.
    base = dict(prefix_k=2, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                mux_beta=0.0, spandec=True, spandec_layers=2, spandec_max_tokens=8,
                tg_restrict=True, tg_restrict_scope="all", tg_geometry="strict")
    base.update(kw)
    return TULConfig(**base)


def _batch(M: int = 4, seed: int = 0):
    spec = TulLayoutSpec(seq_len=64, prefix_k=max(M, 2), max_slots=10, slot_id=4)
    inp, lab, layout, _ = slot_layout_from_ids(_ids(seed=seed), _rule(), spec)
    return inp, lab, layout


def _model(M: int = 4, reads: bool = False, seed: int = 99, **tul_kw):
    kw = dict(prefix_k=max(M, 2))
    if M > 1:
        kw.update(slot_cells=M, slot_cell_init="distinct")
    if reads:
        kw["spandec_reads_cells"] = True
    kw.update(tul_kw)
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**kw)))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.train().float()


def _forward(m, seed: int = 0, M: int = 4):
    x, y, layout = _batch(M, seed)
    torch.manual_seed(7)
    return m(x, labels=y, slot_layout=layout), x, y, layout


class _Spy:
    """Record the `cells` `prefix_project` is given and the `mem` `decode` is given."""

    def __init__(self, m):
        self.m = m
        self.cells = None
        self.mem = None
        self._pp = m.tul.prefix_project
        self._dec = m.tul_spandec.decode

    def __enter__(self):
        def pp(h_slots, layout, l_total, cells=None):
            self.cells = cells
            return self._pp(h_slots, layout, l_total, cells=cells)

        def dec(z, ids, valid, emb, pos=None, mem=None):
            self.mem = mem
            return self._dec(z, ids, valid, emb, pos=pos, mem=mem)

        self.m.tul.prefix_project = pp
        self.m.tul_spandec.decode = dec
        return self

    def __exit__(self, *a):
        self.m.tul.prefix_project = self._pp
        self.m.tul_spandec.decode = self._dec
        return False


# ── 1. OFF IS NOTHING ────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", sorted(OFF_PIN))
def test_off_is_bit_identical_to_the_pre_knob_tree(key):
    """The pin, measured on f89256d and on this tree. See the module docstring."""
    M, seed = key
    got = _pin_run(M, seed)
    want = OFF_PIN[key]
    assert got[4] == want[4], f"state_dict key count moved: {got[4]} != {want[4]}"
    for name, g, w in zip(("loss", "logit_sum", "spandec_ce", "grad_sum"), got, want):
        assert repr(g) == repr(w), f"{name} moved on M={M} seed={seed}: {g!r} != {w!r}"


@pytest.mark.parametrize("M", [1, 4])
def test_off_builds_no_cross_module(M):
    m = _model(M, reads=False)
    assert m.cfg.tul.spandec_reads_cells is False
    assert m.tul_spandec is not None and m.tul_spandec.reads_cells is False
    assert all(b.cross is None for b in m.tul_spandec.blocks)
    assert not any(".cross." in k for k in m.state_dict())


# ── 2. THE WEIGHTS ARE UNMOVED ───────────────────────────────────────────────

def test_reads_cells_adds_cross_tensors_and_moves_no_other_weight():
    off = _model(4, reads=False).state_dict()
    on = _model(4, reads=True).state_dict()
    extra = sorted(set(on) - set(off))
    assert extra, "the knob built nothing"
    assert all(".cross." in k for k in extra), extra
    assert not set(off) - set(on)
    for k, v in off.items():
        assert torch.equal(v, on[k]), f"{k} moved when the knob was turned on"


def test_the_cross_output_projection_is_exactly_zero_at_init():
    m = _model(4, reads=True)
    for b in m.tul_spandec.blocks:
        assert b.cross is not None
        assert float(b.cross.proj.weight.abs().sum()) == 0.0
        # The query/key-value maps are NOT zero: the projection alone is what makes the
        # term vanish, and a zero everywhere would make the module untrainable.
        assert float(b.cross.q.weight.abs().sum()) > 0.0
        assert float(b.cross.kv.weight.abs().sum()) > 0.0


def test_turning_the_reader_on_consumes_no_extra_global_rng():
    """`nn.Linear` kaiming-draws on the GLOBAL stream before the weight is overwritten, so
    three more Linears per block would leave the stream at a different point and shift
    every module constructed after the decoder. `TULSlotRegister` was caught by exactly
    this (sabotage D7, 2026-09-13) and the cross module snapshots and restores the stream
    for the same reason.

    The claim is NOT "the decoder draws nothing" — the base blocks already draw, and this
    change does not alter that. It is "the KNOB draws nothing extra", so the stream sits at
    the same point either way. Sabotage R1-d."""
    kw = dict(d_model=32, n_heads=2, d_ff=64, n_layers=3, max_tokens=8)
    torch.manual_seed(3)
    SpanDecoder(**kw)
    after_off = torch.random.get_rng_state()
    torch.manual_seed(3)
    SpanDecoder(**kw, reads_cells=True)
    after_on = torch.random.get_rng_state()
    assert torch.equal(after_off, after_on), (
        "building the reader moved the global RNG stream: every module constructed after "
        "the decoder would differ between the arm and its ruler")


def test_the_shared_decoder_weights_are_byte_identical_at_three_layers():
    """The cross weights take a THIRD private stream so the self-attention and MLP weights
    of a reads_cells decoder are byte-identical to a mean-only one's. At one layer a leak
    into the block stream has nothing after it to move, so this test runs three.
    Sabotage R1-e."""
    kw = dict(d_model=32, n_heads=2, d_ff=64, n_layers=3, max_tokens=8)
    off = SpanDecoder(**kw).state_dict()
    on = SpanDecoder(**kw, reads_cells=True).state_dict()
    assert not set(off) - set(on)
    assert all(".cross." in k for k in set(on) - set(off))
    for k, v in off.items():
        assert torch.equal(v, on[k]), f"{k} moved when the reader was built"


def test_every_cross_linear_is_excluded_from_ternary_quantisation():
    """The decoder is a TRAINING-ONLY scorer; grading z through a ternarised reader would
    mix the reader's precision into the target the loop is judged on."""
    m = _model(4, reads=True)
    for b in m.tul_spandec.blocks:
        for lin in (b.cross.q, b.cross.kv, b.cross.proj):
            assert getattr(lin, "_ternary_exclude", False) is True


# ── 3. ON AT STEP 0 IS OFF, AND DOES NOT STAY THERE ──────────────────────────

def _loss_and_grad(m, seed: int = 0):
    out, *_ = _forward(m, seed)
    loss = out["loss"]
    sce = float(out["spandec_ce"])
    loss.backward()
    g = sum(float(p.grad.double().sum()) for p in m.parameters() if p.grad is not None)
    return float(loss), sce, g


def test_on_at_step_zero_equals_off_bit_for_bit():
    off = _loss_and_grad(_model(4, reads=False))
    on = _loss_and_grad(_model(4, reads=True))
    # The gradient SUM includes the cross parameters, which OFF does not have. Their grads
    # are part of the difference only if they are non-zero, and at a zero projection the
    # q/kv grads ARE zero while proj's is not, so compare loss and the decoder's own CE
    # bit-for-bit and the base-model gradient separately.
    assert repr(on[0]) == repr(off[0]), (on[0], off[0])
    assert repr(on[1]) == repr(off[1]), (on[1], off[1])


def test_on_at_step_zero_has_the_same_gradient_on_every_shared_parameter():
    a, b = _model(4, reads=False), _model(4, reads=True)
    for m in (a, b):
        out, *_ = _forward(m)
        out["loss"].backward()
    ga = {k: p.grad for k, p in a.named_parameters()}
    for k, p in b.named_parameters():
        if ".cross." in k:
            continue
        assert (ga[k] is None) == (p.grad is None), k
        if p.grad is not None:
            assert torch.equal(ga[k], p.grad), f"{k} gradient moved at step 0"


def test_it_diverges_after_one_optimizer_step():
    a, b = _model(4, reads=False), _model(4, reads=True)
    la0, lb0 = [], []
    for m, store in ((a, la0), (b, lb0)):
        out, *_ = _forward(m)
        store.append(float(out["loss"]))
        out["loss"].backward()
        torch.optim.SGD(m.parameters(), lr=0.5).step()
    assert repr(la0[0]) == repr(lb0[0])            # they started together
    # The cross projection left zero: its gradient is dL/dout x (attention output)', and
    # the attention output is not zero even when the projection is.
    assert float(b.tul_spandec.blocks[0].cross.proj.weight.abs().sum()) > 0.0
    with torch.no_grad():
        la1, lb1 = float(_forward(a)[0]["loss"]), float(_forward(b)[0]["loss"])
    assert la1 != lb1, "one optimizer step left the two arms identical"


# ── 4. THE MEMORY IS THE CODA'S PREFIX CELLS, CELL BY CELL ───────────────────

def test_the_memory_equals_the_codas_prefix_cells_cell_by_cell():
    m = _model(4, reads=True)
    with _Spy(m) as spy:
        _forward(m)
    assert spy.cells is not None, "prefix_project was not handed the register's cells"
    assert spy.mem is not None, "the decoder was not handed a memory"
    cells = spy.cells                                     # [B, S, M, n, C]
    B, S, M = cells.shape[0], cells.shape[1], cells.shape[2]
    assert M == 4 and spy.mem.shape == (B, S, M, cells.shape[-1])
    want = m._readout(cells.reshape(B, S * M, *cells.shape[3:]))
    want = want.reshape(B, S, M, want.shape[-1])
    assert torch.equal(spy.mem, want), (
        "the decoder's memory is not `_readout` of the exact tensor prefix_project writes")


def test_with_a_think_once_stack_the_memory_is_the_stacks_output():
    """`tul.cond_layers > 0`: `_tul_cond_apply` runs BEFORE the cells are formed, so the
    cells the decoder reads are the STACK's output — the same object the coda's prefix
    write projects. The config header claims that; this measures it. The equality below is
    against the tensor `prefix_project` is handed, so it holds whatever the stack does."""
    m = _model(4, reads=True, cond_layers=2)
    assert m.tul_cond is not None and len(m.tul_cond) == 2
    with _Spy(m) as spy:
        _forward(m)
    cells = spy.cells
    B, S, M = cells.shape[0], cells.shape[1], cells.shape[2]
    want = m._readout(cells.reshape(B, S * M, *cells.shape[3:]))
    assert torch.equal(spy.mem, want.reshape(B, S, M, want.shape[-1]))


def test_a_model_with_the_knob_off_is_handed_no_memory():
    m = _model(4, reads=False)
    with _Spy(m) as spy:
        _forward(m)
    assert spy.cells is not None                  # the register still writes per cell
    assert spy.mem is None                        # but the decoder reads only the mean


# ── 5. EVERY CELL IS LOAD-BEARING (and the control that makes that mean something) ──

def _decode_parts(m):
    """Run one forward and return `(dec, z, ids, valid, emb, mem)` for a direct decode."""
    from morph.model.tul_spandec import horizon_span_slots
    x, y, layout = _batch(4, 0)
    captured = {}
    real = m._tul_spandec_loss

    def spy(h_slots, input_ids, lay, stats=None, cells=None):
        captured["h"] = h_slots
        captured["cells"] = cells
        captured["ids"] = input_ids
        captured["layout"] = lay
        return real(h_slots, input_ids, lay, stats=stats, cells=cells)

    m._tul_spandec_loss = spy
    try:
        torch.manual_seed(7)
        m(x, labels=y, slot_layout=layout)
    finally:
        m._tul_spandec_loss = real
    dec = m.tul_spandec
    cells = captured["cells"].detach()
    z = m._readout(captured["h"].detach())
    B, S, M = cells.shape[0], cells.shape[1], cells.shape[2]
    mem = m._readout(cells.reshape(B, S * M, *cells.shape[3:])).reshape(B, S, M, -1)
    ids, valid = horizon_span_slots(captured["ids"], captured["layout"],
                                    dec.per_span_tokens, dec.horizon,
                                    start=dec.target_offset)
    emb = m.embed.lm_weight().detach()
    return dec, z, ids, valid, emb, mem


@pytest.mark.parametrize("j", [0, 1, 2, 3])
def test_blanking_one_memory_cell_moves_the_decoder(j):
    m = _model(4, reads=True)
    # Take the projection off its zero init: at zero the cross term is 0 by construction
    # and NOTHING about the memory could matter. This is the arm as it is after step 1.
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for b in m.tul_spandec.blocks:
            b.cross.proj.weight.copy_(
                torch.empty_like(b.cross.proj.weight).normal_(0.0, 0.05, generator=g))
    dec, z, ids, valid, emb, mem = _decode_parts(m)
    with torch.no_grad():
        base = dec.decode(z, ids, valid, emb, mem=mem)
        blank = mem.clone()
        blank[:, :, j] = 0.0
        got = dec.decode(z, ids, valid, emb, mem=blank)
    assert not torch.equal(base, got), f"cell {j} of the memory is dead"


@pytest.mark.parametrize("j", [0, 1, 2, 3])
def test_at_the_zero_init_blanking_a_cell_moves_nothing(j):
    """The two-sided control for the test above, and the step-0 identity in one line."""
    m = _model(4, reads=True)
    dec, z, ids, valid, emb, mem = _decode_parts(m)
    with torch.no_grad():
        base = dec.decode(z, ids, valid, emb, mem=mem)
        blank = mem.clone()
        blank[:, :, j] = 0.0
        got = dec.decode(z, ids, valid, emb, mem=blank)
    assert torch.equal(base, got)


def test_z_conditioning_is_still_the_mean():
    """With the projection at its zero init the ON decoder's states equal a mean-only
    decoder's, on the SAME weights — which is the statement that z was not changed."""
    m_on = _model(4, reads=True)
    m_off = _model(4, reads=False)
    dec, z, ids, valid, emb, mem = _decode_parts(m_on)
    with torch.no_grad():
        a = dec.decode(z, ids, valid, emb, mem=mem)
        b = m_off.tul_spandec.decode(z, ids, valid, emb)
    assert torch.equal(a, b)


def test_every_cross_parameter_receives_a_finite_gradient():
    m = _model(4, reads=True)
    g = torch.Generator().manual_seed(12)
    with torch.no_grad():
        for b in m.tul_spandec.blocks:
            b.cross.proj.weight.copy_(
                torch.empty_like(b.cross.proj.weight).normal_(0.0, 0.05, generator=g))
    out, *_ = _forward(m)
    out["loss"].backward()
    for name, p in m.named_parameters():
        if ".cross." not in name:
            continue
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} read a non-finite gradient"
        assert float(p.grad.abs().sum()) > 0.0, f"{name} got an all-zero gradient"


# ── 6. THE REFUSALS ──────────────────────────────────────────────────────────

def test_reads_cells_with_one_cell_raises():
    with pytest.raises(ValueError, match="slot_cells=1"):
        _tul(spandec_reads_cells=True)


def test_reads_cells_without_the_decoder_raises():
    with pytest.raises(ValueError, match="spandec=false"):
        TULConfig(prefix_k=2, slot_id=4, spandec=False, spandec_reads_cells=True)


def test_a_reader_with_no_memory_raises():
    dec = SpanDecoder(d_model=16, n_heads=2, d_ff=32, n_layers=1, max_tokens=4,
                      reads_cells=True)
    z = torch.zeros(1, 2, 16)
    ids = torch.zeros(1, 2, 4, dtype=torch.long)
    ok = torch.ones(1, 2, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="no `mem`"):
        dec.decode(z, ids, ok, torch.zeros(8, 16))


def test_a_memory_with_no_reader_raises():
    dec = SpanDecoder(d_model=16, n_heads=2, d_ff=32, n_layers=1, max_tokens=4)
    z = torch.zeros(1, 2, 16)
    ids = torch.zeros(1, 2, 4, dtype=torch.long)
    ok = torch.ones(1, 2, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="reads_cells=False"):
        dec.decode(z, ids, ok, torch.zeros(8, 16), mem=torch.zeros(1, 2, 4, 16))


@pytest.mark.parametrize("shape", [(1, 2, 4), (2, 2, 4, 16), (1, 3, 4, 16), (1, 2, 4, 8)])
def test_a_misshaped_memory_raises(shape):
    dec = SpanDecoder(d_model=16, n_heads=2, d_ff=32, n_layers=1, max_tokens=4,
                      reads_cells=True)
    z = torch.zeros(1, 2, 16)
    ids = torch.zeros(1, 2, 4, dtype=torch.long)
    ok = torch.ones(1, 2, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="must be"):
        dec.decode(z, ids, ok, torch.zeros(8, 16), mem=torch.zeros(*shape))
