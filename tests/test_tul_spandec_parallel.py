"""`tul.spandec_parallel` — LXTUL-E's committed product reader, and its Stage 0 harness.

Module morph/model/tul_spandec_parallel.py; note
.agents/notes/proposed/architecture/2026-09-23-provable-loop-contribution.md; configs
morph/configs/tul_slot_spandec_strict_e0k{1,4}.yaml; scorer
lab/divergence/lxtul_e_stage0_score.py.

Contracts, one test (or one parametrised family) each. Every test asserts the CONTRACT,
computed a second, independent way, never a shape:

  * OFF IS NOTHING: no parameter, no global-RNG movement, and the rest of the forward
    (token CE, span-decoder term, logits) bit-identical to the ruler's.
  * RNG-NEUTRAL: a head model's base tensors are byte-identical to the ruler's and the
    global RNG state after construction is the ruler's.
  * K = 1 is the plain parallel CE, recomputed by hand from the head's pieces with the
    target taken from the RULER DECODER's own mapping; K = 4 is the brute-force mixture;
    K = 4 with identical codes is the K = 1 loss; a shared shift of every code changes
    nothing (mean-free); the K = 1 twin has no code at all.
  * NO TOKEN INPUT: an edit inside the graded span leaves every head state of the graded
    slot bit-exact, while the teacher-forced decoder's states move (positive control).
  * STAGE 0 HARNESS: under train_only + frozen_eval only the head gets a gradient, the
    frozen model is in eval mode and its every tensor is unchanged after a step, with
    model dropout ON; the loaders accept exactly the head's keys as missing; the two
    configs compose, build and differ in K and the run name only.

CPU, fp32, the `tests/test_tul_strict_geometry.py` fixtures (with d_ff set: that
fixture's `d_ff: 0` gives the decoder blocks a zero-width MLP).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import morph.model.tul_spandec_parallel as par_mod
from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_spandec import horizon_span_slots, next_span_slots
from morph.model.tul_spandec_parallel import ParallelSpanHead, mixture_span_nll
from morph.training.freeze import apply_frozen_eval, apply_train_only
from test_tul_strict_geometry import _pack, _runtime, _tiny, _tul

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RULER_CKPT = "/home/wolfe/morph-to/checkpoints/morph/slot-spandec-strict/step_5000.pt"
_D_FF = 96


def _m(seed: int = 1234, dropout: float = 0.0, **tul_kw) -> MORPHTransformer:
    """A strict span-decoder model (the ruler's geometry), fp32, eval mode."""
    kw = dict(tg_geometry="strict", spandec=True)
    kw.update(tul_kw)
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**kw), d_ff=_D_FF, dropout=dropout))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.eval().float()


def _par(k: int = 1, **kw) -> MORPHTransformer:
    return _m(spandec_parallel=True, spandec_parallel_k=k, **kw)


class _Capture:
    """Record the arguments `_tul_spandec_par_loss` is called with (the model's own seam)."""

    def __init__(self, m: MORPHTransformer):
        self.m, self.calls = m, []
        real = m._tul_spandec_par_loss

        def wrap(h_slots, input_ids, layout, stats=None):
            self.calls.append((h_slots.detach().clone(), input_ids.clone(), layout))
            return real(h_slots, input_ids, layout, stats=stats)
        m._tul_spandec_par_loss = wrap

    def z(self, i: int = -1) -> torch.Tensor:
        h, _ids, _lay = self.calls[i]
        with torch.no_grad():
            return self.m._readout(h)


def _hand_logp(m: MORPHTransformer, z: torch.Tensor, codes_off: torch.Tensor | None,
               ids: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """``[R, B, S, J]`` per-position log p(label), recomputed from the head's WEIGHTS with
    full ``[.., V]`` logits (V = 64 here), the slot_id row masked, no helper of the module
    under test except its blocks. Invalid positions hold 0."""
    head = m.tul_spandec_par
    B, S, C = z.shape
    J = ids.shape[-1]
    if codes_off is None:
        zr = z.unsqueeze(0)
    else:
        rms = z.pow(2).mean(-1, keepdim=True).sqrt()
        zr = z.unsqueeze(0) + rms.unsqueeze(0) * codes_off.view(-1, 1, 1, C)
    R = zr.shape[0]
    x = head.z_in(zr).unsqueeze(3) + head.pos[:J].view(1, 1, 1, J, C)
    x = x.reshape(R * B * S, J, C)
    for blk in head.blocks:
        x = blk(x)
    st = head.out_norm(x).reshape(R, B, S, J, C)
    w = m.embed.lm_weight().detach()
    logits = st @ w.t()
    logits[..., m.cfg.tul.slot_id] = float("-inf")
    lp = F.log_softmax(logits, dim=-1).gather(
        -1, ids.clamp(min=0).unsqueeze(0).expand(R, -1, -1, -1).unsqueeze(-1)).squeeze(-1)
    return torch.where(valid.unsqueeze(0), lp, torch.zeros_like(lp))


def _ruler_targets(m: MORPHTransformer, inp, layout):
    """The target the RULER decoder grades, read through its own mapping."""
    dec = m.tul_spandec
    return horizon_span_slots(inp, layout, dec.per_span_tokens, dec.horizon,
                              start=dec.target_offset)


# ── OFF IS NOTHING / RNG-NEUTRAL ─────────────────────────────────────────────

def test_off_builds_nothing_and_knobs_without_the_head_raise():
    m = _m()
    assert m.tul_spandec_par is None
    assert not any(k.startswith("tul_spandec_par") for k in m.state_dict())
    base = dict(prefix_k=2, slot_id=4, tg_restrict=True, tg_restrict_scope="all",
                spandec=True)
    with pytest.raises(ValueError, match="spandec_parallel=false"):
        TULConfig(**base, spandec_parallel_k=4)
    with pytest.raises(ValueError, match="spandec_parallel=false"):
        TULConfig(**base, spandec_parallel_weight=0.5)
    # Stage 1 (2026-09-24): the head may REPLACE the decoder. With `spandec: false` it
    # reads the decoder's four geometry keys itself; every other decoder knob still raises.
    TULConfig(prefix_k=2, slot_id=4, spandec_parallel=True, spandec_max_tokens=16,
              spandec_layers=1)
    with pytest.raises(ValueError, match="spandec=false"):
        TULConfig(prefix_k=2, slot_id=4, spandec_parallel=True, spandec_weight=2.0)
    with pytest.raises(ValueError, match="spandec=false"):
        TULConfig(prefix_k=2, slot_id=4, spandec_max_tokens=16)
    with pytest.raises(ValueError, match="weight > 0"):
        TULConfig(**base, spandec_parallel=True, spandec_parallel_weight=0.0)
    with pytest.raises(NotImplementedError, match="spandec_horizon"):
        TULConfig(**base, spandec_parallel=True, spandec_horizon=2)
    with pytest.raises(ValueError, match="builds no code table"):
        TULConfig(**base, spandec_parallel=True, spandec_parallel_code_init=0.2)
    with pytest.raises(ValueError, match="symmetric"):
        TULConfig(**base, spandec_parallel=True, spandec_parallel_k=4,
                  spandec_parallel_code_init=0.0)


@pytest.mark.parametrize("k", [1, 4])
def test_rng_neutral_base_weights_and_global_stream(k):
    torch.manual_seed(1234)
    ruler = MORPHTransformer(_tiny(tul=_tul(tg_geometry="strict", spandec=True), d_ff=_D_FF))
    after_ruler = torch.random.get_rng_state()
    torch.manual_seed(1234)
    head = MORPHTransformer(_tiny(tul=_tul(tg_geometry="strict", spandec=True,
                                           spandec_parallel=True, spandec_parallel_k=k),
                                  d_ff=_D_FF))
    after_head = torch.random.get_rng_state()
    assert torch.equal(after_ruler, after_head), "building the head moved the global RNG"
    rs, hs = ruler.state_dict(), head.state_dict()
    extra = sorted(set(hs) - set(rs))
    assert extra and all(n.startswith("tul_spandec_par.") for n in extra), extra
    assert set(rs) <= set(hs)
    for n, v in rs.items():
        assert torch.equal(v, hs[n]), f"base tensor {n} differs"


@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_the_rest_of_the_forward_is_the_ruler_bit_for_bit(dropout):
    """Same seed, TRAIN mode (model dropout and token-state dropout both drawing): the
    token CE, the span-decoder term and the logits are the ruler's exactly, and the head
    leaves the global RNG where the ruler's forward leaves it."""
    _ids, inp, lab, layout = _pack()
    kw = dict(dropout=dropout, token_state_dropout=0.15)
    ruler, head = _m(**kw), _par(4, **kw)
    outs = []
    for m in (ruler, head):
        m.train()
        torch.manual_seed(11)
        o = m(inp, labels=lab, slot_layout=layout)
        outs.append((o, torch.random.get_rng_state()))
    (a, ra), (b, rb) = outs
    for key in ("spandec", "spandec_weighted", "n_tokens"):
        assert torch.equal(a[key], b[key]), key
    assert torch.equal(ra, rb), "the head's forward moved the global RNG"
    assert "par" not in a and "par_weighted" in b
    assert torch.allclose(b["loss"], a["loss"] + b["par_weighted"], atol=1e-5, rtol=0)
    ruler.eval()
    head.eval()
    with torch.no_grad():
        ea = ruler(inp, labels=lab, slot_layout=layout)
        eb = head(inp, labels=lab, slot_layout=layout)
        la = ruler(inp, labels=None, slot_layout=layout)["logits"]
        lb = head(inp, labels=None, slot_layout=layout)["logits"]
    assert torch.equal(ea["ce_tokens"], eb["ce_tokens"])
    assert torch.equal(ea["spandec"], eb["spandec"])
    assert torch.equal(la, lb)


# ── THE LOSS ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["eval", "train_dropout"])
def test_k1_is_the_hand_computed_parallel_ce_on_the_ruler_decoders_target(mode):
    _ids, inp, lab, layout = _pack()
    m = _par(1, dropout=0.3 if mode == "train_dropout" else 0.0, token_state_dropout=0.15)
    assert m.tul_spandec_par.codes is None
    assert not any("codes" in n for n, _ in m.tul_spandec_par.named_parameters())
    cap = _Capture(m)
    if mode == "eval":
        with torch.no_grad():
            out = m(inp, labels=lab, slot_layout=layout)
    else:
        m.train()
        torch.manual_seed(3)
        out = m(inp, labels=lab, slot_layout=layout)
    ids, valid = _ruler_targets(m, inp, layout)
    ids1, valid1 = next_span_slots(inp, layout, m.tul_spandec.per_span_tokens)
    assert torch.equal(ids, ids1) and torch.equal(valid, valid1)
    with torch.no_grad():
        lp = _hand_logp(m, cap.z(), None, ids, valid)
    hand = -lp.sum() / valid.sum()
    assert int(valid.sum()) > 20
    assert abs(float(out["par_ce"]) - float(hand)) < 2e-5, (float(out["par_ce"]), float(hand))
    assert float(out["par_n_tokens"]) == float(valid.sum())


def test_k4_is_the_brute_force_mixture():
    _ids, inp, lab, layout = _pack()
    m = _par(4)
    with torch.no_grad():
        m.tul_spandec_par.codes.normal_(0.0, 0.7, generator=torch.Generator().manual_seed(5))
    cap = _Capture(m)
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
        ids, valid = _ruler_targets(m, inp, layout)
        u = m.tul_spandec_par.codes
        lp = _hand_logp(m, cap.z(), u - u.mean(0, keepdim=True), ids, valid)
    per_slot = lp.sum(-1).double()                                     # [4, B, S]
    tot, n = 0.0, int(valid.sum())
    B, S = valid.shape[:2]
    for b in range(B):
        for s in range(S):
            if not bool(valid[b, s].any()):
                continue
            vals = [float(per_slot[k, b, s]) for k in range(4)]
            mx = max(vals)
            tot += mx + math.log(sum(math.exp(v - mx) for v in vals)) - math.log(4)
    brute = -tot / n
    assert abs(float(out["par_ce"]) - brute) < 2e-5, (float(out["par_ce"]), brute)
    # the code-usage readings on the same table
    resp = torch.softmax(per_slot[:, valid.any(-1)], dim=0)
    ent = float(-(resp * resp.log()).sum(0).mean())
    assert abs(float(out["par_resp_entropy"]) - ent) < 1e-4
    single = [-float(per_slot[k].sum()) / n for k in range(4)]
    assert abs(float(out["par_ce_code_best"]) - min(single)) < 2e-5
    wins = torch.bincount(per_slot[:, valid.any(-1)].argmax(0), minlength=4).double()
    for k in range(4):
        assert abs(float(out[f"par_code_win_{k}"]) - float(wins[k] / wins.sum())) < 1e-6


def test_mixture_math_on_random_tables():
    g = torch.Generator().manual_seed(0)
    for R in (1, 2, 4, 7):
        lp = -torch.rand(R, 13, generator=g, dtype=torch.float64) * 30
        n = 97.0
        want = -sum(math.log(sum(math.exp(float(lp[r, m])) for r in range(R)) / R)
                    for m in range(13)) / n
        assert abs(float(mixture_span_nll(lp, n)) - want) < 1e-10
    lp = -torch.rand(1, 9, generator=g, dtype=torch.float64) * 5
    assert float(mixture_span_nll(lp, 9.0)) == float(-lp.sum() / 9.0)


@pytest.mark.parametrize("mode", ["eval", "train_dropout"])
def test_identical_codes_give_the_k1_loss(mode):
    """A K = 4 head whose four codes are equal IS the K = 1 head: the non-code weights are
    byte-identical by construction (the codes have their own generator), the mean-free
    offsets are then exactly zero, and log-mean-exp of four equal values is that value.
    In train mode with model dropout ON the two forwards also draw the same masks."""
    _ids, inp, lab, layout = _pack()
    kw = dict(dropout=0.3 if mode == "train_dropout" else 0.0, token_state_dropout=0.15)
    m1, m4 = _par(1, **kw), _par(4, **kw)
    s1, s4 = m1.state_dict(), m4.state_dict()
    assert set(s4) - set(s1) == {"tul_spandec_par.codes"}
    for n, v in s1.items():
        assert torch.equal(v, s4[n]), n
    with torch.no_grad():
        m4.tul_spandec_par.codes.copy_(torch.randn(1, m4.cfg.d_model).expand(4, -1))
    outs = []
    for m in (m1, m4):
        if mode == "eval":
            with torch.no_grad():
                outs.append(m(inp, labels=lab, slot_layout=layout))
        else:
            m.train()
            torch.manual_seed(9)
            outs.append(m(inp, labels=lab, slot_layout=layout))
    a, b = outs
    assert abs(float(a["par_ce"]) - float(b["par_ce"])) < 1e-6
    assert torch.equal(a["spandec"], b["spandec"])     # the same cells, the same masks
    assert abs(float(b["par_resp_entropy"]) - math.log(4)) < 1e-5


def test_codes_are_mean_free_a_shared_shift_changes_nothing():
    _ids, inp, lab, layout = _pack()
    m = _par(4)
    head = m.tul_spandec_par
    with torch.no_grad():
        head.codes.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(2))
    assert float(head.code_offsets().detach().sum(0).abs().max()) < 1e-6
    with torch.no_grad():
        a = float(m(inp, labels=lab, slot_layout=layout)["par_ce"])
        head.codes.add_(torch.randn(1, m.cfg.d_model) * 3.0)       # the SAME shift, all k
        b = float(m(inp, labels=lab, slot_layout=layout)["par_ce"])
    assert abs(a - b) < 1e-5, (a, b)


def test_the_codes_train():
    """At K = 4 the code table gets a nonzero gradient, and each code a different one."""
    _ids, inp, lab, layout = _pack()
    m = _par(4)
    m.train()
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    g = m.tul_spandec_par.codes.grad
    assert g is not None and float(g.abs().sum()) > 0
    assert float((g[0] - g[1]).abs().max()) > 0


def test_the_tied_table_is_detached_the_term_trains_the_head_alone():
    """The parallel term, backpropagated ALONE from a detached exit state, reaches the
    head's own tensors and the shared readout `z = _readout(h)` passes through
    (`lm_mixer`, `final_norm`: the span decoder's seam, the same on the ruler), and
    nothing else: in particular not the tied embedding table the coda speaks through (the
    module's "detached at both ends" contract)."""
    _ids, inp, lab, layout = _pack()
    m = _par(4)
    cap = _Capture(m)
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout)
    h, ids, lay = cap.calls[0]
    m.zero_grad(set_to_none=True)
    loss = type(m)._tul_spandec_par_loss(m, h, ids, lay)       # the class method, unwrapped
    loss.backward()
    got = [n for n, p in m.named_parameters() if p.grad is not None]
    allowed = ("tul_spandec_par.", "lm_mixer.", "final_norm.")
    assert any(n.startswith("tul_spandec_par.") for n in got), got
    assert all(n.startswith(allowed) for n in got), [n for n in got
                                                      if not n.startswith(allowed)]
    assert not any(n.startswith("embed.") for n in got)


def test_mixture_token_nll_telescopes_to_the_span_term():
    from morph.model.tul_spandec_parallel import mixture_token_nll
    g = torch.Generator().manual_seed(1)
    M, J, R = 7, 6, 4
    lens = torch.randint(1, J + 1, (M,), generator=g)
    val = torch.arange(J).view(1, J) < lens.view(M, 1)
    nv = int(val.sum())
    lp = -torch.rand(R, nv, generator=g, dtype=torch.float64) * 4
    tok = mixture_token_nll(lp, val)
    slot = torch.arange(M).view(M, 1).expand(M, J)[val]
    per_slot = torch.zeros(R, M, dtype=torch.float64).index_add_(1, slot, lp)
    assert abs(float(tok.sum() / nv) - float(mixture_span_nll(per_slot, nv))) < 1e-12
    for m_ in range(M):
        want = -(torch.logsumexp(per_slot[:, m_], 0) - math.log(R))
        assert abs(float(tok[slot == m_].sum()) - float(want)) < 1e-12
    assert torch.allclose(mixture_token_nll(lp[:1], val), -lp[0], atol=1e-12, rtol=0)


# ── NO TOKEN INPUT ───────────────────────────────────────────────────────────

def test_no_token_input_edit_in_the_graded_span_moves_no_head_state(monkeypatch):
    """Edit a token in span 2. Slot 1 is graded on span 2; under strict geometry its exit
    state does not see span 2. The head's states of slots 0..1 (every row that enters the
    vocabulary GEMM) must be bit-exact, while the TEACHER-FORCED decoder's states of slot 1
    move (the positive control: this probe sees a token path when there is one)."""
    from test_tul_strict_geometry import _edit
    from morph.model.tul_layout import slot_layout_from_ids
    from test_tul_strict_geometry import _rule as rule_fn, _spec
    ids, inp, lab, layout = _pack()
    m = _par(4)
    with torch.no_grad():
        m.tul_spandec_par.codes.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(4))
    rows: list[torch.Tensor] = []
    real = par_mod.fused_linear_label_logprob

    def rec(x, w, labels, **kw):
        rows.append(x.detach().clone())
        return real(x, w, labels, **kw)
    monkeypatch.setattr(par_mod, "fused_linear_label_logprob", rec)
    dec_states: list[torch.Tensor] = []
    real_dec = m.tul_spandec.decode

    def rec_dec(z, i, v, e, **kw):
        st = real_dec(z, i, v, e, **kw)
        dec_states.append(st.detach().clone())
        return st
    m.tul_spandec.decode = rec_dec
    cap = _Capture(m)
    ids2 = _edit(ids, layout, row=0, span=2)
    inp2, lab2, layout2, _ = slot_layout_from_ids(ids2, rule_fn(), _spec())
    assert torch.equal(layout2.slot_index, layout.slot_index)
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout)
        m(inp2, labels=lab2, slot_layout=layout2)
    z1, z2 = cap.z(0), cap.z(1)
    assert torch.equal(z1[0, :2], z2[0, :2]), "precondition: slots 0..1 do not see span 2"
    assert not torch.equal(inp[0], inp2[0])
    # rows of the vocab GEMM, [R * Nv, C] in (code, slot-major, offset) order; the layout
    # is unchanged so the order matches. Map each row to its (b, s).
    idsT, valid = m.tul_spandec_par.targets(inp, layout)
    sup = valid.any(-1)
    val_s = valid[sup]
    slot_bs = torch.nonzero(sup)                                    # [M, 2]
    owner = slot_bs.unsqueeze(1).expand(-1, val_s.shape[1], -1)[val_s]   # [Nv, 2]
    keep = (owner[:, 0] == 0) & (owner[:, 1] <= 1)
    keep = keep.repeat(4)
    assert int(keep.sum()) > 0
    a, b = rows[0], rows[1]
    assert a.shape == b.shape
    assert torch.equal(a[keep], b[keep]), "a head state of the graded slot moved"
    assert not torch.equal(a, b)                                    # later slots do move
    # the positive control: the teacher-forced decoder of slot 1 reads span 2's tokens
    assert not torch.equal(dec_states[0][0, 1], dec_states[1][0, 1])


# ── THE STAGE 0 HARNESS ──────────────────────────────────────────────────────

def test_train_only_frozen_eval_only_the_head_learns_and_the_rest_is_frozen():
    """Model dropout 0.3 and token-state dropout 0.15 ON: under frozen_eval the forward is
    the EVAL forward (two calls agree bit-exactly and match `m.eval()`), only the head's
    tensors get a gradient, and after an optimizer step every other tensor (parameters
    AND buffers) is unchanged."""
    _ids, inp, lab, layout = _pack()
    m = _par(4, dropout=0.3, token_state_dropout=0.15)
    apply_train_only(m, ["tul_spandec_par."])
    trained = apply_frozen_eval(m, ["tul_spandec_par."])
    assert m.training is False and m.tul_spandec_par.training is True
    assert trained[0] == "tul_spandec_par" and all(
        n.startswith("tul_spandec_par") for n in trained)
    assert all(not mod.training for n, mod in m.named_modules()
               if n and not n.startswith("tul_spandec_par"))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    torch.manual_seed(1)
    o1 = m(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(2)
    o2 = m(inp, labels=lab, slot_layout=layout)
    assert torch.equal(o1["par_ce"], o2["par_ce"]) and torch.equal(o1["loss"], o2["loss"])
    o1["loss"].backward()
    for n, p in m.named_parameters():
        if n.startswith("tul_spandec_par."):
            assert p.grad is not None and float(p.grad.abs().sum()) > 0, n
        else:
            assert p.grad is None, n
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.5)
    opt.step()
    after = m.state_dict()
    moved = [k for k in before if not torch.equal(before[k], after[k])]
    assert moved and all(k.startswith("tul_spandec_par.") for k in moved), moved
    # the eval forward reads the same cells: the head's term equals a plain eval forward's
    ref = _par(4, dropout=0.3, token_state_dropout=0.15)
    ref.load_state_dict(before)
    with torch.no_grad():
        assert torch.equal(ref.eval()(inp, labels=lab, slot_layout=layout)["par_ce"],
                           o1["par_ce"])
        # and a TRAIN-mode forward with dropout on does not (the control that the
        # dropout is live, so the equality above is not vacuous)
        ref.train()
        torch.manual_seed(1)
        assert not torch.equal(ref(inp, labels=lab, slot_layout=layout)["par_ce"],
                               o1["par_ce"])


def test_frozen_eval_refuses_a_tensor_level_prefix():
    m = _par(1)
    with pytest.raises(ValueError, match="WHOLE modules"):
        apply_frozen_eval(m, ["tul.E_slot"])
    with pytest.raises(ValueError, match="match no module"):
        apply_frozen_eval(m, ["nope."])


def _save_ckpt(m: MORPHTransformer, path: str, drop: str = "", extra: dict | None = None):
    sd = {k: v for k, v in m.state_dict().items() if not (drop and k.startswith(drop))}
    if extra:
        sd.update(extra)
    torch.save({"model": sd, "step": 5000}, path)


def test_init_from_accepts_exactly_the_head_keys(tmp_path):
    from morph.training.train import check_init_from_keys, load_weights_only
    ruler = _m(seed=77)
    path = str(tmp_path / "ruler.pt")
    _save_ckpt(ruler, path)
    m = _par(4)
    missing, unexpected = load_weights_only(path, m, torch.device("cpu"))
    got = check_init_from_keys(missing, unexpected, m, ["tul_spandec_par."], path)
    want = sorted(k for k in m.state_dict() if k.startswith("tul_spandec_par."))
    assert sorted(got) == want
    for k, v in ruler.state_dict().items():
        assert torch.equal(m.state_dict()[k], v), k
    # a missing tensor OUTSIDE the head raises
    bad = str(tmp_path / "bad.pt")
    _save_ckpt(ruler, bad, drop="tul_spandec.blocks.0.")
    mi, un = load_weights_only(bad, _par(4), torch.device("cpu"))
    with pytest.raises(RuntimeError, match="outside training.init_from_new_modules"):
        check_init_from_keys(mi, un, _par(4), ["tul_spandec_par."], bad)
    # a homeless checkpoint tensor raises
    unx = str(tmp_path / "unx.pt")
    _save_ckpt(ruler, unx, extra={"ghost.weight": torch.zeros(2)})
    m2 = _par(4)
    mi, un = load_weights_only(unx, m2, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="no home"):
        check_init_from_keys(mi, un, m2, ["tul_spandec_par."], unx)
    # a prefix the checkpoint DOES carry (partly) raises
    full = str(tmp_path / "full.pt")
    _save_ckpt(_par(4), full, drop="tul_spandec_par.codes")
    m3 = _par(4)
    mi, un = load_weights_only(full, m3, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="listed as NEW"):
        check_init_from_keys(mi, un, m3, ["tul_spandec_par."], full)


@pytest.mark.skipif(not os.path.isfile(_RULER_CKPT), reason="the kept ruler checkpoint")
def test_the_real_ruler_checkpoint_loads_with_exactly_the_head_missing():
    """The kept `slot-spandec-strict` step 5000 into the e0k4 model, built the way the
    trainer builds it (quantisation BEFORE the load, `_build.build_model`)."""
    import sys
    sys.path.insert(0, os.path.join(_ROOT, "lab", "divergence"))
    from _build import build_cfg, build_model
    from morph.training.train import check_init_from_keys, load_weights_only
    cfg = build_cfg("tul_slot_spandec_strict_e0k4", ["model.use_kernels=false"])
    model, _rt = build_model(cfg, device="cpu")
    missing, unexpected = load_weights_only(_RULER_CKPT, model, torch.device("cpu"))
    got = check_init_from_keys(missing, unexpected, model,
                               list(cfg.training.init_from_new_modules), _RULER_CKPT)
    want = sorted(k for k in model.state_dict() if k.startswith("tul_spandec_par."))
    assert sorted(got) == want and "tul_spandec_par.codes" in want


def test_the_stage0_configs_compose_build_and_differ_in_k_only(monkeypatch):
    from omegaconf import OmegaConf
    cfgs = {}
    for name, k, wb in (("tul_slot_spandec_strict_e0k1", 1, "lxtul-e0k1"),
                        ("tul_slot_spandec_strict_e0k4", 4, "lxtul-e0k4")):
        cfg, rt = _runtime(name, monkeypatch)
        tc = rt.model_cfg
        assert tc.spandec and tc.spandec_parallel and tc.spandec_parallel_k == k
        assert tc.tg_geometry == "strict" and tc.spandec_horizon == 1
        assert cfg.wandb.name == wb
        tr = cfg.training
        assert list(tr.train_only) == ["tul_spandec_par."]
        assert list(tr.init_from_new_modules) == ["tul_spandec_par."]
        assert tr.frozen_eval is True and int(tr.steps) == 2000
        assert tr.init_from == _RULER_CKPT
        torch.manual_seed(7)
        m = MORPHTransformer(_tiny(tul=tc, d_ff=_D_FF)).eval().float()
        assert (m.tul_spandec_par.codes is None) == (k == 1)
        _ids0, inp, lab, layout = _pack()
        with torch.no_grad():
            out = m(inp, labels=lab, slot_layout=layout)
        assert torch.isfinite(out["par_ce"])
        cfgs[k] = OmegaConf.to_container(cfg, resolve=True)
    a, b = cfgs[1], cfgs[4]
    a["tul"].pop("spandec_parallel_k")
    b["tul"].pop("spandec_parallel_k")
    a["wandb"].pop("name")
    b["wandb"].pop("name")
    assert a == b, "the two Stage 0 arms differ in something other than K and the name"


def test_the_head_is_never_quantised_and_has_no_sparse_linear():
    m = _par(4)
    for n, mod in m.tul_spandec_par.named_modules():
        if isinstance(mod, torch.nn.Linear):
            assert getattr(mod, "_ternary_exclude", False) is True, n
        assert type(mod).__name__ not in ("MortarLinear", "CMSBlockLinear"), n


def test_head_refuses_a_code_op_at_k1():
    h = ParallelSpanHead(32, 2, 64, 1, 8, n_codes=1)
    with pytest.raises(RuntimeError, match="no code table"):
        h.code_offsets()
    z = torch.randn(5, 32)
    assert torch.equal(h.with_codes(z)[0], z)


def test_a_model_path_that_skips_the_head_raises():
    """The fold refuses a forward that built the head and never computed its term: a
    path that silently skipped the head would read as its own ruler."""
    _ids, inp, lab, layout = _pack()
    m = _par(1)
    m._tul_spandec_par_loss = lambda *a, **k: None       # the term never computed
    with pytest.raises(RuntimeError, match="never computed the term"):
        m(inp, labels=lab, slot_layout=layout)


def test_np_rng_untouched_by_the_head():
    """The head draws no numpy RNG either (the packer and the data path use it)."""
    st = np.random.get_state()[1].copy()
    _par(4)
    assert np.array_equal(np.random.get_state()[1], st)
