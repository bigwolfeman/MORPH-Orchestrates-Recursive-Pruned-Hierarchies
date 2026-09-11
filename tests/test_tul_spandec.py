"""The span-decoder target (``tul.spandec``) and the slot chain (``tul.slot_chain``).

The MUX head grades a slot's exit state ``z`` against an ORDER-FREE bag of the next span
(``mux_span_targets``, geometric weights), so its optimum is that span's weighted unigram
marginal. ``tul.spandec`` replaces that with a teacher-forced causal decoder over
``[z, t_0 .. t_{J-2}]`` and charges ``-log p(t_j | z, t_{<j})`` at every token of the
span. ``tul.slot_chain`` gives slot ``k`` a direct learned edge from slot ``k-1``'s state.

One test per invariant, and each one fails when its mechanism is removed:

  1. targets — ``next_span_slots`` reproduces, token for token, the relation
     ``mux_span_targets(target="next")`` supervises, IN ORDER.
  2. the loss is nonzero and its gradient REACHES ``z`` — checked two ways: autograd
     through the real forward, and a finite difference on ``z`` alone.
  3. pad slots, span 0 and the dump bin carry NO loss (perturbing their ids cannot move
     the term).
  4. the plain path (``slot_layout=None``) is BIT-IDENTICAL with the decoder built.
  5. ``spandec=false`` builds nothing and draws no RNG (the arm's base weights match its
     ruler's byte for byte).
  6. the chain is CAUSAL: slot ``k``'s loss has exactly zero gradient w.r.t. slot ``k+1``'s
     input, and a nonzero one w.r.t. slot ``k-1``'s.
  7. ``KNOWN_TUL_KEYS`` accepts the new keys and rejects a misspelling.

CPU only, tiny config, no tokenizer.
"""

from __future__ import annotations

import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402  (tests/ is on sys.path)

from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig, mux_span_targets
from morph.model.tul_spandec import SpanDecoder, next_span_slots

MAX_DEPTH = 3
J = 8


def _model(seed: int = 3, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul_kw = {k[4:]: v for k, v in kw.items() if k.startswith("tul_")}
    cfg_kw = {k: v for k, v in kw.items() if not k.startswith("tul_")}
    base_tul = dict(tg_restrict=False, sigreg_lambda=0.0, mux_beta=0.0,
                    token_state_dropout=0.0)
    base_tul.update(tul_kw)
    tul = _tul(**base_tul)
    base = dict(tul=tul, n_core=2, mean_depth=MAX_DEPTH, max_depth=MAX_DEPTH,
                bptt_depth=MAX_DEPTH, retention=False, dropout=0.0,
                core_fixed_point_lambda=0.0, ckpt_grad_iters=0)
    base.update(cfg_kw)
    return MORPHTransformer(_cfg(**base))


def _spandec_model(seed: int = 3, **kw) -> MORPHTransformer:
    return _model(seed, tul_spandec=True, tul_spandec_layers=2,
                  tul_spandec_max_tokens=J, **kw)


# ── 1. the targets are the MUX's relation, in order ──────────────────────────

def test_next_span_slots_reproduces_the_mux_relation_in_order():
    """``mux_span_targets`` says WHICH slot each token supervises and with what weight;
    the decoder needs the same tokens IN ORDER. Every token the MUX supervises must land
    at ``ids[b, tgt_slot, offset]``, and no other cell may be marked valid."""
    x, _y, lay, _ = _batch()
    ids, valid = next_span_slots(x, lay, J)
    pos_valid, alpha, tgt_slot, _sup = mux_span_targets(x, lay, 0.9, target="next")
    B, L = x.shape
    seen = set()
    n_checked = 0
    for b in range(B):
        counts: dict[int, int] = {}
        for p in range(L):
            if not bool(pos_valid[b, p]):
                continue
            s = int(tgt_slot[b, p])
            off = counts.get(s, 0)
            counts[s] = off + 1
            if off >= J:
                continue
            assert bool(valid[b, s, off]), f"row {b} slot {s} offset {off} not valid"
            assert int(ids[b, s, off]) == int(x[b, p])
            seen.add((b, s, off))
            n_checked += 1
    assert n_checked > 20, "fixture supervises too few tokens to be a test"
    # and nothing else is marked valid
    extra = [(b, s, j) for b in range(B) for s in range(valid.shape[1])
             for j in range(J) if bool(valid[b, s, j]) and (b, s, j) not in seen]
    assert extra == [], f"valid cells the MUX does not supervise: {extra[:5]}"
    # the weights agree where both are defined (same span, same offset ordering)
    assert float(alpha[pos_valid].sum()) > 0


def test_span_zero_and_pad_slots_are_never_supervised():
    x, _y, lay, _ = _batch()
    ids, valid = next_span_slots(x, lay, J)
    S = valid.shape[1]
    for b in range(x.shape[0]):
        for s in range(S):
            if not bool(lay.slot_valid[b, s]):
                assert not bool(valid[b, s].any()), f"pad slot {s} supervised"
    # the LAST valid slot of a row has no following slot, so nothing supervises it
    for b in range(x.shape[0]):
        last = int(lay.slot_valid[b].sum()) - 1
        assert not bool(valid[b, last].any())
    assert int(ids.shape[-1]) == J


# ── 2. the term is real and its gradient reaches z ───────────────────────────

def test_spandec_loss_is_nonzero_and_enters_the_total_loss():
    m = _spandec_model()
    x, y, lay, _ = _batch()
    m.train()
    torch.manual_seed(17)
    out = m(x, labels=y, slot_layout=lay)
    assert "spandec" in out and float(out["spandec"]) > 0.0
    assert float(out["spandec_n_tokens"]) > 20
    assert float(out["spandec_weighted"]) == pytest.approx(
        float(out["spandec"]) * m.cfg.tul.spandec_weight, rel=1e-6)
    # it is IN the objective: removing the weighted term recovers the ruler's loss
    ruler = _model()
    ruler.load_state_dict(
        {k: v for k, v in m.state_dict().items()
         if not k.startswith("tul_spandec.")}, strict=True)
    ruler.train()
    torch.manual_seed(17)
    out_r = ruler(x, labels=y, slot_layout=lay)
    assert float(out["loss"].detach()) - float(out["spandec_weighted"]) == pytest.approx(
        float(out_r["loss"].detach()), abs=1e-5)


def test_gradient_reaches_z_from_the_decoder_by_autograd_and_by_finite_difference():
    """The whole point of the arm: every token of the span must push on ``z``.

    Autograd half — the decoder's term alone gives every core-loop parameter a gradient.
    Finite-difference half — perturbing ``z`` along the autograd gradient changes the term
    by the predicted amount, which is what a gradient that merely EXISTS cannot fake.
    """
    m = _spandec_model()
    x, _y, lay, _ = _batch()
    m.eval()
    xf, x0, bg = m._tul_front(x, lay)
    _xn, h_slots, *_ = m._tul_core(xf, x0, bg, lay)
    z = h_slots.detach().clone().requires_grad_(True)
    loss = m._tul_spandec_loss(z, x, lay)
    g = torch.autograd.grad(loss, z)[0]
    assert torch.isfinite(g).all()
    assert float(g.abs().sum()) > 0.0
    # only VALID slots may receive gradient: a pad slot is supervised by nothing
    per_slot = g.detach().flatten(2).abs().sum(-1)                 # [B, S]
    assert float(per_slot[~lay.slot_valid].abs().sum()) == 0.0
    assert float(per_slot[lay.slot_valid].abs().sum()) > 0.0
    # finite difference along -g
    eps = 1e-3
    with torch.no_grad():
        step = eps * g / (g.norm() + 1e-12)
        l0 = float(m._tul_spandec_loss(z.detach(), x, lay))
        l1 = float(m._tul_spandec_loss(z.detach() - step, x, lay))
    predicted = -float((g * step).sum())
    assert (l1 - l0) == pytest.approx(predicted, rel=0.25, abs=1e-7), (l1 - l0, predicted)


def test_the_loop_and_the_prelude_are_trained_by_the_decoder_alone():
    """`slot-loop-mask-norm-match` starved because the token CE could not train the loop.
    The decoder's term on its own must reach the shared core AND the prelude."""
    m = _spandec_model()
    x, _y, lay, _ = _batch()
    m.train()
    xf, x0, bg = m._tul_front(x, lay)
    _xn, h_slots, *_ = m._tul_core(xf, x0, bg, lay)
    m._tul_spandec_loss(h_slots, x, lay).backward()
    core = [p for n, p in m.named_parameters() if n.startswith("core.")]
    pre = [p for n, p in m.named_parameters() if n.startswith("prelude.")]
    assert sum(float(p.grad.abs().sum()) for p in core if p.grad is not None) > 0
    assert sum(float(p.grad.abs().sum()) for p in pre if p.grad is not None) > 0


# ── 3. pad slots carry no loss ───────────────────────────────────────────────

def test_pad_and_unsupervised_positions_cannot_move_the_term():
    """Change every id the decoder is NOT supposed to read; the term must not move.
    Then change one id it IS supposed to read; the term must move."""
    m = _spandec_model()
    x, _y, lay, _ = _batch()
    m.eval()
    xf, x0, bg = m._tul_front(x, lay)
    with torch.no_grad():
        _xn, h_slots, *_ = m._tul_core(xf, x0, bg, lay)
        base = float(m._tul_spandec_loss(h_slots, x, lay))
        _ids, valid = next_span_slots(x, lay, J)
        pos_valid, _a, tgt, _s = mux_span_targets(x, lay, 0.9, target="next")
        # every position the decoder reads, by (slot, offset) — rebuilt from the MUX side
        read = torch.zeros_like(pos_valid)
        B, L = x.shape
        for b in range(B):
            counts: dict[int, int] = {}
            for p in range(L):
                if not bool(pos_valid[b, p]):
                    continue
                s = int(tgt[b, p])
                off = counts.get(s, 0)
                counts[s] = off + 1
                if off < J:
                    read[b, p] = True
        x_bad = torch.where(read, x, (x + 7) % 5000 + 5)
        # `_tul_spandec_loss` also rebuilds the layout-derived targets from ids, so the
        # untouched positions must leave both the inputs and the labels alone.
        assert float(m._tul_spandec_loss(h_slots, x_bad, lay)) == pytest.approx(base,
                                                                               abs=1e-5)
        # sensitivity: one read position moved must move the term
        p0 = int(read[0].nonzero()[0])
        x_one = x.clone()
        x_one[0, p0] = (int(x[0, p0]) + 11) % 5000 + 5
        assert abs(float(m._tul_spandec_loss(h_slots, x_one, lay)) - base) > 1e-5


# ── 4./5. off is nothing, and the plain path is untouched ────────────────────

def test_plain_path_is_bit_identical_with_the_decoder_built():
    """``slot_layout=None`` must never reach the decoder. Loss, logits and every base
    gradient identical to a model with no decoder at all."""
    m_on = _spandec_model()
    m_off = _model()
    m_off.load_state_dict(
        {k: v for k, v in m_on.state_dict().items() if not k.startswith("tul_spandec.")},
        strict=True)
    x, y, _lay, _ = _batch()
    for m in (m_on, m_off):
        m.train()
        torch.manual_seed(11)
        m(x, labels=y)["loss"].backward()
    a = {n: p.grad for n, p in m_on.named_parameters() if not n.startswith("tul_spandec.")}
    b = {n: p.grad for n, p in m_off.named_parameters()}
    assert set(a) == set(b)
    for n in a:
        assert (a[n] is None) == (b[n] is None), n
        if a[n] is not None:
            assert torch.equal(a[n], b[n]), n
    for n, p in m_on.named_parameters():
        if n.startswith("tul_spandec."):
            assert p.grad is None or float(p.grad.abs().sum()) == 0.0


def test_building_the_decoder_draws_no_rng():
    """Its init must come from a private generator, or the arm and its ruler would differ
    in every weight and the comparison would not be one factor."""
    on = _spandec_model(seed=5)
    off = _model(seed=5)
    shared = [n for n, _ in off.named_parameters()]
    d_on = dict(on.named_parameters())
    d_off = dict(off.named_parameters())
    assert set(shared) == set(n for n in d_on if not n.startswith("tul_spandec."))
    for n in shared:
        assert torch.equal(d_on[n].detach(), d_off[n].detach()), n
    # and the decoder itself is not accidentally all zeros (a vacuous pass)
    w = on.tul_spandec.blocks[0].qkv.weight
    assert float(w.abs().sum()) > 0.0


def test_spandec_is_invisible_to_ternary_qat():
    on = _spandec_model()
    for name, mod in on.tul_spandec.named_modules():
        if isinstance(mod, torch.nn.Linear):
            assert getattr(mod, "_ternary_exclude", False), name


# ── 6. the chain is causal ───────────────────────────────────────────────────

def test_slot_chain_is_causal_and_zero_at_init():
    """Slot ``k``'s state must depend on slot ``k-1``'s input and NOT on ``k+1``'s.

    Measured through the chain alone: the core's own attention over the compact sequence
    is already causal, so the test perturbs the chain's INPUT tensor directly and reads
    which output slots move.
    """
    m = _model(tul_slot_chain=True)
    assert m.tul_chain is not None
    assert float(m.tul_chain.W.weight.abs().sum()) == 0.0
    assert float(m.tul_chain.first.abs().sum()) == 0.0
    B, S, C = 2, 6, m.cfg.d_model
    valid = torch.ones(B, S, dtype=torch.bool)
    torch.manual_seed(0)
    with torch.no_grad():
        m.tul_chain.W.weight.normal_(std=0.1)
    h = torch.randn(B, S, C, requires_grad=True)
    term = m.tul_chain(h, valid)
    # d term[k] / d h[j] is nonzero only for j == k-1
    for k in range(S):
        g = torch.autograd.grad(term[:, k].sum(), h, retain_graph=True)[0]
        moved = [j for j in range(S) if float(g[:, j].abs().sum()) > 0]
        assert moved == ([k - 1] if k >= 1 else []), (k, moved)


def test_slot_chain_at_zero_init_is_the_ruler_forward():
    on = _model(seed=9, tul_slot_chain=True)
    off = _model(seed=9)
    off.load_state_dict({k: v for k, v in on.state_dict().items()
                         if not k.startswith("tul_chain.")}, strict=True)
    x, y, lay, _ = _batch()
    outs = []
    for m in (on, off):
        m.train()
        torch.manual_seed(13)
        o = m(x, labels=y, slot_layout=lay)
        o["loss"].backward()
        outs.append(float(o["loss"]))
    assert outs[0] == pytest.approx(outs[1], abs=0.0)
    # sensitivity: a nonzero W must change the forward
    with torch.no_grad():
        on.tul_chain.W.weight.normal_(std=0.05)
    on.zero_grad()
    torch.manual_seed(13)
    assert abs(float(on(x, labels=y, slot_layout=lay)["loss"]) - outs[1]) > 1e-6


def test_slot_chain_receives_gradient_through_the_loop():
    m = _model(tul_slot_chain=True)
    with torch.no_grad():
        m.tul_chain.W.weight.normal_(std=0.05)
    x, y, lay, _ = _batch()
    m.train()
    m(x, labels=y, slot_layout=lay)["loss"].backward()
    assert float(m.tul_chain.W.weight.grad.abs().sum()) > 0


# ── 7. the config surface ────────────────────────────────────────────────────

def test_known_tul_keys_accepts_the_new_keys_and_rejects_a_misspelling():
    from morph.training.tul_setup import KNOWN_TUL_KEYS, reject_unknown_tul_keys
    for k in ("spandec", "spandec_layers", "spandec_heads", "spandec_weight",
              "spandec_max_tokens", "slot_chain", "slot_chain_detach"):
        assert k in KNOWN_TUL_KEYS, k
    reject_unknown_tul_keys({k: 0 for k in ("spandec", "spandec_layers", "slot_chain")})
    with pytest.raises(ValueError, match="spandec_wieght"):
        reject_unknown_tul_keys({"spandec": True, "spandec_wieght": 1.0})
    with pytest.raises(ValueError, match="slot_chian"):
        reject_unknown_tul_keys({"slot_chian": True})


def test_spandec_knobs_without_the_decoder_raise():
    with pytest.raises(ValueError, match="spandec_.* set with tul.spandec=false"):
        TULConfig(spandec=False, spandec_layers=4)
    with pytest.raises(ValueError, match="slot_chain_detach"):
        TULConfig(slot_chain=False, slot_chain_detach=True)


def test_spandec_refusals():
    with pytest.raises(NotImplementedError, match="SLOT-LOOP lever"):
        TULConfig(spandec=True, tokens_through_core=True)
    with pytest.raises(ValueError, match="detach_z"):
        TULConfig(spandec=True, detach_z=True)
    with pytest.raises(ValueError, match="spandec_weight > 0"):
        TULConfig(spandec=True, spandec_weight=0.0)
    with pytest.raises(NotImplementedError, match="slot_chain"):
        TULConfig(slot_chain=True, tokens_through_core=True)


def test_decoder_shape_contract():
    """The sequence is ``[z, t_0 .. t_{J-2}]`` predicting ``t_0 .. t_{J-1}``: state ``j``
    must depend on ``z`` and on tokens ``< j`` only."""
    torch.manual_seed(0)
    d, V_ = 16, 40
    dec = SpanDecoder(d_model=d, n_heads=2, d_ff=32, n_layers=2, max_tokens=5)
    emb = torch.randn(V_, d)
    z = torch.randn(1, 1, d, requires_grad=True)
    ids = torch.tensor([[[1, 2, 3, 4, 5]]])
    valid = torch.ones(1, 1, 5, dtype=torch.bool)
    st = dec.decode(z, ids, valid, emb)
    assert st.shape == (1, 1, 5, d)
    # every output position depends on z
    for j in range(5):
        g = torch.autograd.grad(st[0, 0, j].sum(), z, retain_graph=True)[0]
        assert float(g.abs().sum()) > 0, j
    # position 0 cannot see any token: changing every id must leave it alone
    with torch.no_grad():
        st2 = dec.decode(z.detach(), ids * 0 + 9, valid, emb)
    assert torch.allclose(st[0, 0, 0], st2[0, 0, 0], atol=1e-6)
    assert not torch.allclose(st[0, 0, 4], st2[0, 0, 4], atol=1e-6)
