"""Span-level NextLat (`tul.nextlat_weight`, arXiv 2511.05963; 2026-09-25).

Files: morph/model/tul_nextlat.py (`SpanTransition`, `nextlat_pairs`,
`span_token_embeddings`), morph/model/transformer.py (the build, `_tul_nextlat_loss`,
`_nextlat_draft_stats`, the `nextlat_weighted` group), morph/model/tul.py
(`_check_nextlat`), morph/training/tul_setup.py and train.py (the key, the subtraction,
the log).

What each test pins:
  * the transition is a GRU started at z_s over the span's first `length` tokens, then
    `out`: equal to a hand-rolled GRUCell loop with the same weights; padding past
    `length` never reaches the output; `out` is the identity at build;
  * the build is RNG-neutral: every base weight is byte-identical to the model without it,
    and the forward's model loss (loss minus `nextlat_weighted`) is the parent's loss;
  * the term is the SmoothL1 mean over valid (s, s+1) pairs: the target is stop-graded
    (a slot that is only ever a target gets no gradient from it), a slot that is a source
    does, and a pad or unpaired slot does not;
  * under K rollouts the pooled term is the mean of each rollout's own term;
  * the copy baseline equals the term when the transition returns z_s; the draft gap is
    exactly 0 when the transition returns the true z_{s+1} and nonzero when it returns z_s;
  * the tied table gets no gradient from the term;
  * the refusals hold.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from test_tul_lxtul_e import _D_FF, K, _tc
from test_tul_strict_geometry import _pack, _tiny

from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_nextlat import SpanTransition, nextlat_pairs
from morph.model.tul_spandec import span_slots


def _model(k: int = K, weight: float = 1.0, seed: int = 1234) -> MORPHTransformer:
    torch.manual_seed(seed)
    kw = {} if weight == 0.0 else {"nextlat_weight": weight}
    m = MORPHTransformer(_tiny(tul=_tc(k, **kw), d_ff=_D_FF, dropout=0.0))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.float()


def test_transition_is_a_gru_from_z_s_over_the_valid_tokens():
    torch.manual_seed(0)
    C, J = 16, 7
    tr = SpanTransition(C)
    with torch.no_grad():
        tr.out.weight.add_(0.1 * torch.randn(C, C))
        tr.out.bias.normal_()
    z = torch.randn(5, C)
    emb = torch.randn(5, J, C)
    lengths = torch.tensor([1, 3, 7, 4, 2])
    got = tr(z, emb, lengths)
    cell = torch.nn.GRUCell(C, C)
    with torch.no_grad():
        cell.weight_ih.copy_(tr.gru.weight_ih_l0)
        cell.weight_hh.copy_(tr.gru.weight_hh_l0)
        cell.bias_ih.copy_(tr.gru.bias_ih_l0)
        cell.bias_hh.copy_(tr.gru.bias_hh_l0)
    for i in range(5):
        h = z[i:i + 1]
        for t in range(int(lengths[i])):
            h = cell(emb[i:i + 1, t], h)
        assert torch.allclose(got[i], tr.out(h)[0], atol=1e-5), i
    # Padding past `length` never reaches the output.
    emb2 = emb.clone()
    emb2[0, 1:] = 99.0
    assert torch.allclose(tr(z, emb2, lengths)[0], got[0], atol=1e-6)


def test_out_is_the_identity_at_build_and_the_build_draws_no_global_rng():
    s0 = torch.get_rng_state()
    tr = SpanTransition(12)
    assert torch.equal(torch.get_rng_state(), s0)
    assert torch.equal(tr.out.weight, torch.eye(12)) and torch.equal(tr.out.bias, torch.zeros(12))
    assert tr._ternary_exclude and tr.gru._ternary_exclude and tr.out._ternary_exclude


@pytest.mark.parametrize("k", [1, K])
def test_build_is_rng_neutral_and_the_model_loss_is_the_parents(k):
    _ids, inp, lab, layout = _pack()
    off, on = _model(k, 0.0), _model(k, 1.0)
    sd_off, sd_on = off.state_dict(), on.state_dict()
    extra = {n for n in sd_on if n not in sd_off}
    assert extra and all(n.startswith("tul_nextlat.") for n in extra)
    for n, v in sd_off.items():
        assert torch.equal(v, sd_on[n]), n
    off.train(), on.train()
    torch.manual_seed(5)
    o_off = off(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(5)
    o_on = on(inp, labels=lab, slot_layout=layout)
    assert float(o_on["nextlat_weighted"]) > 0.0
    assert torch.allclose(o_on["loss"] - o_on["nextlat_weighted"], o_off["loss"], atol=1e-5)


def _leaf_readout(m: MORPHTransformer, z: torch.Tensor):
    m._readout = lambda h: z


def test_targets_are_stop_graded_sources_get_gradient_and_unpaired_slots_get_none():
    _ids, inp, lab, layout = _pack()
    m = _model(1).train()
    B, S = layout.slot_valid.shape
    C = m.cfg.d_model
    torch.manual_seed(3)
    z = torch.randn(B, S, C, requires_grad=True)
    _leaf_readout(m, z)
    loss = m._tul_nextlat_loss(torch.zeros(B, S, C), inp, layout)
    loss.backward()
    _ids2, valid = span_slots(inp, layout, m._nextlat_J, shift=1)
    pair = nextlat_pairs(valid)
    src = pair
    g = z.grad.abs().sum(-1)                                            # [B, S]
    assert pair.any()
    assert (g[src] > 0).all(), "a source slot got no gradient"
    assert (g[~src] == 0).all(), "a target-only, pad or unpaired slot got gradient"
    # The target slots of the pairs exist and are not all sources (else the test is vacuous).
    tgt = torch.zeros_like(pair)
    tgt[:, 1:] = pair[:, :-1]
    assert (tgt & ~src).any()


def test_the_term_is_the_smoothl1_mean_over_valid_pairs():
    _ids, inp, lab, layout = _pack()
    m = _model(1).train()
    B, S = layout.slot_valid.shape
    C = m.cfg.d_model
    torch.manual_seed(4)
    z = torch.randn(B, S, C)
    _leaf_readout(m, z)
    stats = {}
    loss = m._tul_nextlat_loss(torch.zeros(B, S, C), inp, layout, stats=stats)
    ids, valid = span_slots(inp, layout, m._nextlat_J, shift=1)
    pair = nextlat_pairs(valid)
    terms = []
    for b in range(B):
        for s in range(S - 1):
            if not pair[b, s]:
                continue
            n = int(valid[b, s].sum())
            e = F.rms_norm(F.embedding(ids[b, s, :n], m.embed.lm_weight().detach()).float(), (C,))
            zh = m.tul_nextlat(z[b, s:s + 1], e.unsqueeze(0), torch.tensor([n]))
            terms.append(F.smooth_l1_loss(zh[0], z[b, s + 1].float()))
    assert terms
    assert float(loss) == pytest.approx(float(torch.stack(terms).mean()), rel=1e-4)
    assert float(stats["nextlat_pairs"]) == len(terms)


def test_rollouts_pool_each_rollouts_own_term():
    _ids, inp, lab, layout = _pack()
    m = _model(K).train()
    B, S = layout.slot_valid.shape
    C = m.cfg.d_model
    torch.manual_seed(6)
    z = torch.randn(K * B, S, C)
    lay_k = layout.repeat_rows(K)
    _leaf_readout(m, z)
    pooled = m._tul_nextlat_loss(torch.zeros(1), inp.repeat(K, 1), lay_k, n_rollouts=K)
    per = []
    for r in range(K):
        _leaf_readout(m, z[r * B:(r + 1) * B])
        per.append(m._tul_nextlat_loss(torch.zeros(1), inp, layout))
    assert float(pooled) == pytest.approx(float(torch.stack(per).mean()), rel=1e-5)


def test_copy_baseline_and_the_draft_instrument_read_what_they_claim():
    _ids, inp, lab, layout = _pack()
    m = _model(1).eval()
    assert m.tul_spandec_par is not None
    B, S = layout.slot_valid.shape
    C = m.cfg.d_model
    torch.manual_seed(7)
    z = torch.randn(B, S, C)
    _leaf_readout(m, z)
    ids, valid = span_slots(inp, layout, m._nextlat_J, shift=1)
    pair = nextlat_pairs(valid)[:, :-1]
    z_s = z[:, :-1].reshape(-1, C)
    z_t = z[:, 1:].reshape(-1, C)

    def run(fake):
        m.tul_nextlat.forward = lambda zs, emb, ln: fake.float()
        st = {}
        with torch.no_grad():
            m._tul_nextlat_loss(torch.zeros(1), inp, layout, stats=st)
        return st

    st = run(z_s)          # the no-change guess
    assert float(st["nextlat_l1"]) == pytest.approx(float(st["nextlat_copy_l1"]), rel=1e-6)
    st = run(z_t)          # a perfect draft
    assert float(st["nextlat_l1"]) == 0.0
    assert float(st["nextlat_draft_tokens"]) > 0
    assert float(st["nextlat_draft_gap"]) == pytest.approx(0.0, abs=1e-6)
    # A wrong draft reads a different CE. On random, untrained states its sign is not
    # meaningful (a zero state can read LOWER than a random one), so only |gap| is pinned.
    st = run(z_s)
    assert abs(float(st["nextlat_draft_gap"])) > 1e-3
    assert pair.any()


def test_the_tied_table_gets_no_gradient_from_the_term():
    _ids, inp, lab, layout = _pack()
    m = _model(1).train()
    out = m(inp, labels=lab, slot_layout=layout)
    m.zero_grad()
    B, S = layout.slot_valid.shape
    z = torch.randn(B, S, m.cfg.d_model, requires_grad=True)
    _leaf_readout(m, z)
    m._tul_nextlat_loss(torch.zeros(1), inp, layout).backward()
    # `lm_weight()` may return a non-leaf view of the tied table, whose `.grad` is always
    # None; the parameters themselves are what must stay untouched.
    emb_params = list(m.embed.named_parameters())
    assert emb_params
    for n, p in emb_params:
        assert p.grad is None or float(p.grad.abs().sum()) == 0.0, n
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.tul_nextlat.parameters())
    assert torch.isfinite(out["loss"])


@pytest.mark.parametrize("kw,exc,match", [
    (dict(nextlat_weight=-1.0), ValueError, "nextlat_weight must be >= 0"),
    (dict(nextlat_weight=1.0, nextlat_beta=0.0), ValueError, "nextlat_beta > 0"),
    (dict(nextlat_beta=0.5), ValueError, "silently ignored"),
    (dict(nextlat_weight=1.0, tokens_through_core=True), NotImplementedError, "tokens_through_core"),
    (dict(nextlat_weight=1.0, fan_k=2), (NotImplementedError, ValueError), "fan"),
])
def test_refusals(kw, exc, match):
    with pytest.raises(exc, match=match):
        TULConfig(**kw)


class _AuxStub(torch.nn.Module):
    """The val path's view of a NextLat model: a loss that includes the weighted term."""

    def tul_forward_with_plan_nats(self, x, y, layout):
        return {"loss": torch.tensor(5.0), "nextlat_weighted": torch.tensor(1.25),
                "ce_tokens": 3.75, "layer_passes": 8.0, "n_tokens": 4.0}


class _Layout:
    stats: dict = {}

    def to(self, device):
        return self


def test_the_val_loss_subtracts_the_weighted_term():
    from morph.training.train import evaluate
    x = torch.zeros(1, 4, dtype=torch.long)
    avg, _ppl = evaluate(_AuxStub(), torch.device("cpu"), iter([(x, x, _Layout())]),
                         n_batches=1, tul=True, extra={})
    assert avg == pytest.approx(3.75)
