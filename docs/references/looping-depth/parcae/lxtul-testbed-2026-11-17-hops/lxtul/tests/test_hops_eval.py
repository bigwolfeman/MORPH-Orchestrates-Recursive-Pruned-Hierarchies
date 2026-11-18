"""Hop accuracy (lxtul/hops/eval.py) for the flat and token-loop kinds, CPU.

  1. on hand-built logits (all mass on the answer -> 1.0, on a wrong id -> 0.0, right for 1 hop only ->
     that bucket) the accuracy, p(correct) and the per-bucket counts are right, a flat model is scored once;
  2. the mixture it reads is the one the NLL uses: p(correct) = exp(-token_nll) at every answer position,
     and the prediction is the argmax of that mixture (no other id scores higher through the NLL rule);
  3. the plain loop accepts eval depths past its training depth, and the depth reaches the forward;
  4. eval rows: the same documents for every kind, labels aligned, multiples of the batch size enforced.

    PYTHONPATH=.:$MORPH_ROOT python -m pytest -q lxtul/tests/test_hops_eval.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import parcae_lm
from lxtul.hops.base import FAMILIES
from lxtul.hops.eval import answer_scores, hop_accuracy, make_batches, val_metrics
from lxtul.plain_pointer import PlainPointer
from lxtul.tests.hops_support import SLOT, V, tiny_task

SEQ = 128


def _batches(kind="gpt", B=2):
    task = tiny_task(SEQ, 16, rows_per_bucket=4)
    rows = task.eval_rows()
    return task, rows, make_batches(task, kind, None, [r for _, r in rows], B)


class FakeLM(torch.nn.Module):
    """A flat LM whose logits are put by hand: one-hot on the next token where `ok(x)` says so, else on the
    next token + 1. It records every forward."""

    def __init__(self, ok):
        super().__init__()
        self.ok, self.calls = ok, 0
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, return_logits=False, **kw):
        self.calls += 1
        tgt = torch.roll(x, -1, 1)
        hit = self.ok(x)
        pick = torch.where(hit, tgt, (tgt + 1) % V)
        return {"logits": torch.nn.functional.one_hot(pick, V).float() * 30.0}


def _hop_of(batches):
    """x bytes -> hop label per position, for the fake to key on."""
    return {b.x.numpy().tobytes(): b.hops for b in batches}


def test_accuracy_on_hand_built_logits_counts_and_buckets():
    task, rows, batches = _batches()
    hop_of = _hop_of(batches)
    want = {1: 0, 2: 0}
    for b in batches:
        for h in b.hops[b.hops > 0].tolist():
            want[h] += 1
    assert min(want.values()) > 0
    for name, ok, acc in (("all right", lambda x: torch.ones_like(x, dtype=torch.bool), {1: 1.0, 2: 1.0}),
                          ("all wrong", lambda x: torch.zeros_like(x, dtype=torch.bool), {1: 0.0, 2: 0.0}),
                          ("1 hop only", lambda x: hop_of[x.numpy().tobytes()] <= 1, {1: 1.0, 2: 0.0})):
        m = FakeLM(ok)
        res = hop_accuracy(m, "gpt", batches, [1, 6], SLOT, device="cpu")
        assert m.calls == len(batches), "a flat model has no depth: it is scored once"
        for d in ("1", "6"):
            b = res["buckets"][d]
            assert set(b) == {"chain/h1", "chain/h2"}, b
            for h in (1, 2):
                assert b[f"chain/h{h}"]["n"] == want[h] and b[f"chain/h{h}"]["acc"] == acc[h], (name, d, h)
                assert (b[f"chain/h{h}"]["p"] > 0.99) == (acc[h] == 1.0) and (b[f"chain/h{h}"]["p"] < 1e-6) == (acc[h] == 0.0)
            assert res["family_mean"][d]["chain"] == pytest.approx(np.mean(list(acc.values())))
    keys = val_metrics(res, 6)
    assert keys == {"val/hop/chain/acc_h1": 1.0, "val/hop/chain/acc_h2": 0.0, "val/hop/chain/acc_mean": 0.5}


def test_balanced_reading_scores_always_yes_and_always_no_at_one_half():
    from lxtul.hops.eval import balanced_reading
    b = lambda n, a: {"n": n, "acc": a, "p": a}                                   # noqa: E731
    yes_always = {"reach_yes/h1": b(10, 1.0), "reach_yes/h5": b(3, 1.0), "reach_no/h7": b(50, 0.0), "reach_no/h9": b(50, 0.0),
                  "edge_yes/h1": b(20, 1.0), "edge_no/h1": b(20, 0.0)}
    r = balanced_reading(yes_always)
    assert r["reach_balanced"] == r["edge_balanced"] == r["reach_balanced/h1"] == r["reach_balanced/h5"] == 0.5
    mixed = {"reach_yes/h1": b(10, 1.0), "reach_yes/h5": b(10, 0.5), "reach_no/h7": b(30, 1.0), "reach_no/h9": b(10, 0.0),
             "edge_yes/h1": b(5, 0.2), "edge_no/h1": b(5, 0.6)}
    r = balanced_reading(mixed)
    assert r["reach_balanced"] == pytest.approx(0.5 * (0.75 + 0.75))              # no pooled over its answers: 30 / 40
    assert r["reach_balanced/h1"] == pytest.approx(0.5 * (1.0 + 0.75)) and r["edge_balanced"] == pytest.approx(0.4)
    assert "reach_balanced" not in balanced_reading({"chain/h1": b(3, 1.0)})


def test_an_answer_position_with_no_label_is_refused():
    _, _, batches = _batches()
    b = batches[0]
    b.y[(b.hops > 0).nonzero(as_tuple=True)[0][0], (b.hops > 0).nonzero(as_tuple=True)[1][0]] = -100
    with pytest.raises(AssertionError, match="drifted"):
        answer_scores(FakeLM(lambda x: torch.ones_like(x, dtype=torch.bool)), "gpt", b, 1, SLOT)


# ---------------------------------------------------------------------------------- the real mixture
def _gpt_pointer():
    torch.manual_seed(0)
    gc = parcae_lm.create_config("gpt-small-140m", n_embd=64, num_attention_heads=2, num_key_value_heads=2,
                                 intermediate_size=128, n_layer=2, vocab_size=V, block_size=SEQ, use_fused_head=False)
    m = PlainPointer(gc.construct_model(gradient_checkpointing=False), 4).eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2 and not p.abs().any():
                p.normal_(0, 0.05)
        m.pointer.gate_norm_head.bias.copy_(torch.tensor([0.0, 1.5, 1.0, 0.5, 0.0]))     # a pointer that counts
        m.pointer.q.weight.normal_(0, 0.6)
        m.pointer.k.weight.normal_(0, 0.6)
    return m


def test_p_correct_is_the_nll_mixture_and_the_prediction_is_its_argmax():
    _, _, batches = _batches()
    m = _gpt_pointer()
    b = batches[0]
    sel = (b.hops > 0).nonzero(as_tuple=True)
    pred, p, label, hops, fam = answer_scores(m, "gpt", b, 1, SLOT)
    assert len(label) == len(sel[0]) >= 2 and (fam == FAMILIES["chain"]).all()
    nll = m.token_nll(b.x, b.y)
    assert torch.allclose(p, torch.exp(-nll[sel]), rtol=1e-4), (p, torch.exp(-nll[sel]))
    top = []
    for alt in [pred] + [(label * 7 + k) % V for k in range(1, 12)] + [label]:     # score ids through the NLL rule
        y2 = b.y.clone()
        y2[sel] = alt
        top.append(torch.exp(-m.token_nll(b.x, y2)[sel]))
    best = top[0]
    assert all((best >= t - 1e-6).all() for t in top), "an id scores above the prediction"
    assert (best >= p - 1e-6).all()
    # the pointer matters: with it off the prediction is the plain LM's argmax and the table changes
    pred_on = pred
    m.use_pointer = False
    pred_off = answer_scores(m, "gpt", b, 1, SLOT)[0]
    m.use_pointer = True
    assert (pred_on != pred_off).any(), "the pointer mixture never changed an argmax: the head would be dead weight in this test"


def _plain(mean_rec=2, pointer=0):
    torch.manual_seed(0)
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=64, num_attention_heads=2, num_key_value_heads=2, intermediate_size=128,
        recurrent_embedding_dimension=64, recurrent_intermediation_embedding_dimension=128, n_layers_in_prelude=1,
        n_layers_in_recurrent_block=1, n_layers_in_coda=1, vocab_size=V, block_size=SEQ, mean_recurrence=mean_rec,
        mean_backprop_depth=mean_rec)
    m = cfg.construct_model(gradient_checkpointing=False)
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2 and not p.abs().any():
                p.normal_(0, 0.05)
    return (PlainPointer(m, pointer) if pointer else m).eval()


@pytest.mark.parametrize("pointer", [0, 4])
def test_the_plain_loop_takes_depths_past_its_training_depth(pointer):
    _, _, batches = _batches("plain")
    m = _plain(2, pointer)
    res = hop_accuracy(m, "plain", batches, [1, 2, 12, 16], SLOT, device="cpu")
    ps = {d: tuple(round(v["p"], 9) for v in res["buckets"][d].values()) for d in res["buckets"]}
    assert set(ps) == {"1", "2", "12", "16"} and len(set(ps.values())) == 4, ps        # every depth reaches the forward
    assert all(0.0 <= v["acc"] <= 1.0 and v["n"] > 0 for d in res["buckets"].values() for v in d.values())


# ------------------------------------------------------------------------------------------ eval rows
def test_eval_rows_are_the_same_documents_for_every_kind_and_must_fill_batches():
    task, rows, flat = _batches("plain", B=2)
    again = tiny_task(SEQ, 16, rows_per_bucket=4).eval_rows()
    assert all(a[0] == b[0] and (a[1].tokens == b[1].tokens).all() for a, b in zip(rows, again))
    assert [n for n, _ in rows].count("chain/h1") == [n for n, _ in rows].count("chain/h2") == 4
    train = {tuple(task.train_row(0, r).tokens.tolist()) for r in range(200)}
    assert not train & {tuple(r.tokens.tolist()) for _, r in rows}, "an eval row is a training row"
    gpt = make_batches(task, "gpt", None, [r for _, r in rows], 2)
    assert all(torch.equal(a.x, b.x) and torch.equal(a.hops, b.hops) for a, b in zip(flat, gpt))
    with pytest.raises(ValueError, match="multiple"):
        make_batches(task, "gpt", None, [r for _, r in rows], 3)
