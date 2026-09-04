"""GL1b gates — the MUX write-target loss and the attention-lift instrument.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_tul_gl1b.py -v

Paper: MUX, arXiv 2607.18264, read in full. Eq. 2 (multiplexed target), Eq. 3 (tied
linear-softmax head), Eq. 4 (KL local loss), Table 9 (rho 0.9, tau 1.0, beta 1.0),
§8.3 (reasoning attention lift).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import morph.model.attention as _attn
from morph.model.attn_lift import (
    AttnLiftStats,
    capture_attn_lift,
    first_token_of_span_mask,
)
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig, mux_span_targets
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10
SLOT_ID = 4


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _batch(B: int = 2, n: int = 90, seed: int = 0):
    spec = TulLayoutSpec(seq_len=48, prefix_k=2, max_slots=12, slot_id=SLOT_ID)
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == SLOT_ID] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=SLOT_ID, slot_seed="boundary", tg_restrict=True,
                emit_weight=0.0, plast_weight=1.0, token_state_dropout=0.0,
                sigreg_lambda=0.0, eval_ablations=True,
                mux_beta=1.0, mux_target="own", mux_detach_head=False)
    base.update(kw)
    return TULConfig(**base)


def _model(seed: int = 0, tul=None, **kw) -> MORPHTransformer:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256,
        context_len=256, n_prelude=2, n_core=0, n_coda=2, mean_depth=2, max_depth=2,
        bptt_depth=1, channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=8, bigram_hash_vocab=V,
        use_kernels=False, hc_use_kernel=False, dropout=0.0,
        retention=True, retention_layers=(1,), retention_chunk=8, retention_carry=True,
        tul=tul if tul is not None else _tul(),
    )
    base.update(kw)
    torch.manual_seed(seed)
    return MORPHTransformer(MORPHConfig(**base))


# ── 1. the multiplexed target (Eq. 2) ────────────────────────────────────────

def _dense_targets(ids, layout, rho, target):
    """[B, S, V] dense mux targets, built the slow obvious way for cross-checking."""
    pos_valid, alpha, tgt_slot, sup = mux_span_targets(ids, layout, rho, target=target)
    B, L = ids.shape
    S = layout.slot_index.shape[1]
    out = torch.zeros(B, S, V)
    for b in range(B):
        for p in range(L):
            if bool(pos_valid[b, p]):
                out[b, int(tgt_slot[b, p]), int(ids[b, p])] += float(alpha[b, p])
    return out, sup


def test_own_span_target_is_the_slots_own_span_and_sums_to_one():
    x, _y, lay, _ = _batch()
    tgt, sup = _dense_targets(x, lay, 0.9, "own")
    assert int(sup.sum()) > 4
    tot = tgt.sum(-1)
    assert torch.allclose(tot[sup], torch.ones(int(sup.sum())), atol=1e-5), \
        "Eq. 2's alpha must be normalised within the span"
    assert torch.equal(tgt[~sup], torch.zeros_like(tgt[~sup])), "pad slots got mass"
    # the mass really is the slot's OWN span's tokens, not the next span's
    for b in range(x.shape[0]):
        for i in torch.nonzero(sup[b]).flatten().tolist():
            own = ((lay.bag_id[b] == i) & ~lay.slot_mask[b]).nonzero().flatten()
            assert own.numel() > 0
            ids_own = set(x[b, own].tolist())
            got = set(torch.nonzero(tgt[b, i]).flatten().tolist())
            assert got == ids_own, f"slot {i}: target ids {got} != own-span ids {ids_own}"


def test_own_and_next_targets_differ_and_own_supervises_more_slots():
    x, _y, lay, _ = _batch()
    t_own, s_own = _dense_targets(x, lay, 0.9, "own")
    t_nxt, s_nxt = _dense_targets(x, lay, 0.9, "next")
    assert not torch.allclose(t_own, t_nxt)
    # "next" cannot supervise the last real slot (no span after it); "own" can.
    assert int(s_own.sum()) > int(s_nxt.sum())


def test_geometric_weights_follow_rho_to_the_power_j():
    x, _y, lay, _ = _batch()
    rho = 0.7
    pos_valid, alpha, tgt_slot, sup = mux_span_targets(x, lay, rho, target="own")
    b = 0
    i = int(torch.nonzero(sup[b]).flatten()[1])
    pos = ((lay.bag_id[b] == i) & ~lay.slot_mask[b] & pos_valid[b]).nonzero().flatten()
    assert pos.numel() >= 3
    a = alpha[b, pos].double()
    want = torch.tensor([rho ** j for j in range(pos.numel())], dtype=torch.float64)
    want = want / want.sum()
    assert torch.allclose(a, want, atol=1e-6), f"got {a.tolist()} want {want.tolist()}"


def test_injectivity_same_sequence_matches_and_one_changed_token_does_not():
    """Eq. 2's whole point is losslessness.

    Note what "identical spans" can and cannot mean here: the packer inserts prefix_k
    slot positions between spans, so two spans of the SAME row generally have different
    LENGTHS and their normalised alphas differ for that reason alone. The clean identity
    is between two identical ROWS.
    """
    spec = TulLayoutSpec(seq_len=48, prefix_k=2, max_slots=12, slot_id=SLOT_ID)
    rng = np.random.default_rng(5)
    row = rng.integers(5, V, size=(90,))
    row[row == SLOT_ID] = 5
    row[::6] = DOT
    same = np.stack([row, row]).astype(np.int64)
    x, _y, lay, _ = slot_layout_from_ids(same, _rule(), spec)
    tgt, sup = _dense_targets(x, lay, 0.9, "own")
    assert torch.equal(sup[0], sup[1])
    assert torch.allclose(tgt[0], tgt[1], atol=1e-7), \
        "two identical rows multiplexed differently"

    diff = same.copy()
    j = int(np.nonzero(diff[1] != DOT)[0][7])
    diff[1, j] = 5 if diff[1, j] != 5 else 6
    x2, _y2, lay2, _ = slot_layout_from_ids(diff, _rule(), spec)
    tgt2, _s2 = _dense_targets(x2, lay2, 0.9, "own")
    assert torch.allclose(tgt2[0], tgt[0], atol=1e-7), "the untouched row moved"
    assert not torch.allclose(tgt2[1], tgt[1], atol=1e-6), \
        "changing one token left the multiplexed target unchanged — not injective"


def test_order_within_a_span_changes_the_target():
    """Subset-sum separation: the same multiset in a different ORDER must multiplex
    differently, or the target is not lossless (the paper's uniform-weighting
    counterexample)."""
    from morph.model.tul import mux_span_targets as _mst
    spec = TulLayoutSpec(seq_len=48, prefix_k=2, max_slots=12, slot_id=SLOT_ID)
    rng = np.random.default_rng(6)
    row = rng.integers(12, V, size=(90,))
    row[::6] = DOT
    a = np.stack([row, row]).astype(np.int64)
    b = a.copy()
    b[1, 1], b[1, 3] = b[1, 3], b[1, 1]        # swap two tokens inside span 0
    xa, _ya, la, _ = slot_layout_from_ids(a, _rule(), spec)
    xb, _yb, lb, _ = slot_layout_from_ids(b, _rule(), spec)
    ta, _ = _dense_targets(xa, la, 0.9, "own")
    tb, _ = _dense_targets(xb, lb, 0.9, "own")
    assert not torch.allclose(ta[1, 0], tb[1, 0], atol=1e-6), \
        "swapping two tokens inside a span did not change its multiplexed target"


def test_repeated_tokens_accumulate_mass_rather_than_overwrite():
    """A subword appearing twice carries the SUM of its two alphas (Eq. 2 sums
    one-hots). This is also what makes the entropy computation non-trivial."""
    spec = TulLayoutSpec(seq_len=48, prefix_k=2, max_slots=12, slot_id=SLOT_ID)
    ids = np.full((1, 90), 7, dtype=np.int64)      # every token identical
    ids[0, ::6] = DOT
    x, _y, lay, _ = slot_layout_from_ids(ids, _rule(), spec)
    tgt, sup = _dense_targets(x, lay, 0.9, "own")
    i = int(torch.nonzero(sup[0]).flatten()[1])
    nz = torch.nonzero(tgt[0, i]).flatten()
    assert nz.numel() <= 2, "a single-token span should occupy ~one vocab entry"
    assert float(tgt[0, i].sum()) == pytest.approx(1.0, abs=1e-5)


# ── 2. the loss, the null, and where its gradient goes ───────────────────────

def test_mux_rel_is_exactly_one_at_the_marginal_predictor():
    """The honesty null. `mux_rel = CE(target, model) / CE(target, marginal)`, so a
    model that predicts the batch marginal scores exactly 1.0. Without this the CE alone
    looks like progress it is not."""
    x, y, lay, _ = _batch()
    m = _model()
    m.train()
    out = m(x, y, slot_layout=lay)
    stats = {k: float(out[k]) for k in ("mux_ce", "mux_null", "mux_rel", "mux_entropy",
                                        "mux_kl")}
    assert stats["mux_rel"] == pytest.approx(stats["mux_ce"] / stats["mux_null"], rel=1e-6)
    assert stats["mux_kl"] == pytest.approx(stats["mux_ce"] - stats["mux_entropy"],
                                            rel=1e-6)
    assert stats["mux_entropy"] > 0.0 and stats["mux_kl"] > 0.0

    # Now force the head to emit the marginal exactly and check rel == 1.
    from morph.model.tul import mux_span_targets as _mst
    pos_valid, alpha, tgt_slot, sup = _mst(x, lay, m.cfg.tul.mux_rho, target="own")
    n_sup = sup.sum().double().clamp(min=1.0)
    pbar = torch.zeros(V, dtype=torch.float64)
    ids_flat = x.reshape(-1)[pos_valid.reshape(-1)]
    pbar.scatter_add_(0, ids_flat, alpha.reshape(-1)[pos_valid.reshape(-1)].double())
    pbar = pbar / n_sup
    ce_marginal = -(alpha.reshape(-1)[pos_valid.reshape(-1)].double()
                    * pbar[ids_flat].clamp_min(1e-30).log()).sum() / n_sup
    assert float(ce_marginal) == pytest.approx(stats["mux_null"], rel=1e-6)


def test_the_marginal_null_is_tighter_than_uniform():
    """We picked the batch marginal over uniform; that claim has to hold."""
    x, y, lay, _ = _batch()
    m = _model()
    out = m(x, y, slot_layout=lay)
    assert float(out["mux_null"]) < math.log(V), (
        f"the marginal null ({float(out['mux_null']):.3f}) is not tighter than uniform "
        f"({math.log(V):.3f})")


def test_mux_loss_reaches_the_slot_write_path_and_the_tied_embeddings():
    x, y, lay, _ = _batch()
    m = _model()
    m.train()
    out = m(x, y, slot_layout=lay)
    live = out["mux_local_live"]
    for name, prm in (("W_sent", m.tul.W_sent.weight), ("E_slot", m.tul.E_slot),
                      ("embed", m._euc_embed_leaf())):
        g = torch.autograd.grad(live, prm, retain_graph=True, allow_unused=True)[0]
        assert g is not None and float(g.abs().sum()) > 0, \
            f"the MUX loss does not reach {name}"


def test_detaching_the_head_does_not_isolate_the_embedding_table():
    """A measured caveat that the config states and that must stay true: with
    ``mux_detach_head=True`` the MUX gradient STILL reaches the embedding table, because
    the slot SEED is ``E_slot + W_sent . embed(t_last)``. Detaching the head detaches
    the readout, not the input path."""
    x, y, lay, _ = _batch()
    for det in (True, False):
        m = _model(tul=_tul(mux_detach_head=det))
        m.train()
        live = m(x, y, slot_layout=lay)["mux_local_live"]
        g = torch.autograd.grad(live, m._euc_embed_leaf(), allow_unused=True)[0]
        assert g is not None and float(g.abs().sum()) > 0, \
            f"detach={det}: expected the seed path to carry MUX gradient to the table"


def test_mux_grad_share_is_a_fraction_and_moves_with_beta():
    x, y, lay, _ = _batch()
    m = _model()
    s = m.tul_mux_grad_share(x, y, lay)
    assert 0.0 <= s["mux_embed_grad_share"] <= 1.0
    assert s["mux_embed_grad_norm"] > 0 and s["ce_embed_grad_norm"] > 0
    m2 = _model(tul=_tul(mux_beta=0.01))
    s2 = m2.tul_mux_grad_share(x, y, lay)
    assert s2["mux_embed_grad_share"] < s["mux_embed_grad_share"], \
        "a 100x smaller beta did not lower the auxiliary's share of the table gradient"


def test_mux_beta_zero_is_bit_identical_to_gl1():
    """The whole point of a knob: silencing it must restore the previous arm exactly."""
    x, y, lay, _ = _batch()
    a = _model(seed=3, tul=_tul(mux_beta=0.0))
    b = _model(seed=3, tul=TULConfig(prefix_k=2, slot_id=SLOT_ID, slot_seed="boundary",
                                     tg_restrict=True, emit_weight=0.0, plast_weight=1.0,
                                     token_state_dropout=0.0, sigreg_lambda=0.0,
                                     eval_ablations=True))
    a.eval(); b.eval()
    oa = a(x, y, slot_layout=lay)
    ob = b(x, y, slot_layout=lay)
    assert "mux_local" not in oa and "mux_local" not in ob
    assert float(oa["loss"].detach()) == pytest.approx(float(ob["loss"].detach()),
                                                       rel=1e-12)


def test_pad_slots_are_excluded_from_the_mux_loss():
    x, y, lay, _ = _batch()
    assert bool((~lay.slot_valid).any()), "fixture has no pad slots; the test is inert"
    _pv, _a, _t, sup = mux_span_targets(x, lay, 0.9, target="own")
    assert not bool((sup & ~lay.slot_valid).any()), "a pad slot is being supervised"


# ── 3. the attention-lift instrument (MUX §8.3) ──────────────────────────────

def test_the_lift_wrapper_reproduces_the_shipped_attention():
    """The instrument recomputes the window mask to count mass. If that recomputation
    ever diverges from ``_window_fallback``'s, every lift number is measuring a mask the
    model does not use. Pinned here."""
    from morph.model.attn_lift import _window_mask
    torch.manual_seed(0)
    B, H, S, D = 2, 2, 20, 8
    q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
    extra = torch.rand(B, 1, S, S) > 0.3
    for em in (None, extra):
        for nsr in (0, 3):
            want = _attn._window_fallback(q, k, v, 6, q.device, 0.35, nsr, em)
            mask = _window_mask(S, 6, nsr, q.device, em)
            bias = torch.where(mask, 0.0, float("-inf"))
            sc = torch.einsum("bhid,bhjd->bhij", q, k) * 0.35 + bias
            got = torch.einsum("bhij,bhjd->bhid", torch.softmax(sc, -1), v)
            # A query row with NO visible key: the shipped SDPA ZERO-FILLS it, a bare
            # softmax gives NaN. Compared only where the row has a key — and the
            # instrument excludes exactly those rows for the same reason.
            live = mask.expand(q.shape[0], 1, S, S).any(-1)        # [B, 1, S]
            live = live.expand(-1, q.shape[1], -1)                 # [B, H, S]
            assert torch.allclose(want[live], got[live], atol=1e-5), (
                f"instrument mask diverged from the shipped one "
                f"(extra={em is not None}, n_skip_rope={nsr})")


def test_the_instrument_does_not_change_the_model_output():
    """It must be a measurement, not an intervention. An earlier revision returned its
    own `weights @ v`, a fully-masked row softmaxed to NaN, and the NaN propagated into
    every later layer's measurement."""
    x, _y, lay, _ = _batch()
    m = _model()
    m.eval()
    with torch.no_grad():
        a = m(x, labels=None, slot_layout=lay)["logits"]
    st = AttnLiftStats()
    with capture_attn_lift(lay, st):
        with torch.no_grad():
            b = m(x, labels=None, slot_layout=lay)["logits"]
    assert torch.equal(a, b), "the lift instrument perturbed the forward"
    # The slot_id column is a structural -inf (spec §3.1, masked at generation), so
    # finiteness is checked on every OTHER column.
    keep = torch.ones(b.shape[-1], dtype=torch.bool)
    keep[SLOT_ID] = False
    assert torch.isfinite(b[..., keep]).all()
    assert len(st.calls) == 4                      # 2 prelude + 2 coda window calls


def test_first_span_tokens_see_only_slots_under_the_restriction():
    """The hand-computable case. Under ``tg_restrict`` a span's FIRST token has no
    same-span predecessor, so every key it can see is a slot: share == 1.0 and therefore
    mass == 1.0 and lift == 1.0 exactly. If the instrument reports anything else it is
    not counting what it claims to count."""
    x, _y, lay, _ = _batch()
    m = _model()
    m.eval()
    p = m.tul_attn_lift_probe(x, lay)
    assert p["attn_slot_mass_first_tok"] == pytest.approx(1.0, abs=1e-4)
    assert p["attn_lift_first_tok"] == pytest.approx(1.0, abs=1e-4)
    assert 0.0 < p["attn_slot_share"] < 1.0, "ordinary token queries should see both"
    assert p["attn_lift"] == pytest.approx(
        p["attn_slot_mass"] / p["attn_slot_share"], rel=0.35), \
        "lift is not the mass/share ratio it is defined as"


def test_the_unrestricted_control_gets_a_lower_slot_share():
    """Sanity on the share denominator: without the mask a token query can see many more
    token keys, so slots are a SMALLER fraction of what is visible."""
    x, _y, lay, _ = _batch()
    a = _model(tul=_tul(tg_restrict=True)).eval()
    b = _model(tul=_tul(tg_restrict=False)).eval()
    pa = a.tul_attn_lift_probe(x, lay)
    pb = b.tul_attn_lift_probe(x, lay)
    assert pb["attn_slot_share"] < pa["attn_slot_share"], (
        f"restricted share {pa['attn_slot_share']:.3f} vs unrestricted "
        f"{pb['attn_slot_share']:.3f}")
    for k in ("attn_lift", "attn_lift_first_tok", "attn_slot_mass"):
        assert pb[k] == pb[k], f"{k} is NaN on the unrestricted control"


def test_first_token_of_span_mask_is_what_it_says():
    x, _y, lay, _ = _batch()
    ft = first_token_of_span_mask(lay)
    assert bool(ft.any())
    for b in range(x.shape[0]):
        for p in torch.nonzero(ft[b]).flatten().tolist():
            assert not bool(lay.slot_mask[b, p])
            assert p == 0 or bool(lay.slot_mask[b, p - 1]), \
                f"position {p} is not a span start"
        # every real span contributes exactly one
        n_spans = int(lay.slot_valid[b].sum())
        assert int(ft[b].sum()) >= n_spans - 1


# ── 4. the arm ───────────────────────────────────────────────────────────────

def test_gl1b_config_is_gl1_plus_mux_minus_sigreg():
    import os
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    cd = os.path.abspath("morph/configs")
    with initialize_config_dir(version_base=None, config_dir=cd):
        a = compose(config_name="tul_gl1")
    with initialize_config_dir(version_base=None, config_dir=cd):
        b = compose(config_name="tul_gl1b")

    def _flat(d, pre=""):
        out = {}
        for k, v in d.items():
            out.update(_flat(v, f"{pre}{k}.")) if isinstance(v, dict) else \
                out.update({f"{pre}{k}": v})
        return out

    fa = _flat(OmegaConf.to_container(a, resolve=True))
    fb = _flat(OmegaConf.to_container(b, resolve=True))
    _MISS = object()
    diff = {k for k in set(fa) | set(fb) if fa.get(k, _MISS) != fb.get(k, _MISS)}
    # mux_rho / mux_tau / mux_activate_at are RESTATED in the arm at their code
    # defaults, deliberately: "a run must be reproducible from its config alone", and
    # these are the paper's Table 9 numbers. Asserted equal to the code defaults below
    # so the restatement cannot drift from TULConfig.
    assert diff == {"tul.mux_beta", "tul.mux_target", "tul.mux_detach_head",
                    "tul.mux_rho", "tul.mux_tau", "tul.mux_activate_at",
                    "tul.sigreg_lambda", "wandb.name"}, f"unexpected diff: {diff}"
    _d = TULConfig(prefix_k=2, slot_id=SLOT_ID)
    for k in ("mux_rho", "mux_tau", "mux_activate_at"):
        assert fb[f"tul.{k}"] == getattr(_d, k), (
            f"tul_gl1b restates {k}={fb[f'tul.{k}']} but the code default is "
            f"{getattr(_d, k)} — one of them moved")
    assert fb["tul.mux_beta"] == 1.0 and fb["tul.mux_rho"] == 0.9
    assert fb["tul.mux_tau"] == 1.0            # paper Table 9
    assert fb["tul.mux_target"] == "own"
    assert fb["tul.sigreg_lambda"] == 0.0      # isolate the MUX target
    assert fb["model.n_core"] == 0 and fb["tul.tg_restrict"] is True


def test_the_arm_config_actually_reaches_the_model():
    """Composing a YAML is not the same as the model receiving it.

    ``mux_target: own`` sat in tul_gl1b.yaml for a while and resolved to ``next``,
    because ``build_tul_runtime`` had no line for it — the arm would have trained the
    wrong target with a perfectly correct-looking config. Every knob this arm turns is
    checked on the RESOLVED TULConfig here, not on the composed dict.
    """
    import os
    from hydra import compose, initialize_config_dir
    from morph.training.tul_setup import build_tul_runtime
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("morph/configs")):
        c = compose(config_name="tul_gl1b")
    rt = build_tul_runtime(c)
    mc = rt.model_cfg
    assert mc.mux_target == "own", f"mux_target resolved to {mc.mux_target!r}"
    assert mc.mux_beta == 1.0
    assert mc.mux_rho == 0.9 and mc.mux_tau == 1.0        # paper Table 9
    assert mc.mux_detach_head is False
    assert mc.sigreg_lambda == 0.0
    assert mc.tg_restrict is True and mc.slot_seed == "boundary"
    assert mc.eval_ablations is True
    assert rt.manifest["mux_target"] == "own", "the manifest does not log mux_target"


def test_the_global_detach_default_stays_on():
    """GL1b sets mux_detach_head=false for ITSELF. The global default must remain True —
    detach OFF is a recorded divergence (arm v1a, step 2800)."""
    assert TULConfig(prefix_k=2, slot_id=SLOT_ID).mux_detach_head is True
