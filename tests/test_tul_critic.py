"""`tul.grad_pass_energy: critic` — the within-context improvement critic.

Arm `slot-spandec-strict-critic`, 2026-09-12. Wolfe: "a within-context critic that scores
whether the state after pass t beats the state after pass t-1 on the same coda loss." The
label is a PAIRWISE comparison of two candidate slot states, each written into that slot's
prefix cells and scored by a REPLAY of the real coda, so everything the two candidates
share cancels.

What this file pins, one test per invariant:

* **Off is nothing.** No module, no key, and `_tul_critic_loss` is never called.
* **On is PURELY ADDITIVE and RNG-NEUTRAL.** Every shared parameter is byte-identical to an
  off-model from the same seed (the critic's init stream is private), `loss -
  critic_weighted` equals the off-model's loss BIT FOR BIT, and the label's perturbation
  draw consumes nothing from the run's stream.
* **THE LABEL IS DETACHED.** Every loop, coda, `W_prefix` and span-decoder gradient is
  bit-identical with and without the critic's term — the `tests/test_tul_oracle_z.py`
  contract — while the critic's OWN parameters get gradient and the energy's gradient
  reaches the core.
* **The pair shares its context.** A sabotage that shuffles `ctx` between the two
  candidates is caught.
* **The perturbation is at the requested rms scale**, measured, not asserted.
* **The critic never sees a future token.** Its `forward` is called with the slot state and
  the prelude-entry context and nothing else.
* **`critic_every` skips the LABEL, never the energy.**
* **`critic_replay_groups` substitutes the slots it says it does.**
* **Eval pays nothing**, so the forced-depth sweep reads the ruler's columns.

CPU only, fp32, tiny config — the `tests/test_tul_oracle_z.py` fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_egrad import CriticEnergy
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10


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


def _tul(critic: bool = False, **kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                mux_beta=0.0, spandec=True, spandec_layers=1, spandec_max_tokens=8,
                slot_depth_fixed=3, slot_max_depth=4)
    if critic:
        base.update(grad_pass=True, grad_pass_scale=0.1, grad_pass_norm="rms",
                    grad_pass_energy="critic", pass_residual_lambda=0.01)
    base.update(kw)
    return TULConfig(**base)


def _batch(B: int = 2, n: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::8] = DOT
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _model(seed: int = 99, critic: bool = False, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(critic, **tul_kw)))
    return m.train().float()


def _run(m: MORPHTransformer):
    inp, lab, layout, _ = _batch()
    torch.manual_seed(3)
    return m(inp, labels=lab, slot_layout=layout)


# ── off is nothing ───────────────────────────────────────────────────────────

def test_off_builds_no_critic_and_never_calls_it(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the critic ran on a model that did not ask for it")

    monkeypatch.setattr(MORPHTransformer, "_tul_critic_loss", boom)
    m = _model()
    assert m.tul_egrad is None
    out = _run(m)
    assert "critic" not in out and "critic_weighted" not in out


def test_on_builds_a_critic_and_nothing_else():
    m = _model(critic=True)
    assert isinstance(m.tul_egrad, CriticEnergy)
    names = [n for n, _ in m.named_parameters() if n.startswith("tul_egrad.")]
    assert names, "the critic built no parameter"
    for n in names:
        mod = m.tul_egrad
        for part in n.split(".")[1:-1]:
            mod = getattr(mod, part)
        if isinstance(mod, torch.nn.Linear):
            assert getattr(mod, "_ternary_exclude", False), f"{n} is inside the QAT scope"


# ── additive, RNG-neutral, bit for bit ───────────────────────────────────────

def test_the_critic_shifts_no_shared_weight():
    """Private init stream (`_SEED_CRITIC`), the `TULSlots.W_sent` rule."""
    off, on = _model(), _model(critic=True)
    d_off, d_on = dict(off.named_parameters()), dict(on.named_parameters())
    assert sorted(set(d_off) - set(d_on)) == []
    extra = set(d_on) - set(d_off)
    assert extra and all(n.startswith(("tul_egrad.", "tul_grad_pass.")) for n in extra)
    for n, p in d_off.items():
        assert torch.equal(p, d_on[n]), f"building the critic moved {n}"


def test_the_term_is_positive_and_carries_its_weight():
    m = _model(critic=True, critic_weight=0.25)
    out = _run(m)
    assert float(out["critic"]) > 0.0
    assert abs(float(out["critic_weighted"]) - 0.25 * float(out["critic"])) < 1e-5
    assert float(out["critic_replays"]) == 3.0
    assert float(out["critic_n_traj"]) > 0.0
    assert 0.0 <= float(out["critic_agree"]) <= 1.0


def test_the_label_consumes_nothing_from_the_rng_stream():
    """The perturbation draws `randn_like`. Without the save/restore every later draw in
    the run would shift, and the arm would differ from its ruler by more than the
    mechanism."""
    inp, lab, layout, _ = _batch()
    tails = []
    for critic in (True, False):
        m = _model(critic=critic)
        torch.manual_seed(5)
        m(inp, labels=lab, slot_layout=layout)["loss"].backward()
        tails.append(torch.randn(4))
    assert torch.equal(tails[0], tails[1])


# ── THE LABEL IS DETACHED ────────────────────────────────────────────────────

def _grads(m: MORPHTransformer):
    _run(m)["loss"].backward()
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in m.named_parameters()}


def test_the_label_trains_nothing_but_the_critic():
    """Two models from the same seed: one with `critic_weight` 0-by-skip, one live.

    The comparison that matters is a model whose critic term is REMOVED from the loss
    against one where it is present — every loop, coda, `W_prefix` and decoder gradient
    must be bit-identical, because the label is `no_grad` and the pairwise loss runs on a
    detached `z`.
    """
    class _NoTerm(MORPHTransformer):
        def _tul_critic_loss(self, *a, **k):      # the term, removed; the energy, kept
            return None

    torch.manual_seed(99)
    off = _NoTerm(_tiny(tul=_tul(True))).train().float()
    on = _model(critic=True)
    for n, p in off.named_parameters():
        assert torch.equal(p, dict(on.named_parameters())[n])
    g_off, g_on = _grads(off), _grads(on)
    shared = [n for n in g_off if not n.startswith("tul_egrad.")]
    assert any(n.startswith("core.") for n in shared)
    assert "tul.W_prefix" in shared
    for n in shared:
        a, b = g_off[n], g_on[n]
        assert (a is None) == (b is None), n
        if a is not None:
            assert torch.equal(a, b), (
                f"the critic's LABEL reached {n} — it must be no_grad end to end")
    cr = [n for n in g_on if n.startswith("tul_egrad.")]
    assert cr and any(g_on[n] is not None and float(g_on[n].abs().sum()) > 0 for n in cr), \
        "the critic's own parameters got no gradient — it is not training"


def test_the_replay_builds_no_graph_at_all():
    m = _model(critic=True)
    inp, lab, layout, _ = _batch()
    torch.manual_seed(3)
    x, x0, bg = m._tul_front(inp, layout)
    xn, h, _d, _g, _t, _gr, _mk = m._tul_core(x, x0, bg, layout, input_ids=inp)
    ce, scored = m._critic_replay_ce(
        h.detach(), h.detach(), base=xn.detach(), x0=x0, bigram_emb=bg, input_ids=inp,
        labels=lab, layout=layout, keep=None, coda_kw=None, ret_reset_mask=None, groups=1)
    assert ce.grad_fn is None and not ce.requires_grad
    assert bool(scored.any())


def test_the_energys_gradient_still_reaches_the_core(monkeypatch):
    """The mechanism: the critic reaches the loop ONLY through `W_g`, and it does."""
    off, on = _model(critic=True), _model(critic=True)
    with torch.no_grad():                     # `W_g` is zero-init: move it off zero
        on.tul_grad_pass.W_g.normal_(0.0, 0.02)
    g_off, g_on = _grads(off), _grads(on)
    core = [n for n in g_off if n.startswith("core.")]
    assert any(g_off[n] is not None and g_on[n] is not None
               and not torch.equal(g_off[n], g_on[n]) for n in core), \
        "moving W_g off zero changed no core gradient — the feature is inert"


# ── the pair shares its context ──────────────────────────────────────────────

def test_a_shuffled_context_between_the_candidates_is_caught():
    """The sabotage, run here rather than described: if the two candidates of a pair were
    scored under DIFFERENT contexts the comparison would not be within-context at all, and
    the loss would change. `CriticEnergy.pairwise` takes ONE `ctx` for both, so the only
    way to plant this is to score them separately — which this test does, and the two
    numbers must differ."""
    c = CriticEnergy(16, 0)
    torch.manual_seed(0)
    z_a, z_b = torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    ctx = torch.randn(2, 5, 16)
    ce_a, ce_b = torch.rand(2, 5), torch.rand(2, 5)
    ok = torch.ones(2, 5, dtype=torch.bool)
    same, _n, _a = c.pairwise(z_a, z_b, ctx, ce_a, ce_b, ok)
    # the sabotage: b scored under a rolled context
    s_a = c.score(z_a, ctx).float()
    s_b = c.score(z_b, torch.roll(ctx, 1, dims=1)).float()
    gap = (ce_b - ce_a).float()
    w = ok.to(gap.dtype) * gap.abs()
    bad = (torch.nn.functional.binary_cross_entropy_with_logits(
        s_a - s_b, (gap > 0).to(gap.dtype), reduction="none") * w).sum() / w.sum()
    assert not torch.allclose(same, bad), (
        "scoring the two candidates under different contexts gave the SAME loss — the "
        "pairwise term is not reading its context at all")


def test_the_pairwise_loss_weights_by_the_measured_gap():
    c = CriticEnergy(16, 0)
    torch.manual_seed(1)
    z_a, z_b, ctx = torch.randn(2, 5, 16), torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    ok = torch.ones(2, 5, dtype=torch.bool)
    ce = torch.rand(2, 5)
    tie, _n, agree = c.pairwise(z_a, z_b, ctx, ce, ce.clone(), ok)
    assert float(tie.detach()) == 0.0, (
        "a pair with a zero CE gap must contribute nothing")
    assert float(agree.detach()) == 0.0


# ── the perturbation ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("eps", [0.05, 0.2])
def test_the_perturbation_is_at_the_requested_rms_scale(eps, monkeypatch):
    """Measured off the tensors the replay actually receives, not read off the source."""
    seen: list[torch.Tensor] = []
    real = MORPHTransformer._critic_replay_ce

    def spy(self, cand, exit_h, **k):
        seen.append(cand.detach().clone())
        return real(self, cand, exit_h, **k)

    monkeypatch.setattr(MORPHTransformer, "_critic_replay_ce", spy)
    m = _model(critic=True, critic_eps=eps)
    _run(m)
    assert len(seen) == 3, f"expected 3 candidate replays, got {len(seen)}"
    h_b, h_p = seen[1], seen[2]
    d = (h_p - h_b).float().flatten(2).pow(2).mean(-1).sqrt()
    r = h_b.float().flatten(2).pow(2).mean(-1).sqrt()
    live = r > 0
    assert bool(live.any())
    got = (d[live] / r[live]).mean().item()
    assert abs(got - eps) < 0.02 * max(eps, 0.05), (
        f"the perturbation's rms ratio is {got:.4f}, asked for {eps}")


def test_the_two_pairs_share_h_t(monkeypatch):
    """`(h_{t-1}, h_t)` and `(h_t, h_t + noise)` are both anchored at the SAME state, so
    the critic is fitted where its gradient is read. Using the EXIT state for the
    perturbation would save one replay and fit the critic at a point its gradient is not
    read at; the alternative is recorded in the arm's note."""
    seen: list[torch.Tensor] = []
    real = MORPHTransformer._critic_replay_ce

    def spy(self, cand, exit_h, **k):
        seen.append(cand.detach().clone())
        return real(self, cand, exit_h, **k)

    monkeypatch.setattr(MORPHTransformer, "_critic_replay_ce", spy)
    _run(_model(critic=True, critic_eps=0.1))
    h_a, h_b, h_p = seen
    assert not torch.equal(h_a, h_b), "the trajectory pair is two copies of one state"
    assert not torch.equal(h_b, h_p), "the perturbation moved nothing"
    # h_p is h_b plus a step; the pairs' shared anchor is h_b, not h_a.
    assert float((h_p - h_b).abs().sum()) < float((h_p - h_a).abs().sum())


# ── causality: the critic never sees a future token ──────────────────────────

def test_the_critic_forward_receives_only_the_state_and_its_context(monkeypatch):
    """Two tensors, both causal for the slot, and no token ids at all.

    The conditioning feature has to be computable by a generator at inference, so a label
    token reaching `CriticEnergy.score` would make the whole arm undeployable.
    """
    m = _model(critic=True)
    seen: list[tuple] = []
    real = CriticEnergy.score

    def spy(self, z, ctx):
        seen.append((tuple(z.shape), tuple(ctx.shape)))
        return real(self, z, ctx)

    monkeypatch.setattr(CriticEnergy, "score", spy)
    out = _run(m)
    assert seen, "the critic was never scored"
    inp, _lab, layout, _ = _batch()
    S = int(layout.slot_index.shape[1])
    for zs, cs in seen:
        assert zs == cs, f"z {zs} and ctx {cs} are not the same shape"
        assert zs[:2] == (inp.shape[0], S), (
            f"the critic was handed a {zs} tensor — that is not [B, max_slots, C], so it "
            f"is not reading slot states")
    assert float(out["critic"]) > 0.0


# ── the knobs ────────────────────────────────────────────────────────────────

def test_critic_every_skips_the_label_and_not_the_energy():
    m = _model(critic=True, critic_every=3)
    inp, lab, layout, _ = _batch()
    got = []
    for _ in range(4):
        torch.manual_seed(3)
        out = m(inp, labels=lab, slot_layout=layout)
        got.append("critic" in out)
        # the energy is read EVERY step whatever `critic_every` says
        assert m._loop_gradpass, "the gradient feature did not run"
    assert got == [True, False, False, True], got


def test_replay_groups_substitute_only_their_own_slots(monkeypatch):
    """G > 1 leaves every other slot at its EXIT state, so the nearest confounder sits G
    spans back. Measured on the tensors `prefix_project` receives."""
    seen: list[torch.Tensor] = []
    real = MORPHTransformer._back_region

    def spy(self, x, *a, **k):
        seen.append(x.detach().clone())
        return real(self, x, *a, **k)

    m = _model(critic=True)
    inp, lab, layout, _ = _batch()
    torch.manual_seed(3)
    x, x0, bg = m._tul_front(inp, layout)
    xn, h, _d, _g, _t, _gr, _mk = m._tul_core(x, x0, bg, layout, input_ids=inp)
    cand = torch.randn_like(h)
    monkeypatch.setattr(MORPHTransformer, "_back_region", spy)
    m._critic_replay_ce(cand, h.detach(), base=xn.detach(), x0=x0, bigram_emb=bg,
                        input_ids=inp, labels=lab, layout=layout, keep=None,
                        coda_kw=None, ret_reset_mask=None, groups=3)
    assert len(seen) == 3, f"groups=3 must be 3 replays, got {len(seen)}"
    # In replay g the substituted cells are the ones of slots s = g (mod 3); a cell of a
    # slot NOT in the group must carry the exit state's write, which is what the g = 0
    # and g = 1 replays disagree about.
    S = int(layout.slot_index.shape[1])
    b = 0
    for g in range(3):
        for s in range(S):
            if not bool(layout.slot_valid[b, s]):
                continue
            p = int(layout.slot_index[b, s])
            same = torch.equal(seen[g][b, p], seen[(g + 1) % 3][b, p])
            if s % 3 == g or s % 3 == (g + 1) % 3:
                assert not same, f"slot {s} cell is identical across replays {g} and {g+1}"
            else:
                assert same, f"slot {s} moved in a replay that does not own it"


def test_eval_pays_nothing(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the critic's label ran on an EVAL forward")

    monkeypatch.setattr(MORPHTransformer, "_tul_critic_loss", boom)
    m = _model(critic=True).eval()
    inp, lab, layout, _ = _batch()
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    assert "critic" not in out


# ── the probe that reads this arm ────────────────────────────────────────────

def test_the_direction_probe_reads_the_critics_own_gradient():
    """`lab/divergence/critic_direction_probe.py`'s two direction builders, on the CPU
    fixture. The critic's direction must be unit-RMS and must MOVE with the critic's
    weights; a random direction must be unit-RMS and independent of them."""
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "lab/divergence/critic_direction_probe.py"
    spec = importlib.util.spec_from_file_location("_cdp", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    m = _model(critic=True).eval()
    inp, _lab, layout, _ = _batch()
    with torch.no_grad():
        x, x0, bg = m._tul_front(inp, layout)
        _xn, h, *_ = m._tul_core(x, x0, bg, layout, input_ids=inp)
    ctx = m._tul_egrad_ctx
    assert ctx is not None
    live = layout.slot_valid
    d = mod.critic_direction(m, h, ctx, live)
    r = d.float().flatten(2).pow(2).mean(-1).sqrt()
    # A PAD slot reads a zero feature by construction (the energy's mask excludes it), so
    # its direction is the zero vector and is not unit RMS. That is the contract, not a
    # defect: pinning it here stops a future change from handing a pad a direction.
    assert torch.allclose(r[live], torch.ones_like(r[live]), atol=1e-3), \
        "the direction is not unit RMS on the valid slots"
    assert float(r[~live].abs().max()) == 0.0, "a pad slot was handed a direction"
    with torch.no_grad():
        m.tul_egrad.fc2.weight.mul_(-1.0)
    d2 = mod.critic_direction(m, h, ctx, live)
    assert not torch.allclose(d, d2), "the direction ignores the critic's weights"
    gen = torch.Generator().manual_seed(0)
    rd = mod.random_direction(h, gen)
    rr = rd.float().flatten(2).pow(2).mean(-1).sqrt()
    assert torch.allclose(rr, torch.ones_like(rr), atol=1e-3)
    step = mod.step_along(h, d, 0.1)
    ratio = ((step - h).float().flatten(2).pow(2).mean(-1).sqrt()
             / h.float().flatten(2).pow(2).mean(-1).sqrt().clamp(min=1e-9))
    assert abs(float(ratio[live].mean()) - 0.1) < 1e-3


# ── the refusals ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw,match", [
    (dict(critic=True, critic_weight=0.0), "critic_weight > 0"),
    (dict(critic=True, critic_every=0), "critic_every must be >= 1"),
    (dict(critic=True, critic_replay_groups=0), "critic_replay_groups must be >= 1"),
    (dict(critic=True, critic_eps=0.0), "critic_eps must be > 0"),
    (dict(critic=True, oracle_z=True), "oracle_z"),
    (dict(critic=True, coda_sees_slots=False), "FULL-AXIS coda"),
    (dict(critic=True, detach_z=True), "detach_z"),
    (dict(critic_weight=2.0), "silently ignored"),
    (dict(grad_pass=True, grad_pass_energy="coda_exact", mux_beta=0.5), "REFUSED"),
    (dict(grad_pass=True, grad_pass_energy="nope", mux_beta=0.5), "grad_pass_energy"),
])
def test_critic_config_refusals(kw, match):
    critic = kw.pop("critic", False)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _tul(critic, **kw)


def test_a_coreless_model_refuses_the_critic():
    with pytest.raises(ValueError, match="needs a core loop"):
        MORPHTransformer(_tiny(n_core=0, tul=_tul(True)))


def test_coda_exact_names_all_four_missing_pieces():
    """The refusal is the documentation. If it stops naming what is missing, someone will
    re-derive it — the `reinject_seed_every_pass` precedent."""
    with pytest.raises(NotImplementedError) as e:
        _tul(grad_pass=True, grad_pass_energy="coda_exact", mux_beta=0.5)
    msg = str(e.value)
    for needle in ("CONTEXT IS NOT THERE YET", "OPPOSITE GRADIENT RULES",
                   "MOVES AN RNG DRAW", "THE COST", "critic"):
        assert needle in msg, f"the coda_exact refusal no longer names {needle!r}"
