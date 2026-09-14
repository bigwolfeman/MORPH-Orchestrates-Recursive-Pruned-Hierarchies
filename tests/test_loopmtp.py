"""LoopMTP on the plain looped model (arXiv 2608.03624, Shomali et al. 2026).

One test per contract of the arms `norm-match-20k-d3-loopmtp` / `norm-match-20k-d6-loopmtp`
(`lab/experiments/planned/2026-09-14-arc-loopmtp-token-loop.md`):

* OFF (`core_readout: last` and `loopmtp_weight: 0`) is the tree as it was — proved against
  the PRE-CHANGE SOURCE, exported from the base commit and run in a second process on the
  same CPU fixture, not against a hash somebody typed;
* ON draws no extra RNG, so the arm and its control share every base weight;
* the alignment term is positive, enters the loss, and is recoverable from `ce_main`;
* NO gradient reaches the tied embedding table through the alignment path, while the LM
  head path still trains it;
* iteration `t`'s target is the token `t` ahead — moving the token at `i + t - 1` in the
  labels moves iteration `t`'s term at position `i` and no other iteration's term there;
* the row's last `t - 1` positions carry no target and are masked out of the mean;
* the aggregator's gate sums to one per position, and moving the gate moves what the coda
  reads;
* a Poisson depth, a truncated BPTT window, a TUL model, SCSE and a coreless model are
  REFUSED at build, not silently accepted;
* the gate and the alignment projection are never ternarised;
* every new config key is threaded by `build_morph_config`.

CPU only, tiny config, no tokenizer — the `tests/test_parcae_core.py` fixtures.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F

from morph.model.transformer import MORPHConfig, MORPHTransformer, _LoopMTPGate
from morph.model.tul import TULConfig

V = 64
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The commit this arm was built ON TOP OF. `test_off_is_bit_identical_to_the_base_commit`
# exports `morph/` from it and runs the fixture below against that source in a second
# process, so "off is bit-identical" is a measurement, not an assertion about a hash.
_BASE_COMMIT = "fd2daae"


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=3, max_depth=3, bptt_depth=3,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _model(seed=1234, **cfg_kw) -> MORPHTransformer:
    # A LoopMTP model needs the real fixed depth (mean == max is a clamped Poisson draw).
    if cfg_kw.get("core_readout") == "gated" or float(cfg_kw.get("loopmtp_weight", 0.0)) > 0:
        cfg_kw.setdefault("depth_fixed", True)
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(**cfg_kw))


def _xy(B=2, S=32):
    x = torch.randint(0, V, (B, S), generator=torch.Generator().manual_seed(7))
    y = torch.randint(0, V, (B, S), generator=torch.Generator().manual_seed(8))
    return x, y


def _step(model: MORPHTransformer, seed=99):
    """One labelled training step. Returns the forward's output dict (loss retained)."""
    x, y = _xy()
    model.train()
    torch.manual_seed(seed)
    out = model(x, labels=y)
    out["loss"].backward()
    return out


def _iterates(model: MORPHTransformer, x, labels=None):
    """The per-iteration carriers ``x^(1) .. x^(T)`` the loop produced, in batch order."""
    model._loopmtp_capture = True
    try:
        model.eval()
        with torch.no_grad():
            model(x, labels=labels)
        return list(model._loopmtp_iterates), model._loopmtp_gate_mass
    finally:
        model._loopmtp_capture = False


# ── OFF is the tree as it was ────────────────────────────────────────────────

# The fixture, as a standalone script, so the SAME text runs against the pre-change source
# (PYTHONPATH pointing at a `git archive` of the base commit) and against this tree. It uses
# no config field that LoopMTP added, so the base tree accepts it verbatim.
_FIXTURE = r'''
import sys, torch
torch.set_num_threads(1)
from morph.model.transformer import MORPHConfig, MORPHTransformer
V = 64
cfg = MORPHConfig(
    d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
    n_prelude=1, n_core=2, n_coda=1, mean_depth=3, max_depth=3, bptt_depth=3,
    channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
    hca_compress_ratio=8, top_k=8, window_size=16,
    retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
    dropout=0.0)
torch.manual_seed(1234)
m = MORPHTransformer(cfg)
x = torch.randint(0, V, (2, 32), generator=torch.Generator().manual_seed(7))
y = torch.randint(0, V, (2, 32), generator=torch.Generator().manual_seed(8))
m.train()
torch.manual_seed(99)
out = m(x, labels=y)
out["loss"].backward()
gs = sum(float(p.grad.double().sum()) for _, p in sorted(m.named_parameters())
         if p.grad is not None)
ps = sum(float(p.detach().double().sum()) for _, p in sorted(m.named_parameters()))
m.eval()
torch.manual_seed(5)
with torch.no_grad():
    lg = m(x)["logits"]
print("LOSS=%.17g" % float(out["loss"]))
print("GRADSUM=%.17g" % gs)
print("PARAMSUM=%.17g" % ps)
print("LOGITSUM=%.17g" % float(lg.double().sum()))
'''


def _run_fixture(pythonpath: str) -> dict[str, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(_FIXTURE)
        script = fh.name
    try:
        env = dict(os.environ, PYTHONPATH=pythonpath, OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        env.pop("MORPH_DIAG_CORECOS", None)
        p = subprocess.run([sys.executable, script], capture_output=True, text=True,
                           env=env, cwd=tempfile.gettempdir())
        assert p.returncode == 0, f"fixture failed under {pythonpath}:\n{p.stderr[-4000:]}"
        return dict(line.split("=", 1) for line in p.stdout.strip().splitlines()
                    if line.startswith(("LOSS=", "GRADSUM=", "PARAMSUM=", "LOGITSUM=")))
    finally:
        os.unlink(script)


def test_off_is_bit_identical_to_the_base_commit():
    """The knobs-off model is the ladder rung to the last printed digit.

    Not a pinned hash: the pre-change `morph/` tree is exported from `_BASE_COMMIT` into a
    temp directory and the SAME fixture text runs against it in a second process on this
    same CPU. If a later edit moves the plain forward, this fails with both numbers.
    """
    have_git = subprocess.run(["git", "cat-file", "-e", f"{_BASE_COMMIT}^{{commit}}"],
                              cwd=ROOT, capture_output=True).returncode == 0
    if not have_git:
        pytest.skip(f"base commit {_BASE_COMMIT} not reachable from this checkout")
    base = tempfile.mkdtemp(prefix="loopmtp_base_")
    try:
        tar = subprocess.run(["git", "archive", _BASE_COMMIT, "morph"], cwd=ROOT,
                             capture_output=True)
        assert tar.returncode == 0, tar.stderr.decode()[-2000:]
        subprocess.run(["tar", "-x", "-C", base], input=tar.stdout, check=True)
        old = _run_fixture(base)
        new = _run_fixture(ROOT)
        print(f"\n[loopmtp] base {_BASE_COMMIT}: {old}")
        print(f"[loopmtp] worktree      : {new}")
        assert old == new, (
            "the knobs-off forward moved against the base commit:\n"
            f"  base({_BASE_COMMIT}) {old}\n  worktree        {new}")
        assert set(old) == {"LOSS", "GRADSUM", "PARAMSUM", "LOGITSUM"}
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_knobs_off_build_nothing():
    m = _model()
    assert m._loopmtp_states is False
    assert m.loopmtp_gate is None and m.loopmtp_proj is None
    assert not [n for n, _ in m.named_parameters() if "loopmtp" in n]


def test_knobs_off_explicitly_is_the_same_model():
    """Writing the defaults out is inert, bit for bit."""
    losses, params = [], []
    for kw in ({}, {"core_readout": "last", "loopmtp_weight": 0.0}):
        m = _model(**kw)
        losses.append(_step(m)["loss"].detach().clone())
        params.append(torch.cat([p.detach().flatten()
                                 for _, p in sorted(m.named_parameters())]))
    assert torch.equal(losses[0], losses[1])
    assert torch.equal(params[0], params[1]), "the knob perturbed the init RNG"


def test_knobs_off_and_on_share_every_base_weight():
    """The LoopMTP build restores the RNG stream, so the arm and its control start equal."""
    off = _model()
    on = _model(core_readout="gated", loopmtp_weight=0.05, loopmtp_ponder_weight=0.05)
    base = dict(off.named_parameters())
    for n, p in on.named_parameters():
        if "loopmtp" in n:
            continue
        assert n in base, f"{n} is new but not a loopmtp parameter"
        assert torch.equal(p.detach(), base[n].detach()), f"{n} moved"
    assert set(dict(on.named_parameters())) - set(base) == {
        "loopmtp_gate.proj.weight", "loopmtp_gate.beta",
        "loopmtp_proj.norm.weight", "loopmtp_proj.proj.weight"}


def test_align_only_leaves_the_forward_untouched():
    """`loopmtp_weight > 0` with the 'last' read-out is a LOSS term and nothing else."""
    off = _model()
    on = _model(loopmtp_weight=0.05)
    x, y = _xy()
    for m in (off, on):
        m.eval()
    with torch.no_grad():
        torch.manual_seed(5)
        a = off(x)["logits"]
        torch.manual_seed(5)
        b = on(x)["logits"]
    assert torch.equal(a, b), "the alignment term changed the forward"
    o_off, o_on = _step(off), _step(on)
    assert torch.equal(o_off["loss"].detach(), o_on["ce_main"].detach()), (
        "ce_main is not the untouched next-token CE")


# ── the alignment term ───────────────────────────────────────────────────────

def test_align_term_is_positive_and_enters_the_loss():
    m = _model(loopmtp_weight=0.05)
    out = _step(m)
    al = float(out["loopmtp_align"])
    w = float(out["loopmtp_weighted"].detach())
    assert al > 0.0, f"1 - cos must be positive at init, got {al}"
    assert al <= 2.0
    assert abs(w - 0.05 * al) < 1e-6
    assert torch.allclose(out["loss"].detach(),
                          out["ce_main"].detach() + out["loopmtp_weighted"].detach(),
                          atol=0, rtol=1e-6)
    # and it reaches the core, which is the whole point
    core_grads = [p.grad for _, p in m.core.named_parameters() if p.grad is not None]
    assert core_grads and any(g.abs().sum() > 0 for g in core_grads)


def test_align_term_is_training_only():
    """Eval CE is the plain next-token number every ladder arm is compared on."""
    m = _model(loopmtp_weight=0.05)
    x, y = _xy()
    m.eval()
    with torch.no_grad():
        out = m(x, labels=y)
    assert "loopmtp_align" not in out and "loopmtp_weighted" not in out


def test_no_gradient_reaches_the_embedding_through_the_align_target():
    """`sg[E]` (Eq 13) AND the house rule: an auxiliary head must not train the tied table.

    The claim is about the TARGET side. The alignment term legitimately backpropagates into
    the embeddings through the loop's own states (position i's state was built from token
    embeddings), and that path is the model training normally. What must not exist is a path
    through ``E_{u_{i+t}}`` — the lookup the term compares against. So the states are
    detached and the term is rebuilt: whatever gradient survives can only have come through
    the target.

    Two halves, and both matter: the target side must contribute NOTHING, and the ordinary
    LM-head path must still train the table (otherwise a dead fixture passes trivially).
    """
    m = _model(loopmtp_weight=0.05)
    x, y = _xy()
    states, _ = _iterates(m, x)                 # already detached
    m.train()
    emb = [p for n, p in m.named_parameters() if n.startswith("embed.")]
    assert emb
    al = m._loopmtp_align([s.clone() for s in states], y)
    assert float(al) > 0.0
    g_tgt = torch.autograd.grad(al, emb, retain_graph=True, allow_unused=True)
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in g_tgt), (
        "the alignment target trains the tied embedding table — the detach on "
        "lm_weight() is gone (sg[E] in Eq 13)")
    # fixture sensitivity: the SAME lookup without the detach does reach them.
    w_live = m.embed.lm_weight()
    tgt = F.embedding(y.clamp(min=0), w_live)
    g_live = torch.autograd.grad(tgt.pow(2).sum(), emb, retain_graph=True,
                                 allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in g_live), (
        "fixture is dead: an undetached lm_weight() lookup reaches no embedding either")
    # and the real next-token path still trains them
    torch.manual_seed(99)
    out = m(x, labels=y)
    g_ce = torch.autograd.grad(out["ce_main"], emb, retain_graph=True, allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in g_ce), (
        "fixture is dead: the LM head path does not train the embeddings either")


def test_iteration_t_targets_the_token_t_ahead():
    """Move the label at column `j`; only iteration `t` at position `j - t + 1` moves.

    This is the off-by-one guard. `labels[i]` is `u_{i+1}`, so Eq 13's `u_{i+t}` is
    `labels[i + t - 1]`: iteration 3 at position `i` reads `labels[i + 2]`.
    """
    m = _model(loopmtp_weight=0.05, loopmtp_free_first=False)
    x, y = _xy()
    states, _ = _iterates(m, x)
    assert len(states) == 3
    j = 17
    y2 = y.clone()
    y2[:, j] = (y[:, j] + 1) % V
    a = m._loopmtp_align_maps(states, y)
    b = m._loopmtp_align_maps(states, y2)
    assert [t for t, _, _ in a] == [1, 2, 3]
    for (t, ma, _), (_, mb, _) in zip(a, b):
        moved = (ma - mb).abs().sum(dim=0) > 1e-6          # [S] over the batch
        want = j - (t - 1)
        assert bool(moved[want]), (
            f"iteration {t} did not read labels[{j}] at position {want}")
        assert int(moved.sum()) == 1, (
            f"iteration {t} moved at {moved.nonzero().flatten().tolist()}, "
            f"expected only position {want}")
    # the decisive per-iteration statement: at ONE position, only the iteration whose
    # horizon lands on `j` moves.
    i = j - 2                                              # iteration 3's position
    per_iter = [(t, float((ma[:, i] - mb[:, i]).abs().sum()))
                for (t, ma, _), (_, mb, _) in zip(a, b)]
    print(f"\n[loopmtp] label moved at {j}; per-iteration delta at position {i}: {per_iter}")
    assert per_iter[2][1] > 1e-6
    assert per_iter[0][1] == 0.0 and per_iter[1][1] == 0.0


def test_free_first_drops_exactly_iteration_one():
    m = _model(loopmtp_weight=0.05)
    x, y = _xy()
    states, _ = _iterates(m, x)
    assert [t for t, _, _ in m._loopmtp_align_maps(states, y)] == [2, 3]
    m.cfg.loopmtp_free_first = False
    assert [t for t, _, _ in m._loopmtp_align_maps(states, y)] == [1, 2, 3]


def test_row_end_is_masked_out_of_the_mean():
    """Iteration `t` has no target at the last `t - 1` positions, and they are excluded.

    Fixture sensitivity: the tail positions are given a target the mask must ignore, and
    the term must not move.
    """
    m = _model(loopmtp_weight=0.05, loopmtp_free_first=False)
    x, y = _xy()
    S = y.shape[1]
    states, _ = _iterates(m, x)
    maps = m._loopmtp_align_maps(states, y)
    for t, _, valid in maps:
        assert int(valid.sum()) == y.shape[0] * (S - (t - 1)), (
            f"iteration {t} scored {int(valid.sum())} positions, expected "
            f"{y.shape[0] * (S - (t - 1))} (the paper's S - t normaliser)")
        if t > 1:
            assert not bool(valid[:, S - (t - 1):].any())
    before = float(m._loopmtp_align(states, y))
    # An already-padded label must also be dropped, not scored against row 0 of E.
    y_pad = y.clone()
    y_pad[:, -1] = -100
    after = float(m._loopmtp_align(states, y_pad))
    assert after != before, "masking a label did not change the mean — the mask is inert"
    maps_pad = m._loopmtp_align_maps(states, y_pad)
    assert int(maps_pad[0][2].sum()) == int(maps[0][2].sum()) - y.shape[0]


# ── the aggregator ───────────────────────────────────────────────────────────

def test_gate_sums_to_one_per_position():
    g = _LoopMTPGate(8, 3)
    torch.manual_seed(0)
    states = [torch.randn(2, 5, 4, 8) for _ in range(3)]
    with torch.no_grad():
        g.proj.weight.normal_(std=0.3)
        g.beta.copy_(torch.tensor([-0.4, 0.1, 0.7]))
        raw = g.raw_gates(states)
        tot = raw[0] + raw[1] + raw[2] + g.eps
        norm = torch.stack([r / tot for r in raw], dim=0)
        z, mass = g(states)
    assert torch.allclose(norm.sum(dim=0), torch.ones_like(norm[0]), atol=1e-5), (
        "the gate does not normalise over the ITERATION axis")
    assert torch.allclose(mass.sum(dim=0), torch.ones(mass.shape[1]), atol=1e-5)
    ref = sum(n * s for n, s in zip(norm, states))
    assert torch.allclose(z, ref, atol=1e-6)
    assert mass.shape == (3, 2 * 5)


def test_gate_is_uniform_at_init_and_content_conditional_after():
    g = _LoopMTPGate(8, 4)
    torch.manual_seed(0)
    states = [torch.randn(2, 5, 8) for _ in range(4)]
    _z, mass = g(states)
    assert torch.allclose(mass, torch.full_like(mass, 0.25), atol=1e-4), (
        "zero-init W_g and beta must give the uniform 'All (uniform)' aggregate")
    with torch.no_grad():
        g.proj.weight.normal_(std=0.5)
    _z2, mass2 = g(states)
    assert (mass2 - 0.25).abs().max() > 1e-3, "W_g does not reach the gate"


def test_aggregate_at_the_uniform_gate_is_the_mean_of_every_iterate():
    """Eq 11 sums over t = 1..T. At the zero init the gate is uniform, so the aggregate is
    exactly the arithmetic mean of the T iterates — and is NOT any single one of them."""
    g = _LoopMTPGate(8, 3)
    torch.manual_seed(3)
    states = [torch.randn(2, 5, 8) for _ in range(3)]
    z, _mass = g(states)
    ref = (states[0] + states[1] + states[2]) / 3.0
    assert torch.allclose(z, ref, atol=1e-5), (
        "the aggregate is not the sum over ALL iterates")
    for k, st in enumerate(states):
        assert not torch.allclose(z, st / 3.0, atol=1e-5), (
            f"the aggregate is iterate {k + 1} alone")
        assert not torch.allclose(z, st, atol=1e-5)


def test_gated_readout_moves_what_the_coda_reads():
    """The coda input is the aggregate, not the last iterate, and moving beta moves it."""
    last = _model(core_readout="last")
    gated = _model(core_readout="gated")
    x, _y = _xy()
    for m in (last, gated):
        m.eval()
    with torch.no_grad():
        torch.manual_seed(5)
        a = last(x)["logits"]
        torch.manual_seed(5)
        b = gated(x)["logits"]
    assert not torch.allclose(a, b), "the gated read-out is the last iterate"
    with torch.no_grad():
        gated.loopmtp_gate.beta.copy_(torch.tensor([-8.0, -8.0, 8.0]))
        torch.manual_seed(5)
        c = gated(x)["logits"]
    assert not torch.allclose(b, c), "moving the gate did not move the coda input"
    # beta driven hard onto the last iterate recovers the 'last' read-out
    assert (c - a).abs().max() < (b - a).abs().max(), (
        "a gate pinned to the final iterate did not approach the 'last' read-out")


def test_ponder_is_zero_at_a_uniform_gate_and_positive_when_collapsed():
    m = _model(core_readout="gated", loopmtp_ponder_weight=0.05)
    out = _step(m)
    assert float(out["loopmtp_ponder"]) == pytest.approx(0.0, abs=1e-5)
    with torch.no_grad():
        m.loopmtp_gate.beta.copy_(torch.tensor([-8.0, -8.0, 8.0]))
    m.zero_grad(set_to_none=True)
    out2 = _step(m)
    assert float(out2["loopmtp_ponder"]) > 0.1, (
        "a collapsed gate must cost ponder; the KL is inert")
    assert float(out2["loopmtp_ponder_weighted"]) == pytest.approx(
        0.05 * float(out2["loopmtp_ponder"]), rel=1e-5)


def test_ponder_is_reported_even_at_weight_zero():
    """A gated run without the regulariser still gets the gate-collapse instrument."""
    m = _model(core_readout="gated")
    out = _step(m)
    assert "loopmtp_ponder" in out
    assert "loopmtp_ponder_weighted" not in out


def test_forced_depth_reads_a_gated_model_over_the_available_iterations():
    """The read-out convention `core_depth_sweep.py` relies on.

    At a forced depth `d`, the gate normalises over the `d` iterates that exist; at `d = 0`
    there is no iterate and the aggregate is the entry carrier; above the trained `T`,
    iteration `t > T` reuses `beta_T`.
    """
    m = _model(core_readout="gated")
    x, _y = _xy()
    m.eval()
    for d in (0, 1, 2, 3, 6):
        m.cfg.mean_depth = d
        with torch.no_grad():
            lg = m(x)["logits"]
        assert torch.isfinite(lg).all(), f"forced depth {d} produced non-finite logits"
        if d > 0:
            states, mass = _iterates(m, x)
            assert len(states) == d
            assert mass.shape[0] == d
            assert torch.allclose(mass.sum(dim=0), torch.ones(mass.shape[1]), atol=1e-4)
    m.cfg.mean_depth = 3


# ── refusals ─────────────────────────────────────────────────────────────────

def test_poisson_depth_is_refused():
    with pytest.raises(ValueError, match="FIXED loop depth"):
        _model(core_readout="gated", mean_depth=6, max_depth=8, bptt_depth=8, depth_fixed=False)
    with pytest.raises(ValueError, match="FIXED loop depth"):
        _model(loopmtp_weight=0.05, mean_depth=6, max_depth=8, bptt_depth=8, depth_fixed=False)
    # mean == max is a Poisson draw clamped at max, NOT a constant T (2026-09-14 smoke).
    with pytest.raises(ValueError, match="FIXED loop depth"):
        _model(loopmtp_weight=0.05, mean_depth=3, max_depth=3, bptt_depth=3, depth_fixed=False)
    with pytest.raises(ValueError, match="mean_depth == max_depth"):
        _model(mean_depth=6, max_depth=8, bptt_depth=8, depth_fixed=True)


def test_depth_fixed_runs_every_row_at_max_depth():
    """`depth_fixed` is what LoopMTP's constant T stands on; mean == max is not it."""
    m = _model(depth_fixed=True)
    m.train()
    assert m._sample_depths(64, torch.device("cpu")).tolist() == [3] * 64
    m2 = _model(depth_fixed=False, mean_depth=3, max_depth=3)
    torch.manual_seed(0)
    d2 = m2._sample_depths(4096, torch.device("cpu"))
    assert d2.max().item() == 3 and d2.min().item() < 3, "mean == max is a clamped draw"


def test_truncated_bptt_is_refused():
    with pytest.raises(ValueError, match="full BPTT"):
        _model(loopmtp_weight=0.05, mean_depth=3, max_depth=3, bptt_depth=2)


def test_coreless_model_is_refused():
    with pytest.raises(ValueError, match="needs a core loop"):
        _model(core_readout="gated", n_core=0)


def test_tul_model_is_refused():
    with pytest.raises(ValueError, match="PLAIN looped model"):
        _model(core_readout="gated",
               tul=TULConfig(prefix_k=2, slot_id=4, tokens_through_core=False))


def test_scse_is_refused():
    with pytest.raises(ValueError, match="not defined under SCSE"):
        _model(loopmtp_weight=0.05, scse_enabled=True)


def test_ponder_without_a_gate_is_refused():
    with pytest.raises(ValueError, match="core_readout='gated'"):
        _model(loopmtp_ponder_weight=0.05)


def test_unknown_enum_values_are_refused():
    with pytest.raises(ValueError, match="core_readout"):
        _model(core_readout="gate")
    with pytest.raises(ValueError, match="loopmtp_proj"):
        _model(loopmtp_weight=0.05, loopmtp_proj="mlp")
    with pytest.raises(ValueError, match="loopmtp_weight"):
        _model(loopmtp_weight=-1.0)


def test_bag_labels_are_refused():
    m = _model(loopmtp_weight=0.05)
    x, y = _xy()
    m.train()
    with pytest.raises(RuntimeError, match="3-D TST bag labels"):
        m(x, labels=y.unsqueeze(-1).expand(-1, -1, 2))


# ── quantisation and config plumbing ─────────────────────────────────────────

def test_ternary_qat_skips_the_gate_and_the_align_projection():
    from morph.model.ternary_qat import apply_ternary_qat
    m = _model(core_readout="gated", loopmtp_weight=0.05)
    man = apply_ternary_qat(m, scope="full", scale_mode="norm_match")
    hit = [n for n in man["module_names"] if "loopmtp" in n]
    assert not hit, f"LoopMTP control paths were ternarised: {hit}"
    # fixture sensitivity: the walk under this scope DOES reach ordinary core linears
    assert any(n.startswith("core.") for n in man["module_names"]), man["module_names"][:8]
    # and it sees the LoopMTP modules at all — they are skipped, not invisible
    assert [n for n, _ in m.named_modules() if "loopmtp" in n]


def test_build_morph_config_threads_every_key():
    from hydra import compose, initialize_config_dir
    from morph.training.train import build_morph_config
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.join(ROOT, "morph", "configs")):
        cfg = compose(config_name="notul_norm_match_20k_d3_loopmtp", overrides=[])
    mc = build_morph_config(cfg, tul=None)
    assert mc.core_readout == "gated"
    assert mc.loopmtp_weight == 0.01
    assert mc.loopmtp_free_first is True
    assert mc.loopmtp_proj == "linear"
    assert mc.loopmtp_ponder_weight == 0.05
    assert (mc.mean_depth, mc.max_depth, mc.bptt_depth) == (3, 3, 3)
    assert mc.depth_fixed is True
    assert mc.tul is None


def test_every_new_config_key_is_documented_and_threaded():
    """No silent knob: each new field is named in the dataclass comment AND in train.py."""
    import inspect
    from morph.model import transformer as tmod
    from morph.training import train as trmod
    src_cfg = inspect.getsource(tmod.MORPHConfig)
    src_build = inspect.getsource(trmod.build_morph_config)
    keys = ["core_readout", "loopmtp_weight", "loopmtp_free_first", "loopmtp_proj",
            "loopmtp_ponder_weight", "loopmtp_gate_eps", "depth_fixed"]
    for k in keys:
        assert f"{k}:" in src_cfg, f"{k} is not declared in MORPHConfig"
        assert src_cfg.count(k) >= 2, f"{k} has no prose in the MORPHConfig comment block"
        assert f"{k}=" in src_build, f"{k} is not threaded by build_morph_config"


def test_the_three_arm_configs_compose():
    from hydra import compose, initialize_config_dir
    from morph.training.train import build_morph_config
    want = {
        "notul_norm_match_20k_d3_loopmtp": ("gated", 0.01, 3),
        "notul_norm_match_20k_d3fixed": ("last", 0.0, 3),
        "notul_norm_match_20k_d6fixed": ("last", 0.0, 6),
        "notul_norm_match_20k_d6_loopmtp": ("gated", 0.05, 6),
    }
    for name, (readout, lam, depth) in want.items():
        with initialize_config_dir(version_base=None,
                                   config_dir=os.path.join(ROOT, "morph", "configs")):
            cfg = compose(config_name=name, overrides=[])
        mc = build_morph_config(cfg, tul=None)
        assert mc.core_readout == readout, name
        assert mc.loopmtp_weight == lam, name
        assert (mc.mean_depth, mc.max_depth, mc.bptt_depth) == (depth, depth, depth), name
        assert mc.depth_fixed is True, name
        assert int(cfg.training.steps) == 20000 and int(cfg.training.ckpt_every) == 5000
        assert str(cfg.training.ternary_scale_mode) == "norm_match", name


# ── the iterates are the states the coda would have read ─────────────────────

def test_captured_iterates_end_at_the_last_readout_carrier():
    """`states[-1]` IS what the 'last' read-out hands the coda.

    Guards the collection point: gathering `h_a` (the state going IN to iteration t) instead
    of `h_new` (the state coming OUT) would shift every horizon by one and leave the final
    iterate missing.
    """
    m = _model(loopmtp_weight=0.05)     # 'last' read-out, so the carrier is unmodified
    x, _y = _xy()
    states, _ = _iterates(m, x)
    assert len(states) == 3
    captured: list[torch.Tensor] = []
    orig = m._back_region

    def spy(xx, *a, **kw):
        captured.append(xx.detach().clone())
        return orig(xx, *a, **kw)

    m._back_region = spy
    try:
        m.eval()
        with torch.no_grad():
            m(x)
    finally:
        m._back_region = orig
    assert captured, "the coda was not called"
    assert torch.equal(states[-1], captured[0]), (
        "the last captured iterate is not the carrier the coda read")
    assert not torch.equal(states[0], states[-1]), "the loop did not move the state"
