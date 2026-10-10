"""The gain hinge's per-step pass draw (`model.slot_gain_pass_force`, 2026-10-10; phase-2
tracker items A2/A3, .agents/notes/proposed/process/2026-10-10-phase-2-tracker.md).

Before this key the one-pass hinge drew its pass from the GLOBAL CPU stream and put the
stream back, so every step drew the SAME pass (54 of 54 draws: pass 6 on seed 1, pass 3 on
seed 2; /home/wolfe/morph-scratch/perf/hingepass/). Now the pass is drawn fresh every
training step from the model's private generator. One test per contract:

  1. Fresh per step: consecutive training forwards of one model read different passes, each
     inside the step's grad range, in exactly the order a twin of the generator predicts
     (one float64 uniform per step, pass = n_nograd + floor(u * n_grad)); 8000 draws are
     uniform over 8 passes (chi-square, p = 0.001 bound).
  2. Reproducible from the run seed: two models built after the same torch.manual_seed read
     the same sequence; another seed reads another.
  3. RNG-neutral: a drawing step and a step PINNED to the pass it drew leave the global
     stream bit-identical and give the same loss, gain reading and gradients, so the draw
     consumes nothing but its own generator and the pass is its only effect. Eval and
     no-grad forwards draw nothing.
  4. The generator rides the checkpoint (train.py's own save_checkpoint / load_checkpoint):
     a resumed model of another seed continues the SAME sequence; a checkpoint from before
     the key starts the draw from the run seed.
  5. The force key reads its pass every step and reaches the hinge's site (two forced passes
     read different gains); it is refused when inert, below -1 or past the last pass; a
     pass outside the step's grad range raises.
  6. GraphStep's host draw (`draw_gain_pass` + `pin_gain_pass`, the replayed path) is the
     eager forward's own draw on a `model.graph_safe` model: same passes, same losses and
     gain readings, bit for bit.

CPU, fp32. GraphStep itself (one graph per pass, replay against eager) is pinned on CUDA in
tests/test_graph_step.py.
"""
from __future__ import annotations

import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402  (tests/ is on sys.path)

from morph.model.transformer import _GAIN_PASS_SALT, MORPHTransformer

MAX_DEPTH = 4


def _model(seed: int = 3, **kw) -> MORPHTransformer:
    """The tests/test_slot_gain_reg.py fixture with 4 grad passes and a binding hinge."""
    torch.manual_seed(seed)
    tul = _tul(tg_restrict=False, sigreg_lambda=0.0, mux_beta=1.0, mux_target="next",
               mux_detach_head=False)
    base = dict(tul=tul, n_core=2, mean_depth=MAX_DEPTH, max_depth=MAX_DEPTH,
                bptt_depth=MAX_DEPTH, retention=False, dropout=0.1,
                slot_gain_lambda=1.0, slot_gain_target=0.0)
    base.update(kw)
    return MORPHTransformer(_cfg(**base)).train()


def _fwd(m: MORPHTransformer, seed: int = 7, backward: bool = False):
    """One training forward on the fixed batch, the global stream seeded first. Returns the
    output, the pass the step recorded (cleared, as the trainer clears it) and the global
    CPU stream after the step."""
    x, y, layout, _ = _batch()
    torch.manual_seed(seed)
    out = m(x, labels=y, slot_layout=layout)
    if backward:
        out["loss"].backward()
    t = m._gain_pass_last
    m._gain_pass_last = None
    return out, t, torch.get_rng_state()


def _twin(seed: int, n: int, n_grad: int, n_nograd: int = 0) -> list[int]:
    g = torch.Generator().manual_seed((seed + _GAIN_PASS_SALT) % (1 << 63))
    return [n_nograd + int(float(torch.rand((), generator=g, dtype=torch.float64)) * n_grad)
            for _ in range(n)]


# ── 1. fresh per step ───────────────────────────────────────────────────────────────


def test_every_step_draws_a_fresh_pass_in_the_grad_range():
    m = _model(seed=3)
    seq = [_fwd(m, seed=7 + i)[1] for i in range(16)]
    assert all(t is not None and 0 <= t < MAX_DEPTH for t in seq), seq
    assert len(set(seq)) >= 3, f"the pass did not move: {seq}"
    # one float64 uniform per step from the private generator, nothing else
    assert seq == _twin(3, 16, MAX_DEPTH), (seq, _twin(3, 16, MAX_DEPTH))


def test_the_draw_is_uniform_over_the_grad_passes():
    m = _model(seed=3)
    n, k = 8000, 8
    counts = torch.bincount(torch.tensor([m._gain_pass_for_step(2, k) - 2 for _ in range(n)]),
                            minlength=k).double()
    assert counts.numel() == k and float(counts.min()) > 0, counts
    chi2 = float(((counts - n / k) ** 2 / (n / k)).sum())
    assert chi2 < 24.32, (chi2, counts.tolist())       # chi-square, 7 dof, p = 0.001


# ── 2. reproducible from the run seed ───────────────────────────────────────────────


def test_the_sequence_is_a_function_of_the_run_seed():
    a = [_fwd(_m, seed=7)[1] for _m in [_model(seed=3)] for _ in range(12)]
    b_model = _model(seed=3)
    b = [_fwd(b_model, seed=7)[1] for _ in range(12)]
    c_model = _model(seed=4)
    c = [_fwd(c_model, seed=7)[1] for _ in range(12)]
    assert a == b
    assert a != c, "another run seed drew the same pass sequence"


# ── 3. RNG neutrality ───────────────────────────────────────────────────────────────


def test_the_draw_moves_no_other_stream_and_only_picks_the_pass():
    a, b = _model(seed=3), _model(seed=3)
    gen0 = b.gain_pass_rng_state().clone()
    for i in range(5):
        a.zero_grad(set_to_none=True)
        b.zero_grad(set_to_none=True)
        oa, ta, ra = _fwd(a, seed=10 + i, backward=True)
        b.pin_gain_pass(ta)
        try:
            ob, tb, rb = _fwd(b, seed=10 + i, backward=True)
        finally:
            b.pin_gain_pass(None)
        assert tb is None, "a pinned forward recorded a draw"
        assert torch.equal(ra, rb), f"step {i}: the draw moved the global CPU stream"
        assert torch.equal(oa["loss"], ob["loss"]), i
        assert torch.equal(oa["gain_est"], ob["gain_est"]), i
        for (n, pa), pb in zip(a.named_parameters(), b.parameters()):
            assert (pa.grad is None) == (pb.grad is None), n
            assert pa.grad is None or torch.equal(pa.grad, pb.grad), (i, n)
    # the pinned model never touched its generator
    assert torch.equal(b.gain_pass_rng_state(), gen0)


def test_eval_and_no_grad_forwards_draw_nothing():
    m = _model(seed=3)
    gen0 = m.gain_pass_rng_state().clone()
    m.eval()
    _o, t, _r = _fwd(m)
    assert t is None
    m.train()
    with torch.no_grad():
        _o, t, _r = _fwd(m)
    assert t is None
    assert torch.equal(m.gain_pass_rng_state(), gen0)
    assert _fwd(m)[1] == _twin(3, 1, MAX_DEPTH)[0]


# ── 4. the checkpoint ───────────────────────────────────────────────────────────────


def _save(m, path):
    from morph.training.train import save_checkpoint
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    save_checkpoint(str(path), 5, m, opt, torch.amp.GradScaler("cuda"), pruning=None,
                    next_step=6)


def _load(m, path):
    from morph.training.train import load_checkpoint
    load_checkpoint(str(path), m, torch.amp.GradScaler("cuda"), torch.device("cpu"),
                    pruning=None)


def test_the_generator_rides_the_checkpoint(tmp_path, capsys):
    a = _model(seed=3)
    for i in range(5):
        _fwd(a, seed=20 + i)
    path = tmp_path / "step_5.pt"
    _save(a, path)
    assert "gain_pass_rng" in torch.load(path, map_location="cpu", weights_only=False)
    expect = [_fwd(a, seed=30 + i)[1] for i in range(8)]
    b = _model(seed=99)          # another seed: only the checkpoint can make it agree
    _load(b, path)
    assert [_fwd(b, seed=30 + i)[1] for i in range(8)] == expect
    # A checkpoint from before the key: the draw starts from THIS run's seed.
    raw = torch.load(path, map_location="cpu", weights_only=False)
    raw.pop("gain_pass_rng")
    legacy = tmp_path / "legacy.pt"
    torch.save(raw, legacy)
    c = _model(seed=3)
    capsys.readouterr()
    _load(c, legacy)
    assert "predates the per-step hinge pass draw" in capsys.readouterr().out
    assert [_fwd(c, seed=30 + i)[1] for i in range(8)] == _twin(3, 8, MAX_DEPTH)


def test_a_model_that_draws_nothing_refuses_a_generator_state():
    a = _model(seed=3)
    f = _model(seed=3, slot_gain_pass_force=1)
    assert f.gain_pass_rng_state() is None
    with pytest.raises(RuntimeError, match="draws no pass"):
        f.set_gain_pass_rng_state(a.gain_pass_rng_state())


# ── 5. the force key ────────────────────────────────────────────────────────────────


def test_force_reads_its_pass_every_step_and_reaches_the_hinge():
    m = _model(seed=3, slot_gain_pass_force=2)
    assert m._gain_pass_gen is None and m.gain_pass_choices() == (2,)
    seq = [_fwd(m, seed=7 + i)[1] for i in range(5)]
    assert seq == [2] * 5
    # The forced pass is the hinge's site: pass 0 and pass 3 read the map at different
    # operating points, the same step otherwise.
    g0 = _fwd(_model(seed=3, slot_gain_pass_force=0), seed=7)[0]["gain_est"]
    g3 = _fwd(_model(seed=3, slot_gain_pass_force=3), seed=7)[0]["gain_est"]
    assert not torch.equal(g0, g3), (float(g0), float(g3))
    # ...and a forced pass is exactly the drawing model pinned to it
    p = _model(seed=3)
    p.pin_gain_pass(3)
    assert torch.equal(_fwd(p, seed=7)[0]["gain_est"], g3)


@pytest.mark.parametrize("kw,needle", [
    (dict(slot_gain_lambda=0.0, slot_gain_target=0.9, slot_gain_pass_force=1),
     "needs model.slot_gain_lambda > 0"),
    (dict(slot_gain_all_iters=True, slot_gain_pass_force=1), "needs model.slot_gain_lambda > 0"),
    (dict(slot_gain_pass_force=MAX_DEPTH), "past the slot loop's last pass"),
    (dict(slot_gain_pass_force=-2), "must be -1"),
])
def test_force_is_refused_when_inert_or_out_of_range(kw, needle):
    with pytest.raises(ValueError, match=needle):
        _model(seed=3, **kw)


def test_a_pass_outside_the_steps_grad_range_raises():
    # bptt_depth 2 of 4 passes: passes 0 and 1 run without grad, the hinge reads 2 or 3.
    m = _model(seed=3, bptt_depth=2)
    seq = [_fwd(m, seed=7 + i)[1] for i in range(8)]
    assert set(seq) <= {2, 3} and seq == _twin(3, 8, 2, n_nograd=2), seq
    m.pin_gain_pass(0)
    with pytest.raises(ValueError, match="not a grad pass"):
        _fwd(m)
    m.pin_gain_pass(None)
    with pytest.raises(ValueError, match="not a grad pass"):
        _fwd(_model(seed=3, bptt_depth=2, slot_gain_pass_force=1))


# ── 6. GraphStep's host draw is the eager draw ──────────────────────────────────────


def test_the_host_draw_and_pin_reproduce_the_eager_draw_on_graph_safe():
    from test_graph_safe import _model as _gs_model
    from test_host_shadow import _bare
    from test_tul_fan import _batch as _fan_batch
    from test_tul_lx_credit import M

    def run(host: bool):
        torch.manual_seed(11)
        m = _gs_model(True)
        assert m.gain_pass_choices() == tuple(range(int(m.cfg.max_depth)))
        rows = []
        for s in range(6):
            _ids, inp, lab, layout = _fan_batch(M, seed=s)
            if host:
                t = m.draw_gain_pass()
                m.pin_gain_pass(t)
            torch.manual_seed(7)
            try:
                out = m(inp, labels=lab, slot_layout=_bare(layout).to("cpu"))
            finally:
                m.pin_gain_pass(None)
            rows.append((m._gain_pass_last, out["loss"].detach(), out["gain_est"].detach()))
            m._gain_pass_last = None
        return rows

    eager, host = run(False), run(True)
    assert len({r[0] for r in eager}) >= 2, [r[0] for r in eager]
    for s, (e, h) in enumerate(zip(eager, host)):
        assert e[0] == h[0], (s, e[0], h[0])
        assert torch.equal(e[1], h[1]) and torch.equal(e[2], h[2]), s
