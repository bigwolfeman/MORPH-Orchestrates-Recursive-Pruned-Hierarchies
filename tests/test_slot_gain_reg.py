"""Phase-2 levers on the slot loop's forward map: `model.slot_state_renorm` and
`model.slot_gain_lambda` (design note 2026-09-04-loop-contractivity-as-design).

Contracts, one test each:
  1. the gain penalty leaves the run's random stream and the model's own losses exactly
     where they were (RNG discipline: the two extra core steps draw and put back);
  2. its gradient reaches the core's weights and NOTHING upstream of the loop (the
     operating point and the injection source are detached);
  3. a hinge that never binds adds exactly zero (grads bit-identical to lambda 0) while
     the live gain reading is still produced;
  4. the finite-difference gain is linear in its step (eps 0.02 vs 0.005 agree);
  5. the renorm pins every real slot's exit norm to its entry norm and keeps pads at 0;
  6. both knobs are refused without a slot loop.
"""
from __future__ import annotations

import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402  (tests/ is on sys.path)

from morph.model.transformer import MORPHTransformer

MAX_DEPTH = 3


def _model(seed: int = 0, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul = _tul(tg_restrict=False, sigreg_lambda=0.0, mux_beta=1.0, mux_target="next",
               mux_detach_head=False)
    base = dict(tul=tul, n_core=2, mean_depth=MAX_DEPTH, max_depth=MAX_DEPTH,
                bptt_depth=MAX_DEPTH, retention=False, dropout=0.1)
    base.update(kw)
    return MORPHTransformer(_cfg(**base))


def _run(m: MORPHTransformer, *, seed: int = 7, train: bool = True):
    x, y, layout, _ = _batch()
    m.train(train)
    torch.manual_seed(seed)
    out = m(x, labels=y, slot_layout=layout)
    out["loss"].backward()
    grads = {n: (p.grad.detach().clone() if p.grad is not None else None)
             for n, p in m.named_parameters()}
    return out, grads, torch.get_rng_state()


def _same(ga, gb, names=None) -> bool:
    keys = names or ga.keys()
    return all(((ga[k] is None) == (gb[k] is None)) and (ga[k] is None or torch.equal(ga[k], gb[k]))
               for k in keys)


def test_gain_penalty_is_rng_neutral_and_leaves_the_model_losses_untouched():
    out_r, g_r, rng_r = _run(_model(seed=3))
    out_p, g_p, rng_p = _run(_model(seed=3, slot_gain_lambda=1.0, slot_gain_target=0.0))
    assert torch.equal(rng_r, rng_p), "the penalty moved the run's random stream"
    assert torch.equal(out_r["mux_local"], out_p["mux_local"])
    assert "gain_est" in out_p and float(out_p["gain_est"]) > 0.0
    assert float(out_p["gain_reg_weighted"]) > 0.0
    assert torch.allclose(out_p["loss"] - out_p["gain_reg_weighted"], out_r["loss"], atol=1e-6)


def test_gain_penalty_reaches_the_core_and_nothing_upstream():
    _, g_r, _ = _run(_model(seed=3))
    _, g_p, _ = _run(_model(seed=3, slot_gain_lambda=1.0, slot_gain_target=0.0))
    core = [n for n in g_r if n.startswith("core.")]
    upstream = [n for n in g_r if n.startswith(("prelude.", "embed", "tok_emb", "input_norm"))]
    assert core and upstream
    assert not _same(g_r, g_p, core), "the penalty left the core's gradients untouched"
    assert _same(g_r, g_p, upstream), "the penalty leaked upstream of the loop"


def test_unbinding_hinge_adds_exactly_zero():
    _, g_r, _ = _run(_model(seed=3))
    out_p, g_p, _ = _run(_model(seed=3, slot_gain_lambda=1.0, slot_gain_target=1e6))
    assert float(out_p["gain_reg_weighted"]) == 0.0
    assert float(out_p["gain_est"]) > 0.0
    assert _same(g_r, g_p), "a hinge that never binds changed a gradient"


def test_finite_difference_gain_is_linear_in_its_step():
    a, _, _ = _run(_model(seed=3, slot_gain_lambda=1.0, slot_gain_target=0.0, slot_gain_eps=0.02))
    b, _, _ = _run(_model(seed=3, slot_gain_lambda=1.0, slot_gain_target=0.0, slot_gain_eps=0.005))
    ga, gb = float(a["gain_est"]), float(b["gain_est"])
    assert abs(ga - gb) / max(gb, 1e-6) < 0.10, (ga, gb)


def test_renorm_pins_real_slots_to_their_entry_norm_and_pads_to_zero():
    m = _model(seed=3, slot_state_renorm=True)
    x, y, layout, _ = _batch()
    m.eval()
    torch.manual_seed(7)
    x0 = m.embed(x) if hasattr(m, "embed") else None
    # Read the loop's entry and exit states through the model's own gather.
    captured = {}
    orig = m._tul_core
    def spy(*a, **k):
        res = orig(*a, **k)
        captured["h_exit"] = res[1]
        return res
    m._tul_core = spy
    out = m(x, labels=y, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    h_exit = captured["h_exit"]
    valid = layout.slot_valid
    exit_n = h_exit.detach().flatten(2).float().norm(dim=2)
    # Entry norm: rerun the entry path (core_init of the gathered prelude state).
    from morph.model.tul import gather_valid
    torch.manual_seed(7)
    m._tul_core = orig
    # The entry norm is what `_tul_core` pins to; recover it from a renorm-off twin, whose
    # entry state is identical (same weights, same seed, renorm touches nothing before it).
    twin = _model(seed=3)
    twin.eval()
    entry = {}
    orig_t = twin._tul_core
    def spy_t(x_, x0_, bg_, layout_, **k):
        xn = twin.input_norm(x_)
        e = gather_valid(xn, layout_.slot_index, layout_.slot_valid)
        entry["n0"] = twin.core_init(e).detach().flatten(2).float().norm(dim=2)
        return orig_t(x_, x0_, bg_, layout_, **k)
    twin._tul_core = spy_t
    torch.manual_seed(7)
    twin(x, labels=y, slot_layout=layout)
    n0 = entry["n0"]
    assert torch.allclose(exit_n[valid], n0[valid], rtol=2e-2), (exit_n[valid][:4], n0[valid][:4])
    assert torch.all(exit_n[~valid] == 0.0)


def test_knobs_are_inert_with_a_notice_where_there_is_no_slot_loop(capsys):
    # A coreless TUL model (the GL1 arm and its control) builds with base.yaml's constraint
    # on, and says once that the levers are inert there.
    capsys.readouterr()
    _model(slot_state_renorm=True, slot_gain_lambda=1.0, n_core=0)
    out = capsys.readouterr().out
    assert "[slot-levers]" in out and "INERT" in out and "n_core=0" in out, out
    # base.yaml carries the constraint ON; a plain model (no TUL block at all) must build,
    # carry no slot loop, and say so once.
    from morph.model.transformer import MORPHConfig
    from test_tul_gl1 import V
    torch.manual_seed(0)
    plain = MORPHConfig(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=2, max_depth=2, bptt_depth=1,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4, hca_compress_ratio=8,
        top_k=8, window_size=8, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0, retention=False, slot_gain_lambda=100.0, slot_cot_clip=4.0)
    assert plain.tul is None
    capsys.readouterr()
    m = MORPHTransformer(plain)
    out = capsys.readouterr().out
    assert "[slot-levers]" in out and "INERT" in out and "no TUL block" in out, out
    assert m.tul is None


# ── slot_gain_all_iters: the hinge at EVERY grad iteration (arc E2's rerun rule) ─────────

def test_all_iters_hinge_covers_every_grad_iteration_and_stays_rng_neutral():
    out_a, _, rng_a = _run(_model(seed=3, slot_gain_lambda=1.0, slot_gain_target=0.0,
                                  slot_gain_all_iters=True))
    _, _, rng_0 = _run(_model(seed=3, slot_gain_lambda=0.0))
    assert float(out_a["gain_n_iters"]) == MAX_DEPTH          # bptt_depth == max_depth here
    assert torch.equal(rng_a, rng_0), "the every-iteration hinge moved the global RNG stream"
    assert float(out_a["gain_est_max"]) >= float(out_a["gain_est"])


def test_all_iters_unbinding_hinge_adds_exactly_zero():
    out_p, g_p, _ = _run(_model(seed=3, slot_gain_lambda=1.0, slot_gain_target=1e6,
                                slot_gain_all_iters=True))
    out_0, g_0, _ = _run(_model(seed=3, slot_gain_lambda=0.0))
    assert float(out_p["gain_reg_weighted"]) == 0.0
    assert torch.equal(out_p["loss"], out_0["loss"]) and _same(g_p, g_0)


def test_all_iters_penalty_is_at_least_the_single_sample_penalty_in_expectation():
    """Every iteration's hinge sums, so with the target at 0 the sum over T iterations is
    at least any one iteration's term: the all-iters penalty is never below the
    single-sample penalty of the SAME model at the same RNG seed."""
    single = _run(_model(seed=3, slot_gain_lambda=1.0, slot_gain_target=0.0))[0]
    every = _run(_model(seed=3, slot_gain_lambda=1.0, slot_gain_target=0.0,
                        slot_gain_all_iters=True))[0]
    assert float(every["gain_reg_weighted"]) >= float(single["gain_reg_weighted"]) * 0.999


# ── checkpoint policy of the slot loop and the hinge (graph-step speed work, 2026-10-08) ──

_ON = dict(slot_gain_lambda=1.0, slot_gain_target=0.0)


def test_hinge_without_checkpoint_is_bit_identical():
    """`model.slot_gain_no_ckpt`: the hinge's applications keep their activations. The
    checkpoint recomputes with the RNG state it saved, so the loss, every gradient and the
    run's random stream are bit-identical to the checkpointed hinge (dropout 0.1 here, so
    a mask drawn from the wrong state would show)."""
    out_c, g_c, rng_c = _run(_model(seed=3, **_ON))
    out_n, g_n, rng_n = _run(_model(seed=3, slot_gain_no_ckpt=True, **_ON))
    assert float(out_c["gain_reg_weighted"]) > 0.0
    assert torch.equal(out_c["loss"], out_n["loss"])
    assert torch.equal(out_c["gain_est"], out_n["gain_est"])
    assert _same(g_c, g_n), "the un-checkpointed hinge changed a gradient"
    assert torch.equal(rng_c, rng_n)


def test_slot_loop_without_checkpoint_is_bit_identical():
    """`ckpt_grad_iters` 0 (no pass of the slot loop checkpointed) against -1 (every grad
    pass checkpointed), with the hinge on: the same loss and gradients, bit for bit."""
    out_a, g_a, _ = _run(_model(seed=3, ckpt_grad_iters=-1, **_ON))
    out_b, g_b, _ = _run(_model(seed=3, ckpt_grad_iters=0, slot_gain_no_ckpt=True, **_ON))
    assert torch.equal(out_a["loss"], out_b["loss"])
    assert _same(g_a, g_b), "the un-checkpointed slot loop changed a gradient"


def test_reuse_f0_reads_the_same_gain_at_dropout_zero():
    """`model.slot_gain_reuse_f0` at dropout 0: f(h) is the loop's step output, the same
    map at the same point, and the direction is drawn from the same state, so the gain and
    the penalty are the two-application hinge's bit for bit, and the run's losses and RNG
    stream are untouched. The gradients differ by design (next tests)."""
    out_a, g_a, rng_a = _run(_model(seed=3, dropout=0.0, **_ON))
    out_b, g_b, rng_b = _run(_model(seed=3, dropout=0.0, slot_gain_reuse_f0=True, **_ON))
    assert torch.equal(out_a["gain_est"], out_b["gain_est"])
    assert torch.equal(out_a["gain_reg_weighted"], out_b["gain_reg_weighted"])
    assert torch.equal(out_a["mux_local"], out_b["mux_local"])
    assert torch.equal(rng_a, rng_b)


def test_reuse_f0_replays_the_steps_dropout_masks():
    """At dropout 0.1, f(h + d) must run under the loop step's masks: two applications
    under different masks differ by O(1) of the state, which over a step of size eps reads
    as a gain of order 1/eps. With the masks matched the gain is the map's, close to the
    two-application hinge's (each pair of applications shares its own masks)."""
    out_a, _, rng_a = _run(_model(seed=3, **_ON))
    out_b, _, rng_b = _run(_model(seed=3, slot_gain_reuse_f0=True, **_ON))
    ga, gb = float(out_a["gain_est"]), float(out_b["gain_est"])
    assert abs(ga - gb) / ga < 0.25, (ga, gb)
    assert torch.equal(out_a["mux_local"], out_b["mux_local"])
    assert torch.equal(rng_a, rng_b), "the reused hinge moved the run's random stream"


def test_reuse_f0_gradient_reaches_upstream_of_the_loop_at_the_hinges_scale():
    """The stated cost of `slot_gain_reuse_f0`: f(h) is not detached at its input, so the
    hinge's gradient also reaches the prelude (the two-application hinge never does:
    `test_gain_penalty_reaches_the_core_and_nothing_upstream`). f(h + d) runs at the same
    live point, so the two input gradients cancel to first order and what reaches upstream
    is the gain's own dependence on the operating point: no larger than the hinge's own
    gradient on the core (a detached f(h + d) sent O(1 / eps): 70.9 against 0.28).
    Dropout 0, so the two hinges read one map."""
    _, g_r, _ = _run(_model(seed=3, dropout=0.0))
    _, g_a, _ = _run(_model(seed=3, dropout=0.0, **_ON))
    _, g_b, _ = _run(_model(seed=3, dropout=0.0, slot_gain_reuse_f0=True, **_ON))
    upstream = [n for n in g_r if n.startswith("prelude.") and g_r[n] is not None]
    core = [n for n in g_r if n.startswith("core.") and g_r[n] is not None]
    assert upstream and core
    assert _same(g_r, g_a, upstream) and not _same(g_r, g_b, upstream)
    def _nrm(ga, gb, names):
        return float(sum((ga[n] - gb[n]).double().pow(2).sum() for n in names)) ** 0.5
    hinge_core = _nrm(g_a, g_r, core)            # the two-application hinge on the core
    leak = _nrm(g_b, g_r, upstream)              # the reused hinge upstream of the loop
    assert 0.0 < leak < hinge_core, (leak, hinge_core)
