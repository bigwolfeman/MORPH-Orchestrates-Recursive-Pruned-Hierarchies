"""``training.graph_step`` (2026-10-07, tasks 0.2 and 1.6-1.8 of
.agents/notes/proposed/architecture/2026-10-07-graph-captured-training-step.md).

morph/training/graph_step.py records the training step (forward, backward, clip, found-inf
flag, optimizer step, the fan's post-step update) as a CUDA graph per step kind and replays it.

What each test pins:
  1. THE REPLAY IS THE EAGER STEP, BIT FOR BIT. A tiny winner (`lxtul_pointer_ditto`
     composed by Hydra: LXTUL fan, latent-selected loop, pointer heads, DITTO rows, gain
     hinge, fixed-point term, ternary STE, the EMA twin; `model.graph_safe`,
     `training.capturable_optimizer`) trains 12 steps on batches whose slot counts differ,
     once all eager (the trainer's sequence, written out here) and once through GraphStep:
     warm-up, the recording of both kinds (regular, instrument), replays of both, an EAGER
     regular and an EAGER instrument step between replays, an eval forward with the graphs
     released and their re-recording after it, and a step whose gradients are
     inf (skipped on the device). After every step the loss, the grad norm, every `out`
     reading (the fan's instruments), every gradient, every parameter and buffer (the EMA
     twin included), the optimizer state and the step counters are equal bit for bit.
  2. The CPU generator moving after capture (the gain hinge would draw another pass) raises
     before the replay.
  3. A rebound parameter raises before the replay (the address check); a different phase
     context or a missing DITTO field raises.
  4. Config refusals (CPU): without its two halves, with `ce_compact_rows`, a probe, an
     in-run prune event, step_mix, curriculum; the winner configs are accepted.

Sabotage checks, run and reverted (session report, not committed):
  (a) StaticBatch.load skips `layout.slot_valid` -> test 1 fails at the first replay whose
      batch differs from the capture batch.
  (b) eager steps run on the default stream instead of the capture stream (`_side`
      bypassed) -> the recording fails (cudaErrorStreamCaptureImplicit) and test 1 errors.
  (c) a replay does not bind its graph's gradient buffers to `p.grad` -> test 1 fails at
      the first replay ("grad presence").
CUDA (the capture), fp32 params, bf16 autocast as the trainer runs it.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib

import numpy as np
import pytest
import torch

from morph.model.tul_layout import TulLayoutSpec, pack_tul_batch
from test_tul_strict_geometry import _rule, _StubTok

# Bit-for-bit needs the deterministic kernels on BOTH runs (the eager reference against itself
# differs in scatter/index_add backward atomics otherwise): train.py's
# `training.deterministic`, which needs this before cuBLAS initialises.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

CFG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "morph" / "configs")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture")

V, DOT = 64, 10
# The tiny width of tests/test_tul_fan.py; every other MORPHConfig field is the winner's.
_TINY = dict(d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256,
             context_len=256, n_prelude=2, n_core=2, n_coda=2, channel_dims=(32, 20, 12),
             compression=2, csa_compress_ratio=4, hca_compress_ratio=8, top_k=8,
             window_size=16, bigram_hash_vocab=V, d_ff=96)
B, SEQ = 3, 64
_ON = ("model.graph_safe=true", "training.capturable_optimizer=true")


def _compose(monkeypatch, name="lxtul_pointer_ditto", *overrides):
    import transformers
    from hydra import compose, initialize_config_dir

    from morph.training import tul_setup
    monkeypatch.setattr(transformers, "AutoTokenizer", _StubTok)
    monkeypatch.setattr(tul_setup, "build_boundary_rule",
                        lambda cfg, cache_dir="": (_rule(), _rule().is_boundary, 0, ("\n",)))
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = compose(config_name=name, overrides=list(overrides))
    return cfg, tul_setup.build_tul_runtime(cfg)


def _build(cfg, rt, seed=0):
    """The trainer's build order (train.py: config, model, quantisation, the EMA twin,
    optimizer) at the tiny width."""
    from morph.model.transformer import MORPHTransformer
    from morph.training.optimizer import create_optimizer
    from morph.training.quant_setup import apply_quantization
    from morph.training.train import build_morph_config
    mc = dataclasses.replace(build_morph_config(cfg, tul=rt.model_cfg), **_TINY)
    torch.manual_seed(seed)
    m = MORPHTransformer(mc).cuda()
    apply_quantization(m, cfg)
    m.tul_fan_target_build()
    if hasattr(m, "tul_fan_target_sync"):
        m.tul_fan_target_sync()
    m.train()
    return m, create_optimizer(m, cfg)


def _stream(n: int, seed: int, p_boundary: float) -> list[int]:
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=n)
    ids[ids == 4] = 5
    ids[rng.random(n) < p_boundary] = DOT
    return ids.astype(np.int64).tolist()


def _batches(rt, n: int, ditto: bool = True):
    """``n`` packed batches whose valid slot counts differ (boundary density varies)."""
    spec = TulLayoutSpec(seq_len=SEQ, prefix_k=rt.data_cfg.prefix_k, max_slots=10,
                         slot_id=rt.data_cfg.slot_id)
    out = []
    for i in range(n):
        buf = _stream(B * 400, seed=100 + i, p_boundary=(0.06, 0.12, 0.2)[i % 3])
        x, y, lay = pack_tul_batch(buf, _rule(), spec, B,
                                   ditto_rows=1 if ditto else 0,
                                   ditto_rng=np.random.default_rng(i) if ditto else None)
        out.append((x.cuda(), y.cuda(), lay.to("cuda")))
    counts = {int(b[2].slot_valid.sum()) for b in out}
    assert len(counts) >= 3, f"batches must differ in slot count, got {counts}"
    return out


def _fwd(m):
    return lambda x, y, lay: m(x, labels=y, bag_size=0, slot_layout=lay, tul_step_mode=None)


def _zero_pen():
    """The logging-only spectral penalty's term (lam 0 -> an exact fp32 zero)."""
    return torch.zeros((), device="cuda", dtype=torch.float32)


def _eager_step(m, opt, x, y, lay, clip):
    """train.py's eager sequence under capturable_optimizer (disabled GradScaler), written
    out independently of GraphStep._body."""
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(x, labels=y, bag_size=0, slot_layout=lay, tul_step_mode=None)
        loss = out["loss"]
        loss = loss + _zero_pen().to(loss.dtype)
    (loss / 1).backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
    opt.mark_found_inf(gn)
    opt.step()
    m.tul_fan_after_step()
    return out, loss.detach(), gn.detach()


def _snap(m, opt, out, loss, gn, grads):
    tw = m.__dict__.get("_fan_target")
    return {
        "loss": loss.clone(), "gnorm": gn.clone(),
        "out": {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in out.items()},
        "grads": [None if g is None else g.detach().clone() for g in grads],
        "params": [p.detach().clone() for p in m.parameters()],
        "buffers": {n: b.clone() for n, b in m.named_buffers()},
        "twin": None if tw is None else [t.detach().clone() for t in
                                         list(tw.parameters()) + list(tw.buffers())],
        "state": [{k: v.clone() for k, v in opt.state[p].items() if torch.is_tensor(v)}
                  for p in m.parameters()],
        "steps": [int(cb["step"].item()) for cb in opt._cap],
    }


_BITS = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16,
         torch.float64: torch.int64}


def _eq(u: torch.Tensor, v: torch.Tensor) -> bool:
    """Bit equality: a float tensor is compared as its integer bit pattern, so a NaN (the
    inf step's gradients) equals the same NaN and -0.0 differs from +0.0."""
    if u.shape != v.shape or u.dtype != v.dtype:
        return False
    if u.dtype in _BITS:
        return torch.equal(u.contiguous().view(_BITS[u.dtype]), v.contiguous().view(_BITS[v.dtype]))
    return torch.equal(u, v)


def _same(a, b, where):
    assert _eq(a["loss"], b["loss"]), (where, "loss", a["loss"], b["loss"])
    assert _eq(a["gnorm"], b["gnorm"]), (where, "gnorm", a["gnorm"], b["gnorm"])
    # every reading the trainer logs (the fan's instruments on the instrument step)
    assert a["out"].keys() == b["out"].keys(), (where, a["out"].keys() ^ b["out"].keys())
    for k, u in a["out"].items():
        v = b["out"][k]
        assert (_eq(u, v) if torch.is_tensor(u) else u == v), (where, "out", k)
    for i, (u, v) in enumerate(zip(a["grads"], b["grads"])):
        assert (u is None) == (v is None), (where, "grad presence", i)
        assert u is None or _eq(u, v), (where, "grad", i)
    for i, (u, v) in enumerate(zip(a["params"], b["params"])):
        assert _eq(u, v), (where, "param", i)
    assert a["buffers"].keys() == b["buffers"].keys()
    for n in a["buffers"]:
        assert _eq(a["buffers"][n], b["buffers"][n]), (where, "buffer", n)
    if a["twin"] is not None:
        for i, (u, v) in enumerate(zip(a["twin"], b["twin"])):
            assert _eq(u, v), (where, "twin", i)
    for i, (u, v) in enumerate(zip(a["state"], b["state"])):
        assert u.keys() == v.keys(), (where, "state keys", i)
        for k in u:
            assert _eq(u[k], v[k]), (where, "state", i, k)
    assert a["steps"] == b["steps"], (where, "step counters", a["steps"], b["steps"])


def _poison(m, buf):
    """A device scalar multiplied into one parameter's gradient: 1.0, or inf on the step
    that must be skipped. Registered at build, so the capture records it."""
    p = next(p for n, p in m.named_parameters() if "coda" in n and p.dim() == 2)
    p.register_hook(lambda g: g * buf)


# (kind, how GraphStep runs it). kind True = the instrument step (step % 20 == 0).
# Step 2 records BOTH kinds, then replays its own.
# After step 6 an eval runs with the graphs released (the trainer's eval), so step 8
# records both kinds again.
_PLAN = [(True, "eager"), (False, "eager"), (False, "capture"), (False, "replay"),
         (True, "replay"), (False, "run_eager"), (False, "replay"), (True, "run_eager"),
         (False, "capture"), (False, "replay"), (True, "replay"), (False, "replay")]
_INF_STEP = 8
_EVAL_AFTER = 6        # an eval forward (eval mode, no grad) after this step, in both runs


def _eval_forward(m, x, y, lay):
    m.eval()
    with torch.no_grad():
        ev = m(x, labels=y, slot_layout=dataclasses.replace(lay, ditto_prev=None))["loss"]
    m.train()
    return ev.cpu()           # nothing device-side outlives the eval


@pytest.fixture
def deterministic():
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    yield
    torch.use_deterministic_algorithms(prev)


@cuda
def test_replay_is_the_eager_step_bit_for_bit(monkeypatch, deterministic):
    from morph.training.graph_step import GraphStep
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON)
    clip = float(cfg.training.grad_clip)
    batches = _batches(rt, len(_PLAN))
    lr = [1e-3 * (1 + 0.1 * s) for s in range(len(_PLAN))]

    # ── reference: all eager ───────────────────────────────────────────────────────
    ma, oa = _build(cfg, rt)
    pa = torch.ones((), device="cuda")
    _poison(ma, pa)
    torch.manual_seed(1234)
    ref = []
    for s, ((kind, _how), (x, y, lay)) in enumerate(zip(_PLAN, batches)):
        ma._train_instruments = kind
        for g in oa.param_groups:
            g["lr"] = lr[s]
        oa.write_step_scalars()
        pa.fill_(float("inf") if s == _INF_STEP else 1.0)
        out, loss, gn = _eager_step(ma, oa, x, y, lay, clip)
        ref.append(_snap(ma, oa, out, loss, gn, [p.grad for p in ma.parameters()]))
        if s == _EVAL_AFTER:
            ref_eval = _eval_forward(ma, x, y, lay)
    assert not torch.isfinite(ref[_INF_STEP]["gnorm"])
    assert ref[_INF_STEP]["steps"] == ref[_INF_STEP - 1]["steps"]       # skipped
    del ma, oa

    # ── GraphStep ──────────────────────────────────────────────────────────────────
    mb, ob = _build(cfg, rt)
    pb = torch.ones((), device="cuda")
    _poison(mb, pb)
    gs = GraphStep(mb, ob, grad_clip=clip, forward=_fwd(mb), kinds=(False, True),
                   set_kind=lambda k: setattr(mb, "_train_instruments", k),
                   loss_terms=_zero_pen, after_step=mb.tul_fan_after_step, first_capture=2,
                   keep_grads=(True, False), compiled=False)
    torch.manual_seed(1234)
    for s, ((kind, how), (x, y, lay)) in enumerate(zip(_PLAN, batches)):
        mb._train_instruments = kind
        for g in ob.param_groups:
            g["lr"] = lr[s]
        ob.write_step_scalars()
        pb.fill_(float("inf") if s == _INF_STEP else 1.0)
        ob.zero_grad(set_to_none=True)                  # the trainer's top-of-step zero_grad
        if how == "run_eager":
            out, loss, gn = gs.run_eager(x, y, lay)
            mode = "eager"
        else:
            out, loss, gn, mode = gs.step(s, kind, x, y, lay, context="tul")
        assert mode == ("eager" if how == "run_eager" else how), (s, mode, how)
        # After a replay `p.grad` is the graph's own gradient buffer (keep_grads binds it),
        # which is what the trainer's log block reads.
        grads = [p.grad for p in mb.parameters()]
        _same(ref[s], _snap(mb, ob, out, loss, gn, grads), f"step {s} ({how}, kind={kind})")
        if s == _EVAL_AFTER:
            # eval with the graphs released (the trainer's eval and generation); the
            # re-recording after it must change nothing
            with gs.released():
                assert gs.graphs == {}
                ev = _eval_forward(mb, x, y, lay)
            assert _eq(ev, ref_eval)
    assert gs.stats.replayed == 8 and gs.stats.captures == 4 and gs.stats.eager == 4
    assert gs.stats.releases == 1
    assert {k: c.replays for k, c in gs.graphs.items()} == {False: 3, True: 1}


def _captured(monkeypatch):
    """A tiny winner with both kinds recorded (steps 0-1 eager, step 2 records)."""
    from morph.training.graph_step import GraphStep
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON)
    m, opt = _build(cfg, rt)
    gs = GraphStep(m, opt, grad_clip=float(cfg.training.grad_clip), forward=_fwd(m),
                   kinds=(False, True), set_kind=lambda k: setattr(m, "_train_instruments", k),
                   after_step=m.tul_fan_after_step, first_capture=2, compiled=False)
    batches = _batches(rt, 5)
    for s in range(3):
        m._train_instruments = s == 0
        opt.write_step_scalars()
        gs.step(s, s == 0, *batches[s], context="tul")
    assert gs.stats.captures == 2
    return m, opt, gs, batches


@cuda
def test_cpu_generator_moving_after_capture_raises(monkeypatch):
    m, opt, gs, batches = _captured(monkeypatch)
    gs.step(3, False, *batches[3], context="tul")                  # a replay: fine
    torch.rand(1)                                                    # the CPU generator moves
    with pytest.raises(RuntimeError, match="CPU generator moved"):
        gs.step(4, False, *batches[4], context="tul")


@cuda
def test_rebound_parameter_raises_before_the_replay(monkeypatch):
    m, opt, gs, batches = _captured(monkeypatch)
    gs.run_eager(*batches[3])                                        # eager work: re-check due
    p = next(iter(m.parameters()))
    p.data = p.data.clone()                                          # same values, new storage
    with pytest.raises(RuntimeError, match="rebound"):
        gs.step(4, False, *batches[4], context="tul")


@cuda
def test_a_different_phase_or_layout_raises(monkeypatch):
    m, opt, gs, batches = _captured(monkeypatch)
    with pytest.raises(RuntimeError, match="captured under"):
        gs.step(3, False, *batches[3], context="other phase")
    x, y, lay = batches[3]
    with pytest.raises(RuntimeError, match="ditto_prev"):
        gs.step(3, False, x, y, dataclasses.replace(lay, ditto_prev=None), context="tul")


# ── 4. refusals (CPU) ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("over,needle", [
    ((), "capturable_optimizer must be true"),
    (("training.capturable_optimizer=true",), "graph_safe must be true"),
    (_ON + ("model.ce_compact_rows=true",), "ce_compact_rows"),
    (_ON + ("training.grad_probe_every=20",), "grad_probe_every"),
    (_ON + ("training.prune_start=100",), "prune_start=100"),
    (_ON + ("+training.step_mix={bptt: 1, db1: 1}",), "step_mix"),
])
def test_config_refusals(monkeypatch, over, needle):
    from morph.training.graph_step import graph_step_refusals
    cfg, _rt = _compose(monkeypatch, "lxtul_pointer", *over)
    why = graph_step_refusals(cfg, total_steps=5000, curriculum=False)
    assert any(needle in w for w in why), why


def test_the_winner_with_both_halves_is_accepted(monkeypatch):
    from morph.training.graph_step import graph_step_refusals
    for name in ("lxtul_pointer", "lxtul_pointer_ditto"):
        cfg, _rt = _compose(monkeypatch, name, *_ON, "training.grad_probe_every=0")
        assert graph_step_refusals(cfg, total_steps=int(cfg.training.steps),
                                   curriculum=False) == []
    cfg, _rt = _compose(monkeypatch, "lxtul_pointer", *_ON, "training.grad_probe_every=0")
    assert graph_step_refusals(cfg, total_steps=100, curriculum=True) == [
        "curriculum: the stage changes the sequence length and grad accumulation"]
