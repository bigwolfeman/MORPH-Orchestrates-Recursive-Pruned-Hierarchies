"""CPU tests of the spiral cycle (`spiral.cycle`, lxtul/spiral.py): the whole coarse-to-fine descent repeated as an outer
cycle with a Poisson count, a sampled start level, level skips and two credit modes (end, deep supervision).

fp32, tiny shapes, Parcae's own blocks, like test_spiral.py. Each guard is broken once with the same detector that
guards it, to show the detector can fail. The "no cycle block is byte-identical to HEAD" test lives outside the repo
(it loads `git show HEAD:lxtul/spiral.py`).

    PYTHONPATH=.:$MORPH_ROOT python -m pytest -q lxtul/tests/test_spiral_cycle.py
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import lxtul.ce as ce_mod
import lxtul.spiral as sp
import parcae_lm
from lxtul.plain_pointer import PlainPointer
from lxtul.spiral import CyclePlan, CycleSpec, LevelSpec, SpiralConfig, SpiralParcae
from lxtul.tests.test_spiral import CHUNK, L, V, flip, toy_rule, tokens

@pytest.fixture(autouse=True)
def fp32_head_gemms(monkeypatch):
    """The chunked deep head runs its GEMMs in bf16 in training; the tests pin it to the dtype of the model so the
    references (full fp32 logits) compare to 1e-5."""
    monkeypatch.setattr(ce_mod, "_CDT", torch.float32)


LEVELS = (CHUNK(16), CHUNK(8), CHUNK(4), CHUNK(2))
NL = len(LEVELS)


def mk(cycle: CycleSpec | None, seed: int = 0, levels=LEVELS, train: bool = True) -> SpiralParcae:
    torch.manual_seed(seed)
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=32, num_attention_heads=2, num_key_value_heads=2, intermediate_size=64,
        recurrent_embedding_dimension=32, recurrent_intermediation_embedding_dimension=64,
        n_layers_in_prelude=2, n_layers_in_recurrent_block=2, n_layers_in_coda=2,
        mean_recurrence=3, mean_backprop_depth=3, vocab_size=V, block_size=L + 32)
    m = SpiralParcae(cfg, SpiralConfig(levels=levels, max_spans=24, max_depth=5, cycle=cycle), toy_rule())
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():                    # zero-init weights and zero scorers would hide wiring bugs
        for n, p in m.named_parameters():
            if p.ndim >= 2 and not p.abs().any():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
            if "pool_w" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.4)
    return m.train(train)


END = dict(mean=3, max=6, start="uniform", skip_p=0.3, credit="end")
DEEP = dict(END, credit="deep")
FIXED = dict(mean=3, max=6, start="coarsest", skip_p=0.0, credit="end")


def eager_forward(m, x, step: int = 7, **kw):
    torch.manual_seed(123)                   # the loop's state init is random: same draw for every call
    m.step = step
    return m(x, return_logits=True, **kw)["logits"]


def rich_step(m, B: int) -> int:
    """A step whose sampled plan has more than one cycle AND skips a level: the tests must not pass vacuously."""
    for step in range(200):
        m.step = step
        p = m._cycle_plan(B, None)
        lv = torch.arange(NL)
        skipped = p.active[..., None] & (lv > p.start[..., None]) & ~p.run
        if p.active.shape[0] >= 2 and bool(skipped.any()) and bool((p.start > 0).any()):
            return step
    raise AssertionError("no rich schedule found")


# ---------------------------------------------------------------- config guards
def test_cycle_spec_validates_its_values():
    for bad in (dict(mean=0), dict(max=0), dict(start="random"), dict(skip_p=1.0), dict(skip_p=-0.1),
                dict(credit="mid"), dict(bptt=0), dict(bptt=-2)):
        with pytest.raises(ValueError):
            CycleSpec(**bad)


def test_cycle_refuses_a_second_sampled_depth_source_and_other_topologies():
    cy = CycleSpec(**END)
    with pytest.raises(ValueError, match="sampled-depth"):
        SpiralConfig(levels=(CHUNK(4), LevelSpec("chunk", size=2, depth="poisson")), cycle=cy)
    with pytest.raises(ValueError, match="sampled-depth"):
        SpiralConfig(levels=(LevelSpec("span"), CHUNK(2)), cycle=cy)
    with pytest.raises(ValueError, match="anchor"):
        SpiralConfig(levels=LEVELS, topology="mesh", cycle=cy)
    with pytest.raises(ValueError, match="anchor"):
        SpiralConfig(levels=(LevelSpec("chunk", size=4, offset=0, shift=4),), topology="block", cycle=cy)


def test_deep_credit_needs_its_head_and_the_head_needs_deep_training():
    x = tokens(B=2, seed=1)
    m = mk(CycleSpec(**DEEP))
    with pytest.raises(ValueError, match="head_fn"):
        m(x, return_logits=True)
    with pytest.raises(ValueError, match="head_fn"):
        mk(CycleSpec(**END))(x, head_fn=lambda h: h)
    with pytest.raises(ValueError, match="head_fn"):
        mk(CycleSpec(**DEEP), train=False)(x, head_fn=lambda h: h)
    with torch.no_grad():                    # eval scores the last cycle through the ordinary path
        assert mk(CycleSpec(**DEEP), train=False)(x, return_logits=True)["logits"].shape == (2, L, V)


def test_a_cycle_adds_no_parameter_and_has_a_depth():
    base, cyc = mk(None), mk(CycleSpec(**END))
    assert list(base.state_dict()) == list(cyc.state_dict())
    assert not base.has_sampled_depth and cyc.has_sampled_depth
    with pytest.raises(ValueError, match="no sampled-depth"):
        base(tokens(B=1), depth=2)


def test_arm_configs_compose():
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[1] / "configs" / "arm"
    want = {"spiral_cycle_end": dict(start="uniform", skip_p=0.2, credit="end", bptt=3),
            "spiral_cycle_deep": dict(start="uniform", skip_p=0.2, credit="deep"),
            "spiral_cycle_fixedsched": dict(start="coarsest", skip_p=0.0, credit="end", bptt=3),
            "spiral_cycle_deep_fixedsched": dict(start="coarsest", skip_p=0.0, credit="deep")}
    for name, fields in want.items():
        cfg = OmegaConf.load(root / f"{name}.yaml")
        sc = SpiralConfig.from_cfg(cfg.spiral)
        assert [l.size for l in sc.levels] == [16, 8, 4, 2] and cfg.pointer_heads == 4
        assert sc.cycle == CycleSpec(mean=3, max=6, **fields), name


# ---------------------------------------------------------------- the schedule
def check_plan(p: CyclePlan, skip_p: float, nL: int = NL) -> list[str]:
    """Problems found in a sampled plan (empty = fine): the start level always runs, an inactive row runs nothing, no
    level before the start runs, the realized skip fraction of the later levels is near skip_p."""
    probs = []
    act, lv = p.active, torch.arange(nL)
    on_start = p.run.gather(2, p.start[..., None])[..., 0]
    if not bool(on_start[act].all()):
        probs.append("a start level was skipped")
    if bool(p.run[~act].any()):
        probs.append("a finished row ran a level")
    if bool((p.run & (lv < p.start[..., None])).any()):
        probs.append("a level before the start ran")
    later = act[..., None] & (lv > p.start[..., None])
    frac = float((~p.run & later).sum()) / max(1, int(later.sum()))
    if abs(frac - skip_p) > 0.02:
        probs.append(f"skip fraction {frac:.3f} != {skip_p}")
    return probs


def test_skip_sampling_hits_skip_p_and_never_skips_the_start_level():
    m = mk(CycleSpec(**dict(END, skip_p=0.3)))
    p = m._cycle_plan(4000, None)
    assert check_plan(p, 0.3) == []
    total = p.n + p.k
    assert int(total.min()) >= 1 and int(total.max()) <= 6
    assert abs(float(total.float().mean()) - 2.93) < 0.1 or abs(float(total.float().mean()) - 3.0) < 0.2    # Poisson(3), floor 1, cap 6
    cnt = torch.bincount(p.start[p.active], minlength=NL).float()
    assert float((cnt / cnt.sum() - 1 / NL).abs().max()) < 0.02                  # start is uniform over the levels
    assert int(p.active.sum(0).min()) == int(total.min())                        # a row is active for exactly its count


def test_a_start_that_can_be_skipped_is_caught_by_the_plan_check():
    m = mk(CycleSpec(**END))
    p = m._cycle_plan(4000, None)
    lv = torch.arange(NL)
    bad = CyclePlan(p.n, p.k, p.active, p.start, p.skip,
                    p.active[..., None] & (lv >= p.start[..., None]) & ~p.skip)    # the start level is skippable
    assert any("start level" in s for s in check_plan(bad, 0.3))


def test_eval_ignores_skip_p_and_start_and_depth_k_runs_exactly_k_cycles():
    wild = mk(CycleSpec(**dict(END, skip_p=0.6)), train=False)
    fixed = mk(CycleSpec(**FIXED), train=False)                  # same seed, same weights
    x = tokens(B=2, seed=5)
    with torch.no_grad():
        assert torch.equal(eager_forward(wild, x), eager_forward(fixed, x))
        assert torch.equal(eager_forward(wild, x, depth=4), eager_forward(fixed, x, depth=4))
    p = wild._cycle_plan(3, None)                                # eval default: round(mean) cycles, coarsest, no skip
    assert p.run.shape == (3, 3, NL) and bool(p.run.all()) and bool((p.start == 0).all())
    for k in (1, 2, 5, 7):
        for train in (False, True):
            m = mk(CycleSpec(**FIXED), train=train)
            calls = []
            m._core_pass = lambda *a, _r=m.__class__._core_pass, _m=m, **kw: (calls.append(1), _r(_m, *a, **kw))[1]
            with torch.no_grad():
                eager_forward(m, x, depth=k)
            assert len(calls) == NL * k                          # one pass per level per cycle, exactly k cycles
    sampled = mk(CycleSpec(**END))._cycle_plan(5, 4)             # depth= in training: k cycles in every row
    assert sampled.active.shape[0] == 4 and bool(sampled.active.all())


def test_counts_follow_parcaes_law_and_bptt_splits_the_grad_side():
    m = mk(CycleSpec(mean=4, max=6, start="coarsest", skip_p=0.0, credit="end", bptt=2))
    p = m._cycle_plan(2000, None)
    total = p.n + p.k
    assert int(total.max()) == 6 and int(total.min()) == 1
    assert bool((p.k == total.clamp(max=2)).all()) and bool((p.n == total - p.k).all())
    assert float(p.n.float().mean()) > 0.5                       # the no-grad cycles exist
    d = mk(CycleSpec(**dict(END, credit="deep", mean=4, bptt=2)))._cycle_plan(2000, None)
    assert bool((d.n == 0).all())                                # deep supervision: every cycle on the grad side


# ---------------------------------------------------------------- causality
def collect_logits(m, x, deep: bool, step: int):
    """-> (the [B, L, d] embedding tensor with retained grad, the list of logits the model scores: one per cycle for deep
    supervision, else the single final one)."""
    cap = {}

    def hook(mod, args, out):
        out.retain_grad()
        cap["e"] = out
    h = m.transformer.wte.register_forward_hook(hook)
    try:
        torch.manual_seed(123)
        m.step = step
        if deep:
            outs = []
            m(x, head_fn=lambda hh: outs.append(m.logits_from(hh)) or hh)
        else:
            outs = [m(x, return_logits=True)["logits"]]
    finally:
        h.remove()
    return cap["e"], outs


def autograd_leak(m, x, deep: bool, step: int) -> tuple[float, float]:
    """(largest gradient of a position's logit w.r.t. a LATER input embedding, largest w.r.t. an earlier or equal one)."""
    e, logits = collect_logits(m, x, deep, step)
    leak = seen = 0.0
    for lg in logits:
        for i in (3, 17, 33, 50):
            g, = torch.autograd.grad(lg[:, i].sum(), e, retain_graph=True)
            leak = max(leak, float(g[:, i + 1:].abs().max()))
            seen = max(seen, float(g[:, : i + 1].abs().max()))
    return leak, seen


def flip_leak(m, x, step: int) -> float:
    with torch.no_grad():
        base = eager_forward(m, x, step)
        return max(float((eager_forward(m, flip(x, t), step)[:, :t] - base[:, :t]).abs().max()) for t in range(1, L, 5))


@pytest.mark.parametrize("credit", ["end", "deep"])
def test_every_logit_is_causal_under_a_multi_cycle_skipping_schedule(credit):
    m, x = mk(CycleSpec(**dict(END, credit=credit))), tokens(B=2, seed=9)
    step = rich_step(m, 2)
    leak, seen = autograd_leak(m, x, credit == "deep", step)
    assert leak == 0.0 and seen > 0.0                            # the detector is live: earlier embeddings do get gradient
    if credit == "end":
        assert flip_leak(m, x, step) == 0.0
        base = eager_forward(m, x, step)
        assert float((eager_forward(m, flip(x, 20), step)[:, 20:] - base[:, 20:]).abs().max()) > 1e-4


def test_a_leaky_shift_is_caught_by_the_cycle_causality_detectors(monkeypatch):
    real = sp.chunk_geometry

    def no_shift(L_, spec, device):
        g, _, G = real(L_, spec, device)
        return g, g.clone(), G                                   # a token reads its OWN chunk
    monkeypatch.setattr(sp, "chunk_geometry", no_shift)
    m, x = mk(CycleSpec(**END)), tokens(B=2, seed=9)
    step = rich_step(m, 2)
    assert autograd_leak(m, x, False, step)[0] > 1e-6
    assert flip_leak(m, x, step) > 1e-4


# ---------------------------------------------------------------- equality with the published spiral
def test_one_coarsest_cycle_without_skips_is_the_published_spiral():
    pub = mk(None)
    one = mk(CycleSpec(mean=1, max=1, start="coarsest", skip_p=0.0, credit="end"))
    two = mk(CycleSpec(mean=2, max=2, start="coarsest", skip_p=0.0, credit="end"))
    x = tokens(B=2, seed=3)
    assert all(torch.equal(a, b) for a, b in zip(pub.state_dict().values(), one.state_dict().values()))
    for train in (False, True):
        pub.train(train), one.train(train), two.train(train)
        with torch.no_grad():
            ref = eager_forward(pub, x)
            assert torch.equal(eager_forward(one, x), ref)
            assert torch.equal(eager_forward(one, x, depth=1), ref)
            assert float((eager_forward(two, x, depth=2) - ref).abs().max()) > 1e-4      # a second cycle does change the answer
    pub.train(), one.train()                                     # gradients match too
    a, b = [(eager_forward(m, x).pow(2).mean().backward(), m)[1] for m in (pub, one)]
    assert all(torch.equal(p.grad, q.grad) for p, q in zip(a.parameters(), b.parameters()))


# ---------------------------------------------------------------- per-row gating
def hand_plan(n, k, start, skip) -> CyclePlan:
    n, k = torch.tensor(n), torch.tensor(k)
    Cn, Ck = int(n.max()), int(k.max())
    g = torch.arange(Cn + Ck)[:, None]
    active = torch.where(g < Cn, g < n[None], (g - Cn) < k[None])
    start, skip = torch.tensor(start), torch.tensor(skip, dtype=torch.bool)
    lv = torch.arange(NL)
    run = active[..., None] & (lv >= start[..., None]) & ((lv == start[..., None]) | ~skip)
    return CyclePlan(n, k, active, start, skip, run)


def test_rows_with_different_counts_starts_and_skips_equal_each_row_run_alone():
    m, x = mk(CycleSpec(**END)), tokens(B=2, seed=11)
    m.initialize_state = lambda z: torch.full_like(z, 0.01)      # a random state init would differ with the batch shape
    # row 0: 1 no-grad then 2 grad cycles; row 1: 2 grad cycles only. Different starts and skips per (cycle, row).
    start = [[0, 2], [1, 0], [3, 1]]
    skip = [[[0, 0, 1, 0], [0, 1, 0, 0]], [[0, 0, 0, 0], [0, 1, 1, 0]], [[0, 0, 0, 0], [0, 0, 1, 1]]]
    n, k = [1, 0], [2, 2]
    full = hand_plan(n, k, start, skip)
    assert not bool(full.run.all()) and not bool(full.active.all())
    m._cycle_plan = lambda B, depth: full
    with torch.no_grad():
        both = eager_forward(m, x)
        for b in range(2):
            cols = full.active[:, b]
            sub = hand_plan([n[b]], [k[b]], [[s[b]] for s in start][: int(cols.sum())],
                            [[s[b]] for s in skip][: int(cols.sum())])
            # a row alone has only its own cycles: gather them in order
            rows = cols.nonzero()[:, 0].tolist()
            sub = hand_plan([n[b]], [k[b]], [[start[g][b]] for g in rows], [[skip[g][b]] for g in rows])
            m._cycle_plan = lambda B, depth, _s=sub: _s
            alone = eager_forward(m, x[b: b + 1])
            assert torch.allclose(both[b: b + 1], alone, atol=1e-5, rtol=1e-5), b


def test_the_row_gating_check_is_sensitive_to_a_broken_where(monkeypatch):
    m, x = mk(CycleSpec(**END)), tokens(B=2, seed=11)
    m.initialize_state = lambda z: torch.full_like(z, 0.01)
    full = hand_plan([0, 0], [2, 1], [[0, 0], [0, 0]], [[[0] * 4] * 2] * 2)         # row 1 stops after one cycle
    m._cycle_plan = lambda B, depth: full
    with torch.no_grad():
        good = eager_forward(m, x)
        real_where = torch.where
        monkeypatch.setattr(torch, "where", lambda c, a, b: a if c.dtype == torch.bool and c.shape == (2, 1, 1) else real_where(c, a, b))
        assert float((eager_forward(m, x)[1] - good[1]).abs().max()) > 1e-4         # row 1 ran a cycle it should not have


# ---------------------------------------------------------------- credit: end and deep
def deep_model(seed: int = 0, **kw):
    return mk(CycleSpec(**dict(FIXED, credit="deep", **kw)), seed=seed)


def test_deep_loss_is_the_mean_of_the_per_cycle_ces_and_the_last_cycle_is_the_end_forward():
    x = tokens(B=2, seed=13)
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    m = deep_model()
    pp = PlainPointer(m, 2)
    pp.use_pointer = False                                       # plain CE: the reference is then a one-liner
    pp.train()
    torch.manual_seed(123)
    loss = pp(x, labels=y, depth=3)["loss"]
    # independent reference: the per-cycle logits straight from the model
    per = []
    torch.manual_seed(123)
    m(x, head_fn=lambda h: per.append(m.logits_from(h)) or h, depth=3)
    assert len(per) == 3
    ces = [F.cross_entropy(lg.flatten(0, 1), y.flatten(), ignore_index=-100) for lg in per]
    assert torch.allclose(loss, sum(ces) / 3, rtol=1e-5)
    assert len(pp.last_cycle_ce) == 3
    assert all(torch.allclose(a, b, rtol=1e-5) for a, b in zip(pp.last_cycle_ce, ces))
    assert float((ces[0] - ces[2]).abs()) > 1e-5         # the cycles differ, so the mean is not a one-cycle loss
    end = mk(CycleSpec(**FIXED))                                 # same weights, credit end: its logits are the last cycle's
    with torch.no_grad():
        assert torch.equal(eager_forward(end, x, depth=3), per[-1].detach())


def carried_grads(monkeypatch=None) -> list:
    """Gradient of each cycle's head loss w.r.t. cycle 0's level-3 group states. Cycle 0's own loss must reach them,
    later cycles' must not (the carried state is detached)."""
    m, x = deep_model(), tokens(B=2, seed=13)
    outs = []
    h = m.up_norm[-1].register_forward_hook(lambda mod, a, o: outs.append(o))
    grads = []

    def head(hh):
        ce = m.logits_from(hh).square().mean()
        grads.append(torch.autograd.grad(ce, outs[0], retain_graph=True, allow_unused=True)[0])
        return hh
    try:
        torch.manual_seed(123)
        m(x, head_fn=head, depth=3)
    finally:
        h.remove()
    assert len(grads) == 3
    return grads


def test_deep_supervision_detaches_the_carried_state():
    g = carried_grads()
    assert g[0] is not None and float(g[0].abs().max()) > 0      # cycle 0's CE trains cycle 0
    assert g[1] is None and g[2] is None                         # nothing from cycles 1 and 2 reaches cycle 0's states


def test_a_carry_that_keeps_the_graph_is_caught(monkeypatch):
    monkeypatch.setattr(SpiralParcae, "_carry", staticmethod(lambda u: u))
    g = carried_grads()
    assert g[1] is not None and float(g[1].abs().max()) > 0      # the detector fires: cycle 1's CE now reaches cycle 0


def test_end_credit_does_not_detach_and_both_modes_reach_every_parameter():
    x = tokens(B=2, seed=14)
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    for credit in ("end", "deep"):
        m = mk(CycleSpec(**dict(FIXED, credit=credit)))
        pp = PlainPointer(m, 2).train()
        pp.step = 3
        out = pp(x, labels=y, depth=2)
        loss = out["loss"]
        assert torch.isfinite(loss)
        assert out.get("backward_done", False) == (credit == "deep")      # deep: the backward ran inside the forward
        if credit == "end":
            loss.backward()
        missing = [n for n, p in pp.named_parameters() if p.requires_grad and (p.grad is None or not p.grad.abs().any())]
        # the first-level scorer's gradient can be exactly zero at the zero-update start: only demand the shared core
        assert not [n for n in missing if "core_block" in n or "up_norm" in n or "lm_head" in n or "wte" in n], (credit, missing)
        assert hasattr(pp, "last_cycle_ce") == (credit == "deep")


def test_deep_rows_with_fewer_cycles_average_over_their_own_cycles():
    x = tokens(B=2, seed=15)
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    m = deep_model()
    m.initialize_state = lambda z: torch.full_like(z, 0.01)
    plan = hand_plan([0, 0], [3, 1], [[0, 0]] * 3, [[[0] * 4] * 2] * 3)
    m._cycle_plan = lambda B, depth: plan
    pp = PlainPointer(m, 2)
    pp.use_pointer = False
    pp.train()
    loss = pp(x, labels=y)["loss"]
    per = []
    m(x, head_fn=lambda h: per.append(m.logits_from(h)) or h)
    nll = lambda lg: F.cross_entropy(lg.flatten(0, 1), y.flatten().clamp_min(0), reduction="none").view(y.shape) * (y >= 0)
    want = (sum(nll(lg)[0] for lg in per).sum() / 3 + nll(per[0])[1].sum()) / (y >= 0).sum()
    assert torch.allclose(loss, want, rtol=1e-5)


def test_deep_model_in_eval_scores_the_last_cycle_through_the_ordinary_path():
    x = tokens(B=2, seed=16)
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    m = deep_model()
    pp = PlainPointer(m, 2).eval()
    with torch.no_grad():
        torch.manual_seed(5)
        nll = pp.token_nll(x, y, depth=3)
        torch.manual_seed(5)
        lg = m.eval()(x, return_logits=True, depth=3)["logits"]
    ref = F.cross_entropy(lg.flatten(0, 1), y.flatten().clamp_min(0), reduction="none").view(y.shape)
    assert nll.shape == y.shape and torch.isfinite(nll).all()
    assert not hasattr(pp, "last_cycle_ce")
    assert float(ref[y >= 0].mean()) > 0 and float((nll - torch.where(y >= 0, ref, 0.0)).abs().max()) > 0   # + pointer mixture changes it


# ---------------------------------------------------------------- plain deep supervision (plain.deep_supervision)
def plain_model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=32, num_attention_heads=2, num_key_value_heads=2, intermediate_size=64,
        recurrent_embedding_dimension=32, recurrent_intermediation_embedding_dimension=64,
        n_layers_in_prelude=2, n_layers_in_recurrent_block=2, n_layers_in_coda=2,
        mean_recurrence=3, mean_backprop_depth=3, vocab_size=V, block_size=L + 32)
    m = cfg.construct_model(gradient_checkpointing=False)
    m.initialize_state = lambda z: torch.full_like(z, 0.01)      # same state for every call and batch shape
    return m.train()


def fix_passes(m, per_row):
    """Pin the per-row pass counts of the plain sampler (all on the grad side, as deep supervision treats them)."""
    m._sample_batch_depths = lambda B, seq_len=None: (torch.zeros(B, dtype=torch.long), torch.tensor(per_row))


def plain_xy(seed=21):
    x = tokens(B=2, seed=seed)
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    return x, y


def test_plain_deep_loss_is_the_mean_of_the_per_pass_ces_and_the_last_pass_is_the_ordinary_forward():
    from lxtul.plain_deep import deep_forward, head_logits
    x, y = plain_xy()
    m = plain_model()
    fix_passes(m, [3, 3])
    pp = PlainPointer(m, 2)
    pp.use_pointer = False
    pp.plain_deep_passes = 8
    loss = pp(x, labels=y)["loss"]
    per = []
    deep_forward(m, x, lambda h: per.append(head_logits(m, h)) or h, 8)
    assert len(per) == 3 and len(pp.last_cycle_ce) == 3
    ces = [F.cross_entropy(lg.flatten(0, 1), y.flatten(), ignore_index=-100) for lg in per]
    assert torch.allclose(loss, sum(ces) / 3, rtol=1e-5)
    assert all(torch.allclose(a, b, rtol=1e-5) for a, b in zip(pp.last_cycle_ce, ces))
    assert float((ces[0] - ces[2]).abs()) > 1e-5
    with torch.no_grad():                                        # the last pass is the plain model's 3-pass forward
        ref = m.eval()(x, return_logits=True, num_steps_pair=torch.tensor([0, 3]))["logits"]
    assert torch.allclose(per[-1].detach(), ref, atol=1e-5, rtol=1e-5)


def test_plain_deep_rows_with_fewer_passes_average_over_their_own_passes():
    from lxtul.plain_deep import deep_forward, head_logits
    x, y = plain_xy(22)
    m = plain_model()
    fix_passes(m, [3, 1])
    pp = PlainPointer(m, 2)
    pp.use_pointer = False
    pp.plain_deep_passes = 8
    loss = pp(x, labels=y)["loss"]
    per = []
    deep_forward(m, x, lambda h: per.append(head_logits(m, h)) or h, 8)
    nll = lambda lg: F.cross_entropy(lg.flatten(0, 1), y.flatten().clamp_min(0), reduction="none").view(y.shape) * (y >= 0)
    want = (sum(nll(lg)[0] for lg in per).sum() / 3 + nll(per[0])[1].sum()) / (y >= 0).sum()
    assert torch.allclose(loss, want, rtol=1e-5)
    fix_passes(m, [9, 2])                                        # the cap: 9 draws are cut to deep_max_passes
    pp.plain_deep_passes = 4
    assert len(deep_forward(m, x, lambda h: h.sum((1, 2)), 4)["cycle_nll"]) == 4


def plain_carried_grads() -> list:
    from lxtul.plain_deep import deep_forward, head_logits
    x, _ = plain_xy()
    m = plain_model()
    fix_passes(m, [3, 3])
    outs = []
    h = m.transformer.core_block[-1].register_forward_hook(lambda mod, a, o: outs.append(o))
    grads = []

    def head(hh):
        ce = head_logits(m, hh).square().mean()
        grads.append(torch.autograd.grad(ce, outs[0], retain_graph=True, allow_unused=True)[0])
        return hh.sum((1, 2))
    try:
        deep_forward(m, x, head, 8)
    finally:
        h.remove()
    assert len(grads) == 3
    return grads


def test_plain_deep_detaches_the_carried_core_state():
    g = plain_carried_grads()
    assert g[0] is not None and float(g[0].abs().max()) > 0
    assert g[1] is None and g[2] is None


def test_a_plain_carry_that_keeps_the_graph_is_caught(monkeypatch):
    import lxtul.plain_deep as pdp
    monkeypatch.setattr(pdp, "carry", lambda x: x)
    g = plain_carried_grads()
    assert g[1] is not None and float(g[1].abs().max()) > 0


def test_plain_deep_off_and_eval_leave_the_ordinary_loss_untouched():
    x, y = plain_xy()
    m = plain_model()
    fix_passes(m, [3, 3])
    pp = PlainPointer(m, 2)
    assert pp.plain_deep_passes == 0
    base = pp(x, labels=y, num_steps_pair=torch.tensor([0, 3]))["loss"]
    pp.plain_deep_passes = 8
    with torch.no_grad():
        ev = PlainPointer.eval(pp)                               # eval mode: the ordinary forward even with the key on
        same = pp(x, labels=y, num_steps_pair=torch.tensor([0, 3]))["loss"]
    assert not hasattr(pp, "last_cycle_ce")
    pp.train()
    assert torch.allclose(base, same, rtol=1e-4)                 # train-mode dropout is 0 here


def test_plain_deep_build_wiring(monkeypatch):
    from lxtul.tests.test_spiral import _built
    m, _, _ = _built("plain_pointer", monkeypatch, ["plain.deep_supervision=true"])
    assert m.plain_deep_passes == 8
    off, _, _ = _built("plain_pointer", monkeypatch)
    assert off.plain_deep_passes == 0
    with pytest.raises(ValueError, match="pointer"):
        _built("plain", monkeypatch, ["plain.deep_supervision=true"])
    with pytest.raises(ValueError, match="sync_free"):
        _built("plain_pointer", monkeypatch, ["plain.deep_supervision=true", "plain.sync_free=true"])


@pytest.mark.parametrize("arm", ["spiral_cycle_end", "spiral_cycle_deep", "spiral_cycle_fixedsched",
                                 "spiral_cycle_deep_fixedsched"])
def test_cycle_arms_build_and_take_a_train_step(arm, monkeypatch):
    from lxtul.tests.test_spiral import SEQ, _built
    m, _, _ = _built(arm, monkeypatch)
    assert m.inner.spiral.cycle is not None
    x = tokens(B=2, seed=30, n=SEQ)
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    m.train()
    m.step = 5
    loss = m(x, labels=y)["loss"]
    assert torch.isfinite(loss)
    if m.inner.spiral.cycle.credit == "end":
        loss.backward()
    assert all(p.grad is not None for n, p in m.named_parameters() if "core_block" in n)
    assert (m.inner.spiral.cycle.credit == "deep") == hasattr(m, "last_cycle_ce")


# ---------------------------------------------------------------- per-cycle backward and the chunked head
@pytest.fixture
def fp64_gemms(monkeypatch):
    monkeypatch.setattr(ce_mod, "_CDT", torch.float64)
    monkeypatch.setattr(ce_mod, "_ACC", torch.float64)


def _spiral_deep_case():
    """A double deep spiral wrapped in the pointer head, with a hand plan: unequal cycle counts, shifted starts, skips."""
    m = mk(CycleSpec(**DEEP)).double()
    m.initialize_state = lambda z: torch.full_like(z, 0.01)
    plan = hand_plan([0, 0], [3, 2], [[0, 2], [1, 0], [3, 1]], [[[0, 0, 1, 0], [0, 1, 0, 0]],
                                                              [[0, 0, 0, 0], [0, 1, 1, 0]], [[0, 0, 0, 0], [0, 0, 1, 1]]])
    m._cycle_plan = lambda B, depth: plan
    return m, tokens(B=2, seed=40)


def _plain_deep_case():
    m = plain_model().double()
    fix_passes(m, [3, 1])
    return m, tokens(B=2, seed=41)


def _grads(pp):
    return {n: (None if p.grad is None else p.grad.clone()) for n, p in pp.named_parameters()}


def _same_grads(a, b, rtol: float = 1e-6, atol: float = 1e-9) -> list:
    """Parameters whose gradients differ (None vs tensor, or not allclose)."""
    bad = []
    for n in a:
        if (a[n] is None) != (b[n] is None) or (a[n] is not None and not torch.allclose(a[n], b[n], rtol=rtol, atol=atol)):
            bad.append(n)
    return bad


def _single_and_per_cycle(case):
    m, x = case()
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    pp = PlainPointer(m, 2).double().train()
    if case is _plain_deep_case:
        pp.plain_deep_passes = 8
    n = (y >= 0).sum()
    pp.zero_grad()
    torch.manual_seed(5)
    (pp._nll(x, y).sum() / n).backward()                         # one backward through every cycle's graph
    ref = _grads(pp)
    pp.zero_grad()
    torch.manual_seed(5)
    out = pp(x, labels=y)                                        # a backward after every cycle
    return pp, ref, _grads(pp), out, n


@pytest.mark.parametrize("case", [_spiral_deep_case, _plain_deep_case])
def test_per_cycle_backward_gives_the_gradient_of_one_backward_through_the_sum(fp64_gemms, case):
    pp, ref, got, out, n = _single_and_per_cycle(case)
    assert out["backward_done"] and not out["loss"].requires_grad
    assert _same_grads(ref, got) == []
    assert float(ref[next(k for k in ref if "wte" in k)].abs().max()) > 0         # the prelude and embedding really got gradient
    assert any(float(g.abs().max()) > 0 for k, g in ref.items() if "prelude" in k and g is not None)


@pytest.mark.parametrize("case", [_spiral_deep_case, _plain_deep_case])
def test_a_skipped_prelude_backward_is_caught_by_the_gradient_comparison(fp64_gemms, monkeypatch, case):
    real = torch.autograd.backward
    monkeypatch.setattr(torch.autograd, "backward", lambda t, *a, **kw: real(t, *a, **kw) if isinstance(t, torch.Tensor) else None)
    pp, ref, got, out, n = _single_and_per_cycle(case)           # the final h0 -> prelude backward never runs
    assert _same_grads(ref, got) != []


@pytest.mark.parametrize("case", [_spiral_deep_case, _plain_deep_case])
def test_chunked_deep_head_equals_the_full_fp32_logits_head(fp64_gemms, case):
    import torch.nn.functional as F
    from lxtul.plain_deep import deep_forward, head_logits, head_nll
    m, x = case()
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    pp = PlainPointer(m, 2).double().train()
    mix = lambda h, nll: pp._mix(h, nll, y)
    heads = {"chunked": lambda h: mix(h, head_nll(m, h, y)),
             "full": lambda h: mix(h, F.cross_entropy(head_logits(m, h).flatten(0, 1), y.flatten().clamp_min(0),
                                                      reduction="none").view(y.shape))}
    res = []
    for head in heads.values():
        pp.zero_grad()
        torch.manual_seed(5)
        out = deep_forward(m, x, head, 8) if case is _plain_deep_case else m(x, head_fn=head)
        nll = sum(w[:, None] * a for w, a in zip(out["cycle_weight"], out["cycle_nll"]))
        (nll.sum() / (y >= 0).sum()).backward()
        res.append((nll.detach(), _grads(pp)))
    assert torch.allclose(res[0][0], res[1][0], rtol=1e-5, atol=1e-8) and float(res[0][0].abs().max()) > 0
    scale = max(float(g.abs().max()) for g in res[1][1].values() if g is not None)
    # head_logits casts to fp32 (`.float()`), so the reference is good to ~1e-7 of the largest gradient
    assert _same_grads(res[0][1], res[1][1], rtol=1e-4, atol=1e-6 * scale) == []
    wrong = {k: (None if g is None else g * 1.01) for k, g in res[0][1].items()}          # the tolerance can fail
    assert _same_grads(wrong, res[1][1], rtol=1e-4, atol=1e-6 * scale) != []


@pytest.mark.parametrize("case", [_spiral_deep_case, _plain_deep_case])
def test_the_per_cycle_backward_frees_each_cycles_graph_and_the_single_backward_does_not(fp64_gemms, case):
    from lxtul.plain_deep import deep_forward, head_nll
    m, x = case()
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    n = (y >= 0).sum()
    result = {}
    for mode in ("per_cycle", "single"):
        seen = []

        def head(h):
            nll = head_nll(m, h, y)
            seen.append(nll)
            return nll
        m.zero_grad()
        run = (lambda **kw: deep_forward(m, x, head, 8, **kw)) if case is _plain_deep_case else (lambda **kw: m(x, head_fn=head, **kw))
        run(**({"backward_scale": n} if mode == "per_cycle" else {}))
        try:
            seen[0].sum().backward()                             # backward through cycle 0's graph a second time
            result[mode] = "alive"
        except RuntimeError:
            result[mode] = "freed"
    assert result == {"per_cycle": "freed", "single": "alive"}


def test_backward_scale_needs_a_head_and_grad():
    m, x = _spiral_deep_case()
    with pytest.raises(ValueError, match="backward_scale"):
        m(x, backward_scale=torch.tensor(1.0))
    with torch.no_grad(), pytest.raises(ValueError, match="backward_scale"):
        m(x, head_fn=lambda h: h.sum((1, 2)), backward_scale=torch.tensor(1.0))
