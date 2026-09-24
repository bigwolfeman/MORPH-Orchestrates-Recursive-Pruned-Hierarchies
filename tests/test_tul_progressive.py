"""The progressive loss on the slot loop (``tul.progressive_p``).

Bansal, Schwarzschild et al. 2022, "End-to-end algorithm synthesis with recurrent
networks: logical extrapolation without overthinking". A random number of passes runs
with NO gradient and the rest with gradient, so the shared map must improve ANY state it
is handed. MORPH draws the cut PER SLOT inside one forward: with probability
``progressive_p`` a slot of realised depth ``T_i >= 2`` draws ``k_i`` uniform in
``[1, T_i - 1]`` and runs its first ``k_i`` passes detached.

One test per contract:

  1. ``progressive_p = 0`` is the forward from before the knob existed — no draw, no
     counters, no change to the random stream, and every pass of every live slot still
     carries gradient. (The identity against the PRE-CHANGE code was checked directly,
     off the test tree: this tiny model at seed 0/7 gives loss 9.1006078720092773 and
     sha256 ``32b174ef…2853f8`` over all 208 gradient tensors both at master ``c429e22``
     and at ``progressive_p: 0.0`` after the change. This file keeps the contract that
     would break if the mechanism ever leaked into the p = 0 path.)
  2. at ``progressive_p = 1`` the drawn prefix contributes NO gradient: the cotangent
     arriving at a prefix pass's output is exactly zero for that slot — and the same tap
     shows nonzero gradient at the grad passes, so the check is not vacuous.
  3. the draw stays inside ``[0, T_i - 1]`` and never touches a pad slot, which is what
     keeps a slot's LAST pass (the terminal fixed-point term, the exit state) in the
     gradient window.
  4. the gain hinge probes only GRAD passes — its penalty shapes the core weights, and at
     a detached position the training objective has been cut away.
  5. the knob is refused where a per-slot prefix has no meaning.

CPU only, tiny config, no tokenizer.
"""

from __future__ import annotations

import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402  (tests/ is on sys.path)

from morph.model.transformer import MORPHConfig, MORPHTransformer

MAX_DEPTH = 4


def _model(seed: int = 3, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul_kw = {k[4:]: v for k, v in kw.items() if k.startswith("tul_")}
    cfg_kw = {k: v for k, v in kw.items() if not k.startswith("tul_")}
    tul = _tul(tg_restrict=False, sigreg_lambda=0.0, mux_beta=1.0, mux_target="next",
               mux_detach_head=False, **tul_kw)
    base = dict(tul=tul, n_core=2, mean_depth=MAX_DEPTH, max_depth=MAX_DEPTH,
                bptt_depth=MAX_DEPTH, retention=False, dropout=0.1,
                core_fixed_point_lambda=1.0, ckpt_grad_iters=0)
    base.update(cfg_kw)
    return MORPHTransformer(_cfg(**base))


def _depth_spy(m: MORPHTransformer) -> dict:
    """Capture the realised per-slot depths `_tul_core` drew (its third return value)."""
    seen: dict = {}
    orig = m._tul_core

    def spy(*a, **k):
        res = orig(*a, **k)
        seen["depths"] = res[2].detach().clone()
        return res

    m._tul_core = spy
    return seen


def _run(m: MORPHTransformer, *, seed: int = 7):
    x, y, layout, _ = _batch()
    m.train()
    torch.manual_seed(seed)
    out = m(x, labels=y, slot_layout=layout)
    out["loss"].backward()
    grads = {n: (p.grad.detach().clone() if p.grad is not None else None)
             for n, p in m.named_parameters()}
    return out, grads, layout


def _same(ga, gb) -> bool:
    return all(((ga[k] is None) == (gb[k] is None))
               and (ga[k] is None or torch.equal(ga[k], gb[k])) for k in ga)


def _tap(m: MORPHTransformer) -> dict[int, torch.Tensor]:
    """Record the per-slot cotangent norm arriving at each pass's core-step output.

    A tensor hook on ``_apply_core_step``'s return — the ONE call the slot loop makes per
    pass — keyed by the loop's own ``iter_idx``. Fills ``{t: [B, S] float}`` during the
    backward. Run with the gain hinge OFF: ``_slot_gain_penalty`` applies the same method
    twice more at one iteration and its outputs would land in the same bucket.
    """
    banked: dict[int, torch.Tensor] = {}
    orig = MORPHTransformer._apply_core_step

    def spy(self, *a, **kw):
        h, r = orig(self, *a, **kw)
        t = int(kw["iter_idx"])
        if h.requires_grad:
            def _hook(g, t=t):
                banked[t] = g.detach().float().flatten(2).norm(dim=2)   # [B, S]
            h.register_hook(_hook)
        return h, r

    m._apply_core_step = spy.__get__(m, MORPHTransformer)
    return banked


# ── 1. p = 0 is the forward from before the knob ─────────────────────────────

def test_p_zero_draws_nothing_and_leaves_the_stream_and_the_gradients_alone():
    out_a, g_a, _ = _run(_model())
    rng_a = torch.get_rng_state()
    out_b, g_b, _ = _run(_model(tul_progressive_p=0.0))
    assert torch.equal(rng_a, torch.get_rng_state()), "p=0 consumed random numbers"
    assert torch.equal(out_a["loss"], out_b["loss"])
    assert _same(g_a, g_b), "p=0 changed a gradient"
    m = _model(tul_progressive_p=0.0)
    _run(m)
    assert m._loop_prog is None and m._loop_prog_k is None, \
        "p=0 wrote the progressive counters, so the mechanism ran"


def test_the_check_is_sensitive_p_one_changes_the_gradients():
    """Guards the test above: were the mechanism a no-op at every p, it would pass for
    the wrong reason."""
    _, g_0, _ = _run(_model(tul_progressive_p=0.0))
    _, g_1, _ = _run(_model(tul_progressive_p=1.0))
    assert not _same(g_0, g_1)


def test_p_zero_every_pass_of_every_live_slot_carries_gradient():
    m = _model(slot_gain_lambda=0.0)
    seen = _depth_spy(m)
    banked = _tap(m)
    out, _, layout = _run(m)
    assert torch.isfinite(out["loss"])
    assert banked, "the tap recorded nothing — the loop did not run"
    depths = seen["depths"]
    checked = 0
    for t, rows in banked.items():
        live = layout.slot_valid & (depths > t)
        if not bool(live.any()):
            continue
        assert float(rows[live].min()) > 0.0, \
            f"pass {t} has a zero cotangent at a live slot with progressive OFF"
        checked += 1
    assert checked >= 2, checked


# ── 2. the drawn prefix contributes no gradient ──────────────────────────────

def test_p_one_prefix_passes_receive_exactly_zero_gradient():
    m = _model(slot_gain_lambda=0.0, tul_progressive_p=1.0)
    seen = _depth_spy(m)
    banked = _tap(m)
    out, _, layout = _run(m)
    assert torch.isfinite(out["loss"])
    k, depths = m._loop_prog_k, seen["depths"]
    assert k is not None and int((k > 0).sum()) > 0, "no slot drew a prefix at p=1"
    n_zero = n_live = 0
    for t, rows in banked.items():
        pfx = layout.slot_valid & (depths > t) & (k > t)
        tail = layout.slot_valid & (depths > t) & (k <= t)
        if bool(pfx.any()):
            assert float(rows[pfx].abs().max()) == 0.0, \
                f"pass {t}: a detached prefix position received gradient"
            n_zero += int(pfx.sum())
        if bool(tail.any()):
            assert float(rows[tail].min()) > 0.0, \
                f"pass {t}: a grad pass received no gradient"
            n_live += int(tail.sum())
    assert n_zero > 0 and n_live > 0, (n_zero, n_live)


# ── 3. the draw's bounds ─────────────────────────────────────────────────────

@pytest.mark.parametrize("p", [0.3, 1.0])
def test_the_draw_stays_below_the_slot_depth_and_skips_pads(p):
    m = _model(tul_progressive_p=p)
    seen = _depth_spy(m)
    _, _, layout = _run(m)
    k, depths = m._loop_prog_k, seen["depths"]
    assert torch.all(k >= 0)
    assert torch.all(k <= (depths - 1).clamp(min=0)), \
        "a slot's LAST pass was cut — the fixed-point term would sit on a detached state"
    assert torch.all(k[~layout.slot_valid] == 0), "a pad slot drew a prefix"
    assert torch.all(k[depths < 2] == 0), "a depth-1 slot drew a prefix"
    if p == 1.0:
        assert torch.all(k[layout.slot_valid & (depths >= 2)] >= 1), \
            "p=1 left an eligible slot without a prefix"


def test_counters_report_the_draw():
    m = _model(tul_progressive_p=1.0)
    seen = _depth_spy(m)
    _, _, layout = _run(m)
    k, depths = m._loop_prog_k, seen["depths"]
    assert set(m._loop_prog) == {"prog_frac", "prog_nograd_passes", "prog_grad_depth"}
    valid = layout.slot_valid.float()
    n = float(valid.sum())
    assert float(m._loop_prog["prog_frac"]) == pytest.approx(
        float(((k > 0).float() * valid).sum()) / n, rel=1e-6)
    assert float(m._loop_prog["prog_nograd_passes"]) == pytest.approx(
        float((k.float() * valid).sum()) / n, rel=1e-6)
    assert float(m._loop_prog["prog_grad_depth"]) == pytest.approx(
        float(((depths - k).float() * valid).sum()) / n, rel=1e-6)


def test_eval_and_no_grad_forwards_never_draw():
    m = _model(tul_progressive_p=1.0)
    x, y, layout, _ = _batch()
    m.eval()
    torch.manual_seed(7)
    m(x, labels=y, slot_layout=layout)
    assert m._loop_prog is None, "an eval forward ran the progressive draw"
    m.train()
    with torch.no_grad():
        m(x, labels=y, slot_layout=layout)
    assert m._loop_prog is None, "a no-grad forward ran the progressive draw"


# ── 4. the hinge probes grad passes only ─────────────────────────────────────

def test_the_gain_hinge_never_probes_a_detached_position():
    m = _model(tul_progressive_p=1.0, slot_gain_lambda=1.0, slot_gain_target=0.0,
               slot_gain_all_iters=True)
    seen: list[tuple[int, torch.Tensor]] = []
    orig = m._slot_gain_penalty

    def spy(core_step, h_in, e_arg, inj_arg, ret_state, t, stage_cond, mask, lam, **kw):
        # **kw: the forward passes `carry=` (tul.loop_carry, 6bcbe89) on every call.
        seen.append((int(t), mask.detach().clone()))
        return orig(core_step, h_in, e_arg, inj_arg, ret_state, t, stage_cond, mask, lam,
                    **kw)

    m._slot_gain_penalty = spy
    out, _, _ = _run(m)
    assert seen, "the hinge never ran"
    k = m._loop_prog_k
    for t, mask in seen:
        assert not bool((mask & (k > t)).any()), \
            f"iteration {t}: the hinge measured a slot inside its no-grad prefix"
    assert float(out["gain_est"]) > 0.0


# ── 5. refusals ──────────────────────────────────────────────────────────────

def test_paid_loop_refuses_the_knob():
    with pytest.raises(NotImplementedError, match="SLOT-LOOP lever"):
        _tul(progressive_p=0.5, tokens_through_core=True)


def test_db_loop_and_staged_targets_refuse_the_knob():
    with pytest.raises(ValueError, match="db_loop"):
        _tul(progressive_p=0.5, db_loop=True)
    with pytest.raises(NotImplementedError, match="mux_stage_own_iters"):
        _tul(progressive_p=0.5, mux_beta=1.0, mux_stage_own_iters=2)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_out_of_range_p_raises(bad):
    with pytest.raises(ValueError, match="progressive_p"):
        _tul(progressive_p=bad)


def test_a_coreless_model_refuses_the_knob():
    with pytest.raises(ValueError, match="needs a core loop"):
        _model(n_core=0, tul_progressive_p=0.5)


def test_the_plain_path_is_untouched():
    """`slot_layout=None` must not see the knob at all (runtime-invariants §6b)."""
    x = torch.randint(0, 64, (2, 48))
    y = torch.randint(0, 64, (2, 48))
    outs, grads = [], []
    for p in (0.0, 1.0):
        m = _model(tul_progressive_p=p)
        m.train()
        torch.manual_seed(99)
        out = m(x, labels=y)
        out["loss"].backward()
        outs.append(out["loss"].detach().clone())
        grads.append(torch.cat([q.grad.flatten() for n, q in sorted(m.named_parameters())
                                if q.grad is not None and not n.startswith("tul.")]))
        assert m._loop_prog is None, "the plain path ran the slot-loop draw"
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(grads[0], grads[1])


def test_a_plain_model_without_a_tul_block_builds():
    torch.manual_seed(0)
    m = MORPHTransformer(MORPHConfig(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=64, max_seq_len=256,
        context_len=256, n_prelude=2, n_core=2, n_coda=2, mean_depth=2, max_depth=2,
        bptt_depth=1, channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=8, bigram_hash_vocab=64,
        use_kernels=False, hc_use_kernel=False, dropout=0.0, retention=False))
    assert m.tul is None and m._loop_prog is None
