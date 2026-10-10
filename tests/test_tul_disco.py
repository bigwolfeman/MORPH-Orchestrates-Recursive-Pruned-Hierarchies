"""``tul.disco`` (2026-10-10, phase-2 tracker D2a): DiscoLoop re-embedding between slot-loop
passes (Fu et al. 2026, arXiv 2607.00341). At the INPUT of pass t >= 1, every cell whose
depth continues gets ``alpha * RMSNorm(Phi(h))``, ``Phi(h) = sum_v softmax(head(h)/tau)_v W[v]``,
read off the carried cell through the model's own token head.

Files: morph/model/tul_disco.py (the bridge and the module), morph/model/tul.py (the keys,
`_check_disco`), morph/model/transformer.py (the build, `_disco_term`, the one site in
`_tul_core`, the eval fold), morph/model/embeddings.py (`attend(detach_table=)`),
morph/training/{tul_setup,optimizer,train}.py, morph/inference/tul_generate_cached.py, the
six configs. Note: .agents/notes/proposed/architecture/2026-10-10-discotul.md

What each test pins:
  * OFF builds nothing; ON draws no RNG: every base weight equals the same-seed model
    without the key, and the global RNG state after the build is the same.
  * FIXED alpha 0 is the model without the key: train loss, every base gradient, eval
    logits and eval loss ``torch.equal``, on the one-cell loop and on the latent-selected
    loop, at dropout 0.0 and 0.1, eager and in the graph_safe training form, while the
    term's code provably ran.
  * The DETACHED table: a loss on the loop's exit alone puts no gradient on vocabulary rows
    absent from the batch (the undetached soft bridge does: the control).
  * Finished slots get NO term (mask == depth > t AND valid, the added delta is exactly 0
    outside it and nonzero inside), no term at pass 0, and the exit is the raw output of
    every slot's LAST pass (the paper's "last pass ungated").
  * On the latent-selected loop the M cells entering every pass t >= 1 are bit-equal on
    the slots that get a term, so ONE read per slot equals the per-cell read; a fan WITHOUT
    the selection reads every cell.
  * Each bridge's forward value against a direct reference; the hard bridge's forward is the
    argmax row exactly and its backward is the top-k mixture's.
  * The term inside the model equals an independent reference built from the carried cell
    (stream mean -> lm_mixer -> final_norm -> tied table -> slot id masked -> top-k ->
    RMSNorm -> gate 0.5).
  * The gain hinge probes the pass at the input WITH the term.
  * The learned gate starts at alpha 0.5 and every disco parameter gets a gradient; the
    parameters are no-decay and never ternary.
  * Eval readings exist at eval only and reach train.py's `evaluate`; the eager generator
    runs; the cached generator refuses; the refusals; the configs differ from their parents
    by the disco keys alone and reach MORPHConfig and the manifest.

CPU, fp32, the tests/test_tul_fan.py / test_tul_lxfan.py / test_tul_fan_lsel.py fixtures.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_disco import disco_bridge
from test_tul_fan import V, _batch, _tiny, _tul
from test_tul_fan_lsel import _lsel
from test_tul_lxfan import _build

M = 4
DISCO = dict(disco=True, disco_topk=8)


def _arm(arm: str, dropout: float = 0.0, model_kw: dict | None = None, **kw):
    """``one``: the strict one-cell slot loop with the winner's non-fan pieces (fp01's
    hinge and fixed-point term, the cell RMSNorm, the span decoder, a pointer head).
    ``lx``: the winner's latent-selected loop (joint, router-followed, rank-only head)."""
    mk = {"dropout": dropout, **(model_kw or {})}
    if arm == "one":
        return _build(model_kw=mk, prefix_k=4, slot_cell_pass_norm="rms", spandec=True,
                      pointer_heads=2, **kw)
    return _lsel("joint", model_kw=mk, fan_lsel_train_follow="router", fan_lsel_lambda=1.0,
                 fan_lsel_head_input="detached", slot_cell_pass_norm="rms", pointer_heads=2,
                 **kw)


def _cells(arm: str) -> int:
    return 1 if arm == "one" else M


def _run(m: MORPHTransformer, prefix_k: int):
    """One seeded train step (loss, named grads), then the label-free and labelled eval
    forwards (logits, loss)."""
    _ids, inp, lab, lay = _batch(prefix_k)
    m.train()
    torch.manual_seed(99)
    o = m(inp, labels=lab, slot_layout=lay)
    o["loss"].backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.eval()
    with torch.no_grad():
        lg = m(inp, labels=None, slot_layout=lay)["logits"]
        ev = m(inp, labels=lab, slot_layout=lay)["loss"]
    return o["loss"].detach(), grads, lg, ev


def _depth_table(lay, cells: int, pattern=(1, 2, 3)) -> torch.Tensor:
    """A forced eval depth table: slot s gets ``pattern[s % len]``, constant within a slot."""
    B, S = lay.slot_valid.shape
    d = torch.tensor([pattern[s % len(pattern)] for s in range(S)]).repeat(B, 1)
    return d.repeat_interleave(cells, dim=1)


def _core(m: MORPHTransformer, inp, lay, depths: torch.Tensor):
    """The shipped front and slot loop at a forced depth table (eval)."""
    front_kw, front_reset, _ckw, _creset = m._tul_tg_kwargs(lay)
    x, x0, bg = m._tul_front(inp, lay, attn_kwargs=front_kw, ret_reset_mask=front_reset)
    return m._tul_core(x, x0, bg, lay, slot_depths=depths)


# ── 1. off builds nothing, on draws no RNG ─────────────────────────────────────────


@pytest.mark.parametrize("arm", ["one", "lx"])
def test_off_builds_nothing_and_on_draws_no_rng(arm):
    off = _arm(arm)
    rng_off = torch.get_rng_state()
    on = _arm(arm, **DISCO)
    rng_on = torch.get_rng_state()
    assert off.tul_disco is None and off.cfg.tul.disco is False
    assert not any("tul_disco" in k for k in off.state_dict())
    assert torch.equal(rng_off, rng_on), "building tul.disco moved the global RNG stream"
    so, sn = off.state_dict(), on.state_dict()
    assert set(sn) - set(so) == {"tul_disco.norm.weight", "tul_disco.gate_w",
                                 "tul_disco.gate_b"}
    for k, v in so.items():
        assert torch.equal(v, sn[k]), k


# ── 2. fixed alpha 0 is the model without the key ──────────────────────────────────


@pytest.mark.parametrize("arm", ["one", "lx"])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("graph_safe", [False, True])
def test_fixed_alpha_zero_is_bit_identical_to_the_base(arm, dropout, graph_safe):
    """``graph_safe`` True is the training form the graph-captured step runs (a fixed pass
    count, the fixed-shape fixed-point term), on CPU and uncaptured."""
    mk = {"graph_safe": graph_safe}
    base = _arm(arm, dropout, model_kw=mk)
    zero = _arm(arm, dropout, model_kw=mk, disco_gate="fixed", disco_alpha=0.0, **DISCO)
    zero._disco_capture = []
    pk = 4 if arm == "one" else M
    lb, gb, lgb, evb = _run(base, pk)
    lz, gz, lgz, evz = _run(zero, pk)
    cap = zero._disco_capture
    zero._disco_capture = None
    # the term's code ran (train AND eval forwards) on cells that continue
    assert cap and any(bool(c["mask"].any()) for c in cap)
    assert torch.equal(lb, lz)
    assert torch.equal(lgb, lgz)
    assert torch.equal(evb, evz)
    extra = set(gz) - set(gb)
    assert extra <= {"tul_disco.norm.weight"}, extra
    for n, g in gb.items():
        assert torch.equal(g, gz[n]), n
    for n in extra:
        assert float(gz[n].abs().max()) == 0.0


# ── 3. the detached table ──────────────────────────────────────────────────────────


def _loop_only_table_grads(bridge: str, detach: bool):
    """Gradient on the two tied-table halves from a loss on the loop's EXIT alone (no CE:
    the token head's CE touches every vocabulary row and would hide the difference)."""
    kw = dict(disco=True, disco_bridge=bridge, disco_detach_table=detach)
    if bridge != "soft":
        kw["disco_topk"] = 8
    m = _arm("one", **kw).eval()
    _ids, inp, _lab, lay = _batch(4)
    _xn, h, *_ = _core(m, inp, lay, _depth_table(lay, 1, (3,)))
    h.float().pow(2).sum().backward()
    hy = m.embed.hybrid
    return (hy.euc_embed.weight.grad.clone(), hy.lor_embed.space_embed.weight.grad.clone(),
            inp)


@pytest.mark.parametrize("bridge", ["soft", "topk", "hard"])
def test_detached_table_leaves_absent_vocab_rows_untouched(bridge):
    ge, gl, inp = _loop_only_table_grads(bridge, True)
    present = set(int(v) for v in inp.unique())
    absent = [v for v in range(5, V) if v not in present]
    assert len(absent) >= 3, "fixture has no absent vocabulary rows to test with"
    for g in (ge, gl):
        assert torch.equal(g[absent], torch.zeros_like(g[absent]))
    assert ge.abs().max() > 0, "no embedding gradient at all: the loop path is cut"


def test_undetached_soft_bridge_reaches_absent_rows_the_control():
    ge, _gl, inp = _loop_only_table_grads("soft", False)
    present = set(int(v) for v in inp.unique())
    absent = [v for v in range(5, V) if v not in present]
    assert ge[absent].abs().max() > 0, "the undetached table did not reach absent rows: test void"


# ── 4. which cells get the term, and the ungated last pass ─────────────────────────


@pytest.mark.parametrize("arm", ["one", "lx"])
def test_finished_slots_get_no_term_and_pass_zero_gets_none(arm):
    m = _arm(arm, **DISCO).eval()
    c = _cells(arm)
    _ids, inp, _lab, lay = _batch(4 if arm == "one" else M)
    depths = _depth_table(lay, c)
    m._disco_capture = []
    with torch.no_grad():
        _core(m, inp, lay, depths)
    cap, m._disco_capture = m._disco_capture, None
    assert [e["t"] for e in cap] == [1, 2]
    valid = lay.slot_valid.repeat_interleave(c, dim=1)
    for e in cap:
        want = (depths > e["t"]) & valid
        assert torch.equal(e["mask"], want), e["t"]
        delta = (e["post"] - e["pre"]).flatten(2).abs().amax(-1)              # [B, S*M]
        assert float(delta[~want].max()) == 0.0, "a finished / pad cell got a term"
        assert bool((delta[want] > 0).all()), "a continuing cell got no term"


def test_the_exit_is_the_raw_output_of_every_slots_last_pass():
    m = _arm("one", **DISCO).eval()
    _ids, inp, _lab, lay = _batch(4)
    depths = _depth_table(lay, 1)
    rec: dict[int, torch.Tensor] = {}

    def hook(t, h, h_new, layout):
        rec[t] = h_new.detach().clone()
        return h_new

    m._slot_pass_hook = hook
    m._disco_capture = []
    with torch.no_grad():
        _xn, h, *_ = _core(m, inp, lay, depths)
    m._slot_pass_hook = None
    cap, m._disco_capture = m._disco_capture, None
    assert any(float(e["term"].abs().max()) > 0 for e in cap)
    for b in range(depths.shape[0]):
        for s in range(depths.shape[1]):
            if bool(lay.slot_valid[b, s]):
                d = int(depths[b, s])
                assert torch.equal(h[b, s], rec[d - 1][b, s]), (b, s, d)


# ── 5. once per slot on the latent-selected loop ───────────────────────────────────


def test_lsel_cells_enter_every_pass_equal_so_one_read_per_slot_is_exact():
    m = _arm("lx", **DISCO).eval()
    _ids, inp, _lab, lay = _batch(M)
    depths = _depth_table(lay, M)
    m._disco_capture = []
    with torch.no_grad():
        _core(m, inp, lay, depths)
    cap, m._disco_capture = m._disco_capture, None
    B, S = lay.slot_valid.shape
    tc = m.cfg.tul
    for e in cap:
        pre = e["pre"].view(B, S, M, *e["pre"].shape[2:])
        ms = e["mask"].view(B, S, M)[:, :, 0]
        for i in range(1, M):
            assert torch.equal(pre[:, :, i][ms], pre[:, :, 0][ms]), (e["t"], i)
        # the per-cell read (every cell through the head on its own) equals the per-slot one
        with torch.no_grad():
            z = m._readout(e["pre"])
            lg = m.embed.attend(z, detach_table=True).index_fill(
                -1, torch.tensor([tc.slot_id]), float("-inf"))
            term, *_ = m.tul_disco(lg.reshape(B * S * M, -1), m.embed.lm_weight().detach())
            term = term.view(B, S * M, -1) * e["mask"].unsqueeze(-1).float()
        assert torch.allclose(term, e["term"].float(), atol=1e-6, rtol=1e-6), e["t"]


def test_a_fan_without_the_selection_reads_every_cell():
    """The once-per-slot shortcut is the selection's alone: on a fan whose cells differ the
    term differs per cell."""
    from test_tul_lxfan import _fan_kw
    m = _build(**_fan_kw(M, fan_all_wta_lambda=0.0, spandec=True, **DISCO)).eval()
    assert m._lsel_mode == "off"
    _ids, inp, _lab, lay = _batch(M)
    m._disco_capture = []
    with torch.no_grad():
        _core(m, inp, lay, _depth_table(lay, M, (3,)))
    cap, m._disco_capture = m._disco_capture, None
    B, S = lay.slot_valid.shape
    t1 = cap[0]["term"].view(B, S, M, -1)
    v = lay.slot_valid
    assert float((t1[:, :, 1] - t1[:, :, 0])[v].abs().max()) > 0


# ── 6. the bridges and the term against direct references ──────────────────────────


def _ref_rows(logits, table, k, tau):
    lf = logits.double() / tau
    order = lf.argsort(dim=-1, descending=True)[:, :k]
    val = lf.gather(-1, order)
    p = torch.exp(val - val.max(-1, keepdim=True).values)
    p = p / p.sum(-1, keepdim=True)
    return p, table.double()[order], order


def test_bridge_values_against_a_direct_reference():
    g = torch.Generator().manual_seed(3)
    N, Vv, C, k, tau = 7, 40, 16, 5, 0.7
    logits = torch.randn(N, Vv, generator=g) * 3
    table = torch.randn(Vv, C, generator=g)
    # soft: the explicit sum over the vocabulary
    phi, p, top1 = disco_bridge(logits, table, "soft", k, tau)
    ps = torch.softmax(logits.double() / tau, dim=-1)
    ref = (ps.unsqueeze(-1) * table.double().unsqueeze(0)).sum(1)
    assert torch.allclose(phi.double(), ref, atol=1e-5)
    assert torch.equal(top1, logits.argmax(-1))
    # topk: the top-k rows by a full sort, renormalised
    phi, p, top1 = disco_bridge(logits, table, "topk", k, tau)
    pr, rows, order = _ref_rows(logits, table, k, tau)
    assert torch.allclose(phi.double(), (pr.unsqueeze(-1) * rows).sum(1), atol=1e-5)
    assert torch.allclose(p.double(), pr, atol=1e-6)
    assert torch.equal(top1, order[:, 0])
    # hard: the argmax row EXACTLY in the forward
    phi, _p, top1 = disco_bridge(logits, table, "hard", k, tau)
    assert torch.equal(phi, table[logits.argmax(-1)].float())


def test_hard_bridge_backward_is_the_topk_mixtures():
    g = torch.Generator().manual_seed(4)
    logits = (torch.randn(6, 30, generator=g) * 2).requires_grad_(True)
    table = torch.randn(30, 8, generator=g)
    w = torch.randn(6, 8, generator=g)
    (disco_bridge(logits, table, "hard", 5, 1.0)[0] * w).sum().backward()
    g_hard = logits.grad.clone()
    logits.grad = None
    (disco_bridge(logits, table, "topk", 5, 1.0)[0] * w).sum().backward()
    assert torch.equal(g_hard, logits.grad)
    assert float(g_hard.abs().max()) > 0


def test_the_models_term_equals_an_independent_reference():
    """Stream mean -> lm_mixer -> final_norm -> tied table -> slot id at -inf -> top-k
    softmax mixture -> RMSNorm (gain 1) -> gate 0.5, written from the definitions."""
    m = _arm("one", **DISCO).eval()
    _ids, inp, _lab, lay = _batch(4)
    m._disco_capture = []
    with torch.no_grad():
        _core(m, inp, lay, _depth_table(lay, 1, (3,)))
    e = m._disco_capture[0]
    m._disco_capture = None
    with torch.no_grad():
        z = m.final_norm(m.lm_mixer(e["pre"].double().mean(dim=2).float()))
        logits = z.double() @ m.embed.lm_weight().double().T
        logits[..., m.cfg.tul.slot_id] = float("-inf")
        B, S = logits.shape[:2]
        pr, rows, _o = _ref_rows(logits.reshape(B * S, -1), m.embed.lm_weight(), 8, 1.0)
        phi = (pr.unsqueeze(-1) * rows).sum(1)
        phi_n = phi / phi.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        ref = (0.5 * phi_n).view(B, S, -1) * e["mask"].unsqueeze(-1).double()
    assert torch.allclose(e["term"].double(), ref, atol=1e-5), \
        float((e["term"].double() - ref).abs().max())


# ── 7. the gain hinge sees the term ────────────────────────────────────────────────


def test_the_gain_hinge_probes_the_pass_at_its_input_with_the_term():
    m = _arm("one", model_kw={"slot_gain_all_iters": True}, **DISCO).train()
    _ids, inp, lab, lay = _batch(4)
    seen: dict[int, torch.Tensor] = {}
    orig = m._slot_gain_penalty

    def spy(core_step, h_in, *a, **k):
        seen[int(a[3])] = h_in.detach().clone()
        return orig(core_step, h_in, *a, **k)

    m._slot_gain_penalty = spy
    m._disco_capture = []
    torch.manual_seed(5)
    m(inp, labels=lab, slot_layout=lay)
    cap, m._disco_capture = m._disco_capture, None
    m._slot_gain_penalty = orig
    assert cap, "no pass t >= 1 ran"
    for e in cap:
        assert torch.equal(seen[e["t"]], e["post"]), e["t"]
        assert not torch.equal(seen[e["t"]], e["pre"])


# ── 8. the gate, the gradients, the optimizer groups ───────────────────────────────


def test_learned_gate_starts_open_and_every_disco_parameter_trains():
    m = _arm("one", **DISCO).train()
    _ids, inp, lab, lay = _batch(4)
    with torch.no_grad():
        alpha = m.tul_disco(torch.randn(5, V), m.embed.lm_weight())[1]
    assert torch.equal(alpha, torch.full_like(alpha, 0.5))
    torch.manual_seed(99)
    m(inp, labels=lab, slot_layout=lay)["loss"].backward()
    for n, p in m.tul_disco.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), n
        assert float(p.grad.abs().max()) > 0, n


def test_disco_parameters_are_no_decay_and_never_ternary():
    from morph.model.ternary_qat import _categorize
    from morph.training.optimizer import _split_by_decay
    m = _arm("one", **DISCO)
    _d, _nd, decay_n, no_decay_n = _split_by_decay(m)
    names = [n for n, _p in m.named_parameters() if n.startswith("tul_disco")]
    assert len(names) == 3 and all(n in no_decay_n for n in names)
    for n, mod in m.named_modules():
        if n.startswith("tul_disco"):
            assert _categorize(n, mod, set()) is None, n


# ── 9. eval readings, generation, refusals ─────────────────────────────────────────


def test_eval_readings_exist_at_eval_only_and_reach_evaluate():
    from morph.training.train import evaluate
    m = _arm("one", **DISCO)
    _ids, inp, lab, lay = _batch(4)
    m.train()
    torch.manual_seed(1)
    assert not any(str(k).startswith("disco_") for k in m(inp, labels=lab, slot_layout=lay))
    m.eval()
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=lay)
    for key in ("gate_mean", "entropy", "top1_p", "distinct_frac", "term_ratio", "n"):
        assert f"disco_{key}_t1" in out, key
    assert float(out["disco_gate_mean_t1"]) == 0.5
    # the carried cell is RMS-normed (gain 1) and RMSNorm(Phi) has RMS 1: ratio = alpha
    assert abs(float(out["disco_term_ratio_t1"]) - 0.5) < 1e-3
    assert 0.0 < float(out["disco_distinct_frac_t1"]) <= 1.0
    extra: dict = {}
    evaluate(m, torch.device("cpu"), iter([(inp, lab, lay)]), n_batches=1, tul=True,
             extra=extra)
    assert "val/disco_gate_mean_t1" in extra and "val/disco_distinct_frac_t1" in extra


def test_the_eager_generator_runs_and_the_cached_one_refuses():
    from morph.inference.tul_generate import generate_tul
    from morph.inference.tul_generate_cached import _check_supported
    from test_tul_fan import _rule, _spec
    m = _arm("one", **DISCO).eval()
    m._disco_capture = []
    toks, _builder = generate_tul(m, [5, 6, 7, 10, 8, 9, 11, 12, 10, 13], _rule(), _spec(4),
                                  max_new_tokens=6, temperature=1.0, seed=0,
                                  emit_source="token")
    cap, m._disco_capture = m._disco_capture, None
    assert len(toks) == 6 and all(0 <= int(t) < V for t in toks)
    assert any(bool(e["mask"].any()) for e in cap), "the term never ran during generation"
    from test_tul_generate_cached import _model as _cached_ok
    _check_supported(_cached_ok())                       # the control: a model it serves
    with pytest.raises(NotImplementedError, match="tul.disco"):
        _check_supported(_cached_ok(**DISCO))


@pytest.mark.parametrize("kw,match", [
    (dict(disco_bridge="soft"), "set with tul.disco=false"),
    (dict(disco=True, disco_bridge="soft", disco_topk=8), "disco_topk set"),
    (dict(disco=True, disco_alpha=0.2), "disco_alpha set"),
    (dict(disco=True, disco_gate="fixed", disco_gate_bias=1.0), "disco_gate_bias set"),
    (dict(disco=True, disco_bridge="argmax"), "disco_bridge must be"),
    (dict(disco=True, disco_tau=0.0), "disco_tau must be"),
    (dict(disco=True, tokens_through_core=True), "tokens_through_core"),
    (dict(disco=True, loop_denoise=True), "loop_denoise"),
])
def test_config_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        TULConfig(slot_id=4, **kw)


def test_a_top_k_wider_than_the_vocabulary_is_refused_at_build():
    with pytest.raises(NotImplementedError, match="disco_topk"):
        MORPHTransformer(_tiny(tul=_tul(disco=True, disco_topk=V)))


@pytest.mark.parametrize("name,parent", [
    ("discotul", "tul_onecell_pointer"), ("discotul_fast", "tul_onecell_pointer_fast"),
    ("lxtul_disco", "lxtul_pointer"), ("lxtul_disco_fast", "lxtul_pointer_fast")])
def test_disco_configs_change_the_disco_keys_only_and_reach_the_model(name, parent,
                                                                       monkeypatch):
    from omegaconf import OmegaConf
    from test_slot_gain_tail import _leaves, _MISSING
    from test_tul_strict_geometry import _runtime

    from morph.training.train import build_morph_config

    cfg, rt = _runtime(name, monkeypatch)
    pcfg, prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
    assert diff == {"wandb.name", "tul.disco", "tul.disco_bridge", "tul.disco_topk",
                    "tul.disco_tau", "tul.disco_gate", "tul.disco_gate_bias",
                    "tul.disco_detach_table"}, sorted(diff)
    mc = build_morph_config(cfg, tul=rt.model_cfg)
    pmc = build_morph_config(pcfg, tul=prt.model_cfg)
    tdiff = {f.name for f in dataclasses.fields(mc.tul)
             if getattr(mc.tul, f.name) != getattr(pmc.tul, f.name)}
    assert tdiff == {"disco"}, tdiff
    assert mc.tul.disco and mc.tul.disco_bridge == "topk" and mc.tul.disco_topk == 128
    assert rt.manifest.get("disco") is True and prt.manifest.get("disco") is False


def test_the_one_cell_control_is_the_winner_without_the_fan(monkeypatch):
    from omegaconf import OmegaConf
    from test_slot_gain_tail import _leaves, _MISSING
    from test_tul_strict_geometry import _runtime

    for name, parent in (("tul_onecell_pointer", "lxtul_pointer"),
                         ("tul_onecell_pointer_fast", "lxtul_pointer_fast")):
        cfg, _rt = _runtime(name, monkeypatch)
        pcfg, _prt = _runtime(parent, monkeypatch)
        c = _leaves(OmegaConf.to_container(cfg, resolve=True))
        p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
        diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
        fan = {k for k in diff if k.startswith(("tul.fan_", "tul.slot_cell_init"))}
        rest = diff - fan - {"wandb.name"}
        allowed = set() if name == "tul_onecell_pointer" else {
            "model.tg_fused_attention", "model.slot_compact", "model.fan_twin_compile",
            "model.fan_div_fast", "model.fan_lsel_pick_bf16", "model.fan_target_online"}
        assert rest == allowed, sorted(rest)
        assert all(c.get(k, _MISSING) is _MISSING for k in fan), sorted(fan)
