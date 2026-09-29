"""``tul.fan_all_wta_winner`` ("onewinner", 2026-09-29; restructured "onewinner-perf",
2026-09-29): share ONE WTA winner per slot across every ``tul.code_enum_k`` rollout,
instead of each rollout picking its own — ranked by the model's TRUE per-span mixture
posterior, computed by a coda pass this method shares with the model's own deployed
pass rather than paying for either an extra proxy pass or the deployed pass twice.
``tul.fan_all_wta_grad_rollouts`` (also 2026-09-29): the winner-alone grad pass's own
row count, independently.

Files: morph/model/tul.py (both fields + `__post_init__` refusals), morph/model/transformer.py
(`_tul_fan_all`'s "map" branch and its grad-pass split, `_batch_head`,
`_fan_all_winner_capture`, `_fan_all_deployed_cache`, `_enum_mix_losses`'s `s_out` param,
`accumulate_span_ce_fused`), morph/training/tul_setup.py (`KNOWN_TUL_KEYS`,
`build_tul_runtime`, the wandb manifest, the build-time banner),
morph/configs/tul_slot_spandec_strict_lxfan4_wta_fp01_mapwin{,_s2,_s3}.yaml.
Note: .agents/notes/proposed/architecture/2026-09-29-onewinner-shared-map-rollout.md

WHY. Arm (b)'s WTA table (tests/test_tul_lx_credit.py) runs M = fan_k no-grad coda passes
PER STREAM on the FULL K-fold rollout batch (K*B0 rows each) so every rollout picks its
own per-slot winner independently — 2/3 of the arm's coda cost. "map" shares ONE winner
per slot, picked from the slot's MAP rollout under the model's own EXACT mixture
posterior `S` (`_enum_mix_losses`'s `s_out["S"]` — the same tensor `enum_code_win{k}`
reads) — not a proxy. This posterior comes from the SAME coda pass the model needs
anyway (the write-all 1:1 deployed pass `_forward_tul` would otherwise build again right
after this method returns): `_tul_fan_all` builds that pass itself, WITH grad, and
caches `(xh, groups)` in `_fan_all_deployed_cache` for `_forward_tul` to reuse instead of
recomputing. The M PICKING passes then run on a B0-row batch instead of K*B0 — a
QUARTER the picking table's own rows — legal because the strict coda's prefix cell "sees
ITSELF and nothing else" (`tul_slot_spandec_strict.yaml`): swapping one cell's rollout
only ever changes TOKENS strictly after it, never another cell's own read, so a per-slot
Frankenstein-of-rollouts row is a well-defined probe — exactly what the grad pass already
builds per FAN STREAM. NET (grad_rollouts="all", the default): the arm's total coda row
count drops from `(M+2)*K*B0` to `K*B0 + M*B0 + K*B0` — 24 -> 12 row-units at this arm's
K = M = 4, exactly half (there is no separate deployed pass left to count — the shared
mixture pass IS it). `fan_all_wta_grad_rollouts="map"` drops the grad pass itself to
B0 rows too: 24 -> 9 row-units, at the cost of a real objective change (one rollout's
term per slot instead of the mean over K) — a knob, not merely cheaper.

What each test pins:
  * "per_rollout" (default) runs the UNCHANGED branch (`map_idx is None` in the capture,
    `_fan_all_ce_capture` still populated) and gives finite, deterministic loss/grads —
    the tree's existing LX-Fan/credit tests (test_tul_lxfan.py, test_tul_lx_credit.py,
    test_tul_fan_all.py, test_tul_fan.py) already exercise this exact, textually-unchanged
    code path (only wrapped in an `if`/`else`) and pass unchanged (modulo the fused-CE
    kernel swap those tests' own pins now document — see `accumulate_span_ce_fused` and
    tests/test_tul_wta_fused_ce.py).
  * "map" equals "per_rollout" when every rollout is identical (codes zeroed): the same
    trick tests/test_tul_lx_credit.py uses (`m.tul_code_enum.directions = lambda: ...`).
  * The MAP rollout's winner is shared: every one of the R blocks of `choice` is the SAME
    per-slot table, and `map_idx` equals an independently recomputed ARGMAX of the
    shared mixture pass's own posterior (`S_slots`, captured in `_fan_all_winner_capture`).
  * The assembled pick batch reads ONLY the MAP rollout's cells: a pure gather check
    against an independent reference AND an end-to-end perturbation of a NON-map
    rollout's cells that leaves the pick unchanged (with a non-degeneracy precondition,
    the `test_hard_credit...degenerate case` pattern already used in this file's sibling).
  * The picking table's own row count (a quarter) and the net row count (half, with no
    separate ranking OR deployed pass left) against `_back_region` call spies, and the
    deployed-pass cache is always consumed, never left dangling.
  * `fan_all_wta_grad_rollouts="map"` moves the grad pass to B0 rows too.
  * A real-dropout regression on a non-multiple batch (the bug the 30-step GPU trace
    caught, not any CPU test — see the test's own docstring).
  * The config composes, differs from (b) by exactly the intended keys, and reaches the
    model config.
  * Refusals.

CPU, fp32, dropout 0 (the "test fixtures run at dropout 0" convention unless a test says
otherwise), the `tests/test_tul_fan.py` / `tests/test_tul_lxfan.py` strict-geometry
fixtures.
"""
from __future__ import annotations

import torch

from morph.model.tul_layout import SlotLayout
from test_tul_fan import _batch, _tul
from test_tul_lxfan import K, _build, _Spy
from test_tul_lx_credit import M, _wta_kw


def _table(m, inp, lab, layout, seed: int = 5):
    """One train forward with BOTH capture hooks armed: the per-stream table
    (`_fan_all_ce_capture`, the ``ce``/``cells_d`` precedent — populated under
    "per_rollout" only) and the winner capture (`_fan_all_winner_capture`, this key's
    own hook, populated under both modes)."""
    m._fan_all_ce_capture = []
    m._fan_all_winner_capture = []
    torch.manual_seed(seed)
    out = m(inp, labels=lab, slot_layout=layout)
    cap_ce = m._fan_all_ce_capture[0] if m._fan_all_ce_capture else None
    cap_w = m._fan_all_winner_capture[0]
    m._fan_all_ce_capture = None
    m._fan_all_winner_capture = None
    return out, cap_ce, cap_w


# ── 1. per_rollout is the unchanged branch ──────────────────────────────────────────


def test_default_is_per_rollout_and_runs_the_unchanged_branch():
    _ids, inp, lab, layout = _batch(M)
    m = _build(**_wta_kw(model_kw={"dropout": 0.0}, fan_select_eps=0.0)).train()
    assert m.cfg.tul.fan_all_wta_winner == "per_rollout"
    _out, cap_ce, cap_w = _table(m, inp, lab, layout)
    # `map_idx is None` is the capture's OWN marker that the "map" branch never ran.
    assert cap_w["map_idx"] is None and cap_w["cells_map"] is None
    ce = cap_ce["ce"]
    ok = layout.slot_valid.repeat(K, 1) & (ce.sum(-1) != 0.0)  # every valid slot scored
    torch.testing.assert_close(cap_w["choice"][ok], ce.argmin(-1)[ok])


def test_explicit_per_rollout_matches_the_default_exactly():
    _ids, inp, lab, layout = _batch(M)
    a = _build(**_wta_kw(model_kw={"dropout": 0.1})).train()
    b = _build(**_wta_kw(model_kw={"dropout": 0.1}, fan_all_wta_winner="per_rollout")).train()
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k
    torch.manual_seed(9)
    oa = a(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(9)
    ob = b(inp, labels=lab, slot_layout=layout)
    for key, v in oa.items():
        if torch.is_tensor(v):
            assert torch.equal(v, ob[key]), key
    oa["loss"].backward()
    ob["loss"].backward()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        if pa.grad is None:
            assert pb.grad is None
        else:
            assert torch.equal(pa.grad, pb.grad), na


# ── 2. map == per_rollout when every rollout is identical ──────────────────────────


def test_map_equals_per_rollout_when_the_rollouts_coincide():
    """Zeroed codes make the K rollouts one computation (the
    `test_with_the_codes_zeroed_the_rollouts_tables_are_identical_under_dropout`
    precedent): the MAP rollout is then arbitrary among ties but every rollout's OWN
    table is identical, so the SHARED winner and the INDEPENDENT winner agree everywhere,
    and the WTA term itself is bit-identical."""
    _ids, inp, lab, layout = _batch(M)
    a = _build(**_wta_kw(fan_select_eps=0.0, model_kw={"dropout": 0.0})).train()
    b = _build(**_wta_kw(fan_select_eps=0.0, model_kw={"dropout": 0.0},
                         fan_all_wta_winner="map")).train()
    a.tul_code_enum.directions = lambda: torch.zeros(K, a.cfg.d_model)
    b.tul_code_enum.directions = lambda: torch.zeros(K, b.cfg.d_model)
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k
    oa, cap_ce_a, _ = _table(a, inp, lab, layout)
    ob, _cap_ce_b, cap_w_b = _table(b, inp, lab, layout)
    B0 = inp.shape[0]
    ce_a = cap_ce_a["ce"].view(K, B0, *cap_ce_a["ce"].shape[1:])
    for k in range(1, K):
        assert torch.equal(ce_a[k], ce_a[0]), "precondition: rollouts must coincide"
    torch.testing.assert_close(ob["loss"], oa["loss"])
    assert torch.equal(ob["fan_wta_ce"] if "fan_wta_ce" in ob else ob["loss"],
                       oa["fan_wta_ce"] if "fan_wta_ce" in oa else oa["loss"])
    torch.testing.assert_close(cap_w_b["choice"], cap_w_b["choice"][:B0].repeat(K, 1))


# ── 3. the shared winner is the MAP rollout's own table, for every rollout ─────────


def test_map_winner_is_shared_by_every_rollout_and_matches_the_free_argmin():
    """"Free" here means the SHARED mixture pass's own posterior `S` (one pass, K*B0
    rows — the model's own deployed coda call, which `_forward_tul` would otherwise
    build again right after this method returns; see
    `test_the_picking_passes_run_on_b0_rows_not_kb0`, which pins that no SEPARATE
    ranking pass exists any more) — the TRUE per-span mixture posterior
    `_enum_mix_losses` computes, not a proxy, and the ONLY extra pass "map" pays beyond
    the M picking passes."""
    _ids, inp, lab, layout = _batch(M)
    m = _build(**_wta_kw(fan_all_wta_winner="map", fan_select_eps=0.0,
                        model_kw={"dropout": 0.0})).train()
    _out, cap_ce, cap_w = _table(m, inp, lab, layout)
    assert cap_ce is None, "map mode must not populate the per-stream table hook"
    B0 = inp.shape[0]
    S_slots = cap_w["S_slots"]                                    # [K, B0, S]
    ref_map_idx = S_slots.argmax(dim=0)        # [B0, S] -- S is a posterior DIRECTION:
    # higher = more likely, so the MAP rollout is the argMAX, not argmin (the CE-argmin
    # the old proxy used was the same direction read the other way: CE = -logprob).
    assert torch.equal(cap_w["map_idx"], ref_map_idx)
    choice = cap_w["choice"]
    assert choice.shape == (K * B0, S_slots.shape[-1])
    for r in range(1, K):
        assert torch.equal(choice[r * B0:(r + 1) * B0], choice[:B0]), r
    ok0 = layout.slot_valid[:B0] & (cap_w["ce_map"].sum(-1) != 0.0)
    assert bool(ok0.any()), "degenerate fixture: no valid scored slot to check"


# ── 4. the pick batch reads ONLY the MAP rollout's cells ───────────────────────────


def test_pick_batch_draws_each_slot_from_its_own_map_rollout():
    _ids, inp, lab, layout = _batch(M)
    m = _build(**_wta_kw(fan_all_wta_winner="map", fan_select_eps=0.0,
                        model_kw={"dropout": 0.0})).train()
    spy = _Spy(m, "_tul_fan_all")
    out, cap_ce, cap_w = _table(m, inp, lab, layout)
    assert cap_ce is None
    assert len(spy.calls) == 1
    a, kw = spy.calls[0]
    cells, layout_arg = a[0], a[6]
    assert kw == {"n_rollouts": K}
    cells_d, map_idx = cells.detach(), cap_w["map_idx"]
    B0 = cells.shape[0] // K
    S = map_idx.shape[1]
    cells5 = cells_d.view(K, B0, *cells_d.shape[1:])

    # 4a. the PRODUCTION gather equals an INDEPENDENT hand-built reference.
    ref = torch.stack([torch.stack([cells5[int(map_idx[b, s]), b, s]
                                    for s in range(S)]) for b in range(B0)])
    torch.testing.assert_close(cap_w["cells_map"], ref)

    # 4b. a non-degenerate slot: at least one base row/slot has R > 1 candidate rollouts
    # that are NOT its own MAP rollout (true whenever K > 1, always here at K = 4), and
    # the fixture must actually score it.
    ok0 = layout_arg.slot_valid[:B0] & (cap_w["ce_map"].sum(-1) != 0.0)
    assert bool(ok0.any()), "degenerate fixture: no valid scored slot to perturb"
    b0, s0 = [(int(b), int(s)) for b in range(B0) for s in range(S) if bool(ok0[b, s])][0]
    r_map = int(map_idx[b0, s0])
    r_other = (r_map + 1) % K

    # 4c. perturbing a NON-map rollout's cells at (b0, s0) does not move the gathered
    # entry (the "pick" — a pure data-flow fact, no coda pass needed to see it).
    perturbed = cells_d.clone()
    perturbed[r_other * B0 + b0, s0] += 1000.0
    pert5 = perturbed.view(K, B0, *perturbed.shape[1:])
    got = pert5[int(map_idx[b0, s0]), b0, s0]
    torch.testing.assert_close(got, cap_w["cells_map"][b0, s0])

    # 4d. contrast: perturbing the MAP rollout's OWN cells at (b0, s0) DOES move it —
    # proves 4c is not vacuous (the gather is not simply blind to `cells_d` entirely).
    perturbed2 = cells_d.clone()
    perturbed2[r_map * B0 + b0, s0] += 1000.0
    pert5b = perturbed2.view(K, B0, *perturbed2.shape[1:])
    got2 = pert5b[int(map_idx[b0, s0]), b0, s0]
    assert not torch.allclose(got2, cap_w["cells_map"][b0, s0])

    # 4e. end-to-end: rerunning `_tul_fan_all` DIRECTLY on cells perturbed only at a
    # NON-map rollout of (b0, s0), everything else identical, reproduces the SAME
    # map_idx (precondition — the perturbation must not flip the argmin) and the SAME
    # cells_map / choice at (b0, s0): the picking passes never touched that rollout.
    cells2 = cells.detach().clone()
    cells2[r_other * B0 + b0, s0] += 1e-3
    m._fan_all_winner_capture = []
    with torch.no_grad():
        m._tul_fan_all(cells2, *a[1:], **kw)
    cap2 = m._fan_all_winner_capture[0]
    m._fan_all_winner_capture = None
    assert int(cap2["map_idx"][b0, s0]) == r_map, "precondition: the tiny perturbation flipped the MAP rollout"
    torch.testing.assert_close(cap2["cells_map"][b0, s0], cap_w["cells_map"][b0, s0])
    assert int(cap2["choice"][b0, s0]) == int(cap_w["choice"][b0, s0])


def test_map_picking_passes_survive_real_dropout_on_a_non_multiple_batch():
    """Regression (2026-09-29, caught by the 30-step GPU trace, NOT by any CPU test
    above — every one of them builds its model at ``dropout: 0.0``, the file's own
    "test fixtures run at dropout 0" convention, which makes `RolloutSharedDropout`
    (morph/model/rollout_dropout.py) a no-op and never exercises its
    `rows % n_rep` check at all).

    `self.coda`'s dropout is swapped to `RolloutSharedDropout(n_rep=R)` for the WHOLE
    model once R = `code_enum_k` > 1 (`MORPHTransformer.__init__`), built for the
    R*B0-row rollout-major passes every OTHER coda call in this method makes. The "map"
    picking loop calls `_back_region` on a B0-row batch instead — `_batch(M)` here gives
    B0 = 2, R = K = 4, and 2 % 4 != 0 — so without `bypass_rollout_sharing` (wired into
    `_tul_fan_all`'s "map" branch around exactly this loop) ANY dropout > 0 under "map"
    mode raises `RolloutSharedDropout`'s own `rows % self.n_rep` RuntimeError on every
    real (non-zero-dropout) run, including the one shipped in
    `tul_slot_spandec_strict_lxfan4_wta_fp01_mapwin.yaml` (`model.dropout` inherited from
    arm (b), not 0). This test is the first one in the file to build a "map" model with
    dropout > 0; it must simply not raise, and its loss/grads must be finite."""
    _ids, inp, lab, layout = _batch(M)
    m = _build(**_wta_kw(fan_all_wta_winner="map", model_kw={"dropout": 0.2})).train()
    torch.manual_seed(3)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n


def test_map_grad_pass_survives_real_dropout_on_a_non_multiple_batch():
    """The grad-pass twin of `test_map_picking_passes_survive_real_dropout_on_a_
    non_multiple_batch`: `fan_all_wta_grad_rollouts="map"` runs the responsibility pass
    on the SAME B0-row batch as the picks (B0 = 2, R = K = 4, not a multiple), so it
    needs its OWN `bypass_rollout_sharing` around its OWN `_back_region` call — a
    separate call site from the picks', not exercised by that test (whose model keeps
    `fan_all_wta_grad_rollouts` at the default "all", so ITS grad pass runs on the full
    K*B0-row — always a multiple of K — batch and never touches this code path)."""
    _ids, inp, lab, layout = _batch(M)
    m = _build(**_wta_kw(fan_all_wta_winner="map", fan_all_wta_grad_rollouts="map",
                        model_kw={"dropout": 0.2})).train()
    torch.manual_seed(5)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n


def test_the_picking_passes_run_on_b0_rows_not_kb0(monkeypatch):
    """The cost claim ("onewinner-perf", 2026-09-29): under "map" (grad_rollouts="all",
    the default) the coda pass ORDER over a FULL forward is [1 SHARED mixture pass,
    K*B0 rows] + [M picking passes, B0 rows each] + [1 grad pass, K*B0] — M+2 calls,
    same as "per_rollout"'s own [M picking passes, K*B0 rows each] + [1 grad pass,
    K*B0] + [1 deployed coda pass, K*B0] (the
    `test_wta_under_the_code_runs_one_table_pass_per_stream_then_a2s_two` precedent) —
    but with M of those M+2 calls at a QUARTER the rows: there is no SEPARATE deployed
    pass under "map" any more (`_fan_all_deployed_cache` — the mixture pass IS the
    deployed pass), so the net row count drops from `(M+2)*K*B0` to
    `K*B0 + M*B0 + K*B0` — 24 -> 12 row-units at K = M = 4, exactly half."""
    _ids, inp, lab, layout = _batch(M)
    B0 = inp.shape[0]
    per = _build(**_wta_kw()).train()
    spy_per = _Spy(per, "_back_region")
    per(inp, labels=lab, slot_layout=layout)
    rows_per = [int(a[0].shape[0]) for a, _k in spy_per.calls]
    assert rows_per == [K * B0] * (M + 2), rows_per

    mp = _build(**_wta_kw(fan_all_wta_winner="map")).train()
    spy_map = _Spy(mp, "_back_region")
    mp(inp, labels=lab, slot_layout=layout)
    rows_map = [int(a[0].shape[0]) for a, _k in spy_map.calls]
    assert rows_map == [K * B0] + [B0] * M + [K * B0], rows_map
    assert mp._fan_all_deployed_cache is None, (
        "the cache must be consumed by the deployed-pass call site, not left dangling")

    mg = _build(**_wta_kw(fan_all_wta_winner="map",
                          fan_all_wta_grad_rollouts="map")).train()
    spy_grad = _Spy(mg, "_back_region")
    mg(inp, labels=lab, slot_layout=layout)
    rows_grad = [int(a[0].shape[0]) for a, _k in spy_grad.calls]
    assert rows_grad == [K * B0] + [B0] * M + [B0], rows_grad


# ── 5. the config ────────────────────────────────────────────────────────────────


def test_mapwin_config_composes_and_reaches_the_model(monkeypatch):
    from test_tul_lxfan import _diff

    cfg, rt, diff = _diff("tul_slot_spandec_strict_lxfan4_wta_fp01_mapwin",
                          "tul_slot_spandec_strict_lxfan4_wta_fp01", monkeypatch)
    assert diff == {"tul.fan_all_wta_winner", "training.steps",
                    "training.lr_decay_steps", "wandb.name"}, sorted(diff)
    tc = rt.model_cfg
    assert tc.fan_all_wta_winner == "map"
    assert (tc.code_enum_k, tc.fan_k, tc.fan_mix, tc.fan_all_wta_lambda) == (4, 4, "all", 1.0)
    assert int(cfg.training.steps) == 3000 and int(cfg.training.lr_decay_steps) == 5000
    assert cfg.wandb.name == "lxtul-lxfan4-wta-fp01-mapwin"


def test_mapwin_seed_variants(monkeypatch):
    from test_tul_lxfan import _diff

    for seed, tag in ((2, "s2"), (3, "s3")):
        cfg, rt, diff = _diff(f"tul_slot_spandec_strict_lxfan4_wta_fp01_mapwin_{tag}",
                              "tul_slot_spandec_strict_lxfan4_wta_fp01_mapwin", monkeypatch)
        assert diff == {"training.seed", "wandb.name"}, sorted(diff)
        assert int(cfg.training.seed) == seed
        assert rt.model_cfg.fan_all_wta_winner == "map"
        assert cfg.wandb.name == f"lxtul-lxfan4-wta-fp01-mapwin-{tag}"


def test_grad_rollouts_map_reaches_the_model_through_hydra(monkeypatch):
    """`tul.fan_all_wta_grad_rollouts` has no dedicated production config (no queued
    arm asked for one) — reached here the way an ad-hoc override would reach it
    (`+tul.fan_all_wta_grad_rollouts=map` — the `+` is Hydra's own requirement for a
    key absent from every yaml in the compose chain, confirmed against the live
    `compose()` call, not assumed), same as `wandb.name` staying at its default (the
    key is not otherwise queued so it inherits the mapwin name)."""
    import transformers
    from hydra import compose, initialize_config_dir

    from morph.training import tul_setup
    from test_tul_fan import _rule
    from test_tul_strict_geometry import _CONFIG_DIR, _StubTok

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubTok)
    monkeypatch.setattr(tul_setup, "build_boundary_rule",
                        lambda cfg, cache_dir="": (_rule(), _rule().is_boundary, 0, ("\n",)))
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        cfg = compose(config_name="tul_slot_spandec_strict_lxfan4_wta_fp01_mapwin",
                      overrides=["+tul.fan_all_wta_grad_rollouts=map"])
    rt = tul_setup.build_tul_runtime(cfg)
    assert rt.model_cfg.fan_all_wta_grad_rollouts == "map"
    assert rt.model_cfg.fan_all_wta_winner == "map"          # the override's precondition
    assert "fan_all_wta_grad_rollouts" in tul_setup.KNOWN_TUL_KEYS
    assert rt.manifest["fan_all_wta_grad_rollouts"] == "map"


# ── 6. refusals ──────────────────────────────────────────────────────────────────


def test_bad_value_raises():
    import pytest
    with pytest.raises(ValueError, match="per_rollout' or 'map'"):
        _tul(**_wta_kw(fan_all_wta_winner="mcl"))


def test_map_needs_fan_mix_all():
    import pytest
    from test_tul_lxfan import _lx_kw
    with pytest.raises(ValueError, match="needs tul.fan_mix='all'"):
        _tul(**_lx_kw(fan_all_wta_winner="map"))


def test_map_needs_wta_lambda_positive():
    import pytest
    from test_tul_lxfan import _lxfan_kw
    with pytest.raises(ValueError, match="needs tul.fan_mix='all'"):
        _tul(**_lxfan_kw(M, fan_all_wta_winner="map"))     # lxfan4: wta_lambda 0.0


def test_map_needs_code_enum_k_at_least_2():
    import pytest
    from test_tul_lxfan import _fan_kw
    with pytest.raises(ValueError, match="needs tul.code_enum_k >= 2"):
        _tul(**_fan_kw(M, fan_all_wta_winner="map"))       # a2's fan: no code (K = 1)


def test_map_refuses_bcast():
    import pytest
    with pytest.raises(ValueError, match="no defined interaction with"):
        _tul(**_wta_kw(fan_all_wta_winner="map", bcast=True))


def test_grad_rollouts_bad_value_raises():
    import pytest
    with pytest.raises(ValueError, match="'all' or 'map'"):
        _tul(**_wta_kw(fan_all_wta_winner="map", fan_all_wta_grad_rollouts="mcl"))


def test_grad_rollouts_map_needs_winner_map():
    import pytest
    with pytest.raises(ValueError, match="needs tul.fan_all_wta_winner='map'"):
        _tul(**_wta_kw(fan_all_wta_grad_rollouts="map"))   # winner stays "per_rollout"
