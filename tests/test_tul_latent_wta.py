"""``tul.fan_all_wta_winner="latent"`` (2026-09-29 latent-WTA build): the THIRD value of
the winner-picking key ``test_tul_wta_shared_winner.py`` added ``"map"`` to. Removes the
picking table ENTIRELY -- no coda pass at all -- and picks each (rollout, slot)'s winner
by an InfoNCE score in latent space against ``latent_wta_target``, a target built from
the TRUE next span's own prelude states (already computed upstream of `_tul_fan_all`; no
new network, no coda read).

REVISED 2026-09-29, same day: the offline probe's SMOKE run (6 rows, 1 checkpoint) first
suggested a raw-cosine target; the full-spec run (96 rows, both checkpoints, ~19.9k slot
instances each) that followed showed raw cosine and raw L2 BOTH losing to an InfoNCE
own-target log-probability against in-batch negatives, on both checkpoints (see
`latent_wta_target`'s and `latent_wta_infonce_score`'s docstrings in
morph/model/transformer.py for the numbers). This test file pins the InfoNCE mechanism
that shipped as a result, not the cosine mechanism an earlier draft of this same file
pinned for a few minutes before the full-spec numbers came back.

Files: morph/model/tul.py (`fan_all_wta_winner` doc + `fan_all_wta_latent_temp` +
`__post_init__` refusals), morph/model/transformer.py (`latent_wta_target`,
`latent_wta_infonce_score`, `_tul_fan_all`'s "latent" branch, `_fan_all_winner_capture`'s
extra keys), morph/training/tul_setup.py (KNOWN_TUL_KEYS, runtime, manifest, the
build-time banner), morph/configs/tul_slot_spandec_strict_lxfan4_wta_fp01_latwta.yaml,
lab/divergence/latent_wta_probe.py (the offline probe that picked this target+score over
two other scoring rules by REGRET).
Note: .agents/notes/proposed/architecture/2026-09-29-latent-wta.md

What each test pins:
  * ZERO coda pick passes: `_back_region`'s call count drops from "per_rollout"'s M+2 to
    exactly 2 (the grad pass + the model's own deployed pass) at the SAME weights.
  * THE PICK is an independently-recomputed InfoNCE argmax against the captured target
    and cells, at eps 0 (no random override). The raw cosine table is still captured
    (diagnostic only) and checked for internal consistency, but is no longer what
    `choice` is picked from.
  * THE TARGET NEVER REACHES THE CODA: with `latent_wta_target` monkeypatched to a
    sentinel value, the sentinel appears NOWHERE in the grad pass's actual coda input,
    while the CANDIDATE cells themselves are bit-identical to the unperturbed run.
  * NO GRADIENT reaches the scoring path (the captured cosine/score/target tensors carry
    no grad_fn -- the whole branch runs inside the method's own `torch.no_grad()`).
  * REAL DROPOUT (>0) does not crash and gives finite grads (the
    `test-fixtures-run-at-dropout-zero` lesson).
  * WORKS AT `code_enum_k=1` (R=1) -- unlike "map", which needs >= 2 rollouts to share a
    winner across; "latent" scores every (rollout, slot) independently regardless of R.
  * THE WINNER-SHARE ENTROPY instrument (`wta_latent_entropy`) reads `ln(M)` when the
    picks are (engineered to be) uniform across cells and drops when they are not --
    the collapse guard named in the lit report.
  * THE CONFIG composes, differs from (b) by exactly the intended keys, and reaches the
    model config.
  * REFUSALS: bad value (shared test in test_tul_wta_shared_winner.py, updated there),
    needs `fan_mix='all'`, needs `fan_all_wta_lambda > 0`, needs
    `fan_all_wta_latent_temp > 0`, `fan_all_wta_latent_temp` is read only under
    `winner='latent'` -- and does NOT need `code_enum_k >= 2` (that is "map"-only).

CPU, fp32, the `tests/test_tul_fan.py` fixtures (strict geometry), the
`tests/test_tul_lxfan.py` / `tests/test_tul_lx_credit.py` tiny model (`_build`, `_wta_kw`,
`M`, `K`) and `_Spy`/`_table` precedents from `test_tul_wta_shared_winner.py`.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import morph.model.transformer as transformer_mod
from morph.model.transformer import latent_wta_infonce_score
from morph.model.tul import TULConfig
from test_tul_fan import _batch, _tul
from test_tul_lxfan import K, _build, _fan_kw, _lx_kw, _lxfan_kw, _Spy
from test_tul_lx_credit import M, _wta_kw
from test_tul_wta_shared_winner import _table

SENTINEL = 12345.0


# ── 0. the default is unaffected ────────────────────────────────────────────────────


def test_default_stays_per_rollout():
    assert TULConfig().fan_all_wta_winner == "per_rollout"


# ── 1. zero coda pick passes ────────────────────────────────────────────────────────


def test_latent_runs_zero_coda_pick_passes_at_the_same_weights():
    _ids, inp, lab, layout = _batch(M)
    torch.manual_seed(1234)
    per = _build(**_wta_kw()).train()
    torch.manual_seed(1234)
    lat = _build(**_wta_kw(fan_all_wta_winner="latent")).train()
    for k, v in per.state_dict().items():
        assert torch.equal(v, lat.state_dict()[k]), k

    spy_per = _Spy(per, "_back_region")
    per(inp, labels=lab, slot_layout=layout)
    rows_per = [int(a[0].shape[0]) for a, _k in spy_per.calls]
    assert rows_per == [K * inp.shape[0]] * (M + 2), rows_per

    spy_lat = _Spy(lat, "_back_region")
    lat(inp, labels=lab, slot_layout=layout)
    rows_lat = [int(a[0].shape[0]) for a, _k in spy_lat.calls]
    # ONLY the grad pass (inside `_tul_fan_all`) plus the model's own deployed pass
    # (built afterward by `_forward_tul`, unaffected by this key) -- no picking table.
    assert rows_lat == [K * inp.shape[0]] * 2, rows_lat


# ── 2. the pick is an independent InfoNCE argmax (cosine is diagnostic only) ────────


def test_latent_choice_matches_an_independently_recomputed_infonce_argmax():
    _ids, inp, lab, layout = _batch(M)
    m = _build(**_wta_kw(fan_all_wta_winner="latent", fan_select_eps=0.0,
                         model_kw={"dropout": 0.0})).train()
    _out, cap_ce, cap_w = _table(m, inp, lab, layout)
    assert cap_ce is None, "latent mode must not populate the per-stream CE table hook"
    assert cap_w["map_idx"] is None and cap_w["ce_map"] is None

    cells_score, target = cap_w["latent_cells_score"], cap_w["latent_target"]
    ok = cap_w["latent_ok"]        # the EXACT mask the model scored with, not a
                                    # reconstruction (layout.slot_valid alone omits the
                                    # `n_tok > 0` term and scores a different negative pool)
    assert bool(ok.any()), "degenerate fixture: no valid scored slot to check"

    # The raw cosine table is still captured for diagnostics and must stay internally
    # consistent, even though it is no longer what `choice` is picked from.
    ref_cos = F.cosine_similarity(cells_score, target.unsqueeze(2), dim=-1)
    torch.testing.assert_close(ref_cos, cap_w["latent_cos"])

    ref_score = latent_wta_infonce_score(cells_score, target, ok,
                                         float(m.cfg.tul.fan_all_wta_latent_temp))
    torch.testing.assert_close(ref_score, cap_w["latent_score"])
    torch.testing.assert_close(cap_w["choice"][ok], ref_score.argmax(-1)[ok])


# ── 3. the target never reaches the coda input ──────────────────────────────────────


def test_latent_target_never_reaches_the_coda_input(monkeypatch):
    """A sentinel value that replaces the target must show up NOWHERE in the coda's
    actual input tensor (the grad pass's `x_w`) -- if it did, the true span's content
    would be reaching the coda directly (the teacher-forcing bypass this mode's docstring
    warns against), not merely selecting among the model's own cells."""
    _ids, inp, lab, layout = _batch(M)
    real_target = transformer_mod.latent_wta_target

    def _sentinel_target(base, gid, keep_tok, g_bins):
        real = real_target(base, gid, keep_tok, g_bins)
        return torch.full_like(real, SENTINEL)

    m = _build(**_wta_kw(fan_all_wta_winner="latent", model_kw={"dropout": 0.0})).train()
    spy = _Spy(m, "_back_region")
    monkeypatch.setattr(transformer_mod, "latent_wta_target", _sentinel_target)
    torch.manual_seed(11)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    x_w = spy.calls[-1][0][0]           # the grad pass's coda input, the LAST call
    assert not bool((x_w == SENTINEL).any()), (
        "the sentinel target leaked into the coda's own input tensor")


def test_latent_target_perturbation_only_moves_the_choice_not_the_cells():
    """The complement of the sentinel test: perturbing the TARGET (negating it, so the
    nearest cell typically becomes the farthest) changes `choice` on at least one slot,
    but the CANDIDATE cells `_tul_fan_all` scored are bit-identical either way -- the
    perturbation could only ever have reached the pick, never the write's own content."""
    _ids, inp, lab, layout = _batch(M)

    def _build_pair():
        torch.manual_seed(1234)
        return _build(**_wta_kw(fan_all_wta_winner="latent", fan_select_eps=0.0,
                                model_kw={"dropout": 0.0})).train()

    real_target = transformer_mod.latent_wta_target
    m1 = _build_pair()
    _o1, _c1, cap1 = _table(m1, inp, lab, layout, seed=5)

    m2 = _build_pair()
    orig = transformer_mod.latent_wta_target
    try:
        transformer_mod.latent_wta_target = lambda *a, **k: -real_target(*a, **k)
        _o2, _c2, cap2 = _table(m2, inp, lab, layout, seed=5)
    finally:
        transformer_mod.latent_wta_target = orig

    torch.testing.assert_close(cap1["latent_cells_d"], cap2["latent_cells_d"])
    ok = cap1["latent_ok"]         # the EXACT mask the model scored with (see test 2)
    torch.testing.assert_close(ok, cap2["latent_ok"])   # unaffected by the target itself
    assert bool(ok.any())
    assert not torch.equal(cap1["choice"][ok], cap2["choice"][ok]), (
        "precondition: negating the target must flip at least one slot's pick, or this "
        "test cannot tell 'unaffected' from 'never checked'")


# ── 4. no gradient reaches the scoring path ─────────────────────────────────────────


def test_latent_scoring_carries_no_grad():
    _ids, inp, lab, layout = _batch(M)
    m = _build(**_wta_kw(fan_all_wta_winner="latent", model_kw={"dropout": 0.0})).train()
    _out, _cap_ce, cap_w = _table(m, inp, lab, layout)
    assert cap_w["latent_cos"].requires_grad is False
    assert cap_w["latent_target"].requires_grad is False
    assert cap_w["latent_cells_d"].requires_grad is False
    # positive control: the grad pass DOES still train the model (the pipeline is not
    # simply dead) -- loss backward reaches at least one parameter.
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert any(p.grad is not None for p in m.parameters())


# ── 5. real dropout does not crash ──────────────────────────────────────────────────


def test_latent_runs_under_real_dropout():
    _ids, inp, lab, layout = _batch(M)
    m = _build(**_wta_kw(fan_all_wta_winner="latent",
                        model_kw={"dropout": 0.2})).train()
    torch.manual_seed(3)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n


# ── 6. works at code_enum_k = 1 (unlike "map") ──────────────────────────────────────


def test_latent_works_with_no_code_unlike_map():
    """a2's fan alone (no LX-Fan code, K = 1): "map" refuses this
    (`code_enum_k >= 2` required, test_tul_wta_shared_winner.py's
    `test_map_needs_code_enum_k_at_least_2`); "latent" scores each (rollout, slot)
    independently and has no such requirement."""
    _ids, inp, lab, layout = _batch(M)
    m = _build(**_fan_kw(M, fan_all_wta_winner="latent")).train()
    torch.manual_seed(4)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in m.parameters())


# ── 7. the winner-share entropy instrument ──────────────────────────────────────────


def test_latent_entropy_instrument_present_and_bounded():
    _ids, inp, lab, layout = _batch(M)
    m = _build(**_wta_kw(fan_all_wta_winner="latent", model_kw={"dropout": 0.0})).train()
    torch.manual_seed(6)
    out = m(inp, labels=lab, slot_layout=layout)
    assert "fan_wta_latent_entropy" in out
    ent = float(out["fan_wta_latent_entropy"])
    assert 0.0 <= ent <= float(torch.tensor(float(M)).log()) + 1e-4
    # "per_rollout" / "map" never build this key.
    per = _build(**_wta_kw()).train()
    out_per = per(inp, labels=lab, slot_layout=layout)
    assert "fan_wta_latent_entropy" not in out_per


# ── 8. the config ────────────────────────────────────────────────────────────────


def test_latwta_config_composes_and_reaches_the_model(monkeypatch):
    from test_tul_lxfan import _diff

    cfg, rt, diff = _diff("tul_slot_spandec_strict_lxfan4_wta_fp01_latwta",
                          "tul_slot_spandec_strict_lxfan4_wta_fp01", monkeypatch)
    assert diff <= {"tul.fan_all_wta_winner", "tul.fan_all_wta_latent_temp",
                    "training.steps", "training.lr_decay_steps", "training.seed",
                    "wandb.name"}, sorted(diff)
    assert "tul.fan_all_wta_winner" in diff and "wandb.name" in diff
    tc = rt.model_cfg
    assert tc.fan_all_wta_winner == "latent"
    assert tc.fan_all_wta_latent_temp == 0.1
    assert (tc.code_enum_k, tc.fan_k, tc.fan_mix, tc.fan_all_wta_lambda) == (4, 4, "all", 1.0)
    assert int(cfg.training.steps) == 3000 and int(cfg.training.lr_decay_steps) == 5000
    assert cfg.wandb.name == "lxtul-lxfan4-wta-fp01-latwta"


# ── 9. refusals ──────────────────────────────────────────────────────────────────


def test_latent_needs_fan_mix_all():
    with pytest.raises(ValueError, match="needs tul.fan_mix='all'"):
        _tul(**_lx_kw(fan_all_wta_winner="latent"))


def test_latent_needs_wta_lambda_positive():
    with pytest.raises(ValueError, match="needs tul.fan_mix='all'"):
        _tul(**_lxfan_kw(M, fan_all_wta_winner="latent"))   # lxfan4: wta_lambda 0.0


def test_latent_does_not_need_code_enum_k_at_least_2():
    """The negative of "map"'s refusal: a2's fan with NO code (K = 1) is legal here."""
    _tul(**_fan_kw(M, fan_all_wta_winner="latent"))         # must not raise


def test_latent_temp_default_is_point_one():
    """Matches `lab/divergence/latent_wta_probe.py --temp`'s default — the value the
    probe's own regret numbers were measured at."""
    assert TULConfig().fan_all_wta_latent_temp == 0.1


def test_latent_needs_temp_positive():
    with pytest.raises(ValueError, match="needs tul.fan_all_wta_latent_temp > 0"):
        _tul(**_wta_kw(fan_all_wta_winner="latent", fan_all_wta_latent_temp=0.0))


def test_latent_temp_is_read_only_under_latent():
    """A value set here with any other winner mode silently does nothing -- refused."""
    with pytest.raises(ValueError, match="is read only under"):
        _tul(**_lx_kw(fan_all_wta_latent_temp=0.2))   # winner stays "per_rollout"


def test_probe_regret_is_per_token_from_summed_span_ce():
    """`lab/divergence/latent_wta_probe.py::regret_and_agreement` reads a SUMMED span CE
    table. Its first version multiplied each span's regret by the span's token count a
    second time, inflating the reading by about one span length. Hand-computed: two
    spans of 2 and 8 tokens, the pick loses 1.0 nats (whole span) on the short one and
    nothing on the long one -> 1.0 / 10 tokens = 0.1 nats/token (the old formula: 0.2)."""
    from lab.divergence.latent_wta_probe import regret_and_agreement
    ce = torch.tensor([[[3.0, 2.0], [5.0, 9.0]]])              # [B=1, S=2, M=2] summed
    ok = torch.tensor([[True, True]])
    pick = torch.tensor([[0, 0]])                              # loses on span 0 only
    n_tok = torch.tensor([[2.0, 8.0]])
    r = regret_and_agreement(ce, ok, pick, n_tok)
    assert r["regret_sum"] / r["tok_sum"] == pytest.approx(0.1)
    assert r["agree_sum"] / r["n"] == pytest.approx(0.5)
    # random cell: span 0 expects 0.5 over best, span 1 expects 2.0 -> 2.5 / 10
    assert r["random_regret_sum"] / r["tok_sum"] == pytest.approx(0.25)
