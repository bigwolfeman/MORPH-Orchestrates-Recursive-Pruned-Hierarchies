"""``tul.recon_weight`` — own-span reconstruction from the cell the coda actually reads
(CE queue item 10, 2026-10-05).

Why: nothing in LXTUL charges a penalty for a slot's winning cell losing its OWN span's
exact tokens. `tul.spandec` grades the NEXT span, and even that is graded on the
PRE-route mean of every cell (so every cell gets gradient — `_fan_route_cells`'s
documented choice), not on the one cell the coda reads. This key adds a THIRD,
independent teacher-forced decoder (reusing `morph.model.tul_spandec.SpanDecoder` and
`own_span_slots`) that reconstructs span s from the live, undiluted WINNER cell, read
AFTER the latent-selected loop's route and BEFORE `W_prefix`.

Files: morph/model/tul.py (`recon_weight` / `recon_layers` fields, `_check_recon`),
morph/model/transformer.py (`self.tul_recon` construction with an RNG snapshot/restore,
`_tul_recon_loss`, the call site inside the register/fan write branch, the
`recon_weighted` fold), morph/training/tul_setup.py (`KNOWN_TUL_KEYS`, the Hydra path,
the wandb manifest), morph/training/train.py (both subtraction tuples, the `_mk` val
tuple, the `tul/recon_*` scan), the three new configs.

What each test pins:
  1. `recon_weight: 0.0` (implicit or explicit) builds NO parameters (`m.tul_recon is
     None`) and is BIT-IDENTICAL to the tree before this key: state_dict, train loss,
     eval logits.
  2. Building the decoder at `recon_weight > 0` is RNG-neutral: every OTHER parameter
     (everything that is not `tul_recon.*`) equals the `recon_weight=0` twin's, at the
     same seed — the `TULSlotRegister` snapshot/restore precedent, applied here because
     `SpanDecoder`'s own `nn.Linear`s would otherwise shift the global stream.
  3. The targets are exactly span s's OWN tokens: `_tul_recon_loss` reaches the same loss
     as an independent hand call of `own_span_slots` + `SpanDecoder.decode` +
     `fused_linear_cross_entropy` on the real fixture (which has spans of different
     lengths and at least one pad slot).
  4. IT READS THE WINNER: using the real `_fan_route_cells` + `_tul_recon_loss`,
     perturbing a LOSER's raw register cell leaves the recon loss EXACTLY unchanged
     (the route zeroes it regardless); perturbing the WINNER's raw cell changes it.
  5. GRADIENT REACHES THE LOOP: calling `.backward()` on the live (undetached) recon
     loss alone moves `tul_register.{Q,W_k,W_v,W_o}` and a core-loop block's parameters.
  6. THE WEIGHTED TERM IS ADDITIVE: `recon_on_loss - recon_weighted == recon_off_loss`,
     exactly, on the same seed and batch — so train/loss and val loss stay the model's
     other objective once train.py subtracts `recon_weighted`.
  7. ABSENT AT EVAL AND AT NO-LABEL FORWARDS: `model.eval()` with labels, and
     `model.train()` with `labels=None`, both have no `recon` / `recon_weighted` key.
  8. REFUSALS: every geometry `_check_recon` names; `recon_layers` set at weight 0;
     `recon_weight < 0`; `recon_layers < 1` at weight > 0.
  9. The configs compose, differ from `lxtul.yaml` by exactly their stated keys, and
     reach `TULConfig` through `build_tul_runtime`.
  10. END TO END: forward + backward on the tiny analogues of `lxtul_recon.yaml` and
      `lxtul_pool16_recon.yaml` gives a finite loss.

Sabotage checks (reported in the session, not committed): (a) reading the PRE-route mean
(`h_slots` instead of `_cells.sum(dim=2)`) instead of the post-route winner makes the
"reads the winner" test's loser-perturbation assertion fail (a loser's raw value would
then move the mean and the loss); (b) dropping the RNG snapshot/restore around
`SpanDecoder`'s construction makes test 2 fail (a later-constructed parameter would
differ between the on/off twins).

CPU, fp32, the ``tests/test_tul_fan.py`` / ``tests/test_tul_fan_lsel.py`` fixtures.
"""
from __future__ import annotations

import pytest
import torch

from morph.model.tul import TULConfig
from morph.model.tul_spandec import own_span_slots
from morph.model.transformer import MORPHTransformer
from morph.model.fused_ce import fused_linear_cross_entropy
from morph.training.tul_setup import KNOWN_TUL_KEYS, reject_unknown_tul_keys
from test_tul_fan import _batch
from test_tul_fan_lsel import _lsel
from test_tul_lx_credit import M as FAN_K

FAN_K = int(FAN_K)


# ── 1. off is nothing ────────────────────────────────────────────────────────────────


def test_recon_weight_zero_builds_no_parameters_and_is_bit_identical():
    a = _lsel("joint")                           # no recon key
    b = _lsel("joint", recon_weight=0.0)          # explicit default
    assert a.tul_recon is None and b.tul_recon is None
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    _ids, inp, lab, layout = _batch(FAN_K)
    a.train(); b.train()
    torch.manual_seed(7)
    oa = a(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(7)
    ob = b(inp, labels=lab, slot_layout=layout)
    assert torch.equal(oa["loss"], ob["loss"])
    a.eval(); b.eval()
    with torch.no_grad():
        la = a(inp, labels=None, slot_layout=layout)["logits"]
        lb = b(inp, labels=None, slot_layout=layout)["logits"]
    assert torch.equal(la, lb)


# ── 2. RNG-neutral build ─────────────────────────────────────────────────────────────


def test_recon_build_is_rng_neutral():
    off = _lsel("joint", recon_weight=0.0)
    on = _lsel("joint", recon_weight=1.0)
    s_off, s_on = off.state_dict(), on.state_dict()
    off_keys = set(s_off.keys())
    on_keys = set(s_on.keys())
    assert off_keys.issubset(on_keys)
    new_keys = on_keys - off_keys
    assert new_keys and all(k.startswith("tul_recon.") for k in new_keys), new_keys
    for k in off_keys:
        assert torch.equal(s_off[k], s_on[k]), k


def test_recon_build_leaves_the_global_rng_state_untouched():
    """Direct check of the CLAIM (the `TULSlotRegister` precedent's own test style):
    building `SpanDecoder` without a snapshot/restore DOES move the global RNG stream
    (`nn.Linear`'s default kaiming init runs before the private-generator overwrite), so
    the state right after construction must be identical with and without the key — not
    merely inferred from the state_dict comparison above, which can stay green even if
    the restore is missing whenever nothing else in that particular build happens to
    depend on the raw stream afterward."""
    off = _lsel("joint", recon_weight=0.0)
    after_off = torch.random.get_rng_state()
    on = _lsel("joint", recon_weight=1.0)
    after_on = torch.random.get_rng_state()
    assert torch.equal(after_off, after_on)


# ── 3. targets are span s's own tokens ───────────────────────────────────────────────


def test_recon_loss_matches_an_independent_own_span_slots_hand_call():
    m = _lsel("joint", recon_weight=1.0)
    _ids, inp, lab, layout = _batch(FAN_K)
    S = layout.slot_index.shape[1]
    assert bool((~layout.slot_valid).any()), "fixture must have at least one pad slot"
    span_lens = []
    for s in range(S):
        if bool(layout.slot_valid[0, s]):
            span_lens.append(int(((~layout.slot_mask[0]) & (layout.bag_id[0] == s)).sum()))
    assert len(set(span_lens)) > 1, "fixture must have spans of different lengths"

    captured: list = []
    orig = m._tul_recon_loss

    def _spy(winner_state, input_ids, lay, stats=None):
        captured.append((winner_state.detach().clone(), input_ids, lay))
        return orig(winner_state, input_ids, lay, stats=stats)
    m._tul_recon_loss = _spy
    m.train()
    torch.manual_seed(9)
    out = m(inp, labels=lab, slot_layout=layout)
    assert len(captured) == 1
    winner_state, cap_ids, cap_layout = captured[0]

    dec = m.tul_recon
    z = m._readout(winner_state)
    w_tied = m.embed.lm_weight().detach()
    ids, valid = own_span_slots(cap_ids, cap_layout, dec.max_tokens)
    st = dec.decode(z, ids, valid, w_tied)
    C = st.shape[-1]
    lab2 = torch.where(valid, ids, torch.full_like(ids, -100))
    hand_loss = fused_linear_cross_entropy(
        st.reshape(-1, C), w_tied, lab2.reshape(-1), ignore_index=-100,
        chunk_size=m.cfg.ce_chunk_size, mask_token_id=m.cfg.tul.slot_id, **m._ce_kw)
    assert torch.equal(hand_loss, out["recon"])


def test_real_forward_call_site_passes_the_post_route_undiluted_sum():
    """Pins the CALL SITE in `_forward_tul`, not just `_tul_recon_loss` in isolation:
    captures the `winner_state` the real forward actually hands `_tul_recon_loss` AND
    the routed `cells` the SAME forward hands `prefix_project` (the coda's write, the
    "actually reaches the coda" object), then asserts the former equals the latter's
    cell-axis SUM. A call site that read `h_slots` (the pre-route MEAN, diluted by
    1/fan_k) instead would fail this for any slot with fan_k > 1, since mean != sum."""
    m = _lsel("joint", recon_weight=1.0)
    _ids, inp, lab, layout = _batch(FAN_K)
    recon_cap: list = []
    orig_recon = m._tul_recon_loss

    def _spy_recon(winner_state, input_ids, lay, stats=None):
        recon_cap.append(winner_state.detach().clone())
        return orig_recon(winner_state, input_ids, lay, stats=stats)
    m._tul_recon_loss = _spy_recon

    cells_cap: list = []
    orig_pp = m.tul.prefix_project

    def _spy_pp(h_slots, lay, l_total, cells=None):
        if cells is not None:
            cells_cap.append(cells.detach().clone())
        return orig_pp(h_slots, lay, l_total, cells=cells)
    m.tul.prefix_project = _spy_pp

    m.train()
    torch.manual_seed(37)
    m(inp, labels=lab, slot_layout=layout)
    assert len(recon_cap) == 1 and len(cells_cap) == 1
    expect = cells_cap[0].sum(dim=2)
    assert torch.equal(recon_cap[0], expect)


# ── 4. reads the winner, not the losers ──────────────────────────────────────────────


def _reg_cell_shape(m: MORPHTransformer, layout) -> tuple:
    B, S = layout.slot_valid.shape
    n = m._n_streams if m._is_hc else None
    return (B, S, FAN_K, n, m.cfg.d_model) if n else (B, S, FAN_K, m.cfg.d_model)


def test_recon_reads_the_winner_loser_perturbation_is_invisible():
    m = _lsel("joint", recon_weight=1.0)
    _ids, inp, lab, layout = _batch(FAN_K)
    shape = _reg_cell_shape(m, layout)
    torch.manual_seed(13)
    cells = torch.randn(*shape)
    B, S = layout.slot_valid.shape
    winner = torch.zeros(B, S, dtype=torch.long)
    winner[:] = 1                                 # every slot's winner is cell 1
    route = {"winner": winner, "scale": None}

    routed = MORPHTransformer._fan_route_cells(cells, route)
    base_state = routed.sum(dim=2)
    base_loss = m._tul_recon_loss(base_state, inp, layout)

    cells_loser_pert = cells.clone()
    cells_loser_pert[:, :, 0] += 100.0             # perturb a LOSER cell (index 0 != 1)
    routed_lp = MORPHTransformer._fan_route_cells(cells_loser_pert, route)
    lp_state = routed_lp.sum(dim=2)
    lp_loss = m._tul_recon_loss(lp_state, inp, layout)
    assert torch.equal(base_state, lp_state)
    assert torch.equal(base_loss, lp_loss)

    cells_winner_pert = cells.clone()
    cells_winner_pert[:, :, 1] += 100.0            # perturb the WINNER cell (index 1)
    routed_wp = MORPHTransformer._fan_route_cells(cells_winner_pert, route)
    wp_state = routed_wp.sum(dim=2)
    wp_loss = m._tul_recon_loss(wp_state, inp, layout)
    assert not torch.equal(base_state, wp_state)
    assert not torch.equal(base_loss, wp_loss)


# ── 5. gradient reaches the loop / the cells ─────────────────────────────────────────


def test_recon_gradient_reaches_register_and_core():
    """``W_o`` is ZERO-INIT (the register's step-0 no-op rule), so
    ``d(register output)/d(pooled) = W_o = 0`` identically AT INIT and Q/W_k/W_v get
    EXACTLY zero gradient from ANY per-slot reader on a freshly built model — not a bug
    in recon, a property of the zero-init design every register reader shares. `W_o`
    and `P_cell` themselves DO get gradient at step 0 (the loss depends on them
    linearly), so they are checked on the untouched model; `W_o` is then perturbed off
    zero (simulating "after step 1 of training") to confirm gradient also reaches
    Q/W_k/W_v once the seed is live, and the core loop's own blocks get gradient
    regardless (their path to the exit state does not route through the register's
    output at all)."""
    m = _lsel("joint", recon_weight=1.0)
    with torch.no_grad():
        m.tul_register.W_o.weight.copy_(0.1 * torch.eye(m.tul_register.W_o.weight.shape[0]))
    _ids, inp, lab, layout = _batch(FAN_K)
    captured: list = []
    orig = m._tul_recon_loss

    def _spy(winner_state, input_ids, lay, stats=None):
        r = orig(winner_state, input_ids, lay, stats=stats)
        captured.append(r)
        return r
    m._tul_recon_loss = _spy
    m.train()
    torch.manual_seed(17)
    m(inp, labels=lab, slot_layout=layout)
    assert len(captured) == 1
    captured[0].backward()
    for name in ("Q", "W_k.weight", "W_v.weight", "W_o.weight", "P_cell"):
        obj = m.tul_register
        for part in name.split("."):
            obj = getattr(obj, part)
        assert obj.grad is not None and float(obj.grad.abs().sum()) > 0.0, name
    core_grad = sum(float(p.grad.abs().sum()) for p in m.core[0].parameters()
                    if p.grad is not None)
    assert core_grad > 0.0


# ── 6. the weighted term is additive ─────────────────────────────────────────────────


def test_recon_weighted_term_is_additive():
    off = _lsel("joint", recon_weight=0.0)
    on = _lsel("joint", recon_weight=1.0)
    _ids, inp, lab, layout = _batch(FAN_K)
    off.train(); on.train()
    torch.manual_seed(23)
    out_off = off(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(23)
    out_on = on(inp, labels=lab, slot_layout=layout)
    assert out_on.get("recon_weighted") is not None
    recovered = out_on["loss"] - out_on["recon_weighted"]
    assert torch.allclose(recovered, out_off["loss"], atol=1e-5)


# ── 7. absent at eval and at no-label forwards ───────────────────────────────────────


def test_recon_absent_at_eval_and_no_label_forwards():
    m = _lsel("joint", recon_weight=1.0)
    _ids, inp, lab, layout = _batch(FAN_K)
    m.eval()
    with torch.no_grad():
        out_eval = m(inp, labels=lab, slot_layout=layout)
    assert out_eval.get("recon") is None and out_eval.get("recon_weighted") is None
    m.train()
    out_nolabel = m(inp, labels=None, slot_layout=layout)
    assert out_nolabel.get("recon") is None and out_nolabel.get("recon_weighted") is None


# ── 8. refusals ───────────────────────────────────────────────────────────────────────


def test_recon_weight_negative_raises():
    with pytest.raises(ValueError, match="recon_weight must be >= 0"):
        TULConfig(prefix_k=2, slot_id=4, recon_weight=-1.0)


def test_recon_layers_set_at_weight_zero_raises():
    with pytest.raises(ValueError, match="recon_layers set with tul.recon_weight=0"):
        TULConfig(prefix_k=2, slot_id=4, recon_weight=0.0, recon_layers=3)


def test_recon_layers_lt_1_at_weight_gt_0_raises():
    with pytest.raises(ValueError, match="recon_layers must be >= 1"):
        TULConfig(prefix_k=FAN_K, slot_id=4, fan_k=FAN_K, slot_cells=FAN_K,
                  fan_mix="all", fan_loop_select="joint", recon_weight=1.0,
                  recon_layers=0)


@pytest.mark.parametrize("bad_kw", [
    dict(fan_k=0),                                         # no fan at all
    dict(fan_mix="mean"),                                  # mixing fan, no register write
    dict(fan_loop_select="off"),                           # no latent-selected loop
    dict(fan_lsel_read="all"),                             # no hard winner
])
def test_recon_refuses_every_geometry_without_a_hard_winner(bad_kw):
    base = dict(prefix_k=FAN_K, slot_id=4, fan_k=FAN_K, slot_cells=FAN_K, fan_mix="all",
               fan_loop_select="joint", recon_weight=1.0)
    base.update(bad_kw)
    if base["fan_k"] == 0:
        base["slot_cells"] = 1
        base["prefix_k"] = 2
        base["fan_mix"] = "mean"
        base["fan_loop_select"] = "off"
    with pytest.raises((ValueError, NotImplementedError)):
        TULConfig(**base)


# ── 9. known key, Hydra path, configs ────────────────────────────────────────────────


def test_recon_keys_are_known_tul_keys():
    assert {"recon_weight", "recon_layers"} <= KNOWN_TUL_KEYS
    class _FakeDict(dict):
        def keys(self): return super().keys()
    reject_unknown_tul_keys(_FakeDict(recon_weight=1.0, recon_layers=2))
    with pytest.raises(ValueError, match="unknown"):
        reject_unknown_tul_keys(_FakeDict(recon_wieght=1.0))


_CONFIG_DIR = __import__("os").path.abspath("morph/configs")


def _compose(name: str):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        return compose(config_name=name)


@pytest.mark.parametrize("name,expect_keys,expect_vals", [
    ("lxtul_recon", {"tul.recon_weight", "tul.recon_layers", "training.steps",
                     "wandb.name"},
     {"tul.recon_weight": 1.0, "tul.recon_layers": 2, "training.steps": 5000,
      "wandb.name": "lxtul-recon"}),
    ("lxtul_pool16_recon",
     {"tul.recon_weight", "tul.recon_layers", "tul.slot_pool_heads", "training.steps",
      "wandb.name"},
     {"tul.recon_weight": 1.0, "tul.recon_layers": 2, "tul.slot_pool_heads": 16,
      "training.steps": 5000, "wandb.name": "lxtul-pool16-recon"}),
])
def test_recon_configs_compose_and_differ_from_lxtul_by_exactly_their_keys(
        name, expect_keys, expect_vals):
    from omegaconf import OmegaConf

    from test_slot_gain_tail import _leaves, _MISSING

    c = _leaves(OmegaConf.to_container(_compose(name), resolve=True))
    c1 = _leaves(OmegaConf.to_container(_compose("lxtul"), resolve=True))
    diff = {k for k in c.keys() | c1.keys() if c.get(k, _MISSING) != c1.get(k, _MISSING)}
    assert diff == expect_keys, sorted(diff)
    for k, v in expect_vals.items():
        assert c[k] == v, (k, c[k], v)


def test_recon_config_reaches_the_tulconfig_the_trainer_builds(monkeypatch):
    import transformers
    from hydra import compose, initialize_config_dir

    from morph.training import tul_setup
    from test_tul_strict_geometry import _StubTok, _rule as _strict_rule

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubTok)
    monkeypatch.setattr(tul_setup, "build_boundary_rule",
                        lambda cfg, cache_dir="": (_strict_rule(), _strict_rule().is_boundary,
                                                   0, ("\n",)))
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        cfg = compose(config_name="lxtul_recon")
    rt = tul_setup.build_tul_runtime(cfg)
    assert rt.model_cfg.recon_weight == 1.0
    assert rt.model_cfg.recon_layers == 2
    assert rt.manifest["recon_weight"] == 1.0
    assert rt.manifest["recon_layers"] == 2


# ── 10. end to end: forward + backward on the tiny analogues ───────────────────────


def test_lxtul_recon_tiny_analogue_forward_backward_is_finite():
    m = _lsel("joint", recon_weight=1.0)
    _ids, inp, lab, layout = _batch(FAN_K)
    m.train()
    torch.manual_seed(29)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n


def test_lxtul_pool16_recon_tiny_analogue_forward_backward_is_finite():
    m = _lsel("joint", recon_weight=1.0, slot_pool_heads=16)
    _ids, inp, lab, layout = _batch(FAN_K)
    m.train()
    torch.manual_seed(31)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n
