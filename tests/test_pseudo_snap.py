"""``tul.pseudo_k`` — the trace-free pseudo-token carrier, arm "snap" (2026-10-05).

LXTUL's latent-selected loop writes its hard winner into ONE prefix cell per slot and
leaves the other ``fan_k - 1`` prefix positions exactly zero (``_fan_route_cells``). This
key fills up to ``pseudo_k`` of those zero positions with a PonderLM-style snapped
mixture of the SPAN'S OWN tokens, read off the winner cell (or, under
``pseudo_source="entry"``, the slot's pre-loop seed).

Files: morph/model/tul_snap.py (``TULPseudoSnap``, the math), morph/model/tul.py (the five
keys, ``_check_pseudo_snap``), morph/model/transformer.py (construction beside
``tul_fan_lsel_*``, ``_tul_pseudo_snap_write``, the call site right after
``TULSlots.prefix_project``, the ``_pseudo_snap_capture`` test hook),
morph/training/tul_setup.py (``KNOWN_TUL_KEYS``, the Hydra -> TULConfig path, the wandb
manifest), morph/configs/lxtul_snap{,_entry}.yaml.

Note: .agents/notes/proposed/architecture/2026-10-05-trace-free-pseudo-token-carrier.md
Lean package: lab/theory/tul_pseudotoken/README.md

What each test pins:
  1. ``pseudo_k=0`` is bit-identical to the tree before the key (no key at all vs the
     explicit default): state dict, train loss, eval logits.
  2. ``pseudo_k=3`` with every ``g_n`` at its zero init is BYTE-IDENTICAL to the LXTUL
     ruler (no pseudo keys) — loss, logits — and every shared parameter equals the
     ruler's (RNG-neutral build).
  3. A NONZERO gate fills exactly the ``fan_k - 1`` loser positions and leaves the
     winner's own position completely unchanged.
  4. SUPPORT: the written vector is an exact (hand-recomputed) convex mixture of the
     span's own candidate embeddings; a token OUTSIDE the span does not move it, a token
     INSIDE the span does.
  5. The masked softmax is NaN-safe on an all-masked (pad/invalid) row, and a full model
     forward + backward over a batch with varied span lengths and pad slots is finite
     everywhere.
  6. THE VERTEX LIMIT: a large logit scale and a ``W_q`` aligned with one candidate snaps
     the written vector to (approximately) that candidate's raw embedding, scaled by
     ``g``.
  7. GRADIENT: the tied embedding table handed to the head is DETACHED at the call site
     (the module never re-detaches it itself); the loop's exit cell, ``W_q``, ``beta``
     and ``g`` all get a real gradient once the gate is nonzero.
  8. ``pseudo_source``: ``"entry"``'s captured source is IDENTICAL whatever the forced
     loop depth is; ``"exit"``'s is not (two forced per-slot depths on the same seed).
  9. CONFIG REFUSALS: every value check and every composition check in
     ``_check_pseudo_snap``, each isolated to the one violated condition.
 10. The two configs compose and differ from ``lxtul`` by exactly the keys the note says.

Sabotage checks, actually run and reverted (not committed):
  (a) Dropping the ``* self.beta.float().exp()...`` scaling in ``TULPseudoSnap.forward``
      fails ``test_hard_limit_large_beta_snaps_to_the_aligned_candidate`` (beta can no
      longer sharpen the softmax) AND ``test_gradient_reaches_wq_beta_gate_and_the_loop``
      (``beta`` drops out of the graph entirely, so its ``.grad`` is ``None``) — 2 of 22
      fail, the rest (including the byte-identical and support tests) stay green.
  (b) Dropping the ``* self.g...`` gate multiply fails
      ``test_pseudo_k3_gate_zero_is_byte_identical_to_the_lxtul_ruler`` (step 0 is no
      longer byte-identical: every pseudo position now writes the UNGATED mixture) AND
      ``test_gradient_reaches_wq_beta_gate_and_the_loop`` (``g`` drops out of the graph,
      ``.grad`` is ``None``) — 2 of 22 fail, the rest (support, vertex-limit, masking)
      stay green, confirming those do not already pin the gate either.

CPU, fp32, the ``tests/test_tul_fan.py`` / ``tests/test_tul_lxfan.py`` /
``tests/test_tul_fan_lsel.py`` strict LXTUL fixtures (``M = 4`` fan cells).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from morph.model.tul import TULConfig
from morph.model.tul_snap import TULPseudoSnap, rms_norm
from test_tul_fan import _batch, _tul
from test_tul_fan_lsel import _lsel
from test_tul_lx_credit import M
from test_tul_lxfan import _Spy

M = int(M)


def _pseudo(mode: str = "joint", seed: int = 1234, **kw) -> "object":
    kw.setdefault("pseudo_k", 3)
    kw.setdefault("pseudo_source", "exit")
    kw.setdefault("pseudo_support", "span")
    return _lsel(mode, seed=seed, **kw)


# ── 1. pseudo_k=0 is the tree before the key ────────────────────────────────────────


def test_pseudo_k_zero_is_bit_identical_to_the_tree_before_the_key():
    a = _lsel("joint")                 # no pseudo key at all
    b = _lsel("joint", pseudo_k=0)      # explicit default
    assert a.tul_pseudo_snap is None and b.tul_pseudo_snap is None
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    _ids, inp, lab, layout = _batch(M)
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


# ── 2. pseudo_k=3, gate 0: byte-identical to the LXTUL ruler ────────────────────────


def test_pseudo_k3_gate_zero_is_byte_identical_to_the_lxtul_ruler():
    ruler = _lsel("joint")
    arm = _pseudo("joint")
    assert arm.tul_pseudo_snap is not None
    assert bool((arm.tul_pseudo_snap.g == 0).all())
    s1, s2 = ruler.state_dict(), arm.state_dict()
    assert s1.keys() <= s2.keys()
    for k in s1:
        assert torch.equal(s1[k], s2[k]), k
    extra = s2.keys() - s1.keys()
    assert extra == {"tul_pseudo_snap.W_q", "tul_pseudo_snap.beta", "tul_pseudo_snap.g"}
    _ids, inp, lab, layout = _batch(M)
    ruler.train(); arm.train()
    torch.manual_seed(7)
    o1 = ruler(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(7)
    o2 = arm(inp, labels=lab, slot_layout=layout)
    assert torch.equal(o1["loss"], o2["loss"])
    ruler.eval(); arm.eval()
    with torch.no_grad():
        l1 = ruler(inp, labels=None, slot_layout=layout)["logits"]
        l2 = arm(inp, labels=None, slot_layout=layout)["logits"]
    assert torch.equal(l1, l2)


# ── 3. a nonzero gate fills exactly the loser positions ─────────────────────────────


def test_nonzero_gate_fills_losers_leaves_winner_position_unchanged():
    arm = _pseudo("joint")
    with torch.no_grad():
        arm.tul_pseudo_snap.g.fill_(1.0)
    _ids, inp, lab, layout = _batch(M)
    arm.eval()
    spy = _Spy(arm, "_tul_pseudo_snap_write")
    torch.manual_seed(3)
    with torch.no_grad():
        arm(inp, labels=None, slot_layout=layout)
    assert len(spy.outs) == 1
    (values_before, winner, *_rest), _kw = spy.calls[0]
    values_after = spy.outs[0]
    K = int(arm.cfg.tul.prefix_k)
    B, S = winner.shape
    before = values_before.view(B, S, K, *values_before.shape[2:])
    after = values_after.view(B, S, K, *values_after.shape[2:])
    valid = layout.slot_valid
    k_pseudo = arm.tul_pseudo_snap.k
    for b in range(B):
        for s in range(S):
            if not bool(valid[b, s]):
                continue
            w = int(winner[b, s])
            assert torch.equal(before[b, s, w], after[b, s, w]), "winner position moved"
            pseudo_positions = {(w + n) % K for n in range(1, k_pseudo + 1)}
            for j in pseudo_positions:
                assert bool(after[b, s, j].abs().sum() > 0), (b, s, j, "pseudo not written")
            # every OTHER loser position (beyond pseudo_k, none here at K-1 == k_pseudo)
            # stays exactly zero, as the ordinary LXTUL write already left it.
            untouched = set(range(K)) - pseudo_positions - {w}
            for j in untouched:
                assert torch.equal(after[b, s, j], before[b, s, j]), (b, s, j)


# ── 4. support: the span's own tokens, nothing else ─────────────────────────────────


def test_pseudo_is_an_exact_mixture_of_the_spans_own_candidates():
    torch.manual_seed(0)
    d = 8
    head = TULPseudoSnap(d, pseudo_k=2, scale_init=0.3, gate_init=1.0)
    B, S, J = 1, 1, 5
    c = torch.randn(B, S, d)
    table = torch.randn(20, d)
    ids = torch.randint(5, 20, (B, S, J))
    valid = torch.ones(B, S, J, dtype=torch.bool)
    pseudo, p, _ = head(c, ids, valid, table)
    cand = table[ids]
    hand = torch.einsum("bskj,bsjc->bskc", p, cand) * head.g.view(1, 1, -1, 1)
    assert torch.allclose(pseudo, hand, atol=1e-5)


def test_support_ignores_tokens_outside_the_span_and_reads_tokens_inside_it():
    torch.manual_seed(0)
    d = 8
    head = TULPseudoSnap(d, pseudo_k=1, scale_init=0.5, gate_init=1.0)
    c = torch.randn(1, 1, d)
    table = torch.randn(20, d)
    ids = torch.randint(5, 20, (1, 1, 5))
    valid = torch.ones(1, 1, 5, dtype=torch.bool)
    pseudo0, _, _ = head(c, ids, valid, table)

    table_outside = table.clone()
    table_outside[0] += 5.0            # id 0 never occurs in `ids` (drawn from [5, 20))
    pseudo1, _, _ = head(c, ids, valid, table_outside)
    assert torch.equal(pseudo0, pseudo1)

    table_inside = table.clone()
    table_inside[int(ids[0, 0, 0])] += 5.0
    pseudo2, _, _ = head(c, ids, valid, table_inside)
    assert not torch.equal(pseudo0, pseudo2)


# ── 5. masked softmax is NaN-safe; a real batch is finite end to end ───────────────


def test_all_masked_row_is_nan_safe():
    head = TULPseudoSnap(8, pseudo_k=1, scale_init=0.0, gate_init=1.0)
    c = torch.randn(1, 1, 8)
    table = torch.randn(10, 8)
    ids = torch.zeros(1, 1, 4, dtype=torch.long)
    valid = torch.zeros(1, 1, 4, dtype=torch.bool)       # every candidate masked
    pseudo, p, vertex = head(c, ids, valid, table)
    assert torch.isfinite(pseudo).all()
    assert torch.isfinite(p).all()
    assert torch.allclose(p.sum(dim=-1), torch.ones(1, 1, 1))


def test_real_batch_with_varied_spans_and_pad_slots_is_finite_end_to_end():
    arm = _pseudo("joint")
    with torch.no_grad():
        arm.tul_pseudo_snap.g.fill_(0.7)
    _ids, inp, lab, layout = _batch(M)
    assert bool((~layout.slot_valid).any()), "fixture must have at least one pad slot"
    arm.train()
    torch.manual_seed(5)
    out = arm(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for p in arm.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


# ── 6. the vertex limit ──────────────────────────────────────────────────────────────


def test_hard_limit_large_beta_snaps_to_the_aligned_candidate():
    torch.manual_seed(0)
    d = 16
    head = TULPseudoSnap(d, pseudo_k=1, scale_init=0.0, gate_init=1.0)
    table = F.normalize(torch.randn(10, d), dim=-1)
    ids = torch.arange(10).view(1, 1, 10)
    valid = torch.ones(1, 1, 10, dtype=torch.bool)
    target = 3
    c = table[target].view(1, 1, d).clone()
    with torch.no_grad():
        head.W_q.zero_()
        head.W_q[0] = torch.eye(d)
        head.beta.fill_(20.0)
    pseudo, p, vertex = head(c, ids, valid, table)
    assert float(vertex[0, 0, 0].detach()) > 0.999
    assert torch.allclose(pseudo[0, 0, 0].detach(), table[target], atol=1e-2)


# ── 7. gradient reaches the loop / head; the embedding is detached at the seam ─────


def test_table_handed_to_the_head_is_detached():
    arm = _pseudo("joint")
    _ids, inp, lab, layout = _batch(M)
    arm.train()
    spy = _Spy(arm.tul_pseudo_snap, "forward")
    torch.manual_seed(1)
    arm(inp, labels=lab, slot_layout=layout)
    assert len(spy.calls) == 1
    (c_arg, ids_arg, valid_arg, table_arg), _kw = spy.calls[0]
    assert table_arg.requires_grad is False
    assert c_arg.requires_grad                      # the loop's exit cell carries grad


def test_gradient_reaches_wq_beta_gate_and_the_loop():
    arm = _pseudo("joint")
    with torch.no_grad():
        arm.tul_pseudo_snap.g.fill_(0.3)
    _ids, inp, lab, layout = _batch(M)
    arm.train()
    torch.manual_seed(11)
    out = arm(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    head = arm.tul_pseudo_snap
    assert head.W_q.grad is not None and float(head.W_q.grad.abs().sum()) > 0
    assert head.beta.grad is not None
    assert head.g.grad is not None and float(head.g.grad.abs().sum()) > 0
    assert arm.tul.W_prefix.grad is not None and float(arm.tul.W_prefix.grad.abs().sum()) > 0


# ── 8. pseudo_source: entry is pass-count independent, exit is not ─────────────────


def test_entry_source_is_depth_independent_exit_source_is_not():
    """``slot_depths`` is checked against the CELL-major layout ``_tul_core`` expands to
    internally (``[B, max_slots * slot_cells]``) — one depth per SPAN, broadcast over its
    ``slot_cells`` register cells (``_tul_core``'s own comment); filling every entry with
    the same value trivially satisfies that broadcast rule."""
    _ids, inp, lab, layout = _batch(M)
    B, max_slots = layout.slot_index.shape
    width = max_slots * M
    d1 = torch.ones(B, width, dtype=torch.long)
    d3 = torch.full((B, width), 3, dtype=torch.long)
    for source in ("entry", "exit"):
        m = _pseudo("joint", pseudo_source=source, seed=99)
        with torch.no_grad():
            m.tul_pseudo_snap.g.fill_(1.0)
        m.eval()
        caps = []
        for d in (d1, d3):
            m._pseudo_snap_capture = []
            with torch.no_grad():
                m.tul_forward_ablated(inp, None, layout, slot_depths=d)
            assert len(m._pseudo_snap_capture) == 1
            caps.append(m._pseudo_snap_capture[0]["c"].clone())
        if source == "entry":
            assert torch.equal(caps[0], caps[1]), \
                "pseudo_source='entry' must not depend on the loop's pass count"
        else:
            assert not torch.equal(caps[0], caps[1]), \
                "pseudo_source='exit' must depend on the loop's pass count"


# ── 9. config refusals ───────────────────────────────────────────────────────────────


def _lsel_tul(**kw) -> dict:
    """A minimal valid composition for ``_check_pseudo_snap`` to run against: the fan,
    the write-all mix and the latent-selected loop, all satisfied."""
    base = dict(fan_k=M, slot_cells=M, prefix_k=M, fan_mix="all",
                fan_loop_select="joint", fan_all_wta_lambda=0.0)
    base.update(kw)
    return _tul(**base)


def test_pseudo_k_zero_with_a_non_default_pseudo_key_raises():
    with pytest.raises(ValueError, match="a silent no-op"):
        TULConfig(**_lsel_tul(pseudo_source="entry"))


def test_bad_pseudo_source_raises():
    with pytest.raises(ValueError, match="pseudo_source must be"):
        TULConfig(**_lsel_tul(pseudo_k=3, pseudo_source="bogus"))


def test_bad_pseudo_support_raises():
    with pytest.raises(ValueError, match="pseudo_support must be"):
        TULConfig(**_lsel_tul(pseudo_k=3, pseudo_support="vocab_topk"))


def test_pseudo_without_the_latent_selected_loop_raises():
    with pytest.raises(ValueError, match="needs tul.fan_loop_select"):
        TULConfig(**_tul(fan_k=M, slot_cells=M, prefix_k=M, fan_mix="all",
                         fan_loop_select="off", pseudo_k=3))


def test_pseudo_with_fan_lsel_read_all_raises():
    with pytest.raises(ValueError, match="needs tul.fan_loop_select"):
        TULConfig(**_lsel_tul(fan_lsel_read="all", pseudo_k=3))


def test_pseudo_outside_strict_geometry_raises():
    with pytest.raises(ValueError, match="needs tul.fan_loop_select"):
        TULConfig(**_lsel_tul(tg_geometry="restrict", pseudo_k=3))


def test_pseudo_with_prefix_per_cell_raises():
    with pytest.raises(NotImplementedError, match="prefix_per_cell"):
        TULConfig(**_lsel_tul(prefix_k=M * 2, prefix_per_cell=2, pseudo_k=3))


def test_pseudo_k_too_large_raises():
    with pytest.raises(ValueError, match=r"> tul\.fan_k - 1"):
        TULConfig(**_lsel_tul(pseudo_k=M))


def test_pseudo_k_negative_raises():
    with pytest.raises(ValueError, match="must be >= 0"):
        TULConfig(**_lsel_tul(pseudo_k=-1))


# ── 10. the configs compose ──────────────────────────────────────────────────────────


_CONFIG_DIR = __import__("os").path.abspath("morph/configs")


def _compose(name: str):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        return compose(config_name=name)


def test_lxtul_snap_composes_and_differs_from_lxtul_by_exactly_the_documented_keys():
    from omegaconf import OmegaConf

    from test_slot_gain_tail import _leaves, _MISSING

    c_snap = _leaves(OmegaConf.to_container(_compose("lxtul_snap"), resolve=True))
    c_entry = _leaves(OmegaConf.to_container(_compose("lxtul_snap_entry"), resolve=True))
    c_base = _leaves(OmegaConf.to_container(_compose("lxtul"), resolve=True))

    diff = {k for k in c_snap.keys() | c_base.keys()
            if c_snap.get(k, _MISSING) != c_base.get(k, _MISSING)}
    assert diff == {"tul.pseudo_k", "tul.pseudo_source", "tul.pseudo_support",
                    "training.steps", "wandb.name"}, sorted(diff)
    assert c_snap["tul.pseudo_k"] == 3 and c_snap["tul.pseudo_source"] == "exit"
    assert c_snap["training.steps"] == 5000 and c_snap["wandb.name"] == "lxtul-snap"

    diff_entry = {k for k in c_entry.keys() | c_base.keys()
                  if c_entry.get(k, _MISSING) != c_base.get(k, _MISSING)}
    assert diff_entry == {"tul.pseudo_k", "tul.pseudo_source", "tul.pseudo_support",
                          "training.steps", "wandb.name"}, sorted(diff_entry)
    assert c_entry["tul.pseudo_source"] == "entry"
    assert c_entry["wandb.name"] == "lxtul-snap-entry"

    diff_pair = {k for k in c_snap.keys() | c_entry.keys()
                 if c_snap.get(k, _MISSING) != c_entry.get(k, _MISSING)}
    assert diff_pair == {"tul.pseudo_source", "wandb.name"}, sorted(diff_pair)


def test_lxtul_snap_reaches_the_tulconfig_the_trainer_builds(monkeypatch):
    import transformers

    from morph.training import tul_setup
    from test_tul_strict_geometry import _StubTok, _rule

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubTok)
    monkeypatch.setattr(tul_setup, "build_boundary_rule",
                        lambda cfg, cache_dir="": (_rule(), _rule().is_boundary, 0, ("\n",)))
    cfg = _compose("lxtul_snap")
    rt = tul_setup.build_tul_runtime(cfg)
    assert rt.model_cfg.pseudo_k == 3
    assert rt.model_cfg.pseudo_source == "exit"
    assert rt.model_cfg.pseudo_support == "span"
    assert rt.manifest["pseudo_k"] == 3
    assert rt.manifest["pseudo_source"] == "exit"
