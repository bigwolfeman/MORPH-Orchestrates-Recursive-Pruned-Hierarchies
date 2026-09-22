"""LXTUL-R Step 1 composition checks (`tul_slot_spandec_strict_fan4_all_reach1.yaml`).

Design record: `.agents/notes/proposed/architecture/2026-09-21-lxtul-r-reach-composition.md`
("Step 1: fan4-all plus the chain"). Nothing new is built here — every knob composed
already exists — this file proves three knob interactions the design names as
UNVERIFIED, plus a smoke that the composed config trains and evals without raising:

* 3a. `tul.loop_reach: 1` on a register (`fan_k`/`slot_cells`/`prefix_k` = 4): inside the
  loop, does a cell attend every cell of its OWN slot and of slot k-1 ONLY? Asserted
  DIRECTLY on `slot_cell_relation` (`morph/model/transformer.py`), the pure function the
  loop's core-layer-0 mask is built from — no model needed.
* 3b. `model.slot_state_renorm: true` with K cells: is the entry norm kept PER CELL or
  PER SLOT (mean over cells)? Read from `_tul_core`'s own comment and the S*M cell-axis
  rebinding (`morph/model/transformer.py`), then verified end to end on a forward.
* 3c. `tul.tg_coda_prefix_reach: prev` with `fan_mix: all`: does a token of span k+1
  read the K cells of slot k and NONE of slot k-1's? Asserted DIRECTLY on
  `tg_strict_allow` (`morph/model/tul_layout.py`), the pure function the coda's mask is
  built from, using a REAL packed layout at `prefix_k=4` (== `fan_k`/`slot_cells`).
* 3d. The composed tiny config: a 2-step train forward+backward (loss finite, grads
  finite, no raise) and one eval forward with the depth forced via `tul.slot_mean_depth`
  / `tul.slot_max_depth` (the depth-sweep script's own knobs, TULConfig fields, a
  one-liner — no separate runner needed).

CPU only, fp32, `use_kernels=False` — the `tests/test_tul_fan.py` fixtures (`_tiny`,
`_rule`, `_spec`, `_ids`, `_batch`), reused the way `tests/test_tul_fan_all.py` reuses
them (`from test_tul_fan import ...`).
"""

from __future__ import annotations

import pytest
import torch

from morph.model.transformer import MORPHTransformer, slot_cell_relation
from morph.model.tul import TULConfig
from morph.model.tul_layout import slot_layout_from_ids, tg_strict_allow

from test_tul_fan import _batch, _ids, _rule, _spec, _tiny

# ── the composed tiny model ─────────────────────────────────────────────────


def _r1_tul(**kw) -> TULConfig:
    base = dict(
        prefix_k=4, slot_id=4, tg_restrict=True, tg_restrict_scope="all",
        tg_geometry="strict", emit_weight=0.0, token_state_dropout=0.0, mux_beta=0.0,
        fan_k=4, slot_cells=4, fan_mix="all", fan_all_wta_lambda=1.0,
        fan_repel_mode="epivol", fan_select_eps=0.0,
        loop_reach=1, tg_coda_prefix_reach="prev",
    )
    base.update(kw)
    return TULConfig(**base)


def _r1_model(seed: int = 1234, tul_kw: dict | None = None,
             **model_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    cfg = _tiny(tul=_r1_tul(**(tul_kw or {})),
               core_fixed_point_lambda=0.0, slot_state_renorm=True, **model_kw)
    m = MORPHTransformer(cfg)
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.eval().float()


def _assert_real_logits_finite(logits: torch.Tensor, slot_id: int) -> None:
    """`fused_ce`'s `mask_token_id` forces the vocab row at `tul.slot_id` (the special
    slot/prefix token, never a legitimate prediction target) to exactly -inf at EVERY
    position (`morph/model/fused_ce.py`, `morph/model/CLAUDE.md`) — a by-design, known
    non-finite column, not a leak. Checked directly (not swallowed by `nan_to_num`, the
    convention `tests/test_tul_fan.py::_finite_logit_sum` uses) so a genuine NaN
    elsewhere is not hidden behind it.
    """
    assert torch.isinf(logits[..., slot_id]).all() and (logits[..., slot_id] < 0).all(), \
        f"expected tul.slot_id={slot_id}'s logit column to be exactly -inf everywhere"
    real = torch.cat([logits[..., :slot_id], logits[..., slot_id + 1:]], dim=-1)
    assert torch.isfinite(real).all(), "a REAL vocabulary logit is not finite"


def test_r1_config_composes_a_model_with_no_raise():
    m = _r1_model()
    assert m.cfg.core_fixed_point_lambda == 0.0
    assert m.cfg.slot_state_renorm is True
    assert m.cfg.tul.loop_reach == 1
    assert m.cfg.tul.tg_coda_prefix_reach == "prev"
    assert m.cfg.tul.fan_mix == "all"
    assert m.cfg.tul.slot_cells == 4


# ── 3a. the in-loop relation: own slot + slot k-1 only ─────────────────────

def test_loop_reach1_relation_is_own_slot_plus_one_slot_back():
    """`slot_cell_relation(n_slots, m_cells, device, reach=1)` — the pure builder
    `_tul_core` hands core layer 0 as `tg_relation` at `loop_reach: 1`
    (`morph/model/transformer.py`, the `# depth as reach` section: ``_mask0, _same =
    slot_cell_relation(_n_slots, _m_cells, x.device, _reach, ...)``).

    Hand-built expected relation for 3 slots x 2 cells (small enough to enumerate):
    cell i of slot k must read every cell of slot k (both, regardless of order within
    the slot) and every cell of slot k-1, and NOTHING of slot k-2 or earlier, or of any
    LATER slot. `same` (what layers 1..n-1 run) must be slot-local only, at every reach.
    """
    n_slots, m_cells = 3, 2
    blk, same = slot_cell_relation(n_slots, m_cells, torch.device("cpu"), reach=1)
    blk, same = blk[0, 0], same[0, 0]

    slot_of = torch.tensor([c // m_cells for c in range(n_slots * m_cells)])
    expected_blk = torch.zeros(n_slots * m_cells, n_slots * m_cells, dtype=torch.bool)
    expected_same = torch.zeros_like(expected_blk)
    for i in range(n_slots * m_cells):
        for j in range(n_slots * m_cells):
            si, sj = int(slot_of[i]), int(slot_of[j])
            expected_same[i, j] = si == sj
            expected_blk[i, j] = (si == sj) or (sj == si - 1)

    assert torch.equal(blk, expected_blk), "reach=1: own slot + slot k-1 only"
    assert torch.equal(same, expected_same), "`same` must be slot-local regardless of reach"

    # And the two-slots-back cell must be refused: slot 2's cells never read slot 0's.
    cells_slot2 = [i for i in range(n_slots * m_cells) if slot_of[i] == 2]
    cells_slot0 = [j for j in range(n_slots * m_cells) if slot_of[j] == 0]
    for i in cells_slot2:
        for j in cells_slot0:
            assert not bool(blk[i, j]), \
                f"cell {i} (slot 2) must not read cell {j} (slot 0) at reach=1"
    # slot 1's cells (k-1) MUST be readable by slot 2's cells.
    cells_slot1 = [j for j in range(n_slots * m_cells) if slot_of[j] == 1]
    for i in cells_slot2:
        assert all(bool(blk[i, j]) for j in cells_slot1), \
            f"cell {i} (slot 2) must read every cell of slot 1 (k-1) at reach=1"


def test_loop_reach1_relation_is_a_strict_narrowing_of_reach0():
    """`tul.loop_reach: 0` means UNLIMITED (`morph/model/tul.py`: "0 = unlimited"), so
    `reach=0`'s `blk` is plain block-causal (every earlier slot, no narrowing) and
    `reach=1` REMOVES exactly the pairs where slot j is more than one slot behind slot i
    — it can never ADD a pair `reach=0` did not already allow."""
    n_slots, m_cells = 4, 4
    blk0, _ = slot_cell_relation(n_slots, m_cells, torch.device("cpu"), reach=0)
    blk1, _ = slot_cell_relation(n_slots, m_cells, torch.device("cpu"), reach=1)
    slot_of = torch.arange(n_slots * m_cells) // m_cells
    si = slot_of.unsqueeze(1)
    sj = slot_of.unsqueeze(0)
    assert torch.equal(blk0[0, 0], (si >= sj))
    added = blk1[0, 0] & ~blk0[0, 0]
    assert not added.any(), "reach=1 must never allow a pair reach=0 (unlimited) forbids"
    removed = blk0[0, 0] & ~blk1[0, 0]
    assert torch.equal(removed, (sj < si - 1)), \
        "reach=1 must remove EXACTLY the pairs more than one slot back"


# ── 3b. slot_state_renorm: per CELL, not per slot (mean over cells) ────────

def test_slot_state_renorm_keeps_each_cells_entry_norm_through_a_pass():
    """`slot_state_renorm: true` on the K-cell register: the state that LEAVES a pass has,
    per CELL, the norm it ENTERED the loop with (`_tul_core`: `_n0 = h.detach().flatten(2)
    .norm(dim=2)` at entry, `h_new = h_new * (_n0 / _hn)` after every pass, both on the
    S*M cell axis, no mean over a slot's M cells). Read through the model's own per-pass
    capture (`_jac_capture` appends the state ENTERING each pass), forced to two passes:
    capture[1]["h"] is the state after pass 0's renorm, capture[0]["h"] the entry. On every
    cell that pass 0 CHANGED, the two norms must agree to 1e-3 relative; and pass 0 must
    have changed a real share of the valid cells, so the assertion has teeth."""
    m2 = _r1_model(tul_kw=dict(slot_mean_depth=2, slot_max_depth=2))
    m2.eval()
    ids, inp, lab, layout = _batch(prefix_k=4, seed=2)
    m2._jac_capture = []
    try:
        with torch.no_grad():
            m2(inp, labels=None, slot_layout=layout)
    finally:
        caps, m2._jac_capture = m2._jac_capture, None
    assert len(caps) >= 2, f"forced depth 2 must capture two pass entries, got {len(caps)}"
    h0, h1 = caps[0]["h"].float(), caps[1]["h"].float()
    n0 = h0.flatten(2).norm(dim=2)
    n1 = h1.flatten(2).norm(dim=2)
    valid = n0 > 0
    changed = valid & ((h1 - h0).flatten(2).abs().amax(dim=2) > 1e-6)
    assert changed.float().sum() >= 0.3 * valid.float().sum(), (
        f"pass 0 changed {int(changed.sum())} of {int(valid.sum())} valid cells; the "
        "renorm assertion below would be vacuous")
    rel = ((n1 - n0).abs() / n0.clamp(min=1e-6))[changed]
    assert rel.max() < 1e-3, (
        f"slot_state_renorm: a changed cell's exit norm differs from its entry norm by "
        f"{rel.max().item():.2e} relative (max over {int(changed.sum())} cells)")


def test_coda_prefix_reach_prev_reads_only_the_previous_slots_k_cells():
    """`tg_strict_allow(layout, "coda", coda_prefix_reach="prev")`
    (`morph/model/tul_layout.py`): a TOKEN query at bag i may read a prefix-cell key at
    bag j iff `bag_j == bag_i - 1` (AND `bag_i < max_slots`, the dump-bin guard) — this
    is a BAG-level condition, so at `prefix_k=4` (== `fan_k`/`slot_cells`) it reads ALL
    FOUR prefix positions of slot k and NONE of slot k-1's, regardless of which cell
    index within the slot. Asserted directly on the real packed layout, no model.
    """
    ids = _ids(seed=1)
    spec = _spec(prefix_k=4)
    _, _, layout, _ = slot_layout_from_ids(ids, _rule(), spec)
    allow = tg_strict_allow(layout, "coda", coda_prefix_reach="prev")[0, 0]   # [L, L]
    bag_id = layout.bag_id[0]
    slot_mask = layout.slot_mask[0]

    # Find a bag k with >=1 non-dump-bin token in span k+1, and slot k's own prefix
    # cells (>=1) and slot k-1's prefix cells (>=1) both present, so the assertion is
    # not vacuous.
    max_slots = layout.max_slots
    found = False
    for k in range(1, int(bag_id.max().item())):
        tok_kplus1 = ((bag_id == k + 1) & ~slot_mask).nonzero(as_tuple=True)[0]
        cells_k = ((bag_id == k) & slot_mask).nonzero(as_tuple=True)[0]
        cells_km1 = ((bag_id == k - 1) & slot_mask).nonzero(as_tuple=True)[0]
        if tok_kplus1.numel() == 0 or cells_k.numel() == 0 or cells_km1.numel() == 0:
            continue
        if k + 1 >= max_slots:          # dump-bin guard in the code; skip it here too
            continue
        found = True
        q = int(tok_kplus1[0])
        for j in cells_k.tolist():
            assert bool(allow[q, j]), \
                f"token {q} (span {k+1}) must read cell {j} of slot {k} (prev)"
        for j in cells_km1.tolist():
            assert not bool(allow[q, j]), \
                f"token {q} (span {k+1}) must NOT read cell {j} of slot {k-1} (prev)"
        break
    assert found, "fixture must contain a span k+1 with both slot k and slot k-1 prefix cells"


def test_coda_prefix_reach_all_is_the_control():
    """The same query, under `coda_prefix_reach="all"`, MUST read both slot k and slot
    k-1's cells — proves the "prev" restriction above is doing something, not merely
    matching an already-narrow relation."""
    ids = _ids(seed=1)
    spec = _spec(prefix_k=4)
    _, _, layout, _ = slot_layout_from_ids(ids, _rule(), spec)
    allow = tg_strict_allow(layout, "coda", coda_prefix_reach="all")[0, 0]
    bag_id = layout.bag_id[0]
    slot_mask = layout.slot_mask[0]
    max_slots = layout.max_slots

    for k in range(1, int(bag_id.max().item())):
        tok_kplus1 = ((bag_id == k + 1) & ~slot_mask).nonzero(as_tuple=True)[0]
        cells_km1 = ((bag_id == k - 1) & slot_mask).nonzero(as_tuple=True)[0]
        if tok_kplus1.numel() == 0 or cells_km1.numel() == 0 or k + 1 >= max_slots:
            continue
        q = int(tok_kplus1[0])
        assert all(bool(allow[q, j]) for j in cells_km1.tolist()), \
            "under 'all' a later token must still read an EARLIER-than-previous slot"
        return
    pytest.fail("fixture must contain a span k+1 with slot k-1 prefix cells")


# ── 3d. the composed config trains and evals ────────────────────────────────

def test_r1_two_step_train_forward_backward_is_finite():
    ids, inp, lab, layout = _batch(prefix_k=4, seed=2)
    m = _r1_model(seed=11).train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-4)
    for step in range(2):
        opt.zero_grad()
        out = m(inp, labels=lab, slot_layout=layout)
        loss = out["loss"]
        assert torch.isfinite(loss), f"step {step}: loss is not finite ({loss})"
        loss.backward()
        n_finite_grads = 0
        for p in m.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "a gradient is not finite"
                n_finite_grads += 1
        assert n_finite_grads > 0, f"step {step}: no parameter received a gradient"
        opt.step()


def test_r1_eval_forward_with_forced_depth_is_finite():
    """`tul.slot_mean_depth` / `tul.slot_max_depth` (TULConfig fields, 0 -> falls back
    to `model.mean_depth` / `model.max_depth`): the depth-sweep script's own forced-depth
    knobs, a one-liner to set here — no separate runner needed."""
    ids, inp, lab, layout = _batch(prefix_k=4, seed=3)
    m = _r1_model(seed=12, tul_kw=dict(slot_mean_depth=3, slot_max_depth=3))
    with torch.no_grad():
        out_loss = m(inp, labels=lab, slot_layout=layout)
        # `labels=None` -> real materialised logits (see the renorm test's own note).
        logits = m(inp, labels=None, slot_layout=layout)["logits"]
    assert torch.isfinite(out_loss["loss"])
    _assert_real_logits_finite(logits, m.cfg.tul.slot_id)





# ── 3e. the in-loop reach is ONE SLOT PER PASS through every op, not only attention ──
#
# Found 2026-09-22 on the Step 1 arm: the register branch handed the core layers
# `tg_relation` alone, so the CCA conv (kernel 4) and the value shift, which are
# position-local on the flattened CELL axis and were "left alone" (a choice that holds
# at reach 0 only), read the previous slot's last cells at EVERY core layer and relayed
# about one slot per LAYER. The planted probe read content three spans back at depth 1
# (0.048) where the single-cell chain read exactly 0. The fix passes a per-SLOT `tg_seg`
# with the relation whenever `loop_reach > 0`; reach 0 keeps its forward.


def _span_logit_delta(m, inp, layout, j: int, g: int) -> float:
    """max |logits| change on span j's tokens when span j-g's token ids are perturbed
    (non-boundary, non-slot ids only, so the packing is unchanged)."""
    from test_tul_fan import DOT
    bag = layout.bag_id[0]
    slot = layout.slot_mask[0]
    tgt = ((bag == j) & ~slot).nonzero(as_tuple=True)[0]
    src = ((bag == j - g) & ~slot).nonzero(as_tuple=True)[0]
    assert tgt.numel() > 0 and src.numel() > 0, (j, g)
    inp2 = inp.clone()
    for p in src.tolist():
        v = int(inp2[0, p])
        if v in (0, 4, DOT, 11):
            continue
        inp2[0, p] = 5 + ((v - 5 + 1) % 40)
    with torch.no_grad():
        a = m(inp, labels=None, slot_layout=layout)["logits"][0, tgt]
        b = m(inp2, labels=None, slot_layout=layout)["logits"][0, tgt]
    d = a - b
    d[:, m.cfg.tul.slot_id] = 0.0            # the masked slot-id column is -inf on both
    d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
    return float(d.abs().max())


@pytest.mark.parametrize("depth,first_dark", [(1, 3), (2, 4)])
@pytest.mark.parametrize("loop_carry", ["none", "persist"])
def test_loop_reach1_register_reaches_exactly_one_slot_per_pass(loop_carry, depth,
                                                                 first_dark):
    """Strict geometry, coda prefix reach prev, forced depth T: a token of span j reads
    its own span and slot j-1's cells; slot j-1's cells hold spans j-1 .. j-1-T after T
    in-loop hops. So span j-(T+1) moves span j's logits and span j-(T+2) moves them by
    EXACTLY 0, with the shipped conv kernel (4) and value shift in place.

    Parametrized over ``tul.loop_carry`` (``"none"`` / ``"persist"``, LXTUL-R Step 2):
    the reach is a property of the MAP, and ``persist`` never touches the map (its whole
    accumulator is added once, at the loop's exit, after every in-loop hop has already
    happened) — so the reach must be exactly the same under both.
    """
    m = _r1_model(seed=12, tul_kw=dict(slot_mean_depth=depth, slot_max_depth=depth,
                                       loop_carry=loop_carry))
    ids, inp, lab, layout = _batch(prefix_k=4, seed=3)
    nb = int(layout.bag_id[0].max())
    j = min(nb - 1, 7)
    lit = _span_logit_delta(m, inp, layout, j, first_dark - 1)
    dark = _span_logit_delta(m, inp, layout, j, first_dark)
    assert lit > 1e-4, f"depth {depth}: span j-{first_dark - 1} must reach span j, got {lit}"
    assert dark == 0.0, f"depth {depth}: span j-{first_dark} must NOT reach span j, got {dark}"


def _core_kwargs_seen(m, inp, layout) -> list:
    seen = []
    real = m._apply_core_step

    def spy(*args, attn_kw=None, **kw):
        if attn_kw is not None:
            seen.append(attn_kw)
        return real(*args, attn_kw=attn_kw, **kw)

    m._apply_core_step = spy
    try:
        with torch.no_grad():
            m(inp, labels=None, slot_layout=layout)
    finally:
        m._apply_core_step = real
    return seen


def test_loop_reach1_register_core_kwargs_carry_a_per_slot_segment_and_reach0_does_not():
    ids, inp, lab, layout = _batch(prefix_k=4, seed=3)
    m1 = _r1_model(seed=12, tul_kw=dict(slot_mean_depth=1, slot_max_depth=1))
    seen1 = _core_kwargs_seen(m1, inp, layout)
    assert seen1, "the core must run at least one pass"
    for kws in seen1:
        for kw in kws:
            assert "tg_relation" in kw and "tg_seg" in kw
            seg = kw["tg_seg"]
            M = m1.cfg.tul.slot_cells
            assert seg.shape[1] % M == 0
            assert torch.equal(seg[0], torch.arange(seg.shape[1] // M).repeat_interleave(M))
    m0 = _r1_model(seed=12, tul_kw=dict(slot_mean_depth=1, slot_max_depth=1, loop_reach=0,
                                        tg_coda_prefix_reach="all"))
    seen0 = _core_kwargs_seen(m0, inp, layout)
    assert seen0
    for kws in seen0:
        for kw in kws:
            assert "tg_relation" in kw and "tg_seg" not in kw, \
                "reach 0 must keep the filed register arms' forward (no tg_seg)"


# ── LXTUL-R Step 2: tul.loop_carry="persist" on the shipped fan4-all-reach1 geometry ──
#
# The decay fix: a per-cell state that accumulates the SAME cross-cell read `sum` does,
# but never hands it back into the loop — the whole accumulator is added once, at the
# loop's exit. `tests/test_tul_loop_carry.py` proves the mechanism on the small single-
# cell fixture; these three prove it composes with the SHIPPED register geometry
# (fan_k 4, fan_mix "all", loop_reach 1, tg_coda_prefix_reach "prev",
# slot_state_renorm True, core_fixed_point_lambda 0) end to end.


def test_persist_composes_with_the_shipped_fan4_all_reach1_geometry():
    m = _r1_model(seed=12, tul_kw=dict(loop_carry="persist"))
    assert m.cfg.tul.loop_carry == "persist"
    assert m.cfg.tul.fan_k == 4 and m.cfg.tul.slot_cells == 4
    assert m.cfg.tul.loop_reach == 1
    assert m.tul_carry is not None and m.tul_carry.mode == "persist"


def test_persist_term_differs_with_depth():
    """The accumulated read is a SUM over the realised passes, so the exit's persist term
    — and therefore the logits — must actually depend on how many passes ran."""
    ids, inp, lab, layout = _batch(prefix_k=4, seed=3)
    m1 = _r1_model(seed=12, tul_kw=dict(slot_mean_depth=1, slot_max_depth=1,
                                        loop_carry="persist"))
    m3 = _r1_model(seed=12, tul_kw=dict(slot_mean_depth=3, slot_max_depth=3,
                                        loop_carry="persist"))
    with torch.no_grad():
        l1 = m1(inp, labels=None, slot_layout=layout)["logits"]
        l3 = m3(inp, labels=None, slot_layout=layout)["logits"]
    d = torch.where(torch.isfinite(l1) & torch.isfinite(l3), l1 - l3,
                    torch.zeros_like(l1))
    assert float(d.abs().max()) > 0.0, \
        "depth 1 and depth 3 persist exits must differ — the term is not depth-blind"


def test_persist_shipped_composition_training_step_is_finite():
    ids, inp, lab, layout = _batch(prefix_k=4, seed=2)
    m = _r1_model(seed=11, tul_kw=dict(loop_carry="persist")).train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-4)
    out = m(inp, labels=lab, slot_layout=layout)
    loss = out["loss"]
    assert torch.isfinite(loss), f"loss is not finite ({loss})"
    loss.backward()
    n_finite_grads = 0
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "a gradient is not finite"
            n_finite_grads += 1
    assert n_finite_grads > 0, "no parameter received a gradient"
    opt.step()


# ── the composed config (tul_slot_spandec_strict_fan4_all_reach1_persist.yaml) ───────

_CONFIG_DIR = __import__("os").path.abspath("morph/configs")


class _StubTok:
    """Enough of a tokenizer for `build_tul_runtime`'s id resolution, no download."""

    @staticmethod
    def from_pretrained(_name):
        return _StubTok()

    def convert_tokens_to_ids(self, _tok):
        return 4


def test_the_persist_config_composes_through_hydra_and_build_tul_runtime(monkeypatch):
    """`tul_slot_spandec_strict_fan4_all_reach1_persist.yaml` — Step 2 of the reach
    composition, `tul.loop_carry: persist` over the shipped Step 1 arm — composes through
    Hydra, maps through the SHIPPED `build_tul_runtime` key mapping (the
    `test_tul_strict_geometry.py` pattern, no tokenizer download) and resolves the three
    knobs the design record names."""
    import transformers
    from hydra import compose, initialize_config_dir

    from morph.training import tul_setup

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubTok)
    monkeypatch.setattr(
        tul_setup, "build_boundary_rule",
        lambda cfg, cache_dir="": (_rule(), _rule().is_boundary, 0, ("\n",)))
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        cfg = compose(config_name="tul_slot_spandec_strict_fan4_all_reach1_persist")
    rt = tul_setup.build_tul_runtime(cfg)
    assert rt is not None, "tul is off — the arm would run the plain model"
    tc = rt.model_cfg
    assert tc.loop_carry == "persist"
    assert tc.loop_reach == 1
    assert tc.fan_k == 4
    assert tc.tg_coda_prefix_reach == "prev"
    assert cfg.wandb.name == "slot-spandec-strict-fan4-all-reach1-persist"


# ── 2026-09-22 correction: the READER fix (persist on a register excludes siblings) ──
#
# On a Thought Register (`slot_cells > 1`) core layer 0's window branch runs under
# `_mask0` — own slot's OTHER cells (every one, XSA excludes only literal self) PLUS
# slot k-1's cells. The ORDINARY `tg_win_capture` (what `sum`/`gate` would use, and what
# is refused there — morph/model/tul.py) would therefore mix a slot's M cells toward
# each other through the persist accumulator: a READER problem the repulsion terms
# (which read the raw `db_traj`, never this capture) cannot see. The fix: persist reads a
# SEPARATE relation, `cross = blk & ~same` (own slot excluded ENTIRELY, not just self),
# through one extra layer-0 attention call fired inside `_apply_core_step`
# (`"tg_persist_capture"`), WITH grad. These tests prove, in order: (0) at `slot_cells
# == 1` — where the new mechanism is NOT used — `cross` would equal the relation the OLD
# `tg_win_capture` path already captures under, so the fix is not a special case that
# only happens to work on the register shape; (a)/(b) on the shipped fan4-all-reach1
# geometry at forced depth 1, a SIBLING cell's perturbed SEED (`core_init(e)`'s output at
# that one compact-axis position — the only way to isolate one cell's identity, since all
# M cells of a slot pool the SAME span and an input-token edit cannot separate them)
# leaves cell i's captured read, and therefore the persist TERM added at the exit,
# bit-identical; a slot k-1 perturbation moves both. (c): the full six-file gate is
# re-run at the end of the report.


def test_cross_relation_at_m_cells_1_equals_the_old_captures_relation():
    """`cross = blk & ~same`, evaluated at `m_cells = 1` (where persist does NOT build
    this mechanism — the single-cell arm stays on the ordinary `tg_win_capture` path,
    `_tul_core`), must equal "causal, self excluded": the relation the OLD single-cell
    capture already runs under (its `tg_relation`/`tg_allow` mask includes the diagonal,
    but the window branch's XSA excludes literal self regardless of the mask — so the
    EFFECTIVE relation of the old path is exactly this). This is what makes "cross == blk
    minus self at m_cells == 1" a proven equality and not a coincidence of the register
    shape.
    """
    n_slots, reach = 6, 1
    blk, same = slot_cell_relation(n_slots, 1, torch.device("cpu"), reach=reach)
    cross = blk & ~same
    idx = torch.arange(n_slots)
    ii, jj = idx.unsqueeze(1), idx.unsqueeze(0)
    old_effective = (ii >= jj) & (jj >= ii - reach) & (ii != jj)
    assert torch.equal(cross[0, 0], old_effective)


def _core_init_seed_perturb_hook(cell_idx: int, scale: float = 5.0):
    """A one-shot forward hook on `core_init` that adds a large, deterministic
    perturbation to ONE compact-axis cell's `h_0 = core_init(e)` — the "seed" every pass
    starts from — and to nothing else. Perturbing an INPUT TOKEN cannot isolate one
    cell's identity (all M cells of a slot pool the same span); the seed is the earliest
    point one specific cell's own state can be moved alone.
    """
    gen = torch.Generator().manual_seed(0)

    def hook(module, inputs, output):
        out = output.clone()
        pert = torch.randn(out.shape[-1], generator=gen) * scale
        out[:, cell_idx] = out[:, cell_idx] + pert.to(out.dtype)
        return out
    return hook


def _reads(m, inp, layout):
    m._carry_capture = []
    try:
        with torch.no_grad():
            m(inp, labels=None, slot_layout=layout)
        return list(m._carry_capture)
    finally:
        m._carry_capture = None


def _tul_core_exit_h(m, inp, layout):
    """Spy on `_tul_core`; return its raw return value `h` (the second tuple entry)."""
    real = m._tul_core
    captured: dict = {}

    def spy(*a, **kw):
        ret = real(*a, **kw)
        captured["h"] = ret[1]
        return ret

    m._tul_core = spy
    try:
        with torch.no_grad():
            m(inp, labels=None, slot_layout=layout)
        return captured["h"]
    finally:
        m._tul_core = real


def test_persist_register_capture_is_blind_to_siblings_and_reads_the_previous_slot():
    """(a) + (b) of the 2026-09-22 correction, on the shipped fan4-all-reach1 geometry,
    forced depth 1."""
    from morph.model.tul_carry import carry_rms_match

    kw = dict(slot_mean_depth=1, slot_max_depth=1)
    m = _r1_model(seed=12, tul_kw=dict(loop_carry="persist", **kw))
    m_none = _r1_model(seed=12, tul_kw=dict(**kw))     # same seed/weights, no carry —
                                                        # the raw-exit reference (b) needs
    ids, inp, lab, layout = _batch(prefix_k=4, seed=3)
    M = m.cfg.tul.slot_cells
    valid = layout.slot_valid[0]

    k = None
    for cand in range(1, int(layout.bag_id[0].max())):
        if cand < valid.shape[0] and bool(valid[cand]) and bool(valid[cand - 1]):
            k = cand
            break
    assert k is not None, "fixture must contain two consecutive valid slots"
    target_cell = k * M + 0
    sibling_cell = k * M + 1        # SAME slot, a different stream — must NOT leak
    prev_cell = (k - 1) * M + 0     # the PREVIOUS slot — must reach the capture

    def reads(perturb_idx):
        handle = (m.core_init.register_forward_hook(_core_init_seed_perturb_hook(perturb_idx))
                  if perturb_idx is not None else None)
        try:
            return _reads(m, inp, layout)
        finally:
            if handle is not None:
                handle.remove()

    def exit_h(perturb_idx):
        handle = (m_none.core_init.register_forward_hook(
                      _core_init_seed_perturb_hook(perturb_idx))
                  if perturb_idx is not None else None)
        try:
            return _tul_core_exit_h(m_none, inp, layout)
        finally:
            if handle is not None:
                handle.remove()

    recs_base, recs_sib, recs_prev = reads(None), reads(sibling_cell), reads(prev_cell)
    assert len(recs_base) == len(recs_sib) == len(recs_prev) == 1, "forced depth 1"

    r_base = recs_base[0]["read"][:, target_cell]
    r_sib = recs_sib[0]["read"][:, target_cell]
    r_prev = recs_prev[0]["read"][:, target_cell]
    assert torch.equal(r_base, r_sib), \
        "(a) a SIBLING (same slot, different stream) perturbation changed cell i's read"
    assert not torch.equal(r_base, r_prev), \
        "(a) a slot k-1 perturbation left cell i's read unchanged — the fixture is inert"

    # (b) THE TERM, not the raw exit. The real (unmodified) layer-0 step still mixes
    # siblings under `_mask0` — cell i's own exit state legitimately moves — but its
    # NORM is pinned by `model.slot_state_renorm` to the entry norm `core_init(e)` set,
    # which a sibling-only perturbation never touches. `carry_rms_match`'s scale is
    # exactly that norm ratio, so the TERM (unlike the raw exit tensor) is unaffected.
    h_base, h_sib = exit_h(None), exit_h(sibling_cell)
    c_base, c_sib = recs_base[0]["carry"], recs_sib[0]["carry"]
    # `carry_rms_match` is elementwise on the cell axis (no cross-cell mixing — read the
    # docstring), so comparing the FULL tensor here is the wrong claim: the sibling's OWN
    # accumulated cell (its query row moved, since the perturbation sits on the sibling's
    # own entry state) and every slot >= k+1 within reach of slot k (which now attends a
    # perturbed sibling exit) legitimately differ. The claim under test is narrower and is
    # exactly what (a) already established at the read level: cell i's OWN accumulated
    # carry — built only from `cross` = slot k-1's cells — must not move.
    assert torch.equal(c_base[:, target_cell], c_sib[:, target_cell]), \
        "the accumulated carry at cell i itself must be sibling-blind"
    term_base = carry_rms_match(c_base, h_base)[:, target_cell]
    term_sib = carry_rms_match(c_sib, h_sib)[:, target_cell]
    assert torch.allclose(term_base, term_sib, atol=1e-5), \
        "(b) the persist exit term for cell i moved under a sibling-only perturbation"
