"""``tul.code_target_ref`` — the frozen VAE-stage twin every code target is measured
against (spec §17.1), one test per invariant.

    PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_tul_code_ref.py -q

CPU only, fp32, the strict-geometry tiny fixture.

WHY IT EXISTS, measured. `E` is frozen; its INPUT is not. It pools the LIVE prelude's
states of the next span, so on the code-only arm (`train_only: []`, everything trains) the
prelude drifted under the loop's gradient and E's codes collapsed onto one direction: train
own cosine 0.61 / shuffled 0.56 at steps 500-1000 and 0.99 / 0.98 by 2500-3000, regression
loss 0.016, the frozen coda's oracle CE 13.1 nats against arm A's 1.4 (wandb 38naddpq,
killed at step 3500). No gradient ever reached E; its input drifted.

What this file pins:

* **Off is HEAD.** `code_target_ref: false` builds no twin and is bit-identical.
* **The twin is invisible.** Not in `named_parameters`, not in `state_dict`, not in
  `modules()` — so no walk in the tree (ternary QAT, embedding QAT, the CMS prune / carve /
  route walk, the optimizer) can reach it. Frozen and in eval mode.
* **The target stops moving.** Perturb the live prelude and embeddings: `code_z` is
  bit-identical with the twin and changes without it.
* **The grader stops moving.** Perturb the live coda and the tied head: the grades are
  bit-identical with the twin.
* **The snapshot copies the LOADED weights**, and a RESUME keeps the checkpoint's twin
  instead of re-snapshotting the live model.
* **The refusal.** A code-target model whose front trains without the twin is refused,
  with the collapse numbers in the message.
"""

from __future__ import annotations

import copy

import pytest
import torch

from test_tul_strict_geometry import _pack, _tiny, _tul  # noqa: E402

from morph.model.transformer import MORPHTransformer
from morph.training.train import assert_code_target_front_frozen, read_code_ref_state

_FRONT = ("prelude.", "embed.")


def _model(ref: bool = True, seed: int = 5, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul_kw = dict(tg_geometry="strict", code_target=True, code_target_ref=ref)
    tul_kw.update({k[4:]: v for k, v in kw.items() if k.startswith("tul_")})
    m = MORPHTransformer(_tiny(tul=_tul(**tul_kw), n_core=2, mean_depth=3, max_depth=4,
                               bptt_depth=4, retention=False, dropout=0.0,
                               core_fixed_point_lambda=0.0, ckpt_grad_iters=0)).float()
    if ref:
        m.tul_code_ref_snapshot()
    return m


def _perturb(m: MORPHTransformer, prefixes, scale: float = 0.05) -> None:
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.startswith(prefixes):
                p.add_(torch.randn_like(p) * scale)


def _code_z(m: MORPHTransformer):
    _ids, inp, lab, layout = _pack()
    m.eval()
    with torch.no_grad():
        return m(inp, labels=lab, slot_layout=layout)["code_z"].clone()


# ── off is HEAD ─────────────────────────────────────────────────────────────

def test_off_builds_no_twin_and_is_bit_identical():
    on, off = _model(ref=True), _model(ref=False)
    assert on.code_ref is not None and off.code_ref is None
    off_p = dict(off.named_parameters())
    assert set(off_p) == set(dict(on.named_parameters())), "the twin leaked into the tree"
    for n, p in on.named_parameters():
        assert torch.equal(p, off_p[n]), f"{n}: building the twin drew from the RNG"
    _ids, inp, lab, layout = _pack()
    on.train(), off.train()
    torch.manual_seed(3)
    o_on = on(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(3)
    o_off = off(inp, labels=lab, slot_layout=layout)
    # the twin holds the SAME weights as the live model at step 0, so the two agree
    assert torch.equal(o_on["loss"], o_off["loss"])
    assert torch.equal(o_on["code_target_cos"], o_off["code_target_cos"])


def test_the_twin_is_invisible_to_every_walk_and_is_frozen():
    m = _model(ref=True)
    names = set(dict(m.named_parameters()))
    assert not any(n.startswith("tul_code_ref") or "_code_ref" in n for n in names)
    assert not any("_code_ref" in k for k in m.state_dict())
    assert m.code_ref not in list(m.modules()), "the twin is a registered submodule"
    assert sum(p.numel() for p in m.parameters()) \
        == sum(p.numel() for p in _model(ref=False).parameters())
    assert all(not p.requires_grad for p in m.code_ref.parameters())
    assert not m.code_ref.training
    assert m.code_ref.code_ref is None, "the twin holds a twin of its own"
    # an optimizer built the way the trainer builds one cannot see it
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    seen = {id(p) for g in opt.param_groups for p in g["params"]}
    assert not any(id(p) in seen for p in m.code_ref.parameters())


# ── the target stops moving ─────────────────────────────────────────────────

def test_the_target_is_frozen_against_a_drifting_front():
    on, off = _model(ref=True), _model(ref=False)
    z_on0, z_off0 = _code_z(on), _code_z(off)
    _perturb(on, _FRONT)
    _perturb(off, _FRONT)
    assert torch.equal(z_on0, _code_z(on)), \
        "the twin's code moved when the LIVE prelude did"
    z_off1 = _code_z(off)
    assert not torch.equal(z_off0, z_off1), "the control did not drift: the test is vacuous"
    assert float((z_off0 - z_off1).abs().max()) > 0.1


def test_the_grader_and_sampler_read_the_twin_not_the_live_coda():
    """Perturb the live CODA: with the twin the grades and the winner's code are
    bit-identical, without it they move.

    The coda is the only part of the readout that can be isolated on this arm, and the
    reason the others cannot is worth recording. The head is weight-tied to the input table
    (`morph-lm-head-is-weight-tied`) so moving it moves the live prelude, and `_readout`'s
    `final_norm` / `lm_mixer` are what `TULCodeProj` reads — both are upstream of the loop's
    predicted cell, which the candidates are SUPPOSED to be conditioned on. The coda is
    downstream of all of it and, on this arm, runs ONLY inside the sampler and the grader."""
    kw = dict(tul_code_target_skip_coda=True, tul_code_target_weight=0.0,
              tul_code_grade=True, tul_code_grade_k=3, tul_code_grade_tokens=8,
              tul_code_grade_rows=2, tul_code_grade_every=1)
    _ids, inp, lab, layout = _pack()

    def run(m):
        m.train()
        m.code_grade_step.fill_(0)
        torch.manual_seed(3)
        o = m(inp, labels=lab, slot_layout=layout)
        return {k: float(o[k]) for k in o if str(k).startswith("code_grade")}

    on = _model(ref=True, **kw)
    a = run(on)
    assert a["code_grade_n"] > 0
    _perturb(on, ("coda.",), scale=0.2)
    b = run(on)
    for k in ("code_grade_best", "code_grade_mean", "code_grade_worst", "code_grade_true",
              "code_grade_true_rank", "code_grade_cos_best_true"):
        assert a[k] == b[k], f"{k} moved with the live coda: {a[k]} -> {b[k]}"

    off = _model(ref=False, **kw)
    c = run(off)
    _perturb(off, ("coda.",), scale=0.2)
    d = run(off)
    assert any(c[k] != d[k] for k in ("code_grade_best", "code_grade_true")), \
        "the control's grades did not move either: the test is vacuous"


# ── the snapshot and the resume ─────────────────────────────────────────────

def test_the_snapshot_copies_the_loaded_weights():
    m = _model(ref=False)                      # no twin yet
    m.cfg.tul.code_target_ref = True
    seeded = copy.deepcopy(m.state_dict())
    for k in seeded:
        if k.startswith(_FRONT) and seeded[k].is_floating_point():
            seeded[k] = seeded[k] + 1.0
    m.load_state_dict(seeded, strict=True)     # "training.init_from"
    assert m.tul_code_ref_snapshot() == "snapshot"
    ref_sd = m.code_ref.state_dict()
    for k, v in m.state_dict().items():
        assert torch.equal(ref_sd[k], v), f"{k}: the twin is not the LOADED model"


def test_a_resume_keeps_the_checkpoints_twin():
    """Re-snapshotting the live weights at step N would move the target mid-run, which is
    the whole failure this mechanism exists to prevent."""
    m = _model(ref=True)
    saved = copy.deepcopy(m.tul_code_ref_state())
    z_saved = _code_z(m)
    _perturb(m, _FRONT)                        # the live model has trained since
    assert m.tul_code_ref_snapshot(saved) == "resume"
    assert torch.equal(z_saved, _code_z(m)), "the resume re-snapshotted the live weights"
    for k, v in m.code_ref.state_dict().items():
        assert torch.equal(saved[k], v), f"{k}: the checkpoint's twin was not restored"
    # and a checkpoint with no twin (every init_from seed) is the snapshot case
    assert m.tul_code_ref_snapshot(None) == "snapshot"
    assert not torch.equal(z_saved, _code_z(m))


def test_the_checkpoint_carries_the_twin_and_the_reader_finds_it(tmp_path):
    from morph.training.train import save_checkpoint
    m = _model(ref=True)
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, 7, m, opt, torch.amp.GradScaler("cpu", enabled=False), None)
    got = read_code_ref_state(path)
    assert got is not None and set(got) == set(m.tul_code_ref_state())
    for k, v in m.tul_code_ref_state().items():
        assert torch.equal(got[k], v)
    # a checkpoint from a model with no twin carries none, and that is the snapshot case
    m2 = _model(ref=False)
    opt2 = torch.optim.SGD([p for p in m2.parameters() if p.requires_grad], lr=0.1)
    p2 = str(tmp_path / "ck2.pt")
    save_checkpoint(p2, 7, m2, opt2, torch.amp.GradScaler("cpu", enabled=False), None)
    assert read_code_ref_state(p2) is None


# ── the refusal ─────────────────────────────────────────────────────────────

def test_a_trainable_front_without_the_twin_is_refused():
    off = _model(ref=False)
    with pytest.raises(ValueError, match="trainable front and no tul.code_target_ref"):
        assert_code_target_front_frozen(off)
    # arm A's freeze passes: the front is frozen, so the target cannot drift
    for n, p in off.named_parameters():
        if n.startswith(("embed.", "prelude.", "input_norm.", "value_embed")) \
                or (n.startswith("x0_injects.") and int(n.split(".")[1]) < off.cfg.n_prelude):
            p.requires_grad_(False)
    assert assert_code_target_front_frozen(off) == []
    # and the twin lifts it whatever trains
    assert assert_code_target_front_frozen(_model(ref=True)) == []


def test_the_knob_is_refused_off_a_code_target_model():
    from morph.model.tul import TULConfig
    with pytest.raises(ValueError, match="silently ignored"):
        TULConfig(prefix_k=2, slot_id=4, code_target_ref=True)
    with pytest.raises(RuntimeError, match="built without tul.code_target_ref"):
        _model(ref=False).tul_code_ref_snapshot()
