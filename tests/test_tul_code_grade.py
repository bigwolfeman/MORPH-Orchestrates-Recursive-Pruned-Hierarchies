"""``tul.code_grade`` — the graded-continuation target (spec §17.2), one test per invariant.

    PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_tul_code_grade.py -q

CPU only, fp32, the strict-geometry tiny fixture (tests/test_tul_strict_geometry.py), whose
rows cut a boundary every 8 tokens — so ``code_grade_tokens=8`` covers a whole span and the
eligibility filter admits every slot with a next span.

What this file pins:

* **Off is the code-only arm.** ``code_grade: false`` builds no parameter and, on a step the
  schedule does not grade, an ON model's loss and gradients equal an OFF model's exactly.
* **The candidates are real.** Sampled token ids from the frozen coda, written into exactly
  the graded spans' token windows, every position of each window written (so no true token
  of the span being graded survives in the row the grader reads) and nothing else touched.
* **The grader is blind.** Under ``coda_past`` the graded slot's OWN cell is zeroed, so
  perturbing it cannot move a grade — and the same perturbation DOES move the grade when the
  cell is left in, which is what makes the first half of that statement worth asserting.
* **The term trains the loop and nothing else.** E and the coda take exactly zero gradient;
  the projection and the core take a non-zero one.
* **The loss reconciles**, the mode and both RNG streams are restored, and the two configs
  compose, build and run.
"""

from __future__ import annotations

import pytest
import torch

from test_tul_strict_geometry import _pack, _runtime, _tiny, _tul  # noqa: E402

from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_code import code_grade_distinct2, code_grade_pref_loss

GRADE_CONFIGS = ["tul_code_grade", "tul_code_grade_l2"]


def _model(seed: int = 5, on: bool = True, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul_kw = {k[4:]: v for k, v in kw.items() if k.startswith("tul_")}
    cfg_kw = {k: v for k, v in kw.items() if not k.startswith("tul_")}
    base_tul = dict(tg_geometry="strict", code_target=True, code_target_skip_coda=True,
                    code_target_weight=0.0 if on else 1.0, code_grade=on)
    if on:
        base_tul.update(code_grade_k=3, code_grade_tokens=8, code_grade_rows=2,
                        code_grade_every=1)
    base_tul.update(tul_kw)
    base = dict(tul=_tul(**base_tul), n_core=2, mean_depth=3, max_depth=4, bptt_depth=4,
                retention=False, dropout=0.0, core_fixed_point_lambda=0.0, ckpt_grad_iters=0)
    base.update(cfg_kw)
    return MORPHTransformer(_tiny(**base)).float()


def _run(m: MORPHTransformer, seed: int = 3, step: int = 0, **kw):
    _ids, inp, lab, layout = _pack()
    m.code_grade_step.fill_(int(step))
    torch.manual_seed(seed)
    return m(inp, labels=lab, slot_layout=layout, **kw)


def _grads(m: MORPHTransformer, prefix: str) -> float:
    return sum(float(p.grad.abs().sum()) for n, p in m.named_parameters()
               if n.startswith(prefix) and p.grad is not None)


# ── off is the code-only arm ─────────────────────────────────────────────────

def test_off_builds_no_parameter_and_shares_every_weight():
    on, off = _model(on=True), _model(on=False)
    assert on.cfg.tul.code_grade and not off.cfg.tul.code_grade
    off_p = dict(off.named_parameters())
    assert set(off_p) == set(dict(on.named_parameters())), "the graded term adds no parameter"
    for n, p in on.named_parameters():
        assert torch.equal(p, off_p[n]), f"{n} differs: the graded term drew from the RNG"


def test_a_step_the_schedule_does_not_grade_is_the_code_only_step():
    """The gate is the whole difference on an ungraded step: same loss, same gradients."""
    on = _model(on=True, tul_code_grade_every=4, tul_code_target_weight=1.0)
    off = _model(on=False)
    o_on = _run(on, step=1)          # 1 % 4 != 0 -> not graded
    o_off = _run(off, step=1)
    assert "code_grade" not in o_on, "an ungraded step must expose no graded reading"
    assert torch.equal(o_on["loss"], o_off["loss"])
    o_on["loss"].backward()
    o_off["loss"].backward()
    for (n, p), q in zip(on.named_parameters(), off.parameters()):
        if p.grad is None:
            assert q.grad is None, n
        else:
            assert torch.equal(p.grad, q.grad), n
    # and the SAME model on a graded step is different (the gate is not inert)
    on.zero_grad(set_to_none=True)
    o_g = _run(on, step=4)
    assert "code_grade" in o_g and float(o_g["code_grade_n"]) > 0
    assert not torch.equal(o_g["loss"], o_off["loss"])


def test_refusals():
    ok = dict(prefix_k=2, slot_id=4, tg_restrict=True, tg_geometry="strict",
              code_target=True, code_grade=True)
    with pytest.raises(ValueError, match="silently ignored"):
        TULConfig(prefix_k=2, slot_id=4, code_grade_k=3)
    with pytest.raises(NotImplementedError, match="needs tul.code_target"):
        TULConfig(prefix_k=2, slot_id=4, tg_restrict=True, tg_geometry="strict",
                  code_grade=True)
    with pytest.raises(ValueError, match="code_grade_k must be >= 2"):
        TULConfig(**ok, code_grade_k=1)
    with pytest.raises(ValueError, match="code_grade_tokens must be >= 2"):
        TULConfig(**ok, code_grade_tokens=1)
    with pytest.raises(ValueError, match="code_grade_rows must be >= 1"):
        TULConfig(**ok, code_grade_rows=0)
    with pytest.raises(ValueError, match="code_grade_every must be >= 1"):
        TULConfig(**ok, code_grade_every=0)
    with pytest.raises(ValueError, match="code_grade_weight must be > 0"):
        TULConfig(**ok, code_grade_weight=0.0)
    with pytest.raises(ValueError, match="code_grade_temp must be > 0"):
        TULConfig(**ok, code_grade_temp=0.0)
    with pytest.raises(ValueError, match="code_grade_loss must be"):
        TULConfig(**ok, code_grade_loss="margin")
    with pytest.raises(ValueError, match="code_grade_grader must be"):
        TULConfig(**ok, code_grade_grader="coda_own")
    # the code-only refusal is LIFTED by the graded term and only by it
    with pytest.raises(ValueError, match="nothing would train"):
        TULConfig(prefix_k=2, slot_id=4, tg_restrict=True, tg_geometry="strict",
                  code_target=True, code_target_skip_coda=True, code_target_weight=0.0)
    TULConfig(**ok, code_target_skip_coda=True, code_target_weight=0.0)


# ── the candidates are real, and they cover the graded spans exactly ─────────

def _capture(m: MORPHTransformer, monkeypatch):
    """Record every (ids, cells) the sampler/grader hands the forward."""
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    orig = MORPHTransformer._code_grade_forward

    def spy(self, ids, lay, cells):
        seen.append((ids.detach().clone(), cells.detach().clone()))
        return orig(self, ids, lay, cells)

    monkeypatch.setattr(MORPHTransformer, "_code_grade_forward", spy)
    return seen


def _grading_passes(seen, m):
    """The passes that GRADE the candidates. The call order is: J decode passes over the
    R·K candidate rows, then one pass per grader parity over the same rows, then the same
    parities over the R TRUE rows (the `code_grade_true_rank` instrument). So the candidate
    grading passes are the last `2·n_par`-th to the `n_par`-th, and the assertions below
    fail loudly if that order ever changes."""
    n_par = 1 if m.cfg.tul.code_grade_grader == "coda_zero" else 2
    cand = seen[-2 * n_par:-n_par]
    truth = seen[-n_par:]
    n = m.cfg.tul.code_grade_k * min(m.cfg.tul.code_grade_rows, 2)
    assert all(x[0].shape[0] == n for x in cand), "candidate grading pass width changed"
    assert all(x[0].shape[0] == n // m.cfg.tul.code_grade_k for x in truth), \
        "truth grading pass width changed"
    return cand, truth


def test_candidates_are_sampled_tokens_written_into_the_graded_windows(monkeypatch):
    m = _model()
    seen = _capture(m, monkeypatch)
    ids_raw, inp, lab, layout = _pack()
    out = _run(m)
    K, J = m.cfg.tul.code_grade_k, m.cfg.tul.code_grade_tokens
    assert len(seen) >= J, f"expected at least {J} decode passes, saw {len(seen)}"
    final = _grading_passes(seen, m)[0][0][0]             # a candidate grading pass's row
    B, S = layout.slot_valid.shape
    base = inp.repeat_interleave(K, 0)                    # rows 0..B-1 each K times
    assert final.shape == base.shape
    # every position that moved is inside a graded span's token window ...
    n_tok = ((layout.bag_id.unsqueeze(1) == torch.arange(S).view(1, S, 1))
             & (~layout.slot_mask).unsqueeze(1)).sum(-1)
    nxt = torch.zeros_like(n_tok)
    nxt[:, :S - 1] = n_tok[:, 1:]
    first = layout.slot_index + layout.prefix_k
    win = torch.zeros_like(inp, dtype=torch.bool)
    graded = []
    for b in range(B):
        for s in range(S - 1):
            if bool(layout.slot_valid[b, s]) and bool(layout.slot_valid[b, s + 1]) \
                    and 2 <= int(nxt[b, s]) <= J:
                graded.append((b, s))
                win[b, int(first[b, s]):int(first[b, s]) + int(nxt[b, s])] = True
    assert graded, "the fixture produced no eligible slot — the test would assert nothing"
    win_k = win.repeat_interleave(K, 0)
    assert bool((~win_k & (final != base)).sum() == 0), \
        "the sampler wrote outside the graded spans' token windows"
    # ... and every position of every window carries a REAL token id (not the slot id,
    # in range) — that is what makes the grader's row free of the true span.
    vals = final[win_k]
    assert vals.numel() == int(win.sum()) * K
    assert int(vals.min()) >= 0 and int(vals.max()) < m.cfg.vocab_size
    assert not bool((vals == m.cfg.tul.slot_id).any()), "the structural slot id was emitted"
    # the K copies of a row are not all the same draw (temperature 1.0 samples)
    b0, s0 = graded[0]
    w = slice(int(first[b0, s0]), int(first[b0, s0]) + int(nxt[b0, s0]))
    copies = [final[b0 * K + k, w] for k in range(K)]
    assert any(not torch.equal(copies[0], c) for c in copies[1:]), \
        "every candidate of a slot was the same sequence: the sampler is not sampling"
    assert float(out["code_grade_n"]) == float(len(graded))


def test_the_grader_cannot_read_the_graded_slots_own_cell(monkeypatch):
    """Under `coda_past` the pass that grades span s+1 hands the coda a ZERO at cell s —
    the cell that holds E(span s+1). So the grade cannot depend on E(span s+1), however
    much the coda would read that cell if it were there. Both halves are asserted: the
    grade is blind to it, and leaving it in WOULD change the state (so the zeroing is
    load-bearing and the first half is not vacuous)."""
    m = _model()
    seen = _capture(m, monkeypatch)
    _run(m)
    ids_g, cells_g = _grading_passes(seen, m)[0][1]   # the parity-1 candidate grading pass
    lay = m._code_grade_sub_layout(_pack()[3], torch.arange(2), m.cfg.tul.code_grade_k)
    S = lay.slot_valid.shape[1]
    par = 1
    zeroed = (torch.arange(S) % 2) == par
    assert float(cells_g[:, zeroed].abs().sum()) == 0.0, \
        "the parity-1 grading pass did not zero its parity's cells"
    assert float(cells_g[:, ~zeroed].abs().sum()) > 0.0, \
        "it zeroed EVERY cell: `coda_past` collapsed into `coda_zero`"

    # (1) the cells the grader sees do not carry E(span s+1) for the slots it grades
    z = torch.randn_like(cells_g)
    c0 = m._code_grade_cells(z, par, S)
    z2 = z.clone()
    z2[:, zeroed] = torch.randn_like(z2[:, zeroed])       # every GRADED slot's own code
    assert torch.equal(c0, m._code_grade_cells(z2, par, S)), \
        "the graded slots' own codes reached the grader's cells"
    z3 = z.clone()
    z3[:, ~zeroed] = torch.randn_like(z3[:, ~zeroed])     # the CONTEXT codes
    assert not torch.equal(c0, m._code_grade_cells(z3, par, S)), \
        "no code reaches the grader at all: `coda_past` carries no context"

    # (2) end to end: identical states from z and z2, different from the un-zeroed z
    m.eval()
    first = lay.slot_index + lay.prefix_k
    s = int(torch.nonzero(zeroed & lay.slot_valid[0][:S])[1])
    p = int(first[0, s])
    with torch.no_grad():
        h0, _ = m._code_grade_forward(ids_g, lay, c0)
        h2, _ = m._code_grade_forward(ids_g, lay, m._code_grade_cells(z2, par, S))
        assert torch.equal(h0, h2), "the graded slot's own code moved the grader's states"
        h_leak, _ = m._code_grade_forward(ids_g, lay, z)   # cell s NOT zeroed
        assert not torch.equal(h0[0, p], h_leak[0, p]), \
            "cell s does not reach span s+1 at all — zeroing it would guard nothing, and " \
            "this whole grader design would be pointless"


def test_coda_zero_grader_zeroes_everything_and_grades_every_slot_in_one_pass(monkeypatch):
    m = _model(tul_code_grade_grader="coda_zero")
    seen = _capture(m, monkeypatch)
    out = _run(m)
    cand, truth = _grading_passes(seen, m)
    assert len(cand) == 1 and len(truth) == 1, "coda_zero must need ONE pass per row set"
    assert float(cand[0][1].abs().sum()) == 0.0, "coda_zero left a cell in"
    assert float(out["code_grade_n"]) > 0


# ── the gradient routes ─────────────────────────────────────────────────────

def test_the_encoder_and_the_coda_take_zero_gradient_and_the_loop_takes_one():
    m = _model()
    out = _run(m)
    assert float(out["code_grade_n"]) > 0
    out["loss"].backward()
    for n, p in m.named_parameters():
        if n.startswith("tul_code_enc."):
            assert p.grad is None or float(p.grad.abs().sum()) == 0.0, f"E moved: {n}"
    assert _grads(m, "coda.") == 0.0, "the coda took gradient from the graded term"
    assert _grads(m, "tul_code_proj.") > 0.0, "the projection took none"
    assert _grads(m, "core.") > 0.0, "the loop took none"


def test_the_loss_reconciles_with_its_weighted_parts():
    m = _model(tul_code_target_weight=1.0)
    out = _run(m)
    parts = float(out["code_grade_weighted"]) + float(out["code_target_weighted"])
    if out.get("gain_reg_weighted") is not None:
        parts += float(out["gain_reg_weighted"])
    assert abs(float(out["loss"].detach()) - parts) < 1e-5, (float(out["loss"]), parts)
    w = m.cfg.tul.code_grade_weight
    assert abs(float(out["code_grade_weighted"]) - w * float(out["code_grade"])) < 1e-6


def test_the_graded_term_alone_moves_the_loop():
    """`code_target_weight: 0` — the graded term is the only thing with a gradient."""
    m = _model()
    out = _run(m)
    assert float(out["code_target_weighted"]) == 0.0
    out["loss"].backward()
    assert _grads(m, "tul_code_proj.") > 0.0


def test_mode_and_both_rng_streams_are_restored(monkeypatch):
    """The grader runs a dozen eval forwards inside a training step. It must hand the step
    back in TRAIN mode and leave the global RNG streams exactly where it found them, or
    every later draw of the run shifts and the arm stops being comparable to its control.

    The second half is tested against an INJECTED global draw: the eval forward happens to
    consume nothing today, so asserting "the state is unchanged" would pass with the
    save/restore deleted. Forcing a draw inside the grader is what makes it real.
    """
    m = _model()
    _ids, inp, lab, layout = _pack()
    m.train()
    m.code_grade_step.fill_(0)
    torch.manual_seed(11)
    out = m(inp, labels=lab, slot_layout=layout)
    assert float(out["code_grade_n"]) > 0
    assert m.training, "the grader left the module in eval mode"
    assert not m._in_code_grade

    orig = MORPHTransformer._code_grade_logits

    def greedy_with_a_global_draw(self, h, src, live):
        torch.rand(4)                      # a draw from the GLOBAL stream, on purpose
        return orig(self, h, src, live)

    monkeypatch.setattr(MORPHTransformer, "_code_grade_logits", greedy_with_a_global_draw)
    pred = torch.randn(2, layout.slot_valid.shape[1], layout.prefix_k, m.cfg.d_model)
    z = torch.randn_like(pred)
    ok = layout.slot_valid.clone()
    ok[:, -1] = False
    st = torch.random.get_rng_state()
    loss, stats = m._tul_code_grade(pred, z, ok, layout, inp, None,
                                    torch.full(ok.shape, 4))
    assert float(stats["code_grade_n"]) > 0, "the injected draw broke the grader itself"
    assert torch.equal(torch.random.get_rng_state(), st), \
        "a draw inside the grader shifted the global stream: every later draw of the " \
        "training run would move and the arm would not be comparable to its control"


# ── the pure functions ──────────────────────────────────────────────────────

def test_distinct2_reads_repetition():
    tok = torch.tensor([[7, 7, 7, 7, 7, 7],          # one bigram, repeated 5x
                        [1, 2, 3, 4, 5, 6],          # 5 distinct bigrams
                        [1, 2, 1, 2, 1, 2]])         # 2 distinct, alternating
    lens = torch.tensor([6, 6, 6])
    d2 = code_grade_distinct2(tok, lens)
    assert abs(float(d2[0]) - 1 / 5) < 1e-6
    assert abs(float(d2[1]) - 1.0) < 1e-6
    assert abs(float(d2[2]) - 2 / 5) < 1e-6
    # the length is honoured: only the first `lens` tokens are the candidate
    d2b = code_grade_distinct2(tok, torch.tensor([3, 3, 3]))
    assert abs(float(d2b[0]) - 1 / 2) < 1e-6 and abs(float(d2b[1]) - 1.0) < 1e-6


def test_pref_loss_prefers_the_winner():
    torch.manual_seed(0)
    R, K, S, M, C = 1, 4, 3, 2, 8
    z = torch.nn.functional.normalize(torch.randn(R, K, S, M, C), dim=-1) * (C ** 0.5)
    best = torch.tensor([[0, 1, 2]])
    ok = torch.ones(R, S, dtype=torch.bool)
    aligned = z.gather(1, best.view(R, 1, S, 1, 1).expand(R, 1, S, M, C)).squeeze(1)
    l_al, m_al = code_grade_pref_loss(aligned, z, best, ok, 0.1)
    l_wr, m_wr = code_grade_pref_loss(
        z.gather(1, ((best + 1) % K).view(R, 1, S, 1, 1).expand(R, 1, S, M, C)).squeeze(1),
        z, best, ok, 0.1)
    assert float(l_al) < float(l_wr), (float(l_al), float(l_wr))
    assert float(m_al) > 0.0 > float(m_wr)
    # the gradient reaches the prediction, and no valid slot means an exact 0 with a graph
    p = aligned.clone().requires_grad_(True)
    code_grade_pref_loss(p, z, best, ok, 0.1)[0].backward()
    assert float(p.grad.abs().sum()) > 0.0
    p2 = aligned.clone().requires_grad_(True)
    zero, _ = code_grade_pref_loss(p2, z, best, torch.zeros_like(ok), 0.1)
    assert float(zero) == 0.0 and zero.requires_grad


# ── the configs ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", GRADE_CONFIGS)
def test_configs_compose_build_and_run(name, monkeypatch):
    cfg, rt = _runtime(name, monkeypatch)
    tc = rt.model_cfg
    assert tc.code_grade and tc.code_target and tc.code_target_skip_coda
    assert tc.code_grade_loss == "pref" and tc.code_grade_grader == "coda_past"
    assert (tc.code_target_weight == 0.0) == (name == "tul_code_grade")
    torch.manual_seed(2)
    small = _tul(**{k: getattr(tc, k) for k in
                    ("prefix_k", "slot_id", "tg_restrict", "tg_restrict_scope", "tg_geometry",
                     "emit_weight", "token_state_dropout", "mux_beta", "code_target",
                     "code_target_weight", "code_target_skip_coda", "code_target_detach",
                     "code_grade", "code_grade_k", "code_grade_loss", "code_grade_grader",
                     "code_grade_rows", "code_grade_every", "code_grade_tau",
                     "code_grade_temp", "code_grade_min_distinct2")},
                 code_grade_tokens=8)
    m = MORPHTransformer(_tiny(tul=small, n_core=2, mean_depth=3, max_depth=4, bptt_depth=4,
                               retention=False, dropout=0.0, core_fixed_point_lambda=0.0,
                               ckpt_grad_iters=0)).float()
    out = _run(m)
    assert float(out["code_grade_n"]) > 0
    assert 0.0 <= float(out["code_grade_true_rank"]) <= 1.0
    assert float(out["code_grade_best"]) >= float(out["code_grade_worst"])
    out["loss"].backward()
    assert _grads(m, "tul_code_proj.") > 0.0
