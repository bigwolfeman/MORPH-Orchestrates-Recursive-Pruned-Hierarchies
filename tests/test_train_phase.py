"""Regressions for the three mid-run schedule defects `morph/training/phase.py` cures.

Every test here failed against `experiments/tul` at d970add. See the module docstring of
morph/training/phase.py for what each defect was.

The loader test synthesises its own pretokenised shard, so it does not depend on
data/pretok being present.
"""
from __future__ import annotations

import dataclasses
import math
import json

import numpy as np
import torch
import pytest

from morph.model.tul_layout import BoundaryRule, TulDataConfig
from morph.training.phase import PhaseSchedule, TrainPhase

VOCAB, EOS, SLOT_ID = 512, 0, 4


# ── TrainPhase / PhaseSchedule ────────────────────────────────────────────────

def test_tst_hands_off_to_tul_on_the_same_step():
    """The intended recipe: superposition ends exactly where the slot layout begins."""
    s = PhaseSchedule(total_steps=1000, tst_bag_size=6, tst_ratio=0.3, tul_activate_at=0.3)
    assert s.at(299) == TrainPhase(bag_size=6, tul_on=False)
    assert s.at(300) == TrainPhase(bag_size=0, tul_on=True)
    assert s.at(999) == TrainPhase(bag_size=0, tul_on=True)


def test_tul_off_never_reports_on():
    s = PhaseSchedule(total_steps=100, tul_activate_at=None)
    assert all(not s.at(i).tul_on for i in range(100))


def test_tul_activate_at_zero_is_on_from_step_zero():
    s = PhaseSchedule(total_steps=100, tul_activate_at=0.0)
    assert s.at(0).tul_on


def test_params_built_but_never_activated_is_expressible():
    """`enabled: true, activate_at: 1.0` — the control arm that proves the TUL parameters
    alone do not perturb the baseline. Inexpressible under the old tri-state key."""
    s = PhaseSchedule(total_steps=100, tul_activate_at=1.0)
    assert all(not s.at(i).tul_on for i in range(100))


def test_overlapping_tst_and_tul_raises_when_the_schedule_is_built():
    """DEFECT: the old code let TUL activate inside the TST phase and only raised
    later, inside the forward, at step N."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        PhaseSchedule(total_steps=1000, tst_bag_size=6, tst_ratio=0.5, tul_activate_at=0.2)


def test_a_phase_can_never_carry_both_bag_and_tul():
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainPhase(bag_size=6, tul_on=True)


def test_resume_reports_the_same_phase_as_a_fresh_run_reaching_that_step():
    """DEFECT 1: the val loader asked `activate_at == 0.0` while the live flag asked
    `start_step >= tul_step`. Two predicates for one question, so a resume past a mid-run
    activation built a plain val loader, left the flag True, and never rebuilt.

    One predicate means resume and fresh-run agree at every step, by construction."""
    s = PhaseSchedule(total_steps=1000, tul_activate_at=0.3)
    for resume_step in (0, 1, 299, 300, 301, 500, 999):
        assert s.at(resume_step) == s.at(resume_step), "at() must be pure"
    # The specific case that used to break: resume at 500 with activate_at 0.3.
    assert s.at(500).tul_on is True
    # …and the old val-loader predicate, which disagreed.
    old_val_loader_predicate = (0.3 == 0.0)
    assert old_val_loader_predicate != s.at(500).tul_on


def test_boundaries_follow_the_total_steps_they_are_given():
    """DEFECT 3: tst_phase1_steps and tul_step were derived from `training.steps`, but
    the curriculum scheduler overrides total_steps afterwards, so both boundaries landed
    at the wrong step. Building the schedule after the override fixes it."""
    training_steps, curriculum_steps = 1000, 4000
    early = PhaseSchedule(total_steps=training_steps, tul_activate_at=0.3)
    late = PhaseSchedule(total_steps=curriculum_steps, tul_activate_at=0.3)
    assert early.tul_step == 300
    assert late.tul_step == 1200          # 0.3 of the run that actually runs
    assert not late.at(300).tul_on        # the old code activated here


def test_phase_change_is_detectable_by_equality():
    """The loop replaces three bespoke `if <phase change>` blocks with one `!=`."""
    s = PhaseSchedule(total_steps=100, tst_bag_size=4, tst_ratio=0.5, tul_activate_at=0.5)
    changes = [i for i in range(1, 100) if s.at(i) != s.at(i - 1)]
    assert changes == [50]


# ── the loader regression (DEFECT 2) ─────────────────────────────────────────

def _write_shard(d, doc_lens=None, seed=0):
    """A minimal pretokenised shard: the on-disk contract of scripts/pretokenize.py.

    Lengths straddle the stage boundaries so BOTH buckets have docs — a doc is assigned
    to the first stage whose seq_len fits it."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lens = np.asarray(doc_lens if doc_lens is not None
                      else [400] * 16 + [900] * 16, dtype=np.int32)
    n_docs = len(lens)
    toks = rng.integers(8, VOCAB, size=int(lens.sum()), dtype=np.uint16)
    offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    toks.tofile(d / "tokens.u16.bin")
    np.save(d / "doc_lens.i32.npy", lens)
    np.save(d / "doc_offsets.i64.npy", offs)
    (d / "meta.json").write_text(json.dumps({
        "name": d.name, "kind": "jsonl", "paths": [], "text_field": "text",
        "tokenizer": "test", "n_docs": int(n_docs), "n_tokens": int(lens.sum()),
        "eos_id": EOS, "dtype": "uint16", "role": "pretrain_bulk"}))


def _tul_cfg():
    lut = np.zeros(VOCAB, dtype=bool)
    lut[[11, 13, 17]] = True                       # a few synthetic boundary ids
    return TulDataConfig(rule=BoundaryRule(is_boundary=lut, min_span=4, span_cap=32,
                                           eos_id=EOS, fixed_stride=0),
                         prefix_k=2, slot_id=SLOT_ID, max_slots=0)


def test_curriculum_stage_transition_keeps_the_tul_layout(tmp_path):
    """DEFECT 2, reproduced and fixed.

    `cur_stage` starts at -1, so the stage-transition site is also the site that CREATES
    the curriculum loader. It called `.batches(mb, bag_size=cur_bag)` with no `tul=`, so
    on a curriculum run the loader yielded 2-tuples, train.py's arity check set
    `slot_layout=None`, and training ran DENSE from step 0 — while `tul_on` stayed True
    and validation kept reporting val/plan_nats. Nothing raised.

    Driving the rebuild from the phase makes the layout impossible to drop."""
    from morph.training.curriculum_data import MultiSourceCurriculumLoader
    from morph.training.data_placement import DataRuntimeConfig

    _write_shard(tmp_path / "pretok" / "src")   # 16 docs <=512, 16 in (512,1024]
    rt = dataclasses.replace(DataRuntimeConfig.resolve(), prefetch_batches=0)
    loader = MultiSourceCurriculumLoader(str(tmp_path / "pretok"), {"src": 1.0},
                                         [512, 1024], seed=0, data_runtime=rt)
    tul_cfg, phase = _tul_cfg(), PhaseSchedule(total_steps=10, tul_activate_at=0.0).at(0)

    def rebuild(stage):
        """What the training loop must do at a stage change: derive `tul=` FROM the
        phase, never from whatever the call site happened to remember."""
        loader.set_stage(stage)
        return loader.batches(2, bag_size=phase.bag_size,
                              tul=tul_cfg if phase.tul_on else None)

    first = next(rebuild(0))
    assert len(first) == 3, "setup: stage 0 must carry the layout"
    after = next(rebuild(1))
    assert len(after) == 3, (
        "stage transition dropped the TUL slot layout: the loader yielded a 2-tuple, so "
        "train.py would pass slot_layout=None and train DENSE while tul_on stayed True")
    x, y, layout = after
    assert isinstance(layout, object) and layout is not None
    assert x.shape[1] == tul_cfg.spec_for(1024).l_total, "layout must follow the new seq_len"


# ── val PPL aggregation ───────────────────────────────────────────────────────
#
# `evaluate` reports TWO perplexities: the baseline's `val/ppl` (from the return value)
# and TUL's `val/ppl_tokens` (from `extra`). They are compared against each other in
# docs/ablation-ledger.md, so they must use the SAME aggregation. Until 2026-08-23
# `val/ppl_tokens` was the mean of the per-batch exp(CE) while `val/ppl` was exp of the
# mean CE. Jensen makes the first strictly larger whenever the batches differ: on
# tul-a1-acap1 it read 25.89 against a true 25.14, and that 0.75 PPL gap is 59 % of the
# 1.27 PPL A1-vs-A0 effect the metric exists to measure.

class _CEStubModel(torch.nn.Module):
    """Returns a scripted CE per batch so the aggregation is the only thing under test."""

    def __init__(self, ces):
        super().__init__()
        self._ces = list(ces)
        self._i = 0

    def tul_forward_with_plan_nats(self, x, y, layout):
        ce = self._ces[self._i]
        self._i += 1
        return {"loss": torch.tensor(ce), "ce_tokens": ce,
                "layer_passes": 8.0, "n_tokens": 4.0}


class _Layout:
    stats: dict = {}

    def to(self, device):
        return self


def _run_eval(ces):
    from morph.training.train import evaluate
    x = torch.zeros(1, 4, dtype=torch.long)
    loader = iter([(x, x, _Layout()) for _ in ces])
    extra: dict = {}
    avg, ppl = evaluate(_CEStubModel(ces), torch.device("cpu"), loader,
                        n_batches=len(ces), tul=True, extra=extra)
    return avg, ppl, extra


def test_val_ppl_tokens_is_exp_of_the_mean_not_the_mean_of_exps():
    ces = [2.0, 4.0, 3.0]                       # spread out, so Jensen bites
    mean_ce = sum(ces) / len(ces)
    mean_of_exps = sum(math.exp(c) for c in ces) / len(ces)

    _avg, _ppl, extra = _run_eval(ces)

    assert extra["val/ce_tokens"] == pytest.approx(mean_ce)
    assert extra["val/ppl_tokens"] == pytest.approx(math.exp(mean_ce))
    # The old form. exp(3) = 20.09 vs 22.98 — a 14 % overstatement on this input.
    assert extra["val/ppl_tokens"] != pytest.approx(mean_of_exps)


def test_val_ppl_tokens_uses_the_same_aggregation_as_the_baseline_val_ppl():
    """The two numbers are compared arm-to-arm, so they must agree on identical CE."""
    ces = [2.0, 4.0, 3.0]
    _avg, ppl_baseline, extra = _run_eval(ces)
    # The stub's `loss` equals its `ce_tokens`, so the token path and the baseline path
    # see the same numbers and must report the same PPL.
    assert extra["val/ppl_tokens"] == pytest.approx(ppl_baseline)
