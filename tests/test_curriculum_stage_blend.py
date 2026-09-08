"""The curriculum loader's data-curriculum support (arc E15): stages at ONE seq_len share the
length bucket, and a stage may override the blend. One test per contract."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from morph.training.curriculum_data import MultiSourceCurriculumLoader, _Source


def _shard(root: str, name: str, first_id: int, doc_len: int, n_docs: int) -> None:
    d = os.path.join(root, name)
    os.makedirs(d)
    toks = []
    lens = []
    for _ in range(n_docs):
        body = [first_id + (i % 9) for i in range(doc_len - 1)] + [0]
        toks.extend(body)
        lens.append(len(body))
    np.asarray(toks, dtype=np.uint16).tofile(os.path.join(d, "tokens.u16.bin"))
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    np.save(os.path.join(d, "doc_offsets.i64.npy"), off)
    np.save(os.path.join(d, "doc_lens.i32.npy"), np.asarray(lens, dtype=np.int32))
    json.dump({"eos_id": 0, "role": "pretrain_bulk"}, open(os.path.join(d, "meta.json"), "w"))


@pytest.fixture
def two_sources(tmp_path):
    _shard(str(tmp_path), "alpha", 1, 20, 40)      # ids 1..9
    _shard(str(tmp_path), "beta", 10, 30, 40)      # ids 10..18
    return str(tmp_path)


def _source_of(tokens: list[int]) -> set[str]:
    out = set()
    for t in tokens:
        if 1 <= t <= 9:
            out.add("alpha")
        elif 10 <= t <= 18:
            out.add("beta")
    return out


def test_distinct_lengths_keep_the_old_bucket_rule(two_sources):
    s = _Source("alpha", os.path.join(two_sources, "alpha"), 1.0, ["pretrain_bulk"])
    s.lens = np.asarray([10, 16, 17, 100])
    s.assign_stages([16, 64])
    assert s.bucket_of_doc.tolist() == [0, 0, 1, 1]          # first seq_len >= len; top clamps
    assert s.bucket_of_stage == [0, 1]


def test_same_length_stages_share_every_doc(two_sources):
    s = _Source("alpha", os.path.join(two_sources, "alpha"), 1.0, ["pretrain_bulk"])
    s.assign_stages([64, 64, 64])
    assert s.bucket_of_stage == [0, 0, 0]
    assert bool(s.docs_in_stage(2).all())
    rng = np.random.default_rng(0)
    s.build_stage_queue(2, rng)
    assert s.has_stage(2)


def test_two_stages_at_one_length_both_serve_docs(two_sources):
    ld = MultiSourceCurriculumLoader(two_sources, {"alpha": 1.0, "beta": 1.0}, [64, 64], seed=0)
    ld.set_stage(1)                                           # the old rule raised here: no docs
    assert _source_of(ld._fill(512)) == {"alpha", "beta"}


def test_stage_blend_selects_the_sources_per_stage(two_sources):
    ld = MultiSourceCurriculumLoader(two_sources, {"alpha": 1.0, "beta": 1.0}, [64, 64], seed=0,
                                     stage_weights=[{"alpha": 1.0}, {"beta": 1.0}])
    assert _source_of(ld._fill(600)) == {"alpha"}
    ld.set_stage(1)
    assert _source_of(ld._fill(600)) == {"beta"}
    assert ld._probs.tolist() == [1.0]


def test_stage_blend_none_keeps_the_global_weights(two_sources):
    ld = MultiSourceCurriculumLoader(two_sources, {"alpha": 3.0, "beta": 1.0}, [64, 64], seed=0,
                                     stage_weights=[None, {"beta": 1.0}])
    w = np.array([3.0, 1.0]); mean_len = np.array([20.0, 30.0])
    p = w / mean_len
    assert np.allclose(ld._probs, p / p.sum())
    ld.set_stage(1)
    assert [s.name for s in ld._active] == ["beta"]


def test_stage_blend_refuses_unknown_and_empty(two_sources):
    with pytest.raises(ValueError, match="not in the loaded blend"):
        MultiSourceCurriculumLoader(two_sources, {"alpha": 1.0, "beta": 1.0}, [64, 64],
                                    stage_weights=[{"gamma": 1.0}, None])
    with pytest.raises(ValueError, match="no positive weight"):
        MultiSourceCurriculumLoader(two_sources, {"alpha": 1.0, "beta": 1.0}, [64, 64],
                                    stage_weights=[{"alpha": 0.0}, None])
    with pytest.raises(ValueError, match="entries for 2 stages"):
        MultiSourceCurriculumLoader(two_sources, {"alpha": 1.0, "beta": 1.0}, [64, 64],
                                    stage_weights=[None])


def test_draw_stream_is_unchanged_without_stage_weights(two_sources):
    a = MultiSourceCurriculumLoader(two_sources, {"alpha": 1.0, "beta": 1.0}, [64, 128], seed=3)
    b = MultiSourceCurriculumLoader(two_sources, {"alpha": 1.0, "beta": 1.0}, [64, 128], seed=3,
                                    stage_weights=[None, None])
    assert a._fill(800) == b._fill(800)
