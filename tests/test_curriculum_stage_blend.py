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


# ── held-out validation (arc E16): the holdout guard, rewind(), the shard builder ──────
def _holdout_shard(root: str, name: str, first_id: int, doc_len: int, n_docs: int) -> None:
    """Like _shard but every doc is DISTINCT (doc i is the constant first_id + i), so a
    replayed stream is told apart from a continued one, and split = eval_holdout."""
    d = os.path.join(root, name)
    os.makedirs(d)
    toks: list[int] = []
    lens: list[int] = []
    for i in range(n_docs):
        body = [first_id + i] * (doc_len - 1) + [0]
        toks.extend(body)
        lens.append(len(body))
    np.asarray(toks, dtype=np.uint16).tofile(os.path.join(d, "tokens.u16.bin"))
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    np.save(os.path.join(d, "doc_offsets.i64.npy"), off)
    np.save(os.path.join(d, "doc_lens.i32.npy"), np.asarray(lens, dtype=np.int32))
    json.dump({"eos_id": 0, "role": "pretrain_bulk", "split": "eval_holdout"},
              open(os.path.join(d, "meta.json"), "w"))


def _drain(loader, n: int, batch: int = 2) -> list[list[int]]:
    it = loader.batches(batch, bag_size=0)
    return [next(it)[0].flatten().tolist() for _ in range(n)]


def test_a_holdout_shard_is_refused_in_a_training_blend(tmp_path):
    _holdout_shard(str(tmp_path), "held", 1, 20, 12)
    with pytest.raises(RuntimeError, match="eval-holdout shard"):
        MultiSourceCurriculumLoader(str(tmp_path), {"held": 1.0}, [64], seed=0,
                                    allowed_roles=["pretrain_bulk"])


def test_a_holdout_loader_refuses_a_training_shard(two_sources):
    with pytest.raises(RuntimeError, match="not an eval-holdout shard"):
        MultiSourceCurriculumLoader(two_sources, {"alpha": 1.0}, [64], seed=0,
                                    allowed_roles=["pretrain_bulk"], holdout=True)


def test_rewind_replays_the_same_batches_and_matches_a_fresh_loader(tmp_path, capsys):
    _holdout_shard(str(tmp_path), "held", 1, 20, 12)
    kw = dict(seed=5, allowed_roles=["pretrain_bulk"], holdout=True, verbose=False)
    a = MultiSourceCurriculumLoader(str(tmp_path), {"held": 1.0}, [64, 64], **kw)
    a.set_stage(1)
    a.rewind()
    first = _drain(a, 3)
    a.rewind()
    assert _drain(a, 3) == first                         # a rewind replays the stream
    b = MultiSourceCurriculumLoader(str(tmp_path), {"held": 1.0}, [64, 64], **kw)
    b.set_stage(1)
    b.rewind()
    assert _drain(b, 3) == first                         # and so does a second loader
    assert sum(a.realized_token_fractions().values()) == pytest.approx(1.0)
    a.rewind()
    assert all(v == 0.0 for v in a.realized_token_fractions().values())
    assert "[curriculum] stage" not in capsys.readouterr().out   # verbose=False is silent


def test_rewind_is_not_a_plain_continue(tmp_path):
    _holdout_shard(str(tmp_path), "held", 1, 20, 12)     # 12 docs x 20 = 240 tokens
    from morph.training.data_placement import DataRuntimeConfig
    a = MultiSourceCurriculumLoader(str(tmp_path), {"held": 1.0}, [64], seed=5,
                                    allowed_roles=["pretrain_bulk"], holdout=True,
                                    verbose=False,
                                    data_runtime=DataRuntimeConfig(prefetch_batches=0))
    a.rewind()
    first = _drain(a, 1)
    cont = _drain(a, 1)                                  # the stream continues: new docs
    assert cont != first
    a.rewind()
    assert _drain(a, 1) == first


def test_holdout_shard_builder_round_trips_through_a_holdout_loader(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "olympiad_holdout_shard",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "olympiad_holdout_shard.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    docs = [{"stage": "2.1", "input_ids": [5, 6, 7, 0]},
            {"stage": "11.3", "input_ids": [8, 9]},        # no EOS: the builder appends it
            {"stage": "2.1", "input_ids": [10, 11, 12, 13, 0]}]
    jl = tmp_path / "eval_holdout.jsonl"
    jl.write_text("".join(json.dumps(d) + "\n" for d in docs))
    out = tmp_path / "bands" / "holdout"
    meta = mod.build_holdout_shard([str(jl)], str(out), "holdout")
    assert meta["split"] == "eval_holdout" and meta["n_docs"] == 3 and meta["n_tokens"] == 12
    assert meta["stages"] == ["2.1", "11.3"] and meta["max_len"] == 5
    lens = np.load(out / "doc_lens.i32.npy").tolist()
    assert lens == [4, 3, 5]
    toks = np.fromfile(out / "tokens.u16.bin", dtype=np.uint16).tolist()
    assert toks == [5, 6, 7, 0, 8, 9, 0, 10, 11, 12, 13, 0]
    ld = MultiSourceCurriculumLoader(str(tmp_path / "bands"), {"holdout": 1.0}, [8], seed=0,
                                     allowed_roles=["reasoning_midtrain"], holdout=True,
                                     verbose=False)
    x, y = next(ld.batches(1, bag_size=0))
    assert x.shape == (1, 8) and y.shape == (1, 8)
    assert set(x.flatten().tolist()) <= set(toks)
    with pytest.raises(RuntimeError, match="eval-holdout shard"):
        MultiSourceCurriculumLoader(str(tmp_path / "bands"), {"holdout": 1.0}, [8], seed=0,
                                    allowed_roles=["reasoning_midtrain"])
