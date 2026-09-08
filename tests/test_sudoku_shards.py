"""Tests for scripts/sudoku_shards.py (arc E17): serialize/deserialize, the HRM
augmentation's validity, rating-bucket encoding, and the plain packer's
board-never-splits-a-row invariant (T2's seq_len choice)."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.sudoku_shards import (  # noqa: E402
    ANSWER_CLOSE,
    ANSWER_OPEN,
    ShardWriter,
    augment_board,
    bucket_of_rating,
    deserialize,
    is_valid_solution,
    puzzle_consistent,
    resolve_token_ids,
    serialize_board,
    stage_of,
)
from morph.training.curriculum_data import HOLDOUT_SPLIT, MultiSourceCurriculumLoader  # noqa: E402
from morph.training.data_placement import DataRuntimeConfig  # noqa: E402

# A hand-checked valid solved 9x9 grid (row/col/box "all different"; is_valid_solution
# confirms it below) -- used instead of downloading the CSV, so these tests run offline.
SOLVED = "123456789456789123789123456" "231564897564897231897231564" "312645978645978312978312645"
assert len(SOLVED) == 81


def _puzzle(a: str, blanks: list[int]) -> str:
    chars = list(a)
    for i in blanks:
        chars[i] = "."
    return "".join(chars)


PUZZLE = _puzzle(SOLVED, list(range(0, 81, 3)))  # every 3rd cell blank (27 blanks)


@pytest.fixture(scope="module")
def ids():
    return resolve_token_ids()


def test_solved_fixture_is_actually_valid():
    assert is_valid_solution(SOLVED)
    assert puzzle_consistent(PUZZLE, SOLVED)


# ── serialize / deserialize ────────────────────────────────────────────────────


def test_serialize_deserialize_round_trip(ids):
    board = serialize_board(PUZZLE, SOLVED, ids)
    assert len(board) == 183
    assert board[90] == ANSWER_OPEN  # 9 puzzle rows x (9 digits + newline)
    assert board[-2] == ANSWER_CLOSE
    assert board[-1] == ids.eos
    q, a = deserialize(board, ids)
    assert q == PUZZLE
    assert a == SOLVED


def test_serialize_every_board_is_the_same_length(ids):
    """Every augmented board serializes to exactly the same token count -- this is
    the fixed length the seq_len choice below depends on."""
    rng = np.random.default_rng(0)
    lens = set()
    for _ in range(20):
        qa, aa = augment_board(PUZZLE, SOLVED, rng)
        lens.add(len(serialize_board(qa, aa, ids)))
    assert lens == {183}


def test_serialize_rejects_a_malformed_board(ids):
    with pytest.raises(ValueError):
        serialize_board(PUZZLE[:-1], SOLVED, ids)


def test_deserialize_rejects_missing_markers(ids):
    with pytest.raises(ValueError):
        deserialize([ids.eos], ids)


# ── augmentation validity (HRM recipe: digit relabel + band/stack shuffle + transpose) ──


def test_augmentation_preserves_validity_on_200_samples():
    rng = np.random.default_rng(1)
    base_blanks = PUZZLE.count(".")
    for _ in range(200):
        qa, aa = augment_board(PUZZLE, SOLVED, rng)
        assert is_valid_solution(aa)
        assert puzzle_consistent(qa, aa)
        assert qa.count(".") == base_blanks


def test_augmentation_across_five_puzzles():
    rng = np.random.default_rng(2)
    puzzles = []
    for shift in range(5):
        blanks = [(i + shift * 7) % 81 for i in range(0, 81, 4)]
        puzzles.append(_puzzle(SOLVED, blanks))
    for q in puzzles:
        for _ in range(40):
            qa, aa = augment_board(q, SOLVED, rng)
            assert is_valid_solution(aa)
            assert puzzle_consistent(qa, aa)
            assert qa.count(".") == q.count(".")


def test_is_valid_solution_rejects_a_broken_board():
    assert not is_valid_solution("1" * 81)  # every row/col/box repeats 1
    assert not is_valid_solution(PUZZLE)  # has blanks


def test_puzzle_consistent_rejects_a_wrong_filled_cell():
    bad = list(PUZZLE)
    for i, c in enumerate(PUZZLE):
        if c != ".":
            bad[i] = "1" if SOLVED[i] != "1" else "2"
            break
    assert not puzzle_consistent("".join(bad), SOLVED)


# ── rating buckets ──────────────────────────────────────────────────────────────

EDGES = [0, 1, 11, 23, 38]


@pytest.mark.parametrize(
    "rating,expected_bucket",
    [
        (0, 0),
        (1, 1),
        (10, 1),
        (11, 2),
        (22, 2),
        (23, 3),
        (37, 3),
        (38, 4),
        (1000, 4),
    ],
)
def test_bucket_of_rating(rating, expected_bucket):
    assert bucket_of_rating(rating, EDGES) == expected_bucket


def test_stage_round_trips_the_bucket():
    for rating in [0, 1, 5, 22, 38, 200]:
        stage = stage_of(rating, EDGES)
        assert int(stage.split(".")[0]) == bucket_of_rating(rating, EDGES)
        assert stage.split(".", 1)[1] == str(rating)


# ── plain packing: a board never splits across a row (T2) ──────────────────────


def _tiny_train_shard(root, ids, n_boards=24, seed=3):
    w = ShardWriter(str(root))
    rng = np.random.default_rng(seed)
    for _ in range(n_boards):
        qa, aa = augment_board(PUZZLE, SOLVED, rng)
        w.add(serialize_board(qa, aa, ids), stage="0.0")
    w.close(
        {"eos_id": ids.eos, "role": "reasoning_midtrain", "source_name": "train", "paths": ["test"]}
    )
    return w


def test_packing_never_splits_a_board(tmp_path, ids):
    """seq_len=182=board_len-1 makes ``batch_size*(L+1)`` always a multiple of the
    fixed board length (183), and MultiSourceCurriculumLoader._fill only ever pulls
    whole docs -- so its buffer is a multiple of 183 between calls and the
    carry-split packer never lands mid-board (see the module docstring's proof)."""
    root = tmp_path / "sudoku"
    _tiny_train_shard(root / "train", ids)
    ld = MultiSourceCurriculumLoader(
        str(root),
        {"train": 1.0},
        [182],
        seed=0,
        allowed_roles=["reasoning_midtrain"],
        data_runtime=DataRuntimeConfig(prefetch_batches=0),
        verbose=False,
    )
    it = ld.batches(4, bag_size=0)
    n_boards = 0
    for _ in range(6):
        x, _y = next(it)
        for row in x.numpy():
            opens = int((row == ANSWER_OPEN).sum())
            closes = int((row == ANSWER_CLOSE).sum())
            assert opens == 1 and closes == 1, (opens, closes, row.tolist())
            n_boards += 1
    assert n_boards == 24


def test_a_seq_len_off_the_board_length_does_split(tmp_path, ids):
    """Control for the invariant above: at seq_len=200 (L+1=201, not a multiple of
    183) the SAME loader straddles a board across two rows -- this is why
    sudoku_data.yaml pins seq_len to 182 rather than an arbitrary value >= one board."""
    root = tmp_path / "sudoku"
    _tiny_train_shard(root / "train", ids)
    ld = MultiSourceCurriculumLoader(
        str(root),
        {"train": 1.0},
        [200],
        seed=0,
        allowed_roles=["reasoning_midtrain"],
        data_runtime=DataRuntimeConfig(prefetch_batches=0),
        verbose=False,
    )
    it = ld.batches(4, bag_size=0)
    split_seen = False
    for _ in range(6):
        x, _y = next(it)
        for row in x.numpy():
            if int((row == ANSWER_OPEN).sum()) != 1:
                split_seen = True
    assert split_seen


# ── end-to-end: build train + holdout shards through the script's own functions,
#    then read the holdout through a holdout loader (T6) ───────────────────────


def test_holdout_shard_round_trips_through_a_holdout_loader(tmp_path, ids):
    root = tmp_path / "sudoku"
    rng = np.random.default_rng(5)
    w_train = ShardWriter(str(root / "train"))
    seen: set[tuple[int, ...]] = set()
    for _ in range(6):
        qa, aa = augment_board(PUZZLE, SOLVED, rng)
        tok_ids = serialize_board(qa, aa, ids)
        w_train.add(tok_ids, stage="1.5")
        seen.add(tuple(tok_ids))
    w_train.close(
        {
            "eos_id": ids.eos,
            "role": "reasoning_midtrain",
            "source_name": "train",
            "paths": ["test:train"],
        }
    )

    hold_dir = root / "holdout"
    hold_dir.mkdir(parents=True)
    w_hold = ShardWriter(str(hold_dir))
    held_qa, held_aa = augment_board(PUZZLE, SOLVED, rng)
    held_ids = serialize_board(held_qa, held_aa, ids)
    assert tuple(held_ids) not in seen  # the T1 leak guard, hand-checked here
    w_hold.add(held_ids, stage="1.5")
    jsonl_path = hold_dir / "eval_holdout.jsonl"
    with open(jsonl_path, "w") as jf:
        jf.write(
            json.dumps({"input_ids": held_ids, "stage": "1.5", "rating": 5, "source": "test"})
            + "\n"
        )
    w_hold.close(
        {
            "eos_id": ids.eos,
            "role": "reasoning_midtrain",
            "split": HOLDOUT_SPLIT,
            "source_name": "holdout",
            "paths": ["test:holdout"],
        }
    )

    ld = MultiSourceCurriculumLoader(
        str(root),
        {"holdout": 1.0},
        [182],
        seed=0,
        allowed_roles=["reasoning_midtrain"],
        holdout=True,
        data_runtime=DataRuntimeConfig(prefetch_batches=0),
        verbose=False,
    )
    x, _y = next(ld.batches(1, bag_size=0))
    assert x.shape == (1, 182)
    assert x.flatten().tolist() == held_ids[:-1]
    with pytest.raises(RuntimeError, match="eval-holdout shard"):
        MultiSourceCurriculumLoader(
            str(root), {"holdout": 1.0}, [182], seed=0, allowed_roles=["reasoning_midtrain"]
        )  # refused in a training blend

    rows = [json.loads(line) for line in open(jsonl_path)]
    assert rows[0]["input_ids"] == held_ids
    assert rows[0]["stage"] == "1.5"
