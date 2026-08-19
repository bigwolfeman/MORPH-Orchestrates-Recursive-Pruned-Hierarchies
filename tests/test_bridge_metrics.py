"""Bridge-metric contracts. CPU only, no teacher download.

These are the only cross-family numbers the campaign is allowed to compare (sheet §1.3), so
they need to be right about the thing they claim: rep4 must actually detect degenerate
repetition, and distinct-n must actually rank diversity.
"""

import torch

from morph.posttrain.bridge_metrics import (
    BridgeConfig,
    bridge_report,
    distinct_n,
    rep_ngram_fraction,
)


def test_rep4_is_zero_on_unique_text_and_one_on_a_loop():
    unique = torch.arange(0, 200).view(1, 200)
    assert rep_ngram_fraction(unique, 4) == 0.0
    # a 4-token cycle repeated: every 4-gram after the first few is a repeat
    loop = torch.tensor([[1, 2, 3, 4] * 50])
    assert rep_ngram_fraction(loop, 4) > 0.9


def test_rep4_ranks_partial_repetition_between_the_extremes():
    unique = torch.arange(0, 100).view(1, 100)
    half = torch.cat([torch.arange(0, 50), torch.arange(0, 50)]).view(1, 100)
    loop = torch.tensor([[7, 8, 9, 10] * 25])
    a, b, c = (rep_ngram_fraction(x, 4) for x in (unique, half, loop))
    assert a < b < c, f"rep4 did not rank repetition: {a:.3f} {b:.3f} {c:.3f}"


def test_rep4_respects_the_window():
    """Only the first `window` tokens count, so a tail loop outside it must not register."""
    row = torch.cat([torch.arange(0, 64), torch.tensor([1, 2, 3, 4] * 16)]).view(1, 128)
    assert rep_ngram_fraction(row, 4, window=64) == 0.0
    assert rep_ngram_fraction(row, 4, window=128) > 0.0


def test_distinct_n_ranks_diversity():
    diverse = torch.arange(0, 200).view(2, 100)
    same = torch.zeros(2, 100, dtype=torch.long)
    assert distinct_n(diverse, 2) > distinct_n(same, 2)
    assert distinct_n(same, 2) < 0.05


def test_bridge_report_marks_a_skipped_teacher_instead_of_faking_zero():
    """A missing teacher must be visible, not reported as a score of 0."""
    ids = torch.arange(0, 320).view(4, 80)

    class _Tok:
        def decode(self, r, skip_special_tokens=True):
            return " ".join(str(i) for i in r)

    rep = bridge_report(ids, BridgeConfig(), _Tok(), teacher=None)
    assert rep["bridge/teacher"] == "SKIPPED"
    assert "bridge/gen_ppl_teacher" not in rep
    assert rep["bridge/n_sequences"] == 4
    assert rep["bridge/gen_tokens"] == 80


def test_bridge_report_records_the_decoding_settings():
    """Two arms are comparable only under identical decoding; record it in the row."""
    ids = torch.arange(0, 160).view(2, 80)

    class _Tok:
        def decode(self, r, skip_special_tokens=True):
            return "x"

    cfg = BridgeConfig(top_p=0.9, db_steps=4)
    rep = bridge_report(ids, cfg, _Tok(), teacher=None)
    assert rep["bridge/decode_top_p"] == 0.9
    assert rep["bridge/db_steps"] == 4
