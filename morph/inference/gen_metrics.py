"""Generation-quality metrics — the numbers a good val CE can hide.

docs/tul-gate-spec.md §10 requires these from run 1. §5's teacher-forcing leak is
invisible in val CE by construction: at training the coda is told the true span length,
at generation it is told the model's guess. Only generation exposes the gap.

The repetition pair is not optional. A degenerate repetition loop scores an EXCELLENT
perplexity — a measured 1.46 against real text's 32.44 — so a fluency number is
meaningless unless a diversity number is reported beside it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["generation_metrics", "ngram_stats", "span_stats"]


def ngram_stats(ids: list[int], n: int = 4, window: int = 512) -> tuple[float, float]:
    """``(rep_n, distinct_n)`` over the first ``window`` tokens.

    ``rep_n``      fraction of ``n``-grams that already occurred earlier — the collapse
                   detector. Real text sits low; a repetition loop goes to ~1.
    ``distinct_n`` unique ``n``-grams / total. The same signal from the other side, and
                   the one that is comparable across lengths.
    """
    w = ids[:window]
    if len(w) <= n:
        return 0.0, 1.0
    grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    seen: set = set()
    rep = 0
    for g in grams:
        if g in seen:
            rep += 1
        seen.add(g)
    return rep / len(grams), len(seen) / len(grams)


def span_stats(builder, rule) -> dict[str, float]:
    """Span geometry of one generated row: how long the model's spans were, and how
    often it actually landed on a boundary rather than spending its whole budget.

    ``boundary_frac`` is the direct read on §3.2: a span that ends off-boundary is one
    where the gate's ``k`` ran out first. All-1.0 means the budget never bound (the gate
    is asking for more than the rule needs); a low value means it is cutting units short.
    """
    ids = builder.ids
    firsts = list(builder.slot_first)
    if not firsts:
        return {"n_spans": 0.0, "mean_span": 0.0, "boundary_frac": 0.0}
    prev, lens, on_b = -1, [], 0
    for f in firsts:
        last = f - 1  # the span's final TOKEN position
        lens.append(last - prev)
        if bool(rule.is_boundary[ids[last]]):
            on_b += 1
        prev = last + builder.spec.prefix_k
    return {
        "n_spans": float(len(lens)),
        "mean_span": float(np.mean(lens)),
        "p50_span": float(np.median(lens)),
        "boundary_frac": on_b / len(lens),
    }


def generation_metrics(emitted: list[int], builder=None, rule=None,
                       window: int = 512) -> dict[str, float]:
    """Everything §10 asks for about one generated sample, in one dict.

    ``builder``/``rule`` are OPTIONAL: the A0 baseline arm builds no slots, so it has no
    span geometry to report, and refusing to score it is how the first version of this
    table ended up with no baseline row in it. Absent a builder, the n-gram half is
    still scored by exactly this code, which is what makes the arms comparable.

    ``window`` must be passed explicitly whenever samples are longer than the default:
    rep_n is NOT comparable across lengths (measured on held-out OpenWebText, 256 rows:
    rep4 = 0.015 at 128 tokens with 54 % of rows at exactly 0, against 0.037 at 512
    tokens with 1 % at 0). At 128 tokens the reference itself is on the floor, so the
    metric cannot rank anything.
    """
    rep4, d4 = ngram_stats(emitted, n=4, window=window)
    _r3, d3 = ngram_stats(emitted, n=3, window=window)
    out = {"rep4": rep4, "distinct4": d4, "distinct3": d3, "n_tokens": float(len(emitted))}
    if builder is not None and rule is not None:
        out.update(span_stats(builder, rule))
    return out
