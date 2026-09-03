"""The eager dmorph generator: both heads generate, never emit the slot id, and build the
loader's layout (morph/inference/dmorph_generate.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.inference.dmorph_generate import generate_dmorph
from morph.inference.gen_metrics import generation_metrics
from _dmorph_common import DOT, V, dm_cfg, model, rule, spec, wake_stream


@pytest.mark.parametrize("head", ["clean", "ladder"])
def test_generate_dmorph_emits_tokens_and_the_loaders_layout(head):
    m = model(dm_cfg(n_blocks=2))
    wake_stream(m)
    sp = spec(seq_len=64, max_slots=16)
    prompt = [7, 8, 9, DOT, 12, 13]
    out, builder = generate_dmorph(m, prompt, rule(), sp, max_new_tokens=12,
                                   temperature=1.0, top_k=0, seed=0, head=head)
    assert len(out) == 12
    assert all(0 <= t < V and t != sp.slot_id for t in out)
    # The layout is the loader's: it is built by the SAME TulRowBuilder the TUL generator
    # uses (parity-tested against the packer in tests/test_tul_layout.py), and every
    # slot sits right after a boundary token with prefix_k slot positions.
    r = rule()
    assert builder.n_slots >= 1
    toks = [t for t, s in zip(builder.ids, builder.slot_mask) if not s]
    span, cuts = 0, []
    for i, tok in enumerate(toks):
        c, span = r.cut(np.array([tok], dtype=np.int64), span)
        if c.size:
            cuts.append(i)
    # One slot per cut the rule makes on the token stream (a span_cap cut is not a
    # boundary character, so the check is the rule, not the character table).
    assert builder.n_slots == len(cuts)
    for f, ci in zip(builder.slot_first, cuts):
        assert not builder.slot_mask[f - 1] and builder.ids[f - 1] == toks[ci]
        assert all(builder.slot_mask[f + k] for k in range(sp.prefix_k))
    stats = generation_metrics(out, builder, r)
    assert set(stats) >= {"rep4", "distinct3", "n_spans"}


def test_generate_dmorph_refuses_a_model_without_the_stream_and_an_unknown_head():
    m = model(None)
    with pytest.raises(RuntimeError, match="dmorph"):
        generate_dmorph(m, [7, 8], rule(), spec(), max_new_tokens=1)
    m2 = model(dm_cfg(n_blocks=2))
    with pytest.raises(ValueError, match="head"):
        generate_dmorph(m2, [7, 8], rule(), spec(), max_new_tokens=1, head="soft")


def test_greedy_generation_is_deterministic_per_head():
    m = model(dm_cfg(n_blocks=2))
    wake_stream(m)
    sp = spec(seq_len=64, max_slots=16)
    a, _ = generate_dmorph(m, [7, 8, 9], rule(), sp, max_new_tokens=6, temperature=0.0)
    b, _ = generate_dmorph(m, [7, 8, 9], rule(), sp, max_new_tokens=6, temperature=0.0)
    assert a == b
    torch.manual_seed(0)
