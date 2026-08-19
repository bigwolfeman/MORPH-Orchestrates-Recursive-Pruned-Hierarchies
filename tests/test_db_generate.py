"""DiffusionBlocks sampler smoke test under the faithful two-source concat.

The Euler-chain generator drives ``model(input_ids, db_step=...)`` per step; with
``conditioning='concat'`` each call runs the clean context pass + the noisy
two-source denoise. This confirms the sampler is wired end-to-end (it recomputes the
clean context each step — caching it across the σ-independent chain is a later
optimization).
"""

from __future__ import annotations

import torch

from morph.model.diffusion_blocks import DBConfig
from morph.inference.db_generate import db_sample, SampleTrace

from test_db_forward import _model, _RT, V


def test_db_sample_concat_runs_end_to_end():
    for mode in ("b1", "b3"):
        cfg = DBConfig(mode=mode, conditioning="concat")
        m = _model(cfg)
        rt = _RT(cfg)
        ids = torch.randint(0, V, (2, 16))
        g = torch.Generator().manual_seed(0)
        trace = SampleTrace()

        logits, z = db_sample(m, ids, rt, n_steps=4, generator=g, trace=trace)

        assert logits.shape == (2, 16, V), f"{mode}: bad logits shape {tuple(logits.shape)}"
        assert torch.isfinite(logits).all(), f"{mode}: non-finite sampler logits"
        assert z.shape == (2, 16, m.cfg.d_model)
        assert torch.isfinite(z).all(), f"{mode}: non-finite denoised estimate"
        assert len(trace.sigma) >= 1, f"{mode}: trace empty"
        # σ must have descended along the chain.
        assert all(trace.sigma[i] > trace.sigma[i + 1] for i in range(len(trace.sigma) - 1))


def test_db_sample_concat_is_deterministic_given_seed():
    cfg = DBConfig(mode="b1", conditioning="concat")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(0, V, (2, 16))
    a, _ = db_sample(m, ids, rt, n_steps=4, generator=torch.Generator().manual_seed(7))
    b, _ = db_sample(m, ids, rt, n_steps=4, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b), "same seed must give the same sample"
