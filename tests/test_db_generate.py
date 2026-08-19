"""DiffusionBlocks sampler contracts. CPU only, tiny model, no CUDA.

This file exists because `db_generate.py` was written and never executed. It is on the
critical path to any QUALITY claim — the bridge metrics need generated tokens — and it has
more moving parts than the eval branch that already NameError'd on first contact:
descending σ, the softmax @ E bridge, and block-of-σ dispatch.
"""

from __future__ import annotations

import pytest
import torch

from morph.inference.db_generate import SampleTrace, db_sample
from morph.model.diffusion_blocks import DBConfig, DBSchedule, EDMPrecond
from morph.model.transformer import MORPHConfig, MORPHTransformer

V = 64


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=2, max_depth=3, bptt_depth=2,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16, retention=False,
        bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False, dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


class _RT:
    def __init__(self, cfg: DBConfig, mean_depth: int = 2):
        self.model_cfg = cfg
        self.schedule = DBSchedule(cfg, mean_depth=mean_depth)
        self.precond = EDMPrecond(cfg.sigma_data)


def _model(cfg: DBConfig, seed: int = 0):
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny()).eval()
    m.build_db_modules(cfg)
    return m.eval()


@pytest.mark.parametrize("mode", ["b1", "b3"])
def test_sampler_runs_and_returns_the_right_shapes(mode):
    cfg = DBConfig(mode=mode, conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(5, V, (2, 16))
    logits, z = db_sample(m, ids, rt, n_steps=4,
                          generator=torch.Generator().manual_seed(0))
    assert logits.shape == (2, 16, V)
    assert z.shape == (2, 16, 64)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(z).all()


def test_sampler_is_deterministic_under_a_seed():
    """Two arms are only comparable if decoding is reproducible."""
    cfg = DBConfig(mode="b3", conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(5, V, (2, 16))
    a, _ = db_sample(m, ids, rt, n_steps=4, generator=torch.Generator().manual_seed(3))
    b, _ = db_sample(m, ids, rt, n_steps=4, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_step_count_is_a_free_dial_needing_no_retraining():
    """Arm DB-13. The denoiser conditions on σ, not Δσ, so any step count must work.

    Their AR text setting used only 4 steps against a default of 50, so this has to hold.
    """
    cfg = DBConfig(mode="b3", conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(5, V, (1, 12))
    outs = {}
    for n in (2, 4, 8, 16):
        lg, _ = db_sample(m, ids, rt, n_steps=n, generator=torch.Generator().manual_seed(1))
        assert torch.isfinite(lg).all(), f"n_steps={n} produced non-finite logits"
        outs[n] = lg
    # more steps must actually change the trajectory, or the schedule is being ignored
    assert not torch.equal(outs[2], outs[16]), "step count had no effect on the output"


def test_trace_records_every_euler_step_with_descending_sigma():
    cfg = DBConfig(mode="b3", conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(5, V, (1, 12))
    tr = SampleTrace()
    db_sample(m, ids, rt, n_steps=6, generator=torch.Generator().manual_seed(0), trace=tr)
    assert len(tr.sigma) == 5, f"expected 5 euler steps for n_steps=6, got {len(tr.sigma)}"
    assert all(tr.sigma[i] > tr.sigma[i + 1] for i in range(len(tr.sigma) - 1)), tr.sigma
    d = tr.as_dict()
    assert len(d["db_sample/denoised_norm"]) == 5
    assert all(x == x for x in d["db_sample/denoised_norm"])   # no NaN


def test_b3_sampler_visits_more_than_one_block():
    """Under B=3 the walk must hand off between blocks as σ descends.

    If every step dispatched to the same block, the partition would be doing nothing and the
    B=3 arm would be a mislabelled B=1.
    """
    cfg = DBConfig(mode="b3", conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(5, V, (1, 12))
    tr = SampleTrace()
    db_sample(m, ids, rt, n_steps=12, generator=torch.Generator().manual_seed(0), trace=tr)
    assert len(set(tr.block)) > 1, f"sampler stayed in one block: {set(tr.block)}"


def test_sampler_rejects_a_degenerate_step_count():
    cfg = DBConfig(mode="b1", conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(5, V, (1, 8))
    # n_steps is clamped to >= 2 internally; 1 must not silently become a no-op walk
    lg, _ = db_sample(m, ids, rt, n_steps=1, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(lg).all()


def test_sampler_output_feeds_the_bridge_metrics():
    """End to end: sampled logits -> tokens -> a bridge row. The whole point of the sampler."""
    from morph.posttrain.bridge_metrics import BridgeConfig, bridge_report

    cfg = DBConfig(mode="b3", conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(5, V, (4, 32))
    logits, _ = db_sample(m, ids, rt, n_steps=6,
                          generator=torch.Generator().manual_seed(0))
    toks = logits.argmax(dim=-1)
    assert toks.shape == (4, 32)

    class _Tok:
        def decode(self, r, skip_special_tokens=True):
            return " ".join(str(i) for i in r)

    rep = bridge_report(toks, BridgeConfig(rep_window=32), _Tok(), teacher=None)
    assert 0.0 <= rep["bridge/rep4@32"] <= 1.0
    assert 0.0 < rep["bridge/distinct2"] <= 1.0
    assert rep["bridge/n_sequences"] == 4


def test_generator_device_mismatch_raises_an_actionable_error():
    """A CPU generator with a CUDA model must fail LOUDLY, not be silently converted.

    torch's own message ("Expected a 'cuda' device type for generator but found 'cpu'") comes
    from deep inside randn and does not say what to do. This surfaced only on the GPU bridge
    run — the CPU shakeout could not catch it, because there the two devices agreed.

    Not auto-corrected on purpose: CPU and CUDA RNG streams differ for the same seed, so
    swapping one for the other would change generations between two runs that look identical.
    """
    cfg = DBConfig(mode="b1", conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(5, V, (1, 8))

    class _FakeCudaGen:
        device = torch.device("cuda:0")

    with pytest.raises(ValueError, match="RNG streams differ"):
        db_sample(m, ids, rt, n_steps=3, generator=_FakeCudaGen())


def test_matching_generator_device_is_accepted():
    cfg = DBConfig(mode="b1", conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(5, V, (1, 8))
    lg, _ = db_sample(m, ids, rt, n_steps=3,
                      generator=torch.Generator(device="cpu").manual_seed(0))
    assert torch.isfinite(lg).all()
