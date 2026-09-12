"""Every 2026-09-12 objective arm composes through Hydra, builds, and RUNS.

Green unit tests do not prove an arm can START. Every unit test of these two arms builds
its own tiny `TULConfig` by hand; the 2026-09-12 latent-z batch lost all three of its arms
to a refusal in `TULConfig.__post_init__` that 1,149 green tests never saw, and the
strict-oracle smoke died on a J 8 vs 32 reshape inside `SpanDecoder.decode` because every
fixture had set both budgets to 8.

So this file composes each new config through Hydra, maps it through the SHIPPED
`build_tul_runtime`, asserts the PANEL's real budgets (`spandec_max_tokens` 0 -> 32, the
per-pass term at 8 tokens and cap 6), prints the resolved
knobs, then builds a model and runs a TRAINING forward and backward so both terms execute.

Arms: `.agents/notes/proposed/architecture/2026-09-12-objective-arms-per-pass-plan-and-parallel-coda.md`
Record: `lab/experiments/planned/2026-09-12-arc-objective-arms.md`

CPU only, fp32, tiny model — the `tests/test_tul_oracle_z.py` fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=3, max_depth=4, bptt_depth=4,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)


def _batch(B: int = 2, n: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::8] = DOT
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


_CONFIG_DIR = __import__("os").path.abspath("morph/configs")
NEW_CONFIGS = ["tul_slot_spandec_strict_perpass"]


class _StubTok:
    """Enough of a tokenizer for `build_tul_runtime`'s id resolution, no download."""

    @staticmethod
    def from_pretrained(_name):
        return _StubTok()

    def convert_tokens_to_ids(self, _tok):
        return 4


def _runtime(name: str, monkeypatch):
    import transformers
    from hydra import compose, initialize_config_dir

    from morph.training import tul_setup
    monkeypatch.setattr(transformers, "AutoTokenizer", _StubTok)
    monkeypatch.setattr(tul_setup, "build_boundary_rule",
                        lambda cfg, cache_dir="": (_rule(), _rule().is_boundary, 0,
                                                   ("\n",)))
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        cfg = compose(config_name=name)
    return cfg, tul_setup.build_tul_runtime(cfg)


@pytest.mark.parametrize("name", NEW_CONFIGS)
def test_the_objective_arms_compose_and_build_and_run(name, monkeypatch):
    cfg, rt = _runtime(name, monkeypatch)
    assert rt is not None, f"{name}: tul is off — the arm would run the plain model"
    tc = rt.model_cfg
    # The panel's REAL budgets, not a fixture's: the decoder at span_cap 32 and the
    # per-pass term at 8 tokens and cap 6.
    assert tc.tg_geometry == "strict" and tc.tg_restrict and tc.spandec
    assert tc.spandec_max_tokens == 0 and tc.bound_span_cap == 32
    assert tc.spandec_horizon == 1, "the exit target must stay the next thought"
    assert tc.spandec_per_pass and tc.spandec_pass_tokens == 8
    assert tc.spandec_pass_horizon_max == 6
    assert not bool(cfg.model.use_kernels), "tg_restrict forces model.use_kernels: false"
    print(f"[{name}] spandec_per_pass={tc.spandec_per_pass} "
          f"pass_tokens={tc.spandec_pass_tokens} cap={tc.spandec_pass_horizon_max} "
          f"spandec_max_tokens={tc.spandec_max_tokens or tc.bound_span_cap} "
          f"horizon={tc.spandec_horizon}")

    # Build at the config's OWN budgets (decoder J = 32, per-pass J = 8) and run a
    # TRAINING forward, so the per-pass term executes for real.
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=tc)).train().float()
    inp, lab, layout, _ = _batch()
    torch.manual_seed(3)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"]), f"{name}: loss is not finite"
    if tc.spandec_per_pass:
        assert float(out["spandec_pass"]) > 0.0
        # the horizons the arm actually graded, at the panel's cap
        hs = [int(out[k]) for k in out if str(k).startswith("spandec_pass_h")
              and str(k)[14:].isdigit()]
        assert hs == [min(t, 6) for t in range(1, len(hs) + 1)], hs
    out["loss"].backward()
