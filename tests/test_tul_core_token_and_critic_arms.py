"""Both 2026-09-12 core-token / critic arms compose through Hydra, build, and RUN.

Green unit tests do not prove an arm can START. Every unit test of these two arms builds
its own tiny `TULConfig` by hand; the 2026-09-12 latent-z batch lost all three of its arms
to a refusal in `TULConfig.__post_init__` that 1,149 green tests never saw.

So this file composes each config through Hydra, maps it through the SHIPPED
`build_tul_runtime`, asserts the PANEL's real budgets, prints the resolved knobs, then
builds a model and runs a TRAINING forward and backward so both terms execute — and an
EVAL forward, because arm C's whole claim is that the shipped forward is unchanged.

Arms: `.agents/notes/proposed/architecture/2026-09-12-core-token-gradient-and-within-context-critic.md`
Record: `lab/experiments/planned/2026-09-12-arc-core-token-and-critic.md`

CPU only, fp32, tiny model — the `tests/test_tul_objective_arms.py` fixtures.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul_egrad import CriticEnergy
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10
_CONFIG_DIR = os.path.abspath("morph/configs")
NEW_CONFIGS = ["tul_slot_spandec_strict_critic", "tul_slot_spandec_strict_coretok"]


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
def test_the_arms_compose_at_the_panel_budget(name, monkeypatch):
    cfg, rt = _runtime(name, monkeypatch)
    assert rt is not None, f"{name}: tul is off — the arm would run the plain model"
    tc = rt.model_cfg
    # The PANEL's real budgets, not a fixture's.
    assert int(cfg.data.seq_len) == 1024 and int(cfg.training.batch_size) == 6
    assert int(cfg.training.steps) == 5000
    assert tc.tg_geometry == "strict" and tc.tg_restrict and tc.spandec
    assert tc.tg_coda_prefix_reach == "all" and tc.loop_reach == 0
    assert tc.mux_beta == 0.0 and tc.spandec_max_tokens == 0 and tc.bound_span_cap == 32
    assert tc.spandec_horizon == 1 and not tc.tokens_through_core
    assert not bool(cfg.model.use_kernels), "tg_restrict forces model.use_kernels: false"
    if name.endswith("_coretok"):
        assert tc.core_token_aux and tc.core_token_aux_weight == 1.0
        assert tc.grad_pass_energy == "own_mux" and not tc.grad_pass
    else:
        assert tc.grad_pass and tc.grad_pass_energy == "critic"
        assert tc.grad_pass_scale == 0.1 and tc.grad_pass_norm == "rms"
        assert tc.critic_weight == 1.0 and tc.critic_every == 1
        assert tc.critic_eps == 0.1 and tc.critic_replay_groups == 1
        assert tc.pass_residual_lambda == 0.01
        assert not tc.core_token_aux
    # The wandb manifest is what makes a run reproducible from its config alone.
    for k in ("core_token_aux", "core_token_aux_weight", "grad_pass_energy",
              "critic_weight", "critic_every", "critic_eps", "critic_replay_groups"):
        assert k in rt.manifest, f"{k} is missing from the wandb manifest"
    print(f"[{name}] core_token_aux={tc.core_token_aux}/{tc.core_token_aux_weight} "
          f"grad_pass={tc.grad_pass} energy={tc.grad_pass_energy} "
          f"critic_weight={tc.critic_weight} every={tc.critic_every} "
          f"eps={tc.critic_eps} groups={tc.critic_replay_groups} "
          f"tg_geometry={tc.tg_geometry} reach={tc.tg_coda_prefix_reach} "
          f"spandec_J={tc.spandec_max_tokens or tc.bound_span_cap} "
          f"seq={int(cfg.data.seq_len)} batch={int(cfg.training.batch_size)} "
          f"wandb={cfg.wandb.name}")


@pytest.mark.parametrize("name", NEW_CONFIGS)
def test_the_arms_build_and_run_a_training_step(name, monkeypatch):
    _cfg, rt = _runtime(name, monkeypatch)
    tc = rt.model_cfg
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=tc)).train().float()
    inp, lab, layout, _ = _batch()
    torch.manual_seed(3)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"]), f"{name}: loss is not finite"
    if tc.core_token_aux:
        assert float(out["core_token_aux"]) > 0.0
        assert float(out["core_token_aux_n"]) > 0.0
    else:
        assert isinstance(m.tul_egrad, CriticEnergy)
        assert float(out["critic"]) > 0.0
        assert float(out["critic_replays"]) == 3.0
    out["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for n, p in m.named_parameters() if n.startswith("core."))


@pytest.mark.parametrize("name", NEW_CONFIGS)
def test_the_arms_run_an_eval_forward(name, monkeypatch):
    """The sweeps and `worth_profile` run this path. Arm C must pay nothing here."""
    _cfg, rt = _runtime(name, monkeypatch)
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=rt.model_cfg)).eval().float()
    inp, lab, layout, _ = _batch()
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    assert "core_token_aux" not in out and "critic" not in out
