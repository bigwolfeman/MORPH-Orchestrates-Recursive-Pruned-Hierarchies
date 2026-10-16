"""CPU tests of the minimal slot loop (`configs/arm/minimal.yaml`, `minimal_noptr.yaml`).

What is pinned:
  1. the arm files compose, every key is a real LXTULConfig field, and the values are the minimal ones;
  2. the minimal arm builds NO parameter or buffer of the span decoder, the router, the latent head, the EMA
     twin or the epivol reservoir; the default arm still builds all of them (its parameter set is pinned);
  3. the minimal arm RUNS none of the removed compute: the gain hinge, the fixed-point residual, the cotangent
     clip, the epivol term, the latent-selection exit, the span decoder and activation checkpointing (each
     probe is also shown to fire when the term is on, so a dead probe cannot pass);
  4. the layout follows `cells`: one cell per slot packs a row of SEQ + S positions, and the model runs it;
  5. combinations that make no sense raise (select latent or epivol at one cell).

    PYTHONPATH=.:$MORPH_ROOT python -m pytest -q lxtul/tests/test_minimal.py
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import lxtul.model as lm
import parcae_lm
from lxtul.data import layout_for_cells, packed_length
from lxtul.model import LXTULConfig, LXTULParcae
from lxtul.tests.test_compact import (EOS, J, K, S, SEQ, SLOT, V, _batch, _build, _randomize,  # noqa: F401
                                      _run, cpu_reference_kernels)

ARMS = Path(lm.__file__).parent / "configs" / "arm"
REMOVED = ("sd_blocks", "sd_z_in", "sd_tok_in", "sd_out_norm", "sd_cross", "router", "latent_head",
           "twin_prelude", "epi_W1", "epi_W2")


def _keys(name: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(ARMS / f"{name}.yaml").lxtul, resolve=True)


def _row1(seed: int, B: int = 3):
    """Rows packed with ONE cell per slot (MORPH's packer at prefix_k = 1)."""
    from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, pack_tul_batch
    lut = np.zeros(V, dtype=bool)
    lut[10:20] = True
    lut[EOS] = True
    rule = BoundaryRule(lut, min_span=4, span_cap=J, eos_id=EOS)
    spec = TulLayoutSpec(seq_len=SEQ, prefix_k=1, max_slots=S, slot_id=SLOT)
    rng = np.random.default_rng(seed)
    buf = rng.integers(20, V, size=B * (spec.l_total + 1) * 2).tolist()
    i = 0
    while i < len(buf):
        buf[i] = int(rng.integers(10, 20))
        i += int(rng.integers(8, 14))
    return pack_tul_batch(buf, rule, spec, B), spec


def _model(cells: int, name: str | None = "minimal", **extra) -> LXTULParcae:
    keys = {} if name is None else _keys(name)
    keys.pop("cells", None)
    keys.update(extra)
    torch.manual_seed(0)
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=64, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=128, recurrent_embedding_dimension=64,
        recurrent_intermediation_embedding_dimension=128, n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2, n_layers_in_coda=1, vocab_size=V, block_size=SEQ + cells * S)
    m = LXTULParcae(cfg, LXTULConfig(slot_id=SLOT, max_slots=S, cells=cells, span_cap=J,
                                     spandec_layers=1, **keys))
    m.init_slot_seed()
    return _randomize(m).double().train()


def _names(m) -> set[str]:
    return {n for n, _ in m.named_parameters()} | {n for n, _ in m.named_buffers()}


# ------------------------------------------------------------------ 1. the arm files
@pytest.mark.parametrize("name,ptr", [("minimal", 4), ("minimal_noptr", 0)])
def test_arm_files_are_real_keys_with_the_minimal_values(name, ptr):
    arm = OmegaConf.load(ARMS / f"{name}.yaml")
    assert str(arm.name) == name and str(arm.kind) == "lxtul"
    keys = _keys(name)
    assert set(keys) <= {f.name for f in dataclasses.fields(LXTULConfig)}, set(keys)
    off = dict(spandec_weight=0.0, gain_lambda=0.0, fixed_point_lambda=0.0, cot_clip=0.0,
               repel_lambda=0.0, ckpt_passes=0)
    for k, v in off.items():
        assert keys[k] == v, k
    assert keys["cells"] == 1 and keys["select"] == "none" and keys["pointer_heads"] == ptr
    for k in ("compact_slots", "static_depth", "ce_rows"):
        assert keys[k] is True, k
    assert "token_state_dropout" not in keys, "token-state dropout keeps MORPH's value"


# ------------------------------------------------------------------ 2. what is built
def test_minimal_builds_none_of_the_removed_modules_and_default_builds_all():
    mini = _names(_model(1))
    assert not [n for n in mini if n.split(".")[0] in REMOVED], sorted(mini)
    default = _names(_build())                                    # the default arm: select latent, fan of K
    for stem in (r for r in REMOVED if r != "sd_cross"):          # sd_cross needs spandec_reads_cells
        assert any(n.split(".")[0] == stem for n in default), f"default arm lost {stem}"
    # nothing else of the default arm vanished: the minimal arm's names are the default's, minus the
    # removed modules, minus the role-table rows that only exist for cells 2..K, plus nothing
    kept = {n for n in default if n.split(".")[0] not in REMOVED and not n.startswith("W_prefix.")
            and not n.startswith("pointer")}
    assert kept <= mini, sorted(kept - mini)
    assert {n for n in mini if n.startswith("W_prefix.")} == {"W_prefix.0"}


def test_default_arm_parameter_set_is_pinned():
    m = _build()
    top = {n.split(".")[0] for n in _names(m)}
    assert top == {"transformer", "freqs_cis", "value_embeds", "E_slot", "W_sent", "E_mask", "reg_q_embed",
                   "reg_k", "reg_v", "reg_o", "reg_p_cell_embed", "cell_norm", "W_prefix", "router",
                   "latent_head", "twin_prelude", "sd_blocks", "sd_z_in", "sd_tok_in", "sd_out_norm",
                   "epi_W1", "epi_W2"}, top
    assert len(m.W_prefix) == K and m.reg_q_embed.shape[0] == K
    assert m.tul.lsel_on and m.tul.spandec_on and m.tul.repel_on
    # the removed-by-default-off code did not change the default keys
    d = LXTULConfig(slot_id=SLOT)
    assert (d.gain_lambda, d.fixed_point_lambda, d.cot_clip, d.repel_lambda, d.ckpt_passes, d.select) == \
        (100.0, 0.1, 4.0, 0.1, 4, "latent")


# ------------------------------------------------------------------ 3. what is run
@pytest.fixture
def probes(monkeypatch):
    """Count calls of every removed piece of compute."""
    calls: dict[str, int] = {k: 0 for k in
                             ("gain", "fp_clip_fn", "epivol", "lsel_exit", "spandec", "checkpoint", "hook")}

    def counted(name, fn):
        def wrapper(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return wrapper
    monkeypatch.setattr(LXTULParcae, "_gain_penalty", counted("gain", LXTULParcae._gain_penalty))
    monkeypatch.setattr(LXTULParcae, "_epivol", counted("epivol", LXTULParcae._epivol))
    monkeypatch.setattr(LXTULParcae, "_lsel_exit", counted("lsel_exit", LXTULParcae._lsel_exit))
    monkeypatch.setattr(LXTULParcae, "_spandec", counted("spandec", LXTULParcae._spandec))
    monkeypatch.setattr(lm, "checkpoint", counted("checkpoint", lm.checkpoint))
    monkeypatch.setattr(lm._CotClip, "apply", staticmethod(counted("fp_clip_fn", lm._CotClip.apply)))
    reg = torch.Tensor.register_hook
    monkeypatch.setattr(torch.Tensor, "register_hook", counted("hook", reg))
    return calls


def test_minimal_runs_none_of_the_removed_compute(probes):
    m = _model(1)
    (x, y, lay), _ = _row1(1)
    o, g = _run(m, x, y, lay)
    assert torch.isfinite(o["loss"]) and all(torch.isfinite(v).all() for v in g.values())
    assert probes == {k: 0 for k in probes}, probes
    assert "fp" not in o["metrics"] and "gain" not in o["metrics"], "a removed term left a metric"
    assert not [n for n in g if n.split(".")[0] in REMOVED]
    assert float(o["loss"]) == pytest.approx(float(o["ce"]), abs=0), "the loss is the token CE alone"


def test_the_probes_fire_when_the_terms_are_on(probes):
    """Sabotage check: with the default keys every probe must count, else the zeros above prove nothing."""
    m = _build(cot_clip_fn=True)
    x, y, lay = _batch(1)
    _run(m, x, y, lay)
    assert probes["gain"] >= 1 and probes["epivol"] >= 1 and probes["lsel_exit"] >= 1
    assert probes["spandec"] >= 1 and probes["checkpoint"] >= 1 and probes["fp_clip_fn"] >= 1
    m2 = _build()                                                  # dict-hook clip
    probes["hook"] = 0
    _run(m2, x, y, lay)
    assert probes["hook"] >= 1


def test_each_removal_alone_skips_its_compute(probes):
    x, y, lay = _batch(1)
    for key, val, probe in (("gain_lambda", 0.0, "gain"), ("repel_lambda", 0.0, "epivol"),
                            ("spandec_weight", 0.0, "spandec"), ("ckpt_passes", 0, "checkpoint")):
        # the hinge runs its own two checkpointed steps, so isolate ckpt_passes from it
        m = _build(**{key: val}, **({"gain_lambda": 0.0} if probe == "checkpoint" else {}))
        for k in probes:
            probes[k] = 0
        _run(m, x, y, lay)
        assert probes[probe] == 0, f"{key}={val} still ran {probe}"
    m = _build(select="none")
    for k in probes:
        probes[k] = 0
    _run(m, x, y, lay)
    assert probes["lsel_exit"] == 0 and not hasattr(m, "router") and not hasattr(m, "twin_prelude")
    m = _build(cot_clip=0.0)
    for k in probes:
        probes[k] = 0
    _run(m, x, y, lay)
    assert probes["hook"] == 0 and probes["fp_clip_fn"] == 0
    m = _build(fixed_point_lambda=0.0)
    o, _ = _run(m, x, y, lay)
    assert "fp" not in o["metrics"]


# ------------------------------------------------------------------ 4. the layout follows `cells`
class _RT:
    """A TUL runtime stand-in: MORPH's TulDataConfig inside a dataclass (layout_for_cells uses replace)."""


def _rt(prefix_k: int):
    from morph.model.tul_layout import BoundaryRule, TulDataConfig
    lut = np.zeros(V, dtype=bool)
    lut[10:20] = True
    rule = BoundaryRule(lut, min_span=4, span_cap=J, eos_id=EOS)

    @dataclasses.dataclass
    class RT:
        data_cfg: object
    return RT(TulDataConfig(rule=rule, prefix_k=prefix_k, slot_id=SLOT, max_slots=64))


def test_layout_for_one_cell_is_1088_positions_and_the_default_is_untouched():
    mcfg = type("C", (), {"data": type("D", (), {"seq_len": 1024})()})()
    rt = _rt(4)
    assert packed_length(mcfg, rt) == 1280 and layout_for_cells(rt, 4) is rt
    one = layout_for_cells(rt, 1)
    assert packed_length(mcfg, one) == 1088 and one.data_cfg.prefix_k == 1
    assert rt.data_cfg.prefix_k == 4, "the original runtime must not change"
    assert one.data_cfg.max_slots == 64 and one.data_cfg.slot_id == SLOT
    three = layout_for_cells(rt, 3)
    assert packed_length(mcfg, three) == 1024 + 3 * 64
    with pytest.raises(ValueError, match="cells must be >= 1"):
        layout_for_cells(rt, 0)
    assert layout_for_cells(None, 1) is None


def test_one_cell_row_has_one_prefix_position_per_slot_and_the_model_runs_it():
    (x, y, lay), spec = _row1(2)
    assert x.shape[1] == SEQ + S == spec.l_total and lay.prefix_k == 1
    assert (lay.slot_mask.sum(1) == lay.slot_valid.sum(1)).all(), "one prefix position per valid slot"
    m = _model(1)
    assert m.config.block_size == SEQ + S
    o, g = _run(m, x, y, lay)
    assert torch.isfinite(o["loss"]) and g["W_prefix.0"].abs().sum() > 0
    m.eval()
    with torch.no_grad():
        h = m(x, None, lay)["hidden"]
    assert h.shape == (x.shape[0], SEQ + S, 64)


# ------------------------------------------------------------------ 5. refusals
def test_senseless_combinations_raise():
    with pytest.raises(ValueError, match="select='latent' with cells=1"):
        _model(1, None, select="latent", repel_lambda=0.0)
    with pytest.raises(ValueError, match="repel_lambda > 0 with cells=1"):
        _model(1, None, select="none")
    _model(1, None, select="none", repel_lambda=0.0)               # the sane pair builds


def test_plain_sync_free_and_row_compact_are_exclusive():
    """row_compact runs its own loop, so sync_free would silently do nothing: build() refuses (before any GPU use)."""
    from hydra import compose, initialize_config_dir
    from lxtul.build import build
    with initialize_config_dir(str(Path(lm.__file__).parent / "configs"), version_base=None):
        cfg = compose("train", overrides=["arm=plain", "run.wandb_mode=disabled",
                                          "plain.sync_free=true", "plain.row_compact=true"])
    with pytest.raises(ValueError, match="exclusive"):
        build(cfg, "plain", compile=False)
