"""training.injection_lr_mult — a third optimizer param group for the injection

parameters (DiagonalInjection's B/log_A/log_dt, arc E20 e20-carry-lr20x).

Contract:
  - mult == 1.0 (default): today's two-group layout, byte-identical — no `lr_mult`
    key anywhere, `param_group_names` returns exactly 2 name-lists.
  - mult != 1.0: a THIRD group holding exactly the "injection"-named parameters,
    weight_decay 0.0, `lr_mult` == mult. Its lr is `mult`x the base lr both at
    construction and after every trainer schedule write
    (`pg["lr"] = lr * pg.get("lr_mult", 1.0)`, `morph/training/train.py`).
  - `param_group_names(model, injection_lr_mult=mult)` lines up 1:1 with the live
    optimizer's `param_groups` at the same mult, in both group COUNT and per-group
    parameter identity — this is what `align_optimizer_state` depends on for a resume.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.training.optimizer import create_optimizer, param_group_names

V = 64


def _model_cfg(**kw) -> MORPHConfig:
    base = dict(
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=V,
        max_seq_len=128,
        context_len=128,
        n_prelude=1,
        n_core=2,
        n_coda=1,
        mean_depth=2,
        max_depth=3,
        bptt_depth=2,
        channel_dims=(32, 20, 12),
        compression=2,
        csa_compress_ratio=4,
        hca_compress_ratio=8,
        top_k=8,
        window_size=16,
        retention=False,
        bigram_hash_vocab=V,
        use_kernels=False,
        hc_use_kernel=False,
        dropout=0.0,
        injection_channels="all",
        injection_B=True,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _model(seed=3, **kw) -> MORPHTransformer:
    import torch

    torch.manual_seed(seed)
    return MORPHTransformer(_model_cfg(**kw))


def _train_cfg(injection_lr_mult: float = 1.0, lr: float = 1e-4):
    return OmegaConf.create(
        {
            "training": {
                "lr": lr,
                "weight_decay": 0.1,
                "optimizer": "adamw",  # torch.optim.AdamW fallback — CPU-only, no bnb needed
                "adam8bit": False,
                "injection_lr_mult": injection_lr_mult,
            }
        }
    )


def _injection_param_names(model) -> set[str]:
    return {n for n, p in model.named_parameters() if p.requires_grad and "injection" in n}


def test_mult_one_is_the_old_two_group_layout():
    m = _model()
    opt = create_optimizer(m, _train_cfg(injection_lr_mult=1.0))
    assert len(opt.param_groups) == 2
    for g in opt.param_groups:
        assert "lr_mult" not in g
    names = param_group_names(m, injection_lr_mult=1.0)
    assert len(names) == 2
    # every injection param is still in the no-decay (group 1) name list
    inj = _injection_param_names(m)
    assert inj and inj <= set(names[1])
    assert not (inj & set(names[0]))


def test_mult_20_makes_a_third_group_of_exactly_the_injection_params():
    m = _model()
    inj_names = _injection_param_names(m)
    assert inj_names, "fixture must actually have injection params (injection_B=True)"

    opt = create_optimizer(m, _train_cfg(injection_lr_mult=20.0, lr=1e-4))
    assert len(opt.param_groups) == 3

    g0, g1, g2 = opt.param_groups
    assert "lr_mult" not in g0 and "lr_mult" not in g1
    assert g2["lr_mult"] == 20.0
    assert g2["weight_decay"] == 0.0

    # group 2 holds exactly the injection tensors (by identity), nothing else.
    inj_params_by_id = {id(p) for n, p in m.named_parameters() if n in inj_names}
    g2_ids = {id(p) for p in g2["params"]}
    assert g2_ids == inj_params_by_id
    g0_ids = {id(p) for p in g0["params"]}
    g1_ids = {id(p) for p in g1["params"]}
    assert not (g0_ids & g2_ids) and not (g1_ids & g2_ids)

    # lr set at CONSTRUCTION, before any trainer schedule write.
    assert abs(g2["lr"] - 1e-4 * 20.0) < 1e-12
    assert abs(g0["lr"] - 1e-4) < 1e-12
    assert abs(g1["lr"] - 1e-4) < 1e-12


def test_mult_20_lr_after_one_schedule_application():
    """Simulate the trainer's per-step write: pg["lr"] = lr * pg.get("lr_mult", 1.0)."""
    m = _model()
    opt = create_optimizer(m, _train_cfg(injection_lr_mult=20.0, lr=1e-4))

    new_lr = 3e-4  # a later point on the schedule, distinct from the construction lr
    for pg in opt.param_groups:
        pg["lr"] = new_lr * pg.get("lr_mult", 1.0)

    g0, g1, g2 = opt.param_groups
    assert abs(g0["lr"] - new_lr) < 1e-12
    assert abs(g1["lr"] - new_lr) < 1e-12
    assert abs(g2["lr"] - new_lr * 20.0) < 1e-12


def test_param_group_names_matches_optimizer_groups_one_to_one():
    for mult in (1.0, 20.0):
        m = _model()
        opt = create_optimizer(m, _train_cfg(injection_lr_mult=mult))
        names = param_group_names(m, injection_lr_mult=mult)
        assert len(names) == len(opt.param_groups)
        for name_list, g in zip(names, opt.param_groups):
            assert len(name_list) == len(g["params"])
            ids_by_name = {id(p) for n, p in m.named_parameters() if n in name_list}
            assert ids_by_name == {id(p) for p in g["params"]}
