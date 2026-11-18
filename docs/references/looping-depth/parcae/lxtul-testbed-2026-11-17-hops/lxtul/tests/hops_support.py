"""Shared fixtures of the hop-task tests: the real tokenizer / boundary rule / task config, and a tiny synthetic
vocabulary for the tests that run a model. Not a test file."""
from __future__ import annotations

import functools
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import lxtul.data  # noqa: F401  (puts MORPH_ROOT on sys.path)
from lxtul.data import morph_cfg_of, tul_runtime
from lxtul.hops.base import HopEnv, token_texts

CONF = str(Path(__file__).resolve().parents[1] / "configs")


@functools.lru_cache(maxsize=None)
def compose_cfg(arm: str = "minimal", *extra: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(CONF, version_base=None):
        return compose("train", overrides=[f"arm={arm}", "run.name=t", "run.wandb_mode=disabled", *extra])


@functools.lru_cache(maxsize=1)
def real():
    """(tokenizer, HopEnv, tul runtime, MORPH cfg, testbed cfg with task=hops): the production objects."""
    from transformers import AutoTokenizer
    cfg = compose_cfg("minimal", "task=hops")
    mcfg = morph_cfg_of(cfg)
    rt = tul_runtime(mcfg)
    tok = AutoTokenizer.from_pretrained(str(mcfg.data.tokenizer))
    env = HopEnv(token_texts(tok), lambda s: tok(s, add_special_tokens=False)["input_ids"], rt.data_cfg.rule,
                 int(tok.eos_token_id) if tok.eos_token_id < len(tok) else 0, int(rt.data_cfg.slot_id))
    return tok, env, rt, mcfg, cfg


@functools.lru_cache(maxsize=1)
def real_task():
    from lxtul.hops.mix import build_task
    _, _, rt, mcfg, cfg = real()
    return build_task(cfg.task, mcfg, rt)


# A tiny vocabulary (V ids) whose boundary rule cuts after id STOP: names / values are plain ids, statements
# are 4-token spans. Used where a model has to run (the real vocabulary has 49k ids).
V, SLOT, EOS, STOP = 512, 4, 1, 12


def tiny_rule():
    from morph.model.tul_layout import BoundaryRule
    lut = np.zeros(V, dtype=bool)
    lut[10:20] = True
    lut[EOS] = True
    return BoundaryRule(lut, min_span=4, span_cap=12, eos_id=EOS)


def tiny_chain_vocab():
    from lxtul.hops.chain import ChainVocab
    return ChainVocab(names=np.arange(100, 300), values=np.arange(300, 400), is_=21, stop=STOP, find=22, now=23,
                      it=24, eos=EOS)


def tiny_task(seq_len: int, max_slots: int, share: float = 0.5, k_range=(2, 2), len_caps=(2,), seed: int = 1,
              rows_per_bucket: int = 4, ask: str = "heads_half"):
    """A HopTask over a synthetic chain source (eval buckets: 2 chains of 1 and 2 hops)."""
    from lxtul.hops.chain import ChainSource
    from lxtul.hops.mix import HopTask
    src = ChainSource(tiny_chain_vocab(), k_range, list(len_caps), ask, eval_k=2, eval_hops=[1, 2])
    return HopTask(sources=[src], shares=[share], seed=seed, eval_seed=seed + 1, seq_len=seq_len,
                   max_slots=max_slots, rule=tiny_rule(), slot_id=SLOT, pad_id=EOS, rows_per_bucket=rows_per_bucket,
                   val_rows_per_bucket=rows_per_bucket // 2, depths=[1, 2], manifest={})


def tiny_rt(seq_len: int, cells: int, max_slots: int):
    """What the loader and `HopEval` read of a TUL runtime: `data_cfg.spec_for` and `.rule`."""
    from morph.model.tul_layout import TulDataConfig
    return SimpleNamespace(data_cfg=TulDataConfig(rule=tiny_rule(), prefix_k=cells, slot_id=SLOT, max_slots=max_slots))
