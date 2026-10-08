"""MORPH's OWT stream, tokenizer and TUL packer, served to the Parcae testbed unchanged.

The testbed reads the SAME rows as MORPH: the same StarCoder2 tokenizer, the same
unshuffled OWT shard order, the same boundary rule and the same fixed-shape packer
(`morph.model.tul_layout.pack_tul_batch`). So a CE here is comparable to a MORPH
ledger CE, and a K-sweep here can use MORPH's 480 rows.

`morph_cfg()` composes a MORPH config from MORPH's own config dir (MORPH_ROOT).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MORPH_ROOT = Path(os.environ.get(
    "MORPH_ROOT",
    "/mnt/BigAssDrive/00projects/00DeepNet/00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies"))
if str(MORPH_ROOT) not in sys.path:
    sys.path.append(str(MORPH_ROOT))


def morph_cfg(name: str, overrides: list[str] | tuple[str, ...] = ()):
    """Compose a MORPH config from MORPH's own config dir, in a private Hydra context.

    The caller's Hydra context (a running @hydra.main) is saved and put back, so this
    works inside the testbed's entry points and outside them."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    gh = GlobalHydra.instance()
    saved = gh.hydra if gh.is_initialized() else None
    gh.clear()
    try:
        with initialize_config_dir(str(MORPH_ROOT / "morph" / "configs"), version_base=None):
            return compose(config_name=name, overrides=list(overrides))
    finally:
        if saved is not None:
            GlobalHydra.instance().initialize(saved)


def tul_runtime(mcfg):
    """MORPH's resolved TUL runtime: rule, layout spec, slot id. None for a plain cfg."""
    from morph.training.tul_setup import build_tul_runtime
    return build_tul_runtime(mcfg, cache_dir=str(MORPH_ROOT / "ignore" / "tul_cache"))


def make_loader(mcfg, split: str, batch_size: int, tul_rt=None, prefetch: int = 4):
    """Infinite iterator of MORPH batches: (x, y) plain, (x, y, SlotLayout) with TUL.

    `split="validation"` skips the first 50k documents, as MORPH's val loader does.
    """
    from morph.training.data import create_dataloader
    from morph.training.data_placement import Prefetcher
    val = split == "validation"
    tul = None if tul_rt is None else (tul_rt.val_data_cfg if val else tul_rt.data_cfg)
    it = iter(create_dataloader(str(mcfg.data.tokenizer), str(mcfg.data.dataset),
                                int(mcfg.data.seq_len), batch_size, split=split,
                                skip_samples=50_000 if val else 0, tul=tul))
    if prefetch > 0:
        it = Prefetcher(it, depth=prefetch, name=f"owt-{split}", pin=True)
    return it
