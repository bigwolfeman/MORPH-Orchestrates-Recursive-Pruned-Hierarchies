"""Leak check for output-mixture arms: score a checkpoint with its copy cache / pointer ON and OFF.

    python -m lxtul.mixture_off_eval /home/wolfe/parcae-runs/lxtul-pointer-5k --out f.json

OFF is the model's own distribution. An output-only channel cannot teach the model to copy, so
OFF must read near strict LXTUL (4.12 at 5k); OFF near the ON number would mean a copy path
inside the model.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from lxtul.data import morph_cfg, tul_runtime
from lxtul.evaluate import depth_sweep, eval_rows
from lxtul.gap_probe import load_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    m, kind, cfg = load_run(a.run)
    m.cuda()
    mcfg = morph_cfg(str(cfg.morph.config)); rt = tul_runtime(mcfg)
    rows = eval_rows(mcfg, rt, 480, 6)
    res = {}
    for name in ("on", "off"):
        if kind == "plain":                       # plain + pointer (lxtul/plain_pointer.py)
            m.use_pointer = name == "on"
        else:
            on = m.tul if name == "on" else on
            m.tul = on if name == "on" else dataclasses.replace(on, copy_cache=False, pointer_heads=0)
        sw = depth_sweep(m, kind, rows, [1, 6], int(rt.data_cfg.slot_id))
        res[name] = {k: sum(v) / sum(sw["count"]) for k, v in sw["sums"].items()}
    if kind == "plain":
        m.use_pointer = True
    else:
        m.tul = on
    a.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
