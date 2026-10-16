"""Score several finished runs on ONE run's held-out rows, so every gap is paired per row.

  python -m lxtul.paired_eval --rows-from RUN --out paired.json RUN [RUN ...] [--depths 1 6]

The eval rows of a TUL run follow its cell count (1024 + 64 * cells positions; the packer fills unused
slot positions with tokens), so a 1-cell run and a 4-cell run score DIFFERENT tokens and their
train-time sweeps are not paired. Here every run is scored on the rows of `--rows-from`: a TUL run
with the same cell count reads them as they are, and a plain or gpt run reads their token positions
(`evaluate.plain_view`). A TUL run whose cell count differs from the row source's is refused.

Writes per-run CE per depth and, for every run after the first, the paired bootstrap CI of
(run - first run) at each depth over the rows (MORPH's `paired_bootstrap_ci`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lxtul.data import morph_cfg
from lxtul.evaluate import _stats, depth_sweep, eval_rows
from lxtul.gap_probe import load_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-from", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 6])
    ap.add_argument("runs", type=Path, nargs="+")
    a = ap.parse_args()
    src, src_kind, cfg, rt = load_run(a.rows_from)
    if src_kind != "lxtul":
        raise ValueError("--rows-from must be a TUL run: its cell count fixes the row layout")
    width = int(rt.data_cfg.prefix_k)
    del src
    rows = eval_rows(morph_cfg(str(cfg.morph.config)), rt, a.rows, 6)
    slot_id = int(rt.data_cfg.slot_id)
    sweeps = {}
    for run in a.runs:
        m, kind, _, rrt = load_run(run)
        if kind == "lxtul" and int(rrt.data_cfg.prefix_k) != width:
            raise ValueError(f"{run.name}: {rrt.data_cfg.prefix_k} cells per slot, rows have {width}")
        sweeps[run.name] = depth_sweep(m.cuda(), kind, rows, a.depths, slot_id)
        del m
    ci = _stats().paired_bootstrap_ci
    names = list(sweeps)
    cnt = np.asarray(sweeps[names[0]]["count"])
    for n in names[1:]:
        if not np.array_equal(np.asarray(sweeps[n]["count"]), cnt):
            raise RuntimeError(f"{n}: scored token counts differ from {names[0]} on the same rows")
    res = {"rows_from": a.rows_from.name, "cells_per_slot": width, "tokens": float(cnt.sum()),
           "sweeps": sweeps,
           "ce": {n: {str(k): float(np.sum(sw["sums"][str(k)]) / cnt.sum()) for k in a.depths}
                  for n, sw in sweeps.items()},
           "minus_first": {n: {str(k): ci(np.asarray(sweeps[n]["sums"][str(k)]),
                                          np.asarray(sweeps[names[0]]["sums"][str(k)]), cnt)
                               for k in a.depths} for n in names[1:]}}
    a.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["ce"]), flush=True)
    for n, d in res["minus_first"].items():
        print(n, "-", names[0], {k: "%+.4f [%+.4f, %+.4f]" % (v["point"], v["lo"], v["hi"]) for k, v in d.items()},
              flush=True)


if __name__ == "__main__":
    main()
