"""Read logged scalars straight from a run's LOCAL wandb record (online or offline run).

The runner keeps every run's `.wandb` protobuf under WANDB_DIR (`morph-scratch/wandb/`),
so a series can be read without the wandb API, a sync, or the run finishing. Each history
item is stored as key (0x12 len key) then value (0x82 0x01 len json); the summary repeats
the last value, so consecutive duplicates are dropped.

    python lab/divergence/wandb_local_series.py /home/wolfe/morph-scratch/wandb/run-*-e2zl3cha \
        --keys val/ce_marginal,val/ce_single_mean,val/ce_tf,val/ce_tokens --every 250

Prints one row per logged value, step = row index × --every (the trainer's val cadence).
"""
from __future__ import annotations

import argparse
import glob
import re


def series(run_dir: str, key: str) -> list[float]:
    (f,) = glob.glob(f"{run_dir}/run-*.wandb")
    b = open(f, "rb").read()
    raw = [float(x) for x in re.findall(re.escape(key.encode()) + rb"\x82\x01.([0-9.eE+-]+)", b)]
    out: list[float] = []
    for x in raw:
        if not out or x != out[-1]:
            out.append(x)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--keys", default="val/ce_marginal,val/ce_single_mean,val/ce_tf,val/ce_tokens,val/code_gap")
    ap.add_argument("--every", type=int, default=250, help="steps between logged values")
    a = ap.parse_args()
    keys = a.keys.split(",")
    cols = {k: series(a.run_dir, k) for k in keys}
    n = min(len(v) for v in cols.values())
    print("counts " + " ".join(f"{k}={len(v)}" for k, v in cols.items()))
    print("step  " + "  ".join(k.split("/")[-1][:14].rjust(14) for k in keys))
    for i in range(n):
        print(f"{(i + 1) * a.every:5d} " + "  ".join(f"{cols[k][i]:14.4f}" for k in keys))


if __name__ == "__main__":
    main()
