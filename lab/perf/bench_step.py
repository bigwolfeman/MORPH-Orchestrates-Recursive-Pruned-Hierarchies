"""The one steady-state speed number for a MORPH config.

Runs each arm as its own `python -m morph.training.train` process under the GPU lock, arms
interleaved (A, B, A, B, ...) so slow drift of the machine hits every arm alike. Each run turns
on `training.step_time_print`, so the trainer prints the wall clock at the end of every step.
A step is asynchronous (the host runs ahead of the GPU) and only the trainer's logging steps
(every 20) sync, so one step's own duration is noise. Two stamps taken 20 steps apart at the
same phase are exact, so the bench cuts the steady-state part (step `--skip` to the end) into
20-step blocks between logging steps, turns each block into ms/step, and reports the median,
p10 and p90 of those block means.

    python lab/perf/bench_step.py \
        --arm off=nowta_pointer_fast_spdcells_speed \
        --arm on=nowta_pointer_fast_spdcells_speed:"model.slot_gain_rows=2" \
        --steps 300 --skip 100 --repeats 2 --out result.json

Arm syntax: name=config_name[:hydra overrides separated by spaces, quoted]. The overrides are
split with shlex, so quote one that holds a space.

Why not the trainer's `tok/s=` line: its sps is a rolling mean of the last 100 steps, so up to
step 200 it still holds the compile warm-up, and it prints only every 20 steps.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(os.environ.get("MORPH_BENCH_SCRATCH", "/home/wolfe/morph-scratch/bench-step"))
GPU_LOCK = "/home/wolfe/morph-scratch/gpu.lock"
PYTHON = "/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python"

STEP_RE = re.compile(r"^\[steptime\] step=(\d+) t=([0-9.]+) tokens=(\d+)\s*$")
BLOCK = 20          # the trainer's logging interval: one device sync inside every block


def parse_steptime(lines):
    """[(step, t_seconds, tokens)] from the trainer's `[steptime]` lines; others are ignored."""
    out = []
    for line in lines:
        m = STEP_RE.match(line.rstrip("\n"))
        if m:
            out.append((int(m.group(1)), float(m.group(2)), int(m.group(3))))
    return out


def percentile(sorted_vals, q):
    """Linear-interpolated percentile (q in 0..100) of an already sorted list."""
    if not sorted_vals:
        raise ValueError("percentile of an empty list")
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def block_ms(steps, skip, block=BLOCK):
    """ms/step of every `block`-step window whose two end stamps are at steps >= skip and
    multiples of `block` (the trainer's syncing steps)."""
    stamp = {s: t for s, t, _ in steps if s >= skip and s % block == 0}
    return [(stamp[s] - stamp[s - block]) / block * 1e3
            for s in sorted(stamp) if s - block in stamp]


def steady_state(steps, skip, block=BLOCK):
    """Per-run numbers over the steady-state blocks. Raises if none fit."""
    ms = block_ms(steps, skip, block)
    if not ms:
        raise ValueError(f"no complete {block}-step block at step >= {skip} "
                         f"(got {len(steps)} [steptime] lines)")
    ms = sorted(ms)
    med = statistics.median(ms)
    tokens = steps[0][2]
    return {"n_blocks": len(ms), "median_ms": med, "p10_ms": percentile(ms, 10),
            "p90_ms": percentile(ms, 90), "mean_ms": sum(ms) / len(ms),
            "tokens_per_step": tokens, "tok_s": tokens / (med / 1e3), "_ms": ms}


def aggregate(runs):
    """Per-arm summary from {arm: [per-run steady_state dicts]}.

    The arm's distribution pools every block of every repeat; `repeat_median_ms` keeps each
    repeat apart so run-to-run noise is visible. With exactly two arms, `ratio_of_medians`
    is arms[0] / arms[1] on the pooled median (below 1 means the first arm is faster).
    """
    arms = {}
    for name, rs in runs.items():
        pooled = sorted(ms for r in rs for ms in r["_ms"])
        med = statistics.median(pooled)
        tokens = rs[0]["tokens_per_step"]
        arms[name] = {"median_ms": med, "p10_ms": percentile(pooled, 10),
                      "p90_ms": percentile(pooled, 90), "tokens_per_step": tokens,
                      "tok_s": tokens / (med / 1e3),
                      "repeat_median_ms": [r["median_ms"] for r in rs],
                      "repeat_tok_s": [r["tok_s"] for r in rs], "n_blocks": len(pooled)}
    out = {"arms": arms}
    names = list(arms)
    if len(names) == 2:
        out["ratio_of_medians"] = {"num": names[0], "den": names[1],
                                   "value": arms[names[0]]["median_ms"] / arms[names[1]]["median_ms"]}
    return out


def format_table(result):
    rows = [f"{'arm':<14}{'median ms':>11}{'p10':>9}{'p90':>9}{'tok/s':>10}  per-repeat median ms"]
    for name, a in result["arms"].items():
        reps = ", ".join(f"{m:.2f}" for m in a["repeat_median_ms"])
        rows.append(f"{name:<14}{a['median_ms']:>11.2f}{a['p10_ms']:>9.2f}{a['p90_ms']:>9.2f}"
                    f"{a['tok_s']:>10.0f}  {reps}")
    r = result.get("ratio_of_medians")
    if r:
        rows.append(f"ratio of medians {r['num']}/{r['den']} = {r['value']:.4f}")
    return "\n".join(rows)


def parse_arm(spec):
    if "=" not in spec:
        raise ValueError(f"--arm {spec!r}: expected name=config_name[:overrides]")
    name, rest = spec.split("=", 1)
    config, _, ov = rest.partition(":")
    if not name or not config:
        raise ValueError(f"--arm {spec!r}: empty name or config")
    return name, config, shlex.split(ov)


def run_one(name, config, overrides, steps, tag, workdir):
    """One trainer process. Returns the parsed [steptime] rows; raises if it fails."""
    run_dir = Path(tempfile.mkdtemp(prefix=f"{tag}-", dir=workdir))
    wb_name = f"bench-{tag}-{os.getpid()}"
    env = dict(os.environ, PYTHONPATH=str(ROOT), WANDB_MODE="offline", WANDB_DIR=str(run_dir),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               TORCHINDUCTOR_COMPILE_THREADS="2")
    cmd = ["flock", GPU_LOCK, PYTHON, "-m", "morph.training.train", "--config-name", config,
           f"training.steps={steps}", "training.eval_every=100000", "training.ckpt_every=100000",
           "training.step_time_print=true", f"wandb.name={wb_name}",
           f"hydra.run.dir={run_dir / 'hy'}", *overrides]
    log_path = run_dir / "stdout.log"
    ckpt = ROOT / "checkpoints" / "morph" / wb_name
    try:
        with open(log_path, "w") as lf:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        rows = parse_steptime(open(log_path).readlines())
        if rc != 0:
            tail = "".join(open(log_path).readlines()[-25:])
            raise RuntimeError(f"arm {name} exited {rc}; log {log_path}\n{tail}")
        return rows
    finally:
        shutil.rmtree(ckpt, ignore_errors=True)
        # the log is kept on failure (the raise above names it); removed on success
        if sys.exc_info()[0] is None:
            shutil.rmtree(run_dir, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="name=config[:overrides]")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--skip", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args(argv)
    if not 0 <= a.skip < a.steps:
        ap.error("need 0 <= --skip < --steps")
    arms = [parse_arm(s) for s in a.arm]
    if len({n for n, _, _ in arms}) != len(arms):
        ap.error("arm names must be unique")
    SCRATCH.mkdir(parents=True, exist_ok=True)

    runs = {n: [] for n, _, _ in arms}
    for rep in range(a.repeats):
        for name, config, ov in arms:                       # A, B, A, B, ...
            t0 = time.time()
            rows = run_one(name, config, ov, a.steps, f"{name}-r{rep}", SCRATCH)
            ss = steady_state(rows, a.skip)
            runs[name].append(ss)
            print(f"[bench] {name} repeat {rep}: median {ss['median_ms']:.2f} ms "
                  f"({ss['tok_s']:.0f} tok/s) in {time.time() - t0:.0f}s", flush=True)
    result = aggregate(runs)
    result["config"] = {"arms": a.arm, "steps": a.steps, "skip": a.skip, "repeats": a.repeats}
    print(format_table(result))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
