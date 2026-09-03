"""Score dmorph run logs against the preregs from their ``[VAL step]`` and step lines.

    python lab/dmorph/score_logs.py RESULTS_DIR/dmorph-*-s1-5k.log [--last 3] [--from-step 1000]

For every log: the mean over the last ``--last`` evals of every key the VAL line carries
(loss = the clean head, dm_ce = the one-pass noisy head, ladder_ce, lad_r0/lad_r2/auroc_r2
under FPF, ...) and the mean ``tok/s`` over training steps ≥ ``--from-step``. Reads the
trainer's own text; nothing is recomputed. The prereg reads
(``lab/experiments/failures/2026-09-03-dmorph-v1-panel.md``,
``2026-09-03-dmorph-fpf-tok.md``) are these means — never one eval (the per-eval spread
is ~0.085 nats, memory ``morph-warmup-pair-verdict``).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VAL_RE = re.compile(r"\[VAL\s+(\d+)\]\s+(.*)")
KV_RE = re.compile(r"([A-Za-z0-9_/]+)=([-+0-9.eEna]+)")
STEP_RE = re.compile(r"\[\s*(\d+)/(\d+)\].*?tok/s=(\d+)")
FINAL_RE = re.compile(r"Final val_loss=(.*)")
# The trainer's final eval is printed as ``Final val_loss=... val/<key>=...`` rather than
# as a ``[VAL step]`` line; its keys are renamed onto the VAL line's names here so the
# last-N mean includes the run's last eval.
FINAL_RENAME = {"val_loss": "loss", "val/dm_ce": "dm_ce", "val/dm_fm": "dm_fm",
                "val/dm_fm_band0": "dm_fm_b0", "val/dm_ladder_ce": "ladder_ce",
                "val/dm_cos": "cos", "val/dm_ladder_ce_r0": "lad_r0",
                "val/dm_ladder_ce_r2": "lad_r2", "val/dm_resid_auroc_r2": "auroc_r2",
                "val/first_tok_ce": "first_tok", "val/layer_passes_per_token": "lp/tok",
                "val/ppl_tokens": "ppl_tok", "val/first_tok_counterfactual": "cf"}


def parse(path: Path) -> tuple[list[tuple[int, dict[str, float]]], list[tuple[int, float]]]:
    evals, steps = [], []
    total_steps = 0
    for line in path.read_text(errors="replace").splitlines():
        m = VAL_RE.search(line)
        if m:
            kv = {}
            for k, v in KV_RE.findall(m.group(2)):
                try:
                    kv[k] = float(v)
                except ValueError:
                    pass
            evals.append((int(m.group(1)), kv))
            continue
        m = STEP_RE.search(line)
        if m:
            steps.append((int(m.group(1)), float(m.group(3))))
            total_steps = int(m.group(2))
            continue
        m = FINAL_RE.search(line)
        if m:
            kv = {}
            for k, v in KV_RE.findall("val_loss=" + m.group(1)):
                name = FINAL_RENAME.get(k, k[4:] if k.startswith("val/") else k)
                if name in FINAL_RENAME.values() or name == "ppl":
                    try:
                        kv[name] = float(v)
                    except ValueError:
                        pass
            evals.append((total_steps, kv))     # the final eval runs at training.steps
    return evals, steps


def summarize(path: Path, last: int, from_step: int) -> dict[str, float]:
    evals, steps = parse(path)
    if not evals:
        raise SystemExit(f"{path}: no [VAL] lines")
    tail = evals[-last:]
    keys = sorted({k for _s, kv in tail for k in kv})
    out = {"evals_used": float(len(tail)), "last_eval_step": float(tail[-1][0])}
    for k in keys:
        vals = [kv[k] for _s, kv in tail if k in kv and kv[k] == kv[k]]
        if vals:
            out[k] = sum(vals) / len(vals)
    tps = [t for s, t in steps if s >= from_step]
    if tps:
        out["tok_s"] = sum(tps) / len(tps)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--last", type=int, default=3, help="evals averaged (default 3)")
    ap.add_argument("--from-step", type=int, default=1000, help="tok/s averaged from this step")
    a = ap.parse_args(argv)
    rows = {p.stem: summarize(p, a.last, a.from_step) for p in a.logs}
    keys = sorted({k for r in rows.values() for k in r})
    names = list(rows)
    w = max(len(n) for n in names) + 2
    print("key".ljust(18) + "".join(n.rjust(w) for n in names))
    for k in keys:
        print(k.ljust(18) + "".join(
            (f"{rows[n][k]:.4f}" if k in rows[n] else "-").rjust(w) for n in names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
