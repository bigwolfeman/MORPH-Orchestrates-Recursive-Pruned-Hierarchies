"""Shared readers for scoring `core_depth_sweep.py` panels (E18, E19, ...).

Between-arm CE differences pair at the TOKEN level: every arm scores the same validation
stream, but each cut packs it differently, so ROWS are not the same text across arms (E18:
the row means of two widths correlate at 0.098 across 480 rows against 1.000 within an arm).
A sweep JSON that carries ``tokens_npz`` has a per-token CE array keyed by stream index; the
pairing intersects the indices and bootstraps over 1,024-token stream BLOCKS (tokens in a row
are not independent). Without the token files the row-mean fallback is labelled UNPAIRED.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

BLOCK = 1024  # stream tokens per bootstrap unit
N_BOOT, SEED = 2000, 0
_TOK: dict[str, dict | None] = {}


def load_sweep(path: str) -> dict | None:
    """The one arm inside a `core_depth_sweep.py` JSON, or None if the file is missing."""
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return next(iter(d.values()))


def ci(s: dict, key: str) -> str:
    c = s["ci_ce_tokens"][key]
    return f"{c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"


def tokens(s: dict, search_dirs: tuple[str, ...] = ()) -> dict | None:
    """The per-token CE arrays of a sweep (``tok_index`` + ``ce_<depth>``), if it wrote them.

    The JSON records the path the sweep wrote to; ``search_dirs`` are tried by basename when
    that path has moved (artifacts live under ``ignored/experiment-artifacts/``)."""
    p = s.get("tokens_npz")
    if not p:
        return None
    if p not in _TOK:
        cands = [p] + [os.path.join(d, os.path.basename(p)) for d in search_dirs]
        hit = next((c for c in cands if os.path.exists(c)), None)
        _TOK[p] = dict(np.load(hit)) if hit else None
    return _TOK[p]


def paired(a: dict, da: str, b: dict, db: str, search_dirs: tuple[str, ...] = ()
           ) -> tuple[float, float, float, str]:
    """Token CE of sweep a at depth da minus sweep b at depth db, with a 95 % bootstrap CI.

    Token-level when both sweeps carry per-token arrays (a sweep against itself pairs
    exactly, which is how within-arm K-differences not in ``ci_ce_tokens`` are read);
    otherwise the row-mean fallback (UNPAIRED across cuts)."""
    ta, tb = tokens(a, search_dirs), tokens(b, search_dirs)
    rng = np.random.default_rng(SEED)
    if ta is not None and tb is not None:
        ia, ib = ta["tok_index"], tb["tok_index"]
        common, pa, pb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
        d = ta[f"ce_{da}"][pa].astype(np.float64) - tb[f"ce_{db}"][pb].astype(np.float64)
        blk = common // BLOCK
        ub, inv = np.unique(blk, return_inverse=True)
        bsum = np.bincount(inv, weights=d, minlength=len(ub))
        bcnt = np.bincount(inv, minlength=len(ub)).astype(float)
        idx = rng.integers(0, len(ub), size=(N_BOOT, len(ub)))
        boots = bsum[idx].sum(1) / bcnt[idx].sum(1)
        return (float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)),
                f"tokens n={len(common)} blocks={len(ub)}")
    na, nb = np.asarray(a["row_n_tokens"], float), np.asarray(b["row_n_tokens"], float)
    n = min(len(na), len(nb))
    d = np.asarray(a["row_ce_sum"][da][:n]) / na[:n] - np.asarray(b["row_ce_sum"][db][:n]) / nb[:n]
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = (d * nb[:n])[idx].sum(1) / nb[:n][idx].sum(1)
    return (float((d * nb[:n]).sum() / nb[:n].sum()), float(np.quantile(boots, 0.025)),
            float(np.quantile(boots, 0.975)), "UNPAIRED rows")


def fmt(t: tuple) -> str:
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}] ({t[3]})"


def val_curve(log: str, final: bool = False) -> dict[int, float]:
    """step -> held-out loss from a trainer log's `[VAL step] loss=` lines.

    With ``final`` the `Final val_loss=` line (the eval after the last step) is keyed at
    the run's last step + 1."""
    text = open(log).read()
    v = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"\[VAL\s+(\d+)\] loss=([0-9.]+)", text)}
    m = re.search(r"Final val_loss=([0-9.]+)", text)
    if final and m and v:
        v[max(v) + 1] = float(m.group(1))
    return v


def peak_gb(log: str) -> float:
    return max(float(x) for x in re.findall(r"peak=([0-9.]+)GB", open(log).read()))


def wall_h(queue: str, prefix: str) -> dict[str, float]:
    """Arm -> hours between its START and DONE stamps in the queue log."""
    t: dict[str, dict[str, int]] = {}
    for m in re.finditer(rf"(START|DONE) ({prefix}[\w-]+) .*?epoch=(\d+)", open(queue).read()):
        t.setdefault(m.group(2), {})[m.group(1)] = int(m.group(3))
    return {a: (v["DONE"] - v["START"]) / 3600 for a, v in t.items() if "START" in v and "DONE" in v}


def verdicts(queue: str, prefix: str) -> dict[str, str]:
    return {m.group(1): m.group(2)
            for m in re.finditer(rf"DONE ({prefix}[\w-]+) exit=\d+ epoch=\d+ verdict=(\S+.*?) Final",
                                 open(queue).read())}


def probe_stats(path: str, window: tuple[int, int] = (1000, 5000)) -> dict:
    """Tripwire peak after step 200 and, where the hinge logs, its gain mean / max / active fraction."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    w = [r for r in rows if window[0] <= r["step"] <= window[1]]
    g = [r["loss/gain_est"] for r in w if "loss/gain_est" in r]
    out = {"preclip_max_after_200": max((round(r["preclip/total"]), r["step"]) for r in rows if r["step"] >= 200)}
    if g:
        out.update(gain_mean=round(float(np.mean(g)), 4), gain_max=round(float(max(g)), 4),
                   hinge_frac=round(sum(1 for r in w if r.get("loss/gain_reg_weighted", 0) > 0) / len(w), 4))
    return out


def last_prune(log: str) -> tuple[int, float] | None:
    """(step, density) from the trainer's LAST `[prune] step N: ... density=D` line, or None
    if the run never pruned (the CLAUDE.md gotcha: a 'sparse' run whose density never fell)."""
    hits = re.findall(r"\[prune\] step (\d+): .*?density=([0-9.]+)", open(log).read())
    return (int(hits[-1][0]), float(hits[-1][1])) if hits else None


def prune_at(log: str, step: int) -> float | None:
    """Density after the last prune event at or before `step` (1.0 if none yet), or None if
    the run never pruned."""
    hits = [(int(s), float(d)) for s, d in
            re.findall(r"\[prune\] step (\d+): .*?density=([0-9.]+)", open(log).read())]
    if not hits:
        return None
    before = [d for s, d in hits if s <= step]
    return before[-1] if before else 1.0
