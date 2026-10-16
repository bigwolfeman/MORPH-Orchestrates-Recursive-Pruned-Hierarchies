"""Markdown tables from a bench4 folder (the JSON files of lxtul/bench_all.py).

    python -m lxtul.summarize_bench4 /home/wolfe/parcae-runs/bench4 > bench4.md

One training table and one inference table per batch size. The two ratio columns are tok/s of the arm
divided by tok/s of gpt8 and of flat14 at the same batch (1.00 = as fast as the flat model; below 1 = slower).
A cell that did not fit prints OOM with the peak it reached.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ORDER = ["gpt8", "gpt8_pointer", "flat14", "flat14_pointer", "plain_pointer_rc", "plain_pointer",
         "plain_pointer_rc_ckpt", "plain_pointer_ckpt", "lxtul_pointer_fast", "lxtul_pointer_fast_nockpt",
         "lxtul_pointer", "minimal", "minimal_noptr", "spiral_published", "spiral_span", "spiral_span_pair", "spiral_span_pair_mesh",
         "spiral_chunk4_loop", "spiral_chunk2_loop", "spiral_token_skip_loop",
         "spiral_cycle_end", "spiral_cycle_deep", "spiral_cycle_fixedsched",
         "spiral_cycle_deep_fixedsched", "plain_deep"]


def load(folder: Path) -> dict[tuple[str, int], dict]:
    out = {}
    for p in sorted(Path(folder).glob("*_b*.json")):
        d = json.loads(p.read_text())
        out[(d["arm"], int(d["batch"]))] = d
    return out


def _f(v, spec: str) -> str:
    return "n/a" if v is None else format(v, spec)


def _status(r: dict | None) -> str | None:
    """None when the cell has numbers; else the text for the row."""
    if r is None:
        return "not run"
    if r.get("status") == "ok":
        return None
    if r.get("status") == "OOM":
        return f"OOM (peak {_f(r.get('peak_alloc_gib'), '.1f')} GiB allocated, {_f(r.get('peak_reserved_gib'), '.1f')} reserved)"
    return str(r.get("status", "no result"))


def _ratio(cells, arm, batch, ref, mode) -> str:
    a, b = cells.get((arm, batch), {}).get(mode), cells.get((ref, batch), {}).get(mode)
    if not a or not b or a.get("status") != "ok" or b.get("status") != "ok":
        return "n/a"
    return f"{a['tok_s'] / b['tok_s']:.2f}"


def _arms(cells, batch):
    have = {a for (a, b) in cells if b == batch}
    return [a for a in ORDER if a in have] + sorted(have - set(ORDER))


def train_table(cells, batch: int) -> str:
    rows = ["| arm | params M | step ms | tok/s | peak GiB | fwd / bwd / clip / opt / ema ms | TFLOP fwd+bwd | TFLOP opt | TFLOPS | x gpt8 | x flat14 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in _arms(cells, batch):
        c = cells[(arm, batch)]
        r = c.get("train")
        bad = _status(r)
        if bad:
            rows.append(f"| {arm} | {_f(c.get('params', 0) / 1e6 if c.get('params') else None, '.0f')} | {bad} |||||||||")
            continue
        ph, fl = r.get("phases_ms", {}), r.get("flops", {})
        phs = " / ".join(_f(ph.get(k), ".1f") for k in ("forward", "backward", "clip", "optimizer", "ema"))
        rows.append(f"| {arm} | {c['params'] / 1e6:.0f} | {r['median_ms']:.1f} | {r['tok_s']:,.0f} | {r['peak_alloc_gib']:.1f} | {phs} | "
                    f"{_f(fl.get('tflop_fwd_bwd'), '.2f')} | {_f(fl.get('tflop_optimizer'), '.2f')} | {_f(r.get('achieved_tflops'), '.0f')} | "
                    f"{_ratio(cells, arm, batch, 'gpt8', 'train')} | {_ratio(cells, arm, batch, 'flat14', 'train')} |")
    return "\n".join(rows)


def infer_table(cells, batch: int) -> str:
    rows = ["| arm | ms | tok/s | peak GiB | fwd TFLOP / token | TFLOPS | x gpt8 | x flat14 |", "|---|---|---|---|---|---|---|---|"]
    for arm in _arms(cells, batch):
        r = cells[(arm, batch)].get("infer")
        bad = _status(r)
        if bad:
            rows.append(f"| {arm} | {bad} |||||||")
            continue
        fl = r.get("flops", {})
        rows.append(f"| {arm} | {r['median_ms']:.1f} | {r['tok_s']:,.0f} | {r['peak_alloc_gib']:.1f} | "
                    f"{_f(fl.get('tflop_per_token'), '.2e')} | {_f(r.get('achieved_tflops'), '.0f')} | "
                    f"{_ratio(cells, arm, batch, 'gpt8', 'infer')} | {_ratio(cells, arm, batch, 'flat14', 'infer')} |")
    return "\n".join(rows)


def render(folder: Path) -> str:
    cells = load(folder)
    out = ["# bench4: every arm timed the same way", "",
           "Step ms = median of synced steps after warm-up. tok/s counts 1024 real tokens per row. "
           "x gpt8 / x flat14 = tok/s of the arm over tok/s of that flat model at the same batch. "
           "TFLOP is forward + backward (optimizer separate); TFLOPS = (fwd+bwd+opt) / step time.", ""]
    for batch in sorted({b for (_, b) in cells}):
        out += [f"## Training, batch {batch}", "", train_table(cells, batch), "",
                f"## Inference (eval forward at depth 6), batch {batch}", "", infer_table(cells, batch), ""]
    return "\n".join(out)


if __name__ == "__main__":
    print(render(Path(sys.argv[1] if len(sys.argv) > 1 else "/home/wolfe/parcae-runs/bench4")))
