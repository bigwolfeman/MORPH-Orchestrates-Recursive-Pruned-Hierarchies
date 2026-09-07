"""Score the E10 assay draws from queue.log + probe jsonl + val txt.

Usage: python score_e10.py  -> prints a markdown table and the P10 tallies.
"""
import json, re, glob, os, statistics
Q = "/home/wolfe/morph-scratch/arc"
R = f"{Q}/results/2026-09-07-arc-e10"
E9 = f"{Q}/results/2026-09-07-arc-e9"
done = {}
for line in open(f"{Q}/queue.log"):
    m = re.match(r"\[(\d\d:\d\d:\d\d)\] DONE (\S+) exit=(\d+) verdict=(\S+)(.*)", line)
    if m:
        t, name, ex, verdict, rest = m.groups()
        step = re.search(r" step=(\d+)", rest)
        done[name] = dict(verdict=verdict, exit=int(ex), first=int(step.group(1)) if step else None, rest=rest.strip())

def val(path):
    out = {}
    if os.path.exists(path):
        for l in open(path):
            m = re.search(r"VAL\s+(\d+)\] loss=([\d.]+)", l)
            if m: out[int(m.group(1))] = float(m.group(2))
    return out

def trace(path, key):
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    tr = {r["step"]: r.get(key) for r in rows if key in r}
    pre = [(r["preclip/total"], r["step"]) for r in rows if r["step"] >= 200 and "preclip/total" in r]
    mx = max(pre) if pre else (None, None)
    at600 = tr.get(600)
    last = tr[max(tr)] if tr else None
    vals = [v for v in tr.values() if v is not None]
    return dict(at600=at600, last=last, max=max(vals) if vals else None, n=len(rows), preclip_max=mx)

arms = [
    ("E10a fp",        "fp-s3",     R,  "loss/fixed_point",  "fp"),
    ("E10b (control)", "pgain-s3",  R,  "loss/core_gain_est","gain"),
    ("E10c pgain2",    "pgain2-s4", R,  "loss/core_gain_est","gain"),
    ("E9 ctx control", "carry-ctx", E9, None, None),
]
print("| arm | draw | verdict | first cross | max preclip/total (step>=200) | val400 | val800 | key@600 | key@last | key max |")
print("|---|---|---|---|---|---|---|---|---|---|")
tally = {}
for arm, prefix, root, key, kind in arms:
    for name in sorted(n for n in done if n.startswith(prefix)):
        d = done[name]
        v = val(f"{root}/val_{name}.txt")
        tr = trace(f"{root}/probe_{name}.jsonl", key) if key else trace(f"{root}/probe_{name}.jsonl", "preclip/total")
        pm = tr["preclip_max"]
        pms = f"{pm[0]:.3g}@{pm[1]}" if pm[0] is not None else "-"
        f = lambda x: "-" if x is None else f"{x:.4g}"
        print(f"| {arm} | {name} | {d['verdict']} | {d['first'] or '-'} | {pms} | {f(v.get(400))} | {f(v.get(800))} | "
              f"{f(tr['at600']) if key else '-'} | {f(tr['last']) if key else '-'} | {f(tr['max']) if key else '-'} |")
        tally.setdefault(arm, []).append(d["verdict"])
print()
for arm, vs in tally.items():
    n = len(vs); det = sum(v == "DETONATED" for v in vs)
    print(f"{arm}: {det}/{n} detonated  ({', '.join(vs)})")
