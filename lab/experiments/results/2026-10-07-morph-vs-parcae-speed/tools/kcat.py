#!/usr/bin/env python3
"""Kernel-category breakdown of a kineto chrome trace (model-agnostic), per step.

Usage: kcat.py trace.json n_steps [--top N] [--cat CATNAME]
Categories by kernel name. GPU busy = union of kernel/memcpy/memset intervals.
"""
import collections
import json
import sys

RULES = [  # first match wins
    ("memcpy/memset", lambda n, c: c in ("gpu_memcpy", "gpu_memset")),
    ("attention", lambda n, c: any(s in n for s in ("flash", "fmha", "attention", "attn", "flex", "triton_tem_fused_flex"))),
    ("gemm", lambda n, c: any(s in n for s in ("gemm", "nvjet", "cutlass", "xmma", "cublas", "Kernel2", "sm80_", "sm90_", "sm100_", "sm120_", "splitKreduce"))),
    ("inductor-fused", lambda n, c: n.startswith(("triton_poi", "triton_red", "triton_per", "triton_tem"))),
    ("optimizer/foreach", lambda n, c: "multi_tensor_apply" in n or "foreach" in n.lower() or "ademamix" in n.lower() or "adam" in n.lower()),
    ("custom-triton", lambda n, c: not n.startswith(("void ", "at::", "cutlass")) and "(" not in n),
    ("copy/cast", lambda n, c: any(s in n for s in ("copy_kernel", "direct_copy", "CatArrayBatchedCopy", "_to_copy", "cast"))),
    ("index/scatter/gather", lambda n, c: any(s in n.lower() for s in ("index", "scatter", "gather", "masked"))),
    ("reduce", lambda n, c: "reduce_kernel" in n or "Reduce" in n),
    ("softmax/norm", lambda n, c: any(s in n.lower() for s in ("softmax", "norm", "logsumexp"))),
    ("elementwise", lambda n, c: "elementwise" in n),
    ("other", lambda n, c: True),
]


def cat(n, c):
    for name, f in RULES:
        if f(n, c):
            return name


def union(iv):
    iv.sort()
    tot = 0.0
    cs = ce = None
    for s, e in iv:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs
            cs, ce = s, e
    if cs is not None:
        tot += ce - cs
    return tot


def main():
    tr = json.load(open(sys.argv[1]))
    ns = int(sys.argv[2])
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 0
    want = sys.argv[sys.argv.index("--cat") + 1] if "--cat" in sys.argv else None
    ev = [e for e in tr["traceEvents"] if e.get("ph") == "X"]
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    t0 = min(e["ts"] for e in ev)
    t1 = max(e["ts"] + e["dur"] for e in ev)
    g0 = min(e["ts"] for e in gpu)
    g1 = max(e["ts"] + e["dur"] for e in gpu)
    busy = union([(e["ts"], e["ts"] + e["dur"]) for e in gpu])
    span = g1 - g0
    print(f"trace wall {(t1-t0)/1e3/ns:.1f} ms/step; GPU span {span/1e3/ns:.1f}; busy {busy/1e3/ns:.1f} "
          f"({100*busy/span:.0f}% of span); kernels {len(gpu)/ns:.0f}/step; "
          f"mean kernel {busy/len(gpu):.1f} us")
    t = collections.Counter()
    k = collections.Counter()
    names = collections.defaultdict(collections.Counter)
    ncount = collections.defaultdict(collections.Counter)
    for e in gpu:
        c = cat(e["name"], e.get("cat"))
        t[c] += e["dur"]
        k[c] += 1
        names[c][e["name"][:110]] += e["dur"]
        ncount[c][e["name"][:110]] += 1
    tot = sum(t.values())
    print(f"{'category':22s} {'ms/step':>8s} {'%sum':>5s} {'n/step':>7s} {'us/kern':>8s}")
    for c, d in t.most_common():
        print(f"{c:22s} {d/1e3/ns:8.1f} {100*d/tot:5.1f} {k[c]/ns:7.0f} {d/k[c]:8.1f}")
    small = sum(e["dur"] for e in gpu if e["dur"] < 10)
    nsmall = sum(1 for e in gpu if e["dur"] < 10)
    print(f"kernels < 10 us: {nsmall/ns:.0f}/step, {small/1e3/ns:.1f} ms/step")
    for c in ([want] if want else (list(t) if top else [])):
        print(f"\n== {c} top {top or 25} ==")
        for n, d in names[c].most_common(top or 25):
            print(f"{d/1e3/ns:8.2f} {ncount[c][n]/ns:6.0f}  {n}")


if __name__ == "__main__":
    main()
