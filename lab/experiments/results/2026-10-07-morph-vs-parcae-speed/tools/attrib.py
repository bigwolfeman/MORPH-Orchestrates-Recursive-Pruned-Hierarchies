#!/usr/bin/env python3
"""Charge every GPU kernel (forward AND backward) to the forward R:: range that caused it.

Usage: attrib.py trace.json n_steps [depth]
A kernel launched on the forward thread is charged to its innermost R:: range there. A kernel
launched by the autograd engine is charged through the backward op's "Sequence number" to the
forward op with the same sequence number, then to the R:: ranges open around that forward op.
`depth` = how many nested R:: levels to keep in the label (default 2, outermost first).
Prints ms/step and kernels/step per label, fwd and bwd separately and summed, plus GPU idle.
"""
import collections
import json
import os
import sys


def sweep(events, points):
    """events: [(ts, end, payload)] properly nested per thread; points: [(ts, key)].
    Returns {key: [payloads of events open at ts, outermost first]}."""
    marks = [(s, 1, i) for i, (s, e, _) in enumerate(events)] + \
            [(e, -1, i) for i, (s, e, _) in enumerate(events)] + \
            [(t, 0, k) for t, k in points]
    # at equal ts: close (-1) before point (0) before open (1)? a point at an op's start
    # belongs to it, so: close, open, point
    order = {-1: 0, 1: 1, 0: 2}
    marks.sort(key=lambda m: (m[0], order[m[1]]))
    stack, out = [], {}
    for t, kind, x in marks:
        if kind == 1:
            stack.append(x)
        elif kind == -1:
            if x in stack:
                stack.remove(x)
        else:
            out[x] = [events[i][2] for i in stack]
    return out


def main():
    tr = json.load(open(sys.argv[1]))
    ns = int(sys.argv[2])
    darg = sys.argv[3] if len(sys.argv) > 3 else "2"
    if darg.startswith("o"):           # "o2i2": outermost 2 + innermost 2 levels
        o, i = int(darg[1]), int(darg[3])

        def cut(rs):
            return rs if len(rs) <= o + i else rs[:o] + ["…"] + rs[-i:]
    else:
        depth = int(darg)

        def cut(rs):
            return rs[:depth]
    ev = [e for e in tr["traceEvents"] if e.get("ph") == "X"]
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    rt = {e["args"]["correlation"]: e for e in ev
          if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    by_tid = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") in ("cpu_op", "user_annotation"):
            by_tid[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + e["dur"], e))
    # forward ops with a sequence number: their R:: stack
    fwd_seq_points = collections.defaultdict(list)
    for k, evs in by_tid.items():
        for s, e, x in evs:
            a = x.get("args", {})
            if x.get("cat") == "cpu_op" and "Sequence number" in a and a.get("Fwd thread id", 0) == 0:
                fwd_seq_points[k].append((s, ("seq", a["Sequence number"], s)))
    launch_points = collections.defaultdict(list)
    for g in gpu:
        r = rt.get(g.get("args", {}).get("correlation"))
        if r is not None:
            launch_points[(r["pid"], r["tid"])].append((r["ts"], ("k", id(g))))
    seq_label = {}
    kern_ctx = {}
    for k, evs in by_tid.items():
        pts = fwd_seq_points.get(k, []) + launch_points.get(k, [])
        if not pts:
            continue
        res = sweep(evs, pts)
        for key, stack in res.items():
            if key[0] == "seq":
                rs = [x["name"].replace("R::", "") for x in stack if x["name"].startswith("R::")]
                # first forward occurrence wins (checkpoint recompute reuses seq numbers)
                seq_label.setdefault(key[1], " > ".join(cut(rs)) if rs else "<no range>")
            else:
                kern_ctx[key[1]] = stack
    tf = collections.Counter(); tb = collections.Counter(); nf = collections.Counter(); nb = collections.Counter()
    for g in gpu:
        st = kern_ctx.get(id(g))
        if st is None:
            tf["<no launch>"] += g["dur"]; nf["<no launch>"] += 1
            continue
        bseq = None
        for x in reversed(st):
            a = x.get("args", {})
            if x.get("cat") == "cpu_op" and a.get("Fwd thread id", 0) != 0 and "Sequence number" in a:
                bseq = a["Sequence number"]
                break
        if bseq is not None:
            lab = seq_label.get(bseq, "<seq unmatched>")
            tb[lab] += g["dur"]; nb[lab] += 1
        else:
            rs = [x["name"].replace("R::", "") for x in st if x["name"].startswith("R::")]
            lab = " > ".join(cut(rs)) if rs else "<no range>"
            tf[lab] += g["dur"]; nf[lab] += 1
    if os.environ.get("ATTRIB_GROUP"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from kcat import cat as kcat_cat
        grp = collections.defaultdict(collections.Counter)

        def gkey(lab):
            last = lab.split(" > ")[-1]
            return last if last in ("attention", "hc_mlp", "hc_attn", "norm", "hc") else lab
        for g in gpu:
            st = kern_ctx.get(id(g))
            if st is None:
                continue
            bseq = None
            for x in reversed(st):
                a = x.get("args", {})
                if x.get("cat") == "cpu_op" and a.get("Fwd thread id", 0) != 0 and "Sequence number" in a:
                    bseq = a["Sequence number"]
                    break
            if bseq is not None:
                lab = seq_label.get(bseq, "<seq unmatched>")
            else:
                rs = [x["name"].replace("R::", "") for x in st if x["name"].startswith("R::")]
                lab = " > ".join(cut(rs)) if rs else "<no range>"
            grp[gkey(lab)][kcat_cat(g["name"], g.get("cat"))] += g["dur"]
        want = os.environ.get("ATTRIB_KERNELS")
        if want:
            kc = collections.Counter(); kt = collections.Counter()
            for g in gpu:
                st = kern_ctx.get(id(g))
                if st is None:
                    continue
                bseq = None
                for x in reversed(st):
                    a = x.get("args", {})
                    if x.get("cat") == "cpu_op" and a.get("Fwd thread id", 0) != 0 and "Sequence number" in a:
                        bseq = a["Sequence number"]
                        break
                rs = [x["name"].replace("R::", "") for x in st if x["name"].startswith("R::")]
                lab = seq_label.get(bseq, "") if bseq is not None else " > ".join(cut(rs))
                if all(w in lab for w in want.split("&")):
                    side = "B" if bseq is not None else "F"
                    kc[(side, g["name"][:95])] += 1; kt[(side, g["name"][:95])] += g["dur"]
            print(f"== kernels under labels containing {want!r}: {sum(kc.values())/ns:.0f}/step ==")
            for k, n in kc.most_common(30):
                print(f"{n/ns:7.0f} {kt[k]/1e3/ns:6.2f}ms {k[0]} {k[1]}")
            return
        print("== grouped by sublayer (innermost label), ms/step by kernel category ==")
        for k, c in sorted(grp.items(), key=lambda kv: -sum(kv[1].values()))[:25]:
            tot = sum(c.values())
            print(f"{tot/1e3/ns:7.1f}  {k[-70:]:70s}  " + "  ".join(f"{n}={v/1e3/ns:.1f}" for n, v in c.most_common(5)))
        return
    labs = set(tf) | set(tb)
    rows = sorted(labs, key=lambda l: -(tf[l] + tb[l]))
    print(f"{'fwd ms':>8s} {'bwd ms':>8s} {'sum ms':>8s} {'fwd n':>7s} {'bwd n':>7s}  label   (per step)")
    for l in rows:
        print(f"{tf[l]/1e3/ns:8.1f} {tb[l]/1e3/ns:8.1f} {(tf[l]+tb[l])/1e3/ns:8.1f} {nf[l]/ns:7.0f} {nb[l]/ns:7.0f}  {l}")
    print(f"{sum(tf.values())/1e3/ns:8.1f} {sum(tb.values())/1e3/ns:8.1f} "
          f"{(sum(tf.values())+sum(tb.values()))/1e3/ns:8.1f} {sum(nf.values())/ns:7.0f} {sum(nb.values())/ns:7.0f}  TOTAL")


if __name__ == "__main__":
    main()
