#!/usr/bin/env python3
"""Wall ms/step between the logged steps 200 and 400 (timestamped by tsfilter), plus peak memory."""
import re, sys
for tag in sys.argv[1:]:
    L = open(f"/home/wolfe/morph-scratch/perf/r1007/runs/{tag}/run.log").read().splitlines()
    ts = {}
    for ln in L:
        m = re.match(r"([\d.]+) \[\s*(\d+)/\d+\]", ln)
        if m:
            ts[int(m.group(2))] = float(m.group(1))
    pk = re.findall(r"peak=([\d.]+)GB", "\n".join(L))
    if 200 not in ts or 400 not in ts:
        print(f"{tag}: missing step 200/400 lines ({sorted(ts)})"); continue
    ms = (ts[400] - ts[200]) * 1000 / 200
    print(f"{tag:16s} {ms:7.1f} ms/step  {6144/ms*1000:7.0f} tok/s  peak {pk[-1] if pk else '?'} GB")
