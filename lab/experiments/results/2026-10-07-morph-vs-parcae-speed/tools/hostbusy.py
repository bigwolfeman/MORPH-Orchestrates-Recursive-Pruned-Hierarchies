#!/usr/bin/env python3
"""CPU busy time per thread (union of cpu_op/user_annotation intervals) and launch counts, per step."""
import collections, json, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from kcat import union
tr = json.load(open(sys.argv[1])); ns = int(sys.argv[2])
ev = [e for e in tr["traceEvents"] if e.get("ph") == "X"]
t0 = min(e["ts"] for e in ev); t1 = max(e["ts"] + e["dur"] for e in ev)
by = collections.defaultdict(list); launches = collections.Counter()
for e in ev:
    if e.get("cat") in ("cpu_op", "user_annotation", "cuda_runtime", "cuda_driver"):
        by[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + e["dur"]))
    if e.get("cat") in ("cuda_runtime", "cuda_driver") and "Launch" in e["name"]:
        launches[(e["pid"], e["tid"])] += 1
print(f"wall {(t1-t0)/1e3/ns:.1f} ms/step")
for k, iv in sorted(by.items(), key=lambda kv: -union(list(kv[1]))):
    u = union(list(iv))
    if u / 1e3 / ns < 1:
        continue
    print(f"tid {k[1]}: CPU busy {u/1e3/ns:7.1f} ms/step ({100*u/(t1-t0):.0f}% of wall), launches {launches[k]/ns:.0f}/step, "
          f"{u/max(1,launches[k]):.1f} us CPU per launch")
