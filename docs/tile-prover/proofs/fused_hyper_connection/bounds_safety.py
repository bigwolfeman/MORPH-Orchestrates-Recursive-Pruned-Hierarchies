"""Z3 proof: in-bounds global memory access for fused_hyper_connection kernels (sm_120).

Kernel grid: one program per (b,s) token, pid in [0, B*S). Each program loads/stores a
register block c = arange(0, BLOCK_C) masked by (c < C). The proof checks that for EVERY
valid program id and EVERY UNMASKED lane (c < C), every computed flat address falls inside
the allocated tensor — i.e. the mask (c < C) is sufficient to keep all accesses in-bounds,
for all four kernels (PRE fwd/bwd, POST fwd/bwd).

We prove UNSAT on the negation: "there exists a valid (pid, c<C, j, i) whose address is
out of [0, numel)". UNSAT => no such access => bounds-safe.

Tensors (flat element indexing):
  h        : numel = B*S*N*C          access h_base + j*C + c,  h_base = pid*(N*C)
  xbar/y   : numel = B*S*C            access pid*C + c
  hpre/hpost: numel = B*S*N           access pid*N + j  (or +i)
  hres     : numel = B*S*N*N          access pid*(N*N) + i*N + j
"""
import json, time, os
from z3 import Solver, Int, And, Or, Not, sat, unsat

N = 4
C = 768
BLOCK_C = 1 << (C - 1).bit_length()   # 1024
results = {}


def prove(name, constraints_violation, extra_vars_desc):
    s = Solver()
    s.add(constraints_violation)
    t0 = time.time()
    r = s.check()
    ms = (time.time() - t0) * 1000
    if r == unsat:
        results[name] = {"status": "proven", "property": "in_bounds", "proof_time_ms": round(ms, 2)}
    elif r == sat:
        m = s.model()
        results[name] = {"status": "violated", "counterexample": {str(d): m[d].as_long() for d in m.decls()}}
    else:
        results[name] = {"status": "unknown"}
    print(f"  [{results[name]['status'].upper():8}] {name:28} ({ms:.2f} ms)  {extra_vars_desc}")


# Use representative production sizes B=4, S=4096 (largest gate config). The proof is over
# symbolic pid/c/i/j bounded by these, so it covers every program in the launch.
for (B, S) in [(4, 4096), (2, 512)]:
    tag = f"B{B}_S{S}"
    GS = B * S

    pid = Int("pid"); c = Int("c"); j = Int("j"); i = Int("i")
    valid = And(pid >= 0, pid < GS, c >= 0, c < C, j >= 0, j < N, i >= 0, i < N)

    # --- h access: h_base + j*C + c, numel = GS*N*C ---
    h_numel = GS * N * C
    h_addr = pid * (N * C) + j * C + c
    prove(f"h_inbounds_{tag}", And(valid, Or(h_addr < 0, h_addr >= h_numel)),
          f"h numel={h_numel}")

    # --- xbar / y access: pid*C + c, numel = GS*C ---
    xy_numel = GS * C
    xy_addr = pid * C + c
    prove(f"xbar_y_inbounds_{tag}", And(valid, Or(xy_addr < 0, xy_addr >= xy_numel)),
          f"xbar/y numel={xy_numel}")

    # --- hpre / hpost access: pid*N + j, numel = GS*N ---
    vN_numel = GS * N
    vN_addr = pid * N + j
    prove(f"hpre_hpost_inbounds_{tag}", And(valid, Or(vN_addr < 0, vN_addr >= vN_numel)),
          f"hpre/hpost numel={vN_numel}")

    # --- hres access: pid*(N*N) + i*N + j, numel = GS*N*N ---
    hres_numel = GS * N * N
    hres_addr = pid * (N * N) + i * N + j
    prove(f"hres_inbounds_{tag}", And(valid, Or(hres_addr < 0, hres_addr >= hres_numel)),
          f"hres numel={hres_numel}")

    # --- round-2 PRE-MAP kernels: raw_full/graw [B,S,48] accessed pid*48 + k, k in [0,48) ---
    k48 = Int("k48")
    raw_numel = GS * 48
    raw_addr = pid * 48 + k48
    prove(f"raw48_inbounds_{tag}",
          And(pid >= 0, pid < GS, k48 >= 0, k48 < 48, Or(raw_addr < 0, raw_addr >= raw_numel)),
          f"raw_full/graw numel={raw_numel} (48 mapping floats/token)")

    # --- rms [B,S,1] accessed at pid, numel = GS ---
    rms_addr = pid
    prove(f"rms_inbounds_{tag}",
          And(pid >= 0, pid < GS, Or(rms_addr < 0, rms_addr >= GS)),
          f"rms numel={GS}")

    # --- hprecm / hpostrow [B,S,N] accessed pid*N + j (same form as hpre/hpost) — covered
    #     by hpre_hpost_inbounds above. ghpart [B,S,N,C] is the same layout as h — covered
    #     by h_inbounds above (identical address expr h_base + j*C + c). ---


print(f"\nBLOCK_C={BLOCK_C} (next_pow2 of C={C}); masked lanes (c>=C) never accessed (guarded by tl mask).")
out = os.path.join(os.path.dirname(__file__), "bounds_result.json")
json.dump(results, open(out, "w"), indent=2)
all_proven = all(v["status"] == "proven" for v in results.values())
print("ALL PROVEN" if all_proven else "SOME NOT PROVEN")
assert all_proven
