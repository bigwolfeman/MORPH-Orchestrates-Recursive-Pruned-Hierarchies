"""Z3 formal verification of memory access in-bounds for fused_cca_conv (sm_120).

Proves: for every Triton program / lane in the 6 kernels, every load/store
offset that is NOT masked off lies within the allocated tensor, for the full
production parameter set (Q stream C=384 G=12, K stream C=128 G=4, Cg=32, K=4,
S in {512,1024,2048,4096}, B=2, BLOCK_T=128).

Method: each kernel computes flat offsets from (b, channel/group, time-tile, j,
lane). We encode the offset arithmetic + the kernel's actual mask predicate as
Z3 BitVec/Int constraints, then ask Z3 to find an assignment where the mask is
TRUE but the offset is out of [0, numel). UNSAT == proven in-bounds.

This mirrors the kernel source exactly (see fused_cca_conv.py); if the kernel's
indexing changes, re-run this.
"""
import json
from z3 import (Solver, Int, And, Or, Not, If, sat, unsat)

# production configs: (C, G)
CONFIGS = [(384, 12), (128, 4)]
CG = 32
K = 4
P = K - 1
BLOCK_T = 128
B = 2
S_VALUES = [512, 1024, 2048, 4096]

results = {}


def check(name, build_constraints):
    """build_constraints(s) -> (offset_expr, in_bounds_pred, mask_pred, numel).
    We prove: mask => 0 <= offset < numel, i.e. UNSAT( mask AND NOT inbounds )."""
    s = Solver()
    s.set("timeout", 5000)
    offset, mask, numel = build_constraints(s)
    s.add(mask)
    s.add(Or(offset < 0, offset >= numel))
    r = s.check()
    status = "VIOLATED" if r == sat else ("proven" if r == unsat else "unknown")
    cex = None
    if r == sat:
        m = s.model()
        cex = str(m)
    results[name] = {"status": status, "counterexample": cex}
    print(f"  [{status:8}] {name}")
    return status == "proven"


all_ok = True
for (C, G) in CONFIGS:
    for S in S_VALUES:
        n_tt = (S + BLOCK_T - 1) // BLOCK_T
        tag = f"C{C}_G{G}_S{S}"

        # common symbolic indices, bounded to their kernel ranges
        def base(s, grid_groups):
            b = Int("b"); pid_tile = Int("tt"); unit = Int("unit")
            lane = Int("lane"); j = Int("j")
            s.add(b >= 0, b < B)
            s.add(pid_tile >= 0, pid_tile < n_tt)
            s.add(unit >= 0, unit < grid_groups)      # channel (C) or group (G)
            s.add(lane >= 0, lane < BLOCK_T)          # time lane within tile
            s.add(j >= 0, j < K)
            return b, pid_tile, unit, lane, j

        # ---- depthwise fwd: x load x_ptr + (b*C+c)*S + (t-p+j); mask t<S & 0<=src<S
        def dw_fwd_x(s):
            b, tt, c, lane, j = base(s, C)
            t = tt * BLOCK_T + lane
            src = t - P + j
            off = (b * C + c) * S + src
            mask = And(t < S, src >= 0, src < S)
            return off, mask, B * C * S
        all_ok &= check(f"{tag}/dw_fwd.x_load", dw_fwd_x)

        # ---- depthwise fwd: y store (b*C+c)*S + t ; mask t<S
        def dw_fwd_y(s):
            b, tt, c, lane, j = base(s, C)
            t = tt * BLOCK_T + lane
            off = (b * C + c) * S + t
            return off, t < S, B * C * S
        all_ok &= check(f"{tag}/dw_fwd.y_store", dw_fwd_y)

        # ---- grouped fwd: x load (b*C + g*Cg + ci)*S + (t-p+j); ci<Cg
        def gp_fwd_x(s):
            b, tt, g, lane, j = base(s, G)
            ci = Int("ci"); s.add(ci >= 0, ci < CG)
            t = tt * BLOCK_T + lane
            src = t - P + j
            off = (b * C + g * CG + ci) * S + src
            mask = And(t < S, src >= 0, src < S)
            return off, mask, B * C * S
        all_ok &= check(f"{tag}/gp_fwd.x_load", gp_fwd_x)

        # ---- grouped fwd: weight load (g*Cg+co)*Cg*K + ci*K + j ; co,ci<Cg
        def gp_fwd_w(s):
            b, tt, g, lane, j = base(s, G)
            co = Int("co"); ci = Int("ci")
            s.add(co >= 0, co < CG, ci >= 0, ci < CG)
            off = (g * CG + co) * (CG * K) + ci * K + j
            # weight tile is loaded UNMASKED -> must be unconditionally in-bounds
            return off, True, C * CG * K
        all_ok &= check(f"{tag}/gp_fwd.w_load(unmasked)", gp_fwd_w)

        # ---- grouped fwd: y store (b*C + g*Cg + co)*S + t ; co<Cg, mask t<S
        def gp_fwd_y(s):
            b, tt, g, lane, j = base(s, G)
            co = Int("co"); s.add(co >= 0, co < CG)
            t = tt * BLOCK_T + lane
            off = (b * C + g * CG + co) * S + t
            return off, t < S, B * C * S
        all_ok &= check(f"{tag}/gp_fwd.y_store", gp_fwd_y)

        # ---- grouped bwd dx: go load (b*C + g*Cg + cl)*S + (T+p-j); mask T<S & 0<=gsrc<S
        def gp_bwd_dx_go(s):
            b, tt, g, lane, j = base(s, G)
            cl = Int("cl"); s.add(cl >= 0, cl < CG)
            T = tt * BLOCK_T + lane
            gsrc = T + P - j
            off = (b * C + g * CG + cl) * S + gsrc
            mask = And(T < S, gsrc >= 0, gsrc < S)
            return off, mask, B * C * S
        all_ok &= check(f"{tag}/gp_bwd_dx.go_load", gp_bwd_dx_go)

        # ---- grouped bwd dw: store partial slab[b*n_tt+tt, C, Cg, K]
        #      offset = (b*n_tt+tt)*(C*Cg*K) + (g*Cg+co)*Cg*K + ci*K + j ; unmasked store
        def gp_bwd_dw_store(s):
            b, tt, g, lane, j = base(s, G)
            co = Int("co"); ci = Int("ci")
            s.add(co >= 0, co < CG, ci >= 0, ci < CG)
            slab = b * n_tt + tt
            off = slab * (C * CG * K) + (g * CG + co) * (CG * K) + ci * K + j
            numel = (B * n_tt) * C * CG * K
            return off, True, numel
        all_ok &= check(f"{tag}/gp_bwd_dw.store(unmasked)", gp_bwd_dw_store)

        # ---- depthwise bwd (FUSED dx+dw kernel _dw_bwd_kernel) ----
        # dx path: go load (b*C+c)*S + (T+p-j); mask T<S & 0<=gsrc<S
        def dw_bwd_dx_go(s):
            b, tt, c, lane, j = base(s, C)
            T = tt * BLOCK_T + lane
            gsrc = T + P - j
            off = (b * C + c) * S + gsrc
            mask = And(T < S, gsrc >= 0, gsrc < S)
            return off, mask, B * C * S
        all_ok &= check(f"{tag}/dw_bwd_dx.go_load", dw_bwd_dx_go)

        # dw path x load (b*C+c)*S + (t-p+j); mask t<S & 0<=src<S
        def dw_bwd_dw_x(s):
            b, tt, c, lane, j = base(s, C)
            t = tt * BLOCK_T + lane
            src = t - P + j
            off = (b * C + c) * S + src
            mask = And(t < S, src >= 0, src < S)
            return off, mask, B * C * S
        all_ok &= check(f"{tag}/dw_bwd_dw.x_load", dw_bwd_dw_x)

        # dx store (b*C+c)*S + t ; mask t<S
        def dw_bwd_dx_store(s):
            b, tt, c, lane, j = base(s, C)
            t = tt * BLOCK_T + lane
            off = (b * C + c) * S + t
            return off, t < S, B * C * S
        all_ok &= check(f"{tag}/dw_bwd_dx.store", dw_bwd_dx_store)

        # ---- depthwise bwd dw: store slab[b*n_tt+tt, C, K] at slab*(C*K)+c*K+j
        def dw_bwd_dw_store(s):
            b, tt, c, lane, j = base(s, C)
            slab = b * n_tt + tt
            off = slab * (C * K) + c * K + j
            numel = (B * n_tt) * C * K
            return off, True, numel
        all_ok &= check(f"{tag}/dw_bwd_dw.store(unmasked)", dw_bwd_dw_store)

print()
summary = {"all_proven": all_ok, "n_checks": len(results),
           "configs": CONFIGS, "S": S_VALUES, "CG": CG, "K": K, "BLOCK_T": BLOCK_T,
           "results": results}
with open("tile-prover/proofs/fused_cca_conv/result.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"{len(results)} checks; all_proven={all_ok}")
assert all_ok, "Z3 found an out-of-bounds access — see VIOLATED rows"
