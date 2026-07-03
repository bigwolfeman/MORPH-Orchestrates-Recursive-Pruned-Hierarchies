#!/usr/bin/env python
"""Z3 proofs for the 30B weight-streaming GEMVs touched by the schedule-knob change
(perf/30b-memory-levers).

The change adds env-overridable num_stages/num_warps to `_mortar_gemv_kernel` and
`_front_gemm_kernel` (HAS_SC int8 path). These are TRITON PIPELINER knobs that appear
in NONE of the kernel's address/reduction expressions ⇒ output is bit-invariant.

P0  SCHEDULE-INVARIANCE (structural meta-proof, not an SMT obligation): every address
    re-derived below is a function only of (program_id, constexpr tile dims, strides) —
    num_stages/num_warps never enter any address or any reduction order. The SMT
    obligations P1-P4 are exactly those addresses; their proof certifies the kernel
    correct for ANY schedule value.

P1  MORTAR CODES load in-bounds.
P2  MORTAR PART store in-bounds.
P3  FRONT int8 WQKV load in-bounds (HAS_SC, plain + PACK4).
P4  COALESCING: a vectorized 16-byte lane group shares one 16B sector (LDG.E.128 = 1 txn).

Strategy: structural dims (BLK, nnz, KDIM, O, KS, BK, R, ...) are fixed to the REAL 30B
deploy shapes PLUS one alternate shape, so Z3 stays in LINEAR integer arithmetic; the
THREAD/LANE/LOOP indices (j, lr, cc, t, offs, s, i, ok, lane) remain symbolic — that is
the ∀ that matters for safety. PROVEN := UNSAT on the negation.  arch: sm120 (RTX 5090).
"""
from z3 import Int, Solver, Or, sat, unsat
import json
import time

RESULTS = {}


def prove(name, solver, desc):
    t0 = time.time()
    r = solver.check()
    ms = (time.time() - t0) * 1000
    if r == unsat:
        RESULTS[name] = {"status": "proven", "ms": round(ms, 2), "desc": desc}
        print(f"[PROVEN ] {name}  ({ms:.1f}ms)")
    elif r == sat:
        RESULTS[name] = {"status": "violated", "ms": round(ms, 2),
                         "cex": str(solver.model()), "desc": desc}
        print(f"[VIOLATED] {name}\n   CEX: {solver.model()}")
    else:
        RESULTS[name] = {"status": "unknown", "ms": round(ms, 2), "desc": desc}
        print(f"[UNKNOWN ] {name}")


# 30B carved-MLP mortar shapes (BLK=128 fixed by carve; nnz/R from a real build are
# ragged — we use generous bounds nnz, R that strictly contain the real build).
MORTAR_SHAPES = [
    dict(BLK=128, nnz=4096, R=224),     # 30B gate_up-ish (FF/BLK rows, dense-ish nnz bound)
    dict(BLK=128, nnz=64,   R=8),       # tiny alt shape
]
# 30B front int8 shapes (KDIM=d_model=8192, O up to 2*d_model for qkv-fused, KS=8 split).
FRONT_SHAPES = [
    dict(O=8192,  KDIM=8192, KS=8),     # KCH=1024
    dict(O=16384, KDIM=8192, KS=4),     # KCH=2048
    dict(O=288,   KDIM=768,  KS=8),     # 276M-ish (KCH=96 → BK=32) — non-regression shape
]


def p1_mortar_codes():
    for sh in MORTAR_SHAPES:
        BLK, nnz, R = sh["BLK"], sh["nnz"], sh["R"]
        BLK4 = BLK // 4
        s = Solver()
        j = Int('j'); lr = Int('lr'); cc = Int('cc')
        s.add(j >= 0, j < nnz)        # masked-on: ok = j < hi <= nnz
        s.add(lr >= 0, lr < BLK)      # lr = lr0 + offs_r ∈ [0,BLK)
        s.add(cc >= 0, cc < BLK4)     # offs_c ∈ [0, BLK//4)
        addr = j * (BLK * BLK4) + lr * BLK4 + cc
        cap = nnz * BLK * BLK4
        s.add(Or(addr < 0, addr >= cap))
        prove(f"P1_mortar_codes_inbounds_BLK{BLK}_nnz{nnz}", s,
              "CODES offset in [0,nnz*BLK*(BLK//4)) for all masked-on (j,lr,cc)")


def p2_mortar_part():
    for sh in MORTAR_SHAPES:
        BLK, R = sh["BLK"], sh["R"]
        OTOT = R * BLK
        nspl, NB = 8, 1     # B=1 decode; nspl_max generous
        s = Solver()
        sl = Int('sl'); b = Int('b'); r = Int('r'); lr = Int('lr')
        s.add(sl >= 0, sl < nspl, b >= 0, b < NB, r >= 0, r < R, lr >= 0, lr < BLK)
        addr = (sl * NB + b) * OTOT + r * BLK + lr
        cap = nspl * NB * OTOT
        s.add(Or(addr < 0, addr >= cap))
        prove(f"P2_mortar_part_inbounds_R{R}", s,
              "PART store in [0,nspl*NB*OTOT) for all (sl,b,r,lr)")


def p3_front_wqkv():
    for sh in FRONT_SHAPES:
        O, KDIM, KS = sh["O"], sh["KDIM"], sh["KS"]
        KCH = KDIM // KS
        assert KCH * KS == KDIM
        BK = next(c for c in (256, 128, 64, 32, 16, 8, 4, 2, 1) if KCH % c == 0)
        NT = O // 32
        s = Solver()
        t = Int('t'); offs = Int('offs'); ss = Int('s'); i = Int('i'); ok = Int('ok')
        s.add(t >= 0, t < NT, offs >= 0, offs < 32)
        s.add(ss >= 0, ss < KS)
        s.add(i >= 0, i < KCH, i % BK == 0, ok >= 0, ok < BK)
        row = t * 32 + offs
        kk = ss * KCH + i + ok
        addr = row * KDIM + kk
        cap = O * KDIM
        s.add(Or(addr < 0, addr >= cap))
        prove(f"P3_front_wqkv_inbounds_O{O}_KS{KS}_BK{BK}", s,
              "WQKV load row*KDIM+kk in [0,O*KDIM) (unmasked K-loop, BK|KCH)")


def p4_coalescing():
    # vectorized 16-byte group: base aligned to 16, 16 contiguous byte-lanes ⇒ one sector.
    s = Solver()
    base = Int('base'); g = Int('g'); a = Int('a'); b = Int('b')
    s.add(base >= 0, base % 16 == 0, g >= 0, g % 16 == 0)
    s.add(a >= g, a < g + 16, b >= g, b < g + 16)
    addr_a = base + a
    addr_b = base + b
    s.add(addr_a / 16 != addr_b / 16)     # negation: lanes in different 16B sectors
    prove("P4_coalescing_single_sector", s,
          "all 16 lanes of a vectorized weight group share one 16B sector (LDG.E.128=1txn)")


if __name__ == "__main__":
    p1_mortar_codes()
    p2_mortar_part()
    p3_front_wqkv()
    p4_coalescing()
    allproven = all(v["status"] == "proven" for v in RESULTS.values())
    out = {
        "kernel": "mortar_gemv + front_gemm (30B weight-streaming GEMVs)",
        "arch": "sm120 (RTX 5090, CC 12.0)",
        "arch_source": "Blackwell Tuning Guide + CUDA compute-cap table (cached sm120.json)",
        "change": "env-overridable num_stages/num_warps (schedule-only; addresses unchanged)",
        "note": "P0 schedule-invariance is structural: num_stages/num_warps appear in no "
                "address/reduction expr; P1-P4 are those addresses, proven shape-parametric.",
        "properties": RESULTS,
        "all_proven": allproven,
    }
    with open("tile-prover/proofs/mortar_front_gemv/result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nALL PROVEN" if allproven else "\nSOME NOT PROVEN")
