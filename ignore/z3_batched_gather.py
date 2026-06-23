#!/usr/bin/env python
"""z3_batched_gather.py — formal proof of the BATCHED routed-gather kernel changes
for sm_120 (RTX 5090).

Proves three properties of the B>1 modifications to:
  • _route_flags_kernel  (grid (B,); per-stream GATES/RACT/CACT batch offsets)
  • _mortar_gemv_kernel  (ROWACT + b*ra_b + r ; COLACT + b*ca_b + cb)

PROPERTIES
  P1 IN-BOUNDS    : every batched flag READ (route_flags GATES, mortar ROWACT/COLACT)
                    and every flag WRITE (route_flags RACT/CACT) lands inside its
                    allocated [B, *] buffer for all b in [0,B), all lane in [0,NR|NC).
  P2 INDEPENDENCE : item b's flag accesses touch ONLY rows of GATES/RACT/CACT owned by
                    b. Two distinct batch items b1 != b2 share NO flag address (no
                    cross-stream contamination) — formalizes "item b reads only item b".
  P3 COALESCING   : within route_flags, the NR (or NC) lanes of one program write
                    contiguous int32 addresses RACT[b*NR + i] for i in [0,NR) → a unit
                    stride run (coalesced within the warp tile).

Scratch model (matches kv_cache_static.py): RACT/CACT are [B, SCR] (SCR=512), sliced
[:, :NR]/[:, :NC] so the PER-ITEM STRIDE is SCR (=stride(0)), NOT NR/NC. The kernel
strides ROWACT/COLACT by ra_b=stride(0)=SCR. We prove with the real SCR stride.

GATES is [B, NCLS] CONTIGUOUS → per-item stride NCLS. RLO/RHI/CLO/CHI are cluster
indices in [0, NCLS), shared across batch (topology frozen); the gather GATES[b*NCLS +
cluster] must stay in row b.
"""
from z3 import (Int, Solver, ForAll, Implies, And, Or, Not, sat, unsat, Distinct)

# ── symbolic hardware / shape parameters (kept symbolic so the proof is general) ──
B    = Int("B")      # batch size
SCR  = Int("SCR")    # flag scratch width (512 in kv_cache_static.py)
NR   = Int("NR")     # active gate_up output block-rows  = 2*(d_ff/128)
NC   = Int("NC")     # active down input block-cols      = d_ff/128
NCLS = Int("NCLS")   # n_clusters (router output width)

base = [B >= 1, SCR >= 1, NCLS >= 1,
        NR >= 1, NC >= 1,
        NR <= SCR, NC <= SCR,            # NR,NC fit the [B,SCR] scratch (kv_cache slices [:, :NR])
        NCLS <= SCR]                     # (router gates fit too — generous bound)

# per-item strides as the code computes them
RA_B = SCR    # ROWACT stride(0) for [B, SCR][:, :NR]
CA_B = SCR    # COLACT stride(0) for [B, SCR][:, :NC]
GA_B = NCLS   # GATES  stride(0) for [B, NCLS] contiguous

ok = True


def check(name, solver, claim):
    """claim must be VALID (negation unsat)."""
    global ok
    solver.push()
    solver.add(Not(claim))
    r = solver.verify() if hasattr(solver, "verify") else solver.check()
    solver.pop()
    valid = (r == unsat)
    print(f"  [{'PASS' if valid else 'FAIL'}] {name}")
    ok = ok and valid


# ════════════════════════════════════════════════════════════════════════════════
# P1  IN-BOUNDS  — all batched flag reads/writes inside their buffers.
# ════════════════════════════════════════════════════════════════════════════════
print("P1 in-bounds:")
b   = Int("b")     # batch program id   in [0, B)
i   = Int("i")     # flag lane          in [0, NR) or [0, NC)
cl  = Int("cl")    # cluster index      in [0, NCLS)  (RLO/RHI/CLO/CHI value)
cb  = Int("cb")    # mortar col-block   in [0, NC)    (COLIDX value, ≤ down block-cols)
r   = Int("r")     # mortar row-block   in [0, NR)

# route_flags: RACT write  RACT[b*RA_B + i],  i<NR  →  index in [0, B*SCR)
s = Solver(); s.add(base + [0 <= b, b < B, 0 <= i, i < NR])
check("route_flags RACT write in [0, B*SCR)",
      s, And(b * RA_B + i >= 0, b * RA_B + i < B * SCR))
# CACT write  CACT[b*CA_B + i], i<NC
s = Solver(); s.add(base + [0 <= b, b < B, 0 <= i, i < NC])
check("route_flags CACT write in [0, B*SCR)",
      s, And(b * CA_B + i >= 0, b * CA_B + i < B * SCR))
# route_flags GATES gather  GATES[b*GA_B + cl],  cl<NCLS  →  in [0, B*NCLS)
s = Solver(); s.add(base + [0 <= b, b < B, 0 <= cl, cl < NCLS])
check("route_flags GATES read in [0, B*NCLS)",
      s, And(b * GA_B + cl >= 0, b * GA_B + cl < B * NCLS))
# mortar ROWACT read  ROWACT[b*ra_b + r], r<NR  → in [0, B*SCR)
s = Solver(); s.add(base + [0 <= b, b < B, 0 <= r, r < NR])
check("mortar ROWACT read in [0, B*SCR)",
      s, And(b * RA_B + r >= 0, b * RA_B + r < B * SCR))
# mortar COLACT read  COLACT[b*ca_b + cb], cb<NC → in [0, B*SCR)
s = Solver(); s.add(base + [0 <= b, b < B, 0 <= cb, cb < NC])
check("mortar COLACT read in [0, B*SCR)",
      s, And(b * CA_B + cb >= 0, b * CA_B + cb < B * SCR))

# ════════════════════════════════════════════════════════════════════════════════
# P2  PER-ITEM INDEPENDENCE — distinct items never share a flag address.
#     (b1 != b2)  ⇒  addr(b1) != addr(b2)  for any in-range lanes.
# ════════════════════════════════════════════════════════════════════════════════
print("P2 per-item independence (no cross-stream address sharing):")
b1 = Int("b1"); b2 = Int("b2"); i1 = Int("i1"); i2 = Int("i2")

# RACT: b1*SCR+i1 == b2*SCR+i2 with i<NR<=SCR forces b1==b2.
s = Solver(); s.add(base + [0 <= b1, b1 < B, 0 <= b2, b2 < B,
                            0 <= i1, i1 < NR, 0 <= i2, i2 < NR, b1 != b2])
check("RACT addresses of b1!=b2 are disjoint",
      s, b1 * RA_B + i1 != b2 * RA_B + i2)
# CACT
s = Solver(); s.add(base + [0 <= b1, b1 < B, 0 <= b2, b2 < B,
                            0 <= i1, i1 < NC, 0 <= i2, i2 < NC, b1 != b2])
check("CACT addresses of b1!=b2 are disjoint",
      s, b1 * CA_B + i1 != b2 * CA_B + i2)
# GATES: stride NCLS, cluster < NCLS ⇒ rows disjoint.
c1 = Int("c1"); c2 = Int("c2")
s = Solver(); s.add(base + [0 <= b1, b1 < B, 0 <= b2, b2 < B,
                            0 <= c1, c1 < NCLS, 0 <= c2, c2 < NCLS, b1 != b2])
check("GATES read addresses of b1!=b2 are disjoint",
      s, b1 * GA_B + c1 != b2 * GA_B + c2)

# Cross-property (THE bug this proof guards): the flag a mortar CTA(b) READS must be
# the flag route_flags(b) WROTE. This holds IFF both use the SAME per-item stride.
# The original bug used route_flags stride = NR while mortar read stride = SCR; for
# NR != SCR and b>=1 the addresses DIVERGED (b=0 coincidentally agreed → B=1 passed,
# B>2 failed). The fix passes ract.stride(0) (=SCR) to BOTH. Model both strides as
# free vars and prove agreement is REQUIRED, then that the fix (equal strides) gives it.
wr_b = Int("wr_b")   # route_flags RACT write per-item stride
rd_b = Int("rd_b")   # mortar ROWACT read per-item stride
# (a) if the strides differ, write/read addresses CANNOT agree for any b>=1 (the
#     original bug): prove the agreement constraint is UNSAT under NR!=SCR, b>=1.
s = Solver(); s.add(base + [0 <= b, b < B, 0 <= r, r < NR,
                            wr_b == NR, rd_b == SCR, NR != SCR, b >= 1,
                            b * wr_b + r == b * rd_b + r])   # "they agree"
buggy_agree = s.check()                                     # expect UNSAT (cannot agree)
print(f"  [{'PASS' if buggy_agree == unsat else 'FAIL'}] "
      "mismatched strides (NR vs SCR) CANNOT agree for b>=1 (original bug is real, "
      "diverges)")
ok = ok and (buggy_agree == unsat)
# (b) the FIX: equal strides ⇒ write addr == read addr for ALL b, r (no divergence).
s = Solver(); s.add(base + [0 <= b, b < B, 0 <= r, r < NR, wr_b == rd_b])
check("FIX: equal write/read stride ⇒ mortar reads exactly route_flags' write (all b)",
      s, b * wr_b + r == b * rd_b + r)

# ════════════════════════════════════════════════════════════════════════════════
# P3  COALESCING — route_flags writes RACT[b*SCR + i] for i in [0,NR): consecutive
#     lanes → consecutive int32 addresses (unit stride within the program's tile).
# ════════════════════════════════════════════════════════════════════════════════
print("P3 coalescing (unit-stride flag writes per program):")
ia = Int("ia")
s = Solver(); s.add(base + [0 <= b, b < B, 0 <= ia, ia + 1 < NR])
check("RACT[b, i+1] - RACT[b, i] == 1 (contiguous)",
      s, (b * RA_B + (ia + 1)) - (b * RA_B + ia) == 1)
s = Solver(); s.add(base + [0 <= b, b < B, 0 <= ia, ia + 1 < NC])
check("CACT[b, i+1] - CACT[b, i] == 1 (contiguous)",
      s, (b * CA_B + (ia + 1)) - (b * CA_B + ia) == 1)

# ════════════════════════════════════════════════════════════════════════════════
# B==1 EQUIVALENCE — with B=1 the batched addressing reduces to the old flat layout
# (b=0 ⇒ offset 0 ⇒ identical addresses to the pre-change RACT[i]/COLACT[cb]).
# ════════════════════════════════════════════════════════════════════════════════
print("B==1 reduces to the original flat addressing:")
s = Solver(); s.add(base + [B == 1, b == 0, 0 <= i, i < NR])
check("B=1: RACT[0*SCR + i] == RACT[i] (original flat index)",
      s, 0 * RA_B + i == i)
s = Solver(); s.add(base + [B == 1, b == 0, 0 <= cb, cb < NC])
check("B=1: COLACT[0*SCR + cb] == COLACT[cb] (original flat index)",
      s, 0 * CA_B + cb == cb)

print()
if ok:
    print("Z3_BATCHED_GATHER_PROVED  (P1 in-bounds, P2 independence, P3 coalescing, "
          "B=1 equivalence — all VALID)")
else:
    print("Z3_BATCHED_GATHER_FAILED")
    raise SystemExit(1)
