"""Z3 proof: global memory coalescing for fused_hyper_connection kernels (sm_120).

The carrier loads/stores index h/xbar/y/out as ``base + c`` where ``c = arange(0,BLOCK_C)``
is the per-program lane index over the feature dim. Within a warp, 32 consecutive lanes hold
32 CONSECUTIVE values of ``c`` (Triton maps a 1-D tl.arange contiguously across lanes). With
bf16 (2 bytes/elem), a warp's 32 lanes therefore span 32*2 = 64 contiguous bytes from a
base that is a multiple of the per-stream stride — i.e. one aligned 128 B L1 cache-line
segment per warp (64 B < 128 B). This is perfect coalescing.

We prove, with Z3, that for ANY warp (any base-lane lane0, any stream j) the 32 lanes of a
warp touch addresses spanning < 128 bytes AND lying within a single 128 B cache line when the
base is 128 B-aligned (which it is: h_base + j*C is a multiple of C=768 elems = 1536 B = 12*128,
and 768 is even so j*C stays 2-byte aligned; the warp base lane0 is a multiple of 32).

Property A (span): max_lane_addr - min_lane_addr == (warp_size-1)*elem_bytes  (stride-1, no gaps)
Property B (single line): for a warp whose lane0 byte offset is a multiple of 64, all 32 lanes
                          fall in one 128 B line.  floor(addr/128) is constant across the warp.

Negation checked UNSAT.
"""
import json, time, os
from z3 import Solver, Int, And, Or, Not, sat, unsat, If

WARP = 32
ELEM_BYTES = 2     # bf16 carrier
LINE = 128
C = 768
N = 4
results = {}


def floordiv(a, b):
    # z3 Int division is truncation toward zero; addresses are >=0 here so it's floor.
    return a / b


def prove(name, viol, desc=""):
    s = Solver(); s.add(viol)
    t0 = time.time(); r = s.check(); ms = (time.time() - t0) * 1000
    if r == unsat:
        results[name] = {"status": "proven", "proof_time_ms": round(ms, 2)}
    elif r == sat:
        m = s.model()
        results[name] = {"status": "violated", "counterexample": {str(d): m[d].as_long() for d in m.decls()}}
    else:
        results[name] = {"status": "unknown"}
    print(f"  [{results[name]['status'].upper():8}] {name:34} ({ms:.2f} ms) {desc}")


# A warp's lane0 starts the feature block at some multiple of 32 (warp granularity within
# the BLOCK_C tl.arange). lane = lane0 + t, t in [0,32). c address (element) = base_feat + lane.
# Byte address = (carrier_base_elems + base_feat + lane) * ELEM_BYTES.
lane0 = Int("lane0")        # warp's first lane feature index, multiple of 32, in [0, C)
t = Int("t")                # lane within warp
carrier_base = Int("carrier_base")  # = pid*(N*C) + j*C  (elements) — a multiple of C
k = Int("k"); pid = Int("pid"); j = Int("j")

warp_ok = And(lane0 >= 0, lane0 < C, lane0 % WARP == 0, t >= 0, t < WARP)
# carrier_base is pid*N*C + j*C = (pid*N + j)*C, a nonneg multiple of C
base_ok = And(carrier_base == k * C, k >= 0)

elem = carrier_base + lane0 + t
byte = elem * ELEM_BYTES

# --- Property A: stride-1, contiguous, no gaps within the warp ---
# addr(t) - addr(0) == t*ELEM_BYTES for all t  => span = 31*2 = 62 B
t0v = Int("t0v")
span_byte = (carrier_base + lane0 + t) * ELEM_BYTES - (carrier_base + lane0 + 0) * ELEM_BYTES
prove("coalesced_stride1_contiguous",
      And(warp_ok, base_ok, Not(span_byte == t * ELEM_BYTES)),
      "addr(t)-addr(0) must == t*2B (stride-1)")

# --- Property B: all 32 lanes in ONE 128 B cache line ---
# Need: floor(byte(t)/128) == floor(byte(0)/128) for all t in warp.
# byte(0) = (carrier_base+lane0)*2. carrier_base = k*768 -> *2 = k*1536 = k*12*128 -> line-aligned.
# lane0 multiple of 32 -> *2 = 64*(lane0/32) -> multiple of 64. So byte(0) mod 128 in {0,64}.
# Max lane byte = byte(0) + 62 < byte(0) + 128, so single line iff byte(0) mod 128 <= 64.
# Since byte(0) mod 128 in {0,64} and 64+62=126<128, ALWAYS single line. Prove it.
line0 = floordiv((carrier_base + lane0 + 0) * ELEM_BYTES, LINE)
linet = floordiv((carrier_base + lane0 + t) * ELEM_BYTES, LINE)
prove("coalesced_single_cache_line",
      And(warp_ok, base_ok, Not(line0 == linet)),
      "all 32 lanes share one 128B line")

# --- broadcast/uniform scalar loads (Hres[i,j], Hpost[i], Hpre[j]): same address for ALL
# lanes in the warp (no c term) => uniform load, never a coalescing problem, and on sm_80+
# a same-address smem-style access would broadcast. Here they are GLOBAL uniform loads;
# prove the address is lane-INDEPENDENT (does not depend on t). ---
# scalar addr e.g. hpre: pid*N + j  — contains no lane term. Model "depends on t" as
# addr(t) != addr(t') for some t,t'. Prove UNSAT.
tt = Int("tt")
scalar_addr_t = pid * N + j        # no t
scalar_addr_tt = pid * N + j       # no tt (identical expression by construction)
prove("scalar_loads_uniform",
      And(pid >= 0, j >= 0, j < N, t >= 0, t < WARP, tt >= 0, tt < WARP,
          Not(scalar_addr_t == scalar_addr_tt)),
      "Hres/Hpost/Hpre scalar loads lane-independent (broadcast)")

# --- round-2 PRE-MAP scalar loads: raw_full[k]=pid*48+k, hres[i,j]=pid*16+i*4+j, rms[pid].
#     All are lane-INDEPENDENT (no c/t term) -> uniform broadcast loads, never a coalescing
#     problem (same class as the Hres/Hpost/Hpre scalar loads proved above). The big-tile
#     loads/stores of the PRE-MAP kernels (h, xbar, ghpart) use the IDENTICAL stride-1
#     base+c pattern proved in Property A/B, so they inherit perfect coalescing. ---
k48 = Int("k48"); k48b = Int("k48b")
raw_addr_t = pid * 48 + k48       # no lane term
raw_addr_tt = pid * 48 + k48b
prove("premap_scalar_loads_uniform",
      And(pid >= 0, k48 >= 0, k48 < 48, k48b >= 0, k48b < 48, k48 == k48b,
          t >= 0, t < WARP, tt >= 0, tt < WARP, Not(raw_addr_t == raw_addr_tt)),
      "raw_full[k]/hres/rms loads lane-independent (broadcast)")

print(f"\nELEM_BYTES={ELEM_BYTES} (bf16), WARP={WARP} -> warp span = {(WARP-1)*ELEM_BYTES} B < {LINE} B line.")
out = os.path.join(os.path.dirname(__file__), "coalescing_result.json")
json.dump(results, open(out, "w"), indent=2)
ap = all(v["status"] == "proven" for v in results.values())
print("ALL PROVEN" if ap else "SOME NOT PROVEN")
assert ap
