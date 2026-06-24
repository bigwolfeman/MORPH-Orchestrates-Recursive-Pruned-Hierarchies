#!/usr/bin/env python
"""Z3 proof: the magic-number 2-bit dequant is bit-exact equivalent to the original
`((p>>2st)&3).to(f32)-1` decode for ALL packed bytes and ALL four strides, and the
extracted code is always in {0,1,2,3} (so |0x4B400000 never overflows the mantissa).

Decode under proof (per stride st, per byte p in [0,255]):
  ORIG:  code = (p >> 2st) & 3   ;  w_orig = float(code) - 1     ∈ {-1,0,1,2}
  MAGIC: bits = code | 0x4B400000 ;  w_magic = bitcast<f32>(bits) - 12582913.0

We prove two things over the 8-bit domain with Z3 BitVec/FP:
 1. code = (p>>2st)&3 ∈ [0,3]  AND  (code | 0x4B400000) has the code ONLY in the
    low 2 mantissa bits (no collision with the fixed exponent/mantissa pattern).
 2. The float read-back (1.5*2^23 + code) - 12582913.0 == code - 1 exactly (the
    integers 12582912..12582915 are all exactly representable in fp32, so the
    subtraction is exact) → w_magic == w_orig for every code.

Z3 models fp32 via Real arithmetic on the exactly-representable integers (the whole
construction stays within [12582912, 12582915] ⊂ exactly-representable f32 ints, so
Real == fp32 here — no rounding). The bit-extraction is proven in BitVec(8/32).
"""
from z3 import (BitVec, BitVecVal, Extract, ZeroExt, LShR, And, Or, Not,
                Solver, ForAll, sat, unsat, Int, IntVal, Function, IntSort, RealSort)

MAGIC = 0x4B400000          # 1.5 * 2^23 = 12582912.0 as fp32 bits
BIAS = 12582913.0           # = 12582912.0 + 1.0


def prove_extract_in_range():
    """For every byte p and stride st∈{0,1,2,3}: (p>>2st)&3 ∈ [0,3] and the OR with
    MAGIC sets exactly the code in bits [0:2], leaving 0x4B400000's pattern intact
    (bits [0:2] of MAGIC are 0, so no collision)."""
    results = {}
    for st in range(4):
        s = Solver()
        p = BitVec('p', 8)
        code = (LShR(p, 2 * st)) & BitVecVal(3, 8)          # (p>>2st)&3, 8-bit
        # range: code <= 3 always (mask &3). Negate → search code > 3.
        s.add(ZeroExt(24, code) > BitVecVal(3, 32))
        r1 = s.check()
        # collision check: MAGIC low 2 bits must be 0 so (code|MAGIC) low2 == code
        magic_low2 = BitVecVal(MAGIC & 0x3, 32)
        s2 = Solver()
        s2.add(magic_low2 != BitVecVal(0, 32))
        r2 = s2.check()
        results[st] = {
            'code_le_3': 'proven' if r1 == unsat else 'VIOLATED',
            'no_low2_collision': 'proven' if r2 == unsat else 'VIOLATED',
        }
    return results


def prove_value_equivalence():
    """The float read-back equals code-1 for code∈{0,1,2,3}. The kernel does an
    actual int32→f32 BITCAST of (code | 0x4B400000), NOT a numeric conversion. We
    model that bitcast exactly with struct (the same IEEE-754 reinterpretation the
    GPU does), then in fp32 compute readback - 12582913.0 and compare to code-1.
    The readback float is 12582912.0+code (ULP=1 at this magnitude → exact), and
    12582912..12582915 are all < 2^24 so the subtraction is exact."""
    import struct

    def bitcast_i32_to_f32(i):
        return struct.unpack('<f', struct.pack('<I', i & 0xFFFFFFFF))[0]

    out = {}
    for code in (0, 1, 2, 3):
        bits = code | MAGIC                       # int32 OR (low 2 bits of MAGIC are 0)
        readback_f = bitcast_i32_to_f32(bits)     # ACTUAL bitcast, = 12582912.0+code
        # fp32 subtraction modeled via numpy float32 to be faithful to the kernel
        import numpy as np
        w_magic = float(np.float32(readback_f) - np.float32(BIAS))
        w_orig = float(np.float32(np.float32(code) - np.float32(1.0)))
        exact_repr = (0.0 <= readback_f < float(1 << 24))
        out[code] = {
            'or_bits_hex': hex(bits),
            'readback_f': readback_f,
            'exactly_representable_f32': exact_repr,
            'w_magic': w_magic, 'w_orig': w_orig,
            'equal': (w_magic == w_orig) and exact_repr,
        }
    return out


def prove_or_preserves_code_z3():
    """Z3 (BitVec32): for EVERY byte p∈[0,255] and stride st, the extracted code
    c=(p>>2st)&3 satisfies (c | MAGIC) - MAGIC == c — i.e. the OR injects the code
    into exactly the low 2 bits with no carry/collision (MAGIC's low 2 bits are 0,
    and c<4 so it never touches bit 2+). This is the bit-level guarantee that the
    fp readback value is MAGIC_float + c. Proven over the full 8-bit input domain."""
    res = {}
    for st in range(4):
        s = Solver()
        p = BitVec('p', 8)
        c = ZeroExt(24, (LShR(p, 2 * st)) & BitVecVal(3, 8))   # code in int32, ∈[0,3]
        ored = c | BitVecVal(MAGIC, 32)
        # negate the claim (ored - MAGIC == c) over all p
        s.add(Not(ored - BitVecVal(MAGIC, 32) == c))
        res[st] = 'proven' if s.check() == unsat else 'VIOLATED'
    return res


if __name__ == "__main__":
    import json
    rng = prove_extract_in_range()
    val = prove_value_equivalence()
    orz = prove_or_preserves_code_z3()
    all_ok = (all(v['code_le_3'] == 'proven' and v['no_low2_collision'] == 'proven'
                  for v in rng.values())
              and all(v['equal'] for v in val.values())
              and all(v == 'proven' for v in orz.values()))
    res = {
        'property': 'magic_dequant_bit_exact_equiv',
        'extract_in_range_z3': rng,
        'value_equivalence_exhaustive': val,
        'or_preserves_code_z3': orz,
        'overall': 'PROVEN' if all_ok else 'VIOLATED',
        'note': ('code|0x4B400000 → fp32 readback (1.5*2^23+code); minus 12582913.0 '
                 '= code-1 ∈ {-1,0,1} for code∈{0,1,2}; arch-independent (IEEE-754 '
                 'fp32, exact integers < 2^24).'),
    }
    print(json.dumps(res, indent=2))
    print("MAGIC_DEQUANT_PROOF_" + ("PASS" if all_ok else "FAIL"))
