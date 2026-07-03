#!/usr/bin/env python
"""Compile the weight-streaming GEMV kernels and dump SASS to inspect whether the
packed-code / int8 weight loads are already 128-bit (LDG.E.128 / vectorized).

No-theater: this is the EVIDENCE for Lever 1. If the weight load already emits
LDG.E.128 we do NOT vectorize (no win available); if it emits LDG.E.U8 / LDG.E.32
there is bandwidth left on the table.

Run: PYTHONPATH=$PWD /home/wolfe/.venv/bin/python ignore/inspect_gemv_sass.py
"""
from __future__ import annotations
import re
import subprocess
import tempfile
import os

import torch
import triton

from morph.kernels.triton.fused_decode_step import (
    _ternary_gemv_kernel, _mortar_gemv_kernel, _front_gemm_kernel,
)

CUOBJDUMP = "/opt/cuda/bin/cuobjdump"


def dump_sass(compiled, tag):
    """Write the cubin and disassemble to SASS, return the text."""
    cubin = compiled.asm.get("cubin")
    if cubin is None:
        print(f"[{tag}] no cubin in asm keys: {list(compiled.asm.keys())}")
        return ""
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin)
        path = f.name
    try:
        out = subprocess.run([CUOBJDUMP, "-sass", path],
                             capture_output=True, text=True).stdout
    finally:
        os.unlink(path)
    return out


def classify_loads(sass):
    """Count LDG by width.  Returns dict of width->count and the matching lines."""
    counts = {}
    lines = []
    for ln in sass.splitlines():
        m = re.search(r'\bLDG\.E(?:\.(U8|S8|U16|S16|32|64|128|))?\b', ln)
        if m:
            w = m.group(1) or "32"   # plain LDG.E is 32-bit
            counts[w] = counts.get(w, 0) + 1
            lines.append(ln.strip())
    return counts, lines


def compile_ternary():
    # 30B backbone MLP-ish shape; X_MODE=0 plain.  I, O constexpr.
    I, O = 4096, 4096
    I4 = I // 4
    BO = 8
    BI4 = min(triton.next_power_of_2(I4), 512)
    x = torch.randn(1, I, device="cuda", dtype=torch.float32)
    codes = torch.zeros(O, I4, device="cuda", dtype=torch.uint8)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    out = torch.empty(1, 1, O, device="cuda", dtype=torch.float32)
    c = _ternary_gemv_kernel[(triton.cdiv(O, BO), 1)](
        x, codes, scale, x, out, I=I, O=O, BI4=BI4, BO=BO,
        BC=triton.next_power_of_2(I), X_MODE=0, eps=1e-6,
        num_stages=1, num_warps=4)
    return c, f"ternary I={I} O={O} BO={BO} BI4={BI4}"


def compile_mortar():
    # 30B carved MLP: BLK=128, gate_up shape O=FF, ragged BCSR. Use a small synthetic
    # work-list with the real constexprs (CB blocks/CTA, BO rows/CTA).
    BLK = 128
    BO = 32
    CB = 8
    FF = 28672  # 30B ffn-ish; only constexprs FF/OTOT matter for the load width
    OTOT = FF
    NB = 1
    R = 4
    nnz = 16
    X = torch.randn(1, 2 * FF, device="cuda", dtype=torch.float32)
    CODES = torch.zeros(nnz, BLK * (BLK // 4), device="cuda", dtype=torch.uint8)
    COLIDX = torch.zeros(nnz, device="cuda", dtype=torch.int32)
    OFFS = torch.zeros(R + 1, device="cuda", dtype=torch.int32)
    ROWS = torch.zeros(R * (BLK // BO), device="cuda", dtype=torch.int32)
    J0S = torch.zeros(R * (BLK // BO), device="cuda", dtype=torch.int32)
    SLOTS = torch.zeros(R * (BLK // BO), device="cuda", dtype=torch.int32)
    PART = torch.zeros(NB * OTOT, device="cuda", dtype=torch.float32)
    grid = (R * (BLK // BO), 1)
    c = _mortar_gemv_kernel[grid](
        X, CODES, COLIDX, OFFS, ROWS, J0S, SLOTS, PART, X, X,
        BLK=BLK, BO=BO, CB=CB, OTOT=OTOT, NB=NB,
        SWIGLU=False, FF=FF, HAS_RACT=False, HAS_CACT=False,
        sx_b=2 * FF, ra_b=0, ca_b=0, num_warps=4, num_stages=2)
    return c, f"mortar BLK={BLK} BO={BO} CB={CB} innerload=BLK//4={BLK//4}B"


def compile_front():
    # int8 front (i8row schedule): HAS_SC=True path, PACK4=False. K-split int8 GEMM.
    LQK = 4096; VH = 4096; O = 8192; OP = 8192
    KDIM = 8192; KS = 4; BK = 256
    NT = O // 32; NU = NT * KS
    X = torch.randn(1, 128 * KDIM, device="cuda", dtype=torch.float32)
    XOFF = torch.zeros(7, device="cuda", dtype=torch.int32)
    WQKV = torch.zeros(O, KDIM, device="cuda", dtype=torch.int8)
    WSC = torch.ones(O, device="cuda", dtype=torch.float32)
    PART = torch.zeros(KS * 7 * OP, device="cuda", dtype=torch.float32)
    grid = (NU,)
    c = _front_gemm_kernel[grid](
        X, XOFF, WQKV, WSC, PART,
        LQK=LQK, VH=VH, O=O, OP=OP, KDIM=KDIM, KS=KS, BK=BK,
        NT=NT, NU=NU, HAS_SC=True, PACK4=False,
        sx_b=128 * KDIM, sx_r=KDIM, num_warps=4, num_stages=2)
    return c, f"front_i8 KDIM={KDIM} BK={BK} (int8 weight LDG)"


def main():
    torch.cuda.init()
    for compile_fn in (compile_ternary, compile_mortar, compile_front):
        compiled, tag = compile_fn()
        sass = dump_sass(compiled, tag)
        counts, lines = classify_loads(sass)
        print(f"\n===== {tag} =====")
        print("LDG width histogram:", counts)
        # show a sample of the widest and the U8 loads
        for w in ("128", "64", "U8", "32"):
            sub = [l for l in lines if f".{w}" in l or (w == "32" and re.search(r'LDG\.E\s', l))]
            if sub:
                print(f"  --- {w}-bit sample (n={counts.get(w,0)}) ---")
                for l in sub[:4]:
                    print("   ", l)
        # save full sass for manual grep
        with open(f"ignore/sass_{tag.split()[0]}.txt", "w") as fh:
            fh.write(sass)
        print(f"  full SASS -> ignore/sass_{tag.split()[0]}.txt")


if __name__ == "__main__":
    main()
