# TileProver Journal — 30B MLP compute/occupancy lever

## [2026-06-23] BASELINE — gemv_bw_30b.py (worktree perf/30b-mlp-compute @ 493716f)
**Goal:** establish per-kernel µs baseline for mortar/ternary MLP GEMVs at 30B shapes before optimizing.
**Method:** `PYTHONPATH=/tmp/morph-mlp-lever-wt /home/wolfe/.venv/bin/python ignore/gemv_bw_30b.py` (cold-L2, median of 5×120 iters).
**Result (median µs):**
- mortar_gemv gate_up [43648,8192]: 43.62µs (±0.48) — 28.6% peak BW, COMPUTE headroom
- mortar_gemv down    [8192,21824]: 39.49µs (±0.27) — 16.4% peak BW, COMPUTE headroom
- ternary_gemv gate_up [43648,8192]: 83.10µs (±0.29) — 60.1% peak
- ternary_gemv down   [8192,21824]: 75.04µs (±0.62) — 33.3% peak
- (int8 o_proj 68.05µs 55%, qkv 96.35µs 78% — NOT target)
**Key structural observation:** _mortar_gemv_kernel acc = tl.zeros((BO=32, BLK//4=32)) = 1024 fp32 regs/CTA → the register-limit driver (ncu: 61-64 regs/thread, ~4 blk/SM, 55% occ). Dequant chain `((p>>(2*st))&3).to(f32)-1.0` is the dependent-latency stall.
**Next:** Lever 1 = cheaper dequant (LUT or fold -1). Lever 2 = cut regs (BO/accumulator tiling). Measure each via gemv_bw_30b µs.

## [2026-06-23] FLOOR PROBE — decode IS ~40-45% of mortar wall-clock
**Method:** in-place patched live _mortar_gemv_kernel decode `w=((p>>2st)&3).to(f32)-1` → `w=const 1.0` (no SHR/AND/I2F), measured via real gemv_bw_30b.py.
**Result:** gate_up 43.62→23.87µs (-45%), down 39.49→23.78µs (-40%). Floor is memory-bound (52%/27% peak).
**Verdict:** the 2-bit unpack ALU chain IS the dominant compute cost (ncu execution-dependency stall confirmed in wall-clock). A cheaper CORRECT decode should recover a big chunk of the 20µs gap.
**Discarded:** closure-based ceiling harness (ignore/mlp_ceiling_probe.py) MIS-COMPILED (645µs, 15x) — do not trust it. BO/num_warps sweep = null (all <noise); lowering BO HURTS (grid+partial-sum overhead). Factored -1 (acc+=code*x - Σx) = wash-to-worse (I2F not removed).
**Next:** find cheapest CORRECT {0,1,2}->{-1,0,1} fp decode. Candidates: (a) hoist I2F: cast whole byte p once to f32, extract strides by float mul/floor; (b) bit-trick avoiding per-stride I2F; (c) reduce x-load redundancy.

## [2026-06-23] DECODE COST BREAKDOWN (in-place probes, real harness, gate_up µs)
- 43.62 base  = load p + 4×(SHR+AND+I2F) + 4 variable-FMA + (-1)
- 43.15 P2 (drop -1)        → SUB ~0.5µs (noise)
- 39.41 P3 (w=p.to(f32), no SHR/AND, CSE'd 1 I2F) → SHR/AND extraction ≈ 4µs (9%)
- 23.87 floor (w=1.0 const) → variable-FMA + I2F ≈ 15µs (35%, mostly irreducible dot-product math)
**Conclusion:** recoverable compute = SHR/AND extraction (~4µs, ~9%) + maybe -1 (~0.5µs). The 15µs FMA is the dot product itself (irreducible at this layout). Float-floor extraction was SLOWER (floors cost > SHR/AND). 
**Decision:** pursue cheapest-correct extraction (hoist/minimize SHR/AND) for ~10% mortar win; null on the rest. ~10% of the dominant 23B-param kernel is a real headline at 30B scale.

## [2026-06-23] LEVER 1 LANDED — magic-number 2-bit dequant (bit-exact)
**Method:** replaced `((p>>2st)&3).to(f32)-1` with `((p32>>2st)&3 | 0x4B400000).bitcast(f32) - 12582913.0` in _mortar_gemv_kernel, _ternary_gemv_kernel, _ternary_rs_kernel. Places 2-bit code in mantissa LSBs of 1.5*2^23; readback - bias = code-1. Removes per-element I2F (int OR + bitcast + fp sub instead).
**Parity:** ignore/mortar_decode_parity.py vs /tmp/fds_backup.py original decode → bit_exact=True maxd=0.000e+00 both gate_up & down. PARITY_PASS.
**Per-kernel µs (gemv_bw_30b.py):**
- mortar gate_up 43.62→42.51 (-2.5%) | mortar down 39.49→37.54 (-4.9%)
- ternary gate_up 83.10→83.23 (flat, 60% BW memory-bound) | ternary down 75.04→71.87 (-4.2%)
**Honest:** wins concentrated in compute-bound down-projs (2-5%); memory-bound gate_up flat. All below the ~5% boundary individually but consistent + zero-risk (bit-exact). The bigger 15µs FMA floor is the irreducible dot product.
**Lever 2 (occupancy):** NULL. BO 32→16→8 HURTS (grid+partial overhead); num_warps 8 slower; eager-reduce small-acc wash. Kernel stall not fixable by occupancy knobs at this layout.
**Next:** hard gates — 276M bench_decode non-regression, live 30B tok/s, Z3 on the magic indexing.

## [2026-06-23] HARD GATES + END-TO-END VERDICT
**Z3:** tile-prover/proofs/mortar_gemv/magic_dequant.py → MAGIC_DEQUANT_PROOF_PASS. BitVec32 proves (code|0x4B400000)-0x4B400000==code over ALL 256 bytes×4 strides; exhaustive 4-code struct bitcast → w_magic==w_orig exact. Empirical: maxd=0.000e+00.
**276M non-regression:** DECODE_BENCH_PASS match=256/256 tok/s=473.7 (base 472.3, noise). PASS.
**Per-kernel (gemv_bw_30b):** mortar gate_up 43.62→41.90 (-3.9%), mortar down 39.49→37.41 (-5.3%), ternary down 75.04→71.63 (-4.5%), ternary gate_up flat (memory-bound).
**Live 30B tok/s:** base [44.06, 45.41] vs magic [43.61, 44.92] — OVERLAPPING, ~3% noise band. NO end-to-end change: full decode is HBM-bound (51% peak, 17.97 GB/tok). Parity argmax 8/8 cos 0.99981 PASS.
**HONEST VERDICT:** Lever-1 magic dequant gives a real, Z3-bit-exact, zero-risk per-kernel compute win (mortar/ternary down −4 to −5%) but it does NOT move headline 30B tok/s because the aggregate decode is bandwidth-bound. Lever-2 occupancy = null. The per-kernel win is bankable for any future config where MLP compute becomes the e2e bottleneck (denser carve, B>1, faster attn). ncu confirmation of the SM%/occupancy delta is owed (admin-gated).
