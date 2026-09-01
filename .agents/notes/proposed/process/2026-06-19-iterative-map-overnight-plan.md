# Agent Note: Iterative Map Overnight Plan

Status: proposed

Origin: Ai-notes/06-19-2026/MORPH-Iterative-Map-Dynamics/OVERNIGHT-PLAN.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# Overnight autonomous plan — β1=0 AdEMAMix stability via the nested-dynamical-system frame (2026-06-19)

Sequential GPU runs (one at a time, UPS), CPU builds overlapped with GPU runs, each run digested by a
subagent into a tight report. Frame: see MENTAL-MODEL.md. Goal: β1=0 (memory) + α·m_slow (gain) + STABLE,
≤ AdamW8bit bar (~31–32 ppl). Resume-test window: step_18800, frozen density 0.45, detonates ~20000.

## E1 — ✅ DONE (P1, governor off, detonated step 19600): RESOLVED = POS-b
Result (subagent-digested, numbers verified): **POS-b CONFIRMED, POS-a FALSIFIED.**
- Healthy steady-state profile is a **HUMP** (peak ~1.06@iter3, settle ~1.0), NOT the decaying [1.37→1.12]
 (that was compile-warmup). Healthy training already runs at σ_max≈1.
- Blowup = **single loop iteration spiking 10²–10³× at a VARYING index (i3–i7)**, neighbors ~1.0; does NOT
 compound up the loop index. Max finite pre-detonation gain 4250.
- The real disease axis is **FREQUENCY over training**: spike rate 2.5%→7%→89.5% over steps 19000→19600,
 **~600-step lead** (spikes present while ppl healthy 27.5). Direction flat. → cap WORST-CASE σ_max(J_core)
 of ONE core application via a MAP/weight constraint with σ_target>1 (preserve the hump). E3 below, refined.

## E2 — σ_max(J_core) + ρ(J_core) probe (build during E1; gate; run capturing detonation)
JᵀJ power iteration: FD-JVP (validated) for Jv + autograd-VJP for Jᵀ. Gate must recover svdvals σ_max.
Spectral NORM (transient), not radius (asymptotic) — gate this morning caught the distinction.
- POS σ_max→>1 leading the spike → target confirmed, fix = bound σ_max. ρ<1 vs >1 = non-normal vs eigenvalue.
- NEG σ_max <1 through the spike → one-step expansion NOT the disease → weight-space drift → E4, major rethink.

## ⭐ KEY REALIZATION (2026-06-19, supersedes ordering): test τ≤1 FIRST — it's config-only
M3's realized max_gain (1e4) is a LOWER BOUND on σ_max → **σ_max≥1e4 at the blowup is ALREADY proven**;
the σ_max probe (E2) is no longer needed to establish "σ_max>1" (only to measure ρ, which doesn't change
the fix). And `core_gain_clip` IS a direction-preserving realized-gain contractivity knob — but prior runs
used τ=2.0 and τ=1.3, **BOTH >1 (still expansive!; 1.3^6≈4.8× compounds)**. The frame prescribes τ≤1
(non-expansion). **τ≤1 was NEVER tested and is config-only.** → E3a below leads.
σ_max estimator built+gated: `ignore/verify_sigma_probe.py` → VERIFY_SIGMA_PROBE_PASS (incl. non-normal
ρ=0.5<1, σ_max=6.04>1 recovered exactly). Ready for E2 if E3a is ambiguous. E2 wiring NOT yet built.

## E3a — ✅ DONE = SCRAMBLE (NOT cure). core_gain_clip=1.0 detonated step 19400 DESPITE max_gain≡1.000,
recovered to scrambled ppl~48-60 (≈ τ=2). **Realized-magnitude-clamp family DEFINITIVELY killed as a cure.**
E3a digest: direction metrics (cos/frac_rot) do NOT localize the spike either — they drift continuously through
it. So BOTH forward-side signals (magnitude, direction) are exhausted → go operator/weight-side.

## E2 — ✅ DONE = THE DECISIVE RESULT: σ_max(J_core) IS THE ORDER PARAMETER
`ignore/E2_sigma_on_ckpt.py` (validated estimator + bit-exact core-step parity via new `_apply_core_step`
method). Controlled (same seeded input, vary weights):
- healthy step_18800: realized gain **0.85 (contractive!)**, **σ_max≈22–115** (eps-dep, robustly ≫1) = STRONGLY
 NON-NORMAL, the §5 regime CONFIRMED empirically.
- scrambled E3a/20000: realized 98–2147, **σ_max≈1e5–2e6** (operator WRECKED, loss-masked by RMSNorm).
- detonated P1/20000: realized 5.9e5, **σ_max≈6e7**.
**σ_max rises ~6 ORDERS healthy→blown — the detonation IS the operator running away in WEIGHT SPACE.** No
forward-side realized clamp can cure (E3a clamped the carrier; weights stayed expansive). **Cure naïve form
RULED OUT:** healthy WORKING model runs at σ_max~22 (computes USING the non-normal map) → forcing σ_max≤1
lobotomizes it. Cure = PREVENT σ_max RUNNING AWAY (generous cap ~2-3× healthy), weight-space OR optimizer-side.

## E5 — ▶ RUNNING: IS IT THE OPTIMIZER? (disambiguates the cure fork before any risky build)
`ignore/E5_adamw_sigma_control_2k.sh`: init_from step_18800 (WEIGHTS only) + **AdamW8bit** (the survivor),
frozen-0.45, to 2000 steps (covers the β1=0 detonation window), CORECOS+PERITER. step-0 ppl 24.4 healthy.
NEXT on exit: run E2 σ_max on E5/step_2000.
- POS (expected): AdamW survives + σ_max STAYS ~tens → the ARCHITECTURE tolerates σ_max~22; β1=0 NOISE drives
 the runaway → **cure = OPTIMIZER-SIDE σ_max control (preserves β1=0 memory)** is viable + preferred ('s
 β1=0 fixation). Also opens: AdamW survives BECAUSE momentum smooths updates out of the high-σ direction.
- NEG: AdamW also detonates from these weights → weights already past no-return → cure must be weight-space.

## E3 — CURE BUILD (after E5 disambiguates): cap σ_max RUNAWAY (generous, ~2-3× healthy ~22)
Two branches by E5: (A) optimizer-side — damp the high-σ-direction component of the β1=0 update (what β1=0.9
does implicitly); (B) weight-space — Miyato spectral-norm / soft σ_max-penalty on the core branch ternary
linears (no double-backward: σ=uᵀW_q v power-iter), σ_target generous. Gate bit-exact-off + σ_max-reduction
(measure with E2). DO NOT force σ_max≤1 (lobotomizes). Hold full launch for operator go-ahead on the architecture.

## E4 — stale-momentum analysis (cheap, complements E5): cos(m_slow, g) across the run-up
Does the AdEMAMix slow direction (m₂, β3-EMA) go stale/anti-aligned with g as σ_max climbs? Tests the
"steers on stale directions" half. Can pair with an in-training σ_max logger (reuses `_apply_core_step`).

## Subagent pattern
Runs self-manage (watchdog: blowup ppl>150/inf/nan → +1500 → kill exact PID) and notify parent on exit.
On exit → spawn analysis subagent (read runlog, align fwd→step, extract the decisive stat) → tight report →
parent decides next + launches next + builds the following. Keeps orchestrator context lean.
