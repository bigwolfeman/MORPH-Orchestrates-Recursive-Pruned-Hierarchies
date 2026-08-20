# Agent Note: Iterative Map Dynamics

Status: implemented

Origin: Ai-notes/06-19-2026/MORPH-Iterative-Map-Dynamics/MENTAL-MODEL.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# MORPH as a Nested Dynamical System — a core mental model for optimizing the looped architecture

**Date:** 2026-06-19 · **Origin:** Wolfe's framing during the β1=0 AdEMAMix stability campaign (Task #276).
**Status:** strongly-motivated FRAME with one decisive prediction still to be measured (`ρ(J_core)`, see §5).
This is a *mental model*, not yet a proven theorem — it unifies every data point we have and makes a
falsifiable prediction.

---

## 1. The two nested dynamical systems

MORPH has **two time-axes**, not one:

- **Outer (slow) — optimization.** `θ_{t+1} = θ_t − η·u_t`. This is the dynamicism *every* optimizer is
  built to model. AdEMAMix models it richly: fast EMA (β1), slow EMA (β3·α), "old gradients stay relevant."
- **Inner (fast) — the forward pass itself.** `h_{k+1} = f_θ(h_k)`, k=1…T. The hidden carrier evolves under
  **repeated application of the SAME weight-shared map** (Parcae core, T = Poisson depth mean 6). This axis
  exists *only* because of the loop. A feedforward net does not have it.

**The optimizer's blind spot.** The optimizer's entire input is `∇_θ L` — one vector per parameter that
*integrates over the whole inner trajectory and discards its structure*. It has **no state variable** for
whether the inner map `f_θ` is contractive (stable) or expansive (divergent). For a normal model that loss
is total — the only dynamicism is the outer one, fully captured. **The loop adds a second dynamical system
the optimizer cannot see.** AdEMAMix is therefore not *wrong* here — it is **incomplete**: it models `dθ/dt`
but is blind to `ρ(J_θ)`, the contractivity of the inner map it is simultaneously deforming.

---

## 2. How the iterative map reshapes the loss landscape

A looped loss is `L(θ) = ℓ(f_θ^{∘T}(x))` — the **T-fold self-composition** of one map. The chain rule gives
two consequences that *redefine the landscape's geometry*:

1. **Sensitivity scales as the spectral radius to the T-th power.** The gradient through T steps carries
   `∏ J_k` with `J = ∂f/∂h`. If `ρ(J) < 1` contributions decay (smooth loss); if `ρ(J) > 1` they grow like
   `ρ^T`. A *mild* curvature feature of the single-step map becomes that feature **raised to the power T** in
   the loss. (This is the classic BPTT exploding/vanishing-gradient fact, here re-cast as landscape geometry.)

2. **The landscape inherits the BIFURCATION STRUCTURE of `f_θ`.** θ-space partitions into a **contractive
   region** (`ρ<1`: stable orbit, smooth loss) and an **expansive region** (`ρ>1`: the iterate diverges, loss
   explodes), separated by a manifold where `ρ=1`. That manifold is the "cliff" — and its steepness grows
   like `ρ^T`, so at depth ~6 it is razor-sharp. **It is not a generic landscape feature; it is a
   dynamical-systems bifurcation boundary written into the loss geometry.**

**Unification.** "Navigating the loss landscape" and "the nested inner dynamicism" are the **same object**
viewed from two sides: the inner dynamics *sculpt* the cliffs the optimizer then has to navigate. The
optimizer, blind to `ρ`, walks across the `ρ=1` manifold carrying stale, α-amplified momentum, and the loop
turns that one bad step into `ρ^T`.

---

## 3. Why this matters specifically for AdEMAMix on MORPH

AdEMAMix's premise — *old gradients stay relevant* — may be **structurally false near a bifurcation of a
weight-shared map**: the map's fixed point moves every time θ updates, so the gradient field's *sign* can flip
across `ρ=1`. A long-memory optimizer then confidently steps in a direction the (now-bifurcated) map has
reversed — and the loop amplifies it `ρ^T`. This is the deep reason "incomplete," not "failed."

---

## 4. The evidence chain (all from Task #276, density-0.45 frozen unless noted)

Four-config triangulation — **both AdEMAMix deviations from plain Adam destabilize *independently*:**

| config | numerator | slow boost | outcome |
|---|---|---|---|
| AdamW8bit | m₁ (β1=0.9) | α=0 | **survives** (baseline ~31–32 ppl) |
| ours (T1/M3) | raw g (β1=0) | α=8 | detonates |
| A1 | raw g (β1=0) | **α=0** | detonates (~same step) → α·m_slow NOT the driver |
| ARM D | m₁ (β1=0.9) | α=8 | detonates → adding the boost breaks even a smoothed base |

Only plain-Adam (β1=0.9 **and** α=0) survives. Drop the smoothing **or** add the long-memory boost → death.

**M3 (no governor, captured an actual blowup):** at the detonation the per-token **`max_gain` (core
amplification) explodes to 10³–10⁵×** while bulk direction (`mean_cos`, `frac_rot`) does **not** move; carrier
saturates to `nan`; **NO recovery** (loss pinned 10⁵–10⁶ for 1500 steps). `max_gain` is a crude one-token
Rayleigh-quotient proxy for local expansion `ρ`. First extreme spike fired ~hundreds of steps *before* loss
reacted (RMSNorm `final_norm` masks it) → **early-warning lead.**

**L1g τ=2 (magnitude governor `core_gain_clip=2.0` + CORECOS, the decisive intervention):**
- Governor confirmed biting: `max_gain` pinned at exactly **2.000** at the spike (0 forwards > 2.01).
- **Yet loss still spiked** (ppl 27→926), then **recovered to ~45–58** (vs M3's unrecoverable 10⁵).
- Realized-direction metrics show **no pathological signature** at the spike: healthy `mean_cos 0.207 /
  frac_rot 0.782` vs spike `0.197 / 0.645` (frac_rot even *drops*). (`min_cos` is order-stat-saturated at −1
  in both regimes — not a discriminator.)

**Verdict:** clamping the *realized* expansion (magnitude) **masks the symptom, not the disease.** It limits
severity (unrecoverable→recoverable) but cannot prevent the excursion or return to baseline (settles in a
*scrambled* basin ~45–58, not the healthy ~28 → deploy-unacceptable per Wolfe's "doesn't return to baseline =
periodic scrambling"). The pathology lives in the map's **sensitivity `σ_max(J)`** — which drives the gain
toward 10⁴ and which the governor (a direction-PRESERVING rescale applied AFTER the amplification) doesn't
touch and CORECOS's realized-state metrics cannot see.

**P1 (per-iteration gain, governor OFF `core_gain_clip=0`, captured the blowup at step 19600 — the decisive
mechanism read):** logs per-loop-iteration `max_gain` (`MORPH_DIAG_PERITER`). Three findings that **refine
the frame and correct one earlier assumption:**
- **Corrected baseline:** the healthy steady-state per-iteration profile is **NOT monotonically decaying** (that
  `[1.37→1.12]` was the *compile-warmup* profile). It is a **hump** — contract at iter0–1, peak ~**1.06 @ iter3**,
  settle to ~1.0. So **healthy training already runs at σ_max≈1** (marginal isometry, exactly the HC-Cayley
  design point). Any contractivity fix must allow this hump — forcing σ_max≪1 everywhere would kill expressivity.
- **The blowup is a single-iteration transient spike (POS-b), NOT monotonic ρ^T compounding (POS-a is
  FALSIFIED):** at detonation **one** loop iteration jumps to **10²–10³×** (max finite pre-detonation gain
  **4250**) at a **varying** index (i3–i7); its neighbors stay healthy (~1.0). It does not compound smoothly up
  the loop index, and iterations do not all rise together. → the disease is **occasional worst-case σ_max≫1
  excursions of ONE core application**, not a uniformly-expansive map.
- **The real time-axis of the disease is FREQUENCY over training, with ~600-step lead:** spikes (gain≥1.5)
  first appear at step ~19000 **while ppl is healthy at 27.5**; spike *rate* climbs 2.5%→7%→**89.5%** over steps
  19000→19600; ppl first crosses 150 at **19600**. So the operator excursions lead the loss reaction by ~one-to-two
  200-step buckets (~400–600 steps). Post-blowup → inf/nan, degenerate i0/i1 profile, no recovery.
- Direction metrics (`mean_cos`/`frac_rot`) stay flat across the spike (go nan only *after*), reconfirming the
  pathology is **magnitude/σ_max, not rotation.**

**Refined target (post-P1):** the fix is not "make the map globally contractive" but **cap the worst-case
one-step operator gain `σ_max(J_core)` of a single core application** (e.g. via spectral/Lipschitz control on the
core branch with target σ_target>1, generous enough to preserve the healthy 1.06 hump) so **no input direction
can be amplified catastrophically** — distinct from the governor, which rescales *after* the amplification (and so
preserves the corrupted direction → masks-not-cures). The ~600-step lead also opens an early-warning/early-stop
option if a preventive constraint proves insufficient.

**E2 — σ_max(J_core) MEASURED DIRECTLY (2026-06-19, `ignore/E2_sigma_on_ckpt.py`, the decisive measurement):**
the validated estimator (Gate A non-normal PASS) on the EXACT training core map (Gate B core-step parity
**bit-exact 0.000e+00** vs the real forward, via the new `_apply_core_step` method), controlled design (same
seeded input, vary only the weights). **σ_max(J_core) is the ORDER PARAMETER of the detonation:**

| checkpoint | state | realized gain ‖f(h)‖/‖h‖ | σ_max(J_core) |
|---|---|---|---|
| step_18800 | healthy (ppl~27) | **0.85** (contractive) | **~22–115** (eps-dependent, robustly ≫1) |
| E3a/step_20000 | scrambled (ppl~70) | 98–2147 | **~1e5–2e6** |
| P1/step_20000 | detonated | 5.9e5 | **~6e7–6e8** |

Three conclusions, each hard-numbered:
1. **The healthy operator is STRONGLY NON-NORMAL** — realized gain 0.85 (contractive!) while σ_max≈tens. This is
   EXACTLY the §5 non-normal regime (ρ-like realized contraction, σ_max≫1 transient capacity), now empirically
   confirmed. The carrier rides a benign subspace; the top-σ direction is a loaded gun.
2. **The detonation IS the operator running away in WEIGHT SPACE** — σ_max rises ~6 orders of magnitude (tens →
   10⁷) healthy→blown. Not a forward-state event; a weight/operator event the optimizer drives. The eps-ambiguity
   in the *healthy* absolute value is irrelevant — the 6-order signal dwarfs it.
3. **The "scrambled recovery" is operator-level FALSE** — E3a's loss recovered to ppl~70 but σ_max stayed ~1e6,
   realized gain ~1e3; RMSNorm `final_norm` masks the wrecked operator in the loss. ⇒ Wolfe's "periodic
   scrambling → won't converge" is an OPERATOR fact, and **no forward-side realized clamp can cure** (E3a clamped
   the carrier while the weights stayed catastrophically expansive — confirmed).

**CURE (re-posed by E2, naïve form RULED OUT):** the healthy WORKING model runs at σ_max~22 — it *computes using*
the non-normal map — so **forcing σ_max≤1 would lobotomize the ppl-27 model.** The cure must **prevent σ_max from
RUNNING AWAY** (a generous cap ~2–3× healthy, biting only on the 22→10⁷ blowup), via a weight-space spectral
constraint/penalty on the core branch — OR fix the β1=0 optimizer noise that drives the runaway (β1=0.9/AdamW
survives precisely because momentum-smoothing keeps updates out of the high-σ direction; the "incomplete
optimizer" half, now concrete). **CAVEATS (no-theater):** random-input operating point (real-data h unverified,
expected similar — it's a weight property); FD σ_max is eps-dependent at the healthy point (22@1e-2 → 115@1e-4,
~2×/decade ⇒ sharp local curvature, NOT pure roundoff which would be ~10×/decade) — an exact double-VJP JVP
cross-check would pin the healthy absolute value but does NOT affect the 6-order healthy→blown comparison.

---

## 5. The decisive prediction + the fix family

**The quantity is `σ_max(J_core)` (spectral NORM), NOT `ρ(J_core)` (spectral radius).** Refinement forced by
the estimator gate (`ignore/verify_rho_probe.py`, 2026-06-19): plain finite-difference power iteration
converges to the spectral *radius* (max |eigenvalue|), but our blowup is a **finite-T transient** (depth ~6),
and for **non-normal** maps (neural cores are non-normal) transient amplification is governed by the spectral
*norm* `σ_max` and can be huge even when `ρ<1` (pseudospectra / non-normality). Measuring `ρ` alone risks a
false "frame wrong" when `σ_max>1` is the true driver. `σ_max(J)` needs a JᵀJ power iteration (one JVP + one
VJP per step) — the rigorous probe, to build with Wolfe (autograd-graph care through the compiled core).

**Tonight's validated proxy (built + RUN — see P1 in §4):** per-loop-iteration `max_gain` logging
(`MORPH_DIAG_PERITER`, reuses the trusted CORECOS `_g`). `max_gain` = realized one-step amplification = a
data-direction lower bound on `σ_max`. **RESULT (P1, RESOLVED):** the "compounds monotonically across iteration
index" form is **FALSIFIED**; the actual fingerprint is a **single-iteration σ_max≫1 excursion at a varying
index, growing in FREQUENCY over training (~600-step lead before loss reacts)**. The frame survives in its
*essential* claim — `σ_max(J_core)>1` drives the blowup (lower-bounded at 4250 pre-detonation) and the loop is
the amplifier — but the time-axis of the disease is *training-step frequency of worst-case excursions*, not
*loop-index compounding within one forward*. This sharpens the fix from "global contractivity" to "cap the
worst-case one-step operator gain" (see Refined target in §4).

**If confirmed, the fix family is contractivity control, not symptom-clamping:**
- Give the *model* the guard the optimizer can't: keep `ρ(J_core) ≤ 1` — spectral/Lipschitz control or a
  direction-preserving per-iteration carrier renorm on the core step.
- This addresses the bifurcation itself, and — unlike a magnitude clamp — should let us **keep β1=0 (the
  memory win) AND α·m_slow (the training gain Wolfe won't give up).**
- Note: MORPH was *architected* for `ρ<1` — Parcae diagonal injection (spectral radius <1 by construction) +
  HC-Cayley orthogonal residual (`ρ=1` isometry). The open question is whether the **learned, ternary-
  quantized core under optimizer drift** pushes the *effective* `ρ` past 1 despite those guards. The probe
  answers it.

**Broader payoff (Wolfe):** this reframes optimizer design for *all* looped/recursive/weight-shared
architectures (universal transformers, HRM, diffusion-as-iteration, deep equilibrium) — a complete optimizer
for an iterative map needs awareness of the inner map's contractivity, a dimension current optimizers omit.

---

## 6. Files / runs
- `ignore/M3_corecos_30k.{sh,runlog}` (wandb s371f0ak) — captured detonation, max_gain explosion, no recovery.
- `ignore/L1g_tau2_corecos_24k.{sh,runlog}` (wandb b1zero_L1g_tau2_corecos_24k) — governor masks symptom.
- `ignore/P1_periter_gain_21k.{sh,runlog}` (wandb b1zero_P1_periter_gain_21k) — per-iteration gain, governor
  OFF; detonated step 19600; **resolved POS-b: single-iter excursion growing in frequency, ~600-step lead.**
- `ignore/E3a_contractive_tau1_24k.{sh,runlog}` (wandb b1zero_E3a_contractive_tau1_24k) — τ=1.0 non-expansion
  governor (tightest realized-magnitude clamp); cure-or-scramble test. [RUNNING / verdict pending]
- `ignore/verify_sigma_probe.py` — VERIFY_SIGMA_PROBE_PASS: validated σ_max(J) estimator (JᵀJ power iteration,
  FD-JVP + autograd-VJP), incl. the non-normal ρ=0.5<1<σ_max=6.04 case. Ready to wire as the E2 operator probe.
- CORECOS/PERITER diag: `transformer.py` gated `MORPH_DIAG_CORECOS` / `MORPH_DIAG_PERITER`; governor
  `model.core_gain_clip` (τ).
- Next (if E3a not a clean cure): cap worst-case `σ_max(J_core)` via a MAP/weight constraint (spectral/Lipschitz
  on the core branch, σ_target>1 to preserve the healthy 1.06 hump), gated bit-exact-off + σ_max-reduction proof.
