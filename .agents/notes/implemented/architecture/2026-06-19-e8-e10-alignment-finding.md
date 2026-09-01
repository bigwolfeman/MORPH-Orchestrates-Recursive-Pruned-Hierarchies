# Agent Note: E8 E10 Alignment Finding

Status: implemented

Origin: Ai-notes/06-19-2026/MORPH-Iterative-Map-Dynamics/E8-E10-ALIGNMENT-FINDING.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# E8/E9/E10 — the σ_max(J_core) runaway is INTER-BLOCK SUBSPACE ALIGNMENT

**Date:** 2026-06-19 (Task #276, β1=0 AdEMAMix stability campaign)
**TL;DR:** The weight-space spectral-penalty cure (E8) detonated. Localization (E9/E10) proved the
σ_max(J_core) runaway is **not** in any single operator — it is the **inter-block alignment of the
6 core blocks' top singular subspaces**. Per-operator (per-linear / per-block) spectral penalties are
therefore a dead approach. The cure must be composition-level or optimizer-level.

## E8 — the cure ran, then detonated

Full β1=0 deploy stack (m₂-SNR gate κ=0.3 + update_clip5 + α=8 + β2/β3=0.999) **plus** the soft
per-core-MLP-linear spectral penalty `L = λ·Σ relu(σ_max(Wᵢ) − cap)²`, cap=3.0, λ=1.0 (healthy
gate_up σ_max ~1.5 → cap = 2× headroom). `init_from` step_18800, frozen topology, 2500 steps.
Gate `SPECTRAL_PENALTY_GATE_PASS` (estimator exact to 1e-13; penalty proven to reduce σ; idle below cap).

| step | 0 | 600 | 1200 | **1400** | 1600 | 1800 | 2000 |
|---|---|---|---|---|---|---|---|
| ppl | 25.2 | 30.6 | 28.8 | **497** | 5272 | 31430 | 389434 |

VAL: 21.6 → 20.8 → (1250) 43.6 → (1500) **1491** → (1750) 18418 → detonated. Watchdog killed @2200.
**Same single-step cliff character as the un-cured P1/M3 (~800–1200 past the same resume).** The cure
delayed the cliff marginally at best.

## E9 — the penalty never engaged; MLP weights are innocent

`ignore/E9_diag_sigmas.py` on the healthy seed (step_18800) and E8's last pre-cliff ckpt
(step_1250, VAL 43.6 — already creeping). Reports per-core-linear σ_max (what the penalty acts on)
**and** the full-composition σ_max(J_core) on the same loaded model.

| ckpt | MLP gate_up σ_max (max) | MLP down σ_max (max) | linears over cap=3 | **composition σ_max(J_core)** | realized gain |
|---|---|---|---|---|---|
| seed step_18800 (healthy) | 1.52 | 0.40 | **0/12** | **54.6** | 0.86 |
| E8 step_1250 (cliff edge) | 1.51 | 0.41 | **0/12** | **800.1** | 0.74 |

The composition σ_max climbed **15× (54.6→800)** while **every** MLP linear weight σ_max sat **frozen**
(~1.5 gate_up, ~0.4 down), never crossing cap=3 — penalty **idle 0/12 the entire run**. This is a clean
falsification of M3's "gate_up is the runaway epicenter": M3 located the worst optimizer-*update*
position, which is **not** the σ_max driver.

## E10 — not any single block either: it's their ALIGNMENT

`ignore/E10_localize_sigma.py` decomposes J_core into its 6 per-block Jacobians, each measured at its
true operating-point input (captured via forward hooks on a real core step).

| block | seed σ_max(J_blk) | cliff σ_max(J_blk) |
|---|---|---|
| core.0 | 34.3 | 25.7 |
| core.1 | 48.5 | **84.4** |
| core.2 | 27.2 | 65.0 |
| core.3 | **67.5** | 40.9 |
| core.4 | 35.0 | 55.1 |
| core.5 | 31.1 | 56.2 |
| **composition** | **60.4** | **800.1** |
| worst single block | 67.5 | 84.4 |

**The decisive ratio:** at healthy, composition (60) ≈ the single worst block (67) — the blocks are
**non-aligned**, their worst-case directions don't chain, the composition is contractive overall
(realized gain 0.86). At the cliff, composition (800) is **~10× the worst block (84)** — the blocks'
top singular subspaces have **rotated into alignment**, so amplification now **chains
multiplicatively** across the core instead of cancelling. Individually every block stays in the *tens*.

### E10 on the AdamW control — the optimizer A/B nails the mechanism

Ran E10 on the **stable** AdamW control (E5, β1=0.9 α=0) at step_2000. Alignment ratio = composition
σ_max ÷ worst-single-block σ_max:

| ckpt | optimizer | composition σ | worst block | **ratio** | realized gain | state |
|---|---|---|---|---|---|---|
| seed step_18800 | AdEMAMix (pre-detonation) | 60 | 67 | **0.90** | 0.86 | non-aligned |
| **AdamW step_2000** | **AdamW β1=0.9 α=0** | **47** | **51** | **0.91** | **0.58** | **non-aligned (STABLE)** |
| E8 step_1250 | AdEMAMix (cliff edge) | 800 | 84 | **9.5** | 0.74 | **ALIGNED → detonates** |

AdamW holds the ratio at ~0.9 (composition ≤ worst block) — the healthy non-aligned regime — while
AdEMAMix drives it to ~9.5. **The optimizer is unambiguously the cause of the inter-block subspace
alignment.** Complete causal chain: optimizer coherence (β1=0 no-smoothing + α=8 heavy slow EMA) →
inter-block singular-subspace alignment → composition σ_max runaway → single-step detonation.

## Entry injection — analytically ruled out (no probe needed)

`DiagonalInjection` (the one operator inside J_core but excluded from the per-block measurements):
`A = log_A.exp.clamp(max=0.9999)`; non-ctx channels pass through unchanged (Jacobian = 1); `e` is
independent of `h_in` (the `dt·e_ctx` term is constant wrt the perturbation). ⇒ **σ_max(J_injection) ≡
1.0 by hard clamp** — "spectral radius < 1 guaranteed by construction" holds. Not the amplifier.

## Conclusion + cure implications

- **MLP weights** (E9), **per-block norms** (E10), and the **entry injection** (analytic) are all ruled
 out. The σ_max(J_core) runaway is **inter-block singular-subspace alignment** — a property of the
 blocks' *relative orientation*, not any operator's norm.
- **DEAD:** per-block / per-linear / per-operator spectral penalties (incl. raising λ or extending E8 to
 attention/HC). Each operator is individually fine; there is nothing locally over any cap to penalize.
- **Mechanism hypothesis:** β1=0 (no fast-momentum smoothing) + α=8 (heavy slow EMA) = a coherent,
 sustained directional drift that rotates the blocks' singular subspaces into alignment over ~1250
 steps. AdamW's noisier/shorter-memory updates don't sustain that coherence → it holds σ_max in the
 tens (E5).
- **Next-cure candidates (SURFACED — expensive build + real re-route risk):**
 - **(a) Composition-level σ_max(J_core) penalty** — penalize the order parameter directly via a JVP
 power-iteration through the whole core step (cap ~100 ≈ 2× healthy). Directly de-aligns. Cost
 ~+15–30%/step; risk: alignment may re-route around a single composition cap; differentiable-σ
 through the checkpointed/dynamo-disabled core is non-trivial.
 - **(b) Optimizer-level** — address *why* long-memory AdEMAMix drives coherent subspace alignment
 (e.g. the slow-EMA branch's coherence on the subspace-controlling params).

## E11 — α-sweep: α IS the alignment knob, sharp threshold between 4 and 6

Held E8's EXACT stack (β1=0 + gate κ=0.3 + clip5 + β2/β3=0.999 + eps_inside=false), from step_18800,
frozen, 2500 steps, varied ONLY α. After each arm, E10 on the latest ckpt for the alignment ratio
(composition σ_max ÷ worst-single-block σ_max; healthy/AdamW ≈ 0.9, E8 α=8 cliff = 9.5).

| α | outcome | comp σ_max | worst block | **align ratio** | VAL through window |
|---|---|---|---|---|---|
| 0 | SURVIVED 2500 | 36.3 | 65.9 | **0.55** | healthy 21–26 |
| 2 | SURVIVED 2500 | 40.9 | 58.1 | **0.70** | healthy 21–26 |
| 4 | SURVIVED 2500 | 34.4 | 61.3 | **0.56** | healthy 21–26 |
| 6 | DETONATED @1800 | 114 (step_1250, pre-cliff) | 86.6 | **1.32** (crossing) | 26→3352→53k |
| 8 (E8) | DETONATED @1400 | 800 (step_1250, pre-cliff) | 84 | **9.5** (aligned) | 26→1491→18k |

**The alignment ratio is MONOTONIC in α** with a sharp critical horizon between α=4 (ratio 0.56,
stable) and α=6 (ratio 1.32 at step_1250, detonates @1800). α=8 aligns harder/faster (9.5, @1400).
This is direct evidence for the local-linearity / curved-manifold mechanism : **α = how far
back the slow EMA's stale tangent reaches**; past the critical horizon the accumulated direction
drifts off the current local linearization → rotates the blocks' subspaces into alignment → detonates.
α=0 surviving shows the gate+clip stack stabilizes β1=0 by itself; the α-boost re-introduces the
instability only above the horizon.

**α=4 (β1=0) = largest stable, non-aligned config in this screen** — keeps a real slow-momentum boost.
CAVEATS (no-theater): NO-PRUNE / frozen / 2500-step SCREEN only (α=6 reached 1800 before dying, so a
survivor still needs a longer confirmation); deploy prune-shock UNTESTED; whether α=4 gives a real
convergence WIN over AdamW8bit needs a longer head-to-head (wandb quality call). Scripts:
`ignore/E11_alpha_sweep.sh` + `.summary`, arm runlogs `ignore/E11_alpha{0,2,4,6}.runlog`,
ckpts `checkpoints/morph/b1zero_E11_alpha{0,2,4}.0/step_2500.pt`.

### Next: optimizer-side de-coherence (the plan)
The α-sweep is the empirical map of the tolerable EMA horizon. The de-coherence goal: let α=8's gain
run WITHOUT the alignment — i.e. keep the slow-EMA magnitude but strip its off-manifold COHERENCE
(the component that systematically rotates subspaces). Candidate framings to refine :
- the slow EMA m₂ assumes local linearity over its memory horizon; on the curved looped manifold the
 stale-tangent component is what aligns. A de-coherence step would project/whiten m₂ to remove the
 component that's no longer a valid local descent direction (e.g. orthogonalize m₂ against the current
 gradient's complement, or decay the part of m₂ that has rotated away from the live gradient).

## E12 — slow-EMA geometry: m₂ goes STALE (⊥g) + FROZEN, and detonates when g collapses

Instrumented the detonating α=8 config (E8 stack, step_18800, frozen) with `MORPH_DIAG_M2G` logging,
per core param every step: |m₂|, |g|, cos(m₂,g), and m₂ self-coherence cos(m₂_t, m₂_{t-50}).
188 core params; blowup @1200.

**Median over core params:**

| step | medCos(m₂,g) | med m₂ self-coherence |
|---|---|---|
| 24 | 0.43 | 0.11 |
| 124 | 0.11 | 0.76 |
| 624 | 0.05 | 0.93 |
| 924 | **0.00** | **0.99** |
| 1199 (pre-cliff) | 0.03 | 0.96 |

m₂ converges to a **fixed, stale direction orthogonal to the live gradient**: self-coherence ≈0.99
(m₂ barely rotates over 50 steps = frozen) while cos(m₂,g)≈0 (that frozen direction is not where g
points). The α·m₂ term is a persistent **off-tangent** push the live gradient no longer endorses —
the local-linearity break made concrete (the EMA locked onto an old tangent; the manifold curved away).

**The detonation trigger — g-collapse makes the stale push dominate (core MLP gate_up, α=8):**

| step | cos(m₂,g) | **\|α·m₂\|/\|g\|** | regime |
|---|---|---|---|
| 124 | ~0.13 | ~2.8–3.0 | healthy |
| 624 | ~0.08 | ~1.5 | healthy |
| **999** (pre-cliff) | **~0.00** | **540 – 1200×** | about to blow |
| 1199 | ~0.00 | 6–50 (g already blown) | in cliff |

Near the critical point the **live gradient g collapses** on the core MLP weights while m₂ keeps its
large frozen magnitude → the update becomes an almost-pure stale-momentum jump of ~1000× the live
gradient, off-tangent → tips the detonation at 1200.

**Why update_clip=5 (ACTIVE) didn't save it:** per-COORDINATE clipping cannot stop a COHERENT
directional push. Clipping each coord to 5 still lets the whole tensor slide ~5·√N in m₂'s fixed
direction, and a coherent directional weight move is exactly what rotates the block's singular
subspace. **The disease is directional/coherent; a per-coord magnitude cap is the wrong instrument.**

⇒ **Cure family confirmed = DIRECTIONAL** (gate α by cos(m₂,g), or strip the m₂⊥g component) — it acts
on the coherent direction the per-coord clip can't touch. Discriminator (stable α=4 vs detonating α=8)
being measured in E13 (the α=4 M2G control): cos-gate works if cos(m₂,g) is higher when stable;
otherwise the discriminator is the stale-push MAGNITUDE |α·m₂|/|g| (per-tensor, directional) and the
cure caps that. Probe: `morph/training/train.py::diag_m2g_geometry` (env MORPH_DIAG_M2G), run
`ignore/E12_m2g_geometry_a8.sh` (+ `.m2glog`).

## E14 — de-coherence A/B (both cures built + run): STP HARMFUL, CAP DELAYS-not-cures

Both cures built+gated, run on the α=8 detonating stack (E12 baseline: dies @1200, ratio 9.5),
step_18800, frozen, 1600 steps, with engagement verification + post-run E10.

| arm | setting | outcome | engagement | verdict |
|---|---|---|---|---|
| baseline (E12) | α=8 | detonated @1200 | — | ratio 9.5 |
| **STP** | loop_stp_lambda=0.3 | **detonated @400 (WORSE)** | loop_stp 1.12→0.84 (active, reducing) | **harmful** |
| **CAP** | stale_push_cap=5 | detonated @1400 (delayed ~300) | stale_cap_ev 2948→40907 at cliff | delays, no cure |
| both | STP+CAP | (STP-poisoned) | — | — |

- **Loop-axis STP (the idea) FAILED — actively harmful at λ=0.3.** It *did* flatten the loop
 trajectory (loop_stp 1.12→0.84) but the model detonated FASTER than baseline (400 vs 1200). The aux
 gradient destabilizes the already-fragile optimization — the exact symptom-mask trap pre-registered
 (regularize the realized trajectory, miss/worsen the off-trajectory σ_max). Could be λ-too-high, but
 the direction is bad. Same failure class as the earlier max_gain-clamp and the #266/#268 world-model
 aux objectives (aux on the looped carrier → touchy/divergent).
- **CAP (per-tensor ‖α·m₂‖≤5‖g‖) is the best cure so far but only DELAYS.** Held ppl healthy through
 step 1200 (baseline was ppl ~2071 there), cap firing at a low steady ~2.5 tensors/step. At the cliff
 the event count explodes 2948→40907 (~190/step = every tensor capped every step = GLOBAL g-collapse)
 and it detonates anyway (post-cliff E10 σ_max 2.2e8). ⇒ the cap addresses the FIRST failure mode
 (local g-collapse lets stale m₂ dominate) but a SECOND mechanism takes over at the global cliff. A
 tighter c (2–3) might extend the delay but the re-route suggests it won't fully cure.

**Conclusion:** the loss-side/optimizer-side cures do not cleanly hold α=8 at these settings. PIVOT
: **deploy at the stable α=4** (E11) instead of curing α=8. Two unknowns decide if it's worth
shipping over AdamW8bit: (1) prune-shock survival @29k (untested — all historical deaths were
prune-triggered), (2) does α=4's gain beat AdamW8bit ppl 31-32 (untested — at 2500 steps from a
converged seed all α looked identical). Both die in one real deploy-schedule run. Free upgrade:
**α=4 + stale_push_cap=5** — at α=4 the amR stays ≤1.1 (E13) so the cap is bit-inert in healthy
training (never fires below 5) but auto-catches a prune-shock spike → strictly-dominant insurance at
the one untested risk. Decision: run the full deploy schedule at α=4(+cap) after the E14 'both' arm
completes (locked decision). Scripts: `ignore/E14_decoherence_ab.sh` + `.summary`; cures in
`morph/model/transformer.py` (loop_stp), `morph/training/ademamix_b1zero.py` (stale_push_cap).

## Artifacts
- `ignore/E8_spectral_cure_2500.sh` (+ `.runlog`), ckpt `checkpoints/morph/b1zero_E8_spectral_cure_2500/step_1250.pt`
- `ignore/E9_diag_sigmas.py`, `ignore/E10_localize_sigma.py`
- `morph/training/spectral_penalty.py` (the E8 penalty — gated, but targets the wrong object; keep as the
 building block for a composition-level variant)
