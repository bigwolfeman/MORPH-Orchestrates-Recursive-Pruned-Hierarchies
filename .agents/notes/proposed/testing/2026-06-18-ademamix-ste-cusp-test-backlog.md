# Agent Note: Ademamix Ste Cusp Test Backlog

Status: proposed

Origin: Ai-notes/06-18-2026/MORPH-AdEMAMix-STE-Cusp/TEST-BACKLOG.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# MORPH β1=0 AdEMAMix — Test Backlog (Claude + GPT)

**Task #276** · repo `00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies`, branch
`ablation/ademamix-optimizer` (UNCOMMITTED). Living doc — update STATUS as we run.

## Goal (unchanged, non-negotiable)
β1=0, 8-bit blockwise AdEMAMix both **stable** AND **≤ AdamW8bit loss** on the full deploy/prune
schedule. β1=0 is the memory win (drop `m1` → AdamW8bit parity 2.03 B/param). Stateless fixes
strongly preferred. **Bar to beat:** AdamW8bit (wandb `s4g72u7i`) → post-route VAL ppl ~31–32.

---

## ⚠️ FRAMING CORRECTION (2026-06-18, GPT review + verified by Claude)

My earlier "**STE-cusp-vault confirmed**" claim is **RETRACTED on the trigger**. Evidence:

- The 206k-flip "vault" steps land at **exactly `prune_interval=167`** spacing
  (17535 → 17702 → 17869 → 18036, Δ=167 each). A spontaneous optimizer cusp-crossing would not
  fall on a perfect schedule → **the spikes are PRUNE EVENTS**, not spontaneous vaults.
- **Diagnostic ordering confound**: `diag_forward_norms` computes TERNFLIP **after** prune+optimizer
  step (`train.py:1400/1439/1517`); a block the pruner just zeroed reads ±1→0 = a "flip". So the
  count mixes prune-mask zeroing with optimizer-driven code changes.
- GPT arithmetic: 12 blocks × 128 × 128 = 196,608 ≈ the 206k spike minus ~10k baseline churn. (NB:
  config says `prune_rate=0.005` is INERT at d=768 — 1-block floor — so the exact block count per
  event needs the split diagnostic to confirm; the periodicity is the hard evidence.)
- **What survives**: ternary scale IS a live moving boundary — `scale=mean|w.detach()|` recomputed
  every forward, threshold `|w/scale|>0.5` (`ternary_qat.py:191`). So scale-rethreshold is a *real
  candidate amplifier* — but the headline 206k number is NOT clean evidence for it (it matches
  pruning). **Better current read (GPT): AdEMAMix drives the ternary looped model into a fragile
  state where the scheduled MORTAR mask-shock becomes catastrophic.**

The flip spread is proportional to section block-counts (core 102k / coda 52k / prelude 50k ≈ 6:3:3),
consistent with EITHER direct mask zeroing OR a prune-induced global scale shift. The 3-way flip
split (below) is what disambiguates them.

---

## 🔬 M2 CORECOS RESULT (2026-06-19) — directional hypothesis NOT confirmed; diagnostic decoupled from loss

Measured `cos(h_a, h_new)` per core-step, **min-over-loop**, governor OFF (natural), resume
step_17500 → 18800, frozen prune. RC=0.

- **The run did NOT detonate** — final VAL ppl **25.19** (healthy). Train bounced
  25.8@18000 → 33.3@18400 → 39.2@18600 = the mild bounce, not the T1 blowup.
- CORECOS: fwd 1–16 healthy (mean_cos≈0.84, frac_rot=0.00, max_gain≈1.40); **fwd 17 onward**
  (≈step 17517) a persistent flip → mean_cos≈0.15–0.20, frac_rot≈0.71–0.81, max_gain DROPS to
  ≈1.06 and stays.
- **KILLER:** `frac_rot` is pinned ~0.79 IDENTICALLY at ppl=25.8 (healthy) AND ppl=39.2 (bouncing).
  The rotation signal does **not** track loss — it regime-shifted ~700 steps *before* any loss
  effect and is flat across healthy+degrading.

**VERDICT = (c) inconclusive / confounded, NOT (a) confirmation.** The min-over-loop reduction is
likely picking up *normal* late-loop-iteration orthogonality (carrier converges, increment turns
orthogonal as gain→1 — cos↓ and gain↓ together). Only `max_gain` bounded (magnitude not exploding)
is consistent with the directional framing, and that was already known from the clamp failures.
**Do NOT greenlight a rotation fix on this.**

NEXT DISCRIMINATOR (no run pending, Wolfe's call): log cos **per loop-iteration index** (not
min-over-loop) on a HEALTHY step vs an ACTUAL detonated step (paired) — distinguishes
"late-iter orthogonality, always present" from "a new early-iter rotation appearing only at the
blowup." Also flagged: a real persistent geometry regime change @~step 17517 the model TOLERATES.
Files: `ignore/M2_corecos_natural.{sh,runlog}` (wandb tnifs9np), diag gated `MORPH_DIAG_CORECOS`.

---

## RANKED BACKLOG

Legend — **STATUS**: ☐ todo · ▶ running · ✅ done · ✗ falsified · ⏸ deferred.
**COST**: code-only / short-replay (~30 min, resume step_17500) / full-35k (~hours).
**DECISIVE**: what a pass/fail actually proves.

### Tier 0 — diagnostics (do before any "fix"; cheap, settle the mechanism)

- ☐ **D1. Fix the diagnostic: split ternary flips 3 ways** *(code-only)* — GPT.
  `mask_flips` (newly killed by prune mask) / `alive_flips` (tile alive, code changed) /
  `scale_only_flips` (code changed because scale/threshold moved, not the weight). Plus a **fixed-batch
  post-update probe forward**, clearly labeled, so FWDNORM and TERNFLIP are same-frame causal.
  DECISIVE: separates direct prune-mask shock from optimizer/scale movement. **Root instrument — do first.**

- ☐ **D2. Per-section pre/post-prune residual-norm delta on a fixed batch** *(code-only)*.
  Measure core residual norm immediately before vs after the prune step on a held batch.
  DECISIVE: confirms the prune event (not an inter-step optimizer move) is what detonates the forward.

### Tier 1 — prune-aware tests (GPT's top picks; cheapest decisive experiments)

- ▶ **T1. Pause pruning 17500→19600, same gate+clip** *(short-replay, Wolfe greenlit ~30 min)*.
  DECISIVE: if the detonation DISAPPEARS → trigger is the prune schedule. **NOTE (Wolfe): we've seen
  β1=0 blow up *without* pruning before** (ORIGINAL plain β1=0 pre gate+clip; the gate+clip no-prune
  8k testbed was RC=0 stable). Resume-from-fragile-0.45 no-prune is NEW.
  - ## ✅ T1 VERDICT (v2, FWD-diag off, RC clean trajectory): **DETONATES WITHOUT PRUNING.** With
    pruning FROZEN it blew up at ~step 18800 — *earlier* than the prune-on run's 19400. Trajectory:
    18000 ppl 25.9 (VAL 27.97) → 18400 33.7 (VAL 32.0) → 18600 39.3 → **18800 457** → 19000 37,497
    (VAL 43,188). Killed @19000 (PID 3733808, GPU at 100%). **PRUNING IS NOT THE TRIGGER** → refutes
    the strong form of GPT's "prune-shock is the trigger." Detonation step is FP-knife-edge across
    conditions (18036 heavy-diag / 18800 no-prune / 19400 prune-on) but the INVARIANT holds: β1=0 at
    density ~0.41–0.45, step ~18–19k, detonates regardless of pruning. **What survives of GPT's read:
    the "AdEMAMix drives the model into a fragile state" half** — the fragile RMSNorm-masked
    exploding-core state (4e10 transients, v1) grows until one exceeds the recovery margin → permanent
    divergence, no prune event needed. **CONSEQUENCE: prune-aware fixes T2/T3/T4 are DROPPED** (won't
    help — pruning isn't the cause). Fix must target the OPTIMIZER or the looped-core gain.
  - **v1 (FWD-diag ON) — KILLED BY WATCHDOG @18309** (`cudaErrorLaunchTimeout` in CSA bwd; the
    per-step forward probe = documented footgun; ppl was HEALTHY at death, NOT divergence). v2
    relaunched FWD-diag OFF (PID 3733805) → clean run past 19400 to settle the ppl-detonation question.
  - **v1 PARTIAL DATA (steps 17501→18309) — TWO findings, both durable:**
    1. ✅ **Flip spikes ARE prune-driven** (GPT confirmed): **0** TERNFLIP spikes >50k with pruning
       frozen (max 14,860); sailed past would-be prune boundaries 17869/18036/18203 at ~10–13k churn.
    2. ⚠️ **Core forward EXPLODES anyway, DECOUPLED from pruning AND flips**: core residual norm hit
       **4.06e10 @ step 17815** with only **12,229 flips (normal churn)**, no pruning. Cluster
       17793–17917 (up to 4e10). ppl stayed HEALTHY (27.8→29.0→25.8→31.0, VAL 27.87) — RMSNorm
       `final_norm` pinned ~4547 masks it → recoverable transient.
    - **→ KILLS my "mass-ternary-flip vaults a cusp" mechanism** (flips were normal at the 4e10
      explosion). The forward blow-up is the looped-core T× amplification of an optimizer-driven
      perturbation, RMSNorm-recovered each step. **Supports GPT's synthesis**: β1=0 creates the fragile
      (RMSNorm-masked exploding-core) state; the prune-shock TIPS a recoverable transient past the
      recovery margin into permanent divergence. → elevates **L1 (bound looped-core gain)** and
      prune-aware tests; de-prioritizes flip-clamp (flips aren't the driver of the explosion).

- ☐ **T2. Halve prune_rate / double prune_interval around density 0.45→0.35** *(short→full)*.
  DECISIVE: if stable → practical fix that never touches β1 memory. (Caveat: prune_rate is inert at
  d=768; the real lever is prune_interval → e.g. 167→334 across the danger band.)

- ☐ **T3. Freeze CORE pruning past density ~0.45; prune only prelude/coda** *(short→full)*.
  DECISIVE: if stable → the looped core is the amplification-critical region (mask shock × T× reuse).

- ☐ **T4. Mask-change BUDGET per event** *(code + short)*.
  Cap blocks-pruned-per-event so no single step shocks the topology hard. Stateless.

### Tier 2 — optimizer/governor fixes (only after Tier-0/1 attribute the cause)

- ✗ **ARM D. Numerator momentum (`num_beta1=0.9`, 8-bit m1)** — FAILED, detonated ~18800. Also
  erodes the β1=0 memory win. Confirms we're NOT hitting the paper's β1/β3 instability.
- ⏸ **ARM A. Per-MODULE flip-clamp (`flip_clamp_kappa`)** — INERT at κ=0.03 (vault was 0.57%
  per-module). SUPERSEDED: per-module clamp will chase scheduled pruning and can't prevent the mask
  itself (GPT). Replace with ↓.
- ☐ **G1. ALIVE-only flip governor** *(code + short)* — GPT. Clamp the step only on flips of tiles
  that STAYED alive (exclude prune-mask flips). Only meaningful AFTER D1 proves alive-flips are
  nonzero at the event. Stateless.
- ☐ **G2. Global flip-spike clamp** *(code + short)* — Claude. Scale step down when TOTAL backbone
  flips spike ≥~10× trailing mean. SUPERSEDED-RISK: will chase scheduled pruning (G1 is the corrected
  form). Keep only as a coarse backstop.

### Tier 3 — root-cause / architectural (cheap memory, more invasive)

- ☐ **S1. Ternary scale EMA / freeze** *(code + short)* — both. Update `scale` once per optimizer
  step via EMA (NOT in-forward — compile/repeat-forward makes in-forward EMA state messy, GPT). Per-
  tensor scale buffers are tiny (β1=0 memory win intact). Plausible + cheap, but NOT yet supported by
  the 206k evidence → run after D1/T1. Verify-first (no GPU): does scale already recompute each
  forward? **YES — confirmed `ternary_qat.py:191`.**
- ☐ **L1. Bound looped-core gain** *(code + parity gate)* — both. Spectral/residual-norm clamp on the
  core block so a discrete flip can't amplify T× into 3.5e7. Safety net if prune-aware fixes preserve
  quality but occasional shocks remain. RMSNorm already half-does this (masks magnitude, not direction).
- ☐ **A1. Dial α/β3 toward AdamW** *(short)* — untested lever; diag showed raw-g (not α·m2) dominant,
  so likely a passenger, but cheap to check if needed.

### Tier 4 — stop condition
- ☐ **STOP. Ship AdamW8bit** (proven ~31–32 to density 0.25). Pocket the confirmed mechanism + the
  gate+clip "stable & quality-winning to density 0.45" result; revisit β1=0 memory win later.

---

## RESULTS LOG (runs in copy `00-MORPH-ademamix-b1zero`)
- ✅ **A1 α=0 — DETONATED @~18500** (VAL 117.85@18500 → ppl 2965@18600), same window as α=8. Since α
  scales one additive term monotonically and BOTH endpoints (0 and 8) fail identically → **α·m₂ is
  NOT the driver.** Skipped α=2/4 (confirmatory only — flagged as Claude's call). `ignore/A1_alpha0_noprune.*`
- ✅ **B1 fp32-state — DETONATED @~18200** (VAL 32.81@18000 → ppl 122@18200 → 50,199@19000), even
  EARLIER than 8-bit's 18800. Dequant confirmed active (216 states 8-bit→fp32). **→ 8-bit is NOT the
  culprit** (Wolfe's hypothesis FALSIFIED). Full-precision optimizer state didn't help. `ignore/B1_fp32_state_noprune.*`

- ✅ **L1 τ=2.0 — GOVERNOR WORKS, TOO LOOSE** (2026-06-18): converted the UNRECOVERABLE β1=0
  detonation (T1/A1/B1: 37k-137k, climbing) into a RECOVERABLE spike — 31@18200 → **845@18400** →
  210 → 94 → 60 (VAL 51.9@19000), DESCENDING. Bounding core-gain directly controls the failure
  (mechanism confirmed-by-intervention) but 2⁶=64× cumulative is too loose → tighten τ.
- ❌ **L1 τ=1.3 — MAGNITUDE GOVERNOR FALSIFIED AS CURE** (2026-06-18): peak ppl **784@18400 ≈ τ=2's
  845** (tightening the cap 64×→4.8× barely moved the peak) and recovered SLOWER (669@18600 vs τ=2's
  210). **WHY (no-theater): the magnitude clamp attacks the WRONG quantity.** `final_norm` (RMSNorm
  before the LM head) normalizes magnitude → the LOSS is magnitude-invariant. τ=1.3 forces h_new to
  ≤1.3·‖h_a‖ (a ~3700× downscale during the vault) yet the loss STILL spikes to 784 → the corruption
  is **DIRECTIONAL/structural, not magnitude**. The governor only changes the unnormalized gradient
  flow (hence the recovery-SPEED difference), not the spike. Wolfe's "periodic scrambling → won't
  converge" prediction SUPPORTED. **→ L1-magnitude is a DEAD END for the cure; going tighter (τ=1.2/1.1)
  would only hurt healthy signal more, not lower the peak (data is non-monotonic: 1.3 worse than 2.0).**
  logs `ignore/L1_gain1p3_noprune.*`, `ignore/L1_core_gain_clip2_noprune.*`.

  **WOLFE'S CALL (2026-06-19): don't give up, don't change direction — keep working the lever; his
  cosine/RoPE intuition → the failure is a carrier ROTATION (magnitude-invariant ⇒ directional).**
- ▶ **M1 — MEASURE carrier rotation at the spike** (PID 4030942). Added MORPH_DIAG_CORECOS to the core
  loop (per-step min per-token cos(h_new,h_a) + max gain; transformer.py, tensor-reduced 1 sync/fwd,
  NO hooks → no watchdog). Healthy baseline (random-init fwd): min_cos=0.92, max_gain=1.16. Run:
  resume step_17500, core_gain_clip=2.0 (magnitude BOUNDED ≤2) + DIAG on, to 18800. DECISIVE: if
  min_cos COLLAPSES at the 18400 loss spike WITH magnitude clamped → spike is a DIRECTIONAL rotation →
  build an ANGLE governor (cap per-iteration orthogonal/rotation component, the directional analog of
  the magnitude clamp). If min_cos stays ~0.9 → not a simple rotation → rethink. `ignore/M1_corecos_measure.*`

  **(prior fork, now being worked via M1 — NOT shipping AdamW yet):**
  - **(a) Directional/structural intervention** — the corruption is in the normalized direction, so a
    fix must constrain DIRECTION (e.g. cap per-iteration cosine change of the core output, or bound the
    per-step ternary code-flip COUNT directly = the realized-function change, not the optimizer step or
    the residual magnitude). Speculative, more code.
  - **(b) Accept β1=0 is unstable on this looped ternary core → ship AdamW8bit** (~31-32 to density
    0.25). The magnitude lever masks the symptom (faster recovery) but cannot prevent the directional
    scramble; per Wolfe's bar that's not deployable. The β1=0 memory win (~1 B/param) may be unreachable.
- ◽ (was) **L1 core-gain governor τ=2.0 — RUNNING** (PID 3905725, Wolfe's call). BUILT: per-iteration
  per-sample cap ‖h_new‖/‖h_a‖≤τ in the core loop (transformer.py, cfg `model.core_gain_clip`,
  default 0=off; plumbed train.py + base.yaml). GATED `ignore/verify_core_gain_governor.py` →
  VERIFY_CORE_GAIN_PASS 6/6 (math exact; τ=0≡τ=1e9 BIT-IDENTICAL on real 273M model = transparent
  when not biting; τ small bites + finite). CLEAN A/B vs T1 baseline: identical config (α=8, gate
  κ=0.3, clip5, frozen prune, 8-bit) + ONLY model.core_gain_clip=2.0. Vault was ~4800× single-iter
  gain → τ=2 kills it, healthy (gain≈1) untouched. VERDICT PENDING: survives 19800 healthy → cure +
  mechanism confirmed-by-intervention + β1=0 memory intact → full 35k deploy A/B; still detonates →
  tighter τ / per-token, or amplification isn't proximate. `ignore/L1_core_gain_clip2_noprune.*`

### 🔒 NARROWED CONCLUSION (3 hypotheses closed): the detonation is **β1=0-AdEMAMix-SPECIFIC**, at
density ~0.41–0.45 / step ~18–19k, INVARIANT to: pruning (T1), α·m₂ (A1 α=0), state precision (B1 fp32).
AdamW8bit (β1=0.9) passes through this exact density on the SAME architecture/data without detonating
(baseline s4g72u7i). So the cause is the β1=0 update SHAPE (raw-g numerator, no momentum smoothing) ×
the looped ternary core — NOT a quant/schedule/momentum-term artifact. ARM D (β1=0.9 numerator momentum)
also failed, so it's not a trivial "add momentum back" either. REMAINING FORK:
- **L1 — core-output governor** (GPT's plan, memory-PRESERVING, doubles as mechanism proof). Forward-side
  cap/scalar-gain on core block output when ||core_out||/||core_in|| exceeds threshold. CODE + parity gate.
- **(cheap probe) LR halve** 1e-4→5e-5 from step_17500: if β1=0's noisier raw-g steps are overshooting,
  lower LR defers/cures. CONFIG-ONLY, ~30 min. Low cost, sharpens "step-magnitude vs structural."
- **STOP — ship AdamW8bit** (~31–32 to density 0.25; β1=0 memory win may be unreachable on this stack).

## DECISION ORDER — POST-T1, GPT-REFINED (2026-06-18) — all runs now in the COPY `00-MORPH-ademamix-b1zero`
T1 dropped pruning as the trigger. Prune-aware (T2/T3/T4) DROPPED. New order (GPT):
1. **A1 FIRST — α sweep** {α=0, 2, 4, 8-control} from frozen-prune step_17500, same seed/data.
   *Cheapest, highest-signal, CONFIG-ONLY.* Rationale: the SNR gate damps ~100% of MLP coords
   (T1 optlog `snr<0.3≈1.000`), so the remaining AdEMAMix-specific force is the persistent **α·m₂
   drift**. If lowering α pushes the failure window out or clears it → it's OPTIMIZER DYNAMICS, not
   architecture alone. Sequential plan: run **α=0 first** (the extreme) — if it survives, sweep 2/4
   to find the threshold (then try α-warmup clipped at 2-4, or α-decay after density ~0.45); if α=0
   ALSO detonates in the same window → it's NOT α·m₂ → go to L1.
2. **L1 SECOND — but as an ACTIVATION/RESIDUAL GOVERNOR, not spectral norm** (GPT). A core
   residual-OUTPUT norm cap / learned-scalar residual gain on core-block outputs, triggered only when
   `||core_out||/||core_in|| > threshold`. Cheaper than spectral norm (which is expensive on a looped
   ternary core and may fight the learned representation) and directly attacks the observed failure
   mode. **Doubles as mechanism PROOF**: if the governor prevents detonation, the looped-core
   amplification mechanism is confirmed-by-intervention (currently only INFERRED — see caveat below).
3. **S1 — scale EMA: LOWER priority** (lost its strongest evidence when the flip spike turned out to
   be a prune artifact; still a plausible amplifier). Test after A1/L1 unless A1 is inconclusive.
4. **STOP / ship AdamW8bit** if A1+L1 both fail.

⚠️ MECHANISM IS INFERRED, NOT PROVEN (GPT's no-theater catch): the "looped-core norm amplification"
read rests on v1 partial data (4e10 core norm @ normal ~12k flips), but v1 WATCHDOG-DIED @18309 BEFORE
the 18800 detonation, and the clean v2 detonation run had FWD-diag OFF. So it's SUGGESTIVE. Prove it
via a lighter post-update forward probe (stride-20, core-only) OR by L1-intervention success.

(Superseded pre-T1 order kept for history: D1 split-diagnostic → T1 → prune-aware. D1 still useful as
the light forward probe to PROVE the mechanism, but no longer gates the fix.)

## Key files / ckpts
- `morph/training/ademamix_b1zero.py` — optimizer (gate+clip + `num_beta1`/`flip_clamp_kappa` knobs).
- `morph/training/train.py` — `diag_forward_norms` (FWDNORM/TERNFLIP) ← **D1 edits here**; resume + wandb-id sidecar.
- `morph/training/pruning.py:160-193` — prune loop (block-aligned 128×128, `[prune]` stdout print).
- `morph/configs/base.yaml` — `prune_interval:167`, `prune_rate:0.005` (inert at d=768), target_density 0.25.
- Checkpoints `checkpoints/morph/b1zero_gate_k0.3_clip5_deploy_35k/step_{...}.pt`; `step_17500` =
  healthy pre-danger (density ~0.45); deploy run detonated @~19400.
- `ignore/b1zero_diag_ste_cusp_replay.{sh,optlog}` (the periodicity evidence), `ignore/verify_cusp_fixes.py`.

## wandb-visibility footgun
Resume reads `wandb_id.txt` sidecar next to the ckpt → rejoins the ORIGINAL run → replay steps get
dropped (monotonic-step rule). For any visible replay: resume from a **sidecar-free COPY** of the ckpt
→ fresh wandb run.
