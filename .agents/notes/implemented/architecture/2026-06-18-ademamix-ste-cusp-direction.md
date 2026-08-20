# Agent Note: Ademamix Ste Cusp Direction

Status: implemented

Origin: Ai-notes/06-18-2026/MORPH-AdEMAMix-STE-Cusp/DIRECTION-REVIEW.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# MORPH β1=0 AdEMAMix — STE-Cusp-Vault: Direction Review (for GPT)

**Date:** 2026-06-18 · **Task #276** · repo `00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies`,
branch `ablation/ademamix-optimizer` (UNCOMMITTED). Reviewer: assess the diagnosis and pick/refine
the fix direction. We believe the mechanism is now **confirmed**; the open question is the fix.

---

## 1. Goal (non-negotiable constraint first)

Make a **β1=0, 8-bit blockwise AdEMAMix** optimizer (`AdEMAMixB1Zero`) both **stable** and
**≤ AdamW8bit loss** on MORPH's full deploy schedule. **β1=0 is the whole point**: dropping the
fast-momentum `m1` tensor recovers AdamW8bit memory parity (2.03 B/param). Any full `m1` buffer
(even 8-bit ≈ +1 B/param) erodes the win — so a fix that needs `m1` is a last resort, and a
*stateless* fix is strongly preferred.

**Bar to beat:** AdamW8bit baseline (wandb `s4g72u7i`) trains the full schedule to target density
0.25 stably, post-route VAL **ppl ~31–32**.

## 2. The stack (what makes MORPH different from the AdEMAMix paper's setting)

- **276.5M looped transformer**: 3 prelude blocks + a **core of 6 blocks applied T× per step**
  (Parcae diagonal injection, per-sequence Poisson depth T~mean 6, truncated BPTT depth 4) + 3 coda.
  Weight-shared core ⇒ a per-step weight change is applied **T times** in the forward.
- **Ternary STE QAT backbone**: 40 weight matrices quantized to {−1,0,+1}×scale via a `TernarySTE`
  **parametrization** on `.weight` (symmetric, per-tensor, threshold 0.5·scale). 60.2M params (21.8%).
  int6 embeddings; HyperConnection-Cayley n=4 residual; CCA attention (not ternary).
- **Deploy schedule**: gradual prune from step 3000 → target density 0.25, carve→BCSR @29000,
  ReMoE route @30000, 35k steps total.
- **The AdEMAMix paper (arXiv:2409.03137) trains normal dense transformers** — no STE quantizer,
  no weight-shared loop. Its documented instabilities are slow-EMA (β3) warmup phenomena.

## 3. The journey (compressed, all measured)

1. **Plain β1=0 detonates.** Root cause (earlier, confirmed): β1=0 numerator is raw `g` → on
   noise-dominated coords the update is a unit-variance random walk (lost Adam noise-gating) →
   collective saturation → blow-up.
2. **Fix that works for steady-state — `m2`-SNR gate + per-coord update clamp** (both stateless,
   β1=0 memory intact): gate `= floor + (1−floor)·clamp(|m2|/denom / κ, 0, 1)`, κ=0.3 floor=0.1;
   `update_clip=5`. Result: **first β1=0 config to clear the no-prune 8k testbed (RC=0, zero
   detonation)**, and on the **deploy schedule it was stable AND quality-WINNING** — cleared the
   density-0.74 prune point (where a prior arm died), VAL **26.65 @ step 17000 (BELOW the AdamW
   ~31–32 bar)** at density ~0.45.
3. **Then it detonates at density 0.41 / step ~19400** (same step/density a prior arm died). VAL
   29.8@19000 → ppl 145@19200 → 754k@19400. **Recoverable**, not permanent (754k→63k declining).

## 4. CONFIRMED MECHANISM — STE-cusp-vault (frame-by-frame)

Built forward-side instrumentation (`train.py diag_forward_norms`): per-block residual-stream norm
(`FWDNORM`) + backbone ternary {−1,0,+1} state-flip count via the TernarySTE parametrization
(`TERNFLIP`, per-section). Resumed the healthy pre-spike ckpt (`step_17500`, density 0.45) and
replayed through the vault. Captured the event:

| step  | TERNFLIP total | core flips | FWDNORM core |
|-------|---------------:|-----------:|-------------:|
| 18034 | 11,076         | 4,147      | 7.4e3 (healthy) |
| **18036** | **206,052** | **102,873** | **3.57e7** ← leap |
| 18038 | 11,616         | 4,991      | 7.3e3 (recovered) |

- **Baseline ternary churn ~10–14k flips/step** (normal QAT wiggle near ±0.5·scale thresholds),
  core residual norm steady ~6–8e3.
- **Vault = a single step where ~206k codes flip (≈20× baseline), ~103k of them in the looped
  core**, coincident with the core residual stream leaping 7.4e3 → 3.57e7. Worst single module is a
  SwiGLU **`gate_up`** projection (`core.1.gate_up` ~17.6k flips). Next step it reverts → recovers.
- **`final_norm` (RMSNorm before LM head) stays pinned ~4500 even when core=1e16** → it divides out
  the magnitude blow-up → logits/loss are shielded → loss *lags* the forward explosion and the spike
  is **recoverable**. Sustained divergence only after a cluster of vaults (18080→18082→18088: 3.5e6
  → 9e15 → inf) sticks.
- **Optimizer state is CALM at the vault**: per-coord/per-tensor optimizer-update magnitudes are
  bounded (maxU 5–11, relStep ~1e-5, zero ν-collapse). The detonation is **invisible to
  optimizer-state diagnostics** — it lives in the *discrete forward* (ternary code flips), which the
  optimizer-update magnitude can't see.

**Interpretation:** the ternary STE makes the *effective* loss surface piecewise with cusps/ledges
at the quantization thresholds. β1=0's raw-gradient step occasionally **vaults a cusp** → a mass of
core ternary codes flip together → the realized core weight matrix changes discontinuously → the
**weight-shared core amplifies it T×** → residual-stream explosion → RMSNorm masks magnitude → mostly
recovers, occasionally sticks. AdamW (β1=0.9) survives the identical schedule.

## 5. Fixes tried (both on top of the working gate+clip base, resume step_17500 → replay through vault)

- **ARM D — numerator momentum** (`num_beta1=0.9`, 8-bit m1, ~+1 B/param): smooth the step direction
  (the direct β1 analog). **FAILED** — detonated @~18800. Momentum did not prevent the vault.
  → This makes our optimizer ≈ the paper's AdEMAMix (fast m1 + slow α·m2). It still detonates ⇒
  **we are NOT hitting the paper's β1/β3 instabilities; the cusp-vault is our-stack-specific.**
- **ARM A — flip-rate clamp** (`flip_clamp_kappa=0.03`, stateless): per-tensor, if fraction of codes
  that would flip under `p−lr·u` exceeds κ, scale that tensor's update by κ/frac. **INCONCLUSIVE —
  the clamp never fired**: measured vault worst-module flip *fraction* is only **0.57%**
  (`core.1.gate_up`: 17.6k/3.1M), baseline ~0.03%; κ=3% was ~50× too loose. Idea untested, not falsified.

Both arms: unit-gated (`ignore/verify_cusp_fixes.py` → PASS 8/8, incl. defaults-off byte-identical
to baseline). Diagnostic replays' exact vault step is FP-nondeterministic (seen 18036 / 18800 / 19400).

(Footnote: the one thing surviving-AdamW lacks vs ARM D is the **slow α·m2 term** (α=8, β3=0.999).
The diag showed raw-g, not α·m2, dominant at the vault — so α·m2 looks like a passenger, but dialing
α/β3 toward AdamW is an untested lever if needed.)

## 6. Candidate directions (for review)

The fix should target the **cusp/loop**, not optimizer momentum:

- **(i) Tighter / global flip-clamp** — replace per-module κ with a global flip-spike detector (total
  flips spike a clean ~20×: 11k→206k, far less fragile than the 0.6% per-module fraction), or κ≈0.002.
  Stateless. Risk: narrow window (0.03%→0.6% per-module), may be brittle.
- **(ii) Anneal / slow the ternary threshold+scale** — **hypothesis to verify first (cheap, no GPU):
  does the per-tensor STE scale = mean|w| RECOMPUTE every step?** If so, coherent weight drift moves
  the scale → a whole tensor's codes re-threshold *at once* = the mass-flip vault. Fix = freeze or
  slow-EMA the scale so cusps drift slowly. Possibly the root lever; stateless.
- **(iii) Bound the looped-core gain** — spectral-norm / residual-norm clamp on the core block so a
  discrete flip can't amplify T× into an explosion (makes vaults non-catastrophic regardless of
  trigger). Architectural; needs a parity check. (Note RMSNorm already half-does this — it masks
  magnitude but not the directional corruption that eventually sticks.)
- **(iv) Stop**: ship AdamW8bit (proven ~31–32 to density 0.25); pocket the confirmed mechanism +
  the gate+clip "stable & quality-winning to density 0.45" result; revisit the β1=0 memory win later.

## 7. Specific questions for GPT

1. Is the STE-cusp-vault diagnosis sound given the data (single-step mass core flip ⇄ core-norm leap,
   calm optimizer, RMSNorm-masked recoverability, β1=0.9 not curing it)? Any alternative read?
2. Of (i)–(iii), which is most likely to cure it WITHOUT eroding the β1=0 memory win — and is the
   "ternary scale recomputes each step → coherent mass re-threshold" hypothesis the real root lever?
3. Is there a known-good technique for **STE QAT stability under a weight-shared/looped forward** we're
   missing (e.g. scale EMA, threshold annealing, stochastic rounding, gradient-scaling near thresholds,
   per-step weight-change caps in *weight* space rather than Adam-update space)?
4. Is chasing β1=0 worth it vs accepting AdamW8bit, given the win is memory-only (~1 B/param at 30B)
   and the instability is fundamental to the ternary×loop interaction?

## 8. Key files (UNCOMMITTED)
- `morph/training/ademamix_b1zero.py` — optimizer; gate+clip + the two new knobs (`num_beta1`,
  `flip_clamp_kappa`), de-fused `_foreach` step.
- `morph/training/train.py` — `diag_forward_norms` (FWDNORM/TERNFLIP), resume logic (wandb-id sidecar).
- `morph/training/optimizer.py`, `morph/configs/base.yaml` — plumbing.
- `ignore/verify_cusp_fixes.py` (unit gate), `ignore/b1zero_diag_ste_cusp_replay.{sh,optlog}` (the
  frame-by-frame capture), `ignore/b1zero_fix_arm{D_mom,A_flipclamp}.sh`.
- Checkpoints: `checkpoints/morph/b1zero_gate_k0.3_clip5_deploy_35k/step_{2500..22500}.pt`
  (`step_17500` = healthy pre-vault, density ~0.45; the deploy run detonated @19400).
