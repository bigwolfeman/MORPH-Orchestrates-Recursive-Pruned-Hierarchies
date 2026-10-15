/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import Mathlib.Topology.MetricSpace.HausdorffDistance
import Mathlib.Analysis.Normed.Module.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# PL1: the bounded-step depth law

## Model

A pre-norm residual block moves its stream by a bounded amount whatever the stream holds. So a
loop pass is modelled as a step of length at most `δ` in a metric space `X` (the loop state).
The reader (the coda) accepts a set `G` of good loop states; its loss at depth `T` is a
monotone function `φ` of the distance `infDist (x T) G`. The loop's WORK is the distance
`W = infDist (x 0) G` from its entry state to the good set. A loop is any trajectory with
steps at most `δ`; time-homogeneous (weight-shared) maps are a special case.

## Main results

* **`rem_le_infDist`**: after `T` passes, every bounded-step trajectory is still at least
  `rem W δ T = max (W - T δ) 0` from the good set.
* **`greedy_infDist`**: one time-homogeneous map (step to the target at full speed, then
  stay) attains that bound at EVERY depth at once.
* **`minimizer_forced`**: so for any depth law, every minimizer of the depth-averaged loss
  has the greedy map's per-depth loss at every depth the law draws. The K-curve at the
  optimum is forced: it is a function of `W` and `δ` alone.
* **`kcurve_pos_iff`**: the forced K-curve between depths `a < b` is positive iff
  `W > a δ`. A job that fits in one pass (`W ≤ δ`) gives a flat curve at the optimum.
* **`tokens_kcurve_pos_iff`**: summed over tokens, the curve is positive iff SOME token's
  job is longer than `a` passes.
* **`skip_work_eq_entry_work`**: a coda that reads `h + u` (a skip around the loop) gives the
  loop the work of a series loop entered at `h`: an additive skip IS a prelude entry.
* **`kcurve_mono_sq`**, **`rem_mono`**: more work means a higher curve AND a loss at least
  as high at every depth. Withholding from the skip buys K-curve with a tax at every depth.
* **`rem_thin`**, **`thin_core_earns`**: a core `m` times thinner per pass reaches, in
  `m T` passes, what the full core reaches in `T`; a job that the full core does in one
  pass, but not the thin one, earns depth on the thin core only.
* **`nrm_smul_pos`**, **`normalized_build_flat`**: a normalized read (RMSNorm) erases a
  build that only grows the scale. A reader of `nrm (x T)` sees the same input at every depth.
-/

namespace ParallelLoop

open Metric

/-! ## The remaining distance -/

/-- Distance still to go after `T` passes of length at most `δ` from work `W`. -/
noncomputable def rem (W δ : ℝ) (T : ℕ) : ℝ := max (W - T * δ) 0

theorem rem_nonneg (W δ : ℝ) (T : ℕ) : 0 ≤ rem W δ T := le_max_right _ _

theorem rem_zero {W : ℝ} (hW : 0 ≤ W) (δ : ℝ) : rem W δ 0 = W := by
  simp [rem, hW]

theorem rem_succ {δ : ℝ} (hδ : 0 ≤ δ) (W : ℝ) (T : ℕ) :
    rem W δ (T + 1) = max (rem W δ T - δ) 0 := by
  simp only [rem]
  push_cast
  rcases le_total 0 (W - T * δ) with h | h
  · rw [max_eq_left h]
    congr 1
    ring
  · rw [max_eq_right h, max_eq_right (by linarith), max_eq_right (by nlinarith)]

/-- More work never leaves less to do. -/
theorem rem_mono {W W' : ℝ} (hW : W ≤ W') (δ : ℝ) (T : ℕ) : rem W δ T ≤ rem W' δ T :=
  max_le_max (by linarith) le_rfl

/-- More passes never leave more to do. -/
theorem rem_anti {δ : ℝ} (hδ : 0 ≤ δ) (W : ℝ) {a b : ℕ} (hab : a ≤ b) :
    rem W δ b ≤ rem W δ a := by
  have : (a : ℝ) * δ ≤ b * δ := mul_le_mul_of_nonneg_right (by exact_mod_cast hab) hδ
  exact max_le_max (by linarith) le_rfl

/-! ## The lower bound: no loop does better than `rem` -/

section Lower

variable {X : Type*} [MetricSpace X]

/-- Each pass of length at most `δ` lowers the distance to any set by at most `δ`. -/
theorem infDist_sub_le (G : Set X) (x : ℕ → X) {δ : ℝ}
    (hstep : ∀ t, dist (x (t + 1)) (x t) ≤ δ) (T : ℕ) :
    infDist (x 0) G - T * δ ≤ infDist (x T) G := by
  induction T with
  | zero => simp
  | succ n ih =>
    have h1 := infDist_le_infDist_add_dist (x := x n) (y := x (n + 1)) (s := G)
    have h2 := hstep n
    rw [dist_comm] at h2
    push_cast
    linarith

/-- **The depth law, lower half.** After `T` passes the loop is at least `rem W δ T` from
the good set, `W` its entry distance. -/
theorem rem_le_infDist (G : Set X) (x : ℕ → X) {δ : ℝ}
    (hstep : ∀ t, dist (x (t + 1)) (x t) ≤ δ) (T : ℕ) :
    rem (infDist (x 0) G) δ T ≤ infDist (x T) G :=
  max_le (infDist_sub_le G x hstep T) infDist_nonneg

end Lower

/-! ## The upper bound: one weight-shared map attains `rem` at every depth -/

section Greedy

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Step toward `g` by `δ`, or land on `g` when it is within `δ`. One map, every pass. -/
noncomputable def greedy (g : E) (δ : ℝ) (x : E) : E :=
  if ‖g - x‖ ≤ δ then g else x + (δ / ‖g - x‖) • (g - x)

theorem greedy_step (g : E) {δ : ℝ} (hδ : 0 ≤ δ) (x : E) : ‖greedy g δ x - x‖ ≤ δ := by
  unfold greedy
  split_ifs with h
  · simpa using h
  · push Not at h
    have hr : 0 < ‖g - x‖ := lt_of_le_of_lt hδ h
    rw [add_sub_cancel_left, norm_smul, Real.norm_eq_abs, abs_of_nonneg (div_nonneg hδ hr.le),
      div_mul_cancel₀ _ hr.ne']

theorem greedy_dist (g : E) {δ : ℝ} (hδ : 0 ≤ δ) (x : E) :
    ‖g - greedy g δ x‖ = max (‖g - x‖ - δ) 0 := by
  unfold greedy
  split_ifs with h
  · rw [sub_self, norm_zero, max_eq_right (by linarith)]
  · push Not at h
    have hr : 0 < ‖g - x‖ := lt_of_le_of_lt hδ h
    have hq : δ / ‖g - x‖ ≤ 1 := (div_le_one hr).2 h.le
    have hv : g - (x + (δ / ‖g - x‖) • (g - x)) = (1 - δ / ‖g - x‖) • (g - x) := by
      rw [sub_smul, one_smul]
      abel
    rw [hv, norm_smul, Real.norm_eq_abs, abs_of_nonneg (by linarith), max_eq_left (by linarith),
      sub_mul, one_mul, div_mul_cancel₀ _ hr.ne']

/-- The greedy map's iterates are exactly `rem` from `g` at every depth. -/
theorem greedy_iterate_dist (g x₀ : E) {δ : ℝ} (hδ : 0 ≤ δ) (T : ℕ) :
    ‖g - (greedy g δ)^[T] x₀‖ = rem ‖g - x₀‖ δ T := by
  induction T with
  | zero => simp [rem_zero (norm_nonneg _)]
  | succ n ih =>
    rw [Function.iterate_succ_apply', greedy_dist g hδ, ih, rem_succ hδ]

/-- **The depth law, upper half.** If `g ∈ G` is a nearest good state to the entry `x₀`, the
greedy map toward `g` is exactly `rem` from `G` after every number of passes. -/
theorem greedy_infDist (G : Set E) {g : E} (hg : g ∈ G) (x₀ : E)
    (hnear : ‖g - x₀‖ = infDist x₀ G) {δ : ℝ} (hδ : 0 ≤ δ) (T : ℕ) :
    infDist ((greedy g δ)^[T] x₀) G = rem (infDist x₀ G) δ T := by
  apply le_antisymm
  · calc infDist ((greedy g δ)^[T] x₀) G ≤ dist ((greedy g δ)^[T] x₀) g :=
          infDist_le_dist_of_mem hg
      _ = ‖g - (greedy g δ)^[T] x₀‖ := by rw [dist_eq_norm, norm_sub_rev]
      _ = rem (infDist x₀ G) δ T := by rw [greedy_iterate_dist g x₀ hδ, hnear]
  · have hstep : ∀ t, dist ((greedy g δ)^[t + 1] x₀) ((greedy g δ)^[t] x₀) ≤ δ := by
      intro t
      rw [Function.iterate_succ_apply', dist_eq_norm]
      exact greedy_step g hδ _
    have := rem_le_infDist G (fun t => (greedy g δ)^[t] x₀) hstep T
    simpa using this

end Greedy

/-! ## Every minimizer of a depth-averaged loss has the forced K-curve -/

section Forced

variable {X : Type*} [MetricSpace X]

/-- The depth-averaged loss of a trajectory: depth law `π` on the depths `S`, reader loss `φ`
of the distance to the good set. -/
noncomputable def avgLoss (S : Finset ℕ) (π : ℕ → ℝ) (φ : ℝ → ℝ) (G : Set X) (x : ℕ → X) : ℝ :=
  ∑ T ∈ S, π T * φ (infDist (x T) G)

/-- The loss floor: the same average with `rem` in place of the distance. -/
noncomputable def floorLoss (S : Finset ℕ) (π : ℕ → ℝ) (φ : ℝ → ℝ) (W δ : ℝ) : ℝ :=
  ∑ T ∈ S, π T * φ (rem W δ T)

/-- No bounded-step loop beats the floor. -/
theorem floor_le_avgLoss (S : Finset ℕ) {π : ℕ → ℝ} (hπ : ∀ T ∈ S, 0 ≤ π T) {φ : ℝ → ℝ}
    (hφ : MonotoneOn φ (Set.Ici 0)) (G : Set X) (x : ℕ → X) {δ : ℝ}
    (hstep : ∀ t, dist (x (t + 1)) (x t) ≤ δ) :
    floorLoss S π φ (infDist (x 0) G) δ ≤ avgLoss S π φ G x := by
  apply Finset.sum_le_sum
  intro T hT
  exact mul_le_mul_of_nonneg_left
    (hφ (rem_nonneg _ _ _) infDist_nonneg (rem_le_infDist G x hstep T)) (hπ T hT)

/-- **The K-curve at the optimum is forced.** If a bounded-step loop's averaged loss equals
the floor (it is a minimizer, since the greedy map attains the floor: `greedy_infDist`), then
at every drawn depth with positive weight its loss is the floor's per-depth loss. -/
theorem minimizer_forced (S : Finset ℕ) {π : ℕ → ℝ} (hπ : ∀ T ∈ S, 0 ≤ π T) {φ : ℝ → ℝ}
    (hφ : MonotoneOn φ (Set.Ici 0)) (G : Set X) (x : ℕ → X) {δ : ℝ}
    (hstep : ∀ t, dist (x (t + 1)) (x t) ≤ δ)
    (hmin : avgLoss S π φ G x = floorLoss S π φ (infDist (x 0) G) δ)
    {T : ℕ} (hT : T ∈ S) (hpos : 0 < π T) :
    φ (infDist (x T) G) = φ (rem (infDist (x 0) G) δ T) := by
  set W := infDist (x 0) G
  have hterm : ∀ t ∈ S, 0 ≤ π t * φ (infDist (x t) G) - π t * φ (rem W δ t) := by
    intro t ht
    rw [← mul_sub]
    exact mul_nonneg (hπ t ht)
      (sub_nonneg.2 (hφ (rem_nonneg _ _ _) infDist_nonneg (rem_le_infDist G x hstep t)))
  have hsum : ∑ t ∈ S, (π t * φ (infDist (x t) G) - π t * φ (rem W δ t)) = 0 := by
    rw [Finset.sum_sub_distrib]
    unfold avgLoss floorLoss at hmin
    linarith
  have h0 := (Finset.sum_eq_zero_iff_of_nonneg hterm).1 hsum T hT
  rw [← mul_sub] at h0
  have := (mul_eq_zero.1 h0).resolve_left hpos.ne'
  linarith

end Forced

/-! ## When the forced curve is flat -/

/-- The remaining distance is the same at depths `a < b` iff the job fits in `a` passes. -/
theorem rem_eq_iff {W δ : ℝ} (hδ : 0 < δ) {a b : ℕ} (hab : a < b) :
    rem W δ a = rem W δ b ↔ W ≤ a * δ := by
  have hab' : (a : ℝ) * δ < b * δ := mul_lt_mul_of_pos_right (by exact_mod_cast hab) hδ
  constructor
  · intro h
    by_contra hc
    push Not at hc
    have h1 : rem W δ a = W - a * δ := max_eq_left (by linarith)
    have h2 : rem W δ b < W - a * δ := max_lt (by linarith) (by linarith)
    linarith
  · intro h
    have h1 : rem W δ a = 0 := max_eq_right (by linarith)
    have h2 : rem W δ b = 0 := max_eq_right (by linarith)
    rw [h1, h2]

/-- **Flat iff one-pass.** For a strictly monotone reader loss, the forced K-curve between
depths `a < b` is positive iff the work exceeds `a` passes, and zero otherwise. With `a = 1`:
a loop whose job fits in one pass has a flat curve at every optimum. -/
theorem kcurve_pos_iff {W δ : ℝ} (hδ : 0 < δ) {a b : ℕ} (hab : a < b)
    {φ : ℝ → ℝ} (hφ : StrictMonoOn φ (Set.Ici 0)) :
    0 < φ (rem W δ a) - φ (rem W δ b) ↔ a * δ < W := by
  have hle : rem W δ b ≤ rem W δ a := rem_anti hδ.le W hab.le
  constructor
  · intro h
    by_contra hc
    push Not at hc
    rw [(rem_eq_iff hδ hab).2 hc, sub_self] at h
    exact lt_irrefl _ h
  · intro h
    have hne : rem W δ a ≠ rem W δ b := fun he => by
      have := (rem_eq_iff hδ hab).1 he
      linarith
    have hlt : rem W δ b < rem W δ a := lt_of_le_of_ne hle (Ne.symm hne)
    exact sub_pos.2 (hφ (rem_nonneg _ _ _) (rem_nonneg _ _ _) hlt)

/-- The forced K-curve is never negative. -/
theorem kcurve_nonneg {W δ : ℝ} (hδ : 0 ≤ δ) {a b : ℕ} (hab : a ≤ b)
    {φ : ℝ → ℝ} (hφ : MonotoneOn φ (Set.Ici 0)) :
    0 ≤ φ (rem W δ a) - φ (rem W δ b) :=
  sub_nonneg.2 (hφ (rem_nonneg _ _ _) (rem_nonneg _ _ _) (rem_anti hδ W hab))

/-- **Per-token aggregate.** Tokens `i ∈ s` with weights `q i > 0` and works `W i`: the
summed forced curve is positive iff SOME token's job is longer than `a` passes. -/
theorem tokens_kcurve_pos_iff {ι : Type*} (s : Finset ι) {q W : ι → ℝ}
    (hq : ∀ i ∈ s, 0 < q i) {δ : ℝ} (hδ : 0 < δ)
    {a b : ℕ} (hab : a < b) {φ : ℝ → ℝ} (hφ : StrictMonoOn φ (Set.Ici 0)) :
    0 < ∑ i ∈ s, q i * (φ (rem (W i) δ a) - φ (rem (W i) δ b)) ↔ ∃ i ∈ s, a * δ < W i := by
  have hnn : ∀ i ∈ s, 0 ≤ q i * (φ (rem (W i) δ a) - φ (rem (W i) δ b)) := fun i hi =>
    mul_nonneg (hq i hi).le (kcurve_nonneg hδ.le hab.le hφ.monotoneOn)
  constructor
  · intro h
    by_contra hc
    push Not at hc
    have : ∑ i ∈ s, q i * (φ (rem (W i) δ a) - φ (rem (W i) δ b)) = 0 := by
      apply Finset.sum_eq_zero
      intro i hi
      have hz : φ (rem (W i) δ a) - φ (rem (W i) δ b) = 0 := by
        rw [(rem_eq_iff hδ hab).2 (hc i hi), sub_self]
      rw [hz, mul_zero]
    linarith
  · rintro ⟨i, hi, hwi⟩
    exact Finset.sum_pos' hnn
      ⟨i, hi, mul_pos (hq i hi) ((kcurve_pos_iff hδ hab hφ).2 hwi)⟩

/-! ## The skip around the loop -/

section Skip

variable {E : Type*} [NormedAddCommGroup E]

/-- **An additive skip is a prelude entry.** The coda reads `u + h`; its good set for the
SUM is `G`, so its good set for the loop's output is `Gs` with `G = (· + h) '' Gs`. The work
the skip leaves the loop from entry `u` equals the work of a series loop entered at `u + h`.
With the loop entered at `0` (Parcae's near-zero `like-init` state), the skip arm's job is the
series job from the prelude state `h`. -/
theorem skip_work_eq_entry_work (Gs : Set E) (h u : E) :
    infDist (u + h) ((fun v => v + h) '' Gs) = infDist u Gs :=
  infDist_image (Isometry.of_dist_eq fun x y => dist_add_right x y h)

end Skip

/-! ## Withholding and thinning -/

/-- **Withholding is a tax at every depth and a gift to the curve.** Under squared reader loss,
more work gives a K-curve at least as high (and `rem_mono` gives a loss at least as high at
every depth). A skip that delivers less leaves more work, so it raises K1-K6 only by raising
the loss. -/
theorem kcurve_mono_sq {W W' δ : ℝ} (hWW : W ≤ W') (hδ : 0 ≤ δ) {a b : ℕ} (hab : a ≤ b) :
    rem W δ a ^ 2 - rem W δ b ^ 2 ≤ rem W' δ a ^ 2 - rem W' δ b ^ 2 := by
  have hAB : (a : ℝ) * δ ≤ b * δ := mul_le_mul_of_nonneg_right (by exact_mod_cast hab) hδ
  set A := (a : ℝ) * δ
  set B := (b : ℝ) * δ
  simp only [rem, max_def]
  split_ifs <;> nlinarith

/-- **Equal travel, equal loss.** A core whose per-pass step is `δ / m` reaches in `m T`
passes exactly what the full core reaches in `T`. -/
theorem rem_thin (W δ : ℝ) {m : ℕ} (hm : 0 < m) (T : ℕ) :
    rem W (δ / m) (m * T) = rem W δ T := by
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  simp only [rem]
  push_cast
  congr 2
  field_simp

/-- **A thinner core earns where the full core is flat.** A token whose job fits one full pass
(`W ≤ δ`) but not one thin pass (`δ / m < W`) has a flat forced curve on the full core and a
positive one on the thin core. -/
theorem thin_core_earns {W δ : ℝ} (hδ : 0 < δ) {m : ℕ} (hm : 0 < m)
    (hfit : W ≤ δ) (hthin : δ / m < W) {b : ℕ} (hb : 1 < b)
    {φ : ℝ → ℝ} (hφ : StrictMonoOn φ (Set.Ici 0)) :
    φ (rem W δ 1) - φ (rem W δ b) = 0 ∧ 0 < φ (rem W (δ / m) 1) - φ (rem W (δ / m) b) := by
  have hδm : 0 < δ / m := div_pos hδ (by exact_mod_cast hm)
  refine ⟨?_, (kcurve_pos_iff hδm hb hφ).2 (by simpa using hthin)⟩
  rw [(rem_eq_iff hδ hb).2 (by simpa using hfit), sub_self]

/-! ## A normalized read erases a scale-only build -/

section Norm

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The direction of a vector (`0` for `0`). RMSNorm with a learned gain is a fixed linear map of
this times `√dim`, so it inherits the scale invariance. -/
noncomputable def nrm (v : E) : E := ‖v‖⁻¹ • v

theorem nrm_smul_pos {s : ℝ} (hs : 0 < s) (v : E) : nrm (s • v) = nrm v := by
  unfold nrm
  rw [norm_smul, Real.norm_eq_abs, abs_of_pos hs, smul_smul, mul_inv, mul_comm s⁻¹, mul_assoc,
    inv_mul_cancel₀ hs.ne', mul_one]

/-- **A scale-only build reads flat through a norm.** If the loop state after `T ≥ 1` passes is
`s T • v` with `s T > 0` (Parcae's injection with a uniform decay and a fixed block output:
`s T = (1 - a^T) / (1 - a)`), then any reader of `(skip, nrm (x T))` gets the same input,
hence the same loss, at every depth. -/
theorem normalized_build_flat {Y : Type*} (L : Y → E → ℝ) (y : Y) (s : ℕ → ℝ) (v : E)
    {a b : ℕ} (ha : 0 < s a) (hb : 0 < s b) :
    L y (nrm (s a • v)) = L y (nrm (s b • v)) := by
  rw [nrm_smul_pos ha, nrm_smul_pos hb]

end Norm

end ParallelLoop
