/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
# PL2: what a skip does to a loop channel under gradient flow (linear-Gaussian model)

## Model

One token feature `z` with `E z² = 1`. A depth law draws `T` with weight `π T`. The loop output
after `T` passes is `s T · β · z`: Parcae's injection with a uniform decay builds the state's
SCALE, `s T = (1 - a^T)/(1 - a)`, and `β` is the loop's write. The reader puts weight `w` on the
loop output (so the loop's product weight is `c = w β`) and, in the PARALLEL topology, weight
`v` on a skip that carries `z` itself. Target `κ z`. Squared loss, averaged over the depth law:

* parallel: `Jpar κ v c = ∑ π T (κ - v - c s T)²`;
* series (no skip): `Jser κ c = ∑ π T (κ - c s T)²`.

`S1 = E s`, `S2 = E s²`, `var = S2 - S1²` (positive whenever the drawn depths give different
scales). The gradient components `gv`, `gs` below are minus one half of the partial derivatives
(`hasDerivAt_*` prove it), so `(v̇, ẇ, β̇) = (gv, β gs, w gs)` is gradient flow.

## Main results

* **`par_minimizer`**: the only optimum of the parallel reader puts ZERO weight on the loop.
  A loop output that varies with the drawn depth is pure noise beside a skip that carries the
  same feature.
* **`par_stationary_prunes`**: stronger, for the factorized loop `c = w β`: EVERY stationary
  point of gradient flow has `w β = 0`. No limit of training uses the loop.
* **`flow_prune_slice`**, **`flow_conserve`**: once the skip has fit (`gv = 0`) the flow moves
  the product weight as `ċ = -(w² + β²) var c`: the loop channel decays.
* **`scaled_skip_grad_prunes`**, **`Jpar_fixed_eq_Jser`**: gradient routing that scales the
  skip's backward has the same stationary set; freezing the skip makes the loop a series loop
  on what the skip does not carry.
* **`gs_zero_par`**, **`gs_zero_ser`**: at `c = 0` the loop's gradient is `S1 gv` in parallel
  (zero once the skip fits: the loop cannot start) and `κ S1` in series (it always starts).
* **`ser_minimizer`**, **`ser_loss_at_opt`**, **`ser_kcurve_pos`**: the series optimum keeps
  the loop (`c* = κ S1 / S2`), its per-depth loss is `κ² (1 - q s T)²` with `q = S1 / S2`, and
  its K-curve is positive. It is never below the parallel optimum's loss (zero) at any depth:
  depth dependence without depth value.
* **`par2_minimizer`**, **`unique_share`**: with a second feature `φ` that only the loop can
  build, the parallel optimum prunes the duplicate channel, keeps the unique one at the series
  weight, and its K-curve is the series K-curve times `η² / (κ² + η²)`, the unique share.
* **`drop_minimizer`**, **`drop_eval_loss`**, **`drop_tax`**: dropping the skip with
  probability `p` at train keeps a loop weight `c_p = p κ S1 / ((1-p) var + p S2)`; with the skip
  on at eval the per-depth loss is `c_p² (S1 - s T)²` and the averaged loss `c_p² var > 0`. The
  K-curve it buys is a tax: the plain parallel optimum is zero at every depth.
-/

namespace ParallelLoop

open Finset

/-- A depth law on a finite set of depths: weights `π` and the loop output's scale `s`. -/
structure DepthLaw (ι : Type*) [Fintype ι] where
  π : ι → ℝ
  s : ι → ℝ
  nonneg : ∀ i, 0 ≤ π i
  sum_one : ∑ i, π i = 1

namespace DepthLaw

variable {ι : Type*} [Fintype ι] (D : DepthLaw ι)

/-- Mean scale `E s`. -/
noncomputable def S1 : ℝ := ∑ i, D.π i * D.s i

/-- Second moment `E s²`. -/
noncomputable def S2 : ℝ := ∑ i, D.π i * D.s i ^ 2

/-- Variance of the scale over the depth law. -/
noncomputable def var : ℝ := D.S2 - D.S1 ^ 2

theorem sum_lin (a b : ℝ) : ∑ i, D.π i * (a + b * D.s i) = a + b * D.S1 := by
  have h : ∀ i, D.π i * (a + b * D.s i) = a * D.π i + b * (D.π i * D.s i) := fun i => by ring
  simp only [S1]
  simp_rw [h, Finset.sum_add_distrib, ← Finset.mul_sum, D.sum_one, mul_one]

theorem sum_s_lin (a b : ℝ) :
    ∑ i, D.π i * D.s i * (a + b * D.s i) = a * D.S1 + b * D.S2 := by
  have h : ∀ i, D.π i * D.s i * (a + b * D.s i) = a * (D.π i * D.s i) + b * (D.π i * D.s i ^ 2) :=
    fun i => by ring
  simp only [S1, S2]
  simp_rw [h, Finset.sum_add_distrib, ← Finset.mul_sum]

theorem sum_sq_affine (a b : ℝ) :
    ∑ i, D.π i * (a + b * D.s i) ^ 2 = a ^ 2 + 2 * a * b * D.S1 + b ^ 2 * D.S2 := by
  have h : ∀ i, D.π i * (a + b * D.s i) ^ 2 =
      a ^ 2 * D.π i + (2 * a * b) * (D.π i * D.s i) + b ^ 2 * (D.π i * D.s i ^ 2) :=
    fun i => by ring
  simp only [S1, S2]
  simp_rw [h, Finset.sum_add_distrib, ← Finset.mul_sum, D.sum_one, mul_one]

theorem var_eq_sum : D.var = ∑ i, D.π i * (D.s i - D.S1) ^ 2 := by
  have h : ∀ i, D.π i * (D.s i - D.S1) ^ 2 = D.π i * (-D.S1 + 1 * D.s i) ^ 2 := fun i => by ring
  simp_rw [h, D.sum_sq_affine]
  unfold var
  ring

theorem var_nonneg : 0 ≤ D.var := by
  rw [var_eq_sum]
  exact Finset.sum_nonneg fun i _ => mul_nonneg (D.nonneg i) (sq_nonneg _)

/-! ## The parallel topology -/

/-- Parallel loss: skip weight `v`, loop product weight `c`, target `κ`. -/
noncomputable def Jpar (κ v c : ℝ) : ℝ := ∑ i, D.π i * (κ - v - c * D.s i) ^ 2

/-- Bias-variance split of the parallel loss. -/
theorem Jpar_eq (κ v c : ℝ) : D.Jpar κ v c = (κ - v - c * D.S1) ^ 2 + c ^ 2 * D.var := by
  have h : ∀ i, D.π i * (κ - v - c * D.s i) ^ 2 = D.π i * ((κ - v) + (-c) * D.s i) ^ 2 :=
    fun i => by ring
  unfold Jpar
  simp_rw [h, D.sum_sq_affine]
  unfold var
  ring

theorem Jpar_nonneg (κ v c : ℝ) : 0 ≤ D.Jpar κ v c := by
  rw [Jpar_eq]
  exact add_nonneg (sq_nonneg _) (mul_nonneg (sq_nonneg _) D.var_nonneg)

theorem Jpar_skip (κ : ℝ) : D.Jpar κ κ 0 = 0 := by
  rw [Jpar_eq]
  ring

/-- **The parallel optimum prunes the loop.** If the drawn depths give the loop output
different scales (`var > 0`), the only minimizer of the parallel loss is `v = κ`, `c = 0`. -/
theorem par_minimizer {κ v c : ℝ} (hvar : 0 < D.var)
    (hmin : ∀ v' c', D.Jpar κ v c ≤ D.Jpar κ v' c') : v = κ ∧ c = 0 := by
  have h0 := hmin κ 0
  rw [Jpar_skip, Jpar_eq] at h0
  have h1 := sq_nonneg (κ - v - c * D.S1)
  have h2 : 0 ≤ c ^ 2 * D.var := mul_nonneg (sq_nonneg _) D.var_nonneg
  have hc2 : c ^ 2 * D.var = 0 := by linarith
  have hc : c = 0 := by
    have : c ^ 2 = 0 := (mul_eq_zero.1 hc2).resolve_right hvar.ne'
    exact pow_eq_zero_iff (n := 2) (by norm_num) |>.1 this
  have hv : (κ - v - c * D.S1) ^ 2 = 0 := by linarith
  have := pow_eq_zero_iff (n := 2) (by norm_num) |>.1 hv
  rw [hc] at this
  exact ⟨by linarith, hc⟩

/-- Minus half the `v`-derivative of the parallel loss. -/
noncomputable def gv (κ v c : ℝ) : ℝ := ∑ i, D.π i * (κ - v - c * D.s i)

/-- Minus half the `c`-derivative of the parallel loss (the loop channel's raw gradient). -/
noncomputable def gs (κ v c : ℝ) : ℝ := ∑ i, D.π i * D.s i * (κ - v - c * D.s i)

theorem gv_eq (κ v c : ℝ) : D.gv κ v c = κ - v - c * D.S1 := by
  have h : ∀ i, D.π i * (κ - v - c * D.s i) = D.π i * ((κ - v) + (-c) * D.s i) :=
    fun i => by ring
  unfold gv
  simp_rw [h, D.sum_lin]
  ring

theorem gs_eq (κ v c : ℝ) : D.gs κ v c = D.S1 * D.gv κ v c - c * D.var := by
  have h : ∀ i, D.π i * D.s i * (κ - v - c * D.s i) =
      D.π i * D.s i * ((κ - v) + (-c) * D.s i) := fun i => by ring
  unfold gs
  simp_rw [h, D.sum_s_lin]
  rw [gv_eq]
  unfold var
  ring

/-- `gv` is minus half the `v`-derivative of the factorized parallel loss. -/
theorem hasDerivAt_Jpar_v (κ c v : ℝ) :
    HasDerivAt (fun v => D.Jpar κ v c) (-2 * D.gv κ v c) v := by
  have ha : HasDerivAt (fun v : ℝ => κ - v - c * D.S1) (-1) v := by
    simpa using ((hasDerivAt_id' v).const_sub κ).sub_const (c * D.S1)
  have hb := (ha.fun_pow 2).add_const (c ^ 2 * D.var)
  have hf : (fun v => D.Jpar κ v c) = fun v => (κ - v - c * D.S1) ^ 2 + c ^ 2 * D.var :=
    funext fun v => D.Jpar_eq κ v c
  rw [hf]
  refine hb.congr_deriv ?_
  rw [gv_eq]
  push_cast
  ring

/-- `β gs` is minus half the `w`-derivative of the factorized parallel loss `Jpar κ v (w β)`. -/
theorem hasDerivAt_Jpar_w (κ v β w : ℝ) :
    HasDerivAt (fun w => D.Jpar κ v (w * β)) (-2 * β * D.gs κ v (w * β)) w := by
  have h1 : HasDerivAt (fun w : ℝ => w * β) β w := by
    simpa using (hasDerivAt_id' w).mul_const β
  have ha : HasDerivAt (fun w : ℝ => κ - v - w * β * D.S1) (-(β * D.S1)) w :=
    (h1.mul_const D.S1).const_sub (κ - v)
  have hb := (ha.fun_pow 2).add ((h1.fun_pow 2).mul_const D.var)
  have hf : (fun w => D.Jpar κ v (w * β)) =
      fun w => (κ - v - w * β * D.S1) ^ 2 + (w * β) ^ 2 * D.var :=
    funext fun w => D.Jpar_eq κ v (w * β)
  rw [hf]
  refine hb.congr_deriv ?_
  rw [gs_eq, gv_eq]
  push_cast
  ring

/-- `w gs` is minus half the `β`-derivative of the factorized parallel loss. -/
theorem hasDerivAt_Jpar_β (κ v w β : ℝ) :
    HasDerivAt (fun β => D.Jpar κ v (w * β)) (-2 * w * D.gs κ v (w * β)) β := by
  have h1 : HasDerivAt (fun β : ℝ => w * β) w β := by
    simpa using (hasDerivAt_id' β).const_mul w
  have ha : HasDerivAt (fun β : ℝ => κ - v - w * β * D.S1) (-(w * D.S1)) β :=
    (h1.mul_const D.S1).const_sub (κ - v)
  have hb := (ha.fun_pow 2).add ((h1.fun_pow 2).mul_const D.var)
  have hf : (fun β => D.Jpar κ v (w * β)) =
      fun β => (κ - v - w * β * D.S1) ^ 2 + (w * β) ^ 2 * D.var :=
    funext fun β => D.Jpar_eq κ v (w * β)
  rw [hf]
  refine hb.congr_deriv ?_
  rw [gs_eq, gv_eq]
  push_cast
  ring

/-- **Every stationary point of the parallel flow prunes the loop.** If the skip's gradient and
the write's gradient vanish (`gv = 0`, `β gs = 0`; by `hasDerivAt_*` these are the `v` and `w`
partial derivatives), then `w β = 0`. -/
theorem par_stationary_prunes {κ v w β : ℝ} (hvar : 0 < D.var)
    (hv : D.gv κ v (w * β) = 0) (hw : β * D.gs κ v (w * β) = 0) : w * β = 0 := by
  rw [gs_eq, hv, mul_zero, zero_sub] at hw
  have h : (w * β) ^ 2 * D.var = 0 := by linear_combination (-w) * hw
  have h2 : (w * β) ^ 2 = 0 := (mul_eq_zero.1 h).resolve_right hvar.ne'
  exact pow_eq_zero_iff (n := 2) (by norm_num) |>.1 h2

/-- **Once the skip has fit, the flow shrinks the loop.** On the slice `gv = 0`, the product
weight's time derivative under gradient flow, `ċ = ẇ β + w β̇ = (β gs) β + w (w gs)`, is
`-(w² + β²) var c`. -/
theorem flow_prune_slice {κ v w β : ℝ} (hv : D.gv κ v (w * β) = 0) :
    (β * D.gs κ v (w * β)) * β + w * (w * D.gs κ v (w * β)) =
      -((w ^ 2 + β ^ 2) * D.var * (w * β)) := by
  rw [gs_eq, hv]
  ring

/-- The squared product weight is non-increasing on that slice, strictly while `c ≠ 0`. -/
theorem flow_prune_lyapunov {κ v w β : ℝ} (hvar : 0 < D.var) (hv : D.gv κ v (w * β) = 0)
    (hc : w * β ≠ 0) :
    2 * (w * β) * ((β * D.gs κ v (w * β)) * β + w * (w * D.gs κ v (w * β))) < 0 := by
  rw [flow_prune_slice D hv]
  have hw : w ≠ 0 := left_ne_zero_of_mul hc
  have hsum : 0 < w ^ 2 + β ^ 2 := by positivity
  have hc2 : 0 < (w * β) ^ 2 := by positivity
  nlinarith [mul_pos (mul_pos hsum hvar) hc2]

/-- Gradient flow conserves `w² - β²` (balanced factorization). -/
theorem flow_conserve (κ v w β : ℝ) :
    w * (β * D.gs κ v (w * β)) - β * (w * D.gs κ v (w * β)) = 0 := by ring

/-- At `c = 0` the parallel loop's gradient is `S1 gv`: zero once the skip has fit, so a loop
that starts near zero after the skip has fit gets no first-order signal. -/
theorem gs_zero_par (κ v : ℝ) : D.gs κ v 0 = D.S1 * (κ - v) := by
  rw [gs_eq, gv_eq]
  ring

/-- **Gradient routing does not move the stationary set.** If the skip's own gradient is
scaled by any `λ ≠ 0` (a stop-gradient into the prelude, or a scaled backward through the
skip, leaves the skip reader's weight trainable), stationarity still forces `gv = 0`, hence
`w β = 0`. -/
theorem scaled_skip_grad_prunes {κ v w β lam : ℝ} (hvar : 0 < D.var) (hlam : lam ≠ 0)
    (hv : lam * D.gv κ v (w * β) = 0) (hw : β * D.gs κ v (w * β) = 0) : w * β = 0 :=
  D.par_stationary_prunes hvar ((mul_eq_zero.1 hv).resolve_left hlam) hw

/-! ## The series topology -/

/-- Series loss: the reader sees the loop output alone. -/
noncomputable def Jser (κ c : ℝ) : ℝ := ∑ i, D.π i * (κ - c * D.s i) ^ 2

theorem Jser_eq (κ c : ℝ) : D.Jser κ c = κ ^ 2 - 2 * κ * c * D.S1 + c ^ 2 * D.S2 := by
  have h : ∀ i, D.π i * (κ - c * D.s i) ^ 2 = D.π i * (κ + (-c) * D.s i) ^ 2 :=
    fun i => by ring
  unfold Jser
  simp_rw [h, D.sum_sq_affine]
  ring

/-- **A frozen skip is a series loop on the remainder.** With the skip weight held at `v₀`
(its gradient zeroed), the parallel loss is the series loss of the target `κ - v₀`: the loop is
kept, but with the series depth dependence on what the skip does not carry. -/
theorem Jpar_fixed_eq_Jser (κ v₀ c : ℝ) : D.Jpar κ v₀ c = D.Jser (κ - v₀) c := by
  unfold Jpar Jser
  exact Finset.sum_congr rfl fun i _ => by ring

/-- Series has no skip: its loop gradient at `c = 0` is `κ S1`, nonzero for any target. -/
theorem gs_zero_ser (κ : ℝ) : D.gs κ 0 0 = κ * D.S1 := by
  rw [gs_zero_par]
  ring

theorem Jser_vertex {κ c : ℝ} (hS2 : 0 < D.S2) :
    D.Jser κ c = D.S2 * (c - κ * D.S1 / D.S2) ^ 2 + (κ ^ 2 - κ ^ 2 * D.S1 ^ 2 / D.S2) := by
  rw [Jser_eq]
  field_simp
  ring

/-- **The series optimum keeps the loop**: its unique minimizer is `c* = κ S1 / S2`. -/
theorem ser_minimizer {κ c : ℝ} (hS2 : 0 < D.S2) (hmin : ∀ c', D.Jser κ c ≤ D.Jser κ c') :
    c = κ * D.S1 / D.S2 := by
  have h := hmin (κ * D.S1 / D.S2)
  rw [Jser_vertex D hS2, Jser_vertex D hS2, sub_self] at h
  have h2 : D.S2 * (c - κ * D.S1 / D.S2) ^ 2 = 0 := by
    have := mul_nonneg hS2.le (sq_nonneg (c - κ * D.S1 / D.S2))
    nlinarith
  have h3 := (mul_eq_zero.1 h2).resolve_left hS2.ne'
  have := pow_eq_zero_iff (n := 2) (by norm_num) |>.1 h3
  linarith

/-- The series optimum's loss at depth `i`: `κ² (1 - q s i)²`, `q = S1 / S2`. -/
theorem ser_loss_at_opt (κ : ℝ) (i : ι) :
    (κ - κ * D.S1 / D.S2 * D.s i) ^ 2 = κ ^ 2 * (1 - D.S1 / D.S2 * D.s i) ^ 2 := by ring

/-- **The series K-curve is positive** between a shallow depth `i` and a deeper `j` whose
scale is still below `S2 / S1`. -/
theorem ser_kcurve_pos {κ : ℝ} (hκ : κ ≠ 0) {i j : ι} (hq : 0 < D.S1 / D.S2)
    (hij : D.s i < D.s j) (hj : D.S1 / D.S2 * D.s j ≤ 1) :
    0 < (κ - κ * D.S1 / D.S2 * D.s i) ^ 2 - (κ - κ * D.S1 / D.S2 * D.s j) ^ 2 := by
  rw [ser_loss_at_opt, ser_loss_at_opt]
  have hk : 0 < κ ^ 2 := by positivity
  have h1 : D.S1 / D.S2 * D.s i < D.S1 / D.S2 * D.s j := mul_lt_mul_of_pos_left hij hq
  nlinarith [mul_pos hk (sub_pos.2 h1)]

/-- **Dependence without value.** The parallel optimum's loss is zero at every depth; the
series optimum's is `κ² (1 - q s i)² ≥ 0`. A positive series K-curve sits on a loss that is
never below the parallel one. -/
theorem par_le_ser_every_depth (κ : ℝ) (i : ι) :
    (κ - κ - 0 * D.s i) ^ 2 ≤ (κ - κ * D.S1 / D.S2 * D.s i) ^ 2 := by
  simp only [sub_self, zero_mul, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow]
  exact sq_nonneg _

/-! ## A feature only the loop can build -/

/-- Parallel loss with a skip feature (target `κ`) and a loop-only feature (target `η`),
orthonormal, the loop writing both with scale `s`. -/
noncomputable def Jpar2 (κ η v c₁ c₂ : ℝ) : ℝ := D.Jpar κ v c₁ + D.Jser η c₂

/-- `Jpar2` is the depth average of the two squared residuals. -/
theorem Jpar2_eq (κ η v c₁ c₂ : ℝ) :
    D.Jpar2 κ η v c₁ c₂ =
      ∑ i, D.π i * ((κ - v - c₁ * D.s i) ^ 2 + (η - c₂ * D.s i) ^ 2) := by
  unfold Jpar2 Jpar Jser
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- **The parallel optimum prunes the duplicate and keeps the unique channel at the series
weight.** -/
theorem par2_minimizer {κ η v c₁ c₂ : ℝ} (hvar : 0 < D.var) (hS2 : 0 < D.S2)
    (hmin : ∀ v' c₁' c₂', D.Jpar2 κ η v c₁ c₂ ≤ D.Jpar2 κ η v' c₁' c₂') :
    v = κ ∧ c₁ = 0 ∧ c₂ = η * D.S1 / D.S2 := by
  have hA : ∀ v' c₁', D.Jpar κ v c₁ ≤ D.Jpar κ v' c₁' := by
    intro v' c₁'
    have := hmin v' c₁' c₂
    unfold Jpar2 at this
    linarith
  have hB : ∀ c₂', D.Jser η c₂ ≤ D.Jser η c₂' := by
    intro c₂'
    have := hmin v c₁ c₂'
    unfold Jpar2 at this
    linarith
  obtain ⟨h1, h2⟩ := D.par_minimizer hvar hA
  exact ⟨h1, h2, D.ser_minimizer hS2 hB⟩

/-- Per-depth loss of the two-feature PARALLEL reader at its optimum. -/
noncomputable def lossPar2 (κ η : ℝ) (i : ι) : ℝ :=
  (κ - κ - 0 * D.s i) ^ 2 + (η - η * D.S1 / D.S2 * D.s i) ^ 2

/-- Per-depth loss of the two-feature SERIES reader at its optimum (both channels at
`· S1 / S2`, the series minimizer of each separable part). -/
noncomputable def lossSer2 (κ η : ℝ) (i : ι) : ℝ :=
  (κ - κ * D.S1 / D.S2 * D.s i) ^ 2 + (η - η * D.S1 / D.S2 * D.s i) ^ 2

/-- **The parallel loop keeps exactly the unique share of the series K-curve.** For any two
depths, `(κ² + η²) (Kpar) = η² (Kser)`. -/
theorem unique_share (κ η : ℝ) (i j : ι) :
    (κ ^ 2 + η ^ 2) * (D.lossPar2 κ η i - D.lossPar2 κ η j) =
      η ^ 2 * (D.lossSer2 κ η i - D.lossSer2 κ η j) := by
  unfold lossPar2 lossSer2
  ring

/-! ## Stochastic skip: drop the skip with probability `p` at train -/

/-- Training loss with the skip dropped with probability `p` (the reader's weights are shared
between the two modes, as a linear coda's are). -/
noncomputable def Jdrop (p κ v c : ℝ) : ℝ := (1 - p) * D.Jpar κ v c + p * D.Jser κ c

/-- The loop weight the stochastic skip keeps. -/
noncomputable def cDrop (p κ : ℝ) : ℝ := p * κ * D.S1 / ((1 - p) * D.var + p * D.S2)

theorem Jdrop_on_slice {p κ c : ℝ} (hD : 0 < (1 - p) * D.var + p * D.S2) :
    D.Jdrop p κ (κ - c * D.S1) c =
      ((1 - p) * D.var + p * D.S2) * (c - D.cDrop p κ) ^ 2 +
        (p * κ ^ 2 - ((1 - p) * D.var + p * D.S2) * D.cDrop p κ ^ 2) := by
  unfold Jdrop cDrop
  rw [Jpar_eq, Jser_eq]
  field_simp
  unfold var
  ring

/-- **The stochastic skip's optimum.** For `0 ≤ p < 1`, every minimizer has `c = c_p` and
`v = κ - c_p S1`. -/
theorem drop_minimizer {p κ v c : ℝ} (hp1 : p < 1) (hD : 0 < (1 - p) * D.var + p * D.S2)
    (hmin : ∀ v' c', D.Jdrop p κ v c ≤ D.Jdrop p κ v' c') :
    c = D.cDrop p κ ∧ v = κ - D.cDrop p κ * D.S1 := by
  have hp : 0 < 1 - p := by linarith
  -- the skip weight sits on the slice
  have hslice : v = κ - c * D.S1 := by
    have h := hmin (κ - c * D.S1) c
    unfold Jdrop at h
    rw [Jpar_eq, Jpar_eq] at h
    have h0 : (1 - p) * (κ - v - c * D.S1) ^ 2 ≤ 0 := by
      have : κ - (κ - c * D.S1) - c * D.S1 = 0 := by ring
      rw [this] at h
      nlinarith
    have h1 : (κ - v - c * D.S1) ^ 2 = 0 := by
      have := mul_nonneg hp.le (sq_nonneg (κ - v - c * D.S1))
      have h2 : (1 - p) * (κ - v - c * D.S1) ^ 2 = 0 := le_antisymm h0 this
      exact (mul_eq_zero.1 h2).resolve_left hp.ne'
    have := pow_eq_zero_iff (n := 2) (by norm_num) |>.1 h1
    linarith
  -- the loop weight is the vertex of a positive quadratic
  have hc : c = D.cDrop p κ := by
    have h := hmin (κ - D.cDrop p κ * D.S1) (D.cDrop p κ)
    rw [hslice, Jdrop_on_slice D hD, Jdrop_on_slice D hD, sub_self] at h
    have h2 : ((1 - p) * D.var + p * D.S2) * (c - D.cDrop p κ) ^ 2 = 0 := by
      have := mul_nonneg hD.le (sq_nonneg (c - D.cDrop p κ))
      nlinarith
    have h3 := (mul_eq_zero.1 h2).resolve_left hD.ne'
    have := pow_eq_zero_iff (n := 2) (by norm_num) |>.1 h3
    linarith
  exact ⟨hc, by rw [hslice, hc]⟩

/-- **Eval with the skip on.** At the stochastic-skip optimum the per-depth loss is
`c_p² (S1 - s i)²`: zero only at depths whose scale equals the mean. -/
theorem drop_eval_loss (p κ : ℝ) (i : ι) :
    (κ - (κ - D.cDrop p κ * D.S1) - D.cDrop p κ * D.s i) ^ 2 =
      D.cDrop p κ ^ 2 * (D.S1 - D.s i) ^ 2 := by ring

/-- **The stochastic skip's K-curve is a tax.** Its depth-averaged eval loss (skip on) is
`c_p² var`, strictly positive for `0 < p < 1` with `κ, S1 ≠ 0` and `var > 0`, while the plain
parallel optimum is zero (`Jpar_skip`). -/
theorem drop_tax {p κ : ℝ} (hp0 : 0 < p) (hD : 0 < (1 - p) * D.var + p * D.S2)
    (hκ : κ ≠ 0) (hS1 : D.S1 ≠ 0) (hvar : 0 < D.var) :
    D.Jpar κ (κ - D.cDrop p κ * D.S1) (D.cDrop p κ) = D.cDrop p κ ^ 2 * D.var ∧
      0 < D.cDrop p κ ^ 2 * D.var := by
  refine ⟨by rw [Jpar_eq]; ring, ?_⟩
  have hc : D.cDrop p κ ≠ 0 := by
    unfold cDrop
    exact div_ne_zero (mul_ne_zero (mul_ne_zero hp0.ne' hκ) hS1) hD.ne'
  positivity

end DepthLaw

end ParallelLoop
