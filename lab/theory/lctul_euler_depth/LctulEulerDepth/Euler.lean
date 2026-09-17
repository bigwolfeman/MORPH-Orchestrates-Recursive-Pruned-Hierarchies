/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import Mathlib.Analysis.Normed.Module.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

/-!
# The Euler sampler of LCTUL, as algebra

`morph/model/tul_code.py::euler_sample` is

```
z = z0
for j in range(k):
    z = z + (1/k) * v(z, j/k)
```

and `eulerEndpoint v k z0` below is exactly that loop. `v z t` is the thinker's velocity
field at one fixed context (tape + seed); the context is not written because nothing in
this file varies it.

## Main results

* `eulerEndpoint_one` : one step from `t = 0` returns `z0 + v z0 0`.
* `eulerEndpoint_one_of_meanField` : if the field at `t = 0` is `m - z` (the L²-optimal
  field, `SecondMoment.lean`), one step returns `m` from EVERY source draw. The `k = 1`
  sample is the conditional mean and carries no randomness.
* `eulerEndpoint_dirac` : for a code fully determined by the context (`z* `), the optimal
  field `(z* - z)/(1 - t)` is integrated EXACTLY by Euler at every `k ≥ 1`. Depth is worth
  nothing because the first step already lands.
* `euler_affine` : an affine field `a t • z + b t` makes every `k`-step endpoint an affine
  image of the source draw, `λ_k • z0 + c_k`.
* `eulerEndpoint_gauss` : the isotropic-Gaussian field `a t • z + (1 - t * a t) • μ` makes
  the `k`-step endpoint `μ + λ_k • z0` for EVERY `k ≥ 1`: the mean component is exact at
  every depth, the noise direction is untouched, and the only thing `k` sets is the
  scalar radius `λ_k`. With `λ_1 = 0` (`affScale_gauss_one`).
* `affScale_gauss_two` and `affScale_gauss_pos` : `λ_2 = 1/2` and `λ_k > 0` for `k ≥ 2`
  on matched scales. So the `k = 1 → 2` step is "mean → mean plus noise", and nothing
  after it changes the family.
* `euler_error_bound` : the discrete Grönwall bound. If the exact trajectory has local
  defect at most `δ` against one Euler step and the field is `L`-Lipschitz, the `k`-step
  endpoint is within `δ * ∑_{i<k} (1 + L/k)^i` of the exact endpoint. `δ` is the
  curvature of the field along the path; `euler_exact_of_zero_defect` is the `δ = 0` case.
* `euler_error_bound_curvature` : the `O(M/k)` form, `‖z_k - φ_k‖ ≤ M * exp L / (2 k)`
  when `δ = M / (2 k²)`.
-/

noncomputable section

namespace LctulEulerDepth

open Finset

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-! ## The sampler -/

/-- One Euler step of the `k`-step schedule at step index `j`, i.e. at time `t_j = j / k`. -/
def eulerStep (v : E → ℝ → E) (k : ℕ) (j : ℕ) (z : E) : E :=
  z + ((k : ℝ)⁻¹) • v z ((j : ℝ) / k)

/-- The state after `j` Euler steps from `z0`, `k` steps in the whole schedule. -/
def euler (v : E → ℝ → E) (k : ℕ) (z0 : E) : ℕ → E
  | 0 => z0
  | j + 1 => eulerStep v k j (euler v k z0 j)

/-- The `k`-step Euler endpoint: `euler_sample(velocity, z0, k)`. -/
def eulerEndpoint (v : E → ℝ → E) (k : ℕ) (z0 : E) : E := euler v k z0 k

@[simp] lemma euler_zero (v : E → ℝ → E) (k : ℕ) (z0 : E) : euler v k z0 0 = z0 := rfl

lemma euler_succ (v : E → ℝ → E) (k : ℕ) (z0 : E) (j : ℕ) :
    euler v k z0 (j + 1) = euler v k z0 j + ((k : ℝ)⁻¹) • v (euler v k z0 j) ((j : ℝ) / k) :=
  rfl

/-! ## `k = 1`: the endpoint is the field at the source, and for the optimal field the mean -/

/-- One Euler step from `t = 0` returns `z0 + v z0 0`. -/
theorem eulerEndpoint_one (v : E → ℝ → E) (z0 : E) : eulerEndpoint v 1 z0 = z0 + v z0 0 := by
  simp [eulerEndpoint, euler, eulerStep]

/-- **The `k = 1` sample is the conditional mean.**  If the field at `t = 0` is the
mean-reverting field `m - z` (the L²-optimal field at `t = 0`, `fm_loss_t0_optimal`), one
Euler step returns `m` whatever the source draw. The endpoint is constant in `z0`: it has
no randomness at all. -/
theorem eulerEndpoint_one_of_meanField (v : E → ℝ → E) (m : E) (h : ∀ z, v z 0 = m - z)
    (z0 : E) : eulerEndpoint v 1 z0 = m := by
  rw [eulerEndpoint_one, h]; abel

/-! ## The Dirac conditional: Euler is exact at every `k` -/

/-- The optimal marginal field when the code is determined by the context: `z* ` is the
only target, the paths are straight lines to it, and `v z t = (z* - z) / (1 - t)`. -/
def diracField (zs : E) : E → ℝ → E := fun z t => (1 - t)⁻¹ • (zs - z)

/-- Along the Dirac field the Euler iterates are the exact straight line `z0 + (j/k)(z* - z0)`. -/
theorem euler_dirac (zs z0 : E) (k : ℕ) (hk : 0 < k) :
    ∀ j, j ≤ k → euler (diracField zs) k z0 j = z0 + ((j : ℝ) / k) • (zs - z0) := by
  intro j
  induction j with
  | zero => intro _; simp
  | succ j ih =>
    intro hj
    have hjk : j < k := Nat.lt_of_succ_le hj
    have hne : (1 : ℝ) - (j : ℝ) / k ≠ 0 := by
      have hk' : (0 : ℝ) < k := by exact_mod_cast hk
      have : (j : ℝ) / k < 1 := by
        rw [div_lt_one hk']; exact_mod_cast hjk
      linarith
    rw [euler_succ, ih (Nat.le_of_succ_le hj)]
    simp only [diracField]
    have hv : (1 - (j : ℝ) / k)⁻¹ • (zs - (z0 + ((j : ℝ) / k) • (zs - z0))) = zs - z0 := by
      have : zs - (z0 + ((j : ℝ) / k) • (zs - z0)) = (1 - (j : ℝ) / k) • (zs - z0) := by
        rw [sub_smul, one_smul]; abel
      rw [this, smul_smul, inv_mul_cancel₀ hne, one_smul]
    rw [hv]
    have hk' : (k : ℝ) ≠ 0 := by exact_mod_cast hk.ne'
    rw [add_assoc, ← add_smul]
    congr 2
    push_cast
    field_simp

/-- **Depth is worth nothing on a determined code.**  For every `k ≥ 1` the Euler endpoint
along the Dirac field is `z* ` exactly. -/
theorem eulerEndpoint_dirac (zs z0 : E) (k : ℕ) (hk : 0 < k) :
    eulerEndpoint (diracField zs) k z0 = zs := by
  rw [eulerEndpoint, euler_dirac zs z0 k hk k le_rfl]
  have hk' : (k : ℝ) ≠ 0 := by exact_mod_cast hk.ne'
  rw [div_self hk', one_smul]; abel

/-! ## Affine fields: the endpoint is an affine image of the source draw -/

/-- The scalar `λ_j = ∏_{i<j} (1 + a(t_i)/k)` that an affine field puts in front of `z0`. -/
def affScale (a : ℝ → ℝ) (k : ℕ) : ℕ → ℝ
  | 0 => 1
  | j + 1 => affScale a k j * (1 + (k : ℝ)⁻¹ * a ((j : ℝ) / k))

/-- The shift `c_j` an affine field accumulates. -/
def affShift (a : ℝ → ℝ) (b : ℝ → E) (k : ℕ) : ℕ → E
  | 0 => 0
  | j + 1 => affShift a b k j
      + (k : ℝ)⁻¹ • (a ((j : ℝ) / k) • affShift a b k j + b ((j : ℝ) / k))

/-- **Affine field, affine sampler.**  If `v z t = a t • z + b t`, then after `j` Euler steps
`z_j = λ_j • z0 + c_j` with `λ_j`, `c_j` independent of `z0`. Every endpoint distribution over
`k` is an affine image of ONE source distribution. -/
theorem euler_affine (a : ℝ → ℝ) (b : ℝ → E) (k : ℕ) (z0 : E) :
    ∀ j, euler (fun z t => a t • z + b t) k z0 j = affScale a k j • z0 + affShift a b k j := by
  intro j
  induction j with
  | zero => simp [affScale, affShift]
  | succ j ih =>
    rw [euler_succ, ih]
    simp only [affScale, affShift]
    simp only [smul_add, smul_smul, add_smul, mul_add, mul_one]
    abel_nf
    simp only [mul_comm, mul_left_comm]
    abel

/-- The endpoint form of `euler_affine`. -/
theorem eulerEndpoint_affine (a : ℝ → ℝ) (b : ℝ → E) (k : ℕ) (z0 : E) :
    eulerEndpoint (fun z t => a t • z + b t) k z0 = affScale a k k • z0 + affShift a b k k :=
  euler_affine a b k z0 k

/-! ## The isotropic Gaussian conditional -/

/-- The scalar coefficient of the L²-optimal (conditional-expectation) field when the source
is `N(0, s0² I)` and the code given the context is `N(μ, s1² I)`, independent:
`E[z1 - z0 | z_t = z] = a t • (z - t • μ) + μ` with
`a t = (t s1² - (1-t) s0²) / ((1-t)² s0² + t² s1²)` (the covariance of `z1 - z0` with
`z_t` over the variance of `z_t`). That this affine form IS the Gaussian
conditional expectation is standard Gaussian conditioning and is NOT proved here; everything
below is a consequence of the form. -/
noncomputable def gaussCoeff (s0 s1 : ℝ) (t : ℝ) : ℝ :=
  (t * s1 ^ 2 - (1 - t) * s0 ^ 2) / ((1 - t) ^ 2 * s0 ^ 2 + t ^ 2 * s1 ^ 2)

/-- The isotropic-Gaussian optimal field, written as `a t • z + (1 - t * a t) • μ`. -/
noncomputable def gaussField (s0 s1 : ℝ) (μ : E) : E → ℝ → E :=
  fun z t => gaussCoeff s0 s1 t • z + (1 - t * gaussCoeff s0 s1 t) • μ

lemma gaussCoeff_zero (s0 s1 : ℝ) (hs0 : s0 ≠ 0) : gaussCoeff s0 s1 0 = -1 := by
  simp only [gaussCoeff]
  have : s0 ^ 2 ≠ 0 := pow_ne_zero 2 hs0
  field_simp
  ring

/-- Along the Gaussian field the shift after `j` steps is exactly `(j/k) • μ`: the mean
component of the Euler iterate is the exact linear interpolation `t_j • μ` at EVERY `k`. -/
theorem euler_gauss (s0 s1 : ℝ) (μ z0 : E) (k : ℕ) :
    ∀ j, euler (gaussField s0 s1 μ) k z0 j
      = affScale (gaussCoeff s0 s1) k j • z0 + ((j : ℝ) / k) • μ := by
  intro j
  induction j with
  | zero => simp [affScale]
  | succ j ih =>
    rw [euler_succ, ih]
    simp only [gaussField, affScale]
    set a := gaussCoeff s0 s1 ((j : ℝ) / k) with ha
    have hshift : ((j : ℝ) / k) • μ
        + (k : ℝ)⁻¹ • (a • (((j : ℝ) / k) • μ) + (1 - ((j : ℝ) / k) * a) • μ)
        = (((j + 1 : ℕ) : ℝ) / k) • μ := by
      rw [smul_smul, ← add_smul,
        show a * ((j : ℝ) / k) + (1 - (j : ℝ) / k * a) = 1 by ring, one_smul, ← add_smul]
      congr 1
      push_cast
      ring
    calc affScale (gaussCoeff s0 s1) k j • z0 + ((j : ℝ) / k) • μ
          + (k : ℝ)⁻¹ • (a • (affScale (gaussCoeff s0 s1) k j • z0 + ((j : ℝ) / k) • μ)
            + (1 - (j : ℝ) / k * a) • μ)
        = (affScale (gaussCoeff s0 s1) k j * (1 + (k : ℝ)⁻¹ * a)) • z0
          + (((j : ℝ) / k) • μ
            + (k : ℝ)⁻¹ • (a • (((j : ℝ) / k) • μ) + (1 - ((j : ℝ) / k) * a) • μ)) := by
          simp only [smul_add, smul_smul, add_smul, mul_add, mul_one]
          abel_nf
          simp only [mul_comm, mul_left_comm]
          abel
      _ = _ := by rw [hshift]

/-- **The Gaussian sampler at every depth is `μ + λ_k • z0`.**  The mean is exact at every
`k ≥ 1`; the noise direction is the source direction at every `k`; `k` sets only the scalar
`λ_k = affScale (gaussCoeff s0 s1) k k`. -/
theorem eulerEndpoint_gauss (s0 s1 : ℝ) (μ z0 : E) (k : ℕ) (hk : 0 < k) :
    eulerEndpoint (gaussField s0 s1 μ) k z0 = μ + affScale (gaussCoeff s0 s1) k k • z0 := by
  rw [eulerEndpoint, euler_gauss]
  have hk' : (k : ℝ) ≠ 0 := by exact_mod_cast hk.ne'
  rw [div_self hk', one_smul, add_comm]

/-- `λ_1 = 0`: the one-step Gaussian sample is the bare mean. -/
theorem affScale_gauss_one (s0 s1 : ℝ) (hs0 : s0 ≠ 0) :
    affScale (gaussCoeff s0 s1) 1 1 = 0 := by
  simp [affScale, gaussCoeff_zero s0 s1 hs0]

/-- The `k = 1` Gaussian endpoint is `μ` (the special case of `eulerEndpoint_one_of_meanField`). -/
theorem eulerEndpoint_gauss_one (s0 s1 : ℝ) (hs0 : s0 ≠ 0) (μ z0 : E) :
    eulerEndpoint (gaussField s0 s1 μ) 1 z0 = μ := by
  rw [eulerEndpoint_gauss s0 s1 μ z0 1 Nat.one_pos, affScale_gauss_one s0 s1 hs0, zero_smul,
    add_zero]

/-- On matched scales (`s0 = s1 = 1`) the two-step radius is `λ_2 = 1/2`: the `k = 2` sample is
`μ + z0 / 2`. -/
theorem affScale_gauss_two : affScale (gaussCoeff 1 1) 2 2 = 1 / 2 := by
  simp [affScale, gaussCoeff]
  norm_num

lemma gaussCoeff_matched (t : ℝ) :
    gaussCoeff 1 1 t = (2 * t - 1) / (2 * t ^ 2 - 2 * t + 1) := by
  simp only [gaussCoeff, one_pow, mul_one]
  have h : (1 - t) ^ 2 + t ^ 2 = 2 * t ^ 2 - 2 * t + 1 := by ring
  rw [h]
  congr 1
  ring

/-- Every Euler factor `1 + a(t)/k` is positive on matched scales once `k ≥ 2`. -/
lemma one_add_gaussCoeff_div_pos (k : ℕ) (hk : 2 ≤ k) (t : ℝ) :
    0 < 1 + (k : ℝ)⁻¹ * gaussCoeff 1 1 t := by
  rw [gaussCoeff_matched]
  have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
  have hden : 0 < 2 * t ^ 2 - 2 * t + 1 := by nlinarith [sq_nonneg (2 * t - 1)]
  have hkpos : (0 : ℝ) < k := by linarith
  have hq : (k : ℝ)⁻¹ * ((2 * t - 1) / (2 * t ^ 2 - 2 * t + 1))
      = (2 * t - 1) / ((k : ℝ) * (2 * t ^ 2 - 2 * t + 1)) := by
    field_simp
  rw [hq]
  have hlt : -1 < (2 * t - 1) / ((k : ℝ) * (2 * t ^ 2 - 2 * t + 1)) := by
    rw [lt_div_iff₀ (by positivity)]
    nlinarith [sq_nonneg (4 * t - 1), mul_le_mul_of_nonneg_right hk' hden.le]
  linarith

/-- `λ_j > 0` for every `j` on matched scales once `k ≥ 2`: from two steps on, the sample is
never the mean; it is the mean plus a strictly positive multiple of the source noise. -/
theorem affScale_gauss_pos (k : ℕ) (hk : 2 ≤ k) : ∀ j, 0 < affScale (gaussCoeff 1 1) k j := by
  intro j
  induction j with
  | zero => simp [affScale]
  | succ j ih =>
    simp only [affScale]
    exact mul_pos ih (one_add_gaussCoeff_div_pos k hk _)

/-! ## The general Euler error: curvature along the path is the whole of it -/

/-- **Discrete Grönwall for the Euler sampler.**  `φ` is any reference trajectory sampled at
the step times, `φ 0 = z0`. If one Euler step from `φ j` misses `φ (j+1)` by at most `δ`
(the local defect: zero when the field is constant along the path) and the field is
`L`-Lipschitz in `z` at every step time, the Euler iterate is within
`δ * ∑_{i<j} (1 + L/k)^i` of `φ j`. -/
theorem euler_error_bound (v : E → ℝ → E) (k : ℕ) (z0 : E) (L δ : ℝ) (hL : 0 ≤ L)
    (φ : ℕ → E) (hφ0 : φ 0 = z0)
    (hlip : ∀ (j : ℕ) (z z' : E), ‖v z ((j : ℝ) / k) - v z' ((j : ℝ) / k)‖ ≤ L * ‖z - z'‖)
    (hdef : ∀ j, ‖φ (j + 1) - (φ j + (k : ℝ)⁻¹ • v (φ j) ((j : ℝ) / k))‖ ≤ δ) :
    ∀ j, ‖euler v k z0 j - φ j‖ ≤ δ * ∑ i ∈ range j, (1 + L / k) ^ i := by
  intro j
  induction j with
  | zero => simp [hφ0]
  | succ j ih =>
    have hkinv : (0 : ℝ) ≤ (k : ℝ)⁻¹ := by positivity
    have hδ : 0 ≤ δ := le_trans (norm_nonneg _) (hdef 0)
    set zj := euler v k z0 j with hzj
    set tj : ℝ := (j : ℝ) / k with htj
    have hsplit : euler v k z0 (j + 1) - φ (j + 1)
        = (zj - φ j) + (k : ℝ)⁻¹ • (v zj tj - v (φ j) tj)
          - (φ (j + 1) - (φ j + (k : ℝ)⁻¹ • v (φ j) tj)) := by
      rw [euler_succ, smul_sub]; abel
    have h1 : ‖(zj - φ j) + (k : ℝ)⁻¹ • (v zj tj - v (φ j) tj)‖
        ≤ (1 + L / k) * ‖zj - φ j‖ := by
      calc ‖(zj - φ j) + (k : ℝ)⁻¹ • (v zj tj - v (φ j) tj)‖
          ≤ ‖zj - φ j‖ + ‖(k : ℝ)⁻¹ • (v zj tj - v (φ j) tj)‖ := norm_add_le _ _
        _ = ‖zj - φ j‖ + (k : ℝ)⁻¹ * ‖v zj tj - v (φ j) tj‖ := by
          rw [norm_smul, Real.norm_of_nonneg hkinv]
        _ ≤ ‖zj - φ j‖ + (k : ℝ)⁻¹ * (L * ‖zj - φ j‖) := by
          gcongr
          exact hlip j zj (φ j)
        _ = (1 + L / k) * ‖zj - φ j‖ := by ring
    have hbound : ‖euler v k z0 (j + 1) - φ (j + 1)‖
        ≤ (1 + L / k) * ‖zj - φ j‖ + δ := by
      rw [hsplit]
      calc ‖(zj - φ j) + (k : ℝ)⁻¹ • (v zj tj - v (φ j) tj)
            - (φ (j + 1) - (φ j + (k : ℝ)⁻¹ • v (φ j) tj))‖
          ≤ ‖(zj - φ j) + (k : ℝ)⁻¹ • (v zj tj - v (φ j) tj)‖
            + ‖φ (j + 1) - (φ j + (k : ℝ)⁻¹ • v (φ j) tj)‖ := norm_sub_le _ _
        _ ≤ (1 + L / k) * ‖zj - φ j‖ + δ := add_le_add h1 (hdef j)
    have hgeom : δ * ∑ i ∈ range (j + 1), (1 + L / k) ^ i
        = (1 + L / k) * (δ * ∑ i ∈ range j, (1 + L / k) ^ i) + δ := by
      rw [Finset.sum_range_succ']
      simp only [pow_succ, pow_zero]
      rw [← Finset.sum_mul]
      ring
    rw [hgeom]
    have hpos : 0 ≤ 1 + L / k := by positivity
    calc ‖euler v k z0 (j + 1) - φ (j + 1)‖
        ≤ (1 + L / k) * ‖zj - φ j‖ + δ := hbound
      _ ≤ (1 + L / k) * (δ * ∑ i ∈ range j, (1 + L / k) ^ i) + δ := by
        gcongr

/-- **Zero curvature along the path, zero error at every depth.**  If the field is constant
along the exact trajectory between step times (`δ = 0`), the Euler iterate IS the exact
trajectory for every `k`. The Dirac field is the model case (`eulerEndpoint_dirac`). -/
theorem euler_exact_of_zero_defect (v : E → ℝ → E) (k : ℕ) (z0 : E) (L : ℝ) (hL : 0 ≤ L)
    (φ : ℕ → E) (hφ0 : φ 0 = z0)
    (hlip : ∀ (j : ℕ) (z z' : E), ‖v z ((j : ℝ) / k) - v z' ((j : ℝ) / k)‖ ≤ L * ‖z - z'‖)
    (hdef : ∀ j, φ (j + 1) = φ j + (k : ℝ)⁻¹ • v (φ j) ((j : ℝ) / k)) :
    ∀ j, euler v k z0 j = φ j := by
  intro j
  have h := euler_error_bound v k z0 L 0 hL φ hφ0 hlip (fun j => by rw [hdef j]; simp) j
  simp only [zero_mul] at h
  exact sub_eq_zero.mp (norm_le_zero_iff.mp h)

lemma one_add_div_pow_le_exp (L : ℝ) (hL : 0 ≤ L) (k : ℕ) (hk : 0 < k) :
    (1 + L / k) ^ k ≤ Real.exp L := by
  have hkpos : (0 : ℝ) < k := by exact_mod_cast hk
  have h1 : 1 + L / k ≤ Real.exp (L / k) := by
    have := Real.add_one_le_exp (L / k); linarith
  have h0 : 0 ≤ 1 + L / k := by positivity
  calc (1 + L / k) ^ k ≤ (Real.exp (L / k)) ^ k := pow_le_pow_left₀ h0 h1 k
    _ = Real.exp L := by
      rw [← Real.exp_nat_mul]
      congr 1
      field_simp

/-- **The `O(M/k)` Euler bound.**  With the local defect of a `C²` path, `δ = M / (2 k²)`
(`M` bounds the second derivative of the exact trajectory, i.e. the rate at which the field
turns along the path), the endpoint error is at most `M * exp L / (2 k)`. Depth pays at the
rate the field CURVES along the path; on a straight path (`M = 0`) it pays nothing. -/
theorem euler_error_bound_curvature (v : E → ℝ → E) (k : ℕ) (hk : 0 < k) (z0 : E)
    (L M : ℝ) (hL : 0 ≤ L) (hM : 0 ≤ M)
    (φ : ℕ → E) (hφ0 : φ 0 = z0)
    (hlip : ∀ (j : ℕ) (z z' : E), ‖v z ((j : ℝ) / k) - v z' ((j : ℝ) / k)‖ ≤ L * ‖z - z'‖)
    (hdef : ∀ j, ‖φ (j + 1) - (φ j + (k : ℝ)⁻¹ • v (φ j) ((j : ℝ) / k))‖
      ≤ M / (2 * (k : ℝ) ^ 2)) :
    ‖eulerEndpoint v k z0 - φ k‖ ≤ M * Real.exp L / (2 * k) := by
  have hkpos : (0 : ℝ) < k := by exact_mod_cast hk
  have h := euler_error_bound v k z0 L (M / (2 * (k : ℝ) ^ 2)) hL φ hφ0 hlip hdef k
  have hpos : 0 ≤ 1 + L / k := by positivity
  -- every term of the geometric sum is at most `(1 + L/k)^k ≤ exp L`, and there are `k` terms
  have hsum : ∑ i ∈ range k, (1 + L / k) ^ i ≤ (k : ℝ) * Real.exp L := by
    have hterm : ∀ i ∈ range k, (1 + L / k) ^ i ≤ Real.exp L := by
      intro i hi
      have hik : i ≤ k := (Finset.mem_range.mp hi).le
      have h1 : 1 ≤ 1 + L / k := by
        have : 0 ≤ L / k := by positivity
        linarith
      calc (1 + L / k) ^ i ≤ (1 + L / k) ^ k := pow_le_pow_right₀ h1 hik
        _ ≤ Real.exp L := one_add_div_pow_le_exp L hL k hk
    calc ∑ i ∈ range k, (1 + L / k) ^ i ≤ ∑ _i ∈ range k, Real.exp L :=
          Finset.sum_le_sum hterm
      _ = (k : ℝ) * Real.exp L := by simp
  have hδ : 0 ≤ M / (2 * (k : ℝ) ^ 2) := by positivity
  calc ‖eulerEndpoint v k z0 - φ k‖ ≤ M / (2 * (k : ℝ) ^ 2) * ∑ i ∈ range k, (1 + L / k) ^ i := h
    _ ≤ M / (2 * (k : ℝ) ^ 2) * ((k : ℝ) * Real.exp L) := by gcongr
    _ = M * Real.exp L / (2 * k) := by field_simp

end LctulEulerDepth

end
