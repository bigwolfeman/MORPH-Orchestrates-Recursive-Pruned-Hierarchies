/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.DerivativeTest
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Ring.Finset
import TulExploration.HardMin

/-!
# T1: the smooth multi-sample bound drives a learned noise scale to zero

## Model

The latent of one slot is `x = μ + σ n`.  A draw `o` (finite law `ρ`) places the `K`
latents at `μ + σ w o k`; `w` can encode independent samples, antithetic pairs or an
enumeration of fixed codes.  The target `y` has a finite law `P i` on values `y i`.  The
reader's log-likelihood of target `i` at latent `x` is `S i x`.  The smooth multi-sample bound
(XM's smooth form; the GK loss) is

  `J = ∑ᵢ P i ∑ₒ ρ o log ((1/K) ∑ₖ exp (S i (x o k)))`.

## Main results

General reader:
* `smoothObj_det`, `smoothObj_le_common_max`, `smoothObj_lt_common_max` : if one latent `x*`
  maximises the reader's score for EVERY target, the deterministic latent `x*` is optimal at
  every `K`, and any draw that puts a sample elsewhere is strictly worse.
* `smoothObj_one`, `single_draw_le_best` : at `K = 1` the objective is linear in the law of
  the latent, so some deterministic latent is at least as good as any noise.

Gaussian reader `S i x = -a (x - y i)²` (curvature `a = 1/(2s²)`):
* `quadObj_deriv_zero` : `J'(0) = -2a E[μ - y] · E[mean w]`, zero for centred noise.
* **`quadObj_deriv2_zero`** : `J''(0) = 4a² V · spread - 2a · m₂`, with `V = E(μ - y)²`,
  `m₂ = E[mean w²]` and `spread = E[mean w² - (mean w)²]` (the within-draw spread).
  Cost `2a m₂` against gain `4a² V spread`: both second order, exactly.
* `quadObj_isLocalMax` / `quadObj_isLocalMin` : the sign of `2aV·spread - m₂` decides.
* `matched_curvature_optimal` : with its normaliser, the reader's best curvature at `σ = 0` is
  `a = 1/(2V)` ("matched": predictive variance = residual spread).
* `iid_sum_sgn`, `iid_sum_sgn_sq` : `K` iid fair signs have `E mean = 0`, `E mean² = 1/K`.
* **`iid_matched_isLocalMax`**, `iid_matched_strict` : with `K` iid two-point draws and the
  reader's curvature matched to the target spread (`a = 1/(2V)`, the Gaussian reader's own
  optimum at `σ = 0`), `σ = 0` is a strict local maximum at EVERY `K ≥ 1`:
  `J''(0) = -2a/K < 0`.
* `iid_deriv2` : for `K` iid two-point draws, `J''(0) = 2a(2aV(1 - 1/K) - 1)`.
* `iid_sharp_isLocalMin`, `iid_sharp_not_isLocalMax` : a reader sharper than
  `2aV(1 - 1/K) > 1` makes `σ = 0` a local minimum and not a maximiser: the noise grows.
* `sharp_reader_counterexample` : a concrete instance (targets `±1`, `K = 2`, `a = 2`).  The
  global claim "`σ = 0` maximises the bound at every `K`" is FALSE once the target varies.
* `enumerated_pair_deriv2` : an enumerated antithetic pair (`K = 2`, codes `±1` always both
  present) has `J''(0) = 2a(2aV - 1)`: zero at the matched reader, positive for a sharper one.
-/

namespace TulExploration

open Finset

section General

variable {ι Ω : Type*} [Fintype ι] [Fintype Ω] {K : ℕ}

/-- **The smooth multi-sample objective.** -/
noncomputable def smoothObj (P : ι → ℝ) (ρ : Ω → ℝ) (S : ι → ℝ → ℝ) (x : Ω → Fin K → ℝ) :
    ℝ :=
  ∑ i, P i * ∑ o, ρ o * Real.log ((∑ k, Real.exp (S i (x o k))) / K)

/-- A deterministic latent `xs` scores `∑ P i S i xs` at every `K`. -/
theorem smoothObj_det (hK : 0 < K) (P : ι → ℝ) (ρ : Ω → ℝ) (hρ1 : ∑ o, ρ o = 1)
    (S : ι → ℝ → ℝ) (xs : ℝ) :
    smoothObj P ρ S (fun (_ : Ω) (_ : Fin K) => xs) = ∑ i, P i * S i xs := by
  have hKne : (K : ℝ) ≠ 0 := by exact_mod_cast hK.ne'
  unfold smoothObj
  apply Finset.sum_congr rfl
  intro i _
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [mul_div_cancel_left₀ _ hKne, Real.log_exp, ← Finset.sum_mul, hρ1, one_mul]

lemma log_mean_exp_le (hK : 0 < K) (s : Fin K → ℝ) (m : ℝ) (hs : ∀ k, s k ≤ m) :
    Real.log ((∑ k, Real.exp (s k)) / K) ≤ m := by
  have hKpos : (0 : ℝ) < K := by exact_mod_cast hK
  have : Nonempty (Fin K) := ⟨⟨0, hK⟩⟩
  have hpos : 0 < (∑ k, Real.exp (s k)) / K :=
    div_pos (Finset.sum_pos (fun k _ => Real.exp_pos _) univ_nonempty) hKpos
  rw [Real.log_le_iff_le_exp hpos, div_le_iff₀ hKpos]
  calc ∑ k, Real.exp (s k) ≤ ∑ _k : Fin K, Real.exp m :=
        Finset.sum_le_sum (fun k _ => Real.exp_le_exp.2 (hs k))
    _ = Real.exp m * K := by simp [mul_comm]

lemma log_mean_exp_lt (hK : 0 < K) (s : Fin K → ℝ) (m : ℝ) (hs : ∀ k, s k ≤ m)
    (k0 : Fin K) (hk0 : s k0 < m) :
    Real.log ((∑ k, Real.exp (s k)) / K) < m := by
  have hKpos : (0 : ℝ) < K := by exact_mod_cast hK
  have : Nonempty (Fin K) := ⟨⟨0, hK⟩⟩
  have hpos : 0 < (∑ k, Real.exp (s k)) / K :=
    div_pos (Finset.sum_pos (fun k _ => Real.exp_pos _) univ_nonempty) hKpos
  rw [Real.log_lt_iff_lt_exp hpos, div_lt_iff₀ hKpos]
  calc ∑ k, Real.exp (s k) < ∑ _k : Fin K, Real.exp m :=
        Finset.sum_lt_sum (fun k _ => Real.exp_le_exp.2 (hs k))
          ⟨k0, mem_univ _, Real.exp_lt_exp.2 hk0⟩
    _ = Real.exp m * K := by simp [mul_comm]

/-- **A common maximiser makes the deterministic latent optimal at every `K`.**  If one
latent `xs` maximises the reader's score for every target, no draw of `K` latents beats it.
This is XM's "unimodal data yields unimodal minimisers at any `K`" in its exact form: what
matters is not the shape of the data but whether one latent serves every target. -/
theorem smoothObj_le_common_max (hK : 0 < K) (P : ι → ℝ) (hP : ∀ i, 0 ≤ P i) (ρ : Ω → ℝ)
    (hρ : ∀ o, 0 ≤ ρ o) (hρ1 : ∑ o, ρ o = 1) (S : ι → ℝ → ℝ) (x : Ω → Fin K → ℝ) (xs : ℝ)
    (hmax : ∀ i z, S i z ≤ S i xs) :
    smoothObj P ρ S x ≤ smoothObj P ρ S (fun (_ : Ω) (_ : Fin K) => xs) := by
  rw [smoothObj_det hK P ρ hρ1 S xs]
  unfold smoothObj
  apply Finset.sum_le_sum
  intro i _
  apply mul_le_mul_of_nonneg_left _ (hP i)
  calc ∑ o, ρ o * Real.log ((∑ k, Real.exp (S i (x o k))) / K)
      ≤ ∑ o, ρ o * S i xs := Finset.sum_le_sum (fun o _ => mul_le_mul_of_nonneg_left
          (log_mean_exp_le hK _ _ (fun k => hmax i _)) (hρ o))
    _ = S i xs := by rw [← Finset.sum_mul, hρ1, one_mul]

/-- **Strictly worse off the maximiser.**  If some target of positive weight has its score
strictly below the maximum at some sample of some draw of positive weight, the noisy
objective is strictly below the deterministic one. -/
theorem smoothObj_lt_common_max (hK : 0 < K) (P : ι → ℝ) (hP : ∀ i, 0 ≤ P i) (ρ : Ω → ℝ)
    (hρ : ∀ o, 0 ≤ ρ o) (hρ1 : ∑ o, ρ o = 1) (S : ι → ℝ → ℝ) (x : Ω → Fin K → ℝ) (xs : ℝ)
    (hmax : ∀ i z, S i z ≤ S i xs) (i0 : ι) (o0 : Ω) (k0 : Fin K) (hPi : 0 < P i0)
    (hρo : 0 < ρ o0) (hlt : S i0 (x o0 k0) < S i0 xs) :
    smoothObj P ρ S x < smoothObj P ρ S (fun (_ : Ω) (_ : Fin K) => xs) := by
  rw [smoothObj_det hK P ρ hρ1 S xs]
  unfold smoothObj
  have inner_le : ∀ i, ∑ o, ρ o * Real.log ((∑ k, Real.exp (S i (x o k))) / K) ≤ S i xs := by
    intro i
    calc ∑ o, ρ o * Real.log ((∑ k, Real.exp (S i (x o k))) / K)
        ≤ ∑ o, ρ o * S i xs := Finset.sum_le_sum (fun o _ => mul_le_mul_of_nonneg_left
            (log_mean_exp_le hK _ _ (fun k => hmax i _)) (hρ o))
      _ = S i xs := by rw [← Finset.sum_mul, hρ1, one_mul]
  have inner_lt : ∑ o, ρ o * Real.log ((∑ k, Real.exp (S i0 (x o k))) / K) < S i0 xs := by
    calc ∑ o, ρ o * Real.log ((∑ k, Real.exp (S i0 (x o k))) / K)
        < ∑ o, ρ o * S i0 xs := by
          apply Finset.sum_lt_sum
          · exact fun o _ => mul_le_mul_of_nonneg_left
              (log_mean_exp_le hK _ _ (fun k => hmax i0 _)) (hρ o)
          · exact ⟨o0, mem_univ _, mul_lt_mul_of_pos_left
              (log_mean_exp_lt hK _ _ (fun k => hmax i0 _) k0 hlt) hρo⟩
      _ = S i0 xs := by rw [← Finset.sum_mul, hρ1, one_mul]
  apply Finset.sum_lt_sum
  · exact fun i _ => mul_le_mul_of_nonneg_left (inner_le i) (hP i)
  · exact ⟨i0, mem_univ _, mul_lt_mul_of_pos_left inner_lt hPi⟩

/-- At `K = 1` the objective is the expected score of the one latent: linear in its law. -/
theorem smoothObj_one (P : ι → ℝ) (ρ : Ω → ℝ) (S : ι → ℝ → ℝ) (x : Ω → Fin 1 → ℝ) :
    smoothObj P ρ S x = ∑ o, ρ o * ∑ i, P i * S i (x o 0) := by
  unfold smoothObj
  simp only [Fin.sum_univ_one, Nat.cast_one, div_one, Real.log_exp]
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl; intro o _
  apply Finset.sum_congr rfl; intro i _
  ring

/-- **At `K = 1` a deterministic latent is always at least as good as noise.**  The draw
that maximises the expected score, held fixed, does at least as well as the random draw. -/
theorem single_draw_le_best [Nonempty Ω] (P : ι → ℝ) (ρ : Ω → ℝ) (hρ : ∀ o, 0 ≤ ρ o)
    (hρ1 : ∑ o, ρ o = 1) (S : ι → ℝ → ℝ) (x : Ω → Fin 1 → ℝ) :
    ∃ o0, smoothObj P ρ S x ≤ smoothObj P ρ S (fun (_ : Ω) (_ : Fin 1) => x o0 0) := by
  obtain ⟨o0, -, ho0⟩ := Finset.exists_max_image univ (fun o => ∑ i, P i * S i (x o 0))
    univ_nonempty
  refine ⟨o0, ?_⟩
  rw [smoothObj_one, smoothObj_det Nat.one_pos P ρ hρ1 S (x o0 0)]
  calc ∑ o, ρ o * ∑ i, P i * S i (x o 0) ≤ ∑ o, ρ o * ∑ i, P i * S i (x o0 0) :=
        Finset.sum_le_sum (fun o _ => mul_le_mul_of_nonneg_left (ho0 o (mem_univ _)) (hρ o))
    _ = ∑ i, P i * S i (x o0 0) := by rw [← Finset.sum_mul, hρ1, one_mul]

end General

/-! ## The Gaussian reader: the exact second-order comparison -/

section Quadratic

variable {K : ℕ}

/-- The per-draw sum `D(s) = ∑ₖ exp (-a (μ + s wₖ - y)²)`. -/
noncomputable def qD (a μ y : ℝ) (w : Fin K → ℝ) (s : ℝ) : ℝ :=
  ∑ k, Real.exp (-a * (μ + s * w k - y) ^ 2)

/-- Its derivative `N(s)`. -/
noncomputable def qN (a μ y : ℝ) (w : Fin K → ℝ) (s : ℝ) : ℝ :=
  ∑ k, Real.exp (-a * (μ + s * w k - y) ^ 2) * (-2 * a * (μ + s * w k - y) * w k)

/-- The derivative of `N`. -/
noncomputable def qM (a μ y : ℝ) (w : Fin K → ℝ) (s : ℝ) : ℝ :=
  ∑ k, (Real.exp (-a * (μ + s * w k - y) ^ 2) * (-2 * a * (μ + s * w k - y) * w k)
      * (-2 * a * (μ + s * w k - y) * w k)
    + Real.exp (-a * (μ + s * w k - y) ^ 2) * (-2 * a * w k * w k))

lemma hasDerivAt_lin (μ y w s : ℝ) : HasDerivAt (fun t : ℝ => μ + t * w - y) w s := by
  have := (((hasDerivAt_id s).mul_const w).const_add μ).sub_const y
  simpa using this

lemma hasDerivAt_quad (a μ y w s : ℝ) :
    HasDerivAt (fun t : ℝ => -a * (μ + t * w - y) ^ 2) (-2 * a * (μ + s * w - y) * w) s := by
  have h := ((hasDerivAt_lin μ y w s).fun_pow 2).const_mul (-a)
  exact h.congr_deriv (by norm_num; ring)

lemma hasDerivAt_quad' (a μ y w s : ℝ) :
    HasDerivAt (fun t : ℝ => -2 * a * (μ + t * w - y) * w) (-2 * a * w * w) s := by
  exact ((hasDerivAt_lin μ y w s).const_mul (-2 * a)).mul_const w

lemma qD_pos (hK : 0 < K) (a μ y : ℝ) (w : Fin K → ℝ) (s : ℝ) : 0 < qD a μ y w s := by
  have : Nonempty (Fin K) := ⟨⟨0, hK⟩⟩
  exact Finset.sum_pos (fun k _ => Real.exp_pos _) univ_nonempty

lemma hasDerivAt_qD (a μ y : ℝ) (w : Fin K → ℝ) (s : ℝ) :
    HasDerivAt (qD a μ y w) (qN a μ y w s) s := by
  unfold qD qN
  exact HasDerivAt.fun_sum (fun k _ => (hasDerivAt_quad a μ y (w k) s).exp)

lemma hasDerivAt_qN (a μ y : ℝ) (w : Fin K → ℝ) (s : ℝ) :
    HasDerivAt (qN a μ y w) (qM a μ y w s) s := by
  unfold qN qM
  exact HasDerivAt.fun_sum (fun k _ =>
    ((hasDerivAt_quad a μ y (w k) s).exp).fun_mul (hasDerivAt_quad' a μ y (w k) s))

/-- The per-draw log-mean-exp has derivative `N/D`. -/
lemma hasDerivAt_logmean (hK : 0 < K) (a μ y : ℝ) (w : Fin K → ℝ) (s : ℝ) :
    HasDerivAt (fun t => Real.log (qD a μ y w t / K)) (qN a μ y w s / qD a μ y w s) s := by
  have hKne : (K : ℝ) ≠ 0 := by exact_mod_cast hK.ne'
  have h := ((hasDerivAt_qD a μ y w s).div_const (K : ℝ)).log
    (div_pos (qD_pos hK a μ y w s) (by exact_mod_cast hK)).ne'
  refine h.congr_deriv ?_
  field_simp

lemma hasDerivAt_ratio (hK : 0 < K) (a μ y : ℝ) (w : Fin K → ℝ) (s : ℝ) :
    HasDerivAt (fun t => qN a μ y w t / qD a μ y w t)
      ((qM a μ y w s * qD a μ y w s - qN a μ y w s * qN a μ y w s) / qD a μ y w s ^ 2) s :=
  (hasDerivAt_qN a μ y w s).fun_div (hasDerivAt_qD a μ y w s) (qD_pos hK a μ y w s).ne'

variable {ι Ω : Type*} [Fintype ι] [Fintype Ω]

/-- The smooth objective of the Gaussian reader, as a function of the noise scale `σ`. -/
noncomputable def quadObj (P : ι → ℝ) (ρ : Ω → ℝ) (y : ι → ℝ) (w : Ω → Fin K → ℝ)
    (a μ : ℝ) (σ : ℝ) : ℝ :=
  ∑ i, P i * ∑ o, ρ o * Real.log (qD a μ (y i) (w o) σ / K)

lemma quadObj_eq_smoothObj (P : ι → ℝ) (ρ : Ω → ℝ) (y : ι → ℝ) (w : Ω → Fin K → ℝ)
    (a μ σ : ℝ) :
    quadObj P ρ y w a μ σ
      = smoothObj P ρ (fun i x => -a * (x - y i) ^ 2) (fun o k => μ + σ * w o k) := by
  unfold quadObj smoothObj qD
  rfl

/-- **What "matched" means.**  With its normaliser the Gaussian reader scores target `y` at
latent `μ` by `-a (μ - y)² + (1/2) log a` (up to a constant).  Averaged over the target, the
curvature that maximises this at `σ = 0` is `a = 1/(2V)`, `V = E (μ - y)²`: the reader's
predictive variance equals the residual spread.  A reader trained at `σ ≈ 0` sits here. -/
theorem matched_curvature_optimal (V : ℝ) (hV : 0 < V) (a : ℝ) (ha : 0 < a) :
    -a * V + Real.log a / 2 ≤ -(1 / (2 * V)) * V + Real.log (1 / (2 * V)) / 2 := by
  have hs : 0 < 1 / (2 * V) := by positivity
  have hq : 0 < a / (1 / (2 * V)) := div_pos ha hs
  have hlog := Real.log_le_sub_one_of_pos hq
  rw [Real.log_div ha.ne' hs.ne'] at hlog
  have e : a / (1 / (2 * V)) = 2 * V * a := by field_simp
  rw [e] at hlog
  have e2 : (1 / (2 * V)) * V = 1 / 2 := by field_simp
  nlinarith

/-- The first derivative of the objective, as a function. -/
noncomputable def quadObj1 (P : ι → ℝ) (ρ : Ω → ℝ) (y : ι → ℝ) (w : Ω → Fin K → ℝ)
    (a μ : ℝ) (σ : ℝ) : ℝ :=
  ∑ i, P i * ∑ o, ρ o * (qN a μ (y i) (w o) σ / qD a μ (y i) (w o) σ)

lemma hasDerivAt_quadObj (hK : 0 < K) (P : ι → ℝ) (ρ : Ω → ℝ) (y : ι → ℝ)
    (w : Ω → Fin K → ℝ) (a μ σ : ℝ) :
    HasDerivAt (quadObj P ρ y w a μ) (quadObj1 P ρ y w a μ σ) σ := by
  unfold quadObj quadObj1
  exact HasDerivAt.fun_sum (fun i _ => (HasDerivAt.fun_sum (fun o _ =>
    (hasDerivAt_logmean hK a μ (y i) (w o) σ).const_mul (ρ o))).const_mul (P i))

lemma deriv_quadObj (hK : 0 < K) (P : ι → ℝ) (ρ : Ω → ℝ) (y : ι → ℝ) (w : Ω → Fin K → ℝ)
    (a μ : ℝ) : deriv (quadObj P ρ y w a μ) = quadObj1 P ρ y w a μ := by
  funext σ
  exact (hasDerivAt_quadObj hK P ρ y w a μ σ).deriv

lemma hasDerivAt_quadObj1 (hK : 0 < K) (P : ι → ℝ) (ρ : Ω → ℝ) (y : ι → ℝ)
    (w : Ω → Fin K → ℝ) (a μ σ : ℝ) :
    HasDerivAt (quadObj1 P ρ y w a μ)
      (∑ i, P i * ∑ o, ρ o * ((qM a μ (y i) (w o) σ * qD a μ (y i) (w o) σ
        - qN a μ (y i) (w o) σ * qN a μ (y i) (w o) σ) / qD a μ (y i) (w o) σ ^ 2)) σ := by
  unfold quadObj1
  exact HasDerivAt.fun_sum (fun i _ => (HasDerivAt.fun_sum (fun o _ =>
    (hasDerivAt_ratio hK a μ (y i) (w o) σ).const_mul (ρ o))).const_mul (P i))

/-- The within-draw mean `m₁ = (1/K) ∑ wₖ`. -/
noncomputable def m1 (w : Fin K → ℝ) : ℝ := (∑ k, w k) / K

/-- The within-draw second moment `m₂ = (1/K) ∑ wₖ²`. -/
noncomputable def m2 (w : Fin K → ℝ) : ℝ := (∑ k, w k ^ 2) / K

/-- The per-draw values at `σ = 0`. -/
lemma q_at_zero (hK : 0 < K) (a μ y : ℝ) (w : Fin K → ℝ) :
    qN a μ y w 0 / qD a μ y w 0 = -2 * a * (μ - y) * m1 w
      ∧ (qM a μ y w 0 * qD a μ y w 0 - qN a μ y w 0 * qN a μ y w 0) / qD a μ y w 0 ^ 2
        = (4 * a ^ 2 * (μ - y) ^ 2 - 2 * a) * m2 w - 4 * a ^ 2 * (μ - y) ^ 2 * m1 w ^ 2 := by
  have hKne : (K : ℝ) ≠ 0 := by exact_mod_cast hK.ne'
  set E := Real.exp (-a * (μ - y) ^ 2) with hE
  have hEpos : 0 < E := Real.exp_pos _
  have hD : qD a μ y w 0 = K * E := by
    unfold qD; simp [hE]
  have hN : qN a μ y w 0 = E * (-2 * a * (μ - y)) * ∑ k, w k := by
    unfold qN
    simp only [zero_mul, add_zero, Finset.mul_sum]
    apply Finset.sum_congr rfl; intro k _; rw [← hE]; ring
  have hM : qM a μ y w 0 = E * (4 * a ^ 2 * (μ - y) ^ 2 - 2 * a) * ∑ k, w k ^ 2 := by
    unfold qM
    simp only [zero_mul, add_zero, Finset.mul_sum]
    apply Finset.sum_congr rfl; intro k _; rw [← hE]; ring
  rw [hD, hN, hM]
  unfold m1 m2
  constructor
  · field_simp
  · field_simp
    ring

/-- **The first derivative at `σ = 0`.** -/
theorem quadObj_deriv_zero (hK : 0 < K) (P : ι → ℝ) (ρ : Ω → ℝ) (y : ι → ℝ)
    (w : Ω → Fin K → ℝ) (a μ : ℝ) :
    deriv (quadObj P ρ y w a μ) 0
      = -2 * a * (∑ i, P i * (μ - y i)) * (∑ o, ρ o * m1 (w o)) := by
  rw [deriv_quadObj hK]
  unfold quadObj1
  simp_rw [fun i o => (q_at_zero hK a μ (y i) (w o)).1]
  have hi : ∀ i, ∑ o, ρ o * (-2 * a * (μ - y i) * m1 (w o))
      = (-2 * a * (μ - y i)) * ∑ o, ρ o * m1 (w o) := by
    intro i
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl; intro o _; ring
  simp_rw [hi]
  generalize ∑ o, ρ o * m1 (w o) = M
  rw [Finset.mul_sum, Finset.sum_mul]
  apply Finset.sum_congr rfl; intro i _; ring

/-- **The second derivative at `σ = 0`, exactly.**
`J''(0) = 4 a² V · spread - 2 a · m₂`, with `V = ∑ P (μ - y)²`, `m₂ = ∑ ρ m₂(w o)` and
`spread = ∑ ρ (m₂(w o) - m₁(w o)²)`.  The first term is the gain of the mixture (the
reader's scores disagreeing across the `K` samples), the second the curvature cost. -/
theorem quadObj_deriv2_zero (hK : 0 < K) (P : ι → ℝ) (hP1 : ∑ i, P i = 1) (ρ : Ω → ℝ)
    (y : ι → ℝ) (w : Ω → Fin K → ℝ) (a μ : ℝ) :
    deriv (deriv (quadObj P ρ y w a μ)) 0
      = 4 * a ^ 2 * (∑ i, P i * (μ - y i) ^ 2) * (∑ o, ρ o * (m2 (w o) - m1 (w o) ^ 2))
        - 2 * a * (∑ o, ρ o * m2 (w o)) := by
  rw [deriv_quadObj hK, (hasDerivAt_quadObj1 hK P ρ y w a μ 0).deriv]
  simp_rw [fun i o => (q_at_zero hK a μ (y i) (w o)).2]
  have inner : ∀ i, ∑ o, ρ o * ((4 * a ^ 2 * (μ - y i) ^ 2 - 2 * a) * m2 (w o)
      - 4 * a ^ 2 * (μ - y i) ^ 2 * m1 (w o) ^ 2)
      = 4 * a ^ 2 * (μ - y i) ^ 2 * (∑ o, ρ o * (m2 (w o) - m1 (w o) ^ 2))
        - 2 * a * (∑ o, ρ o * m2 (w o)) := by
    intro i
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl; intro o _; ring
  simp_rw [inner]
  generalize ∑ o, ρ o * (m2 (w o) - m1 (w o) ^ 2) = Sp
  generalize ∑ o, ρ o * m2 (w o) = Sm
  have e : ∑ i, P i * (4 * a ^ 2 * (μ - y i) ^ 2 * Sp - 2 * a * Sm)
      = 4 * a ^ 2 * Sp * (∑ i, P i * (μ - y i) ^ 2) - 2 * a * Sm * (∑ i, P i) := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl; intro i _; ring
  rw [e, hP1]
  ring

lemma quadObj_continuousAt (hK : 0 < K) (P : ι → ℝ) (ρ : Ω → ℝ) (y : ι → ℝ)
    (w : Ω → Fin K → ℝ) (a μ : ℝ) : ContinuousAt (quadObj P ρ y w a μ) 0 :=
  (hasDerivAt_quadObj hK P ρ y w a μ 0).continuousAt

/-- **Collapse.**  With centred draws (`∑ ρ m₁ = 0`) and `2 a V · spread < m₂`, `σ = 0` is a
local maximum of the smooth objective: a learned noise scale is pushed back to zero. -/
theorem quadObj_isLocalMax (hK : 0 < K) (P : ι → ℝ) (hP1 : ∑ i, P i = 1) (ρ : Ω → ℝ)
    (y : ι → ℝ) (w : Ω → Fin K → ℝ) (a μ : ℝ) (ha : 0 < a)
    (hc : ∑ o, ρ o * m1 (w o) = 0)
    (hcond : 2 * a * (∑ i, P i * (μ - y i) ^ 2) * (∑ o, ρ o * (m2 (w o) - m1 (w o) ^ 2))
      < ∑ o, ρ o * m2 (w o)) :
    IsLocalMax (quadObj P ρ y w a μ) 0 := by
  apply isLocalMax_of_deriv_deriv_neg _ _ (quadObj_continuousAt hK P ρ y w a μ)
  · rw [quadObj_deriv2_zero hK P hP1 ρ y w a μ]
    nlinarith
  · rw [quadObj_deriv_zero hK P ρ y w a μ, hc, mul_zero]

/-- **Escape.**  With centred draws and `2 a V · spread > m₂` (a reader sharper than the
threshold), `σ = 0` is a local minimum: the noise scale grows. -/
theorem quadObj_isLocalMin (hK : 0 < K) (P : ι → ℝ) (hP1 : ∑ i, P i = 1) (ρ : Ω → ℝ)
    (y : ι → ℝ) (w : Ω → Fin K → ℝ) (a μ : ℝ) (ha : 0 < a)
    (hc : ∑ o, ρ o * m1 (w o) = 0)
    (hcond : ∑ o, ρ o * m2 (w o)
      < 2 * a * (∑ i, P i * (μ - y i) ^ 2) * (∑ o, ρ o * (m2 (w o) - m1 (w o) ^ 2))) :
    IsLocalMin (quadObj P ρ y w a μ) 0 := by
  apply isLocalMin_of_deriv_deriv_pos _ _ (quadObj_continuousAt hK P ρ y w a μ)
  · rw [quadObj_deriv2_zero hK P hP1 ρ y w a μ]
    nlinarith
  · rw [quadObj_deriv_zero hK P ρ y w a μ, hc, mul_zero]

end Quadratic

/-! ## Independent two-point draws: `spread = 1 - 1/K` -/

section IID

variable {K : ℕ}

/-- `∑ₒ ∏ₘ f m (o m) = ∏ₘ ∑_b f m b` over all sign patterns. -/
lemma sum_prod_signs (f : Fin K → Bool → ℝ) :
    ∑ o : Fin K → Bool, ∏ m, f m (o m) = ∏ m, ∑ b, f m b :=
  (Fintype.prod_sum f).symm

lemma sum_sgn_bool : ∑ b : Bool, sgn b = 0 := by simp [sgn]

lemma sgn_sq (b : Bool) : sgn b ^ 2 = 1 := by cases b <;> simp [sgn]

/-- One coordinate of a uniform sign pattern has mean zero. -/
lemma sum_sgn_coord (k : Fin K) : ∑ o : Fin K → Bool, sgn (o k) = 0 := by
  have h : ∀ o : Fin K → Bool, sgn (o k) = ∏ m, (if m = k then sgn (o m) else 1) := by
    intro o
    rw [Finset.prod_ite_eq']
    simp
  simp_rw [h]
  rw [sum_prod_signs (fun m b => if m = k then sgn b else 1)]
  apply Finset.prod_eq_zero (mem_univ k)
  simp [sgn]

/-- Two different coordinates are uncorrelated. -/
lemma sum_sgn_mul_coord (k l : Fin K) (hkl : k ≠ l) :
    ∑ o : Fin K → Bool, sgn (o k) * sgn (o l) = 0 := by
  have h : ∀ o : Fin K → Bool, sgn (o k) * sgn (o l)
      = ∏ m, ((if m = k then sgn (o m) else 1) * (if m = l then sgn (o m) else 1)) := by
    intro o
    rw [Finset.prod_mul_distrib, Finset.prod_ite_eq', Finset.prod_ite_eq']
    simp
  simp_rw [h]
  rw [sum_prod_signs (fun m b => (if m = k then sgn b else 1) * (if m = l then sgn b else 1))]
  apply Finset.prod_eq_zero (mem_univ k)
  simp [hkl, sgn]

/-- **`K` iid fair signs: the within-draw sum has mean zero.** -/
theorem iid_sum_sgn : ∑ o : Fin K → Bool, ∑ k, sgn (o k) = 0 := by
  rw [Finset.sum_comm]
  exact Finset.sum_eq_zero (fun k _ => sum_sgn_coord k)

/-- **`K` iid fair signs: the within-draw sum has second moment `K`** (summed over the `2^K`
patterns: `K · 2^K`). -/
theorem iid_sum_sgn_sq : ∑ o : Fin K → Bool, (∑ k, sgn (o k)) ^ 2 = K * 2 ^ K := by
  have h : ∀ o : Fin K → Bool, (∑ k, sgn (o k)) ^ 2 = ∑ k, ∑ l, sgn (o k) * sgn (o l) := by
    intro o
    rw [sq, Finset.sum_mul_sum]
  simp_rw [h]
  rw [Finset.sum_comm]
  have hk : ∀ k : Fin K, ∑ o : Fin K → Bool, ∑ l, sgn (o k) * sgn (o l) = 2 ^ K := by
    intro k
    rw [Finset.sum_comm]
    rw [Finset.sum_eq_single k]
    · simp_rw [← sq, sgn_sq]
      simp [Finset.card_univ, Fintype.card_bool, Fintype.card_fin]
    · intro l _ hlk
      exact sum_sgn_mul_coord k l (Ne.symm hlk)
    · intro h; exact absurd (mem_univ k) h
  simp_rw [hk]
  simp

/-- The iid two-point draw: `ρ = 2^{-K}` on each sign pattern, `w o k = sgn (o k)`. -/
noncomputable def iidρ (_o : Fin K → Bool) : ℝ := 1 / 2 ^ K

/-- The iid draw's sample positions. -/
def iidw (o : Fin K → Bool) (k : Fin K) : ℝ := sgn (o k)

lemma iid_m2 (hK : 0 < K) (o : Fin K → Bool) : m2 (iidw o) = 1 := by
  have hKne : (K : ℝ) ≠ 0 := by exact_mod_cast hK.ne'
  unfold m2 iidw
  simp_rw [sgn_sq]
  simp [hKne]

lemma iid_stats (hK : 0 < K) :
    (∑ o, iidρ o * m1 (iidw (K := K) o) = 0)
      ∧ (∑ o, iidρ o * m2 (iidw (K := K) o) = 1)
      ∧ (∑ o, iidρ o * (m2 (iidw (K := K) o) - m1 (iidw o) ^ 2) = 1 - 1 / K) := by
  have hKne : (K : ℝ) ≠ 0 := by exact_mod_cast hK.ne'
  have h2 : (2 : ℝ) ^ K ≠ 0 := by positivity
  have hcard : ∑ _o : Fin K → Bool, (1 : ℝ) / 2 ^ K = 1 := by
    simp [Finset.card_univ, Fintype.card_bool, Fintype.card_fin]
  refine ⟨?_, ?_, ?_⟩
  · unfold iidρ m1 iidw
    rw [show (∑ o : Fin K → Bool, 1 / 2 ^ K * ((∑ k, sgn (o k)) / K))
        = (1 / 2 ^ K / K) * ∑ o : Fin K → Bool, ∑ k, sgn (o k) by
      rw [Finset.mul_sum]; apply Finset.sum_congr rfl; intro o _; ring]
    rw [iid_sum_sgn, mul_zero]
  · simp_rw [iid_m2 hK, mul_one]
    exact hcard
  · simp_rw [iid_m2 hK]
    unfold iidρ m1 iidw
    rw [show (∑ o : Fin K → Bool, 1 / 2 ^ K * (1 - ((∑ k, sgn (o k)) / K) ^ 2))
        = ∑ _o : Fin K → Bool, (1 : ℝ) / 2 ^ K
          - (1 / 2 ^ K / K ^ 2) * ∑ o : Fin K → Bool, (∑ k, sgn (o k)) ^ 2 by
      rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl; intro o _; field_simp]
    rw [hcard, iid_sum_sgn_sq]
    field_simp

variable {ι : Type*} [Fintype ι]

/-- **The matched reader collapses independent noise at every `K`.**  `K` iid two-point
draws; a Gaussian reader whose curvature is matched to the target spread, `a = 1/(2V)` with
`V = ∑ P (μ - y)² > 0` (the reader's own optimum at `σ = 0`).  Then `σ = 0` is a local
maximum of the multi-sample bound for every `K ≥ 1`.  The mixture's gain is
`(1 - 1/K)` of the cost, never all of it. -/
theorem iid_matched_isLocalMax (hK : 0 < K) (P : ι → ℝ) (hP1 : ∑ i, P i = 1) (y : ι → ℝ)
    (μ : ℝ) (hV : 0 < ∑ i, P i * (μ - y i) ^ 2) :
    IsLocalMax (quadObj P iidρ y (iidw (K := K)) (1 / (2 * ∑ i, P i * (μ - y i) ^ 2)) μ) 0 := by
  obtain ⟨h1, h2, h3⟩ := iid_stats (K := K) hK
  have hKpos : (0 : ℝ) < K := by exact_mod_cast hK
  apply quadObj_isLocalMax hK P hP1 iidρ y iidw _ μ (by positivity) h1
  rw [h2, h3]
  have : 2 * (1 / (2 * ∑ i, P i * (μ - y i) ^ 2)) * (∑ i, P i * (μ - y i) ^ 2) = 1 := by
    field_simp
  rw [this, one_mul]
  have : 0 < 1 / (K : ℝ) := by positivity
  linarith

/-- The second derivative for `K` iid draws: `J''(0) = 2a (2aV(1 - 1/K) - 1)`. -/
theorem iid_deriv2 (hK : 0 < K) (P : ι → ℝ) (hP1 : ∑ i, P i = 1) (y : ι → ℝ) (a μ : ℝ) :
    deriv (deriv (quadObj P iidρ y (iidw (K := K)) a μ)) 0
      = 2 * a * (2 * a * (∑ i, P i * (μ - y i) ^ 2) * (1 - 1 / K) - 1) := by
  obtain ⟨_, h2, h3⟩ := iid_stats (K := K) hK
  rw [quadObj_deriv2_zero hK P hP1 iidρ y iidw a μ, h2, h3]
  ring

/-- **A sharp reader lets independent noise grow.**  If `2 a V (1 - 1/K) > 1`, `σ = 0` is a
local minimum. -/
theorem iid_sharp_isLocalMin (hK : 0 < K) (P : ι → ℝ) (hP1 : ∑ i, P i = 1) (y : ι → ℝ)
    (a μ : ℝ) (ha : 0 < a) (hsharp : 1 < 2 * a * (∑ i, P i * (μ - y i) ^ 2) * (1 - 1 / K)) :
    IsLocalMin (quadObj P iidρ y (iidw (K := K)) a μ) 0 := by
  obtain ⟨h1, h2, h3⟩ := iid_stats (K := K) hK
  apply quadObj_isLocalMin hK P hP1 iidρ y iidw a μ ha h1
  rw [h2, h3]
  linarith

/-- A positive second derivative at a critical point rules out a local maximum. -/
lemma not_isLocalMax_of_deriv2_pos {f : ℝ → ℝ} (hdiff : Differentiable ℝ f)
    (hd : deriv f 0 = 0) (h2 : 0 < deriv (deriv f) 0) : ¬ IsLocalMax f 0 := by
  intro hmax
  have hs := eventually_nhdsWithin_sign_eq_of_deriv_pos (f := deriv f) (x₀ := 0) h2 hd
  have hpos : ∀ᶠ x in nhdsWithin (0 : ℝ) (Set.Ioi 0), 0 < deriv f x := by
    filter_upwards [nhdsWithin_le_nhds hs, self_mem_nhdsWithin] with x hx (hx0 : 0 < x)
    rwa [sub_zero, sign_pos hx0, sign_eq_one_iff] at hx
  obtain ⟨u, hu, hsub⟩ := mem_nhdsGT_iff_exists_Ioo_subset.1 hpos
  have hgt : ∀ᶠ x in nhdsWithin (0 : ℝ) (Set.Ioi 0), f 0 < f x := by
    filter_upwards [Ioo_mem_nhdsGT (Set.mem_Ioi.1 hu)] with x hx
    have hmono : StrictMonoOn f (Set.Icc 0 x) := strictMonoOn_of_deriv_pos (convex_Icc 0 x)
      hdiff.continuous.continuousOn (fun t ht => by
        rw [interior_Icc] at ht
        exact hsub ⟨ht.1, ht.2.trans hx.2⟩)
    exact hmono ⟨le_rfl, hx.1.le⟩ ⟨hx.1.le, le_rfl⟩ hx.1
  have hle : ∀ᶠ x in nhdsWithin (0 : ℝ) (Set.Ioi 0), f x ≤ f 0 :=
    hmax.filter_mono nhdsWithin_le_nhds
  obtain ⟨x, h1, h3⟩ := (hgt.and hle).exists
  linarith

/-- A negative second derivative at a critical point gives a STRICT local maximum:
every nearby point other than `0` scores strictly less. -/
lemma strict_localMax_of_deriv2_neg {f : ℝ → ℝ} (hdiff : Differentiable ℝ f)
    (hd : deriv f 0 = 0) (h2 : deriv (deriv f) 0 < 0) :
    ∀ᶠ x in nhdsWithin (0 : ℝ) {0}ᶜ, f x < f 0 := by
  have hs := eventually_nhdsWithin_sign_eq_of_deriv_neg (f := deriv f) (x₀ := 0) h2 hd
  have hneg : ∀ᶠ x in nhdsWithin (0 : ℝ) (Set.Ioi 0), deriv f x < 0 := by
    filter_upwards [nhdsWithin_le_nhds hs, self_mem_nhdsWithin] with x hx (hx0 : 0 < x)
    rwa [zero_sub, sign_neg (neg_neg_of_pos hx0), sign_eq_neg_one_iff] at hx
  have hpos : ∀ᶠ x in nhdsWithin (0 : ℝ) (Set.Iio 0), 0 < deriv f x := by
    filter_upwards [nhdsWithin_le_nhds hs, self_mem_nhdsWithin] with x hx (hx0 : x < 0)
    rwa [zero_sub, sign_pos (neg_pos.2 hx0), sign_eq_one_iff] at hx
  obtain ⟨u, hu, hsubu⟩ := mem_nhdsGT_iff_exists_Ioo_subset.1 hneg
  obtain ⟨l, hl, hsubl⟩ := mem_nhdsLT_iff_exists_Ioo_subset.1 hpos
  have right : ∀ᶠ x in nhdsWithin (0 : ℝ) (Set.Ioi 0), f x < f 0 := by
    filter_upwards [Ioo_mem_nhdsGT (Set.mem_Ioi.1 hu)] with x hx
    have hanti : StrictAntiOn f (Set.Icc 0 x) := strictAntiOn_of_deriv_neg (convex_Icc 0 x)
      hdiff.continuous.continuousOn (fun t ht => by
        rw [interior_Icc] at ht
        exact hsubu ⟨ht.1, ht.2.trans hx.2⟩)
    exact hanti ⟨le_rfl, hx.1.le⟩ ⟨hx.1.le, le_rfl⟩ hx.1
  have left : ∀ᶠ x in nhdsWithin (0 : ℝ) (Set.Iio 0), f x < f 0 := by
    filter_upwards [Ioo_mem_nhdsLT (Set.mem_Iio.1 hl)] with x hx
    have hmono : StrictMonoOn f (Set.Icc x 0) := strictMonoOn_of_deriv_pos (convex_Icc x 0)
      hdiff.continuous.continuousOn (fun t ht => by
        rw [interior_Icc] at ht
        exact hsubl ⟨hx.1.trans ht.1, ht.2⟩)
    exact hmono ⟨le_rfl, hx.2.le⟩ ⟨hx.2.le, le_rfl⟩ hx.2
  rw [← nhdsLT_sup_nhdsGT, Filter.eventually_sup]
  exact ⟨left, right⟩

lemma quadObj_differentiable (hK : 0 < K) {ι Ω : Type*} [Fintype ι] [Fintype Ω]
    (P : ι → ℝ) (ρ : Ω → ℝ) (y : ι → ℝ) (w : Ω → Fin K → ℝ) (a μ : ℝ) :
    Differentiable ℝ (quadObj P ρ y w a μ) :=
  fun σ => (hasDerivAt_quadObj hK P ρ y w a μ σ).differentiableAt

/-- **A sharp reader makes `σ = 0` NOT a maximiser** (the strict form of
`iid_sharp_isLocalMin`): arbitrarily small noise scales score strictly higher. -/
theorem iid_sharp_not_isLocalMax (hK : 0 < K) {ι : Type*} [Fintype ι] (P : ι → ℝ)
    (hP1 : ∑ i, P i = 1) (y : ι → ℝ) (a μ : ℝ) (ha : 0 < a)
    (hsharp : 1 < 2 * a * (∑ i, P i * (μ - y i) ^ 2) * (1 - 1 / K)) :
    ¬ IsLocalMax (quadObj P iidρ y (iidw (K := K)) a μ) 0 := by
  obtain ⟨h1, _, _⟩ := iid_stats (K := K) hK
  apply not_isLocalMax_of_deriv2_pos (quadObj_differentiable hK P iidρ y iidw a μ)
  · rw [quadObj_deriv_zero hK P iidρ y iidw a μ, h1, mul_zero]
  · rw [iid_deriv2 hK P hP1 y a μ]
    have : 0 < 2 * a * (∑ i, P i * (μ - y i) ^ 2) * (1 - 1 / K) - 1 := by linarith
    positivity

/-- **The matched reader collapses independent noise, strictly.**  Same setting as
`iid_matched_isLocalMax`: every small nonzero noise scale scores strictly below `σ = 0`,
since `J''(0) = -2a/K < 0`. -/
theorem iid_matched_strict (hK : 0 < K) {ι : Type*} [Fintype ι] (P : ι → ℝ)
    (hP1 : ∑ i, P i = 1) (y : ι → ℝ) (μ : ℝ) (hV : 0 < ∑ i, P i * (μ - y i) ^ 2) :
    ∀ᶠ σ in nhdsWithin (0 : ℝ) {0}ᶜ,
      quadObj P iidρ y (iidw (K := K)) (1 / (2 * ∑ i, P i * (μ - y i) ^ 2)) μ σ
        < quadObj P iidρ y (iidw (K := K)) (1 / (2 * ∑ i, P i * (μ - y i) ^ 2)) μ 0 := by
  obtain ⟨h1, _, _⟩ := iid_stats (K := K) hK
  have hKpos : (0 : ℝ) < K := by exact_mod_cast hK
  apply strict_localMax_of_deriv2_neg (quadObj_differentiable hK P iidρ y iidw _ μ)
  · rw [quadObj_deriv_zero hK P iidρ y iidw _ μ, h1, mul_zero]
  · rw [iid_deriv2 hK P hP1 y _ μ]
    have e : 2 * (1 / (2 * ∑ i, P i * (μ - y i) ^ 2)) * (∑ i, P i * (μ - y i) ^ 2) = 1 := by
      field_simp
    rw [e]
    have : 0 < 1 / (2 * ∑ i, P i * (μ - y i) ^ 2) := by positivity
    have : 0 < 1 / (K : ℝ) := by positivity
    nlinarith

/-- **T1 is false as a global claim once the target varies.**  Two equally likely targets
at `±1`, the latent mean at `0`, two iid draws, a reader of curvature `a = 2` (predictive
std `1/2`, sharper than the target spread `1`): `σ = 0` is not a local maximum of the
multi-sample bound. -/
theorem sharp_reader_counterexample :
    ¬ IsLocalMax (quadObj (fun _ : Bool => (1 / 2 : ℝ)) iidρ (fun b => sgn b) (iidw (K := 2))
      2 0) 0 := by
  apply iid_sharp_not_isLocalMax (by norm_num) _ (by simp) _ 2 0 (by norm_num)
  simp [sgn]
  norm_num

end IID

/-! ## An enumerated antithetic pair -/

section Enumerated

variable {ι : Type*} [Fintype ι]

/-- Two codes at `±1`, both always present: one draw, `K = 2`. -/
def pairW (_o : Unit) : Fin 2 → ℝ := ![1, -1]

/-- **Enumeration removes the `1/K` shrink.**  For the antithetic pair,
`J''(0) = 2a(2aV - 1)`: exactly zero at the matched reader `a = 1/(2V)`, positive for any
sharper reader.  Enumerating the codes turns the smooth bound into the exact mixture
likelihood, so spread is neutral to second order at the matched reader instead of a cost. -/
theorem enumerated_pair_deriv2 (P : ι → ℝ) (hP1 : ∑ i, P i = 1) (y : ι → ℝ) (a μ : ℝ) :
    deriv (deriv (quadObj P (fun _ : Unit => (1 : ℝ)) y pairW a μ)) 0
      = 2 * a * (2 * a * (∑ i, P i * (μ - y i) ^ 2) - 1) := by
  rw [quadObj_deriv2_zero (by norm_num) P hP1 _ y pairW a μ]
  simp [m1, m2, pairW, Fin.sum_univ_two]
  ring

end Enumerated

end TulExploration
