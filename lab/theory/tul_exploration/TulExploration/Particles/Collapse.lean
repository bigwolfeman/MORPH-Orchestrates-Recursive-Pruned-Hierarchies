/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent (ParticleProver)
-/
import Mathlib.Topology.MetricSpace.Contracting
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import TulExploration.Washout
import TulExploration.MapCause

/-!
# P1, P2: a contractive coarse loop collapses its particles; basins need expansion

## Model

`G z` is one coarse pass for context `z`, `x ↦ F(D ⊙ x + Δ ⊙ B z)` in the Parcae testbed.
Here it is any self-map `G` of a metric space.  `K` particles are `K` starts
`x : Fin K → E`; particle `k` after `T` passes is `G^[T] (x k)`.  A *set read* is any
function `R` of the `K` exit states; it is `Lr`-Lipschitz in the sup distance.  Softmax
attention over the coarse memory is `attn s v = (∑ eˢ)⁻¹ ∑ eˢ v`.

## Main results (P1, collapse under contraction)

* `particles_spread` : if `G` is `ρ`-Lipschitz, every pairwise distance after `T` passes is
  at most `ρᵀ` times the initial spread bound.  This is `washout` applied to particles.
* `particles_to_fixedPoint` : if `G` is a contraction (`ρ < 1`) on a complete space, every
  particle is within `C ρᵀ / (1 - ρ)` of the ONE fixed point (Banach, from Mathlib).
* **`set_read_gap`** : any `Lr`-Lipschitz set read of the `K` exit states is within
  `Lr C ρᵀ / (1 - ρ)` of the read of `K` copies of the fixed point.
* `mean_read_const` : a mean-pooled read of `K` copies equals the read of one.
* **`attn_duplicates`** : softmax attention over `K` copies of one key (logit `s₀ + b`) and
  any other keys equals attention over ONE copy with logit `s₀ + b + log K`.  So collapsed
  particles are NOT the read of one particle under plain attention: they are one particle
  up-weighted by `log K`.
* **`attn_duplicates_corrected`** : with the bias `b = -log K` the read of `K` collapsed
  particles equals the read of one, for every `K`.

## Main results (P2, necessity of non-contraction)

* `fixed_points_lip_ge_one` : two distinct fixed points force every Lipschitz constant of
  `G` on a set holding both to be at least `1`.  `no_contraction_two_fixed` : so no `ρ < 1`.
* **`parcae_fixed_points`** : for `x ↦ F(D x + b)` two distinct fixed points force
  `1 ≤ L_F · d`, where `d` bounds the decay `‖D v‖ ≤ d ‖v‖`.  With `d < 1` the blocks must
  EXPAND, `L_F ≥ 1/d > 1`, along the inter-basin direction.
* `residual_fixed_points` : for MORPH's residual form `h ↦ D h + u h + c`, two distinct
  fixed points force `‖x - y‖ ≤ ‖D(x - y)‖ + L_u ‖x - y‖` (reuses `gain_sandwich`).
* **`stable_pair_forces_expansion`** : a differentiable 1-D pass with two fixed points
  `x₁ < x₂`, both locally stable (`G' < 1` there), has `G'(c) > 1` at some `c ∈ (x₁, x₂)`.
-/

namespace TulExploration

open Finset

/-! ## P1: collapse under contraction -/

section Collapse

/-- **Particle spread shrinks like `ρᵀ`.** -/
theorem particles_spread {E : Type*} [PseudoMetricSpace E] (G : E → E) (ρ : ℝ) (hρ : 0 ≤ ρ)
    (hG : ∀ x y, dist (G x) (G y) ≤ ρ * dist x y) {K : ℕ} (x : Fin K → E) (δ : ℝ)
    (hδ : ∀ j k, dist (x j) (x k) ≤ δ) (T : ℕ) (j k : Fin K) :
    dist (G^[T] (x j)) (G^[T] (x k)) ≤ ρ ^ T * δ :=
  (washout G ρ hρ hG (x j) (x k) T).trans
    (mul_le_mul_of_nonneg_left (hδ j k) (pow_nonneg hρ T))

variable {E : Type*} [MetricSpace E] [Nonempty E] [CompleteSpace E]

/-- **Every particle converges to the one fixed point.**  `C` bounds each particle's first
step, `dist (x k) (G (x k)) ≤ C`. -/
theorem particles_to_fixedPoint {ρ : NNReal} {G : E → E} (hG : ContractingWith ρ G)
    {K : ℕ} (x : Fin K → E) (C : ℝ) (hC : ∀ k, dist (x k) (G (x k)) ≤ C) (T : ℕ)
    (k : Fin K) :
    dist (G^[T] (x k)) (ContractingWith.fixedPoint G hG) ≤ C * (ρ : ℝ) ^ T / (1 - ρ) := by
  have h1 : (0 : ℝ) < 1 - ρ := by
    have : (ρ : ℝ) < 1 := by exact_mod_cast hG.1
    linarith
  have hb := ContractingWith.apriori_dist_iterate_fixedPoint_le hG (x k) T
  have hnn : (0 : ℝ) ≤ (ρ : ℝ) ^ T / (1 - ρ) := div_nonneg (pow_nonneg ρ.2 T) h1.le
  calc dist (G^[T] (x k)) (ContractingWith.fixedPoint G hG)
      ≤ dist (x k) (G (x k)) * (ρ : ℝ) ^ T / (1 - ρ) := hb
    _ = dist (x k) (G (x k)) * ((ρ : ℝ) ^ T / (1 - ρ)) := by ring
    _ ≤ C * ((ρ : ℝ) ^ T / (1 - ρ)) := mul_le_mul_of_nonneg_right (hC k) hnn
    _ = C * (ρ : ℝ) ^ T / (1 - ρ) := by ring

/-- **The set read of `K` contracted particles is the read of `K` copies of one point, up to
`O(ρᵀ)`.**  `R` is any read of the `K` exit states that is `Lr`-Lipschitz in the sup
distance. -/
theorem set_read_gap {ρ : NNReal} {G : E → E} (hG : ContractingWith ρ G) {K : ℕ}
    (x : Fin K → E) (C : ℝ) (hC : ∀ k, dist (x k) (G (x k)) ≤ C) (T : ℕ)
    (R : (Fin K → E) → ℝ) (Lr : ℝ)
    (hR : ∀ u v : Fin K → E, ∀ δ : ℝ, (∀ k, dist (u k) (v k) ≤ δ) → |R u - R v| ≤ Lr * δ) :
    |R (fun k => G^[T] (x k)) - R (fun _ => ContractingWith.fixedPoint G hG)|
      ≤ Lr * (C * (ρ : ℝ) ^ T / (1 - ρ)) :=
  hR _ _ _ (fun k => particles_to_fixedPoint hG x C hC T k)

end Collapse

/-- A mean-pooled read of `K ≥ 1` copies of one state is the read of that state. -/
theorem mean_read_const {E : Type*} {K : ℕ} (hK : 0 < K) (r : E → ℝ) (x : E) :
    (∑ _k : Fin K, r x) / K = r x := by
  have : (K : ℝ) ≠ 0 := by exact_mod_cast hK.ne'
  simp [Finset.sum_const, Finset.card_univ]
  field_simp

section Attention

variable {ι V : Type*} [Fintype ι] [AddCommGroup V] [Module ℝ V]

/-- Softmax attention: `(∑ eˢ)⁻¹ ∑ eˢ v`. -/
noncomputable def attn {κ : Type*} [Fintype κ] (s : κ → ℝ) (v : κ → V) : V :=
  (∑ i, Real.exp (s i))⁻¹ • ∑ i, Real.exp (s i) • v i

/-- **`K` collapsed particles are one particle with logit `+ log K`.**  The memory holds the
other keys `s, v` and `K` copies of the particle key (logit `s₀ + b`, value `v₀`).  Attention
over it equals attention over the other keys plus ONE copy with logit `s₀ + b + log K`. -/
theorem attn_duplicates (s : ι → ℝ) (v : ι → V) (s₀ b : ℝ) (v₀ : V) {K : ℕ} (hK : 0 < K) :
    attn (Sum.elim s (fun _ : Fin K => s₀ + b)) (Sum.elim v (fun _ : Fin K => v₀))
      = attn (fun o : Option ι => o.elim (s₀ + b + Real.log K) s)
          (fun o : Option ι => o.elim v₀ v) := by
  have hKR : (0 : ℝ) < K := by exact_mod_cast hK
  have he : Real.exp (s₀ + b + Real.log K) = K * Real.exp (s₀ + b) := by
    rw [Real.exp_add, Real.exp_log hKR]; ring
  unfold attn
  rw [Fintype.sum_sum_type, Fintype.sum_sum_type, Fintype.sum_option, Fintype.sum_option]
  simp only [Sum.elim_inl, Sum.elim_inr, Option.elim, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul, he]
  rw [add_comm (∑ x, Real.exp (s x)) _, add_comm (∑ x, Real.exp (s x) • v x) _, mul_smul,
    Nat.cast_smul_eq_nsmul]

/-- **With the bias `-log K`, `K` collapsed particles read as one, for every `K`.** -/
theorem attn_duplicates_corrected (s : ι → ℝ) (v : ι → V) (s₀ : ℝ) (v₀ : V) {K : ℕ}
    (hK : 0 < K) :
    attn (Sum.elim s (fun _ : Fin K => s₀ + -Real.log K)) (Sum.elim v (fun _ : Fin K => v₀))
      = attn (fun o : Option ι => o.elim s₀ s) (fun o : Option ι => o.elim v₀ v) := by
  rw [attn_duplicates s v s₀ _ v₀ hK]
  congr 2
  funext o
  cases o <;> simp

end Attention

/-! ## P2: per-context diversity needs a non-contractive pass -/

section Necessity

/-- **Two distinct fixed points force Lipschitz constant `≥ 1`.** -/
theorem fixed_points_lip_ge_one {E : Type*} [MetricSpace E] (G : E → E) (L : ℝ) (x y : E)
    (hx : G x = x) (hy : G y = y) (hxy : x ≠ y) (hL : dist (G x) (G y) ≤ L * dist x y) :
    1 ≤ L := by
  rw [hx, hy] at hL
  have hd : 0 < dist x y := dist_pos.2 hxy
  nlinarith

/-- **So a pass with two basin points is not a contraction on any set holding both.** -/
theorem no_contraction_two_fixed {E : Type*} [MetricSpace E] (G : E → E) (S : Set E)
    (x y : E) (hxS : x ∈ S) (hyS : y ∈ S) (hx : G x = x) (hy : G y = y) (hxy : x ≠ y)
    (ρ : ℝ) (hρ : ρ < 1) : ¬ ∀ a ∈ S, ∀ b ∈ S, dist (G a) (G b) ≤ ρ * dist a b := by
  intro h
  have := fixed_points_lip_ge_one G ρ x y hx hy hxy (h x hxS y hyS)
  linarith

/-- **Parcae form: the blocks must expand along the inter-basin direction.**  The pass is
`x ↦ F (D x + b)` with `F` `L`-Lipschitz and `D` additive (the decay `exp(-Δ ⊙ A)` is
diagonal).  Two fixed points `x`, `y` give `‖x - y‖ ≤ L ‖D (x - y)‖`. -/
theorem parcae_fixed_points {E : Type*} [SeminormedAddCommGroup E] (F : E → E) (D : E →+ E)
    (b : E) (L : ℝ) (hF : ∀ u w, ‖F u - F w‖ ≤ L * ‖u - w‖) (x y : E)
    (hx : F (D x + b) = x) (hy : F (D y + b) = y) : ‖x - y‖ ≤ L * ‖D (x - y)‖ := by
  have h := hF (D x + b) (D y + b)
  rw [hx, hy, add_sub_add_right_eq_sub, ← map_sub] at h
  exact h

/-- **With a decay bounded by `d`, two distinct fixed points need `L · d ≥ 1`.**  For Parcae,
`dᵢ = exp(-Δᵢ Aᵢ) < 1` whenever `Δᵢ Aᵢ > 0` (a hypothesis on the trained values, not checked
here), so `L ≥ 1/d > 1`. -/
theorem parcae_needs_expansion {E : Type*} [NormedAddCommGroup E] (F : E → E) (D : E →+ E)
    (b : E) (L d : ℝ) (hF : ∀ u w, ‖F u - F w‖ ≤ L * ‖u - w‖) (hL : 0 ≤ L)
    (hD : ∀ v, ‖D v‖ ≤ d * ‖v‖) (x y : E) (hx : F (D x + b) = x) (hy : F (D y + b) = y)
    (hxy : x ≠ y) : 1 ≤ L * d := by
  have h := parcae_fixed_points F D b L hF x y hx hy
  have h2 : L * ‖D (x - y)‖ ≤ L * (d * ‖x - y‖) := mul_le_mul_of_nonneg_left (hD _) hL
  have hpos : 0 < ‖x - y‖ := norm_pos_iff.2 (sub_ne_zero.2 hxy)
  nlinarith

/-- **MORPH residual form.**  For `h ↦ D h + u h + c` with `u` `L`-Lipschitz, two fixed points
satisfy `‖x - y‖ ≤ ‖D(x - y)‖ + L ‖x - y‖`.  Along a direction where `D` is the identity
(the 704 channels `DiagonalInjection` leaves alone) this asks nothing of the blocks; along
the context slice (`D = A ≈ 0.447`) it asks `L ≥ 1 - 0.447`. -/
theorem residual_fixed_points {E : Type*} [SeminormedAddCommGroup E] (D : E →+ E) (u : E → E)
    (c : E) (L : ℝ) (hu : ∀ x y, ‖u x - u y‖ ≤ L * ‖x - y‖) (x y : E)
    (hx : D x + u x + c = x) (hy : D y + u y + c = y) :
    ‖x - y‖ ≤ ‖D (x - y)‖ + L * ‖x - y‖ := by
  have h := (gain_sandwich D u c L hu y (x - y)).2
  rw [add_sub_cancel, hx, hy] at h
  exact h

/-- Below a locally stable fixed point's right side: `G y < y` just right of `x`. -/
lemma eventually_below_right {G : ℝ → ℝ} {g x : ℝ} (hG : HasDerivAt G g x) (hx : G x = x)
    (hg : g < 1) : ∀ᶠ y in nhdsWithin x (Set.Ioi x), G y < y := by
  have ht := hasDerivAt_iff_tendsto_slope.1 hG
  have hev : ∀ᶠ y in nhdsWithin x {x}ᶜ, slope G x y < 1 := ht.eventually_lt_const hg
  have hev' : ∀ᶠ y in nhdsWithin x (Set.Ioi x), slope G x y < 1 :=
    nhdsWithin_mono x (fun y hy => ne_of_gt (Set.mem_Ioi.1 hy)) hev
  filter_upwards [hev', self_mem_nhdsWithin] with y hy hyx
  rw [slope_def_field, hx] at hy
  have hpos : 0 < y - x := sub_pos.2 hyx
  rw [div_lt_one hpos] at hy
  linarith

/-- Above a locally stable fixed point's left side: `G y > y` just left of `x`. -/
lemma eventually_above_left {G : ℝ → ℝ} {g x : ℝ} (hG : HasDerivAt G g x) (hx : G x = x)
    (hg : g < 1) : ∀ᶠ y in nhdsWithin x (Set.Iio x), y < G y := by
  have ht := hasDerivAt_iff_tendsto_slope.1 hG
  have hev : ∀ᶠ y in nhdsWithin x {x}ᶜ, slope G x y < 1 := ht.eventually_lt_const hg
  have hev' : ∀ᶠ y in nhdsWithin x (Set.Iio x), slope G x y < 1 :=
    nhdsWithin_mono x (fun y hy => ne_of_lt (Set.mem_Iio.1 hy)) hev
  filter_upwards [hev', self_mem_nhdsWithin] with y hy hyx
  rw [slope_def_field, hx] at hy
  have hneg : y - x < 0 := sub_neg.2 hyx
  rw [div_lt_one_of_neg hneg] at hy
  linarith

/-- **Two locally stable basin points on a line force local expansion between them.**  A
differentiable pass `G : ℝ → ℝ` with fixed points `x₁ < x₂`, both with `G' < 1`, has a point
`c ∈ (x₁, x₂)` with `G'(c) > 1`.  (The double well `tanh(βx) + c` has it at the saddle.) -/
theorem stable_pair_forces_expansion {G : ℝ → ℝ} (hG : Differentiable ℝ G) {x₁ x₂ : ℝ}
    (h12 : x₁ < x₂) (h1 : G x₁ = x₁) (h2 : G x₂ = x₂) (hs1 : deriv G x₁ < 1)
    (hs2 : deriv G x₂ < 1) : ∃ c ∈ Set.Ioo x₁ x₂, 1 < deriv G c := by
  set m := (x₁ + x₂) / 2 with hm
  have hm1 : x₁ < m := by rw [hm]; linarith
  have hm2 : m < x₂ := by rw [hm]; linarith
  obtain ⟨a, ha, ha'⟩ := ((eventually_below_right (hG x₁).hasDerivAt h1 hs1).and
    (Ioo_mem_nhdsGT hm1)).exists
  obtain ⟨b, hb, hb'⟩ := ((eventually_above_left (hG x₂).hasDerivAt h2 hs2).and
    (Ioo_mem_nhdsLT hm2)).exists
  have hab : a < b := lt_trans ha'.2 hb'.1
  obtain ⟨c, hc, hc'⟩ := exists_hasDerivAt_eq_slope (fun t => G t - t)
    (fun t => deriv G t - 1) hab
    ((hG.continuous.sub continuous_id).continuousOn)
    (fun t _ => ((hG t).hasDerivAt.sub (hasDerivAt_id t)))
  refine ⟨c, ⟨lt_trans ha'.1 hc.1, lt_trans hc.2 hb'.2⟩, ?_⟩
  have hpos : 0 < ((G b - b) - (G a - a)) / (b - a) :=
    div_pos (by linarith) (sub_pos.2 hab)
  linarith

end Necessity

end TulExploration
