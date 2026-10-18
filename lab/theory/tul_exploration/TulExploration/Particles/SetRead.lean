/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent (ParticleProver)
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Field

/-!
# P4: the set read, its `log K` ceiling, and the sequential Bayes read

## Model

One span of `n` tokens and `K` particles.  `a k t > 0` is particle `k`'s likelihood of the
TRUE token `t` given the true prefix (teacher forcing): `p(y_t | y_{<t}, z_k)`.  A
*past-only weighting* is any `w t k` in the simplex that may depend on the tokens before `t`;
its per-token read of the span is `∑_t log ∑_k w t k · a k t`.  The *span mixture* is
`log ((1/K) ∑_k ∏_t a k t)`, the uniform mixture of the particles' span likelihoods.

## Main results

* `mixture_le_best`, **`best_le_mixture_add_logK`** : the span oracle (best particle) minus
  the uniform mixture lies in `[0, log K]`, per span and per draw.
* `post_sum_one`, **`post_past_only`** : the sequential Bayes weights (posterior over
  particles from a uniform prior, updated on the true prefix) sum to one and depend only on
  tokens already seen.
* **`seq_bayes_eq_span_mixture`** : the per-token read with those weights EQUALS the span
  mixture (the chain rule).  Training on the span mixture and deploying the token-by-token
  posterior read are the same objective: deploy-consistent.
* `seqRead_le_token_oracle` : any past-only weighting is at most the per-TOKEN oracle.
* **`switching_beats_span_oracle`** : a past-only weighting that switches particles between
  tokens can beat the span oracle, and so the span mixture, by an unbounded amount.  "Between
  the mixture and the oracle" is FALSE for a general past-only reader.
* `arith_pool_floor` : a probability mixture keeps the floor `w_k p_k(y)`.
* **`value_mix_loses_floor`** : attention that mixes VALUES, read by a linear head (logits
  mix, a normalised geometric pool), has no such floor: one particle that is confidently
  wrong drives the read to `-∞` while the probability mixture stays above `-log 4`.
-/

namespace TulExploration

open Finset

/-! ## The `log K` ceiling -/

/-- The uniform mixture is at most the best candidate. -/
theorem mixture_le_best {K : ℕ} (hK : 0 < K) (p : Fin K → ℝ) (hp : ∀ k, 0 < p k) (m : ℝ)
    (hm : ∀ k, p k ≤ m) : Real.log ((∑ k, p k) / K) ≤ Real.log m := by
  have hKR : (0 : ℝ) < K := by exact_mod_cast hK
  have : Nonempty (Fin K) := ⟨⟨0, hK⟩⟩
  have hS : 0 < ∑ k, p k := Finset.sum_pos (fun k _ => hp k) univ_nonempty
  apply Real.log_le_log (div_pos hS hKR)
  rw [div_le_iff₀ hKR]
  calc ∑ k, p k ≤ ∑ _k : Fin K, m := Finset.sum_le_sum (fun k _ => hm k)
    _ = m * K := by simp [Finset.sum_const, Finset.card_univ]; ring

/-- **The oracle beats the uniform mixture by at most `log K`.**  For every candidate `j`,
`log p_j ≤ log ((1/K) ∑ p) + log K`. -/
theorem best_le_mixture_add_logK {K : ℕ} (hK : 0 < K) (p : Fin K → ℝ) (hp : ∀ k, 0 < p k)
    (j : Fin K) : Real.log (p j) ≤ Real.log ((∑ k, p k) / K) + Real.log K := by
  have hKR : (0 : ℝ) < K := by exact_mod_cast hK
  have hS : 0 < ∑ k, p k := Finset.sum_pos (fun k _ => hp k) ⟨j, mem_univ j⟩
  rw [← Real.log_mul (div_pos hS hKR).ne' hKR.ne', div_mul_cancel₀ _ hKR.ne']
  exact Real.log_le_log (hp j) (Finset.single_le_sum (fun k _ => (hp k).le) (mem_univ j))

/-! ## The sequential Bayes read -/

section Sequential

variable {K : ℕ}

/-- Particle `k`'s likelihood of the first `t` true tokens. -/
noncomputable def cum (a : Fin K → ℕ → ℝ) (k : Fin K) (t : ℕ) : ℝ := ∏ s ∈ range t, a k s

/-- The posterior over particles before token `t` (uniform prior, true prefix). -/
noncomputable def post (a : Fin K → ℕ → ℝ) (t : ℕ) (k : Fin K) : ℝ :=
  cum a k t / ∑ j, cum a j t

/-- The per-token read of a weighting `w`. -/
noncomputable def seqRead (w : ℕ → Fin K → ℝ) (a : Fin K → ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ t ∈ range n, Real.log (∑ k, w t k * a k t)

lemma cum_pos (a : Fin K → ℕ → ℝ) (ha : ∀ k t, 0 < a k t) (k : Fin K) (t : ℕ) :
    0 < cum a k t := Finset.prod_pos (fun s _ => ha k s)

lemma sum_cum_pos (hK : 0 < K) (a : Fin K → ℕ → ℝ) (ha : ∀ k t, 0 < a k t) (t : ℕ) :
    0 < ∑ k, cum a k t := by
  have : Nonempty (Fin K) := ⟨⟨0, hK⟩⟩
  exact Finset.sum_pos (fun k _ => cum_pos a ha k t) univ_nonempty

/-- The posterior weights sum to one. -/
theorem post_sum_one (hK : 0 < K) (a : Fin K → ℕ → ℝ) (ha : ∀ k t, 0 < a k t) (t : ℕ) :
    ∑ k, post a t k = 1 := by
  unfold post
  rw [← Finset.sum_div, div_self (sum_cum_pos hK a ha t).ne']

/-- **The posterior is past-only**: it depends only on the likelihoods of tokens `s < t`. -/
theorem post_past_only (a a' : Fin K → ℕ → ℝ) (t : ℕ)
    (h : ∀ k, ∀ s < t, a k s = a' k s) : post a t = post a' t := by
  have hc : ∀ k, cum a k t = cum a' k t := fun k =>
    Finset.prod_congr rfl (fun s hs => h k s (Finset.mem_range.1 hs))
  funext k
  unfold post
  simp_rw [hc]

/-- One step of the posterior read: `∑ post · a = (∑ cum (t+1)) / (∑ cum t)`. -/
lemma post_step (a : Fin K → ℕ → ℝ) (t : ℕ) :
    ∑ k, post a t k * a k t = (∑ k, cum a k (t + 1)) / ∑ j, cum a j t := by
  unfold post cum
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro k _
  rw [Finset.prod_range_succ]
  ring

/-- **The sequential Bayes read IS the span mixture (chain rule).**  Reading the span token
by token, each token from the posterior over particles given the true prefix, scores exactly
`log ((1/K) ∑_k ∏_t a k t)`. -/
theorem seq_bayes_eq_span_mixture (hK : 0 < K) (a : Fin K → ℕ → ℝ) (ha : ∀ k t, 0 < a k t)
    (n : ℕ) : seqRead (post a) a n = Real.log ((∑ k, cum a k n) / K) := by
  have hKR : (0 : ℝ) < K := by exact_mod_cast hK
  have hstep : ∀ t, Real.log (∑ k, post a t k * a k t)
      = Real.log (∑ k, cum a k (t + 1)) - Real.log (∑ k, cum a k t) := by
    intro t
    rw [post_step, Real.log_div (sum_cum_pos hK a ha (t + 1)).ne'
      (sum_cum_pos hK a ha t).ne']
  unfold seqRead
  simp_rw [hstep]
  rw [Finset.sum_range_sub (fun t => Real.log (∑ k, cum a k t)) n,
    Real.log_div (sum_cum_pos hK a ha n).ne' hKR.ne']
  congr 1
  simp [cum, Finset.sum_const, Finset.card_univ]

/-- **Any past-only weighting is at most the per-token oracle.** -/
theorem seqRead_le_token_oracle (w : ℕ → Fin K → ℝ) (hw : ∀ t k, 0 ≤ w t k)
    (hw1 : ∀ t, ∑ k, w t k = 1) (a : Fin K → ℕ → ℝ) (m : ℕ → ℝ)
    (hm : ∀ k t, a k t ≤ m t) (n : ℕ)
    (hpos : ∀ t, 0 < ∑ k, w t k * a k t) :
    seqRead w a n ≤ ∑ t ∈ range n, Real.log (m t) := by
  unfold seqRead
  apply Finset.sum_le_sum
  intro t _
  apply Real.log_le_log (hpos t)
  calc ∑ k, w t k * a k t ≤ ∑ k, w t k * m t :=
        Finset.sum_le_sum (fun k _ => mul_le_mul_of_nonneg_left (hm k t) (hw t k))
    _ = m t := by rw [← Finset.sum_mul, hw1, one_mul]

end Sequential

/-- The switching example: particle `k` is right (likelihood `1`) on token `k` only. -/
noncomputable def switchA (ε : ℝ) (k : Fin 2) (t : ℕ) : ℝ := if (k : ℕ) = t then 1 else ε

/-- A fixed past-only schedule: read particle `t` at token `t`. -/
noncomputable def switchW (t : ℕ) (k : Fin 2) : ℝ := if (k : ℕ) = t then 1 else 0

/-- **A switching past-only read beats the span oracle by `-log ε`.**  Two particles, a
two-token span, particle `k` right only on token `k`.  The schedule that reads particle `t` at
token `t` scores `0`; every single particle scores `log ε` on the span, and the span mixture
scores `log ε` too.  So the gain of a past-only reader over the span mixture is not capped by
`log K`, and is not below the span oracle.  (Per draw.  In expectation over the true law a
past-only reader still defines a pmf over spans and stays below the Bayes read; that bound
is not formalised here.)  The identities hold for every `ε`; the claim is about `0 < ε < 1`,
where `log ε < 0`. -/
theorem switching_beats_span_oracle (ε : ℝ) :
    seqRead switchW (switchA ε) 2 = 0
      ∧ (∀ k : Fin 2, ∑ t ∈ range 2, Real.log (switchA ε k t) = Real.log ε)
      ∧ Real.log ((∑ k, cum (switchA ε) k 2) / 2) = Real.log ε := by
  refine ⟨?_, ?_, ?_⟩
  · simp [seqRead, switchW, switchA, Finset.sum_range_succ, Fin.sum_univ_two]
  · intro k
    fin_cases k <;> simp [switchA, Finset.sum_range_succ]
  · simp [cum, switchA, Finset.prod_range_succ, Fin.sum_univ_two]

/-! ## Mixing probabilities vs mixing values -/

section Pool

variable {Y : Type*} [Fintype Y] {K : ℕ}

/-- The softmax pmf of logits `l`. -/
noncomputable def softmaxP (l : Y → ℝ) (y : Y) : ℝ := Real.exp (l y) / ∑ u, Real.exp (l u)

/-- **Probability mixture** of the particles' predictive laws. -/
noncomputable def arithPool (w : Fin K → ℝ) (ℓ : Fin K → Y → ℝ) (y : Y) : ℝ :=
  ∑ k, w k * softmaxP (ℓ k) y

/-- **Value mixture read by a linear head**: attention averages the particles' values, the
head's logits are linear in the value, so the logits mix (a normalised geometric pool). -/
noncomputable def geoPool (w : Fin K → ℝ) (ℓ : Fin K → Y → ℝ) (y : Y) : ℝ :=
  softmaxP (fun u => ∑ k, w k * ℓ k u) y

/-- The probability mixture keeps every particle's weighted floor `w_k p_k(y)`. -/
theorem arith_pool_floor (w : Fin K → ℝ) (hw : ∀ k, 0 ≤ w k) (ℓ : Fin K → Y → ℝ) (y : Y)
    (j : Fin K) : w j * softmaxP (ℓ j) y ≤ arithPool w ℓ y := by
  have hs : ∀ k, 0 ≤ softmaxP (ℓ k) y := fun k =>
    div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg (fun u _ => (Real.exp_pos _).le))
  exact Finset.single_le_sum (f := fun k => w k * softmaxP (ℓ k) y)
    (fun k _ => mul_nonneg (hw k) (hs k)) (mem_univ j)

/-- The two particles of the counterexample: particle 0 favours token `0` by `c`,
particle 1 favours token `1` by `d`. -/
def vmLogits (c d : ℝ) : Fin 2 → Fin 2 → ℝ := ![![0, -c], ![-d, 0]]

/-- **Value mixing loses the floor.**  Uniform weights, true token `0`, `c ≥ 0`.  The
probability mixture scores at least `-log 4` for every `d`, while the value mixture scores
at most `(c - d)/2`, which tends to `-∞` as particle 1 becomes confidently wrong. -/
theorem value_mix_loses_floor (c d : ℝ) (hc : 0 ≤ c) :
    -Real.log 4 ≤ Real.log (arithPool (fun _ => (1 / 2 : ℝ)) (vmLogits c d) 0)
      ∧ Real.log (geoPool (fun _ => (1 / 2 : ℝ)) (vmLogits c d) 0) ≤ (c - d) / 2 := by
  constructor
  · -- arith ≥ (1/2) · softmax(particle 0)(0) ≥ 1/4
    have hfloor := arith_pool_floor (fun _ => (1 / 2 : ℝ)) (fun _ => by norm_num)
      (vmLogits c d) 0 0
    have h0 : softmaxP (vmLogits c d 0) 0 = 1 / (1 + Real.exp (-c)) := by
      simp [softmaxP, vmLogits, Fin.sum_univ_two]
    have he : Real.exp (-c) ≤ 1 := Real.exp_le_one_iff.2 (by linarith)
    have h12 : 1 / 2 ≤ 1 / (1 + Real.exp (-c)) := by
      rw [div_le_div_iff₀ (by norm_num) (by positivity)]
      linarith
    rw [h0] at hfloor
    have hA : 1 / 4 ≤ arithPool (fun _ => (1 / 2 : ℝ)) (vmLogits c d) 0 := by linarith
    have := Real.log_le_log (by norm_num) hA
    rw [one_div, Real.log_inv] at this
    exact this
  · have hg : geoPool (fun _ => (1 / 2 : ℝ)) (vmLogits c d) 0
        = Real.exp (-d / 2) / (Real.exp (-d / 2) + Real.exp (-c / 2)) := by
      simp [geoPool, softmaxP, vmLogits, Fin.sum_univ_two]
      ring_nf
    rw [hg]
    have hpos : 0 < Real.exp (-d / 2) / (Real.exp (-d / 2) + Real.exp (-c / 2)) := by
      positivity
    have hle : Real.exp (-d / 2) / (Real.exp (-d / 2) + Real.exp (-c / 2))
        ≤ Real.exp ((c - d) / 2) := by
      rw [div_le_iff₀ (by positivity)]
      have : Real.exp ((c - d) / 2) * Real.exp (-c / 2) = Real.exp (-d / 2) := by
        rw [← Real.exp_add]; ring_nf
      nlinarith [Real.exp_pos ((c - d) / 2), Real.exp_pos (-d / 2)]
    calc Real.log (Real.exp (-d / 2) / (Real.exp (-d / 2) + Real.exp (-c / 2)))
        ≤ Real.log (Real.exp ((c - d) / 2)) := Real.log_le_log hpos hle
      _ = (c - d) / 2 := Real.log_exp _

/-- Two particles confidently wrong in opposite directions (three tokens, true token `0`). -/
def cmLogits (M : ℝ) : Fin 2 → Fin 3 → ℝ := ![![0, M, -M], ![0, -M, M]]

/-- **Neither read dominates: the value mixture can beat the probability mixture.**  The two
particles' errors cancel in the mean logits, so the value mixture reads the true token at
`1/3`, while the probability mixture reads it below `exp(-M)`.  With
`value_mix_loses_floor`, this says the attention read is a different read (a committee),
not a worse or better mixture. -/
theorem committee_can_beat_mixture (M : ℝ) :
    Real.log (arithPool (fun _ => (1 / 2 : ℝ)) (cmLogits M) 0) ≤ -M
      ∧ Real.log (geoPool (fun _ => (1 / 2 : ℝ)) (cmLogits M) 0) = -Real.log 3 := by
  constructor
  · have ha : arithPool (fun _ => (1 / 2 : ℝ)) (cmLogits M) 0
        = 1 / (1 + Real.exp M + Real.exp (-M)) := by
      simp [arithPool, softmaxP, cmLogits, Fin.sum_univ_two, Fin.sum_univ_three]
      ring
    rw [ha]
    have hpos : 0 < 1 / (1 + Real.exp M + Real.exp (-M)) := by positivity
    have hle : 1 / (1 + Real.exp M + Real.exp (-M)) ≤ Real.exp (-M) := by
      rw [div_le_iff₀ (by positivity), Real.exp_neg]
      have h1 : 0 < Real.exp M := Real.exp_pos M
      have h2 : 0 < (Real.exp M)⁻¹ := inv_pos.2 h1
      have h3 : (Real.exp M)⁻¹ * Real.exp M = 1 := inv_mul_cancel₀ h1.ne'
      nlinarith
    calc Real.log (1 / (1 + Real.exp M + Real.exp (-M))) ≤ Real.log (Real.exp (-M)) :=
          Real.log_le_log hpos hle
      _ = -M := Real.log_exp _
  · have hg : geoPool (fun _ => (1 / 2 : ℝ)) (cmLogits M) 0 = 1 / 3 := by
      simp [geoPool, softmaxP, cmLogits, Fin.sum_univ_two, Fin.sum_univ_three]
      ring_nf
    rw [hg, one_div, Real.log_inv]

end Pool

end TulExploration
