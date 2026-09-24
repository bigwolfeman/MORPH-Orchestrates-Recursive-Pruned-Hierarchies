/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import Mathlib.Data.Finset.Card
import Mathlib.Data.List.Basic
import Mathlib.Logic.Function.Iterate
import Mathlib.Algebra.Order.Group.Nat

/-!
# T4: when depth is necessary, and why a reader restriction is not what makes it so

## Model

The context holds a table `f : V → V` (edges as tokens, facts as tokens: `f v` is "what
`v` points to").  The target is the `h`-th hop from a known start, `f^[h] x₀`.

A computation is an ADAPTIVE LOOKUP PROGRAM (`Prog`): at each step it either returns a node
or looks the table up at ANY node it likes, chosen from everything it has seen so far, and
continues.  Nothing restricts which node it may look at: the program's VIEW of the table is
unrestricted.  What is bounded is the number of SEQUENTIAL lookups on the path the program
actually runs (`cost`).  One lookup is one composition: one attention read keyed by the
current state, then any computation on the result.

A MORPH forward in this model is a prelude program, `T` passes of a step program and a coda
program, glued with `bind`.  A pass whose core has `b` blocks makes at most `b` lookups.

## Main results

* `eval_hop`, `cost_hop` : following the pointer `h` times computes `f^[h] x₀` with `h`
  lookups (sufficiency).
* **`hop_lower_bound`** : a program that makes fewer than `h` lookups on the chain table
  `v ↦ v + 1` returns the wrong node on the chain table or on a table that differs from it
  at ONE node the program never looked at.  So `h` sequential lookups are necessary, even
  with an unrestricted view.
* **`passes_needed`** : a pipeline (prelude `Lp` lookups, `T` passes of at most `b` lookups,
  coda `Lc` lookups) that computes the `h`-th hop on every table has `h ≤ Lp + b T + Lc`.
  The loop is necessary exactly when `h > Lp + Lc`, and needs `T ≥ (h - Lp - Lc)/b` passes.
* `pointer_loop_exact` : one lookup per pass, `h` passes, computes the target: the bound is
  tight.
* `fixed_after` : if the chain ends in an absorbing node, the pointer loop reaches that
  fixed point after `h` passes and stays there.  Depth used = distance to the fixed point.
* `teacher_forced_one_lookup` : given the true previous node (teacher forcing), the target is
  ONE lookup away at every `h`.  Teacher forcing makes every target shallow.
* `fixed_table_no_lookups` : a relation that is the same on every input (stored in the
  weights, or a positional shortcut) is precomposed with zero lookups.  Only relations that
  vary with the context force depth.
* `double_iterate` : if a step may compose two partial results (parallel cells that read
  each other: pointer doubling), `t` steps reach hop `2^t`.  The single-thread lower bound
  does not cover this model.
-/

namespace TulExploration

/-- An adaptive lookup program over nodes `V`. -/
inductive Prog (V : Type) : Type where
  /-- Return a node. -/
  | ret : V → Prog V
  /-- Look the table up at a node and continue with what it returns. -/
  | ask : V → (V → Prog V) → Prog V

namespace Prog

variable {V : Type}

/-- Run the program on table `f`. -/
def eval (f : V → V) : Prog V → V
  | ret v => v
  | ask x k => eval f (k (f x))

/-- The number of lookups on the path the program runs on table `f`. -/
def cost (f : V → V) : Prog V → ℕ
  | ret _ => 0
  | ask x k => cost f (k (f x)) + 1

/-- The nodes looked up on the path the program runs on table `f`. -/
def asked (f : V → V) : Prog V → List V
  | ret _ => []
  | ask x k => x :: asked f (k (f x))

theorem length_asked (f : V → V) (p : Prog V) : (asked f p).length = cost f p := by
  induction p with
  | ret v => rfl
  | ask x k ih => simp [asked, cost, ih]

/-- Two tables that agree on every node the program looked up give the same run. -/
theorem run_congr (f g : V → V) (p : Prog V) (h : ∀ x ∈ asked f p, g x = f x) :
    eval g p = eval f p ∧ asked g p = asked f p := by
  induction p with
  | ret v => simp [eval, asked]
  | ask x k ih =>
    have hx : g x = f x := h x (by simp [asked])
    have := ih (f x) (fun z hz => h z (by simp [asked, hz]))
    simp [eval, asked, hx, this]

/-- Sequential composition: run `p`, then `c` on its answer. -/
def bind : Prog V → (V → Prog V) → Prog V
  | ret v, c => c v
  | ask x k, c => ask x (fun v => bind (k v) c)

theorem eval_bind (f : V → V) (p : Prog V) (c : V → Prog V) :
    eval f (bind p c) = eval f (c (eval f p)) := by
  induction p with
  | ret v => rfl
  | ask x k ih => simp [bind, eval, ih]

theorem cost_bind (f : V → V) (p : Prog V) (c : V → Prog V) :
    cost f (bind p c) = cost f p + cost f (c (eval f p)) := by
  induction p with
  | ret v => simp [bind, cost, eval]
  | ask x k ih => simp [bind, cost, eval, ih]; omega

/-- `T` passes of a step program. -/
def loop (step : V → Prog V) : ℕ → V → Prog V
  | 0, v => ret v
  | t + 1, v => bind (step v) (loop step t)

theorem cost_loop_le (f : V → V) (step : V → Prog V) (b : ℕ)
    (hb : ∀ v, cost f (step v) ≤ b) : ∀ t v, cost f (loop step t v) ≤ b * t
  | 0, v => by simp [loop, cost]
  | t + 1, v => by
    simp only [loop, cost_bind]
    have := cost_loop_le f step b hb t (eval f (step v))
    have := hb v
    rw [Nat.mul_succ]
    omega

/-- Follow the pointer `t` times. -/
def hop : ℕ → V → Prog V
  | 0, v => ret v
  | t + 1, v => ask v (hop t)

theorem eval_hop (f : V → V) : ∀ t v, eval f (hop t v) = f^[t] v
  | 0, v => rfl
  | t + 1, v => by
    simp only [hop, eval, Function.iterate_succ_apply]
    exact eval_hop f t (f v)

theorem cost_hop (f : V → V) : ∀ t v, cost f (hop t v) = t
  | 0, _ => rfl
  | t + 1, v => by simp [hop, cost, cost_hop f t]

end Prog

open Prog

/-! ## The lower bound -/

/-- The chain table on `ℕ`: `v ↦ v + 1`. -/
def succT : ℕ → ℕ := fun v => v + 1

lemma succT_iter : ∀ i v, succT^[i] v = v + i
  | 0, v => rfl
  | i + 1, v => by
    rw [Function.iterate_succ_apply', succT_iter i v]
    simp [succT]
    omega

/-- The detour table: the chain, except that node `j` points to `h + 1`. -/
def detour (j h : ℕ) : ℕ → ℕ := fun v => if v = j then h + 1 else v + 1

lemma detour_iter_le (j h : ℕ) : ∀ i, i ≤ j → (detour j h)^[i] 0 = i
  | 0, _ => rfl
  | i + 1, hi => by
    rw [Function.iterate_succ_apply', detour_iter_le j h i (by omega)]
    have : i ≠ j := by omega
    simp [detour, this]

lemma detour_iter_after (j h : ℕ) (hjh : j < h) :
    ∀ m, (detour j h)^[j + 1 + m] 0 = h + 1 + m
  | 0 => by
    rw [Nat.add_zero, Function.iterate_succ_apply', detour_iter_le j h j le_rfl]
    simp [detour]
  | m + 1 => by
    rw [show j + 1 + (m + 1) = (j + 1 + m) + 1 by omega, Function.iterate_succ_apply',
      detour_iter_after j h hjh m]
    have : h + 1 + m ≠ j := by omega
    simp [detour, this]
    omega

/-- **`h` sequential lookups are necessary for the `h`-th hop.**  If a program makes fewer
than `h` lookups when run on the chain table, then it returns the wrong node either on the
chain table or on a detour table that changes ONE node the program never looked at.  The
program may look at any node it likes at every step; only the number of sequential lookups
is bounded. -/
theorem hop_lower_bound (p : Prog ℕ) (h : ℕ) (hc : cost succT p < h) :
    ∃ f : ℕ → ℕ, eval f p ≠ f^[h] 0 := by
  have hpig : ∃ j, j < h ∧ j ∉ asked succT p := by
    by_contra hcon
    simp only [not_exists, not_and, not_not] at hcon
    have hsub : Finset.range h ⊆ (asked succT p).toFinset :=
      fun j hj => List.mem_toFinset.2 (hcon j (Finset.mem_range.1 hj))
    have h1 := Finset.card_le_card hsub
    rw [Finset.card_range] at h1
    have h2 := h1.trans (List.toFinset_card_le _)
    rw [length_asked] at h2
    omega
  obtain ⟨j, hj, hnot⟩ := hpig
  by_cases hs : eval succT p = h
  · refine ⟨detour j h, ?_⟩
    have hagree : ∀ x ∈ asked succT p, detour j h x = succT x := by
      intro x hx
      have : x ≠ j := fun hxj => hnot (hxj ▸ hx)
      simp [detour, succT, this]
    rw [(run_congr succT (detour j h) p hagree).1, hs]
    have := detour_iter_after j h hj (h - j - 1)
    rw [show j + 1 + (h - j - 1) = h by omega] at this
    rw [this]
    omega
  · refine ⟨succT, ?_⟩
    rw [succT_iter]
    simpa using hs

/-- **Passes needed.**  A prelude program, `T` passes of a step program, and a coda program,
glued with `bind`.  If, run on the chain table, the prelude makes at most `Lp` lookups, every
pass at most `b`, and the coda at most `Lc`, and the pipeline returns the `h`-th hop on EVERY
table, then `h ≤ Lp + b T + Lc`. -/
theorem passes_needed (pre : Prog ℕ) (step coda : ℕ → Prog ℕ) (Lp b Lc T h : ℕ)
    (hpre : cost succT pre ≤ Lp) (hstep : ∀ v, cost succT (step v) ≤ b)
    (hcoda : ∀ v, cost succT (coda v) ≤ Lc)
    (hcorrect : ∀ f : ℕ → ℕ,
      eval f (bind pre (fun v => bind (loop step T v) coda)) = f^[h] 0) :
    h ≤ Lp + b * T + Lc := by
  by_contra hlt
  push Not at hlt
  have hcost : cost succT (bind pre (fun v => bind (loop step T v) coda)) < h := by
    rw [cost_bind, cost_bind]
    have := cost_loop_le succT step b hstep T (eval succT pre)
    have := hcoda (eval succT (loop step T (eval succT pre)))
    omega
  obtain ⟨f, hf⟩ := hop_lower_bound _ h hcost
  exact hf (hcorrect f)

/-- **The bound is tight.**  One lookup per pass, `h` passes, no prelude or coda lookups. -/
theorem pointer_loop_exact {V : Type} (f : V → V) (x₀ : V) (h : ℕ) :
    eval f (loop (fun v => ask v ret) h x₀) = f^[h] x₀ ∧
      cost f (loop (fun v => ask v ret) h x₀) = h := by
  induction h generalizing x₀ with
  | zero => exact ⟨rfl, rfl⟩
  | succ t ih =>
    simp only [loop, eval_bind, cost_bind, eval, cost, Function.iterate_succ_apply]
    obtain ⟨h1, h2⟩ := ih (f x₀)
    exact ⟨h1, by rw [h2]; omega⟩

/-- **A contractive picture: the loop's fixed point is the answer.**  If the chain ends in an
absorbing node at hop `h`, every later pass leaves the state there.  The number of passes a
loop uses is its distance to the fixed point. -/
theorem fixed_after {V : Type} (f : V → V) (x₀ : V) (h : ℕ) (hfix : f (f^[h] x₀) = f^[h] x₀) :
    ∀ m, f^[h + m] x₀ = f^[h] x₀
  | 0 => rfl
  | m + 1 => by
    rw [show h + (m + 1) = (h + m) + 1 by omega, Function.iterate_succ_apply',
      fixed_after f x₀ h hfix m, hfix]

/-- **Teacher forcing makes every target one lookup deep.**  A reader handed the true
previous node `f^[h] x₀` (the true prefix) reaches `f^[h+1] x₀` with one lookup, for every
`h`.  Under teacher forcing no target needs depth; a parallel read of position `h` from the
start needs `h` lookups (`hop_lower_bound`). -/
theorem teacher_forced_one_lookup {V : Type} (f : V → V) (x₀ : V) (h : ℕ) :
    eval f (ask (f^[h] x₀) ret) = f^[h + 1] x₀ ∧ cost f (ask (f^[h] x₀) ret) = 1 := by
  refine ⟨?_, rfl⟩
  simp [eval, Function.iterate_succ_apply']

/-- **A relation fixed across examples needs no lookups.**  If the table is the same on every
input (a relation stored in the weights, or a positional shortcut like `p ↦ p + 1`), the
`h`-th hop is a constant the network can precompose: zero lookups.  The lower bound bites
only for relations that VARY with the context (`hop_lower_bound` quantifies over tables). -/
theorem fixed_table_no_lookups {V : Type} (f₀ : V → V) (x₀ : V) (h : ℕ) :
    eval f₀ (ret (f₀^[h] x₀)) = f₀^[h] x₀ ∧ cost f₀ (ret (f₀^[h] x₀) : Prog V) = 0 :=
  ⟨rfl, rfl⟩

/-- **Pointer doubling.**  A step that composes two partial results reaches hop `2^t` after
`t` steps.  Parallel cells that read each other's pointers do this; the single-thread
lower bound above does not apply to them. -/
theorem double_iterate {α : Type*} (f : α → α) : ∀ t, (fun g : α → α => g ∘ g)^[t] f = f^[2 ^ t]
  | 0 => by simp
  | t + 1 => by
    rw [Function.iterate_succ_apply', double_iterate f t, pow_succ, Nat.mul_two,
      Function.iterate_add]

end TulExploration
