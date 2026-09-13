/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent
-/
import TulInformation.Basic

/-!
# The corollaries the MORPH slot loop needs

## The model of the forward pass

Taken from `morph/model/transformer.py::_tul_core` on 2026-09-13
(`tokens_through_core = false`, `reread = false`, `slot_chain = false`, `db_loop = false`,
full BPTT):

```
h_0[k]     = core_init (input_norm (prelude x) [slot_index k])
h_{t+1}[k] = f (h_t[j] for k - w <= j <= k, e[k], inj[k], ret_state_t, t)  if depth k > t
           = h_t[k]                                                       otherwise
z[k]       = h_T[k]
```

`e[k]` and `inj[k]` are frozen: they are computed once from the prelude output at cell `k`
and re-fed at every pass. `w` is `tul.loop_reach` (0 means "every earlier cell"). No token
position is read at any pass. So the whole array of slot states after pass `t` is a
deterministic function of ONE object, the loop's entry data
`E = (h_0, e, inj, ret_state_0, depths)`, and the passes are iterations of one map
`f : E → E` on that object.

Everything below is a statement about that deterministic iteration. `α` is the alphabet of
the entry data, `β` the alphabet of the target (a token, a span, anything).

## Main results

* `Joint.mi_iterate_le` : no number of passes adds information.
* `Joint.mi_traj` : the trajectory carries exactly what the entry carries.
* `Joint.mi_factor_le` : a pass carries at most what it can reach.
* `Joint.mi_relay_redundant` : a reader that already sees the pass's inputs gains nothing
  from the relayed state.
-/

namespace TulInformation

namespace Joint

open Finset

variable {α β γ ν : Type*}

/-! ## One pass, and any number of passes, add nothing -/

/-- **No pass adds information.**  `f` is the one-pass core map on the loop's entry data.
After `n` passes the state carries at most what the entry carried about the target.
This is `mi_map_le` applied to `f^[n]`; it holds for every `n`, every target, every core. -/
theorem mi_iterate_le [Fintype α] [Fintype β] [DecidableEq α]
    (P : Joint α β) (f : α → α) (n : ℕ) : (P.map f^[n]).mi ≤ P.mi :=
  P.mi_map_le _

/-- The log-loss reading of `mi_iterate_le`: the Bayes conditional entropy of the target
given the state after `n` passes is at least its value given the entry state.  Any CE that a
deeper read wins is therefore won against a SUBOPTIMAL reader, never against the entry's
information. -/
theorem condEntropy_iterate_ge [Fintype α] [Fintype β] [DecidableEq α]
    (P : Joint α β) (f : α → α) (n : ℕ) :
    P.condEntropy ≤ (P.map f^[n]).condEntropy :=
  P.condEntropy_map_ge _

/-! ## The trajectory carries exactly what the entry carries -/

/-- The trajectory of the first `n + 1` states: `traj a = fun i => f^[i] a`. -/
def traj (f : α → α) (n : ℕ) (a : α) : Fin (n + 1) → α := fun i => f^[(i : ℕ)] a

@[simp]
lemma traj_zero (f : α → α) (n : ℕ) (a : α) : traj f n a 0 = a := rfl

/-- **The trajectory adds nothing over the entry, and loses nothing either.**  Writing every
pass's state into the reader (a "trajectory as prefix" arm) hands it exactly the entry's
information about the target: no more, and — unlike the exit state alone — no less. -/
theorem mi_traj [Fintype α] [Fintype β] [DecidableEq α]
    (P : Joint α β) (f : α → α) (n : ℕ) : (P.map (traj f n)).mi = P.mi := by
  refine le_antisymm (P.mi_map_le _) ?_
  have hcomp : (fun t : Fin (n + 1) → α => t 0) ∘ (traj f n) = id := by
    funext a; rfl
  calc P.mi = ((P.map (traj f n)).map fun t : Fin (n + 1) → α => t 0).mi := by
        rw [map_comp, hcomp, map_id]
    _ ≤ (P.map (traj f n)).mi := (P.map (traj f n)).mi_map_le _

/-- The exit state carries no more than the trajectory. -/
theorem mi_exit_le_traj [Fintype α] [Fintype β] [DecidableEq α]
    (P : Joint α β) (f : α → α) (n : ℕ) :
    (P.map f^[n]).mi ≤ (P.map (traj f n)).mi := by
  rw [mi_traj]
  exact P.mi_iterate_le f n

/-! ## Reachability: a pass can only carry what it can reach -/

/-- **The reach bound.**  If the state read out (`φ`) depends on the entry only through a
projection `π` — for the slot loop with `tul.loop_reach = w`, `π` is the entry data of the
cells within `t * w` positions — then the read-out carries at most what that projection
carries.  Depth buys information only by enlarging `π`. -/
theorem mi_factor_le {ω : Type*} [Fintype α] [Fintype β] [Fintype γ] [Fintype ω]
    [DecidableEq γ] [DecidableEq ω]
    (P : Joint α β) (π : α → ω) (F : ω → γ) (φ : α → γ) (hfac : ∀ a, φ a = F (π a)) :
    (P.map φ).mi ≤ (P.map π).mi := by
  have : φ = F ∘ π := funext hfac
  rw [this, ← map_comp]
  exact (P.map π).mi_map_le F

/-! ## The relay, and why an unrestricted reader makes it worthless -/

/-- **The relay bound.**  A pass whose inputs are the cell's own previous state together
with the states of the cells it can reach carries at most what those inputs carry.  `ν` is
the alphabet of the reachable neighbour states, `F` the pass. -/
theorem mi_relay_le [Fintype α] [Fintype β] [Fintype γ] [Fintype ν] [DecidableEq γ]
    (P : Joint (α × ν) β) (F : α × ν → γ) :
    (P.map F).mi ≤ P.mi :=
  P.mi_map_le F

/-- **Reach "all" makes the relay free of information.**  If the reader already observes the
inputs of the pass — under `tg_coda_prefix_reach: all` the coda reads every slot's cells
directly — then also handing it the relayed state changes the mutual information with the
target by exactly zero.  The relay can then only re-organise; it cannot deliver.

With `tg_coda_prefix_reach: prev` the reader does NOT observe the other cells, this theorem
does not apply, and the loop becomes the only route.  That is the one geometry in the ledger
with a forced K-curve. -/
theorem mi_relay_redundant [Fintype α] [Fintype β] [Fintype γ] [Fintype ν]
    [DecidableEq α] [DecidableEq γ] [DecidableEq ν]
    (P : Joint (α × ν) β) (F : α × ν → γ) :
    (P.map fun x => (F x, x.1, x.2)).mi = P.mi := by
  refine le_antisymm (P.mi_map_le _) ?_
  have hcomp : ((fun y : γ × α × ν => (y.2.1, y.2.2)) ∘ fun x : α × ν => (F x, x.1, x.2))
      = id := by
    funext x
    cases x
    rfl
  calc P.mi = ((P.map fun x : α × ν => (F x, x.1, x.2)).map
        fun y : γ × α × ν => (y.2.1, y.2.2)).mi := by
        rw [map_comp, hcomp, map_id]
    _ ≤ (P.map fun x : α × ν => (F x, x.1, x.2)).mi :=
        (P.map fun x : α × ν => (F x, x.1, x.2)).mi_map_le _

end Joint

end TulInformation
