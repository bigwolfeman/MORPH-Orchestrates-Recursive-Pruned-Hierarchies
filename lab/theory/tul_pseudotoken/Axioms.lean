/-
Sorry check.  Run with

  lake env lean Axioms.lean

Every line of the output must read `depends on axioms: [propext, Classical.choice, Quot.sound]`
or a subset of those three.  `sorryAx` in any line means the theorem is unproved.
-/
import TulPseudotoken

-- P1: Carrier.lean (what one carrier position can hold)
#print axioms TulPseudotoken.abs_strength_le_one
#print axioms TulPseudotoken.strength_eq_one_iff
#print axioms TulPseudotoken.sum_sq_strength_le_one
#print axioms TulPseudotoken.card_strong_mul_sq_le_one
#print axioms TulPseudotoken.inner_mixture
#print axioms TulPseudotoken.norm_sq_mixture
#print axioms TulPseudotoken.vertex_mixture_eq
#print axioms TulPseudotoken.mixture_strength_eq_one_iff
#print axioms TulPseudotoken.affine_write_card_le
#print axioms TulPseudotoken.slot_change_linearIndependent
#print axioms TulPseudotoken.exact_tuple_write_needs_dim
#print axioms TulPseudotoken.softmax_nonneg
#print axioms TulPseudotoken.sum_softmax
#print axioms TulPseudotoken.sum_exp_split
#print axioms TulPseudotoken.softmax_ge
#print axioms TulPseudotoken.one_sub_softmax_le
#print axioms TulPseudotoken.snap_close
#print axioms TulPseudotoken.exists_scale_commit
#print axioms TulPseudotoken.softmax_le
#print axioms TulPseudotoken.passes_to_commit
-- P2: Signals.lean (which trace-free signal pays for copy content)
#print axioms TulPseudotoken.CopyTask.ceValue_eq_next_add_far
#print axioms TulPseudotoken.CopyTask.ceValue_empty
#print axioms TulPseudotoken.CopyTask.next_blind
#print axioms TulPseudotoken.rec_misaligned
#print axioms TulPseudotoken.aux_price
#print axioms TulPseudotoken.norm_sq_sum_orth
#print axioms TulPseudotoken.kdLoss_orth
#print axioms TulPseudotoken.kd_misaligned
-- P3: Teacher.lean (what a raw-context teacher can add)
#print axioms TulPseudotoken.kd_mean_eq
#print axioms TulPseudotoken.sq_wsum_le
#print axioms TulPseudotoken.kd_variance_le
#print axioms TulPseudotoken.transcript_replay
#print axioms TulPseudotoken.distilled_carrier_le
-- P4: Bypass.lean (when a carrier takes the loop's job)
#print axioms TulPseudotoken.bypass_zeroes_passes
#print axioms TulPseudotoken.deeper_never_more_informative
#print axioms TulPseudotoken.raw_span_sufficient
