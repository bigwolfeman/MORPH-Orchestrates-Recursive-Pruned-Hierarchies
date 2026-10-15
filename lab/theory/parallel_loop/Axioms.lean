/-
Sorry check.  Run with

  lake env lean Axioms.lean

Every line of the output must read `depends on axioms: [propext, Classical.choice, Quot.sound]`
or a subset of those three.  `sorryAx` in any line means the theorem is unproved.
-/
import ParallelLoop

-- PL1: DepthLaw.lean
#print axioms ParallelLoop.rem_nonneg
#print axioms ParallelLoop.rem_zero
#print axioms ParallelLoop.rem_succ
#print axioms ParallelLoop.rem_mono
#print axioms ParallelLoop.rem_anti
#print axioms ParallelLoop.infDist_sub_le
#print axioms ParallelLoop.rem_le_infDist
#print axioms ParallelLoop.greedy_step
#print axioms ParallelLoop.greedy_dist
#print axioms ParallelLoop.greedy_iterate_dist
#print axioms ParallelLoop.greedy_infDist
#print axioms ParallelLoop.floor_le_avgLoss
#print axioms ParallelLoop.minimizer_forced
#print axioms ParallelLoop.rem_eq_iff
#print axioms ParallelLoop.kcurve_pos_iff
#print axioms ParallelLoop.kcurve_nonneg
#print axioms ParallelLoop.tokens_kcurve_pos_iff
#print axioms ParallelLoop.skip_work_eq_entry_work
#print axioms ParallelLoop.kcurve_mono_sq
#print axioms ParallelLoop.rem_thin
#print axioms ParallelLoop.thin_core_earns
#print axioms ParallelLoop.nrm_smul_pos
#print axioms ParallelLoop.normalized_build_flat
-- PL2: Prune.lean
#print axioms ParallelLoop.DepthLaw.sum_lin
#print axioms ParallelLoop.DepthLaw.sum_s_lin
#print axioms ParallelLoop.DepthLaw.sum_sq_affine
#print axioms ParallelLoop.DepthLaw.var_eq_sum
#print axioms ParallelLoop.DepthLaw.var_nonneg
#print axioms ParallelLoop.DepthLaw.Jpar_eq
#print axioms ParallelLoop.DepthLaw.Jpar_nonneg
#print axioms ParallelLoop.DepthLaw.Jpar_skip
#print axioms ParallelLoop.DepthLaw.par_minimizer
#print axioms ParallelLoop.DepthLaw.gv_eq
#print axioms ParallelLoop.DepthLaw.gs_eq
#print axioms ParallelLoop.DepthLaw.hasDerivAt_Jpar_v
#print axioms ParallelLoop.DepthLaw.hasDerivAt_Jpar_w
#print axioms ParallelLoop.DepthLaw.hasDerivAt_Jpar_β
#print axioms ParallelLoop.DepthLaw.par_stationary_prunes
#print axioms ParallelLoop.DepthLaw.flow_prune_slice
#print axioms ParallelLoop.DepthLaw.flow_prune_lyapunov
#print axioms ParallelLoop.DepthLaw.flow_conserve
#print axioms ParallelLoop.DepthLaw.gs_zero_par
#print axioms ParallelLoop.DepthLaw.scaled_skip_grad_prunes
#print axioms ParallelLoop.DepthLaw.Jser_eq
#print axioms ParallelLoop.DepthLaw.Jpar_fixed_eq_Jser
#print axioms ParallelLoop.DepthLaw.gs_zero_ser
#print axioms ParallelLoop.DepthLaw.Jser_vertex
#print axioms ParallelLoop.DepthLaw.ser_minimizer
#print axioms ParallelLoop.DepthLaw.ser_loss_at_opt
#print axioms ParallelLoop.DepthLaw.ser_kcurve_pos
#print axioms ParallelLoop.DepthLaw.par_le_ser_every_depth
#print axioms ParallelLoop.DepthLaw.Jpar2_eq
#print axioms ParallelLoop.DepthLaw.par2_minimizer
#print axioms ParallelLoop.DepthLaw.unique_share
#print axioms ParallelLoop.DepthLaw.Jdrop_on_slice
#print axioms ParallelLoop.DepthLaw.drop_minimizer
#print axioms ParallelLoop.DepthLaw.drop_eval_loss
#print axioms ParallelLoop.DepthLaw.drop_tax
