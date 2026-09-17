/-
Sorry check.  Run with

  lake env lean Axioms.lean

Every line of the output must read `depends on axioms: [propext, Classical.choice, Quot.sound]`.
`sorryAx` in any line means the theorem is unproved.
-/
import LctulEulerDepth

#print axioms LctulEulerDepth.eulerEndpoint_one
#print axioms LctulEulerDepth.eulerEndpoint_one_of_meanField
#print axioms LctulEulerDepth.euler_dirac
#print axioms LctulEulerDepth.eulerEndpoint_dirac
#print axioms LctulEulerDepth.euler_affine
#print axioms LctulEulerDepth.eulerEndpoint_affine
#print axioms LctulEulerDepth.euler_gauss
#print axioms LctulEulerDepth.eulerEndpoint_gauss
#print axioms LctulEulerDepth.affScale_gauss_one
#print axioms LctulEulerDepth.eulerEndpoint_gauss_one
#print axioms LctulEulerDepth.affScale_gauss_two
#print axioms LctulEulerDepth.affScale_gauss_pos
#print axioms LctulEulerDepth.euler_error_bound
#print axioms LctulEulerDepth.euler_exact_of_zero_defect
#print axioms LctulEulerDepth.euler_error_bound_curvature
#print axioms LctulEulerDepth.sum_norm_sub_sq
#print axioms LctulEulerDepth.sum_norm_sub_sq_ge
#print axioms LctulEulerDepth.fm_loss_t0
#print axioms LctulEulerDepth.fm_loss_t0_ge
#print axioms LctulEulerDepth.fm_loss_t0_optimal
#print axioms LctulEulerDepth.fm_loss_t0_optimal_unique
#print axioms LctulEulerDepth.k1_endpoint_of_fm_optimal
#print axioms LctulEulerDepth.residual_two_draws
#print axioms LctulEulerDepth.total_variance
#print axioms LctulEulerDepth.residual_ratio_conditional_draw
#print axioms LctulEulerDepth.explained_of_residual_ratio
#print axioms LctulEulerDepth.norm_wmean_le
#print axioms LctulEulerDepth.norm_wmean_le_of_shell
#print axioms TulInformation.Joint.mi_prodIndep
#print axioms TulInformation.Joint.sample_mi_le
#print axioms TulInformation.Joint.sample_mi_le_depth
#print axioms TulInformation.Joint.sample_condEntropy_ge
