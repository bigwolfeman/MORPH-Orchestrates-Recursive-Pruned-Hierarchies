/-
Sorry check.  Run with

  lake env lean Axioms.lean

Every line of the output must read `depend on axioms: [propext, Classical.choice, Quot.sound]`.
`sorryAx` in any line means the theorem is unproved.
-/
import TulInformation

#print axioms TulInformation.sum_mul_log_div_le_zero
#print axioms TulInformation.Joint.mi_nonneg
#print axioms TulInformation.Joint.mi_map_le
#print axioms TulInformation.Joint.condEntropy_map_ge
#print axioms TulInformation.Joint.map_id
#print axioms TulInformation.Joint.map_comp
#print axioms TulInformation.Joint.mi_iterate_le
#print axioms TulInformation.Joint.condEntropy_iterate_ge
#print axioms TulInformation.Joint.mi_traj
#print axioms TulInformation.Joint.mi_exit_le_traj
#print axioms TulInformation.Joint.mi_factor_le
#print axioms TulInformation.Joint.mi_relay_le
#print axioms TulInformation.Joint.mi_relay_redundant
