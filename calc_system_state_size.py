# Calculate the actual size of SystemState structure

MAX_PHASES = 4
MAX_COMPONENTS = 4
MAX_STATEVARS = 4
MAX_DOF_PER_PHASE = 4
MAX_INTERNAL_CONSTRAINTS = 4
MAX_FIXED_MOLE_FRACTION_CONDITIONS = 4

# CompositionSet size
comp_set_size = (
    8 +  # phase_record pointer
    MAX_DOF_PER_PHASE * 8 +  # dof array
    MAX_COMPONENTS * 8 +  # X array
    8 +  # energy
    8 +  # NP
    1    # fixed bool (rounded to 8 bytes for alignment)
)

# CompsetState size
compset_state_size = (
    (MAX_STATEVARS + MAX_DOF_PER_PHASE) * 8 +  # x array
    4 +  # x_length int
    8 +  # energy double
    (MAX_STATEVARS + MAX_DOF_PER_PHASE) * 8 +  # grad array
    4 +  # grad_length int
    ((MAX_STATEVARS + MAX_DOF_PER_PHASE) * (MAX_STATEVARS + MAX_DOF_PER_PHASE)) * 8 +  # hess array
    4 + 4 +  # hess_rows, hess_cols ints
    MAX_COMPONENTS * 8 +  # masses array
    4 +  # masses_length int
    (MAX_COMPONENTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE)) * 8 +  # mass_jac array
    4 + 4 +  # mass_jac_rows, mass_jac_cols ints
    ((MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS) * (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS)) * 8 +  # phase_matrix
    4 +  # phase_matrix_dim int
    ((MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS) * (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS)) * 8 +  # full_e_matrix
    4 +  # full_e_matrix_dim int
    MAX_DOF_PER_PHASE * 8 +  # c_G array
    4 +  # c_G_length int
    (MAX_DOF_PER_PHASE * MAX_STATEVARS) * 8 +  # c_statevars array
    4 + 4 +  # c_statevars_rows, c_statevars_cols ints
    (MAX_COMPONENTS * MAX_DOF_PER_PHASE) * 8 +  # c_component array
    4 + 4 +  # c_component_rows, c_component_cols ints
    MAX_DOF_PER_PHASE * 8 +  # delta_y array
    4 +  # delta_y_length int
    8 +  # moles_normalization double
    MAX_INTERNAL_CONSTRAINTS * 8 +  # internal_cons array
    4 +  # internal_cons_length int
    (MAX_STATEVARS + MAX_DOF_PER_PHASE) * 8 +  # moles_normalization_grad array
    4 +  # moles_normalization_grad_length int
    (MAX_INTERNAL_CONSTRAINTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE)) * 8 +  # cons_jac_tmp array
    4 + 4  # cons_jac_tmp_rows, cons_jac_tmp_cols ints
)

# SystemState size
system_state_fields = (
    MAX_PHASES * comp_set_size +  # compsets array
    4 +  # num_compsets int
    MAX_PHASES * compset_state_size +  # cs_states array
    4 +  # iteration int
    4 +  # iterations_since_last_phase_change int
    MAX_PHASES * 4 +  # metastable_phase_iterations array of ints
    MAX_PHASES * 4 +  # times_compset_removed array of ints
    8 +  # mass_residual double
    MAX_PHASES * 8 +  # phase_amt array
    MAX_COMPONENTS * 8 +  # chemical_potentials array
    MAX_COMPONENTS * 8 +  # mole_fractions array
    8 +  # system_amount double
    4 +  # converged bool (rounded to 4 for alignment)
    4 +  # phases_changed bool (rounded to 4 for alignment)
    8 +  # largest_statevar_change double
    8 +  # largest_phase_amt_change double
    8 +  # largest_y_change double
    MAX_PHASES * 8 +  # phase_compositions array
    MAX_PHASES * 4 +  # free_stable_compset_indices array of ints
    4 +  # num_free_stable_compsets int
    4    # condition_idx int
)

print(f"CompositionSet size: {comp_set_size} bytes")
print(f"CompsetState size: {compset_state_size} bytes")
print(f"SystemState size: {system_state_fields} bytes")
print(f"SystemState size in doubles: {system_state_fields // 8}")
print(f"Current allocation (50000 doubles): {50000 * 8} bytes")
print(f"Sufficient? {50000 * 8 >= system_state_fields}")
