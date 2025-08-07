# Calculate SystemState size with MAX_PHASES=6, MAX_COMPONENTS=4

MAX_PHASES = 6
MAX_COMPONENTS = 4
MAX_STATEVARS = 4
MAX_DOF_PER_PHASE = 6

# Size of CompositionSet (rough estimate)
# phase_record pointer (8) + NP (8) + X[MAX_COMPONENTS] + dof[MAX_STATEVARS + MAX_DOF_PER_PHASE] + etc
compset_size = 8 + 8 + (MAX_COMPONENTS * 8) + ((MAX_STATEVARS + MAX_DOF_PER_PHASE) * 8) + 32  # Extra for other fields
compset_size = 8 + 8 + 32 + 80 + 32  # = 160 bytes per CompositionSet

# Size of CompsetState (estimate)
compset_state_size = 80  # Rough estimate

# SystemState fields:
systemstate_size = 0
systemstate_size += compset_size * MAX_PHASES  # compsets[MAX_PHASES]
systemstate_size += 4  # num_compsets
systemstate_size += compset_state_size * MAX_PHASES  # cs_states[MAX_PHASES]
systemstate_size += 4  # iteration
systemstate_size += 4  # iterations_since_last_phase_change
systemstate_size += 4 * MAX_PHASES  # metastable_phase_iterations[MAX_PHASES]
systemstate_size += 4 * MAX_PHASES  # times_compset_removed[MAX_PHASES]
systemstate_size += 8  # mass_residual
systemstate_size += 8 * MAX_PHASES  # phase_amt[MAX_PHASES]
systemstate_size += 8 * MAX_COMPONENTS  # chemical_potentials[MAX_COMPONENTS]
systemstate_size += 4  # condition_idx
systemstate_size += 8 * MAX_COMPONENTS  # previous_chemical_potentials[MAX_COMPONENTS]
systemstate_size += 8  # largest_chemical_potential_difference
systemstate_size += 8 * MAX_PHASES * MAX_COMPONENTS  # delta_ms[MAX_PHASES * MAX_COMPONENTS]
systemstate_size += 4  # delta_ms_rows
systemstate_size += 4  # delta_ms_cols
systemstate_size += 8 * MAX_STATEVARS  # delta_statevars[MAX_STATEVARS]
systemstate_size += 8 * MAX_PHASES * MAX_COMPONENTS  # phase_compositions[MAX_PHASES * MAX_COMPONENTS]
systemstate_size += 4  # phase_compositions_rows
systemstate_size += 4  # phase_compositions_cols

print(f"Estimated SystemState size: {systemstate_size} bytes = {systemstate_size/1024:.1f} KB")
print(f"CompositionSet array alone: {compset_size * MAX_PHASES} bytes")
print(f"GPU thread stack limit is typically ~1KB")
