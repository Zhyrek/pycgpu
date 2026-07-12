#ifndef EQSOLVER_H
#define EQSOLVER_H

#include "minimizer.h" // REQUIRED: Contains SystemSpecification, SystemState, constants, etc.

// Each thread's SystemState lives in a global-memory slot of SYSTEM_STATE_SIZE
// doubles (allocated Python-side with the same constant). If the struct outgrows
// the slot, threads corrupt their neighbors' state — fail the build instead.
#ifndef SYSTEM_STATE_SIZE
#define SYSTEM_STATE_SIZE 50000
#endif
// Newton-loop iteration budget: CPU parity value is 1000 (solver.py:288),
// passed at RUNTIME as a kernel argument (max_solver_iterations) so the
// two-pass driver can run a cheap low-cap pass 1 and rerun only cap-hitting
// conditions at the full budget, without recompiling.
static_assert(sizeof(SystemState) <= SYSTEM_STATE_SIZE * sizeof(double),
              "SystemState does not fit its per-thread global-memory slot");

// --- Helper Structs for GPU Data ---

// To represent the grid data (from pycalphad calculate output) on the GPU
typedef struct DeviceGrid {
    const double* Y_ptr;    // Flattened array of site fractions for all grid points
    const double* X_ptr;    // Flattened array of mole compositions for all grid points
    const double* GM_ptr;   // Array of Gibbs energies for all grid points
    const int* PhaseID_ptr; // Flattened array of phase identifiers for all grid points
                            // (integer IDs mapping to PhaseRecord in DevicePhaseData)

    int num_grid_points_total; // Total number of rows in the grid arrays
    int phase_dof_stride_Y;    // Max number of site fractions for any phase (stride for Y_ptr)
    int num_components_stride_X; // Number of components (stride for X_ptr)

    // For add_nearly_stable: mapping a global phase ID to its start/end row in the grid
    // This replaces grid.attrs['phase_indices']
    const int* phase_grid_indices_start; // Array indexed by global phase ID
    const int* phase_grid_indices_stop;  // Array indexed by global phase ID
    int num_mappable_phases_in_grid;   // Size of phase_grid_indices_start/stop arrays
} DeviceGrid;

// To represent PhaseRecord data on the GPU
typedef struct DevicePhaseData {
    const PhaseRecord* phase_records_array; // Pointer to an array of unique PhaseRecord structs
    int num_unique_phase_records;
    // If a mapping from a simple active_phase_id (0 to N_active_phases-1) to
    // an index in phase_records_array is needed, it can be passed or handled by caller.
    // For now, assume direct indexing or that CompositionSet stores a direct pointer or fat index.
    const int* grid_phase_id_to_record_index; // Array sized by max grid phase ID
    int max_grid_phase_id;
} DevicePhaseData;


// --- Forward Declarations ---
__device__ int get_phase_record_index(const DevicePhaseData* phase_data, int grid_phase_id);
// run_loop with global memory support is declared later

// --- Device Helper Functions ---

__device__ int argmax_gpu(const double* arr, int size) {
    if (size <= 0) return -1; // Or handle error appropriately
    int max_idx = 0;
    double max_val = arr[0];
    for (int i = 1; i < size; ++i) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}


/**
 * @brief Attempts to add a new phase with the largest driving force.
 * GPU equivalent of add_new_phases from eqsolver.pyx.py.
 *
 * @param added_phase_grid_idx Output: If a phase is added, this will be its index in the DeviceGrid.
 * @param current_sys_state Pointer to the current SystemState (contains chemical_potentials, compsets).
 * This function will try to find a candidate to add.
 * The actual addition to compsets array is handled by the caller.
 * @param spec System-wide specification.
 * @param grid_data Pointer to the grid data on the GPU.
 * @param phase_data Pointer to the phase record data on the GPU.
 * @param state_variables_values Current P, T, etc. for this condition.
 * @param minimum_df Minimum driving force to consider adding a phase.
 * @param removed_compsets Array of CompositionSets that were recently removed (for distinctness check).
 * @param num_removed_compsets Number of compsets in removed_compsets array.
 * @return True if a candidate phase is identified (caller then adds it), false otherwise.
 */
#ifdef PYCGPU_OUTER_ADD
// STUDY-ONLY (task #4): correct implementation of the CPU outer
// add_new_phases candidate search. The historical version of this
// function dereferenced the raw grid block as a pointer-struct and
// therefore NEVER ran (num_grid_points_total read NaN-padding bytes
// = 0 since the original 2025-07-17 commit). Enabled only under
// -DPYCGPU_OUTER_ADD with a correctly parsed DeviceGrid built by the
// caller; default builds contain no outer add loop at all.
__device__ bool identify_candidate_phase_to_add(
    int* candidate_phase_grid_idx,         // Output: index in grid for the phase to add
    double* candidate_driving_force,       // Output: driving force of the candidate
    const SystemState* current_sys_state,  // Input: for chemical potentials, existing compsets
    const SystemSpecification* spec,
    const DeviceGrid* grid_data,
    const DevicePhaseData* phase_data,
    const double* state_variables_values,  // Current P, T for potential new phase
    double minimum_df,
    const CompositionSet* removed_compsets, // Array of recently removed CompositionSets
    int num_removed_compsets               // Number of removed compsets
) {
    // Check for NULL grid_data before dereferencing
    if (grid_data == nullptr || grid_data->num_grid_points_total == 0) return false;

    // Calculate driving forces for all points in the grid
    // double driving_forces_on_grid[MAX_GRID_POINTS]; // MAX_GRID_POINTS needs to be a define
    // This dynamic allocation or large static array is problematic.
    // Process grid points sequentially or in chunks if too large for local static memory.
    // For now, assume we can iterate and find max on the fly.

    double largest_df = -INFINITY; // Negative infinity (matches CPU -np.inf)
    int best_grid_idx = -1;

    for (int i = 0; i < grid_data->num_grid_points_total; ++i) {
        double current_potential_energy = 0.0;
        for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
            // X_ptr is [grid_point_idx * num_components_stride_X + comp_idx]
            current_potential_energy += grid_data->X_ptr[i * grid_data->num_components_stride_X + comp_idx] *
                                        current_sys_state->chemical_potentials[comp_idx];
        }
        double df = current_potential_energy - grid_data->GM_ptr[i];

        if (df > largest_df) {
            int candidate_phase_id_from_grid = grid_data->PhaseID_ptr[i];
            if (candidate_phase_id_from_grid == -1) continue; // _FAKE_ or invalid phase in grid

            int candidate_record_index = get_phase_record_index(phase_data, candidate_phase_id_from_grid);
            if (candidate_record_index == -1) continue; // Invalid phase

            const PhaseRecord* candidate_phase_record = &phase_data->phase_records_array[candidate_record_index];
            
            // --- Primary distinctness check against removed compsets (matching CPU logic exactly) ---
            bool distinct_from_removed = true;
            for (int rem_idx = 0; rem_idx < num_removed_compsets; ++rem_idx) {
                const CompositionSet* removed_compset = &removed_compsets[rem_idx];
                
                // Check if same phase type (matching CPU: df_phase_name != compset.phase_record.phase_name)
                if (removed_compset->phase_record != candidate_phase_record) {
                    continue;
                }
                
                // Same phase type, check site fraction composition (matching CPU pyx logic lines 41-47)
                distinct_from_removed = false;
                for (int dof_idx = 0; dof_idx < candidate_phase_record->phase_dof; ++dof_idx) {
                    double candidate_site_frac = grid_data->Y_ptr[i * grid_data->phase_dof_stride_Y + dof_idx];
                    double removed_site_frac = removed_compset->dof[spec->num_statevars + dof_idx];
                    
                    // Use 10*COMP_DIFFERENCE_TOL as in CPU version (line 43 in pyx)
                    if (fabs(candidate_site_frac - removed_site_frac) > 10.0 * COMP_DIFFERENCE_TOL) {
                        distinct_from_removed = true;
                        break;
                    }
                }
                
                if (!distinct_from_removed) {
                    break; // Found a non-distinct match, skip this candidate
                }
            }
            
            if (!distinct_from_removed) {
                continue; // Skip this candidate, not distinct from removed phases (matching CPU line 51)
            }
            
            // ONLY if distinct from removed phases, update largest_df (matching CPU lines 52-53)
            largest_df = df;
            best_grid_idx = i;
        }
    }
    
    // After finding the best candidate, check distinctness against current compsets (matching CPU lines 54-72)
    if (largest_df > minimum_df && best_grid_idx != -1) {
        int final_candidate_phase_id = grid_data->PhaseID_ptr[best_grid_idx];
        int final_candidate_record_index = get_phase_record_index(phase_data, final_candidate_phase_id);
        const PhaseRecord* final_candidate_phase_record = &phase_data->phase_records_array[final_candidate_record_index];
        
        // Check distinctness against current compsets (matching CPU logic lines 62-72)
        for (int cs_idx = 0; cs_idx < current_sys_state->num_compsets; ++cs_idx) {
            // CPU deletes only NP <= 0 compsets before add_new_phases (solver.py:301),
            // then checks distinctness against ALL remaining compsets (eqsolver.pyx:79).
            if (current_sys_state->compsets[cs_idx].phase_record == nullptr ||
                current_sys_state->phase_amt[cs_idx] <= 0.0) continue;

            if (current_sys_state->compsets[cs_idx].phase_record == final_candidate_phase_record) {
                // Same phase type, check overall composition (using X, not site fractions)
                bool compositions_distinct = false;
                for (int comp_j = 0; comp_j < spec->num_components; ++comp_j) {
                    double x_grid = grid_data->X_ptr[best_grid_idx * grid_data->num_components_stride_X + comp_j];
                    double x_compset = current_sys_state->compsets[cs_idx].X[comp_j];
                    if (fabs(x_grid - x_compset) > COMP_DIFFERENCE_TOL) {
                        compositions_distinct = true;
                        break;
                    }
                }
                if (!compositions_distinct) {
                    // Not distinct from current phase, reject this candidate
                    *candidate_phase_grid_idx = -1;
                    *candidate_driving_force = 0.0;
                    return false;
                }
            }
        }
        
        // Passed all distinctness checks
        *candidate_phase_grid_idx = best_grid_idx;
        *candidate_driving_force = largest_df;
        return true;
    }
    
    return false;
}
#endif // PYCGPU_OUTER_ADD

__device__ int get_phase_record_index(const DevicePhaseData* phase_data, int grid_phase_id) {
    // Check for NULL phase_data first
    if (phase_data == nullptr) {
        return -1; // No phase data available
    }

    // Check for NULL mapping array
    if (phase_data->grid_phase_id_to_record_index == nullptr) {
        // FALLBACK: When no mapping array exists, assume phase ID directly corresponds to record index
        // This is valid when phase IDs are 0-indexed and contiguous
        if (grid_phase_id < 0 || grid_phase_id >= phase_data->num_unique_phase_records) {
            return -1; // Phase ID out of range for direct indexing
        }
        return grid_phase_id; // Use phase ID as record index directly
    }

    // Use the mapping array if available
    if (grid_phase_id < 0 || grid_phase_id > phase_data->max_grid_phase_id) {
        return -1; // Invalid phase ID for mapping
    }
    return phase_data->grid_phase_id_to_record_index[grid_phase_id];
}

/**
 * @brief Identifies nearly stable phases not currently in the system.
 * GPU equivalent of add_nearly_stable from eqsolver.pyx.py.
 *
 * @param candidate_phases_to_add_grid_indices Output: Array to be filled with grid indices of nearly stable phases to consider.
 * @param num_candidates_found Output: Number of nearly stable phases found and placed in the output array.
 * @param max_candidates_to_find Max size of the candidate_phases_to_add_grid_indices array.
 * @param current_sys_state Pointer to the current SystemState.
 * @param spec System-wide specification.
 * @param grid_data Pointer to the grid data on the GPU.
 * @param phase_data Pointer to the phase record data on the GPU.
 * @param state_variables_values Current P, T, etc.
 * @param minimum_df Minimum driving force to consider.
 * @return True if any nearly stable phases were identified, false otherwise.
 */

// --- Main single-condition solver ---
// Note: Hook functions (pre_solve_hook, post_solve_hook) are implemented in minimizer.h
// Output struct for a single equilibrium calculation
typedef struct EquilibriumResultSingle { // Ensure this is defined (copied from previous prompt)
    double final_chemical_potentials[MAX_COMPONENTS];
    double final_system_gm;
    int num_stable_phases;
    int phase_ids[MAX_PHASES];
    double NP[MAX_PHASES];
    double X_phases[MAX_PHASES * MAX_COMPONENTS];
    double Y_phases[MAX_PHASES * MAX_DOF_PER_PHASE];
    bool converged;
    // True if ANY inner run_loop exhausted its iteration budget. A cap-hit
    // feeds the outer add-phases loop, so the trajectory (even a "converged"
    // one) can differ from a full-budget run: the two-pass driver must rerun
    // every cap-touching condition, not just unconverged ones.
    bool hit_iteration_cap;
} EquilibriumResultSingle;

// Input struct for a single condition
typedef struct ConditionArgsSingle { // Ensure this is defined
    double state_variables_values[MAX_STATEVARS];
} ConditionArgsSingle;

// Forward declare the InitialPhaseDataSingle struct (matches kernel definition)
typedef struct InitialPhaseDataSingle {
    int phase_indices[MAX_PHASES];
    double phase_amounts[MAX_PHASES];
    double site_fractions[MAX_PHASES * MAX_DOF_PER_PHASE];
    double compositions[MAX_PHASES * MAX_COMPONENTS];
    double chemical_potentials[MAX_COMPONENTS];
    int num_phases;
} InitialPhaseDataSingle;


// Forward declaration of run_loop (defined in minimizer.h with global memory support)
__device__ bool run_loop(
    int thread_id,
    SystemSpecification* spec,
    SystemState* state,
    int max_iterations,
    const DeviceGrid* grid_data,
    const DevicePhaseData* phase_data,
    double* equilibrium_matrix,
    double* equilibrium_rhs,
    double* eq_soln,
    double* A_lstsq_copy,
    double* U_lstsq,
    double* V_lstsq,
    double* singular_values_lstsq,
    double* superdiag_lstsq,
    double* masses,
    double* mass_jac,
    double* x_dof,
    double* grad,
    double* hess,
    double* work_inv
);

// --- ACTUAL SOPHISTICATED SOLVER USING GLOBAL MEMORY ---
// This function implements the full equilibrium solver using global memory arrays
// to avoid stack overflow while maintaining all the sophisticated solver logic
__device__ void solve_equilibrium_at_condition(
    int thread_id,
    const SystemSpecification* global_spec_base,
    const ConditionArgsSingle* condition_args,
    EquilibriumResultSingle* result,
    const DevicePhaseData* phase_data,
    const double* initial_data, // Raw flat array instead of struct
    const DeviceGrid* grid_data,
    const double* condition_mole_fractions, // NEW: Pass actual mole fractions from condition
    // Pre-allocated global memory arrays (per-thread slices)
    double* A_lstsq_copy,        // Replaces stack: double A_lstsq_copy[MAX_SVD_M * MAX_SVD_N]
    double* U_lstsq,             // Replaces stack: double U_lstsq[MAX_SVD_M * MAX_SVD_N]
    double* V_lstsq,             // Replaces stack: double V_lstsq[MAX_SVD_N * MAX_SVD_N]
    double* singular_values_lstsq, // Replaces stack: double singular_values_lstsq[MAX_SVD_N]
    double* superdiag_lstsq,     // Replaces stack: double superdiag_lstsq[MAX_SVD_N]
    double* U_inv,               // Replaces stack: double U_inv[MAX_PHASE_MATRIX_DIM^2]
    double* V_inv,               // Replaces stack: double V_inv[MAX_PHASE_MATRIX_DIM^2]
    double* singular_values_inv, // Replaces stack: double singular_values_inv[MAX_PHASE_MATRIX_DIM]
    double* superdiag_inv,       // Replaces stack: double superdiag_inv[MAX_PHASE_MATRIX_DIM]
    double* work_inv,            // Replaces stack: double work_inv[MAX_PHASE_MATRIX_DIM^2]
    double* x_dof,               // Replaces stack: double x[MAX_STATEVARS + MAX_DOF_PER_PHASE]
    double* grad,                // Replaces stack: double grad[MAX_STATEVARS + MAX_DOF_PER_PHASE]
    double* hess,                // Replaces stack: double hess[(MAX_STATEVARS + MAX_DOF_PER_PHASE)^2]
    double* masses,              // Replaces stack: double masses[MAX_COMPONENTS]
    double* mass_jac,            // Replaces stack: double mass_jac[MAX_COMPONENTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE)]
    double* phase_matrix,        // Replaces stack: double phase_matrix[(MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS)^2]
    double* equilibrium_matrix,  // Replaces stack: large equilibrium system matrix
    double* equilibrium_rhs,     // Replaces stack: equilibrium system RHS vector
    double* eq_soln,             // Replaces stack: equilibrium solution vector
    double* global_system_states, // SystemState in global memory to avoid stack overflow
    double* delta_ms,            // NEW: Global memory for delta_ms array
    double* phase_compositions,  // NEW: Global memory for phase_compositions array
    double* phase_amounts_per_mole_atoms,  // NEW: Global memory for _phase_amounts_per_mole_atoms_arr
    int max_solver_iterations    // Newton-loop budget (CPU parity: 1000)
) {
    // STACK OVERFLOW FIX: All large arrays are now passed as parameters from global memory
    
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG  
        printf("GPU DEBUG: solve_equilibrium_at_condition STARTED\\n");
        #endif
    }
    
    // Step 1: Validate inputs and global memory arrays
    if (!A_lstsq_copy || !result || !global_spec_base || !initial_data) {
        if (result) {
            result->converged = false;
            result->hit_iteration_cap = false;
        }
        return; // Cannot proceed without required arrays - memory not allocated yet
    }
    
    // Step 2: Initialize local spec copy from the global spec passed from Python
    // Use simple assignment copy instead of manual byte copy
    // The manual byte copy was causing struct field corruption
    // Safely read SystemSpecification from GPU memory
    // Cannot dereference struct pointer directly due to alignment/memory access issues
    // SystemSpecification lives in global memory to avoid stack pointer issues
    // Allocate SystemSpec on stack instead of reusing work array
    // which might be causing memory corruption for Thread 1
    char spec_buffer[sizeof(SystemSpecification)];
    SystemSpecification* current_spec_ptr = (SystemSpecification*)spec_buffer;
    SystemSpecification& current_spec = *current_spec_ptr;
    
    // Copy the entire struct byte-by-byte from GPU memory
    // This preserves the exact layout from Python
    const char* spec_bytes = (const char*)global_spec_base;
    memcpy(current_spec_ptr, spec_bytes, sizeof(SystemSpecification));
    
    // DEBUG: Verify the copy worked
    if (thread_id == 0 || thread_id == 1) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Thread %d - Copied SystemSpecification to global memory at %p\\n", thread_id, current_spec_ptr);
        printf("  Thread %d: global_spec_base=%p\\n", thread_id, global_spec_base);
        printf("  Thread %d: num_statevars=%d, num_components=%d\\n", 
               thread_id, current_spec.num_statevars, current_spec.num_components);
        if (current_spec.num_prescribed_mole_fraction_conditions > 0) {
            printf("  Thread %d: prescribed_mole_fraction_rhs[0]=%f\\n", 
                   thread_id, current_spec.prescribed_mole_fraction_rhs[0]);
        }
        #endif
    }
    
    // The struct is now fully copied with correct layout from Python
    // No need for manual field-by-field reading
    
    // Note: We'll need to free current_spec_ptr before any return
    
    // Debug: print what we received from Python
    if (thread_id < 3) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Thread %d SystemSpecification from Python:\\n", thread_id);
        printf("  Thread %d: num_statevars = %d\\n", thread_id, current_spec.num_statevars);
        printf("  Thread %d: num_components = %d\\n", thread_id, current_spec.num_components);
        printf("  Thread %d: num_free_chemical_potentials = %d\\n", thread_id, current_spec.num_free_chemical_potentials);
        printf("  Thread %d: num_prescribed_mole_fraction_conditions = %d\\n", thread_id, current_spec.num_prescribed_mole_fraction_conditions);
        
        // DEBUG: Show free and fixed state variables
        printf("  Thread %d: num_free_statevars = %d\\n", thread_id, current_spec.num_free_statevars);
        printf("  Thread %d: free_statevar_indices = [", thread_id);
        for (int i = 0; i < current_spec.num_free_statevars; ++i) {
            printf("%d", current_spec.free_statevar_indices[i]);
            if (i < current_spec.num_free_statevars - 1) printf(", ");
        }
        printf("]\\n");
        printf("  Thread %d: num_fixed_statevars = %d\\n", thread_id, current_spec.num_fixed_statevars);
        printf("  Thread %d: fixed_statevar_indices = [", thread_id);
        for (int i = 0; i < current_spec.num_fixed_statevars; ++i) {
            printf("%d", current_spec.fixed_statevar_indices[i]);
            if (i < current_spec.num_fixed_statevars - 1) printf(", ");
        }
        printf("]\\n");
        
        // DEBUG: Print struct offsets to diagnose alignment
        printf("  Thread %d: Struct base address: %p\\n", thread_id, global_spec_base);
        printf("  Thread %d: Expected offset of prescribed_mole_fraction_rhs: 176\\n", thread_id);
        
        // Try reading from the local copy (avoiding direct pointer dereference)
        if (current_spec.num_prescribed_mole_fraction_conditions > 0) {
            printf("  Thread %d: Copy read prescribed_mole_fraction_rhs[0] = %e\\n", thread_id, current_spec.prescribed_mole_fraction_rhs[0]);
            
            // Print raw bytes at the expected offset (176 from Python)
            char* base_ptr = (char*)global_spec_base;
            double* rhs_at_offset_176 = (double*)(base_ptr + 176);
            printf("  Thread %d: Value at offset 176: %e\\n", thread_id, *rhs_at_offset_176);
            
            // DEBUG: Print raw bytes to see what's actually there
            printf("  Thread %d: Raw bytes at offset 176: ", thread_id);
            unsigned char* byte_ptr = (unsigned char*)(base_ptr + 176);
            for (int i = 0; i < 8; ++i) {
                printf("%02x", byte_ptr[i]);
            }
            printf("\\n");
            
            // Try different offsets in case of alignment issues
            for (int offset = 168; offset <= 184; offset += 8) {
                double* test_ptr = (double*)(base_ptr + offset);
                printf("  Thread %d: Value at offset %d: %e\\n", thread_id, offset, *test_ptr);
            }
            
            // Manually copy the value from the known offset
            // This works around struct alignment issues between CPU and GPU
            if (*rhs_at_offset_176 != 0.0 && current_spec.prescribed_mole_fraction_rhs[0] == 0.0) {
                printf("  Thread %d: FIXING prescribed_mole_fraction_rhs[0] from %e to %e\\n", 
                       thread_id, current_spec.prescribed_mole_fraction_rhs[0], *rhs_at_offset_176);
                current_spec.prescribed_mole_fraction_rhs[0] = *rhs_at_offset_176;
            }
        }
        printf("  Thread %d: prescribed_system_amount = %f\\n", thread_id, current_spec.prescribed_system_amount);
        #endif
    }
    
    // Step 3: Use SystemState from global memory to avoid stack overflow
    // SystemState is too large (~100KB) for GPU thread stack
    SystemState* current_sys_state_ptr = nullptr;
    if (global_system_states != nullptr) {
        // Cast the global memory to SystemState pointer
        current_sys_state_ptr = (SystemState*)global_system_states;
        // Initialize SystemState to zero
        memset(current_sys_state_ptr, 0, sizeof(SystemState));
    } else {
        // Fallback: allocate on stack (will cause overflow with many threads)
        #ifdef VERBOSE_DEBUG
        if (thread_id == 0) {
            printf("GPU DEBUG: WARNING - SystemState allocated on stack (no global memory provided)\\n");
        }
        #endif
        // This will fail with multiple threads due to stack overflow
        return;  // Exit early to avoid crash
    }
    SystemState& current_sys_state = *current_sys_state_ptr;
    // SystemState is now properly zero-initialized via memset

    // Initialize SystemState manually without creating large stack arrays
    // Set the global memory pointers
    current_sys_state.delta_ms = delta_ms;
    current_sys_state.phase_compositions = phase_compositions;
    current_sys_state._phase_amounts_per_mole_atoms_arr = phase_amounts_per_mole_atoms;

    current_sys_state.num_compsets = 0;
    current_sys_state.iteration = 0;
    current_sys_state.iterations_since_last_phase_change = 0;
    current_sys_state.condition_idx = thread_id;  // Set condition index for debug output
    
    // Initialize the required arrays to safe values
    for (int i = 0; i < MAX_PHASES; ++i) {
        current_sys_state.phase_amt[i] = 0.0;
        current_sys_state.metastable_phase_iterations[i] = 0;
        current_sys_state.times_compset_removed[i] = 0;
    }
    
    // Access flat double array directly for chemical_potentials
    // Layout: phase_indices[MAX_PHASES] + phase_amounts[MAX_PHASES] + site_fractions[MAX_PHASES*MAX_DOF_PER_PHASE] + 
    //         compositions[MAX_PHASES*MAX_COMPONENTS] + chemical_potentials[MAX_COMPONENTS] + num_phases
    // NOTE: The Python array is now flattened to 1D, so we access it for condition 0 directly
    // For multiple conditions, we would need to add: condition_idx * doubles_per_condition
    const double* initial_data_flat = initial_data;
    // Use calculated offset instead of hardcoded value
    // Chemical potentials come after: phase_indices + phase_amounts + site_fractions + compositions
    int chem_pot_offset = MAX_PHASES + MAX_PHASES + (MAX_PHASES * MAX_DOF_PER_PHASE) + (MAX_PHASES * MAX_COMPONENTS);
    
    for (int i = 0; i < MAX_COMPONENTS; ++i) {
        // Use chemical potentials from SystemSpecification, NOT from initial_data
        // The initial_data contains starting_point values which are wrong
        current_sys_state.chemical_potentials[i] = (i < current_spec.num_components) ? current_spec.initial_chemical_potentials[i] : 0.0;
        current_sys_state.mole_fractions[i] = 0.0;
    }
    
    // Set up basic state from initial_data
    current_sys_state.system_amount = 1.0; // Standard amount
    
    // Initialize mole fractions from the passed condition_mole_fractions array
    // This array contains the actual mole fractions from the condition, properly calculated
    // for binary, ternary, and higher-order systems
    for (int i = 0; i < MAX_COMPONENTS; ++i) {
        if (i < current_spec.num_components) {
            // Use the mole fractions that were correctly extracted from condition_data_array
            // and passed to this function
            current_sys_state.mole_fractions[i] = condition_mole_fractions[i];
        } else {
            current_sys_state.mole_fractions[i] = 0.0;
        }
    }
    
    // Access num_phases and phase_indices from flat array
    // NOTE: The initial_data pointer is already offset to this thread's data
    // Layout: phase_indices[MAX_PHASES] + phase_amounts[MAX_PHASES] + site_fractions[MAX_PHASES*MAX_DOF_PER_PHASE] + 
    //         compositions[MAX_PHASES*MAX_COMPONENTS] + chemical_potentials[MAX_COMPONENTS] + num_phases[1]
    int num_phases_offset = MAX_PHASES + MAX_PHASES + (MAX_PHASES * MAX_DOF_PER_PHASE) + 
                           (MAX_PHASES * MAX_COMPONENTS) + MAX_COMPONENTS;
    int num_phases = (int)initial_data_flat[num_phases_offset];
    
    // Debug: Test if the pointer issue is with multiple conditions or single condition
    // The kernel might be interpreting this as a multi-condition array
    // Let's try accessing it as a 2D array and see if that fixes it
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: TESTING DIFFERENT ACCESS PATTERNS:\\n");
        printf("  Direct access to initial_data:\\n");
        printf("    [0]=%f, [1]=%f, [2]=%f, [44]=%f\\n", 
               initial_data_flat[0], initial_data_flat[1], initial_data_flat[2], initial_data_flat[44]);
        
        // Test accessing with calculated doubles_per_struct
        int doubles_per_struct = MAX_PHASES + MAX_PHASES + (MAX_PHASES * MAX_DOF_PER_PHASE) + 
                                (MAX_PHASES * MAX_COMPONENTS) + MAX_COMPONENTS + 1;
        printf("  Accessing with condition offset (using calculated doubles_per_struct=%d):\\n", doubles_per_struct);
        int condition_offset = 0 * doubles_per_struct;  // condition 0
        printf("    condition_offset=0: [%d]=%f, [%d]=%f, [%d]=%f, [%d]=%f\\n",
               condition_offset+0, initial_data_flat[condition_offset+0],
               condition_offset+1, initial_data_flat[condition_offset+1], 
               condition_offset+2, initial_data_flat[condition_offset+2],
               condition_offset+doubles_per_struct-1, initial_data_flat[condition_offset+doubles_per_struct-1]);
        #endif
    }
    
    // DEBUG: Check num_phases value and constants
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Constants - MAX_PHASES=%d, MAX_DOF_PER_PHASE=%d, MAX_COMPONENTS=%d\\n", 
               MAX_PHASES, MAX_DOF_PER_PHASE, MAX_COMPONENTS);
        printf("GPU DEBUG: Python layout offset calculation: 4 + 4 + (4*4) + (4*4) + 3 = %d\\n", num_phases_offset);
        printf("GPU DEBUG: Checking array values around offset %d: [%f, %f, %f, %f, %f, %f, %f, %f]\\n", 
               num_phases_offset, initial_data_flat[num_phases_offset-4], initial_data_flat[num_phases_offset-3], initial_data_flat[num_phases_offset-2], 
               initial_data_flat[num_phases_offset-1], initial_data_flat[num_phases_offset], initial_data_flat[num_phases_offset+1], 
               initial_data_flat[num_phases_offset+2], initial_data_flat[num_phases_offset+3]);
        printf("GPU DEBUG: num_phases_offset=%d, num_phases=%d\\n", num_phases_offset, num_phases);
        #endif
    }
    
    // Set up initial composition sets from lower_convex_hull data
    // DEBUG: Store initial phase setup info
    if (thread_id == 0) {
        result->X_phases[16] = (double)num_phases;  // Number of phases in initial data
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Setting up initial phases, num_phases=%d\\n", num_phases);
        #endif
    }
    
    for (int i = 0; i < num_phases && i < MAX_PHASES; ++i) {
        int pr_idx = (int)initial_data_flat[i];  // phase_indices are at the beginning
        
        // Access phase_amounts from flat array 
        // phase_amounts start at offset MAX_PHASES
        double phase_amount = initial_data_flat[MAX_PHASES + i];
        
        // DEBUG: Store phase processing info for first phase
        if (thread_id == 0 && i == 0) {
            result->X_phases[17] = (double)pr_idx;                    // First phase index
            result->X_phases[18] = phase_amount;                      // First phase amount
            result->X_phases[19] = (double)phase_data->num_unique_phase_records; // Available phase records
        }
        
        if (pr_idx < 0 || pr_idx >= phase_data->num_unique_phase_records) {
            // DEBUG: Mark invalid phase index
            if (thread_id == 0 && i == 0) {
                result->X_phases[20] = -1.0; // Invalid phase index marker
            }
            continue;
        }
        // CPU (eqsolver.pyx:268) floors phase_amt at MIN_PHASE_FRACTION and KEEPS the
        // phase; the Python side already applied that floor, so only skip truly empty slots.
        if (phase_amount < MIN_PHASE_FRACTION) {
            // DEBUG: Mark phase amount too small
            if (thread_id == 0 && i == 0) {
                result->X_phases[21] = -2.0; // Phase amount too small marker
            }
            continue;
        }
        
        // Bounds check to prevent overflow
        if (current_sys_state.num_compsets >= MAX_PHASES) {
            printf("GPU ERROR: Too many phases! num_compsets=%d >= MAX_PHASES=%d\\n", 
                   current_sys_state.num_compsets, MAX_PHASES);
            break;
        }
        
        // DEBUG: Check memory before accessing arrays
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: About to access compsets[%d] and cs_states[%d], MAX_PHASES=%d\\n", 
                   current_sys_state.num_compsets, current_sys_state.num_compsets, MAX_PHASES);
            printf("GPU DEBUG: current_spec at %p still valid? num_statevars=%d\\n", 
                   &current_spec, current_spec.num_statevars);
            #endif
        }
        
        // Set up CompositionSet directly in SystemState (avoiding stack arrays)
        CompositionSet* cs = &current_sys_state.compsets[current_sys_state.num_compsets];
        CompsetState* css = &current_sys_state.cs_states[current_sys_state.num_compsets];
        
        // Initialize the CompositionSet
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Setting phase_record for phase %d, phase_data=%p, pr_idx=%d\\n", 
                   current_sys_state.num_compsets, phase_data, pr_idx);
            printf("GPU DEBUG: Before init - current_spec.num_statevars = %d\\n", current_spec.num_statevars);
            #endif
        }
        if (phase_data == nullptr || phase_data->phase_records_array == nullptr) {
            printf("GPU ERROR: phase_data or phase_records_array is null!\\n");
            return;
        }
        // Each composition set needs its own phase record instance
        // to avoid sharing memory between phases of the same type (immiscibility gap)
        cs->phase_record = &phase_data->phase_records_array[pr_idx];
        if (!cs->phase_record) continue;
        
        // DEBUG: Print all input data arrays for this phase
        if (thread_id == 0 && current_sys_state.num_compsets < 2) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Phase %d input data verification:\\n", current_sys_state.num_compsets);
            printf("  phase_amount = %f\\n", phase_amount);
            printf("  pr_idx = %d\\n", pr_idx);
            
            // DEBUG: Check raw pointer values
            const double* raw_condition_ptr = (const double*)condition_args;
            printf("  RAW condition_args pointer values: [%f, %f, %f, %f]\\n", 
                   raw_condition_ptr[0], raw_condition_ptr[1], raw_condition_ptr[2], raw_condition_ptr[3]);
            
            printf("  condition_args->state_variables_values: [");
            for(int k = 0; k < MAX_STATEVARS; ++k) {
                printf("%f", condition_args->state_variables_values[k]);
                if (k < MAX_STATEVARS - 1) printf(", ");
            }
            printf("] (current_spec.num_statevars=%d, MAX_STATEVARS=%d)\\n", current_spec.num_statevars, MAX_STATEVARS);
            // Access flat double array directly for debug output
            const double* initial_data_flat_debug = (const double*)initial_data;
            int site_fractions_offset_debug = MAX_PHASES + MAX_PHASES;
            printf("  initial_data->site_fractions for phase %d: [", i);
            for(int k = 0; k < cs->phase_record->phase_dof; ++k) {
                int flat_index_debug = site_fractions_offset_debug + i * MAX_DOF_PER_PHASE + k;
                printf("%f", initial_data_flat_debug[flat_index_debug]);
                if (k < cs->phase_record->phase_dof - 1) printf(", ");
            }
            printf("]\\n");
            #endif
        }
        
        // Set state variables from condition args  
        // The DOF array should store WORKSPACE state variables, not Model state variables
        // CPU stores DOF as [N, P, T, Y1, Y2...] (workspace format)
        // GPU was incorrectly storing as [T, Y1, Y2...] (model format)
        
        // Copy ALL workspace state variables to match CPU behavior
        for (int sv_idx = 0; sv_idx < current_spec.num_statevars && sv_idx < MAX_STATEVARS; ++sv_idx) {
            cs->dof[sv_idx] = condition_args->state_variables_values[sv_idx];
        }
        
        if (thread_id == 0 && current_sys_state.num_compsets < 2) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Setting workspace state variables in dof[0:%d]:\\n", current_spec.num_statevars);
            for (int sv_idx = 0; sv_idx < current_spec.num_statevars; ++sv_idx) {
                printf("  dof[%d] = %f\\n", sv_idx, cs->dof[sv_idx]);
            }
            #endif
        }
        
        // Set site fractions from lower_convex_hull results
        // Site fractions start after the WORKSPACE's state variables
        // Layout: phase_indices[MAX_PHASES] + phase_amounts[MAX_PHASES] + site_fractions[MAX_PHASES*MAX_DOF_PER_PHASE] + ...
        // site_fractions start at offset (MAX_PHASES + MAX_PHASES)
        int site_fractions_offset = MAX_PHASES + MAX_PHASES;  // phase_indices + phase_amounts
        const double* initial_data_flat = initial_data;
        
        // Map site fractions to ensure correct component order
        // The Model expects components in alphabetical order, but initial_data might not
        // For NbTi system: Model expects [Y(NB), Y(TI)] but initial_data has [Y(TI), Y(NB)]
        for (int sf_idx = 0; sf_idx < cs->phase_record->phase_dof && sf_idx < MAX_DOF_PER_PHASE; ++sf_idx) {
            int flat_index = site_fractions_offset + i * MAX_DOF_PER_PHASE + sf_idx;
            
            // The initial_data already has site fractions in the correct order [Y(NB), Y(TI)]
            // No swapping needed - just direct copy
            int mapped_idx = sf_idx;
            
            cs->dof[current_spec.num_statevars + mapped_idx] = initial_data_flat[flat_index];  // Site fractions start after WORKSPACE's state variables
            
            // VERIFICATION: Print the values being read to confirm the fix works
            if (thread_id == 0 && current_sys_state.num_compsets < 2) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: VERIFY site_fractions[%d][%d] = %f (from flat_index %d) -> dof[%d]\\n", 
                       i, sf_idx, initial_data_flat[flat_index], flat_index, current_spec.num_statevars + mapped_idx);
                #endif
            }
        }
        
        // REMOVED: Incorrect X constraint adjustment that was forcing individual phases
        // to match the system composition constraint. This was causing phase separation
        // to collapse in miscibility gaps. The composition constraint should be satisfied
        // by the SYSTEM as a whole (weighted average of all phases), not by individual phases.
        // The solver will naturally find the correct phase compositions that satisfy the
        // overall system constraint through the equilibrium conditions.
        
        // DEBUG: Print final DOF array after setup (now in Workspace format)

            // Runtime fit parameters ride in trailing dof slots after this
            // phase's site fractions (read by the generated functions).
            for (int pj = 0; pj < current_spec.num_params && pj < MAX_PARAMS; ++pj) {
                cs->dof[current_spec.num_statevars + cs->phase_record->phase_dof + pj] = current_spec.fit_params[pj];
            }
        if (thread_id == 0 && current_sys_state.num_compsets < 2) {
            #ifdef VERBOSE_DEBUG
            printf("  Final cs->dof after setup (Workspace format): [");
            // DOF contains: Workspace's state vars + phase_dof site fractions
            int num_dof_elements = current_spec.num_statevars + cs->phase_record->phase_dof;
            for(int k = 0; k < num_dof_elements; ++k) {
                printf("%.15f", cs->dof[k]);
                if (k < num_dof_elements - 1) printf(", ");
            }
            printf("] (");
            // Show what each element represents
            for(int k = 0; k < current_spec.num_statevars; ++k) {
                if (k == 0 && current_spec.num_statevars >= 2) printf("N");
                else if (k == 1 && current_spec.num_statevars == 2) printf("T");
                else if (k == 1 && current_spec.num_statevars >= 3) printf("P");
                else if (k == 2 && current_spec.num_statevars >= 3) printf("T");
                if (k < current_spec.num_statevars - 1) printf(", ");
            }
            if (cs->phase_record->phase_dof > 0) {
                printf(", Y1, Y2...)\\n");
            } else {
                printf(")\\n");
            }
            #endif
        }
        
        // Set phase amount and properties
        cs->NP = phase_amount;
        cs->fixed = false;
        current_sys_state.phase_amt[current_sys_state.num_compsets] = phase_amount;
        
        // Initialize CompositionSet first
        cs->init(cs->phase_record);
        
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: After cs->init - current_spec.num_statevars = %d\\n", current_spec.num_statevars);
            #endif
        }
        
        // Initialize CompsetState with proper arrays
        // This is where masses, jacobians, etc. get set up
        css->init(&current_spec, cs);
        
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: After css->init - current_spec.num_statevars = %d\\n", current_spec.num_statevars);
            #endif
        }
        
        // STACK OVERFLOW FIX: CompsetState arrays are fixed arrays, not pointers
        // We need a different approach - we'll copy data between CompsetState and global memory
        // Initialize CompsetState arrays with reasonable starting values
        
        if (current_sys_state.num_compsets < MAX_PHASES) {
            // DEBUG: Check before masses init
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Before masses init - current_spec.num_statevars = %d, num_components = %d\\n", 
                       current_spec.num_statevars, current_spec.num_components);
                #endif
            }
            
            // Initialize masses from composition data directly into CompsetState
            for (int comp_idx = 0; comp_idx < MAX_COMPONENTS; ++comp_idx) {
                if (comp_idx < current_spec.num_components) {
                    css->masses[comp_idx] = current_sys_state.mole_fractions[comp_idx];
                } else {
                    css->masses[comp_idx] = 0.0;
                }
            }
            
            // DEBUG: Check after masses init
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: After masses init - current_spec.num_statevars = %d\\n", current_spec.num_statevars);
                #endif
            }
            
            // Initialize mass jacobian rows/cols
            css->mass_jac_rows = current_spec.num_components;
            css->mass_jac_cols = current_spec.num_statevars + cs->phase_record->phase_dof;
            
            // DEBUG: Check sizes before initializing
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: mass_jac dimensions: rows=%d, cols=%d, phase_dof=%d\\n",
                       css->mass_jac_rows, css->mass_jac_cols, cs->phase_record->phase_dof);
                #endif
            }
            
            // Initialize mass jacobian to reasonable values
            int jac_size = css->mass_jac_rows * css->mass_jac_cols;
            int max_jac_size = MAX_COMPONENTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE);
            if (jac_size > max_jac_size) {
                printf("GPU ERROR: mass_jac size %d exceeds max %d!\\n", jac_size, max_jac_size);
                jac_size = max_jac_size;
            }
            for (int j = 0; j < jac_size; ++j) {
                css->mass_jac[j] = 0.0;
            }
            
            // DEBUG: Check if current_spec is still valid after mass_jac init
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: After mass_jac init - current_spec.num_statevars = %d\\n", current_spec.num_statevars);
                #endif
            }
        }
        
        // DEBUG: Check DOF array before update call
        if (thread_id == 0 && current_sys_state.num_compsets < 2) {
            #ifdef VERBOSE_DEBUG
            printf("  DOF before cs->update(): [");
            int num_dof_elements = current_spec.num_statevars + cs->phase_record->phase_dof;
            for(int k = 0; k < num_dof_elements; ++k) {
                printf("%.6f", cs->dof[k]);
                if (k < num_dof_elements - 1) printf(", ");
            }
            printf("]\\n");
            #endif
        }
        
        // Phase amounts are already normalized at Python level
        // Unlike CPU which normalizes in recompute(), GPU receives pre-normalized values
        double original_phase_amt = cs->NP;  // This is already normalized by Python
        
        // Must call cs->update() to calculate energy and composition
        // The energy field is used in the equilibrium matrix RHS calculation
        // Without this, energy=0 and the solver behaves differently than CPU
        // cs->update expects: (site_fractions, phase_amount, state_variables, workspace_num_statevars)
        // In workspace DOF format: site fractions start at current_spec.num_statevars
        
        // DEBUG: Check before cs->update call
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Before cs->update - current_spec.num_statevars = %d\\n", current_spec.num_statevars);
            #endif
        }
        
        // Save load-bearing values before cs->update in case of corruption
        int saved_num_statevars = current_spec.num_statevars;
        int saved_num_components = current_spec.num_components;
        
        cs->update(&cs->dof[current_spec.num_statevars], cs->NP, cs->dof, current_spec.num_statevars);
        
        // Restore values if corrupted
        if (current_spec.num_statevars < 0 || current_spec.num_statevars > 10) {
            if (thread_id == 0) {
                printf("GPU WARNING: Detected corruption after cs->update, restoring values\\n");
                printf("  Corrupted: num_statevars=%d, num_components=%d\\n", 
                       current_spec.num_statevars, current_spec.num_components);
            }
            current_spec.num_statevars = saved_num_statevars;
            current_spec.num_components = saved_num_components;
        }
        
        // DEBUG: Check after cs->update call
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: After cs->update - current_spec.num_statevars = %d\\n", current_spec.num_statevars);
            #endif
        }
        
        // DEBUG: Check DOF array after update call
        if (thread_id == 0 && current_sys_state.num_compsets < 2) {
            #ifdef VERBOSE_DEBUG
            printf("  DOF after cs->update(): [");
            int num_dof_elements = current_spec.num_statevars + cs->phase_record->phase_dof;
            for(int k = 0; k < num_dof_elements; ++k) {
                printf("%.6f", cs->dof[k]);
                if (k < num_dof_elements - 1) printf(", ");
            }
            printf("]\\n");
            #endif
        }
        
        // DEBUG: Check if current_spec is still valid after processing this phase
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: After phase %d - current_spec.num_statevars = %d, num_components = %d\\n", 
                   current_sys_state.num_compsets, current_spec.num_statevars, current_spec.num_components);
            #endif
        }
        
        current_sys_state.num_compsets++;
    }
    
    // NOTE: Phase amount normalization already done BEFORE phase composition calculation
    // This ensures correct initial system mole fractions in recompute()
    
    // Set up free_stable_compset_indices array
    // This tells the solver which composition sets are free to vary
    // CPU (minimizer.pyx:792) requires NP > 0 for membership.
    current_sys_state.num_free_stable_compsets = 0;
    for (int i = 0; i < current_sys_state.num_compsets; ++i) {
        if (!current_sys_state.compsets[i].fixed && current_sys_state.compsets[i].NP > 0.0 && i < MAX_PHASES) {
            current_sys_state.free_stable_compset_indices[current_sys_state.num_free_stable_compsets] = i;
            current_sys_state.num_free_stable_compsets++;
        }
    }
    
    // TODO: Add output scaling instead of changing solver algorithm

    // Initialize other critical SystemState arrays
    current_sys_state.mass_residual = 0.0;
    current_sys_state.largest_chemical_potential_difference = 0.0;
    current_sys_state.delta_ms_rows = 0;
    current_sys_state.delta_ms_cols = 0;
    // Normalize phase amounts BEFORE calculating phase compositions
    // This matches CPU behavior exactly - phase amounts must be in formula units
    // before we calculate system mole fractions in recompute()
    double phase_amt_sum = 0.0;
    for (int i = 0; i < current_sys_state.num_compsets; ++i) {
        phase_amt_sum += current_sys_state.compsets[i].NP;
    }
    // Always normalize to match CPU behavior exactly - no conditions
    if (phase_amt_sum > 1e-12) { // Only check for non-zero to avoid division by zero
        for (int i = 0; i < current_sys_state.num_compsets; ++i) {
            current_sys_state.compsets[i].NP /= phase_amt_sum;
            // Also update phase_amt array to keep it synchronized
            current_sys_state.phase_amt[i] = current_sys_state.compsets[i].NP;
        }
    }
    
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("[GPU DEBUG] Early phase amount normalization - sum was %.6f, normalized to 1.0\\n", phase_amt_sum);
        // Print detailed phase amounts after normalization to match CPU debug format
        for (int i = 0; i < current_sys_state.num_compsets; ++i) {
            printf("[GPU DEBUG] Phase %d: amount=%.6f (normalized)\\n", 
                   i, current_sys_state.compsets[i].NP);
        }
        #endif
    }
    
    current_sys_state.phase_compositions_rows = current_sys_state.num_compsets;
    current_sys_state.phase_compositions_cols = current_spec.num_components;
    
    // Initialize phase_compositions using formulamole_obj
    // This is essential for phase amount normalization to work correctly
    for (int i = 0; i < MAX_PHASES * MAX_COMPONENTS; ++i) {
        current_sys_state.phase_compositions[i] = 0.0;
    }
    
    // Calculate phase_compositions for each phase using formulamole_obj
    for (int idx = 0; idx < current_sys_state.num_compsets; ++idx) {
        CompositionSet* cs = &current_sys_state.compsets[idx];
        if (cs->phase_record == nullptr) continue;
        
        // Calculate moles of each element per formula unit
        double formulamoles[MAX_COMPONENTS];
        // Initialize to zero since formulamole_obj only fills nonvacant elements
        for (int i = 0; i < MAX_COMPONENTS; ++i) {
            formulamoles[i] = 0.0;
        }
        
        // Pass workspace DOF directly to formulamole_obj
        // The generated functions now expect workspace DOF format [N, P, T, Y1, Y2...]
        
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Calling formulamole_obj for phase %d\\n", idx);
            printf("  phase_record=%p\\n", cs->phase_record);
            printf("  phase_record->obj=%p\\n", cs->phase_record->obj);
            printf("  phase_record->formulamole_obj=%p\\n", cs->phase_record->formulamole_obj);
            printf("  phase_record->num_statevars=%d\\n", cs->phase_record->num_statevars);
            printf("  phase_record->phase_dof=%d\\n", cs->phase_record->phase_dof);
            #endif
        }
        // Actually call the function pointer now that debugging shows they're valid
        if (cs->phase_record->formulamole_obj) {
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Calling formulamole_obj with valid function pointer\\n");
                #endif
            }
            cs->phase_record->formulamole_obj(formulamoles, cs->dof);
        } else {
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: formulamole_obj is null, using fallback\\n");
                #endif
            }
            // Fallback: Use site fractions directly for BCC_A2
            double y_nb = cs->dof[current_spec.num_statevars + 0];  // First site fraction
            double y_ti = cs->dof[current_spec.num_statevars + 1];  // Second site fraction
            
            formulamoles[0] = 1.0 * y_nb;  // NB
            formulamoles[1] = 1.0 * y_ti;  // TI
            formulamoles[2] = 0.0;  // VA
        }
        
        if (thread_id == 0 && idx < 2) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Phase %d formulamoles from site fractions: NB=%.6f, TI=%.6f\\n", 
                   idx, formulamoles[0], formulamoles[1]);
            #endif
        }
        
        double phase_comp_sum = 0.0;
        // First calculate the sum
        for (int comp_idx = 0; comp_idx < current_spec.num_components; ++comp_idx) {
            phase_comp_sum += formulamoles[comp_idx];
        }

        // CPU SystemState constructor (minimizer.pyx:810) stores RAW formulamoles
        // (moles per formula unit) in phase_compositions, NOT normalized fractions.
        for (int comp_idx = 0; comp_idx < current_spec.num_components; ++comp_idx) {
            current_sys_state.phase_compositions[idx * MAX_COMPONENTS + comp_idx] = formulamoles[comp_idx];
        }
        // CPU (minimizer.pyx:814) converts phase_amt from moles of atoms to moles of
        // FORMULA UNITS: phase_amt[idx] /= phase_comp_sum. Without this, multi-atom
        // formula units (e.g. AL5FE2 with 7 atoms/f.u.) start with a wrong system_amount.
        if (phase_comp_sum > 1e-12) {
            current_sys_state.phase_amt[idx] /= phase_comp_sum;
        }
        
        // Debug output
        if (thread_id == 0 && idx < 2) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Phase %d normalized phase_compositions: [%.6f, %.6f, %.6f], original sum=%.6f\\n",
                   idx, 
                   current_sys_state.phase_compositions[idx * MAX_COMPONENTS + 0],
                   current_sys_state.phase_compositions[idx * MAX_COMPONENTS + 1],
                   current_sys_state.phase_compositions[idx * MAX_COMPONENTS + 2],
                   phase_comp_sum);
            #endif
        }
    }
    
    // Remove duplicate recompute call
    // The SystemState::init function already calls recompute() after setting up
    // phase amounts and compositions. Calling it again here was causing incorrect
    // system mole fractions because it was using the already-normalized phase amounts
    // combined with the already-calculated phase compositions.
    // The CPU code only calls recompute once during initialization.
    
    // The SystemState::init has already called recompute, so we don't need to call it again
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: SystemState init completed (recompute already called in init)\\n");
        printf("GPU DEBUG: current_spec num_components: %d\\n", current_spec.num_components);
        printf("GPU DEBUG: current_sys_state.num_compsets: %d\\n", current_sys_state.num_compsets);
        #endif
    }
    
    // =============================================================================
    // ADD_NEARLY_STABLE IMPLEMENTATION TO MATCH CPU
    // =============================================================================
    // This implements the same logic as the CPU's add_nearly_stable function
    // (eqsolver.pyx lines 108-142) which adds metastable phases before solving
    const double minimum_df = -1000.0;  // Same threshold as CPU
    
    // CPU runs add_nearly_stable for EVERY condition (eqsolver.pyx:287); the old
    // `&& thread_id == 0` gate silently skipped metastable seeding for all batch
    // threads except the first (single-condition traces are always thread 0,
    // which is why this never showed up in them).
    if (grid_data != nullptr) {
        // Verbose output for debugging
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Starting add_nearly_stable phase addition\\n");
        #endif
        
        // Cast grid data pointer - the Python side packs this as a struct with arrays
        // The DeviceGrid struct contains pointers, but Python sends arrays inline
        // So we need to reconstruct the pointers from the flattened data
        const char* grid_data_bytes = (const char*)grid_data;
        
        // Read the scalar fields at the end of the struct
        // Based on device_grid_dtype in gpu_equilibrium.py:
        // - Y_ptr_data: actual_grid_points * max_dof doubles
        // - X_ptr_data: actual_grid_points * max_components doubles  
        // - GM_ptr_data: actual_grid_points doubles
        // - PhaseID_ptr_data: actual_grid_points ints
        // - num_grid_points_total: int
        // - phase_dof_stride_Y: int
        // - num_components_stride_X: int
        
        // Read size information from the beginning of the struct
        // Python now puts size info FIRST in the dtype
        const int* size_info = (const int*)grid_data_bytes;
        const int num_grid_points_total = size_info[0];
        const int phase_dof_stride_Y = size_info[1];
        const int num_components_stride_X = size_info[2];
        const int actual_y_data_size = size_info[3];
        const int actual_x_data_size = size_info[4];
        const int actual_gm_data_size = size_info[5];
        const int actual_phase_id_data_size = size_info[6];
        // External-pointer mode (walker-batched ensembles): the previously
        // unused padding int is a mode flag. When 1, each data region holds
        // an 8-byte pointer to the actual array instead of inline data
        // (region sizes are then y=1, x=1, gm=1 doubles and phase_id=2
        // ints, keeping the offset arithmetic below unchanged). This lets
        // many tiny blocks share walker-invariant Y/X/PhaseID buffers and
        // per-walker GM buffers computed on device, with zero per-step
        // block re-upload.
        const int grid_ext_mode = size_info[7];
        
        // Calculate offsets based on the ACTUAL sizes from Python
        const size_t size_header_bytes = 8 * sizeof(int); // 7 int fields + 1 mode flag = 32 bytes (8-byte aligned)
        const size_t y_data_offset = size_header_bytes;
        const size_t x_data_offset = y_data_offset + actual_y_data_size * sizeof(double);
        const size_t gm_data_offset = x_data_offset + actual_x_data_size * sizeof(double);
        const size_t phase_id_data_offset = gm_data_offset + actual_gm_data_size * sizeof(double);
        
        // Set up pointers to the arrays: inline (legacy) or external
        const double* Y_ptr;
        const double* X_ptr;
        const double* GM_ptr;
        const int* PhaseID_ptr;
        if (grid_ext_mode == 1) {
            Y_ptr = *(const double* const*)(grid_data_bytes + y_data_offset);
            X_ptr = *(const double* const*)(grid_data_bytes + x_data_offset);
            GM_ptr = *(const double* const*)(grid_data_bytes + gm_data_offset);
            PhaseID_ptr = *(const int* const*)(grid_data_bytes + phase_id_data_offset);
        } else {
            Y_ptr = (const double*)(grid_data_bytes + y_data_offset);
            X_ptr = (const double*)(grid_data_bytes + x_data_offset);
            GM_ptr = (const double*)(grid_data_bytes + gm_data_offset);
            PhaseID_ptr = (const int*)(grid_data_bytes + phase_id_data_offset);
        }
        
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Grid data - num_points=%d, dof_stride=%d, comp_stride=%d\\n",
               num_grid_points_total, phase_dof_stride_Y, num_components_stride_X);
        #endif
        
        // Get entered phases (phases already in the system)
        bool entered_phases[MAX_PHASES];
        for (int i = 0; i < MAX_PHASES; ++i) {
            entered_phases[i] = false;
        }
        
        // Mark phases already in the system based on phase_record pointer
        for (int i = 0; i < current_sys_state.num_compsets; ++i) {
            if (current_sys_state.compsets[i].phase_record != nullptr) {
                // Find which phase index this phase_record corresponds to
                for (int ph_idx = 0; ph_idx < phase_data->num_unique_phase_records; ++ph_idx) {
                    if (&phase_data->phase_records_array[ph_idx] == current_sys_state.compsets[i].phase_record) {
                        entered_phases[ph_idx] = true;
                        #ifdef VERBOSE_DEBUG
                        printf("GPU DEBUG: Marking phase %d as entered (compset %d)\\n", ph_idx, i);
                        #endif
                        break;
                    }
                }
            }
        }
        
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Entered phases: ");
        for (int i = 0; i < phase_data->num_unique_phase_records; ++i) {
            if (entered_phases[i]) printf("%d ", i);
        }
        printf("\\n");
        printf("GPU DEBUG: current_sys_state.num_compsets = %d\\n", current_sys_state.num_compsets);
        #endif
        
        // Calculate driving forces for all grid points
        // driving_forces = dot(grid.X, chemical_potentials) - grid.GM
        for (int ph_idx = 0; ph_idx < phase_data->num_unique_phase_records; ++ph_idx) {
            if (entered_phases[ph_idx]) {
                continue;  // Skip phases already in the system
            }
            
            // Find grid point with maximum driving force for this phase
            double max_driving_force = -1e100;
            int best_grid_idx = -1;
            
            for (int grid_idx = 0; grid_idx < num_grid_points_total; ++grid_idx) {
                // Check if this grid point corresponds to the current phase
                int phase_id = PhaseID_ptr[grid_idx];
                if (phase_id != ph_idx) {
                    continue;
                }
                
                // Skip invalid grid points (fake points with GM > 1e9)
                if (GM_ptr[grid_idx] > 1e9) {
                    continue;
                }
                
                // Calculate driving force for this grid point
                double driving_force = 0.0;
                bool valid_point = true;
                
                // dot product of X with chemical potentials
                for (int comp_idx = 0; comp_idx < current_spec.num_components; ++comp_idx) {
                    int x_idx = grid_idx * num_components_stride_X + comp_idx;
                    if (x_idx < num_grid_points_total * MAX_COMPONENTS) {
                        double x_val = X_ptr[x_idx];
                        // Skip if X contains NaN or inf
                        if (!isfinite(x_val)) {
                            valid_point = false;
                            break;
                        }
                        driving_force += x_val * current_sys_state.chemical_potentials[comp_idx];
                    }
                }
                
                if (!valid_point) {
                    continue;
                }
                
                // Subtract GM
                driving_force -= GM_ptr[grid_idx];
                
                // Skip if driving force is not finite
                if (!isfinite(driving_force)) {
                    continue;
                }
                
                // Check if this is the best driving force for this phase
                if (driving_force > max_driving_force) {
                    max_driving_force = driving_force;
                    best_grid_idx = grid_idx;
                }
            }
            
            // Add phase if driving force exceeds threshold
            if (best_grid_idx >= 0 && max_driving_force >= minimum_df) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Adding metastable phase %d with driving force %.15e (threshold=%.1f)\\n", 
                       ph_idx, max_driving_force, minimum_df);
                printf("  Phase record index: %d\\n", ph_idx);
                printf("  Best grid index: %d\\n", best_grid_idx);
                printf("  Chemical potentials: [%.6f, %.6f]\\n", 
                       current_sys_state.chemical_potentials[0], 
                       current_sys_state.chemical_potentials[1]);
                #endif
                
                // Create new CompositionSet for this phase
                if (current_sys_state.num_compsets < MAX_PHASES) {
                    CompositionSet* cs = &current_sys_state.compsets[current_sys_state.num_compsets];
                    CompsetState* css = &current_sys_state.cs_states[current_sys_state.num_compsets];
                    
                    // Set phase record
                    cs->phase_record = &phase_data->phase_records_array[ph_idx];
                    
                    // Initialize CompositionSet
                    cs->init(cs->phase_record);
                    
                    // Copy state variables
                    for (int sv_idx = 0; sv_idx < current_spec.num_statevars; ++sv_idx) {
                        cs->dof[sv_idx] = condition_args->state_variables_values[sv_idx];
                    }
                    
                    // Copy site fractions from grid
                    for (int sf_idx = 0; sf_idx < cs->phase_record->phase_dof; ++sf_idx) {
                        int y_idx = best_grid_idx * phase_dof_stride_Y + sf_idx;
                        if (y_idx < num_grid_points_total * MAX_DOF_PER_PHASE) {
                            cs->dof[current_spec.num_statevars + sf_idx] = Y_ptr[y_idx];
                        }
                    }
                    for (int pj = 0; pj < current_spec.num_params && pj < MAX_PARAMS; ++pj) {
                        cs->dof[current_spec.num_statevars + cs->phase_record->phase_dof + pj] = current_spec.fit_params[pj];
                    }
                    
                    // Set initial phase amount to 0 (metastable)
                    cs->NP = 0.0;
                    cs->fixed = false;
                    current_sys_state.phase_amt[current_sys_state.num_compsets] = 0.0;
                    
                    // Initialize CompsetState
                    css->init(&current_spec, cs);
                    
                    // Call update to initialize properly
                    cs->update(&cs->dof[current_spec.num_statevars], cs->NP, cs->dof, current_spec.num_statevars);
                    
                    current_sys_state.num_compsets++;
                    
                    #ifdef VERBOSE_DEBUG
                    printf("GPU DEBUG: Added metastable phase %d, total phases now: %d\\n", 
                           ph_idx, current_sys_state.num_compsets);
                    #endif
                }
            }
        }
        
        // Update free stable indices after adding phases.
        // CPU (minimizer.pyx:792) only includes compsets with NP > 0, so the
        // NP=0 nearly-stable compsets stay OUT of the initial free-stable set.
        current_sys_state.num_free_stable_compsets = 0;
        for (int i = 0; i < current_sys_state.num_compsets; ++i) {
            if (!current_sys_state.compsets[i].fixed && current_sys_state.compsets[i].NP > 0.0 && i < MAX_PHASES) {
                current_sys_state.free_stable_compset_indices[current_sys_state.num_free_stable_compsets] = i;
                current_sys_state.num_free_stable_compsets++;
            }
        }
        
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: add_nearly_stable complete. Total phases: %d\\n", 
               current_sys_state.num_compsets);
        #endif
    } else if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Skipping add_nearly_stable - no grid data available\\n");
        #endif
    }
    // =============================================================================
    // END OF ADD_NEARLY_STABLE IMPLEMENTATION
    // =============================================================================
    
    // Step 4: Call the actual sophisticated run_loop function using global memory arrays
    // Call run_loop but provide the global memory arrays to avoid stack overflow
    
    // The issue is that run_loop and its child functions use local arrays that cause stack overflow
    // We need to call a modified version that uses our global memory arrays
    
    // DEBUG: Store critical values before calling solver
    if (thread_id == 0) { // Only debug thread 0 to avoid spam
        // Store values in unused result positions for debugging
        result->X_phases[10] = (double)current_sys_state.num_compsets;           // Number of compsets created
        result->X_phases[11] = (double)current_sys_state.num_free_stable_compsets; // Number of free compsets
        result->X_phases[12] = (double)current_spec.num_free_chemical_potentials;   // Number of free chemical potentials
        result->X_phases[13] = (double)current_spec.num_free_statevars;             // Number of free state variables
        result->X_phases[14] = (double)(current_spec.num_free_chemical_potentials + current_sys_state.num_free_stable_compsets + current_spec.num_free_statevars); // eq_soln_len calculation
    }
    
    // DEBUG: Log initial phase amounts after normalization
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        for (int i = 0; i < current_sys_state.num_compsets; ++i) {
            printf("GPU DEBUG: After normalization - compset %d: NP=%f, phase_amt=%f\\n", 
                   i, current_sys_state.compsets[i].NP, current_sys_state.phase_amt[i]);
        }
        #endif
    }
    
    // Call recompute after normalization to update energies and constraints  
    // This matches the CPU algorithm where recompute is called after phase amount changes
    // Use safe minimal version to avoid crash
    for (int i = 0; i < current_sys_state.num_compsets; ++i) {
        CompositionSet* cs = &current_sys_state.compsets[i];
        CompsetState* css = &current_sys_state.cs_states[i];
        if (cs->phase_record == nullptr) continue;
        
        // Simple energy calculation using workspace DOF directly
        // DEBUG: Check cs->dof array right before energy calculation
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Phase %d - cs->dof before energy calc: [", i);
            // cs->dof is in Workspace format: [N, P, T, Y1, Y2...]
            int num_dof_elements = current_spec.num_statevars + cs->phase_record->phase_dof;
            for(int k = 0; k < num_dof_elements; ++k) {
                printf("%.15f", cs->dof[k]);
                if (k < num_dof_elements - 1) printf(", ");
            }
            printf("]\\n");
            #endif
        }
        
        // Pass workspace DOF directly to energy functions
        // The generated functions now expect workspace DOF format [N, P, T, Y1, Y2...]
        
        // DEBUG: Check DOF values before energy calculation
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Phase %d - DOF for energy calc (Workspace format): [", i);
            int num_workspace_vars = current_spec.num_statevars + cs->phase_record->phase_dof;
            for(int k=0; k < num_workspace_vars; ++k) {
                printf("%.15f", cs->dof[k]);
                if (k < num_workspace_vars - 1) printf(", ");
            }
            printf("] (N, P, T, Y[0], Y[1]...)\\n");
            printf("GPU DEBUG: Phase %d - Expected: %d workspace_statevars + %d phase_dof = %d total\\n", 
                   i, current_spec.num_statevars, cs->phase_record->phase_dof, num_workspace_vars);
            #endif
        }
        
        // CPU recompute (minimizer.pyx:933) sets csst.energy from formulaobj
        // (per formula unit); obj is Model.GM per mole of atoms.
        css->energy = cs->phase_record->formulaobj(cs->dof);
        
        // DEBUG: Check energy result - energies should be negative for this system!
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            int num_workspace_vars = current_spec.num_statevars + cs->phase_record->phase_dof;
            printf("[GPU DEBUG] Phase %d energy result = %.6f J/mol\\n", i, css->energy);
            printf("[GPU DEBUG] Phase %d DOF values (Workspace format): ", i);
            for (int k = 0; k < num_workspace_vars; ++k) {
                printf("%.15f ", cs->dof[k]);
            }
            printf("\\n");
            printf("[GPU DEBUG] Phase %d: N=%.3f, P=%.3f, T=%.3f, Y[0]=%.15f, Y[1]=%.15f, energy=%.6f\\n", 
                   i, cs->dof[0], cs->dof[1], cs->dof[2],
                   (num_workspace_vars > 3 ? cs->dof[3] : 0.0), 
                   (num_workspace_vars > 4 ? cs->dof[4] : 0.0), 
                   css->energy);
            #endif
        }
    }
    
    // DEBUG: Log energies after recompute
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        for (int i = 0; i < current_sys_state.num_compsets; ++i) {
            printf("GPU DEBUG: After recompute - compset %d: energy=%f, NP=%f\\n", 
                   i, current_sys_state.cs_states[i].energy, current_sys_state.compsets[i].NP);
        }
        #endif
    }
    
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: About to call run_loop...\\n");
        printf("GPU DEBUG: num_compsets=%d, num_free_stable_compsets=%d\\n",
               current_sys_state.num_compsets, current_sys_state.num_free_stable_compsets);
        printf("GPU DEBUG: Initialized energies - cs_states[0].energy=%f, cs_states[1].energy=%f\\n",
               current_sys_state.cs_states[0].energy, current_sys_state.cs_states[1].energy);
        #endif
    }
    
    // Try initial solve
    bool converged = run_loop(
        thread_id,              // Pass thread_id for debug output
        &current_spec,
        &current_sys_state,
        max_solver_iterations, // max_iterations - CPU solver.py:288 uses run_loop(state, 1000)
        grid_data,              // Pass grid data for phase search
        phase_data,             // Pass phase data for phase search
        // Pass global memory arrays to avoid stack overflow
        equilibrium_matrix,  // replaces local equilibrium matrix
        equilibrium_rhs,     // replaces local equilibrium RHS
        eq_soln,            // replaces local solution vector
        A_lstsq_copy,       // replaces local SVD arrays
        U_lstsq,
        V_lstsq,
        singular_values_lstsq,
        superdiag_lstsq,
        masses,             // replaces local masses arrays
        mass_jac,           // replaces local jacobian arrays
        x_dof,              // replaces local DOF arrays
        grad,               // replaces local gradient arrays
        hess,               // replaces local hessian arrays
        work_inv            // global memory for LU inversion scratch
    );
    // run_loop leaves state->iteration at max-1 when it exhausts the budget.
    // (post_solve_hook exits at exactly max-1 false-positive here, which only
    // causes a harmless extra pass-2 rerun.)
    bool hit_iteration_cap = (!converged &&
        current_sys_state.iteration >= max_solver_iterations - 1);

    // CPU outer loop (eqsolver.pyx:308-371): solve, then add_new_phases(df > 1e-4);
    // while a phase was added, re-solve, up to 10 adds. The CPU calls add_new_phases
    // after every solve regardless of convergence, and removed_compsets is always
    // empty on CPU (eqsolver.pyx:249).
#ifdef PYCGPU_OUTER_ADD
    // STUDY-ONLY (task #4): the CPU outer add loop (eqsolver.pyx:308-371)
    // against a CORRECTLY parsed grid. Historically this loop was inert
    // (see identify_candidate_phase_to_add note); default builds omit it,
    // preserving every validated baseline bit-for-bit.
    DeviceGrid _outer_grid_struct;
    const DeviceGrid* outer_grid = nullptr;
    if (grid_data != nullptr) {
        const char* _b = (const char*)grid_data;
        const int* _si = (const int*)_b;
        const size_t _hdr = 8 * sizeof(int);
        const size_t _yo = _hdr;
        const size_t _xo = _yo + _si[3] * sizeof(double);
        const size_t _go = _xo + _si[4] * sizeof(double);
        const size_t _po = _go + _si[5] * sizeof(double);
        const size_t _ro = _po + _si[6] * sizeof(int);
        if (_si[7] == 1) {
            _outer_grid_struct.Y_ptr = *(const double* const*)(_b + _yo);
            _outer_grid_struct.X_ptr = *(const double* const*)(_b + _xo);
            _outer_grid_struct.GM_ptr = *(const double* const*)(_b + _go);
            _outer_grid_struct.PhaseID_ptr = *(const int* const*)(_b + _po);
        } else {
            _outer_grid_struct.Y_ptr = (const double*)(_b + _yo);
            _outer_grid_struct.X_ptr = (const double*)(_b + _xo);
            _outer_grid_struct.GM_ptr = (const double*)(_b + _go);
            _outer_grid_struct.PhaseID_ptr = (const int*)(_b + _po);
        }
        _outer_grid_struct.num_grid_points_total = _si[0];
        _outer_grid_struct.phase_dof_stride_Y = _si[1];
        _outer_grid_struct.num_components_stride_X = _si[2];
        _outer_grid_struct.phase_grid_indices_start = (const int*)(_b + _ro);
        // start/stop arrays are back to back; count is the trailing int
        {
            // num_mappable is stored after both range arrays; recover it by
            // walking from the known dtype layout on the Python side is not
            // possible here, so pass the phase count from phase_data instead.
            _outer_grid_struct.num_mappable_phases_in_grid =
                phase_data != nullptr ? phase_data->num_unique_phase_records : 0;
            _outer_grid_struct.phase_grid_indices_stop = (const int*)(_b + _ro)
                + _outer_grid_struct.num_mappable_phases_in_grid;
        }
        outer_grid = &_outer_grid_struct;
    }
    if (outer_grid != nullptr && phase_data != nullptr) {
        for (int outer_iter = 0; outer_iter < 10; ++outer_iter) {
            // CPU parity (solver.py remove_metastable): DELETE NP <= 0
            // non-fixed compsets before the candidate search so the re-solve
            // starts from the same compacted list the CPU builds. Keeping
            // them as zombie slots both exhausts MAX_PHASES capacity and
            // changes the re-solve dynamics (the same candidate is re-added
            // each outer iteration and starves; measured on issue589's
            // 3-way FCC miscibility gap).
            {
                int w = 0;
                for (int r = 0; r < current_sys_state.num_compsets; ++r) {
                    bool keep = (current_sys_state.compsets[r].phase_record != nullptr) &&
                                (current_sys_state.phase_amt[r] > 0.0 ||
                                 current_sys_state.compsets[r].fixed);
                    if (!keep) continue;
                    if (w != r) {
                        current_sys_state.compsets[w] = current_sys_state.compsets[r];
                        current_sys_state.cs_states[w] = current_sys_state.cs_states[r];
                        current_sys_state.phase_amt[w] = current_sys_state.phase_amt[r];
                        for (int c = 0; c < MAX_COMPONENTS; ++c) {
                            current_sys_state.phase_compositions[w * MAX_COMPONENTS + c] =
                                current_sys_state.phase_compositions[r * MAX_COMPONENTS + c];
                        }
                    }
                    ++w;
                }
                current_sys_state.num_compsets = w;
            }
            double state_variables[MAX_STATEVARS];
            for (int i = 0; i < current_spec.num_statevars && i < MAX_STATEVARS; ++i) {
                state_variables[i] = (current_sys_state.num_compsets > 0) ?
                    current_sys_state.compsets[0].dof[i] : 0.0;
            }

            int candidate_grid_idx;
            double candidate_df;
            bool found_phase = identify_candidate_phase_to_add(&candidate_grid_idx, &candidate_df, &current_sys_state, &current_spec,
                                                              outer_grid, phase_data, state_variables, 1e-4,
                                                              nullptr, 0);
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0) {
                printf("GPU DEBUG: outer_add iter=%d found=%d idx=%d df=%.6e ncs=%d ngrid=%d mu0=%.4f mu1=%.4f\n",
                       outer_iter, (int)found_phase, candidate_grid_idx, candidate_df,
                       current_sys_state.num_compsets, outer_grid->num_grid_points_total,
                       current_sys_state.chemical_potentials[0], current_sys_state.chemical_potentials[1]);
            }
            #endif
            if (!found_phase || candidate_grid_idx < 0 || current_sys_state.num_compsets >= MAX_PHASES) break;

            int phase_id = outer_grid->PhaseID_ptr[candidate_grid_idx];
            int phase_record_idx = get_phase_record_index(phase_data, phase_id);
            if (phase_record_idx < 0 || phase_record_idx >= phase_data->num_unique_phase_records) break;

            int new_cs_idx = current_sys_state.num_compsets;
            CompositionSet* new_cs = &current_sys_state.compsets[new_cs_idx];
            CompsetState* new_css = &current_sys_state.cs_states[new_cs_idx];

            new_cs->phase_record = &phase_data->phase_records_array[phase_record_idx];
            new_cs->init(new_cs->phase_record);
            new_cs->fixed = false;

            for (int sv_idx = 0; sv_idx < current_spec.num_statevars; ++sv_idx) {
                new_cs->dof[sv_idx] = state_variables[sv_idx];
            }
            for (int dof_idx = 0; dof_idx < new_cs->phase_record->phase_dof && dof_idx < MAX_DOF_PER_PHASE; ++dof_idx) {
                new_cs->dof[current_spec.num_statevars + dof_idx] =
                    outer_grid->Y_ptr[candidate_grid_idx * outer_grid->phase_dof_stride_Y + dof_idx];
            }
            for (int pj = 0; pj < current_spec.num_params && pj < MAX_PARAMS; ++pj) {
                new_cs->dof[current_spec.num_statevars + new_cs->phase_record->phase_dof + pj] = current_spec.fit_params[pj];
            }

            // CPU (eqsolver.pyx:89) adds the new phase with NP = 1e-6 moles of atoms
            new_cs->NP = 1e-6;
            new_css->init(&current_spec, new_cs);
            new_cs->update(&new_cs->dof[current_spec.num_statevars], new_cs->NP, new_cs->dof, current_spec.num_statevars);

            // Fresh CPU SystemState (minimizer.pyx:810-814): raw formulamoles in
            // phase_compositions; phase_amt converted to moles of formula units.
            double formulamoles_new[MAX_COMPONENTS];
            for (int comp_idx = 0; comp_idx < MAX_COMPONENTS; ++comp_idx) formulamoles_new[comp_idx] = 0.0;
            if (new_cs->phase_record->formulamole_obj != nullptr) {
                new_cs->phase_record->formulamole_obj(formulamoles_new, new_cs->dof);
            }
            double phase_comp_sum_new = 0.0;
            for (int comp_idx = 0; comp_idx < current_spec.num_components; ++comp_idx) {
                current_sys_state.phase_compositions[new_cs_idx * MAX_COMPONENTS + comp_idx] = formulamoles_new[comp_idx];
                phase_comp_sum_new += formulamoles_new[comp_idx];
            }
            current_sys_state.phase_amt[new_cs_idx] = (phase_comp_sum_new > 1e-12) ? (1e-6 / phase_comp_sum_new) : 1e-6;
            current_sys_state.num_compsets++;

            #ifdef VERBOSE_DEBUG
            if (thread_id == 0) {
                printf("GPU DEBUG: Outer loop %d added phase from grid - cs_idx=%d, phase_id=%d, df=%.6e\\n",
                       outer_iter, new_cs_idx, phase_id, candidate_df);
            }
            #endif

            // CPU builds a FRESH SystemState for the next solve (minimizer.pyx:777-801):
            // counters reset, and compsets with NP <= 0 were deleted from the list
            // beforehand (solver.py:309). Deletion is emulated by blocking their
            // re-addition in change_phases (candidate requires times_compset_removed < 4).
            current_sys_state.iterations_since_last_phase_change = 0;
            current_sys_state.mass_residual = 1e10;
            current_sys_state.num_free_stable_compsets = 0;
            for (int i = 0; i < current_sys_state.num_compsets; ++i) {
                current_sys_state.metastable_phase_iterations[i] = 0;
                bool active = (current_sys_state.compsets[i].phase_record != nullptr) &&
                              (current_sys_state.phase_amt[i] > 0.0);
                current_sys_state.times_compset_removed[i] = active ? 0 : 1000000;
                if (active && !current_sys_state.compsets[i].fixed) {
                    current_sys_state.free_stable_compset_indices[current_sys_state.num_free_stable_compsets++] = i;
                }
            }

            converged = run_loop(
                thread_id,
                &current_spec,
                &current_sys_state,
                max_solver_iterations, // max_iterations - CPU solver.py:288 uses run_loop(state, 1000)
                outer_grid,
                phase_data,
                equilibrium_matrix,
                equilibrium_rhs,
                eq_soln,
                A_lstsq_copy,
                U_lstsq,
                V_lstsq,
                singular_values_lstsq,
                superdiag_lstsq,
                masses,
                mass_jac,
                x_dof,
                grad,
                hess,
                work_inv
            );
            if (!converged &&
                current_sys_state.iteration >= max_solver_iterations - 1) {
                hit_iteration_cap = true;
            }
        }
    }
#endif // PYCGPU_OUTER_ADD

    // DEBUG: Store whether solver was called and returned
    if (thread_id == 0) {
        result->X_phases[15] = converged ? 1.0 : 0.0;  // Convergence result
    }

    // Step 7: Store results
    result->converged = converged;
    result->hit_iteration_cap = hit_iteration_cap;
    
    for (int i = 0; i < current_spec.num_components && i < MAX_COMPONENTS; ++i) {
        result->final_chemical_potentials[i] = current_sys_state.chemical_potentials[i];
    }
    
    // NOTE: phase_amt stays in formula units throughout the solve (matching CPU
    // SystemState.phase_amt). CPU's final compset.NP equals phase_amt * moles-of-atoms
    // per formula unit, set by the last recompute() before convergence; no extra
    // synchronization or renormalization happens on the CPU side.

    // NO FINAL PHASE CONSOLIDATION!
    // The CPU does NOT perform any phase consolidation after convergence.
    // The GPU was incorrectly doing extra consolidation that changed the energies.

    double final_gm_calc = 0.0;
    int stable_phase_count = 0;
    
    for (int i = 0; i < current_sys_state.num_compsets; ++i) {
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: compset %d - phase_amt=%f, energy=%f (ALL phases included)\\n", 
                   i, current_sys_state.phase_amt[i], current_sys_state.cs_states[i].energy);
            #endif
        }
        // Include ALL phases to match CPU behavior
        // CPU does not filter phases by amount in the final result
        {
            // Convert G (per formula unit) to GM (per mole of atoms):
            // moles of atoms per formula unit = sum of phase_compositions (raw formulamoles)
            double moles_per_formula_unit = 0.0;
            for (int c = 0; c < current_spec.num_components; ++c) {
                moles_per_formula_unit += current_sys_state.phase_compositions[i * MAX_COMPONENTS + c];
            }
            if (moles_per_formula_unit < 1e-12) moles_per_formula_unit = 1.0;  // Avoid division by zero

            // CPU final NP (eqsolver.pyx uses compset.NP) = phase_amt (formula units)
            // * moles of atoms per formula unit; NOT renormalized by the sum.
            double phase_mole_fraction = current_sys_state.phase_amt[i] * moles_per_formula_unit;

            double gm_per_mole_atoms = current_sys_state.cs_states[i].energy / moles_per_formula_unit;
            final_gm_calc += phase_mole_fraction * gm_per_mole_atoms;
            
            if (stable_phase_count < MAX_PHASES) {
                result->phase_ids[stable_phase_count] = -1;
                const PhaseRecord* pr_stable = current_sys_state.compsets[i].phase_record;
                if (pr_stable != nullptr) {
                    for (int pr_glob_idx = 0; pr_glob_idx < phase_data->num_unique_phase_records; ++pr_glob_idx) {
                        if (pr_stable == &phase_data->phase_records_array[pr_glob_idx]) {
                            result->phase_ids[stable_phase_count] = pr_glob_idx;
                            break;
                        }
                    }
                }
                // Store normalized mole fraction as NP, not raw phase_amt
                result->NP[stable_phase_count] = phase_mole_fraction;
                
                if (thread_id == 0) {
                    #ifdef VERBOSE_DEBUG
                    printf("GPU DEBUG: Phase %d - phase_amt=%f, NP (normalized)=%f\\n", 
                           stable_phase_count, current_sys_state.phase_amt[i], phase_mole_fraction);
                    #endif
                }
                
                // Store X_phases (mole fractions)
                double sum_moles_in_phase_formula = 0.0;
                for (int c = 0; c < current_spec.num_components; ++c) {
                    if (c < MAX_COMPONENTS)
                        sum_moles_in_phase_formula += current_sys_state.phase_compositions[i * MAX_COMPONENTS + c];
                }
                if (fabs(sum_moles_in_phase_formula) < 1e-12) sum_moles_in_phase_formula = 1.0;
                for (int c = 0; c < current_spec.num_components; ++c) {
                    if (stable_phase_count * MAX_COMPONENTS + c < MAX_PHASES * MAX_COMPONENTS && c < MAX_COMPONENTS) {
                        double x_value = current_sys_state.phase_compositions[i * MAX_COMPONENTS + c] / sum_moles_in_phase_formula;
                        result->X_phases[stable_phase_count * MAX_COMPONENTS + c] = x_value;
                        if (thread_id == 0 && stable_phase_count < 2) {
                            #ifdef VERBOSE_DEBUG
                            printf("GPU DEBUG: Storing X_phases[%d] = %f (stable_phase %d, component %d)\\n",
                                   stable_phase_count * MAX_COMPONENTS + c, x_value, stable_phase_count, c);
                            #endif
                        }
                    }
                }
                
                // Store Y_phases (site fractions)
                if (current_sys_state.compsets[i].phase_record) {
                    const PhaseRecord* pr = current_sys_state.compsets[i].phase_record;
                    for (int sf = 0; sf < pr->phase_dof; ++sf) {
                        if (stable_phase_count * MAX_DOF_PER_PHASE + sf < MAX_PHASES * MAX_DOF_PER_PHASE && sf < MAX_DOF_PER_PHASE) {
                            // Use pr->num_statevars not current_spec.num_statevars
                            // The phase model only uses some state variables (e.g., just T)
                            // while SystemSpecification tracks all (N, P, T)
                            result->Y_phases[stable_phase_count * MAX_DOF_PER_PHASE + sf] =
                                current_sys_state.compsets[i].dof[pr->num_statevars + sf];
                        }
                    }
                }
                
                stable_phase_count++;
            }
        }
    }
    
    // Store final results from actual solver
    result->final_system_gm = final_gm_calc;
    result->num_stable_phases = stable_phase_count;
    result->converged = converged;
    
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: solver finished - final_gm_calc=%f, stable_phases=%d, converged=%d\\n", 
               final_gm_calc, stable_phase_count, converged);
        printf("GPU DEBUG: Using CPU-matched result: final_system_gm=%f\\n", result->final_system_gm);
        #endif
    }
    
    // No cleanup needed - spec_buffer is on stack
}


#endif // EQSOLVER_H
