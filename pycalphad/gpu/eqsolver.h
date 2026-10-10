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
    PYCGPU_GLOBAL const double* Y_ptr;    // Flattened array of site fractions for all grid points
    PYCGPU_GLOBAL const double* X_ptr;    // Flattened array of mole compositions for all grid points
    PYCGPU_GLOBAL const double* GM_ptr;   // Array of Gibbs energies for all grid points
    PYCGPU_GLOBAL const int* PhaseID_ptr; // Flattened array of phase identifiers for all grid points
                            // (integer IDs mapping to PhaseRecord in DevicePhaseData)

    int num_grid_points_total; // Total number of rows in the grid arrays
    int phase_dof_stride_Y;    // Max number of site fractions for any phase (stride for Y_ptr)
    int num_components_stride_X; // Number of components (stride for X_ptr)

    // For add_nearly_stable: mapping a global phase ID to its start/end row in the grid
    // This replaces grid.attrs['phase_indices']
    PYCGPU_GLOBAL const int* phase_grid_indices_start; // Array indexed by global phase ID
    PYCGPU_GLOBAL const int* phase_grid_indices_stop;  // Array indexed by global phase ID
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
// The CPU outer add_new_phases candidate search. (Historical note: an
// early version dereferenced the raw grid block as a pointer-struct and
// never ran; this implementation parses the DeviceGrid correctly.)
// DEFAULT ON: the current reference exercises the outer add loop, e.g.
// re-seeding the second FCC_L12 composition set in gamma/gamma-prime
// regions. PYCGPU_OUTER_ADD=0 compiles it out.
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
typedef struct EquilibriumResultSingle {
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
typedef struct ConditionArgsSingle {
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
    PYCGPU_GLOBAL const DeviceGrid* grid_data,
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
    PYCGPU_GLOBAL const DeviceGrid* grid_data,  // packed grid block (see gpu_equilibrium)
    const double* condition_mole_fractions, // the condition's prescribed mole fractions
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
    double* delta_ms,            // work memory for SystemState.delta_ms
    double* phase_compositions,  // work memory for SystemState.phase_compositions
    double* phase_amounts_per_mole_atoms,  // work memory for SystemState._phase_amounts_per_mole_atoms_arr
    int max_solver_iterations,   // Newton-loop budget (CPU parity: 1000)
    double* jansson_out          // optional Jansson delta block (PYCGPU_JANSSON_TARGET)
) {
    
    
    // Step 1: Validate inputs and global memory arrays
    if (!A_lstsq_copy || !result || !global_spec_base || !initial_data) {
        if (result) {
            result->converged = false;
            result->hit_iteration_cap = false;
        }
        return; // Cannot proceed without required arrays - memory not allocated yet
    }
    
    // Step 2: Copy this condition's SystemSpecification (packed by Python
    // with the same layout) into thread-local storage.
    char spec_buffer[sizeof(SystemSpecification)];
    SystemSpecification* current_spec_ptr = (SystemSpecification*)spec_buffer;
    SystemSpecification& current_spec = *current_spec_ptr;
    const char* spec_bytes = (const char*)global_spec_base;
    memcpy(current_spec_ptr, spec_bytes, sizeof(SystemSpecification));

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
    current_sys_state.condition_idx = thread_id;  // Condition index (used by the PYCGPU_TRACE_* output)
    
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

    // CPU parity (eqsolver.pyx:200-212): a condition whose prescribed mole
    // fractions sum above one is infeasible and is skipped silently (NaN
    // result; the CPU allows these to make 2-D composition mapping easy).
    // Without the guard these conditions burn the full iteration budget —
    // 1000 inner iterations plus 10 outer phase-add re-solves — before
    // failing; on composition-grid workloads that was most of the runtime.
    //
    // The sum is taken over the spec's prescribed mole-fraction constraint
    // rows (identity rows only — those are the plain X() conditions the CPU
    // counts; W()/linear-combination rows are excluded, as on the CPU).
    // Do NOT compute this from condition_mole_fractions: that array carries
    // backend-transformed values whose meaning varies by path (dilute
    // rewrites push its sum slightly above one; on charged systems it does
    // not hold mole fractions at all).
    {
        double indep_sum = 0.0;
        int ncols = current_spec.num_prescribed_mole_fraction_coefficients_cols;
        if (ncols > MAX_COMPONENTS) ncols = MAX_COMPONENTS;
        for (int r = 0; r < current_spec.num_prescribed_mole_fraction_conditions; ++r) {
            int ones = 0;
            bool identity = true;
            for (int c = 0; c < ncols; ++c) {
                double coef = current_spec.prescribed_mole_fraction_coefficients[r][c];
                if (fabs(coef - 1.0) < 1e-12) ones++;
                else if (fabs(coef) > 1e-12) { identity = false; break; }
            }
            if (identity && ones == 1) {
                indep_sum += current_spec.prescribed_mole_fraction_rhs[r];
            }
        }
        if (indep_sum > 1.0) {
            result->converged = false;
            result->hit_iteration_cap = false;
            result->num_stable_phases = 0;
            return;
        }
    }
    
    // Access num_phases and phase_indices from flat array
    // NOTE: The initial_data pointer is already offset to this thread's data
    // Layout: phase_indices[MAX_PHASES] + phase_amounts[MAX_PHASES] + site_fractions[MAX_PHASES*MAX_DOF_PER_PHASE] + 
    //         compositions[MAX_PHASES*MAX_COMPONENTS] + chemical_potentials[MAX_COMPONENTS] + num_phases[1]
    int num_phases_offset = MAX_PHASES + MAX_PHASES + (MAX_PHASES * MAX_DOF_PER_PHASE) + 
                           (MAX_PHASES * MAX_COMPONENTS) + MAX_COMPONENTS;
    int num_phases = (int)initial_data_flat[num_phases_offset];
    
    // Set up initial composition sets from lower_convex_hull data
    
    for (int i = 0; i < num_phases && i < MAX_PHASES; ++i) {
        int pr_idx = (int)initial_data_flat[i];  // phase_indices are at the beginning
        
        // Access phase_amounts from flat array 
        // phase_amounts start at offset MAX_PHASES
        double phase_amount = initial_data_flat[MAX_PHASES + i];
        
        if (pr_idx < 0 || pr_idx >= phase_data->num_unique_phase_records) {
            continue;
        }
        // CPU (eqsolver.pyx:268) floors phase_amt at MIN_PHASE_FRACTION and KEEPS the
        // phase; the Python side already applied that floor, so only skip truly empty slots.
        if (phase_amount < MIN_PHASE_FRACTION) {
            continue;
        }
        
        // Bounds check to prevent overflow
        if (current_sys_state.num_compsets >= MAX_PHASES) {
            break;
        }
        
        // Set up CompositionSet directly in SystemState (avoiding stack arrays)
        CompositionSet* cs = &current_sys_state.compsets[current_sys_state.num_compsets];
        CompsetState* css = &current_sys_state.cs_states[current_sys_state.num_compsets];
        
        // Initialize the CompositionSet
        if (phase_data == nullptr || phase_data->phase_records_array == nullptr) {
            return;
        }
        // Each composition set needs its own phase record instance
        // to avoid sharing memory between phases of the same type (immiscibility gap)
        cs->phase_record = &phase_data->phase_records_array[pr_idx];
        if (!cs->phase_record) continue;
        
        // Set state variables from condition args  
        // The DOF array should store WORKSPACE state variables, not Model state variables
        // CPU stores DOF as [N, P, T, Y1, Y2...] (workspace format)
        // GPU was incorrectly storing as [T, Y1, Y2...] (model format)
        
        // Copy ALL workspace state variables to match CPU behavior
        for (int sv_idx = 0; sv_idx < current_spec.num_statevars && sv_idx < MAX_STATEVARS; ++sv_idx) {
            cs->dof[sv_idx] = condition_args->state_variables_values[sv_idx];
        }
        
        
        // Set site fractions from lower_convex_hull results
        // Site fractions start after the WORKSPACE's state variables
        // Layout: phase_indices[MAX_PHASES] + phase_amounts[MAX_PHASES] + site_fractions[MAX_PHASES*MAX_DOF_PER_PHASE] + ...
        // site_fractions start at offset (MAX_PHASES + MAX_PHASES)
        int site_fractions_offset = MAX_PHASES + MAX_PHASES;  // phase_indices + phase_amounts
        const double* initial_data_flat = initial_data;
        
        // Site fractions arrive in the Model's variable order; they follow the
        // workspace's state variables in the DOF vector.
        for (int sf_idx = 0; sf_idx < cs->phase_record->phase_dof && sf_idx < MAX_DOF_PER_PHASE; ++sf_idx) {
            int flat_index = site_fractions_offset + i * MAX_DOF_PER_PHASE + sf_idx;
            cs->dof[current_spec.num_statevars + sf_idx] = initial_data_flat[flat_index];
        }
        
            // Runtime fit parameters ride in trailing dof slots after this
            // phase's site fractions (read by the generated functions).
            for (int pj = 0; pj < current_spec.num_params && pj < MAX_PARAMS; ++pj) {
                cs->dof[current_spec.num_statevars + cs->phase_record->phase_dof + pj] = current_spec.fit_params[pj];
            }
            pycgpu_attach_local_conditions(cs, &current_spec, pr_idx);
        
        // Set phase amount and properties
        cs->NP = phase_amount;
        cs->fixed = false;
        current_sys_state.phase_amt[current_sys_state.num_compsets] = phase_amount;
        
        // Initialize CompositionSet first
        cs->init(cs->phase_record);
        
        
        // Initialize CompsetState with proper arrays
        // This is where masses, jacobians, etc. get set up
        css->init(&current_spec, cs);
        
        
        // STACK OVERFLOW FIX: CompsetState arrays are fixed arrays, not pointers
        // We need a different approach - we'll copy data between CompsetState and global memory
        // Initialize CompsetState arrays with reasonable starting values
        
        if (current_sys_state.num_compsets < MAX_PHASES) {
            // Initialize masses from composition data directly into CompsetState
            for (int comp_idx = 0; comp_idx < MAX_COMPONENTS; ++comp_idx) {
                if (comp_idx < current_spec.num_components) {
                    css->masses[comp_idx] = current_sys_state.mole_fractions[comp_idx];
                } else {
                    css->masses[comp_idx] = 0.0;
                }
            }
            
            // Initialize mass jacobian rows/cols
            css->mass_jac_rows = current_spec.num_components;
            css->mass_jac_cols = current_spec.num_statevars + cs->phase_record->phase_dof;
            
            // Initialize mass jacobian to reasonable values
            int jac_size = css->mass_jac_rows * css->mass_jac_cols;
            int max_jac_size = MAX_COMPONENTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE);
            if (jac_size > max_jac_size) {
                jac_size = max_jac_size;
            }
            for (int j = 0; j < jac_size; ++j) {
                css->mass_jac[j] = 0.0;
            }
        }
        
        // Phase amounts are already normalized at Python level
        // Unlike CPU which normalizes in recompute(), GPU receives pre-normalized values
        double original_phase_amt = cs->NP;  // This is already normalized by Python
        
        // Must call cs->update() to calculate energy and composition
        // The energy field is used in the equilibrium matrix RHS calculation
        // Without this, energy=0 and the solver behaves differently than CPU
        // cs->update expects: (site_fractions, phase_amount, state_variables, workspace_num_statevars)
        // In workspace DOF format: site fractions start at current_spec.num_statevars
        
        cs->update(&cs->dof[current_spec.num_statevars], cs->NP, cs->dof, current_spec.num_statevars);
        
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
        
        if (cs->phase_record->formulamole_obj) {
            cs->phase_record->formulamole_obj(formulamoles, cs->dof);
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
    }
    
    // Remove duplicate recompute call
    // The SystemState::init function already calls recompute() after setting up
    // phase amounts and compositions. Calling it again here was causing incorrect
    // system mole fractions because it was using the already-normalized phase amounts
    // combined with the already-calculated phase compositions.
    // The CPU code only calls recompute once during initialization.
    
    // The SystemState::init has already called recompute, so we don't need to call it again
    
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
        // Cast grid data pointer - the Python side packs this as a struct with arrays
        // The DeviceGrid struct contains pointers, but Python sends arrays inline
        // So we need to reconstruct the pointers from the flattened data
        PYCGPU_GLOBAL const char* grid_data_bytes = (PYCGPU_GLOBAL const char*)grid_data;
        
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
        PYCGPU_GLOBAL const int* size_info = (PYCGPU_GLOBAL const int*)grid_data_bytes;
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
        const size_t x_data_offset = y_data_offset + actual_y_data_size * PYCGPU_F64_BYTES;
        const size_t gm_data_offset = x_data_offset + actual_x_data_size * PYCGPU_F64_BYTES;
        const size_t phase_id_data_offset = gm_data_offset + actual_gm_data_size * PYCGPU_F64_BYTES;
        
        // Set up pointers to the arrays: inline (legacy) or external
        PYCGPU_GLOBAL const double* Y_ptr;
        PYCGPU_GLOBAL const double* X_ptr;
        PYCGPU_GLOBAL const double* GM_ptr;
        PYCGPU_GLOBAL const int* PhaseID_ptr;
#ifndef PYCGPU_METAL
        if (grid_ext_mode == 1) {
            Y_ptr = *(const double* const*)(grid_data_bytes + y_data_offset);
            X_ptr = *(const double* const*)(grid_data_bytes + x_data_offset);
            GM_ptr = *(const double* const*)(grid_data_bytes + gm_data_offset);
            PhaseID_ptr = *(const int* const*)(grid_data_bytes + phase_id_data_offset);
        } else
#endif
        {
            Y_ptr = (PYCGPU_GLOBAL const double*)(grid_data_bytes + y_data_offset);
            X_ptr = (PYCGPU_GLOBAL const double*)(grid_data_bytes + x_data_offset);
            GM_ptr = (PYCGPU_GLOBAL const double*)(grid_data_bytes + gm_data_offset);
            PhaseID_ptr = (PYCGPU_GLOBAL const int*)(grid_data_bytes + phase_id_data_offset);
        }
        
        
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
                        break;
                    }
                }
            }
        }
        
        
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
                    pycgpu_attach_local_conditions(cs, &current_spec, ph_idx);
                    
                    // Set initial phase amount to 0 (metastable)
                    cs->NP = 0.0;
                    cs->fixed = false;
                    current_sys_state.phase_amt[current_sys_state.num_compsets] = 0.0;
                    
                    // Initialize CompsetState
                    css->init(&current_spec, cs);
                    
                    // Call update to initialize properly
                    cs->update(&cs->dof[current_spec.num_statevars], cs->NP, cs->dof, current_spec.num_statevars);
                    
                    current_sys_state.num_compsets++;
                    
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
    }
    // =============================================================================
    // END OF ADD_NEARLY_STABLE IMPLEMENTATION
    // =============================================================================
    
    // Step 4: Call the actual sophisticated run_loop function using global memory arrays
    // Call recompute after normalization to update energies and constraints  
    // This matches the CPU algorithm where recompute is called after phase amount changes
    // Use safe minimal version to avoid crash
    for (int i = 0; i < current_sys_state.num_compsets; ++i) {
        CompositionSet* cs = &current_sys_state.compsets[i];
        CompsetState* css = &current_sys_state.cs_states[i];
        if (cs->phase_record == nullptr) continue;
        
        // Simple energy calculation using workspace DOF directly
        
        // Pass workspace DOF directly to energy functions
        // The generated functions now expect workspace DOF format [N, P, T, Y1, Y2...]
        
        // CPU recompute (minimizer.pyx:933) sets csst.energy from formulaobj
        // (per formula unit); obj is Model.GM per mole of atoms.
        css->energy = cs->phase_record->formulaobj(cs->dof);
    }
    
    // Try initial solve
    bool converged = run_loop(
        thread_id,
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
    // The CPU outer add loop (eqsolver.pyx: solve, add_new_phases(df >
    // 1e-4), re-solve while phases were added, up to 10 adds). DEFAULT ON;
    // PYCGPU_OUTER_ADD=0 compiles it out.
    DeviceGrid _outer_grid_struct;
    const DeviceGrid* outer_grid = nullptr;
    if (grid_data != nullptr) {
        PYCGPU_GLOBAL const char* _b = (PYCGPU_GLOBAL const char*)grid_data;
        PYCGPU_GLOBAL const int* _si = (PYCGPU_GLOBAL const int*)_b;
        const size_t _hdr = 8 * sizeof(int);
        const size_t _yo = _hdr;
        const size_t _xo = _yo + _si[3] * PYCGPU_F64_BYTES;
        const size_t _go = _xo + _si[4] * PYCGPU_F64_BYTES;
        const size_t _po = _go + _si[5] * PYCGPU_F64_BYTES;
        const size_t _ro = _po + _si[6] * sizeof(int);
#ifndef PYCGPU_METAL
        if (_si[7] == 1) {
            _outer_grid_struct.Y_ptr = *(const double* const*)(_b + _yo);
            _outer_grid_struct.X_ptr = *(const double* const*)(_b + _xo);
            _outer_grid_struct.GM_ptr = *(const double* const*)(_b + _go);
            _outer_grid_struct.PhaseID_ptr = *(const int* const*)(_b + _po);
        } else
#endif
        {
            _outer_grid_struct.Y_ptr = (PYCGPU_GLOBAL const double*)(_b + _yo);
            _outer_grid_struct.X_ptr = (PYCGPU_GLOBAL const double*)(_b + _xo);
            _outer_grid_struct.GM_ptr = (PYCGPU_GLOBAL const double*)(_b + _go);
            _outer_grid_struct.PhaseID_ptr = (PYCGPU_GLOBAL const int*)(_b + _po);
        }
        _outer_grid_struct.num_grid_points_total = _si[0];
        _outer_grid_struct.phase_dof_stride_Y = _si[1];
        _outer_grid_struct.num_components_stride_X = _si[2];
        _outer_grid_struct.phase_grid_indices_start = (PYCGPU_GLOBAL const int*)(_b + _ro);
        // start/stop arrays are back to back; count is the trailing int
        {
            // num_mappable is stored after both range arrays; recover it by
            // walking from the known dtype layout on the Python side is not
            // possible here, so pass the phase count from phase_data instead.
            _outer_grid_struct.num_mappable_phases_in_grid =
                phase_data != nullptr ? phase_data->num_unique_phase_records : 0;
            _outer_grid_struct.phase_grid_indices_stop = (PYCGPU_GLOBAL const int*)(_b + _ro)
                + _outer_grid_struct.num_mappable_phases_in_grid;
        }
        outer_grid = &_outer_grid_struct;
    }
    if (outer_grid != nullptr && phase_data != nullptr) {
        #ifdef PYCGPU_OUTER_STATS
        int outer_adds_count = 0;
        long long outer_iters_spent = 0;
        #endif
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
                // Rebuild the stable-index list NOW: the block above moved
                // compsets to new slots, and on the no-candidate break below
                // nothing else refreshes these indices — leaving them stale
                // poisons any post-solve reader of the state (the Jansson
                // epilogue's fill read zeroed global rows through a
                // out-of-range slot; found via matrix dump).
                current_sys_state.num_free_stable_compsets = 0;
                for (int r = 0; r < current_sys_state.num_compsets; ++r) {
                    if (!current_sys_state.compsets[r].fixed) {
                        current_sys_state.free_stable_compset_indices[
                            current_sys_state.num_free_stable_compsets++] = r;
                    }
                }
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
#ifdef PYCGPU_LOCKSTEP
            // gpu-fast frozen-set mode: a positive-driving-force candidate
            // (grid scan) OR a suppressed in-loop add (continuous driving
            // force of a metastable compset — catches ordering-gap splits
            // the grid cannot see) means the assemblage needs growth. Do
            // NOT add — flag for a faithful-kernel rerun and stop.
            if ((found_phase && candidate_grid_idx >= 0)
                || current_sys_state.lockstep_needs_add) {
                hit_iteration_cap = true;
            }
            break;
#else
            if (!found_phase || candidate_grid_idx < 0 || current_sys_state.num_compsets >= MAX_PHASES) break;
#endif

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
            pycgpu_attach_local_conditions(new_cs, &current_spec, phase_record_idx);

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


            // CPU builds a FRESH SystemState for the next solve (minimizer.pyx:777-801):
            // counters reset, and compsets with NP <= 0 were deleted from the list
            // beforehand (solver.py:309). Deletion is emulated by blocking their
            // re-addition in change_phases (candidate requires times_compset_removed < 4).
            current_sys_state.iterations_since_last_phase_change = 0;
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
                grid_data,  // run_loop only forwards it (unused downstream)
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
            #ifdef PYCGPU_OUTER_STATS
            outer_adds_count++;
            outer_iters_spent += current_sys_state.iteration + 1;
            #endif
        }
        #ifdef PYCGPU_OUTER_STATS
        printf("OUTERSTAT tid=%d adds=%d resolve_iters=%lld conv=%d\n",
               thread_id, outer_adds_count, outer_iters_spent, (int)converged);
        #endif
    }
#endif // PYCGPU_OUTER_ADD

#ifdef PYCGPU_JANSSON_TARGET
    // Jansson-derivative epilogue: at the converged state, solve the
    // perturbed equilibrium system for d(MU, NP, statevars)/d(target) and
    // back-substitute the per-compset site-fraction deltas (Sundman 2015
    // Eq. 74/78; reference minimizer.pyx state_variable_differential /
    // site_fraction_differential).  Runs on the converged, recompute'd
    // state with the same per-thread buffers as the Newton solve.
    if (jansson_out != (double*)0) {
        if (converged) {
            // Refresh the per-compset state at the FINAL dof before solving
            // the differentials: run_loop's last iteration leaves grad/hess/
            // c-blocks one advance_state stale, which pollutes the delta
            // solve (~1e-3 relative on d_amt).  The reference does the same
            // via state.recompute(spec) in JanssonDerivative.compute_property.
            current_sys_state.recompute(&current_spec, work_inv);
            pyjan_compute_deltas(&current_spec, &current_sys_state,
                                 PYCGPU_JANSSON_TARGET,
                                 equilibrium_matrix, equilibrium_rhs,
                                 U_lstsq, V_lstsq, singular_values_lstsq,
                                 superdiag_lstsq, jansson_out);
        } else {
            for (int _ji = 0; _ji < PYJAN_COND_STRIDE; ++_ji) jansson_out[_ji] = 0.0;
        }
    }
#endif

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
    
    
    // No cleanup needed - spec_buffer is on stack
}


#endif // EQSOLVER_H
