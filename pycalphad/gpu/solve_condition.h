// Per-condition equilibrium entry point shared by every backend.
//
// One call solves one condition: it unpacks that condition's rows of the
// packed input buffers (system specification, state variables/compositions,
// starting point), runs solve_equilibrium_at_condition, and writes the
// condition's results row. The CUDA kernel, the C++ driver and the Metal
// kernel are thin wrappers that supply the per-condition work memory.
//
// PYCGPU_GLOBAL marks pointers into buffers shared by all conditions; it is
// empty on CUDA/C++ and expands to Metal's `device` address space.
#ifndef PYCGPU_GLOBAL
#define PYCGPU_GLOBAL
#endif

// Results row layout (doubles): GM, MU[MAX_COMPONENTS], NP[MAX_PHASES],
// converged, num_stable_phases, T, P, hit_iteration_cap,
// Y[MAX_PHASES * MAX_DOF_PER_PHASE], X[MAX_PHASES * MAX_COMPONENTS],
// phase_ids[MAX_PHASES].
#define PYCGPU_RESULTS_PER_CONDITION \
    (7 + MAX_COMPONENTS + MAX_PHASES + (MAX_PHASES * MAX_DOF_PER_PHASE) \
     + (MAX_PHASES * MAX_COMPONENTS) + MAX_PHASES)

// Starting-point row layout (doubles): phase_indices[MAX_PHASES],
// phase_amounts[MAX_PHASES], Y[MAX_PHASES * MAX_DOF_PER_PHASE],
// X[MAX_PHASES * MAX_COMPONENTS], MU[MAX_COMPONENTS], num_phases.
#define PYCGPU_IPD_ROW_LEN \
    (2 * MAX_PHASES + MAX_PHASES * MAX_DOF_PER_PHASE + MAX_PHASES * MAX_COMPONENTS \
     + MAX_COMPONENTS + 1)

// Per-condition scratch memory: device slices on CUDA, host slices on C++,
// thread-private arrays on Metal.
typedef struct PycgpuWork {
    double* A_lstsq_copy;
    double* U_lstsq;
    double* V_lstsq;
    double* singular_values_lstsq;
    double* superdiag_lstsq;
    double* U_inv;
    double* V_inv;
    double* singular_values_inv;
    double* superdiag_inv;
    double* work_inv;
    double* x_dof;
    double* grad;
    double* hess;
    double* masses;
    double* mass_jac;
    double* phase_matrix;
    double* equilibrium_matrix;
    double* equilibrium_rhs;
    double* eq_soln;
    double* system_state;
    double* delta_ms;
    double* phase_compositions;
    double* phase_amounts_per_mole_atoms;
} PycgpuWork;

__device__ void pycgpu_solve_condition(
    int condition_idx,
    PYCGPU_GLOBAL const double* system_specs, int system_spec_stride,
    PYCGPU_GLOBAL const double* condition_data, int condition_stride, int python_max_statevars,
    PYCGPU_GLOBAL double* results,
    PYCGPU_GLOBAL const double* initial_phase_data, int initial_phase_data_stride,
    PYCGPU_GLOBAL const char* grid_data, PYCGPU_GLOBAL const int* grid_block_indices,
    long long grid_block_stride_bytes,
    const PhaseRecord* phase_records, int num_phase_records,
    PycgpuWork work,
    int max_solver_iterations,
    double* jansson_out)
{
    const long long base = (long long)condition_idx * PYCGPU_RESULTS_PER_CONDITION;
    for (int i = 0; i < PYCGPU_RESULTS_PER_CONDITION; ++i) {
        results[base + i] = 0.0;
    }

    PYCGPU_GLOBAL const double* spec_row = system_specs + (long long)condition_idx * system_spec_stride;
    PYCGPU_GLOBAL const double* cond_row = condition_data + (long long)condition_idx * condition_stride;

    // State variables in the packed condition row are [N, P, T] (or [N, T]).
    const int num_statevars = (int)spec_row[0];
    const int num_components = (int)spec_row[1];
    double pressure = 101325.0;
    double temp = 298.15;
    if (num_statevars == 2) {
        temp = cond_row[1];
    } else if (num_statevars >= 3) {
        pressure = cond_row[1];
        temp = cond_row[2];
    }

    // Prescribed mole fractions follow the state variables; the one
    // component without a prescription (negative sentinel) gets the balance,
    // and the trailing slot (VA) is zero.
    double mole_fractions[MAX_COMPONENTS];
    for (int i = 0; i < MAX_COMPONENTS; ++i) {
        mole_fractions[i] = (i < num_components) ? (double)cond_row[python_max_statevars + i] : 0.0;
    }
    {
        double prescribed_sum = 0.0;
        int unprescribed_idx = -1;
        for (int i = 0; i < num_components - 1; ++i) {
            if (mole_fractions[i] > -1e-10) {
                prescribed_sum += mole_fractions[i];
            } else {
                unprescribed_idx = i;
            }
        }
        if (unprescribed_idx >= 0) {
            mole_fractions[unprescribed_idx] = 1.0 - prescribed_sum;
        }
        if (num_components > 0) {
            mole_fractions[num_components - 1] = 0.0;
        }
    }

    // This condition's starting point (copied to local memory).
    double ipd[PYCGPU_IPD_ROW_LEN];
    {
        PYCGPU_GLOBAL const double* ipd_row = initial_phase_data + (long long)condition_idx * initial_phase_data_stride;
        const int n = initial_phase_data_stride < PYCGPU_IPD_ROW_LEN ? initial_phase_data_stride : PYCGPU_IPD_ROW_LEN;
        for (int i = 0; i < n; ++i) ipd[i] = ipd_row[i];
        for (int i = n; i < PYCGPU_IPD_ROW_LEN; ++i) ipd[i] = 0.0;
    }
    const int num_phases_offset = 2 * MAX_PHASES + MAX_PHASES * MAX_DOF_PER_PHASE
                                  + MAX_PHASES * MAX_COMPONENTS + MAX_COMPONENTS;
    const int mu_offset = 2 * MAX_PHASES + MAX_PHASES * MAX_DOF_PER_PHASE + MAX_PHASES * MAX_COMPONENTS;
    const int raw_num_phases = (int)ipd[num_phases_offset];
    const int num_phases = (raw_num_phases > 0 && raw_num_phases <= MAX_PHASES) ? raw_num_phases : 0;
    if (num_phases == 0) {
        return;  // no starting assemblage: reported as not converged
    }

    int phase_indices[MAX_PHASES];
    double phase_amounts[MAX_PHASES];
    for (int i = 0; i < MAX_PHASES; ++i) {
        phase_indices[i] = (int)ipd[i];
        phase_amounts[i] = ipd[MAX_PHASES + i];
    }
    double chemical_potentials[MAX_COMPONENTS];
    for (int i = 0; i < MAX_COMPONENTS; ++i) {
        chemical_potentials[i] = (i < num_components) ? ipd[mu_offset + i] : 0.0;
    }

    // System specification.
    SystemSpecification spec;
    memset(&spec, 0, sizeof(SystemSpecification));
    {
        int k = 0;
        spec.num_statevars = (int)spec_row[k++];
        spec.num_components = (int)spec_row[k++];
        spec.prescribed_system_amount = spec_row[k++];
        for (int i = 0; i < MAX_COMPONENTS; ++i) spec.initial_chemical_potentials[i] = spec_row[k++];
        for (int i = 0; i < MAX_FIXED_MOLE_FRACTION_CONDITIONS; ++i)
            for (int j = 0; j < MAX_COMPONENTS; ++j)
                spec.prescribed_mole_fraction_coefficients[i][j] = spec_row[k++];
        for (int i = 0; i < MAX_FIXED_MOLE_FRACTION_CONDITIONS; ++i) spec.prescribed_mole_fraction_rhs[i] = spec_row[k++];
        spec.num_prescribed_mole_fraction_conditions = (int)spec_row[k++];
        spec.num_prescribed_mole_fraction_coefficients_cols = (int)spec_row[k++];
        for (int i = 0; i < MAX_COMPONENTS; ++i) spec.free_chemical_potential_indices[i] = (int)spec_row[k++];
        spec.num_free_chemical_potentials = (int)spec_row[k++];
        for (int i = 0; i < MAX_STATEVARS; ++i) spec.free_statevar_indices[i] = (int)spec_row[k++];
        spec.num_free_statevars = (int)spec_row[k++];
        for (int i = 0; i < MAX_COMPONENTS; ++i) spec.fixed_chemical_potential_indices[i] = (int)spec_row[k++];
        spec.num_fixed_chemical_potentials = (int)spec_row[k++];
        for (int i = 0; i < MAX_STATEVARS; ++i) spec.fixed_statevar_indices[i] = (int)spec_row[k++];
        spec.num_fixed_statevars = (int)spec_row[k++];
        for (int i = 0; i < MAX_PHASES; ++i) spec.fixed_stable_compset_indices[i] = (int)spec_row[k++];
        spec.num_fixed_stable_compsets = (int)spec_row[k++];
        spec.max_num_free_stable_phases = (int)spec_row[k++];
        spec.num_determined_components = (int)spec_row[k++];
        for (int i = 0; i < MAX_COMPONENTS; ++i) spec.determined_component_indices[i] = (int)spec_row[k++];
        for (int i = 0; i < MAX_COMPONENTS; ++i) spec.determined_component_mole_fractions[i] = spec_row[k++];
        spec.num_undetermined_condition_rows = (int)spec_row[k++];
        for (int i = 0; i < MAX_FIXED_MOLE_FRACTION_CONDITIONS; ++i)
            spec.undetermined_condition_row_indices[i] = (int)spec_row[k++];
        spec.num_phase_local_conditions_total = (int)spec_row[k++];
        for (int i = 0; i < MAX_PHASE_LOCAL_CONDITIONS; ++i) {
            spec.plc_model[i] = (int)spec_row[k++];
            spec.plc_type[i] = (int)spec_row[k++];
            spec.plc_target[i] = (int)spec_row[k++];
            spec.plc_value[i] = spec_row[k++];
        }
        for (int i = 0; i < MAX_PARAMS; ++i) spec.fit_params[i] = spec_row[k++];
        spec.num_params = (int)spec_row[k++];
    }

    // Validate the starting assemblage and evaluate its energy (the solver
    // starts from it; implausible inputs are reported as not converged).
    double system_gm = 0.0;
    for (int ph = 0; ph < num_phases; ++ph) {
        const int record_idx = phase_indices[ph];
        const double amount = phase_amounts[ph];
        if (!(amount > 1e-12 && record_idx >= 0 && record_idx < num_phase_records)) continue;
        const PhaseRecord* rec = &phase_records[record_idx];
        double dof[MAX_STATEVARS + MAX_DOF_PER_PHASE + MAX_PARAMS];
        double n_val = 1.0, p_val = 101325.0, t_val = 298.15;
        if (num_statevars == 2) {
            n_val = cond_row[0];
            t_val = cond_row[1];
        } else if (num_statevars >= 3) {
            n_val = cond_row[0];
            p_val = cond_row[1];
            t_val = cond_row[2];
        }
        if (isnan(n_val) || isinf(n_val) || n_val <= 0.0) return;
        if (isnan(t_val) || isinf(t_val) || t_val <= 0.0) return;
        dof[0] = n_val;
        dof[1] = p_val;
        dof[2] = t_val;
        const int y_offset = 2 * MAX_PHASES + ph * MAX_DOF_PER_PHASE;
        for (int sf = 0; sf < rec->phase_dof && sf < MAX_DOF_PER_PHASE; ++sf) {
            const double y = ipd[y_offset + sf];
            if (isnan(y) || isinf(y) || y < 0.0 || y > 1.0) return;
            dof[3 + sf] = y;
        }
#if MAX_PARAMS > 0
        for (int j = 0; j < spec.num_params && j < MAX_PARAMS; ++j) {
            dof[3 + rec->phase_dof + j] = spec.fit_params[j];
        }
#endif
        double g = 0.0;
        if (rec->obj != nullptr) {
            g = rec->obj(dof);
            if (isnan(g) || isinf(g) || g > 1e8 || g < -1e8) return;
        }
        if (isnan(amount) || isinf(amount) || amount < 0.0) return;
        const double contribution = amount * g;
        if (isnan(contribution) || isinf(contribution)) return;
        system_gm += contribution;
    }

    ConditionArgsSingle condition_args;
    for (int i = 0; i < MAX_STATEVARS; ++i) {
        condition_args.state_variables_values[i] = (i < spec.num_statevars) ? (double)cond_row[i] : 0.0;
    }

    DevicePhaseData phase_data;
    phase_data.phase_records_array = phase_records;
    phase_data.num_unique_phase_records = num_phase_records;
    phase_data.grid_phase_id_to_record_index = nullptr;
    phase_data.max_grid_phase_id = 0;

    PYCGPU_GLOBAL const char* grid_block = grid_data;
    if (grid_data != nullptr && grid_block_indices != nullptr && grid_block_stride_bytes > 0) {
        grid_block = grid_data + (long long)grid_block_indices[condition_idx] * grid_block_stride_bytes;
    }

    EquilibriumResultSingle result;
    result.converged = false;
    result.hit_iteration_cap = false;
    result.final_system_gm = system_gm;
    result.num_stable_phases = 0;
    for (int i = 0; i < MAX_COMPONENTS; ++i) result.final_chemical_potentials[i] = chemical_potentials[i];
    for (int i = 0; i < MAX_PHASES; ++i) {
        result.phase_ids[i] = -1;
        result.NP[i] = 0.0;
    }
    for (int i = 0; i < MAX_PHASES * MAX_COMPONENTS; ++i) result.X_phases[i] = 0.0;
    for (int i = 0; i < MAX_PHASES * MAX_DOF_PER_PHASE; ++i) result.Y_phases[i] = 0.0;

    solve_equilibrium_at_condition(
        condition_idx, &spec, &condition_args, &result, &phase_data, ipd,
        (PYCGPU_GLOBAL const DeviceGrid*)grid_block, mole_fractions,
        work.A_lstsq_copy, work.U_lstsq, work.V_lstsq,
        work.singular_values_lstsq, work.superdiag_lstsq,
        work.U_inv, work.V_inv, work.singular_values_inv, work.superdiag_inv, work.work_inv,
        work.x_dof, work.grad, work.hess, work.masses, work.mass_jac, work.phase_matrix,
        work.equilibrium_matrix, work.equilibrium_rhs, work.eq_soln,
        work.system_state, work.delta_ms, work.phase_compositions,
        work.phase_amounts_per_mole_atoms,
        max_solver_iterations, jansson_out);

    if (isnan(result.final_system_gm) || isinf(result.final_system_gm)) return;
    if (isnan(result.final_chemical_potentials[0]) || isinf(result.final_chemical_potentials[0])) return;

    results[base + 0] = result.final_system_gm;
    for (int i = 0; i < MAX_COMPONENTS; ++i) results[base + 1 + i] = result.final_chemical_potentials[i];
    for (int i = 0; i < MAX_PHASES; ++i) results[base + 1 + MAX_COMPONENTS + i] = result.NP[i];
    results[base + 1 + MAX_COMPONENTS + MAX_PHASES] = result.converged ? 1.0 : 0.0;
    results[base + 2 + MAX_COMPONENTS + MAX_PHASES] = (double)result.num_stable_phases;
    results[base + 3 + MAX_COMPONENTS + MAX_PHASES] = temp;
    results[base + 4 + MAX_COMPONENTS + MAX_PHASES] = pressure;
    results[base + 5 + MAX_COMPONENTS + MAX_PHASES] = result.hit_iteration_cap ? 1.0 : 0.0;
    const long long y_base = base + 6 + MAX_COMPONENTS + MAX_PHASES;
    for (int i = 0; i < MAX_PHASES * MAX_DOF_PER_PHASE; ++i) results[y_base + i] = result.Y_phases[i];
    const long long x_base = y_base + MAX_PHASES * MAX_DOF_PER_PHASE;
    for (int i = 0; i < MAX_PHASES * MAX_COMPONENTS; ++i) results[x_base + i] = result.X_phases[i];
    const long long id_base = x_base + MAX_PHASES * MAX_COMPONENTS;
    for (int i = 0; i < MAX_PHASES; ++i) results[id_base + i] = (double)result.phase_ids[i];
}
