// Metal kernel entry points (compiled only by the Metal backend, after the
// solver headers and the generated model functions).
//
// Each thread solves one condition entirely in thread-private memory: the
// work arrays and SystemState that the CUDA backend slices out of shared
// device allocations are local arrays here. The generator defines
// PYCGPU_NUM_RECORDS and pycgpu_init_phase_records().
//
// Builds with PYCGPU_JANSSON_TARGET defined also return Jansson derivative
// deltas: one PYJAN_COND_STRIDE block per condition in an extra buffer.

struct PycgpuEqParams {
    int num_conditions;
    int condition_offset;        // first condition of this launch (chunking)
    int condition_stride;
    int python_max_statevars;
    int initial_phase_data_stride;
    int system_spec_stride;
    int max_solver_iterations;
    int has_grid;
    long grid_block_stride_bytes;
};

kernel void pycgpu_equilibrium_kernel(
    PYCGPU_GLOBAL const double* system_specs [[buffer(0)]],
    PYCGPU_GLOBAL const double* condition_data [[buffer(1)]],
    PYCGPU_GLOBAL double* results [[buffer(2)]],
    PYCGPU_GLOBAL const double* initial_phase_data [[buffer(3)]],
    PYCGPU_GLOBAL const char* grid_data [[buffer(4)]],
    PYCGPU_GLOBAL const int* grid_block_indices [[buffer(5)]],
#ifdef PYCGPU_JANSSON_TARGET
    PYCGPU_GLOBAL double* jansson_results [[buffer(6)]],
    constant PycgpuEqParams& p [[buffer(7)]],
#else
    constant PycgpuEqParams& p [[buffer(6)]],
#endif
    uint gid [[thread_position_in_grid]])
{
    const int tid = p.condition_offset + (int)gid;
    if (tid >= p.num_conditions) return;

    PhaseRecord records[PYCGPU_NUM_RECORDS];
    pycgpu_init_phase_records(records);

    double A_lstsq_copy[MAX_SVD_M * MAX_SVD_N];
    double U_lstsq[MAX_SVD_M * MAX_SVD_N];
    double V_lstsq[MAX_SVD_N * MAX_SVD_N];
    double singular_values_lstsq[MAX_SVD_N];
    double superdiag_lstsq[MAX_SVD_N];
    double U_inv[MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM];
    double V_inv[MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM];
    double singular_values_inv[MAX_PHASE_MATRIX_DIM];
    double superdiag_inv[MAX_PHASE_MATRIX_DIM];
    double work_inv[MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM];
    double x_dof[MAX_STATEVARS + MAX_DOF_PER_PHASE];
    double grad[MAX_STATEVARS + MAX_DOF_PER_PHASE];
    double hess[(MAX_STATEVARS + MAX_DOF_PER_PHASE) * (MAX_STATEVARS + MAX_DOF_PER_PHASE)];
    double masses[MAX_COMPONENTS];
    double mass_jac[MAX_COMPONENTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE)];
    double phase_matrix[(MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS) * (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS)];
    double equilibrium_matrix[MAX_EQ_MATRIX_SIZE];
    double equilibrium_rhs[MAX_EQ_MATRIX_ROWS];
    double eq_soln[MAX_EQ_SOLN_LEN];
    SystemState system_state;
    double delta_ms[MAX_PHASES * MAX_COMPONENTS];
    double phase_compositions[MAX_PHASES * MAX_COMPONENTS];
    double phase_amounts_per_mole_atoms[MAX_PHASES * MAX_COMPONENTS];

    PycgpuWork work;
    work.A_lstsq_copy = A_lstsq_copy;
    work.U_lstsq = U_lstsq;
    work.V_lstsq = V_lstsq;
    work.singular_values_lstsq = singular_values_lstsq;
    work.superdiag_lstsq = superdiag_lstsq;
    work.U_inv = U_inv;
    work.V_inv = V_inv;
    work.singular_values_inv = singular_values_inv;
    work.superdiag_inv = superdiag_inv;
    work.work_inv = work_inv;
    work.x_dof = x_dof;
    work.grad = grad;
    work.hess = hess;
    work.masses = masses;
    work.mass_jac = mass_jac;
    work.phase_matrix = phase_matrix;
    work.equilibrium_matrix = equilibrium_matrix;
    work.equilibrium_rhs = equilibrium_rhs;
    work.eq_soln = eq_soln;
    work.system_state = (double*)&system_state;
    work.delta_ms = delta_ms;
    work.phase_compositions = phase_compositions;
    work.phase_amounts_per_mole_atoms = phase_amounts_per_mole_atoms;

    double* jansson_out = (double*)nullptr;
#ifdef PYCGPU_JANSSON_TARGET
    double jansson_local[PYJAN_COND_STRIDE];
    jansson_out = jansson_local;
#endif

    pycgpu_solve_condition(
        tid,
        system_specs, p.system_spec_stride,
        condition_data, p.condition_stride, p.python_max_statevars,
        results,
        initial_phase_data, p.initial_phase_data_stride,
        p.has_grid ? grid_data : (PYCGPU_GLOBAL const char*)nullptr,
        p.has_grid ? grid_block_indices : (PYCGPU_GLOBAL const int*)nullptr,
        p.grid_block_stride_bytes,
        records, PYCGPU_NUM_RECORDS,
        work, p.max_solver_iterations, jansson_out);
#ifdef PYCGPU_JANSSON_TARGET
    for (int i = 0; i < PYJAN_COND_STRIDE; ++i)
        jansson_results[(long)tid * PYJAN_COND_STRIDE + i] = jansson_local[i];
#endif
}
