cimport cython
import numpy as np
cimport numpy as np
from pycalphad.core.composition_set cimport CompositionSet
from pycalphad.core.constants import MIN_SITE_FRACTION as _MIN_SITE_FRACTION
cimport scipy.linalg.cython_lapack as cython_lapack
cimport scipy.linalg.cython_blas as cython_blas
from libc.stdlib cimport malloc, free
from libc.math cimport isnan, INFINITY

@cython.boundscheck(False)
cdef void lstsq(double *A, int M, int N, double* x, double rcond) noexcept nogil:
    # Note: This function will destroy input matrix A
    cdef int i
    cdef int NRHS = 1
    cdef int iwork = 0
    cdef int info = 0
    cdef int SMLSIZ = 50  # this is a guess
    cdef int NLVL = 10  # this is also a guess
    cdef int lwork = 12*N + 2*N*SMLSIZ + 8*N*NLVL + N*NRHS + (SMLSIZ+1)**2
    cdef int rank = 0
    cdef double* work = <double*>malloc(lwork * sizeof(double))
    cdef double* singular_values = <double*>malloc(N * sizeof(double))
    cdef bint isfinite = True

    for i in range(M*N):
        if isnan(A[i]):
            isfinite = False

    if not isfinite:
        for i in range(N):
            x[i] = 0
    else:
        cython_lapack.dgelsd(&M, &N, &NRHS, A, &N, x, &M, singular_values, &rcond, &rank,
                            work, &lwork, &iwork, &info)
    free(singular_values)
    free(work)
    if info != 0:
        for i in range(N):
            x[i] = -1e19

cpdef void lstsq_check_infeasible(double[:,::] A, double[::] b, double[::] out_x):
    # Note: This function will destroy input matrix A
    A_copy = np.copy(A)
    b_copy = np.copy(b)
    lstsq(&A[0,0], A.shape[0], A.shape[1],
        &out_x[0], 1e-16)
    residual = np.sum(np.square(np.dot(A_copy, out_x) - b_copy))
    if residual > 1e-6:
        # lstsq solution is spurious; throw it away
        out_x[:] = np.nan

@cython.boundscheck(False)
cdef void invert_matrix(double *A, int N, int* ipiv) noexcept nogil:
    "A will be overwritten."
    cdef int info = 0
    cdef int i
    cdef double* work = <double*>malloc(N * sizeof(double))
    cdef bint isfinite = True

    for i in range(N**2):
        if isnan(A[i]):
            isfinite = False

    if not isfinite:
        for i in range(N**2):
            A[i] = 0
    else:
        cython_lapack.dgetrf(&N, &N, A, &N, ipiv, &info)
        cython_lapack.dgetri(&N, A, &N, ipiv, work, &N, &info)

    free(work)
    if info != 0:
        for i in range(N**2):
            A[i] = -1e19

@cython.boundscheck(False)
cdef void compute_phase_matrix(double[:,::1] phase_matrix, double[:,::1] hess,
                               double[:, ::1] cons_jac_tmp, double[:, ::1] phase_local_jac_tmp,
                               double[:, ::1] mass_hess_tmp,
                               CompositionSet compset, int num_statevars, double[::1] chemical_potentials,
                               double[::1] phase_dof) noexcept nogil:
    "Compute the LHS of Eq. 41, Sundman 2015."
    cdef int comp_idx, i, j
    cdef int num_components = chemical_potentials.shape[0]
    compset.phase_record.solver_internal_cons_jac(&cons_jac_tmp[0, 0], &phase_dof[0])
    if compset.num_phase_local_conditions > 0:
        compset.phase_record.call_function(compset.phase_local_cons_jac, &phase_local_jac_tmp[0, 0], &phase_dof[0])

    for i in range(compset.phase_record.phase_dof):
        for j in range(compset.phase_record.phase_dof):
            phase_matrix[i, j] = hess[num_statevars+i, num_statevars+j]

    # Special case for ionic liquids where P/Q make the moles non-linear in the
    # site fractions. Eq. 37 of Sundman only does a Taylor expansion of G, but for this
    # model we should also linearize the $\partial M / \partial y$ term by an expansion.
    # This introduces chemical potentials into the phase matrix and we use the ones
    # from the previous iteration.
    for comp_idx in range(num_components):
        compset.phase_record.solver_formulamole_hess(&mass_hess_tmp[0, 0], &phase_dof[0], comp_idx)
        for i in range(compset.phase_record.phase_dof):
            for j in range(compset.phase_record.phase_dof):
                phase_matrix[i, j] -= chemical_potentials[comp_idx] * mass_hess_tmp[num_statevars+i, num_statevars+j]

    for i in range(compset.phase_record.num_internal_cons):
        for j in range(compset.phase_record.phase_dof):
            phase_matrix[compset.phase_record.phase_dof+i, j] = cons_jac_tmp[i, num_statevars+j]
            phase_matrix[j, compset.phase_record.phase_dof+i] = cons_jac_tmp[i, num_statevars+j]

    for i in range(compset.num_phase_local_conditions):
        for j in range(compset.phase_record.phase_dof):
            phase_matrix[compset.phase_record.phase_dof+compset.phase_record.num_internal_cons+i, j] = phase_local_jac_tmp[i, num_statevars+j]
            phase_matrix[j, compset.phase_record.phase_dof+compset.phase_record.num_internal_cons+i] = phase_local_jac_tmp[i, num_statevars+j]


cdef int write_row_stable_phase(double* out_row, int row_stride, double* out_rhs, int[::1] free_chemical_potential_indices,
                                int[::1] free_stable_compset_indices, int[::1] free_statevar_indices,
                                int[::1] fixed_chemical_potential_indices, double[::1] chemical_potentials,
                                double[:, ::1] masses, double[::1] grad, double energy) except -1 nogil:
    # 1a. This phase row: free chemical potentials
    cdef int free_variable_column_offset = 0
    cdef int chempot_idx, statevar_idx, i
    for i in range(free_chemical_potential_indices.shape[0]):
        chempot_idx = free_chemical_potential_indices[i]
        out_row[(free_variable_column_offset + i) * row_stride] = masses[chempot_idx, 0]
    free_variable_column_offset += free_chemical_potential_indices.shape[0]
    # 1a. This phase row: free stable composition sets = zero contribution
    free_variable_column_offset += free_stable_compset_indices.shape[0]
    # 1a. This phase row: free state variables
    for i in range(free_statevar_indices.shape[0]):
        statevar_idx = free_statevar_indices[i]
        out_row[(free_variable_column_offset + i) * row_stride] = -grad[statevar_idx]
    out_rhs[0] = energy
    # 4. Subtract fixed chemical potentials from each phase RHS
    for i in range(fixed_chemical_potential_indices.shape[0]):
        chempot_idx = fixed_chemical_potential_indices[i]
        out_rhs[0] -= masses[chempot_idx, 0] * chemical_potentials[chempot_idx]
    return 0

cdef int write_row_fixed_mole_fraction(double* out_row, int row_stride, double* out_rhs, int component_idx,
                                       int[::1] free_chemical_potential_indices, int[::1] free_stable_compset_indices,
                                       int[::1] free_statevar_indices, int[::1] fixed_chemical_potential_indices,
                                       double[::1] chemical_potentials,
                                       double [::1] system_mole_fractions, double current_system_amount,
                                       double[:, ::1] mass_jac, double[:, ::1] c_component,
                                       double[:, ::1] c_statevars, double[::1] c_G, double[:, ::1] masses,
                                       double moles_normalization, double[::1] moles_normalization_grad,
                                       double[::1] phase_amt, int idx, double prefactor) except -1 nogil:
    if prefactor == 0.0:
        return 0
    cdef int free_variable_column_offset = 0
    cdef int num_statevars = c_statevars.shape[1]
    cdef int chempot_idx, compset_idx, statevar_idx, i, j
    # 2a. This component row: free chemical potentials
    for i in range(free_chemical_potential_indices.shape[0]):
        chempot_idx = free_chemical_potential_indices[i]
        for j in range(c_component.shape[1]):
            out_row[(free_variable_column_offset + i) * row_stride] += prefactor * \
                (phase_amt[idx]/current_system_amount) * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
        for j in range(c_component.shape[1]):
            out_row[(free_variable_column_offset + i) * row_stride] += prefactor * \
                (phase_amt[idx]/current_system_amount) * (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_component[chempot_idx, j]
    free_variable_column_offset += free_chemical_potential_indices.shape[0]
    # 2a. This component row: free stable composition sets
    for i in range(free_stable_compset_indices.shape[0]):
        compset_idx = free_stable_compset_indices[i]
        # Only fill this out if the current idx is equal to a free composition set
        if compset_idx == idx:
            out_row[(free_variable_column_offset + i) * row_stride] += prefactor * \
                (1./current_system_amount)*(masses[component_idx, 0] - system_mole_fractions[component_idx] * moles_normalization)
    free_variable_column_offset += free_stable_compset_indices.shape[0]
    # 2a. This component row: free state variables
    for i in range(free_statevar_indices.shape[0]):
        statevar_idx = free_statevar_indices[i]
        for j in range(c_statevars.shape[0]):
            out_row[(free_variable_column_offset + i) * row_stride] += prefactor * \
                (phase_amt[idx]/current_system_amount) * mass_jac[component_idx, num_statevars+j] * c_statevars[j, statevar_idx]
        for j in range(c_statevars.shape[0]):
            out_row[(free_variable_column_offset + i) * row_stride] += prefactor * \
                (phase_amt[idx]/current_system_amount) * (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_statevars[j, statevar_idx]
    # 3.
    for j in range(c_G.shape[0]):
        out_rhs[0] += -prefactor * (phase_amt[idx]/current_system_amount) * \
            mass_jac[component_idx, num_statevars+j] * c_G[j]
    for j in range(c_G.shape[0]):
        out_rhs[0] += -prefactor * (phase_amt[idx]/current_system_amount) * \
            (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_G[j]
    # 4. Subtract fixed chemical potentials from phase RHS
    for i in range(fixed_chemical_potential_indices.shape[0]):
        chempot_idx = fixed_chemical_potential_indices[i]
        # 5. Subtract fixed chemical potentials from fixed component RHS
        for j in range(c_component.shape[1]):
            out_rhs[0] -= prefactor * (phase_amt[idx]/current_system_amount) * chemical_potentials[
                chempot_idx] * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
        for j in range(c_component.shape[1]):
            out_rhs[0] -= prefactor * (phase_amt[idx]/current_system_amount) * chemical_potentials[
                chempot_idx] * (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_component[chempot_idx, j]
    return 0

cdef int write_row_fixed_mole_amount(double* out_row, int row_stride, double* out_rhs, int component_idx,
                                     int[::1] free_chemical_potential_indices, int[::1] free_stable_compset_indices,
                                     int[::1] free_statevar_indices, int[::1] fixed_chemical_potential_indices,
                                     double[::1] chemical_potentials,
                                     double[:, ::1] mass_jac, double[:, ::1] c_component,
                                     double[:, ::1] c_statevars, double[::1] c_G, double[:, ::1] masses,
                                     double[::1] phase_amt, int idx) except -1 nogil:
    cdef int free_variable_column_offset = 0
    cdef int num_statevars = c_statevars.shape[1]
    cdef int i, j, chempot_idx, compset_idx, statevar_idx
    # 2a. This component row: free chemical potentials
    for i in range(free_chemical_potential_indices.shape[0]):
        chempot_idx = free_chemical_potential_indices[i]
        for j in range(c_component.shape[1]):
            out_row[(free_variable_column_offset + i) * row_stride] += \
                phase_amt[idx] * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
    free_variable_column_offset += free_chemical_potential_indices.shape[0]
    # 2a. This component row: free stable composition sets
    for i in range(free_stable_compset_indices.shape[0]):
        compset_idx = free_stable_compset_indices[i]
        # Only fill this out if the current idx is equal to a free composition set
        if compset_idx == idx:
            out_row[(free_variable_column_offset + i) * row_stride] += masses[component_idx, 0]
    free_variable_column_offset += free_stable_compset_indices.shape[0]
    # 2a. This component row: free state variables
    for i in range(free_statevar_indices.shape[0]):
        statevar_idx = free_statevar_indices[i]
        for j in range(c_statevars.shape[0]):
            out_row[(free_variable_column_offset + i) * row_stride] += \
                phase_amt[idx] * mass_jac[component_idx, num_statevars+j] * c_statevars[j, statevar_idx]
    # 3.
    for j in range(c_G.shape[0]):
        out_rhs[0] += -phase_amt[idx] * mass_jac[component_idx, num_statevars+j] * c_G[j]
    # 4. Subtract fixed chemical potentials from each phase RHS
    for i in range(fixed_chemical_potential_indices.shape[0]):
        chempot_idx = fixed_chemical_potential_indices[i]
        # 6. Subtract fixed chemical potentials from the N=1 row
        for j in range(c_component.shape[1]):
            out_rhs[0] -= phase_amt[idx] * chemical_potentials[
                chempot_idx] * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
    return 0


cdef double dot(double* x, double* y, int n) noexcept nogil:
    "Dot product of two contiguous vectors, computed the way numpy.dot computes it"
    cdef int inc = 1
    cdef double result = 0.
    result += cython_blas.ddot(&n, x, &inc, y, &inc)
    return result


cdef int write_phase_row(double* equilibrium_matrix, int num_rows, double* equilibrium_rhs, int row,
                         SystemSpecification spec, SystemState state, CompsetState csst) except -1 nogil:
    write_row_stable_phase(&equilibrium_matrix[row], num_rows, &equilibrium_rhs[row], spec.free_chemical_potential_indices,
                           state.free_stable_compset_indices, spec.free_statevar_indices, spec.fixed_chemical_potential_indices,
                           state.chemical_potentials, csst.masses, csst.grad, csst.energy)
    return 0


cdef int write_component_rows(double* equilibrium_matrix, int num_rows, double* equilibrium_rhs,
                              SystemSpecification spec, SystemState state, CompsetState csst, int idx) except -1 nogil:
    cdef int component_row_offset, component_idx, system_amount_index, fixed_molefrac_cond_idx
    cdef int num_components = state.chemical_potentials.shape[0]
    cdef int num_stable_phases = state.free_stable_compset_indices.shape[0]
    cdef int num_fixed_phases = spec.fixed_stable_compset_indices.shape[0]
    cdef int num_fixed_mole_fraction_conditions = spec.prescribed_mole_fraction_rhs.shape[0]
    cdef double prefactor
    # 2. Contribute to the row of all fixed mole fraction conditions
    component_row_offset = num_stable_phases + num_fixed_phases
    for fixed_molefrac_cond_idx in range(num_fixed_mole_fraction_conditions):
        for component_idx in range(spec.prescribed_mole_fraction_coefficients.shape[1]):
            prefactor = spec.prescribed_mole_fraction_coefficients[fixed_molefrac_cond_idx, component_idx]
            write_row_fixed_mole_fraction(&equilibrium_matrix[component_row_offset + fixed_molefrac_cond_idx], num_rows,
                                          &equilibrium_rhs[component_row_offset + fixed_molefrac_cond_idx],
                                          component_idx, spec.free_chemical_potential_indices,
                                          state.free_stable_compset_indices,
                                          spec.free_statevar_indices, spec.fixed_chemical_potential_indices,
                                          state.chemical_potentials,
                                          state.mole_fractions, state.system_amount, csst.mass_jac,
                                          csst.c_component, csst.c_statevars,
                                          csst.c_G, csst.masses, csst.moles_normalization,
                                          csst.moles_normalization_grad, state.phase_amt, idx, prefactor)

    system_amount_index = component_row_offset + num_fixed_mole_fraction_conditions
    # 2X. Also handle the N=1 row
    for component_idx in range(num_components):
        write_row_fixed_mole_amount(&equilibrium_matrix[system_amount_index], num_rows,
                                    &equilibrium_rhs[system_amount_index], component_idx,
                                    spec.free_chemical_potential_indices, state.free_stable_compset_indices,
                                    spec.free_statevar_indices, spec.fixed_chemical_potential_indices,
                                    state.chemical_potentials, csst.mass_jac, csst.c_component,
                                    csst.c_statevars, csst.c_G, csst.masses,
                                    state.phase_amt, idx)
    return 0


cdef int fill_equilibrium_system(double* equilibrium_matrix, int num_rows, double* equilibrium_rhs,
                                 SystemSpecification spec, SystemState state) except -1 nogil:
    """
    Fill the equilibrium matrix (Fortran order, num_rows rows) and right-hand side.
    Both must be zeroed by the caller.
    """
    cdef int stable_idx, idx, component_row_offset, fixed_idx
    cdef int system_amount_index, fixed_molefrac_cond_idx
    cdef int num_stable_phases = state.free_stable_compset_indices.shape[0]
    cdef int num_fixed_phases = spec.fixed_stable_compset_indices.shape[0]
    cdef int num_fixed_mole_fraction_conditions = spec.prescribed_mole_fraction_rhs.shape[0]
    cdef double component_residual, system_residual

    for stable_idx in range(state.free_stable_compset_indices.shape[0]):
        idx = state.free_stable_compset_indices[stable_idx]
        write_phase_row(equilibrium_matrix, num_rows, equilibrium_rhs, stable_idx, spec, state,
                        <CompsetState>state._cs_state_ptrs[idx])

    # Handle phases which are fixed to be stable at some amount
    # Example shown in Eq. 60, Sundman et al 2015
    for fixed_idx in range(spec.fixed_stable_compset_indices.shape[0]):
        idx = spec.fixed_stable_compset_indices[fixed_idx]
        write_phase_row(equilibrium_matrix, num_rows, equilibrium_rhs, num_stable_phases + fixed_idx, spec, state,
                        <CompsetState>state._cs_state_ptrs[idx])

    for stable_idx in range(state.free_stable_compset_indices.shape[0]):
        idx = state.free_stable_compset_indices[stable_idx]
        write_component_rows(equilibrium_matrix, num_rows, equilibrium_rhs, spec, state,
                             <CompsetState>state._cs_state_ptrs[idx], idx)

    for fixed_idx in range(spec.fixed_stable_compset_indices.shape[0]):
        idx = spec.fixed_stable_compset_indices[fixed_idx]
        write_component_rows(equilibrium_matrix, num_rows, equilibrium_rhs, spec, state,
                             <CompsetState>state._cs_state_ptrs[idx], idx)

    # Add mass residual to fixed component row RHS, plus N=1 row
    component_row_offset = num_stable_phases + num_fixed_phases
    system_amount_index = component_row_offset + num_fixed_mole_fraction_conditions
    for fixed_molefrac_cond_idx in range(num_fixed_mole_fraction_conditions):
        component_residual = dot(&spec.prescribed_mole_fraction_coefficients[fixed_molefrac_cond_idx, 0], &state.mole_fractions[0],
                                 state.mole_fractions.shape[0]) - spec.prescribed_mole_fraction_rhs[fixed_molefrac_cond_idx]
        equilibrium_rhs[component_row_offset + fixed_molefrac_cond_idx] -= component_residual
    system_residual = state.system_amount - spec.prescribed_system_amount
    equilibrium_rhs[system_amount_index] -= system_residual
    return 0


# Convergence criteria
cdef double ALLOWED_DELTA_Y = 5e-09
cdef double ALLOWED_DELTA_PHASE_AMT = 1e-8
cdef double ALLOWED_DELTA_STATEVAR = 1e-5  # changes defined as percent change
cdef double MASS_RESIDUAL_RTOL = 1e-6
cdef double MASS_RESIDUAL_ATOL = 1e-11
cdef double MIN_SITE_FRACTION = _MIN_SITE_FRACTION


cdef void print_convergence_criteria(SystemSpecification spec, SystemState state):
    # Recomputed with NumPy here so the converged path stays free of debugging work
    mole_fractions = np.asarray(state.mole_fractions)
    determined_indices = np.asarray(spec.determined_component_indices)
    determined_targets = np.asarray(spec.determined_component_mole_fractions)
    determined_residuals = np.abs(mole_fractions[determined_indices] - determined_targets)
    determined_passing = determined_residuals < MASS_RESIDUAL_RTOL * np.abs(determined_targets) + MASS_RESIDUAL_ATOL
    undetermined_indices = np.asarray(spec.undetermined_condition_row_indices)
    undetermined_coefficients = np.asarray(spec.prescribed_mole_fraction_coefficients)[undetermined_indices]
    undetermined_residuals = np.abs(undetermined_coefficients @ mole_fractions - np.asarray(spec.prescribed_mole_fraction_rhs)[undetermined_indices])
    undetermined_scales = np.abs(undetermined_coefficients * mole_fractions).sum(axis=1)
    undetermined_passing = undetermined_residuals < MASS_RESIDUAL_RTOL * undetermined_scales + MASS_RESIDUAL_ATOL
    print(
        f"state.largest_phase_amt_change[0]  ={state.largest_phase_amt_change[0]} (passing={state.largest_phase_amt_change[0] < ALLOWED_DELTA_PHASE_AMT})\n" \
        f"state.largest_y_change[0]          ={state.largest_y_change[0]} (passing={state.largest_y_change[0] < ALLOWED_DELTA_Y})\n" \
        f"state.largest_statevar_change[0]   ={state.largest_statevar_change[0]} (passing={state.largest_statevar_change[0] < ALLOWED_DELTA_STATEVAR})\n" \
        f"determined component residuals     ={determined_residuals} (components={determined_indices}, passing={determined_passing})\n" \
        f"undetermined condition residuals   ={undetermined_residuals} (rows={undetermined_indices}, passing={undetermined_passing})"
        )


cdef bint has_overridden_hooks(SystemSpecification spec, SystemState state):
    "Whether a subclass overrides the solver hooks, which must then be called with the GIL"
    if type(spec) is SystemSpecification and type(state) is SystemState:
        return False
    for name in ('check_convergence', 'pre_solve_hook', 'post_solve_hook'):
        if getattr(type(spec), name) is not getattr(SystemSpecification, name):
            return True
    return getattr(type(state), 'recompute') is not getattr(SystemState, 'recompute')


cdef class SystemSpecification:
    def __init__(self, int num_statevars, int num_components, double prescribed_system_amount,
                   double[::1] initial_chemical_potentials, double[:, ::1] prescribed_mole_fraction_coefficients,
                   double[::1] prescribed_mole_fraction_rhs, int[::1] free_chemical_potential_indices,
                   int[::1] free_statevar_indices, int[::1] fixed_chemical_potential_indices,
                   int[::1] fixed_statevar_indices, int[::1] fixed_stable_compset_indices, bint debugging_output=False):
        self.num_statevars = num_statevars
        self.num_components = num_components
        self.prescribed_system_amount = prescribed_system_amount
        self.initial_chemical_potentials = initial_chemical_potentials
        self.prescribed_mole_fraction_coefficients = prescribed_mole_fraction_coefficients
        self.prescribed_mole_fraction_rhs = prescribed_mole_fraction_rhs
        self.free_chemical_potential_indices = free_chemical_potential_indices
        self.free_statevar_indices = free_statevar_indices
        self.fixed_chemical_potential_indices = fixed_chemical_potential_indices
        self.fixed_statevar_indices = fixed_statevar_indices
        self.fixed_stable_compset_indices = fixed_stable_compset_indices
        self.max_num_free_stable_phases = num_components + len(free_statevar_indices) - len(fixed_stable_compset_indices)
        self.debugging_output = debugging_output

        # The prescribed mole fraction conditions, together with the
        # normalization condition (mole fractions sum to one), may uniquely
        # determine the mole fractions of some components. Convergence of the
        # composition residuals (check_convergence) tests each determined
        # component against its unique value. Prescribed rows involving any
        # undetermined component are not covered by those tests and are
        # checked directly as row residuals.
        num_conditions = self.prescribed_mole_fraction_rhs.shape[0]
        condition_matrix = np.empty((num_conditions + 1, num_components))
        condition_matrix[:num_conditions, :] = self.prescribed_mole_fraction_coefficients
        condition_matrix[num_conditions, :] = 1.0
        condition_rhs = np.empty(num_conditions + 1)
        condition_rhs[:num_conditions] = self.prescribed_mole_fraction_rhs
        condition_rhs[num_conditions] = 1.0
        u, singular_values, vt = np.linalg.svd(condition_matrix)
        rank = int(np.sum(singular_values > max(condition_matrix.shape) * np.finfo(np.float64).eps * singular_values[0]))
        # Minimum-norm least-squares solution of the conditions; unique for determined components
        target_mole_fractions = vt[:rank].T @ ((u[:, :rank].T @ condition_rhs) / singular_values[:rank])
        # A component is determined iff no null space direction of the conditions can change it
        determined_components = ~np.any(np.abs(vt[rank:]) > 1e-10, axis=0)
        self.determined_component_indices = np.array(np.nonzero(determined_components)[0], dtype=np.int32)
        self.determined_component_mole_fractions = target_mole_fractions[np.nonzero(determined_components)[0]]
        undetermined_coefficients = np.array(self.prescribed_mole_fraction_coefficients)[:, ~determined_components]
        self.undetermined_condition_row_indices = np.array(np.nonzero(np.any(np.abs(undetermined_coefficients) > 1e-14, axis=1))[0], dtype=np.int32)

    def __getstate__(self):
        return (self.num_statevars, self.num_components, self.prescribed_system_amount,
                np.array(self.initial_chemical_potentials), np.array(self.prescribed_mole_fraction_coefficients),
                np.array(self.prescribed_mole_fraction_rhs), np.array(self.free_chemical_potential_indices),
                np.array(self.free_statevar_indices), np.array(self.fixed_chemical_potential_indices),
                np.array(self.fixed_statevar_indices), np.array(self.fixed_stable_compset_indices))
    def __setstate__(self, state):
        self.__init__(*state)

    cpdef bint check_convergence(self, SystemState state):
        return self._check_convergence(state)

    cdef int _check_convergence(self, SystemState state) except -1 nogil:
        "Returns 1 if the solution is converged, otherwise 0"
        cdef int i, comp_idx, row_idx
        cdef double target, residual, scale
        cdef bint mass_residuals_converged = True
        cdef bint solution_is_feasible
        # Determined components are checked against their unique prescribed values
        for i in range(self.determined_component_indices.shape[0]):
            comp_idx = self.determined_component_indices[i]
            target = self.determined_component_mole_fractions[i]
            mass_residuals_converged = mass_residuals_converged and (
                abs(state.mole_fractions[comp_idx] - target) < MASS_RESIDUAL_RTOL * abs(target) + MASS_RESIDUAL_ATOL)
        # Prescribed rows involving undetermined components are checked as row
        # residuals, relative to the magnitudes of the terms in the linear combination
        for i in range(self.undetermined_condition_row_indices.shape[0]):
            row_idx = self.undetermined_condition_row_indices[i]
            residual = -self.prescribed_mole_fraction_rhs[row_idx]
            scale = 0.0
            for comp_idx in range(self.num_components):
                residual += self.prescribed_mole_fraction_coefficients[row_idx, comp_idx] * state.mole_fractions[comp_idx]
                scale += abs(self.prescribed_mole_fraction_coefficients[row_idx, comp_idx] * state.mole_fractions[comp_idx])
            mass_residuals_converged = mass_residuals_converged and (
                abs(residual) < MASS_RESIDUAL_RTOL * scale + MASS_RESIDUAL_ATOL)
        if self.debugging_output:
            with gil:
                print_convergence_criteria(self, state)

        solution_is_feasible = (
            (state.largest_phase_amt_change[0] < ALLOWED_DELTA_PHASE_AMT) and
            (state.largest_y_change[0] < ALLOWED_DELTA_Y) and
            (state.largest_statevar_change[0] < ALLOWED_DELTA_STATEVAR) and
            mass_residuals_converged
        )
        if solution_is_feasible and (state.iterations_since_last_phase_change >= 10):
            return True
        else:
            return False

    cpdef bint pre_solve_hook(self, SystemState state):
        return True

    cpdef bint post_solve_hook(self, SystemState state):
        return True

    cpdef bint run_loop(self, SystemState state, int max_iterations):
        cdef bint use_overrides = has_overridden_hooks(self, state)
        cdef int converged
        if state._prepare(self):
            with nogil:
                converged = self._run_loop(state, max_iterations, use_overrides)
        else:
            # Some phase functions must not be called from several threads at once
            converged = self._run_loop(state, max_iterations, use_overrides)
        return converged

    cdef int _run_loop(self, SystemState state, int max_iterations, bint use_overrides) except -1 nogil:
        """
        Iterate the solver from state, without holding the GIL except to call hooks
        overridden in Python (use_overrides) and to print debugging output.
        Returns 1 if converged, otherwise 0.
        """
        cdef double step_size = 1.0
        cdef double target_step_size = 1.0
        cdef double step_size_initial
        cdef bint converged = False
        cdef bint phases_changed = False
        cdef bint proceed = True
        cdef size_t iteration
        for iteration in range(max_iterations):
            if self.debugging_output:
                with gil:
                    print(f"---------------")
                    print(f"iteration {iteration+1}")
            state.iteration = iteration
            if use_overrides:
                with gil:
                    proceed = self.pre_solve_hook(state)
                    state._prepare(self)
            if not proceed:
                break
            solve_state_nogil(self, state, use_overrides)
            if use_overrides:
                with gil:
                    proceed = self.post_solve_hook(state)
                    state._prepare(self)
            if not proceed:
                break
            if state.iterations_since_last_phase_change >= 5:
                phases_changed = remove_and_consolidate_phases(self, state)
            else:
                phases_changed = False
            if use_overrides:
                with gil:
                    converged = self.check_convergence(state)
            else:
                converged = self._check_convergence(state)
            if converged:
                phases_changed = phases_changed or change_phases(self, state)
                if not phases_changed:
                    break
            state.increment_phase_metastability_counters()
            if phases_changed:
                state.iterations_since_last_phase_change = 0
            else:
                state.iterations_since_last_phase_change += 1
                # slowly ramp up step size at the beginning of solves to
                # prevent large changes from the starting point
                step_size_initial = target_step_size * (state.iteration + 1) / (20)
                step_size = min(target_step_size, step_size_initial)
                advance_state_nogil(self, state, &state._equilibrium_soln[0], step_size)
        if state.free_stable_compset_indices.shape[0] > self.max_num_free_stable_phases:
            # Gibbs phase rule violation in solution
            converged = False
        return converged

    cpdef SystemState get_new_state(self, list compsets):
        return SystemState(self, compsets)

cdef class CompsetState:
    cdef double[::1] x
    cdef double energy
    cdef double[::1] grad
    cdef double[:,::1] hess
    cdef double[:,::1] masses
    cdef double[:,::1] mass_jac
    cdef double[:,::1] phase_matrix
    cdef double[:,::1] full_e_matrix
    cdef double[::1] c_G
    cdef double[:, ::1] c_statevars
    cdef double[:, ::1] c_component
    cdef double[::1] delta_y
    cdef double moles_normalization
    cdef double[::1] internal_cons
    cdef double[::1] moles_normalization_grad
    cdef int[::1] fixed_phase_dof_indices
    cdef int[::1] ipiv
    cdef double[:, ::1] cons_jac_tmp
    cdef double[:, ::1] phase_local_jac_tmp
    cdef double[:, ::1] mass_hess_tmp
    cdef double[::1] new_y
    # Bound by SystemState._prepare
    cdef CompositionSet compset
    cdef double[::1] dof

    def __init__(self, SystemSpecification spec, CompositionSet compset):
        self.x = np.zeros(spec.num_statevars + compset.phase_record.phase_dof)
        self.energy = 0
        self.grad = np.zeros(spec.num_statevars + compset.phase_record.phase_dof)
        self.hess = np.zeros((spec.num_statevars + compset.phase_record.phase_dof,
                             spec.num_statevars + compset.phase_record.phase_dof))
        self.masses = np.zeros((spec.num_components, 1))
        self.mass_jac = np.zeros((spec.num_components,
                                  spec.num_statevars + compset.phase_record.phase_dof))
        self.phase_matrix = np.zeros((compset.phase_record.phase_dof + compset.phase_record.num_internal_cons + compset.num_phase_local_conditions,
                                      compset.phase_record.phase_dof + compset.phase_record.num_internal_cons + compset.num_phase_local_conditions))
        self.full_e_matrix = np.zeros((compset.phase_record.phase_dof + compset.phase_record.num_internal_cons + compset.num_phase_local_conditions,
                                       compset.phase_record.phase_dof + compset.phase_record.num_internal_cons + compset.num_phase_local_conditions))
        self.c_G = np.zeros(compset.phase_record.phase_dof)
        self.c_statevars = np.zeros((compset.phase_record.phase_dof, spec.num_statevars))
        self.c_component = np.zeros((spec.num_components, compset.phase_record.phase_dof))
        self.moles_normalization = 0.0
        self.internal_cons = np.zeros(compset.phase_record.num_internal_cons)
        self.moles_normalization_grad = np.zeros(spec.num_statevars+compset.phase_record.phase_dof)
        self.fixed_phase_dof_indices = np.array([], dtype=np.int32)
        self.ipiv = np.empty(self.phase_matrix.shape[0], dtype=np.int32)
        self.delta_y = np.zeros(compset.phase_record.phase_dof)
        self.cons_jac_tmp = np.zeros((compset.phase_record.num_internal_cons, spec.num_statevars + compset.phase_record.phase_dof))
        self.phase_local_jac_tmp = np.zeros((compset.num_phase_local_conditions, spec.num_statevars + compset.phase_record.phase_dof))
        self.mass_hess_tmp = np.zeros((spec.num_statevars + compset.phase_record.phase_dof,
                                       spec.num_statevars + compset.phase_record.phase_dof))
        self.new_y = np.zeros(spec.num_statevars + compset.phase_record.phase_dof)

    def __getstate__(self):
        return (np.array(self.x), self.energy, np.array(self.grad), np.array(self.hess),
                np.array(self.phase_matrix), np.array(self.full_e_matrix),
                np.array(self.masses), np.array(self.mass_jac), np.array(self.c_G), np.array(self.c_statevars),
                np.array(self.c_component), self.moles_normalization, np.array(self.internal_cons), np.array(self.moles_normalization_grad),
                np.array(self.fixed_phase_dof_indices, dtype=np.int32), np.array(self.ipiv, dtype=np.int32))
    def __setstate__(self, state):
        (self.x, self.energy, self.grad, self.hess, self.phase_matrix, self.full_e_matrix,
         self.masses, self.mass_jac, self.c_G, self.c_statevars,
         self.c_component, self.moles_normalization, self.internal_cons, self.moles_normalization_grad, self.fixed_phase_dof_indices,
         self.ipiv) = state
        self.new_y = np.zeros(self.x.shape[0])


cdef class SystemState:
    def __init__(self, SystemSpecification spec, list compsets):
        cdef CompositionSet compset
        cdef int idx, comp_idx
        self.compsets = compsets
        cdef double phase_comp_sum
        for compset in compsets:
            compset.fixed = False
        for idx in spec.fixed_stable_compset_indices:
            compset = compsets[idx]
            compset.fixed = True
        self.cs_states = [CompsetState(spec, compset) for compset in compsets]
        self.dof = [np.array(compset.dof) for compset in compsets]
        self.iteration = 0
        self.iterations_since_last_phase_change = 0
        self.metastable_phase_iterations = np.zeros(len(compsets), dtype=np.int32)
        self.times_compset_removed = np.zeros(len(compsets), dtype=np.int32)
        # Phase fractions need to be converted to moles of formula
        self.phase_amt = np.array([compset.NP for compset in compsets])
        self.chemical_potentials = np.zeros(spec.num_components)
        self.previous_chemical_potentials = np.zeros(spec.num_components)
        self.largest_chemical_potential_difference = -np.inf
        self.delta_ms = np.zeros((len(compsets), spec.num_components))
        self.delta_statevars = np.zeros(spec.num_statevars)
        self.phase_compositions = np.zeros((len(compsets), spec.num_components))
        self.free_stable_compset_indices = np.array(np.nonzero([((compset.fixed==False) and (compset.NP>0))
                                                                for compset in compsets])[0], dtype=np.int32)
        self.largest_statevar_change[0] = 0
        self.largest_phase_amt_change[0] = 0
        self.largest_y_change[0] = 0
        self.system_amount = 0
        self.mole_fractions = np.zeros(spec.num_components)
        self._driving_forces = np.zeros(len(compsets))
        self._phase_energies_per_mole_atoms = np.zeros((len(compsets), 1))
        self._phase_amounts_per_mole_atoms = np.zeros((len(compsets), spec.num_components, 1))

        cdef double[:, ::1] masses_tmp = np.zeros((spec.num_components, 1))
        for idx in range(self.phase_amt.shape[0]):
            compset = self.compsets[idx]
            x = self.dof[idx]
            phase_comp_sum = 0.0
            for comp_idx in range(spec.num_components):
                compset.phase_record.formulamole_obj(masses_tmp[comp_idx, :], x, comp_idx)
                self.phase_compositions[idx, comp_idx] = masses_tmp[comp_idx, 0]
                phase_comp_sum += self.phase_compositions[idx, comp_idx]
                masses_tmp[:,:] = 0
            # Convert phase fractions to formula units
            self.phase_amt[idx] /= phase_comp_sum

    def __getstate__(self):
        return (self.compsets, self.cs_states, self.dof, self.iteration, self.iterations_since_last_phase_change,
                self.metastable_phase_iterations, self.times_compset_removed,
                np.array(self.phase_amt), np.array(self.chemical_potentials), np.array(self.previous_chemical_potentials),
                np.array(self.delta_ms), np.array(self.phase_compositions), self.largest_chemical_potential_difference,
                self.largest_statevar_change[0], self.largest_phase_amt_change[0], self.largest_y_change[0],
                np.array(self.free_stable_compset_indices), self.system_amount, np.array(self.mole_fractions))
    def __setstate__(self, state):
        (self.compsets, self.cs_states, self.dof, self.iteration, self.iterations_since_last_phase_change,
         self.metastable_phase_iterations, self.times_compset_removed,
         self.phase_amt, self.chemical_potentials, self.previous_chemical_potentials,
         self.delta_ms, self.phase_compositions, self.largest_chemical_potential_difference, self.largest_statevar_change[0],
         self.largest_phase_amt_change[0], self.largest_y_change[0], self.free_stable_compset_indices, self.system_amount, self.mole_fractions) = state

    cdef bint _prepare(self, SystemSpecification spec) except -1:
        """
        Bind the composition sets and their states for the solver functions that run without
        the GIL, and size the work buffers for spec. Must be called again if compsets, cs_states
        or dof change.

        Returns True if the functions of all phases can be called from several threads at once.
        """
        cdef CompsetState csst
        cdef CompositionSet compset
        cdef int idx, num_free_stable, max_num_free_variables
        cdef bint reentrant = True
        cdef dict phase_ids = {}
        if not (len(self.compsets) == len(self.cs_states) == len(self.dof)):
            raise ValueError('SystemState has inconsistent numbers of composition sets')
        self.num_compsets = len(self.compsets)
        cs_state_refs = np.empty(self.num_compsets, dtype=object)
        self._phase_ids = np.empty(self.num_compsets, dtype=np.int32)
        for idx in range(self.num_compsets):
            csst = self.cs_states[idx]
            compset = self.compsets[idx]
            csst.compset = compset
            csst.dof = self.dof[idx]
            reentrant = compset.phase_record.prepare_solver_functions() and reentrant
            if compset.num_phase_local_conditions > 0:
                reentrant = reentrant and compset.phase_local_cons_jac.reentrant
            self._phase_ids[idx] = phase_ids.setdefault(compset.phase_record.phase_name, len(phase_ids))
            cs_state_refs[idx] = csst
        self._cs_state_refs = cs_state_refs
        self._cs_state_ptrs = <void**> (<np.ndarray>cs_state_refs).data

        if (not self._buffers_ready) or self._free_stable_buffer.shape[0] != self.num_compsets:
            self._free_stable_buffer = np.empty(self.num_compsets, dtype=np.int32)
            self._new_free_stable_buffer = np.empty(self.num_compsets, dtype=np.int32)
            self._remove_flags = np.empty(self.num_compsets, dtype=np.int32)
            self._add_flags = np.empty(self.num_compsets, dtype=np.int32)
        # The solver changes the stable composition sets in place, within this buffer
        num_free_stable = self.free_stable_compset_indices.shape[0]
        if num_free_stable > self.num_compsets:
            raise ValueError('SystemState has more stable composition sets than composition sets')
        for idx in range(num_free_stable):
            self._free_stable_buffer[idx] = self.free_stable_compset_indices[idx]
        self.free_stable_compset_indices = self._free_stable_buffer[:num_free_stable]

        max_num_free_variables = spec.free_chemical_potential_indices.shape[0] + self.num_compsets + \
                                 spec.free_statevar_indices.shape[0]
        if (not self._buffers_ready) or self._equilibrium_soln.shape[0] < max_num_free_variables:
            self._equilibrium_matrix = np.empty(max_num_free_variables * max_num_free_variables)
            self._equilibrium_soln = np.empty(max_num_free_variables)
        self._buffers_ready = True
        return reentrant

    cpdef void recompute(self, SystemSpecification spec):
        if self._prepare(spec):
            with nogil:
                self._recompute(spec)
        else:
            self._recompute(spec)

    @cython.boundscheck(False)
    cdef int _recompute(self, SystemSpecification spec) except -1 nogil:
        cdef int idx, comp_idx
        for comp_idx in range(self.mole_fractions.shape[0]):
            self.mole_fractions[comp_idx] = 0
        for idx in range(self.delta_ms.shape[0]):
            for comp_idx in range(self.delta_ms.shape[1]):
                self.delta_ms[idx, comp_idx] = 0
        self.system_amount = 0
        # Compute normalized global quantities
        for idx in range(self.num_compsets):
            add_compset_amounts(self, spec, <CompsetState>self._cs_state_ptrs[idx], idx)
        for comp_idx in range(self.mole_fractions.shape[0]):
            self.mole_fractions[comp_idx] /= self.system_amount

        for idx in range(self.num_compsets):
            recompute_compset(self, spec, <CompsetState>self._cs_state_ptrs[idx], idx)
        return 0

    cdef int compute_driving_forces(self) except -1 nogil:
        "Compute _driving_forces"
        cdef int idx
        # This needs to be done per mole of atoms, not per formula unit, since we compare phases to each other
        for idx in range(self.num_compsets):
            self._driving_forces[idx] = 0
            compute_driving_force(self, <CompsetState>self._cs_state_ptrs[idx], idx)
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void increment_phase_metastability_counters(self) noexcept nogil:
        cdef int idx, i
        cdef bint is_stable
        for idx in range(self.num_compsets):
            is_stable = False
            for i in range(self.free_stable_compset_indices.shape[0]):
                if self.free_stable_compset_indices[i] == idx:
                    is_stable = True
                    break
            if is_stable or (<CompsetState>self._cs_state_ptrs[idx]).compset.fixed:
                self.metastable_phase_iterations[idx] = 0
            else:
                self.metastable_phase_iterations[idx] += 1


@cython.boundscheck(False)
cdef void add_compset_amounts(SystemState state, SystemSpecification spec, CompsetState csst, int idx) noexcept nogil:
    cdef int comp_idx, i
    for comp_idx in range(csst.masses.shape[0]):
        for i in range(csst.masses.shape[1]):
            csst.masses[comp_idx, i] = 0
    for comp_idx in range(spec.num_components):
        csst.compset.phase_record.solver_formulamole_obj(&csst.masses[comp_idx, 0], &csst.dof[0], comp_idx)
        if state.phase_amt[idx] > 0:
            state.mole_fractions[comp_idx] += state.phase_amt[idx] * csst.masses[comp_idx, 0]
            state.system_amount += state.phase_amt[idx] * csst.masses[comp_idx, 0]
        state.phase_compositions[idx, comp_idx] = csst.masses[comp_idx, 0]


@cython.boundscheck(False)
cdef void recompute_compset(SystemState state, SystemSpecification spec, CompsetState csst, int idx) noexcept nogil:
    cdef int num_components = spec.num_components
    cdef int num_statevars = spec.num_statevars
    cdef int comp_idx, statevar_idx, i, j, num_phase_dof
    cdef double mu_c_sum
    cdef double phase_comp_sum
    cdef double* x = &csst.dof[0]
    # Calculate key phase quantities starting here
    phase_comp_sum = 0.0
    for comp_idx in range(num_components):
        phase_comp_sum += state.phase_compositions[idx, comp_idx]
    csst.compset.solver_update(&x[num_statevars], csst.dof.shape[0] - num_statevars, state.phase_amt[idx] * phase_comp_sum,
                               x, num_statevars)
    csst.energy = 0
    for comp_idx in range(csst.mass_jac.shape[0]):
        for i in range(csst.mass_jac.shape[1]):
            csst.mass_jac[comp_idx, i] = 0
    # Compute phase matrix (LHS of Eq. 41, Sundman 2015)
    for i in range(csst.phase_matrix.shape[0]):
        for j in range(csst.phase_matrix.shape[1]):
            csst.phase_matrix[i, j] = 0
    for i in range(csst.internal_cons.shape[0]):
        csst.internal_cons[i] = 0
    for i in range(csst.hess.shape[0]):
        for j in range(csst.hess.shape[1]):
            csst.hess[i, j] = 0
    for i in range(csst.grad.shape[0]):
        csst.grad[i] = 0

    csst.compset.phase_record.solver_formulaobj(&csst.energy, x)
    for comp_idx in range(num_components):
        csst.compset.phase_record.solver_formulamole_grad(&csst.mass_jac[comp_idx, 0], x, comp_idx)
    csst.compset.phase_record.solver_formulahess(&csst.hess[0, 0], x)
    csst.compset.phase_record.solver_formulagrad(&csst.grad[0], x)
    csst.compset.phase_record.solver_internal_cons_func(&csst.internal_cons[0], x)

    compute_phase_matrix(csst.phase_matrix, csst.hess, csst.cons_jac_tmp, csst.phase_local_jac_tmp, csst.mass_hess_tmp, csst.compset, num_statevars, state.chemical_potentials, csst.dof)
    # Copy the phase matrix into the e matrix and invert the e matrix
    for i in range(csst.full_e_matrix.shape[0]):
        for j in range(csst.full_e_matrix.shape[1]):
            csst.full_e_matrix[i,j] = csst.phase_matrix[i,j]
    invert_matrix(&csst.full_e_matrix[0,0], csst.full_e_matrix.shape[0], &csst.ipiv[0])

    num_phase_dof = csst.compset.phase_record.phase_dof
    for i in range(csst.c_G.shape[0]):
        csst.c_G[i] = 0
    for i in range(csst.c_statevars.shape[0]):
        for j in range(csst.c_statevars.shape[1]):
            csst.c_statevars[i, j] = 0
    for i in range(csst.c_component.shape[0]):
        for j in range(csst.c_component.shape[1]):
            csst.c_component[i, j] = 0
    csst.moles_normalization = 0
    for i in range(csst.moles_normalization_grad.shape[0]):
        csst.moles_normalization_grad[i] = 0
    for i in range(num_phase_dof):
        for j in range(num_phase_dof):
            csst.c_G[i] -= csst.full_e_matrix[i, j] * csst.grad[num_statevars+j]
    for i in range(num_phase_dof):
        for j in range(num_phase_dof):
            for statevar_idx in range(num_statevars):
                csst.c_statevars[i, statevar_idx] -= csst.full_e_matrix[i, j] * csst.hess[num_statevars + j, statevar_idx]
    for comp_idx in range(num_components):
        for i in range(num_phase_dof):
            for j in range(num_phase_dof):
                csst.c_component[comp_idx, i] += csst.mass_jac[comp_idx, num_statevars + j] * csst.full_e_matrix[i, j]
    for comp_idx in range(num_components):
        for i in range(num_phase_dof):
            mu_c_sum = 0
            for j in range(state.chemical_potentials.shape[0]):
                mu_c_sum += csst.c_component[j, i] * state.chemical_potentials[j]
            state.delta_ms[idx, comp_idx] += csst.mass_jac[comp_idx, num_statevars + i] * (mu_c_sum + csst.c_G[i])
    for comp_idx in range(num_components):
        csst.moles_normalization += csst.masses[comp_idx, 0]
        for i in range(num_phase_dof+num_statevars):
            csst.moles_normalization_grad[i] += csst.mass_jac[comp_idx, i]


cdef int compute_driving_force(SystemState state, CompsetState csst, int idx) except -1 nogil:
    cdef int comp_idx
    for comp_idx in range(state.chemical_potentials.shape[0]):
        csst.compset.phase_record.solver_mass_obj(&state._phase_amounts_per_mole_atoms[idx, comp_idx, 0], &csst.dof[0], comp_idx)
        state._driving_forces[idx] += state.chemical_potentials[comp_idx] * state._phase_amounts_per_mole_atoms[idx, comp_idx, 0]
    csst.compset.phase_record.solver_obj(&state._phase_energies_per_mole_atoms[idx, 0], &csst.dof[0])
    state._driving_forces[idx] -= state._phase_energies_per_mole_atoms[idx, 0]
    return 0


cpdef construct_equilibrium_system(SystemSpecification spec, SystemState state, int num_reserved_rows) except *:
    cdef double[::1,:] equilibrium_matrix  # Fortran ordering required by call into lapack
    cdef double[::1] equilibrium_rhs
    cdef int num_stable_phases, num_fixed_phases, num_fixed_mole_fraction_conditions, num_free_variables

    num_stable_phases = state.free_stable_compset_indices.shape[0]
    num_fixed_phases = spec.fixed_stable_compset_indices.shape[0]
    num_fixed_mole_fraction_conditions = spec.prescribed_mole_fraction_rhs.shape[0]
    num_free_variables = spec.free_chemical_potential_indices.shape[0] + num_stable_phases + \
                         spec.free_statevar_indices.shape[0]

    equilibrium_matrix = np.zeros((num_stable_phases + num_fixed_phases + num_fixed_mole_fraction_conditions + num_reserved_rows + 1,
                                   num_free_variables), order='F')
    equilibrium_rhs = np.zeros(equilibrium_matrix.shape[0])
    if (equilibrium_matrix.shape[0] != equilibrium_matrix.shape[1]):
        raise ValueError('Conditions do not obey Gibbs Phase Rule')
    state._prepare(spec)
    fill_equilibrium_system(&equilibrium_matrix[0, 0], equilibrium_matrix.shape[0], &equilibrium_rhs[0], spec, state)
    return np.asarray(equilibrium_matrix), np.asarray(equilibrium_rhs)

cpdef state_variable_differential(SystemSpecification spec, SystemState state, int target_statevar_index):
    # Sundman et al 2015, Eq. 74
    cdef double[::1,:] equilibrium_matrix  # Fortran ordering required by call into lapack
    cdef double[::1] equilibrium_soln, delta_chemical_potentials, delta_statevars, delta_phase_amounts
    cdef int[::1] orig_fixed_statevar_indices, orig_free_statevar_indices
    cdef int chempot_idx, statevar_idx, cs_idx, i

    orig_fixed_statevar_indices = np.array(spec.fixed_statevar_indices)
    orig_free_statevar_indices = np.array(spec.free_statevar_indices)
    delta_chemical_potentials = np.zeros(spec.num_components)
    delta_statevars = np.zeros(spec.num_statevars)
    delta_phase_amounts = np.zeros(len(state.compsets))
    spec.fixed_statevar_indices = np.setdiff1d(spec.fixed_statevar_indices, np.array(target_statevar_index))
    spec.free_statevar_indices = np.append(spec.free_statevar_indices, target_statevar_index).astype(np.int32)

    try:
        equilibrium_matrix, equilibrium_soln = construct_equilibrium_system(spec, state, 1)
        equilibrium_soln[:] = 0
        # target_statevar_index is the last column of the matrix, by construction
        equilibrium_matrix[-1, -1] = 1
        equilibrium_soln[-1] = 1
        lstsq_check_infeasible(equilibrium_matrix, equilibrium_soln, equilibrium_soln)
        for i in range(spec.free_chemical_potential_indices.shape[0]):
            chempot_idx = spec.free_chemical_potential_indices[i]
            delta_chemical_potentials[chempot_idx] = equilibrium_soln[i]
        for i in range(state.free_stable_compset_indices.shape[0]):
            cs_idx = state.free_stable_compset_indices[i]
            delta_phase_amounts[cs_idx] = equilibrium_soln[spec.free_chemical_potential_indices.shape[0] + i]
        for i in range(spec.free_statevar_indices.shape[0]):
            statevar_idx = spec.free_statevar_indices[i]
            delta_statevars[statevar_idx] = equilibrium_soln[spec.free_chemical_potential_indices.shape[0] +
                                                             state.free_stable_compset_indices.shape[0] + i]
        return np.asarray(delta_chemical_potentials), np.asarray(delta_statevars), np.asarray(delta_phase_amounts)
    finally:
        spec.fixed_statevar_indices = orig_fixed_statevar_indices
        spec.free_statevar_indices = orig_free_statevar_indices

cpdef fixed_component_differential(SystemSpecification spec, SystemState state, int target_component_index):
    # Based on Sundman et al 2015, Eq. 74, with some modifications
    cdef double[::1,:] equilibrium_matrix  # Fortran ordering required by call into lapack
    cdef double[::1] equilibrium_soln, delta_chemical_potentials, delta_statevars, delta_phase_amounts
    cdef np.ndarray comparison_array = np.zeros(spec.prescribed_mole_fraction_coefficients.shape[1])
    comparison_array[target_component_index] = 1
    cdef int num_stable_phases = state.free_stable_compset_indices.shape[0]
    cdef int num_fixed_phases = spec.fixed_stable_compset_indices.shape[0]
    cdef int num_fixed_mole_fraction_conditions = spec.prescribed_mole_fraction_rhs.shape[0]
    cdef int chempot_idx, statevar_idx, cs_idx, i
    cdef bint component_was_fixed = False

    for i in range(spec.prescribed_mole_fraction_coefficients.shape[0]):
        if np.all(np.asarray(spec.prescribed_mole_fraction_coefficients[i]) == comparison_array):
            component_was_fixed = True
    if not component_was_fixed:
        raise ValueError('Target component was not fixed in the present calculation')

    delta_chemical_potentials = np.zeros(spec.num_components)
    delta_statevars = np.zeros(spec.num_statevars)
    delta_phase_amounts = np.zeros(len(state.compsets))

    equilibrium_matrix, equilibrium_soln = construct_equilibrium_system(spec, state, 0)
    equilibrium_soln[:] = 0

    # delta mole fractions must sum to zero; we have degrees of freedom to decide how to distribute
    # for now, specify one dependent component
    for i in range(spec.prescribed_mole_fraction_coefficients.shape[0]):
        if np.all(np.asarray(spec.prescribed_mole_fraction_coefficients[i]) == comparison_array):
            equilibrium_soln[num_stable_phases + num_fixed_phases + i] = 1
        else:
            equilibrium_soln[num_stable_phases + num_fixed_phases + i] = 0
    lstsq_check_infeasible(equilibrium_matrix, equilibrium_soln, equilibrium_soln)
    for i in range(spec.free_chemical_potential_indices.shape[0]):
        chempot_idx = spec.free_chemical_potential_indices[i]
        delta_chemical_potentials[chempot_idx] = equilibrium_soln[i]
    for i in range(state.free_stable_compset_indices.shape[0]):
        cs_idx = state.free_stable_compset_indices[i]
        delta_phase_amounts[cs_idx] = equilibrium_soln[spec.free_chemical_potential_indices.shape[0] + i]
    for i in range(spec.free_statevar_indices.shape[0]):
        statevar_idx = spec.free_statevar_indices[i]
        delta_statevars[statevar_idx] = equilibrium_soln[spec.free_chemical_potential_indices.shape[0] +
                                                            state.free_stable_compset_indices.shape[0] + i]
    return np.asarray(delta_chemical_potentials), np.asarray(delta_statevars), np.asarray(delta_phase_amounts)

cpdef chemical_potential_differential(SystemSpecification spec, SystemState state, int target_component_index):
    # Sundman et al 2015, Eq. 74
    cdef double[::1,:] equilibrium_matrix  # Fortran ordering required by call into lapack
    cdef double[::1] equilibrium_soln, delta_chemical_potentials, delta_statevars, delta_phase_amounts
    cdef int[::1] orig_fixed_statevar_indices, orig_free_statevar_indices
    cdef int chempot_idx, statevar_idx, cs_idx, i
    cdef bint component_was_fixed = False

    for i in range(spec.fixed_chemical_potential_indices.shape[0]):
        if spec.fixed_chemical_potential_indices[i] == target_component_index:
            component_was_fixed = True
    if not component_was_fixed:
        raise ValueError('Target chemical potential was not fixed in the present calculation')

    # Release chemical potential condition
    orig_fixed_chemical_potential_indices = np.array(spec.fixed_chemical_potential_indices)
    orig_free_chemical_potential_indices = np.array(spec.free_chemical_potential_indices)
    delta_chemical_potentials = np.zeros(spec.num_components)
    delta_statevars = np.zeros(spec.num_statevars)
    delta_phase_amounts = np.zeros(len(state.compsets))
    spec.fixed_chemical_potential_indices = np.setdiff1d(spec.fixed_chemical_potential_indices, np.array(target_component_index))
    spec.free_chemical_potential_indices = np.append(spec.free_chemical_potential_indices, target_component_index).astype(np.int32)

    try:
        equilibrium_matrix, equilibrium_soln = construct_equilibrium_system(spec, state, 1)
        equilibrium_soln[:] = 0
        equilibrium_matrix[-1, target_component_index] = 1
        equilibrium_soln[-1] = 1
        lstsq_check_infeasible(equilibrium_matrix, equilibrium_soln, equilibrium_soln)
        for i in range(spec.free_chemical_potential_indices.shape[0]):
            chempot_idx = spec.free_chemical_potential_indices[i]
            delta_chemical_potentials[chempot_idx] = equilibrium_soln[i]
        for i in range(state.free_stable_compset_indices.shape[0]):
            cs_idx = state.free_stable_compset_indices[i]
            delta_phase_amounts[cs_idx] = equilibrium_soln[spec.free_chemical_potential_indices.shape[0] + i]
        for i in range(spec.free_statevar_indices.shape[0]):
            statevar_idx = spec.free_statevar_indices[i]
            delta_statevars[statevar_idx] = equilibrium_soln[spec.free_chemical_potential_indices.shape[0] +
                                                             state.free_stable_compset_indices.shape[0] + i]
        return np.asarray(delta_chemical_potentials), np.asarray(delta_statevars), np.asarray(delta_phase_amounts)
    finally:
        spec.fixed_chemical_potential_indices = orig_fixed_chemical_potential_indices
        spec.free_chemical_potential_indices = orig_free_chemical_potential_indices

cpdef site_fraction_differential(CompsetState csst, double[::1] delta_chempots, double[::1] delta_statevars):
    # Sundman et al 2015, Eq. 78
    cdef double[::1] delta_y = np.zeros(csst.delta_y.shape[0])
    cdef int chempot_idx, statevar_idex

    for i in range(delta_y.shape[0]):
        for statevar_idx in range(delta_statevars.shape[0]):
            delta_y[i] += csst.c_statevars[i, statevar_idx] * delta_statevars[statevar_idx]
        for chempot_idx in range(delta_chempots.shape[0]):
            delta_y[i] += csst.c_component[chempot_idx, i] * delta_chempots[chempot_idx]
    return np.asarray(delta_y)

cpdef solve_state(SystemSpecification spec, SystemState state):
    cdef double[::1] equilibrium_soln
    cdef bint use_overrides = has_overridden_hooks(spec, state)
    if state._prepare(spec):
        with nogil:
            solve_state_nogil(spec, state, use_overrides)
    else:
        solve_state_nogil(spec, state, use_overrides)
    equilibrium_soln = np.array(state._equilibrium_soln[:count_free_variables(spec, state)])
    return equilibrium_soln


cdef int count_free_variables(SystemSpecification spec, SystemState state) noexcept nogil:
    return spec.free_chemical_potential_indices.shape[0] + state.free_stable_compset_indices.shape[0] + \
           spec.free_statevar_indices.shape[0]


cdef void print_phase_matrices(SystemState state):
    cdef CompsetState csst
    for i in range(state.free_stable_compset_indices.shape[0]):
        cs_idx = state.free_stable_compset_indices[i]
        csst = state.cs_states[cs_idx]
        print(state.compsets[cs_idx])
        print(f"phase matrix:\n{np.asarray(csst.phase_matrix)}")
        print(f"inv phase matrix:\n{np.asarray(csst.full_e_matrix)}")


cdef int solve_state_nogil(SystemSpecification spec, SystemState state, bint use_overrides) except -1 nogil:
    """
    Solve the equilibrium system at the current state for the step direction, into state._equilibrium_soln.
    Requires state._prepare(spec).
    """
    cdef int chempot_idx, comp_idx, i
    cdef int num_rows = state.free_stable_compset_indices.shape[0] + spec.fixed_stable_compset_indices.shape[0] + \
                        spec.prescribed_mole_fraction_rhs.shape[0] + 1
    cdef int num_columns = count_free_variables(spec, state)
    cdef double* equilibrium_matrix = &state._equilibrium_matrix[0]  # Fortran ordering required by call into lapack
    cdef double* equilibrium_soln = &state._equilibrium_soln[0]

    for comp_idx in range(state.chemical_potentials.shape[0]):
        state.previous_chemical_potentials[comp_idx] = state.chemical_potentials[comp_idx]
    if use_overrides:
        with gil:
            state.recompute(spec)
            state._prepare(spec)
    else:
        state._recompute(spec)

    if spec.debugging_output:
        with gil:
            print_phase_matrices(state)

    if num_rows != num_columns:
        with gil:
            raise ValueError('Conditions do not obey Gibbs Phase Rule')
    for i in range(num_rows * num_columns):
        equilibrium_matrix[i] = 0
    for i in range(num_rows):
        equilibrium_soln[i] = 0
    fill_equilibrium_system(equilibrium_matrix, num_rows, equilibrium_soln, spec, state)

    if spec.debugging_output:
        with gil:
            print(f"equilibrium matrix:\n{np.asarray(state._equilibrium_matrix[:num_rows*num_columns]).reshape((num_rows, num_columns), order='F')}")
            print(f"equilibrium rhs: {np.asarray(state._equilibrium_soln[:num_rows])}")

    lstsq(equilibrium_matrix, num_rows, num_columns, equilibrium_soln, 1e-16)

    if spec.debugging_output:
        with gil:
            print(f"equilibrium solution: {np.asarray(state._equilibrium_soln[:num_rows])}")

    # set the chemical potentials from the solution
    for i in range(spec.free_chemical_potential_indices.shape[0]):
        chempot_idx = spec.free_chemical_potential_indices[i]
        state.chemical_potentials[chempot_idx] = equilibrium_soln[i]

    # Force some chemical potentials to adopt their fixed values
    for chempot_idx in range(spec.fixed_chemical_potential_indices.shape[0]):
        comp_idx = spec.fixed_chemical_potential_indices[chempot_idx]
        state.chemical_potentials[comp_idx] = spec.initial_chemical_potentials[comp_idx]

    state.largest_chemical_potential_difference = -INFINITY
    for comp_idx in range(spec.num_components):
        state.largest_chemical_potential_difference = max(state.largest_chemical_potential_difference, abs(state.chemical_potentials[comp_idx] - state.previous_chemical_potentials[comp_idx]))
    return 0


# TODO: should we store equilibrium_soln in the state(?)
cpdef advance_state(SystemSpecification spec, SystemState state, double[::1] equilibrium_soln, double step_size):
    cdef int num_used = count_free_variables(spec, state)
    if num_used > spec.free_chemical_potential_indices.shape[0] and equilibrium_soln.shape[0] < num_used:
        raise IndexError('Out of bounds on buffer access (axis 0)')
    cdef double* soln = &equilibrium_soln[0] if equilibrium_soln.shape[0] > 0 else NULL
    if state._prepare(spec):
        with nogil:
            advance_state_nogil(spec, state, soln, step_size)
    else:
        advance_state_nogil(spec, state, soln, step_size)


cdef int advance_state_nogil(SystemSpecification spec, SystemState state, double* equilibrium_soln, double step_size) except -1 nogil:
    "Requires state._prepare(spec)."
    # Apply linear corrections in phase amounts, state variables and site fractions
    cdef double psc, phase_amt_step_size, x0
    cdef int i, idx, compset_idx, statevar_idx
    cdef int soln_index_offset = spec.free_chemical_potential_indices.shape[0]  # Chemical potentials handled after solving

    cdef double MIN_PHASE_AMOUNT = 1e-16

    # 1. Step in phase amounts
    # Determine largest allowable step size such that the smallest phase amount is zero
    phase_amt_step_size = step_size
    for i in range(state.free_stable_compset_indices.shape[0]):
        compset_idx = state.free_stable_compset_indices[i]
        if state.phase_amt[compset_idx] + equilibrium_soln[soln_index_offset + i] < MIN_PHASE_AMOUNT:
            # Assuming:
            # 1. NP>0 (the phase would not be a free_stable_compset if not) and
            # 2. delta_NP<0 (must be true if assumption #1 is true and this condition is true)
            # The largest allowable step size satisfies the equation: (NP + step_size * delta_NP = MIN_PHASE_AMOUNT)
            if abs(equilibrium_soln[soln_index_offset + i]) > MIN_PHASE_AMOUNT:
                phase_amt_step_size = min(phase_amt_step_size, (MIN_PHASE_AMOUNT - state.phase_amt[compset_idx]) / equilibrium_soln[soln_index_offset + i])
    # enforce a non-negative step size in case floating point precision near
    # MIN_PHASE_AMOUNT causes phase_amt_step_size to be a negative step
    phase_amt_step_size = max(phase_amt_step_size, 0.0)
    # Update the phase amounts using the largest allowable step size
    state.largest_phase_amt_change[0] = 0
    for i in range(state.free_stable_compset_indices.shape[0]):
        compset_idx = state.free_stable_compset_indices[i]
        state.phase_amt[compset_idx] += phase_amt_step_size * equilibrium_soln[soln_index_offset + i]
        if state.phase_amt[compset_idx] < MIN_PHASE_AMOUNT:
            # force non-zero, positive amounts to avoid divide by zero
            state.phase_amt[compset_idx] = MIN_PHASE_AMOUNT
        state.largest_phase_amt_change[0] = max(state.largest_phase_amt_change[0], abs(phase_amt_step_size * equilibrium_soln[soln_index_offset + i]))
    soln_index_offset += state.free_stable_compset_indices.shape[0]

    # 2. Step in state variables
    state.largest_statevar_change[0] = 0
    for statevar_idx in range(state.delta_statevars.shape[0]):
        state.delta_statevars[statevar_idx] = 0
    for i in range(spec.free_statevar_indices.shape[0]):
        statevar_idx = spec.free_statevar_indices[i]
        state.delta_statevars[statevar_idx] = equilibrium_soln[soln_index_offset + i]
        x0 = (<CompsetState>state._cs_state_ptrs[0]).dof[statevar_idx]
        if x0 == 0:
            psc = INFINITY
        else:
            psc = abs(state.delta_statevars[statevar_idx] / x0)
        state.largest_statevar_change[0] = max(state.largest_statevar_change[0], psc)
    # Update state variables in the `x` array
    for idx in range(state.num_compsets):
        for statevar_idx in range(state.delta_statevars.shape[0]):
            (<CompsetState>state._cs_state_ptrs[idx]).dof[statevar_idx] += state.delta_statevars[statevar_idx]
        # We need real state variable bounds support

    # 3. Step in phase internal degrees of freedom
    state.largest_y_change[0] = 0.0
    for idx in range(state.num_compsets):
        step_size = advance_compset(spec, state, <CompsetState>state._cs_state_ptrs[idx], idx, step_size)
    return 0


cdef double advance_compset(SystemSpecification spec, SystemState state, CompsetState csst, int idx, double step_size) except? -1 nogil:
    "Step the internal degrees of freedom of one composition set. Returns the step size, which may be reduced."
    cdef bint exceeded_bounds
    cdef double minimum_step_size
    cdef int i, cons_idx, statevar_idx, chempot_idx
    cdef double* x = &csst.dof[0]
    cdef double* new_y = &csst.new_y[0]
    cdef int num_dof = csst.dof.shape[0]

    # Construct delta_y from Eq. 43 in Sundman 2015
    for i in range(csst.delta_y.shape[0]):
        csst.delta_y[i] = 0

    for i in range(csst.delta_y.shape[0]):
        csst.delta_y[i] += csst.c_G[i]
        for statevar_idx in range(state.delta_statevars.shape[0]):
            csst.delta_y[i] += csst.c_statevars[i, statevar_idx] * state.delta_statevars[statevar_idx]
        for chempot_idx in range(state.chemical_potentials.shape[0]):
            csst.delta_y[i] += csst.c_component[chempot_idx, i] * state.chemical_potentials[chempot_idx]
        for cons_idx in range(csst.internal_cons.shape[0]):
            csst.delta_y[i] -= csst.full_e_matrix[csst.delta_y.shape[0] + cons_idx, i] * csst.internal_cons[cons_idx]

    for i in range(num_dof):
        new_y[i] = x[i]
    minimum_step_size = 1e-20 * step_size
    while step_size >= minimum_step_size:
        exceeded_bounds = False
        for i in range(spec.num_statevars, num_dof):
            new_y[i] = x[i] + step_size * csst.delta_y[i - spec.num_statevars]
            if new_y[i] > 1:
                if (new_y[i] - 1) > 1e-11:
                    # Allow some tolerance in the name of progress
                    exceeded_bounds = True
                new_y[i] = 1
            elif new_y[i] < MIN_SITE_FRACTION:
                if (MIN_SITE_FRACTION - new_y[i]) > 1e-11:
                    # Allow some tolerance in the name of progress
                    exceeded_bounds = True
                # Reduce by two orders of magnitude, or MIN_SITE_FRACTION, whichever is larger
                new_y[i] = max(x[i]/100, MIN_SITE_FRACTION)
        if exceeded_bounds:
            step_size *= 0.5
            continue
        break
    # Only allow stable or fixed phases can contribute to state.largest_y_change
    if (state.phase_amt[idx] > 0) or csst.compset.fixed:
        for i in range(spec.num_statevars, num_dof):
            state.largest_y_change[0] = max(state.largest_y_change[0], abs(x[i] - new_y[i]))
    for i in range(num_dof):
        x[i] = new_y[i]
    return step_size


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint is_fixed_stable(SystemSpecification spec, int idx) noexcept nogil:
    cdef int i
    for i in range(spec.fixed_stable_compset_indices.shape[0]):
        if spec.fixed_stable_compset_indices[i] == idx:
            return True
    return False


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint remove_and_consolidate_phases(SystemSpecification spec, SystemState state) noexcept nogil:
    """Remove phases that have become unstable (phase amount <= 0) and consolidate composition sets in an artificial misicbility gap.

    Updates the state in place.
    """
    cdef int i, j, idx, idx2, cp_idx, comp_idx, phase_idx, num_kept
    cdef int num_to_remove = 0
    cdef int num_free_stable = state.free_stable_compset_indices.shape[0]
    cdef int[::1] to_remove = state._remove_flags
    cdef bint phases_changed = False
    cdef bint compsets_should_be_consolidated
    cdef double composition_difference
    cdef double COMPSET_CONSOLIDATE_DISTANCE = 1e-4

    for idx in range(state.num_compsets):
        to_remove[idx] = False
    for i in range(num_free_stable):
        idx = state.free_stable_compset_indices[i]
        if (<CompsetState>state._cs_state_ptrs[idx]).compset.fixed:
            continue
        if to_remove[idx]:
            continue
        # Remove unstable phases
        if state.phase_amt[idx] < 1e-10:
            to_remove[idx] = True
            num_to_remove += 1
            state.phase_amt[idx] = 0
            continue
        for j in range(num_free_stable):
            idx2 = state.free_stable_compset_indices[j]
            if idx == idx2:
                continue
            if (<CompsetState>state._cs_state_ptrs[idx2]).compset.fixed:
                continue
            if state._phase_ids[idx] != state._phase_ids[idx2]:
                continue
            if to_remove[idx2]:
                continue
            # Detect if these compsets describe the same internal configuration inside a miscibility gap
            compsets_should_be_consolidated = True
            # Detected based on composition, we may miss gaps that have nearly
            # the same composition, but different site fractions (such as ordering)
            for comp_idx in range(spec.num_components):
                composition_difference = abs(state.phase_compositions[idx, comp_idx] - state.phase_compositions[idx2, comp_idx])
                if composition_difference > COMPSET_CONSOLIDATE_DISTANCE:
                    compsets_should_be_consolidated = False
                    break
            if compsets_should_be_consolidated:
                to_remove[idx2] = True
                num_to_remove += 1
                if not is_fixed_stable(spec, idx):
                    # ensure that the consolidated phase is stable
                    state.phase_amt[idx] = max(state.phase_amt[idx] + state.phase_amt[idx2], 1e-8)
                state.phase_amt[idx2] = 0
    if num_to_remove > 0:
        if num_to_remove - num_free_stable == 0:
            # Do not allow all phases to leave the system
            for i in range(num_free_stable):
                phase_idx = state.free_stable_compset_indices[i]
                state.phase_amt[phase_idx] = 1
            for comp_idx in range(state.chemical_potentials.shape[0]):
                state.chemical_potentials[comp_idx] = 0
            # Force some chemical potentials to adopt their fixed values
            for cp_idx in range(spec.fixed_chemical_potential_indices.shape[0]):
                comp_idx = spec.fixed_chemical_potential_indices[cp_idx]
                state.chemical_potentials[comp_idx] = spec.initial_chemical_potentials[comp_idx]
        else:
            # Keep the remaining indices in (sorted) order, in place
            num_kept = 0
            for i in range(num_free_stable):
                idx = state.free_stable_compset_indices[i]
                if to_remove[idx]:
                    state.times_compset_removed[idx] += 1
                else:
                    state._free_stable_buffer[num_kept] = idx
                    num_kept += 1
            state.free_stable_compset_indices = state._free_stable_buffer[:num_kept]
            phases_changed = True
    return phases_changed


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int change_phases(SystemSpecification spec, SystemState state) except -1 nogil:
    "Returns 1 if the stable composition sets changed, otherwise 0."
    cdef int idx, i, cs_idx, least_removed_cs_idx, smallest_df_cs_idx, num_new_free_stable
    cdef int num_free_stable = state.free_stable_compset_indices.shape[0]
    cdef int num_to_remove = 0
    cdef int num_to_add = 0
    cdef int max_allowed_to_add
    cdef bint phases_changed, is_new_free_stable, should_add_compset
    cdef int[::1] to_remove = state._remove_flags
    cdef int[::1] to_add = state._add_flags
    cdef double[::1] driving_forces = state._driving_forces
    cdef double MIN_PHASE_AMOUNT = 1e-9
    cdef int MIN_REQUIRED_METASTABLE_PHASE_ITERATIONS_TO_ADD = 5
    cdef double MIN_DRIVING_FORCE_TO_ADD = 1e-5
    cdef int MAX_ALLOWED_TIMES_COMPSET_REMOVED = 4
    state.compute_driving_forces()
    if num_free_stable > spec.max_num_free_stable_phases:
        # Gibbs phase rule is currently being violated
        # Try forcing phases with small amounts out of the equilibrium
        MIN_PHASE_AMOUNT = 1e-4
    for cs_idx in range(state.num_compsets):
        to_remove[cs_idx] = False
        to_add[cs_idx] = False
    for i in range(num_free_stable):
        cs_idx = state.free_stable_compset_indices[i]
        if state.phase_amt[cs_idx] < MIN_PHASE_AMOUNT:
            to_remove[cs_idx] = True
            num_to_remove += 1

    # Only add phases with positive driving force which have been metastable for at least 5 iterations, which have been removed fewer than 4 times
    for cs_idx in range(state.metastable_phase_iterations.shape[0]):
        should_add_compset = (
            (state.metastable_phase_iterations[cs_idx] >= MIN_REQUIRED_METASTABLE_PHASE_ITERATIONS_TO_ADD)
            and (driving_forces[cs_idx] > MIN_DRIVING_FORCE_TO_ADD)
            and (state.times_compset_removed[cs_idx] < MAX_ALLOWED_TIMES_COMPSET_REMOVED)
        )
        if should_add_compset:
            to_add[cs_idx] = True
    # Finally, remove all currently stable compsets as candidates
    for i in range(num_free_stable):
        to_add[state.free_stable_compset_indices[i]] = False
    for cs_idx in range(state.num_compsets):
        num_to_add += to_add[cs_idx]
    max_allowed_to_add = spec.max_num_free_stable_phases + num_to_remove - num_free_stable
    # We must obey the Gibbs phase rule
    if num_to_add > 0:
        if max_allowed_to_add < 1:
            # We are at the maximum number of allowed phases, yet there is still positive driving force
            # Destabilize one phase and add only one phase
            # Destabilize the one that has been removed the least
            least_removed_cs_idx = -1
            for i in range(num_free_stable):
                cs_idx = state.free_stable_compset_indices[i]
                if to_remove[cs_idx]:
                    continue
                if (least_removed_cs_idx == -1) or (state.times_compset_removed[cs_idx] < state.times_compset_removed[least_removed_cs_idx]):
                    least_removed_cs_idx = cs_idx
            if least_removed_cs_idx == -1:
                with gil:
                    raise IndexError('No stable composition set can be destabilized')
            to_remove[least_removed_cs_idx] = True
            num_to_remove += 1
            state.phase_amt[least_removed_cs_idx] = 0
        # Add the compset with least amount (but still positive) driving force
        smallest_df_cs_idx = -1
        for cs_idx in range(state.num_compsets):
            if not to_add[cs_idx]:
                continue
            if (smallest_df_cs_idx == -1) or (driving_forces[cs_idx] < driving_forces[smallest_df_cs_idx]):
                smallest_df_cs_idx = cs_idx
            to_add[cs_idx] = False
        to_add[smallest_df_cs_idx] = True
    # New stable compsets: the current ones which are not removed, and the added one, in sorted order
    num_new_free_stable = 0
    for cs_idx in range(state.num_compsets):
        is_new_free_stable = to_add[cs_idx]
        for i in range(num_free_stable):
            if state.free_stable_compset_indices[i] == cs_idx:
                is_new_free_stable = not to_remove[cs_idx]
                break
        if is_new_free_stable:
            state._new_free_stable_buffer[num_new_free_stable] = cs_idx
            num_new_free_stable += 1
    for i in range(num_free_stable):
        cs_idx = state.free_stable_compset_indices[i]
        if to_remove[cs_idx]:
            state.times_compset_removed[cs_idx] += 1
    for idx in range(state.num_compsets):
        if to_add[idx] or (not to_remove[idx] and is_free_stable(state, idx)):
            # Force some amount of newly stable phases
            if state.phase_amt[idx] < 1e-10:
                state.phase_amt[idx] = 1e-10
        # Force unstable phase amounts to zero
        else:
            state.phase_amt[idx] = 0
    # feasible system, and no phases to add or remove, if the sets are equal
    phases_changed = (num_to_remove > 0) or (num_to_add > 0)
    for i in range(num_new_free_stable):
        state._free_stable_buffer[i] = state._new_free_stable_buffer[i]
    state.free_stable_compset_indices = state._free_stable_buffer[:num_new_free_stable]
    return phases_changed


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint is_free_stable(SystemState state, int idx) noexcept nogil:
    cdef int i
    for i in range(state.free_stable_compset_indices.shape[0]):
        if state.free_stable_compset_indices[i] == idx:
            return True
    return False
