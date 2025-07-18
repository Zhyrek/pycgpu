cimport cython
import numpy as np
cimport numpy as np
import sys
from pycalphad.core.composition_set cimport CompositionSet
from pycalphad.core.constants import MIN_SITE_FRACTION
cimport scipy.linalg.cython_lapack as cython_lapack
from libc.stdlib cimport malloc, free
from libc.math cimport isnan
from libc.stdio cimport printf

@cython.boundscheck(False)
cdef void lstsq(double *A, int M, int N, double* x, double rcond) nogil:
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
cdef void invert_matrix(double *A, int N, int* ipiv) nogil:
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
                               CompositionSet compset, int num_statevars, double[::1] chemical_potentials,
                               double[::1] phase_dof) nogil:
    "Compute the LHS of Eq. 41, Sundman 2015."
    cdef int comp_idx, i, j
    cdef int num_components = chemical_potentials.shape[0]
    compset.phase_record.internal_cons_jac(cons_jac_tmp, phase_dof)
    if compset.num_phase_local_conditions > 0:
        compset.phase_record.phase_local_cons_jac(phase_local_jac_tmp, phase_dof, compset.phase_local_cons_jac)

    # DEBUG: Print Hessian values
    printf("[CPU HESSIAN DEBUG] compute_phase_matrix called\n")
    for i in range(compset.phase_record.phase_dof):
        for j in range(compset.phase_record.phase_dof):
            phase_matrix[i, j] = hess[num_statevars+i, num_statevars+j]
            if i < 2 and j < 2:
                printf("  hess[%d,%d] = %e\n", num_statevars+i, num_statevars+j, hess[num_statevars+i, num_statevars+j])

    for i in range(compset.phase_record.num_internal_cons):
        for j in range(compset.phase_record.phase_dof):
            phase_matrix[compset.phase_record.phase_dof+i, j] = cons_jac_tmp[i, num_statevars+j]
            phase_matrix[j, compset.phase_record.phase_dof+i] = cons_jac_tmp[i, num_statevars+j]

    for i in range(compset.num_phase_local_conditions):
        for j in range(compset.phase_record.phase_dof):
            phase_matrix[compset.phase_record.phase_dof+compset.phase_record.num_internal_cons+i, j] = phase_local_jac_tmp[i, num_statevars+j]
            phase_matrix[j, compset.phase_record.phase_dof+compset.phase_record.num_internal_cons+i] = phase_local_jac_tmp[i, num_statevars+j]


cdef void write_row_stable_phase(double[:] out_row, double* out_rhs, int[::1] free_chemical_potential_indices,
                                 int[::1] free_stable_compset_indices, int[::1] free_statevar_indices,
                                 int[::1] fixed_chemical_potential_indices, double[::1] chemical_potentials,
                                 double[:, ::1] masses, double[::1] grad, double energy):
    # DEBUG: Print the row being written
    cdef int debug_idx
    printf("[CPU EQUILIBRIUM MATRIX DEBUG] Writing row for stable phase:\n")
    printf("  Energy: %e\n", energy)
    printf("  Masses: ")
    for debug_idx in range(masses.shape[0]):
        printf("%e ", masses[debug_idx, 0])
    printf("\n")
    
    # 1a. This phase row: free chemical potentials
    cdef int free_variable_column_offset = 0
    cdef int chempot_idx, statevar_idx, i
    for i in range(free_chemical_potential_indices.shape[0]):
        chempot_idx = free_chemical_potential_indices[i]
        out_row[free_variable_column_offset + i] = masses[chempot_idx, 0]
    free_variable_column_offset += free_chemical_potential_indices.shape[0]
    # 1a. This phase row: free stable composition sets = zero contribution
    free_variable_column_offset += free_stable_compset_indices.shape[0]
    # 1a. This phase row: free state variables
    for i in range(free_statevar_indices.shape[0]):
        statevar_idx = free_statevar_indices[i]
        out_row[free_variable_column_offset + i] = -grad[statevar_idx]
    out_rhs[0] = energy
    # 4. Subtract fixed chemical potentials from each phase RHS
    for i in range(fixed_chemical_potential_indices.shape[0]):
        chempot_idx = fixed_chemical_potential_indices[i]
        out_rhs[0] -= masses[chempot_idx, 0] * chemical_potentials[chempot_idx]
    
    # DEBUG: Print the final row values and RHS
    printf("  Row values (first %d): ", 
           free_chemical_potential_indices.shape[0] + free_stable_compset_indices.shape[0] + free_statevar_indices.shape[0])
    for debug_idx in range(free_chemical_potential_indices.shape[0] + free_stable_compset_indices.shape[0] + free_statevar_indices.shape[0]):
        printf("%e ", out_row[debug_idx])
    printf("\n")
    printf("  RHS: %e\n", out_rhs[0])

cdef void write_row_fixed_mole_fraction(double[:] out_row, double* out_rhs, int component_idx,
                                        int[::1] free_chemical_potential_indices, int[::1] free_stable_compset_indices,
                                        int[::1] free_statevar_indices, int[::1] fixed_chemical_potential_indices,
                                        double[::1] chemical_potentials,
                                        double [::1] system_mole_fractions, double current_system_amount,
                                        double[:, ::1] mass_jac, double[:, ::1] c_component,
                                        double[:, ::1] c_statevars, double[::1] c_G, double[:, ::1] masses,
                                        double moles_normalization, double[::1] moles_normalization_grad,
                                        double[::1] phase_amt, int idx, double prefactor):
    if prefactor == 0.0:
        return
    
    # DEBUG: Print when this function is called
    import sys
    sys.stderr.write(f"[CPU] write_row_fixed_mole_fraction called: phase_idx={idx}, component_idx={component_idx}, prefactor={prefactor}\n")
    sys.stderr.write(f"  out_rhs[0] on entry: {out_rhs[0]}\n")
    sys.stderr.flush()
    cdef int free_variable_column_offset = 0
    cdef int num_statevars = c_statevars.shape[1]
    cdef int chempot_idx, compset_idx, statevar_idx, i, j
    
    # DEBUG: Print mass_jac structure for first call
    if idx == 0 and component_idx == 1:
        printf("[CPU MOLE FRAC DEBUG] Phase %d, Component %d:\n", idx, component_idx)
        printf("  mass_jac shape: (%d, %d)\n", mass_jac.shape[0], mass_jac.shape[1])
        printf("  num_statevars: %d\n", num_statevars)
        printf("  mass_jac[1,:]: ")
        for j in range(mass_jac.shape[1]):
            printf("%e ", mass_jac[component_idx, j])
        printf("\n")
    # 2a. This component row: free chemical potentials
    for i in range(free_chemical_potential_indices.shape[0]):
        chempot_idx = free_chemical_potential_indices[i]
        for j in range(c_component.shape[1]):
            out_row[free_variable_column_offset + i] += prefactor * \
                (phase_amt[idx]/current_system_amount) * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
        for j in range(c_component.shape[1]):
            out_row[free_variable_column_offset + i] += prefactor * \
                (phase_amt[idx]/current_system_amount) * (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_component[chempot_idx, j]
    free_variable_column_offset += free_chemical_potential_indices.shape[0]
    # 2a. This component row: free stable composition sets
    for i in range(free_stable_compset_indices.shape[0]):
        compset_idx = free_stable_compset_indices[i]
        # Only fill this out if the current idx is equal to a free composition set
        if compset_idx == idx:
            out_row[free_variable_column_offset + i] += prefactor * \
                (1./current_system_amount)*(masses[component_idx, 0] - system_mole_fractions[component_idx] * moles_normalization)
    free_variable_column_offset += free_stable_compset_indices.shape[0]
    # 2a. This component row: free state variables
    for i in range(free_statevar_indices.shape[0]):
        statevar_idx = free_statevar_indices[i]
        for j in range(c_statevars.shape[0]):
            out_row[free_variable_column_offset + i] += prefactor * \
                (phase_amt[idx]/current_system_amount) * mass_jac[component_idx, num_statevars+j] * c_statevars[j, statevar_idx]
        for j in range(c_statevars.shape[0]):
            out_row[free_variable_column_offset + i] += prefactor * \
                (phase_amt[idx]/current_system_amount) * (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_statevars[j, statevar_idx]
    # 3.
    # DEBUG: Print RHS calculation details
    if idx < 2 and component_idx == 1 and prefactor != 0.0:
        print(f"[CPU MOLE FRAC RHS DEBUG] Phase {idx}, Component 1:")
        print(f"  c_G.shape[0]={c_G.shape[0]}")
        if c_G.shape[0] > 0:
            print(f"  c_G values: {np.asarray(c_G)}")
        print(f"  mass_jac[1,:]: {np.asarray(mass_jac[1,:])}")
        print(f"  moles_normalization_grad: {np.asarray(moles_normalization_grad)}")
        
    cdef double rhs_term1 = 0.0
    cdef double rhs_term2 = 0.0
    
    for j in range(c_G.shape[0]):
        if idx < 2 and component_idx == 1 and prefactor != 0.0:
            print(f"    j={j}: mass_jac[1,{num_statevars+j}]={mass_jac[component_idx, num_statevars+j]:.6e}, c_G[{j}]={c_G[j]:.6e}")
        rhs_term1 += mass_jac[component_idx, num_statevars+j] * c_G[j]
        out_rhs[0] += -prefactor * (phase_amt[idx]/current_system_amount) * \
            mass_jac[component_idx, num_statevars+j] * c_G[j]
    for j in range(c_G.shape[0]):
        rhs_term2 += (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_G[j]
        out_rhs[0] += -prefactor * (phase_amt[idx]/current_system_amount) * \
            (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_G[j]
            
    if idx < 2 and component_idx == 1 and prefactor != 0.0:
        print(f"  [CPU] Phase {idx} rhs_term1={rhs_term1}, rhs_term2={rhs_term2}")
        print(f"  phase_amt={phase_amt[idx]}, system_amt={current_system_amount}, prefactor={prefactor}")
        print(f"  RHS contribution: {-prefactor * (phase_amt[idx]/current_system_amount) * (rhs_term1 + rhs_term2)}")
        print(f"  out_rhs[0] after this phase: {out_rhs[0]}")
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
    

cdef void write_row_fixed_mole_amount(double[:] out_row, double* out_rhs, int component_idx,
                                      int[::1] free_chemical_potential_indices, int[::1] free_stable_compset_indices,
                                      int[::1] free_statevar_indices, int[::1] fixed_chemical_potential_indices,
                                      double[::1] chemical_potentials,
                                      double[:, ::1] mass_jac, double[:, ::1] c_component,
                                      double[:, ::1] c_statevars, double[::1] c_G, double[:, ::1] masses,
                                      double[::1] phase_amt, int idx):
    cdef int free_variable_column_offset = 0
    cdef int num_statevars = c_statevars.shape[1]
    cdef int i, j, chempot_idx, compset_idx, statevar_idx
    # 2a. This component row: free chemical potentials
    for i in range(free_chemical_potential_indices.shape[0]):
        chempot_idx = free_chemical_potential_indices[i]
        for j in range(c_component.shape[1]):
            out_row[free_variable_column_offset + i] += \
                phase_amt[idx] * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
    free_variable_column_offset += free_chemical_potential_indices.shape[0]
    # 2a. This component row: free stable composition sets
    for i in range(free_stable_compset_indices.shape[0]):
        compset_idx = free_stable_compset_indices[i]
        # Only fill this out if the current idx is equal to a free composition set
        if compset_idx == idx:
            out_row[free_variable_column_offset + i] += masses[component_idx, 0]
    free_variable_column_offset += free_stable_compset_indices.shape[0]
    # 2a. This component row: free state variables
    for i in range(free_statevar_indices.shape[0]):
        statevar_idx = free_statevar_indices[i]
        for j in range(c_statevars.shape[0]):
            out_row[free_variable_column_offset + i] += \
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


cdef void fill_equilibrium_system(double[::1,:] equilibrium_matrix, double[::1] equilibrium_rhs,
                                  SystemSpecification spec, SystemState state):
    from pycalphad.core.debug_output import debug_log
    cdef int stable_idx, idx, component_row_offset, component_idx, fixed_idx, free_idx
    cdef int fixed_component_idx, comp_idx, system_amount_index, fixed_molefrac_cond_idx
    cdef CompositionSet compset
    cdef CompsetState csst
    cdef int num_components = state.chemical_potentials.shape[0]
    cdef int num_stable_phases = state.free_stable_compset_indices.shape[0]
    cdef int num_fixed_phases = spec.fixed_stable_compset_indices.shape[0]
    cdef int num_fixed_mole_fraction_conditions = spec.prescribed_mole_fraction_rhs.shape[0]
    cdef double prefactor
    cdef bint debug_enabled = False  # Debug controlled globally via debug_output.py

    # SEGMENT 28: FILL EQUILIBRIUM SYSTEM - PHASE ROWS
    debug_log(28, f"Fill equilibrium system - phase rows (iteration {state.iteration})")
    debug_log(f"  num_stable_phases: {num_stable_phases}", debug_enabled)
    debug_log(f"  num_fixed_phases: {num_fixed_phases}", debug_enabled)

    for stable_idx in range(state.free_stable_compset_indices.shape[0]):
        idx = state.free_stable_compset_indices[stable_idx]
        compset = state.compsets[idx]
        csst = state.cs_states[idx]

        write_row_stable_phase(equilibrium_matrix[stable_idx, :], &equilibrium_rhs[stable_idx], spec.free_chemical_potential_indices,
                               state.free_stable_compset_indices, spec.free_statevar_indices, spec.fixed_chemical_potential_indices,
                               state.chemical_potentials, csst.masses, csst.grad, csst.energy)
        
        # DEBUG: Print gradient details for first iteration
        if state.iteration < 3 and stable_idx < 2:
            print(f"[CPU] Phase {stable_idx} gradient (iteration {state.iteration}): {np.asarray(csst.grad)}")
            print(f"[CPU] Phase {stable_idx} energy: {csst.energy:.15e}")
            print(f"[CPU] Phase {stable_idx} masses: {np.asarray(csst.masses)}")
            print(f"[CPU] Phase {stable_idx} equilibrium_rhs before: {equilibrium_rhs[stable_idx]:.15e}")
            
        debug_log(f"  phase_row_{stable_idx}_rhs: {equilibrium_rhs[stable_idx]:.15e}", debug_enabled)

    # Handle phases which are fixed to be stable at some amount
    # Example shown in Eq. 60, Sundman et al 2015
    for fixed_idx in range(spec.fixed_stable_compset_indices.shape[0]):
        idx = spec.fixed_stable_compset_indices[fixed_idx]
        compset = state.compsets[idx]
        csst = state.cs_states[idx]
        write_row_stable_phase(equilibrium_matrix[num_stable_phases + fixed_idx, :],
                               &equilibrium_rhs[num_stable_phases + fixed_idx], spec.free_chemical_potential_indices,
                               state.free_stable_compset_indices, spec.free_statevar_indices, spec.fixed_chemical_potential_indices,
                               state.chemical_potentials, csst.masses, csst.grad, csst.energy)

    # SEGMENT 29: FILL EQUILIBRIUM SYSTEM - CONSTRAINT ROWS
    debug_log(29, f"Fill equilibrium system - constraint rows (iteration {state.iteration})")
    debug_log(f"  num_fixed_mole_fraction_conditions: {num_fixed_mole_fraction_conditions}", debug_enabled)
    
    # DEBUG: Always print to check if mole fraction constraints are being filled
    import sys
    sys.stderr.write(f"[CPU DEBUG] Iteration {state.iteration}: num_fixed_mole_fraction_conditions = {num_fixed_mole_fraction_conditions}\n")
    sys.stderr.write(f"[CPU DEBUG] num_stable_phases = {state.free_stable_compset_indices.shape[0]}\n")
    sys.stderr.flush()
    
    for stable_idx in range(state.free_stable_compset_indices.shape[0]):
        idx = state.free_stable_compset_indices[stable_idx]
        compset = state.compsets[idx]
        csst = state.cs_states[idx]
        # 2. Contribute to the row of all fixed mole fraction conditions
        component_row_offset = num_stable_phases + num_fixed_phases
        for fixed_molefrac_cond_idx in range(num_fixed_mole_fraction_conditions):
            for component_idx in range(spec.prescribed_mole_fraction_coefficients.shape[1]):
                prefactor = spec.prescribed_mole_fraction_coefficients[fixed_molefrac_cond_idx, component_idx]
                write_row_fixed_mole_fraction(equilibrium_matrix[component_row_offset + fixed_molefrac_cond_idx, :],
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
            write_row_fixed_mole_amount(equilibrium_matrix[system_amount_index, :],
                                        &equilibrium_rhs[system_amount_index], component_idx,
                                        spec.free_chemical_potential_indices, state.free_stable_compset_indices,
                                        spec.free_statevar_indices, spec.fixed_chemical_potential_indices,
                                        state.chemical_potentials, csst.mass_jac, csst.c_component,
                                        csst.c_statevars, csst.c_G, csst.masses,
                                        state.phase_amt, idx)

    for fixed_idx in range(spec.fixed_stable_compset_indices.shape[0]):
        idx = spec.fixed_stable_compset_indices[fixed_idx]
        compset = state.compsets[idx]
        csst = state.cs_states[idx]
        # 2. Contribute to the row of all fixed mole fraction conditions
        component_row_offset = num_stable_phases + num_fixed_phases
        for fixed_molefrac_cond_idx in range(num_fixed_mole_fraction_conditions):
            for component_idx in range(spec.prescribed_mole_fraction_coefficients.shape[1]):
                prefactor = spec.prescribed_mole_fraction_coefficients[fixed_molefrac_cond_idx, component_idx]
                write_row_fixed_mole_fraction(equilibrium_matrix[component_row_offset + fixed_molefrac_cond_idx, :],
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
            write_row_fixed_mole_amount(equilibrium_matrix[system_amount_index, :],
                                        &equilibrium_rhs[system_amount_index], component_idx,
                                        spec.free_chemical_potential_indices, state.free_stable_compset_indices,
                                        spec.free_statevar_indices, spec.fixed_chemical_potential_indices,
                                        state.chemical_potentials, csst.mass_jac, csst.c_component,
                                        csst.c_statevars, csst.c_G, csst.masses,
                                        state.phase_amt, idx)


    # SEGMENT 30: FILL EQUILIBRIUM SYSTEM - RESIDUALS
    debug_log(30, f"Fill equilibrium system - residuals (iteration {state.iteration})")
    
    # Show first few residuals for debugging
    debug_log(f"  equilibrium_rhs[0:3]: {equilibrium_rhs[0]:.15e}, {equilibrium_rhs[1]:.15e}, {equilibrium_rhs[2]:.15e}", debug_enabled)
    
    # Add mass residual to fixed component row RHS, plus N=1 row
    component_row_offset = num_stable_phases + num_fixed_phases
    system_amount_index = component_row_offset + num_fixed_mole_fraction_conditions
    for fixed_molefrac_cond_idx in range(num_fixed_mole_fraction_conditions):
        component_residual = np.dot(spec.prescribed_mole_fraction_coefficients[fixed_molefrac_cond_idx, :], state.mole_fractions) - spec.prescribed_mole_fraction_rhs[fixed_molefrac_cond_idx]
        equilibrium_rhs[component_row_offset + fixed_molefrac_cond_idx] -= component_residual
        debug_log(f"  constraint_{fixed_molefrac_cond_idx}_residual: {component_residual:.15e}", debug_enabled)
        # DEBUG: Print mole fraction constraint RHS
        if state.iteration < 3:
            print(f"[CPU] Mole fraction constraint {fixed_molefrac_cond_idx} RHS before residual: {equilibrium_rhs[component_row_offset + fixed_molefrac_cond_idx] + component_residual:.6f}")
            print(f"[CPU] Mole fraction constraint {fixed_molefrac_cond_idx} residual: {component_residual:.6f}")
            print(f"[CPU] Mole fraction constraint {fixed_molefrac_cond_idx} RHS after residual: {equilibrium_rhs[component_row_offset + fixed_molefrac_cond_idx]:.6f}")
    
    system_residual = state.system_amount - spec.prescribed_system_amount
    equilibrium_rhs[system_amount_index] -= system_residual
    
    debug_log(f"  system_amount_residual: {system_residual:.15e}", debug_enabled)
    debug_log(f"  mass_residual: {state.mass_residual:.15e}", debug_enabled)


cdef class SystemSpecification:
    def __init__(self, int num_statevars, int num_components, double prescribed_system_amount,
                   double[::1] initial_chemical_potentials, double[:, ::1] prescribed_mole_fraction_coefficients,
                   double[::1] prescribed_mole_fraction_rhs, int[::1] free_chemical_potential_indices,
                   int[::1] free_statevar_indices, int[::1] fixed_chemical_potential_indices,
                   int[::1] fixed_statevar_indices, int[::1] fixed_stable_compset_indices):
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

        # Assuming the prescribed_mole_fraction_rhs doesn't change, this is
        # constant and we can keep extra computation (especially calls into
        # NumPy out of the run loop)
        if self.prescribed_mole_fraction_rhs.shape[0] > 0:
            # With linear combinations of conditions, RHS can now be exactly zero
            # This means the smallest allowed mass residual needs to be limited to prevent instability
            self.ALLOWED_MASS_RESIDUAL = max(1e-12, min(1e-8, np.min(np.abs(self.prescribed_mole_fraction_rhs))/10.0))
            # Also adjust mass residual if we are near the edge of composition space
            self.ALLOWED_MASS_RESIDUAL = min(self.ALLOWED_MASS_RESIDUAL, (1-np.sum(np.abs(self.prescribed_mole_fraction_rhs)))/10.0)
        else:
            self.ALLOWED_MASS_RESIDUAL = 1e-8

    def __getstate__(self):
        return (self.num_statevars, self.num_components, self.prescribed_system_amount,
                np.array(self.initial_chemical_potentials), np.array(self.prescribed_mole_fraction_coefficients),
                np.array(self.prescribed_mole_fraction_rhs), np.array(self.free_chemical_potential_indices),
                np.array(self.free_statevar_indices), np.array(self.fixed_chemical_potential_indices),
                np.array(self.fixed_statevar_indices), np.array(self.fixed_stable_compset_indices))
    def __setstate__(self, state):
        self.__init__(*state)

    cpdef bint check_convergence(self, SystemState state):
        from pycalphad.core.debug_output import debug_log
        cdef bint debug_enabled = False  # Debug controlled globally via debug_output.py
        
        # SEGMENT 35: CHECK CONVERGENCE
        debug_log(35, f"Check convergence (iteration {state.iteration})")
        
        # convergence criteria
        cdef double ALLOWED_DELTA_Y = 5e-09
        cdef double ALLOWED_DELTA_PHASE_AMT = 1e-10
        cdef double ALLOWED_DELTA_STATEVAR = 1e-5  # changes defined as percent change
        
        debug_log(f"  largest_phase_amt_change: {state.largest_phase_amt_change[0]:.15e}", debug_enabled)
        debug_log(f"  largest_y_change: {state.largest_y_change[0]:.15e}", debug_enabled)
        debug_log(f"  largest_statevar_change: {state.largest_statevar_change[0]:.15e}", debug_enabled)
        debug_log(f"  mass_residual: {state.mass_residual:.15e}", debug_enabled)
        debug_log(f"  iterations_since_last_phase_change: {state.iterations_since_last_phase_change}", debug_enabled)
        
        cdef bint solution_is_feasible = (
            (state.largest_phase_amt_change[0] < ALLOWED_DELTA_PHASE_AMT) and
            (state.largest_y_change[0] < ALLOWED_DELTA_Y) and
            (state.largest_statevar_change[0] < ALLOWED_DELTA_STATEVAR) and
            (state.mass_residual < self.ALLOWED_MASS_RESIDUAL)
        )
        
        debug_log(f"  solution_is_feasible: {solution_is_feasible}", debug_enabled)
        
        if solution_is_feasible and (state.iterations_since_last_phase_change >= 5):
            debug_log(f"  converged: True", debug_enabled)
            return True
        else:
            debug_log(f"  converged: False", debug_enabled)
            return False

    cpdef bint pre_solve_hook(self, SystemState state):
        return True

    cpdef bint post_solve_hook(self, SystemState state):
        return True

    cpdef bint run_loop(self, SystemState state, int max_iterations):
        from pycalphad.core.debug_output import debug_log
        
        # Run loop entry
        
        cdef double step_size = 1.0
        cdef bint converged = False
        cdef bint phases_changed = False
        cdef size_t iteration
        
        # Use False for verbose flag in debug messages (debug output controlled globally)
        debug_log(f"  max_iterations: {max_iterations}", False)
        debug_log(f"  initial_step_size: {step_size}", False) 
        debug_log(f"  num_phases: {len(state.free_stable_compset_indices)}", False)
        debug_log(f"  free_stable_compset_indices: {state.free_stable_compset_indices}", False)
        
        # DEBUG: Enable debug logging for state object
        # Commented out - would need to declare _debug_enabled in cdef class
        # try:
        #     state._debug_enabled = True
        # except:
        #     pass  # Ignore if we can't set the attribute
            
        # DEBUG: Initial state logging
        # Commented out - would need _debug_enabled attribute
        # if hasattr(state, '_debug_enabled') and state._debug_enabled:
        #     print(f"\n[CPU DEBUG] ===== SOLVER STARTING =====")
        #     print(f"[CPU DEBUG] Initial state:")
        #     print(f"[CPU DEBUG]   Number of phases: {len(state.free_stable_compset_indices)}")
        #     print(f"[CPU DEBUG]   Chemical potentials: {np.array(state.chemical_potentials)}")
        #     for idx in state.free_stable_compset_indices:
        #         compset = state.compsets[idx]
        #         print(f"[CPU DEBUG]   Phase {idx} ({compset.phase_record.phase_name}): amount={state.phase_amt[idx]:.6f}")
        
        for iteration in range(max_iterations):
            state.iteration = iteration
            
            # DEBUG: Log iteration start  
            if False and iteration == 0:  # Detailed for first iteration - disabled
                print(f"\n[CPU DEBUG] ===== ITERATION {iteration} (DETAILED) =====")
                print(f"[CPU DEBUG] State before iteration:")
                print(f"[CPU DEBUG]   Chemical potentials: {np.array(state.chemical_potentials)}")
                print(f"[CPU DEBUG]   Number of phases: {len(state.free_stable_compset_indices)}")
                print(f"[CPU DEBUG]   Free stable indices: {state.free_stable_compset_indices}")
                print(f"[CPU DEBUG]   System amount: {state.system_amount}")
                print(f"[CPU DEBUG]   Mole fractions: {state.mole_fractions}")
                for idx in state.free_stable_compset_indices:
                    compset = state.compsets[idx]
                    print(f"[CPU DEBUG]   Phase {idx} ({compset.phase_record.phase_name}):")
                    print(f"[CPU DEBUG]     NP={compset.NP:.6f}")
                    print(f"[CPU DEBUG]     phase_amt={state.phase_amt[idx]:.6f} (formula units)")
                    print(f"[CPU DEBUG]     energy={state.cs_states[idx].energy:.6f}")
                    print(f"[CPU DEBUG]     dof={np.array(state.dof[idx])}")
                    print(f"[CPU DEBUG]     phase_compositions={state.phase_compositions[idx]}")
                    # Calculate phase_comp_sum for this phase
                    phase_comp_sum = np.sum(state.phase_compositions[idx])
                    print(f"[CPU DEBUG]     phase_comp_sum={phase_comp_sum:.6f}")
                    print(f"[CPU DEBUG]     phase_amt * phase_comp_sum={state.phase_amt[idx] * phase_comp_sum:.6f}")
            elif False:  # disabled
                print(f"\n[CPU DEBUG] ===== ITERATION {iteration} =====")
                print(f"[CPU DEBUG]   Chemical potentials: {np.array(state.chemical_potentials)}")
                for idx in state.free_stable_compset_indices:
                    print(f"[CPU DEBUG]   Phase {idx}: amount={state.phase_amt[idx]:.6f}, energy={state.cs_states[idx].energy:.6f}")
            
            if not self.pre_solve_hook(state):
                break
            eq_soln = solve_state(self, state)
            
            # DEBUG: Log equilibrium solution
            if False and iteration == 0:  # Detailed for first iteration - disabled
                print(f"[CPU DEBUG] Equilibrium solution: {np.array(eq_soln)}")
                print(f"[CPU DEBUG] After solve_state:")
                print(f"[CPU DEBUG]   Chemical potentials: {np.array(state.chemical_potentials)}")
                for idx in state.free_stable_compset_indices:
                    compset = state.compsets[idx]
                    print(f"[CPU DEBUG]   Phase {idx} ({compset.phase_record.phase_name}):")
                    print(f"[CPU DEBUG]     phase_amt={state.phase_amt[idx]:.6f} (formula units)")
                    print(f"[CPU DEBUG]     NP={compset.NP:.6f}")
                    print(f"[CPU DEBUG]     dof={np.array(state.dof[idx])}")
            elif False:  # disabled
                print(f"[CPU DEBUG] Equilibrium solution: {np.array(eq_soln)}")
            
            # SEGMENT 33: POST SOLVE HOOK
            debug_log(33, f"Post solve hook (iteration {state.iteration})")
            
            if not self.post_solve_hook(state):
                debug_log(f"  post_solve_hook_returned_false", True)
                break
            
            debug_log(f"  post_solve_hook_returned_true", True)
            phases_changed = remove_and_consolidate_phases(self, state)
            converged = self.check_convergence(state)
            
            # DEBUG: Log convergence status
            # if state._debug_enabled:
            #     print(f"[CPU DEBUG] Phases changed: {phases_changed}, Converged: {converged}")
            
            if converged:
                phases_changed = phases_changed or change_phases(self, state)
                if phases_changed:
                    print(f"[CPU DEBUG] Phases changed at iteration {iteration}: num_phases={len(state.free_stable_compset_indices)}")
                if phases_changed:
                    # TODO: this preserves old logic about phase changes, but should we
                    # reset the counter `if phases_changed and not converged` -
                    # i.e. phases were changed by remove_and_consolidate_phases?
                    state.iterations_since_last_phase_change = 0
                else:
                    break
            state.iterations_since_last_phase_change += 1
            state.increment_phase_metastability_counters()
            if not phases_changed:
                advance_state(self, state, eq_soln, step_size)
            
            # DEBUG: Add detailed output after first iteration
            if iteration == 0:
                print(f"\n[CPU TRACE] ===== AFTER ITERATION 0 =====")
                print(f"[CPU TRACE] Chemical potentials: {np.array(state.chemical_potentials)}")
                print(f"[CPU TRACE] System amount: {state.system_amount:.15e}")
                print(f"[CPU TRACE] Mole fractions: {np.array(state.mole_fractions)}")
                print(f"[CPU TRACE] Mass residual: {state.mass_residual:.15e}")
                print(f"[CPU TRACE] Number of active phases: {len(state.free_stable_compset_indices)}")
                print(f"[CPU TRACE] Free stable indices: {state.free_stable_compset_indices}")
                
                for idx in state.free_stable_compset_indices:
                    compset = state.compsets[idx]
                    print(f"\n[CPU TRACE] Phase {idx} ({compset.phase_record.phase_name}):")
                    print(f"  NP (mole fraction): {compset.NP:.15e}")
                    print(f"  phase_amt (formula units): {state.phase_amt[idx]:.15e}")
                    print(f"  energy: {compset.energy:.15e}")
                    print(f"  phase_compositions: {state.phase_compositions[idx]}")
                    phase_comp_sum = np.sum(state.phase_compositions[idx])
                    print(f"  phase_comp_sum: {phase_comp_sum:.15e}")
                    num_sv = len(compset.phase_record.state_variables)
                    print(f"  Site fractions: {np.array(state.dof[idx][num_sv:])}")
                    print(f"  State variables: {np.array(state.dof[idx][:num_sv])}")
                    
                print(f"\n[CPU TRACE] Convergence status:")
                print(f"  converged: {converged}")
                print(f"  phases_changed: {phases_changed}")
                print(f"  largest_phase_amt_change: {state.largest_phase_amt_change[0]:.15e}")
                print(f"  largest_y_change: {state.largest_y_change[0]:.15e}")
                print(f"  largest_statevar_change: {state.largest_statevar_change[0]:.15e}")
                print(f"[CPU TRACE] ===== END ITERATION 0 =====\n")
                
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


cdef class SystemState:
    def __init__(self, SystemSpecification spec, list compsets):
        cdef CompositionSet compset
        cdef int idx, comp_idx
        self.compsets = compsets
        # Debug flag will be set later if needed
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
        self.mass_residual = 1e10
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
                self.metastable_phase_iterations, self.times_compset_removed, self.mass_residual,
                np.array(self.phase_amt), np.array(self.chemical_potentials), np.array(self.previous_chemical_potentials),
                np.array(self.delta_ms), np.array(self.phase_compositions), self.largest_chemical_potential_difference,
                self.largest_statevar_change[0], self.largest_phase_amt_change[0], self.largest_y_change[0],
                np.array(self.free_stable_compset_indices), self.system_amount, np.array(self.mole_fractions))
    def __setstate__(self, state):
        (self.compsets, self.cs_states, self.dof, self.iteration, self.iterations_since_last_phase_change,
         self.metastable_phase_iterations, self.times_compset_removed, self.mass_residual,
         self.phase_amt, self.chemical_potentials, self.previous_chemical_potentials,
         self.delta_ms, self.phase_compositions, self.largest_chemical_potential_difference, self.largest_statevar_change[0],
         self.largest_phase_amt_change[0], self.largest_y_change[0], self.free_stable_compset_indices, self.system_amount, self.mole_fractions) = state

    @cython.boundscheck(False)
    cpdef void recompute(self, SystemSpecification spec):
        from pycalphad.core.debug_output import debug_log
        
        # SEGMENT 22: STATE RECOMPUTE - GLOBAL QUANTITIES
        debug_log(22, f"State recompute - global quantities (iteration {self.iteration})")
        debug_log(f"  chemical_potentials: {np.array(self.chemical_potentials)}", True)
        debug_log(f"  system_amount: {self.system_amount:.15e}", True)
        debug_log(f"  mole_fractions: {np.array(self.mole_fractions)}", True)
        
        cdef int num_components = spec.num_components
        cdef CompositionSet compset
        cdef CompsetState csst
        cdef double[::1] x
        cdef int idx, comp_idx, cons_idx, i, j, stable_idx, fixed_idx, fixed_molefrac_cond_idx, num_phase_dof
        cdef double mu_c_sum
        cdef double phase_comp_sum
        self.mole_fractions[:] = 0
        self.delta_ms[:, :] = 0
        self.system_amount = 0
        
        debug_log(f"  zeroed_arrays", True)
        
        # DEBUG: Track total phase amounts through iterations
        cdef double phase_amt_sum = 0.0
        for idx in range(len(self.compsets)):
            phase_amt_sum += self.phase_amt[idx]
        print(f"[CPU MASS BALANCE] recompute() - iteration {self.iteration}: sum(phase_amt) = {phase_amt_sum:.15e}")
        
        # Compute normalized global quantities
        for idx in range(len(self.compsets)):
            x = self.dof[idx]
            compset = self.compsets[idx]
            csst = self.cs_states[idx]
            csst.masses[:,:] = 0
            for comp_idx in range(num_components):
                compset.phase_record.formulamole_obj(csst.masses[comp_idx, :], x, comp_idx)
                if self.phase_amt[idx] > 0:
                    self.mole_fractions[comp_idx] += self.phase_amt[idx] * csst.masses[comp_idx, 0]
                    self.system_amount += self.phase_amt[idx] * csst.masses[comp_idx, 0]
                self.phase_compositions[idx, comp_idx] = csst.masses[comp_idx, 0]
        
        debug_log(f"  system_amount: {self.system_amount:.15e}", True)
        
        # DEBUG: Show phase details before normalization
        debug_log(f"  num_phases_active: {len([idx for idx in range(len(self.compsets)) if self.phase_amt[idx] > 1e-10])}", True)
        for idx in range(len(self.compsets)):
            if self.phase_amt[idx] > 1e-10:
                compset = self.compsets[idx]
                debug_log(f"  phase_{idx}_{compset.phase_record.phase_name}: NP={self.phase_amt[idx]:.15e}, X={self.phase_compositions[idx]}", True)
        
        for comp_idx in range(self.mole_fractions.shape[0]):
            self.mole_fractions[comp_idx] /= self.system_amount
        
        debug_log(f"  mole_fractions: {np.array(self.mole_fractions)}", True)

        self.mass_residual = 0.0
        for fixed_molefrac_cond_idx in range(spec.prescribed_mole_fraction_rhs.shape[0]):
            # DEBUG: Print mass residual calculation details
            if self.iteration < 3:
                coef = spec.prescribed_mole_fraction_coefficients[fixed_molefrac_cond_idx,:]
                dot_product = np.dot(coef, self.mole_fractions)
                rhs = spec.prescribed_mole_fraction_rhs[fixed_molefrac_cond_idx]
                print(f"[CPU] Mass residual calc (iteration {self.iteration}):")
                print(f"  Coefficients: {coef}")
                print(f"  Mole fractions: {self.mole_fractions}")
                print(f"  Dot product: {dot_product:.15e}")
                print(f"  RHS: {rhs:.15e}")
                print(f"  Residual contribution: {abs(dot_product - rhs):.15e}")
            self.mass_residual += abs(np.dot(spec.prescribed_mole_fraction_coefficients[fixed_molefrac_cond_idx,:], self.mole_fractions) - spec.prescribed_mole_fraction_rhs[fixed_molefrac_cond_idx])
        
        debug_log(f"  mass_residual: {self.mass_residual:.15e}", True)

        # SEGMENT 23: STATE RECOMPUTE - PHASE QUANTITIES  
        debug_log(23, f"State recompute - phase quantities (iteration {self.iteration})")
        debug_log(f"  num_phases_active: {len([idx for idx in range(len(self.compsets)) if self.phase_amt[idx] > 1e-10])}", True)
        for idx in range(len(self.compsets)):
            if self.phase_amt[idx] > 1e-10:
                compset = self.compsets[idx]
                debug_log(f"  phase_{idx}_{compset.phase_record.phase_name}: NP={self.phase_amt[idx]:.15e}, X={self.phase_compositions[idx]}", True)
        
        for idx in range(len(self.compsets)):
            compset = self.compsets[idx]
            csst = self.cs_states[idx]
            # TODO: Use better dof storage
            # Calculate key phase quantities starting here
            x = self.dof[idx]
            phase_comp_sum = 0.0
            for comp_idx in range(num_components):
                phase_comp_sum += self.phase_compositions[idx, comp_idx]
            
            debug_log(f"  phase_{idx}_{compset.phase_record.phase_name}_comp_sum: {phase_comp_sum:.15e}", True)
            
            compset.update(x[spec.num_statevars:], self.phase_amt[idx] * phase_comp_sum, x[:spec.num_statevars])
            csst.energy = 0
            csst.mass_jac[:,:] = 0
            # Compute phase matrix (LHS of Eq. 41, Sundman 2015)
            csst.phase_matrix[:,:] = 0
            csst.internal_cons[:] = 0
            csst.hess[:,:] = 0
            csst.grad[:] = 0

            compset.phase_record.formulaobj(<double[:1]>&csst.energy, x)
            
            debug_log(f"  phase_{idx}_{compset.phase_record.phase_name}_energy: {csst.energy:.15e}", True)
            
            for comp_idx in range(num_components):
                compset.phase_record.formulamole_grad(csst.mass_jac[comp_idx, :], x, comp_idx)
            if state.iteration < 3:
                printf("[CPU FORMULAHESS INPUT] Phase %d iteration %d, DOF: ", idx, state.iteration);
                for i in range(5):
                    printf("%.15e ", x[i])
                printf("\n")
            compset.phase_record.formulahess(csst.hess, x)
            
            # DEBUG: Print CPU Hessian values after calculation
            if True:  # Always print for debugging
                print(f"[CPU HESSIAN] Phase {idx} ({compset.phase_record.phase_name}) Hessian after formulahess:")
                for i in range(spec.num_statevars, min(spec.num_statevars + 2, csst.hess.shape[0])):
                    print(f"  Row {i}: ", end="")
                    for j in range(spec.num_statevars, min(spec.num_statevars + 2, csst.hess.shape[1])):
                        print(f"{csst.hess[i,j]:e} ", end="")
                    print()
            
            compset.phase_record.formulagrad(csst.grad, x)
            compset.phase_record.internal_cons_func(csst.internal_cons, x)
            
            debug_log(f"  phase_{idx}_{compset.phase_record.phase_name}_internal_cons: {np.array(csst.internal_cons)}", True)

            # Compute phase matrix
            
            compute_phase_matrix(csst.phase_matrix, csst.hess, csst.cons_jac_tmp, csst.phase_local_jac_tmp, compset, spec.num_statevars, self.chemical_potentials, x)
            
            debug_log(f"  phase_matrix_shape: {csst.phase_matrix.shape}", True)
            
            # Invert phase matrix
            
            # Copy the phase matrix into the e matrix and invert the e matrix
            for i in range(csst.full_e_matrix.shape[0]):
                for j in range(csst.full_e_matrix.shape[1]):
                    csst.full_e_matrix[i,j] = csst.phase_matrix[i,j]
            invert_matrix(&csst.full_e_matrix[0,0], csst.full_e_matrix.shape[0], &csst.ipiv[0])
            
            debug_log(f"  matrix_inverted", True)

            # Compute C matrices
            
            num_phase_dof = compset.phase_record.phase_dof
            csst.c_G[:] = 0
            csst.c_statevars[:,:] = 0
            csst.c_component[:,:] = 0
            csst.moles_normalization = 0
            csst.moles_normalization_grad[:] = 0
            
            # Calculate c_G
            # DEBUG: Print c_G calculation details
            if idx < 2 and self.iteration < 2:
                print(f"[CPU c_G DEBUG] Phase {idx} calculation:")
                print(f"  gradient values: {np.asarray(csst.grad[spec.num_statevars:spec.num_statevars+num_phase_dof])}")
                print(f"  full_e_matrix diagonal: {[csst.full_e_matrix[i,i] for i in range(num_phase_dof)]}")
                print(f"  Before c_G calc, c_G = {np.asarray(csst.c_G)}")
            for i in range(num_phase_dof):
                for j in range(num_phase_dof):
                    if idx < 2 and self.iteration < 2 and i == 0:
                        print(f"    c_G[{i}] -= {csst.full_e_matrix[i, j]:.6e} * {csst.grad[spec.num_statevars+j]:.6e} = {csst.full_e_matrix[i, j] * csst.grad[spec.num_statevars+j]:.6e}")
                    csst.c_G[i] -= csst.full_e_matrix[i, j] * csst.grad[spec.num_statevars+j]
            if idx < 2 and self.iteration < 2:
                print(f"  After c_G calc, c_G = {np.asarray(csst.c_G[:num_phase_dof])}")
            
            debug_log(f"  c_G: {np.array(csst.c_G[:num_phase_dof])}", True)
            
            # Calculate c_statevars
            for i in range(num_phase_dof):
                for j in range(num_phase_dof):
                    for statevar_idx in range(spec.num_statevars):
                        csst.c_statevars[i, statevar_idx] -= csst.full_e_matrix[i, j] * csst.hess[spec.num_statevars + j, statevar_idx]
            
            debug_log(f"  c_statevars_shape: {csst.c_statevars.shape}", True)
            
            # Calculate c_component
            for comp_idx in range(num_components):
                for i in range(num_phase_dof):
                    for j in range(num_phase_dof):
                        csst.c_component[comp_idx, i] += csst.mass_jac[comp_idx, spec.num_statevars + j] * csst.full_e_matrix[i, j]
            
            debug_log(f"  c_component_shape: {csst.c_component.shape}", True)
            
            # Calculate delta_ms
            for comp_idx in range(num_components):
                for i in range(num_phase_dof):
                    mu_c_sum = 0
                    for j in range(self.chemical_potentials.shape[0]):
                        mu_c_sum += csst.c_component[j, i] * self.chemical_potentials[j]
                    self.delta_ms[idx, comp_idx] += csst.mass_jac[comp_idx, spec.num_statevars + i] * (mu_c_sum + csst.c_G[i])
            
            debug_log(f"  delta_ms_{idx}: {np.array(self.delta_ms[idx])}", True)
            
            # Calculate moles normalization
            for comp_idx in range(num_components):
                csst.moles_normalization += csst.masses[comp_idx, 0]
                for i in range(num_phase_dof+spec.num_statevars):
                    csst.moles_normalization_grad[i] += csst.mass_jac[comp_idx, i]
            
            debug_log(f"  moles_normalization: {csst.moles_normalization:.15e}", True)

    cdef double[::1] driving_forces(self):
        cdef int idx, comp_idx
        cdef CompositionSet compset
        cdef double[::1] x
        cdef int num_components = self.chemical_potentials.shape[0]
        # This needs to be done per mole of atoms, not per formula unit, since we compare phases to each other
        self._driving_forces[:] = 0
        for idx in range(len(self.compsets)):
            compset = self.compsets[idx]
            x = self.dof[idx]
            for comp_idx in range(num_components):
                compset.phase_record.mass_obj(self._phase_amounts_per_mole_atoms[idx, comp_idx, :], x, comp_idx)
                self._driving_forces[idx] += self.chemical_potentials[comp_idx] * self._phase_amounts_per_mole_atoms[idx, comp_idx, 0]
            compset.phase_record.obj(self._phase_energies_per_mole_atoms[idx, :], x)
            self._driving_forces[idx] -= self._phase_energies_per_mole_atoms[idx, 0]
        return self._driving_forces

    cdef void increment_phase_metastability_counters(self):
        cdef int idx
        for idx in range(len(self.compsets)):
            if idx in self.free_stable_compset_indices or self.compsets[idx].fixed:
                self.metastable_phase_iterations[idx] = 0
            else:
                self.metastable_phase_iterations[idx] += 1

cpdef construct_equilibrium_system(SystemSpecification spec, SystemState state, int num_reserved_rows) except *:
    from pycalphad.core.debug_output import debug_log
    
    import sys
    sys.stderr.write(f"[CPU MATRIX DEBUG] construct_equilibrium_system called, state.iteration={state.iteration}\n")
    sys.stderr.flush()
    
    cdef double[::1,:] equilibrium_matrix  # Fortran ordering required by call into lapack
    cdef double[::1] equilibrium_soln
    cdef int num_stable_phases, num_fixed_phases, num_fixed_mole_fraction_conditions, num_free_variables

    num_stable_phases = state.free_stable_compset_indices.shape[0]
    num_fixed_phases = spec.fixed_stable_compset_indices.shape[0]
    num_fixed_mole_fraction_conditions = spec.prescribed_mole_fraction_rhs.shape[0]
    num_free_variables = spec.free_chemical_potential_indices.shape[0] + num_stable_phases + \
                         spec.free_statevar_indices.shape[0]

    # SEGMENT 27: CONSTRUCT EQUILIBRIUM SYSTEM
    debug_log(27, f"Construct equilibrium system (iteration {state.iteration})")
    debug_log(f"  num_stable_phases: {num_stable_phases}", True)
    debug_log(f"  num_fixed_phases: {num_fixed_phases}", True)
    debug_log(f"  num_fixed_mole_fraction_conditions: {num_fixed_mole_fraction_conditions}", True)
    debug_log(f"  num_free_variables: {num_free_variables}", True)

    equilibrium_matrix = np.zeros((num_stable_phases + num_fixed_phases + num_fixed_mole_fraction_conditions + num_reserved_rows + 1,
                                   num_free_variables), order='F')
    equilibrium_rhs = np.zeros(equilibrium_matrix.shape[0])
    
    debug_log(f"  equilibrium_matrix_shape: {equilibrium_matrix.shape}", True)
    debug_log(f"  equilibrium_matrix_condition_number: {np.linalg.cond(np.array(equilibrium_matrix)) if equilibrium_matrix.shape[0] < 50 else 'too large':.15e}", True)
    
    if (equilibrium_matrix.shape[0] != equilibrium_matrix.shape[1]):
        raise ValueError('Conditions do not obey Gibbs Phase Rule')
    fill_equilibrium_system(equilibrium_matrix, equilibrium_rhs, spec, state)
    
    # DEBUG: Print equilibrium matrix for first few iterations
    if state.iteration < 3:
        print(f"\n[CPU MATRIX DEBUG] Equilibrium matrix at iteration {state.iteration} (rows={equilibrium_matrix.shape[0]}, cols={equilibrium_matrix.shape[1]}):")
        for i in range(min(equilibrium_matrix.shape[0], 5)):
            row_str = f"  Row {i}: "
            for j in range(min(equilibrium_matrix.shape[1], 5)):
                row_str += f"{equilibrium_matrix[i,j]:+.6e} "
            row_str += f"| RHS: {equilibrium_rhs[i]:+.6e}"
            print(row_str)
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
    from pycalphad.core.debug_output import debug_log
    cdef double[::1,:] equilibrium_matrix  # Fortran ordering required by call into lapack
    cdef double[::1] equilibrium_soln
    cdef int chempot_idx, comp_idx

    state.previous_chemical_potentials[:] = state.chemical_potentials[:]
    
    # DEBUG: Log state before recompute (CPU minimizer)
    # Note: verbose flag not directly available here, but we can check if state has debug attributes
    cdef bint debug_enabled = False  # Debug controlled globally via debug_output.py
    if debug_enabled:
        print(f"\n--- CPU Minimizer solve_state start ---")
        print(f"Previous chemical potentials: {np.array(state.previous_chemical_potentials)}")
        print(f"Current chemical potentials: {np.array(state.chemical_potentials)}")
        for i, compset in enumerate(state.compsets):
            if compset.NP > 1e-12:
                print(f"Active phase {i} ({compset.phase_record.phase_name}): amount={compset.NP:.6e}")
    
    state.recompute(spec)

    # Construct equilibrium system
    
    equilibrium_matrix, equilibrium_soln = construct_equilibrium_system(spec, state, 0)

    # System constructed
    
    # DEBUG: Log equilibrium system details
    if debug_enabled:
        print(f"Equilibrium matrix shape: {equilibrium_matrix.shape}")
        print(f"Matrix condition number estimate: {np.linalg.cond(np.array(equilibrium_matrix)) if equilibrium_matrix.shape[0] < 100 else 'skipped (large matrix)'}")

    # SEGMENT 31: SOLVE LINEAR SYSTEM
    debug_log(31, f"Solve linear system (iteration {state.iteration})")
    debug_log(f"  matrix_dimensions: {equilibrium_matrix.shape[0]}x{equilibrium_matrix.shape[1]}", True)
    debug_log(f"  rhs_norm_before: {np.linalg.norm(np.array(equilibrium_soln)):.15e}", True)
    
    lstsq(&equilibrium_matrix[0,0], equilibrium_matrix.shape[0], equilibrium_matrix.shape[1],
          &equilibrium_soln[0], 1e-16)
    
    debug_log(f"  solution_norm: {np.linalg.norm(np.array(equilibrium_soln)):.15e}", True)
    debug_log(f"  solution_first_3_values: {equilibrium_soln[0]:.15e}, {equilibrium_soln[1]:.15e}, {equilibrium_soln[2]:.15e}", True)

    # SEGMENT 32: UPDATE CHEMICAL POTENTIALS
    debug_log(32, f"Update chemical potentials (iteration {state.iteration})")
    
    # set the chemical potentials from the solution
    for i in range(spec.free_chemical_potential_indices.shape[0]):
        chempot_idx = spec.free_chemical_potential_indices[i]
        state.chemical_potentials[chempot_idx] = equilibrium_soln[i]
        debug_log(f"  free_mu_{chempot_idx}: {equilibrium_soln[i]:.15e}", debug_enabled)

    # Force some chemical potentials to adopt their fixed values
    for chempot_idx in range(spec.fixed_chemical_potential_indices.shape[0]):
        comp_idx = spec.fixed_chemical_potential_indices[chempot_idx]
        state.chemical_potentials[comp_idx] = spec.initial_chemical_potentials[comp_idx]
        debug_log(f"  fixed_mu_{comp_idx}: {spec.initial_chemical_potentials[comp_idx]:.15e}", debug_enabled)
    
    debug_log(f"  final_chemical_potentials: {np.array(state.chemical_potentials)}", debug_enabled)
    
    # DEBUG: Log updated chemical potentials
    if debug_enabled:
        print(f"Updated chemical potentials: {np.array(state.chemical_potentials)}")

    state.largest_chemical_potential_difference = -np.inf
    for comp_idx in range(spec.num_components):
        state.largest_chemical_potential_difference = max(state.largest_chemical_potential_difference, abs(state.chemical_potentials[comp_idx] - state.previous_chemical_potentials[comp_idx]))

    # DEBUG: Log convergence metrics
    if debug_enabled:
        print(f"Largest chemical potential change: {state.largest_chemical_potential_difference:.6e}")
        print(f"Mass residual: {state.mass_residual:.6e}")
        print(f"--- CPU Minimizer solve_state end ---\n")

    return equilibrium_soln


# TODO: should we store equilibrium_soln in the state(?)
cpdef advance_state(SystemSpecification spec, SystemState state, double[::1] equilibrium_soln, double step_size):
    from pycalphad.core.debug_output import debug_log
    
    # SEGMENT 37: ADVANCE STATE - NUMERICAL VALUES
    debug_log(37, f"Advance state (iteration {state.iteration})")
    
    # Apply linear corrections in phase amounts, state variables and site fractions
    cdef bint exceeded_bounds
    cdef double minimum_step_size, psc, phase_amt_step_size
    cdef int i, idx, cons_idx, compset_idx, statevar_idx, chempot_idx
    cdef int soln_index_offset = spec.free_chemical_potential_indices.shape[0]  # Chemical potentials handled after solving
    cdef double[::1] new_y, x
    cdef CompsetState csst

    cdef double MIN_PHASE_AMOUNT = 1e-16
    
    # DEBUG: Log advance_state start
    cdef bint debug_enabled = False  # Debug controlled globally via debug_output.py
    
    debug_log(f"  initial_step_size: {step_size:.15e}", True)
    debug_log(f"  equilibrium_soln_norm: {np.linalg.norm(np.array(equilibrium_soln)):.15e}", True)
    
    if debug_enabled:
        print(f"\n--- CPU Minimizer advance_state start ---")
        print(f"Initial step size: {step_size:.6e}")
        for i, compset in enumerate(state.compsets):
            if compset.NP > 1e-12:
                print(f"Pre-advance phase {i} ({compset.phase_record.phase_name}): amount={compset.NP:.6e}")

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
    # Update the phase amounts using the largest allowable step size
    debug_log(f"  phase_amt_step_size: {phase_amt_step_size:.15e}", True)
    
    state.largest_phase_amt_change[0] = 0
    for i in range(state.free_stable_compset_indices.shape[0]):
        compset_idx = state.free_stable_compset_indices[i]
        old_amt = state.phase_amt[compset_idx]
        delta = equilibrium_soln[soln_index_offset + i]
        print(f"[CPU ADVANCE] Phase {compset_idx}: old={old_amt:.6e}, delta={delta:.6e}, step_size={phase_amt_step_size:.6e}, actual_change={phase_amt_step_size * delta:.6e}")
        state.phase_amt[compset_idx] += phase_amt_step_size * equilibrium_soln[soln_index_offset + i]
        state.largest_phase_amt_change[0] = max(state.largest_phase_amt_change[0], abs(phase_amt_step_size * equilibrium_soln[soln_index_offset + i]))
        
        # DEBUG: Log phase amount changes
        if debug_enabled and abs(old_amt - state.phase_amt[compset_idx]) > 1e-12:
            print(f"Phase amount change {compset_idx}: {old_amt:.6e} -> {state.phase_amt[compset_idx]:.6e} (delta: {state.phase_amt[compset_idx] - old_amt:.6e})")
    soln_index_offset += state.free_stable_compset_indices.shape[0]
    
    # DEBUG: Check total phase amounts after update
    cdef double phase_amt_sum_after = 0.0
    for idx in range(len(state.compsets)):
        phase_amt_sum_after += state.phase_amt[idx]
    print(f"[CPU MASS BALANCE] advance_state() - after phase update: sum(phase_amt) = {phase_amt_sum_after:.15e}")

    # 2. Step in state variables
    debug_log(f"  largest_phase_amt_change: {state.largest_phase_amt_change[0]:.15e}", True)
    
    state.largest_statevar_change[0] = 0
    state.delta_statevars[:] = 0
    for i in range(spec.free_statevar_indices.shape[0]):
        statevar_idx = spec.free_statevar_indices[i]
        state.delta_statevars[statevar_idx] = equilibrium_soln[soln_index_offset + i]
        if state.dof[0][statevar_idx] == 0:
            psc = np.inf
        else:
            psc = abs(state.delta_statevars[statevar_idx] / state.dof[0][statevar_idx])
        state.largest_statevar_change[0] = max(state.largest_statevar_change[0], psc)
    # Update state variables in the `x` array
    for idx in range(len(state.compsets)):
        x = state.dof[idx]
        for statevar_idx in range(state.delta_statevars.shape[0]):
            x[statevar_idx] += state.delta_statevars[statevar_idx]
        # We need real state variable bounds support

    # 3. Step in phase internal degrees of freedom
    for idx in range(len(state.compsets)):
        # TODO: Use better dof storage
        x = state.dof[idx]
        csst = state.cs_states[idx]

        # Construct delta_y from Eq. 43 in Sundman 2015
        csst.delta_y[:] = 0
        
        # DEBUG: Print delta_y calculation at iteration 1 for single phase
        if state.iteration == 1 and len(state.free_stable_compset_indices) == 1 and idx == 0:
            print(f"\n[CPU DELTA_Y DEBUG] Iteration 1, Phase {idx}:")
            print(f"  c_G values: {np.asarray(csst.c_G)}")
            print(f"  Chemical potentials: {np.asarray(state.chemical_potentials)}")

        for i in range(csst.delta_y.shape[0]):
            csst.delta_y[i] += csst.c_G[i]
            for statevar_idx in range(state.delta_statevars.shape[0]):
                csst.delta_y[i] += csst.c_statevars[i, statevar_idx] * state.delta_statevars[statevar_idx]
            for chempot_idx in range(state.chemical_potentials.shape[0]):
                csst.delta_y[i] += csst.c_component[chempot_idx, i] * state.chemical_potentials[chempot_idx]
            for cons_idx in range(csst.internal_cons.shape[0]):
                csst.delta_y[i] -= csst.full_e_matrix[csst.delta_y.shape[0] + cons_idx, i] * csst.internal_cons[cons_idx]
        
        # DEBUG: Print final delta_y at iteration 1
        if state.iteration == 1 and len(state.free_stable_compset_indices) == 1 and idx == 0:
            print(f"  Calculated delta_y: {np.asarray(csst.delta_y)}")

        new_y = np.array(x)
        minimum_step_size = 1e-20 * step_size
        while step_size >= minimum_step_size:
            exceeded_bounds = False
            for i in range(spec.num_statevars, new_y.shape[0]):
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
        state.largest_y_change[0] = 0.0
        for i in range(spec.num_statevars, new_y.shape[0]):
            state.largest_y_change[0] = max(state.largest_y_change[0], abs(x[i] - new_y[i]))
        
        # DEBUG: Show site fraction update at iteration 1
        if state.iteration == 1 and len(state.free_stable_compset_indices) == 1 and idx == 0:
            print(f"  Step size used: {step_size}")
            print(f"  Site fractions before update: {x[spec.num_statevars:]}")
            print(f"  Site fractions after update: {new_y[spec.num_statevars:]}")
            print(f"  Largest y change: {state.largest_y_change[0]}")
            
        x[:] = new_y
        
        # DEBUG: Log site fraction changes
        if debug_enabled and state.largest_y_change[0] > 1e-12:
            print(f"Phase {idx} largest site fraction change: {state.largest_y_change[0]:.6e}")
    
    # DEBUG: Log advance_state end
    debug_log(f"  final_largest_phase_amt_change: {state.largest_phase_amt_change[0]:.15e}", True)
    debug_log(f"  final_largest_y_change: {state.largest_y_change[0]:.15e}", True)  
    debug_log(f"  final_largest_statevar_change: {state.largest_statevar_change[0]:.15e}", True)
    
    if debug_enabled:
        print(f"Final largest phase amount change: {state.largest_phase_amt_change[0]:.6e}")
        print(f"Final largest y change: {state.largest_y_change[0]:.6e}")
        print(f"Final largest statevar change: {state.largest_statevar_change[0]:.6e}")
        print(f"--- CPU Minimizer advance_state end ---\n")


cdef bint remove_and_consolidate_phases(SystemSpecification spec, SystemState state):
    """Remove phases that have become unstable (phase amount <= 0) and consolidate composition sets in an artificial misicbility gap.

    Updates the state in place.
    """
    from pycalphad.core.debug_output import debug_log
    cdef int i, j, idx, idx2, cp_idx, comp_idx, dof_idx, phase_idx
    cdef CompositionSet compset, compset2
    cdef bint phases_changed = False
    cdef double composition_difference
    cdef double COMPSET_CONSOLIDATE_DISTANCE = 1e-4
    cdef bint debug_enabled = False  # Debug controlled globally via debug_output.py

    # SEGMENT 34: REMOVE AND CONSOLIDATE PHASES
    debug_log(34, f"Remove and consolidate phases (iteration {state.iteration})")
    debug_log(f"  initial_phase_count: {len(state.free_stable_compset_indices)}", True)
    
    compset_indices_to_remove = set()
    for i in range(len(state.free_stable_compset_indices)):
        idx = state.free_stable_compset_indices[i]
        compset = state.compsets[idx]
        if compset.fixed:
            continue
        if idx in compset_indices_to_remove:
            continue
        # Remove unstable phases
        if state.phase_amt[idx] < 1e-10:
            compset_indices_to_remove.add(idx)
            state.phase_amt[idx] = 0
            debug_log(f"  removing_phase_{idx}_{compset.phase_record.phase_name}: amount={state.phase_amt[idx]:.15e}", True)
            continue
        for j in range(len(state.free_stable_compset_indices)):
            idx2 = state.free_stable_compset_indices[j]
            compset2 = state.compsets[idx2]
            if idx == idx2:
                continue
            if compset2.fixed:
                continue
            if compset.phase_record.phase_name != compset2.phase_record.phase_name:
                continue
            if idx2 in compset_indices_to_remove:
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
                compset_indices_to_remove.add(idx2)
                debug_log(f"  consolidating_phases_{idx}_{idx2}: {compset.phase_record.phase_name}", True)
                print(f"[CPU DEBUG] CONSOLIDATING phases {idx} and {idx2}: max_diff={max(abs(state.phase_compositions[idx, comp_idx] - state.phase_compositions[idx2, comp_idx]) for comp_idx in range(spec.num_components)):.6f}")
                if idx not in spec.fixed_stable_compset_indices:
                    # ensure that the consolidated phase is stable
                    state.phase_amt[idx] = max(state.phase_amt[idx] + state.phase_amt[idx2], 1e-8)
                state.phase_amt[idx2] = 0
    
    debug_log(f"  phases_to_remove: {len(compset_indices_to_remove)}", True)
    
    if len(compset_indices_to_remove) > 0:
        if len(compset_indices_to_remove) - len(state.free_stable_compset_indices) == 0:
            # Do not allow all phases to leave the system
            for phase_idx in state.free_stable_compset_indices:
                state.phase_amt[phase_idx] = 1
            state.chemical_potentials[:] = 0
            # Force some chemical potentials to adopt their fixed values
            for cp_idx in range(spec.fixed_chemical_potential_indices.shape[0]):
                comp_idx = spec.fixed_chemical_potential_indices[cp_idx]
                state.chemical_potentials[comp_idx] = spec.initial_chemical_potentials[comp_idx]
        else:
            state.free_stable_compset_indices = np.array(sorted(set(state.free_stable_compset_indices) - compset_indices_to_remove), dtype=np.int32)
            phases_changed = True
    return phases_changed

cdef bint change_phases(SystemSpecification spec, SystemState state):
    from pycalphad.core.debug_output import debug_log
    cdef bint debug_enabled = False  # Debug controlled globally via debug_output.py
    
    # SEGMENT 36: CHANGE PHASES  
    debug_log(36, f"Change phases (iteration {state.iteration})")
    # Calculate driving forces for all phases
    cdef double[::1] driving_forces = state.driving_forces()
    debug_log(f"  driving_forces: {np.array(driving_forces)}", True)
    
    cdef int idx, i, cs_idx, least_removed_cs_idx, smallest_df_cs_idx
    # Already calculated above
    cdef double MIN_PHASE_AMOUNT = 1e-9
    cdef int MIN_REQUIRED_METASTABLE_PHASE_ITERATIONS_TO_ADD = 5
    cdef double MIN_DRIVING_FORCE_TO_ADD = 1e-5
    cdef int MAX_ALLOWED_TIMES_COMPSET_REMOVED = 4
    
    debug_log(f"  current_phase_count: {state.free_stable_compset_indices.shape[0]}", True)
    debug_log(f"  max_allowed_phases: {spec.max_num_free_stable_phases}", True)
    
    if state.free_stable_compset_indices.shape[0] > spec.max_num_free_stable_phases:
        # Gibbs phase rule is currently being violated
        # Try forcing phases with small amounts out of the equilibrium
        MIN_PHASE_AMOUNT = 1e-4
        debug_log(f"  gibbs_phase_rule_violation: increasing MIN_PHASE_AMOUNT to {MIN_PHASE_AMOUNT}", True)
    
    phase_amt = state.phase_amt
    current_free_stable_compset_indices = state.free_stable_compset_indices
    compsets_to_remove = set()
    for i in range(current_free_stable_compset_indices.shape[0]):
        cs_idx = current_free_stable_compset_indices[i]
        if phase_amt[cs_idx] < MIN_PHASE_AMOUNT:
            compsets_to_remove.add(cs_idx)
            debug_log(f"  removing_phase_{cs_idx}: amount={phase_amt[cs_idx]:.15e}, driving_force={driving_forces[cs_idx]:.15e}", True)
            print(f"[CPU DEBUG] REMOVING phase {cs_idx}: amount={phase_amt[cs_idx]:.6e} < {MIN_PHASE_AMOUNT:.6e}")

    # Only add phases with positive driving force which have been metastable for at least 5 iterations, which have been removed fewer than 4 times
    compsets_to_add = set()
    for cs_idx in range(state.metastable_phase_iterations.shape[0]):
        should_add_compset = (
            (state.metastable_phase_iterations[cs_idx] >= MIN_REQUIRED_METASTABLE_PHASE_ITERATIONS_TO_ADD)
            and (driving_forces[cs_idx] > MIN_DRIVING_FORCE_TO_ADD)
            and (state.times_compset_removed[cs_idx] < MAX_ALLOWED_TIMES_COMPSET_REMOVED)
        )
        if should_add_compset:
            compsets_to_add.add(cs_idx)
            debug_log(f"  candidate_phase_{cs_idx}: df={driving_forces[cs_idx]:.15e}, metastable_iters={state.metastable_phase_iterations[cs_idx]}", True)
    
    # Finally, remove all currently stable compsets as candidates
    compsets_to_add -= set(current_free_stable_compset_indices)
    max_allowed_to_add = spec.max_num_free_stable_phases + len(compsets_to_remove) - len(current_free_stable_compset_indices)
    
    debug_log(f"  phases_to_add_count: {len(compsets_to_add)}", True)
    debug_log(f"  max_allowed_to_add: {max_allowed_to_add}", True)
    # We must obey the Gibbs phase rule
    if len(compsets_to_add) > 0:
        if max_allowed_to_add < 1:
            # We are at the maximum number of allowed phases, yet there is still positive driving force
            # Destabilize one phase and add only one phase
            possible_phases_to_destabilize = sorted(set(current_free_stable_compset_indices) - compsets_to_add - compsets_to_remove)
            # Destabilize the one that has been removed the least
            least_removed_cs_idx = possible_phases_to_destabilize[0]
            for i in range(1, len(possible_phases_to_destabilize)):
                cs_idx = possible_phases_to_destabilize[i]
                if state.times_compset_removed[cs_idx] < state.times_compset_removed[least_removed_cs_idx]:
                    least_removed_cs_idx = cs_idx
            compsets_to_remove.add(least_removed_cs_idx)
            phase_amt[least_removed_cs_idx] = 0
        # Add the compset with least amount (but still positive) driving force
        possible_phases_to_add = sorted(compsets_to_add)
        smallest_df_cs_idx = possible_phases_to_add[0]
        for i in range(1, len(possible_phases_to_add)):
            cs_idx = possible_phases_to_add[i]
            if driving_forces[cs_idx] < driving_forces[smallest_df_cs_idx]:
                smallest_df_cs_idx = cs_idx
        compsets_to_add = {smallest_df_cs_idx}
    new_free_stable_compset_indices = np.array(sorted((set(current_free_stable_compset_indices) - compsets_to_remove)
                                                      | compsets_to_add
                                                      ),
                                               dtype=np.int32)
    removed_compset_indices = set(current_free_stable_compset_indices) - set(new_free_stable_compset_indices)
    for idx in removed_compset_indices:
        state.times_compset_removed[idx] += 1
    for idx in range(len(state.compsets)):
        if idx in new_free_stable_compset_indices:
            # Force some amount of newly stable phases
            if state.phase_amt[idx] < 1e-10:
                state.phase_amt[idx] = 1e-10
        # Force unstable phase amounts to zero
        else:
            state.phase_amt[idx] = 0
    state.free_stable_compset_indices = new_free_stable_compset_indices
    if set(current_free_stable_compset_indices) == set(new_free_stable_compset_indices):
        # feasible system, and no phases to add or remove
        phases_changed = False
    else:
        phases_changed = True
    
    debug_log(f"  phases_changed: {phases_changed}", True)
    debug_log(f"  final_phase_count: {len(new_free_stable_compset_indices)}", True)
    
    return phases_changed
