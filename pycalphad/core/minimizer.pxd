cdef class SystemState:
    cdef public list compsets
    cdef public list cs_states
    cdef object dof
    cdef public int iteration
    cdef int num_statevars, iterations_since_last_phase_change
    cdef int[::1] metastable_phase_iterations
    cdef int[::1] times_compset_removed
    cdef double largest_chemical_potential_difference
    cdef double[::1] phase_amt, previous_chemical_potentials, delta_statevars
    cdef public double[::1] chemical_potentials
    cdef double[:, ::1] phase_compositions, delta_ms
    cdef double[1] largest_statevar_change, largest_phase_amt_change, largest_y_change
    cdef int[::1] free_stable_compset_indices
    cdef double system_amount
    cdef double[::1] mole_fractions
    cdef double[::1] _driving_forces
    cdef double[:, ::1] _phase_energies_per_mole_atoms
    cdef double[:, :, ::1] _phase_amounts_per_mole_atoms
    # Bound by _prepare, for the solver functions that run without the GIL
    cdef int num_compsets
    cdef object _cs_state_refs
    cdef void** _cs_state_ptrs
    cdef int[::1] _phase_ids
    cdef int[::1] _free_stable_buffer, _new_free_stable_buffer, _remove_flags, _add_flags
    cdef double[::1] _equilibrium_matrix, _equilibrium_soln
    cdef bint _buffers_ready
    cdef bint _prepare(self, SystemSpecification spec) except -1
    cpdef void recompute(self, SystemSpecification spec)
    cdef int _recompute(self, SystemSpecification spec) except -1 nogil
    cdef int compute_driving_forces(self) except -1 nogil
    cdef void increment_phase_metastability_counters(self) noexcept nogil

cdef class SystemSpecification:
    cdef int num_statevars, num_components, max_num_free_stable_phases
    cdef double prescribed_system_amount
    cdef int[::1] determined_component_indices, undetermined_condition_row_indices
    cdef double[::1] determined_component_mole_fractions
    cdef double[::1] initial_chemical_potentials, prescribed_mole_fraction_rhs
    cdef double[:,::1] prescribed_mole_fraction_coefficients
    cdef int[::1] free_chemical_potential_indices, free_statevar_indices
    cdef int[::1] fixed_chemical_potential_indices, fixed_statevar_indices, fixed_stable_compset_indices
    cdef bint debugging_output
    cpdef bint check_convergence(self, SystemState state)
    cdef int _check_convergence(self, SystemState state) except -1 nogil
    cpdef bint pre_solve_hook(self, SystemState state)
    cpdef bint post_solve_hook(self, SystemState state)
    cpdef bint run_loop(self, SystemState state, int max_iterations)
    cdef int _run_loop(self, SystemState state, int max_iterations, bint use_overrides) except -1 nogil
    cpdef SystemState get_new_state(self, list compsets)