# distutils: language = c++
cimport numpy as np
import numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport fabs, fmin, fmax, INFINITY
cimport scipy.linalg.cython_lapack as cython_lapack

# Points are pruned in blocks of this many consecutive points (see find_lowest_driving_force)
cdef enum:
    BLOCK_SIZE = 32
# Block states
cdef enum:
    BLOCK_UNSEEN = 0     # box not computed yet
    BLOCK_BOXED = 1      # box computed, no reference potentials yet
    BLOCK_BOUNDED = 2    # box and reference potentials available: the block can be skipped
    BLOCK_UNBOUNDED = 3  # non-finite or huge values: always evaluated
# Bounds are only used when all magnitudes are below this, so the bound arithmetic cannot overflow
cdef double BOUND_MAGNITUDE_LIMIT = 1e100
# Relative safety margin on bounds; many orders of magnitude above the rounding error of a driving force
cdef double BOUND_ROUNDING_MARGIN = 1e-9


@cython.boundscheck(False)
cdef void solve(double* A, int N, double* x, int* ipiv) nogil:
    cdef int i
    cdef int info = 0
    cdef int NRHS = 1
    cython_lapack.dgesv(&N, &NRHS, A, &N, ipiv, x, &N, &info)
    # Special for our case: singular matrix results get set to a special value
    if info != 0:
        for i in range(N):
            x[i] = -1e19


@cython.boundscheck(False)
cdef void prodsum(double[::1] chempots, double[:,::1] points, double[::1] result) nogil:
    for i in range(chempots.shape[0]):
        for j in range(result.shape[0]):
            result[j] -= chempots[i]*points[j,i]


@cython.boundscheck(False)
cdef double _min(double* a, int a_shape) nogil:
    cdef int i
    cdef double result = 1e300
    for i in range(a_shape):
        if a[i] < result:
            result = a[i]
    return result

@cython.boundscheck(False)
cdef int argmin(double* a, int a_shape, double* lowest) nogil:
    cdef int i
    cdef int result = 0
    for i in range(a_shape):
        if a[i] < lowest[0]:
            lowest[0] = a[i]
            result = i
    return result


@cython.boundscheck(False)
cdef int argmax(double* a, int a_shape) nogil:
    cdef int i
    cdef int result = 0
    cdef double highest = -1e30
    for i in range(a_shape):
        if a[i] > highest:
            highest = a[i]
            result = i
    return result

@cython.boundscheck(False)
cpdef void hyperplane_coefficients(double[:,::1] compositions,
                                   size_t[::1] fixed_chempot_indices,
                                   int[::1] trial_simplex,
                                   double[::1] out_plane_coefs) except * nogil:
    cdef int i, j
    cdef int plane_rows = trial_simplex.shape[0] + fixed_chempot_indices.shape[0]
    if plane_rows != compositions.shape[1]:
        raise ValueError('Hyperplane coefficient matrix is not square')
    cdef double* f_plane_matrix = <double*>malloc(plane_rows * compositions.shape[1] * sizeof(double))
    cdef int* int_tmp = <int*>malloc(plane_rows * sizeof(int))
    for i in range(trial_simplex.shape[0]):
        for j in range(compositions.shape[1]):
            f_plane_matrix[i + j*plane_rows] = compositions[trial_simplex[i], j]
        out_plane_coefs[i] = 1
    for i in range(fixed_chempot_indices.shape[0]):
        for j in range(compositions.shape[1]):
            f_plane_matrix[i + trial_simplex.shape[0] + j*plane_rows] = 0
        f_plane_matrix[i + trial_simplex.shape[0] + fixed_chempot_indices[i]*plane_rows] = 1
        out_plane_coefs[i + trial_simplex.shape[0]] = 0
    solve(f_plane_matrix, plane_rows, &out_plane_coefs[0], int_tmp)
    free(f_plane_matrix)
    free(int_tmp)

@cython.boundscheck(False)
cpdef void intersecting_point(double[:,::1] compositions,
                              size_t[::1] fixed_chempot_indices,
                              int[::1] trial_simplex,
                              double[:,::1] fixed_lincomb_molefrac_coefs,
                              double[::1] fixed_lincomb_molefrac_rhs,
                              double[::1] out_intersecting_point) except * nogil:
    cdef int i, j
    if trial_simplex.shape[0] == 1:
        # Simplex is zero-dimensional, so there is no intersection; just return the point defining the 0-simplex
        for i in range(compositions.shape[1]):
            out_intersecting_point[i] = compositions[trial_simplex[0], i]
        return
    if (fixed_lincomb_molefrac_rhs.shape[0] + 1 != compositions.shape[1]) and fixed_chempot_indices.shape[0] > 0:
        raise ValueError('Constraint matrix is not square')
    cdef int* int_tmp = <int*>malloc(compositions.shape[1] * sizeof(int))
    cdef double* constraint_matrix = <double*>malloc((fixed_lincomb_molefrac_rhs.shape[0] + 1) * compositions.shape[1] * sizeof(double))
    cdef double* constraint_rhs = <double*>malloc((fixed_lincomb_molefrac_rhs.shape[0] + 1) * sizeof(double))
    out_intersecting_point[:] = 0
    cdef double[::1] plane_coefs = out_intersecting_point
    hyperplane_coefficients(compositions, fixed_chempot_indices, trial_simplex, plane_coefs)
    for j in range(compositions.shape[1]):
        for i in range(fixed_lincomb_molefrac_rhs.shape[0]):
            constraint_matrix[i + j*compositions.shape[1]] = fixed_lincomb_molefrac_coefs[i, j]
            constraint_rhs[i] = fixed_lincomb_molefrac_rhs[i]
        constraint_matrix[fixed_lincomb_molefrac_rhs.shape[0] + j*compositions.shape[1]] = plane_coefs[j]
        constraint_rhs[fixed_lincomb_molefrac_rhs.shape[0]] = 1
    solve(constraint_matrix, compositions.shape[1], constraint_rhs, int_tmp)
    for i in range(compositions.shape[1]):
        out_intersecting_point[i] = constraint_rhs[i]
    free(int_tmp)
    free(constraint_matrix)
    free(constraint_rhs)

@cython.boundscheck(False)
cdef void simplex_fractions(double[:,::1] compositions,
                             size_t[::1] fixed_chempot_indices,
                             int[::1] trial_simplex,
                             double[:,::1] fixed_lincomb_molefrac_coefs,
                             double[::1] fixed_lincomb_molefrac_rhs,
                             double* out_fractions) except *:
    cdef int simplex_size = trial_simplex.shape[0]
    # Note that compositions.shape[1] = simplex_size + fixed_chempot_indices.shape[0], by construction
    cdef int i, j
    cdef double* f_coord_matrix = <double*>malloc(simplex_size * simplex_size * sizeof(double))
    cdef double* target_point = <double*>malloc(compositions.shape[1] * sizeof(double))
    cdef int* int_tmp = <int*>malloc(simplex_size * sizeof(int))
    cdef size_t[::1] free_chempot_indices = np.array(list(set(range(compositions.shape[1])) - set(fixed_chempot_indices)), dtype=np.uintp)
    # Get target point for calculation
    intersecting_point(compositions, fixed_chempot_indices, trial_simplex,
                       fixed_lincomb_molefrac_coefs, fixed_lincomb_molefrac_rhs,
                       <double[:compositions.shape[1]]>target_point)
    # Fill coordinate matrix
    for j in range(simplex_size):
        for i in range(simplex_size):
            f_coord_matrix[j + simplex_size*i] = compositions[trial_simplex[i], free_chempot_indices[j]]
        out_fractions[j] = target_point[free_chempot_indices[j]]
    solve(f_coord_matrix, simplex_size, &out_fractions[0], int_tmp)
    free(f_coord_matrix)
    free(target_point)
    free(int_tmp)


cdef inline double driving_force_of(double energy, const double* composition, const double* potentials,
                                    Py_ssize_t num_components) noexcept nogil:
    cdef Py_ssize_t comp_idx
    cdef double driving_force = energy
    for comp_idx in range(num_components):
        driving_force -= potentials[comp_idx] * composition[comp_idx]
    return driving_force


cdef bint compute_block_box(const double* compositions, const double* energies, Py_ssize_t num_components,
                            Py_ssize_t block_start, Py_ssize_t block_stop, double* block_record) noexcept nogil:
    """
    Fill the composition bounding box (lower, upper), the largest absolute energy and the sum over components
    of the largest absolute composition of the points [block_start, block_stop) into block_record.
    Returns False if any value is non-finite or at least BOUND_MAGNITUDE_LIMIT.
    """
    cdef Py_ssize_t idx, comp_idx
    cdef double value, lower, upper
    cdef double energy_scale = 0
    cdef double composition_scale = 0
    cdef bint bounded = True
    for idx in range(block_start, block_stop):
        bounded &= fabs(energies[idx]) < BOUND_MAGNITUDE_LIMIT
        energy_scale = fmax(energy_scale, fabs(energies[idx]))
    for comp_idx in range(num_components):
        lower = compositions[block_start*num_components + comp_idx]
        upper = lower
        for idx in range(block_start, block_stop):
            value = compositions[idx*num_components + comp_idx]
            bounded &= fabs(value) < BOUND_MAGNITUDE_LIMIT
            lower = fmin(lower, value)
            upper = fmax(upper, value)
        block_record[comp_idx] = lower
        block_record[num_components + comp_idx] = upper
        composition_scale += fmax(fabs(lower), fabs(upper))
    block_record[3*num_components + 2] = energy_scale
    block_record[3*num_components + 3] = composition_scale
    return bounded


cdef Py_ssize_t find_lowest_driving_force(const double* compositions, const double* energies,
                                          Py_ssize_t num_points, Py_ssize_t num_components,
                                          const double* potentials, const int* simplex, Py_ssize_t simplex_size,
                                          int* block_state, double* block_data, double* block_bounds,
                                          double* out_lowest_df) noexcept nogil:
    """
    Index of the point with the lowest driving force below 1e10 for the given potentials (the first such index
    on ties, -1 if there is none), as a plain scan over all points would find it. The driving force is written
    to out_lowest_df (1e10 if there is no such point).

    Points are grouped in blocks of BLOCK_SIZE consecutive points. Driving forces are linear in the potentials,
    so for a point x in a block whose driving forces were evaluated at reference potentials mu_ref,
        df(x, mu) = df(x, mu_ref) - (mu - mu_ref).x >= min_block df(mu_ref) - max_(x in block box) (mu - mu_ref).x
    A block whose bound, less a margin for rounding, is above the lowest driving force found so far cannot
    contain the minimizer, so it is skipped. Evaluated blocks take the current potentials as their reference.
    block_state (one per block) and block_data (3*num_components + 4 per block) carry this between calls;
    block_bounds (one per block) is scratch space.
    """
    cdef Py_ssize_t i, idx, comp_idx, block_idx, block_start, block_stop, block_min_idx
    cdef Py_ssize_t record_size = 3*num_components + 4
    cdef Py_ssize_t num_blocks = (num_points + BLOCK_SIZE - 1) // BLOCK_SIZE
    cdef Py_ssize_t min_idx = -1
    cdef double lowest_df = 1e10
    cdef double driving_force, block_lowest_df, bound, delta, lower_term, upper_term
    cdef double potential_scale = 0
    cdef bint potentials_bounded = True
    cdef double* block_record
    # Driving forces of the current simplex are near zero; starting from them lets more blocks be skipped
    for i in range(simplex_size):
        idx = simplex[i]
        if idx < 0 or idx >= num_points:
            continue
        driving_force = driving_force_of(energies[idx], &compositions[idx*num_components], potentials, num_components)
        if driving_force < lowest_df or (driving_force == lowest_df and idx < min_idx):
            lowest_df = driving_force
            min_idx = idx
    for comp_idx in range(num_components):
        if not (fabs(potentials[comp_idx]) < BOUND_MAGNITUDE_LIMIT):
            potentials_bounded = False
        elif fabs(potentials[comp_idx]) > potential_scale:
            potential_scale = fabs(potentials[comp_idx])
    # Lower bounds on the driving forces in each block
    for block_idx in range(num_blocks):
        block_record = &block_data[block_idx*record_size]
        if block_state[block_idx] == BLOCK_BOUNDED and potentials_bounded:
            # block_record: lower[N], upper[N], reference potentials[N],
            #               min df at reference minus fixed margin, margin per unit potential scale, scales[2]
            bound = block_record[3*num_components] - block_record[3*num_components + 1] * potential_scale
            for comp_idx in range(num_components):
                delta = potentials[comp_idx] - block_record[2*num_components + comp_idx]
                lower_term = delta * block_record[comp_idx]
                upper_term = delta * block_record[num_components + comp_idx]
                bound -= lower_term if lower_term > upper_term else upper_term
            block_bounds[block_idx] = bound
        else:
            block_bounds[block_idx] = -INFINITY
    for block_idx in range(num_blocks):
        if block_bounds[block_idx] > lowest_df:
            continue
        block_start = block_idx * BLOCK_SIZE
        block_stop = min(block_start + BLOCK_SIZE, num_points)
        block_record = &block_data[block_idx*record_size]
        if block_state[block_idx] == BLOCK_UNSEEN:
            if compute_block_box(compositions, energies, num_components, block_start, block_stop, block_record):
                block_state[block_idx] = BLOCK_BOXED
            else:
                block_state[block_idx] = BLOCK_UNBOUNDED
        block_lowest_df = INFINITY
        block_min_idx = -1
        for idx in range(block_start, block_stop):
            driving_force = driving_force_of(energies[idx], &compositions[idx*num_components], potentials, num_components)
            if driving_force < block_lowest_df:
                block_lowest_df = driving_force
                block_min_idx = idx
        if block_lowest_df < lowest_df or (block_lowest_df == lowest_df and block_min_idx < min_idx):
            lowest_df = block_lowest_df
            min_idx = block_min_idx
        if potentials_bounded and block_state[block_idx] != BLOCK_UNBOUNDED:
            # Rounding margin: 2*margin*(energy scale + composition scale * (reference + current potential scale))
            block_state[block_idx] = BLOCK_BOUNDED
            block_record[3*num_components] = block_lowest_df - 2 * BOUND_ROUNDING_MARGIN * (
                block_record[3*num_components + 2] + block_record[3*num_components + 3] * potential_scale)
            block_record[3*num_components + 1] = 2 * BOUND_ROUNDING_MARGIN * block_record[3*num_components + 3]
            for comp_idx in range(num_components):
                block_record[2*num_components + comp_idx] = potentials[comp_idx]
    out_lowest_df[0] = lowest_df
    return min_idx


@cython.boundscheck(False)
cpdef double hyperplane(double[:,::1] compositions,
                        double[::1] energies,
                        double[::1] chemical_potentials,
                        size_t[::1] fixed_chempot_indices,
                        double[:, ::1] fixed_lincomb_molefrac_coefs,
                        double[::1] fixed_lincomb_molefrac_rhs,
                        double[::1] result_fractions,
                        int[::1] result_simplex) except *:
    """
    Find chemical potentials which approximate the tangent hyperplane
    at the given composition.

    Parameters
    ----------
    compositions : ndarray
        A sample of the energy surface of the system.
        Aligns with 'energies'.
        Shape of (M, N)
    energies : ndarray
        A sample of the energy surface of the system.
        Aligns with 'compositions'.
        Shape of (M,)
    chemical_potentials : ndarray
        Shape of (N,)
        Will be overwritten
    fixed_chempot_indices : ndarray
        Variable shape from (0,) to (N-1,)
    fixed_lincomb_molefrac_coefs : ndarray
        Variable shape from (0,P) to (N-1, P)
    fixed_lincomb_molefrac_rhs : ndarray
        Variable shape from (0,) to (N-1,)
    result_fractions : ndarray
        Relative amounts of the points making up the hyperplane simplex. Shape of (P,).
        Will be overwritten. Output sums to 1.
    result_simplex : ndarray
        Energies of the points making up the hyperplane simplex. Shape of (P,).
        Will be overwritten. Output*result_fractions sums to out_energy (return value).

    Returns
    -------
    out_energy : double
        Energy of the output configuration.

    Examples
    --------
    None yet.

    Notes
    -----
    M: number of energy points that have been sampled
    N: number of components
    P: N+1, max phases by gibbs phase rule that we can find in a point calculations
    """
    # Scalars
    cdef Py_ssize_t num_points = compositions.shape[0]
    cdef Py_ssize_t num_components = compositions.shape[1]
    cdef Py_ssize_t num_fixed_chempots = fixed_chempot_indices.shape[0]
    cdef Py_ssize_t simplex_size = num_components - num_fixed_chempots
    cdef Py_ssize_t fixed_index = 0
    cdef Py_ssize_t saved_trial = 0
    cdef Py_ssize_t min_df
    cdef int max_iterations = 1000
    cdef int iterations = 0
    cdef Py_ssize_t i, j, idx, ici, simplex_idx, trial_idx, chempot_idx
    cdef bint skip_index = False
    cdef double lowest_df = 0
    cdef double out_energy = 0
    cdef Py_ssize_t num_blocks = (num_points + BLOCK_SIZE - 1) // BLOCK_SIZE
    cdef const double* compositions_ptr = NULL
    cdef const double* energies_ptr = NULL
    if num_points > 0:
        compositions_ptr = &compositions[0, 0]
        energies_ptr = &energies[0]
    # 1-D
    cdef int* best_guess_simplex = <int*>malloc(simplex_size * sizeof(int))
    for i in range(num_components):
        skip_index = False
        for j in range(num_fixed_chempots):
            if i == fixed_chempot_indices[j]:
                skip_index = True
        if skip_index:
            pass
        else:
            best_guess_simplex[fixed_index] = i
            fixed_index += 1
    cdef int* free_chempot_indices = <int*>malloc(simplex_size * sizeof(int))
    cdef int* candidate_simplex = <int*>malloc(simplex_size * sizeof(int))
    for i in range(simplex_size):
        free_chempot_indices[i] = best_guess_simplex[i]
        candidate_simplex[i] = best_guess_simplex[i]
    cdef int* int_tmp = <int*>malloc(simplex_size * sizeof(int))
    cdef double* free_candidate_potentials = <double*>malloc(simplex_size * sizeof(double))
    cdef double* candidate_potentials = <double*>malloc(num_components * sizeof(double))
    for ici in range(num_fixed_chempots):
        chempot_idx = fixed_chempot_indices[ici]
        candidate_potentials[chempot_idx] = chemical_potentials[chempot_idx]
    cdef double* smallest_fractions = <double*>malloc(simplex_size * sizeof(double))
    # 2-D
    cdef int* trial_simplices = <int*>malloc(simplex_size * simplex_size * sizeof(int))
    cdef double* fractions = <double*>malloc(simplex_size * simplex_size * sizeof(double))
    for i in range(simplex_size):
        for j in range(simplex_size):
            trial_simplices[i*simplex_size + j] = best_guess_simplex[j]
    cdef double* f_candidate_tieline = <double*>malloc(simplex_size * simplex_size * sizeof(double))
    # Pruning state for find_lowest_driving_force
    cdef int* block_state = <int*>malloc(num_blocks * sizeof(int))
    for i in range(num_blocks):
        block_state[i] = BLOCK_UNSEEN
    cdef double* block_data = <double*>malloc(num_blocks * (3*num_components + 4) * sizeof(double))
    cdef double* block_bounds = <double*>malloc(num_blocks * sizeof(double))


    while iterations < max_iterations:
        iterations += 1

        for trial_idx in range(simplex_size):
            for simplex_idx in range(simplex_size):
                fractions[trial_idx*simplex_size + simplex_idx] = 0
            simplex_fractions(compositions, fixed_chempot_indices, <int[:simplex_size]>&trial_simplices[trial_idx*simplex_size],
                              fixed_lincomb_molefrac_coefs, fixed_lincomb_molefrac_rhs, &fractions[trial_idx*simplex_size])
            smallest_fractions[trial_idx] = _min(&fractions[trial_idx*simplex_size], simplex_size)
        # Choose simplex with the largest smallest-fraction
        saved_trial = argmax(smallest_fractions, simplex_size)
        if smallest_fractions[saved_trial] < -simplex_size:
            break
        # Should be exactly one candidate simplex
        for i in range(simplex_size):
            candidate_simplex[i] = trial_simplices[saved_trial*simplex_size + i]
        for i in range(simplex_size):
            idx = candidate_simplex[i]
            for ici in range(simplex_size):
                chempot_idx = free_chempot_indices[ici]
                f_candidate_tieline[i + simplex_size*ici] = compositions[idx, chempot_idx]
            free_candidate_potentials[i] = energies[idx]
            for ici in range(num_fixed_chempots):
                chempot_idx = fixed_chempot_indices[ici]
                free_candidate_potentials[i] -= chemical_potentials[chempot_idx] * compositions[idx, chempot_idx]
        solve(f_candidate_tieline, simplex_size, free_candidate_potentials, int_tmp)
        if free_candidate_potentials[0] == -1e19:
            break
        for ici in range(simplex_size):
            candidate_potentials[free_chempot_indices[ici]] = free_candidate_potentials[ici]
        min_df = find_lowest_driving_force(compositions_ptr, energies_ptr, num_points, num_components,
                                           candidate_potentials, candidate_simplex, simplex_size,
                                           block_state, block_data, block_bounds, &lowest_df)
        for i in range(simplex_size):
            best_guess_simplex[i] = candidate_simplex[i]
        for i in range(simplex_size):
            for j in range(simplex_size):
                trial_simplices[i*simplex_size + j] = best_guess_simplex[j]

        # Trial simplices will be the current simplex with each vertex
        #     replaced by the trial point
        # Exactly one of those simplices will contain a given test point,
        #     excepting edge cases
        for i in range(simplex_size):
            trial_simplices[i*simplex_size + i] = min_df
        if lowest_df > -1e-8:
            break
    out_energy = 0
    for i in range(simplex_size):
        idx = best_guess_simplex[i]
        out_energy += fractions[saved_trial*simplex_size + i] * energies[idx]
    for i in range(simplex_size):
        result_fractions[i] = fractions[saved_trial*simplex_size + i]
    for ici in range(simplex_size):
        chempot_idx = free_chempot_indices[ici]
        chemical_potentials[chempot_idx] = free_candidate_potentials[ici]
        result_simplex[ici] = best_guess_simplex[ici]

    # Hack to enforce Gibbs phase rule, shape of result is comp+1, shape of hyperplane is comp
    result_fractions[simplex_size:] = 0.0
    result_simplex[simplex_size:] = 0

    # 1-D
    free(best_guess_simplex)
    free(free_chempot_indices)
    free(candidate_simplex)
    free(int_tmp)
    free(free_candidate_potentials)
    free(candidate_potentials)
    free(smallest_fractions)
    # 2-D
    free(trial_simplices)
    free(fractions)
    free(f_candidate_tieline)
    free(block_state)
    free(block_data)
    free(block_bounds)

    return out_energy
