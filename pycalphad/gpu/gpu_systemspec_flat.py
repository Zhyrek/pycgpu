"""Create SystemSpecification as a flat double array to ensure proper memory layout."""

import numpy as np


def mole_fraction_convergence_data(coefficients, rhs):
    """Composition convergence data of the reference SystemSpecification
    (pycalphad/core/minimizer.pyx), for one coefficient matrix and many
    right-hand sides.

    The prescribed mole fraction conditions, together with the normalization
    condition, may uniquely determine the mole fractions of some components;
    check_convergence tests those against their unique values and checks the
    prescribed rows involving any other component as row residuals.

    Parameters
    ----------
    coefficients : (num_conditions, num_components) array
    rhs : (n, num_conditions) array, one row per condition point

    Returns
    -------
    determined_component_indices : (k,) int array
    undetermined_condition_row_indices : (m,) int array
    determined_component_mole_fractions : (n, k) array
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    num_conditions, num_components = coefficients.shape
    rhs = np.asarray(rhs, dtype=np.float64)
    if rhs.ndim != 2 or rhs.shape[1] != num_conditions:
        raise ValueError(f"rhs must have shape (n, {num_conditions}), got {rhs.shape}")
    condition_matrix = np.empty((num_conditions + 1, num_components))
    condition_matrix[:num_conditions, :] = coefficients
    condition_matrix[num_conditions, :] = 1.0
    condition_rhs = np.ones((rhs.shape[0], num_conditions + 1))
    condition_rhs[:, :num_conditions] = rhs
    u, singular_values, vt = np.linalg.svd(condition_matrix)
    rank = int(np.sum(singular_values > max(condition_matrix.shape) * np.finfo(np.float64).eps * singular_values[0]))
    # Minimum-norm least-squares solution of the conditions; unique for determined components
    target_mole_fractions = ((condition_rhs @ u[:, :rank]) / singular_values[:rank]) @ vt[:rank]
    # A component is determined iff no null space direction of the conditions can change it
    determined_components = ~np.any(np.abs(vt[rank:]) > 1e-10, axis=0)
    determined = np.nonzero(determined_components)[0]
    undetermined_coefficients = coefficients[:, ~determined_components]
    undetermined_rows = np.nonzero(np.any(np.abs(undetermined_coefficients) > 1e-14, axis=1))[0]
    return determined, undetermined_rows, target_mole_fractions[:, determined]


def spec_layout(dynamic_sizes):
    """Offsets (in doubles) of the flat SystemSpecification fields that are
    written per condition, and the length of the core section. Mirrors the
    sequential unpack in solve_condition.h."""
    MC = int(dynamic_sizes["MAX_COMPONENTS"])
    MSV = int(dynamic_sizes["MAX_STATEVARS"])
    MP = int(dynamic_sizes["MAX_PHASES"])
    MFIX = MC  # MAX_FIXED_MOLE_FRACTION_CONDITIONS
    MPAR = int(dynamic_sizes.get("MAX_PARAMS", 0))
    MPLC = int(dynamic_sizes.get("MAX_PHASE_LOCAL_CONDITIONS", 0))
    layout = {}
    idx = 3
    layout['initial_chemical_potentials'] = idx; idx += MC
    layout['prescribed_mole_fraction_coefficients'] = idx; idx += MFIX * MC
    layout['prescribed_mole_fraction_rhs'] = idx; idx += MFIX
    idx += 2 + (MC + 1) + (MSV + 1) + (MC + 1) + (MSV + 1) + (MP + 1) + 1
    layout['num_determined_components'] = idx; idx += 1
    layout['determined_component_indices'] = idx; idx += MC
    layout['determined_component_mole_fractions'] = idx; idx += MC
    layout['num_undetermined_condition_rows'] = idx; idx += 1
    layout['undetermined_condition_row_indices'] = idx; idx += MFIX
    idx += 1 + 4 * MPLC
    layout['fit_params'] = idx; idx += MPAR
    layout['num_params'] = idx; idx += 1
    layout['core_len'] = idx
    return layout


def write_mole_fraction_convergence_data(rows, coefficients, rhs, dynamic_sizes):
    """Fill the composition convergence fields of flat spec `rows` ((n, stride)
    array, NumPy or CuPy) for conditions sharing one coefficient matrix."""
    L = spec_layout(dynamic_sizes)
    determined, undetermined_rows, targets = mole_fraction_convergence_data(coefficients, rhs)
    xp = np if isinstance(rows, np.ndarray) else __import__('cupy')
    k, m = determined.size, undetermined_rows.size
    rows[:, L['num_determined_components']] = k
    start = L['determined_component_indices']
    rows[:, start:start + k] = xp.asarray(determined, dtype=np.float64)
    start = L['determined_component_mole_fractions']
    rows[:, start:start + k] = xp.asarray(targets)
    rows[:, L['num_undetermined_condition_rows']] = m
    start = L['undetermined_condition_row_indices']
    rows[:, start:start + m] = xp.asarray(undetermined_rows, dtype=np.float64)


def create_flat_system_specification(global_spec_scalars, global_spec_arrays, dynamic_sizes):
    """
    Create a SystemSpecification as a flat double array.
    
    This matches the exact memory layout expected by the C kernel,
    avoiding numpy structured array alignment issues.
    """
    # Get sizes
    MAX_COMPONENTS = int(dynamic_sizes["MAX_COMPONENTS"])
    MAX_STATEVARS = int(dynamic_sizes["MAX_STATEVARS"])
    MAX_PHASES = int(dynamic_sizes["MAX_PHASES"])
    MAX_DOF_PER_PHASE = int(dynamic_sizes["MAX_DOF_PER_PHASE"])
    MAX_INTERNAL_CONSTRAINTS = int(dynamic_sizes["MAX_INTERNAL_CONSTRAINTS"])
    MAX_FIXED_MOLE_FRACTION_CONDITIONS = MAX_COMPONENTS
    
    # Calculate SVD dimensions
    MAX_SVD_DIM = MAX_PHASES + MAX_FIXED_MOLE_FRACTION_CONDITIONS + MAX_COMPONENTS + MAX_STATEVARS + 2
    MAX_SVD_M = MAX_SVD_DIM
    MAX_SVD_N = MAX_SVD_DIM
    MAX_PHASE_MATRIX_DIM = MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS
    
    # Calculate total size in doubles
    MAX_PARAMS = int(dynamic_sizes.get("MAX_PARAMS", 0))
    MAX_PHASE_LOCAL_CONDITIONS = int(dynamic_sizes.get("MAX_PHASE_LOCAL_CONDITIONS", 0))
    spec_core_doubles = (
        3 +  # num_statevars, num_components, prescribed_system_amount
        MAX_COMPONENTS +  # initial_chemical_potentials
        (MAX_FIXED_MOLE_FRACTION_CONDITIONS * MAX_COMPONENTS) +  # prescribed_mole_fraction_coefficients
        MAX_FIXED_MOLE_FRACTION_CONDITIONS +  # prescribed_mole_fraction_rhs
        2 +  # num_prescribed_mole_fraction_conditions, num_prescribed_mole_fraction_coefficients_cols
        (MAX_COMPONENTS + 1) +  # free_chemical_potential_indices + num_free_chemical_potentials
        (MAX_STATEVARS + 1) +  # free_statevar_indices + num_free_statevars
        (MAX_COMPONENTS + 1) +  # fixed_chemical_potential_indices + num_fixed_chemical_potentials
        (MAX_STATEVARS + 1) +  # fixed_statevar_indices + num_fixed_statevars
        (MAX_PHASES + 1) +  # fixed_stable_compset_indices + num_fixed_stable_compsets
        1 +  # max_num_free_stable_phases
        (1 + 2 * MAX_COMPONENTS) +  # determined components: count, indices, mole fractions
        (1 + MAX_FIXED_MOLE_FRACTION_CONDITIONS) +  # undetermined condition rows: count, indices
        (1 + 4 * MAX_PHASE_LOCAL_CONDITIONS) +  # phase-local conditions block
        (MAX_PARAMS + 1)  # fit_params[MAX_PARAMS] + num_params
    )
    
    # NOTE: this array used to append per-condition SVD/inverse workspaces
    # (~5800 doubles for a 22-phase system) after the core fields. The kernel
    # parser only reads the core fields and all work arrays moved to the
    # separate WorkArrays allocation long ago, so the tail was pure dead
    # weight: ~75x spec-buffer bloat (640MB at 1M conditions). The kernel
    # slices by the RUNTIME system_spec_stride argument, so shrinking here
    # requires no kernel change.
    total_doubles = spec_core_doubles
    
    # Create flat array
    spec_doubles = np.zeros(total_doubles, dtype=np.float64)
    
    # Fill in values in the exact order expected by C
    idx = 0
    
    # Basic fields (as doubles)
    spec_doubles[idx] = float(global_spec_scalars[0])  # num_statevars
    idx += 1
    spec_doubles[idx] = float(global_spec_scalars[1])  # num_components
    idx += 1
    spec_doubles[idx] = global_spec_scalars[2]  # prescribed_system_amount
    idx += 1
    
    # Arrays
    # initial_chemical_potentials
    for i in range(MAX_COMPONENTS):
        if i < len(global_spec_arrays['initial_chemical_potentials']):
            spec_doubles[idx] = global_spec_arrays['initial_chemical_potentials'][i]
        idx += 1
    
    # prescribed_mole_fraction_coefficients (2D array flattened)
    pmfc = global_spec_arrays['prescribed_mole_fraction_coefficients']
    for i in range(MAX_FIXED_MOLE_FRACTION_CONDITIONS):
        for j in range(MAX_COMPONENTS):
            if i < pmfc.shape[0] and j < pmfc.shape[1]:
                spec_doubles[idx] = pmfc[i, j]
            idx += 1
    
    # prescribed_mole_fraction_rhs
    pmfr = global_spec_arrays['prescribed_mole_fraction_rhs']
    for i in range(MAX_FIXED_MOLE_FRACTION_CONDITIONS):
        if i < len(pmfr):
            spec_doubles[idx] = pmfr[i]
        idx += 1
    
    # More integer fields (as doubles)
    spec_doubles[idx] = float(global_spec_scalars[3])  # num_prescribed_mole_fraction_conditions
    idx += 1
    spec_doubles[idx] = float(global_spec_scalars[4])  # num_prescribed_mole_fraction_coefficients_cols
    idx += 1
    
    # Index arrays (store ints as doubles)
    # free_chemical_potential_indices
    fcpi = global_spec_arrays['free_chemical_potential_indices']
    for i in range(MAX_COMPONENTS):
        if i < len(fcpi) and fcpi[i] >= 0:
            spec_doubles[idx] = float(fcpi[i])
        else:
            spec_doubles[idx] = -1.0
        idx += 1
    
    # num_free_chemical_potentials
    spec_doubles[idx] = float(global_spec_scalars[5])
    idx += 1
    
    # free_statevar_indices
    fsvi = global_spec_arrays['free_statevar_indices']
    for i in range(MAX_STATEVARS):
        if i < len(fsvi) and fsvi[i] >= 0:
            spec_doubles[idx] = float(fsvi[i])
        else:
            spec_doubles[idx] = -1.0
        idx += 1
    
    # num_free_statevars
    spec_doubles[idx] = float(global_spec_scalars[6])
    idx += 1
    
    # fixed_chemical_potential_indices
    fcpi_fixed = global_spec_arrays['fixed_chemical_potential_indices']
    for i in range(MAX_COMPONENTS):
        if i < len(fcpi_fixed) and fcpi_fixed[i] >= 0:
            spec_doubles[idx] = float(fcpi_fixed[i])
        else:
            spec_doubles[idx] = -1.0
        idx += 1
    
    # num_fixed_chemical_potentials
    spec_doubles[idx] = float(global_spec_scalars[7])
    idx += 1
    
    # fixed_statevar_indices
    fsvi_fixed = global_spec_arrays['fixed_statevar_indices']
    for i in range(MAX_STATEVARS):
        if i < len(fsvi_fixed) and fsvi_fixed[i] >= 0:
            spec_doubles[idx] = float(fsvi_fixed[i])
        else:
            spec_doubles[idx] = -1.0
        idx += 1
    
    # num_fixed_statevars
    spec_doubles[idx] = float(global_spec_scalars[8])
    idx += 1
    
    # fixed_stable_compset_indices
    fsci = global_spec_arrays['fixed_stable_compset_indices']
    for i in range(MAX_PHASES):
        if i < len(fsci) and fsci[i] >= 0:
            spec_doubles[idx] = float(fsci[i])
        else:
            spec_doubles[idx] = -1.0
        idx += 1
    
    # num_fixed_stable_compsets
    spec_doubles[idx] = float(global_spec_scalars[9])
    idx += 1
    
    # max_num_free_stable_phases
    spec_doubles[idx] = float(global_spec_scalars[10])
    idx += 1
    
    # Composition convergence data (see mole_fraction_convergence_data)
    determined = np.asarray(global_spec_arrays['determined_component_indices'], dtype=np.int64)
    targets = np.asarray(global_spec_arrays['determined_component_mole_fractions'], dtype=np.float64)
    undetermined_rows = np.asarray(global_spec_arrays['undetermined_condition_row_indices'], dtype=np.int64)
    spec_doubles[idx] = float(determined.size)
    idx += 1
    spec_doubles[idx:idx + determined.size] = determined
    idx += MAX_COMPONENTS
    spec_doubles[idx:idx + targets.size] = targets
    idx += MAX_COMPONENTS
    spec_doubles[idx] = float(undetermined_rows.size)
    idx += 1
    spec_doubles[idx:idx + undetermined_rows.size] = undetermined_rows
    idx += MAX_FIXED_MOLE_FRACTION_CONDITIONS

    # Phase-local conditions: count then (model_idx, type, target, value)
    # per slot. Attached at compset creation to compsets of the matching
    # model; type 0 = mole fraction, 1 = site fraction.
    plc = global_spec_arrays.get('phase_local_conditions', []) or []
    spec_doubles[idx] = float(min(len(plc), MAX_PHASE_LOCAL_CONDITIONS))
    idx += 1
    for i in range(MAX_PHASE_LOCAL_CONDITIONS):
        if i < len(plc):
            model_idx, plc_type, target, value = plc[i]
            spec_doubles[idx] = float(model_idx)
            spec_doubles[idx + 1] = float(plc_type)
            spec_doubles[idx + 2] = float(target)
            spec_doubles[idx + 3] = float(value)
        idx += 4

    # Runtime fit parameters (trailing dof slots in the generated functions)
    fit_params = np.asarray(global_spec_arrays.get('fit_params', []), dtype=np.float64).reshape(-1)
    n_params = min(fit_params.size, MAX_PARAMS)
    for i in range(MAX_PARAMS):
        spec_doubles[idx] = fit_params[i] if i < n_params else 0.0
        idx += 1
    spec_doubles[idx] = float(n_params)
    idx += 1

    return spec_doubles


def apply_safe_padding(spec_doubles, verbose=False):
    """
    Apply padding to SystemSpec array to avoid cache conflicts.
    
    This function pads the array to avoid stride patterns that cause
    cache conflicts on GPUs, particularly the stride-7 pattern that
    causes threads with (tid % 7 == 3) to fail convergence.
    
    Parameters:
    -----------
    spec_doubles : np.ndarray
        The SystemSpec data as a double array
    verbose : bool
        Print padding information
        
    Returns:
    --------
    np.ndarray
        Padded array safe from cache conflicts
    """
    base_size = len(spec_doubles)
    CACHE_LINE_DOUBLES = 8
    
    # Round up to cache line boundary
    padded_size = ((base_size + CACHE_LINE_DOUBLES - 1) // CACHE_LINE_DOUBLES) * CACHE_LINE_DOUBLES
    
    # Avoid problematic patterns
    # Key insight: avoid sizes where (size % small_prime) creates patterns
    # Especially avoid size % 7 = 6, which creates 7-stride conflicts
    
    while padded_size < base_size * 2:  # Don't more than double
        # Check for problematic patterns
        has_conflict = False
        
        # Avoid exact multiples of small primes
        for prime in [3, 5, 7]:
            if padded_size % prime == 0:
                has_conflict = True
                break
        
        # Avoid size % 7 = 6 (the specific AuBi problem where 83 % 7 = 6)
        if padded_size % 7 == 6:
            has_conflict = True
        
        # Avoid exact multiples of GPU warp size
        if padded_size % 32 == 0:
            has_conflict = True
        
        if not has_conflict:
            break
            
        padded_size += CACHE_LINE_DOUBLES
    
    if verbose and padded_size != base_size:
        print(f"[GPU] Padding SystemSpec from {base_size} to {padded_size} doubles to avoid cache conflicts")
    
    # Create padded array
    padded_array = np.zeros(padded_size, dtype=np.float64)
    padded_array[:base_size] = spec_doubles
    
    return padded_array