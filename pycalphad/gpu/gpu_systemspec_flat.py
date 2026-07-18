"""Create SystemSpecification as a flat double array to ensure proper memory layout."""

import numpy as np


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
        1 +  # ALLOWED_MASS_RESIDUAL
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
    
    # ALLOWED_MASS_RESIDUAL
    spec_doubles[idx] = global_spec_scalars[11]
    idx += 1

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