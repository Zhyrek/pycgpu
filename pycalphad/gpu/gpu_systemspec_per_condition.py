# gpu_systemspec_per_condition.py
# Modified functions to create one SystemSpecification per condition

import numpy as np
from typing import Dict, List, Any
import pycalphad.variables as v

def create_system_specifications_array(wks_obj, num_conditions: int, dynamic_sizes: dict, properties=None, verbose=False):
    """
    Create an array of SystemSpecifications, one per condition.
    
    Returns:
        system_specs_array: Array of shape (num_conditions, spec_size_in_doubles)
    """
    if verbose:
        print(f"[GPU] Creating {num_conditions} SystemSpecifications...")
    
    # Get sizes from dynamic_sizes or defaults
    if dynamic_sizes is not None:
        max_components = int(dynamic_sizes["MAX_COMPONENTS"])
        max_statevars = int(dynamic_sizes["MAX_STATEVARS"])
        max_phases = int(dynamic_sizes["MAX_PHASES"])
        max_fixed_mole = int(dynamic_sizes["MAX_FIXED_MOLE_FRACTION_CONDITIONS"])
    else:
        from pycalphad.gpu.gpu_codegen import _get_c_define
        max_components = int(_get_c_define("MAX_COMPONENTS"))
        max_statevars = int(_get_c_define("MAX_STATEVARS"))
        max_phases = int(_get_c_define("MAX_PHASES"))
        max_fixed_mole = int(_get_c_define("MAX_FIXED_MOLE_FRACTION_CONDITIONS"))
    
    # Calculate SystemSpecification size in doubles
    # This must match the C struct layout exactly INCLUDING work arrays
    
    # Core fields
    spec_core_doubles = (
        3 +  # num_statevars, num_components (as int->double), prescribed_system_amount
        max_components +  # initial_chemical_potentials
        max_fixed_mole * max_components +  # prescribed_mole_fraction_coefficients
        max_fixed_mole +  # prescribed_mole_fraction_rhs
        2 +  # num_prescribed_mole_fraction_conditions, num_prescribed_mole_fraction_coefficients_cols
        max_components + 1 +  # free_chemical_potential_indices + count
        max_statevars + 1 +   # free_statevar_indices + count
        max_components + 1 +  # fixed_chemical_potential_indices + count
        max_statevars + 1 +   # fixed_statevar_indices + count
        max_phases + 1 +      # fixed_stable_compset_indices + count
        1 +  # max_num_free_stable_phases
        1    # ALLOWED_MASS_RESIDUAL
    )
    
    # Work arrays that are part of the C struct (even though passed separately)
    MAX_SVD_DIM = max_phases + max_fixed_mole + max_components + max_statevars + 2
    MAX_SVD_M = MAX_SVD_DIM
    MAX_SVD_N = MAX_SVD_DIM
    MAX_PHASE_MATRIX_DIM = max_components + max_components  # Approximation
    
    spec_work_doubles = (
        MAX_SVD_M * MAX_SVD_N +     # A_lstsq_copy
        MAX_SVD_M * MAX_SVD_N +     # U_lstsq
        MAX_SVD_N * MAX_SVD_N +     # V_lstsq
        MAX_SVD_N +                 # singular_values_lstsq
        MAX_SVD_N +                 # superdiag_lstsq
        MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM +  # U_inv
        MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM +  # V_inv
        MAX_PHASE_MATRIX_DIM +      # singular_values_inv
        MAX_PHASE_MATRIX_DIM +      # superdiag_inv
        MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM    # work_inv
    )
    
    spec_doubles_per_condition = spec_core_doubles + spec_work_doubles
    
    # Create array to hold all SystemSpecifications
    system_specs_array = np.zeros((num_conditions, spec_doubles_per_condition), dtype=np.float64)
    
    # Get condition arrays
    import pycalphad.variables as v
    unitless_conds = OrderedDict((key, wks_obj.conditions[key]) for key in wks_obj.conditions.keys())
    
    # Process each condition
    for cond_idx in range(num_conditions):
        if verbose and cond_idx % 100 == 0:
            print(f"[GPU] Processing condition {cond_idx}/{num_conditions}...")
        
        # Create spec data for this condition
        spec_data = np.zeros(spec_doubles_per_condition, dtype=np.float64)
        offset = 0
        
        # Basic fields
        spec_data[offset] = float(len(wks_obj.phase_record_factory.state_variables))  # num_statevars
        offset += 1
        spec_data[offset] = float(len(wks_obj.components))  # num_components
        offset += 1
        spec_data[offset] = 1.0  # prescribed_system_amount (default)
        offset += 1
        
        # Initial chemical potentials (from properties if available)
        if properties is not None and hasattr(properties, 'MU'):
            # Extract chemical potentials for this condition
            multi_idx = np.unravel_index(cond_idx, properties.GM.shape)
            # MU typically has shape (..., n_components-1) since last component is dependent
            mu_shape = properties.MU.shape
            n_mu_components = mu_shape[-1] if len(mu_shape) > 0 else 0
            
            for comp_idx in range(len(wks_obj.components)):
                if comp_idx < max_components:
                    if comp_idx < n_mu_components:
                        mu_val = properties.MU[multi_idx + (comp_idx,)]
                        spec_data[offset + comp_idx] = float(mu_val) if not np.isnan(mu_val) else 0.0
                    else:
                        spec_data[offset + comp_idx] = 0.0  # Dependent component
        offset += max_components
        
        # Prescribed mole fraction constraints - CONDITION SPECIFIC!
        constraint_count = 0
        pmf_coeffs_start = offset
        offset += max_fixed_mole * max_components  # Skip coefficients for now
        pmf_rhs_start = offset
        
        for comp_idx, component in enumerate(wks_obj.components[:max_components]):
            x_var = v.MoleFraction(component)
            if x_var in unitless_conds and constraint_count < max_fixed_mole:
                # Get the value for THIS SPECIFIC CONDITION
                x_values = np.asarray(unitless_conds[x_var])
                if x_values.size > 1:
                    # Multi-point condition - use the value for this condition index
                    x_scalar = float(x_values.flat[cond_idx % x_values.size])
                else:
                    x_scalar = float(x_values.item())
                
                # Set coefficient (1.0 for component comp_idx)
                coeff_idx = pmf_coeffs_start + constraint_count * max_components + comp_idx
                spec_data[coeff_idx] = 1.0
                
                # Set RHS value (condition-specific!)
                spec_data[pmf_rhs_start + constraint_count] = x_scalar
                constraint_count += 1
        
        offset = pmf_rhs_start + max_fixed_mole
        
        # Number of constraints
        spec_data[offset] = float(constraint_count)  # num_prescribed_mole_fraction_conditions
        offset += 1
        spec_data[offset] = float(len(wks_obj.components))  # num_prescribed_mole_fraction_coefficients_cols
        offset += 1
        
        # Index arrays (simplified - assuming all are free for now)
        # Free chemical potential indices
        # CRITICAL FIX: Only include non-VA components as free chemical potentials
        # VA is typically excluded from chemical potential calculations
        free_chem_count = 0
        for comp_idx, comp in enumerate(wks_obj.components):
            if comp_idx < max_components and comp.name != 'VA':
                spec_data[offset + free_chem_count] = float(comp_idx)
                free_chem_count += 1
        # Fill remaining with -1
        for i in range(free_chem_count, max_components):
            spec_data[offset + i] = -1.0
        offset += max_components
        spec_data[offset] = float(free_chem_count)  # num_free_chemical_potentials
        offset += 1
        
        # Free statevar indices - CRITICAL FIX: Only free if NOT in conditions
        free_sv_indices_start = offset
        free_sv_count = 0
        state_variables = wks_obj.phase_record_factory.state_variables
        for sv_idx, state_var in enumerate(state_variables[:max_statevars]):
            if state_var not in wks_obj.conditions:  # Free only if NOT specified
                spec_data[offset + free_sv_count] = float(sv_idx)
                free_sv_count += 1
        # Fill remaining with -1
        for i in range(free_sv_count, max_statevars):
            spec_data[offset + i] = -1.0
        offset += max_statevars
        spec_data[offset] = float(free_sv_count)  # num_free_statevars
        offset += 1
        
        # Fixed chemical potential indices (empty for now)
        for i in range(max_components):
            spec_data[offset + i] = -1.0
        offset += max_components
        spec_data[offset] = 0.0  # num_fixed_chemical_potentials
        offset += 1
        
        # Fixed statevar indices - CRITICAL FIX: Fixed if IN conditions
        fixed_sv_indices_start = offset
        fixed_sv_count = 0
        for sv_idx, state_var in enumerate(state_variables[:max_statevars]):
            if state_var in wks_obj.conditions:  # Fixed if specified
                spec_data[offset + fixed_sv_count] = float(sv_idx)
                fixed_sv_count += 1
        # Fill remaining with -1
        for i in range(fixed_sv_count, max_statevars):
            spec_data[offset + i] = -1.0
        offset += max_statevars
        spec_data[offset] = float(fixed_sv_count)  # num_fixed_statevars
        offset += 1
        offset += max_phases + 1       # fixed_stable_compset_indices + count
        
        # Final fields
        spec_data[offset] = float(max_phases)  # max_num_free_stable_phases
        offset += 1
        spec_data[offset] = 1e-12  # ALLOWED_MASS_RESIDUAL
        
        # Store in array
        system_specs_array[cond_idx] = spec_data
    
    if verbose:
        print(f"[GPU] Created SystemSpecifications array: shape={system_specs_array.shape}")
        print(f"[GPU] Each SystemSpec size: {spec_doubles_per_condition} doubles = {spec_doubles_per_condition * 8} bytes")
        print(f"[GPU] sizeof(SystemSpecification) should match this!")
        print(f"[GPU] First condition X(TI) constraint: {system_specs_array[0, pmf_rhs_start]}")
        if num_conditions > 1:
            print(f"[GPU] Second condition X(TI) constraint: {system_specs_array[1, pmf_rhs_start]}")
    
    return system_specs_array


from collections import OrderedDict

# Add any other necessary functions...