"""Create an array of SystemSpecification structs for multi-condition GPU calculations."""

import numpy as np
from .gpu_equilibrium import (_get_c_define, _populate_system_specification, 
                             _create_system_specification_struct)
from .gpu_systemspec_flat import create_flat_system_specification, apply_safe_padding
from .gpu_properties_subset import PropertiesSubset


def create_system_specifications_array(wks_obj, num_conditions, dynamic_sizes, properties, verbose=False):
    """
    Create an array of SystemSpecification structs, one per condition.
    
    This fixes the multi-condition GPU issue where all threads were sharing the same
    SystemSpecification, causing all conditions to use the same prescribed_mole_fraction_rhs.
    
    Parameters:
    -----------
    wks_obj : WorkspaceState
        The workspace object containing condition data
    num_conditions : int
        Number of conditions to process
    dynamic_sizes : dict
        Dynamic sizes for arrays
    properties : object
        Properties object with initial values
    verbose : bool
        Enable verbose output
        
    Returns:
    --------
    np.ndarray
        Flattened array of doubles containing all SystemSpecification structs
    """
    if verbose:
        print(f"[GPU] Creating SystemSpecification array for {num_conditions} conditions")
    
    # Get sizes
    max_components = dynamic_sizes["MAX_COMPONENTS"]
    max_statevars = dynamic_sizes["MAX_STATEVARS"]
    max_phases = dynamic_sizes["MAX_PHASES"]
    max_fixed_mole = dynamic_sizes["MAX_FIXED_MOLE_FRACTION_CONDITIONS"]
    
    # Create arrays to hold all SystemSpecs
    all_specs = []
    
    # Extract condition arrays
    import pycalphad.variables as v
    
    # Get the X(TI) condition values
    x_ti_values = None
    for comp in wks_obj.components:
        x_var = v.X(comp)
        if x_var in wks_obj.conditions:
            condition_value = wks_obj.conditions[x_var]
            x_ti_values = np.asarray(condition_value).flatten()
            if verbose:
                print(f"[GPU] Found X({comp}) condition with {len(x_ti_values)} values: {x_ti_values}")
            break
    
    if x_ti_values is None:
        # No mole fraction conditions, create single spec for all conditions
        x_ti_values = np.zeros(num_conditions)
        if verbose:
            print(f"[GPU] No mole fraction conditions found, creating {num_conditions} identical specs")
    
    # Get temperature and composition arrays
    temp_values = wks_obj.conditions[v.T]
    if not hasattr(temp_values, '__len__'):
        temp_values = [temp_values]
    temp_values = np.asarray(temp_values).flatten()
    
    # Get composition array size (for X(BI) or similar)
    comp_values_len = 1
    for key in wks_obj.conditions:
        if hasattr(key, 'species') and key.species != 'VA':
            comp_array = np.asarray(wks_obj.conditions[key]).flatten()
            comp_values_len = len(comp_array)
            break
    
    # Create one SystemSpecification per condition
    for condition_idx in range(num_conditions):
        if verbose:
            print(f"\n[GPU] Creating SystemSpecification for condition {condition_idx}")
        
        # Calculate temperature and composition indices for this condition
        temp_idx = condition_idx // comp_values_len
        comp_idx = condition_idx % comp_values_len
        
        if verbose:
            print(f"  Condition {condition_idx}: temp_idx={temp_idx}, comp_idx={comp_idx} (comp_values_len={comp_values_len})")
        
        # Create base arrays for this condition
        global_spec_np = np.zeros(50, dtype=np.float64)  # Scalar fields
        global_spec_arrays = {
            'initial_chemical_potentials': np.zeros(max_components, dtype=np.float64),
            'prescribed_mole_fraction_coefficients': np.zeros((max_fixed_mole, max_components), dtype=np.float64),
            'prescribed_mole_fraction_rhs': np.zeros(max_fixed_mole, dtype=np.float64),
            'free_chemical_potential_indices': np.full(max_components, -1, dtype=np.int32),
            'free_statevar_indices': np.full(max_statevars, -1, dtype=np.int32),
            'fixed_chemical_potential_indices': np.full(max_components, -1, dtype=np.int32),
            'fixed_statevar_indices': np.full(max_statevars, -1, dtype=np.int32),
            'fixed_stable_compset_indices': np.full(max_phases, -1, dtype=np.int32)
        }
        
        # Create a temporary workspace object with single-point conditions
        class TempWorkspace:
            def __init__(self, original_wks, condition_idx, x_ti_value, temp_idx, comp_idx):
                self.components = original_wks.components
                self.phase_record_factory = original_wks.phase_record_factory
                self.verbose = original_wks.verbose
                
                # Copy conditions but use single-point values based on proper indices
                self.conditions = {}
                for key, value in original_wks.conditions.items():
                    value_array = np.asarray(value)
                    if value_array.size > 1:
                        # Multi-point condition - use appropriate index based on variable type
                        if key == v.T:
                            # Temperature - use temp_idx
                            self.conditions[key] = float(value_array.flatten()[temp_idx])
                        elif hasattr(key, 'species') and key.species != 'VA':
                            # Composition variable - use comp_idx
                            self.conditions[key] = float(value_array.flatten()[comp_idx])
                        else:
                            # Other multi-point conditions - use condition_idx as fallback
                            if condition_idx < value_array.size:
                                self.conditions[key] = float(value_array.flatten()[condition_idx])
                            else:
                                self.conditions[key] = float(value_array.flatten()[-1])
                    else:
                        # Single-point condition - use for all
                        self.conditions[key] = float(value_array.item())
                
                if verbose:
                    print(f"[GPU] Condition {condition_idx} - X(TI) = {x_ti_value}")
                    print(f"[GPU] Full conditions: {self.conditions}")
        
        # Get the X(TI) value for this condition
        x_ti_value = x_ti_values[condition_idx] if condition_idx < len(x_ti_values) else x_ti_values[-1]
        
        # Create temporary workspace with single-point conditions
        temp_wks = TempWorkspace(wks_obj, condition_idx, x_ti_value, temp_idx, comp_idx)
        
        # Create properties subset for this specific condition
        properties_subset = PropertiesSubset(properties, condition_idx, temp_idx, comp_idx, verbose=verbose)
        
        # Populate the SystemSpecification for this condition
        _populate_system_specification(global_spec_np, global_spec_arrays, temp_wks, 
                                     dynamic_sizes, properties_subset)
        
        # Create flat double array instead of struct to avoid alignment issues
        spec_doubles = create_flat_system_specification(global_spec_np, global_spec_arrays, 
                                                       dynamic_sizes)
        
        # Apply padding to avoid cache conflicts
        spec_doubles_padded = apply_safe_padding(spec_doubles, verbose=verbose)
        
        if verbose:
            print(f"[GPU] Condition {condition_idx} - SystemSpec fields from flat array:")
            print(f"  temp_idx={temp_idx}, comp_idx={comp_idx}")
            print(f"  num_statevars: {int(spec_doubles_padded[0])}")
            print(f"  num_components: {int(spec_doubles_padded[1])}")
            print(f"  prescribed_system_amount: {spec_doubles_padded[2]}")
            print(f"  initial_chemical_potentials[0]: {spec_doubles_padded[3]}")
            print(f"  initial_chemical_potentials[1]: {spec_doubles_padded[4]}")
            
            # Calculate offset to prescribed_mole_fraction_rhs
            offset = 3 + dynamic_sizes["MAX_COMPONENTS"] + (dynamic_sizes["MAX_FIXED_MOLE_FRACTION_CONDITIONS"] * dynamic_sizes["MAX_COMPONENTS"])
            print(f"  prescribed_mole_fraction_rhs[0]: {spec_doubles_padded[offset]} (should be X(TI) for this condition)")
            print(f"  First 10 doubles: {spec_doubles_padded[:10]}")
        
        all_specs.append(spec_doubles_padded)
    
    # Stack all specs into a single array
    # Shape: (num_conditions, spec_size_in_doubles)
    specs_array = np.vstack(all_specs)
    
    if verbose:
        print(f"\n[GPU] Created SystemSpecification array:")
        print(f"  Shape: {specs_array.shape}")
        print(f"  Total size: {specs_array.nbytes} bytes")
        print(f"  Specs per condition: {specs_array.shape[1]} doubles")
    
    # Flatten for GPU transfer
    specs_flat = specs_array.flatten()
    
    return specs_flat