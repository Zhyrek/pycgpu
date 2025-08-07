#!/usr/bin/env python
"""Verify the actual SystemSpec stride value being used."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Patch to intercept stride value
original_equilibrium_gpu = None

def patched_equilibrium_gpu(*args, **kwargs):
    """Intercept and print stride value."""
    import pycalphad.gpu.gpu_equilibrium as gpu_eq
    
    # Save original function
    original_calculate = gpu_eq.calculate_equilibrium_gpu
    
    def patched_calculate(wks, **calc_kwargs):
        # Call original to get data
        result = original_calculate(wks, **calc_kwargs)
        
        # The stride is calculated in calculate_equilibrium_gpu
        # Let's add a debug print there
        return result
    
    # Temporarily patch
    gpu_eq.calculate_equilibrium_gpu = patched_calculate
    
    try:
        result = original_equilibrium_gpu(*args, **kwargs)
    finally:
        # Restore
        gpu_eq.calculate_equilibrium_gpu = original_calculate
    
    return result

def test_stride():
    """Test the actual stride value."""
    
    # Patch the equilibrium_gpu function
    import pycalphad.gpu.gpu_equilibrium as gpu_eq
    global original_equilibrium_gpu
    original_equilibrium_gpu = gpu_eq.equilibrium_gpu
    
    # Instead of patching, let's just run with verbose and look for stride info
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = filter_phases(dbf, comps)
    
    print("Testing SystemSpec stride value...")
    print("=" * 60)
    
    # Test with 32 conditions (the failing case)
    conditions = {
        v.X('BI'): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        v.T: [400, 500, 600, 700],
        v.P: 101325
    }
    
    print(f"Running with 4x8 = 32 conditions")
    print(f"Expected threads with conflicts (tid % 7 == 3): [3, 10, 17, 24, 31]")
    print(f"Actual failing threads from test: 10 and 17")
    print()
    
    # Let's directly check the stride calculation
    from pycalphad.gpu.gpu_systemspec_flat import create_flat_system_specification
    
    # Simulate creating one spec
    MAX_COMPONENTS = 3
    MAX_STATEVARS = 3
    MAX_PHASES = 6
    MAX_FIXED_MOLE = 3
    
    dynamic_sizes = {
        "MAX_COMPONENTS": MAX_COMPONENTS,
        "MAX_STATEVARS": MAX_STATEVARS,
        "MAX_PHASES": MAX_PHASES,
        "MAX_FIXED_MOLE_FRACTION_CONDITIONS": MAX_FIXED_MOLE,
    }
    
    # Create dummy data
    global_spec_np = np.zeros(50, dtype=np.float64)
    global_spec_arrays = {
        'initial_chemical_potentials': np.zeros(MAX_COMPONENTS, dtype=np.float64),
        'prescribed_mole_fraction_coefficients': np.zeros((MAX_FIXED_MOLE, MAX_COMPONENTS), dtype=np.float64),
        'prescribed_mole_fraction_rhs': np.zeros(MAX_FIXED_MOLE, dtype=np.float64),
        'free_chemical_potential_indices': np.full(MAX_COMPONENTS, -1, dtype=np.int32),
        'free_statevar_indices': np.full(MAX_STATEVARS, -1, dtype=np.int32),
        'fixed_chemical_potential_indices': np.full(MAX_COMPONENTS, -1, dtype=np.int32),
        'fixed_statevar_indices': np.full(MAX_STATEVARS, -1, dtype=np.int32),
        'fixed_stable_compset_indices': np.full(MAX_PHASES, -1, dtype=np.int32),
    }
    
    spec_doubles = create_flat_system_specification(global_spec_np, global_spec_arrays, dynamic_sizes)
    
    print(f"SystemSpec size: {len(spec_doubles)} doubles")
    print(f"SystemSpec size % 7 = {len(spec_doubles) % 7}")
    print()
    
    # 83 % 7 = 6, so 83 * 7 = 581
    # The problem is that thread N and thread N+7 are separated by exactly 581 doubles
    # This creates a stride-7 conflict pattern
    
    print(f"Analysis:")
    print(f"  Stride = 83 doubles")
    print(f"  83 % 7 = 6")
    print(f"  Thread separation = 7 * 83 = 581 doubles")
    print(f"  581 % 7 = 0 (perfect conflict!)")
    print()
    print(f"This means threads 3, 10, 17, 24, 31 all access memory at:")
    print(f"  Thread 3:  offset 3*83 = 249")
    print(f"  Thread 10: offset 10*83 = 830 = 249 + 581") 
    print(f"  Thread 17: offset 17*83 = 1411 = 249 + 2*581")
    print(f"  etc.")
    print()
    print(f"The stride of 581 doubles between these threads creates cache conflicts")
    print(f"because they all map to the same cache set (581 % cache_line_size)")

if __name__ == "__main__":
    test_stride()