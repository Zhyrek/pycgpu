#!/usr/bin/env python
"""Check what MAX_PHASES value is calculated for different phase counts."""

from pycalphad import Database, Workspace, variables as v
from pycalphad.gpu.gpu_codegen import compute_dynamic_kernel_sizes

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Test with different phase counts
for num_phases in [4, 5, 6]:
    # Use actual phases from database
    all_phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'AU2BI_HEX', 'RHOMBOHEDRAL_A7', 'BCC_A2']
    phases = all_phases[:num_phases]
    
    wks = Workspace(dbf, comps, phases, {v.X('BI'): 0.3, v.T: 600, v.P: 101325})
    sizes = compute_dynamic_kernel_sizes(wks)
    
    print(f"\nWith {num_phases} actual phases:")
    print(f"  Calculation: max(4, int({num_phases} * 1.2)) = max(4, {int(num_phases * 1.2)}) = {sizes['MAX_PHASES']}")
    print(f"  MAX_PHASES = {sizes['MAX_PHASES']}")
    print(f"  MAX_COMPONENTS = {sizes['MAX_COMPONENTS']}")
    print(f"  MAX_DOF_PER_PHASE = {sizes['MAX_DOF_PER_PHASE']}")
    print(f"  MAX_INTERNAL_CONSTRAINTS = {sizes['MAX_INTERNAL_CONSTRAINTS']}")
    
    # Calculate memory layout sizes
    doubles_per_condition = (sizes['MAX_PHASES'] + sizes['MAX_PHASES'] + 
                           (sizes['MAX_PHASES'] * sizes['MAX_DOF_PER_PHASE']) + 
                           (sizes['MAX_PHASES'] * sizes['MAX_COMPONENTS']) + 
                           sizes['MAX_COMPONENTS'] + 1)
    print(f"  doubles_per_condition = {doubles_per_condition}")
    
    # Is MAX_PHASES sufficient?
    if sizes['MAX_PHASES'] >= num_phases:
        print(f"  ✓ MAX_PHASES ({sizes['MAX_PHASES']}) >= actual phases ({num_phases})")
    else:
        print(f"  ✗ MAX_PHASES ({sizes['MAX_PHASES']}) < actual phases ({num_phases}) - PROBLEM!")