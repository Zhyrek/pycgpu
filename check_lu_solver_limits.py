#!/usr/bin/env python
"""Check if LU solver hard-coded limits are sufficient."""

from pycalphad import Database, Workspace, variables as v
from pycalphad.gpu.gpu_codegen import compute_dynamic_kernel_sizes

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Test with different phase counts
for num_phases in [4, 6, 8, 10]:
    # Take first N phases
    all_phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'AU2BI_HEX', 'RHOMBOHEDRAL_A7', 'BCC_A2']
    phases = all_phases[:min(num_phases, len(all_phases))]
    
    # Add dummy phases if needed
    while len(phases) < num_phases:
        phases.append(f'DUMMY_{len(phases)}')
    
    try:
        wks = Workspace(dbf, comps, phases[:6], {v.X('BI'): 0.3, v.T: 600, v.P: 101325})  # Max 6 real phases
        sizes = compute_dynamic_kernel_sizes(wks)
        
        print(f"\nWith {num_phases} phases:")
        print(f"  MAX_DOF_PER_PHASE = {sizes['MAX_DOF_PER_PHASE']}")
        print(f"  MAX_INTERNAL_CONSTRAINTS = {sizes['MAX_INTERNAL_CONSTRAINTS']}")
        
        max_matrix_dim = sizes['MAX_DOF_PER_PHASE'] + sizes['MAX_INTERNAL_CONSTRAINTS']
        print(f"  Maximum matrix dimension = {max_matrix_dim}")
        print(f"  LU solver limit = 32")
        
        if max_matrix_dim > 32:
            print(f"  WARNING: Matrix dimension {max_matrix_dim} exceeds LU solver hard-coded limit of 32!")
        else:
            print(f"  OK: Matrix dimension {max_matrix_dim} is within LU solver limit")
            
    except Exception as e:
        print(f"\nError with {num_phases} phases: {e}")