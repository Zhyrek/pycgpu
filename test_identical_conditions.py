#!/usr/bin/env python
"""Test properties indexing with identical conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Test with 4 phases (working case)
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'RHOMBOHEDRAL_A7']

# Test with three IDENTICAL conditions
conditions = {
    v.X('BI'): [0.3, 0.3, 0.3],  # Three identical values
    v.T: 600, 
    v.P: 101325
}

print("Testing with 3 IDENTICAL conditions to check properties indexing")

# Run CPU calculation with verbose=False first
result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

print(f"\nCPU equilibrium calculation complete")
print(f"Result shape: {result.GM.shape}")
print(f"GM values: {result.GM.values.flatten()}")
print(f"NP shape: {result.NP.shape}")
print(f"Phase shape: {result.Phase.shape}")

# Check if properties vary by condition
print(f"\nChecking NP values for each condition:")
for i in range(3):
    # Get phase amounts for condition i  
    # When conditions are identical, we need different indexing
    if result.NP.shape[-2] == 1:
        # All conditions collapsed to one
        np_cond = result.NP.values[0, 0, 0, 0, :] 
        phases_cond = result.Phase.values[0, 0, 0, 0, :]
    else:
        np_cond = result.NP.isel(X_BI=i).values.flatten()
        phases_cond = result.Phase.isel(X_BI=i).values.flatten()
    active_phases = [(j, phases_cond[j], np_cond[j]) for j in range(len(phases_cond)) if phases_cond[j] != '']
    print(f"  Condition {i}: {active_phases}")

# Now test what happens inside GPU prepare function
print("\n" + "="*60)
print("Testing multi_idx calculation with identical conditions:")

# Simulate what happens in _prepare_gpu_data
gm_array = result.GM.values
print(f"gm_array.shape = {gm_array.shape}")

for cond_idx in range(3):
    multi_idx = np.unravel_index(cond_idx, gm_array.shape)
    print(f"\ncond_idx={cond_idx} -> multi_idx={multi_idx}")
    
    # This is the issue! When conditions are identical, gm_array.shape might be (1,1,1,1)
    # But we're trying to index with cond_idx=0,1,2
    
    if cond_idx > 0 and gm_array.shape[-1] == 1:
        print("  WARNING: Trying to access condition index beyond array bounds!")
        print("  This explains why all conditions get the same data!")
        
print("\nThe problem: When conditions are identical, pycalphad collapses them into a single calculation")
print("but GPU code expects separate data for each condition point.")