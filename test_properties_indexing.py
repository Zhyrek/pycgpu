#!/usr/bin/env python
"""Test properties indexing issue with multiple conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Test with 4 phases (working case)
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'RHOMBOHEDRAL_A7']

# Test with three different conditions
conditions = {
    v.X('BI'): [0.3, 0.4, 0.5],  # Three different values
    v.T: 600, 
    v.P: 101325
}

print("Testing with 3 different conditions to check properties indexing")

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
    np_cond = result.NP.isel(X_BI=i).values.flatten()
    phases_cond = result.Phase.isel(X_BI=i).values.flatten()
    active_phases = [(j, phases_cond[j], np_cond[j]) for j in range(len(phases_cond)) if phases_cond[j] != '']
    print(f"  Condition {i} (X_BI={conditions[v.X('BI')][i]}): {active_phases}")

print(f"\nChecking MU values for each condition:")    
for i in range(3):
    mu_cond = result.MU.isel(X_BI=i).values.flatten()
    print(f"  Condition {i}: MU = {mu_cond}")

# Now test what happens inside GPU prepare function
print("\n" + "="*60)
print("Testing multi_idx calculation:")

# Simulate what happens in _prepare_gpu_data
gm_array = result.GM.values
print(f"gm_array.shape = {gm_array.shape}")

for cond_idx in range(3):
    multi_idx = np.unravel_index(cond_idx, gm_array.shape)
    print(f"\ncond_idx={cond_idx} -> multi_idx={multi_idx}")
    
    # Try accessing properties with multi_idx
    try:
        mu_at_idx = result.MU.values[multi_idx]
        print(f"  result.MU.values[{multi_idx}] = {mu_at_idx}")
    except Exception as e:
        print(f"  Error accessing MU: {e}")
        
    # Check NP access
    try:
        np_at_idx = result.NP.values[multi_idx]
        print(f"  result.NP.values[{multi_idx}] shape = {np_at_idx.shape if hasattr(np_at_idx, 'shape') else 'scalar'}")
        print(f"  result.NP.values[{multi_idx}] = {np_at_idx}")
    except Exception as e:
        print(f"  Error accessing NP: {e}")