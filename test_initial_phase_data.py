#!/usr/bin/env python
"""Test initial phase data population for multiple conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Test with 6 phases and multiple conditions
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'AU2BI_HEX', 'RHOMBOHEDRAL_A7', 'BCC_A2']

# Test different conditions
conditions = {
    v.X('BI'): [0.3, 0.3, 0.3],  # Three identical conditions
    v.T: 600, 
    v.P: 101325
}

print(f"Testing initial phase data with {len(phases)} phases and 3 conditions")

# Run CPU calculation with verbose to see initial phase data
result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)

print("\nCPU Results:")
print(f"  GM values: {result.GM.values.flatten()}")
print(f"  Phase amounts shape: {result.NP.values.shape}")
print(f"  Number of phases per condition: {[np.sum(result.Phase.values[..., i, :] != '') for i in range(3)]}")

# Now run GPU calculation with verbose
print("\n" + "="*60)
print("GPU calculation with verbose output:")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)

print("\nGPU Results:")
print(f"  GM values: {result_gpu.GM.values.flatten()}")