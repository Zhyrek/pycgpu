#!/usr/bin/env python
"""Test single condition GPU calculation to debug SystemSpec issue."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Test with single condition
conditions = {
    v.T: 600,
    v.P: 101325,
    v.X('TI'): 0.1,  # Single composition
    v.N: 1
}

print("Testing GPU with single condition...")
print("Composition: X(TI) = 0.1")
print("=" * 60)

# GPU calculation
try:
    gpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50}, gpu=True, verbose=True)
    print(f"\nGPU calculation successful!")
    print(f"Result shape: {gpu_result.GM.shape}")
    print(f"GM value: {gpu_result.GM.values[0, 0, 0, 0]:.3f}")
    print(f"MU(NB): {gpu_result.MU.values[0, 0, 0, 0, 0]:.3f}")
    print(f"MU(TI): {gpu_result.MU.values[0, 0, 0, 0, 1]:.3f}")
    
except Exception as e:
    print(f"GPU calculation FAILED: {e}")
    import traceback
    traceback.print_exc()

# Compare with CPU
print("\n" + "=" * 60)
print("Testing CPU with same condition...")
try:
    cpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50})
    print(f"CPU calculation successful!")
    print(f"GM value: {cpu_result.GM.values[0, 0, 0, 0]:.3f}")
    print(f"MU(NB): {cpu_result.MU.values[0, 0, 0, 0, 0]:.3f}")
    print(f"MU(TI): {cpu_result.MU.values[0, 0, 0, 0, 1]:.3f}")
    
    # Compare
    gm_diff = abs(gpu_result.GM.values[0, 0, 0, 0] - cpu_result.GM.values[0, 0, 0, 0])
    print(f"\nGM difference: {gm_diff:.6f} J/mol")
    
except Exception as e:
    print(f"CPU calculation FAILED: {e}")