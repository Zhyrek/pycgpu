#!/usr/bin/env python
"""Verify the GPU fix for single phase site fraction reset."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v

# Load database
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Test cases that require phase removal/consolidation
test_cases = [
    {'T': 600, 'X_TI': 0.1},   # Original failing case
    {'T': 600, 'X_TI': 0.2},   # Another composition
    {'T': 700, 'X_TI': 0.15},  # Different temperature
]

print("Testing GPU fix for single phase site fraction reset...")
print("=" * 70)

all_passed = True

for i, case in enumerate(test_cases):
    print(f"\nTest {i+1}: T={case['T']}K, X(TI)={case['X_TI']}")
    
    conds = {v.T: case['T'], v.P: 101325, v.X('TI'): case['X_TI'], v.N: 1}
    
    # CPU calculation
    cpu_result = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 50}, verbose=False)
    cpu_gm = float(cpu_result.GM.values)
    cpu_mu_nb = float(cpu_result.MU.sel(component='NB').values)
    cpu_mu_ti = float(cpu_result.MU.sel(component='TI').values)
    
    # GPU calculation
    gpu_result = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 50}, gpu=True, verbose=False)
    gpu_gm = float(gpu_result.GM.values)
    gpu_mu_nb = float(gpu_result.MU.sel(component='NB').values)
    gpu_mu_ti = float(gpu_result.MU.sel(component='TI').values)
    
    # Calculate differences
    gm_diff = abs(cpu_gm - gpu_gm)
    mu_nb_diff = abs(cpu_mu_nb - gpu_mu_nb)
    mu_ti_diff = abs(cpu_mu_ti - gpu_mu_ti)
    
    # Check if differences are within tolerance
    tolerance = 1e-6
    passed = gm_diff < tolerance and mu_nb_diff < tolerance and mu_ti_diff < tolerance
    
    print(f"  CPU: GM={cpu_gm:.6f}, MU(NB)={cpu_mu_nb:.6f}, MU(TI)={cpu_mu_ti:.6f}")
    print(f"  GPU: GM={gpu_gm:.6f}, MU(NB)={gpu_mu_nb:.6f}, MU(TI)={gpu_mu_ti:.6f}")
    print(f"  Differences: GM={gm_diff:.9f}, MU(NB)={mu_nb_diff:.9f}, MU(TI)={mu_ti_diff:.9f}")
    
    if passed:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
        all_passed = False

print("\n" + "=" * 70)
if all_passed:
    print("✓ All tests PASSED! GPU fix is working correctly.")
else:
    print("✗ Some tests FAILED. Further investigation needed.")