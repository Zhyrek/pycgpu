#!/usr/bin/env python
"""Quick test of GPU vs CPU across key conditions."""

from pycalphad import Database, equilibrium
from pycalphad.core.debug_output import reset_debug_session, close_debug_output
import numpy as np

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Test conditions - focusing on key cases
test_conditions = [
    # Low Ti fraction (previous divergence case)
    {'T': 1000, 'X(TI)': 0.01},
    {'T': 1500, 'X(TI)': 0.01},
    
    # Medium Ti fraction
    {'T': 1000, 'X(TI)': 0.1},
    {'T': 1500, 'X(TI)': 0.1},
    {'T': 2000, 'X(TI)': 0.5},
    
    # High Ti fraction
    {'T': 1000, 'X(TI)': 0.9},
    {'T': 1500, 'X(TI)': 0.9},
    
    # Edge cases
    {'T': 500, 'X(TI)': 0.001},
    {'T': 2500, 'X(TI)': 0.999},
]

print("GPU vs CPU Quick Comparison")
print("="*60)

passed = 0
failed = 0
max_diff = 0.0

for i, conditions in enumerate(test_conditions):
    conditions['P'] = 101325
    
    # Reset debug output
    reset_debug_session()
    
    # CPU calculation
    eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = eq_cpu.GM.values.item()
    
    # GPU calculation
    eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = eq_gpu.GM.values.item()
    
    # Compare
    diff = abs(cpu_gm - gpu_gm)
    
    print(f"T={conditions['T']:4d}K, X(TI)={conditions['X(TI)']:5.3f}: ", end='')
    print(f"CPU={cpu_gm:10.1f}, GPU={gpu_gm:10.1f}, Diff={diff:8.1e}", end='')
    
    if diff < 1e-6:
        print(" ✓")
        passed += 1
    else:
        print(" ✗")
        failed += 1
        
    if diff > max_diff:
        max_diff = diff

print("="*60)
print(f"Passed: {passed}/{len(test_conditions)}")
print(f"Failed: {failed}/{len(test_conditions)}")
print(f"Max difference: {max_diff:.2e} J/mol")

if failed == 0:
    print("\n✓ SUCCESS: GPU matches CPU perfectly!")
else:
    print(f"\n✗ {failed} conditions showed differences")