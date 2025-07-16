#!/usr/bin/env python
"""Test specific divergent cases after GPU consolidation fix."""

from pycalphad import Database, equilibrium
import numpy as np

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Test cases that showed divergence
test_cases = [
    {'T': 1000, 'X(TI)': 0.01},  # Original divergent case
    {'T': 1000, 'X(TI)': 0.05},
    {'T': 1500, 'X(TI)': 0.01},
    {'T': 1500, 'X(TI)': 0.05},
    {'T': 500, 'X(TI)': 0.1},
]

print("="*80)
print("TESTING DIVERGENT CASES AFTER GPU CONSOLIDATION FIX")
print("="*80)

max_diff = 0.0
num_divergent = 0

for i, conditions in enumerate(test_cases):
    print(f"\nTest case {i+1}: T={conditions['T']}K, X(TI)={conditions['X(TI)']}")
    
    # Add pressure
    conditions['P'] = 101325
    
    # Run CPU calculation
    eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = eq_cpu.GM.values.item()
    
    # Run GPU calculation
    eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = eq_gpu.GM.values.item()
    
    # Compare results
    diff = abs(cpu_gm - gpu_gm)
    print(f"  CPU GM: {cpu_gm:.6f} J/mol")
    print(f"  GPU GM: {gpu_gm:.6f} J/mol")
    print(f"  Difference: {diff:.6f} J/mol")
    
    if diff > 1e-6:
        num_divergent += 1
        print(f"  *** DIVERGENCE DETECTED ***")
    
    if diff > max_diff:
        max_diff = diff

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print(f"Total test cases: {len(test_cases)}")
print(f"Divergent cases: {num_divergent}")
print(f"Maximum difference: {max_diff:.6f} J/mol")
print(f"Tolerance: 1e-6 J/mol")

if num_divergent == 0:
    print("\n✓ ALL TESTS PASSED - GPU matches CPU within tolerance")
else:
    print(f"\n✗ {num_divergent} cases still show divergence")