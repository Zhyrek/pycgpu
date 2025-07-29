#!/usr/bin/env python
"""Final Al-Cu-Fe comparison test with GPU code fix."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v
import warnings
import time
import sys
import os

# Redirect stdout to suppress debug output
old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

warnings.filterwarnings('ignore')
os.environ['PYCALPHAD_CPU_DEBUG'] = '0'

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = list(db.phases.keys())

# Define conditions
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): (0, 1, 0.1),
    v.X('CU'): (0, 1, 0.1),
    v.N: 1
}

# Run calculations
cpu_success = False
gpu_success = False

try:
    start_cpu = time.time()
    cpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50})
    cpu_time = time.time() - start_cpu
    cpu_success = True
except Exception as e:
    cpu_error = str(e)

try:
    start_gpu = time.time()
    gpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50}, gpu=True, verbose=False)
    gpu_time = time.time() - start_gpu
    gpu_success = True
except Exception as e:
    gpu_error = str(e)

# Restore stdout
sys.stdout = old_stdout

# Print results
print("="*60)
print("Al-Cu-Fe System Final Test Results")
print("="*60)
print(f"Temperature: 1000°C (1273.15K)")
print(f"Composition grid: X(AL) and X(CU) from 0 to 1 in steps of 0.1")
print(f"Number of phases in database: {len(phases)}")

print(f"\nCPU calculation: {'SUCCESS' if cpu_success else 'FAILED'}")
if cpu_success:
    print(f"  Time: {cpu_time:.1f} seconds")
    print(f"  Result shape: {cpu_result.GM.shape}")

print(f"\nGPU calculation: {'SUCCESS' if gpu_success else 'FAILED'}")
if gpu_success:
    print(f"  Time: {gpu_time:.1f} seconds")
    print(f"  Result shape: {gpu_result.GM.shape}")
    if cpu_success:
        print(f"  Speedup: {cpu_time/gpu_time:.1f}x")

# Compare if both succeeded
if cpu_success and gpu_success:
    print("\n" + "="*60)
    print("Comparison Results:")
    print("="*60)
    
    # Flatten results
    gm_cpu = cpu_result.GM.values.flatten()
    gm_gpu = gpu_result.GM.values.flatten()
    
    # Calculate differences
    diff = np.abs(gm_cpu - gm_gpu)
    tolerance = 1.0  # 1 J/mol
    
    matching = np.sum(diff < tolerance)
    total = len(diff)
    
    print(f"\nTotal points: {total}")
    print(f"Points within {tolerance} J/mol: {matching} ({100*matching/total:.1f}%)")
    print(f"\nStatistics:")
    print(f"  Max difference: {np.max(diff):.3f} J/mol")
    print(f"  Mean difference: {np.mean(diff):.3f} J/mol")
    print(f"  Median difference: {np.median(diff):.3f} J/mol")
    
    # Check if results match
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    
    if matching == total:
        print("✓ PERFECT MATCH: GPU results match CPU within tolerance!")
        print("✓ The GPU code generation fix for multi-sublattice phases is working correctly!")
    elif matching / total > 0.95:
        print(f"✓ EXCELLENT: {100*matching/total:.1f}% of points match within tolerance")
        print("✓ The GPU code generation fix is working well")
    elif matching / total > 0.80:
        print(f"⚠ GOOD: {100*matching/total:.1f}% of points match within tolerance")
        print("⚠ The GPU code mostly works but may need refinement")
    else:
        print(f"✗ POOR: Only {100*matching/total:.1f}% of points match within tolerance")
        print("✗ The GPU code needs further debugging")
else:
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    if not cpu_success:
        print("✗ CPU calculation failed")
    if not gpu_success:
        print("✗ GPU calculation failed - code generation may still have issues")