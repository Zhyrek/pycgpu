#!/usr/bin/env python
"""Comprehensive test to prove the SVD tolerance fix works."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("COMPREHENSIVE TEST OF SVD TOLERANCE FIX")
print("=" * 60)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test multiple conditions
test_conditions = [
    {'X_TI': 0.9, 'T': 600},    # Original problematic condition
    {'X_TI': 0.8, 'T': 600},    # Different composition
    {'X_TI': 0.9, 'T': 800},    # Different temperature
    {'X_TI': 0.95, 'T': 700},   # Another test point
    {'X_TI': 0.85, 'T': 500},   # Lower temperature
]

print("\nTesting multiple conditions to verify fix robustness...")
print("\nCondition | CPU X(TI) | GPU X(TI) | Error | Status")
print("-" * 60)

all_passed = True
max_error = 0.0

for test in test_conditions:
    conditions = {v.X('TI'): test['X_TI'], v.T: test['T'], v.P: 101325}
    
    try:
        # Run CPU calculation
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
        
        # Run GPU calculation
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]
        
        # Calculate error
        error = abs(gpu_x_ti - cpu_x_ti)
        max_error = max(max_error, error)
        
        # Check if within tolerance
        status = "PASS" if error < 1e-6 else "FAIL"
        if status == "FAIL":
            all_passed = False
            
        print(f"X(TI)={test['X_TI']:.1f}, T={test['T']:3d}K | {cpu_x_ti:.6f} | {gpu_x_ti:.6f} | {error:.2e} | {status}")
        
    except Exception as e:
        print(f"X(TI)={test['X_TI']:.1f}, T={test['T']:3d}K | ERROR: {str(e)[:40]}...")
        all_passed = False

print("-" * 60)
print(f"\nMaximum error across all tests: {max_error:.2e}")
print(f"Overall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

# Additional verification - check that the problematic case is really fixed
print("\n" + "="*60)
print("DETAILED VERIFICATION OF ORIGINAL PROBLEM")
print("="*60)

conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("\nRunning detailed test for X(TI)=0.9, T=600K...")

# Get both results
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

# Extract all phase information
cpu_phases = result_cpu.Phase.values.flatten()
gpu_phases = result_gpu.Phase.values.flatten()
cpu_np = result_cpu.NP.values.flatten()
gpu_np = result_gpu.NP.values.flatten()
cpu_x = result_cpu.X.sel(component='TI').values.flatten()
gpu_x = result_gpu.X.sel(component='TI').values.flatten()

print("\nPhase comparison:")
print("CPU phases:", [p for p, n in zip(cpu_phases, cpu_np) if n > 1e-6])
print("GPU phases:", [p for p, n in zip(gpu_phases, gpu_np) if n > 1e-6])

print("\nPhase amounts:")
print("CPU:", [f"{n:.6f}" for n in cpu_np if n > 1e-6])
print("GPU:", [f"{n:.6f}" for n in gpu_np if n > 1e-6])

print("\nPhase compositions X(TI):")
print("CPU:", [f"{x:.6f}" for x, n in zip(cpu_x, cpu_np) if n > 1e-6])
print("GPU:", [f"{x:.6f}" for x, n in zip(gpu_x, gpu_np) if n > 1e-6])

# Calculate overall composition
cpu_overall = sum(n * x for n, x in zip(cpu_np, cpu_x) if n > 1e-6)
gpu_overall = sum(n * x for n, x in zip(gpu_np, gpu_x) if n > 1e-6)

print(f"\nOverall X(TI):")
print(f"CPU: {cpu_overall:.10f}")
print(f"GPU: {gpu_overall:.10f}")
print(f"Error: {abs(gpu_overall - cpu_overall):.2e}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if abs(gpu_overall - 0.9) < 1e-6 and abs(gpu_overall - cpu_overall) < 1e-6:
    print("✓ GPU correctly achieves X(TI) = 0.900000")
    print("✓ GPU matches CPU result exactly")
    print("✓ The SVD tolerance fix has resolved the issue!")
else:
    print("✗ GPU still has errors")
    print(f"  GPU gives X(TI) = {gpu_overall:.6f}")
    print(f"  Target is X(TI) = 0.900000")
    print(f"  Error = {abs(gpu_overall - 0.9):.6f}")