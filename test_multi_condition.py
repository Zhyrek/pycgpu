#!/usr/bin/env python
"""Test multiple conditions to check for data overlap errors."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v
import time

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Test multiple conditions at once
# Test compositions from 0 to 1 in steps of 0.1
conditions = {
    v.T: 600,
    v.P: 101325,
    v.X('TI'): (0, 1, 0.1),  # 11 compositions: 0.0, 0.1, 0.2, ..., 1.0
    v.N: 1
}

print("Testing multi-condition equilibrium calculation...")
print(f"Temperature: 600K")
print(f"Compositions: X(TI) from 0.0 to 1.0 in steps of 0.1")
print("=" * 80)

# CPU calculation
print("\nCPU Calculation:")
cpu_start = time.time()
try:
    cpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50})
    cpu_time = time.time() - cpu_start
    print(f"CPU calculation successful! Time: {cpu_time:.3f}s")
    print(f"Result shape: {cpu_result.GM.shape}")
    print(f"GM values shape: {cpu_result.GM.values.shape}")
    
    # Extract results for each composition
    print("\nCPU Results:")
    for i in range(11):
        x_ti = i * 0.1
        gm = float(cpu_result.GM.values.flat[i])
        mu_nb = float(cpu_result.MU.values[i, 0])
        mu_ti = float(cpu_result.MU.values[i, 1])
        print(f"  X(TI)={x_ti:.1f}: GM={gm:.3f}, MU(NB)={mu_nb:.3f}, MU(TI)={mu_ti:.3f}")
except Exception as e:
    print(f"CPU calculation FAILED: {e}")
    import traceback
    traceback.print_exc()

# GPU calculation
print("\nGPU Calculation:")
gpu_start = time.time()
try:
    gpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50}, gpu=True)
    gpu_time = time.time() - gpu_start
    print(f"GPU calculation successful! Time: {gpu_time:.3f}s")
    print(f"Result shape: {gpu_result.GM.shape}")
    print(f"GM values shape: {gpu_result.GM.values.shape}")
    
    # Extract results for each composition
    print("\nGPU Results:")
    for i in range(11):
        x_ti = i * 0.1
        gm = float(gpu_result.GM.values.flat[i])
        mu_nb = float(gpu_result.MU.values[i, 0])
        mu_ti = float(gpu_result.MU.values[i, 1])
        print(f"  X(TI)={x_ti:.1f}: GM={gm:.3f}, MU(NB)={mu_nb:.3f}, MU(TI)={mu_ti:.3f}")
except Exception as e:
    print(f"GPU calculation FAILED: {e}")
    import traceback
    traceback.print_exc()

# Compare results if both succeeded
if 'cpu_result' in locals() and 'gpu_result' in locals():
    print("\n" + "=" * 80)
    print("COMPARISON:")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    print("\nDifferences:")
    max_gm_diff = 0
    max_mu_diff = 0
    
    for i in range(11):
        x_ti = i * 0.1
        cpu_gm = float(cpu_result.GM.values.flat[i])
        gpu_gm = float(gpu_result.GM.values.flat[i])
        gm_diff = abs(cpu_gm - gpu_gm)
        max_gm_diff = max(max_gm_diff, gm_diff)
        
        cpu_mu_nb = float(cpu_result.MU.values[i, 0])
        gpu_mu_nb = float(gpu_result.MU.values[i, 0])
        cpu_mu_ti = float(cpu_result.MU.values[i, 1])
        gpu_mu_ti = float(gpu_result.MU.values[i, 1])
        
        mu_nb_diff = abs(cpu_mu_nb - gpu_mu_nb)
        mu_ti_diff = abs(cpu_mu_ti - gpu_mu_ti)
        max_mu_diff = max(max_mu_diff, mu_nb_diff, mu_ti_diff)
        
        if gm_diff > 0.01 or mu_nb_diff > 0.01 or mu_ti_diff > 0.01:
            print(f"  X(TI)={x_ti:.1f}: GM diff={gm_diff:.6f}, MU(NB) diff={mu_nb_diff:.6f}, MU(TI) diff={mu_ti_diff:.6f}")
    
    print(f"\nMax GM difference: {max_gm_diff:.6f} J/mol")
    print(f"Max MU difference: {max_mu_diff:.6f} J/mol")
    
    if max_gm_diff < 0.1 and max_mu_diff < 0.1:
        print("\n✓ All differences are within acceptable tolerance (< 0.1 J/mol)")
    else:
        print("\n✗ Some differences exceed tolerance")

# Test a larger range to stress test
print("\n" + "=" * 80)
print("STRESS TEST: 101 compositions")
conditions_stress = {
    v.T: 600,
    v.P: 101325,
    v.X('TI'): (0, 1, 0.01),  # 101 compositions
    v.N: 1
}

print("\nGPU Stress Test:")
gpu_stress_start = time.time()
try:
    gpu_stress_result = equilibrium(db, components, phases, conditions_stress, calc_opts={'pdens': 50}, gpu=True)
    gpu_stress_time = time.time() - gpu_stress_start
    print(f"GPU stress test successful! Time: {gpu_stress_time:.3f}s")
    print(f"Result shape: {gpu_stress_result.GM.shape}")
    print(f"Processed {gpu_stress_result.GM.size} conditions in parallel")
except Exception as e:
    print(f"GPU stress test FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")