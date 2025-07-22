#!/usr/bin/env python
"""Test multiple conditions to check for data overlap errors - fixed version."""

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
    
    # Extract results for each composition
    print("\nCPU Results:")
    x_ti_values = np.linspace(0, 1, 11)
    for i, x_ti in enumerate(x_ti_values):
        # Access the data correctly from 4D array
        gm = cpu_result.GM.values[0, 0, 0, i]
        mu_nb = cpu_result.MU.values[0, 0, 0, i, 0]
        mu_ti = cpu_result.MU.values[0, 0, 0, i, 1]
        print(f"  X(TI)={x_ti:.1f}: GM={gm:.3f}, MU(NB)={mu_nb:.3f}, MU(TI)={mu_ti:.3f}")
        
    cpu_gm_values = cpu_result.GM.values[0, 0, 0, :]
    cpu_mu_values = cpu_result.MU.values[0, 0, 0, :, :]
except Exception as e:
    print(f"CPU calculation FAILED: {e}")
    import traceback
    traceback.print_exc()
    cpu_gm_values = None
    cpu_mu_values = None

# GPU calculation
print("\nGPU Calculation:")
gpu_start = time.time()
try:
    gpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50}, gpu=True)
    gpu_time = time.time() - gpu_start
    print(f"GPU calculation successful! Time: {gpu_time:.3f}s")
    print(f"Result shape: {gpu_result.GM.shape}")
    
    # Extract results for each composition - note GPU may have different shape
    print("\nGPU Results:")
    # GPU returns 10 conditions (skips pure endpoints)
    if gpu_result.GM.shape[-1] == 10:
        x_ti_values_gpu = np.linspace(0.1, 0.9, 9)  # GPU computed 0.1 to 0.9
        print("Note: GPU skipped pure endpoints (X=0 and X=1)")
        for i in range(9):
            x_ti = x_ti_values_gpu[i]
            gm = gpu_result.GM.values[0, 0, 0, i]
            mu_nb = gpu_result.MU.values[0, 0, 0, i, 0]
            mu_ti = gpu_result.MU.values[0, 0, 0, i, 1]
            print(f"  X(TI)={x_ti:.1f}: GM={gm:.3f}, MU(NB)={mu_nb:.3f}, MU(TI)={mu_ti:.3f}")
    else:
        x_ti_values_gpu = np.linspace(0, 1, gpu_result.GM.shape[-1])
        for i, x_ti in enumerate(x_ti_values_gpu):
            gm = gpu_result.GM.values[0, 0, 0, i]
            mu_nb = gpu_result.MU.values[0, 0, 0, i, 0]
            mu_ti = gpu_result.MU.values[0, 0, 0, i, 1]
            print(f"  X(TI)={x_ti:.1f}: GM={gm:.3f}, MU(NB)={mu_nb:.3f}, MU(TI)={mu_ti:.3f}")
    
    gpu_gm_values = gpu_result.GM.values[0, 0, 0, :]
    gpu_mu_values = gpu_result.MU.values[0, 0, 0, :, :]
except Exception as e:
    print(f"GPU calculation FAILED: {e}")
    import traceback
    traceback.print_exc()
    gpu_gm_values = None
    gpu_mu_values = None

# Compare results if both succeeded
if cpu_gm_values is not None and gpu_gm_values is not None:
    print("\n" + "=" * 80)
    print("COMPARISON:")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    print("\nDifferences (for overlapping compositions):")
    max_gm_diff = 0
    max_mu_diff = 0
    
    # Compare only the overlapping range (X=0.1 to X=0.9)
    for i in range(1, 10):  # Skip X=0.0 and X=1.0
        x_ti = i * 0.1
        cpu_gm = cpu_gm_values[i]
        # GPU index mapping: CPU index 1-9 maps to GPU index 0-8
        gpu_idx = i - 1
        if gpu_idx < len(gpu_gm_values):
            gpu_gm = gpu_gm_values[gpu_idx]
            gm_diff = abs(cpu_gm - gpu_gm)
            max_gm_diff = max(max_gm_diff, gm_diff)
            
            cpu_mu_nb = cpu_mu_values[i, 0]
            gpu_mu_nb = gpu_mu_values[gpu_idx, 0]
            cpu_mu_ti = cpu_mu_values[i, 1]
            gpu_mu_ti = gpu_mu_values[gpu_idx, 1]
            
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

# Test with fewer conditions to check for overlaps
print("\n" + "=" * 80)
print("SMALLER TEST: 5 compositions")
conditions_small = {
    v.T: 600,
    v.P: 101325,
    v.X('TI'): [0.1, 0.3, 0.5, 0.7, 0.9],  # 5 specific compositions
    v.N: 1
}

print("\nGPU Small Test:")
gpu_small_start = time.time()
try:
    gpu_small_result = equilibrium(db, components, phases, conditions_small, calc_opts={'pdens': 50}, gpu=True)
    gpu_small_time = time.time() - gpu_small_start
    print(f"GPU small test successful! Time: {gpu_small_time:.3f}s")
    print(f"Result shape: {gpu_small_result.GM.shape}")
    print(f"Processed {gpu_small_result.GM.values[0,0,0,:].size} conditions")
    
    # Show results
    x_ti_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    for i in range(len(x_ti_list)):
        if i < gpu_small_result.GM.shape[-1]:
            x_ti = x_ti_list[i]
            gm = gpu_small_result.GM.values[0, 0, 0, i]
            print(f"  X(TI)={x_ti}: GM={gm:.3f} J/mol")
except Exception as e:
    print(f"GPU small test FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Multi-condition test complete!")