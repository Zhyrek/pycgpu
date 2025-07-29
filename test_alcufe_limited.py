#!/usr/bin/env python
"""Test CPU vs GPU for Al-Cu-Fe with limited phases."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v
import warnings
warnings.filterwarnings('ignore')

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Use only simple phases that might compile
simple_phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

print("=" * 80)
print("Al-Cu-Fe System Test with Limited Phases")
print("=" * 80)
print(f"Testing phases: {simple_phases}")
print(f"Temperature: 1000°C (1273.15K)")

# Test grid
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): (0, 1, 0.1),  # 11 points
    v.X('CU'): (0, 1, 0.1),  # 11 points
    v.N: 1
}

# Run CPU
print("\nRunning CPU calculation...")
try:
    cpu_result = equilibrium(db, components, simple_phases, conditions, calc_opts={'pdens': 50})
    print(f"CPU Success! Result shape: {cpu_result.GM.shape}")
    cpu_success = True
except Exception as e:
    print(f"CPU Failed: {e}")
    cpu_success = False

# Run GPU
print("\nRunning GPU calculation...")
try:
    gpu_result = equilibrium(db, components, simple_phases, conditions, calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print(f"GPU Success! Result shape: {gpu_result.GM.shape}")
    gpu_success = True
except Exception as e:
    print(f"GPU Failed: {e}")
    gpu_success = False
    
    # Try even simpler - just LIQUID
    print("\nTrying with just LIQUID phase...")
    try:
        gpu_result = equilibrium(db, components, ['LIQUID'], conditions, calc_opts={'pdens': 50}, gpu=True, verbose=False)
        print(f"GPU Success with LIQUID only! Result shape: {gpu_result.GM.shape}")
        gpu_success = True
        simple_phases = ['LIQUID']
        # Also recalculate CPU with just LIQUID
        cpu_result = equilibrium(db, components, ['LIQUID'], conditions, calc_opts={'pdens': 50})
    except Exception as e2:
        print(f"GPU Failed even with just LIQUID: {e2}")

# Compare if both succeeded
if cpu_success and gpu_success:
    print("\n" + "=" * 80)
    print("Comparison Results:")
    print("=" * 80)
    
    # Grid points
    al_values = np.linspace(0, 1, 11)
    cu_values = np.linspace(0, 1, 11)
    
    matching = 0
    total = 0
    max_diff = 0.0
    
    for i, x_al in enumerate(al_values):
        for j, x_cu in enumerate(cu_values):
            x_fe = 1 - x_al - x_cu
            if x_fe < -1e-10:
                continue
                
            idx = j * 11 + i
            gm_cpu = cpu_result.GM.values.flatten()[idx]
            gm_gpu = gpu_result.GM.values.flatten()[idx]
            diff = abs(gm_cpu - gm_gpu)
            
            total += 1
            if diff < 1.0:  # 1 J/mol tolerance
                matching += 1
            else:
                if diff > max_diff:
                    max_diff = diff
                    max_diff_comp = (x_al, x_cu, x_fe)
            
            # Print sample points
            if (i % 5 == 0 and j % 5 == 0) or diff > 1.0:
                print(f"X(AL)={x_al:.1f}, X(CU)={x_cu:.1f}: GM_CPU={gm_cpu:.1f}, GM_GPU={gm_gpu:.1f}, Δ={diff:.3f}")
    
    print(f"\nTotal valid points: {total}")
    print(f"Matching points: {matching} ({100*matching/total:.1f}%)")
    if max_diff > 1.0:
        print(f"Maximum difference: {max_diff:.1f} J/mol at X(AL)={max_diff_comp[0]:.3f}, X(CU)={max_diff_comp[1]:.3f}")
else:
    print("\nCannot compare - one or both calculations failed")

print("\nConclusion:")
if not gpu_success:
    print("GPU compilation fails for Al-Cu-Fe system with complex phases.")
    print("This appears to be a compilation issue with the generated CUDA code.")
elif matching == total:
    print("GPU and CPU results match within tolerance!")
else:
    print(f"GPU and CPU results differ for {total-matching} out of {total} points.")