#!/usr/bin/env python
"""Quick test to compare CPU and GPU equilibrium results"""

from pycalphad import Database, equilibrium
import numpy as np

# Load database
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI']
phases = ['BCC_A2']

# Test condition
T = 500.0
P = 101325.0
X_TI = 0.5

# CPU test
print("Running CPU equilibrium...")
cpu_result = equilibrium(dbf, comps, phases, {'T': T, 'P': P, 'X(TI)': X_TI})

print(f"\nCPU GM: {cpu_result.GM.values}")
print(f"CPU MU: {cpu_result.MU.values[0][0]}")

# GPU test
try:
    from pycalphad.gpu import gpu_equilibrium
    print("\nRunning GPU equilibrium...")
    gpu_result = equilibrium(dbf, comps, phases, {'T': T, 'P': P, 'X(TI)': X_TI}, gpu=True)
    
    print(f"\nGPU GM: {gpu_result.GM.values}")
    print(f"GPU MU: {gpu_result.MU.values[0][0]}")
    
    # Print difference
    cpu_gm = cpu_result.GM.values[0][0][0]
    gpu_gm = gpu_result.GM.values[0][0][0]
    print(f"\nGM difference (GPU - CPU): {gpu_gm - cpu_gm:.6f} J/mol")
    print(f"Absolute GM difference: {abs(gpu_gm - cpu_gm):.6f} J/mol")
    
    if abs(gpu_gm - cpu_gm) <= 0.001:
        print("SUCCESS: GPU and CPU agree within 0.001 J/mol tolerance")
    else:
        print("FAIL: GPU and CPU differ by more than 0.001 J/mol")
except Exception as e:
    print(f"\nGPU test failed: {e}")
    import traceback
    traceback.print_exc()