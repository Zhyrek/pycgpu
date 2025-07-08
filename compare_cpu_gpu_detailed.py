#!/usr/bin/env python3
"""Detailed comparison of CPU vs GPU equilibrium calculations"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("=== Initial Conditions ===")
print(f"Components: {comps}")
print(f"Phases: {phases}")
print(f"Conditions: {eq_conditions}")

# Run CPU calculation with verbose output
print("\n=== CPU Calculation ===")
cpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=True, gpu=False, to='GM', calc_opts={'pdens': 50})
cpu_gm = float(cpu_result.GM.values)
cpu_np = cpu_result.NP.values.squeeze()
cpu_x = cpu_result.X.values.squeeze()
cpu_phase_name = cpu_result.Phase.values.squeeze()

print(f"\nCPU Results:")
print(f"  GM: {cpu_gm:.6f} J/mol")
print(f"  NP: {cpu_np}")
print(f"  X: {cpu_x}")
print(f"  Phase: {cpu_phase_name}")

# Run GPU calculation with verbose output
print("\n=== GPU Calculation ===")
try:
    gpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=True, gpu=True, to='GM', calc_opts={'pdens': 50})
    gpu_gm = float(gpu_result.GM.values)
    gpu_np = gpu_result.NP.values.squeeze()
    gpu_x = gpu_result.X.values.squeeze()
    gpu_phase_name = gpu_result.Phase.values.squeeze()
    
    print(f"\nGPU Results:")
    print(f"  GM: {gpu_gm:.6f} J/mol")
    print(f"  NP: {gpu_np}")
    print(f"  X: {gpu_x}")
    print(f"  Phase: {gpu_phase_name}")
    
    print(f"\n=== Comparison ===")
    print(f"GM difference: {abs(gpu_gm - cpu_gm):.6f} J/mol ({abs(gpu_gm - cpu_gm)/abs(cpu_gm)*100:.2f}%)")
    print(f"Target accuracy: 0.001 J/mol")
    print(f"PASS: {abs(gpu_gm - cpu_gm) < 0.001}")
    
except Exception as e:
    print(f"GPU calculation failed: {e}")
    import traceback
    traceback.print_exc()