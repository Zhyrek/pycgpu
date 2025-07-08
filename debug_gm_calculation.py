#!/usr/bin/env python3
"""Debug GPU vs CPU GM calculation by adding detailed logging"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.gpu.gpu_equilibrium import clear_gpu_cache

# Clear GPU cache to ensure fresh functions
clear_gpu_cache()

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("=== Debugging GM Calculation ===")

# Run CPU calculation with verbose output to see algorithm steps
print("\n=== CPU Calculation ===")
cpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=True, gpu=False, to='GM', calc_opts={'pdens': 50})
cpu_gm = float(cpu_result.GM.values)
print(f"Final CPU GM: {cpu_gm:.6f} J/mol")

print("\n" + "="*50)

# Run GPU calculation - the debug output should show where it diverges
print("\n=== GPU Calculation ===")
gpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=True, gpu=True, to='GM', calc_opts={'pdens': 50})
gpu_gm = float(gpu_result.GM.values)
print(f"Final GPU GM: {gpu_gm:.6f} J/mol")

print(f"\n=== Final Comparison ===")
print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"GPU GM: {gpu_gm:.6f} J/mol")
print(f"Difference: {abs(gpu_gm - cpu_gm):.6f} J/mol")
print(f"Ratio: {gpu_gm / cpu_gm:.6f}")