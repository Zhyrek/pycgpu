#!/usr/bin/env python3
"""Test infeasible equilibrium - X(TI)=0.4 with only TI-rich phases available"""
import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions - X(TI)=0.4 but BCC_A2 will converge to nearly pure TI
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("Testing infeasible equilibrium calculation...")
print("Condition: X(TI) = 0.4, but only BCC_A2 phase available")
print("This creates an infeasible system since BCC_A2 converges to ~pure TI")
print()

# Run CPU calculation
print("=== CPU Calculation ===")
cpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=False, gpu=False, calc_opts={'pdens': 50})
cpu_gm = float(cpu_result.GM.values.flatten()[0])
cpu_phase = str(cpu_result.Phase.values.flatten()[0])
cpu_x_ti = float(cpu_result.X.sel(component='TI').values.flatten()[0])
cpu_np = float(cpu_result.NP.values.flatten()[0])
print(f"GM: {cpu_gm:.6f} J/mol")
print(f"Phase: {cpu_phase}")
print(f"X(TI) in phase: {cpu_x_ti:.6f}")
print(f"Phase amount: {cpu_np:.6f}")
print(f"System X(TI): {cpu_x_ti * cpu_np:.6f} (should be 0.4)")

print("\n=== GPU Calculation ===")
gpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=False, gpu=True, calc_opts={'pdens': 50})
gpu_gm = float(gpu_result.GM.values.flatten()[0])
gpu_phase = str(gpu_result.Phase.values.flatten()[0])
gpu_x_ti = float(gpu_result.X.sel(component='TI').values.flatten()[0])
gpu_np = float(gpu_result.NP.values.flatten()[0])
print(f"GM: {gpu_gm:.6f} J/mol")
print(f"Phase: {gpu_phase}")
print(f"X(TI) in phase: {gpu_x_ti:.6f}")
print(f"Phase amount: {gpu_np:.6f}")
print(f"System X(TI): {gpu_x_ti * gpu_np:.6f} (should be 0.4)")

print(f"\n=== Comparison ===")
print(f"GM Difference: {abs(gpu_gm - cpu_gm):.6f} J/mol")
print(f"Phase amount difference: {abs(gpu_np - cpu_np):.6f}")
print(f"X(TI) difference: {abs(gpu_x_ti - cpu_x_ti):.6f}")