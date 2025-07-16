#!/usr/bin/env python3
"""Double-check that the spurious term fix is working correctly"""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v

# Set up the calculation
dbf = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = ["BCC_A2"]
conditions = {
    v.X("TI"): 0.4,
    v.T: 1000,
    v.P: 101325,
    v.N: 1,
}

print("=== Running CPU calculation ===")
cpu_eq = equilibrium(dbf, comps, phases, conditions, verbose=False)
cpu_gm = float(cpu_eq.GM.values[0])

# Get Y values - may be single phase or two-phase
cpu_y_data = cpu_eq.Y.sel(vertex=0)
print(f"CPU Results:")
print(f"  GM = {cpu_gm:.6f} J/mol")
print(f"  Number of phases: {cpu_eq.Phase.size}")

# Print site fractions for each phase
for phase_idx in range(cpu_eq.Phase.size):
    phase_name = str(cpu_eq.Phase.values[phase_idx])
    y_values = cpu_eq.Y.sel(vertex=0, Phase=phase_name).values
    if y_values.size >= 2:
        print(f"  Phase {phase_name}: Y(NB) = {y_values[0]:.6f}, Y(TI) = {y_values[1]:.6f}")

print("\n=== Running GPU calculation ===")
gpu_eq = equilibrium(dbf, comps, phases, conditions, verbose=True, gpu=True)
gpu_gm = float(gpu_eq.GM.values[0])

print(f"\nGPU Results:")
print(f"  GM = {gpu_gm:.6f} J/mol")
print(f"  Number of phases: {gpu_eq.Phase.size}")

# Print site fractions for each phase
for phase_idx in range(gpu_eq.Phase.size):
    phase_name = str(gpu_eq.Phase.values[phase_idx])
    y_values = gpu_eq.Y.sel(vertex=0, Phase=phase_name).values
    if y_values.size >= 2:
        print(f"  Phase {phase_name}: Y(NB) = {y_values[0]:.6f}, Y(TI) = {y_values[1]:.6f}")

print("\n=== Comparison ===")
gm_diff = abs(gpu_gm - cpu_gm)
print(f"GM difference: {gm_diff:.6f} J/mol ({gm_diff/abs(cpu_gm)*100:.4f}%)")

# Check if differences are acceptable (within 0.1%)
tolerance = 0.001
if gm_diff/abs(cpu_gm) > tolerance:
    print(f"\nERROR: GM difference exceeds {tolerance*100}% tolerance!")
else:
    print(f"\nSUCCESS: GM values within {tolerance*100}% tolerance!")

# Compare phase amounts if same number of phases
if cpu_eq.Phase.size == gpu_eq.Phase.size and cpu_eq.Phase.size > 1:
    print("\n=== Phase Amounts ===")
    cpu_np = cpu_eq.NP.values
    gpu_np = gpu_eq.NP.values
    print(f"CPU phase amounts: {cpu_np}")
    print(f"GPU phase amounts: {gpu_np}")
    for i in range(len(cpu_np)):
        if cpu_np[i] > 1e-10:  # Only compare non-zero phases
            diff = abs(cpu_np[i] - gpu_np[i])
            print(f"  Phase {i} difference: {diff:.6f} ({diff/cpu_np[i]*100:.4f}%)")