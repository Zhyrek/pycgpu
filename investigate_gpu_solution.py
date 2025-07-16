#!/usr/bin/env python3
"""Investigate why GPU gives lower energy than CPU"""

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

print("=== Investigating GPU vs CPU Solutions ===\n")

# Run CPU calculation
print("CPU Calculation:")
cpu_eq = equilibrium(dbf, comps, phases, conditions, verbose=False)
cpu_gm = float(cpu_eq.GM.values[0])
cpu_np = cpu_eq.NP.values[0]
cpu_y = cpu_eq.Y.values[0]

print(f"  GM = {cpu_gm:.6f} J/mol")
print(f"  Phase amounts: {cpu_np}")
print(f"  Site fractions:")
for i, phase in enumerate(cpu_np):
    if phase[0] > 1e-10:
        print(f"    Phase {i}: Y(NB) = {cpu_y[i][0]:.6f}, Y(TI) = {cpu_y[i][1]:.6f}")

# Run GPU calculation
print("\nGPU Calculation:")
gpu_eq = equilibrium(dbf, comps, phases, conditions, verbose=False, gpu=True)
gpu_gm = float(gpu_eq.GM.values[0])
gpu_np = gpu_eq.NP.values[0]
gpu_y = gpu_eq.Y.values[0]

print(f"  GM = {gpu_gm:.6f} J/mol")
print(f"  Phase amounts: {gpu_np}")
print(f"  Site fractions:")
for i, phase in enumerate(gpu_np):
    if phase[0] > 1e-10:
        print(f"    Phase {i}: Y(NB) = {gpu_y[i][0]:.6f}, Y(TI) = {gpu_y[i][1]:.6f}")

# Compare
print("\n=== Comparison ===")
print(f"Energy difference: {gpu_gm - cpu_gm:.6f} J/mol")
print(f"GPU energy is {'LOWER' if gpu_gm < cpu_gm else 'HIGHER'} than CPU")

# Check if site fractions are different
print("\nSite fraction differences:")
for i in range(len(cpu_np)):
    if cpu_np[i][0] > 1e-10 and gpu_np[i][0] > 1e-10:
        y_nb_diff = gpu_y[i][0] - cpu_y[i][0]
        y_ti_diff = gpu_y[i][1] - cpu_y[i][1]
        print(f"  Phase {i}: ΔY(NB) = {y_nb_diff:.6f}, ΔY(TI) = {y_ti_diff:.6f}")

# Check phase amounts
print("\nPhase amount differences:")
for i in range(len(cpu_np)):
    if cpu_np[i][0] > 1e-10 or gpu_np[i][0] > 1e-10:
        diff = gpu_np[i][0] - cpu_np[i][0]
        print(f"  Phase {i}: CPU = {cpu_np[i][0]:.6f}, GPU = {gpu_np[i][0]:.6f}, Δ = {diff:.6f}")

# Calculate energies manually to verify
print("\n=== Manual Energy Verification ===")
from pycalphad import Model

# Create model
mod = Model(dbf, comps, "BCC_A2")

# For single phase at equilibrium composition
if cpu_np[0][0] > 0.99:  # Single phase
    print("Single phase detected")
    
    # CPU energy
    state_vars = {'T': 1000, 'P': 101325, 'Y(BCC_A2,0,NB)': cpu_y[0][0], 'Y(BCC_A2,0,TI)': cpu_y[0][1]}
    cpu_manual = float(mod.G.subs(state_vars))
    print(f"CPU energy (manual): {cpu_manual:.6f} J/mol")
    print(f"CPU energy (equilibrium): {cpu_gm:.6f} J/mol")
    print(f"Difference: {abs(cpu_manual - cpu_gm):.6f} J/mol")
    
    # GPU energy
    state_vars = {'T': 1000, 'P': 101325, 'Y(BCC_A2,0,NB)': gpu_y[0][0], 'Y(BCC_A2,0,TI)': gpu_y[0][1]}
    gpu_manual = float(mod.G.subs(state_vars))
    print(f"\nGPU energy (manual): {gpu_manual:.6f} J/mol")
    print(f"GPU energy (equilibrium): {gpu_gm:.6f} J/mol")
    print(f"Difference: {abs(gpu_manual - gpu_gm):.6f} J/mol")
    
    print(f"\nManual calculation difference: {gpu_manual - cpu_manual:.6f} J/mol")
    
else:  # Two phase
    print("Two phase detected - checking each phase...")
    # Would need to do weighted average