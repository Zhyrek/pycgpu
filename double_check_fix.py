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
cpu_y = cpu_eq.Y.sel(vertex=0).values
print(f"CPU Results:")
print(f"  GM = {cpu_gm:.6f} J/mol")
if len(cpu_y.shape) > 1:
    print(f"  Y(NB) = {float(cpu_y[0,0]):.6f}")
    print(f"  Y(TI) = {float(cpu_y[0,1]):.6f}")
else:
    print(f"  Y(NB) = {float(cpu_y[0]):.6f}")
    print(f"  Y(TI) = {float(cpu_y[1]):.6f}")

print("\n=== Running GPU calculation ===")
gpu_eq = equilibrium(dbf, comps, phases, conditions, verbose=True, gpu=True)
gpu_gm = float(gpu_eq.GM.values[0])
gpu_y = gpu_eq.Y.sel(vertex=0).values
print(f"\nGPU Results:")
print(f"  GM = {gpu_gm:.6f} J/mol")
if len(gpu_y.shape) > 1:
    print(f"  Y(NB) = {float(gpu_y[0,0]):.6f}")
    print(f"  Y(TI) = {float(gpu_y[0,1]):.6f}")
else:
    print(f"  Y(NB) = {float(gpu_y[0]):.6f}")
    print(f"  Y(TI) = {float(gpu_y[1]):.6f}")

print("\n=== Comparison ===")
gm_diff = abs(gpu_gm - cpu_gm)
if len(gpu_y.shape) > 1 and len(cpu_y.shape) > 1:
    y_nb_diff = abs(float(gpu_y[0,0]) - float(cpu_y[0,0]))
    y_ti_diff = abs(float(gpu_y[0,1]) - float(cpu_y[0,1]))
    cpu_y_nb = float(cpu_y[0,0])
    cpu_y_ti = float(cpu_y[0,1])
else:
    y_nb_diff = abs(float(gpu_y[0]) - float(cpu_y[0]))
    y_ti_diff = abs(float(gpu_y[1]) - float(cpu_y[1]))
    cpu_y_nb = float(cpu_y[0])
    cpu_y_ti = float(cpu_y[1])

print(f"GM difference: {gm_diff:.6f} J/mol ({gm_diff/abs(cpu_gm)*100:.4f}%)")
print(f"Y(NB) difference: {y_nb_diff:.6f} ({y_nb_diff/cpu_y_nb*100:.4f}%)")
print(f"Y(TI) difference: {y_ti_diff:.6f} ({y_ti_diff/cpu_y_ti*100:.4f}%)")

# Check if differences are acceptable (within 0.1%)
tolerance = 0.001
success = True
if gm_diff/abs(cpu_gm) > tolerance:
    print(f"\nERROR: GM difference exceeds {tolerance*100}% tolerance!")
    success = False
if y_nb_diff/cpu_y_nb > tolerance:
    print(f"\nERROR: Y(NB) difference exceeds {tolerance*100}% tolerance!")
    success = False
if y_ti_diff/cpu_y_ti > tolerance:
    print(f"\nERROR: Y(TI) difference exceeds {tolerance*100}% tolerance!")
    success = False

if success:
    print(f"\nSUCCESS: All values within {tolerance*100}% tolerance!")
else:
    print(f"\nFAILURE: Some values exceed {tolerance*100}% tolerance!")

# Also check phase amounts if it's a two-phase region
if cpu_eq.Phase.size > 1:
    print("\n=== Phase Amounts ===")
    cpu_np = cpu_eq.NP.values
    gpu_np = gpu_eq.NP.values
    print(f"CPU phase amounts: {cpu_np}")
    print(f"GPU phase amounts: {gpu_np}")
    if len(cpu_np) == len(gpu_np):
        for i in range(len(cpu_np)):
            diff = abs(cpu_np[i] - gpu_np[i])
            print(f"  Phase {i} difference: {diff:.6f} ({diff/cpu_np[i]*100:.4f}%)")