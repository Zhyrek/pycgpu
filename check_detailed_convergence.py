#!/usr/bin/env python
"""Check detailed CPU vs GPU convergence behavior"""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v

# Simple binary system for testing
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'HCP_A3', 'LIQUID']

# Single condition for easy comparison
conds = {v.T: 1000, v.P: 101325, v.X('TI'): 0.4}

print("=== DETAILED CPU VS GPU CONVERGENCE ===\n")

# Run CPU calculation with verbose output
print("=== CPU CALCULATION ===")
cpu_result = equilibrium(dbf, comps, phases, conds, verbose=True, gpu=False)

# Extract CPU results
cpu_gm = cpu_result.GM.values.flatten()[0]
cpu_np = cpu_result.NP.values.flatten()
cpu_mu = cpu_result.MU.values.flatten()
cpu_phases = [p for p in cpu_result.Phase.values.flatten() if p != '' and p != '_FAKE_']

print(f"\nCPU FINAL RESULTS:")
print(f"  GM: {cpu_gm:.6f} J/mol")
print(f"  Phase amounts: {cpu_np[~np.isnan(cpu_np)]}")
print(f"  Chemical potentials: {cpu_mu}")
print(f"  Active phases: {cpu_phases}")

print("\n" + "="*80 + "\n")

# Run GPU calculation with verbose output
print("=== GPU CALCULATION ===")
try:
    gpu_result = equilibrium(dbf, comps, phases, conds, verbose=True, gpu=True)
    
    # Extract GPU results
    gpu_gm = gpu_result.GM.values.flatten()[0]
    gpu_np = gpu_result.NP.values.flatten()
    gpu_mu = gpu_result.MU.values.flatten()
    gpu_phases = [p for p in gpu_result.Phase.values.flatten() if p != '' and p != '_FAKE_']
    
    print(f"\nGPU FINAL RESULTS:")
    print(f"  GM: {gpu_gm:.6f} J/mol")
    print(f"  Phase amounts: {gpu_np[~np.isnan(gpu_np)]}")
    print(f"  Chemical potentials: {gpu_mu}")
    print(f"  Active phases: {gpu_phases}")
    
    # Compare results
    print(f"\n=== COMPARISON ===")
    print(f"GM difference: {abs(cpu_gm - gpu_gm):.6f} J/mol")
    print(f"MU difference: {np.max(np.abs(cpu_mu - gpu_mu)):.6f}")
    
except Exception as e:
    print(f"GPU calculation failed: {e}")