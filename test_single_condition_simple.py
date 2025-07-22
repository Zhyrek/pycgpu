#!/usr/bin/env python
"""Test single condition X(TI)=0.1, T=600K."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test condition: X(TI)=0.1, T=600K
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("Testing X(TI)=0.1, T=600K")
print("="*50)

# CPU result
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
gm_cpu = float(result_cpu.GM.values)
print(f"\nCPU Result:")
print(f"  GM: {gm_cpu:.6f}")

# Check active phases
phase_mask = result_cpu.NP.values.flatten() > 1e-6
active_phases_cpu = result_cpu.Phase.values.flatten()[phase_mask]
amounts_cpu = result_cpu.NP.values.flatten()[phase_mask]
print(f"  Active phases: {active_phases_cpu}")
print(f"  Phase amounts: {amounts_cpu}")

# Get compositions
x_values_cpu = result_cpu.X.values.reshape(-1, 3)[phase_mask]
for i, (phase, amt) in enumerate(zip(active_phases_cpu, amounts_cpu)):
    print(f"  {phase}: X(TI)={x_values_cpu[i, 1]:.6f}")

# GPU result
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gm_gpu = float(result_gpu.GM.values)
print(f"\nGPU Result:")
print(f"  GM: {gm_gpu:.6f}")

# Check active phases
phase_mask = result_gpu.NP.values.flatten() > 1e-6
active_phases_gpu = result_gpu.Phase.values.flatten()[phase_mask]
amounts_gpu = result_gpu.NP.values.flatten()[phase_mask]
print(f"  Active phases: {active_phases_gpu}")
print(f"  Phase amounts: {amounts_gpu}")

# Get compositions
x_values_gpu = result_gpu.X.values.reshape(-1, 3)[phase_mask]
for i, (phase, amt) in enumerate(zip(active_phases_gpu, amounts_gpu)):
    print(f"  {phase}: X(TI)={x_values_gpu[i, 1]:.6f}")

# Compare
print(f"\nComparison:")
print(f"  GM difference: {abs(gm_gpu - gm_cpu):.9f}")
print(f"  PASS" if abs(gm_gpu - gm_cpu) < 0.001 else f"  FAIL")

# Check chemical potentials
mu_cpu = result_cpu.MU.values.flatten()
mu_gpu = result_gpu.MU.values.flatten()
print(f"\nChemical potentials:")
print(f"  CPU: μ(NB)={mu_cpu[0]:.6f}, μ(TI)={mu_cpu[1]:.6f}")
print(f"  GPU: μ(NB)={mu_gpu[0]:.6f}, μ(TI)={mu_gpu[1]:.6f}")
print(f"  Differences: μ(NB)={abs(mu_gpu[0]-mu_cpu[0]):.9f}, μ(TI)={abs(mu_gpu[1]-mu_cpu[1]):.9f}")