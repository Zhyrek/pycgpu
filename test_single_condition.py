#!/usr/bin/env python
import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import time

# Load the database
db = Database('NbTi.tdb')

# Set up the components and phases
components = ['NB', 'TI', 'VA']
phases = filter_phases(db, components)

# Test single condition: X(TI)=0.1, T=600K
# This is outside the miscibility gap and should give single-phase BCC_A2
conditions = {
    v.P: 101325,
    v.T: 600,
    v.X('TI'): 0.1
}

print(f"Testing condition: X(TI)={conditions[v.X('TI')]}, T={conditions[v.T]}K")
print("Expected: Single phase BCC_A2")
print()

# Run CPU calculation
print("CPU Calculation:")
start_cpu = time.time()
cpu_result = equilibrium(db, components, phases, conditions, verbose=True)
cpu_time = time.time() - start_cpu

# Extract CPU results
cpu_gm = float(cpu_result.GM.values)
cpu_mu = cpu_result.MU.values.flatten()
cpu_phases = []
for phase in phases:
    mask = cpu_result.Phase.values.flatten() == phase
    if np.any(mask):
        phase_amt = float(cpu_result.NP.values.flatten()[mask][0])
        if phase_amt > 1e-6:
            cpu_phases.append((phase, phase_amt))

print(f"CPU GM: {cpu_gm:.6f}")
print(f"CPU MU(NB): {cpu_mu[0]:.6f}")
print(f"CPU MU(TI): {cpu_mu[1]:.6f}")
print(f"CPU Phases: {cpu_phases}")
print(f"CPU Time: {cpu_time:.3f}s")
print()

# Run GPU calculation
print("GPU Calculation:")
start_gpu = time.time()
gpu_result = equilibrium(db, components, phases, conditions, verbose=True, gpu=True)
gpu_time = time.time() - start_gpu

# Extract GPU results
gpu_gm = float(gpu_result.GM.values)
gpu_mu = gpu_result.MU.values.flatten()
gpu_phases = []
for phase in phases:
    mask = gpu_result.Phase.values.flatten() == phase
    if np.any(mask):
        phase_amt = float(gpu_result.NP.values.flatten()[mask][0])
        if phase_amt > 1e-6:
            gpu_phases.append((phase, phase_amt))

print(f"GPU GM: {gpu_gm:.6f}")
print(f"GPU MU(NB): {gpu_mu[0]:.6f}")
print(f"GPU MU(TI): {gpu_mu[1]:.6f}")
print(f"GPU Phases: {gpu_phases}")
print(f"GPU Time: {gpu_time:.3f}s")
print()

# Compare results
gm_diff = abs(gpu_gm - cpu_gm)
mu_nb_diff = abs(gpu_mu[0] - cpu_mu[0])
mu_ti_diff = abs(gpu_mu[1] - cpu_mu[1])

print("Differences:")
print(f"GM difference: {gm_diff:.6f}")
print(f"MU(NB) difference: {mu_nb_diff:.6f}")
print(f"MU(TI) difference: {mu_ti_diff:.6f}")

if gm_diff < 0.001:
    print("✓ GM values match (within tolerance)")
else:
    print("✗ GM values differ significantly!")

# Check phase stability
print()
print("Phase analysis:")
if len(cpu_phases) == 1 and len(gpu_phases) == 1:
    print("✓ Both calculations found single phase")
else:
    print(f"✗ Phase count mismatch: CPU={len(cpu_phases)}, GPU={len(gpu_phases)}")