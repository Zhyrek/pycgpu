#!/usr/bin/env python
"""Trace GM divergence for X(TI)=0.1, T=600K where GPU gives ~2x CPU value."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Failed condition: X(TI)=0.1, T=600K
# CPU GM: -24602.651695, GPU GM: -49222.648127 (almost exactly 2x)
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("="*60)
print("TRACING GM DIVERGENCE: X(TI)=0.1, T=600K")
print("Expected: CPU GM = -24602.65, GPU GM = -49222.65 (2x)")
print("="*60)

# Run CPU calculation with verbose output
print("\nCPU CALCULATION:")
print("-"*30)
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)

# Extract CPU results
cpu_gm = result_cpu.GM.values.flatten()[0]
cpu_phases = result_cpu.Phase.values.flatten()
cpu_np = result_cpu.NP.values.flatten()
cpu_x = result_cpu.X.values.flatten()
cpu_y = result_cpu.Y.values.flatten()

print(f"\nCPU RESULTS:")
print(f"  GM = {cpu_gm:.6f} J/mol")
print(f"  X(NB) = {cpu_x[0]:.6f}, X(TI) = {cpu_x[1]:.6f}")

# Find active phases
active_phases_cpu = []
for i, (phase, np_val) in enumerate(zip(cpu_phases, cpu_np)):
    if np_val > 1e-6:
        active_phases_cpu.append((phase, np_val, i))
        print(f"  Phase {phase}: NP = {np_val:.6f}")

# For each active phase, show site fractions and phase energy
print("\nCPU PHASE DETAILS:")
for phase, np_val, idx in active_phases_cpu:
    # Get site fractions for this phase
    y_start = idx * 2  # Assuming 2 site fractions per phase
    y_phase = cpu_y[y_start:y_start+2]
    print(f"  {phase}: Y = [{y_phase[0]:.6f}, {y_phase[1]:.6f}], NP = {np_val:.6f}")

# Calculate weighted energy
print(f"\nCPU ENERGY CALCULATION:")
total_energy = 0
for phase, np_val, idx in active_phases_cpu:
    # Note: Individual phase energies not directly available in result
    # But GM should be weighted average
    print(f"  {phase}: weight = {np_val:.6f}")

print(f"\n  Total GM = {cpu_gm:.6f} J/mol")

# Run GPU calculation with verbose output
print("\n" + "="*60)
print("GPU CALCULATION:")
print("-"*30)
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)

# Extract GPU results
gpu_gm = result_gpu.GM.values.flatten()[0]
gpu_phases = result_gpu.Phase.values.flatten()
gpu_np = result_gpu.NP.values.flatten()
gpu_x = result_gpu.X.values.flatten()
gpu_y = result_gpu.Y.values.flatten()

print(f"\nGPU RESULTS:")
print(f"  GM = {gpu_gm:.6f} J/mol")
print(f"  X(NB) = {gpu_x[0]:.6f}, X(TI) = {gpu_x[1]:.6f}")

# Find active phases
active_phases_gpu = []
for i, (phase, np_val) in enumerate(zip(gpu_phases, gpu_np)):
    if np_val > 1e-6:
        active_phases_gpu.append((phase, np_val, i))
        print(f"  Phase {phase}: NP = {np_val:.6f}")

# For each active phase, show site fractions
print("\nGPU PHASE DETAILS:")
for phase, np_val, idx in active_phases_gpu:
    # Get site fractions for this phase
    y_start = idx * 2  # Assuming 2 site fractions per phase
    y_phase = gpu_y[y_start:y_start+2]
    print(f"  {phase}: Y = [{y_phase[0]:.6f}, {y_phase[1]:.6f}], NP = {np_val:.6f}")

# Compare results
print("\n" + "="*60)
print("COMPARISON:")
print("-"*30)
print(f"GM difference: {gpu_gm - cpu_gm:.6f} J/mol")
print(f"GM ratio: {gpu_gm / cpu_gm:.6f} (expected ~2.0)")
print(f"X(TI) difference: {abs(gpu_x[1] - cpu_x[1]):.6e}")

# Check if phases match
print("\nPhase comparison:")
cpu_active = [(p, np) for p, np, _ in active_phases_cpu]
gpu_active = [(p, np) for p, np, _ in active_phases_gpu]
print(f"  CPU phases: {cpu_active}")
print(f"  GPU phases: {gpu_active}")

# Site fraction comparison
print("\nSite fraction comparison:")
for i, ((p_cpu, np_cpu, idx_cpu), (p_gpu, np_gpu, idx_gpu)) in enumerate(zip(active_phases_cpu, active_phases_gpu)):
    y_cpu = cpu_y[idx_cpu*2:idx_cpu*2+2]
    y_gpu = gpu_y[idx_gpu*2:idx_gpu*2+2]
    print(f"  Phase {i}: Y_CPU = [{y_cpu[0]:.6f}, {y_cpu[1]:.6f}], Y_GPU = [{y_gpu[0]:.6f}, {y_gpu[1]:.6f}]")
    print(f"           Y diff = [{abs(y_gpu[0]-y_cpu[0]):.6e}, {abs(y_gpu[1]-y_cpu[1]):.6e}]")

print("\n" + "="*60)
print("HYPOTHESIS: GPU is calculating GM = sum(NP * G) instead of weighted average")
print("This would give exactly 2x the energy when there are 2 phases with equal amounts")
print("="*60)