#!/usr/bin/env python
"""Trace GPU convergence issue in detail."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TRACING GPU CONVERGENCE ISSUE")
print("=" * 60)

# Clear cache
cache_dir = os.path.expanduser('~/.cache/pycalphad_gpu')
if os.path.exists(cache_dir):
    import shutil
    shutil.rmtree(cache_dir)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test the problematic condition
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("\nTest condition: X(TI)=0.9, T=600K")

# First run CPU to get expected result
print("\n" + "="*40)
print("CPU CALCULATION")
print("="*40)
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = result_cpu.GM.values.flatten()[0]
print(f"CPU GM: {cpu_gm:.6f} J/mol")

# Check CPU composition
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
cpu_phases = result_cpu.Phase.values.flatten()

print("\nCPU phases:")
for i, (phase, np_val, x_ti) in enumerate(zip(cpu_phases, cpu_np, cpu_x_ti)):
    if np_val > 1e-12:
        print(f"  Phase {i}: {phase}, NP={np_val:.6f}, X(TI)={x_ti:.6f}")

overall_x_ti = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)
print(f"\nCPU overall X(TI): {overall_x_ti:.6f}")

# Now run GPU
print("\n" + "="*40)
print("GPU CALCULATION")  
print("="*40)
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gpu_gm = result_gpu.GM.values.flatten()[0]
print(f"GPU GM: {gpu_gm:.6f} J/mol")

# Check GPU composition
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
gpu_np = result_gpu.NP.values.flatten()
gpu_phases = result_gpu.Phase.values.flatten()

print("\nGPU phases:")
for i, (phase, np_val, x_ti) in enumerate(zip(gpu_phases, gpu_np, gpu_x_ti)):
    if np_val > 1e-12:
        print(f"  Phase {i}: {phase}, NP={np_val:.6f}, X(TI)={x_ti:.6f}")

overall_x_ti_gpu = sum(np_val * x_ti for np_val, x_ti in zip(gpu_np, gpu_x_ti) if np_val > 1e-12)
print(f"\nGPU overall X(TI): {overall_x_ti_gpu:.6f}")

print("\n" + "="*40)
print("COMPARISON")
print("="*40)
print(f"GM difference: {abs(cpu_gm - gpu_gm):.6f} J/mol")
print(f"X(TI) difference: {abs(overall_x_ti - overall_x_ti_gpu):.6f}")
print(f"Composition constraint error (GPU): {abs(overall_x_ti_gpu - 0.9):.6f}")

# Check if both have same phases
cpu_active_phases = [(phase, np_val) for phase, np_val in zip(cpu_phases, cpu_np) if np_val > 1e-12]
gpu_active_phases = [(phase, np_val) for phase, np_val in zip(gpu_phases, gpu_np) if np_val > 1e-12]

print(f"\nCPU active phases: {[p[0] for p in cpu_active_phases]}")
print(f"GPU active phases: {[p[0] for p in gpu_active_phases]}")

if len(cpu_active_phases) != len(gpu_active_phases):
    print("\n✗ WARNING: Different number of phases!")
else:
    print("\n✓ Same number of phases")

# Check phase amounts
print("\nPhase amount differences:")
for i, ((cpu_phase, cpu_np), (gpu_phase, gpu_np)) in enumerate(zip(cpu_active_phases, gpu_active_phases)):
    print(f"  Phase {i}: CPU NP={cpu_np:.6f}, GPU NP={gpu_np:.6f}, diff={abs(cpu_np - gpu_np):.6f}")