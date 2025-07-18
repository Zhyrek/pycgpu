#!/usr/bin/env python
"""Simple trace of X(TI)=0.9, T=600K to find divergence point."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA'] 
phases = filter_phases(dbf, comps)

# Test X(TI)=0.9, T=600K which shows ~19 J/mol error
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("COMPARING X(TI)=0.9, T=600K")
print("=" * 60)

# Create output file
with open('/tmp/trace_output.txt', 'w') as f:
    f.write("X(TI)=0.9, T=600K CPU vs GPU Trace\n")
    f.write("=" * 60 + "\n\n")

# Run CPU with verbose output to capture trace
print("\n1. Running CPU calculation...")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)

cpu_gm = result_cpu.GM.values.flatten()[0]
cpu_phases = result_cpu.NP.values.flatten()

print(f"\nCPU Results:")
print(f"  GM = {cpu_gm:.6f} J/mol")
print(f"  Phase fractions: {[f'{p:.6f}' for p in cpu_phases if p > 1e-6]}")

# Run GPU with verbose output
print("\n2. Running GPU calculation...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)

gpu_gm = result_gpu.GM.values.flatten()[0]
gpu_phases = result_gpu.NP.values.flatten()

print(f"\nGPU Results:")
print(f"  GM = {gpu_gm:.6f} J/mol")
print(f"  Phase fractions: {[f'{p:.6f}' for p in gpu_phases if p > 1e-6]}")

error = abs(cpu_gm - gpu_gm)
print(f"\nError: {error:.6f} J/mol")

# Save key comparison points
with open('/tmp/trace_output.txt', 'a') as f:
    f.write(f"CPU GM: {cpu_gm:.6f} J/mol\n")
    f.write(f"GPU GM: {gpu_gm:.6f} J/mol\n")
    f.write(f"Error: {error:.6f} J/mol\n\n")
    
    f.write("CPU Phase fractions:\n")
    for i, p in enumerate(cpu_phases):
        if p > 1e-6:
            f.write(f"  Phase {i}: {p:.6f}\n")
    
    f.write("\nGPU Phase fractions:\n")
    for i, p in enumerate(gpu_phases):
        if p > 1e-6:
            f.write(f"  Phase {i}: {p:.6f}\n")
    
    cpu_active = sum(1 for p in cpu_phases if p > 1e-6)
    gpu_active = sum(1 for p in gpu_phases if p > 1e-6)
    
    if cpu_active != gpu_active:
        f.write(f"\nWARNING: Different number of phases! CPU={cpu_active}, GPU={gpu_active}\n")

print("\nTrace saved to /tmp/trace_output.txt")
print("\nLook in the verbose output above for divergence points:")