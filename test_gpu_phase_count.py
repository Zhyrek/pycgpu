#!/usr/bin/env python
"""Test GPU phase counting after consolidation."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import cupy as cp

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test condition: X(TI)=0.1, T=600K
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("Testing GPU phase counting for X(TI)=0.1, T=600K")
print("="*80)

# Clear CuPy cache
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

# Run GPU calculation with debug output
import sys
import io
gpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = gpu_output
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
finally:
    sys.stdout = old_stdout

gpu_text = gpu_output.getvalue()
lines = gpu_text.split('\n')

# Search for phase counting debug output
print("\nSearching for phase counting in final result assembly:")
print("-"*80)

for i, line in enumerate(lines):
    if 'Collecting stable phases' in line:
        print(f"\nLine {i}: {line}")
        # Print next 20 lines
        for j in range(i+1, min(len(lines), i+20)):
            if 'phase_amt' in lines[j] or 'threshold' in lines[j] or 'stable_phase_count' in lines[j]:
                print(f"Line {j}: {lines[j]}")
    
    if 'stable_phase_count' in line:
        print(f"\nLine {i}: {line}")
    
    if 'result->num_stable_phases' in line:
        print(f"\nLine {i}: {line}")

# Also look for the actual consolidation happening
print("\n\nSearching for consolidation events:")
print("-"*80)

for i, line in enumerate(lines):
    if 'CONSOLIDATION' in line and 'Consolidated phases' in line:
        print(f"\nLine {i}: {line}")
        # Print surrounding lines
        for j in range(max(0, i-5), min(len(lines), i+10)):
            print(f"  Line {j}: {lines[j]}")

# Extract final results
gpu_gm = float(result_gpu.GM.values)
gpu_phases = result_gpu.Phase.values.flatten()
gpu_np = result_gpu.NP.values.flatten()

print(f"\n\nFinal GPU Results:")
print(f"GM: {gpu_gm:.6f} J/mol")
print(f"Phases: {gpu_phases}")
print(f"NP: {gpu_np}")

# Count non-zero phases
stable_count = 0
for phase, amt in zip(gpu_phases, gpu_np):
    if phase and amt > 1e-6:
        stable_count += 1

print(f"\nStable phases counted from results: {stable_count}")

# Run CPU for comparison
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = float(result_cpu.GM.values)
cpu_stable_count = 0
for phase, amt in zip(result_cpu.Phase.values.flatten(), result_cpu.NP.values.flatten()):
    if phase and amt > 1e-6:
        cpu_stable_count += 1

print(f"\nCPU stable phases: {cpu_stable_count}")
print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"Error: {gpu_gm - cpu_gm:.6f} J/mol")