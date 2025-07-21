#!/usr/bin/env python
"""Test to see when 'all phases removed' code is triggered."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import sys
import io

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test condition: X(TI)=0.1, T=600K
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("Testing 'all phases removed' trigger for X(TI)=0.1, T=600K")
print("="*80)

# Capture GPU output
gpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = gpu_output
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
finally:
    sys.stdout = old_stdout

gpu_text = gpu_output.getvalue()
lines = gpu_text.split('\n')

# Search for the "all phases removed" message
print("\nSearching for 'all phases removed' trigger:")
print("-"*80)

found_trigger = False
for i, line in enumerate(lines):
    if 'All phases would be removed' in line:
        found_trigger = True
        print(f"\n!!! FOUND TRIGGER at line {i}: {line}")
        # Print context
        print("\nContext (10 lines before):")
        for j in range(max(0, i-10), i):
            print(f"  Line {j}: {lines[j]}")
        print("\nContext (10 lines after):")
        for j in range(i+1, min(len(lines), i+10)):
            print(f"  Line {j}: {lines[j]}")

if not found_trigger:
    print("'All phases would be removed' trigger NOT FOUND")

# Also search for consolidation events
print("\n\nSearching for consolidation events:")
print("-"*80)

for i, line in enumerate(lines):
    if 'Consolidated phases' in line:
        print(f"\nLine {i}: {line}")
    if 'marked for removal' in line:
        print(f"Line {i}: {line}")
    if 'Updated num_free_stable_compsets' in line:
        print(f"Line {i}: {line}")

# Check final results
gpu_gm = float(result_gpu.GM.values)
gpu_phases = result_gpu.Phase.values.flatten()
gpu_np = result_gpu.NP.values.flatten()

print(f"\n\nFinal GPU Results:")
print(f"GM: {gpu_gm:.6f} J/mol")
print(f"Phases: {gpu_phases}")
print(f"NP: {gpu_np}")

# Run CPU for comparison
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = float(result_cpu.GM.values)
print(f"\nCPU GM: {cpu_gm:.6f} J/mol")
print(f"Error: {gpu_gm - cpu_gm:.6f} J/mol")