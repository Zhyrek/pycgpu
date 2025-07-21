#!/usr/bin/env python
"""Trace the consolidation bug in detail."""

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

print("Tracing consolidation bug for X(TI)=0.1, T=600K")
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

# Look for key events in order
print("\n1. Initial phase amounts:")
print("-"*40)
for i, line in enumerate(lines[:200]):  # Look in first 200 lines
    if 'After normalization - compset' in line and 'phase_amt' in line:
        print(f"Line {i}: {line}")

print("\n2. Consolidation event:")
print("-"*40)
consolidation_line = -1
for i, line in enumerate(lines):
    if 'Consolidated phases' in line:
        consolidation_line = i
        print(f"Line {i}: {line}")
        # Print a few lines after
        for j in range(i+1, min(len(lines), i+5)):
            if 'amount=' in lines[j] or 'Phase' in lines[j]:
                print(f"  Line {j}: {lines[j]}")
        break

print("\n3. What happens after consolidation:")
print("-"*40)
if consolidation_line > 0:
    # Look for phase amount updates after consolidation
    for i in range(consolidation_line, min(len(lines), consolidation_line + 100)):
        if ('phase_amt' in lines[i] or 
            'NP=' in lines[i] or
            'num_free_stable_compsets' in lines[i] or
            'Phase.*old=' in lines[i]):
            print(f"Line {i}: {lines[i]}")

print("\n4. The problematic sync:")
print("-"*40)
for i, line in enumerate(lines):
    if 'After solver sync' in line:
        print(f"Line {i}: {line}")

print("\n5. Search for where NP becomes 1.0:")
print("-"*40)
found_np_one = False
for i, line in enumerate(lines):
    if 'NP=1.00000' in line and not found_np_one:
        found_np_one = True
        print(f"\nFirst occurrence of NP=1.0 at line {i}: {line}")
        # Print context
        for j in range(max(0, i-10), min(len(lines), i+5)):
            if j == i:
                print(f">>> Line {j}: {lines[j]}")
            else:
                print(f"    Line {j}: {lines[j]}")

# Extract final results
gpu_gm = float(result_gpu.GM.values)
print(f"\n\nFinal GPU Results:")
print(f"GM: {gpu_gm:.6f} J/mol")

# Run CPU for comparison
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = float(result_cpu.GM.values)
print(f"\nCPU GM: {cpu_gm:.6f} J/mol")
print(f"Error: {gpu_gm - cpu_gm:.6f} J/mol")