#!/usr/bin/env python
"""Trace GPU kernel result handling."""

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

print("Tracing GPU kernel result handling for X(TI)=0.1, T=600K")
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

# Search for where equilibrium_result is handled after solve_equilibrium_at_condition
print("\nSearching for equilibrium_result handling after solver:")
print("-"*80)

for i, line in enumerate(lines):
    # Look for lines after solve_equilibrium_at_condition
    if 'solve_equilibrium_at_condition_global_mem COMPLETE' in line:
        print(f"\nLine {i}: {line}")
        print("=== AFTER SOLVER COMPLETES ===")
        # Print next 30 lines to see what happens with results
        for j in range(i+1, min(len(lines), i+30)):
            if 'equilibrium_result' in lines[j] or 'Storing final result' in lines[j] or 'results_array' in lines[j]:
                print(f"Line {j}: {lines[j]}")
    
    # Look for "Storing final result"
    if 'Storing final result' in line:
        print(f"\n=== STORING RESULTS ===")
        print(f"Line {i}: {line}")
        # Print surrounding context
        for j in range(max(0, i-5), min(len(lines), i+15)):
            print(f"  {j}: {lines[j]}")

# Also look for phase amount storage
print("\n\nSearching for phase amount storage:")
print("-"*80)
for i, line in enumerate(lines):
    if 'phase_amounts[' in line or 'equilibrium_result.NP' in line:
        print(f"Line {i}: {line}")
        # Context
        for j in range(max(0, i-2), min(len(lines), i+3)):
            if j != i:
                print(f"  Context {j}: {lines[j]}")