#!/usr/bin/env python
"""Trace where GPU final results are assembled."""

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

print("Tracing GPU final result assembly for X(TI)=0.1, T=600K")
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

# Search for key phrases about final results
print("\nSearching for final result assembly:")
print("-"*80)

in_final_section = False
for i, line in enumerate(lines):
    # Look for final result assembly markers
    if 'SEGMENT 40' in line or 'Final Gibbs energy' in line:
        in_final_section = True
        print(f"\n=== FINAL RESULT SECTION STARTS ===")
    
    if in_final_section:
        print(f"Line {i}: {line}")
        
        # Stop at the end of final results
        if 'solve_equilibrium_at_condition_global_mem COMPLETE' in line:
            break
    
    # Also look for where results are stored
    if 'Storing final result' in line:
        print(f"\nLine {i}: {line}")
        # Print surrounding lines
        for j in range(max(0, i-5), min(len(lines), i+10)):
            print(f"  Line {j}: {lines[j]}")
    
    # Look for equilibrium_result usage
    if 'equilibrium_result' in line and ('NP' in line or 'num_stable' in line):
        print(f"\nLine {i}: {line}")
        # Print surrounding lines  
        for j in range(max(0, i-3), min(len(lines), i+3)):
            print(f"  Context {j}: {lines[j]}")

# Also search for where phase amounts are collected
print("\n\nSearching for phase amount collection:")
print("-"*80)
for i, line in enumerate(lines):
    if 'Collecting stable phases' in line or 'phase_amt[' in line and 'threshold' in line:
        print(f"Line {i}: {line}")
        # Print next few lines
        for j in range(i+1, min(len(lines), i+10)):
            if 'phase' in lines[j].lower() or 'NP' in lines[j]:
                print(f"  Line {j}: {lines[j]}")