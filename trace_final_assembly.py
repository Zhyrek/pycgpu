#!/usr/bin/env python
"""Trace GPU final result assembly in detail."""

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

# Search for SEGMENT 40 and the phase collection
print("\nSearching for SEGMENT 40 (Final Gibbs energy calculation):")
print("-"*80)

in_segment_40 = False
for i, line in enumerate(lines):
    if 'SEGMENT 40' in line:
        in_segment_40 = True
        print(f"\nLine {i}: {line}")
        print("=== SEGMENT 40 STARTS ===")
    
    if in_segment_40:
        # Print all lines in SEGMENT 40
        if ('Collecting stable phases' in line or 
            'compset' in line and 'phase_amt' in line or
            'phase_' in line and 'contribution' in line or
            'final_GM' in line or
            'stable_phase_count' in line or
            'Final result assembly' in line):
            print(f"Line {i}: {line}")
        
        # Stop at the end
        if 'solve_equilibrium_at_condition_global_mem COMPLETE' in line:
            print(f"\nLine {i}: {line}")
            print("=== SEGMENT 40 ENDS ===")
            break

# Look for what happens after consolidation
print("\n\nSearching for consolidation and subsequent iterations:")
print("-"*80)

consolidation_found = False
for i, line in enumerate(lines):
    if 'Consolidated phases' in line:
        consolidation_found = True
        print(f"\nLine {i}: {line}")
        # Print next 50 lines to see what happens after consolidation
        for j in range(i+1, min(len(lines), i+50)):
            if ('num_compsets' in lines[j] or 
                'num_free_stable_compsets' in lines[j] or
                'phase_amt' in lines[j] or
                'SEGMENT 40' in lines[j]):
                print(f"Line {j}: {lines[j]}")

# Extract final results
gpu_gm = float(result_gpu.GM.values)
gpu_phases = result_gpu.Phase.values.flatten()
gpu_np = result_gpu.NP.values.flatten()

print(f"\n\nFinal GPU Results:")
print(f"GM: {gpu_gm:.6f} J/mol")
print(f"Phases: {gpu_phases}")
print(f"NP: {gpu_np}")

# Count stable phases
stable_count = 0
for phase, amt in zip(gpu_phases, gpu_np):
    if phase and amt > 1e-6:
        stable_count += 1
print(f"Stable phases: {stable_count}")

# Run CPU for comparison
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = float(result_cpu.GM.values)
print(f"\nCPU GM: {cpu_gm:.6f} J/mol")
print(f"Error: {gpu_gm - cpu_gm:.6f} J/mol")