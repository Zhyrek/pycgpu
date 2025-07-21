#!/usr/bin/env python
"""Detailed trace of GPU GM calculation."""

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

print("Tracing GPU GM calculation for X(TI)=0.1, T=600K")
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

# Search for the final GM calculation section
print("\nSearching for SEGMENT 40 (Final Gibbs energy calculation):")
print("-"*80)

in_segment_40 = False
for i, line in enumerate(lines):
    if 'SEGMENT 40' in line:
        in_segment_40 = True
        print(f"\nLine {i}: {line}")
        print("=== SEGMENT 40 STARTS ===")
    
    if in_segment_40:
        # Print relevant lines
        if any(keyword in line for keyword in ['phase_', 'NP=', 'energy=', 'contribution', 'final_GM', 'compset', 'phase_amt']):
            print(f"Line {i}: {line}")
        
        # Stop when we reach the end
        if 'Final result assembly' in line or 'solve_equilibrium_at_condition' in line and 'COMPLETE' in line:
            print(f"\nLine {i}: {line}")
            print("=== SEGMENT 40 ENDS ===")
            break

# Also search for what phase energies are being used
print("\n\nSearching for phase energy calculations:")
print("-"*80)
for i, line in enumerate(lines):
    if 'cs_states[' in line and 'energy' in line:
        print(f"Line {i}: {line}")
        # Print context
        for j in range(max(0, i-2), min(len(lines), i+3)):
            if j != i and ('phase' in lines[j].lower() or 'energy' in lines[j].lower()):
                print(f"  Context {j}: {lines[j]}")

# Calculate what the GPU GM should be based on the reported values
print("\n\nAnalysis:")
print("-"*80)
gpu_gm = float(result_gpu.GM.values)
print(f"GPU reported GM: {gpu_gm:.6f} J/mol")

# From the debug output, we know:
# Initial phase 0: Y=[0.907760, 0.092240], energy=-24642.974769
# Initial phase 1: Y=[0.898305, 0.101695], energy=-24593.600729
# These are the energies from lower_convex_hull

# Calculate GM using initial energies with phase amounts [0.5, 0.5]
initial_gm = 0.5 * (-24642.974769) + 0.5 * (-24593.600729)
print(f"\nGM using initial phase energies with [0.5, 0.5]: {initial_gm:.6f} J/mol")
print(f"Difference from GPU reported: {abs(gpu_gm - initial_gm):.6f} J/mol")

# The converged single phase should have energy around -24602.65 based on CPU
converged_energy = -24602.651695  # from CPU calculation
print(f"\nExpected GM for converged single phase: {converged_energy:.6f} J/mol")
print(f"Difference from GPU reported: {gpu_gm - converged_energy:.6f} J/mol")