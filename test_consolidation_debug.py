#!/usr/bin/env python
"""Debug GPU consolidation issue for X(TI)=0.1, T=600K."""

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

print("Testing GPU consolidation for X(TI)=0.1, T=600K")
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

# Look for consolidation-related messages
print("\nSearching for consolidation events in GPU output:")
print("-"*80)

lines = gpu_text.split('\n')
for i, line in enumerate(lines):
    if 'consolidat' in line.lower() or 'CONSOLIDAT' in line:
        print(f"Line {i}: {line}")
    elif 'phases %d and %d' in line:
        print(f"Line {i}: {line}")
    elif 'Phase %d marked for removal' in line:
        print(f"Line {i}: {line}")
    elif 'Should consolidate:' in line:
        print(f"Line {i}: {line}")
    elif 'Max composition diff:' in line:
        print(f"Line {i}: {line}")
    elif 'final_free_count=' in line:
        print(f"Line {i}: {line}")
    elif 'Free stable indices:' in line:
        print(f"Line {i}: {line}")
        # Also print the next few lines
        for j in range(1, 4):
            if i+j < len(lines):
                print(f"  Line {i+j}: {lines[i+j]}")

# Look for phase amounts
print("\n\nPhase amounts through iterations:")
print("-"*80)
for i, line in enumerate(lines):
    if 'phase_amt[' in line or 'Phase amounts:' in line:
        print(f"Line {i}: {line}")
    elif 'NP=' in line or 'phase_amt=' in line:
        print(f"Line {i}: {line}")

# Look for final results
print("\n\nFinal phase configuration:")
print("-"*80)
gm = float(result_gpu.GM.values)
print(f"Final GM: {gm:.6f}")

# Count phases
phase_counts = {}
for phase in phases:
    mask = result_gpu.Phase.values.flatten() == phase
    if np.any(mask):
        phase_amt = result_gpu.NP.values.flatten()[mask]
        phase_counts[phase] = phase_amt[phase_amt > 1e-6]

print(f"Active phases: {len([p for p,v in phase_counts.items() if len(v) > 0])}")
for phase, amounts in phase_counts.items():
    if len(amounts) > 0:
        print(f"  {phase}: {amounts}")

# Look for specific iteration info
print("\n\nIteration 0 details:")
print("-"*80)
in_iter0 = False
for line in lines:
    if 'iteration 0' in line.lower() or 'Iteration 0' in line:
        in_iter0 = True
    elif 'iteration 1' in line.lower() or 'Iteration 1' in line:
        in_iter0 = False
    
    if in_iter0 and ('phase' in line.lower() or 'Phase' in line):
        print(line)