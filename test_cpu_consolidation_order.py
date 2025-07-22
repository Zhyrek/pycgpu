#!/usr/bin/env python
"""Test which phase the CPU keeps during consolidation."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test condition: X(TI)=0.1, T=600K
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("Testing CPU consolidation for X(TI)=0.1, T=600K")
print("="*80)

# Run with extra verbose output to capture consolidation details
import sys
from io import StringIO

# Redirect stdout to capture output
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

try:
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
finally:
    sys.stdout = old_stdout

output = mystdout.getvalue()

# Search for consolidation messages
lines = output.split('\n')
consolidating = False
for i, line in enumerate(lines):
    if 'CONSOLIDATING phases' in line:
        consolidating = True
        print(f"\nFound consolidation at line {i}:")
        print(line)
        # Look for which phases are involved
        if 'phases 0 and 1' in line:
            print("  -> CPU is consolidating phase 1 INTO phase 0")
        elif 'phases 1 and 0' in line:
            print("  -> CPU is consolidating phase 0 INTO phase 1")
    elif consolidating and 'Phase 0' in line and 'amount' in line:
        print(f"  After consolidation: {line}")
    elif consolidating and 'Phase 1' in line and 'amount' in line:
        print(f"  After consolidation: {line}")
        consolidating = False

# Also check iteration traces
print("\n\nLooking for phase amount changes:")
for i, line in enumerate(lines):
    if 'Phase 0' in line and ('amount=' in line or 'phase_amt' in line):
        if i < 100:  # Only first few iterations
            print(f"Line {i}: {line.strip()}")
    elif 'Phase 1' in line and ('amount=' in line or 'phase_amt' in line):
        if i < 100:  # Only first few iterations
            print(f"Line {i}: {line.strip()}")

# Check final result
print(f"\n\nFinal CPU Result:")
print(f"GM: {float(result_cpu.GM.values):.6f}")
print(f"Active phases: {result_cpu.Phase.values.flatten()[result_cpu.NP.values.flatten() > 1e-6]}")
print(f"Phase amounts: {result_cpu.NP.values.flatten()[result_cpu.NP.values.flatten() > 1e-6]}")