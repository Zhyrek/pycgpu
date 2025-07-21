#!/usr/bin/env python
"""Trace the final GM calculation to find source of error."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import sys
import io

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test single-phase condition
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("Tracing GM calculation for single-phase region")
print("="*80)

# Run CPU calculation first
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = float(result_cpu.GM.values)
cpu_phases = result_cpu.Phase.values.flatten()
cpu_np = result_cpu.NP.values.flatten()

print(f"CPU Result:")
print(f"  GM = {cpu_gm:.15f} J/mol")
for i, (phase, amt) in enumerate(zip(cpu_phases, cpu_np)):
    if phase and amt > 1e-6:
        print(f"  Phase {i}: {phase} = {amt:.15f}")

# Now run GPU with debugging to see final GM calculation
gpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = gpu_output
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
finally:
    sys.stdout = old_stdout

gpu_text = gpu_output.getvalue()
gpu_gm = float(result_gpu.GM.values)

print(f"\nGPU Result:")
print(f"  GM = {gpu_gm:.15f} J/mol")
print(f"  Error = {gpu_gm - cpu_gm:.15e} J/mol")

# Look for final GM calculation in GPU output
print("\nSearching for final GM calculation in GPU output:")
lines = gpu_text.split('\n')
for i, line in enumerate(lines):
    if "final GM" in line or "GM calculation" in line or "weighted_sum" in line:
        print(f"  Line {i}: {line}")
        # Print context
        for j in range(max(0, i-2), min(len(lines), i+3)):
            if j != i and ("energy" in lines[j].lower() or "phase_amt" in lines[j] or "NP" in lines[j]):
                print(f"    Context: {lines[j]}")
    elif "Transferring results" in line:
        print(f"\n  Line {i}: {line}")
        # Look for energy values being transferred
        for j in range(i, min(len(lines), i+10)):
            if "energy" in lines[j].lower() or "GM" in lines[j]:
                print(f"    {lines[j]}")

# Check if it's a floating point accumulation issue
print("\nHypothesis: Error accumulates during iterative solution")
print("- Two-phase regions converge without consolidation")
print("- Single-phase regions require consolidation and more iterations")
print("- Each iteration may introduce small floating-point errors")
print("- The 9e-8 J/mol error is consistent with ~10 iterations of 1e-8 errors")