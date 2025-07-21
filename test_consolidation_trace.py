#!/usr/bin/env python
"""Trace phase consolidation for failing single-phase condition."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import sys
import io
import warnings
warnings.filterwarnings("ignore")

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test a failing condition X(TI)=0.1, T=600K
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("TRACING PHASE CONSOLIDATION FOR X(TI)=0.1, T=600K")
print("=" * 80)

# CPU calculation with verbose output
print("\n[CPU CALCULATION]")
print("-" * 40)
cpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = cpu_output
try:
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
finally:
    sys.stdout = old_stdout

cpu_text = cpu_output.getvalue()

# Look for consolidation in CPU output
cpu_consolidated = False
for line in cpu_text.split('\n'):
    if 'Consolidating phases' in line or 'consolidate' in line.lower():
        print(f"CPU: {line}")
        cpu_consolidated = True
    elif 'Removing phase' in line or 'phase_amt < ' in line:
        print(f"CPU: {line}")
    elif 'num_compsets' in line:
        print(f"CPU: {line}")

# GPU calculation with verbose output  
print("\n[GPU CALCULATION]")
print("-" * 40)
gpu_output = io.StringIO()
sys.stdout = gpu_output
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
finally:
    sys.stdout = old_stdout

gpu_text = gpu_output.getvalue()

# Look for consolidation in GPU output
gpu_consolidated = False
for line in gpu_text.split('\n'):
    if 'Consolidating' in line or 'consolidat' in line.lower():
        print(f"GPU: {line}")
        gpu_consolidated = True
    elif 'Removing phase' in line or 'phase_amt < ' in line:
        print(f"GPU: {line}")
    elif 'stable_phases' in line:
        print(f"GPU: {line}")

# Final comparison
print("\n[FINAL RESULTS]")
print("-" * 40)
cpu_phases = [(p, np) for p, np in zip(result_cpu.Phase.values.flatten(), 
                                       result_cpu.NP.values.flatten()) if np > 1e-6]
gpu_phases = [(p, np) for p, np in zip(result_gpu.Phase.values.flatten(), 
                                       result_gpu.NP.values.flatten()) if np > 1e-6]

print(f"CPU: {len(cpu_phases)} phases - {cpu_phases}")
print(f"GPU: {len(gpu_phases)} phases - {gpu_phases}")
print(f"CPU GM: {result_cpu.GM.values[0]:.2f} J/mol")
print(f"GPU GM: {result_gpu.GM.values[0]:.2f} J/mol")
print(f"Difference: {float(result_gpu.GM.values[0] - result_cpu.GM.values[0]):.2f} J/mol")

print("\n[ANALYSIS]")
print("-" * 40)
print(f"CPU showed consolidation: {cpu_consolidated}")
print(f"GPU showed consolidation: {gpu_consolidated}")
print(f"CPU phases all same: {len(set(p for p,_ in cpu_phases)) == 1}")
print(f"GPU phases all same: {len(set(p for p,_ in gpu_phases)) == 1}")

# Check if this is really a single-phase region
if len(cpu_phases) == 1:
    print("\nThis IS a single-phase region (CPU has 1 phase)")
elif len(set(p for p,_ in cpu_phases)) == 1:
    print("\nThis SHOULD BE a single-phase region (all CPU phases are same type)")
else:
    print("\nThis is a TRUE two-phase region")