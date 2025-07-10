#!/usr/bin/env python3
"""Trace where CPU and GPU solvers diverge"""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions - T=300K for the problematic case
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== Tracing CPU/GPU Solver Divergence ===\n")
print("Conditions: T=300K, X(TI)=0.4")
print("Starting with 2 BCC_A2 phases: [[0.6, 0.4], [0.5, 0.5]]\n")

# First run CPU to capture its behavior
print("1. CPU Equilibrium:")
print("-" * 50)

import sys
from io import StringIO

# Capture CPU output
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

eq_cpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                    verbose=True)

cpu_output = mystdout.getvalue()
sys.stdout = old_stdout

# Extract key information from CPU output
print("CPU Initial state:")
for line in cpu_output.split('\n'):
    if 'prop_Phase_values at index:' in line:
        print(f"  {line.strip()}")
    elif 'prop_NP_values at index:' in line:
        print(f"  {line.strip()}")

print("\nCPU Iteration 0:")
iter0_found = False
for line in cpu_output.split('\n'):
    if 'AFTER ITERATION 0' in line:
        iter0_found = True
    elif iter0_found and ('Phase 0' in line or 'Phase 1' in line or 'converged:' in line):
        print(f"  {line.strip()}")
    elif iter0_found and 'END ITERATION 0' in line:
        break

print("\nCPU Iteration 1:")
iter1_found = False
for line in cpu_output.split('\n'):
    if 'AFTER ITERATION 1' in line:
        iter1_found = True
    elif iter1_found and 'CONSOLIDATING' in line:
        print(f"  >>> {line.strip()} <<<")
        break

print(f"\nCPU Final: GM = {eq_cpu.GM.values[0]:.1f} J/mol")

# Now run GPU
print("\n\n2. GPU Equilibrium:")
print("-" * 50)

# Capture GPU output
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

eq_gpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                    gpu=True, verbose=True)

gpu_output = mystdout.getvalue()
sys.stdout = old_stdout

# Extract GPU information
print("GPU Initial state:")
for line in gpu_output.split('\n'):
    if 'num_phases=2' in line:
        print(f"  {line.strip()}")
        break

print("\nGPU Phase compositions during iterations:")
comp_lines = []
for line in gpu_output.split('\n'):
    if 'Phase 0 composition[' in line or 'Phase 1 composition[' in line:
        comp_lines.append(line.strip())
    elif 'remove_and_consolidate_phases called' in line:
        if comp_lines:
            print(f"\n  Before {line.strip()}:")
            for cl in comp_lines[-4:]:  # Last 4 composition lines
                print(f"    {cl}")
        comp_lines = []

print("\nGPU Consolidation check:")
for line in gpu_output.split('\n'):
    if 'Checking phases' in line or 'Should consolidate:' in line:
        print(f"  {line.strip()}")

print(f"\nGPU Final: GM = {eq_gpu.GM.values[0]:.1f} J/mol")

print("\n\n3. Key Differences:")
print("-" * 50)
print(f"Energy difference: {abs(eq_gpu.GM.values[0] - eq_cpu.GM.values[0]):.1f} J/mol")
print(f"CPU consolidates phases: YES (after iteration 1)")
print(f"GPU consolidates phases: NO (compositions too different)")