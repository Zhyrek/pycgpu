#!/usr/bin/env python
"""Test to compare correction calculations between CPU and GPU."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import subprocess

# Clear GPU cache
subprocess.run(['rm', '-f', '/home/user/.cache/pycalphad/*.cu'], capture_output=True)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("=== Testing Correction Calculations ===")
print("\nRunning calculations to examine corrections after consolidation...")

# Run GPU calculation with verbose output to see what corrections are calculated
print("\n--- GPU Calculation ---")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]
print(f"\nGPU Final X(TI) = {gpu_x_ti:.10f}")

# Run CPU calculation with verbose output
print("\n--- CPU Calculation ---")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
print(f"\nCPU Final X(TI) = {cpu_x_ti:.10f}")

# Check the equilibrium matrix dimensions
print("\n--- Equilibrium System Analysis ---")
print(f"After consolidation to single phase:")
print(f"  Free chemical potentials: 2 (NB, TI)")
print(f"  Free stable phases: 1")
print(f"  Free state variables: 0")
print(f"  Total unknowns: 2 + 1 + 0 = 3")
print(f"\n  Constraints:")
print(f"  - 1 stable phase equation")
print(f"  - 1 mole fraction constraint (X(TI) = 0.9)")
print(f"  - 1 system amount constraint (N = 1)")
print(f"  Total constraints: 3")
print(f"\n  System is square (3x3)")

# The key issue: after consolidation, GPU produces X(TI) = 0.9029604
# but needs to correct to X(TI) = 0.9000000
print(f"\nRequired correction: {0.9 - 0.9029604:.10f}")
print(f"This correction should come from adjusting:")
print(f"  1. Chemical potentials")
print(f"  2. Phase amount") 
print(f"  3. Site fractions")