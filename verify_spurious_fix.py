#!/usr/bin/env python3
"""Verify that spurious entropy terms are properly removed from GPU Hessian"""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v

# Set up the calculation
dbf = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = ["BCC_A2"]
conditions = {
    v.X("TI"): 0.4,
    v.T: 1000,
    v.P: 101325,
    v.N: 1,
}

# Run CPU calculation
print("Running CPU calculation...")
cpu_eq = equilibrium(dbf, comps, phases, conditions, verbose=False)
cpu_gm = float(cpu_eq.GM.values[0])
print(f"CPU GM = {cpu_gm:.6f} J/mol")

# Run GPU calculation
print("\nRunning GPU calculation...")
gpu_eq = equilibrium(dbf, comps, phases, conditions, verbose=True, gpu=True)
gpu_gm = float(gpu_eq.GM.values[0])
print(f"\nGPU GM = {gpu_gm:.6f} J/mol")

# Compare results
difference = abs(gpu_gm - cpu_gm)
print(f"\nDifference: {difference:.6f} J/mol")

if difference < 0.1:  # Less than 0.1 J/mol difference
    print("SUCCESS: GPU and CPU results match!")
else:
    print(f"ERROR: GPU and CPU results differ by {difference:.6f} J/mol")