#!/usr/bin/env python
"""Test Au-Bi system to ensure no regression after adding debug code."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1']

# Binary condition
conditions = {
    v.T: 800,
    v.P: 101325,
    v.X('BI'): 0.3
}

print("Testing Au-Bi system (binary) for regression...")

# Test CPU
result_cpu = equilibrium(dbf, comps, phases, conditions, verbose=False, calc_opts={'pdens': 50})
cpu_gm = float(result_cpu.GM.values)

# Test GPU  
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False, calc_opts={'pdens': 50})
gpu_gm = float(result_gpu.GM.values)

# Check results
difference = abs(cpu_gm - gpu_gm)
print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"GPU GM: {gpu_gm:.6f} J/mol")
print(f"Absolute difference: {difference:.6f} J/mol")

if difference < 0.001:
    print("✓ Au-Bi test PASSED - No regression detected")
else:
    print(f"✗ Au-Bi test FAILED - Difference {difference:.6f} J/mol exceeds threshold")
    exit(1)

print("\nAu-Bi system working correctly after debug code additions.")