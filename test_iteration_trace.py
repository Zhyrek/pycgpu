#!/usr/bin/env python
"""
Trace solver iterations to find first CPU-GPU deviation
"""

import pycalphad as cp
import numpy as np

# Load database
db = cp.Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = ['FCC_A1', 'ALCU_ZETA']

# Test conditions
conditions = {
    cp.v.X('AL'): 0.7,
    cp.v.X('CU'): 0.2,
    cp.v.T: 600,
    cp.v.P: 101325
}

print("Testing ALCU_ZETA phase CPU vs GPU - Iteration trace")
print("=" * 60)

# CPU calculation with verbose output
print("\nCPU calculation...")
eq_cpu = cp.equilibrium(db, components, phases, conditions, verbose=True, debug=True)

print("\n" + "="*60)
print("GPU calculation...")
eq_gpu = cp.equilibrium(db, components, phases, conditions, gpu=True, verbose=True, debug=True)