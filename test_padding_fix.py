#!/usr/bin/env python
"""Test the padding fix for threads 10 and 17."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database and set up calculation
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = filter_phases(dbf, comps)

print(f"Testing with {len(phases)} phases: {phases}")

# Create test conditions - focusing on the failing ones
test_conditions = []
x_bi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
temperatures = [400, 500, 600, 700]

# Generate all combinations
for t in temperatures:
    for x in x_bi_values:
        test_conditions.append({'T': t, 'X(BI)': x, 'P': 101325})

# Test conditions 10 and 17 specifically
print(f"Condition 10: {test_conditions[10]}")  # X(BI)=0.3, T=500
print(f"Condition 17: {test_conditions[17]}")  # X(BI)=0.2, T=600

# Run with conditions that fail
conditions = {
    v.X('BI'): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    v.T: [400, 500, 600, 700],
    v.P: 101325
}

print("\nRunning GPU calculation with padding fix...")
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
    print("GPU calculation completed successfully!")
    
    # Check the specific failing conditions
    # Flatten the result to linear index
    gm_gpu = result_gpu.GM.values.flatten()
    print(f"\nCondition 10 GM (GPU): {gm_gpu[10]}")
    print(f"Condition 17 GM (GPU): {gm_gpu[17]}")
    
except Exception as e:
    print(f"GPU calculation failed: {e}")
    import traceback
    traceback.print_exc()