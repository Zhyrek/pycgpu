#!/usr/bin/env python
"""Test to track solver iterations for failing conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = filter_phases(dbf, comps)

# Create a custom GPU kernel that prints iteration info
import os
os.environ['VERBOSE_DEBUG'] = '1'

# Test just the failing conditions
print("Testing conditions 10 and 17 in isolation...")
print("="*60)

# Condition 10: X(BI)=0.3, T=500
print("\n=== Condition 10 (X(BI)=0.3, T=500) ===")
cond_10 = {v.X('BI'): 0.3, v.T: 500, v.P: 101325}
result_10 = equilibrium(dbf, comps, phases, cond_10, gpu=True, verbose=True)
print(f"GPU GM: {float(result_10.GM.values):.6f}")

# Condition 17: X(BI)=0.2, T=600
print("\n=== Condition 17 (X(BI)=0.2, T=600) ===")
cond_17 = {v.X('BI'): 0.2, v.T: 600, v.P: 101325}
result_17 = equilibrium(dbf, comps, phases, cond_17, gpu=True, verbose=True)
print(f"GPU GM: {float(result_17.GM.values):.6f}")

# Now test in a batch
print("\n\n=== Testing in 32-condition batch ===")
cond_batch = {
    v.X('BI'): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    v.T: [400, 500, 600, 700],
    v.P: 101325
}

result_batch = equilibrium(dbf, comps, phases, cond_batch, gpu=True, verbose=False)
gm_batch = result_batch.GM.values.flatten()

print(f"\nCondition 10 in batch: GM={gm_batch[10]:.6f}")
print(f"Condition 17 in batch: GM={gm_batch[17]:.6f}")