#!/usr/bin/env python
"""Test that the stride fix resolves the failing conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = filter_phases(dbf, comps)

print(f"Testing with {len(phases)} phases")

# Create conditions that include the failing ones
conditions = {
    v.X('BI'): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    v.T: [400, 500, 600, 700],
    v.P: 101325
}

# Expected failures are at indices 10 (X(BI)=0.3, T=500) and 17 (X(BI)=0.2, T=600)
print("\nRunning GPU calculation...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

print("\nRunning CPU calculation...")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

# Check results
gm_gpu = result_gpu.GM.values.flatten()
gm_cpu = result_cpu.GM.values.flatten()

print(f"\nCondition 10 (X(BI)=0.3, T=500):")
print(f"  CPU: {gm_cpu[10]:.6f}")
print(f"  GPU: {gm_gpu[10]:.6f}")
print(f"  Diff: {abs(gm_cpu[10] - gm_gpu[10]):.6f}")

print(f"\nCondition 17 (X(BI)=0.2, T=600):")
print(f"  CPU: {gm_cpu[17]:.6f}")
print(f"  GPU: {gm_gpu[17]:.6f}")
print(f"  Diff: {abs(gm_cpu[17] - gm_gpu[17]):.6f}")

# Check all conditions
tolerance = 1e-3
failures = []
for i in range(len(gm_cpu)):
    diff = abs(gm_cpu[i] - gm_gpu[i])
    if diff > tolerance:
        failures.append((i, diff))

if failures:
    print(f"\n{len(failures)} conditions failed with tolerance {tolerance}:")
    for idx, diff in failures[:5]:  # Show first 5
        print(f"  Index {idx}: diff = {diff:.6f}")
else:
    print(f"\n✓ All conditions passed with tolerance {tolerance}!")