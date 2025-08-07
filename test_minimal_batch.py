#!/usr/bin/env python
"""Minimal test for conditions 10 and 17."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Clear any GPU cache
import cupy as cp
cp.get_default_memory_pool().free_all_blocks()

# Load database
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = filter_phases(dbf, comps)

# Full 32-condition batch
cond_full = {
    v.X('BI'): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    v.T: [400, 500, 600, 700],
    v.P: 101325
}

print("Running GPU calculation...")
result_gpu = equilibrium(dbf, comps, phases, cond_full, gpu=True, verbose=False)
gm_gpu = result_gpu.GM.values.flatten()

print("Running CPU calculation...")
result_cpu = equilibrium(dbf, comps, phases, cond_full, gpu=False, verbose=False)
gm_cpu = result_cpu.GM.values.flatten()

print(f"\nCondition 10: GPU={gm_gpu[10]:.6f}, CPU={gm_cpu[10]:.6f}, Diff={abs(gm_gpu[10]-gm_cpu[10]):.6f}")
print(f"Condition 17: GPU={gm_gpu[17]:.6f}, CPU={gm_cpu[17]:.6f}, Diff={abs(gm_gpu[17]-gm_cpu[17]):.6f}")

# Check all differences
failures = []
for i in range(len(gm_cpu)):
    diff = abs(gm_cpu[i] - gm_gpu[i])
    if diff > 1e-3:
        failures.append((i, diff))

print(f"\nTotal failures: {len(failures)}")
if failures:
    for idx, diff in failures:
        print(f"  Index {idx}: diff = {diff:.6f}")