#!/usr/bin/env python
"""Test if conditions 10 and 17 fail when run individually vs in batch."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = filter_phases(dbf, comps)

print(f"Testing with {len(phases)} phases: {phases}")

# Define the specific failing conditions
# Condition 10: X(BI)=0.3, T=500
# Condition 17: X(BI)=0.2, T=600

# Test 1: Run condition 10 alone
print("\n=== TEST 1: Condition 10 (X(BI)=0.3, T=500) alone ===")
cond_10_single = {v.X('BI'): 0.3, v.T: 500, v.P: 101325}

result_cpu_10 = equilibrium(dbf, comps, phases, cond_10_single, gpu=False, verbose=False)
result_gpu_10 = equilibrium(dbf, comps, phases, cond_10_single, gpu=True, verbose=False)

gm_cpu_10 = float(result_cpu_10.GM.values)
gm_gpu_10 = float(result_gpu_10.GM.values)
diff_10 = abs(gm_cpu_10 - gm_gpu_10)

print(f"CPU: {gm_cpu_10:.6f}")
print(f"GPU: {gm_gpu_10:.6f}")
print(f"Diff: {diff_10:.6f}")
print(f"Status: {'PASS' if diff_10 < 1e-3 else 'FAIL'}")

# Test 2: Run condition 17 alone
print("\n=== TEST 2: Condition 17 (X(BI)=0.2, T=600) alone ===")
cond_17_single = {v.X('BI'): 0.2, v.T: 600, v.P: 101325}

result_cpu_17 = equilibrium(dbf, comps, phases, cond_17_single, gpu=False, verbose=False)
result_gpu_17 = equilibrium(dbf, comps, phases, cond_17_single, gpu=True, verbose=False)

gm_cpu_17 = float(result_cpu_17.GM.values)
gm_gpu_17 = float(result_gpu_17.GM.values)
diff_17 = abs(gm_cpu_17 - gm_gpu_17)

print(f"CPU: {gm_cpu_17:.6f}")
print(f"GPU: {gm_gpu_17:.6f}")
print(f"Diff: {diff_17:.6f}")
print(f"Status: {'PASS' if diff_17 < 1e-3 else 'FAIL'}")

# Test 3: Run them in a batch of just these 2 conditions
print("\n=== TEST 3: Both conditions in a small batch ===")
cond_batch_small = {
    v.X('BI'): [0.3, 0.2],
    v.T: [500, 600],
    v.P: 101325
}

result_cpu_batch = equilibrium(dbf, comps, phases, cond_batch_small, gpu=False, verbose=False)
result_gpu_batch = equilibrium(dbf, comps, phases, cond_batch_small, gpu=True, verbose=False)

gm_cpu_batch = result_cpu_batch.GM.values.flatten()
gm_gpu_batch = result_gpu_batch.GM.values.flatten()

# Note: With X_BI x T grid, the conditions are NOT what we think
# Let me print the actual grid
print("Small batch grid shape:", result_cpu_batch.GM.shape)
print("Flattened conditions:")
for i in range(len(gm_cpu_batch)):
    print(f"  Index {i}: CPU={gm_cpu_batch[i]:.6f}, GPU={gm_gpu_batch[i]:.6f}, Diff={abs(gm_cpu_batch[i]-gm_gpu_batch[i]):.6f}")

# Test 4: Run in the full 32-condition batch
print("\n=== TEST 4: Full 32-condition batch ===")
cond_full = {
    v.X('BI'): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    v.T: [400, 500, 600, 700],
    v.P: 101325
}

result_cpu_full = equilibrium(dbf, comps, phases, cond_full, gpu=False, verbose=False)
result_gpu_full = equilibrium(dbf, comps, phases, cond_full, gpu=True, verbose=False)

gm_cpu_full = result_cpu_full.GM.values.flatten()
gm_gpu_full = result_gpu_full.GM.values.flatten()

# Condition 10: X(BI)=0.3, T=500 -> indices [2, 1] -> linear index 10
# Condition 17: X(BI)=0.2, T=600 -> indices [1, 2] -> linear index 17

print(f"Condition 10 (X=0.3,T=500): CPU={gm_cpu_full[10]:.6f}, GPU={gm_gpu_full[10]:.6f}, Diff={abs(gm_cpu_full[10]-gm_gpu_full[10]):.6f}")
print(f"Condition 17 (X=0.2,T=600): CPU={gm_cpu_full[17]:.6f}, GPU={gm_gpu_full[17]:.6f}, Diff={abs(gm_cpu_full[17]-gm_gpu_full[17]):.6f}")

# Also check which thread didn't converge
print("\nChecking all conditions for large differences:")
for i in range(len(gm_cpu_full)):
    diff = abs(gm_cpu_full[i] - gm_gpu_full[i])
    if diff > 1.0:
        print(f"  Condition {i}: CPU={gm_cpu_full[i]:.6f}, GPU={gm_gpu_full[i]:.6f}, Diff={diff:.6f}")

# Summary
print("\n=== SUMMARY ===")
print(f"Condition 10 single: {'PASS' if diff_10 < 1e-3 else 'FAIL'}")
print(f"Condition 17 single: {'PASS' if diff_17 < 1e-3 else 'FAIL'}")
print(f"Small batch: C10={'PASS' if abs(gm_cpu_batch[0]-gm_gpu_batch[0]) < 1e-3 else 'FAIL'}, C17={'PASS' if abs(gm_cpu_batch[1]-gm_gpu_batch[1]) < 1e-3 else 'FAIL'}")
print(f"Full batch: C10={'PASS' if abs(gm_cpu_full[10]-gm_gpu_full[10]) < 1e-3 else 'FAIL'}, C17={'PASS' if abs(gm_cpu_full[17]-gm_gpu_full[17]) < 1e-3 else 'FAIL'}")

print("\nConclusion:")
if diff_10 < 1e-3 and diff_17 < 1e-3:
    print("Both conditions PASS when run individually, but FAIL in multi-condition batch")
    print("This confirms it's a batch-specific issue, likely related to thread interactions")
else:
    print("Conditions fail even when run individually - not batch-specific")