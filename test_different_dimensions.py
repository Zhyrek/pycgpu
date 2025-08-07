#!/usr/bin/env python
"""Test if the failure pattern is tied to thread position or actual conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = filter_phases(dbf, comps)

print("Test 1: Original 8x4 grid (32 conditions)")
print("="*60)

# Original conditions
cond_8x4 = {
    v.X('BI'): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    v.T: [400, 500, 600, 700],
    v.P: 101325
}

result_gpu_8x4 = equilibrium(dbf, comps, phases, cond_8x4, gpu=True, verbose=False)
result_cpu_8x4 = equilibrium(dbf, comps, phases, cond_8x4, gpu=False, verbose=False)

gm_gpu_8x4 = result_gpu_8x4.GM.values.flatten()
gm_cpu_8x4 = result_cpu_8x4.GM.values.flatten()

print("Failing conditions in 8x4 grid:")
for i in range(32):
    diff = abs(gm_cpu_8x4[i] - gm_gpu_8x4[i])
    if diff > 1e-3:
        t_idx = i // 8
        x_idx = i % 8
        x_val = cond_8x4[v.X('BI')][x_idx]
        t_val = cond_8x4[v.T][t_idx]
        print(f"  Thread {i}: X(BI)={x_val}, T={t_val}K, diff={diff:.6f}, {i}%7={i%7}")

print("\n\nTest 2: Different 5x7 grid (35 conditions)")
print("="*60)

# Different grid dimensions
cond_5x7 = {
    v.X('BI'): [0.15, 0.25, 0.35, 0.45, 0.55],  # 5 values
    v.T: [350, 425, 500, 575, 650, 725, 800],   # 7 values
    v.P: 101325
}

result_gpu_5x7 = equilibrium(dbf, comps, phases, cond_5x7, gpu=True, verbose=False)
result_cpu_5x7 = equilibrium(dbf, comps, phases, cond_5x7, gpu=False, verbose=False)

gm_gpu_5x7 = result_gpu_5x7.GM.values.flatten()
gm_cpu_5x7 = result_cpu_5x7.GM.values.flatten()

print("Failing conditions in 5x7 grid:")
failures_5x7 = []
for i in range(35):
    diff = abs(gm_cpu_5x7[i] - gm_gpu_5x7[i])
    if diff > 1e-3:
        t_idx = i // 5
        x_idx = i % 5
        x_val = cond_5x7[v.X('BI')][x_idx]
        t_val = cond_5x7[v.T][t_idx]
        failures_5x7.append(i)
        print(f"  Thread {i}: X(BI)={x_val}, T={t_val}K, diff={diff:.6f}, {i}%7={i%7}")

print("\n\nTest 3: Different 7x5 grid (35 conditions)")
print("="*60)

# Swap dimensions
cond_7x5 = {
    v.X('BI'): [0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84],  # 7 values
    v.T: [420, 490, 560, 630, 700],                          # 5 values
    v.P: 101325
}

result_gpu_7x5 = equilibrium(dbf, comps, phases, cond_7x5, gpu=True, verbose=False)
result_cpu_7x5 = equilibrium(dbf, comps, phases, cond_7x5, gpu=False, verbose=False)

gm_gpu_7x5 = result_gpu_7x5.GM.values.flatten()
gm_cpu_7x5 = result_cpu_7x5.GM.values.flatten()

print("Failing conditions in 7x5 grid:")
failures_7x5 = []
for i in range(35):
    diff = abs(gm_cpu_7x5[i] - gm_gpu_7x5[i])
    if diff > 1e-3:
        t_idx = i // 7
        x_idx = i % 7
        x_val = cond_7x5[v.X('BI')][x_idx]
        t_val = cond_7x5[v.T][t_idx]
        failures_7x5.append(i)
        print(f"  Thread {i}: X(BI)={x_val}, T={t_val}K, diff={diff:.6f}, {i}%7={i%7}")

print("\n\nTest 4: Different 6x6 grid (36 conditions)")
print("="*60)

# Square grid
cond_6x6 = {
    v.X('BI'): [0.1, 0.25, 0.4, 0.55, 0.7, 0.85],     # 6 values
    v.T: [375, 450, 525, 600, 675, 750],              # 6 values
    v.P: 101325
}

result_gpu_6x6 = equilibrium(dbf, comps, phases, cond_6x6, gpu=True, verbose=False)
result_cpu_6x6 = equilibrium(dbf, comps, phases, cond_6x6, gpu=False, verbose=False)

gm_gpu_6x6 = result_gpu_6x6.GM.values.flatten()
gm_cpu_6x6 = result_cpu_6x6.GM.values.flatten()

print("Failing conditions in 6x6 grid:")
failures_6x6 = []
for i in range(36):
    diff = abs(gm_cpu_6x6[i] - gm_gpu_6x6[i])
    if diff > 1e-3:
        t_idx = i // 6
        x_idx = i % 6
        x_val = cond_6x6[v.X('BI')][x_idx]
        t_val = cond_6x6[v.T][t_idx]
        failures_6x6.append(i)
        print(f"  Thread {i}: X(BI)={x_val}, T={t_val}K, diff={diff:.6f}, {i}%7={i%7}")

print("\n\nSummary:")
print("="*60)
print("Original 8x4: threads 10 and 17 fail (both have thread_id % 7 = 3)")
print(f"5x7 grid: {len(failures_5x7)} failures at threads {failures_5x7}")
print(f"7x5 grid: {len(failures_7x5)} failures at threads {failures_7x5}")
print(f"6x6 grid: {len(failures_6x6)} failures at threads {failures_6x6}")

# Check if the pattern holds
print("\nChecking if failures have thread_id % 7 = 3:")
all_failures = []
if failures_5x7:
    for tid in failures_5x7:
        all_failures.append((tid, tid % 7))
if failures_7x5:
    for tid in failures_7x5:
        all_failures.append((tid, tid % 7))
if failures_6x6:
    for tid in failures_6x6:
        all_failures.append((tid, tid % 7))

for tid, mod7 in sorted(set(all_failures)):
    print(f"  Thread {tid}: {tid} % 7 = {mod7}")