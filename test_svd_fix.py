#!/usr/bin/env python
"""Test if SVD improvements fix the GPU convergence issue for X(TI)=0.9, T=600K."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TESTING SVD ROBUSTNESS FIX")
print("=" * 60)

# Clear cache to ensure recompilation with new SVD code
cache_dir = os.path.expanduser('~/.cache/pycalphad_gpu')
if os.path.exists(cache_dir):
    import shutil
    shutil.rmtree(cache_dir)
    print("Cleared GPU cache for recompilation")

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test the problematic condition
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print(f"\nTesting X(TI)={conditions[v.X('TI')]}, T={conditions[v.T]}K")
print("-" * 60)

# Run CPU first
print("\nRunning CPU calculation...")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = result_cpu.GM.values.flatten()[0]
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti_cpu = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)

print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"CPU overall X(TI): {overall_x_ti_cpu:.10f}")

# Run GPU with fixed SVD
print("\nRunning GPU calculation with improved SVD...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gpu_gm = result_gpu.GM.values.flatten()[0]
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
gpu_np = result_gpu.NP.values.flatten()
overall_x_ti_gpu = sum(np_val * x_ti for np_val, x_ti in zip(gpu_np, gpu_x_ti) if np_val > 1e-12)

print(f"GPU GM: {gpu_gm:.6f} J/mol")
print(f"GPU overall X(TI): {overall_x_ti_gpu:.10f}")

# Check results
error = abs(gpu_gm - cpu_gm)
x_error = abs(overall_x_ti_gpu - 0.9)

print(f"\n{'='*60}")
print("RESULTS:")
print(f"GM error: {error:.6f} J/mol")
print(f"X(TI) constraint error: {x_error:.10f}")
print(f"X(TI) satisfaction: {overall_x_ti_gpu:.10f} vs target 0.9")

if error < 1.0 and x_error < 1e-6:
    print("\n✓ SUCCESS: SVD fix resolved the convergence issue!")
    print("  GPU now converges to the correct solution.")
else:
    print("\n✗ ISSUE REMAINS: GPU still not converging properly")
    print("  Need to investigate further...")

# Test a few more conditions to ensure robustness
print(f"\n{'='*60}")
print("Testing additional conditions for robustness...")

test_conditions = [
    {v.X('TI'): 0.5, v.T: 700, v.P: 101325},
    {v.X('TI'): 0.1, v.T: 1000, v.P: 101325},
    {v.X('TI'): 0.95, v.T: 800, v.P: 101325},
]

all_good = True
for cond in test_conditions:
    print(f"\nTesting X(TI)={cond[v.X('TI')]}, T={cond[v.T]}K:")
    
    result_cpu = equilibrium(dbf, comps, phases, cond, gpu=False, verbose=False)
    result_gpu = equilibrium(dbf, comps, phases, cond, gpu=True, verbose=False)
    
    cpu_gm = result_cpu.GM.values.flatten()[0]
    gpu_gm = result_gpu.GM.values.flatten()[0]
    error = abs(gpu_gm - cpu_gm)
    
    print(f"  GM error: {error:.6f} J/mol", end="")
    if error < 1.0:
        print(" ✓")
    else:
        print(" ✗")
        all_good = False

if all_good:
    print("\n✓ ALL TESTS PASSED: SVD improvements working correctly!")
else:
    print("\n✗ Some tests still failing, may need additional fixes")