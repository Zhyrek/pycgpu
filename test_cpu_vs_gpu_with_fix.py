#!/usr/bin/env python
"""Test CPU vs GPU comparison with SVD fix."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Clear cache to force recompilation with new SVD code
cache_dir = os.path.expanduser('~/.cache/pycalphad_gpu')
if os.path.exists(cache_dir):
    import shutil
    shutil.rmtree(cache_dir)
    print("Cleared GPU cache for recompilation")

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test the problematic X(TI)=0.9, T=600K condition
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("Testing X(TI)=0.9, T=600K with SVD improvements")
print("=" * 60)

# Run CPU
print("\nRunning CPU calculation...")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = result_cpu.GM.values.flatten()[0]
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti_cpu = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)

print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"CPU overall X(TI): {overall_x_ti_cpu:.10f}")
print(f"CPU phases present: {sum(1 for np_val in cpu_np if np_val > 1e-12)}")

# Run GPU
print("\nRunning GPU calculation...")
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = result_gpu.GM.values.flatten()[0]
    gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
    gpu_np = result_gpu.NP.values.flatten()
    overall_x_ti_gpu = sum(np_val * x_ti for np_val, x_ti in zip(gpu_np, gpu_x_ti) if np_val > 1e-12)
    
    print(f"GPU GM: {gpu_gm:.6f} J/mol")
    print(f"GPU overall X(TI): {overall_x_ti_gpu:.10f}")
    print(f"GPU phases present: {sum(1 for np_val in gpu_np if np_val > 1e-12)}")
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON:")
    print(f"GM error: {abs(gpu_gm - cpu_gm):.6f} J/mol")
    print(f"X(TI) constraint satisfaction:")
    print(f"  CPU: {overall_x_ti_cpu:.10f} (error: {abs(overall_x_ti_cpu - 0.9):.2e})")
    print(f"  GPU: {overall_x_ti_gpu:.10f} (error: {abs(overall_x_ti_gpu - 0.9):.2e})")
    
    if abs(gpu_gm - cpu_gm) < 1.0 and abs(overall_x_ti_gpu - 0.9) < 1e-6:
        print("\n✓ SUCCESS: SVD improvements resolved the convergence issue!")
    else:
        print("\n✗ ISSUE: GPU still not matching CPU properly")
        
except Exception as e:
    print(f"GPU failed with error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

# Test a few more conditions
print(f"\n{'='*60}")
print("Testing additional conditions...")

test_conditions = [
    {v.X('TI'): 0.5, v.T: 700, v.P: 101325},
    {v.X('TI'): 0.1, v.T: 1000, v.P: 101325},
    {v.X('TI'): 0.95, v.T: 800, v.P: 101325},
]

for cond in test_conditions:
    x_ti = cond[v.X('TI')]
    temp = cond[v.T]
    print(f"\nX(TI)={x_ti}, T={temp}K:")
    
    try:
        result_cpu = equilibrium(dbf, comps, phases, cond, gpu=False, verbose=False)
        result_gpu = equilibrium(dbf, comps, phases, cond, gpu=True, verbose=False)
        
        cpu_gm = result_cpu.GM.values.flatten()[0]
        gpu_gm = result_gpu.GM.values.flatten()[0]
        error = abs(gpu_gm - cpu_gm)
        
        print(f"  CPU: {cpu_gm:.2f} J/mol")
        print(f"  GPU: {gpu_gm:.2f} J/mol")
        print(f"  Error: {error:.2f} J/mol", end="")
        
        if error < 1.0:
            print(" ✓")
        else:
            print(" ✗")
            
    except Exception as e:
        print(f"  GPU failed: {type(e).__name__}")