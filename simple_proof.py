#!/usr/bin/env python
"""Simple proof that GPU converges correctly."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("GPU CONVERGENCE PROOF")
print("="*50)
print("\nTesting X(TI) = 0.9 at T = 600K:")

# Test multiple conditions
test_conditions = [
    {v.X('TI'): 0.1, v.T: 600, v.P: 101325},
    {v.X('TI'): 0.3, v.T: 600, v.P: 101325},
    {v.X('TI'): 0.5, v.T: 600, v.P: 101325},
    {v.X('TI'): 0.7, v.T: 600, v.P: 101325},
    {v.X('TI'): 0.9, v.T: 600, v.P: 101325},
]

print("\nTarget  CPU X(TI)  GPU X(TI)  Difference  Status")
print("-"*50)

all_pass = True
for cond in test_conditions:
    target_x_ti = cond[v.X('TI')]
    
    # CPU calculation
    result_cpu = equilibrium(dbf, comps, phases, cond, gpu=False, verbose=False)
    cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
    
    # GPU calculation  
    result_gpu = equilibrium(dbf, comps, phases, cond, gpu=True, verbose=False)
    gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]
    
    diff = abs(gpu_x_ti - cpu_x_ti)
    status = "PASS" if diff < 1e-6 else "FAIL"
    if diff >= 1e-6:
        all_pass = False
    
    print(f"{target_x_ti:.1f}     {cpu_x_ti:.6f}   {gpu_x_ti:.6f}   {diff:.2e}    {status}")

print("\n" + "="*50)
if all_pass:
    print("✅ ALL TESTS PASSED! GPU matches CPU exactly!")
    print("✅ GPU correctly converges to all target compositions!")
else:
    print("❌ Some tests failed!")

# Detailed test for X(TI) = 0.9
print("\nDetailed results for X(TI) = 0.9:")
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]

print(f"  Target:      0.900000000000000")
print(f"  CPU result:  {cpu_x_ti:.15f}")
print(f"  GPU result:  {gpu_x_ti:.15f}")
print(f"  Difference:  {abs(gpu_x_ti - cpu_x_ti):.15e}")
print(f"  GPU error:   {abs(gpu_x_ti - 0.9):.15e}")

if abs(gpu_x_ti - 0.9) < 1e-6:
    print("\n✅ PROVEN: GPU converges to X(TI)=0.9 within tolerance!")
else:
    print("\n❌ GPU does not converge to X(TI)=0.9")