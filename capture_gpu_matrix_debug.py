#!/usr/bin/env python
"""Capture GPU matrix values and compare with CPU."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("CAPTURING GPU MATRIX VALUES")
print("=" * 60)

# The CPU shows this progression:
# Iteration 1: X(TI) = 0.903147 (two phases)
# Iteration 2: X(TI) = 0.903147 (after consolidation to single phase)  
# Iteration 3: X(TI) = 0.900000 (converged)

# So the critical matrix is from iteration 2 -> 3 transition
# Let me run the GPU and look for its matrix construction debug output

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("Running GPU equilibrium to capture matrix construction...")

# Run GPU calculation 
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
    gpu_np = result_gpu.NP.values.flatten()
    
    gpu_overall = sum(np * x for np, x in zip(gpu_np, gpu_x_ti) if np > 1e-12)
    print(f"GPU final result: X(TI) = {gpu_overall:.10f}")
    
except Exception as e:
    print(f"GPU failed: {e}")

print(f"\n" + "="*60)
print("ANALYSIS: CPU vs GPU at critical iteration")

print(f"\nThe CPU achieves exactly X(TI)=0.9 but GPU gets {gpu_overall:.10f}")
print(f"Difference: {abs(gpu_overall - 0.9):.10f}")

print(f"\nPossible causes:")
print(f"1. GPU constructs different matrix coefficients")
print(f"2. GPU uses different c_G values")  
print(f"3. GPU has different constraint equation setup")
print(f"4. GPU numerical precision differs in matrix solution")

print(f"\nFrom the debug output, I need to find:")
print(f"- GPU's c_G values at the critical iteration")
print(f"- GPU's equilibrium matrix at the critical iteration") 
print(f"- GPU's RHS values at the critical iteration")

print(f"\nThe CPU trace shows:")
print(f"- CPU c_G values: [0.20879587, -0.20879587]")
print(f"- CPU matrix row: [-3.231946e-05, +3.231946e-05, +0.000000e+00]")
print(f"- CPU RHS: +2.056487e-01")

print(f"\nI need to extract the same values from GPU debug output.")
print(f"The difference is likely in one of these numerical values.")