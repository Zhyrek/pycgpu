#!/usr/bin/env python
"""Debug GPU RHS calculation by examining the exact values."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("DEBUGGING GPU RHS CALCULATION")
print("=" * 60)

# Clear cache
cache_dir = os.path.expanduser('~/.cache/pycalphad_gpu')
if os.path.exists(cache_dir):
    import shutil
    shutil.rmtree(cache_dir)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("Running GPU to examine RHS calculation...")

# Run GPU and capture the debug output
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

print(f"\nGPU Final result:")
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
gpu_np = result_gpu.NP.values.flatten()
overall_x_ti_gpu = sum(np_val * x_ti for np_val, x_ti in zip(gpu_np, gpu_x_ti) if np_val > 1e-12)
print(f"GPU overall X(TI): {overall_x_ti_gpu:.10f}")
print(f"Constraint error: {abs(overall_x_ti_gpu - 0.9):.10f}")

print(f"\n{'='*60}")
print("ANALYSIS OF GPU OUTPUT")
print("="*60)
print("From the GPU debug output above, look for:")
print("1. 'GPU MOLE FRAC RHS DEBUG' messages showing c_G values")
print("2. RHS contributions from each phase")
print("3. Final matrix RHS values")
print()
print("Key patterns to find:")
print("- GPU c_G values: should be around [0.209, -0.209] like CPU")
print("- GPU RHS contributions: should sum to ~0.209 before residual")
print("- GPU final RHS: currently shows exactly 0.100 (wrong)")
print()
print("Expected CPU calculation:")
print("- CPU c_G ≈ [0.20879587, -0.20879587]")
print("- CPU RHS before residual ≈ 0.20879587")
print("- CPU RHS after residual ≈ 0.205649")
print()
print("The fix should ensure GPU calculates RHS the same way as CPU.")
print("Check for hard-coded values, wrong formulas, or missing terms.")