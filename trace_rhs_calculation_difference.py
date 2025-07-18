#!/usr/bin/env python
"""Trace why GPU RHS differs from CPU RHS in single-phase constraint."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TRACING RHS CALCULATION DIFFERENCE")
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

# First, let's run both CPU and GPU and capture their RHS values
print("Running CPU to get reference RHS...")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

print(f"\nCPU result:")
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti_cpu = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)
print(f"CPU overall X(TI): {overall_x_ti_cpu:.10f}")

print(f"\n{'='*60}")
print("Running GPU to get its RHS...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

print(f"\nGPU result:")
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
gpu_np = result_gpu.NP.values.flatten()
overall_x_ti_gpu = sum(np_val * x_ti for np_val, x_ti in zip(gpu_np, gpu_x_ti) if np_val > 1e-12)
print(f"GPU overall X(TI): {overall_x_ti_gpu:.10f}")

print(f"\n{'='*60}")
print("ANALYSIS FROM OUTPUT ABOVE")
print("="*60)
print("From the output above, we can see:")
print("1. CPU single-phase matrix Row 1 RHS: +2.056487e-01 ≈ 0.206")
print("2. GPU single-phase matrix Row 1 RHS: +1.000000e-01 = 0.100")
print()
print("The mole fraction constraint equation is:")
print("sum(phase_amt * c_component * y_change) = constraint_rhs")
print()
print("The RHS represents the current composition error and c_G contributions.")
print("CPU RHS ≈ 0.206 vs GPU RHS = 0.100 suggests:")
print()
print("Key differences to investigate:")
print("1. Different c_G values during constraint RHS calculation")
print("2. Different phase compositions at consolidation point") 
print("3. Different mass_jac values")
print("4. Different moles_normalization_grad values")
print()
print("From CPU debug output, we see:")
print("- CPU: c_G = [ 0.20879587 -0.20879587] at single-phase")
print("- Need to check what GPU c_G values are")
print()
print("The formula for constraint RHS involves:")
print("rhs += phase_amt * c_G[component] / system_amt")
print("So if c_G differs, RHS will differ")
print()
print("Next step: Compare c_G calculation between CPU and GPU")
print("to find why CPU gets ~0.209 but GPU gets 0.100")