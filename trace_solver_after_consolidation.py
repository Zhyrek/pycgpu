#!/usr/bin/env python
"""Trace solver behavior at iteration 1 after consolidation."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TRACING SOLVER BEHAVIOR AFTER CONSOLIDATION")
print("=" * 80)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

# Run both calculations
print("\nRunning CPU calculation...")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

print("\nRunning GPU calculation...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

print("\n" + "="*80)
print("FINAL RESULTS:")
print("="*80)

cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]

print(f"\nFinal X(TI):")
print(f"  CPU: {cpu_x_ti:.10f}")
print(f"  GPU: {gpu_x_ti:.10f}")
print(f"  Difference: {abs(cpu_x_ti - gpu_x_ti):.10f}")

print("\nKEY OBSERVATIONS:")
print("1. GPU consolidates phases correctly (same as CPU)")
print("2. After consolidation, GPU has single phase with X(TI) ≈ 0.9031")
print("3. This is already > 0.900, so solver makes tiny corrections")
print("4. CPU also starts with X(TI) ≈ 0.9031 but converges to 0.900")
print("\nThe issue is in how the solver handles the consolidated state!")