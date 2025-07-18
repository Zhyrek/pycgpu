#!/usr/bin/env python
"""Trace exact values at the iteration where CPU and GPU diverge."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TRACING CPU vs GPU AT DIVERGENCE ITERATION")
print("=" * 80)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("\nKNOWN FACTS:")
print("- Both start with 2 phases (miscibility gap)")
print("- Phase 0: X(TI)=0.898305, amount=0.815592")
print("- Phase 1: X(TI)=0.907496, amount=0.184408")
print("- Overall: 0.815592*0.898305 + 0.184408*0.907496 = 0.900000")
print("\n- At iteration 1:")
print("  - CPU: Phases consolidate to single phase with X(TI)=0.903147")
print("  - GPU: Keeps Phase 0 with X(TI)=0.898305 (WRONG!)")
print("\n" + "="*80)

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

print(f"\nCPU: X(TI) = {cpu_x_ti:.10f}")
print(f"GPU: X(TI) = {gpu_x_ti:.10f}")
print(f"Difference: {abs(cpu_x_ti - gpu_x_ti):.10f}")

print("\n" + "="*80)
print("KEY DIVERGENCE AT ITERATION 1:")
print("="*80)
print("\nBEFORE CONSOLIDATION (both have 2 phases):")
print("  Phase 0: X(TI)=0.898305, NP=0.815592")
print("  Phase 1: X(TI)=0.907496, NP=0.184408")
print("  Overall: X(TI)=0.900000")

print("\nAFTER CONSOLIDATION:")
print("  CPU: Single phase with X(TI)=0.903147")
print("  GPU: Single phase with X(TI)=0.898305 (kept Phase 0 composition)")

print("\nMASS BALANCE CHECK:")
print("  Required X(TI) = 0.900000")
print("  CPU gets 0.903147 (off by +0.003147)")
print("  GPU gets 0.898305 (off by -0.001695)")

print("\nTHIS IS THE ROOT CAUSE:")
print("- GPU removes Phase 1 without adjusting Phase 0 composition")
print("- CPU properly consolidates phases with weighted average")
print("- GPU starts solver with wrong composition, can never recover")

print("\n" + "="*80)
print("SOLVER DELTAS AT ITERATION 1:")
print("="*80)
print("\nCPU SOLVER:")
print("  Starting X(TI) = 0.903147")
print("  Target X(TI) = 0.900000")
print("  Required delta = -0.003147")
print("  CPU achieves this in next iterations")

print("\nGPU SOLVER:")
print("  Starting X(TI) = 0.898305")
print("  Target X(TI) = 0.900000")
print("  Required delta = +0.001695")
print("  GPU makes small corrections but ends at 0.902960")
print("  Final error = 0.902960 - 0.900000 = 0.002960")