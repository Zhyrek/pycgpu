#!/usr/bin/env python
"""Trace exact solver values at divergence point."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TRACING EXACT SOLVER VALUES AT DIVERGENCE")
print("=" * 80)

# Add custom debug prints to capture exact values
import pycalphad.core.minimizer as cpu_min
import pycalphad.gpu.gpu_equilibrium as gpu_eq

# Store original functions
orig_cpu_advance = cpu_min.advance_state
orig_gpu_print = print

# Track CPU values
cpu_iteration_data = {}

def cpu_advance_with_trace(state, eq_soln, step_size):
    it = state.iteration
    if it == 0 or it == 1:
        # Extract phase data before advance
        print(f"\n[CPU TRACE ITERATION {it}] Before advance_state:")
        print(f"  num_stable_phases: {len(state.stable_phases)}")
        for i, idx in enumerate(state.free_stable_compset_indices):
            cs = state.compsets[idx]
            print(f"  Phase {i}: NP={cs.NP:.10f}, X(TI)={cs.X[1]:.10f}")
        
        # Extract solution vector
        print(f"  Solution vector (first 4): {eq_soln[:4]}")
        print(f"  Step size: {step_size}")
        
    # Call original
    result = orig_cpu_advance(state, eq_soln, step_size)
    
    if it == 0 or it == 1:
        # Extract phase data after advance
        print(f"\n[CPU TRACE ITERATION {it}] After advance_state:")
        for i, idx in enumerate(state.free_stable_compset_indices):
            cs = state.compsets[idx]
            print(f"  Phase {i}: NP={cs.NP:.10f}, X(TI)={cs.X[1]:.10f}")
            
    return result

# Monkey patch
cpu_min.advance_state = cpu_advance_with_trace

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("\nRunning CPU calculation with tracing...")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

print("\n" + "="*80)
print("Running GPU calculation...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

# Restore
cpu_min.advance_state = orig_cpu_advance

print("\n" + "="*80)
print("FINAL COMPARISON:")
print("="*80)

cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]

print(f"\nFinal X(TI):")
print(f"  CPU: {cpu_x_ti:.10f}")
print(f"  GPU: {gpu_x_ti:.10f}")
print(f"  Difference: {abs(cpu_x_ti - gpu_x_ti):.10f}")