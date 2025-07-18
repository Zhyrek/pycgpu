#!/usr/bin/env python
"""Trace specifically what happens when CPU gets tiny delta_y values."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TRACING CPU TINY DELTA HANDLING")
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

# Patch the CPU solver to track what happens after tiny delta_y solutions
import pycalphad.core.minimizer
original_advance = pycalphad.core.minimizer.advance_state

tiny_delta_iterations = []

def patched_advance(spec, state, equilibrium_matrix, delta):
    global tiny_delta_iterations
    
    # Check if we have a single phase system with tiny delta_y
    if len(state.free_stable_compset_indices) == 1:
        idx = state.free_stable_compset_indices[0]
        if hasattr(state, 'dof') and len(state.dof[idx]) > 4:
            y_ti = state.dof[idx][4]  # Y(TI) is the 5th element (index 4)
            if 0.901 < y_ti < 0.905:  # Near the problematic region
                # Check delta_y magnitude
                if len(delta) > 4:
                    delta_y = delta[4]  # Y(TI) change
                    if abs(delta_y) < 1e-6:  # Very tiny delta
                        print(f"\n[CPU TINY DELTA] Iteration {state.iteration}: Y(TI) = {y_ti:.10f}")
                        print(f"[CPU TINY DELTA] Delta_y(TI): {delta_y:.6e}")
                        print(f"[CPU TINY DELTA] Full delta: {delta}")
                        print(f"[CPU TINY DELTA] Mass residual before advance: {state.mass_residual:.6e}")
                        
                        # Store before advancing
                        old_dof = state.dof[idx].copy()
                        
                        # Call original advance
                        result = original_advance(spec, state, equilibrium_matrix, delta)
                        
                        # Check what changed
                        new_dof = state.dof[idx]
                        actual_change = new_dof[4] - old_dof[4]
                        
                        print(f"[CPU TINY DELTA] Old Y(TI): {old_dof[4]:.10f}")
                        print(f"[CPU TINY DELTA] New Y(TI): {new_dof[4]:.10f}")
                        print(f"[CPU TINY DELTA] Actual change: {actual_change:.6e}")
                        print(f"[CPU TINY DELTA] Mass residual after advance: {state.mass_residual:.6e}")
                        
                        # Check convergence criteria
                        print(f"[CPU TINY DELTA] Checking convergence after advance:")
                        print(f"  - Mass residual < 1e-8: {state.mass_residual < 1e-8}")
                        print(f"  - Actual change < 5e-9: {abs(actual_change) < 5e-9}")
                        
                        tiny_delta_iterations.append({
                            'iteration': state.iteration,
                            'old_y_ti': old_dof[4],
                            'new_y_ti': new_dof[4],
                            'delta_y': delta_y,
                            'actual_change': actual_change,
                            'mass_residual': state.mass_residual
                        })
                        
                        return result
    
    # Default case
    return original_advance(spec, state, equilibrium_matrix, delta)

# Patch and run CPU
pycalphad.core.minimizer.advance_state = patched_advance
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

# Restore original
pycalphad.core.minimizer.advance_state = original_advance

print(f"\n{'='*60}")
print("CPU TINY DELTA ANALYSIS")
print("="*60)
print(f"CPU Final GM: {result_cpu.GM.values.flatten()[0]:.6f} J/mol")

# Check CPU composition
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)
print(f"CPU overall X(TI): {overall_x_ti:.10f}")
print(f"Constraint error: {abs(overall_x_ti - 0.9):.10f}")

if tiny_delta_iterations:
    print(f"\nCaptured {len(tiny_delta_iterations)} tiny delta iterations:")
    for data in tiny_delta_iterations:
        print(f"Iter {data['iteration']}: old_Y(TI)={data['old_y_ti']:.10f}, delta_y={data['delta_y']:.6e}, actual_change={data['actual_change']:.6e}, mass_res={data['mass_residual']:.6e}")
        
    # Check if CPU actually converged despite tiny deltas
    final_data = tiny_delta_iterations[-1]
    print(f"\nFinal tiny delta iteration analysis:")
    print(f"  - Final Y(TI): {final_data['new_y_ti']:.10f}")
    print(f"  - Final mass residual: {final_data['mass_residual']:.6e}")
    print(f"  - Converged with mass_res < 1e-8: {final_data['mass_residual'] < 1e-8}")
else:
    print("No tiny delta iterations captured - CPU may converge differently")

print(f"\n{'='*60}")
print("KEY INSIGHT")
print("="*60)
print("The question is: Does CPU get similar tiny delta_y values as GPU?")
print("If YES: Why does CPU converge but GPU doesn't?")
print("If NO: What's different about CPU's matrix/solver behavior?")