#!/usr/bin/env python
"""Investigate why CPU converges with tiny delta_y values but GPU doesn't."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("INVESTIGATING CPU CONVERGENCE WITH TINY DELTA_Y")
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

# Patch the CPU solver to trace what happens when delta_y is tiny
import pycalphad.core.minimizer
original_construct = pycalphad.core.minimizer.construct_equilibrium_system

cpu_convergence_data = []

def patched_construct(spec, state, equilibrium_matrix, equilibrium_rhs):
    global cpu_convergence_data
    
    # Call original function
    result = original_construct(spec, state, equilibrium_matrix, equilibrium_rhs)
    
    # Check if we have a single phase with Y(TI) ≈ 0.903
    if len(state.free_stable_compset_indices) == 1:
        idx = state.free_stable_compset_indices[0]
        compset = state.compsets[idx]
        # Assuming BCC_A2 phase with site fractions at dof[3] and dof[4]
        if hasattr(state, 'dof') and len(state.dof[idx]) > 4:
            y_ti = state.dof[idx][4]  # Y(TI) is the 5th element (index 4)
            if 0.901 < y_ti < 0.905:
                # Solve the equilibrium system
                num_rows = len(state.free_stable_compset_indices) + spec.num_free_chemical_potentials + spec.prescribed_mole_fraction_rhs.shape[0] + 1
                A = np.array(equilibrium_matrix[:num_rows, :num_rows])
                b = np.array(equilibrium_rhs[:num_rows])
                
                try:
                    delta = np.linalg.solve(A, b)
                    
                    # Extract delta_y for site fractions
                    delta_y = None
                    if len(delta) > 4:
                        delta_y = delta[4]  # Y(TI) change
                    
                    print(f"\n[CPU] Iteration {state.iteration}: Y(TI) = {y_ti:.10f}")
                    print(f"[CPU] Mass residual: {state.mass_residual:.6e}")
                    print(f"[CPU] Delta_y(TI): {delta_y:.6e}" if delta_y is not None else "[CPU] Delta_y: N/A")
                    print(f"[CPU] Solution vector: {delta}")
                    
                    # Check convergence criteria
                    print(f"[CPU] Checking convergence:")
                    print(f"  - Mass residual < 1e-8: {state.mass_residual < 1e-8}")
                    print(f"  - Max delta < threshold: {np.max(np.abs(delta)) < 5e-9}")
                    
                    # Store data
                    cpu_convergence_data.append({
                        'iteration': state.iteration,
                        'y_ti': y_ti,
                        'mass_residual': state.mass_residual,
                        'delta_y': delta_y,
                        'max_delta': np.max(np.abs(delta)),
                        'solution': delta.copy()
                    })
                    
                except np.linalg.LinAlgError as e:
                    print(f"[CPU] Matrix solve failed: {e}")
    
    return result

# Patch and run CPU
pycalphad.core.minimizer.construct_equilibrium_system = patched_construct
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

# Restore original
pycalphad.core.minimizer.construct_equilibrium_system = original_construct

print(f"\n{'='*60}")
print("CPU CONVERGENCE ANALYSIS")
print("="*60)
print(f"CPU Final GM: {result_cpu.GM.values.flatten()[0]:.6f} J/mol")

# Check CPU composition
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)
print(f"CPU overall X(TI): {overall_x_ti:.10f}")
print(f"Constraint error: {abs(overall_x_ti - 0.9):.10f}")

if cpu_convergence_data:
    print(f"\nCaptured {len(cpu_convergence_data)} iterations near Y(TI) ≈ 0.903")
    for data in cpu_convergence_data[-3:]:  # Show last 3 iterations
        print(f"Iter {data['iteration']}: Y(TI)={data['y_ti']:.10f}, mass_res={data['mass_residual']:.6e}, delta_y={data['delta_y']:.6e}, max_delta={data['max_delta']:.6e}")

# Now check what the GPU does at the same point
print(f"\n{'='*60}")
print("GPU COMPARISON")
print("="*60)
print("Running GPU to see convergence behavior...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
print(f"GPU Final GM: {result_gpu.GM.values.flatten()[0]:.6f} J/mol")

gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
gpu_np = result_gpu.NP.values.flatten()
overall_x_ti_gpu = sum(np_val * x_ti for np_val, x_ti in zip(gpu_np, gpu_x_ti) if np_val > 1e-12)
print(f"GPU overall X(TI): {overall_x_ti_gpu:.10f}")
print(f"Constraint error: {abs(overall_x_ti_gpu - 0.9):.10f}")

print(f"\nGM difference: {abs(result_cpu.GM.values.flatten()[0] - result_gpu.GM.values.flatten()[0]):.6f} J/mol")
print(f"X(TI) difference: {abs(overall_x_ti - overall_x_ti_gpu):.10f}")