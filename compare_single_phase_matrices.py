#!/usr/bin/env python
"""Compare single-phase equilibrium matrices between CPU and GPU."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("COMPARING SINGLE-PHASE MATRICES: CPU vs GPU")
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

# Patch the CPU solver to capture single-phase matrices
import pycalphad.core.minimizer
original_construct = pycalphad.core.minimizer.construct_equilibrium_system

cpu_single_phase_data = []

def patched_construct(spec, state, equilibrium_matrix, equilibrium_rhs):
    global cpu_single_phase_data
    
    # Call original function
    result = original_construct(spec, state, equilibrium_matrix, equilibrium_rhs)
    
    # Check if we have a single phase
    if len(state.free_stable_compset_indices) == 1:
        idx = state.free_stable_compset_indices[0]
        if hasattr(state, 'dof') and len(state.dof[idx]) > 4:
            y_ti = state.dof[idx][4]  # Y(TI) is the 5th element (index 4)
            
            # Solve the equilibrium system
            num_rows = len(state.free_stable_compset_indices) + spec.num_free_chemical_potentials + spec.prescribed_mole_fraction_rhs.shape[0] + 1
            A = np.array(equilibrium_matrix[:num_rows, :num_rows])
            b = np.array(equilibrium_rhs[:num_rows])
            
            try:
                delta = np.linalg.solve(A, b)
                
                print(f"\n[CPU SINGLE PHASE] Iteration {state.iteration}: Y(TI) = {y_ti:.10f}")
                print(f"[CPU SINGLE PHASE] Mass residual: {state.mass_residual:.6e}")
                
                # Print matrix in detail
                print(f"[CPU SINGLE PHASE] Matrix (3x3):")
                for i in range(3):
                    row_str = "  "
                    for j in range(3):
                        row_str += f"{A[i,j]:+.6e} "
                    row_str += f"| RHS: {b[i]:+.6e}"
                    print(row_str)
                
                print(f"[CPU SINGLE PHASE] Solution: {delta}")
                
                # Extract delta_y for site fractions  
                if len(delta) > 2:
                    delta_y = delta[2]  # In single phase, Y(TI) change should be at index 2
                    print(f"[CPU SINGLE PHASE] Delta_y(TI): {delta_y:.6e}")
                
                # Check matrix conditioning
                cond_num = np.linalg.cond(A)
                print(f"[CPU SINGLE PHASE] Matrix condition number: {cond_num:.6e}")
                
                # Store data
                cpu_single_phase_data.append({
                    'iteration': state.iteration,
                    'y_ti': y_ti,
                    'mass_residual': state.mass_residual,
                    'matrix': A.copy(),
                    'rhs': b.copy(),
                    'solution': delta.copy(),
                    'condition_number': cond_num
                })
                
            except np.linalg.LinAlgError as e:
                print(f"[CPU SINGLE PHASE] Matrix solve failed: {e}")
    
    return result

# Patch and run CPU
pycalphad.core.minimizer.construct_equilibrium_system = patched_construct
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

# Restore original
pycalphad.core.minimizer.construct_equilibrium_system = original_construct

print(f"\n{'='*60}")
print("CPU SINGLE-PHASE ANALYSIS")
print("="*60)
print(f"CPU Final GM: {result_cpu.GM.values.flatten()[0]:.6f} J/mol")

# Check CPU composition
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)
print(f"CPU overall X(TI): {overall_x_ti:.10f}")
print(f"Constraint error: {abs(overall_x_ti - 0.9):.10f}")

if cpu_single_phase_data:
    print(f"\nCaptured {len(cpu_single_phase_data)} single-phase iterations")
    
    # Show the first few single-phase iterations
    for i, data in enumerate(cpu_single_phase_data[:3]):
        print(f"\nIteration {data['iteration']}:")
        print(f"  Y(TI): {data['y_ti']:.10f}")
        print(f"  Mass residual: {data['mass_residual']:.6e}")
        print(f"  Condition number: {data['condition_number']:.6e}")
        if len(data['solution']) > 2:
            print(f"  Delta_y(TI): {data['solution'][2]:.6e}")
        print(f"  Solution: {data['solution']}")
        
    # Compare with what GPU should get
    print(f"\n{'='*40}")
    print("KEY COMPARISON WITH GPU:")
    print("="*40)
    if len(cpu_single_phase_data) > 0:
        first_single = cpu_single_phase_data[0]
        print(f"CPU first single-phase Y(TI): {first_single['y_ti']:.10f}")
        print(f"GPU gets stuck around Y(TI): 0.9029600000")
        print(f"Difference: {abs(first_single['y_ti'] - 0.902960):.10f}")
        
        if len(first_single['solution']) > 2:
            print(f"CPU delta_y(TI): {first_single['solution'][2]:.6e}")
            print("GPU delta_y(TI): ~3e-07 (too small)")
            
        print(f"CPU mass residual: {first_single['mass_residual']:.6e}")
        print("GPU mass residual: ~3e-03 (fails to converge)")
else:
    print("No single-phase iterations captured")

print(f"\n{'='*60}")
print("GPU COMPARISON")
print("="*60)
print("Running GPU to see what matrices it gets...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
print(f"GPU Final GM: {result_gpu.GM.values.flatten()[0]:.6f} J/mol")

gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
gpu_np = result_gpu.NP.values.flatten()
overall_x_ti_gpu = sum(np_val * x_ti for np_val, x_ti in zip(gpu_np, gpu_x_ti) if np_val > 1e-12)
print(f"GPU overall X(TI): {overall_x_ti_gpu:.10f}")
print(f"Constraint error: {abs(overall_x_ti_gpu - 0.9):.10f}")

print(f"\nGM difference: {abs(result_cpu.GM.values.flatten()[0] - result_gpu.GM.values.flatten()[0]):.6f} J/mol")
print(f"X(TI) difference: {abs(overall_x_ti - overall_x_ti_gpu):.10f}")