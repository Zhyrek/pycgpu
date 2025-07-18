#!/usr/bin/env python
"""Compare matrix conditioning between CPU and GPU after consolidation."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("COMPARING MATRIX CONDITIONING: CPU vs GPU")
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

# Patch CPU to capture post-consolidation matrix details
import pycalphad.core.minimizer
original_construct = pycalphad.core.minimizer.construct_equilibrium_system

cpu_matrix_data = []

def patched_construct(spec, state, equilibrium_matrix, equilibrium_rhs):
    result = original_construct(spec, state, equilibrium_matrix, equilibrium_rhs)
    
    # Look for single-phase iterations after consolidation
    if len(state.free_stable_compset_indices) == 1 and state.iteration >= 2:
        idx = state.free_stable_compset_indices[0]
        if hasattr(state, 'dof') and len(state.dof[idx]) > 4:
            y_ti = state.dof[idx][4]
            
            # Get the equilibrium matrix
            num_rows = len(state.free_stable_compset_indices) + spec.num_free_chemical_potentials + spec.prescribed_mole_fraction_rhs.shape[0] + 1
            A = np.array(equilibrium_matrix[:num_rows, :num_rows])
            b = np.array(equilibrium_rhs[:num_rows])
            
            print(f"\n[CPU MATRIX] Iteration {state.iteration}: Y(TI) = {y_ti:.10f}")
            print(f"[CPU MATRIX] Mass residual: {state.mass_residual:.6e}")
            print(f"[CPU MATRIX] Matrix size: {A.shape}")
            
            # Detailed matrix analysis
            cond_num = np.linalg.cond(A)
            det = np.linalg.det(A)
            
            print(f"[CPU MATRIX] Condition number: {cond_num:.6e}")
            print(f"[CPU MATRIX] Determinant: {det:.6e}")
            
            # Print the actual matrix values
            print(f"[CPU MATRIX] Matrix:")
            for i in range(A.shape[0]):
                row_str = "  "
                for j in range(A.shape[1]):
                    row_str += f"{A[i,j]:+.6e} "
                row_str += f"| RHS: {b[i]:+.6e}"
                print(row_str)
            
            # Solve and analyze solution
            try:
                delta = np.linalg.solve(A, b)
                max_delta = np.max(np.abs(delta))
                print(f"[CPU MATRIX] Solution: {delta}")
                print(f"[CPU MATRIX] Max |delta|: {max_delta:.6e}")
                
                # Store for comparison
                cpu_matrix_data.append({
                    'iteration': state.iteration,
                    'y_ti': y_ti,
                    'mass_residual': state.mass_residual,
                    'condition_number': cond_num,
                    'determinant': det,
                    'matrix': A.copy(),
                    'rhs': b.copy(),
                    'solution': delta.copy(),
                    'max_delta': max_delta
                })
                
            except np.linalg.LinAlgError as e:
                print(f"[CPU MATRIX] Solve failed: {e}")
    
    return result

# Run CPU with matrix analysis
pycalphad.core.minimizer.construct_equilibrium_system = patched_construct
print("Running CPU calculation...")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
pycalphad.core.minimizer.construct_equilibrium_system = original_construct

print(f"\n{'='*60}")
print("CPU RESULTS")
print("="*60)
print(f"CPU Final GM: {result_cpu.GM.values.flatten()[0]:.6f} J/mol")

cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)
print(f"CPU overall X(TI): {overall_x_ti:.10f}")
print(f"Constraint error: {abs(overall_x_ti - 0.9):.10f}")

if cpu_matrix_data:
    print(f"\nCPU single-phase matrix analysis:")
    for data in cpu_matrix_data:
        print(f"  Iter {data['iteration']}: cond={data['condition_number']:.2e}, max_delta={data['max_delta']:.2e}, Y(TI)={data['y_ti']:.10f}")

print(f"\n{'='*60}")
print("GPU COMPARISON")  
print("="*60)
print("Running GPU calculation...")

# The GPU should print its own matrix details when it gets stuck
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

print(f"\nGPU Final GM: {result_gpu.GM.values.flatten()[0]:.6f} J/mol")
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
gpu_np = result_gpu.NP.values.flatten()
overall_x_ti_gpu = sum(np_val * x_ti for np_val, x_ti in zip(gpu_np, gpu_x_ti) if np_val > 1e-12)
print(f"GPU overall X(TI): {overall_x_ti_gpu:.10f}")
print(f"Constraint error: {abs(overall_x_ti_gpu - 0.9):.10f}")

print(f"\n{'='*60}")
print("CONDITIONING ANALYSIS")
print("="*60)
print("Key differences to look for:")
print("1. CPU condition numbers should be reasonable (< 1e12)")
print("2. GPU condition numbers should be very large (> 1e12)")
print("3. CPU deltas should decrease each iteration")
print("4. GPU deltas should be tiny and make no progress")
print("\nThe question is: WHY do the matrices have different conditioning?")
print("Possible causes:")
print("- Different Hessian values")
print("- Different c_G values") 
print("- Different mass_jac values")
print("- Different site fraction handling")
print("Look at the GPU matrix output above to compare with CPU matrices.")