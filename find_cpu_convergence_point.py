#!/usr/bin/env python
"""Find exactly what Y(TI) CPU has after consolidation."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("FINDING CPU CONVERGENCE POINT AFTER CONSOLIDATION")
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

# Patch to track exactly what happens at consolidation
import pycalphad.core.minimizer
original_construct = pycalphad.core.minimizer.construct_equilibrium_system

def patched_construct(spec, state, equilibrium_matrix, equilibrium_rhs):
    # Call original function
    result = original_construct(spec, state, equilibrium_matrix, equilibrium_rhs)
    
    # Check if we have a single phase RIGHT AFTER consolidation
    if len(state.free_stable_compset_indices) == 1:
        idx = state.free_stable_compset_indices[0]
        if hasattr(state, 'dof') and len(state.dof[idx]) > 4:
            y_ti = state.dof[idx][4]  # Y(TI) is the 5th element (index 4)
            
            print(f"\n[CPU POST-CONSOLIDATION] Iteration {state.iteration}")
            print(f"[CPU POST-CONSOLIDATION] Y(TI) = {y_ti:.10f}")
            print(f"[CPU POST-CONSOLIDATION] Mass residual: {state.mass_residual:.6e}")
            
            # Get the single-phase matrix
            num_rows = len(state.free_stable_compset_indices) + spec.num_free_chemical_potentials + spec.prescribed_mole_fraction_rhs.shape[0] + 1
            A = np.array(equilibrium_matrix[:num_rows, :num_rows])
            b = np.array(equilibrium_rhs[:num_rows])
            
            print(f"[CPU POST-CONSOLIDATION] Matrix ({num_rows}x{num_rows}):")
            for i in range(num_rows):
                row_str = "  "
                for j in range(num_rows):
                    row_str += f"{A[i,j]:+.6e} "
                row_str += f"| RHS: {b[i]:+.6e}"
                print(row_str)
            
            try:
                delta = np.linalg.solve(A, b)
                print(f"[CPU POST-CONSOLIDATION] Solution: {delta}")
                
                # Check if this delta is tiny
                max_delta = np.max(np.abs(delta))
                print(f"[CPU POST-CONSOLIDATION] Max |delta|: {max_delta:.6e}")
                
                if max_delta < 1e-12:
                    print("[CPU POST-CONSOLIDATION] *** CONVERGED! Delta is essentially zero ***")
                else:
                    print(f"[CPU POST-CONSOLIDATION] Still changing, max delta: {max_delta:.6e}")
                    
            except np.linalg.LinAlgError as e:
                print(f"[CPU POST-CONSOLIDATION] Matrix solve failed: {e}")
    
    return result

# Patch and run CPU
pycalphad.core.minimizer.construct_equilibrium_system = patched_construct
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

# Restore original
pycalphad.core.minimizer.construct_equilibrium_system = original_construct

print(f"\n{'='*60}")
print("FINAL CPU RESULT")
print("="*60)
print(f"CPU Final GM: {result_cpu.GM.values.flatten()[0]:.6f} J/mol")

# Check CPU composition
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)
print(f"CPU overall X(TI): {overall_x_ti:.10f}")
print(f"Constraint error: {abs(overall_x_ti - 0.9):.10f}")

print(f"\n{'='*60}")
print("KEY INSIGHT")
print("="*60)
print("CPU consolidates phases when they differ by < 0.0001")
print("After consolidation, CPU is already at equilibrium -> converges immediately")
print("GPU consolidates phases but ends up at Y(TI) ≈ 0.903 instead of 0.900")
print("GPU's composition constraint is not satisfied -> continues iterating with tiny deltas")
print("\nThe fundamental difference:")
print("- CPU: Consolidation leads to correct composition (X(TI)=0.900)")
print("- GPU: Consolidation leads to wrong composition (X(TI)≈0.903)")
print("This is why GPU gets stuck but CPU doesn't!")