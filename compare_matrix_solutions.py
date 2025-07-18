#!/usr/bin/env python
"""Compare CPU and GPU equilibrium matrices and solutions when Y(TI) ≈ 0.903."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("COMPARING CPU AND GPU MATRICES AT Y(TI) ≈ 0.903")
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

# First, let's check what iteration the CPU has Y(TI) ≈ 0.903
print("\nChecking CPU matrices...")
print("Running CPU calculation to find when Y(TI) ≈ 0.903...")

# We need to patch the CPU solver to capture the matrix at the right iteration
import pycalphad.core.minimizer
original_construct = pycalphad.core.minimizer.construct_equilibrium_system

cpu_matrix_data = {}
cpu_found_target = False

def patched_construct(spec, state, equilibrium_matrix, equilibrium_rhs):
    global cpu_found_target
    
    # Call original function
    result = original_construct(spec, state, equilibrium_matrix, equilibrium_rhs)
    
    # Check if we have a single phase with Y(TI) ≈ 0.903
    if len(state.free_stable_compset_indices) == 1 and not cpu_found_target:
        idx = state.free_stable_compset_indices[0]
        compset = state.compsets[idx]
        # Assuming BCC_A2 phase with site fractions at dof[3] and dof[4]
        if hasattr(state, 'dof') and len(state.dof[idx]) > 4:
            y_ti = state.dof[idx][4]  # Y(TI) is the 5th element (index 4)
            if 0.902 < y_ti < 0.904:
                print(f"\n[CPU] Found target at iteration {state.iteration}: Y(TI) = {y_ti:.10f}")
                
                # Save matrix data
                cpu_matrix_data['iteration'] = state.iteration
                cpu_matrix_data['y_ti'] = y_ti
                cpu_matrix_data['matrix'] = np.array(equilibrium_matrix).copy()
                cpu_matrix_data['rhs'] = np.array(equilibrium_rhs).copy()
                cpu_matrix_data['num_rows'] = len(state.free_stable_compset_indices) + spec.num_free_chemical_potentials + spec.prescribed_mole_fraction_rhs.shape[0] + 1
                
                # Print matrix
                print(f"[CPU] Equilibrium matrix at Y(TI)={y_ti:.10f}:")
                for i in range(cpu_matrix_data['num_rows']):
                    print(f"  Row {i}: ", end="")
                    for j in range(cpu_matrix_data['num_rows']):
                        print(f"{equilibrium_matrix[i,j]:+.6e} ", end="")
                    print(f"| RHS: {equilibrium_rhs[i]:+.6e}")
                
                cpu_found_target = True
    
    return result

# Patch and run CPU
pycalphad.core.minimizer.construct_equilibrium_system = patched_construct
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

# Restore original
pycalphad.core.minimizer.construct_equilibrium_system = original_construct

if not cpu_found_target:
    print("[CPU] Never found Y(TI) ≈ 0.903")
else:
    print(f"\n[CPU] Matrix captured at iteration {cpu_matrix_data['iteration']}")

# Now run GPU and capture its matrix
print("\n" + "="*60)
print("Running GPU calculation...")

# The GPU already prints its matrix, so we just need to run it
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"CPU GM: {result_cpu.GM.values.flatten()[0]:.6f} J/mol")
print(f"GPU GM: {result_gpu.GM.values.flatten()[0]:.6f} J/mol")
print(f"Error: {abs(result_cpu.GM.values.flatten()[0] - result_gpu.GM.values.flatten()[0]):.6f} J/mol")

# To compare matrices more carefully, we need to look at the GPU output
print("\nTo compare matrices, look for GPU output with Y(TI) ≈ 0.903")
print("The GPU should print its equilibrium matrix at each iteration.")
print("\nKey things to compare:")
print("1. Matrix coefficients (especially in row 1 for mole fraction constraint)")
print("2. RHS values")
print("3. Solution vector from SVD")
print("4. Whether the matrices are identical to within numerical precision")