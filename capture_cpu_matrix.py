#!/usr/bin/env python
"""Capture the CPU equilibrium matrix when it converges to X(TI)=0.9."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Patch to capture the matrix
import pycalphad.core.minimizer

original_solve = pycalphad.core.minimizer.solve_equilibrium
iteration_count = [0]
captured_matrix = [None]
captured_rhs = [None]

def capture_solve(spec, state, equilibrium_matrix, equilibrium_rhs):
    """Capture the matrix and RHS when solving."""
    
    # Check if this is after consolidation (single phase)
    if len(state.free_stable_compset_indices) == 1:
        print(f"\n[CPU CAPTURE] Iteration {state.iteration} - Single phase system")
        
        # Get matrix dimensions
        n_phases = len(state.free_stable_compset_indices)
        n_chem_pot = spec.num_free_chemical_potentials  
        n_constraints = spec.prescribed_mole_fraction_rhs.shape[0]
        n_rows = n_phases + n_chem_pot + n_constraints + 1  # +1 for system amount
        n_cols = n_phases + n_chem_pot + len(spec.free_statevar_indices)
        
        print(f"  Matrix dimensions: {n_rows}x{n_cols}")
        
        # Convert memoryviews to numpy arrays for easier printing
        matrix = np.asarray(equilibrium_matrix)[:n_rows, :n_cols]
        rhs = np.asarray(equilibrium_rhs)[:n_rows]
        
        # Store for later use
        if captured_matrix[0] is None:
            captured_matrix[0] = matrix.copy()
            captured_rhs[0] = rhs.copy()
        
        # Print the full matrix
        print(f"\n[CPU MATRIX] Full equilibrium matrix ({n_rows}x{n_cols}):")
        for i in range(n_rows):
            print(f"  Row {i}: ", end="")
            for j in range(n_cols):
                print(f"{matrix[i,j]:+.10e} ", end="")
            print(f"| RHS: {rhs[i]:+.10e}")
        
        # Check current composition
        idx = state.free_stable_compset_indices[0]
        x_ti = state.phase_compositions[idx, 1]
        print(f"\n  Current X(TI): {x_ti:.10f}")
        print(f"  Target X(TI): 0.9")
        print(f"  Error: {x_ti - 0.9:.10e}")
    
    # Call original
    result = original_solve(spec, state, equilibrium_matrix, equilibrium_rhs)
    
    # Print solution if single phase
    if len(state.free_stable_compset_indices) == 1 and result is not None:
        print(f"\n[CPU SOLUTION] Solution vector:")
        for i in range(len(result)):
            print(f"  x[{i}] = {result[i]:+.15e}")
    
    return result

# Apply patch
pycalphad.core.minimizer.solve_equilibrium = capture_solve

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("CAPTURING CPU EQUILIBRIUM MATRIX")
print("=" * 60)

result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

# Restore
pycalphad.core.minimizer.solve_equilibrium = original_solve

cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)

print(f"\nFINAL: X(TI) = {overall_x_ti:.10f}")

# Save the captured matrix and RHS
if captured_matrix[0] is not None:
    np.save('cpu_matrix.npy', captured_matrix[0])
    np.save('cpu_rhs.npy', captured_rhs[0])
    print(f"\nSaved matrix ({captured_matrix[0].shape}) and RHS to cpu_matrix.npy and cpu_rhs.npy")