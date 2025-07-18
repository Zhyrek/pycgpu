#!/usr/bin/env python
"""Capture the CPU equilibrium matrix when it converges to X(TI)=0.9."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Import cython functions
from pycalphad.core.minimizer import lstsq

# Store captured data
captured_data = {'matrix': None, 'rhs': None, 'solution': None, 'iteration': -1}

# Monkey patch the lstsq function
original_lstsq = lstsq

def capture_lstsq(matrix_ptr, nrows, ncols, rhs_ptr, tolerance):
    """Capture matrix and RHS, then call original lstsq."""
    
    # Only capture after consolidation (3x3 system)
    if nrows == 3 and ncols == 3:
        # Create numpy arrays from pointers
        # This is a bit hacky but works for debugging
        import ctypes
        
        # Convert to numpy arrays
        matrix_arr = np.zeros((nrows, ncols))
        rhs_arr = np.zeros(nrows)
        
        print(f"\n[CPU LSTSQ CAPTURE] {nrows}x{ncols} system")
        print("Note: Direct pointer access not available in Python, showing conceptual capture")
        
        if captured_data['matrix'] is None:
            captured_data['matrix'] = matrix_arr
            captured_data['rhs'] = rhs_arr
            captured_data['iteration'] = 0
    
    # Call original
    return original_lstsq(matrix_ptr, nrows, ncols, rhs_ptr, tolerance)

# Instead, let's add debug output to the CPU code
# First run with verbose to see the matrix
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("CAPTURING CPU EQUILIBRIUM MATRIX")
print("=" * 60)
print("\nRunning CPU equilibrium with verbose output...")
print("Look for the 3x3 matrix after consolidation")

result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)

cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)

print(f"\nFINAL: X(TI) = {overall_x_ti:.10f}")