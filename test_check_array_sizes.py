#!/usr/bin/env python
"""Check array sizes and memory allocations for GPU."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.gpu.gpu_codegen import compute_dynamic_kernel_sizes
from pycalphad.core.workspace import Workspace

def check_array_sizes():
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases_5 = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID']
    
    # Create workspace
    from pycalphad import Model
    models = {phase: Model(dbf, comps, phase) for phase in phases_5}
    wks = Workspace(database=dbf, components=comps, phases=phases_5, models=models, 
                    conditions={v.T: 300, v.P: 101325, v.X('BI'): 0.1})
    
    # Get dynamic sizes
    sizes = compute_dynamic_kernel_sizes(wks)
    
    # Calculate all array dimensions
    MAX_SVD_DIM = sizes['MAX_COMPONENTS'] + sizes['MAX_PHASES'] + sizes['MAX_STATEVARS'] + sizes['MAX_FIXED_MOLE_FRACTION_CONDITIONS'] + 2
    MAX_PHASE_MATRIX_DIM = sizes['MAX_DOF_PER_PHASE'] + sizes['MAX_INTERNAL_CONSTRAINTS']
    MAX_DOF_SIZE = sizes['MAX_STATEVARS'] + sizes['MAX_DOF_PER_PHASE']
    MAX_EQ_MATRIX_ROWS = 2 * sizes['MAX_PHASES'] + sizes['MAX_COMPONENTS'] + 1
    MAX_EQ_MATRIX_COLS = sizes['MAX_COMPONENTS'] + sizes['MAX_PHASES'] + sizes['MAX_STATEVARS']
    MAX_EQ_MATRIX_SIZE = MAX_EQ_MATRIX_ROWS * MAX_EQ_MATRIX_COLS
    
    print("Array dimensions:")
    print(f"  MAX_SVD_DIM: {MAX_SVD_DIM}")
    print(f"  MAX_PHASE_MATRIX_DIM: {MAX_PHASE_MATRIX_DIM}")
    print(f"  MAX_DOF_SIZE: {MAX_DOF_SIZE}")
    print(f"  MAX_EQ_MATRIX_ROWS: {MAX_EQ_MATRIX_ROWS}")
    print(f"  MAX_EQ_MATRIX_COLS: {MAX_EQ_MATRIX_COLS}")
    print(f"  MAX_EQ_MATRIX_SIZE: {MAX_EQ_MATRIX_SIZE}")
    
    # Check specific array sizes
    print("\nGlobal memory array sizes (elements per condition):")
    arrays = {
        'A_lstsq_copy': MAX_SVD_DIM * MAX_SVD_DIM,
        'U_lstsq': MAX_SVD_DIM * MAX_SVD_DIM,
        'V_lstsq': MAX_SVD_DIM * MAX_SVD_DIM,
        'singular_values_lstsq': MAX_SVD_DIM,
        'superdiag_lstsq': MAX_SVD_DIM,
        'U_inv': MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM,
        'V_inv': MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM,
        'singular_values_inv': MAX_PHASE_MATRIX_DIM,
        'superdiag_inv': MAX_PHASE_MATRIX_DIM,
        'work_inv': MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM,
        'x_dof': MAX_DOF_SIZE,
        'grad': MAX_DOF_SIZE,
        'hess': MAX_DOF_SIZE * MAX_DOF_SIZE,
        'masses': sizes['MAX_COMPONENTS'],
        'mass_jac': sizes['MAX_COMPONENTS'] * MAX_DOF_SIZE,
        'phase_matrix': MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM,
        'equilibrium_matrix': MAX_EQ_MATRIX_SIZE,
        'equilibrium_rhs': MAX_EQ_MATRIX_ROWS,
        'eq_soln': MAX_EQ_MATRIX_COLS
    }
    
    for name, size in arrays.items():
        print(f"  {name}: {size} elements")
    
    # Check memory for different condition counts
    print("\nMemory requirements:")
    for num_conds in [1, 10, 14, 15, 20, 40]:
        total_elements = sum(arrays.values()) * num_conds
        total_mb = (total_elements * 8) / (1024 * 1024)
        print(f"  {num_conds} conditions: {total_mb:.2f} MB")
    
    # Check if any sizes might cause issues
    print("\nPotential issues:")
    if MAX_SVD_DIM > 32:
        print(f"  WARNING: MAX_SVD_DIM={MAX_SVD_DIM} is quite large")
    if MAX_EQ_MATRIX_SIZE > 1000:
        print(f"  WARNING: MAX_EQ_MATRIX_SIZE={MAX_EQ_MATRIX_SIZE} is quite large")

if __name__ == "__main__":
    check_array_sizes()