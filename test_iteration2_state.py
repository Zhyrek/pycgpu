#!/usr/bin/env python
"""Test CPU vs GPU state at iteration 2."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import numpy as np

# Modify equilibrium to stop after 2 iterations
import pycalphad.core.eqsolver
original_max_iter = pycalphad.core.eqsolver.MAX_SOLVE_ITERATIONS
pycalphad.core.eqsolver.MAX_SOLVE_ITERATIONS = 3

# Also need to modify GPU max iterations
with open('pycalphad/gpu/gpu_equilibrium.py', 'r') as f:
    gpu_content = f.read()

# Replace max_iterations=200 with max_iterations=3 temporarily
gpu_content_modified = gpu_content.replace('max_iterations=200', 'max_iterations=3')
with open('pycalphad/gpu/gpu_equilibrium.py', 'w') as f:
    f.write(gpu_content_modified)

try:
    # Create test case
    dbf = Database('NbTi.tdb')
    comps = ['NB', 'TI', 'VA']
    phases = filter_phases(dbf, comps)

    print("=== Testing CPU vs GPU at Iteration 2 ===")
    print("Testing with T=1000K, X(TI)=0.01")
    print("Limited to 3 iterations\n")

    # Test conditions
    conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

    # Run CPU
    print("Running CPU...")
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, calc_opts={'pdens': 50})
    
    # Run GPU  
    print("\nRunning GPU...")
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, calc_opts={'pdens': 50})

    # Compare results
    print("\n=== Results after 3 iterations ===")
    
    # Phase fractions
    cpu_np = cpu_result.NP.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    
    print("\nPhase fractions (NP):")
    print(f"  CPU: {cpu_np}")
    print(f"  GPU: {gpu_np}")
    
    # Filter out zeros
    cpu_nonzero = cpu_np[cpu_np > 1e-6]
    gpu_nonzero = gpu_np[gpu_np > 1e-6]
    
    if len(cpu_nonzero) > 0 and len(gpu_nonzero) > 0:
        print(f"\nNon-zero phase fractions:")
        print(f"  CPU: {cpu_nonzero}")
        print(f"  GPU: {gpu_nonzero}")
        
        if len(cpu_nonzero) == len(gpu_nonzero):
            diff = np.abs(cpu_nonzero - gpu_nonzero)
            print(f"  Difference: {diff}")
            print(f"  Max difference: {np.max(diff):.2e}")
        
    # Compositions
    cpu_x_ti = cpu_result.X.sel(component='TI').values.flatten()
    gpu_x_ti = gpu_result.X.sel(component='TI').values.flatten()
    
    print("\nX(TI) values:")
    print(f"  CPU: {cpu_x_ti}")
    print(f"  GPU: {gpu_x_ti}")
    
    # Filter NaN
    cpu_x_valid = cpu_x_ti[~np.isnan(cpu_x_ti)]
    gpu_x_valid = gpu_x_ti[~np.isnan(gpu_x_ti)][:len(cpu_x_valid)]
    
    if len(cpu_x_valid) > 0 and len(gpu_x_valid) > 0:
        x_diff = np.abs(cpu_x_valid - gpu_x_valid)
        print(f"\nValid X(TI) difference: {x_diff}")
        print(f"Max X(TI) difference: {np.max(x_diff):.2e}")
        
        if np.max(x_diff) > 1e-10:
            print("\n⚠️  DIVERGENCE ALREADY PRESENT AT ITERATION 2!")
        else:
            print("\n✓ CPU and GPU still match at iteration 2")

finally:
    # Restore original settings
    pycalphad.core.eqsolver.MAX_SOLVE_ITERATIONS = original_max_iter
    with open('pycalphad/gpu/gpu_equilibrium.py', 'w') as f:
        f.write(gpu_content)