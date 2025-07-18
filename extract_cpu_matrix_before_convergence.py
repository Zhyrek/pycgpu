#!/usr/bin/env python
"""Extract CPU's equilibrium matrix from iteration before reaching X(TI)=0.9."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Add debugging to CPU minimizer to capture matrix values
import sys
import os

# Patch the CPU minimizer to capture matrix values
def patch_cpu_minimizer():
    import pycalphad.core.minimizer
    original_solve_equilibrium_at_condition = pycalphad.core.minimizer.solve_equilibrium_at_condition
    
    # Store matrix data
    matrix_data = {}
    
    def patched_solve_equilibrium_at_condition(*args, **kwargs):
        result = original_solve_equilibrium_at_condition(*args, **kwargs)
        
        # Try to access the solver's internal state
        # This is a bit hacky but we need to see the matrix before final convergence
        if hasattr(result, '_debug_matrix_data'):
            matrix_data.update(result._debug_matrix_data)
            
        return result
    
    pycalphad.core.minimizer.solve_equilibrium_at_condition = patched_solve_equilibrium_at_condition
    return matrix_data

print("EXTRACTING CPU MATRIX VALUES BEFORE CONVERGENCE")
print("=" * 60)

# Set up the problem
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

# We need to add debug prints to the actual CPU code to capture the matrix
# Let's create a modified version that traces the exact values

print("\nTo extract the CPU matrix values, we need to modify the CPU minimizer code.")
print("The matrix we want is from iteration 2 (after consolidation, before final convergence).")
print("\nFrom the trace output, we know:")
print("- Iteration 2: Single phase, X(TI)=0.903147")
print("- Iteration 3: Converges to X(TI)=0.900000")
print("\nWe need the equilibrium matrix from iteration 2 that produces the update")
print("leading to the final X(TI)=0.900000 result.")

# Let's analyze what we know about the final state
print("\n\nANALYZING FINAL CONVERGENCE:")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
cpu_phases = result_cpu.Phase.values.flatten()

print("CPU final result:")
for i, (np_val, x_ti, phase) in enumerate(zip(cpu_np, cpu_x_ti, cpu_phases)):
    if np_val > 1e-12:
        print(f"  {phase}: NP={np_val:.10f}, X(TI)={x_ti:.10f}")

overall = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)
print(f"Overall X(TI): {overall:.10f}")

print("\n\nTO GET THE EXACT MATRIX VALUES:")
print("We need to add debug prints to pycalphad/core/minimizer.pyx")
print("Specifically in the solve_equilibrium_at_condition function")
print("to capture the equilibrium_matrix and equilibrium_rhs values")
print("from the iteration that produces the final convergence.")

print("\nThe key insight is that the CPU's final iteration must have:")
print("1. A 3x3 matrix (after consolidation to single phase)")
print("2. An RHS that when solved gives the update leading to X(TI)=0.9")
print("3. The solution vector: [delta_μ_NB, delta_μ_TI, delta_phase_amount]")

print("\nOnce we have these exact values, we can test if the GPU's SVD solver")
print("produces the same solution for the same matrix.")