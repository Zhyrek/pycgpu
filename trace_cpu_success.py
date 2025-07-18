#!/usr/bin/env python
"""Trace exactly why CPU succeeds where GPU fails."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Patch CPU code to trace the exact solution path
import pycalphad.core.minimizer

original_advance_state = pycalphad.core.minimizer.advance_state
original_solve_equilibrium = pycalphad.core.minimizer.solve_equilibrium

def trace_advance_state(state, delta_statevars, max_step_size):
    """Trace what happens during state advancement."""
    print(f"\n[CPU TRACE] advance_state called:")
    print(f"  Iteration: {state.iteration}")
    print(f"  Number of phases: {len(state.free_stable_compset_indices)}")
    
    # Print delta values
    for i, idx in enumerate(state.free_stable_compset_indices):
        if i < len(delta_statevars):
            print(f"  Phase {idx} delta: {delta_statevars[i]}")
            print(f"  Phase {idx} current amount: {state.phase_amt[idx]}")
    
    # Call original
    result = original_advance_state(state, delta_statevars, max_step_size)
    
    # Print results
    print(f"  After advance:")
    for idx in state.free_stable_compset_indices:
        print(f"    Phase {idx} new amount: {state.phase_amt[idx]}")
    
    return result

def trace_solve_equilibrium(spec, state, equilibrium_matrix, equilibrium_rhs):
    """Trace the linear solve."""
    print(f"\n[CPU TRACE] solve_equilibrium called at iteration {state.iteration}")
    
    # Get matrix dimensions
    n_dof = len(state.free_stable_compset_indices) + spec.num_free_chemical_potentials + spec.prescribed_mole_fraction_rhs.shape[0] + 1
    
    # Print matrix info
    print(f"  Matrix dimensions: {n_dof}x{n_dof}")
    print(f"  Number of phases: {len(state.free_stable_compset_indices)}")
    
    # Print RHS for mole fraction constraint
    if spec.prescribed_mole_fraction_rhs.shape[0] > 0:
        constraint_row = len(state.free_stable_compset_indices) + spec.num_free_chemical_potentials
        print(f"  Mole fraction constraint RHS: {equilibrium_rhs[constraint_row]}")
    
    # Call original
    result = original_solve_equilibrium(spec, state, equilibrium_matrix, equilibrium_rhs)
    
    # Print solution
    if result is not None and len(result) > 0:
        print(f"  Solution delta_phase_amt: {result[:len(state.free_stable_compset_indices)]}")
        if len(state.free_stable_compset_indices) == 1:
            print(f"  Single phase - delta = {result[0]}")
    
    return result

# Apply patches
pycalphad.core.minimizer.advance_state = trace_advance_state
pycalphad.core.minimizer.solve_equilibrium = trace_solve_equilibrium

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("TRACING CPU SUCCESS PATH")
print("=" * 60)

result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

# Restore original functions
pycalphad.core.minimizer.advance_state = original_advance_state
pycalphad.core.minimizer.solve_equilibrium = original_solve_equilibrium

cpu_gm = result_cpu.GM.values.flatten()[0]
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)

print(f"\nFINAL RESULT:")
print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"CPU X(TI): {overall_x_ti:.10f}")
print(f"Phases present: {sum(1 for np in cpu_np if np > 1e-12)}")