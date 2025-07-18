#!/usr/bin/env python
"""Trace CPU behavior around consolidation - corrected version."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Patch to trace what happens
import pycalphad.core.minimizer

# Global flag to track when we've seen consolidation
consolidation_happened = [False]
single_phase_iteration = [-1]

original_recompute = pycalphad.core.minimizer.recompute

def trace_recompute(spec, state):
    """Trace recompute to see phase compositions."""
    result = original_recompute(spec, state)
    
    # Check if we just went to single phase
    if len(state.free_stable_compset_indices) == 1 and not consolidation_happened[0]:
        consolidation_happened[0] = True
        single_phase_iteration[0] = state.iteration
        print(f"\n[CPU SINGLE PHASE] Detected at iteration {state.iteration}")
        idx = state.free_stable_compset_indices[0]
        print(f"  Phase {idx}:")
        print(f"    Amount: {state.phase_amt[idx]:.10f}")
        print(f"    X(TI): {state.phase_compositions[idx, 1]:.10f}")
        print(f"    Mass residual: {state.mass_residual:.6e}")
        
        # Check if we're already at the target
        x_ti = state.phase_compositions[idx, 1]
        print(f"    Distance from target: {abs(x_ti - 0.9):.10f}")
        
    return result

# Patch solve_equilibrium to see solutions
original_solve = pycalphad.core.minimizer.solve_equilibrium

def trace_solve(spec, state, equilibrium_matrix, equilibrium_rhs):
    """Trace the linear solve for single phase."""
    
    # Only trace single phase
    if len(state.free_stable_compset_indices) == 1 and consolidation_happened[0]:
        print(f"\n[CPU SOLVE] Iteration {state.iteration} (single phase)")
        
        # Get dimensions
        n_phases = len(state.free_stable_compset_indices)
        n_chem_pot = spec.num_free_chemical_potentials  
        n_constraints = spec.prescribed_mole_fraction_rhs.shape[0]
        
        # The constraint RHS is at position: n_phases + n_chem_pot
        if n_constraints > 0:
            constraint_row = n_phases + n_chem_pot
            print(f"  Constraint RHS: {equilibrium_rhs[constraint_row]:.10f}")
            
    result = original_solve(spec, state, equilibrium_matrix, equilibrium_rhs)
    
    if len(state.free_stable_compset_indices) == 1 and consolidation_happened[0] and result is not None:
        print(f"  Solution: delta_phase_amt = {result[0]:.6e}")
        # The site fraction changes would be in result[n_phases + n_chem_pot + n_constraints:]
        # But for single phase with constraint, there might not be any DOF left
        
    return result

# Patch construct_equilibrium_system to see matrix details
original_construct = pycalphad.core.minimizer.construct_equilibrium_system

def trace_construct(spec, state, equilibrium_matrix, equilibrium_rhs):
    """Trace matrix construction for single phase."""
    
    result = original_construct(spec, state, equilibrium_matrix, equilibrium_rhs)
    
    # Only trace single phase
    if len(state.free_stable_compset_indices) == 1 and consolidation_happened[0]:
        print(f"\n[CPU MATRIX] Iteration {state.iteration} (single phase)")
        
        idx = state.free_stable_compset_indices[0]
        compset = state.compsets[idx]
        
        # Print c_component values
        print(f"  c_component values:")
        for i in range(spec.num_components):
            for j in range(compset.phase_record.phase_dof):
                print(f"    c_component[{i},{j}] = {compset.c_component[i,j]:.6e}")
                
        # Print current composition
        print(f"  Current X(TI): {state.phase_compositions[idx, 1]:.10f}")
        print(f"  Phase amount: {state.phase_amt[idx]:.10f}")
        
    return result

# Apply patches
pycalphad.core.minimizer.recompute = trace_recompute
pycalphad.core.minimizer.solve_equilibrium = trace_solve
pycalphad.core.minimizer.construct_equilibrium_system = trace_construct

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("TRACING CPU SINGLE-PHASE BEHAVIOR")
print("=" * 60)

result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

# Restore
pycalphad.core.minimizer.recompute = original_recompute
pycalphad.core.minimizer.solve_equilibrium = original_solve
pycalphad.core.minimizer.construct_equilibrium_system = original_construct

cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)

print(f"\nFINAL: X(TI) = {overall_x_ti:.10f}")
print(f"Number of phases: {sum(1 for np in cpu_np if np > 1e-12)}")