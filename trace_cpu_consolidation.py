#!/usr/bin/env python
"""Trace CPU behavior around consolidation and after."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Patch to trace consolidation details
import pycalphad.core.minimizer

original_remove_consolidate = pycalphad.core.minimizer.remove_and_consolidate_phases
iteration_count = [0]

def trace_remove_consolidate(spec, state):
    """Trace what happens during consolidation."""
    print(f"\n[CPU CONSOLIDATION] Iteration {state.iteration}")
    print(f"  Before: {len(state.free_stable_compset_indices)} phases")
    
    # Print phase details before
    for idx in state.free_stable_compset_indices:
        compset = state.compsets[idx]
        print(f"  Phase {idx} ({compset.phase_record.phase_name}):")
        print(f"    Amount: {state.phase_amt[idx]:.6e}")
        print(f"    Y(TI): {state.dof[idx][4]:.10f}")  # Site fraction
        print(f"    X(TI): {state.phase_compositions[idx, 1]:.10f}")  # Mole fraction
    
    # Call original
    result = original_remove_consolidate(spec, state)
    
    # Print phase details after
    print(f"  After: {len(state.free_stable_compset_indices)} phases")
    for idx in state.free_stable_compset_indices:
        compset = state.compsets[idx]
        print(f"  Phase {idx} ({compset.phase_record.phase_name}):")
        print(f"    Amount: {state.phase_amt[idx]:.6e}")
        print(f"    Y(TI): {state.dof[idx][4]:.10f}")  # Site fraction
        print(f"    X(TI): {state.phase_compositions[idx, 1]:.10f}")  # Mole fraction
        
    # Store which iteration consolidation happened
    if result and len(state.free_stable_compset_indices) == 1:
        iteration_count[0] = state.iteration
        print(f"  CONSOLIDATION RESULTED IN SINGLE PHASE!")
    
    return result

# Also patch compute_equilibrium_rhs to see what RHS values are used
original_compute_rhs = pycalphad.core.minimizer.compute_equilibrium_rhs

def trace_compute_rhs(spec, state, equilibrium_rhs):
    """Trace RHS computation."""
    result = original_compute_rhs(spec, state, equilibrium_rhs)
    
    # Only trace for single phase after consolidation
    if len(state.free_stable_compset_indices) == 1 and state.iteration >= iteration_count[0] and iteration_count[0] > 0:
        print(f"\n[CPU RHS] Iteration {state.iteration} (single phase):")
        
        # Find constraint row
        constraint_row = len(state.free_stable_compset_indices) + spec.num_free_chemical_potentials
        if spec.prescribed_mole_fraction_rhs.shape[0] > 0:
            print(f"  Mole fraction constraint RHS: {equilibrium_rhs[constraint_row]:.10f}")
            
            # Also print the c_G values
            idx = state.free_stable_compset_indices[0]
            compset = state.compsets[idx]
            print(f"  Phase {idx} c_G values: {compset.c_G}")
            print(f"  Phase amount: {state.phase_amt[idx]}")
            print(f"  Current X(TI): {state.phase_compositions[idx, 1]:.10f}")
            print(f"  Target X(TI): 0.9")
            print(f"  Error: {state.phase_compositions[idx, 1] - 0.9:.10f}")
    
    return result

# Apply patches
pycalphad.core.minimizer.remove_and_consolidate_phases = trace_remove_consolidate
pycalphad.core.minimizer.compute_equilibrium_rhs = trace_compute_rhs

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("TRACING CPU CONSOLIDATION AND SINGLE-PHASE BEHAVIOR")
print("=" * 60)

result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

# Restore
pycalphad.core.minimizer.remove_and_consolidate_phases = original_remove_consolidate
pycalphad.core.minimizer.compute_equilibrium_rhs = original_compute_rhs

cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)

print(f"\nFINAL: X(TI) = {overall_x_ti:.10f}")