#!/usr/bin/env python
"""Trace exactly how CPU reaches X(TI)=0.9 during consolidation."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TRACING CPU CONSOLIDATION MECHANISM")
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

# Patch the CPU consolidation routine to see exactly what it does
import pycalphad.core.minimizer
original_consolidate = pycalphad.core.minimizer.consolidate_statevar_dependent_phases

def patched_consolidate(state, tolerance=1e-4):
    print(f"\n[CPU CONSOLIDATE] Called at iteration {state.iteration}")
    print(f"[CPU CONSOLIDATE] Tolerance: {tolerance}")
    print(f"[CPU CONSOLIDATE] Number of phases before: {len(state.free_stable_compset_indices)}")
    
    if len(state.free_stable_compset_indices) >= 2:
        # Show phases before consolidation
        for i, idx in enumerate(state.free_stable_compset_indices):
            if hasattr(state, 'dof') and len(state.dof[idx]) > 4:
                y_ti = state.dof[idx][4]
                phase_amt = state.phase_amt[idx] 
                print(f"[CPU CONSOLIDATE] Phase {i} (idx {idx}): amount={phase_amt:.6f}, Y(TI)={y_ti:.10f}")
                
                # Calculate composition of this phase
                compset = state.compsets[idx]
                if hasattr(compset, 'X') and len(compset.X) > 1:
                    x_ti_phase = compset.X[1]  # TI is component 1
                    print(f"[CPU CONSOLIDATE] Phase {i} X(TI): {x_ti_phase:.10f}")
        
        # Calculate overall composition before consolidation
        total_ti = 0.0
        total_amt = 0.0
        for idx in state.free_stable_compset_indices:
            compset = state.compsets[idx]
            phase_amt = state.phase_amt[idx]
            if hasattr(compset, 'X') and len(compset.X) > 1:
                x_ti_phase = compset.X[1]
                ti_contribution = phase_amt * x_ti_phase
                total_ti += ti_contribution
                total_amt += phase_amt
                print(f"[CPU CONSOLIDATE] Phase {idx}: amt={phase_amt:.6f} * X(TI)={x_ti_phase:.6f} = {ti_contribution:.6f}")
        
        overall_x_ti_before = total_ti / total_amt if total_amt > 0 else 0.0
        print(f"[CPU CONSOLIDATE] Overall X(TI) BEFORE consolidation: {overall_x_ti_before:.10f}")
    
    # Call original consolidation
    result = original_consolidate(state, tolerance)
    
    print(f"[CPU CONSOLIDATE] Number of phases after: {len(state.free_stable_compset_indices)}")
    
    if len(state.free_stable_compset_indices) == 1:
        # Show the consolidated phase
        idx = state.free_stable_compset_indices[0]
        if hasattr(state, 'dof') and len(state.dof[idx]) > 4:
            y_ti = state.dof[idx][4]
            phase_amt = state.phase_amt[idx]
            print(f"[CPU CONSOLIDATE] Consolidated phase: amount={phase_amt:.6f}, Y(TI)={y_ti:.10f}")
            
            # Calculate composition of consolidated phase
            compset = state.compsets[idx]
            if hasattr(compset, 'X') and len(compset.X) > 1:
                x_ti_phase = compset.X[1]  # TI is component 1
                print(f"[CPU CONSOLIDATE] Consolidated phase X(TI): {x_ti_phase:.10f}")
                
        # Calculate overall composition after consolidation
        total_ti = 0.0
        total_amt = 0.0
        for idx in state.free_stable_compset_indices:
            compset = state.compsets[idx]
            phase_amt = state.phase_amt[idx]
            if hasattr(compset, 'X') and len(compset.X) > 1:
                x_ti_phase = compset.X[1]
                ti_contribution = phase_amt * x_ti_phase
                total_ti += ti_contribution
                total_amt += phase_amt
        
        overall_x_ti_after = total_ti / total_amt if total_amt > 0 else 0.0
        print(f"[CPU CONSOLIDATE] Overall X(TI) AFTER consolidation: {overall_x_ti_after:.10f}")
        print(f"[CPU CONSOLIDATE] Constraint error: {abs(overall_x_ti_after - 0.9):.10f}")
        
        if abs(overall_x_ti_after - 0.9) < 1e-6:
            print("[CPU CONSOLIDATE] *** COMPOSITION CONSTRAINT SATISFIED! ***")
        else:
            print("[CPU CONSOLIDATE] *** COMPOSITION CONSTRAINT VIOLATED! ***")
    
    return result

# Also patch the composition calculation to see how CPU sets Y(TI)
original_recompute = pycalphad.core.minimizer.recompute_state_variables

def patched_recompute(state):
    print(f"\n[CPU RECOMPUTE] Called at iteration {state.iteration}")
    
    # Call original recompute
    result = original_recompute(state)
    
    # Show what recompute did to site fractions
    if len(state.free_stable_compset_indices) == 1:
        idx = state.free_stable_compset_indices[0]
        if hasattr(state, 'dof') and len(state.dof[idx]) > 4:
            y_ti = state.dof[idx][4]
            print(f"[CPU RECOMPUTE] Single phase Y(TI) after recompute: {y_ti:.10f}")
            
            # Check if recompute enforced the constraint
            compset = state.compsets[idx]
            if hasattr(compset, 'X') and len(compset.X) > 1:
                x_ti_phase = compset.X[1]
                print(f"[CPU RECOMPUTE] Single phase X(TI) after recompute: {x_ti_phase:.10f}")
    
    return result

# Patch both functions
pycalphad.core.minimizer.consolidate_statevar_dependent_phases = patched_consolidate
pycalphad.core.minimizer.recompute_state_variables = patched_recompute

result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

# Restore originals
pycalphad.core.minimizer.consolidate_statevar_dependent_phases = original_consolidate
pycalphad.core.minimizer.recompute_state_variables = original_recompute

print(f"\n{'='*60}")
print("FINAL ANALYSIS")
print("="*60)
print(f"CPU Final GM: {result_cpu.GM.values.flatten()[0]:.6f} J/mol")

cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)
print(f"CPU overall X(TI): {overall_x_ti:.10f}")
print(f"Constraint error: {abs(overall_x_ti - 0.9):.10f}")

print(f"\n{'='*60}")
print("KEY QUESTION")
print("="*60)
print("HOW does CPU's consolidation process result in exactly X(TI)=0.9?")
print("Does it:")
print("1. Weight the phases by amount and get lucky?")
print("2. Explicitly enforce the constraint during consolidation?")
print("3. Use some other mechanism?")
print("Look at the trace above to find the answer!")