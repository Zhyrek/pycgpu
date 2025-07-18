#!/usr/bin/env python
"""Verify exactly what CPU consolidation does to site fractions."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("VERIFYING CPU CONSOLIDATION MECHANISM")
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

# Patch the CPU consolidation to track exactly what happens to Y values
import pycalphad.core.minimizer
original_consolidate = pycalphad.core.minimizer.remove_and_consolidate_phases

def patched_consolidate(spec, state):
    print(f"\n[CONSOLIDATION TRACKER] Before consolidation (iteration {state.iteration}):")
    print(f"[CONSOLIDATION TRACKER] Number of phases: {len(state.free_stable_compset_indices)}")
    
    # Record phase data before consolidation
    before_data = []
    for i, idx in enumerate(state.free_stable_compset_indices):
        if hasattr(state, 'dof') and len(state.dof[idx]) > 4:
            y_ti = state.dof[idx][4]
            phase_amt = state.phase_amt[idx]
            phase_comps = state.phase_compositions[idx]
            before_data.append({
                'idx': idx, 
                'y_ti': y_ti, 
                'phase_amt': phase_amt,
                'x_ti': phase_comps[1] if len(phase_comps) > 1 else 0.0
            })
            print(f"[CONSOLIDATION TRACKER]   Phase {idx}: Y(TI)={y_ti:.10f}, amount={phase_amt:.6f}, X(TI)={phase_comps[1]:.10f}")
    
    # Call original consolidation
    result = original_consolidate(spec, state)
    
    print(f"\n[CONSOLIDATION TRACKER] After consolidation:")
    print(f"[CONSOLIDATION TRACKER] Number of phases: {len(state.free_stable_compset_indices)}")
    
    # Record phase data after consolidation
    after_data = []
    for i, idx in enumerate(state.free_stable_compset_indices):
        if hasattr(state, 'dof') and len(state.dof[idx]) > 4:
            y_ti = state.dof[idx][4]
            phase_amt = state.phase_amt[idx]
            phase_comps = state.phase_compositions[idx]
            after_data.append({
                'idx': idx, 
                'y_ti': y_ti, 
                'phase_amt': phase_amt,
                'x_ti': phase_comps[1] if len(phase_comps) > 1 else 0.0
            })
            print(f"[CONSOLIDATION TRACKER]   Phase {idx}: Y(TI)={y_ti:.10f}, amount={phase_amt:.6f}, X(TI)={phase_comps[1]:.10f}")
    
    # Analysis
    if len(before_data) == 2 and len(after_data) == 1:
        print(f"\n[CONSOLIDATION TRACKER] CONSOLIDATION ANALYSIS:")
        print(f"  Before: Phase {before_data[0]['idx']} had Y(TI)={before_data[0]['y_ti']:.10f}")
        print(f"  Before: Phase {before_data[1]['idx']} had Y(TI)={before_data[1]['y_ti']:.10f}")
        print(f"  After:  Phase {after_data[0]['idx']} has Y(TI)={after_data[0]['y_ti']:.10f}")
        
        # Check which phase survived
        surviving_idx = after_data[0]['idx']
        if surviving_idx == before_data[0]['idx']:
            print(f"  *** Phase {before_data[0]['idx']} survived with UNCHANGED Y(TI) ***")
            print(f"  *** Phase {before_data[1]['idx']} was removed ***")
        elif surviving_idx == before_data[1]['idx']:
            print(f"  *** Phase {before_data[1]['idx']} survived with UNCHANGED Y(TI) ***")
            print(f"  *** Phase {before_data[0]['idx']} was removed ***")
        
        # Check amount addition
        total_before = before_data[0]['phase_amt'] + before_data[1]['phase_amt']
        total_after = after_data[0]['phase_amt']
        print(f"  Amount before: {before_data[0]['phase_amt']:.6f} + {before_data[1]['phase_amt']:.6f} = {total_before:.6f}")
        print(f"  Amount after:  {total_after:.6f}")
        print(f"  Amount conserved: {abs(total_before - total_after) < 1e-10}")
        
        # Most importantly - what X(TI) does this result in?
        final_x_ti = after_data[0]['x_ti']
        print(f"\n  CRITICAL: Final overall X(TI) = {final_x_ti:.10f}")
        print(f"  CRITICAL: Constraint error = {abs(final_x_ti - 0.9):.10f}")
        if abs(final_x_ti - 0.9) < 1e-8:
            print(f"  *** CONSTRAINT SATISFIED BY CHANCE! ***")
        else:
            print(f"  *** CONSTRAINT VIOLATED! ***")
    
    return result

# Patch and run CPU
pycalphad.core.minimizer.remove_and_consolidate_phases = patched_consolidate
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

# Restore original
pycalphad.core.minimizer.remove_and_consolidate_phases = original_consolidate

print(f"\n{'='*60}")
print("FINAL RESULT")
print("="*60)
print(f"CPU Final GM: {result_cpu.GM.values.flatten()[0]:.6f} J/mol")

cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)
print(f"CPU overall X(TI): {overall_x_ti:.10f}")
print(f"Constraint error: {abs(overall_x_ti - 0.9):.10f}")

print(f"\n{'='*60}")
print("KEY INSIGHT")
print("="*60)
print("CPU consolidation mechanism:")
print("1. Adds phase amounts: phase_amt[idx] = phase_amt[idx] + phase_amt[idx2]")
print("2. Sets removed phase amount to zero: phase_amt[idx2] = 0")
print("3. Does NOT change the site fractions (Y values) of the surviving phase")
print("4. Which phase survives depends on the loop order (idx vs idx2)")
print("5. The final composition depends entirely on which phase happens to survive")
print("\nIf CPU reaches X(TI)=0.9 exactly, it's because the surviving phase")
print("happened to have site fractions that give X(TI)=0.9 when it becomes")
print("the only phase (with amount=1.0).")