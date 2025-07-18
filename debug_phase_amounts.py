#!/usr/bin/env python
"""Debug phase amounts at consolidation point."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Patch the CPU solver to print phase amounts
import pycalphad.core.eqsolver
original_solve = pycalphad.core.eqsolver._solve_eq_at_conditions

def patched_solve(dbf, comps, phases, conditions, *args, **kwargs):
    print(f"\n[CPU SOLVER] Called with conditions: {conditions}")
    result = original_solve(dbf, comps, phases, conditions, *args, **kwargs)
    
    # Print phase amounts from result
    if hasattr(result, 'eq') and result.eq is not None:
        state = result.eq.get_system_state()
        print(f"[CPU SOLVER] After solving - num_phases: {len(state.free_stable_compset_indices)}")
        for i, idx in enumerate(state.free_stable_compset_indices):
            phase_name = state.compsets[idx].phase_record.phase_name
            amount = state.phase_amt[idx]
            print(f"[CPU SOLVER]   Phase {idx} ({phase_name}): amount = {amount:.15e}")
    
    return result

pycalphad.core.eqsolver._solve_eq_at_conditions = patched_solve

# Now run the test
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("Running CPU calculation with debug...")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
print(f"\nFinal CPU GM: {result_cpu.GM.values.flatten()[0]:.6f} J/mol")