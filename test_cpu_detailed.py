#!/usr/bin/env python
"""Test CPU in detail to understand constraint vs equilibrium behavior."""

import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("Detailed CPU analysis for different X(TI) constraints...")
print("=" * 60)

for x_ti_target in [0.005, 0.01, 0.02]:
    print(f"\nAnalyzing X(TI) = {x_ti_target:.3f} constraint:")
    print("-" * 40)
    
    conditions = {v.X('TI'): x_ti_target, v.T: 1000, v.P: 101325}
    
    result = equilibrium(dbf, comps, phases, conditions, verbose=False)
    
    # Get detailed results
    x_ti_final = result.X.sel(component='TI').values.flatten()[0]
    x_nb_final = result.X.sel(component='NB').values.flatten()[0]
    
    # Get phase information
    phases_present = []
    phase_amounts = []
    phase_compositions = []
    
    for i, phase_name in enumerate(result.Phase.values.flatten()):
        if phase_name != '' and phase_name != '_FAKE_':
            phase_amt = result.NP.values.flatten()[i]
            if phase_amt > 1e-10:  # Only include phases with significant amounts
                phases_present.append(phase_name)
                phase_amounts.append(phase_amt)
                
                # Get composition of this phase
                y_values = result.Y.values[0, i, :]  # Site fractions
                phase_compositions.append(y_values)
    
    print(f"  Constraint: X(TI) = {x_ti_target:.6f}")
    print(f"  Result:     X(TI) = {x_ti_final:.6f}")
    print(f"  Result:     X(NB) = {x_nb_final:.6f}")
    print(f"  Constraint satisfied: {abs(x_ti_final - x_ti_target) < 1e-10}")
    
    print(f"  Phases present: {len(phases_present)}")
    for i, (phase, amt, comp) in enumerate(zip(phases_present, phase_amounts, phase_compositions)):
        print(f"    Phase {i}: {phase}, amount = {amt:.6f}")
        print(f"             Site fractions: {comp[:2]}")  # Only show first 2 site fractions
    
    print(f"  Total system check: X(NB) + X(TI) = {x_nb_final + x_ti_final:.6f}")
    
    # Check if this is actually satisfying the constraint properly
    mass_balance_error = abs(x_ti_final - x_ti_target)
    if mass_balance_error > 1e-10:
        print(f"  WARNING: Mass balance error = {mass_balance_error:.2e}")
    else:
        print(f"  Mass balance satisfied (error = {mass_balance_error:.2e})")

print("\n" + "=" * 60)
print("Analysis complete.")
print("\nConclusion:")
print("If CPU X(TI) exactly equals the constraint value, then CPU is")
print("correctly satisfying the constraint and GPU results are wrong.")
print("If CPU X(TI) differs from constraint, then both may be finding") 
print("different equilibria and we need to understand which is correct.")