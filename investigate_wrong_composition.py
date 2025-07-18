#!/usr/bin/env python
"""Investigate why equilibrium settles at X(TI)=0.903 instead of 0.900."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("INVESTIGATING WRONG COMPOSITION")
print("=" * 60)

# Test different conditions
test_conditions = [
    {v.X('TI'): 0.9, v.T: 600, v.P: 101325},
    {v.X('TI'): 0.9, v.T: 800, v.P: 101325},
    {v.X('TI'): 0.9, v.T: 1000, v.P: 101325},
]

for cond in test_conditions:
    print(f"\nCondition: X(TI)={cond[v.X('TI')]}, T={cond[v.T]}K")
    
    # Run equilibrium
    result = equilibrium(dbf, comps, phases, cond, verbose=False)
    
    # Extract results
    x_ti = result.X.sel(component='TI').values.flatten()
    np_vals = result.NP.values.flatten()
    phase_names = result.Phase.values.flatten()
    
    # Calculate overall composition
    overall_x_ti = 0
    active_phases = []
    for i, (np_val, x, phase) in enumerate(zip(np_vals, x_ti, phase_names)):
        if np_val > 1e-12:
            overall_x_ti += np_val * x
            active_phases.append((phase, np_val, x))
    
    print(f"  Target X(TI): {cond[v.X('TI')]:.6f}")
    print(f"  Actual X(TI): {overall_x_ti:.6f}")
    print(f"  Error: {overall_x_ti - cond[v.X('TI')]:.6f} ({100*(overall_x_ti - cond[v.X('TI')])/cond[v.X('TI')]:.2f}%)")
    print(f"  Active phases: {len(active_phases)}")
    for phase, np_val, x in active_phases:
        print(f"    {phase}: NP={np_val:.6f}, X(TI)={x:.6f}")

# Check if this is a fundamental issue with the phase diagram
print("\n\nPOSSIBLE EXPLANATIONS:")
print("1. Single-phase region at X(TI)=0.9, T=600K might not exist")
print("2. The solver might be finding a local minimum")
print("3. The mass balance constraint might be incorrectly formulated")
print("4. The phase consolidation might be merging phases prematurely")

# Let's check what happens if we force a single BCC phase
print("\n\nFORCING SINGLE BCC PHASE:")
result_single = equilibrium(dbf, comps, ['BCC_A2'], {v.X('TI'): 0.9, v.T: 600, v.P: 101325}, verbose=False)
x_ti_single = result_single.X.sel(component='TI').values.flatten()[0]
print(f"  Single BCC phase X(TI): {x_ti_single:.6f}")
print(f"  This should be exactly 0.900000 if the constraint is working")

# Check with more phases allowed
print("\n\nWITH ALL PHASES:")
all_phases = list(dbf.phases.keys())
all_phases.remove('VA')  # Remove VA pseudo-phase
result_all = equilibrium(dbf, comps, all_phases, {v.X('TI'): 0.9, v.T: 600, v.P: 101325}, verbose=False)

x_ti_all = result_all.X.sel(component='TI').values.flatten()
np_all = result_all.NP.values.flatten()
phase_all = result_all.Phase.values.flatten()

overall_all = 0
print("  Active phases:")
for x, np_val, phase in zip(x_ti_all, np_all, phase_all):
    if np_val > 1e-12:
        overall_all += np_val * x
        print(f"    {phase}: NP={np_val:.6f}, X(TI)={x:.6f}")
        
print(f"  Overall X(TI): {overall_all:.6f}")