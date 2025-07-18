#!/usr/bin/env python
"""Trace how phases are consolidated and averaged."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("TRACING CONSOLIDATION AVERAGING")
print("=" * 60)

# From the CPU trace output, before consolidation:
print("\nBefore consolidation (iteration 1):")
print("Phase 0: NP=1.0, Y(TI)=0.90314714, X(TI)=0.9031471")
print("Phase 1: NP=8.3e-17, Y(TI)=0.90313275, X(TI)=0.9031328")

# The phases have almost identical compositions
# When consolidated, we'd expect: X(TI) = 1.0 * 0.9031471 + 0.0 * 0.9031328 = 0.9031471

print("\nAfter consolidation:")
print("Single phase: X(TI)=0.9031471")

print("\nBUT WAIT! This is the composition of the phase, not the overall system!")
print("The issue is that X(TI)=0.9031471 is the mole fraction within the BCC phase")
print("But we constrained the OVERALL system to have X(TI)=0.9")

print("\nThe fundamental issue:")
print("- Two BCC phases form with X(TI) ≈ 0.903")
print("- They get consolidated into one phase with X(TI) = 0.903")
print("- But this violates the constraint that the system should have X(TI) = 0.9")

print("\nWhy does this happen?")
print("At T=600K, X(TI)=0.9, the equilibrium state might actually be:")
print("- A two-phase region (miscibility gap)")
print("- Both phases have X(TI) ≈ 0.903")
print("- The constraint X(TI)=0.9 cannot be satisfied by any combination of these phases")

# Let's check the phase diagram more systematically
print("\n\nCHECKING PHASE DIAGRAM AT T=600K:")
x_ti_values = np.linspace(0.85, 0.95, 11)
for x_ti in x_ti_values:
    result = equilibrium(dbf, comps, phases, {v.X('TI'): x_ti, v.T: 600, v.P: 101325}, verbose=False)
    
    # Get phase info
    np_vals = result.NP.values.flatten()
    x_vals = result.X.sel(component='TI').values.flatten()
    phases_active = result.Phase.values.flatten()
    
    # Count active phases and their compositions
    active = []
    for i, np_val in enumerate(np_vals):
        if np_val > 1e-12:
            active.append((phases_active[i], np_val, x_vals[i]))
    
    overall = sum(np_val * x for _, np_val, x in active)
    
    print(f"X(TI)={x_ti:.3f}: ", end="")
    if len(active) == 1:
        print(f"Single phase - {active[0][0]} with X(TI)={active[0][2]:.6f}")
    else:
        print(f"{len(active)} phases - ", end="")
        for phase, np_val, x in active:
            print(f"{phase}({np_val:.3f},X={x:.4f}) ", end="")
        print(f"→ overall={overall:.6f}")