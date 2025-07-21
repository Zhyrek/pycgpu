#!/usr/bin/env python
"""Test to understand GPU GM calculation issue."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test condition: X(TI)=0.1, T=600K
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("Testing X(TI)=0.1, T=600K")
print("="*80)

# Run CPU calculation
print("\nCPU Calculation:")
cpu_result = equilibrium(dbf, comps, phases, conditions, verbose=False)
cpu_gm = float(cpu_result.GM.values)
cpu_phases = []
for i, (phase, amt) in enumerate(zip(cpu_result.Phase.values.flatten(), 
                                     cpu_result.NP.values.flatten())):
    if phase and amt > 1e-6:
        cpu_phases.append((phase, amt))

print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"CPU Phases: {cpu_phases}")

# Now let's manually calculate what GM should be
# For a single BCC_A2 phase with X(TI)=0.1 at 600K
# We need the energy of that phase
print("\nManual verification:")
print("For single-phase BCC_A2 with site fractions Y(NB)=0.9, Y(TI)=0.1")

# The GPU shows it has the correct final state internally:
# Phase 0 with amount=1.0 and Y=[0.9, 0.1]
# But reports phase amounts of [0.5, 0.5]

# If GPU uses phase amounts [0.5, 0.5] with same energy:
energy_per_mole = cpu_gm  # Since CPU has 1 phase with amount 1.0
gm_with_wrong_amounts = 0.5 * energy_per_mole + 0.5 * energy_per_mole
print(f"GM with wrong phase amounts [0.5, 0.5]: {gm_with_wrong_amounts:.6f} J/mol")
print(f"This should equal CPU GM: {cpu_gm:.6f} J/mol")

# But GPU reports -24611.324065, which is different
gpu_gm = -24611.324065
print(f"\nGPU reported GM: {gpu_gm:.6f} J/mol")
print(f"Difference: {gpu_gm - cpu_gm:.6f} J/mol")

# This suggests GPU is using different phase compositions or energies
# Let's check what the energy would be for the initial two-phase state
print("\nHypothesis: GPU is using initial two-phase energies")
print("Initial phases from lower_convex_hull:")
print("  Phase 0: Y=[0.907760, 0.092240], amount=0.179268") 
print("  Phase 1: Y=[0.898305, 0.101695], amount=0.820732")

# The fact that GPU GM differs by 8.67 J/mol suggests it's calculating
# the energy of a different state than the converged single-phase state