#!/usr/bin/env python3
"""Trace why GPU removes phase while CPU keeps it"""
import os
os.environ['PYCALPHAD_DEBUG'] = '1'

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("=== Testing Phase Removal Logic ===")
print(f"Conditions: {conditions}")
print("\nThe CPU keeps 2 BCC_A2 phases through the calculation")
print("The GPU removes one BCC_A2 phase early")
print("\nLet's trace why...")

# First check what the starting point gives us
from pycalphad import calculate
from pycalphad.core.starting_point import starting_point

# Get starting point
grid = calculate(db, comps, phases, output='GM', T=1000, P=101325, N=1, pdens=50)
# Convert grid to properties format
from pycalphad.core.workspace import Workspace
wks = Workspace(db, ['NB', 'TI'], phases, conditions)
properties = wks.eq

print("\n=== Starting Point Analysis ===")
np_values = properties.NP.values.flatten()
phase_values = properties.Phase.values.flatten()
active_mask = np_values > 1e-10

print(f"Number of active phases: {np.sum(active_mask)}")
for i, (phase, amount) in enumerate(zip(phase_values[active_mask], np_values[active_mask])):
    print(f"Phase {i}: {phase}, amount={amount:.6f}")

# Check if they're the same phase
unique_phases = []
phase_indices = []
for i, phase in enumerate(phase_values[active_mask]):
    if phase not in unique_phases:
        unique_phases.append(phase)
    phase_indices.append(unique_phases.index(phase))

print(f"\nPhase types: {unique_phases}")
print(f"Phase type indices: {phase_indices}")

if len(set(phase_indices)) < len(phase_indices):
    print("\n⚠️  Multiple phases of the same type detected (immiscibility gap)")
    print("This is where CPU and GPU behavior diverges!")

# Check the site fractions to see how similar the phases are
y_values = properties.Y.values
print("\n=== Site Fractions ===")
for i in range(len(phase_values)):
    if np_values[i] > 1e-10:
        phase_name = phase_values[i]
        print(f"\nPhase {i} ({phase_name}):")
        print(f"  Y(NB) = {y_values[i,0,0]:.6f}")
        print(f"  Y(TI) = {y_values[i,0,1]:.6f}")

# Check how close the duplicate phases are
if len(phase_indices) > 1 and phase_indices[0] == phase_indices[1]:
    y_diff = abs(y_values[0,0,0] - y_values[1,0,0])
    print(f"\nSite fraction difference between BCC_A2 phases: {y_diff:.6f}")
    print(f"This is {'very small' if y_diff < 0.01 else 'significant'}")

print("\n=== Phase Removal Logic ===")
print("CPU: Keeps both phases even if they're similar")
print("GPU: Likely removes duplicate phases with similar compositions")
print("\nThe key difference is in how duplicates are handled!")