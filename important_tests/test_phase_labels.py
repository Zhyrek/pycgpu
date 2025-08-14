#!/usr/bin/env python
"""Test phase labels from results.Phase."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

dbf = Database('../Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2']

conditions = {
    v.X('AL'): 0.20,
    v.X('CU'): 0.50,
    v.T: 900,
    v.P: 101325
}

print("Testing X(AL)=0.20, X(CU)=0.50, T=900K")
print("=" * 60)

# CPU calculation WITHOUT pdens
print("\nCPU calculation (no pdens):")
cpu_result = equilibrium(dbf, comps, phases, conditions,
                        gpu=False, verbose=False)

# Get phase labels from results.Phase
print("\nPhase labels from results.Phase:")
phase_labels = cpu_result.Phase.values.flatten()
np_vals = cpu_result.NP.values.flatten()

for i, (phase_label, amount) in enumerate(zip(phase_labels, np_vals)):
    if not np.isnan(amount):
        print(f"  Index {i}: {phase_label:12s} = {amount:.6f}")

print("\nActive phases (from results.Phase, amount > 0.001):")
for i, (phase_label, amount) in enumerate(zip(phase_labels, np_vals)):
    if not np.isnan(amount) and amount > 0.001:
        print(f"  {phase_label}: {amount:.6f} ({amount*100:.1f}%)")

print(f"\nGM = {cpu_result.GM.values.item():.2f} J/mol")

# Also check with pdens=50
print("\n" + "=" * 60)
print("CPU calculation WITH pdens=50:")
cpu_result2 = equilibrium(dbf, comps, phases, conditions,
                         calc_opts={'pdens': 50},
                         gpu=False, verbose=False)

phase_labels2 = cpu_result2.Phase.values.flatten()
np_vals2 = cpu_result2.NP.values.flatten()

print("\nActive phases (from results.Phase, amount > 0.001):")
for i, (phase_label, amount) in enumerate(zip(phase_labels2, np_vals2)):
    if not np.isnan(amount) and amount > 0.001:
        print(f"  {phase_label}: {amount:.6f} ({amount*100:.1f}%)")

print(f"\nGM = {cpu_result2.GM.values.item():.2f} J/mol")