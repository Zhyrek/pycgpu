#!/usr/bin/env python
"""Test exact conditions without pdens."""

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

# Check phases by index
np_vals = cpu_result.NP.values.flatten()
print("\nPhase amounts by index:")
for i, phase in enumerate(phases):
    if i < len(np_vals):
        print(f"  Index {i}: {phase:12s} = {np_vals[i]:.6f}")

# Find which phases are active
print("\nActive phases (amount > 0.001):")
for i, phase in enumerate(phases):
    if i < len(np_vals) and np_vals[i] > 0.001:
        print(f"  {phase}: {np_vals[i]:.6f} ({np_vals[i]*100:.1f}%)")

print(f"\nGM = {cpu_result.GM.values.item():.2f} J/mol")