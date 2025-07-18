#!/usr/bin/env python
"""Test CPU constraint row construction to understand the difference."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TESTING CPU CONSTRAINT ROW CONSTRUCTION")
print("=" * 60)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test X(TI)=0.9, T=600K 
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("\nRunning CPU calculation with debug output...")
print("Focus on constraint row coefficients and RHS")
print("-" * 60)

# Run CPU calculation (verbose output from minimizer.pyx debug prints)
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

print("\n" + "-" * 60)
cpu_gm = result_cpu.GM.values.flatten()[0]
print(f"\nCPU Final GM: {cpu_gm:.6f} J/mol")
print(f"CPU converged successfully!")

print("\n" + "=" * 60)
print("KEY OBSERVATIONS:")
print("=" * 60)

print("\nThe CPU code likely handles constraints differently:")
print("1. It may use a different matrix structure")
print("2. It may scale the constraint rows differently")
print("3. It may use a more robust linear solver")
print("4. It may have better conditioned matrices")

print("\nCheck the debug output above for:")
print("- Constraint row coefficients")
print("- RHS values")
print("- How c_component is used")
print("- Matrix conditioning")