#!/usr/bin/env python
"""Test single Al-Cu-Fe condition for GPU vs CPU."""

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import warnings

warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = filter_phases(dbf, comps)

print(f"Testing single Al-Cu-Fe condition")
print(f"Available phases: {phases}")

# Single test condition - Al-Cu binary edge
conditions = {v.T: 800, v.P: 101325, v.N: 1, v.X('CU'): 0.3, v.X('FE'): 0.0}

print("\nRunning CPU calculation...")
cpu_result = equilibrium(dbf, comps, phases, conditions, calc_opts={'pdens': 5}, verbose=False)
cpu_gm = float(cpu_result.GM.values)

print("Running GPU calculation...")
gpu_result = equilibrium(dbf, comps, phases, conditions, calc_opts={'pdens': 5}, verbose=False, gpu=True)
gpu_gm = float(gpu_result.GM.values)

diff = abs(gpu_gm - cpu_gm)
status = "PASS" if diff < 1.0 else "FAIL"

print(f"\nResults:")
print(f"  CPU GM: {cpu_gm:.6f} J/mol")
print(f"  GPU GM: {gpu_gm:.6f} J/mol")
print(f"  Difference: {diff:.6f} J/mol - {status}")