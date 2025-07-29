#!/usr/bin/env python
"""Compare equilibrium matrix construction between CPU and GPU for multiple sublattices."""

import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = 'matrix'

from pycalphad import Database, equilibrium, variables as v
import warnings

warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']

# Test with LIQUID (1 sublattice) and ALCU_ZETA (2 sublattices with 9,11 sites)
phases = ['LIQUID', 'ALCU_ZETA']

print(f"Testing equilibrium matrix construction with phases: {phases}")
for phase in phases:
    if phase in dbf.phases:
        p = dbf.phases[phase]
        print(f"  {phase}: sublattices={p.sublattices}, constituents={p.constituents}")
print()

# Simple test condition
conditions = {v.T: 900, v.P: 101325, v.N: 1, v.X('CU'): 0.17, v.X('FE'): 0.0}

print(f"Test conditions: X(AL)=0.83, X(CU)=0.17, X(FE)=0.00, T=900K")
print()

# Capture matrix output by looking for patterns
print("Running CPU calculation to capture matrix structure...")
try:
    cpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 5}, verbose=False)
    print("CPU calculation completed")
except Exception as e:
    print(f"CPU FAILED: {type(e).__name__}: {str(e)}")

print("\nRunning GPU calculation to capture matrix structure...")
try:
    gpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 5}, verbose=False, gpu=True)
    print("GPU calculation completed")
except Exception as e:
    print(f"GPU FAILED: {type(e).__name__}: {str(e)}")

print("\nCheck the debug output above to compare matrix dimensions and structure.")