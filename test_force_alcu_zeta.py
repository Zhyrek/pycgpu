#!/usr/bin/env python
"""Test that forces ALCU_ZETA phase to be stable by using conditions in its stability region."""

import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = 'matrix'

from pycalphad import Database, equilibrium, variables as v
import warnings

warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']

# Only allow ALCU_ZETA phase to force it to be stable
phases = ['ALCU_ZETA']

print(f"Testing matrix construction with only ALCU_ZETA phase (forced)")
p = dbf.phases['ALCU_ZETA']
print(f"ALCU_ZETA: sublattices={p.sublattices}, constituents={p.constituents}")
print(f"Site fraction DOF: 1 (Y(ALCU_ZETA,1,CU) since first sublattice is pure AL)")
print()

# Use composition where ALCU_ZETA should be stable
conditions = {v.T: 700, v.P: 101325, v.N: 1, v.X('CU'): 0.45, v.X('FE'): 0.0}

print(f"Test conditions: X(AL)=0.55, X(CU)=0.45, X(FE)=0.00, T=700K")
print()

# Run CPU
print("Running CPU calculation...")
try:
    cpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 5}, verbose=False)
    print("CPU calculation completed")
    print(f"CPU GM: {float(cpu_result.GM.values):.2f} J/mol")
except Exception as e:
    print(f"CPU FAILED: {type(e).__name__}: {str(e)}")

print("\nRunning GPU calculation...")
try:
    gpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 5}, verbose=False, gpu=True)
    print("GPU calculation completed")
    print(f"GPU GM: {float(gpu_result.GM.values):.2f} J/mol")
    
    if 'cpu_result' in locals():
        diff = abs(float(cpu_result.GM.values) - float(gpu_result.GM.values))
        print(f"\nDifference: {diff:.6f} J/mol - {'PASS' if diff < 1.0 else 'FAIL'}")
        
except Exception as e:
    print(f"GPU FAILED: {type(e).__name__}: {str(e)}")

print("\nCheck the debug output above to verify matrix handling for multi-sublattice phase.")