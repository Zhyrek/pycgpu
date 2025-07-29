#\!/usr/bin/env python
"""Trace AlCu divergence between CPU and GPU line by line."""

import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = '1'  # Enable debug output

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load AlCuFe database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'ALCU_ZETA']

# Test condition that shows 806 J/mol error
conditions = {
    v.T: 900,  
    v.P: 101325, 
    v.N: 1, 
    v.X('AL'): 0.6,
    v.X('CU'): 0.3
}

print("="*80)
print("TRACING CPU VS GPU DIVERGENCE FOR ALCU SYSTEM")
print("="*80)
print("Condition: T=900K, X(AL)=0.6, X(CU)=0.3")
print("Looking for first deviation in solver iterations...")
print("="*80)

print("\n" + "="*40 + " CPU RUN " + "="*40)
cpu_result = equilibrium(dbf, comps, phases, conditions, 
                       calc_opts={'pdens': 100}, verbose=False)
cpu_gm = float(cpu_result.GM.values.item())

print("\n" + "="*40 + " GPU RUN " + "="*40)
gpu_result = equilibrium(dbf, comps, phases, conditions, 
                       calc_opts={'pdens': 100}, verbose=False, gpu=True)
gpu_gm = float(gpu_result.GM.values.item())

print("\n" + "="*80)
print("FINAL RESULTS:")
print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"GPU GM: {gpu_gm:.6f} J/mol")
print(f"Difference: {abs(cpu_gm - gpu_gm):.2f} J/mol")
print("="*80)
EOF < /dev/null
