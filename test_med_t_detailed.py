#!/usr/bin/env python
"""Detailed test of Med T balanced condition where CPU/GPU are closest."""

import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = 'all'  # Maximum debug output

from pycalphad import Database, equilibrium, variables as v
import warnings

warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']

# Test with LIQUID and ALCU_ZETA
phases = ['LIQUID', 'ALCU_ZETA']

print("="*80)
print("Testing Med T, balanced condition - closest CPU/GPU agreement")
print("Conditions: X(AL)=0.60, X(CU)=0.30, X(FE)=0.10, T=900K")
print("="*80)

# Check phase properties
for phase in phases:
    p = dbf.phases[phase]
    print(f"\n{phase}:")
    print(f"  Sublattices: {p.sublattices}")
    print(f"  Constituents: {p.constituents}")
    print(f"  Site ratio sum: {sum(p.sublattices)}")

conditions = {v.T: 900, v.P: 101325, v.N: 1, v.X('CU'): 0.3, v.X('FE'): 0.1}

print("\n" + "="*80)
print("CPU CALCULATION")
print("="*80)
try:
    cpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 10}, verbose=True)
    cpu_gm = float(cpu_result.GM.values)
    print(f"\nCPU RESULT: GM={cpu_gm:.2f} J/mol")
    
    # Extract phases
    for idx in range(cpu_result.dims['vertex']):
        np_val = float(cpu_result.NP.values[0,0,0,0,0,idx])
        if np_val > 1e-6:
            phase_name = str(cpu_result.Phase.values[0,0,0,0,0,idx])
            if phase_name and phase_name != '':
                print(f"  {phase_name}: NP={np_val:.4f}")
                # Get site fractions
                Y = cpu_result.Y.sel(vertex=idx).values
                print(f"    Site fractions: {Y[0,0,0,0]}")
                
except Exception as e:
    print(f"CPU FAILED: {type(e).__name__}: {str(e)}")
    cpu_gm = None

print("\n" + "="*80)
print("GPU CALCULATION")
print("="*80)
try:
    gpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 10}, verbose=True, gpu=True)
    gpu_gm = float(gpu_result.GM.values)
    print(f"\nGPU RESULT: GM={gpu_gm:.2f} J/mol")
    
    # Extract phases
    for idx in range(gpu_result.dims['vertex']):
        np_val = float(gpu_result.NP.values[0,0,0,0,0,idx])
        if np_val > 1e-6:
            phase_name = str(gpu_result.Phase.values[0,0,0,0,0,idx])
            if phase_name and phase_name != '':
                print(f"  {phase_name}: NP={np_val:.4f}")
                # Get site fractions
                Y = gpu_result.Y.sel(vertex=idx).values
                print(f"    Site fractions: {Y[0,0,0,0]}")
                
    if cpu_gm is not None:
        diff = abs(cpu_gm - gpu_gm)
        print(f"\nDIFFERENCE: {diff:.6f} J/mol")
        
except Exception as e:
    print(f"GPU FAILED: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)