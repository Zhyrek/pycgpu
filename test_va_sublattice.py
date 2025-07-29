#!/usr/bin/env python
"""Test if GPU can handle phases with VA in sublattices."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v

# Test with Al-Cu-Fe which has VA in sublattices
print("Testing Al-Cu-Fe with VA in sublattices")
print("="*60)

db_alcufe = Database('Al-Cu-Fe.tdb')
components_alcufe = ['AL', 'CU', 'FE', 'VA']

# Look at phase structures
print("\nPhase structures in Al-Cu-Fe:")
for phase_name in ['FCC_A1', 'BCC_A2']:
    if phase_name in db_alcufe.phases:
        phase = db_alcufe.phases[phase_name]
        print(f"\n{phase_name}:")
        print(f"  Sublattices: {phase.sublattices}")
        print(f"  Constituents: {phase.constituents}")

# Simple test condition
conditions = {
    v.T: 1000,
    v.P: 101325,
    v.X('AL'): 0.5,
    v.X('CU'): 0.3,
    v.N: 1
}

print("\nTest condition: X(AL)=0.5, X(CU)=0.3, X(FE)=0.2 at 1000K")

# Test individual phases
for phase in ['FCC_A1', 'BCC_A2', 'LIQUID']:
    print(f"\n{phase} phase only:")
    
    # CPU
    try:
        cpu_result = equilibrium(db_alcufe, components_alcufe, [phase], conditions, calc_opts={'pdens': 50})
        print(f"  CPU: Success, GM = {cpu_result.GM.values[0,0,0,0]:.1f} J/mol")
        cpu_success = True
    except Exception as e:
        print(f"  CPU: Failed - {e}")
        cpu_success = False
    
    # GPU
    try:
        gpu_result = equilibrium(db_alcufe, components_alcufe, [phase], conditions, calc_opts={'pdens': 50}, gpu=True, verbose=False)
        print(f"  GPU: Success, GM = {gpu_result.GM.values[0,0,0,0]:.1f} J/mol")
        if cpu_success:
            diff = abs(gpu_result.GM.values[0,0,0,0] - cpu_result.GM.values[0,0,0,0])
            print(f"  Difference: {diff:.3f} J/mol")
    except Exception as e:
        print(f"  GPU: Failed - {e}")

# Now compare with Nb-Ti which doesn't have VA in BCC
print("\n" + "="*60)
print("Testing Nb-Ti without VA in sublattices")
print("="*60)

db_nbti = Database('NbTi.tdb')
components_nbti = ['NB', 'TI', 'VA']

print("\nPhase structures in Nb-Ti:")
for phase_name in ['BCC_A2', 'HCP_A3']:
    if phase_name in db_nbti.phases:
        phase = db_nbti.phases[phase_name]
        print(f"\n{phase_name}:")
        print(f"  Sublattices: {phase.sublattices}")
        print(f"  Constituents: {phase.constituents}")

conditions_nbti = {
    v.T: 1000,
    v.P: 101325,
    v.X('TI'): 0.5,
    v.N: 1
}

print("\nTest condition: X(TI)=0.5 at 1000K")

# Test BCC_A2 in Nb-Ti
print(f"\nBCC_A2 phase only:")

# CPU
try:
    cpu_result = equilibrium(db_nbti, components_nbti, ['BCC_A2'], conditions_nbti, calc_opts={'pdens': 50})
    print(f"  CPU: Success, GM = {cpu_result.GM.values[0,0,0,0]:.1f} J/mol")
    cpu_success = True
except Exception as e:
    print(f"  CPU: Failed - {e}")
    cpu_success = False

# GPU
try:
    gpu_result = equilibrium(db_nbti, components_nbti, ['BCC_A2'], conditions_nbti, calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print(f"  GPU: Success, GM = {gpu_result.GM.values[0,0,0,0]:.1f} J/mol")
    if cpu_success:
        diff = abs(gpu_result.GM.values[0,0,0,0] - cpu_result.GM.values[0,0,0,0])
        print(f"  Difference: {diff:.3f} J/mol")
except Exception as e:
    print(f"  GPU: Failed - {e}")

print("\n" + "="*60)
print("Key difference:")
print("- Al-Cu-Fe FCC_A1: (AL,CU,FE) : (VA)")
print("- Al-Cu-Fe BCC_A2: (AL,CU,FE,VA) : (VA)")  
print("- Nb-Ti BCC_A2: (NB,TI) : no second sublattice")
print("The GPU appears to fail with phases that have VA in sublattices!")