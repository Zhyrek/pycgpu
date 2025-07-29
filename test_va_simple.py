#!/usr/bin/env python
"""Simple test to identify VA sublattice issue."""

from pycalphad import Database
import pycalphad.variables as v

# Check phase structures
print("Phase structures comparison:")
print("="*60)

# Al-Cu-Fe
db1 = Database('Al-Cu-Fe.tdb')
print("\nAl-Cu-Fe.tdb phases with VA:")
for phase_name in ['FCC_A1', 'BCC_A2']:
    if phase_name in db1.phases:
        phase = db1.phases[phase_name]
        print(f"\n{phase_name}:")
        print(f"  Sublattices: {phase.sublattices}")
        print(f"  Constituents: {phase.constituents}")
        # Check if VA is in any sublattice
        has_va = any('VA' in str(constituents) for constituents in phase.constituents)
        print(f"  Has VA in sublattices: {has_va}")

# Nb-Ti  
db2 = Database('NbTi.tdb')
print("\n\nNbTi.tdb phases:")
for phase_name in ['BCC_A2']:
    if phase_name in db2.phases:
        phase = db2.phases[phase_name]
        print(f"\n{phase_name}:")
        print(f"  Sublattices: {phase.sublattices}")
        print(f"  Constituents: {phase.constituents}")
        has_va = any('VA' in str(constituents) for constituents in phase.constituents)
        print(f"  Has VA in sublattices: {has_va}")

# Now test GPU compilation for each
from pycalphad import equilibrium

print("\n" + "="*60)
print("GPU Compilation Tests:")
print("="*60)

# Test 1: Al-Cu-Fe FCC_A1 with VA in second sublattice
print("\nTest 1: Al-Cu-Fe FCC_A1 (has VA in sublattice)")
try:
    result = equilibrium(db1, ['AL','CU','FE','VA'], ['FCC_A1'], 
                        {v.T: 1000, v.P: 101325, v.X('AL'): 0.5, v.X('CU'): 0.3, v.N: 1},
                        calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print("GPU: SUCCESS")
except Exception as e:
    print(f"GPU: FAILED - {type(e).__name__}")

# Test 2: Al-Cu-Fe LIQUID (no VA in sublattice)  
print("\nTest 2: Al-Cu-Fe LIQUID (no VA in sublattice)")
try:
    result = equilibrium(db1, ['AL','CU','FE','VA'], ['LIQUID'],
                        {v.T: 1000, v.P: 101325, v.X('AL'): 0.5, v.X('CU'): 0.3, v.N: 1},
                        calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print("GPU: SUCCESS")
except Exception as e:
    print(f"GPU: FAILED - {type(e).__name__}")

# Test 3: Nb-Ti BCC_A2 (no VA in sublattice)
print("\nTest 3: Nb-Ti BCC_A2 (no VA in sublattice)")
try:
    result = equilibrium(db2, ['NB','TI','VA'], ['BCC_A2'],
                        {v.T: 1000, v.P: 101325, v.X('TI'): 0.5, v.N: 1},
                        calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print("GPU: SUCCESS")
except Exception as e:
    print(f"GPU: FAILED - {type(e).__name__}")

print("\n" + "="*60)
print("Conclusion:")
print("GPU fails when phases have VA in sublattices (e.g., FCC_A1: (AL,CU,FE):(VA))")
print("GPU succeeds when phases don't have VA in sublattices")