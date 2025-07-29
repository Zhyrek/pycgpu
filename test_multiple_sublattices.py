#!/usr/bin/env python
"""Test GPU with phases having multiple sublattices but no vacancy."""

from pycalphad import Database, equilibrium
import pycalphad.variables as v

db = Database('Al-Cu-Fe.tdb')

# First, let's examine ALCU_ZETA structure
print("Phase structure analysis:")
print("="*60)

phases_to_check = ['LIQUID', 'ALCU_ZETA', 'FCC_A1', 'BCC_A2']

for phase_name in phases_to_check:
    if phase_name in db.phases:
        phase = db.phases[phase_name]
        print(f"\n{phase_name}:")
        print(f"  Sublattices: {phase.sublattices}")
        print(f"  Constituents: {phase.constituents}")
        print(f"  Number of sublattices: {len(phase.sublattices)}")
        has_va = any('VA' in str(constituents) for constituents in phase.constituents)
        print(f"  Has VA in sublattices: {has_va}")

# Test conditions
conditions = {
    v.T: 1000,
    v.P: 101325,
    v.X('AL'): 0.5,
    v.X('CU'): 0.3,
    v.N: 1
}

print("\n" + "="*60)
print("GPU Compilation Tests:")
print("="*60)
print(f"Test conditions: X(AL)=0.5, X(CU)=0.3, X(FE)=0.2 at 1000K")

# Test 1: LIQUID (1 sublattice, no VA)
print("\n1. LIQUID (1 sublattice, no VA):")
try:
    result = equilibrium(db, ['AL','CU','FE','VA'], ['LIQUID'], conditions,
                        calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print("   GPU: SUCCESS")
    gm = result.GM.values[0,0,0,0]
    print(f"   GM = {gm:.1f} J/mol")
except Exception as e:
    print(f"   GPU: FAILED - {type(e).__name__}: {str(e)[:50]}...")

# Test 2: ALCU_ZETA (2 sublattices, no VA)
print("\n2. ALCU_ZETA (2 sublattices, no VA):")
try:
    result = equilibrium(db, ['AL','CU','FE','VA'], ['ALCU_ZETA'], conditions,
                        calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print("   GPU: SUCCESS")
    gm = result.GM.values[0,0,0,0]
    print(f"   GM = {gm:.1f} J/mol")
except Exception as e:
    print(f"   GPU: FAILED - {type(e).__name__}: {str(e)[:50]}...")

# Test 3: FCC_A1 (2 sublattices, with VA)
print("\n3. FCC_A1 (2 sublattices, with VA):")
try:
    result = equilibrium(db, ['AL','CU','FE','VA'], ['FCC_A1'], conditions,
                        calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print("   GPU: SUCCESS")
    gm = result.GM.values[0,0,0,0]
    print(f"   GM = {gm:.1f} J/mol")
except Exception as e:
    print(f"   GPU: FAILED - {type(e).__name__}: {str(e)[:50]}...")

# Test 4: Mix of phases
print("\n4. LIQUID + ALCU_ZETA (both without VA sublattices):")
try:
    result = equilibrium(db, ['AL','CU','FE','VA'], ['LIQUID', 'ALCU_ZETA'], conditions,
                        calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print("   GPU: SUCCESS")
    gm = result.GM.values[0,0,0,0]
    print(f"   GM = {gm:.1f} J/mol")
except Exception as e:
    print(f"   GPU: FAILED - {type(e).__name__}: {str(e)[:50]}...")

print("\n" + "="*60)
print("Conclusion:")
print("- If ALCU_ZETA (2 sublattices, no VA) works: Issue is specifically with VA in sublattices")
print("- If ALCU_ZETA fails: Issue is with multiple sublattices in general")