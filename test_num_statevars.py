#!/usr/bin/env python
"""Test to understand num_statevars difference between binary and ternary systems."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

print("COMPARING BINARY VS TERNARY num_statevars")
print("=" * 60)

# Binary system: Au-Bi
print("\nBINARY SYSTEM (Au-Bi):")
tdb_binary = Database('AuBi-07Wan.tdb')
comps_binary = ['AU', 'BI', 'VA']
phases_binary = ['LIQUID', 'FCC_A1']

conditions_binary = {
    v.T: 800,
    v.P: 101325,
    v.X('BI'): 0.3
}

print(f"Components: {comps_binary}")
print(f"Conditions: {list(conditions_binary.keys())}")
print(f"Number of independent components: {len(comps_binary) - 1}")  # Excluding VA
print(f"State variables: T, P")
print(f"Composition variables: X(BI) [X(AU) is dependent]")

# Ternary system: Al-Cu-Fe
print("\nTERNARY SYSTEM (Al-Cu-Fe):")
tdb_ternary = Database('Al-Cu-Fe.tdb')
comps_ternary = ['AL', 'CU', 'FE', 'VA']
phases_ternary = ['LIQUID', 'BCC_A2']

conditions_ternary = {
    v.T: 1200,
    v.P: 101325,
    v.X('CU'): 0.3,
    v.X('FE'): 0.2
}

print(f"Components: {comps_ternary}")
print(f"Conditions: {list(conditions_ternary.keys())}")
print(f"Number of independent components: {len(comps_ternary) - 1}")  # Excluding VA  
print(f"State variables: T, P")
print(f"Composition variables: X(CU), X(FE) [X(AL) is dependent]")

print("\nDEGREES OF FREEDOM ANALYSIS:")
print("-" * 30)

# For a phase with sublattices (M)(VA), DOF = num_sublattices * (num_species_in_sublattice - 1)
# Plus potential ordering variables

print("Binary system DOF:")
print("  - For phases like FCC_A1 or LIQUID with (AU,BI)(VA): DOF = 1 * (2-1) = 1")
print("  - Plus T, P: total workspace state vars = 2")
print("  - So formulagrad expects: [dG/dT, dG/dY1] -> 2 values")
print("  - grad array mapping: temp_grad[0]->grad[2], temp_grad[1]->grad[3]")

print("\nTernary system DOF:")
print("  - For phases like BCC_A2 or LIQUID with (AL,CU,FE)(VA): DOF = 1 * (3-1) = 2")  
print("  - Plus T, P: total workspace state vars = 2")
print("  - So formulagrad expects: [dG/dT, dG/dY1, dG/dY2] -> 3 values")
print("  - grad array mapping: temp_grad[0]->grad[2], temp_grad[1]->grad[3], temp_grad[2]->grad[4]")

print("\nPOTENTIAL ISSUE:")
print("-" * 15)
print("The c_G calculation uses:")
print("  c_G[i] -= full_e_matrix[i,j] * grad[spec->num_statevars + j]")
print("  where spec->num_statevars = 2 for both binary and ternary (T,P)")
print()
print("So for binary: grad[2+0] = grad[2], grad[2+1] = grad[3]")  
print("For ternary: grad[2+0] = grad[2], grad[2+1] = grad[3], grad[2+2] = grad[4]")
print()
print("If formulagrad is not providing the correct number of terms")
print("or if the indexing is wrong, ternary c_G will be incorrect!")

print("\nTEST HYPOTHESIS:")
print("-" * 16)
print("1. Check what formulagrad actually outputs for binary vs ternary")
print("2. Verify the gradient array indexing in both cases")  
print("3. Check if phase_dof is correctly set for ternary phases")
print("4. Verify the c_G loop bounds use the right DOF count")