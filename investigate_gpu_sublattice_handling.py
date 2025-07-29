#!/usr/bin/env python
"""Investigate how GPU handles multi-sublattice phases differently from CPU."""

print("="*80)
print("INVESTIGATING GPU MULTI-SUBLATTICE HANDLING")
print("="*80)

print("\n1. KNOWN DIFFERENCES:")
print("-" * 50)
print("a) Site ratio sum handling:")
print("   - ALCU_ZETA has site ratios (9.0, 11.0), sum = 20.0")
print("   - LIQUID has site ratio 1.0")
print("   - This creates very different matrix coefficients")

print("\nb) Matrix conditioning:")
print("   - dgelsd_device helps but doesn't fully solve the problem")
print("   - Error went from ~6.94 J/mol to ~806 J/mol (worse!)")

print("\n2. POTENTIAL ISSUES TO INVESTIGATE:")
print("-" * 50)

print("\na) Phase amount normalization:")
print("   - CPU may normalize phase amounts by site ratio sums")
print("   - GPU might use raw phase amounts")
print("   - Check: phase_amt vs NP (mole fraction)")

print("\nb) Gradient/Hessian calculations:")
print("   - Multi-sublattice phases have more complex derivatives")
print("   - Check if GPU correctly handles all sublattice contributions")

print("\nc) Constraint formulation:")
print("   - System amount constraint coefficients")
print("   - Mole fraction constraints")
print("   - Check if site ratio sums are properly accounted for")

print("\nd) Initial guess scaling:")
print("   - Starting values for multi-sublattice phases")
print("   - Phase amounts initialization")

print("\ne) formulamole calculations:")
print("   - How masses are calculated for multi-sublattice phases")
print("   - Check if site occupancies are properly weighted")

print("\n3. SPECIFIC CODE LOCATIONS TO CHECK:")
print("-" * 50)

print("\na) In minimizer.h:")
print("   - compute_formulamole() function")
print("   - How it handles multiple sublattices")

print("\nb) In eqsolver.h:")
print("   - System amount constraint construction")
print("   - Check coefficients for each phase")

print("\nc) In comp_set.h:")
print("   - CompositionSet initialization")
print("   - phase_amt vs NP handling")

print("\nd) Phase record initialization:")
print("   - How sublattices are stored/accessed")
print("   - Site fraction ordering")

print("\n4. NEXT STEPS:")
print("-" * 50)
print("1. Add debug output to trace matrix coefficients")
print("2. Compare formulamole calculations between CPU/GPU")
print("3. Check if phase amounts need normalization")
print("4. Verify gradient calculations for multi-sublattice phases")