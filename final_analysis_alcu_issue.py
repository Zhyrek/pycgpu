#!/usr/bin/env python
"""Final analysis of remaining AlCu multi-sublattice issue."""

print("="*80)
print("REMAINING ALCU MULTI-SUBLATTICE ISSUE")
print("="*80)

print("\nWhat we fixed:")
print("1. dgelsd_device for better numerical stability")
print("2. Normalized system amount constraint coefficients by moles_normalization")

print("\nWhat's still wrong:")
print("- GPU is still removing phases that CPU keeps")
print("- Errors are still large (806 J/mol for Med T, balanced)")

print("\nPossible remaining issues:")
print("-" * 50)

print("\n1. System amount calculation inconsistency:")
print("   - In recompute(), system_amount += phase_amt * masses[comp]")
print("   - But masses[comp] already includes site ratio effects")
print("   - For ALCU_ZETA: masses sum to 20.0")
print("   - This might cause system_amount to be incorrect")

print("\n2. Phase amount interpretation:")
print("   - phase_amt should be in formula units")
print("   - But the meaning of 'formula unit' differs between phases")
print("   - LIQUID: 1 formula unit = 1 atom")
print("   - ALCU_ZETA: 1 formula unit = 20 atoms")

print("\n3. Mole fraction calculation:")
print("   - mole_fractions[comp] += phase_amt * masses[comp]")
print("   - Then normalized by system_amount")
print("   - But this might not account for site ratios correctly")

print("\n4. Chemical potential calculation:")
print("   - The gradient calculations might not account for site ratios")
print("   - This could lead to incorrect driving forces")

print("\nThe core issue:")
print("-" * 50)
print("The GPU and CPU are interpreting phase amounts differently")
print("when multi-sublattice phases are present. The normalization")
print("we applied to the matrix coefficients helps conditioning")
print("but doesn't fix the fundamental interpretation issue.")

print("\nNext steps:")
print("1. Check how CPU calculates system_amount")
print("2. Verify phase_amt interpretation matches between CPU/GPU")
print("3. Consider if NP (mole fraction) should be used instead of phase_amt")