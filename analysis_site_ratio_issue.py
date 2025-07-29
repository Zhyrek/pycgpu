#!/usr/bin/env python
"""Analysis of the site ratio issue in GPU multi-sublattice handling."""

print("="*80)
print("SITE RATIO ISSUE IDENTIFIED")
print("="*80)

print("\nThe Problem:")
print("-" * 50)
print("In the system amount constraint row, the coefficients are:")
print("- LIQUID phases: coefficient = 1.0")
print("- ALCU_ZETA phase: coefficient = 20.0 (site ratio sum)")

print("\nThis happens because:")
print("1. formulamole calculations return mole fractions per formula unit")
print("2. For ALCU_ZETA: 1 formula unit = 20 atoms (9 + 11 site ratios)")
print("3. For LIQUID: 1 formula unit = 1 atom")

print("\nThe matrix row looks like:")
print("1.0 * phase_amt[LIQUID_1] + 20.0 * phase_amt[ALCU_ZETA] + 1.0 * phase_amt[LIQUID_2] = 1.0")

print("\nWhy this causes problems:")
print("1. The matrix becomes poorly conditioned (coefficients vary by 20x)")
print("2. Small errors in ALCU_ZETA are amplified 20x")
print("3. The solver may incorrectly remove phases")

print("\nHow the CPU likely handles this:")
print("-" * 50)
print("The CPU code probably normalizes by moles_normalization:")
print("- Uses NP (mole fraction) instead of raw phase_amt")
print("- Or divides phase_amt by moles_normalization in the constraint")

print("\nThe fix needed in GPU code:")
print("-" * 50)
print("In write_row_fixed_mole_amount, instead of:")
print("  coefficient = phase_amt * masses[comp]")
print("")
print("Should be:")
print("  coefficient = phase_amt * masses[comp] / moles_normalization")
print("")
print("This would give:")
print("- LIQUID: phase_amt * 1.0 / 1.0 = phase_amt")
print("- ALCU_ZETA: phase_amt * 20.0 / 20.0 = phase_amt")
print("")
print("Making all coefficients equal to 1.0 in the matrix!")