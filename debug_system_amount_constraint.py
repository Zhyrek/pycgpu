#!/usr/bin/env python
"""Debug system amount constraint handling for multi-sublattice phases."""

print("="*80)
print("SYSTEM AMOUNT CONSTRAINT ANALYSIS")
print("="*80)

print("\nThe Issue:")
print("-" * 50)
print("For ALCU_ZETA with site ratios (9.0, 11.0):")
print("- Site ratio sum = 20.0")
print("- This affects how phase amounts contribute to system amount")
print("- CPU may handle this differently than GPU")

print("\nIn the equilibrium matrix, the system amount constraint row has coefficients:")
print("- For LIQUID: coefficient = 1.0 (site ratio sum = 1.0)")
print("- For ALCU_ZETA: coefficient = 20.0 (site ratio sum = 20.0)")
print("- This creates very different magnitudes in the matrix")

print("\nPossible differences between CPU and GPU:")
print("1. CPU might normalize phase amounts by site ratio sums")
print("2. GPU might use raw phase amounts without normalization")
print("3. The matrix coefficients might be constructed differently")

print("\nSpecific code to check:")
print("-" * 50)
print("1. In write_row_fixed_mole_amount (GPU):")
print("   - Currently uses: phase_amt_sys[idx] * masses[comp_idx]")
print("   - Should it be: phase_amt_sys[idx] * masses[comp_idx] / site_ratio_sum?")

print("\n2. In CPU code (minimizer.pyx):")
print("   - Check if moles_normalization includes site ratio sum")
print("   - Check if phase amounts are normalized before use")

print("\n3. Phase amount interpretation:")
print("   - phase_amt: formula units of the phase")
print("   - NP: mole fraction of phase (normalized)")
print("   - For multi-sublattice: 1 formula unit = site_ratio_sum atoms")

print("\nNext steps:")
print("-" * 50)
print("1. Add debug output to print matrix coefficients for system amount row")
print("2. Compare CPU vs GPU matrix construction")
print("3. Check if site ratio normalization is missing in GPU")
print("4. Verify formulamole calculations account for site ratios")