#!/usr/bin/env python
"""Analyze why CPU doesn't suffer from the matrix conditioning issue."""

print("="*80)
print("WHY CPU HANDLES MULTI-SUBLATTICE PHASES BETTER THAN GPU")
print("="*80)

print("\n1. THE KEY DIFFERENCE: How Phase Amounts Are Used")
print("-" * 50)

print("\nGPU Approach:")
print("- Uses raw phase amounts (phase_amt) in the system amount constraint")
print("- For ALCU_ZETA: phase_amt * 20.0 (site ratio sum)")
print("- For LIQUID: phase_amt * 1.0")
print("- This creates coefficients [1.0, 20.0, 1.0] in the matrix")

print("\nCPU Approach:")
print("- The CPU code calculates 'moles_normalization' for each phase")
print("- This is the sum of masses, which effectively accounts for site ratios")
print("- The CPU solver likely uses normalized quantities internally")

print("\n2. EVIDENCE FROM THE CODE:")
print("-" * 50)

print("\nGPU Matrix (Row 5 - System Amount):")
print("  Coefficients: [1.0, 20.0, 1.0]")
print("  This directly uses site ratio sums as multipliers")

print("\nCPU Code (minimizer.pyx):")
print("  - Calculates: moles_normalization = sum(masses)")
print("  - Uses moles_normalization_grad in mole fraction constraints")
print("  - This suggests internal normalization")

print("\n3. THE MATHEMATICAL DIFFERENCE:")
print("-" * 50)

print("\nWhat the GPU does:")
print("  System amount = φ₀*1.0 + φ₁*20.0 + φ₂*1.0 = 1.0")
print("  (where φᵢ are phase amounts)")

print("\nWhat the CPU effectively does:")
print("  System amount = NP₀ + NP₁ + NP₂ = 1.0")
print("  (where NPᵢ are normalized phase amounts)")
print("  or")
print("  It may scale the constraint row by 1/site_ratio_sum internally")

print("\n4. WHY THIS MATTERS:")
print("-" * 50)

print("\n- The GPU's approach creates a poorly conditioned matrix (cond # = 4e19)")
print("- Large coefficients (20.0) dominate the solution")
print("- Small phases get eliminated due to numerical errors")

print("\n- The CPU's approach maintains better conditioning:")
print("  - Either by using normalized amounts (NP)")
print("  - Or by internal scaling/preconditioning")
print("  - This prevents extreme solutions")

print("\n5. THE SOLVER DIFFERENCE:")
print("-" * 50)

print("\nCPU likely uses:")
print("- LAPACK routines with built-in equilibration/scaling")
print("- These routines automatically scale matrix rows/columns")
print("- This improves numerical stability for poorly conditioned systems")

print("\nGPU uses:")
print("- Custom SVD implementation")
print("- May not have automatic equilibration")
print("- More sensitive to poor conditioning")

print("\n6. CONCLUSION:")
print("-" * 50)
print("\nThe CPU doesn't suffer from this issue because:")
print("1. It may internally normalize phase amounts by site ratio sums")
print("2. LAPACK solvers have built-in matrix equilibration")
print("3. The CPU code structure suggests awareness of this issue")
print("   (via moles_normalization calculations)")
print("\nThe GPU needs to either:")
print("- Normalize phase amounts in the system constraint")
print("- Implement matrix equilibration/scaling")
print("- Use NP (mole fractions) instead of raw phase amounts")

print("\n" + "="*80)