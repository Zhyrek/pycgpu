#!/usr/bin/env python
"""Analyze the equilibrium matrix scaling issue."""

import numpy as np

def analyze_matrix_condition():
    """Analyze the GPU equilibrium matrix at iteration 0."""
    print("="*80)
    print("EQUILIBRIUM MATRIX SCALING ANALYSIS")
    print("="*80)
    
    # GPU Matrix at iteration 0
    print("\nGPU Equilibrium Matrix (6x5):")
    print("Rows: [Phase 0 energy, Phase 1 energy, Phase 2 energy, Mole frac 1, Mole frac 2, System amount]")
    print("Cols: [μ₀, μ₁, φ₀, φ₁, φ₂]")
    print()
    
    # Matrix coefficients
    A = np.array([
        [7.678741e-01, 4.349367e-02, 0.0, 0.0, 0.0],  # Phase 0 (LIQUID)
        [9.000000e+00, 1.100000e+01, 0.0, 0.0, 0.0],  # Phase 1 (ALCU_ZETA)
        [6.666667e-01, 1.000000e-14, 0.0, 0.0, 0.0],  # Phase 2 (LIQUID)
        [-2.006752e-05, 2.567157e-05, -4.831627e-01, 4.668717e-01, -5.266564e-01],  # X_CU constraint
        [-1.587859e-06, -2.432237e-06, 1.792948e-01, -1.867487e-01, 3.239959e-01],  # X_FE constraint
        [-4.710418e-05, -4.664067e-05, 1.000000e+00, 2.000000e+01, 1.000000e+00]   # System amount
    ])
    
    # RHS
    b = np.array([-5.077997e+04, -5.830183e+04, -5.398409e+04, -7.789637e-02, -3.097390e-02, 1.034507e+01])
    
    print("Matrix A:")
    for i, row in enumerate(A):
        print(f"Row {i}: {row}")
    print(f"\nRHS b: {b}")
    
    # Analyze scaling
    print("\n\nSCALING ANALYSIS:")
    print("1. Row 1 (ALCU_ZETA) has coefficients ~10x larger than other phase rows")
    print("2. Row 5 (System amount) has coefficient 20.0 for ALCU_ZETA vs 1.0 for LIQUID")
    print("3. This 20:1 ratio comes from the site ratio sum (20.0 for ALCU_ZETA)")
    
    # Condition number
    # Extract the 5x5 submatrix (removing one row for square matrix)
    A_square = A[:5, :]
    try:
        cond = np.linalg.cond(A_square)
        print(f"\n4. Condition number of 5x5 submatrix: {cond:.2e}")
        if cond > 1e10:
            print("   *** POORLY CONDITIONED MATRIX! ***")
    except:
        print("\n4. Could not compute condition number")
    
    # Show the solution
    print("\n\nSOLUTION ANALYSIS:")
    solution = np.array([-7.44e+04, 5.56e+04, 1.59e+01, 1.16e-01, -8.75e+00])
    print(f"Solution x: {solution}")
    print(f"\nPhase amount changes:")
    print(f"  Δφ₀ (LIQUID 1) = {solution[2]:.2f} (large positive)")
    print(f"  Δφ₁ (ALCU_ZETA) = {solution[3]:.2f} (small positive)")  
    print(f"  Δφ₂ (LIQUID 2) = {solution[4]:.2f} (large negative)")
    
    print("\n\nWHY PHASE 2 IS REMOVED:")
    print("The large coefficient 20.0 in Row 5 for ALCU_ZETA causes:")
    print("1. The solver to heavily weight ALCU_ZETA in the system amount balance")
    print("2. Small changes in ALCU_ZETA have 20x the effect on mass balance")
    print("3. The solver compensates by removing the small LIQUID phase")
    
    # Compare what would happen with normalized coefficients
    print("\n\nIF WE NORMALIZED THE MATRIX:")
    print("If Row 5 used normalized phase amounts (dividing by site ratio sum):")
    print("  Coefficients would be: [1.0, 1.0, 1.0] instead of [1.0, 20.0, 1.0]")
    print("  This would likely prevent the extreme solution that removes phase 2")

def suggest_fix():
    """Suggest how to fix the issue."""
    print("\n" + "="*80)
    print("SUGGESTED FIX")
    print("="*80)
    print("\n1. NORMALIZE PHASE AMOUNTS in the equilibrium system:")
    print("   - Use NP (mole fraction) instead of phase_amt in the system amount constraint")
    print("   - Or divide by site_ratio_sum when constructing the matrix")
    print("\n2. SCALE THE MATRIX ROWS:")
    print("   - Pre-scale rows to have similar magnitudes")
    print("   - This improves numerical stability")
    print("\n3. USE DIFFERENT SOLVER TOLERANCES:")
    print("   - The GPU solver may need different tolerances for multi-sublattice systems")
    print("   - Consider adaptive tolerances based on site ratio sums")

def main():
    """Main analysis."""
    print("Analysis of GPU/CPU Divergence Due to Matrix Scaling\n")
    
    analyze_matrix_condition()
    suggest_fix()
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("The divergence is caused by poor matrix conditioning when phases have")
    print("very different site ratio sums (1.0 vs 20.0). The GPU solver produces") 
    print("an extreme solution that removes small phases. The CPU likely handles")
    print("this better through different numerical methods or scaling.")

if __name__ == "__main__":
    main()