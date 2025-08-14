#!/usr/bin/env python
"""Debug the formulamole function generated for BCC_B2."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pycalphad import Database, Model, variables as v
import numpy as np

def main():
    """Debug BCC_B2 formulamole generation."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    print("=" * 80)
    print("BCC_B2 FORMULAMOLE ANALYSIS")
    print("=" * 80)
    
    # Create model for BCC_B2
    model = Model(dbf, comps, 'BCC_B2')
    
    print("\nBCC_B2 Phase Structure:")
    print(f"  Sublattices: {dbf.phases['BCC_B2'].sublattices}")
    print(f"  Total sites: {sum(dbf.phases['BCC_B2'].sublattices)}")
    
    print("\nSite fraction variables:")
    for i, subl in enumerate(dbf.phases['BCC_B2'].sublattices):
        constituents = dbf.phases['BCC_B2'].constituents[i]
        print(f"  Sublattice {i+1} ({subl} sites): {constituents}")
    
    print("\nFormula mole expressions for each component:")
    for comp in ['AL', 'CU', 'FE']:
        moles_expr = model.moles(comp)
        print(f"\n  moles({comp}) = {moles_expr}")
        
        # Check for potential issues
        str_expr = str(moles_expr)
        
        # Check for divisions that could cause issues at X=0.5
        if '(1 - 2*' in str_expr or '(1.0 - 2.0*' in str_expr:
            print(f"    ⚠ WARNING: Expression contains (1 - 2*...) which is zero when variable = 0.5")
        
        # Check for divisions by site fractions
        if 'Y_' in str_expr and '/' in str_expr:
            print(f"    ⚠ WARNING: Expression may divide by site fractions")
    
    print("\n" + "-" * 80)
    print("Testing specific site fraction combinations:")
    print("-" * 80)
    
    # Test what happens with specific site fractions that match the failing composition
    # At X(AL)=0.2, X(CU)=0.5, X(FE)=0.3
    # We need to figure out what site fractions would give these bulk compositions
    
    print("\nFor bulk composition X(AL)=0.2, X(CU)=0.5, X(FE)=0.3:")
    print("Sublattices: (0.5, 0.5, 3.0) sites")
    print("Need to solve for site fractions Y that give the right bulk composition")
    
    # Simple approximation: put each element preferentially in one sublattice
    # This is just to see if certain combinations cause numerical issues
    test_cases = [
        # (Y_AL_0, Y_CU_0, Y_FE_0), (Y_AL_1, Y_CU_1, Y_FE_1), (Y_AL_2, Y_CU_2, Y_FE_2)
        ([0.4, 0.5, 0.1], [0.0, 1.0, 0.0], [0.2, 0.5, 0.3]),  # Mixed
        ([0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]),  # Special case with 0.5
        ([0.2, 0.5, 0.3], [0.2, 0.5, 0.3], [0.2, 0.5, 0.3]),  # All sublattices same
    ]
    
    for i, (subl0, subl1, subl2) in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print(f"  Sublattice 0: AL={subl0[0]:.2f}, CU={subl0[1]:.2f}, FE={subl0[2]:.2f}")
        print(f"  Sublattice 1: AL={subl1[0]:.2f}, CU={subl1[1]:.2f}, FE={subl1[2]:.2f}")
        print(f"  Sublattice 2: AL={subl2[0]:.2f}, CU={subl2[1]:.2f}, FE={subl2[2]:.2f}")
        
        # Calculate bulk composition from these site fractions
        # X(i) = sum over sublattices of (site_fraction_i * sublattice_sites) / total_sites
        x_al = (subl0[0]*0.5 + subl1[0]*0.5 + subl2[0]*3.0) / 4.0
        x_cu = (subl0[1]*0.5 + subl1[1]*0.5 + subl2[1]*3.0) / 4.0
        x_fe = (subl0[2]*0.5 + subl1[2]*0.5 + subl2[2]*3.0) / 4.0
        
        print(f"  Resulting bulk: X(AL)={x_al:.3f}, X(CU)={x_cu:.3f}, X(FE)={x_fe:.3f}")
        
        # Check for numerical issues
        if abs(x_cu - 0.5) < 0.01:
            print(f"    ⚠ This gives X(CU) ≈ 0.5, potential for numerical issues")
    
    print("\n" + "=" * 80)
    print("HYPOTHESIS:")
    print("The BCC_B2 phase with fractional sublattice sites (0.5, 0.5, 3.0)")
    print("may have formulamole expressions that become singular or poorly")
    print("conditioned when certain bulk compositions (like X(CU)=0.5) are reached.")
    print("This could explain why the GPU solver diverges at these specific points.")
    print("=" * 80)

if __name__ == "__main__":
    main()