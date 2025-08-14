#!/usr/bin/env python
"""Check how many constraints are applied for ternary system."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    """Check constraint setup for ternary system."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1']  # Just 2 phases to simplify
    
    # The failing condition
    conditions = {
        v.X('AL'): 0.2,
        v.X('CU'): 0.5,
        v.T: 900,
        v.P: 101325
    }
    
    print("=" * 80)
    print("CONSTRAINT ANALYSIS FOR TERNARY SYSTEM")
    print("=" * 80)
    
    print("\nSystem setup:")
    print(f"  Components: {comps}")
    print(f"  Non-VA components: {[c for c in comps if c != 'VA']}")
    print(f"  Number of non-VA components: {len([c for c in comps if c != 'VA'])}")
    
    print("\nConditions:")
    print(f"  X(AL) = 0.2 (prescribed)")
    print(f"  X(CU) = 0.5 (prescribed)")
    print(f"  X(FE) = 0.3 (implicit, = 1 - 0.2 - 0.5)")
    print(f"  T = 900 K")
    print(f"  P = 101325 Pa")
    
    print("\nFor a ternary system (3 non-VA components):")
    print("  - Total mole fractions must sum to 1: X(AL) + X(CU) + X(FE) = 1")
    print("  - This gives us 1 degree of freedom removed")
    print("  - We prescribe 2 mole fractions (AL and CU)")
    print("  - The third (FE) is determined by the sum constraint")
    
    print("\nChemical potential variables:")
    print("  - We have 3 chemical potentials: μ(AL), μ(CU), μ(FE)")
    print("  - With 2 prescribed mole fractions, we should have:")
    print("    * 1 free chemical potential")
    print("    * 2 fixed chemical potentials")
    
    print("\nExpected equilibrium matrix structure:")
    print("  For 2 phases (LIQUID, FCC_A1) with 2 mole fraction constraints:")
    print("  - Rows 0-1: Mass balance for each phase")
    print("  - Rows 2-3: Mole fraction constraints (X_AL and X_CU)")
    print("  - Row 4: System amount constraint (N=1)")
    print("  - Total: 5 rows")
    print("")
    print("  - Columns 0-2: Chemical potentials (μ_AL, μ_CU, μ_FE)")
    print("  - Columns 3-4: Phase amounts (NP_LIQUID, NP_FCC)")
    print("  - Total: 5 columns")
    
    print("\nCRITICAL CHECK:")
    print("  In a binary system (2 non-VA components):")
    print("    - Only 1 mole fraction needs to be prescribed")
    print("    - 1 free chemical potential, 1 fixed")
    print("  In a ternary system (3 non-VA components):")
    print("    - 2 mole fractions need to be prescribed")
    print("    - 1 free chemical potential, 2 fixed")
    print("")
    print("  If the GPU solver is hardcoded for binary, it might:")
    print("    - Only apply 1 mole fraction constraint instead of 2")
    print("    - Have wrong number of free vs fixed chemical potentials")
    print("    - Not properly handle the second prescribed mole fraction")
    
    # Run calculations to see the actual dimensions
    print("\n" + "=" * 80)
    print("ACTUAL CALCULATION:")
    print("-" * 80)
    
    print("\nCPU calculation:")
    result_cpu = equilibrium(dbf, comps, phases, conditions,
                            calc_opts={'pdens': 50},
                            gpu=False, verbose=False)
    print(f"  Result shape: {result_cpu.GM.shape}")
    print(f"  GM: {result_cpu.GM.values.item():.2f} J/mol")
    
    print("\nGPU calculation:")
    result_gpu = equilibrium(dbf, comps, phases, conditions,
                            calc_opts={'pdens': 50},
                            gpu=True, verbose=False)
    print(f"  Result shape: {result_gpu.GM.shape}")
    print(f"  GM: {result_gpu.GM.values.item():.2f} J/mol")
    
    diff = abs(result_gpu.GM.values.item() - result_cpu.GM.values.item())
    print(f"\n  Difference: {diff:.2f} J/mol")
    
    if diff > 100:
        print("\n⚠ SIGNIFICANT DIFFERENCE DETECTED")
        print("  This suggests the constraint handling differs between CPU and GPU")
    
    print("=" * 80)

if __name__ == "__main__":
    main()