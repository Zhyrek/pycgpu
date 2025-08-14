#!/usr/bin/env python
"""Trace solver iterations to find where divergence occurs."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    """Trace where solver diverges."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    print("=" * 80)
    print("SOLVER ITERATION TRACE")
    print("=" * 80)
    
    # The exact failing condition
    conditions = {
        v.X('AL'): 0.2,
        v.X('CU'): 0.5,
        v.T: 900,
        v.P: 101325
    }
    
    print("\nCondition: X(AL)=0.2, X(CU)=0.5, X(FE)=0.3, T=900K")
    
    # Test with BCC_B2 included
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2']
    
    print("\nPhases: LIQUID, FCC_A1, BCC_A2, BCC_B2")
    print("\nBCC_B2 properties:")
    print("  - 3 sublattices: (0.5, 0.5, 3.0) sites")
    print("  - Total sites: 4.0")
    print("  - Fractional site ratios: 0.125, 0.125, 0.75")
    
    # Check what happens at the boundary compositions
    print("\n" + "-" * 80)
    print("Testing numerical boundary at X(AL)=0.2, X(CU)=0.5:")
    print("-" * 80)
    
    # These specific values might trigger edge cases
    print("\nNumerical properties of the composition:")
    print(f"  X(AL) = 0.2 = 1/5 = 0.200000...")
    print(f"  X(CU) = 0.5 = 1/2 = 0.500000...")
    print(f"  X(FE) = 0.3 = 3/10 = 0.300000...")
    
    print("\nWith BCC_B2 sublattice fractions:")
    print("  If all AL goes to sublattice 1 (0.5 sites): 0.2/0.125 = 1.6 (impossible)")
    print("  If all CU goes to sublattice 2 (0.5 sites): 0.5/0.125 = 4.0 (impossible)")
    print("  This forces mixing across sublattices")
    
    print("\nPossible numerical issue:")
    print("  At X(CU)=0.5 exactly, some calculation might divide by (1-2*X(CU))")
    print("  This would give division by zero at X(CU)=0.5")
    
    # Test nearby compositions to check for singularity
    print("\n" + "-" * 80)
    print("Testing for numerical singularity around X(CU)=0.5:")
    print("-" * 80)
    
    test_cu_values = [0.48, 0.49, 0.495, 0.499, 0.4999, 0.5, 0.5001, 0.501, 0.505, 0.51, 0.52]
    
    print("\nX(CU)   | CPU GM    | GPU GM    | Diff    | Status")
    print("--------|-----------|-----------|---------|--------")
    
    for x_cu in test_cu_values:
        conditions = {
            v.X('AL'): 0.2,
            v.X('CU'): x_cu,
            v.T: 900,
            v.P: 101325
        }
        
        try:
            cpu_result = equilibrium(dbf, comps, phases, conditions,
                                    calc_opts={'pdens': 50},
                                    gpu=False, verbose=False)
            cpu_gm = cpu_result.GM.values.item()
            
            gpu_result = equilibrium(dbf, comps, phases, conditions,
                                    calc_opts={'pdens': 50},
                                    gpu=True, verbose=False)
            gpu_gm = gpu_result.GM.values.item()
            
            diff = abs(gpu_gm - cpu_gm)
            status = "✓" if diff < 100 else "✗"
            
            print(f"{x_cu:7.4f} | {cpu_gm:9.1f} | {gpu_gm:9.1f} | {diff:7.1f} | {status}")
        except Exception as e:
            print(f"{x_cu:7.4f} | ERROR: {str(e)[:40]}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("If divergence occurs ONLY at X(CU)=0.5000 exactly, this suggests")
    print("a division by zero or numerical singularity in the GPU code")
    print("when handling BCC_B2's sublattice site fractions.")
    print("=" * 80)

if __name__ == "__main__":
    main()