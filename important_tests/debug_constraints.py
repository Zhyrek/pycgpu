#!/usr/bin/env python
"""Debug constraint setup for the failing condition."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    conditions = {
        v.X('AL'): 0.60,
        v.X('CU'): 0.10,
        v.T: 600,
        v.P: 101325
    }
    
    print("=" * 80)
    print("DEBUGGING CONSTRAINT SETUP")
    print("=" * 80)
    print("\nConditions:")
    print(f"  X(AL) = 0.60 (prescribed)")
    print(f"  X(CU) = 0.10 (prescribed)")
    print(f"  X(FE) = 0.30 (calculated as 1 - 0.60 - 0.10)")
    print(f"  T = 600 K")
    print(f"  P = 101325 Pa")
    
    print("\nComponents: AL, CU, FE, VA")
    print("  VA is excluded from mass balance")
    print("  So we have 3 components in mass balance")
    
    print("\nExpected constraints:")
    print("  1. X(AL) = 0.60")
    print("  2. X(CU) = 0.10")
    print("  Note: X(FE) is not explicitly constrained (it's dependent)")
    
    print("\nExpected equilibrium matrix structure:")
    print("  Rows for phase energy equations (one per active phase)")
    print("  Row for X(AL) constraint")
    print("  Row for X(CU) constraint")
    print("  Row for system amount constraint (sum of phases = 1)")
    
    print("\n" + "=" * 80)
    print("RUNNING GPU CALCULATION WITH VERBOSE OUTPUT")
    print("=" * 80)
    
    # Run with verbose to see constraint details
    gpu_result = equilibrium(dbf, comps, phases, conditions,
                            gpu=True, verbose=True)
    
    print("\n" + "=" * 80)
    print("GPU RESULT")
    print("=" * 80)
    
    # Check result
    gpu_phases = gpu_result.Phase.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    
    print("\nActive phases:")
    for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
        if not np.isnan(amount) and amount > 0.001:
            print(f"  {phase}: {amount:.4f}")
    
    # Calculate overall composition
    x_vals = gpu_result.X.values
    np_vals = gpu_result.NP.values.flatten()
    
    total_al = 0
    total_cu = 0
    total_fe = 0
    
    for i, amount in enumerate(np_vals):
        if not np.isnan(amount) and amount > 0.001:
            if i < len(x_vals.reshape(-1, 3)):
                x_reshaped = x_vals.reshape(-1, 3)
                total_al += amount * x_reshaped[i][0]
                total_cu += amount * x_reshaped[i][1]
                total_fe += amount * x_reshaped[i][2]
    
    print("\nCalculated bulk composition from phases:")
    print(f"  X(AL) = {total_al:.6f} (target: 0.600000, error: {abs(total_al - 0.60):.6f})")
    print(f"  X(CU) = {total_cu:.6f} (target: 0.100000, error: {abs(total_cu - 0.10):.6f})")
    print(f"  X(FE) = {total_fe:.6f} (target: 0.300000, error: {abs(total_fe - 0.30):.6f})")
    
    if abs(total_al - 0.60) > 0.001 or abs(total_cu - 0.10) > 0.001:
        print("\n⚠️ MASS CONSERVATION VIOLATED!")
        print("The GPU is not properly enforcing the composition constraints.")

if __name__ == "__main__":
    main()