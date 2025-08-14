#!/usr/bin/env python
"""Test component indexing in GPU vs CPU."""

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
    phases = ['BCC_B2', 'AL5FE2']  # Just the 2 phases that should be stable
    
    conditions = {
        v.X('AL'): 0.60,
        v.X('CU'): 0.10,
        v.T: 600,
        v.P: 101325
    }
    
    print("=" * 80)
    print("COMPONENT INDEXING TEST")
    print("=" * 80)
    
    print("\nComponents list:", comps)
    print("Component indices:")
    for i, comp in enumerate(comps):
        print(f"  {i}: {comp}")
    
    print("\nNon-vacant components:")
    nonvacant = [c for c in comps if c != 'VA']
    for i, comp in enumerate(nonvacant):
        print(f"  {i}: {comp}")
    
    print("\nConditions:")
    for key, val in conditions.items():
        if isinstance(key, v.MoleFraction):
            comp_name = str(key)[2:]
            comp_idx_full = comps.index(comp_name)
            comp_idx_nonva = nonvacant.index(comp_name) if comp_name in nonvacant else -1
            print(f"  {key} = {val}")
            print(f"    Index in full components list: {comp_idx_full}")
            print(f"    Index in non-vacant list: {comp_idx_nonva}")
    
    print("\n" + "=" * 80)
    print("EXPECTED CONSTRAINT MATRIX")
    print("=" * 80)
    
    print("\nFor ternary system with VA, the constraint matrix should be:")
    print("  Constraint 0 (X(AL)=0.60): coefficients = [1.0, 0.0, 0.0, 0.0], rhs = 0.60")
    print("  Constraint 1 (X(CU)=0.10): coefficients = [0.0, 1.0, 0.0, 0.0], rhs = 0.10")
    print("\nNote: Coefficients array has size 4 (all components including VA)")
    print("But coefficient for VA should always be 0 since VA doesn't participate in mass balance")
    
    print("\n" + "=" * 80)
    print("TESTING GPU CALCULATION")
    print("=" * 80)
    
    # Run GPU calculation with verbose to see what's happening
    gpu_result = equilibrium(dbf, comps, phases, conditions,
                            gpu=True, verbose=False)
    
    # Check result
    gpu_phases = gpu_result.Phase.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    x_vals = gpu_result.X.values
    
    print("\nGPU Result:")
    for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
        if not np.isnan(amount) and amount > 0.001:
            print(f"  {phase}: {amount:.4f}")
            if i < len(x_vals.reshape(-1, 3)):
                x_reshaped = x_vals.reshape(-1, 3)
                print(f"    X(AL)={x_reshaped[i][0]:.4f}, X(CU)={x_reshaped[i][1]:.4f}, X(FE)={x_reshaped[i][2]:.4f}")
    
    # Calculate bulk composition
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
    
    print("\nCalculated bulk composition:")
    print(f"  X(AL) = {total_al:.6f} (target: 0.600000, error: {abs(total_al - 0.60):.6f})")
    print(f"  X(CU) = {total_cu:.6f} (target: 0.100000, error: {abs(total_cu - 0.10):.6f})")
    print(f"  X(FE) = {total_fe:.6f} (target: 0.300000, error: {abs(total_fe - 0.30):.6f})")
    
    if abs(total_al - 0.60) > 0.001 or abs(total_cu - 0.10) > 0.001:
        print("\n⚠️ MASS CONSERVATION VIOLATED!")
    else:
        print("\n✓ Mass conservation satisfied")

if __name__ == "__main__":
    main()