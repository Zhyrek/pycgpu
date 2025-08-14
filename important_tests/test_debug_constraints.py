#!/usr/bin/env python
"""Debug constraint setup with verbose output."""

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
    phases = ['BCC_B2', 'AL5FE2']
    
    conditions = {
        v.X('AL'): 0.60,
        v.X('CU'): 0.10,
        v.T: 600,
        v.P: 101325
    }
    
    print("=" * 80)
    print("TESTING CONSTRAINT SETUP WITH DEBUG OUTPUT")
    print("=" * 80)
    
    # Run GPU with verbose to see constraint setup
    gpu_result = equilibrium(dbf, comps, phases, conditions,
                            gpu=True, verbose=True)
    
    print("\n" + "=" * 80)
    print("GPU RESULT SUMMARY")
    print("=" * 80)
    
    gpu_phases = gpu_result.Phase.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    x_vals = gpu_result.X.values.reshape(-1, 3)
    
    for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
        if not np.isnan(amount) and amount > 0.001:
            print(f"{phase}: {amount:.4f}")
            if i < len(x_vals):
                print(f"  X(AL)={x_vals[i][0]:.4f}, X(CU)={x_vals[i][1]:.4f}, X(FE)={x_vals[i][2]:.4f}")
    
    # Calculate bulk composition
    total_al = 0
    total_cu = 0
    total_fe = 0
    
    for i, amount in enumerate(gpu_np):
        if not np.isnan(amount) and amount > 0.001:
            if i < len(x_vals):
                total_al += amount * x_vals[i][0]
                total_cu += amount * x_vals[i][1]
                total_fe += amount * x_vals[i][2]
    
    print(f"\nBulk composition:")
    print(f"  X(AL) = {total_al:.6f} (target: 0.600000, error: {abs(total_al - 0.60):.6f})")
    print(f"  X(CU) = {total_cu:.6f} (target: 0.100000, error: {abs(total_cu - 0.10):.6f})")
    print(f"  X(FE) = {total_fe:.6f} (target: 0.300000, error: {abs(total_fe - 0.30):.6f})")

if __name__ == "__main__":
    main()