#!/usr/bin/env python
"""Test single phase to isolate constraint issue."""

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
    
    # Test with just BCC_B2
    phases = ['BCC_B2']
    
    conditions = {
        v.X('AL'): 0.60,
        v.X('CU'): 0.10,
        v.T: 600,
        v.P: 101325
    }
    
    print("=" * 80)
    print("SINGLE PHASE TEST (BCC_B2 only)")
    print("=" * 80)
    print("\nWith only one phase, the composition should match the bulk exactly.")
    print("Target: X(AL)=0.60, X(CU)=0.10, X(FE)=0.30")
    
    # CPU
    print("\nCPU Result:")
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False)
    cpu_phases = cpu_result.Phase.values.flatten()
    cpu_np = cpu_result.NP.values.flatten()
    x_vals = cpu_result.X.values.reshape(-1, 3)
    
    for i, (phase, amount) in enumerate(zip(cpu_phases, cpu_np)):
        if not np.isnan(amount) and amount > 0.001:
            print(f"  {phase}: {amount:.4f}")
            print(f"    X(AL)={x_vals[i][0]:.4f}, X(CU)={x_vals[i][1]:.4f}, X(FE)={x_vals[i][2]:.4f}")
    
    # GPU
    print("\nGPU Result:")
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True)
    gpu_phases = gpu_result.Phase.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    x_vals = gpu_result.X.values.reshape(-1, 3)
    
    for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
        if not np.isnan(amount) and amount > 0.001:
            print(f"  {phase}: {amount:.4f}")
            print(f"    X(AL)={x_vals[i][0]:.4f}, X(CU)={x_vals[i][1]:.4f}, X(FE)={x_vals[i][2]:.4f}")
    
    print("\n" + "=" * 80)
    print("EXPLANATION:")
    print("With only one phase allowed, it must have the bulk composition.")
    print("If GPU shows different composition, the constraint is not working.")

if __name__ == "__main__":
    main()