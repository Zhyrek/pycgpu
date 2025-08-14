#!/usr/bin/env python
"""Test a specific failing condition individually vs in batch."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_condition(x_al, x_cu, temp, batch=False):
    """Test a specific condition either individually or in a small batch."""
    
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    if batch:
        # Test in a batch with 2 other conditions
        conditions = {
            v.X('AL'): [x_al, 0.1, 0.3],
            v.X('CU'): [x_cu, 0.1, 0.3],
            v.T: temp,
            v.P: 101325
        }
        test_idx = 0  # Our test condition is first
    else:
        # Test individually
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: temp,
            v.P: 101325
        }
        test_idx = None
    
    # Run CPU
    result_cpu = equilibrium(dbf, comps, phases, conditions,
                            calc_opts={'pdens': 50},
                            gpu=False, verbose=False)
    
    # Run GPU
    result_gpu = equilibrium(dbf, comps, phases, conditions,
                            calc_opts={'pdens': 50},
                            gpu=True, verbose=False)
    
    # Extract results
    if batch:
        cpu_gm = result_cpu.GM.values.flatten()[test_idx]
        gpu_gm = result_gpu.GM.values.flatten()[test_idx]
    else:
        cpu_gm = result_cpu.GM.values.item()
        gpu_gm = result_gpu.GM.values.item()
    
    return cpu_gm, gpu_gm

def main():
    # Test the failing conditions
    failing_conditions = [
        (0.50, 0.20, 600),  # X(AL)=0.50, X(CU)=0.20, T=600K
        (0.80, 0.10, 600),  # X(AL)=0.80, X(CU)=0.10, T=600K
        (0.60, 0.30, 900),  # X(AL)=0.60, X(CU)=0.30, T=900K
        (0.10, 0.50, 1200), # X(AL)=0.10, X(CU)=0.50, T=1200K
        (0.20, 0.50, 1200), # X(AL)=0.20, X(CU)=0.50, T=1200K
    ]
    
    print("=" * 80)
    print("TESTING FAILING CONDITIONS INDIVIDUALLY VS IN BATCH")
    print("=" * 80)
    
    for x_al, x_cu, temp in failing_conditions:
        x_fe = 1.0 - x_al - x_cu
        print(f"\nCondition: X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, X(FE)={x_fe:.2f}, T={temp}K")
        print("-" * 60)
        
        # Test individually
        cpu_single, gpu_single = test_condition(x_al, x_cu, temp, batch=False)
        diff_single = abs(gpu_single - cpu_single)
        
        print(f"SINGLE execution:")
        print(f"  CPU GM: {cpu_single:.6f}")
        print(f"  GPU GM: {gpu_single:.6f}")
        print(f"  Difference: {diff_single:.6f} J/mol")
        
        # Test in batch
        cpu_batch, gpu_batch = test_condition(x_al, x_cu, temp, batch=True)
        diff_batch = abs(gpu_batch - cpu_batch)
        
        print(f"BATCH execution:")
        print(f"  CPU GM: {cpu_batch:.6f}")
        print(f"  GPU GM: {gpu_batch:.6f}")
        print(f"  Difference: {diff_batch:.6f} J/mol")
        
        # Compare
        print(f"COMPARISON:")
        print(f"  CPU changes by: {abs(cpu_batch - cpu_single):.6f} J/mol")
        print(f"  GPU changes by: {abs(gpu_batch - gpu_single):.6f} J/mol")
        
        if diff_single < 1.0 and diff_batch > 1.0:
            print("  ✗ FAILS only in batch mode!")
        elif diff_single > 1.0 and diff_batch > 1.0:
            print("  ✗ Fails in both modes")
        else:
            print("  ✓ Passes in both modes")

if __name__ == "__main__":
    main()