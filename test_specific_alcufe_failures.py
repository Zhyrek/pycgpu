#!/usr/bin/env python
"""Test the specific compositions that showed large failures in the data file."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_composition(dbf, comps, phases, x_cu, x_fe, x_al, t, expected_cpu_gm, expected_gpu_gm):
    """Test a specific composition and compare with expected values."""
    
    conditions = {
        v.X('AL'): x_al,
        v.X('CU'): x_cu,
        v.T: t,
        v.P: 101325
    }
    
    # Run CPU and GPU
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    
    cpu_gm = float(cpu_result.GM.values.squeeze())
    gpu_gm = float(gpu_result.GM.values.squeeze())
    diff = gpu_gm - cpu_gm
    
    # Compare with expected values from file
    cpu_match = abs(cpu_gm - expected_cpu_gm) < 1.0
    gpu_match = abs(gpu_gm - expected_gpu_gm) < 1.0
    
    print(f"X(AL)={x_al:.1f} X(CU)={x_cu:.1f} X(FE)={x_fe:.1f} T={t:4d}K | "
          f"CPU: {cpu_gm:10.2f} (exp: {expected_cpu_gm:10.2f}) {'✓' if cpu_match else '✗'} | "
          f"GPU: {gpu_gm:10.2f} (exp: {expected_gpu_gm:10.2f}) {'✓' if gpu_match else '✗'} | "
          f"Δ={diff:8.2f}")
    
    return diff, cpu_match, gpu_match

def main():
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12_FCC', 
              'ALCU_THETA', 'AL13FE4_D03', 'AL5FE2_D82']
    
    print("=" * 80)
    print("TESTING SPECIFIC ALCUFE FAILURES FROM DATA FILE")
    print("=" * 80)
    print("\nComparing current results with values from gpu_cpu_alcufe_all_phases_results_multi.txt")
    print("exp = expected value from file, ✓ = matches, ✗ = different\n")
    
    # Test cases from the file showing large failures
    # Format: (X(CU), X(FE), X(AL), T, expected_CPU_GM, expected_GPU_GM)
    test_cases = [
        # T=600K catastrophic failures (GPU returned -16689.08)
        (0.1, 0.1, 0.8, 600, -29212.38, -16689.08),  # Line 2: 12523 J/mol diff
        (0.1, 0.2, 0.7, 600, -33333.37, -16689.08),  # Line 3: 16644 J/mol diff
        (0.1, 0.3, 0.6, 600, -36370.07, -16689.08),  # Line 4: 19681 J/mol diff
        (0.4, 0.1, 0.5, 600, -38103.61, -16689.08),  # Line 14: 21415 J/mol diff
        (0.4, 0.2, 0.4, 600, -39049.95, -16689.08),  # Line 15: 22361 J/mol diff
        
        # T=800K mixed results
        (0.1, 0.1, 0.8, 800, -41331.36, -41628.93),  # Line 18: 298 J/mol diff
        (0.1, 0.2, 0.7, 800, -45767.72, -28539.20),  # Line 19: 17229 J/mol diff (catastrophic)
        (0.4, 0.1, 0.5, 800, -49790.56, -48904.15),  # Line 30: 886 J/mol diff
        
        # T=1000K failures
        (0.1, 0.1, 0.8, 1000, -56143.45, -60732.05), # Line 34: 4589 J/mol diff
        (0.1, 0.4, 0.5, 1000, -63210.22, -41934.74), # Line 37: 21275 J/mol diff (catastrophic)
        (0.4, 0.1, 0.5, 1000, -63709.65, -63842.42), # Line 46: 133 J/mol diff
    ]
    
    print("Testing compositions that showed large failures:")
    print("-" * 80)
    
    fixed_gpu_value_count = 0
    catastrophic_count = 0
    
    for x_cu, x_fe, x_al, t, exp_cpu, exp_gpu in test_cases:
        diff, cpu_match, gpu_match = test_composition(dbf, comps, phases, 
                                                      x_cu, x_fe, x_al, t, 
                                                      exp_cpu, exp_gpu)
        
        # Check for the suspicious fixed GPU value
        if abs(exp_gpu - (-16689.08)) < 1.0:
            fixed_gpu_value_count += 1
        
        if abs(diff) > 1000:
            catastrophic_count += 1
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    print(f"\nFixed GPU value (-16689.08) appeared in {fixed_gpu_value_count} cases from the file")
    print(f"Current catastrophic divergences (>1000 J/mol): {catastrophic_count}")
    
    print("\nKey observations:")
    print("1. The file data shows GPU returning -16689.08 J/mol for many T=600K cases")
    print("2. This value appears to be a solver failure or default value")
    print("3. Current runs may show different behavior if solver has been updated")
    print("4. Some failures at T=800K and T=1000K also show GPU value of -28539.20 or -41934.74")
    print("   These may also be default/failure values")
    
    # Test if these are actual fixed values
    print("\n" + "=" * 80)
    print("INVESTIGATING SUSPICIOUS FIXED VALUES")
    print("=" * 80)
    
    suspicious_values = [-16689.08, -28539.20, -41934.74]
    
    print("\nTesting if these values appear consistently for failed convergence:")
    for val in suspicious_values:
        print(f"\nValue {val:.2f} J/mol:")
        count = 0
        # Check multiple random compositions
        for i in range(5):
            x_al = 0.3 + i * 0.1
            x_cu = 0.3
            x_fe = 1 - x_al - x_cu
            
            if x_fe < 0 or x_fe > 1:
                continue
                
            conditions = {
                v.X('AL'): x_al,
                v.X('CU'): x_cu,
                v.T: 600,
                v.P: 101325
            }
            
            gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
            gpu_gm = float(gpu_result.GM.values.squeeze())
            
            if abs(gpu_gm - val) < 1.0:
                count += 1
                print(f"  Found at X(AL)={x_al:.1f}, X(CU)={x_cu:.1f}")
        
        if count == 0:
            print(f"  Not found in current tests")

if __name__ == "__main__":
    main()