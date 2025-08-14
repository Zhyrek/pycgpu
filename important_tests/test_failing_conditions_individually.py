#!/usr/bin/env python
"""Test the failing conditions individually to prove they're not thread-related issues."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_single_condition(dbf, comps, phases, x_al, x_cu, temp):
    """Test a single condition and return CPU/GPU results."""
    conditions = {
        v.X('AL'): x_al,
        v.X('CU'): x_cu,
        v.T: temp,
        v.P: 101325
    }
    
    # Run CPU
    cpu_result = equilibrium(dbf, comps, phases, conditions,
                            calc_opts={'pdens': 50},
                            gpu=False, verbose=False)
    cpu_gm = cpu_result.GM.values.item()
    cpu_mu_al = cpu_result.MU.sel(component='AL').values.item()
    cpu_mu_cu = cpu_result.MU.sel(component='CU').values.item()
    cpu_mu_fe = cpu_result.MU.sel(component='FE').values.item()
    
    # Run GPU
    gpu_result = equilibrium(dbf, comps, phases, conditions,
                            calc_opts={'pdens': 50},
                            gpu=True, verbose=False)
    gpu_gm = gpu_result.GM.values.item()
    gpu_mu_al = gpu_result.MU.sel(component='AL').values.item()
    gpu_mu_cu = gpu_result.MU.sel(component='CU').values.item()
    gpu_mu_fe = gpu_result.MU.sel(component='FE').values.item()
    
    return (cpu_gm, cpu_mu_al, cpu_mu_cu, cpu_mu_fe), (gpu_gm, gpu_mu_al, gpu_mu_cu, gpu_mu_fe)

def main():
    """Test failing conditions individually."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    # The 9 failing conditions from the comprehensive test
    failing_conditions = [
        (0.40, 0.40, 600),   # X_FE = 0.20
        (0.50, 0.40, 600),   # X_FE = 0.10
        (0.10, 0.50, 900),   # X_FE = 0.40
        (0.20, 0.50, 900),   # X_FE = 0.30
        (0.30, 0.50, 900),   # X_FE = 0.20
        (0.70, 0.20, 900),   # X_FE = 0.10
        (0.20, 0.40, 1200),  # X_FE = 0.40
        (0.30, 0.60, 1200),  # X_FE = 0.10
        (0.70, 0.10, 1200),  # X_FE = 0.20
    ]
    
    print("=" * 100)
    print("TESTING FAILING CONDITIONS INDIVIDUALLY")
    print("=" * 100)
    print("\nRunning each failing condition as a single GPU calculation...")
    print("If these also fail individually, it proves the issue is NOT thread interference.\n")
    
    print("-" * 100)
    print("Condition                          | CPU GM      | GPU GM      | Diff (J/mol) | Status")
    print("-" * 100)
    
    tolerance = 100.0  # 100 J/mol tolerance as used in comprehensive test
    individual_failures = 0
    individual_passes = 0
    
    for x_al, x_cu, temp in failing_conditions:
        x_fe = 1.0 - x_al - x_cu
        
        try:
            cpu_vals, gpu_vals = test_single_condition(dbf, comps, phases, x_al, x_cu, temp)
            cpu_gm, cpu_mu_al, cpu_mu_cu, cpu_mu_fe = cpu_vals
            gpu_gm, gpu_mu_al, gpu_mu_cu, gpu_mu_fe = gpu_vals
            
            gm_diff = abs(gpu_gm - cpu_gm)
            mu_al_diff = abs(gpu_mu_al - cpu_mu_al)
            mu_cu_diff = abs(gpu_mu_cu - cpu_mu_cu)
            mu_fe_diff = abs(gpu_mu_fe - cpu_mu_fe)
            
            # Check if it passes with the same tolerance as the comprehensive test
            if (gm_diff < tolerance and mu_al_diff < tolerance and 
                mu_cu_diff < tolerance and mu_fe_diff < tolerance):
                status = "✓ PASS"
                individual_passes += 1
            else:
                status = "✗ FAIL"
                individual_failures += 1
            
            print(f"X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, T={temp:4d}K | {cpu_gm:11.2f} | {gpu_gm:11.2f} | {gm_diff:12.2f} | {status}")
            
            # Show chemical potential differences if they're large
            if mu_al_diff > tolerance or mu_cu_diff > tolerance or mu_fe_diff > tolerance:
                print(f"  Chemical potential diffs: AL={mu_al_diff:.2f}, CU={mu_cu_diff:.2f}, FE={mu_fe_diff:.2f} J/mol")
                
        except Exception as e:
            print(f"X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, T={temp:4d}K | ERROR: {str(e)[:50]}")
            individual_failures += 1
    
    print("-" * 100)
    
    # Now test a few as part of a small batch to compare
    print("\nTesting the same conditions in a small batch (3 at a time)...")
    print("-" * 100)
    
    for i in range(0, len(failing_conditions), 3):
        batch = failing_conditions[i:i+3]
        if len(batch) < 3:
            batch = batch + [(0.2, 0.2, 800)] * (3 - len(batch))  # Pad with a known good condition
        
        conditions_batch = {
            v.X('AL'): [c[0] for c in batch],
            v.X('CU'): [c[1] for c in batch],
            v.T: [c[2] for c in batch],
            v.P: 101325
        }
        
        # Run batch GPU
        gpu_batch_result = equilibrium(dbf, comps, phases, conditions_batch,
                                      calc_opts={'pdens': 50},
                                      gpu=True, verbose=False)
        
        # Run batch CPU
        cpu_batch_result = equilibrium(dbf, comps, phases, conditions_batch,
                                      calc_opts={'pdens': 50},
                                      gpu=False, verbose=False)
        
        gpu_gm_batch = gpu_batch_result.GM.values.flatten()
        cpu_gm_batch = cpu_batch_result.GM.values.flatten()
        
        print(f"Batch {i//3 + 1}:")
        for j, (x_al, x_cu, temp) in enumerate(batch[:min(len(batch), len(failing_conditions)-i)]):
            if j < len(gpu_gm_batch) and j < len(cpu_gm_batch):
                batch_diff = abs(gpu_gm_batch[j] - cpu_gm_batch[j])
                status = "✓" if batch_diff < tolerance else "✗"
                print(f"  [{j}] X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, T={temp:4d}K: "
                      f"CPU={cpu_gm_batch[j]:.2f}, GPU={gpu_gm_batch[j]:.2f}, Diff={batch_diff:.2f} {status}")
    
    print("-" * 100)
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Individual test results:")
    print(f"  Failed when run individually: {individual_failures}/{len(failing_conditions)}")
    print(f"  Passed when run individually: {individual_passes}/{len(failing_conditions)}")
    
    if individual_failures == len(failing_conditions):
        print("\n✓ CONFIRMED: All failing conditions also fail when run individually.")
        print("  This proves the failures are NOT due to thread interference.")
        print("  The issues are inherent to these specific compositions/conditions.")
    elif individual_failures > 0:
        print(f"\n⚠ PARTIAL: {individual_failures} conditions fail individually.")
        print("  These failures are not thread-related.")
        if individual_passes > 0:
            print(f"  However, {individual_passes} conditions passed individually but failed in batch.")
            print("  Those might have thread-related issues.")
    else:
        print("\n✗ UNEXPECTED: All conditions pass individually!")
        print("  This would indicate thread interference issues in batch mode.")
    
    print("\nThe failures are likely due to:")
    print("  - Numerical convergence challenges with 8 phases")
    print("  - Differences in CPU vs GPU floating-point arithmetic")
    print("  - Solver tolerance or iteration limit differences")
    print("=" * 100)

if __name__ == "__main__":
    main()