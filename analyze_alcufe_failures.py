#!/usr/bin/env python
"""Analyze multiple AlCuFe test failures to identify patterns."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def analyze_failure(dbf, comps, phases, x_al, x_cu, x_fe, t, test_name):
    """Analyze a single failure case."""
    
    conditions = {
        v.X('AL'): x_al,
        v.X('CU'): x_cu,
        v.T: t,
        v.P: 101325
    }
    
    print(f"\n{test_name}: X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, X(FE)={x_fe:.2f}, T={t}K")
    print("-" * 60)
    
    # Run CPU and GPU
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    
    cpu_gm = float(cpu_result.GM.values.squeeze())
    gpu_gm = float(gpu_result.GM.values.squeeze())
    diff = gpu_gm - cpu_gm
    
    print(f"CPU GM: {cpu_gm:.2f} J/mol")
    print(f"GPU GM: {gpu_gm:.2f} J/mol")
    print(f"Difference: {diff:.2f} J/mol ({abs(diff/cpu_gm)*100:.2f}%)")
    
    # Check if it's a catastrophic failure or minor divergence
    if abs(diff) > 1000:
        print("STATUS: CATASTROPHIC DIVERGENCE")
    elif abs(diff) > 100:
        print("STATUS: SIGNIFICANT DIVERGENCE")
    elif abs(diff) > 10:
        print("STATUS: MODERATE DIVERGENCE")
    else:
        print("STATUS: MINOR DIVERGENCE")
    
    # Get phase information
    cpu_phases = []
    gpu_phases = []
    
    try:
        phase_data = cpu_result.Phase.values
        unique_phases = np.unique(phase_data[phase_data != ''])
        if len(unique_phases) > 0:
            cpu_phases = list(unique_phases)
    except:
        pass
        
    try:
        phase_data = gpu_result.Phase.values
        unique_phases = np.unique(phase_data[phase_data != ''])
        if len(unique_phases) > 0:
            gpu_phases = list(unique_phases)
    except:
        pass
    
    if cpu_phases or gpu_phases:
        print(f"CPU phases: {cpu_phases if cpu_phases else 'None'}")
        print(f"GPU phases: {gpu_phases if gpu_phases else 'None'}")
    
    return diff

def test_neighborhood(dbf, comps, phases, x_al_center, x_cu_center, t):
    """Test points around a failing composition."""
    
    print(f"\nNeighborhood test around X(AL)={x_al_center:.2f}, X(CU)={x_cu_center:.2f}, T={t}K")
    print("=" * 60)
    
    deltas = [-0.01, 0, 0.01]
    results = []
    
    for dal in deltas:
        for dcu in deltas:
            x_al = x_al_center + dal
            x_cu = x_cu_center + dcu
            x_fe = 1 - x_al - x_cu
            
            if x_al < 0 or x_cu < 0 or x_fe < 0 or x_al > 1 or x_cu > 1 or x_fe > 1:
                continue
            
            conditions = {
                v.X('AL'): x_al,
                v.X('CU'): x_cu,
                v.T: t,
                v.P: 101325
            }
            
            cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
            gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
            
            cpu_gm = float(cpu_result.GM.values.squeeze())
            gpu_gm = float(gpu_result.GM.values.squeeze())
            diff = gpu_gm - cpu_gm
            
            status = "MATCH" if abs(diff) < 1.0 else "DIFF"
            marker = "*" if dal == 0 and dcu == 0 else " "
            
            results.append((x_al, x_cu, cpu_gm, gpu_gm, diff, status, marker))
    
    # Print results in grid format
    print("X(AL)  X(CU)   CPU GM      GPU GM      Diff     Status")
    for x_al, x_cu, cpu_gm, gpu_gm, diff, status, marker in results:
        print(f"{x_al:.2f}   {x_cu:.2f}  {cpu_gm:10.2f}  {gpu_gm:10.2f}  {diff:8.2f}  {status:5s} {marker}")
    
    return results

def main():
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12_FCC', 
              'ALCU_THETA', 'AL13FE4_D03', 'AL5FE2_D82']
    
    print("=" * 80)
    print("ANALYSIS OF ALCUFE 8-PHASE TEST FAILURES")
    print("=" * 80)
    
    # Select representative failures from the data
    # Format: (X(AL), X(CU), X(FE), T, expected_diff, test_name)
    test_cases = [
        # T=600K cases (many catastrophic failures)
        (0.8, 0.1, 0.1, 600, 12523.3, "Test #2 (T=600K)"),
        (0.5, 0.4, 0.1, 600, 21414.5, "Test #13 (T=600K)"),
        (0.4, 0.4, 0.2, 600, 17846.1, "Test #17 (T=600K)"),
        
        # T=800K cases (mixed failures)
        (0.8, 0.1, 0.1, 800, 297.6, "Test #18 (T=800K)"),
        (0.5, 0.4, 0.1, 800, 886.4, "Test #30 (T=800K)"),
        
        # T=1000K cases
        (0.8, 0.1, 0.1, 1000, 4588.6, "Test #34 (T=1000K)"),
        (0.5, 0.4, 0.1, 1000, 132.8, "Test #46 (T=1000K)"),
    ]
    
    catastrophic = []
    significant = []
    moderate = []
    minor = []
    
    for x_al, x_cu, x_fe, t, expected_diff, test_name in test_cases:
        diff = analyze_failure(dbf, comps, phases, x_al, x_cu, x_fe, t, test_name)
        
        if abs(diff) > 1000:
            catastrophic.append((test_name, diff))
        elif abs(diff) > 100:
            significant.append((test_name, diff))
        elif abs(diff) > 10:
            moderate.append((test_name, diff))
        else:
            minor.append((test_name, diff))
    
    # Detailed analysis of one catastrophic case
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS OF CATASTROPHIC FAILURE")
    print("=" * 80)
    
    # Test #2: X(AL)=0.8, X(CU)=0.1, X(FE)=0.1, T=600K
    test_neighborhood(dbf, comps, phases, 0.8, 0.1, 600)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF FAILURE CATEGORIES")
    print("=" * 80)
    
    print(f"\nCatastrophic (>1000 J/mol): {len(catastrophic)} cases")
    for name, diff in catastrophic:
        print(f"  {name}: {diff:.1f} J/mol")
    
    print(f"\nSignificant (100-1000 J/mol): {len(significant)} cases")
    for name, diff in significant:
        print(f"  {name}: {diff:.1f} J/mol")
    
    print(f"\nModerate (10-100 J/mol): {len(moderate)} cases")
    for name, diff in moderate:
        print(f"  {name}: {diff:.1f} J/mol")
    
    print(f"\nMinor (<10 J/mol): {len(minor)} cases")
    
    print("\n" + "=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)
    print("""
    1. T=600K shows many catastrophic failures (>10,000 J/mol differences)
       - These appear to be cases where GPU fails to converge properly
       - GPU often returns a fixed value (-16689.08 J/mol) suggesting solver failure
    
    2. T=800K shows mixed behavior:
       - Some minor divergences (<1000 J/mol) - likely different local minima
       - Some catastrophic failures with very large differences
    
    3. T=1000K shows mostly significant but not catastrophic divergences
       - Differences in 100-5000 J/mol range
       - Suggests different phase selections at high temperature
    
    4. The fixed GPU value of -16689.08 J/mol appears repeatedly in failures
       - This may be a default/fallback value when GPU solver fails
       - Need to investigate GPU solver convergence criteria
    """)

if __name__ == "__main__":
    main()