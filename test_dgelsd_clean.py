#!/usr/bin/env python
"""Clean test of dgelsd implementation."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'HCP_A3']

# Test conditions
ti_fractions = np.arange(0.1, 1.0, 0.1)
temperatures = [400, 500, 600, 700, 800, 900]

results = []
max_diff = 0.0
max_diff_condition = None

print("Testing dgelsd_device implementation...")
print("="*60)

for T in temperatures:
    for x_ti in ti_fractions:
        conditions = {v.T: T, v.P: 101325, v.N: 1, v.X('TI'): x_ti}
        
        try:
            # CPU calculation
            cpu_result = equilibrium(dbf, comps, phases, conditions, 
                                   calc_opts={'pdens': 100}, verbose=False)
            cpu_gm = float(cpu_result.GM.values)
            
            # GPU calculation  
            gpu_result = equilibrium(dbf, comps, phases, conditions, 
                                   calc_opts={'pdens': 100}, verbose=False, gpu=True)
            gpu_gm = float(gpu_result.GM.values)
            
            # Compare
            diff = abs(cpu_gm - gpu_gm)
            results.append({
                'T': T,
                'X_TI': x_ti,
                'diff': diff
            })
            
            if diff > max_diff:
                max_diff = diff
                max_diff_condition = (T, x_ti)
                
        except Exception as e:
            print(f"ERROR at T={T}, X(TI)={x_ti}: {e}")

# Summary
print("\nRESULTS SUMMARY")
print("="*60)
print(f"Total test cases: {len(results)}")

if results:
    diffs = [r['diff'] for r in results]
    print(f"Average difference: {np.mean(diffs):.2e} J/mol")
    print(f"Maximum difference: {max_diff:.2e} J/mol")
    if max_diff_condition:
        print(f"  at T={max_diff_condition[0]}K, X(TI)={max_diff_condition[1]:.1f}")
    
    # Check accuracy levels
    below_1e3 = sum(1 for d in diffs if d < 1e-3)
    below_1e4 = sum(1 for d in diffs if d < 1e-4) 
    below_1e5 = sum(1 for d in diffs if d < 1e-5)
    below_1e6 = sum(1 for d in diffs if d < 1e-6)
    below_1e7 = sum(1 for d in diffs if d < 1e-7)
    below_1e8 = sum(1 for d in diffs if d < 1e-8)
    
    print(f"\nAccuracy breakdown:")
    print(f"  < 1e-3 J/mol: {below_1e3}/{len(results)} ({below_1e3/len(results)*100:.1f}%)")
    print(f"  < 1e-4 J/mol: {below_1e4}/{len(results)} ({below_1e4/len(results)*100:.1f}%)")
    print(f"  < 1e-5 J/mol: {below_1e5}/{len(results)} ({below_1e5/len(results)*100:.1f}%)")
    print(f"  < 1e-6 J/mol: {below_1e6}/{len(results)} ({below_1e6/len(results)*100:.1f}%)")
    print(f"  < 1e-7 J/mol: {below_1e7}/{len(results)} ({below_1e7/len(results)*100:.1f}%)")
    print(f"  < 1e-8 J/mol: {below_1e8}/{len(results)} ({below_1e8/len(results)*100:.1f}%)")
    
    print(f"\nComparison to previous error (~6e-4 J/mol):")
    if max_diff < 6e-4:
        improvement = 6e-4 / max_diff
        print(f"✓ IMPROVED! {improvement:.0f}x better accuracy")
        print(f"  Previous maximum error: ~6.0e-04 J/mol") 
        print(f"  New maximum error:      {max_diff:.1e} J/mol")
    else:
        print(f"✗ No improvement detected")