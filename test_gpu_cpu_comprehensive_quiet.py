#!/usr/bin/env python
"""Run comprehensive GPU/CPU comparison test with dgelsd implementation."""

import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = ''  # Disable debug output

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

print("Running comprehensive GPU/CPU comparison with dgelsd...")
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
                'CPU_GM': cpu_gm,
                'GPU_GM': gpu_gm,
                'diff': diff
            })
            
            if diff > max_diff:
                max_diff = diff
                max_diff_condition = (T, x_ti)
                
        except Exception as e:
            print(f"ERROR at T={T}, X(TI)={x_ti}: {e}")

# Summary
print("\nTEST SUMMARY")
print("="*60)
print(f"Total test cases: {len(results)}")

if results:
    diffs = [r['diff'] for r in results]
    print(f"Average difference: {np.mean(diffs):.10f} J/mol")
    print(f"Maximum difference: {max_diff:.10f} J/mol")
    if max_diff_condition:
        print(f"  at T={max_diff_condition[0]}K, X(TI)={max_diff_condition[1]:.1f}")
    
    # Check how many are below old threshold
    below_001 = sum(1 for d in diffs if d < 0.001)
    below_0001 = sum(1 for d in diffs if d < 0.0001)
    below_00001 = sum(1 for d in diffs if d < 0.00001)
    
    print(f"\nAccuracy breakdown:")
    print(f"  < 0.001 J/mol:   {below_001}/{len(results)} ({below_001/len(results)*100:.1f}%)")
    print(f"  < 0.0001 J/mol:  {below_0001}/{len(results)} ({below_0001/len(results)*100:.1f}%)")
    print(f"  < 0.00001 J/mol: {below_00001}/{len(results)} ({below_00001/len(results)*100:.1f}%)")
    
    if max_diff < 0.0006:
        print(f"\n✓ EXCELLENT! Maximum error improved from ~0.0006 to {max_diff:.10f} J/mol")
        print(f"  That's a {0.0006/max_diff:.0f}x improvement!")
    else:
        print(f"\n✗ Maximum error still {max_diff:.6f} J/mol")