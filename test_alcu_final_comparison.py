#!/usr/bin/env python
"""Final comparison of AlCu system with all fixes applied."""

import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = ''  # Disable debug output

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load AlCuFe database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'ALCU_ZETA']

print("Final AlCu Test with All Fixes")
print("="*70)
print("Fixes applied:")
print("1. dgelsd_device for better numerical stability")
print("2. Normalized system amount constraint (divide by moles_normalization)")
print("="*70)

# Test conditions
test_conditions = [
    ("Low T, Al-rich", 0.7, 0.2, 600),
    ("Med T, balanced", 0.6, 0.3, 900),  # Previously ~806 J/mol error
    ("High T, Cu-rich", 0.3, 0.6, 1200),
    ("Med T, more Cu", 0.4, 0.5, 900),
    ("Med T, more Al", 0.5, 0.4, 900),
]

results = []

for name, x_al, x_cu, T in test_conditions:
    print(f"\n{name}: X(AL)={x_al}, X(CU)={x_cu}, T={T}K")
    print("-" * 50)
    
    conditions = {
        v.T: T, 
        v.P: 101325, 
        v.N: 1, 
        v.X('AL'): x_al,
        v.X('CU'): x_cu
    }
    
    try:
        # CPU calculation
        cpu_result = equilibrium(dbf, comps, phases, conditions, 
                               calc_opts={'pdens': 100}, verbose=False)
        cpu_gm = float(cpu_result.GM.values.item())
        
        # GPU calculation
        gpu_result = equilibrium(dbf, comps, phases, conditions, 
                               calc_opts={'pdens': 100}, verbose=False, gpu=True)
        gpu_gm = float(gpu_result.GM.values.item())
        
        # Compare
        diff = abs(cpu_gm - gpu_gm)
        
        print(f"CPU GM: {cpu_gm:.6f} J/mol")
        print(f"GPU GM: {gpu_gm:.6f} J/mol")
        print(f"Difference: {diff:.2e} J/mol")
        
        results.append({
            'name': name,
            'diff': diff,
            'cpu_gm': cpu_gm,
            'gpu_gm': gpu_gm
        })
        
    except Exception as e:
        print(f"ERROR: {e}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if results:
    diffs = [r['diff'] for r in results]
    avg_diff = np.mean(diffs)
    max_diff = np.max(diffs)
    max_case = next(r for r in results if r['diff'] == max_diff)
    
    print(f"Average difference: {avg_diff:.2e} J/mol")
    print(f"Maximum difference: {max_diff:.2e} J/mol (at {max_case['name']})")
    
    # Compare to previous results
    print(f"\nImprovement Summary:")
    print(f"- Med T, balanced previous error: ~806 J/mol")
    med_t_result = next((r for r in results if r['name'] == "Med T, balanced"), None)
    if med_t_result:
        print(f"- Med T, balanced current error: {med_t_result['diff']:.2e} J/mol")
        if med_t_result['diff'] < 806:
            improvement = 806 / med_t_result['diff']
            print(f"- Improvement factor: {improvement:.1f}x")
    
    print(f"\nAccuracy levels:")
    below_10 = sum(1 for d in diffs if d < 10)
    below_1 = sum(1 for d in diffs if d < 1)
    below_0_1 = sum(1 for d in diffs if d < 0.1)
    
    print(f"- < 10 J/mol: {below_10}/{len(results)}")
    print(f"- < 1 J/mol: {below_1}/{len(results)}")
    print(f"- < 0.1 J/mol: {below_0_1}/{len(results)}")
    
    if max_diff < 10:
        print("\n✓ EXCELLENT! All errors < 10 J/mol")
    elif max_diff < 100:
        print("\n✓ Good! All errors < 100 J/mol")
    else:
        print(f"\n⚠️  Some errors still large (max: {max_diff:.0f} J/mol)")