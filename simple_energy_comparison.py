#!/usr/bin/env python
"""Simple CPU vs GPU free energy comparison - final results only."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("CPU vs GPU Free Energy Comparison (Final Results Only)")
print("=" * 60)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test condition sets
test_conditions = [
    # Temperature variations
    {v.X('TI'): 0.01, v.T: 800, v.P: 101325, 'name': 'T=800K, X(TI)=0.01'},
    {v.X('TI'): 0.01, v.T: 1000, v.P: 101325, 'name': 'T=1000K, X(TI)=0.01'},
    {v.X('TI'): 0.01, v.T: 1200, v.P: 101325, 'name': 'T=1200K, X(TI)=0.01'},
    
    # Composition variations at 1000K
    {v.X('TI'): 0.005, v.T: 1000, v.P: 101325, 'name': 'T=1000K, X(TI)=0.005'},
    {v.X('TI'): 0.02, v.T: 1000, v.P: 101325, 'name': 'T=1000K, X(TI)=0.02'},
    
    # Pressure variations
    {v.X('TI'): 0.01, v.T: 1000, v.P: 50000, 'name': 'P=50kPa, T=1000K, X(TI)=0.01'},
    {v.X('TI'): 0.01, v.T: 1000, v.P: 200000, 'name': 'P=200kPa, T=1000K, X(TI)=0.01'},
]

results = []

print(f"{'Test Name':<35} {'CPU GM (J/mol)':<15} {'GPU GM (J/mol)':<15} {'|Diff|':<12} {'Status'}")
print("-" * 95)

for i, test_case in enumerate(test_conditions):
    # Remove 'name' key for equilibrium calculation
    conditions = {k: v for k, v in test_case.items() if k != 'name'}
    
    try:
        # CPU calculation (silent)
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        
        # GPU calculation (silent)
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        
        # Extract results
        cpu_gm = result_cpu.GM.values.flatten()[0]  # Total Gibbs energy
        gpu_gm = result_gpu.GM.values.flatten()[0]
        
        # Calculate differences
        gm_diff = abs(cpu_gm - gpu_gm)
        gm_rel_diff = gm_diff / abs(cpu_gm) if abs(cpu_gm) > 1e-10 else 0
        
        # Status check
        if gm_rel_diff < 1e-10:
            status = "EXCELLENT"
        elif gm_rel_diff < 1e-6:
            status = "GOOD"
        elif gm_rel_diff < 1e-3:
            status = "ACCEPTABLE"
        else:
            status = "POOR"
        
        print(f"{test_case['name']:<35} {cpu_gm:<15.3f} {gpu_gm:<15.3f} {gm_diff:<12.2e} {status}")
        
        # Store results
        results.append({
            'name': test_case['name'],
            'cpu_gm': cpu_gm,
            'gpu_gm': gpu_gm,
            'gm_diff': gm_diff,
            'gm_rel_diff': gm_rel_diff,
            'success': True,
            'status': status
        })
        
    except Exception as e:
        print(f"{test_case['name']:<35} {'ERROR':<15} {'ERROR':<15} {'N/A':<12} {'FAILED'}")
        results.append({
            'name': test_case['name'],
            'success': False,
            'error': str(e)
        })

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

successful_tests = [r for r in results if r['success']]
failed_tests = [r for r in results if not r['success']]

print(f"Total tests: {len(results)}")
print(f"Successful: {len(successful_tests)}")
print(f"Failed: {len(failed_tests)}")

if successful_tests:
    status_counts = {}
    for result in successful_tests:
        status = result['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\nStatus Distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    max_gm_diff = max(r['gm_rel_diff'] for r in successful_tests)
    avg_gm_diff = np.mean([r['gm_rel_diff'] for r in successful_tests])
    
    print(f"\nFree Energy Differences:")
    print(f"  Max relative difference: {max_gm_diff:.2e}")
    print(f"  Average relative difference: {avg_gm_diff:.2e}")

if failed_tests:
    print(f"\nFAILED TESTS:")
    for result in failed_tests:
        print(f"  {result['name']}: {result['error']}")

print(f"\n" + "=" * 60)
if successful_tests and all(r['gm_rel_diff'] < 1e-6 for r in successful_tests):
    print("✓ ALL TESTS PASSED - CPU and GPU show excellent agreement")
elif successful_tests and all(r['gm_rel_diff'] < 1e-3 for r in successful_tests):
    print("✓ ALL TESTS ACCEPTABLE - CPU and GPU show reasonable agreement")
else:
    print("⚠ SOME TESTS SHOW DISCREPANCIES - Investigation needed")