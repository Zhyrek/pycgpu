#!/usr/bin/env python
"""Compare CPU and GPU free energy values across multiple condition sets."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("CPU vs GPU Free Energy Comparison")
print("=" * 50)

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
    {v.X('TI'): 0.01, v.T: 1000, v.P: 101325, 'name': 'T=1000K, X(TI)=0.01'},
    {v.X('TI'): 0.02, v.T: 1000, v.P: 101325, 'name': 'T=1000K, X(TI)=0.02'},
    {v.X('TI'): 0.05, v.T: 1000, v.P: 101325, 'name': 'T=1000K, X(TI)=0.05'},
    
    # Pressure variations
    {v.X('TI'): 0.01, v.T: 1000, v.P: 50000, 'name': 'P=50kPa, T=1000K, X(TI)=0.01'},
    {v.X('TI'): 0.01, v.T: 1000, v.P: 101325, 'name': 'P=101kPa, T=1000K, X(TI)=0.01'},
    {v.X('TI'): 0.01, v.T: 1000, v.P: 200000, 'name': 'P=200kPa, T=1000K, X(TI)=0.01'},
]

results = []

for i, test_case in enumerate(test_conditions):
    print(f"\n{i+1}. Testing: {test_case['name']}")
    print("-" * 40)
    
    # Remove 'name' key for equilibrium calculation
    conditions = {k: v for k, v in test_case.items() if k != 'name'}
    
    try:
        # CPU calculation
        print("  Running CPU calculation...")
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        
        # GPU calculation
        print("  Running GPU calculation...")
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        
        # Extract results
        cpu_gm = result_cpu.GM.values.flatten()[0]  # Total Gibbs energy
        gpu_gm = result_gpu.GM.values.flatten()[0]
        
        cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
        gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]
        
        # Calculate differences
        gm_diff = abs(cpu_gm - gpu_gm)
        gm_rel_diff = gm_diff / abs(cpu_gm) if abs(cpu_gm) > 1e-10 else 0
        x_ti_diff = abs(cpu_x_ti - gpu_x_ti)
        
        print(f"  CPU GM = {cpu_gm:.6f} J/mol")
        print(f"  GPU GM = {gpu_gm:.6f} J/mol")
        print(f"  |Difference| = {gm_diff:.2e} J/mol")
        print(f"  Relative diff = {gm_rel_diff:.2e}")
        print(f"  X(TI) CPU = {cpu_x_ti:.8f}")
        print(f"  X(TI) GPU = {gpu_x_ti:.8f}")
        print(f"  X(TI) diff = {x_ti_diff:.2e}")
        
        # Store results
        results.append({
            'name': test_case['name'],
            'conditions': conditions,
            'cpu_gm': cpu_gm,
            'gpu_gm': gpu_gm,
            'gm_diff': gm_diff,
            'gm_rel_diff': gm_rel_diff,
            'cpu_x_ti': cpu_x_ti,
            'gpu_x_ti': gpu_x_ti,
            'x_ti_diff': x_ti_diff,
            'success': True
        })
        
        # Status check
        if gm_rel_diff < 1e-10 and x_ti_diff < 1e-10:
            print("  ✓ EXCELLENT agreement")
        elif gm_rel_diff < 1e-6 and x_ti_diff < 1e-6:
            print("  ✓ GOOD agreement")
        elif gm_rel_diff < 1e-3 and x_ti_diff < 1e-3:
            print("  ⚠ ACCEPTABLE agreement")
        else:
            print("  ✗ POOR agreement - needs investigation")
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        results.append({
            'name': test_case['name'],
            'conditions': conditions,
            'success': False,
            'error': str(e)
        })

print("\n" + "=" * 50)
print("SUMMARY REPORT")
print("=" * 50)

successful_tests = [r for r in results if r['success']]
failed_tests = [r for r in results if not r['success']]

print(f"Total tests: {len(results)}")
print(f"Successful: {len(successful_tests)}")
print(f"Failed: {len(failed_tests)}")

if successful_tests:
    print(f"\nFREE ENERGY COMPARISON RESULTS:")
    print(f"{'Test Name':<35} {'GM Rel Diff':<12} {'X(TI) Diff':<12} {'Status'}")
    print("-" * 75)
    
    excellent_count = 0
    good_count = 0
    acceptable_count = 0
    poor_count = 0
    
    for result in successful_tests:
        gm_rel_diff = result['gm_rel_diff']
        x_ti_diff = result['x_ti_diff']
        
        if gm_rel_diff < 1e-10 and x_ti_diff < 1e-10:
            status = "EXCELLENT"
            excellent_count += 1
        elif gm_rel_diff < 1e-6 and x_ti_diff < 1e-6:
            status = "GOOD"
            good_count += 1
        elif gm_rel_diff < 1e-3 and x_ti_diff < 1e-3:
            status = "ACCEPTABLE"
            acceptable_count += 1
        else:
            status = "POOR"
            poor_count += 1
        
        print(f"{result['name']:<35} {gm_rel_diff:<12.2e} {x_ti_diff:<12.2e} {status}")
    
    print(f"\nSTATISTICS:")
    print(f"  Excellent (< 1e-10 rel): {excellent_count}")
    print(f"  Good (< 1e-6 rel):       {good_count}")
    print(f"  Acceptable (< 1e-3 rel): {acceptable_count}")
    print(f"  Poor (>= 1e-3 rel):      {poor_count}")
    
    if successful_tests:
        max_gm_diff = max(r['gm_rel_diff'] for r in successful_tests)
        max_x_diff = max(r['x_ti_diff'] for r in successful_tests)
        avg_gm_diff = np.mean([r['gm_rel_diff'] for r in successful_tests])
        avg_x_diff = np.mean([r['x_ti_diff'] for r in successful_tests])
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Max GM relative difference: {max_gm_diff:.2e}")
        print(f"  Average GM relative difference: {avg_gm_diff:.2e}")
        print(f"  Max X(TI) difference: {max_x_diff:.2e}")
        print(f"  Average X(TI) difference: {avg_x_diff:.2e}")

if failed_tests:
    print(f"\nFAILED TESTS:")
    for result in failed_tests:
        print(f"  {result['name']}: {result['error']}")

print(f"\n" + "=" * 50)
if successful_tests and all(r['gm_rel_diff'] < 1e-6 and r['x_ti_diff'] < 1e-6 for r in successful_tests):
    print("✓ ALL TESTS PASSED - CPU and GPU show excellent agreement")
elif successful_tests and all(r['gm_rel_diff'] < 1e-3 and r['x_ti_diff'] < 1e-3 for r in successful_tests):
    print("✓ ALL TESTS ACCEPTABLE - CPU and GPU show reasonable agreement")
else:
    print("⚠ SOME TESTS SHOW DISCREPANCIES - Investigation needed")