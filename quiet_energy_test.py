#!/usr/bin/env python
"""Completely silent CPU vs GPU free energy comparison."""

import os
import sys
import numpy as np
import contextlib
import io
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Function to completely suppress output
@contextlib.contextmanager
def suppress_all_output():
    """Suppress all stdout, stderr and any other output."""
    # Save original stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        # Redirect to devnull
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        # Restore original stdout and stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

print("CPU vs GPU Free Energy Comparison")
print("=" * 50)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test condition sets
test_conditions = [
    {v.X('TI'): 0.01, v.T: 800, v.P: 101325, 'name': 'T=800K, X(TI)=0.01'},
    {v.X('TI'): 0.01, v.T: 1000, v.P: 101325, 'name': 'T=1000K, X(TI)=0.01'},
    {v.X('TI'): 0.01, v.T: 1200, v.P: 101325, 'name': 'T=1200K, X(TI)=0.01'},
    {v.X('TI'): 0.005, v.T: 1000, v.P: 101325, 'name': 'T=1000K, X(TI)=0.005'},
    {v.X('TI'): 0.02, v.T: 1000, v.P: 101325, 'name': 'T=1000K, X(TI)=0.02'},
    {v.X('TI'): 0.01, v.T: 1000, v.P: 50000, 'name': 'P=50kPa, T=1000K, X(TI)=0.01'},
    {v.X('TI'): 0.01, v.T: 1000, v.P: 200000, 'name': 'P=200kPa, T=1000K, X(TI)=0.01'},
]

results = []

print(f"{'Test Name':<35} {'CPU GM':<12} {'GPU GM':<12} {'|Diff|':<10} {'Status'}")
print("-" * 75)

for test_case in test_conditions:
    conditions = {k: v for k, v in test_case.items() if k != 'name'}
    
    try:
        # CPU calculation (completely silent)
        with suppress_all_output():
            result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        
        # GPU calculation (completely silent) 
        with suppress_all_output():
            result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        
        # Extract results
        cpu_gm = result_cpu.GM.values.flatten()[0]
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
        
        print(f"{test_case['name']:<35} {cpu_gm:<12.1f} {gpu_gm:<12.1f} {gm_diff:<10.2e} {status}")
        
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
        print(f"{test_case['name']:<35} {'ERROR':<12} {'ERROR':<12} {'N/A':<10} {'FAILED'}")
        results.append({
            'name': test_case['name'],
            'success': False,
            'error': str(e)
        })

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)

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
    
    print(f"\nStatus distribution:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    
    max_rel_diff = max(r['gm_rel_diff'] for r in successful_tests)
    avg_rel_diff = np.mean([r['gm_rel_diff'] for r in successful_tests])
    
    print(f"\nFree energy accuracy:")
    print(f"  Max relative difference: {max_rel_diff:.2e}")
    print(f"  Average relative difference: {avg_rel_diff:.2e}")

print(f"\n" + "=" * 50)
if successful_tests and all(r['gm_rel_diff'] < 1e-6 for r in successful_tests):
    print("✓ EXCELLENT - CPU and GPU agree within 1e-6 relative error")
elif successful_tests and all(r['gm_rel_diff'] < 1e-3 for r in successful_tests):
    print("✓ ACCEPTABLE - CPU and GPU agree within 1e-3 relative error")
else:
    print("⚠ DISCREPANCIES FOUND - Investigation needed")