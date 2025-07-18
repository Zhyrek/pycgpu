#!/usr/bin/env python
"""CPU vs GPU free energy comparison with actual values and absolute errors."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("CPU vs GPU Free Energy Values - Absolute Error Analysis")
print("=" * 70)

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

print(f"{'Condition':<35} {'CPU GM (J/mol)':<18} {'GPU GM (J/mol)':<18} {'Absolute Error':<15}")
print("-" * 90)

for test_case in test_conditions:
    conditions = {k: v for k, v in test_case.items() if k != 'name'}
    
    try:
        # CPU calculation
        import warnings
        import contextlib
        import io
        
        # Suppress all possible output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        
        # GPU calculation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        
        # Extract results
        cpu_gm = result_cpu.GM.values.flatten()[0]
        gpu_gm = result_gpu.GM.values.flatten()[0]
        
        # Calculate absolute error
        abs_error = abs(cpu_gm - gpu_gm)
        
        # Format and display
        print(f"{test_case['name']:<35} {cpu_gm:<18.6f} {gpu_gm:<18.6f} {abs_error:<15.6f}")
        
        results.append({
            'name': test_case['name'],
            'cpu_gm': cpu_gm,
            'gpu_gm': gpu_gm,
            'abs_error': abs_error,
            'success': True
        })
        
    except Exception as e:
        print(f"{test_case['name']:<35} {'ERROR':<18} {'ERROR':<18} {'N/A':<15}")
        results.append({
            'name': test_case['name'],
            'success': False,
            'error': str(e)
        })

print("\n" + "=" * 70)
print("DETAILED ANALYSIS")
print("=" * 70)

successful_tests = [r for r in results if r['success']]

if successful_tests:
    print(f"\nActual Free Energy Values:")
    print(f"{'Condition':<35} {'CPU (J/mol)':<15} {'GPU (J/mol)':<15} {'Difference':<15}")
    print("-" * 85)
    
    for result in successful_tests:
        cpu_val = result['cpu_gm']
        gpu_val = result['gpu_gm']
        diff = gpu_val - cpu_val  # Signed difference (GPU - CPU)
        
        print(f"{result['name']:<35} {cpu_val:<15.3f} {gpu_val:<15.3f} {diff:<+15.3f}")
    
    print(f"\nAbsolute Error Statistics:")
    abs_errors = [r['abs_error'] for r in successful_tests]
    print(f"  Minimum absolute error: {min(abs_errors):.6f} J/mol")
    print(f"  Maximum absolute error: {max(abs_errors):.6f} J/mol") 
    print(f"  Average absolute error: {np.mean(abs_errors):.6f} J/mol")
    print(f"  Standard deviation:     {np.std(abs_errors):.6f} J/mol")
    
    print(f"\nError Classification:")
    small_errors = [e for e in abs_errors if e < 0.1]
    medium_errors = [e for e in abs_errors if 0.1 <= e < 1.0]
    large_errors = [e for e in abs_errors if e >= 1.0]
    
    print(f"  < 0.1 J/mol:   {len(small_errors)} tests")
    print(f"  0.1-1.0 J/mol: {len(medium_errors)} tests") 
    print(f"  ≥ 1.0 J/mol:   {len(large_errors)} tests")
    
    if large_errors:
        print(f"\nLarge errors (≥ 1.0 J/mol) found in:")
        for result in successful_tests:
            if result['abs_error'] >= 1.0:
                print(f"  {result['name']}: {result['abs_error']:.3f} J/mol")

print(f"\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if successful_tests:
    max_error = max(r['abs_error'] for r in successful_tests)
    avg_error = np.mean([r['abs_error'] for r in successful_tests])
    
    print(f"CPU and GPU free energy calculations compared across {len(successful_tests)} conditions:")
    print(f"• Maximum absolute difference: {max_error:.6f} J/mol")
    print(f"• Average absolute difference: {avg_error:.6f} J/mol")
    
    if max_error < 0.1:
        print("✓ EXCELLENT agreement - all differences < 0.1 J/mol")
    elif max_error < 1.0:
        print("✓ GOOD agreement - all differences < 1.0 J/mol")
    elif max_error < 10.0:
        print("✓ ACCEPTABLE agreement - all differences < 10.0 J/mol")
    else:
        print("⚠ SIGNIFICANT discrepancies detected - investigation needed")
        
    # Typical thermodynamic context
    typical_gm_magnitude = np.mean([abs(r['cpu_gm']) for r in successful_tests])
    relative_error_percent = (avg_error / typical_gm_magnitude) * 100
    print(f"• Relative to typical |GM| ≈ {typical_gm_magnitude:.0f} J/mol: {relative_error_percent:.4f}%")

print(f"\n" + "=" * 70)