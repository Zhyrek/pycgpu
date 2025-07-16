#!/usr/bin/env python
"""Comprehensive test to verify GPU matches CPU across a broad range of conditions."""

from pycalphad import Database, equilibrium
import numpy as np
import time

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Create a broad range of test conditions
temperatures = [300, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000]
ti_fractions = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]

print("="*80)
print("COMPREHENSIVE GPU-CPU EQUILIBRIUM CALCULATION COMPARISON")
print("="*80)
print(f"Testing {len(temperatures)} temperatures × {len(ti_fractions)} compositions = {len(temperatures) * len(ti_fractions)} conditions")
print("="*80)

# Track results
total_tests = 0
passed_tests = 0
failed_tests = 0
max_diff = 0.0
max_diff_conditions = None
tolerance = 1e-6  # 1 microjoule/mol tolerance

# Store all differences for statistical analysis
all_differences = []

# Progress tracking
start_time = time.time()

for T in temperatures:
    print(f"\nTesting T={T}K...")
    row_passed = 0
    row_failed = 0
    
    for x_ti in ti_fractions:
        total_tests += 1
        
        # Skip unphysical conditions
        if x_ti >= 0.999:  # Leave some NB
            x_ti = 0.999
            
        conditions = {
            'T': T,
            'P': 101325,
            'X(TI)': x_ti
        }
        
        try:
            # Run CPU calculation
            eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
            cpu_gm = eq_cpu.GM.values.item()
            
            # Run GPU calculation
            eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
            gpu_gm = eq_gpu.GM.values.item()
            
            # Compare results
            diff = abs(cpu_gm - gpu_gm)
            all_differences.append(diff)
            
            if diff <= tolerance:
                passed_tests += 1
                row_passed += 1
            else:
                failed_tests += 1
                row_failed += 1
                print(f"  FAIL: X(TI)={x_ti:.3f} - CPU: {cpu_gm:.6f}, GPU: {gpu_gm:.6f}, Diff: {diff:.2e} J/mol")
                
            if diff > max_diff:
                max_diff = diff
                max_diff_conditions = conditions.copy()
                
        except Exception as e:
            failed_tests += 1
            row_failed += 1
            print(f"  ERROR: X(TI)={x_ti:.3f} - {str(e)}")
    
    print(f"  Summary: {row_passed} passed, {row_failed} failed")

# Calculate statistics
elapsed_time = time.time() - start_time
all_differences = np.array(all_differences)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Total conditions tested: {total_tests}")
print(f"Passed: {passed_tests} ({100*passed_tests/total_tests:.1f}%)")
print(f"Failed: {failed_tests} ({100*failed_tests/total_tests:.1f}%)")
print(f"Time elapsed: {elapsed_time:.1f} seconds ({total_tests/elapsed_time:.1f} tests/sec)")

print("\n" + "="*80)
print("DIFFERENCE STATISTICS")
print("="*80)
if len(all_differences) > 0:
    print(f"Mean difference: {np.mean(all_differences):.2e} J/mol")
    print(f"Median difference: {np.median(all_differences):.2e} J/mol")
    print(f"Max difference: {max_diff:.2e} J/mol")
    print(f"Min difference: {np.min(all_differences):.2e} J/mol")
    print(f"Std deviation: {np.std(all_differences):.2e} J/mol")
    
    # Check how many are within machine precision
    machine_precision = 1e-12
    within_precision = np.sum(all_differences < machine_precision)
    print(f"\nWithin machine precision (<1e-12): {within_precision} ({100*within_precision/len(all_differences):.1f}%)")
    
    # Percentiles
    print(f"\nPercentiles:")
    for p in [50, 90, 95, 99, 99.9, 100]:
        val = np.percentile(all_differences, p)
        print(f"  {p}%: {val:.2e} J/mol")

if max_diff_conditions:
    print(f"\nWorst case conditions: T={max_diff_conditions['T']}K, X(TI)={max_diff_conditions['X(TI)']:.3f}")
    print(f"Maximum difference: {max_diff:.2e} J/mol")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
if failed_tests == 0:
    print("✓ SUCCESS: GPU and CPU produce identical results across all tested conditions!")
    print(f"  All {total_tests} tests passed with differences < {tolerance:.0e} J/mol")
else:
    print(f"✗ FAILURE: {failed_tests} conditions showed differences > {tolerance:.0e} J/mol")
    print("  Further investigation needed for failed cases")