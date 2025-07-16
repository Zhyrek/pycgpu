#!/usr/bin/env python
"""Broad test to verify GPU matches CPU across diverse conditions."""

from pycalphad import Database, equilibrium
import numpy as np
import time

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Create a broad but manageable range of test conditions
temperatures = [500, 1000, 1500, 2000, 2500]  # 5 temperatures
ti_fractions = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]  # 10 compositions

print("="*80)
print("BROAD GPU-CPU EQUILIBRIUM CALCULATION COMPARISON")
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
failed_conditions = []

# Progress tracking
start_time = time.time()

for T in temperatures:
    print(f"\nT={T}K: ", end='', flush=True)
    
    for x_ti in ti_fractions:
        total_tests += 1
        
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
                print(".", end='', flush=True)
            else:
                failed_tests += 1
                print("F", end='', flush=True)
                failed_conditions.append({
                    'T': T, 
                    'X(TI)': x_ti,
                    'cpu_gm': cpu_gm,
                    'gpu_gm': gpu_gm,
                    'diff': diff
                })
                
            if diff > max_diff:
                max_diff = diff
                max_diff_conditions = conditions.copy()
                
        except Exception as e:
            failed_tests += 1
            print("E", end='', flush=True)
            failed_conditions.append({
                'T': T, 
                'X(TI)': x_ti,
                'error': str(e)
            })
    
    print(f" ({passed_tests}/{total_tests} passed so far)")

# Calculate statistics
elapsed_time = time.time() - start_time
all_differences = np.array(all_differences)

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"Total conditions tested: {total_tests}")
print(f"Passed: {passed_tests} ({100*passed_tests/total_tests:.1f}%)")
print(f"Failed: {failed_tests} ({100*failed_tests/total_tests:.1f}%)")
print(f"Time elapsed: {elapsed_time:.1f} seconds ({total_tests/elapsed_time:.1f} tests/sec)")

if len(all_differences) > 0:
    print(f"\nDifference statistics:")
    print(f"  Mean: {np.mean(all_differences):.2e} J/mol")
    print(f"  Max:  {max_diff:.2e} J/mol") 
    print(f"  Min:  {np.min(all_differences):.2e} J/mol")
    
    # Check how many are exactly zero
    exactly_zero = np.sum(all_differences == 0.0)
    print(f"  Exactly zero: {exactly_zero} ({100*exactly_zero/len(all_differences):.1f}%)")
    
    # Check how many are within machine precision
    machine_precision = 1e-12
    within_precision = np.sum(all_differences < machine_precision)
    print(f"  Within machine precision (<1e-12): {within_precision} ({100*within_precision/len(all_differences):.1f}%)")

# Show failed conditions if any
if failed_conditions:
    print("\n" + "="*80)
    print("FAILED CONDITIONS")
    print("="*80)
    for fc in failed_conditions[:10]:  # Show first 10
        if 'error' in fc:
            print(f"T={fc['T']}K, X(TI)={fc['X(TI)']:.3f}: ERROR - {fc['error']}")
        else:
            print(f"T={fc['T']}K, X(TI)={fc['X(TI)']:.3f}: CPU={fc['cpu_gm']:.6f}, GPU={fc['gpu_gm']:.6f}, Diff={fc['diff']:.2e}")
    if len(failed_conditions) > 10:
        print(f"... and {len(failed_conditions)-10} more")

print("\n" + "="*80)
if failed_tests == 0:
    print("✓ SUCCESS: GPU and CPU produce identical results across all tested conditions!")
else:
    print(f"✗ FAILURE: {failed_tests} conditions showed differences > {tolerance:.0e} J/mol")