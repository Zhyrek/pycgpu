#!/usr/bin/env python
"""Silent broad test to verify GPU matches CPU."""

import os
import sys
# Suppress all output during calculations
os.environ['PYCALPHAD_DEBUG'] = '0'

# Redirect stderr to devnull during imports
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from pycalphad import Database, equilibrium
import numpy as np
import time

# Restore stderr
sys.stderr = stderr

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Test conditions
temperatures = [500, 1000, 1500, 2000, 2500]
ti_fractions = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]

print("Testing", len(temperatures) * len(ti_fractions), "conditions...")

# Track results
total_tests = 0
passed_tests = 0
max_diff = 0.0
tolerance = 1e-6

# Suppress output during calculations
original_stdout = sys.stdout
original_stderr = sys.stderr

for T in temperatures:
    for x_ti in ti_fractions:
        total_tests += 1
        
        conditions = {
            'T': T,
            'P': 101325,
            'X(TI)': x_ti
        }
        
        try:
            # Suppress output
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            
            # Run calculations
            eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
            cpu_gm = eq_cpu.GM.values.item()
            
            eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
            gpu_gm = eq_gpu.GM.values.item()
            
            # Restore output
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            # Compare
            diff = abs(cpu_gm - gpu_gm)
            
            if diff <= tolerance:
                passed_tests += 1
            else:
                print(f"FAIL: T={T}K, X(TI)={x_ti:.3f}, Diff={diff:.2e} J/mol")
                
            if diff > max_diff:
                max_diff = diff
                
        except Exception as e:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            print(f"ERROR: T={T}K, X(TI)={x_ti:.3f}: {str(e)}")

# Results
print(f"\nTotal: {total_tests}, Passed: {passed_tests}, Failed: {total_tests-passed_tests}")
print(f"Success rate: {100*passed_tests/total_tests:.1f}%")
print(f"Max difference: {max_diff:.2e} J/mol")

if passed_tests == total_tests:
    print("\n✓ SUCCESS: GPU matches CPU perfectly!")
else:
    print(f"\n✗ {total_tests-passed_tests} conditions failed")