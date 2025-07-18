#!/usr/bin/env python
"""Quick test of a few condition sets with simplified output."""

import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Redirect stdout/stderr temporarily to suppress debug output
class SuppressOutput:
    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self.stdout
        sys.stderr = self.stderr

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test a few key conditions
test_conditions = [
    (1000, 0.005),
    (1000, 0.01), 
    (1000, 0.02),
    (1100, 0.01),
]

print("Testing CPU vs GPU across multiple condition sets...")
print("=" * 60)

total_tests = 0
passed_tests = 0
max_difference = 0.0

for T, x_ti in test_conditions:
    try:
        conditions = {v.X('TI'): x_ti, v.T: T, v.P: 101325}
        
        # Test CPU (suppress output)
        with SuppressOutput():
            result_cpu = equilibrium(dbf, comps, phases, conditions, verbose=False)
        cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
        
        # Test GPU (suppress output)
        with SuppressOutput():
            result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)
        gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]
        
        difference = abs(cpu_x_ti - gpu_x_ti)
        max_difference = max(max_difference, difference)
        
        total_tests += 1
        if difference < 1e-6:
            passed_tests += 1
            status = "PASS"
        else:
            status = "FAIL"
        
        print(f"T={T:4d}K, X(TI)={x_ti:.3f}: CPU={cpu_x_ti:.8f}, GPU={gpu_x_ti:.8f}, "
              f"diff={difference:.2e} [{status}]")
              
    except Exception as e:
        print(f"T={T:4d}K, X(TI)={x_ti:.3f}: ERROR - {str(e)}")
        total_tests += 1

print("=" * 60)
print(f"Results: {passed_tests}/{total_tests} tests passed")
print(f"Maximum difference: {max_difference:.2e}")

if passed_tests == total_tests:
    print("SUCCESS: All condition sets match within numerical precision!")
else:
    print(f"FAILURE: {total_tests - passed_tests} condition sets failed")