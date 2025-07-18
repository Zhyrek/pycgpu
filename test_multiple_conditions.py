#!/usr/bin/env python
"""Test CPU vs GPU across multiple condition sets."""

import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test multiple X(TI) values
x_ti_values = [0.005, 0.01, 0.02, 0.05, 0.1]
temperatures = [900, 1000, 1100]

print("Testing CPU vs GPU across multiple condition sets...")
print("=" * 60)

total_tests = 0
passed_tests = 0
max_difference = 0.0

for T in temperatures:
    for x_ti in x_ti_values:
        conditions = {v.X('TI'): x_ti, v.T: T, v.P: 101325}
        
        try:
            # Test CPU
            result_cpu = equilibrium(dbf, comps, phases, conditions, verbose=False)
            cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
            
            # Test GPU  
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
            
            print(f"T={T}K, X(TI)={x_ti:.3f}: CPU={cpu_x_ti:.8f}, GPU={gpu_x_ti:.8f}, "
                  f"diff={difference:.2e} [{status}]")
                  
        except Exception as e:
            print(f"T={T}K, X(TI)={x_ti:.3f}: ERROR - {str(e)}")
            total_tests += 1

print("=" * 60)
print(f"Results: {passed_tests}/{total_tests} tests passed")
print(f"Maximum difference: {max_difference:.2e}")

if passed_tests == total_tests:
    print("SUCCESS: All condition sets match within numerical precision!")
else:
    print(f"FAILURE: {total_tests - passed_tests} condition sets failed")