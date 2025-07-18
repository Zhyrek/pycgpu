#!/usr/bin/env python
"""Test CPU vs GPU across multiple condition sets with suppressed debug output."""

import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import subprocess

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test multiple X(TI) values at different temperatures
test_conditions = [
    (900, 0.005),
    (900, 0.01), 
    (900, 0.02),
    (1000, 0.005),
    (1000, 0.01),
    (1000, 0.02),
    (1100, 0.005),
    (1100, 0.01),
    (1100, 0.02),
]

print("Testing CPU vs GPU across multiple condition sets...")
print("=" * 70)

total_tests = 0
passed_tests = 0
max_difference = 0.0

for T, x_ti in test_conditions:
    try:
        # Run CPU test in subprocess to avoid debug output
        cpu_script = f'''
import sys
sys.path.insert(0, ".")
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
dbf = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = filter_phases(dbf, comps)
conditions = {{v.X("TI"): {x_ti}, v.T: {T}, v.P: 101325}}
result = equilibrium(dbf, comps, phases, conditions, verbose=False)
print(f"CPU:{{result.X.sel(component='TI').values.flatten()[0]:.12f}}")
'''
        
        cpu_proc = subprocess.run([sys.executable, '-c', cpu_script], 
                                 capture_output=True, text=True)
        cpu_output = cpu_proc.stdout.strip()
        cpu_x_ti = float(cpu_output.split(':')[1])
        
        # Run GPU test in subprocess  
        gpu_script = f'''
import sys
sys.path.insert(0, ".")
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
dbf = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = filter_phases(dbf, comps)
conditions = {{v.X("TI"): {x_ti}, v.T: {T}, v.P: 101325}}
result = equilibrium(dbf, comps, phases, conditions, gpu=True)
print(f"GPU:{{result.X.sel(component='TI').values.flatten()[0]:.12f}}")
'''
        
        gpu_proc = subprocess.run([sys.executable, '-c', gpu_script], 
                                 capture_output=True, text=True)
        gpu_output = gpu_proc.stdout.strip()
        # Extract just the final GPU result line
        gpu_lines = [line for line in gpu_output.split('\n') if line.startswith('GPU:')]
        if gpu_lines:
            gpu_x_ti = float(gpu_lines[-1].split(':')[1])
        else:
            raise ValueError("No GPU result found")
        
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
        print(f"T={T:4d}K, X(TI)={x_ti:.3f}: ERROR - {str(e)[:50]}...")
        total_tests += 1

print("=" * 70)
print(f"Results: {passed_tests}/{total_tests} tests passed")
print(f"Maximum difference: {max_difference:.2e}")

if passed_tests == total_tests:
    print("SUCCESS: All condition sets match within numerical precision!")
else:
    print(f"FAILURE: {total_tests - passed_tests} condition sets failed")