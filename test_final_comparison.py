#!/usr/bin/env python
"""Final test of GPU vs CPU with clean output."""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Completely suppress debug output
os.environ['PYCALPHAD_DEBUG'] = '0'

from pycalphad import Database, equilibrium
import numpy as np

# Redirect all output during calculations
class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
        
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

# Load TDB
with SuppressOutput():
    tdb = Database('NbTi.tdb')
    
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Test a good range of conditions
print("Testing GPU vs CPU across multiple conditions...")
print("="*50)

results = []
temperatures = [500, 1000, 1500, 2000, 2500]
ti_fractions = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99]

for T in temperatures:
    for x_ti in ti_fractions:
        conditions = {'T': T, 'P': 101325, 'X(TI)': x_ti}
        
        with SuppressOutput():
            try:
                # CPU calculation
                eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
                cpu_gm = float(eq_cpu.GM.values.item())
                
                # GPU calculation
                eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
                gpu_gm = float(eq_gpu.GM.values.item())
                
                diff = abs(cpu_gm - gpu_gm)
                results.append((T, x_ti, cpu_gm, gpu_gm, diff))
                
            except Exception as e:
                results.append((T, x_ti, None, None, None))

# Display results
passed = 0
failed = 0
tolerance = 1e-6

print(f"{'T(K)':>6} {'X(TI)':>6} {'CPU(J/mol)':>12} {'GPU(J/mol)':>12} {'Diff':>10} {'Status':>8}")
print("-"*60)

for T, x_ti, cpu_gm, gpu_gm, diff in results:
    if cpu_gm is None:
        print(f"{T:6d} {x_ti:6.3f} {'ERROR':>12} {'ERROR':>12} {'N/A':>10} {'FAIL':>8}")
        failed += 1
    else:
        status = "PASS" if diff < tolerance else "FAIL"
        if diff < tolerance:
            passed += 1
        else:
            failed += 1
        print(f"{T:6d} {x_ti:6.3f} {cpu_gm:12.1f} {gpu_gm:12.1f} {diff:10.1e} {status:>8}")

print("="*60)
print(f"Summary: {passed} passed, {failed} failed out of {len(results)} tests")
print(f"Success rate: {100*passed/len(results):.1f}%")

# Find max difference
valid_diffs = [d for _, _, _, _, d in results if d is not None]
if valid_diffs:
    print(f"Maximum difference: {max(valid_diffs):.2e} J/mol")

if failed == 0:
    print("\n✓ ALL TESTS PASSED - GPU matches CPU within tolerance!")
else:
    print(f"\n✗ {failed} tests failed")