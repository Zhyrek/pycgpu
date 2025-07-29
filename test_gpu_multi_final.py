#!/usr/bin/env python
"""Final test to verify GPU multi-condition results match CPU."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v
import os
import sys

# Suppress debug output
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Redirect stderr temporarily
class SuppressOutput:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Test with just 3 conditions
conditions = {
    v.T: 600,
    v.P: 101325,
    v.X('TI'): [0.1, 0.5, 0.9],  # 3 specific compositions
    v.N: 1
}

print("=" * 70)
print("GPU MULTI-CONDITION TEST RESULTS")
print("=" * 70)
print(f"Testing 3 conditions at T=600K: X(TI) = [0.1, 0.5, 0.9]")

# GPU calculation
with SuppressOutput():
    gpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50}, gpu=True, verbose=False)

# CPU calculation
with SuppressOutput():
    cpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50})

# Display results
x_ti_list = [0.1, 0.5, 0.9]
print("\nCondition Results:")
print("-" * 70)
print(f"{'X(TI)':<8} {'GM (GPU)':<12} {'GM (CPU)':<12} {'ΔGM':<10} {'Result':<10}")
print("-" * 70)

all_pass = True
for i, x_ti in enumerate(x_ti_list):
    gm_gpu = gpu_result.GM.values[0, 0, 0, i]
    gm_cpu = cpu_result.GM.values[0, 0, 0, i]
    diff = abs(gm_gpu - gm_cpu)
    status = "PASS" if diff < 1.0 else "FAIL"
    if diff >= 1.0:
        all_pass = False
    print(f"{x_ti:<8.1f} {gm_gpu:<12.1f} {gm_cpu:<12.1f} {diff:<10.2f} {status:<10}")

print("-" * 70)
print(f"\nChemical Potentials:")
print("-" * 70)
print(f"{'X(TI)':<8} {'Component':<12} {'GPU':<12} {'CPU':<12} {'Δ':<10} {'Result':<10}")
print("-" * 70)

for i, x_ti in enumerate(x_ti_list):
    for j, comp in enumerate(['NB', 'TI']):
        mu_gpu = gpu_result.MU.values[0, 0, 0, i, j]
        mu_cpu = cpu_result.MU.values[0, 0, 0, i, j]
        diff = abs(mu_gpu - mu_cpu)
        status = "PASS" if diff < 1.0 else "FAIL"
        if diff >= 1.0:
            all_pass = False
        print(f"{x_ti:<8.1f} {comp:<12} {mu_gpu:<12.1f} {mu_cpu:<12.1f} {diff:<10.2f} {status:<10}")
    if i < len(x_ti_list) - 1:
        print()  # Space between conditions

print("-" * 70)
print(f"\n{'OVERALL TEST RESULT: PASS' if all_pass else 'OVERALL TEST RESULT: FAIL'}")
print("=" * 70)