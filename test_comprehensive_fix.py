#!/usr/bin/env python
"""Comprehensive test of the GPU fix across the phase diagram."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test multiple compositions and temperatures
test_compositions = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
test_temperatures = [600, 800, 1000, 1200]

print("Comprehensive test of GPU vs CPU equilibrium calculations")
print("="*80)
print(f"Testing {len(test_compositions)} compositions × {len(test_temperatures)} temperatures = {len(test_compositions) * len(test_temperatures)} conditions")
print()

errors = []
max_error = 0
max_error_condition = None

for T in test_temperatures:
    for x_ti in test_compositions:
        conditions = {v.X('TI'): x_ti, v.T: T, v.P: 101325}
        
        # Run GPU calculation
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        gpu_gm = float(result_gpu.GM.values)
        
        # Run CPU calculation
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        cpu_gm = float(result_cpu.GM.values)
        
        # Calculate error
        error = abs(gpu_gm - cpu_gm)
        errors.append(error)
        
        if error > max_error:
            max_error = error
            max_error_condition = (x_ti, T)
        
        # Print results for problematic cases
        if error > 0.1:  # More than 0.1 J/mol error
            print(f"X(TI)={x_ti:.2f}, T={T}K: GPU={gpu_gm:.2f}, CPU={cpu_gm:.2f}, Error={error:.2f} J/mol")

# Summary statistics
print("\nSummary:")
print("-"*40)
print(f"Total conditions tested: {len(errors)}")
print(f"Mean absolute error: {np.mean(errors):.6f} J/mol")
print(f"Maximum error: {max_error:.6f} J/mol")
if max_error_condition:
    print(f"  at X(TI)={max_error_condition[0]:.2f}, T={max_error_condition[1]}K")
print(f"Conditions with error < 0.001 J/mol: {sum(1 for e in errors if e < 0.001)}/{len(errors)} ({100*sum(1 for e in errors if e < 0.001)/len(errors):.1f}%)")
print(f"Conditions with error < 0.01 J/mol: {sum(1 for e in errors if e < 0.01)}/{len(errors)} ({100*sum(1 for e in errors if e < 0.01)/len(errors):.1f}%)")
print(f"Conditions with error < 0.1 J/mol: {sum(1 for e in errors if e < 0.1)}/{len(errors)} ({100*sum(1 for e in errors if e < 0.1)/len(errors):.1f}%)")

if max_error < 0.1:
    print("\n✓ SUCCESS! All conditions have errors < 0.1 J/mol")
else:
    print(f"\n✗ Some conditions still have large errors (max: {max_error:.2f} J/mol)")