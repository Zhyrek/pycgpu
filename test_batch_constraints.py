#!/usr/bin/env python
"""Test GPU constraint handling with batch processing of multiple condition sets."""

import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("Testing GPU constraint handling: Batch vs Individual processing")
print("=" * 65)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']  
phases = filter_phases(dbf, comps)

# Test multiple constraint values
x_ti_values = [0.005, 0.010, 0.020]

print("\n1. INDIVIDUAL GPU TESTS (separate equilibrium calls)")
print("-" * 50)
individual_results = []
for x_ti in x_ti_values:
    conditions = {v.X('TI'): x_ti, v.T: 1000, v.P: 101325}
    result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    x_ti_final = result.X.sel(component='TI').values.flatten()[0]
    individual_results.append(x_ti_final)
    
    error = abs(x_ti_final - x_ti)
    status = "✓" if error < 1e-10 else "✗"
    print(f"X(TI) = {x_ti:.3f} → GPU result = {x_ti_final:.8f} (error = {error:.2e}) {status}")

print("\n2. BATCH GPU TEST (single equilibrium call with arrays)")
print("-" * 50)
# Create batch conditions
conditions_batch = {
    v.X('TI'): x_ti_values,  # Array of constraint values
    v.T: 1000, 
    v.P: 101325
}

try:
    result_batch = equilibrium(dbf, comps, phases, conditions_batch, gpu=True, verbose=False)
    batch_results = []
    
    for i, x_ti_target in enumerate(x_ti_values):
        x_ti_final = result_batch.X.sel(component='TI').values.flatten()[i]
        batch_results.append(x_ti_final)
        
        error = abs(x_ti_final - x_ti_target)
        status = "✓" if error < 1e-10 else "✗"
        print(f"X(TI) = {x_ti_target:.3f} → GPU result = {x_ti_final:.8f} (error = {error:.2e}) {status}")
        
except Exception as e:
    print(f"Batch processing failed: {e}")
    batch_results = [0.0] * len(x_ti_values)

print("\n3. CPU REFERENCE (batch processing)")
print("-" * 40)
result_cpu_batch = equilibrium(dbf, comps, phases, conditions_batch, gpu=False, verbose=False)
cpu_results = []

for i, x_ti_target in enumerate(x_ti_values):
    x_ti_final = result_cpu_batch.X.sel(component='TI').values.flatten()[i]
    cpu_results.append(x_ti_final)
    
    error = abs(x_ti_final - x_ti_target)
    status = "✓" if error < 1e-10 else "✗"
    print(f"X(TI) = {x_ti_target:.3f} → CPU result = {x_ti_final:.8f} (error = {error:.2e}) {status}")

print("\n4. COMPARISON SUMMARY")
print("-" * 40)
print(f"{'Target':<8} {'Individual':<12} {'Batch GPU':<12} {'Batch CPU':<12} {'Status'}")
print("-" * 60)

all_passed = True
for i, x_ti_target in enumerate(x_ti_values):
    ind_error = abs(individual_results[i] - x_ti_target)
    batch_error = abs(batch_results[i] - x_ti_target) 
    cpu_error = abs(cpu_results[i] - x_ti_target)
    
    ind_ok = ind_error < 1e-10
    batch_ok = batch_error < 1e-10
    cpu_ok = cpu_error < 1e-10
    
    if ind_ok and batch_ok and cpu_ok:
        status = "✓ PASS"
    else:
        status = "✗ FAIL"
        all_passed = False
    
    print(f"{x_ti_target:<8.3f} {individual_results[i]:<12.8f} {batch_results[i]:<12.8f} {cpu_results[i]:<12.8f} {status}")

print("\n" + "=" * 65)
if all_passed:
    print("✓ ALL TESTS PASSED - GPU constraint handling works for both individual and batch processing")
else:
    print("✗ SOME TESTS FAILED - GPU constraint handling has issues")
    
    # Detailed diagnosis
    print("\nDIAGNOSIS:")
    individual_ok = all(abs(individual_results[i] - x_ti_values[i]) < 1e-10 for i in range(len(x_ti_values)))
    batch_ok = all(abs(batch_results[i] - x_ti_values[i]) < 1e-10 for i in range(len(x_ti_values)))
    
    if individual_ok and not batch_ok:
        print("- Individual GPU processing works correctly")
        print("- Batch GPU processing has issues (data isolation problem)")
    elif not individual_ok and not batch_ok:
        print("- Both individual and batch GPU processing have constraint issues")
    elif not individual_ok and batch_ok:
        print("- Individual GPU processing has issues but batch works (unusual)")
    
    print("\nISSUES TO FIX:")
    if not individual_ok:
        print("- GPU constraint handling needs fixing for individual calls")
    if not batch_ok:
        print("- GPU batch processing needs fixing for multiple condition sets")