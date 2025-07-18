#!/usr/bin/env python
"""Comprehensive proof that GPU converges correctly to X(TI)=0.9"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("PROOF OF GPU CONVERGENCE TO X(TI)=0.9")
print("="*60)

# Run CPU calculation
print("\n1. CPU CALCULATION:")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
cpu_x_nb = result_cpu.X.sel(component='NB').values.flatten()[0]
cpu_gm = result_cpu.GM.values.flatten()[0]
cpu_phases = result_cpu.Phase.values.flatten()
cpu_np = result_cpu.NP.values.flatten()

print(f"   X(TI) = {cpu_x_ti:.15f}")
print(f"   X(NB) = {cpu_x_nb:.15f}")
print(f"   GM = {cpu_gm:.6f} J/mol")
print(f"   Phases present: {[p for p, np in zip(cpu_phases, cpu_np) if np > 1e-6]}")
print(f"   Phase amounts: {[f'{np:.6f}' for p, np in zip(cpu_phases, cpu_np) if np > 1e-6]}")

# Run GPU calculation
print("\n2. GPU CALCULATION:")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]
gpu_x_nb = result_gpu.X.sel(component='NB').values.flatten()[0]
gpu_gm = result_gpu.GM.values.flatten()[0]
gpu_phases = result_gpu.Phase.values.flatten()
gpu_np = result_gpu.NP.values.flatten()

print(f"   X(TI) = {gpu_x_ti:.15f}")
print(f"   X(NB) = {gpu_x_nb:.15f}")
print(f"   GM = {gpu_gm:.6f} J/mol")
print(f"   Phases present: {[p for p, np in zip(gpu_phases, gpu_np) if np > 1e-6]}")
print(f"   Phase amounts: {[f'{np:.6f}' for p, np in zip(gpu_phases, gpu_np) if np > 1e-6]}")

# Detailed comparison
print("\n3. DETAILED COMPARISON:")
print(f"   X(TI) difference: {abs(gpu_x_ti - cpu_x_ti):.15e}")
print(f"   X(NB) difference: {abs(gpu_x_nb - cpu_x_nb):.15e}")
print(f"   GM difference: {abs(gpu_gm - cpu_gm):.15e} J/mol")
print(f"   X(TI) + X(NB) for CPU: {cpu_x_ti + cpu_x_nb:.15f}")
print(f"   X(TI) + X(NB) for GPU: {gpu_x_ti + gpu_x_nb:.15f}")

# Convergence criteria
print("\n4. CONVERGENCE CRITERIA:")
x_ti_target = 0.9
cpu_error = abs(cpu_x_ti - x_ti_target)
gpu_error = abs(gpu_x_ti - x_ti_target)

print(f"   Target X(TI): {x_ti_target:.15f}")
print(f"   CPU error from target: {cpu_error:.15e}")
print(f"   GPU error from target: {gpu_error:.15e}")
print(f"   CPU/GPU agreement: {abs(gpu_x_ti - cpu_x_ti):.15e}")

# Pass/Fail criteria
tolerance = 1e-6
print(f"\n5. PASS/FAIL CRITERIA (tolerance = {tolerance}):")

tests_passed = 0
tests_total = 0

# Test 1: GPU converges to target
tests_total += 1
if gpu_error < tolerance:
    print(f"   ✓ TEST 1 PASSED: GPU X(TI) within {tolerance} of target")
    print(f"     GPU: {gpu_x_ti:.15f}, Target: {x_ti_target:.15f}, Error: {gpu_error:.15e}")
    tests_passed += 1
else:
    print(f"   ✗ TEST 1 FAILED: GPU X(TI) not within {tolerance} of target")
    print(f"     GPU: {gpu_x_ti:.15f}, Target: {x_ti_target:.15f}, Error: {gpu_error:.15e}")

# Test 2: GPU agrees with CPU
tests_total += 1
if abs(gpu_x_ti - cpu_x_ti) < tolerance:
    print(f"   ✓ TEST 2 PASSED: GPU agrees with CPU within {tolerance}")
    print(f"     GPU: {gpu_x_ti:.15f}, CPU: {cpu_x_ti:.15f}, Diff: {abs(gpu_x_ti - cpu_x_ti):.15e}")
    tests_passed += 1
else:
    print(f"   ✗ TEST 2 FAILED: GPU does not agree with CPU within {tolerance}")
    print(f"     GPU: {gpu_x_ti:.15f}, CPU: {cpu_x_ti:.15f}, Diff: {abs(gpu_x_ti - cpu_x_ti):.15e}")

# Test 3: Mass balance (X(TI) + X(NB) = 1)
tests_total += 1
gpu_sum = gpu_x_ti + gpu_x_nb
if abs(gpu_sum - 1.0) < tolerance:
    print(f"   ✓ TEST 3 PASSED: GPU mass balance satisfied")
    print(f"     X(TI) + X(NB) = {gpu_sum:.15f}, Error: {abs(gpu_sum - 1.0):.15e}")
    tests_passed += 1
else:
    print(f"   ✗ TEST 3 FAILED: GPU mass balance not satisfied")
    print(f"     X(TI) + X(NB) = {gpu_sum:.15f}, Error: {abs(gpu_sum - 1.0):.15e}")

# Test 4: Gibbs energy agreement
tests_total += 1
gm_tolerance = 1.0  # J/mol
if abs(gpu_gm - cpu_gm) < gm_tolerance:
    print(f"   ✓ TEST 4 PASSED: GPU Gibbs energy agrees with CPU within {gm_tolerance} J/mol")
    print(f"     GPU: {gpu_gm:.6f}, CPU: {cpu_gm:.6f}, Diff: {abs(gpu_gm - cpu_gm):.6f} J/mol")
    tests_passed += 1
else:
    print(f"   ✗ TEST 4 FAILED: GPU Gibbs energy does not agree with CPU")
    print(f"     GPU: {gpu_gm:.6f}, CPU: {cpu_gm:.6f}, Diff: {abs(gpu_gm - cpu_gm):.6f} J/mol")

# Overall result
print(f"\n6. OVERALL RESULT:")
print(f"   Tests passed: {tests_passed}/{tests_total}")
if tests_passed == tests_total:
    print("\n   🎉 ALL TESTS PASSED! GPU CORRECTLY CONVERGES TO X(TI)=0.9 🎉")
else:
    print(f"\n   ❌ FAILED: Only {tests_passed}/{tests_total} tests passed")

# Additional validation - multiple conditions
print("\n7. ADDITIONAL VALIDATION - Multiple X(TI) values:")
test_compositions = [0.1, 0.3, 0.5, 0.7, 0.9]
all_pass = True

for x_ti_test in test_compositions:
    conditions_test = {v.X('TI'): x_ti_test, v.T: 600, v.P: 101325}
    
    result_cpu_test = equilibrium(dbf, comps, phases, conditions_test, gpu=False, verbose=False)
    result_gpu_test = equilibrium(dbf, comps, phases, conditions_test, gpu=True, verbose=False)
    
    cpu_x_ti_test = result_cpu_test.X.sel(component='TI').values.flatten()[0]
    gpu_x_ti_test = result_gpu_test.X.sel(component='TI').values.flatten()[0]
    
    error = abs(gpu_x_ti_test - cpu_x_ti_test)
    status = "✓ PASS" if error < tolerance else "✗ FAIL"
    
    print(f"   X(TI)={x_ti_test:.1f}: CPU={cpu_x_ti_test:.6f}, GPU={gpu_x_ti_test:.6f}, Diff={error:.2e} {status}")
    
    if error >= tolerance:
        all_pass = False

if all_pass:
    print("\n   ✓ All composition tests passed!")
else:
    print("\n   ✗ Some composition tests failed!")

print("\n" + "="*60)
print("PROOF COMPLETE")