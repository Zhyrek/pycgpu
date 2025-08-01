#!/usr/bin/env python
"""
Comprehensive test of GPU equilibrium calculations for Nb-Ti system.
Tests multiple conditions and compares GPU vs CPU results.
"""

from pycalphad import Database, equilibrium
import numpy as np
import os
import sys

# Clear GPU cache
os.system('rm -f ~/.cupy/kernel_cache/*')

# Load database
db = Database('/mnt/c/users/scott/Documents/pycalphad/tests/databases/Ni-Ti_Dupin_2018.tdb')
components = ['NI', 'TI', 'VA']
phases = ['LIQUID', 'FCC_L12', 'HCP_A3', 'BCC_B2']

# Define test conditions
test_conditions = [
    # Single phase regions
    {'T': 1800, 'P': 101325, 'X(TI)': 0.1, 'name': 'High T, Ni-rich (liquid)'},
    {'T': 1800, 'P': 101325, 'X(TI)': 0.9, 'name': 'High T, Ti-rich (liquid)'},
    
    # Two-phase regions
    {'T': 1200, 'P': 101325, 'X(TI)': 0.3, 'name': 'Medium T, two-phase'},
    {'T': 1000, 'P': 101325, 'X(TI)': 0.5, 'name': 'Low T, equiatomic'},
    {'T': 800, 'P': 101325, 'X(TI)': 0.7, 'name': 'Low T, Ti-rich'},
    
    # Near phase boundaries
    {'T': 1273, 'P': 101325, 'X(TI)': 0.25, 'name': 'Near FCC boundary'},
    {'T': 1273, 'P': 101325, 'X(TI)': 0.75, 'name': 'Near HCP boundary'},
    
    # Edge cases
    {'T': 500, 'P': 101325, 'X(TI)': 0.01, 'name': 'Very low T, almost pure Ni'},
    {'T': 500, 'P': 101325, 'X(TI)': 0.99, 'name': 'Very low T, almost pure Ti'},
    {'T': 1273, 'P': 101325, 'X(TI)': 0.5, 'name': 'Exact center composition'},
]

print("Comprehensive Nb-Ti System Test")
print("=" * 80)
print(f"Testing {len(test_conditions)} conditions with {len(phases)} phases")
print(f"Components: {components}")
print(f"Phases: {phases}")
print("=" * 80)

# Track results
passed_tests = 0
failed_tests = 0
phase_match_tests = 0
amount_match_tests = 0

for idx, test in enumerate(test_conditions):
    print(f"\nTest {idx+1}/{len(test_conditions)}: {test['name']}")
    print("-" * 60)
    
    conditions = {k: v for k, v in test.items() if k != 'name'}
    
    # GPU calculation
    try:
        result_gpu = equilibrium(db, components, phases, conditions, 
                               calc_opts={'pdens': 2000}, gpu=True)
        
        gpu_phases = {}
        for phase in np.unique(result_gpu.Phase.values):
            if phase != '':
                mask = result_gpu.Phase.values == phase
                amount = result_gpu.NP.values[mask][0]
                if amount > 1e-10:
                    gpu_phases[phase] = amount
        
        print(f"GPU: {', '.join([f'{p}({a:.3f})' for p, a in gpu_phases.items()]) if gpu_phases else 'No phases'}")
        
    except Exception as e:
        print(f"GPU Error: {type(e).__name__}: {str(e)[:100]}")
        gpu_phases = {}
        failed_tests += 1
        continue
    
    # CPU calculation
    try:
        result_cpu = equilibrium(db, components, phases, conditions, 
                               calc_opts={'pdens': 2000})
        
        cpu_phases = {}
        for phase in np.unique(result_cpu.Phase.values):
            if phase != '':
                mask = result_cpu.Phase.values == phase
                amount = result_cpu.NP.values[mask][0]
                if amount > 1e-10:
                    cpu_phases[phase] = amount
        
        print(f"CPU: {', '.join([f'{p}({a:.3f})' for p, a in cpu_phases.items()]) if cpu_phases else 'No phases'}")
        
    except Exception as e:
        print(f"CPU Error: {type(e).__name__}: {str(e)[:100]}")
        cpu_phases = {}
        failed_tests += 1
        continue
    
    # Compare results
    gpu_phase_names = set(gpu_phases.keys())
    cpu_phase_names = set(cpu_phases.keys())
    
    if gpu_phase_names == cpu_phase_names:
        print("✓ Phases match")
        phase_match_tests += 1
        
        # Check amounts
        max_diff = 0
        for phase in gpu_phase_names:
            diff = abs(gpu_phases[phase] - cpu_phases[phase])
            max_diff = max(max_diff, diff)
        
        if max_diff < 1e-4:
            print(f"✓ Amounts match (max diff: {max_diff:.2e})")
            amount_match_tests += 1
            passed_tests += 1
        else:
            print(f"✗ Amounts differ (max diff: {max_diff:.2e})")
            for phase in gpu_phase_names:
                gpu_amt = gpu_phases[phase]
                cpu_amt = cpu_phases[phase]
                diff = abs(gpu_amt - cpu_amt)
                if diff > 1e-6:
                    print(f"  {phase}: GPU={gpu_amt:.6f}, CPU={cpu_amt:.6f}, diff={diff:.6f}")
            failed_tests += 1
    else:
        print("✗ Phases differ")
        print(f"  GPU phases: {gpu_phase_names}")
        print(f"  CPU phases: {cpu_phase_names}")
        print(f"  Missing in GPU: {cpu_phase_names - gpu_phase_names}")
        print(f"  Extra in GPU: {gpu_phase_names - cpu_phase_names}")
        failed_tests += 1

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total tests: {len(test_conditions)}")
print(f"Passed: {passed_tests}")
print(f"Failed: {failed_tests}")
print(f"Phase matches: {phase_match_tests}/{len(test_conditions)}")
print(f"Amount matches: {amount_match_tests}/{len(test_conditions)}")
print(f"Success rate: {100 * passed_tests / len(test_conditions):.1f}%")

if failed_tests == 0:
    print("\n✓ All tests passed!")
    sys.exit(0)
else:
    print(f"\n✗ {failed_tests} tests failed")
    sys.exit(1)