#!/usr/bin/env python
"""Test single-phase constraint enforcement with prescribed mole fractions."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_single_phase():
    """Test that single-phase systems maintain prescribed mole fractions."""
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    # Test with a single phase - should maintain prescribed composition exactly
    phases = ['LIQUID']
    
    conditions = {
        v.X('AL'): 0.25,
        v.X('CU'): 0.25,
        v.T: 2000,
        v.P: 101325
    }
    
    print("=" * 80)
    print("SINGLE-PHASE CONSTRAINT ENFORCEMENT TEST")
    print("=" * 80)
    print(f"\nPhases: {phases}")
    print(f"Prescribed: X(AL)={conditions[v.X('AL')]:.3f}, X(CU)={conditions[v.X('CU')]:.3f}, X(FE)=0.500")
    print()
    
    # Run CPU
    print("CPU Result:")
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_x = cpu_result.X.sel(component=['AL', 'CU', 'FE']).values.squeeze()
    # Handle potential multi-dimensional array
    if len(cpu_x.shape) > 1:
        cpu_x = cpu_x[0]
    print(f"  X(AL)={float(cpu_x[0]):.6f}, X(CU)={float(cpu_x[1]):.6f}, X(FE)={float(cpu_x[2]):.6f}")
    print(f"  GM = {float(cpu_result.GM.values.squeeze()):.2f} J/mol")
    
    # Run GPU
    print("\nGPU Result:")
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_x = gpu_result.X.sel(component=['AL', 'CU', 'FE']).values.squeeze()
    # Handle potential multi-dimensional array
    if len(gpu_x.shape) > 1:
        gpu_x = gpu_x[0]
    print(f"  X(AL)={float(gpu_x[0]):.6f}, X(CU)={float(gpu_x[1]):.6f}, X(FE)={float(gpu_x[2]):.6f}")
    print(f"  GM = {float(gpu_result.GM.values.squeeze()):.2f} J/mol")
    
    # Check differences
    print("\nDifferences:")
    print(f"  ΔX(AL) = {gpu_x[0] - cpu_x[0]:+.6f}")
    print(f"  ΔX(CU) = {gpu_x[1] - cpu_x[1]:+.6f}")
    print(f"  ΔX(FE) = {gpu_x[2] - cpu_x[2]:+.6f}")
    print(f"  ΔGM = {float(gpu_result.GM.values.squeeze()) - float(cpu_result.GM.values.squeeze()):+.2f} J/mol")
    
    # Verify prescribed values are maintained
    print("\nConstraint Satisfaction:")
    cpu_error_al = abs(cpu_x[0] - conditions[v.X('AL')])
    cpu_error_cu = abs(cpu_x[1] - conditions[v.X('CU')])
    gpu_error_al = abs(gpu_x[0] - conditions[v.X('AL')])
    gpu_error_cu = abs(gpu_x[1] - conditions[v.X('CU')])
    
    print(f"  CPU: |X(AL) - 0.25| = {cpu_error_al:.6f}, |X(CU) - 0.25| = {cpu_error_cu:.6f}")
    print(f"  GPU: |X(AL) - 0.25| = {gpu_error_al:.6f}, |X(CU) - 0.25| = {gpu_error_cu:.6f}")
    
    # Test result
    max_error = max(gpu_error_al, gpu_error_cu)
    if max_error < 1e-6:
        print(f"\n✓ TEST PASSED: GPU maintains prescribed mole fractions (max error = {max_error:.2e})")
    else:
        print(f"\n✗ TEST FAILED: GPU does not maintain prescribed mole fractions (max error = {max_error:.2e})")
    
    return max_error < 1e-6

def test_multi_phase():
    """Test that multi-phase systems still optimize correctly."""
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    # Test with multiple phases - should find equilibrium
    phases = ['LIQUID', 'BCC_B2', 'AL5FE2']
    
    conditions = {
        v.X('AL'): 0.60,
        v.X('CU'): 0.10,
        v.T: 600,
        v.P: 101325
    }
    
    print("\n" + "=" * 80)
    print("MULTI-PHASE EQUILIBRIUM TEST")
    print("=" * 80)
    print(f"\nPhases: {phases}")
    print(f"Prescribed: X(AL)={conditions[v.X('AL')]:.3f}, X(CU)={conditions[v.X('CU')]:.3f}, X(FE)=0.300")
    print()
    
    # Run CPU
    print("CPU Result:")
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_x = cpu_result.X.sel(component=['AL', 'CU', 'FE']).values.squeeze()
    # Handle potential multi-dimensional array
    if len(cpu_x.shape) > 1:
        cpu_x = cpu_x[0]
    print(f"  X(AL)={float(cpu_x[0]):.6f}, X(CU)={float(cpu_x[1]):.6f}, X(FE)={float(cpu_x[2]):.6f}")
    print(f"  GM = {float(cpu_result.GM.values.squeeze()):.2f} J/mol")
    
    # Run GPU
    print("\nGPU Result:")
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_x = gpu_result.X.sel(component=['AL', 'CU', 'FE']).values.squeeze()
    # Handle potential multi-dimensional array
    if len(gpu_x.shape) > 1:
        gpu_x = gpu_x[0]
    print(f"  X(AL)={float(gpu_x[0]):.6f}, X(CU)={float(gpu_x[1]):.6f}, X(FE)={float(gpu_x[2]):.6f}")
    print(f"  GM = {float(gpu_result.GM.values.squeeze()):.2f} J/mol")
    
    # Check differences
    print("\nDifferences:")
    print(f"  ΔX(AL) = {gpu_x[0] - cpu_x[0]:+.6f}")
    print(f"  ΔX(CU) = {gpu_x[1] - cpu_x[1]:+.6f}")
    print(f"  ΔX(FE) = {gpu_x[2] - cpu_x[2]:+.6f}")
    print(f"  ΔGM = {float(gpu_result.GM.values.squeeze()) - float(cpu_result.GM.values.squeeze()):+.2f} J/mol")
    
    # Test result
    max_diff = max(abs(gpu_x[0] - cpu_x[0]), abs(gpu_x[1] - cpu_x[1]), abs(gpu_x[2] - cpu_x[2]))
    if max_diff < 1e-6:
        print(f"\n✓ TEST PASSED: GPU matches CPU for multi-phase equilibrium (max diff = {max_diff:.2e})")
    else:
        print(f"\n✗ TEST FAILED: GPU differs from CPU for multi-phase equilibrium (max diff = {max_diff:.2e})")
    
    return max_diff < 1e-6

if __name__ == "__main__":
    test1_passed = test_single_phase()
    test2_passed = test_multi_phase()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if test1_passed and test2_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
        if not test1_passed:
            print("  - Single-phase constraint enforcement FAILED")
        if not test2_passed:
            print("  - Multi-phase equilibrium FAILED")