#!/usr/bin/env python
"""Test specific failing condition from AlCuFe 8-phase test."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    # All 8 phases from the comprehensive test
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12_FCC', 
              'ALCU_THETA', 'AL13FE4_D03', 'AL5FE2_D82']
    
    # Test #26 from the results - shows 41.16 J/mol difference
    conditions = {
        v.X('AL'): 0.40,
        v.X('CU'): 0.40,
        v.T: 600,
        v.P: 101325
    }
    
    print("=" * 80)
    print("TESTING FAILING CONDITION #26")
    print("=" * 80)
    print(f"X(AL)={conditions[v.X('AL')]:.2f}, X(CU)={conditions[v.X('CU')]:.2f}, X(FE)=0.20, T={conditions[v.T]}K")
    print(f"Phases: {phases}")
    print()
    
    # Run CPU
    print("CPU Result:")
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = float(cpu_result.GM.values.squeeze())
    cpu_x = cpu_result.X.sel(component=['AL', 'CU', 'FE']).values.squeeze()
    if len(cpu_x.shape) > 1:
        cpu_x = cpu_x[0]
    print(f"  GM = {cpu_gm:.6f} J/mol")
    print(f"  X = [AL:{cpu_x[0]:.6f}, CU:{cpu_x[1]:.6f}, FE:{cpu_x[2]:.6f}]")
    
    # Check phase fractions
    cpu_phases = {}
    for phase in phases:
        try:
            np_val = float(cpu_result.NP.sel(phase=phase).values.squeeze())
            if np_val > 1e-10:
                cpu_phases[phase] = np_val
        except:
            pass
    print(f"  Active phases: {cpu_phases}")
    
    # Run GPU
    print("\nGPU Result:")
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = float(gpu_result.GM.values.squeeze())
    gpu_x = gpu_result.X.sel(component=['AL', 'CU', 'FE']).values.squeeze()
    if len(gpu_x.shape) > 1:
        gpu_x = gpu_x[0]
    print(f"  GM = {gpu_gm:.6f} J/mol")
    print(f"  X = [AL:{gpu_x[0]:.6f}, CU:{gpu_x[1]:.6f}, FE:{gpu_x[2]:.6f}]")
    
    # Check phase fractions
    gpu_phases = {}
    for phase in phases:
        try:
            np_val = float(gpu_result.NP.sel(phase=phase).values.squeeze())
            if np_val > 1e-10:
                gpu_phases[phase] = np_val
        except:
            pass
    print(f"  Active phases: {gpu_phases}")
    
    # Compare
    print("\nDifferences:")
    gm_diff = gpu_gm - cpu_gm
    print(f"  ΔGM = {gm_diff:.6f} J/mol")
    print(f"  ΔX(AL) = {gpu_x[0] - cpu_x[0]:+.6f}")
    print(f"  ΔX(CU) = {gpu_x[1] - cpu_x[1]:+.6f}")
    print(f"  ΔX(FE) = {gpu_x[2] - cpu_x[2]:+.6f}")
    
    # Compare phase assemblages
    print("\nPhase differences:")
    all_phases = set(cpu_phases.keys()) | set(gpu_phases.keys())
    for phase in sorted(all_phases):
        cpu_val = cpu_phases.get(phase, 0.0)
        gpu_val = gpu_phases.get(phase, 0.0)
        if abs(cpu_val - gpu_val) > 1e-6:
            print(f"  {phase}: CPU={cpu_val:.6f}, GPU={gpu_val:.6f}, diff={gpu_val-cpu_val:+.6f}")
    
    # Test result
    if abs(gm_diff) > 1.0:  # More than 1 J/mol difference
        print(f"\n✗ LARGE DIFFERENCE DETECTED: {abs(gm_diff):.2f} J/mol")
        print("\nRunning again with verbose output to capture debug info...")
        print("\n" + "=" * 80)
        print("VERBOSE CPU RUN")
        print("=" * 80)
        cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
        print("\n" + "=" * 80)
        print("VERBOSE GPU RUN")  
        print("=" * 80)
        gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
    else:
        print(f"\n✓ Results match within tolerance")

if __name__ == "__main__":
    main()